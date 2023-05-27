use super::{gen_deps, Monotone, YConstrain};
use crate::ir::comp::{CompKind, Field, FieldMethods};
use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::traversal::EdgeKind;
use crate::ir::ty::TypeKind;
use crate::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub(crate) struct HasDestructorAnalysis<'ctx> {
    ctx: &'ctx BindgenContext,

    have_destructor: HashSet<ItemId>,

    dependencies: HashMap<ItemId, Vec<ItemId>>,
}

impl<'ctx> HasDestructorAnalysis<'ctx> {
    fn consider_edge(kind: EdgeKind) -> bool {
        matches!(
            kind,
            EdgeKind::TypeReference
                | EdgeKind::BaseMember
                | EdgeKind::Field
                | EdgeKind::TemplateArgument
                | EdgeKind::TemplateDeclaration
        )
    }

    fn insert<Id: Into<ItemId>>(&mut self, id: Id) -> YConstrain {
        let id = id.into();
        let was_not_already_in_set = self.have_destructor.insert(id);
        assert!(
            was_not_already_in_set,
            "We shouldn't try and insert {:?} twice because if it was \
             already in the set, `constrain` should have exited early.",
            id
        );
        YConstrain::Changed
    }
}

impl<'ctx> Monotone for HasDestructorAnalysis<'ctx> {
    type Node = ItemId;
    type Extra = &'ctx BindgenContext;
    type Output = HashSet<ItemId>;

    fn new(ctx: &'ctx BindgenContext) -> Self {
        let have_destructor = HashSet::default();
        let dependencies = gen_deps(ctx, Self::consider_edge);

        HasDestructorAnalysis {
            ctx,
            have_destructor,
            dependencies,
        }
    }

    fn initial_worklist(&self) -> Vec<ItemId> {
        self.ctx.allowed_items().iter().cloned().collect()
    }

    fn constrain(&mut self, id: ItemId) -> YConstrain {
        if self.have_destructor.contains(&id) {
            return YConstrain::Same;
        }

        let item = self.ctx.resolve_item(id);
        let ty = match item.as_type() {
            None => return YConstrain::Same,
            Some(ty) => ty,
        };

        match *ty.kind() {
            TypeKind::TemplateAlias(t, _) | TypeKind::Alias(t) | TypeKind::ResolvedTypeRef(t) => {
                if self.have_destructor.contains(&t.into()) {
                    self.insert(id)
                } else {
                    YConstrain::Same
                }
            },

            TypeKind::Comp(ref info) => {
                if info.has_own_destructor() {
                    return self.insert(id);
                }

                match info.kind() {
                    CompKind::Union => YConstrain::Same,
                    CompKind::Struct => {
                        let base_or_field_destructor = info
                            .base_members()
                            .iter()
                            .any(|base| self.have_destructor.contains(&base.ty.into()))
                            || info.fields().iter().any(|field| match *field {
                                Field::DataMember(ref data) => self.have_destructor.contains(&data.ty().into()),
                                Field::Bitfields(_) => false,
                            });
                        if base_or_field_destructor {
                            self.insert(id)
                        } else {
                            YConstrain::Same
                        }
                    },
                }
            },

            TypeKind::TemplateInstantiation(ref inst) => {
                let definition_or_arg_destructor = self.have_destructor.contains(&inst.template_definition().into())
                    || inst
                        .template_arguments()
                        .iter()
                        .any(|arg| self.have_destructor.contains(&arg.into()));
                if definition_or_arg_destructor {
                    self.insert(id)
                } else {
                    YConstrain::Same
                }
            },

            _ => YConstrain::Same,
        }
    }

    fn each_depending_on<F>(&self, id: ItemId, mut f: F)
    where
        F: FnMut(ItemId),
    {
        if let Some(edges) = self.dependencies.get(&id) {
            for item in edges {
                trace!("enqueue {:?} into worklist", item);
                f(*item);
            }
        }
    }
}

impl<'ctx> From<HasDestructorAnalysis<'ctx>> for HashSet<ItemId> {
    fn from(analysis: HasDestructorAnalysis<'ctx>) -> Self {
        analysis.have_destructor
    }
}
