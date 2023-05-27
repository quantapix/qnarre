use super::{gen_deps, Monotone, YConstrain};
use crate::ir::comp::{CompKind, Field, FieldMethods};
use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::traversal::EdgeKind;
use crate::ir::ty::TypeKind;
use crate::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub(crate) struct HasDestructorAnalysis<'ctx> {
    ctx: &'ctx BindgenContext,
    ys: HashSet<ItemId>,
    deps: HashMap<ItemId, Vec<ItemId>>,
}

impl<'ctx> HasDestructorAnalysis<'ctx> {
    fn check_edge(k: EdgeKind) -> bool {
        matches!(
            k,
            EdgeKind::TypeReference
                | EdgeKind::BaseMember
                | EdgeKind::Field
                | EdgeKind::TemplateArgument
                | EdgeKind::TemplateDeclaration
        )
    }

    fn insert<Id: Into<ItemId>>(&mut self, id: Id) -> YConstrain {
        let id = id.into();
        let newly = self.ys.insert(id);
        assert!(newly);
        YConstrain::Changed
    }
}

impl<'ctx> Monotone for HasDestructorAnalysis<'ctx> {
    type Node = ItemId;
    type Extra = &'ctx BindgenContext;
    type Output = HashSet<ItemId>;
    fn new(ctx: &'ctx BindgenContext) -> Self {
        let ys = HashSet::default();
        let deps = gen_deps(ctx, Self::check_edge);
        HasDestructorAnalysis { ctx, ys, deps }
    }

    fn initial_worklist(&self) -> Vec<ItemId> {
        self.ctx.allowed_items().iter().cloned().collect()
    }

    fn constrain(&mut self, id: ItemId) -> YConstrain {
        if self.ys.contains(&id) {
            return YConstrain::Same;
        }
        let item = self.ctx.resolve_item(id);
        let ty = match item.as_type() {
            None => return YConstrain::Same,
            Some(ty) => ty,
        };
        match *ty.kind() {
            TypeKind::TemplateAlias(t, _) | TypeKind::Alias(t) | TypeKind::ResolvedTypeRef(t) => {
                if self.ys.contains(&t.into()) {
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
                        let base_or_field_destructor =
                            info.base_members().iter().any(|base| self.ys.contains(&base.ty.into()))
                                || info.fields().iter().any(|field| match *field {
                                    Field::DataMember(ref data) => self.ys.contains(&data.ty().into()),
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
                let definition_or_arg_destructor = self.ys.contains(&inst.template_definition().into())
                    || inst
                        .template_arguments()
                        .iter()
                        .any(|arg| self.ys.contains(&arg.into()));
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
        if let Some(es) = self.deps.get(&id) {
            for e in es {
                trace!("enqueue {:?} into worklist", e);
                f(*e);
            }
        }
    }
}

impl<'ctx> From<HasDestructorAnalysis<'ctx>> for HashSet<ItemId> {
    fn from(x: HasDestructorAnalysis<'ctx>) -> Self {
        x.ys
    }
}
