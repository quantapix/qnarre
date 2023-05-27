use super::{gen_deps, Monotone, YConstrain};
use crate::ir::comp::{CompKind, Field, FieldMethods};
use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::traversal::EdgeKind;
use crate::ir::ty::TypeKind;
use crate::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub(crate) struct HasDestructorAnalysis<'ctx> {
    ctx: &'ctx BindgenContext,
    deps: HashMap<ItemId, Vec<ItemId>>,
    ys: HashSet<ItemId>,
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
        let i = self.ctx.resolve_item(id);
        let ty = match i.as_type() {
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
            TypeKind::Comp(ref x) => {
                if x.has_own_destructor() {
                    return self.insert(id);
                }
                match x.kind() {
                    CompKind::Union => YConstrain::Same,
                    CompKind::Struct => {
                        let destr = x.base_members().iter().any(|x| self.ys.contains(&x.ty.into()))
                            || x.fields().iter().any(|x| match *x {
                                Field::DataMember(ref x) => self.ys.contains(&x.ty().into()),
                                Field::Bitfields(_) => false,
                            });
                        if destr {
                            self.insert(id)
                        } else {
                            YConstrain::Same
                        }
                    },
                }
            },
            TypeKind::TemplateInstantiation(ref t) => {
                let destr = self.ys.contains(&t.template_definition().into())
                    || t.template_arguments().iter().any(|x| self.ys.contains(&x.into()));
                if destr {
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
