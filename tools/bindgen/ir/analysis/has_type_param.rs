use super::{gen_deps, Monotone, YConstrain};
use crate::ir::comp::Field;
use crate::ir::comp::FieldMethods;
use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::traversal::EdgeKind;
use crate::ir::ty::TypeKind;
use crate::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub(crate) struct HasTyParamInArrayAnalysis<'ctx> {
    ctx: &'ctx BindgenContext,
    ys: HashSet<ItemId>,
    deps: HashMap<ItemId, Vec<ItemId>>,
}

impl<'ctx> HasTyParamInArrayAnalysis<'ctx> {
    fn check_edge(k: EdgeKind) -> bool {
        match k {
            EdgeKind::BaseMember
            | EdgeKind::Field
            | EdgeKind::TypeReference
            | EdgeKind::VarType
            | EdgeKind::TemplateArgument
            | EdgeKind::TemplateDeclaration
            | EdgeKind::TemplateParameterDefinition => true,

            EdgeKind::Constructor
            | EdgeKind::Destructor
            | EdgeKind::FunctionReturn
            | EdgeKind::FunctionParameter
            | EdgeKind::InnerType
            | EdgeKind::InnerVar
            | EdgeKind::Method => false,
            EdgeKind::Generic => false,
        }
    }

    fn insert<Id: Into<ItemId>>(&mut self, id: Id) -> YConstrain {
        let id = id.into();
        let newly = self.ys.insert(id);
        assert!(newly);
        YConstrain::Changed
    }
}

impl<'ctx> Monotone for HasTyParamInArrayAnalysis<'ctx> {
    type Node = ItemId;
    type Extra = &'ctx BindgenContext;
    type Output = HashSet<ItemId>;

    fn new(ctx: &'ctx BindgenContext) -> HasTyParamInArrayAnalysis<'ctx> {
        let ys = HashSet::default();
        let deps = gen_deps(ctx, Self::check_edge);
        HasTyParamInArrayAnalysis { ctx, ys, deps }
    }

    fn initial_worklist(&self) -> Vec<ItemId> {
        self.ctx.allowed_items().iter().cloned().collect()
    }

    fn constrain(&mut self, id: ItemId) -> YConstrain {
        trace!("constrain: {:?}", id);
        if self.ys.contains(&id) {
            trace!("    already know it do not have array");
            return YConstrain::Same;
        }
        let item = self.ctx.resolve_item(id);
        let ty = match item.as_type() {
            Some(ty) => ty,
            None => {
                trace!("    not a type; ignoring");
                return YConstrain::Same;
            },
        };
        match *ty.kind() {
            TypeKind::Void
            | TypeKind::NullPtr
            | TypeKind::Int(..)
            | TypeKind::Float(..)
            | TypeKind::Vector(..)
            | TypeKind::Complex(..)
            | TypeKind::Function(..)
            | TypeKind::Enum(..)
            | TypeKind::Reference(..)
            | TypeKind::TypeParam
            | TypeKind::Opaque
            | TypeKind::Pointer(..)
            | TypeKind::UnresolvedTypeRef(..) => {
                trace!("    simple type that do not have array");
                YConstrain::Same
            },
            TypeKind::Array(t, _) => {
                let inner_ty = self.ctx.resolve_type(t).canonical_type(self.ctx);
                match *inner_ty.kind() {
                    TypeKind::TypeParam => {
                        trace!("    Array with Named type has type parameter");
                        self.insert(id)
                    },
                    _ => {
                        trace!("    Array without Named type does have type parameter");
                        YConstrain::Same
                    },
                }
            },

            TypeKind::ResolvedTypeRef(t)
            | TypeKind::TemplateAlias(t, _)
            | TypeKind::Alias(t)
            | TypeKind::BlockPointer(t) => {
                if self.ys.contains(&t.into()) {
                    trace!(
                        "    aliases and type refs to T which have array \
                         also have array"
                    );
                    self.insert(id)
                } else {
                    trace!(
                        "    aliases and type refs to T which do not have array \
                            also do not have array"
                    );
                    YConstrain::Same
                }
            },

            TypeKind::Comp(ref info) => {
                let bases_have = info.base_members().iter().any(|base| self.ys.contains(&base.ty.into()));
                if bases_have {
                    trace!("    bases have array, so we also have");
                    return self.insert(id);
                }
                let fields_have = info.fields().iter().any(|f| match *f {
                    Field::DataMember(ref data) => self.ys.contains(&data.ty().into()),
                    Field::Bitfields(..) => false,
                });
                if fields_have {
                    trace!("    fields have array, so we also have");
                    return self.insert(id);
                }

                trace!("    comp doesn't have array");
                YConstrain::Same
            },

            TypeKind::TemplateInstantiation(ref template) => {
                let args_have = template
                    .template_arguments()
                    .iter()
                    .any(|arg| self.ys.contains(&arg.into()));
                if args_have {
                    trace!(
                        "    template args have array, so \
                         insantiation also has array"
                    );
                    return self.insert(id);
                }

                let def_has = self.ys.contains(&template.template_definition().into());
                if def_has {
                    trace!(
                        "    template definition has array, so \
                         insantiation also has"
                    );
                    return self.insert(id);
                }

                trace!("    template instantiation do not have array");
                YConstrain::Same
            },
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

impl<'ctx> From<HasTyParamInArrayAnalysis<'ctx>> for HashSet<ItemId> {
    fn from(x: HasTyParamInArrayAnalysis<'ctx>) -> Self {
        x.ys
    }
}
