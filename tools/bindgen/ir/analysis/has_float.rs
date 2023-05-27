use super::{gen_deps, Monotone, YConstrain};
use crate::ir::comp::Field;
use crate::ir::comp::FieldMethods;
use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::traversal::EdgeKind;
use crate::ir::ty::TypeKind;
use crate::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub(crate) struct HasFloat<'ctx> {
    ctx: &'ctx BindgenContext,
    has_float: HashSet<ItemId>,
    deps: HashMap<ItemId, Vec<ItemId>>,
}

impl<'ctx> HasFloat<'ctx> {
    fn consider_edge(k: EdgeKind) -> bool {
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
        trace!("inserting {:?} into the has_float set", id);
        let was_not_already_in_set = self.has_float.insert(id);
        assert!(
            was_not_already_in_set,
            "We shouldn't try and insert {:?} twice because if it was \
             already in the set, `constrain` should have exited early.",
            id
        );
        YConstrain::Changed
    }
}

impl<'ctx> Monotone for HasFloat<'ctx> {
    type Node = ItemId;
    type Extra = &'ctx BindgenContext;
    type Output = HashSet<ItemId>;

    fn new(ctx: &'ctx BindgenContext) -> HasFloat<'ctx> {
        let has_float = HashSet::default();
        let deps = gen_deps(ctx, Self::consider_edge);
        HasFloat { ctx, has_float, deps }
    }

    fn initial_worklist(&self) -> Vec<ItemId> {
        self.ctx.allowed_items().iter().cloned().collect()
    }

    fn constrain(&mut self, id: ItemId) -> YConstrain {
        trace!("constrain: {:?}", id);
        if self.has_float.contains(&id) {
            trace!("    already know it do not have float");
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
            | TypeKind::Function(..)
            | TypeKind::Enum(..)
            | TypeKind::Reference(..)
            | TypeKind::TypeParam
            | TypeKind::Opaque
            | TypeKind::Pointer(..)
            | TypeKind::UnresolvedTypeRef(..) => {
                trace!("    simple type that do not have float");
                YConstrain::Same
            },
            TypeKind::Float(..) | TypeKind::Complex(..) => {
                trace!("    float type has float");
                self.insert(id)
            },
            TypeKind::Array(t, _) => {
                if self.has_float.contains(&t.into()) {
                    trace!("    Array with type T that has float also has float");
                    return self.insert(id);
                }
                trace!("    Array with type T that do not have float also do not have float");
                YConstrain::Same
            },
            TypeKind::Vector(t, _) => {
                if self.has_float.contains(&t.into()) {
                    trace!("    Vector with type T that has float also has float");
                    return self.insert(id);
                }
                trace!("    Vector with type T that do not have float also do not have float");
                YConstrain::Same
            },
            TypeKind::ResolvedTypeRef(t)
            | TypeKind::TemplateAlias(t, _)
            | TypeKind::Alias(t)
            | TypeKind::BlockPointer(t) => {
                if self.has_float.contains(&t.into()) {
                    trace!(
                        "    aliases and type refs to T which have float \
                         also have float"
                    );
                    self.insert(id)
                } else {
                    trace!(
                        "    aliases and type refs to T which do not have float \
                            also do not have floaarrayt"
                    );
                    YConstrain::Same
                }
            },
            TypeKind::Comp(ref info) => {
                let bases_have = info
                    .base_members()
                    .iter()
                    .any(|base| self.has_float.contains(&base.ty.into()));
                if bases_have {
                    trace!("    bases have float, so we also have");
                    return self.insert(id);
                }
                let fields_have = info.fields().iter().any(|f| match *f {
                    Field::DataMember(ref data) => self.has_float.contains(&data.ty().into()),
                    Field::Bitfields(ref bfu) => {
                        bfu.bitfields().iter().any(|b| self.has_float.contains(&b.ty().into()))
                    },
                });
                if fields_have {
                    trace!("    fields have float, so we also have");
                    return self.insert(id);
                }
                trace!("    comp doesn't have float");
                YConstrain::Same
            },
            TypeKind::TemplateInstantiation(ref template) => {
                let args_have = template
                    .template_arguments()
                    .iter()
                    .any(|arg| self.has_float.contains(&arg.into()));
                if args_have {
                    trace!(
                        "    template args have float, so \
                         insantiation also has float"
                    );
                    return self.insert(id);
                }

                let def_has = self.has_float.contains(&template.template_definition().into());
                if def_has {
                    trace!(
                        "    template definition has float, so \
                         insantiation also has"
                    );
                    return self.insert(id);
                }

                trace!("    template instantiation do not have float");
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

impl<'ctx> From<HasFloat<'ctx>> for HashSet<ItemId> {
    fn from(x: HasFloat<'ctx>) -> Self {
        x.has_float
    }
}
