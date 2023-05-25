use super::{generate_dependencies, ConstrainResult, MonotoneFramework};
use crate::ir::comp::Field;
use crate::ir::comp::FieldMethods;
use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::traversal::EdgeKind;
use crate::ir::ty::TypeKind;
use crate::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub(crate) struct HasTypeParameterInArray<'ctx> {
    ctx: &'ctx BindgenContext,

    has_type_parameter_in_array: HashSet<ItemId>,

    dependencies: HashMap<ItemId, Vec<ItemId>>,
}

impl<'ctx> HasTypeParameterInArray<'ctx> {
    fn consider_edge(kind: EdgeKind) -> bool {
        match kind {
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

    fn insert<Id: Into<ItemId>>(&mut self, id: Id) -> ConstrainResult {
        let id = id.into();
        trace!("inserting {:?} into the has_type_parameter_in_array set", id);

        let was_not_already_in_set = self.has_type_parameter_in_array.insert(id);
        assert!(
            was_not_already_in_set,
            "We shouldn't try and insert {:?} twice because if it was \
             already in the set, `constrain` should have exited early.",
            id
        );

        ConstrainResult::Changed
    }
}

impl<'ctx> MonotoneFramework for HasTypeParameterInArray<'ctx> {
    type Node = ItemId;
    type Extra = &'ctx BindgenContext;
    type Output = HashSet<ItemId>;

    fn new(ctx: &'ctx BindgenContext) -> HasTypeParameterInArray<'ctx> {
        let has_type_parameter_in_array = HashSet::default();
        let dependencies = generate_dependencies(ctx, Self::consider_edge);

        HasTypeParameterInArray {
            ctx,
            has_type_parameter_in_array,
            dependencies,
        }
    }

    fn initial_worklist(&self) -> Vec<ItemId> {
        self.ctx.allowlisted_items().iter().cloned().collect()
    }

    fn constrain(&mut self, id: ItemId) -> ConstrainResult {
        trace!("constrain: {:?}", id);

        if self.has_type_parameter_in_array.contains(&id) {
            trace!("    already know it do not have array");
            return ConstrainResult::Same;
        }

        let item = self.ctx.resolve_item(id);
        let ty = match item.as_type() {
            Some(ty) => ty,
            None => {
                trace!("    not a type; ignoring");
                return ConstrainResult::Same;
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
            | TypeKind::UnresolvedTypeRef(..)
            | TypeKind::ObjCInterface(..)
            | TypeKind::ObjCId
            | TypeKind::ObjCSel => {
                trace!("    simple type that do not have array");
                ConstrainResult::Same
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
                        ConstrainResult::Same
                    },
                }
            },

            TypeKind::ResolvedTypeRef(t)
            | TypeKind::TemplateAlias(t, _)
            | TypeKind::Alias(t)
            | TypeKind::BlockPointer(t) => {
                if self.has_type_parameter_in_array.contains(&t.into()) {
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
                    ConstrainResult::Same
                }
            },

            TypeKind::Comp(ref info) => {
                let bases_have = info
                    .base_members()
                    .iter()
                    .any(|base| self.has_type_parameter_in_array.contains(&base.ty.into()));
                if bases_have {
                    trace!("    bases have array, so we also have");
                    return self.insert(id);
                }
                let fields_have = info.fields().iter().any(|f| match *f {
                    Field::DataMember(ref data) => self.has_type_parameter_in_array.contains(&data.ty().into()),
                    Field::Bitfields(..) => false,
                });
                if fields_have {
                    trace!("    fields have array, so we also have");
                    return self.insert(id);
                }

                trace!("    comp doesn't have array");
                ConstrainResult::Same
            },

            TypeKind::TemplateInstantiation(ref template) => {
                let args_have = template
                    .template_arguments()
                    .iter()
                    .any(|arg| self.has_type_parameter_in_array.contains(&arg.into()));
                if args_have {
                    trace!(
                        "    template args have array, so \
                         insantiation also has array"
                    );
                    return self.insert(id);
                }

                let def_has = self
                    .has_type_parameter_in_array
                    .contains(&template.template_definition().into());
                if def_has {
                    trace!(
                        "    template definition has array, so \
                         insantiation also has"
                    );
                    return self.insert(id);
                }

                trace!("    template instantiation do not have array");
                ConstrainResult::Same
            },
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

impl<'ctx> From<HasTypeParameterInArray<'ctx>> for HashSet<ItemId> {
    fn from(analysis: HasTypeParameterInArray<'ctx>) -> Self {
        analysis.has_type_parameter_in_array
    }
}
