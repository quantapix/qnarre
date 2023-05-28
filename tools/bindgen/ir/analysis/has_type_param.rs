use super::{gen_deps, Monotone, YConstrain};
use crate::ir::comp::Field;
use crate::ir::comp::FieldMethods;
use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::traversal::EdgeKind;
use crate::ir::ty::TyKind;
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
        if self.ys.contains(&id) {
            return YConstrain::Same;
        }
        let i = self.ctx.resolve_item(id);
        let ty = match i.as_type() {
            Some(ty) => ty,
            None => {
                return YConstrain::Same;
            },
        };
        match *ty.kind() {
            TyKind::Void
            | TyKind::NullPtr
            | TyKind::Int(..)
            | TyKind::Float(..)
            | TyKind::Vector(..)
            | TyKind::Complex(..)
            | TyKind::Function(..)
            | TyKind::Enum(..)
            | TyKind::Reference(..)
            | TyKind::TypeParam
            | TyKind::Opaque
            | TyKind::Pointer(..)
            | TyKind::UnresolvedTypeRef(..) => YConstrain::Same,
            TyKind::Array(t, _) => {
                let ty2 = self.ctx.resolve_type(t).canonical_type(self.ctx);
                match *ty2.kind() {
                    TyKind::TypeParam => self.insert(id),
                    _ => YConstrain::Same,
                }
            },
            TyKind::ResolvedTypeRef(t) | TyKind::TemplateAlias(t, _) | TyKind::Alias(t) | TyKind::BlockPointer(t) => {
                if self.ys.contains(&t.into()) {
                    self.insert(id)
                } else {
                    YConstrain::Same
                }
            },
            TyKind::Comp(ref info) => {
                let bases = info.base_members().iter().any(|x| self.ys.contains(&x.ty.into()));
                if bases {
                    return self.insert(id);
                }
                let fields = info.fields().iter().any(|x| match *x {
                    Field::DataMember(ref x) => self.ys.contains(&x.ty().into()),
                    Field::Bitfields(..) => false,
                });
                if fields {
                    return self.insert(id);
                }
                YConstrain::Same
            },
            TyKind::TemplateInstantiation(ref t) => {
                let args = t.template_arguments().iter().any(|x| self.ys.contains(&x.into()));
                if args {
                    return self.insert(id);
                }
                let def = self.ys.contains(&t.template_definition().into());
                if def {
                    return self.insert(id);
                }
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
