use super::{gen_deps, Monotone, YConstrain};
use crate::ir::comp::Field;
use crate::ir::comp::FieldMethods;
use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::traversal::EdgeKind;
use crate::ir::ty::TypeKind;
use crate::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub(crate) struct HasFloatAnalysis<'ctx> {
    ctx: &'ctx BindgenContext,
    ys: HashSet<ItemId>,
    deps: HashMap<ItemId, Vec<ItemId>>,
}

impl<'ctx> HasFloatAnalysis<'ctx> {
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

impl<'ctx> Monotone for HasFloatAnalysis<'ctx> {
    type Node = ItemId;
    type Extra = &'ctx BindgenContext;
    type Output = HashSet<ItemId>;

    fn new(ctx: &'ctx BindgenContext) -> HasFloatAnalysis<'ctx> {
        let ys = HashSet::default();
        let deps = gen_deps(ctx, Self::check_edge);
        HasFloatAnalysis { ctx, ys, deps }
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
            TypeKind::Void
            | TypeKind::NullPtr
            | TypeKind::Int(..)
            | TypeKind::Function(..)
            | TypeKind::Enum(..)
            | TypeKind::Reference(..)
            | TypeKind::TypeParam
            | TypeKind::Opaque
            | TypeKind::Pointer(..)
            | TypeKind::UnresolvedTypeRef(..) => YConstrain::Same,
            TypeKind::Float(..) | TypeKind::Complex(..) => self.insert(id),
            TypeKind::Array(t, _) => {
                if self.ys.contains(&t.into()) {
                    return self.insert(id);
                }
                YConstrain::Same
            },
            TypeKind::Vector(t, _) => {
                if self.ys.contains(&t.into()) {
                    return self.insert(id);
                }
                YConstrain::Same
            },
            TypeKind::ResolvedTypeRef(t)
            | TypeKind::TemplateAlias(t, _)
            | TypeKind::Alias(t)
            | TypeKind::BlockPointer(t) => {
                if self.ys.contains(&t.into()) {
                    self.insert(id)
                } else {
                    YConstrain::Same
                }
            },
            TypeKind::Comp(ref x) => {
                let bases = x.base_members().iter().any(|x| self.ys.contains(&x.ty.into()));
                if bases {
                    return self.insert(id);
                }
                let fields = x.fields().iter().any(|x| match *x {
                    Field::DataMember(ref x) => self.ys.contains(&x.ty().into()),
                    Field::Bitfields(ref x) => x.bitfields().iter().any(|x| self.ys.contains(&x.ty().into())),
                });
                if fields {
                    return self.insert(id);
                }
                YConstrain::Same
            },
            TypeKind::TemplateInstantiation(ref t) => {
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

impl<'ctx> From<HasFloatAnalysis<'ctx>> for HashSet<ItemId> {
    fn from(x: HasFloatAnalysis<'ctx>) -> Self {
        x.ys
    }
}
