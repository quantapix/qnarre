use super::{gen_deps, HasVtable, Monotone, YConstrain};
use crate::ir::context::{BindgenContext, TypeId};
use crate::ir::item::IsOpaque;
use crate::ir::traversal::EdgeKind;
use crate::ir::ty::TyKind;
use crate::{Entry, HashMap};
use std::{cmp, ops};

pub(crate) trait Sizedness {
    fn sizedness(&self, ctx: &BindgenContext) -> YSizedness;
    fn is_zero_sized(&self, ctx: &BindgenContext) -> bool {
        self.sizedness(ctx) == YSizedness::ZeroSized
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum YSizedness {
    ZeroSized,
    DependsOnTypeParam,
    NonZeroSized,
}

impl Default for YSizedness {
    fn default() -> Self {
        YSizedness::ZeroSized
    }
}

impl YSizedness {
    pub(crate) fn join(self, rhs: Self) -> Self {
        cmp::max(self, rhs)
    }
}

impl ops::BitOr for YSizedness {
    type Output = Self;
    fn bitor(self, rhs: YSizedness) -> Self::Output {
        self.join(rhs)
    }
}

impl ops::BitOrAssign for YSizedness {
    fn bitor_assign(&mut self, rhs: YSizedness) {
        *self = self.join(rhs)
    }
}

#[derive(Debug)]
pub(crate) struct SizednessAnalysis<'ctx> {
    ctx: &'ctx BindgenContext,
    deps: HashMap<TypeId, Vec<TypeId>>,
    ys: HashMap<TypeId, YSizedness>,
}

impl<'ctx> SizednessAnalysis<'ctx> {
    fn check_edge(k: EdgeKind) -> bool {
        matches!(
            k,
            EdgeKind::TemplateArgument
                | EdgeKind::TemplateParameterDefinition
                | EdgeKind::TemplateDeclaration
                | EdgeKind::TypeReference
                | EdgeKind::BaseMember
                | EdgeKind::Field
        )
    }

    fn insert(&mut self, id: TypeId, y: YSizedness) -> YConstrain {
        if let YSizedness::ZeroSized = y {
            return YConstrain::Same;
        }
        match self.ys.entry(id) {
            Entry::Occupied(mut x) => {
                if *x.get() < y {
                    x.insert(y);
                    YConstrain::Changed
                } else {
                    YConstrain::Same
                }
            },
            Entry::Vacant(x) => {
                x.insert(y);
                YConstrain::Changed
            },
        }
    }

    fn forward(&mut self, from: TypeId, to: TypeId) -> YConstrain {
        match self.ys.get(&from).cloned() {
            None => YConstrain::Same,
            Some(x) => self.insert(to, x),
        }
    }
}

impl<'ctx> Monotone for SizednessAnalysis<'ctx> {
    type Node = TypeId;
    type Extra = &'ctx BindgenContext;
    type Output = HashMap<TypeId, YSizedness>;

    fn new(ctx: &'ctx BindgenContext) -> SizednessAnalysis<'ctx> {
        let deps = gen_deps(ctx, Self::check_edge)
            .into_iter()
            .filter_map(|(i, i2)| {
                i.as_type_id(ctx)
                    .map(|i| (i, i2.into_iter().filter_map(|x| x.as_type_id(ctx)).collect::<Vec<_>>()))
            })
            .collect();
        let ys = HashMap::default();
        SizednessAnalysis { ctx, deps, ys }
    }

    fn initial_worklist(&self) -> Vec<TypeId> {
        self.ctx
            .allowed_items()
            .iter()
            .cloned()
            .filter_map(|x| x.as_type_id(self.ctx))
            .collect()
    }

    fn constrain(&mut self, id: TypeId) -> YConstrain {
        if let Some(YSizedness::NonZeroSized) = self.ys.get(&id).cloned() {
            return YConstrain::Same;
        }
        if id.has_vtable_ptr(self.ctx) {
            return self.insert(id, YSizedness::NonZeroSized);
        }
        let ty = self.ctx.resolve_type(id);
        if id.is_opaque(self.ctx, &()) {
            let y = ty.layout(self.ctx).map_or(YSizedness::ZeroSized, |x| {
                if x.size == 0 {
                    YSizedness::ZeroSized
                } else {
                    YSizedness::NonZeroSized
                }
            });
            return self.insert(id, y);
        }
        match *ty.kind() {
            TyKind::Void => self.insert(id, YSizedness::ZeroSized),
            TyKind::TypeParam => self.insert(id, YSizedness::DependsOnTypeParam),
            TyKind::Int(..)
            | TyKind::Float(..)
            | TyKind::Complex(..)
            | TyKind::Function(..)
            | TyKind::Enum(..)
            | TyKind::Reference(..)
            | TyKind::NullPtr
            | TyKind::Pointer(..) => self.insert(id, YSizedness::NonZeroSized),
            TyKind::TemplateAlias(t, _) | TyKind::Alias(t) | TyKind::BlockPointer(t) | TyKind::ResolvedTypeRef(t) => {
                self.forward(t, id)
            },
            TyKind::TemplateInstantiation(ref inst) => self.forward(inst.template_definition(), id),
            TyKind::Array(_, 0) => self.insert(id, YSizedness::ZeroSized),
            TyKind::Array(..) => self.insert(id, YSizedness::NonZeroSized),
            TyKind::Vector(..) => self.insert(id, YSizedness::NonZeroSized),
            TyKind::Comp(ref x) => {
                if !x.fields().is_empty() {
                    return self.insert(id, YSizedness::NonZeroSized);
                }
                let y = x
                    .base_members()
                    .iter()
                    .filter_map(|x| self.ys.get(&x.ty))
                    .fold(YSizedness::ZeroSized, |a, b| a.join(*b));

                self.insert(id, y)
            },
            TyKind::Opaque => {
                unreachable!("covered by the .is_opaque() check above")
            },
            TyKind::UnresolvedTypeRef(..) => {
                unreachable!("Should have been resolved after parsing!");
            },
        }
    }

    fn each_depending_on<F>(&self, id: TypeId, mut f: F)
    where
        F: FnMut(TypeId),
    {
        if let Some(es) = self.deps.get(&id) {
            for e in es {
                f(*e);
            }
        }
    }
}

impl<'ctx> From<SizednessAnalysis<'ctx>> for HashMap<TypeId, YSizedness> {
    fn from(x: SizednessAnalysis<'ctx>) -> Self {
        extra_assert!(x.ys.values().all(|x| { *x != YSizedness::ZeroSized }));
        x.ys
    }
}
