use std::fmt;

use super::{gen_deps, Monotone, YConstrain};
use crate::ir::analysis::has_vtable::HasVtable;
use crate::ir::comp::CompKind;
use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::derive::YDerive;
use crate::ir::function::FnSig;
use crate::ir::item::{IsOpaque, Item};
use crate::ir::layout::Layout;
use crate::ir::template::TemplParams;
use crate::ir::traversal::{EdgeKind, Trace};
use crate::ir::ty::RUST_DERIVE_IN_ARRAY_LIMIT;
use crate::ir::ty::{Type, TypeKind};
use crate::{Entry, HashMap, HashSet};

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum DeriveTrait {
    Copy,
    Debug,
    Default,
    Hash,
    PartialEqOrPartialOrd,
}

type EdgePred = fn(EdgeKind) -> bool;

fn check_edge_default(k: EdgeKind) -> bool {
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
        | EdgeKind::Method
        | EdgeKind::Generic => false,
    }
}

impl DeriveTrait {
    fn not_by_name(&self, ctx: &BindgenContext, i: &Item) -> bool {
        match self {
            DeriveTrait::Copy => ctx.no_copy_by_name(i),
            DeriveTrait::Debug => ctx.no_debug_by_name(i),
            DeriveTrait::Default => ctx.no_default_by_name(i),
            DeriveTrait::Hash => ctx.no_hash_by_name(i),
            DeriveTrait::PartialEqOrPartialOrd => ctx.no_partialeq_by_name(i),
        }
    }

    fn check_edge_comp(&self) -> EdgePred {
        match self {
            DeriveTrait::PartialEqOrPartialOrd => check_edge_default,
            _ => |k| matches!(k, EdgeKind::BaseMember | EdgeKind::Field),
        }
    }

    fn check_edge_typeref(&self) -> EdgePred {
        match self {
            DeriveTrait::PartialEqOrPartialOrd => check_edge_default,
            _ => |k| k == EdgeKind::TypeReference,
        }
    }

    fn check_edge_tmpl_inst(&self) -> EdgePred {
        match self {
            DeriveTrait::PartialEqOrPartialOrd => check_edge_default,
            _ => |k| matches!(k, EdgeKind::TemplateArgument | EdgeKind::TemplateDeclaration),
        }
    }

    fn can_derive_large_array(&self, ctx: &BindgenContext) -> bool {
        !matches!(self, DeriveTrait::Default)
    }

    fn can_derive_union(&self) -> bool {
        matches!(self, DeriveTrait::Copy)
    }

    fn can_derive_compound_with_destructor(&self) -> bool {
        !matches!(self, DeriveTrait::Copy)
    }

    fn can_derive_compound_with_vtable(&self) -> bool {
        !matches!(self, DeriveTrait::Default)
    }

    fn can_derive_compound_forward_decl(&self) -> bool {
        matches!(self, DeriveTrait::Copy | DeriveTrait::Debug)
    }

    fn can_derive_incomplete_array(&self) -> bool {
        !matches!(
            self,
            DeriveTrait::Copy | DeriveTrait::Hash | DeriveTrait::PartialEqOrPartialOrd
        )
    }

    fn can_derive_fnptr(&self, f: &FnSig) -> YDerive {
        match (self, f.function_pointers_can_derive()) {
            (DeriveTrait::Copy, _) | (DeriveTrait::Default, _) | (_, true) => YDerive::Yes,
            (DeriveTrait::Debug, false) => YDerive::Manually,
            (_, false) => YDerive::No,
        }
    }

    fn can_derive_vec(&self) -> YDerive {
        match self {
            DeriveTrait::PartialEqOrPartialOrd => YDerive::No,
            _ => YDerive::Yes,
        }
    }

    fn can_derive_ptr(&self) -> YDerive {
        match self {
            DeriveTrait::Default => YDerive::No,
            _ => YDerive::Yes,
        }
    }

    fn can_derive_simple(&self, k: &TypeKind) -> YDerive {
        match (self, k) {
            (DeriveTrait::Default, TypeKind::Void)
            | (DeriveTrait::Default, TypeKind::NullPtr)
            | (DeriveTrait::Default, TypeKind::Enum(..))
            | (DeriveTrait::Default, TypeKind::Reference(..))
            | (DeriveTrait::Default, TypeKind::TypeParam) => YDerive::No,
            (DeriveTrait::Default, TypeKind::UnresolvedTypeRef(..)) => {
                unreachable!("Type with unresolved type ref can't reach derive default")
            },
            (DeriveTrait::Hash, TypeKind::Float(..)) | (DeriveTrait::Hash, TypeKind::Complex(..)) => YDerive::No,
            _ => YDerive::Yes,
        }
    }
}

impl fmt::Display for DeriveTrait {
    fn fmt(&self, x: &mut fmt::Formatter) -> fmt::Result {
        let y = match self {
            DeriveTrait::Copy => "Copy",
            DeriveTrait::Debug => "Debug",
            DeriveTrait::Default => "Default",
            DeriveTrait::Hash => "Hash",
            DeriveTrait::PartialEqOrPartialOrd => "PartialEq/PartialOrd",
        };
        y.fmt(x)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DeriveAnalysis<'ctx> {
    ctx: &'ctx BindgenContext,
    deps: HashMap<ItemId, Vec<ItemId>>,
    ys: HashMap<ItemId, YDerive>,
    derive: DeriveTrait,
}

impl<'ctx> DeriveAnalysis<'ctx> {
    fn insert<Id: Into<ItemId>>(&mut self, id: Id, y: YDerive) -> YConstrain {
        if let YDerive::Yes = y {
            return YConstrain::Same;
        }
        let id = id.into();
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

    fn constrain_type(&mut self, i: &Item, ty: &Type) -> YDerive {
        if !self.ctx.allowed_items().contains(&i.id()) {
            let y = self.ctx.blocklisted_type_implements_trait(i, self.derive);
            return y;
        }
        if self.derive.not_by_name(self.ctx, i) {
            return YDerive::No;
        }
        if i.is_opaque(self.ctx, &()) {
            if !self.derive.can_derive_union() && ty.is_union() && self.ctx.opts().untagged_union {
                return YDerive::No;
            }
            let y = ty
                .layout(self.ctx)
                .map_or(YDerive::Yes, |x| x.opaque().array_size_within_derive_limit(self.ctx));
            return y;
        }
        match *ty.kind() {
            TypeKind::Void
            | TypeKind::NullPtr
            | TypeKind::Int(..)
            | TypeKind::Complex(..)
            | TypeKind::Float(..)
            | TypeKind::Enum(..)
            | TypeKind::TypeParam
            | TypeKind::UnresolvedTypeRef(..)
            | TypeKind::Reference(..) => {
                return self.derive.can_derive_simple(ty.kind());
            },
            TypeKind::Pointer(x) => {
                let ty2 = self.ctx.resolve_type(x).canonical_type(self.ctx);
                if let TypeKind::Function(ref sig) = *ty2.kind() {
                    self.derive.can_derive_fnptr(sig)
                } else {
                    self.derive.can_derive_ptr()
                }
            },
            TypeKind::Function(ref sig) => self.derive.can_derive_fnptr(sig),
            TypeKind::Array(t, len) => {
                let ty2 = self.ys.get(&t.into()).cloned().unwrap_or_default();
                if ty2 != YDerive::Yes {
                    return YDerive::No;
                }
                if len == 0 && !self.derive.can_derive_incomplete_array() {
                    return YDerive::No;
                }
                if self.derive.can_derive_large_array(self.ctx) {
                    return YDerive::Yes;
                }
                if len > RUST_DERIVE_IN_ARRAY_LIMIT {
                    return YDerive::Manually;
                }
                YDerive::Yes
            },
            TypeKind::Vector(t, len) => {
                let ty2 = self.ys.get(&t.into()).cloned().unwrap_or_default();
                if ty2 != YDerive::Yes {
                    return YDerive::No;
                }
                self.derive.can_derive_vec()
            },
            TypeKind::Comp(ref x) => {
                assert!(!x.has_non_type_template_params());
                if !self.derive.can_derive_compound_forward_decl() && x.is_forward_declaration() {
                    return YDerive::No;
                }
                if !self.derive.can_derive_compound_with_destructor()
                    && self.ctx.lookup_has_destructor(i.id().expect_type_id(self.ctx))
                {
                    return YDerive::No;
                }
                if x.kind() == CompKind::Union {
                    if self.derive.can_derive_union() {
                        if self.ctx.opts().untagged_union
                            && (!x.self_template_params(self.ctx).is_empty()
                                || !i.all_template_params(self.ctx).is_empty())
                        {
                            return YDerive::No;
                        }
                    } else {
                        if self.ctx.opts().untagged_union {
                            return YDerive::No;
                        }
                        let y = ty
                            .layout(self.ctx)
                            .map_or(YDerive::Yes, |l| l.opaque().array_size_within_derive_limit(self.ctx));
                        return y;
                    }
                }
                if !self.derive.can_derive_compound_with_vtable() && i.has_vtable(self.ctx) {
                    return YDerive::No;
                }
                if !self.derive.can_derive_large_array(self.ctx)
                    && x.has_too_large_bitfield_unit()
                    && !i.is_opaque(self.ctx, &())
                {
                    return YDerive::No;
                }
                self.constrain_join(i, self.derive.check_edge_comp())
            },
            TypeKind::ResolvedTypeRef(..)
            | TypeKind::TemplateAlias(..)
            | TypeKind::Alias(..)
            | TypeKind::BlockPointer(..) => self.constrain_join(i, self.derive.check_edge_typeref()),
            TypeKind::TemplateInstantiation(..) => self.constrain_join(i, self.derive.check_edge_tmpl_inst()),
            TypeKind::Opaque => unreachable!("The early ty.is_opaque check should have handled this case"),
        }
    }

    fn constrain_join(&mut self, i: &Item, f: EdgePred) -> YDerive {
        let mut y = None;
        i.trace(
            self.ctx,
            &mut |i2, kind| {
                if i2 == i.id() || !f(kind) {
                    return;
                }
                let y2 = self.ys.get(&i2).cloned().unwrap_or_default();
                *y.get_or_insert(YDerive::Yes) |= y2;
            },
            &(),
        );
        y.unwrap_or_default()
    }
}

impl<'ctx> Monotone for DeriveAnalysis<'ctx> {
    type Node = ItemId;
    type Extra = (&'ctx BindgenContext, DeriveTrait);
    type Output = HashMap<ItemId, YDerive>;

    fn new((ctx, derive_trait): (&'ctx BindgenContext, DeriveTrait)) -> DeriveAnalysis<'ctx> {
        let ys = HashMap::default();
        let deps = gen_deps(ctx, check_edge_default);
        DeriveAnalysis {
            ctx,
            derive: derive_trait,
            ys,
            deps,
        }
    }

    fn initial_worklist(&self) -> Vec<ItemId> {
        self.ctx
            .allowed_items()
            .iter()
            .cloned()
            .flat_map(|i| {
                let mut ys = vec![i];
                i.trace(
                    self.ctx,
                    &mut |i2, _| {
                        ys.push(i2);
                    },
                    &(),
                );
                ys
            })
            .collect()
    }

    fn constrain(&mut self, id: ItemId) -> YConstrain {
        if let Some(YDerive::No) = self.ys.get(&id).cloned() {
            return YConstrain::Same;
        }
        let i = self.ctx.resolve_item(id);
        let y = match i.as_type() {
            Some(ty) => {
                let mut y = self.constrain_type(i, ty);
                if let YDerive::Yes = y {
                    let at_limit = |x: Layout| x.align > RUST_DERIVE_IN_ARRAY_LIMIT;
                    if !self.derive.can_derive_large_array(self.ctx) && ty.layout(self.ctx).map_or(false, at_limit) {
                        y = YDerive::Manually;
                    }
                }
                y
            },
            None => self.constrain_join(i, check_edge_default),
        };
        self.insert(id, y)
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

impl<'ctx> From<DeriveAnalysis<'ctx>> for HashMap<ItemId, YDerive> {
    fn from(x: DeriveAnalysis<'ctx>) -> Self {
        extra_assert!(x.ys.values().all(|v| *v != YDerive::Yes));
        x.ys
    }
}

pub(crate) fn as_cannot_derive_set(xs: HashMap<ItemId, YDerive>) -> HashSet<ItemId> {
    xs.into_iter()
        .filter_map(|(k, v)| if v != YDerive::Yes { Some(k) } else { None })
        .collect()
}
