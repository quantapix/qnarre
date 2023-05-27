use std::fmt;

use super::{gen_deps, Monotone, YConstrain};
use crate::ir::analysis::has_vtable::HasVtable;
use crate::ir::comp::CompKind;
use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::derive::YDerive;
use crate::ir::function::FunctionSig;
use crate::ir::item::{IsOpaque, Item};
use crate::ir::layout::Layout;
use crate::ir::template::TemplateParameters;
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

    fn can_derive_fnptr(&self, f: &FunctionSig) -> YDerive {
        match (self, f.function_pointers_can_derive()) {
            (DeriveTrait::Copy, _) | (DeriveTrait::Default, _) | (_, true) => {
                trace!("    function pointer can derive {}", self);
                YDerive::Yes
            },
            (DeriveTrait::Debug, false) => {
                trace!("    function pointer cannot derive {}, but it may be implemented", self);
                YDerive::Manually
            },
            (_, false) => {
                trace!("    function pointer cannot derive {}", self);
                YDerive::No
            },
        }
    }

    fn can_derive_vec(&self) -> YDerive {
        match self {
            DeriveTrait::PartialEqOrPartialOrd => {
                trace!("    vectors cannot derive PartialOrd");
                YDerive::No
            },
            _ => {
                trace!("    vector can derive {}", self);
                YDerive::Yes
            },
        }
    }

    fn can_derive_ptr(&self) -> YDerive {
        match self {
            DeriveTrait::Default => {
                trace!("    pointer cannot derive Default");
                YDerive::No
            },
            _ => {
                trace!("    pointer can derive {}", self);
                YDerive::Yes
            },
        }
    }

    fn can_derive_simple(&self, k: &TypeKind) -> YDerive {
        match (self, k) {
            (DeriveTrait::Default, TypeKind::Void)
            | (DeriveTrait::Default, TypeKind::NullPtr)
            | (DeriveTrait::Default, TypeKind::Enum(..))
            | (DeriveTrait::Default, TypeKind::Reference(..))
            | (DeriveTrait::Default, TypeKind::TypeParam) => {
                trace!("    types that always cannot derive Default");
                YDerive::No
            },
            (DeriveTrait::Default, TypeKind::UnresolvedTypeRef(..)) => {
                unreachable!("Type with unresolved type ref can't reach derive default")
            },
            (DeriveTrait::Hash, TypeKind::Float(..)) | (DeriveTrait::Hash, TypeKind::Complex(..)) => {
                trace!("    float cannot derive Hash");
                YDerive::No
            },
            _ => {
                trace!("    simple type that can always derive {}", self);
                YDerive::Yes
            },
        }
    }
}

impl fmt::Display for DeriveTrait {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let y = match self {
            DeriveTrait::Copy => "Copy",
            DeriveTrait::Debug => "Debug",
            DeriveTrait::Default => "Default",
            DeriveTrait::Hash => "Hash",
            DeriveTrait::PartialEqOrPartialOrd => "PartialEq/PartialOrd",
        };
        y.fmt(f)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DeriveAnalysis<'ctx> {
    ctx: &'ctx BindgenContext,
    derive_trait: DeriveTrait,
    ys: HashMap<ItemId, YDerive>,
    deps: HashMap<ItemId, Vec<ItemId>>,
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

    fn constrain_type(&mut self, item: &Item, ty: &Type) -> YDerive {
        if !self.ctx.allowed_items().contains(&item.id()) {
            let can_derive = self.ctx.blocklisted_type_implements_trait(item, self.derive_trait);
            match can_derive {
                YDerive::Yes => trace!("    blocklisted type explicitly implements {}", self.derive_trait),
                YDerive::Manually => trace!(
                    "    blocklisted type requires manual implementation of {}",
                    self.derive_trait
                ),
                YDerive::No => trace!("    cannot derive {} for blocklisted type", self.derive_trait),
            }
            return can_derive;
        }
        if self.derive_trait.not_by_name(self.ctx, item) {
            trace!("    cannot derive {} for explicitly excluded type", self.derive_trait);
            return YDerive::No;
        }
        trace!("ty: {:?}", ty);
        if item.is_opaque(self.ctx, &()) {
            if !self.derive_trait.can_derive_union() && ty.is_union() && self.ctx.options().untagged_union {
                trace!("    cannot derive {} for Rust unions", self.derive_trait);
                return YDerive::No;
            }
            let layout_can_derive = ty
                .layout(self.ctx)
                .map_or(YDerive::Yes, |l| l.opaque().array_size_within_derive_limit(self.ctx));
            match layout_can_derive {
                YDerive::Yes => {
                    trace!("    we can trivially derive {} for the layout", self.derive_trait);
                },
                _ => {
                    trace!("    we cannot derive {} for the layout", self.derive_trait);
                },
            };
            return layout_can_derive;
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
                return self.derive_trait.can_derive_simple(ty.kind());
            },
            TypeKind::Pointer(inner) => {
                let inner_type = self.ctx.resolve_type(inner).canonical_type(self.ctx);
                if let TypeKind::Function(ref sig) = *inner_type.kind() {
                    self.derive_trait.can_derive_fnptr(sig)
                } else {
                    self.derive_trait.can_derive_ptr()
                }
            },
            TypeKind::Function(ref sig) => self.derive_trait.can_derive_fnptr(sig),

            TypeKind::Array(t, len) => {
                let inner_type = self.ys.get(&t.into()).cloned().unwrap_or_default();
                if inner_type != YDerive::Yes {
                    trace!(
                        "    arrays of T for which we cannot derive {} \
                         also cannot derive {}",
                        self.derive_trait,
                        self.derive_trait
                    );
                    return YDerive::No;
                }

                if len == 0 && !self.derive_trait.can_derive_incomplete_array() {
                    trace!("    cannot derive {} for incomplete arrays", self.derive_trait);
                    return YDerive::No;
                }

                if self.derive_trait.can_derive_large_array(self.ctx) {
                    trace!("    array can derive {}", self.derive_trait);
                    return YDerive::Yes;
                }

                if len > RUST_DERIVE_IN_ARRAY_LIMIT {
                    trace!(
                        "    array is too large to derive {}, but it may be implemented",
                        self.derive_trait
                    );
                    return YDerive::Manually;
                }
                trace!("    array is small enough to derive {}", self.derive_trait);
                YDerive::Yes
            },
            TypeKind::Vector(t, len) => {
                let inner_type = self.ys.get(&t.into()).cloned().unwrap_or_default();
                if inner_type != YDerive::Yes {
                    trace!(
                        "    vectors of T for which we cannot derive {} \
                         also cannot derive {}",
                        self.derive_trait,
                        self.derive_trait
                    );
                    return YDerive::No;
                }
                assert_ne!(len, 0, "vectors cannot have zero length");
                self.derive_trait.can_derive_vec()
            },

            TypeKind::Comp(ref info) => {
                assert!(
                    !info.has_non_type_template_params(),
                    "The early ty.is_opaque check should have handled this case"
                );

                if !self.derive_trait.can_derive_compound_forward_decl() && info.is_forward_declaration() {
                    trace!("    cannot derive {} for forward decls", self.derive_trait);
                    return YDerive::No;
                }

                if !self.derive_trait.can_derive_compound_with_destructor()
                    && self.ctx.lookup_has_destructor(item.id().expect_type_id(self.ctx))
                {
                    trace!("    comp has destructor which cannot derive {}", self.derive_trait);
                    return YDerive::No;
                }

                if info.kind() == CompKind::Union {
                    if self.derive_trait.can_derive_union() {
                        if self.ctx.options().untagged_union
                            && (!info.self_template_params(self.ctx).is_empty()
                                || !item.all_template_params(self.ctx).is_empty())
                        {
                            trace!(
                                "    cannot derive {} for Rust union because issue 36640",
                                self.derive_trait
                            );
                            return YDerive::No;
                        }
                    } else {
                        if self.ctx.options().untagged_union {
                            trace!("    cannot derive {} for Rust unions", self.derive_trait);
                            return YDerive::No;
                        }

                        let layout_can_derive = ty
                            .layout(self.ctx)
                            .map_or(YDerive::Yes, |l| l.opaque().array_size_within_derive_limit(self.ctx));
                        match layout_can_derive {
                            YDerive::Yes => {
                                trace!("    union layout can trivially derive {}", self.derive_trait);
                            },
                            _ => {
                                trace!("    union layout cannot derive {}", self.derive_trait);
                            },
                        };
                        return layout_can_derive;
                    }
                }

                if !self.derive_trait.can_derive_compound_with_vtable() && item.has_vtable(self.ctx) {
                    trace!("    cannot derive {} for comp with vtable", self.derive_trait);
                    return YDerive::No;
                }

                if !self.derive_trait.can_derive_large_array(self.ctx)
                    && info.has_too_large_bitfield_unit()
                    && !item.is_opaque(self.ctx, &())
                {
                    trace!(
                        "    cannot derive {} for comp with too large bitfield unit",
                        self.derive_trait
                    );
                    return YDerive::No;
                }

                let pred = self.derive_trait.check_edge_comp();
                self.constrain_join(item, pred)
            },

            TypeKind::ResolvedTypeRef(..)
            | TypeKind::TemplateAlias(..)
            | TypeKind::Alias(..)
            | TypeKind::BlockPointer(..) => {
                let pred = self.derive_trait.check_edge_typeref();
                self.constrain_join(item, pred)
            },

            TypeKind::TemplateInstantiation(..) => {
                let pred = self.derive_trait.check_edge_tmpl_inst();
                self.constrain_join(item, pred)
            },

            TypeKind::Opaque => unreachable!("The early ty.is_opaque check should have handled this case"),
        }
    }

    fn constrain_join(&mut self, item: &Item, check_edge: EdgePred) -> YDerive {
        let mut candidate = None;

        item.trace(
            self.ctx,
            &mut |sub_id, edge_kind| {
                if sub_id == item.id() || !check_edge(edge_kind) {
                    return;
                }
                let can_derive = self.ys.get(&sub_id).cloned().unwrap_or_default();
                match can_derive {
                    YDerive::Yes => trace!("    member {:?} can derive {}", sub_id, self.derive_trait),
                    YDerive::Manually => trace!(
                        "    member {:?} cannot derive {}, but it may be implemented",
                        sub_id,
                        self.derive_trait
                    ),
                    YDerive::No => trace!("    member {:?} cannot derive {}", sub_id, self.derive_trait),
                }
                *candidate.get_or_insert(YDerive::Yes) |= can_derive;
            },
            &(),
        );
        if candidate.is_none() {
            trace!("    can derive {} because there are no members", self.derive_trait);
        }
        candidate.unwrap_or_default()
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
            derive_trait,
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
        trace!("constrain: {:?}", id);
        if let Some(YDerive::No) = self.ys.get(&id).cloned() {
            trace!("    already know it cannot derive {}", self.derive_trait);
            return YConstrain::Same;
        }
        let item = self.ctx.resolve_item(id);
        let can_derive = match item.as_type() {
            Some(ty) => {
                let mut can_derive = self.constrain_type(item, ty);
                if let YDerive::Yes = can_derive {
                    let is_reached_limit = |l: Layout| l.align > RUST_DERIVE_IN_ARRAY_LIMIT;
                    if !self.derive_trait.can_derive_large_array(self.ctx)
                        && ty.layout(self.ctx).map_or(false, is_reached_limit)
                    {
                        can_derive = YDerive::Manually;
                    }
                }
                can_derive
            },
            None => self.constrain_join(item, check_edge_default),
        };
        self.insert(id, can_derive)
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
