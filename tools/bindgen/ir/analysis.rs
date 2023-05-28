use crate::ir::context::{BindgenContext, ItemId};
use crate::ir::traversal::{EdgeKind, Trace};
use crate::HashMap;
use std::fmt;
use std::ops;

pub(crate) trait Monotone: Sized + fmt::Debug {
    type Node: Copy;
    type Extra: Sized;
    type Output: From<Self> + fmt::Debug;
    fn new(x: Self::Extra) -> Self;
    fn initial_worklist(&self) -> Vec<Self::Node>;
    fn constrain(&mut self, n: Self::Node) -> YConstrain;
    fn each_depending_on<F>(&self, n: Self::Node, f: F)
    where
        F: FnMut(Self::Node);
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum YConstrain {
    Changed,
    Same,
}

impl Default for YConstrain {
    fn default() -> Self {
        YConstrain::Same
    }
}

impl ops::BitOr for YConstrain {
    type Output = Self;
    fn bitor(self, rhs: YConstrain) -> Self::Output {
        if self == YConstrain::Changed || rhs == YConstrain::Changed {
            YConstrain::Changed
        } else {
            YConstrain::Same
        }
    }
}

impl ops::BitOrAssign for YConstrain {
    fn bitor_assign(&mut self, rhs: YConstrain) {
        *self = *self | rhs;
    }
}

pub(crate) fn analyze<T>(x: T::Extra) -> T::Output
where
    T: Monotone,
{
    let mut y = T::new(x);
    let mut ns = y.initial_worklist();
    while let Some(n) = ns.pop() {
        if let YConstrain::Changed = y.constrain(n) {
            y.each_depending_on(n, |x| {
                ns.push(x);
            });
        }
    }
    y.into()
}

pub(crate) fn gen_deps<F>(ctx: &BindgenContext, f: F) -> HashMap<ItemId, Vec<ItemId>>
where
    F: Fn(EdgeKind) -> bool,
{
    let mut ys = HashMap::default();
    for &i in ctx.allowed_items() {
        ys.entry(i).or_insert_with(Vec::new);
        {
            i.trace(
                ctx,
                &mut |i2: ItemId, kind| {
                    if ctx.allowed_items().contains(&i2) && f(kind) {
                        ys.entry(i2).or_insert_with(Vec::new).push(i);
                    }
                },
                &(),
            );
        }
    }
    ys
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum DeriveTrait {
    Copy,
    Debug,
    Default,
    Hash,
    PartialEqOrPartialOrd,
}

pub(crate) mod derive {
    use std::fmt;

    use super::{gen_deps, DeriveTrait, HasVtable, Monotone, YConstrain};
    use crate::ir::comp::CompKind;
    use crate::ir::context::{BindgenContext, ItemId};
    use crate::ir::derive::YDerive;
    use crate::ir::function::FnSig;
    use crate::ir::item::{IsOpaque, Item};
    use crate::ir::layout::Layout;
    use crate::ir::template::TemplParams;
    use crate::ir::traversal::{EdgeKind, Trace};
    use crate::ir::ty::RUST_DERIVE_IN_ARRAY_LIMIT;
    use crate::ir::ty::{TyKind, Type};
    use crate::{Entry, HashMap, HashSet};

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

        fn can_derive_simple(&self, k: &TyKind) -> YDerive {
            match (self, k) {
                (DeriveTrait::Default, TyKind::Void)
                | (DeriveTrait::Default, TyKind::NullPtr)
                | (DeriveTrait::Default, TyKind::Enum(..))
                | (DeriveTrait::Default, TyKind::Reference(..))
                | (DeriveTrait::Default, TyKind::TypeParam) => YDerive::No,
                (DeriveTrait::Default, TyKind::UnresolvedTypeRef(..)) => {
                    unreachable!("Type with unresolved type ref can't reach derive default")
                },
                (DeriveTrait::Hash, TyKind::Float(..)) | (DeriveTrait::Hash, TyKind::Complex(..)) => YDerive::No,
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
    pub struct Analysis<'ctx> {
        ctx: &'ctx BindgenContext,
        deps: HashMap<ItemId, Vec<ItemId>>,
        ys: HashMap<ItemId, YDerive>,
        derive: DeriveTrait,
    }

    impl<'ctx> Analysis<'ctx> {
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
                TyKind::Void
                | TyKind::NullPtr
                | TyKind::Int(..)
                | TyKind::Complex(..)
                | TyKind::Float(..)
                | TyKind::Enum(..)
                | TyKind::TypeParam
                | TyKind::UnresolvedTypeRef(..)
                | TyKind::Reference(..) => {
                    return self.derive.can_derive_simple(ty.kind());
                },
                TyKind::Pointer(x) => {
                    let ty2 = self.ctx.resolve_type(x).canonical_type(self.ctx);
                    if let TyKind::Function(ref sig) = *ty2.kind() {
                        self.derive.can_derive_fnptr(sig)
                    } else {
                        self.derive.can_derive_ptr()
                    }
                },
                TyKind::Function(ref sig) => self.derive.can_derive_fnptr(sig),
                TyKind::Array(t, len) => {
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
                TyKind::Vector(t, len) => {
                    let ty2 = self.ys.get(&t.into()).cloned().unwrap_or_default();
                    if ty2 != YDerive::Yes {
                        return YDerive::No;
                    }
                    self.derive.can_derive_vec()
                },
                TyKind::Comp(ref x) => {
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
                TyKind::ResolvedTypeRef(..)
                | TyKind::TemplateAlias(..)
                | TyKind::Alias(..)
                | TyKind::BlockPointer(..) => self.constrain_join(i, self.derive.check_edge_typeref()),
                TyKind::TemplateInstantiation(..) => self.constrain_join(i, self.derive.check_edge_tmpl_inst()),
                TyKind::Opaque => unreachable!("The early ty.is_opaque check should have handled this case"),
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

    impl<'ctx> Monotone for Analysis<'ctx> {
        type Node = ItemId;
        type Extra = (&'ctx BindgenContext, DeriveTrait);
        type Output = HashMap<ItemId, YDerive>;

        fn new((ctx, derive): (&'ctx BindgenContext, DeriveTrait)) -> Analysis<'ctx> {
            let ys = HashMap::default();
            let deps = gen_deps(ctx, check_edge_default);
            Analysis { ctx, derive, ys, deps }
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
                        if !self.derive.can_derive_large_array(self.ctx) && ty.layout(self.ctx).map_or(false, at_limit)
                        {
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

    impl<'ctx> From<Analysis<'ctx>> for HashMap<ItemId, YDerive> {
        fn from(x: Analysis<'ctx>) -> Self {
            extra_assert!(x.ys.values().all(|v| *v != YDerive::Yes));
            x.ys
        }
    }

    pub fn as_cannot_derive_set(xs: HashMap<ItemId, YDerive>) -> HashSet<ItemId> {
        xs.into_iter()
            .filter_map(|(k, v)| if v != YDerive::Yes { Some(k) } else { None })
            .collect()
    }
}
pub(crate) use self::derive::as_cannot_derive_set;

pub(crate) mod has_destructor {
    use super::{gen_deps, Monotone, YConstrain};
    use crate::ir::comp::{CompKind, Field, FieldMethods};
    use crate::ir::context::{BindgenContext, ItemId};
    use crate::ir::traversal::EdgeKind;
    use crate::ir::ty::TyKind;
    use crate::{HashMap, HashSet};

    #[derive(Debug, Clone)]
    pub struct Analysis<'ctx> {
        ctx: &'ctx BindgenContext,
        deps: HashMap<ItemId, Vec<ItemId>>,
        ys: HashSet<ItemId>,
    }

    impl<'ctx> Analysis<'ctx> {
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

    impl<'ctx> Monotone for Analysis<'ctx> {
        type Node = ItemId;
        type Extra = &'ctx BindgenContext;
        type Output = HashSet<ItemId>;
        fn new(ctx: &'ctx BindgenContext) -> Self {
            let ys = HashSet::default();
            let deps = gen_deps(ctx, Self::check_edge);
            Analysis { ctx, ys, deps }
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
                TyKind::TemplateAlias(t, _) | TyKind::Alias(t) | TyKind::ResolvedTypeRef(t) => {
                    if self.ys.contains(&t.into()) {
                        self.insert(id)
                    } else {
                        YConstrain::Same
                    }
                },
                TyKind::Comp(ref x) => {
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
                TyKind::TemplateInstantiation(ref t) => {
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

    impl<'ctx> From<Analysis<'ctx>> for HashSet<ItemId> {
        fn from(x: Analysis<'ctx>) -> Self {
            x.ys
        }
    }
}

pub(crate) mod has_float {
    use super::{gen_deps, Monotone, YConstrain};
    use crate::ir::comp::Field;
    use crate::ir::comp::FieldMethods;
    use crate::ir::context::{BindgenContext, ItemId};
    use crate::ir::traversal::EdgeKind;
    use crate::ir::ty::TyKind;
    use crate::{HashMap, HashSet};

    #[derive(Debug, Clone)]
    pub struct Analysis<'ctx> {
        ctx: &'ctx BindgenContext,
        ys: HashSet<ItemId>,
        deps: HashMap<ItemId, Vec<ItemId>>,
    }

    impl<'ctx> Analysis<'ctx> {
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

    impl<'ctx> Monotone for Analysis<'ctx> {
        type Node = ItemId;
        type Extra = &'ctx BindgenContext;
        type Output = HashSet<ItemId>;

        fn new(ctx: &'ctx BindgenContext) -> Analysis<'ctx> {
            let ys = HashSet::default();
            let deps = gen_deps(ctx, Self::check_edge);
            Analysis { ctx, ys, deps }
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
                | TyKind::Function(..)
                | TyKind::Enum(..)
                | TyKind::Reference(..)
                | TyKind::TypeParam
                | TyKind::Opaque
                | TyKind::Pointer(..)
                | TyKind::UnresolvedTypeRef(..) => YConstrain::Same,
                TyKind::Float(..) | TyKind::Complex(..) => self.insert(id),
                TyKind::Array(t, _) => {
                    if self.ys.contains(&t.into()) {
                        return self.insert(id);
                    }
                    YConstrain::Same
                },
                TyKind::Vector(t, _) => {
                    if self.ys.contains(&t.into()) {
                        return self.insert(id);
                    }
                    YConstrain::Same
                },
                TyKind::ResolvedTypeRef(t)
                | TyKind::TemplateAlias(t, _)
                | TyKind::Alias(t)
                | TyKind::BlockPointer(t) => {
                    if self.ys.contains(&t.into()) {
                        self.insert(id)
                    } else {
                        YConstrain::Same
                    }
                },
                TyKind::Comp(ref x) => {
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

    impl<'ctx> From<Analysis<'ctx>> for HashSet<ItemId> {
        fn from(x: Analysis<'ctx>) -> Self {
            x.ys
        }
    }
}

pub(crate) mod has_ty_param {
    use super::{gen_deps, Monotone, YConstrain};
    use crate::ir::comp::Field;
    use crate::ir::comp::FieldMethods;
    use crate::ir::context::{BindgenContext, ItemId};
    use crate::ir::traversal::EdgeKind;
    use crate::ir::ty::TyKind;
    use crate::{HashMap, HashSet};

    #[derive(Debug, Clone)]
    pub struct Analysis<'ctx> {
        ctx: &'ctx BindgenContext,
        ys: HashSet<ItemId>,
        deps: HashMap<ItemId, Vec<ItemId>>,
    }

    impl<'ctx> Analysis<'ctx> {
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

    impl<'ctx> Monotone for Analysis<'ctx> {
        type Node = ItemId;
        type Extra = &'ctx BindgenContext;
        type Output = HashSet<ItemId>;

        fn new(ctx: &'ctx BindgenContext) -> Analysis<'ctx> {
            let ys = HashSet::default();
            let deps = gen_deps(ctx, Self::check_edge);
            Analysis { ctx, ys, deps }
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
                TyKind::ResolvedTypeRef(t)
                | TyKind::TemplateAlias(t, _)
                | TyKind::Alias(t)
                | TyKind::BlockPointer(t) => {
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

    impl<'ctx> From<Analysis<'ctx>> for HashSet<ItemId> {
        fn from(x: Analysis<'ctx>) -> Self {
            x.ys
        }
    }
}

pub(crate) trait HasVtable {
    fn has_vtable(&self, ctx: &BindgenContext) -> bool;
    fn has_vtable_ptr(&self, ctx: &BindgenContext) -> bool;
}

pub(crate) mod has_vtable {
    use super::{gen_deps, Monotone, YConstrain};
    use crate::ir::context::{BindgenContext, ItemId};
    use crate::ir::traversal::EdgeKind;
    use crate::ir::ty::TyKind;
    use crate::{Entry, HashMap};
    use std::cmp;
    use std::ops;

    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub enum Result {
        No,
        SelfHasVtable,
        BaseHasVtable,
    }

    impl Default for Result {
        fn default() -> Self {
            Result::No
        }
    }

    impl Result {
        pub fn join(self, rhs: Self) -> Self {
            cmp::max(self, rhs)
        }
    }

    impl ops::BitOr for Result {
        type Output = Self;
        fn bitor(self, rhs: Result) -> Self::Output {
            self.join(rhs)
        }
    }

    impl ops::BitOrAssign for Result {
        fn bitor_assign(&mut self, rhs: Result) {
            *self = self.join(rhs)
        }
    }

    #[derive(Debug, Clone)]
    pub struct Analysis<'ctx> {
        ctx: &'ctx BindgenContext,
        ys: HashMap<ItemId, Result>,
        deps: HashMap<ItemId, Vec<ItemId>>,
    }

    impl<'ctx> Analysis<'ctx> {
        fn check_edge(k: EdgeKind) -> bool {
            matches!(
                k,
                EdgeKind::TypeReference | EdgeKind::BaseMember | EdgeKind::TemplateDeclaration
            )
        }

        fn insert<Id: Into<ItemId>>(&mut self, id: Id, y: Result) -> YConstrain {
            if let Result::No = y {
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

        fn forward<Id1, Id2>(&mut self, from: Id1, to: Id2) -> YConstrain
        where
            Id1: Into<ItemId>,
            Id2: Into<ItemId>,
        {
            let from = from.into();
            let to = to.into();
            match self.ys.get(&from).cloned() {
                None => YConstrain::Same,
                Some(x) => self.insert(to, x),
            }
        }
    }

    impl<'ctx> Monotone for Analysis<'ctx> {
        type Node = ItemId;
        type Extra = &'ctx BindgenContext;
        type Output = HashMap<ItemId, Result>;

        fn new(ctx: &'ctx BindgenContext) -> Analysis<'ctx> {
            let ys = HashMap::default();
            let deps = gen_deps(ctx, Self::check_edge);
            Analysis { ctx, ys, deps }
        }

        fn initial_worklist(&self) -> Vec<ItemId> {
            self.ctx.allowed_items().iter().cloned().collect()
        }

        fn constrain(&mut self, id: ItemId) -> YConstrain {
            let i = self.ctx.resolve_item(id);
            let ty = match i.as_type() {
                None => return YConstrain::Same,
                Some(ty) => ty,
            };
            match *ty.kind() {
                TyKind::TemplateAlias(t, _) | TyKind::Alias(t) | TyKind::ResolvedTypeRef(t) | TyKind::Reference(t) => {
                    self.forward(t, id)
                },
                TyKind::Comp(ref info) => {
                    let mut y = Result::No;
                    if info.has_own_virtual_method() {
                        y |= Result::SelfHasVtable;
                    }
                    let has_vtable = info.base_members().iter().any(|x| self.ys.contains_key(&x.ty.into()));
                    if has_vtable {
                        y |= Result::BaseHasVtable;
                    }
                    self.insert(id, y)
                },
                TyKind::TemplateInstantiation(ref x) => self.forward(x.template_definition(), id),
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

    impl<'ctx> From<Analysis<'ctx>> for HashMap<ItemId, Result> {
        fn from(x: Analysis<'ctx>) -> Self {
            extra_assert!(x.ys.values().all(|x| { *x != Result::No }));
            x.ys
        }
    }
}

pub(crate) trait Sizedness {
    fn sizedness(&self, ctx: &BindgenContext) -> sizedness::Result;
    fn is_zero_sized(&self, ctx: &BindgenContext) -> bool {
        self.sizedness(ctx) == sizedness::Result::ZeroSized
    }
}

pub(crate) mod sizedness {
    use super::{gen_deps, HasVtable, Monotone, YConstrain};
    use crate::ir::context::{BindgenContext, TypeId};
    use crate::ir::item::IsOpaque;
    use crate::ir::traversal::EdgeKind;
    use crate::ir::ty::TyKind;
    use crate::{Entry, HashMap};
    use std::{cmp, ops};

    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub enum Result {
        ZeroSized,
        DependsOnTypeParam,
        NonZeroSized,
    }

    impl Default for Result {
        fn default() -> Self {
            Result::ZeroSized
        }
    }

    impl Result {
        pub fn join(self, rhs: Self) -> Self {
            cmp::max(self, rhs)
        }
    }

    impl ops::BitOr for Result {
        type Output = Self;
        fn bitor(self, rhs: Result) -> Self::Output {
            self.join(rhs)
        }
    }

    impl ops::BitOrAssign for Result {
        fn bitor_assign(&mut self, rhs: Result) {
            *self = self.join(rhs)
        }
    }

    #[derive(Debug)]
    pub struct Analysis<'ctx> {
        ctx: &'ctx BindgenContext,
        deps: HashMap<TypeId, Vec<TypeId>>,
        ys: HashMap<TypeId, Result>,
    }

    impl<'ctx> Analysis<'ctx> {
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

        fn insert(&mut self, id: TypeId, y: Result) -> YConstrain {
            if let Result::ZeroSized = y {
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

    impl<'ctx> Monotone for Analysis<'ctx> {
        type Node = TypeId;
        type Extra = &'ctx BindgenContext;
        type Output = HashMap<TypeId, Result>;

        fn new(ctx: &'ctx BindgenContext) -> Analysis<'ctx> {
            let deps = gen_deps(ctx, Self::check_edge)
                .into_iter()
                .filter_map(|(i, i2)| {
                    i.as_type_id(ctx)
                        .map(|i| (i, i2.into_iter().filter_map(|x| x.as_type_id(ctx)).collect::<Vec<_>>()))
                })
                .collect();
            let ys = HashMap::default();
            Analysis { ctx, deps, ys }
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
            if let Some(Result::NonZeroSized) = self.ys.get(&id).cloned() {
                return YConstrain::Same;
            }
            if id.has_vtable_ptr(self.ctx) {
                return self.insert(id, Result::NonZeroSized);
            }
            let ty = self.ctx.resolve_type(id);
            if id.is_opaque(self.ctx, &()) {
                let y = ty.layout(self.ctx).map_or(Result::ZeroSized, |x| {
                    if x.size == 0 {
                        Result::ZeroSized
                    } else {
                        Result::NonZeroSized
                    }
                });
                return self.insert(id, y);
            }
            match *ty.kind() {
                TyKind::Void => self.insert(id, Result::ZeroSized),
                TyKind::TypeParam => self.insert(id, Result::DependsOnTypeParam),
                TyKind::Int(..)
                | TyKind::Float(..)
                | TyKind::Complex(..)
                | TyKind::Function(..)
                | TyKind::Enum(..)
                | TyKind::Reference(..)
                | TyKind::NullPtr
                | TyKind::Pointer(..) => self.insert(id, Result::NonZeroSized),
                TyKind::TemplateAlias(t, _)
                | TyKind::Alias(t)
                | TyKind::BlockPointer(t)
                | TyKind::ResolvedTypeRef(t) => self.forward(t, id),
                TyKind::TemplateInstantiation(ref x) => self.forward(x.template_definition(), id),
                TyKind::Array(_, 0) => self.insert(id, Result::ZeroSized),
                TyKind::Array(..) => self.insert(id, Result::NonZeroSized),
                TyKind::Vector(..) => self.insert(id, Result::NonZeroSized),
                TyKind::Comp(ref x) => {
                    if !x.fields().is_empty() {
                        return self.insert(id, Result::NonZeroSized);
                    }
                    let y = x
                        .base_members()
                        .iter()
                        .filter_map(|x| self.ys.get(&x.ty))
                        .fold(Result::ZeroSized, |a, b| a.join(*b));

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

    impl<'ctx> From<Analysis<'ctx>> for HashMap<TypeId, Result> {
        fn from(x: Analysis<'ctx>) -> Self {
            extra_assert!(x.ys.values().all(|x| { *x != Result::ZeroSized }));
            x.ys
        }
    }
}

pub(crate) mod used_templ_param {
    use super::{Monotone, YConstrain};
    use crate::ir::context::{BindgenContext, ItemId};
    use crate::ir::item::{Item, ItemSet};
    use crate::ir::template::{TemplInstantiation, TemplParams};
    use crate::ir::traversal::{EdgeKind, Trace};
    use crate::ir::ty::TyKind;
    use crate::{HashMap, HashSet};

    #[derive(Debug, Clone)]
    pub struct Analysis<'ctx> {
        ctx: &'ctx BindgenContext,
        ys: HashMap<ItemId, Option<ItemSet>>,
        deps: HashMap<ItemId, Vec<ItemId>>,
        alloweds: HashSet<ItemId>,
    }

    impl<'ctx> Analysis<'ctx> {
        fn check_edge(k: EdgeKind) -> bool {
            match k {
                EdgeKind::TemplateArgument
                | EdgeKind::BaseMember
                | EdgeKind::Field
                | EdgeKind::Constructor
                | EdgeKind::Destructor
                | EdgeKind::VarType
                | EdgeKind::FunctionReturn
                | EdgeKind::FunctionParameter
                | EdgeKind::TypeReference => true,
                EdgeKind::InnerVar | EdgeKind::InnerType => false,
                EdgeKind::Method => false,
                EdgeKind::TemplateDeclaration | EdgeKind::TemplateParameterDefinition => false,
                EdgeKind::Generic => false,
            }
        }

        fn take_this_id_usage_set<Id: Into<ItemId>>(&mut self, id: Id) -> ItemSet {
            let id = id.into();
            self.ys
                .get_mut(&id)
                .expect(
                    "Should have a set of used template params for every item \
                     id",
                )
                .take()
                .expect(
                    "Should maintain the invariant that all used template param \
                     sets are `Some` upon entry of `constrain`",
                )
        }

        fn constrain_instantiation_of_blocklisted_template(
            &self,
            id: ItemId,
            y: &mut ItemSet,
            inst: &TemplInstantiation,
        ) {
            let args = inst
                .template_arguments()
                .iter()
                .map(|x| {
                    x.into_resolver()
                        .through_type_refs()
                        .through_type_aliases()
                        .resolve(self.ctx)
                        .id()
                })
                .filter(|x| *x != id)
                .flat_map(|x| {
                    self.ys
                        .get(&x)
                        .expect("Should have a used entry for the template arg")
                        .as_ref()
                        .expect(
                            "Because a != this_id, and all used template \
                             param sets other than this_id's are `Some`, \
                             a's used template param set should be `Some`",
                        )
                        .iter()
                        .cloned()
                });
            y.extend(args);
        }

        fn constrain_instantiation(&self, id: ItemId, y: &mut ItemSet, inst: &TemplInstantiation) {
            let decl = self.ctx.resolve_type(inst.template_definition());
            let args = inst.template_arguments();
            let ps = decl.self_template_params(self.ctx);
            debug_assert!(id != inst.template_definition());
            let used_by_def = self
                .ys
                .get(&inst.template_definition().into())
                .expect("Should have a used entry for instantiation's template definition")
                .as_ref()
                .expect(
                    "And it should be Some because only this_id's set is None, and an \
                         instantiation's template definition should never be the \
                         instantiation itself",
                );
            for (arg, p) in args.iter().zip(ps.iter()) {
                if used_by_def.contains(&p.into()) {
                    let arg = arg
                        .into_resolver()
                        .through_type_refs()
                        .through_type_aliases()
                        .resolve(self.ctx)
                        .id();
                    if arg == id {
                        continue;
                    }
                    let used_by_arg = self
                        .ys
                        .get(&arg)
                        .expect("Should have a used entry for the template arg")
                        .as_ref()
                        .expect(
                            "Because arg != this_id, and all used template \
                             param sets other than this_id's are `Some`, \
                             arg's used template param set should be \
                             `Some`",
                        )
                        .iter()
                        .cloned();
                    y.extend(used_by_arg);
                }
            }
        }

        fn constrain_join(&self, y: &mut ItemSet, i: &Item) {
            i.trace(
                self.ctx,
                &mut |i2, kind| {
                    if i2 == i.id() || !Self::check_edge(kind) {
                        return;
                    }
                    let y2 = self
                        .ys
                        .get(&i2)
                        .expect("Should have a used set for the sub_id successor")
                        .as_ref()
                        .expect(
                            "Because sub_id != id, and all used template \
                             param sets other than id's are `Some`, \
                             sub_id's used template param set should be \
                             `Some`",
                        )
                        .iter()
                        .cloned();
                    y.extend(y2);
                },
                &(),
            );
        }
    }

    impl<'ctx> Monotone for Analysis<'ctx> {
        type Node = ItemId;
        type Extra = &'ctx BindgenContext;
        type Output = HashMap<ItemId, ItemSet>;

        fn new(ctx: &'ctx BindgenContext) -> Analysis<'ctx> {
            let mut ys = HashMap::default();
            let mut deps = HashMap::default();
            let alloweds: HashSet<_> = ctx.allowed_items().iter().cloned().collect();
            let allowed_and_blocklisted_items: ItemSet = alloweds
                .iter()
                .cloned()
                .flat_map(|i| {
                    let mut ys = vec![i];
                    i.trace(
                        ctx,
                        &mut |i2, _| {
                            ys.push(i2);
                        },
                        &(),
                    );
                    ys
                })
                .collect();
            for i in allowed_and_blocklisted_items {
                deps.entry(i).or_insert_with(Vec::new);
                ys.entry(i).or_insert_with(|| Some(ItemSet::new()));
                {
                    i.trace(
                        ctx,
                        &mut |i2: ItemId, _| {
                            ys.entry(i2).or_insert_with(|| Some(ItemSet::new()));
                            deps.entry(i2).or_insert_with(Vec::new).push(i);
                        },
                        &(),
                    );
                }
                let k = ctx.resolve_item(i).as_type().map(|ty| ty.kind());
                if let Some(TyKind::TemplateInstantiation(inst)) = k {
                    let decl = ctx.resolve_type(inst.template_definition());
                    let args = inst.template_arguments();
                    let ps = decl.self_template_params(ctx);
                    for (arg, p) in args.iter().zip(ps.iter()) {
                        let arg = arg
                            .into_resolver()
                            .through_type_aliases()
                            .through_type_refs()
                            .resolve(ctx)
                            .id();
                        let p = p
                            .into_resolver()
                            .through_type_aliases()
                            .through_type_refs()
                            .resolve(ctx)
                            .id();
                        ys.entry(arg).or_insert_with(|| Some(ItemSet::new()));
                        ys.entry(p).or_insert_with(|| Some(ItemSet::new()));
                        deps.entry(arg).or_insert_with(Vec::new).push(p);
                    }
                }
            }
            if cfg!(feature = "__testing_only_extra_assertions") {
                for i in alloweds.iter() {
                    extra_assert!(ys.contains_key(i));
                    extra_assert!(deps.contains_key(i));
                    i.trace(
                        ctx,
                        &mut |i2, _| {
                            extra_assert!(ys.contains_key(&i2));
                            extra_assert!(deps.contains_key(&i2));
                        },
                        &(),
                    )
                }
            }
            Analysis {
                ctx,
                ys,
                deps,
                alloweds,
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
            extra_assert!(self.ys.values().all(|v| v.is_some()));
            let mut y = self.take_this_id_usage_set(id);
            let len = y.len();
            let i = self.ctx.resolve_item(id);
            let ty_kind = i.as_type().map(|x| x.kind());
            match ty_kind {
                Some(&TyKind::TypeParam) => {
                    y.insert(id);
                },
                Some(TyKind::TemplateInstantiation(x)) => {
                    if self.alloweds.contains(&x.template_definition().into()) {
                        self.constrain_instantiation(id, &mut y, x);
                    } else {
                        self.constrain_instantiation_of_blocklisted_template(id, &mut y, x);
                    }
                },
                _ => self.constrain_join(&mut y, i),
            }
            let len2 = y.len();
            assert!(len2 >= len);
            debug_assert!(self.ys[&id].is_none());
            self.ys.insert(id, Some(y));
            extra_assert!(self.ys.values().all(|v| v.is_some()));
            if len2 != len {
                YConstrain::Changed
            } else {
                YConstrain::Same
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

    impl<'ctx> From<Analysis<'ctx>> for HashMap<ItemId, ItemSet> {
        fn from(x: Analysis<'ctx>) -> Self {
            x.ys.into_iter().map(|(k, v)| (k, v.unwrap())).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HashMap, HashSet};

    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    struct Node(usize);

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct Graph(HashMap<Node, Vec<Node>>);

    impl Graph {
        fn make_test_graph() -> Graph {
            let mut y = Graph::default();
            y.0.insert(Node(1), vec![Node(3)]);
            y.0.insert(Node(2), vec![Node(2)]);
            y.0.insert(Node(3), vec![Node(4), Node(5)]);
            y.0.insert(Node(4), vec![Node(7)]);
            y.0.insert(Node(5), vec![Node(6), Node(7)]);
            y.0.insert(Node(6), vec![Node(8)]);
            y.0.insert(Node(7), vec![Node(3)]);
            y.0.insert(Node(8), vec![]);
            y
        }

        fn reverse(&self) -> Graph {
            let mut y = Graph::default();
            for (node, edges) in self.0.iter() {
                y.0.entry(*node).or_insert_with(Vec::new);
                for referent in edges.iter() {
                    y.0.entry(*referent).or_insert_with(Vec::new).push(*node);
                }
            }
            y
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ReachableFrom<'a> {
        reachable: HashMap<Node, HashSet<Node>>,
        graph: &'a Graph,
        reversed: Graph,
    }

    impl<'a> Monotone for ReachableFrom<'a> {
        type Node = Node;
        type Extra = &'a Graph;
        type Output = HashMap<Node, HashSet<Node>>;

        fn new(graph: &'a Graph) -> ReachableFrom {
            let reversed = graph.reverse();
            ReachableFrom {
                reachable: Default::default(),
                graph,
                reversed,
            }
        }

        fn initial_worklist(&self) -> Vec<Node> {
            self.graph.0.keys().cloned().collect()
        }

        fn constrain(&mut self, n: Node) -> YConstrain {
            let s = self.reachable.entry(n).or_insert_with(HashSet::default).len();
            for n2 in self.graph.0[&n].iter() {
                self.reachable.get_mut(&n).unwrap().insert(*n2);
                let r2 = self.reachable.entry(*n2).or_insert_with(HashSet::default).clone();
                for transitive in r2 {
                    self.reachable.get_mut(&n).unwrap().insert(transitive);
                }
            }
            let s2 = self.reachable[&n].len();
            if s != s2 {
                YConstrain::Changed
            } else {
                YConstrain::Same
            }
        }

        fn each_depending_on<F>(&self, n: Node, mut f: F)
        where
            F: FnMut(Node),
        {
            for dep in self.reversed.0[&n].iter() {
                f(*dep);
            }
        }
    }

    impl<'a> From<ReachableFrom<'a>> for HashMap<Node, HashSet<Node>> {
        fn from(x: ReachableFrom<'a>) -> Self {
            x.reachable
        }
    }

    #[test]
    fn monotone() {
        let g = Graph::make_test_graph();
        let y = analyze::<ReachableFrom>(&g);
        println!("reachable = {:#?}", y);
        fn nodes<A>(x: A) -> HashSet<Node>
        where
            A: AsRef<[usize]>,
        {
            x.as_ref().iter().cloned().map(Node).collect()
        }
        let mut y2 = HashMap::default();
        y2.insert(Node(1), nodes([3, 4, 5, 6, 7, 8]));
        y2.insert(Node(2), nodes([2]));
        y2.insert(Node(3), nodes([3, 4, 5, 6, 7, 8]));
        y2.insert(Node(4), nodes([3, 4, 5, 6, 7, 8]));
        y2.insert(Node(5), nodes([3, 4, 5, 6, 7, 8]));
        y2.insert(Node(6), nodes([8]));
        y2.insert(Node(7), nodes([3, 4, 5, 6, 7, 8]));
        y2.insert(Node(8), nodes([]));
        println!("expected = {:#?}", y2);
        assert_eq!(y, y2);
    }
}
