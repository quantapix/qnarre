use super::super::codegen::{EnumVariation, CONSTIFIED_ENUM_MODULE_REPR_NAME};
use super::analysis::{HasVtable, Sizedness, *};
use super::annotations::Annotations;
use super::comp::{CompKind, MethodKind};
use super::context::{BindgenContext, ItemId, PartialType, TypeId};
use super::derive::{
    CanDeriveCopy, CanDeriveDebug, CanDeriveDefault, CanDeriveEq, CanDeriveHash, CanDeriveOrd, CanDerivePartialEq,
    CanDerivePartialOrd,
};
use super::dot::DotAttrs;
use super::function::{FnKind, Function};
use super::item_kind::ItemKind;
use super::layout::Opaque;
use super::module::Module;
use super::template::{AsTemplParam, TemplParams};
use super::traversal::{EdgeKind, Trace, Tracer};
use super::ty::{TyKind, Type};
use crate::clang;
use crate::parse;

use lazycell::LazyCell;

use std::cell::Cell;
use std::collections::BTreeSet;
use std::fmt::Write;
use std::io;
use std::iter;

pub(crate) trait CanonicalName {
    fn canonical_name(&self, ctx: &BindgenContext) -> String;
}

pub(crate) trait CanonicalPath {
    fn namespace_aware_canonical_path(&self, ctx: &BindgenContext) -> Vec<String>;

    fn canonical_path(&self, ctx: &BindgenContext) -> Vec<String>;
}

pub(crate) trait IsOpaque {
    type Extra;

    fn is_opaque(&self, ctx: &BindgenContext, extra: &Self::Extra) -> bool;
}

pub(crate) trait HasTypeParamInArray {
    fn has_type_param_in_array(&self, ctx: &BindgenContext) -> bool;
}

pub(crate) trait HasFloat {
    fn has_float(&self, ctx: &BindgenContext) -> bool;
}

pub(crate) trait Ancestors {
    fn ancestors<'a>(&self, ctx: &'a BindgenContext) -> AncestorsIter<'a>;
}

struct DebugOnlyItemSet;

impl DebugOnlyItemSet {
    fn new() -> Self {
        DebugOnlyItemSet
    }

    fn contains(&self, _id: &ItemId) -> bool {
        false
    }

    fn insert(&mut self, _id: ItemId) {}
}

pub(crate) struct AncestorsIter<'a> {
    item: ItemId,
    ctx: &'a BindgenContext,
    seen: DebugOnlyItemSet,
}

impl<'a> AncestorsIter<'a> {
    fn new<Id: Into<ItemId>>(ctx: &'a BindgenContext, id: Id) -> Self {
        AncestorsIter {
            item: id.into(),
            ctx,
            seen: DebugOnlyItemSet::new(),
        }
    }
}

impl<'a> Iterator for AncestorsIter<'a> {
    type Item = ItemId;
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.ctx.resolve_item(self.item);
        if item.parent_id() == self.item {
            None
        } else {
            self.item = item.parent_id();
            self.seen.insert(item.id());
            Some(item.id())
        }
    }
}

impl<T> AsTemplParam for T
where
    T: Copy + Into<ItemId>,
{
    type Extra = ();
    fn as_template_param(&self, ctx: &BindgenContext, _: &()) -> Option<TypeId> {
        ctx.resolve_item((*self).into()).as_template_param(ctx, &())
    }
}

impl AsTemplParam for Item {
    type Extra = ();
    fn as_template_param(&self, ctx: &BindgenContext, _: &()) -> Option<TypeId> {
        self.kind.as_template_param(ctx, self)
    }
}

impl AsTemplParam for ItemKind {
    type Extra = Item;
    fn as_template_param(&self, ctx: &BindgenContext, item: &Item) -> Option<TypeId> {
        match *self {
            ItemKind::Type(ref ty) => ty.as_template_param(ctx, item),
            ItemKind::Module(..) | ItemKind::Function(..) | ItemKind::Var(..) => None,
        }
    }
}

impl<T> CanonicalName for T
where
    T: Copy + Into<ItemId>,
{
    fn canonical_name(&self, ctx: &BindgenContext) -> String {
        debug_assert!(ctx.in_codegen_phase(), "You're not supposed to call this yet");
        ctx.resolve_item(*self).canonical_name(ctx)
    }
}

impl<T> CanonicalPath for T
where
    T: Copy + Into<ItemId>,
{
    fn namespace_aware_canonical_path(&self, ctx: &BindgenContext) -> Vec<String> {
        debug_assert!(ctx.in_codegen_phase(), "You're not supposed to call this yet");
        ctx.resolve_item(*self).namespace_aware_canonical_path(ctx)
    }

    fn canonical_path(&self, ctx: &BindgenContext) -> Vec<String> {
        debug_assert!(ctx.in_codegen_phase(), "You're not supposed to call this yet");
        ctx.resolve_item(*self).canonical_path(ctx)
    }
}

impl<T> Ancestors for T
where
    T: Copy + Into<ItemId>,
{
    fn ancestors<'a>(&self, ctx: &'a BindgenContext) -> AncestorsIter<'a> {
        AncestorsIter::new(ctx, *self)
    }
}

impl Ancestors for Item {
    fn ancestors<'a>(&self, ctx: &'a BindgenContext) -> AncestorsIter<'a> {
        self.id().ancestors(ctx)
    }
}

impl<Id> Trace for Id
where
    Id: Copy + Into<ItemId>,
{
    type Extra = ();
    fn trace<T>(&self, ctx: &BindgenContext, tracer: &mut T, extra: &())
    where
        T: Tracer,
    {
        ctx.resolve_item(*self).trace(ctx, tracer, extra);
    }
}

impl Trace for Item {
    type Extra = ();
    fn trace<T>(&self, ctx: &BindgenContext, tracer: &mut T, _extra: &())
    where
        T: Tracer,
    {
        match *self.kind() {
            ItemKind::Type(ref ty) => {
                if ty.should_be_traced_unconditionally() || !self.is_opaque(ctx, &()) {
                    ty.trace(ctx, tracer, self);
                }
            },
            ItemKind::Function(ref fun) => {
                tracer.visit(fun.signature().into());
            },
            ItemKind::Var(ref var) => {
                tracer.visit_kind(var.ty().into(), EdgeKind::VarType);
            },
            ItemKind::Module(_) => {},
        }
    }
}

impl CanDeriveDebug for Item {
    fn can_derive_debug(&self, ctx: &BindgenContext) -> bool {
        self.id().can_derive_debug(ctx)
    }
}

impl CanDeriveDefault for Item {
    fn can_derive_default(&self, ctx: &BindgenContext) -> bool {
        self.id().can_derive_default(ctx)
    }
}

impl CanDeriveCopy for Item {
    fn can_derive_copy(&self, ctx: &BindgenContext) -> bool {
        self.id().can_derive_copy(ctx)
    }
}

impl CanDeriveHash for Item {
    fn can_derive_hash(&self, ctx: &BindgenContext) -> bool {
        self.id().can_derive_hash(ctx)
    }
}

impl CanDerivePartialOrd for Item {
    fn can_derive_partialord(&self, ctx: &BindgenContext) -> bool {
        self.id().can_derive_partialord(ctx)
    }
}

impl CanDerivePartialEq for Item {
    fn can_derive_partialeq(&self, ctx: &BindgenContext) -> bool {
        self.id().can_derive_partialeq(ctx)
    }
}

impl CanDeriveEq for Item {
    fn can_derive_eq(&self, ctx: &BindgenContext) -> bool {
        self.id().can_derive_eq(ctx)
    }
}

impl CanDeriveOrd for Item {
    fn can_derive_ord(&self, ctx: &BindgenContext) -> bool {
        self.id().can_derive_ord(ctx)
    }
}

#[derive(Debug)]
pub(crate) struct Item {
    id: ItemId,
    local_id: LazyCell<usize>,
    next_child_local_id: Cell<usize>,
    canonical_name: LazyCell<String>,
    path_for_allowlisting: LazyCell<Vec<String>>,
    comment: Option<String>,
    annotations: Annotations,
    parent_id: ItemId,
    kind: ItemKind,
    location: Option<clang::SrcLoc>,
}

impl AsRef<ItemId> for Item {
    fn as_ref(&self) -> &ItemId {
        &self.id
    }
}

impl Item {
    pub(crate) fn new(
        id: ItemId,
        comment: Option<String>,
        annotations: Option<Annotations>,
        parent_id: ItemId,
        kind: ItemKind,
        location: Option<clang::SrcLoc>,
    ) -> Self {
        debug_assert!(id != parent_id || kind.is_module());
        Item {
            id,
            local_id: LazyCell::new(),
            next_child_local_id: Cell::new(1),
            canonical_name: LazyCell::new(),
            path_for_allowlisting: LazyCell::new(),
            parent_id,
            comment,
            annotations: annotations.unwrap_or_default(),
            kind,
            location,
        }
    }

    pub(crate) fn new_opaque_type(with_id: ItemId, ty: &clang::Type, ctx: &mut BindgenContext) -> TypeId {
        let location = ty.declaration().location();
        let ty = Opaque::from_clang_ty(ty, ctx);
        let kind = ItemKind::Type(ty);
        let parent = ctx.root_module().into();
        ctx.add_item(Item::new(with_id, None, None, parent, kind, Some(location)), None, None);
        with_id.as_type_id_unchecked()
    }

    pub(crate) fn id(&self) -> ItemId {
        self.id
    }

    pub(crate) fn parent_id(&self) -> ItemId {
        self.parent_id
    }

    pub(crate) fn set_parent_for_replacement<Id: Into<ItemId>>(&mut self, id: Id) {
        self.parent_id = id.into();
    }

    pub(crate) fn codegen_depth(&self, ctx: &BindgenContext) -> usize {
        if !ctx.opts().enable_cxx_namespaces {
            return 0;
        }
        self.ancestors(ctx)
            .filter(|id| {
                ctx.resolve_item(*id).as_module().map_or(false, |module| {
                    !module.is_inline() || ctx.opts().conservative_inline_namespaces
                })
            })
            .count()
            + 1
    }

    pub(crate) fn comment(&self, ctx: &BindgenContext) -> Option<String> {
        if !ctx.opts().generate_comments {
            return None;
        }
        self.comment.as_ref().map(|comment| ctx.opts().process_comment(comment))
    }

    pub(crate) fn kind(&self) -> &ItemKind {
        &self.kind
    }

    pub(crate) fn kind_mut(&mut self) -> &mut ItemKind {
        &mut self.kind
    }

    pub(crate) fn location(&self) -> Option<&clang::SrcLoc> {
        self.location.as_ref()
    }

    pub(crate) fn local_id(&self, ctx: &BindgenContext) -> usize {
        *self.local_id.borrow_with(|| {
            let parent = ctx.resolve_item(self.parent_id);
            parent.next_child_local_id()
        })
    }

    pub(crate) fn next_child_local_id(&self) -> usize {
        let local_id = self.next_child_local_id.get();
        self.next_child_local_id.set(local_id + 1);
        local_id
    }

    pub(crate) fn is_toplevel(&self, ctx: &BindgenContext) -> bool {
        if ctx.opts().enable_cxx_namespaces && self.kind().is_module() && self.id() != ctx.root_module() {
            return false;
        }
        let mut parent = self.parent_id;
        loop {
            let parent_item = match ctx.resolve_item_fallible(parent) {
                Some(item) => item,
                None => return false,
            };
            if parent_item.id() == ctx.root_module() {
                return true;
            } else if ctx.opts().enable_cxx_namespaces || !parent_item.kind().is_module() {
                return false;
            }
            parent = parent_item.parent_id();
        }
    }

    pub(crate) fn expect_type(&self) -> &Type {
        self.kind().expect_type()
    }

    pub(crate) fn as_type(&self) -> Option<&Type> {
        self.kind().as_type()
    }

    pub(crate) fn expect_function(&self) -> &Function {
        self.kind().expect_function()
    }

    pub(crate) fn is_module(&self) -> bool {
        matches!(self.kind, ItemKind::Module(..))
    }

    pub(crate) fn annotations(&self) -> &Annotations {
        &self.annotations
    }

    pub(crate) fn is_blocklisted(&self, ctx: &BindgenContext) -> bool {
        debug_assert!(ctx.in_codegen_phase(), "You're not supposed to call this yet");
        if self.annotations.hide() {
            return true;
        }
        if !ctx.opts().blocklisted_files.is_empty() {
            if let Some(location) = &self.location {
                let (file, _, _, _) = location.location();
                if let Some(filename) = file.name() {
                    if ctx.opts().blocklisted_files.matches(filename) {
                        return true;
                    }
                }
            }
        }
        let path = self.path_for_allowlisting(ctx);
        let name = path[1..].join("::");
        ctx.opts().blocklisted_items.matches(&name)
            || match self.kind {
                ItemKind::Type(..) => {
                    ctx.opts().blocklisted_types.matches(&name) || ctx.is_replaced_type(path, self.id)
                },
                ItemKind::Function(..) => ctx.opts().blocklisted_fns.matches(&name),
                ItemKind::Var(..) | ItemKind::Module(..) => false,
            }
    }

    pub(crate) fn name<'a>(&'a self, ctx: &'a BindgenContext) -> NameOpts<'a> {
        NameOpts::new(self, ctx)
    }

    fn name_target(&self, ctx: &BindgenContext) -> ItemId {
        let mut targets_seen = DebugOnlyItemSet::new();
        let mut i = self;
        loop {
            targets_seen.insert(i.id());
            if self.annotations().use_instead_of().is_some() {
                return self.id();
            }
            match *i.kind() {
                ItemKind::Type(ref ty) => match *ty.kind() {
                    TyKind::ResolvedTypeRef(inner) => {
                        i = ctx.resolve_item(inner);
                    },
                    TyKind::TemplateInstantiation(ref x) => {
                        i = ctx.resolve_item(x.template_definition());
                    },
                    _ => return i.id(),
                },
                _ => return i.id(),
            }
        }
    }

    pub(crate) fn full_disambiguated_name(&self, ctx: &BindgenContext) -> String {
        let mut s = String::new();
        let level = 0;
        self.push_disambiguated_name(ctx, &mut s, level);
        s
    }

    fn push_disambiguated_name(&self, ctx: &BindgenContext, to: &mut String, level: u8) {
        to.push_str(&self.canonical_name(ctx));
        if let ItemKind::Type(ref ty) = *self.kind() {
            if let TyKind::TemplateInstantiation(ref x) = *ty.kind() {
                to.push_str(&format!("_open{}_", level));
                for arg in x.template_arguments() {
                    arg.into_resolver()
                        .through_type_refs()
                        .resolve(ctx)
                        .push_disambiguated_name(ctx, to, level + 1);
                    to.push('_');
                }
                to.push_str(&format!("close{}", level));
            }
        }
    }

    fn func_name(&self) -> Option<&str> {
        match *self.kind() {
            ItemKind::Function(ref func) => Some(func.name()),
            _ => None,
        }
    }

    fn overload_index(&self, ctx: &BindgenContext) -> Option<usize> {
        self.func_name().and_then(|func_name| {
            let parent = ctx.resolve_item(self.parent_id());
            if let ItemKind::Type(ref ty) = *parent.kind() {
                if let TyKind::Comp(ref ci) = *ty.kind() {
                    return ci.constructors().iter().position(|c| *c == self.id()).or_else(|| {
                        ci.methods()
                            .iter()
                            .filter(|m| {
                                let item = ctx.resolve_item(m.signature());
                                let func = item.expect_function();
                                func.name() == func_name
                            })
                            .position(|m| m.signature() == self.id())
                    });
                }
            }

            None
        })
    }

    fn base_name(&self, ctx: &BindgenContext) -> String {
        if let Some(path) = self.annotations().use_instead_of() {
            return path.last().unwrap().clone();
        }
        match *self.kind() {
            ItemKind::Var(ref var) => var.name().to_owned(),
            ItemKind::Module(ref module) => module
                .name()
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("_bindgen_mod_{}", self.exposed_id(ctx))),
            ItemKind::Type(ref ty) => ty
                .sanitized_name(ctx)
                .map(Into::into)
                .unwrap_or_else(|| format!("_bindgen_ty_{}", self.exposed_id(ctx))),
            ItemKind::Function(ref fun) => {
                let mut name = fun.name().to_owned();
                if let Some(idx) = self.overload_index(ctx) {
                    if idx > 0 {
                        write!(&mut name, "{}", idx).unwrap();
                    }
                }
                name
            },
        }
    }

    fn is_anon(&self) -> bool {
        match self.kind() {
            ItemKind::Module(module) => module.name().is_none(),
            ItemKind::Type(ty) => ty.name().is_none(),
            ItemKind::Function(_) => false,
            ItemKind::Var(_) => false,
        }
    }

    pub(crate) fn real_canonical_name(&self, ctx: &BindgenContext, opt: &NameOpts) -> String {
        let target = ctx.resolve_item(self.name_target(ctx));

        if let Some(path) = target.annotations.use_instead_of() {
            if ctx.opts().enable_cxx_namespaces {
                return path.last().unwrap().clone();
            }
            return path.join("_");
        }

        let base_name = target.base_name(ctx);

        if target.is_template_param(ctx, &()) {
            return base_name;
        }

        let mut ids_iter = target
            .parent_id()
            .ancestors(ctx)
            .filter(|id| *id != ctx.root_module())
            .take_while(|id| !opt.within_namespaces || !ctx.resolve_item(*id).is_module())
            .filter(|id| {
                if !ctx.opts().conservative_inline_namespaces {
                    if let ItemKind::Module(ref module) = *ctx.resolve_item(*id).kind() {
                        return !module.is_inline();
                    }
                }

                true
            });

        let ids: Vec<_> = if ctx.opts().disable_nested_struct_naming {
            let mut ids = Vec::new();

            if target.is_anon() {
                for id in ids_iter.by_ref() {
                    ids.push(id);

                    if !ctx.resolve_item(id).is_anon() {
                        break;
                    }
                }
            }

            ids
        } else {
            ids_iter.collect()
        };

        let mut names: Vec<_> = ids
            .into_iter()
            .map(|id| {
                let item = ctx.resolve_item(id);
                let target = ctx.resolve_item(item.name_target(ctx));
                target.base_name(ctx)
            })
            .filter(|name| !name.is_empty())
            .collect();

        names.reverse();

        if !base_name.is_empty() {
            names.push(base_name);
        }

        if ctx.opts().c_naming {
            if let Some(prefix) = self.c_naming_prefix() {
                names.insert(0, prefix.to_string());
            }
        }

        let name = names.join("_");

        let name = if opt.user_mangled == UserMangled::Yes {
            ctx.opts()
                .last_callback(|callbacks| callbacks.item_name(&name))
                .unwrap_or(name)
        } else {
            name
        };

        ctx.rust_mangle(&name).into_owned()
    }

    pub(crate) fn exposed_id(&self, ctx: &BindgenContext) -> String {
        let ty_kind = self.kind().as_type().map(|t| t.kind());
        if let Some(ty_kind) = ty_kind {
            match *ty_kind {
                TyKind::Comp(..) | TyKind::TemplateInstantiation(..) | TyKind::Enum(..) => {
                    return self.local_id(ctx).to_string()
                },
                _ => {},
            }
        }

        format!("id_{}", self.id().as_usize())
    }

    pub(crate) fn as_module(&self) -> Option<&Module> {
        match self.kind {
            ItemKind::Module(ref module) => Some(module),
            _ => None,
        }
    }

    pub(crate) fn as_module_mut(&mut self) -> Option<&mut Module> {
        match self.kind {
            ItemKind::Module(ref mut module) => Some(module),
            _ => None,
        }
    }

    fn is_constified_enum_module(&self, ctx: &BindgenContext) -> bool {
        let item = self.id.into_resolver().through_type_refs().resolve(ctx);
        let type_ = match *item.kind() {
            ItemKind::Type(ref type_) => type_,
            _ => return false,
        };

        match *type_.kind() {
            TyKind::Enum(ref enum_) => enum_.computed_enum_variation(ctx, self) == EnumVariation::ModuleConsts,
            TyKind::Alias(inner_id) => {
                let inner_item = ctx.resolve_item(inner_id);
                let name = item.canonical_name(ctx);

                if inner_item.canonical_name(ctx) == name {
                    inner_item.is_constified_enum_module(ctx)
                } else {
                    false
                }
            },
            _ => false,
        }
    }

    pub(crate) fn is_enabled_for_codegen(&self, ctx: &BindgenContext) -> bool {
        let cc = &ctx.opts().codegen_config;
        match *self.kind() {
            ItemKind::Module(..) => true,
            ItemKind::Var(_) => cc.vars(),
            ItemKind::Type(_) => cc.types(),
            ItemKind::Function(ref f) => match f.kind() {
                FnKind::Function => cc.functions(),
                FnKind::Method(MethodKind::Constructor) => cc.constructors(),
                FnKind::Method(MethodKind::Destructor) | FnKind::Method(MethodKind::VirtualDestructor { .. }) => {
                    cc.destructors()
                },
                FnKind::Method(MethodKind::Static)
                | FnKind::Method(MethodKind::Normal)
                | FnKind::Method(MethodKind::Virtual { .. }) => cc.methods(),
            },
        }
    }

    pub(crate) fn path_for_allowlisting(&self, ctx: &BindgenContext) -> &Vec<String> {
        self.path_for_allowlisting
            .borrow_with(|| self.compute_path(ctx, UserMangled::No))
    }

    fn compute_path(&self, ctx: &BindgenContext, mangled: UserMangled) -> Vec<String> {
        if let Some(path) = self.annotations().use_instead_of() {
            let mut ret = vec![ctx.resolve_item(ctx.root_module()).name(ctx).get()];
            ret.extend_from_slice(path);
            return ret;
        }

        let target = ctx.resolve_item(self.name_target(ctx));
        let mut path: Vec<_> = target
            .ancestors(ctx)
            .chain(iter::once(ctx.root_module().into()))
            .map(|id| ctx.resolve_item(id))
            .filter(|item| {
                item.id() == target.id()
                    || item.as_module().map_or(false, |module| {
                        !module.is_inline() || ctx.opts().conservative_inline_namespaces
                    })
            })
            .map(|item| {
                ctx.resolve_item(item.name_target(ctx))
                    .name(ctx)
                    .within_namespaces()
                    .user_mangled(mangled)
                    .get()
            })
            .collect();
        path.reverse();
        path
    }

    fn c_naming_prefix(&self) -> Option<&str> {
        let ty = match self.kind {
            ItemKind::Type(ref ty) => ty,
            _ => return None,
        };

        Some(match ty.kind() {
            TyKind::Comp(ref ci) => match ci.kind() {
                CompKind::Struct => "struct",
                CompKind::Union => "union",
            },
            TyKind::Enum(..) => "enum",
            _ => return None,
        })
    }

    pub(crate) fn must_use(&self, ctx: &BindgenContext) -> bool {
        self.annotations().must_use_type() || ctx.must_use_type_by_name(self)
    }
}

impl<T> IsOpaque for T
where
    T: Copy + Into<ItemId>,
{
    type Extra = ();

    fn is_opaque(&self, ctx: &BindgenContext, _: &()) -> bool {
        debug_assert!(ctx.in_codegen_phase(), "You're not supposed to call this yet");
        ctx.resolve_item((*self).into()).is_opaque(ctx, &())
    }
}

impl IsOpaque for Item {
    type Extra = ();

    fn is_opaque(&self, ctx: &BindgenContext, _: &()) -> bool {
        debug_assert!(ctx.in_codegen_phase(), "You're not supposed to call this yet");
        self.annotations.opaque()
            || self.as_type().map_or(false, |ty| ty.is_opaque(ctx, self))
            || ctx.opaque_by_name(self.path_for_allowlisting(ctx))
    }
}

impl<T> HasVtable for T
where
    T: Copy + Into<ItemId>,
{
    fn has_vtable(&self, ctx: &BindgenContext) -> bool {
        let id: ItemId = (*self).into();
        id.as_type_id(ctx)
            .map_or(false, |id| !matches!(ctx.lookup_has_vtable(id), has_vtable::Result::No))
    }

    fn has_vtable_ptr(&self, ctx: &BindgenContext) -> bool {
        let id: ItemId = (*self).into();
        id.as_type_id(ctx).map_or(false, |id| {
            matches!(ctx.lookup_has_vtable(id), has_vtable::Result::SelfHasVtable)
        })
    }
}

impl HasVtable for Item {
    fn has_vtable(&self, ctx: &BindgenContext) -> bool {
        self.id().has_vtable(ctx)
    }

    fn has_vtable_ptr(&self, ctx: &BindgenContext) -> bool {
        self.id().has_vtable_ptr(ctx)
    }
}

impl<T> Sizedness for T
where
    T: Copy + Into<ItemId>,
{
    fn sizedness(&self, ctx: &BindgenContext) -> sizedness::Result {
        let id: ItemId = (*self).into();
        id.as_type_id(ctx)
            .map_or(sizedness::Result::default(), |x| ctx.lookup_sizedness(x))
    }
}

impl Sizedness for Item {
    fn sizedness(&self, ctx: &BindgenContext) -> sizedness::Result {
        self.id().sizedness(ctx)
    }
}

impl<T> HasTypeParamInArray for T
where
    T: Copy + Into<ItemId>,
{
    fn has_type_param_in_array(&self, ctx: &BindgenContext) -> bool {
        debug_assert!(ctx.in_codegen_phase(), "You're not supposed to call this yet");
        ctx.lookup_has_type_param_in_array(*self)
    }
}

impl HasTypeParamInArray for Item {
    fn has_type_param_in_array(&self, ctx: &BindgenContext) -> bool {
        debug_assert!(ctx.in_codegen_phase(), "You're not supposed to call this yet");
        ctx.lookup_has_type_param_in_array(self.id())
    }
}

impl<T> HasFloat for T
where
    T: Copy + Into<ItemId>,
{
    fn has_float(&self, ctx: &BindgenContext) -> bool {
        debug_assert!(ctx.in_codegen_phase(), "You're not supposed to call this yet");
        ctx.lookup_has_float(*self)
    }
}

impl HasFloat for Item {
    fn has_float(&self, ctx: &BindgenContext) -> bool {
        debug_assert!(ctx.in_codegen_phase(), "You're not supposed to call this yet");
        ctx.lookup_has_float(self.id())
    }
}

pub(crate) type ItemSet = BTreeSet<ItemId>;

impl DotAttrs for Item {
    fn dot_attributes<W>(&self, ctx: &BindgenContext, out: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        writeln!(
            out,
            "<tr><td>{:?}</td></tr>
                       <tr><td>name</td><td>{}</td></tr>",
            self.id,
            self.name(ctx).get()
        )?;

        if self.is_opaque(ctx, &()) {
            writeln!(out, "<tr><td>opaque</td><td>true</td></tr>")?;
        }

        self.kind.dot_attributes(ctx, out)
    }
}

impl<T> TemplParams for T
where
    T: Copy + Into<ItemId>,
{
    fn self_template_params(&self, ctx: &BindgenContext) -> Vec<TypeId> {
        ctx.resolve_item_fallible(*self)
            .map_or(vec![], |item| item.self_template_params(ctx))
    }
}

impl TemplParams for Item {
    fn self_template_params(&self, ctx: &BindgenContext) -> Vec<TypeId> {
        self.kind.self_template_params(ctx)
    }
}

impl TemplParams for ItemKind {
    fn self_template_params(&self, ctx: &BindgenContext) -> Vec<TypeId> {
        match *self {
            ItemKind::Type(ref ty) => ty.self_template_params(ctx),
            ItemKind::Function(_) | ItemKind::Module(_) | ItemKind::Var(_) => {
                vec![]
            },
        }
    }
}

fn visit_child(
    cur: clang::Cursor,
    id: ItemId,
    ty: &clang::Type,
    parent_id: Option<ItemId>,
    ctx: &mut BindgenContext,
    result: &mut Result<TypeId, parse::Error>,
) -> clang_lib::CXChildVisitResult {
    use clang_lib::*;
    if result.is_ok() {
        return CXChildVisit_Break;
    }

    *result = Item::from_ty_with_id(id, ty, cur, parent_id, ctx);

    match *result {
        Ok(..) => CXChildVisit_Break,
        Err(parse::Error::Recurse) => {
            cur.visit(|c| visit_child(c, id, ty, parent_id, ctx, result));
            CXChildVisit_Continue
        },
        Err(parse::Error::Continue) => CXChildVisit_Continue,
    }
}

impl Item {
    pub(crate) fn builtin_type(kind: TyKind, is_const: bool, ctx: &mut BindgenContext) -> TypeId {
        match kind {
            TyKind::Void | TyKind::Int(..) | TyKind::Pointer(..) | TyKind::Float(..) => {},
            _ => panic!("Unsupported builtin type"),
        }

        let ty = Type::new(None, None, kind, is_const);
        let id = ctx.next_item_id();
        let module = ctx.root_module().into();
        ctx.add_item(Item::new(id, None, None, module, ItemKind::Type(ty), None), None, None);
        id.as_type_id_unchecked()
    }

    pub(crate) fn parse(
        cursor: clang::Cursor,
        parent_id: Option<ItemId>,
        ctx: &mut BindgenContext,
    ) -> Result<ItemId, parse::Error> {
        use crate::ir::var::Var;
        use clang_lib::*;

        if !cursor.is_valid() {
            return Err(parse::Error::Continue);
        }

        let comment = cursor.raw_comment();
        let annotations = Annotations::new(&cursor);

        let current_module = ctx.current_module().into();
        let relevant_parent_id = parent_id.unwrap_or(current_module);

        macro_rules! try_parse {
            ($what:ident) => {
                match $what::parse(cursor, ctx) {
                    Ok(parse::Result::New(item, declaration)) => {
                        let id = ctx.next_item_id();
                        ctx.add_item(
                            Item::new(
                                id,
                                comment,
                                annotations,
                                relevant_parent_id,
                                ItemKind::$what(item),
                                Some(cursor.location()),
                            ),
                            declaration,
                            Some(cursor),
                        );
                        return Ok(id);
                    },
                    Ok(parse::Result::AlreadyResolved(id)) => {
                        return Ok(id);
                    },
                    Err(parse::Error::Recurse) => return Err(parse::Error::Recurse),
                    Err(parse::Error::Continue) => {},
                }
            };
        }
        try_parse!(Module);
        try_parse!(Function);
        try_parse!(Var);
        {
            let definition = cursor.definition();
            let applicable_cursor = definition.unwrap_or(cursor);

            let relevant_parent_id = match definition {
                Some(definition) => {
                    if definition != cursor {
                        ctx.add_semantic_parent(definition, relevant_parent_id);
                        return Ok(Item::from_ty_or_ref(applicable_cursor.cur_type(), cursor, parent_id, ctx).into());
                    }
                    ctx.known_semantic_parent(definition)
                        .or(parent_id)
                        .unwrap_or_else(|| ctx.current_module().into())
                },
                None => relevant_parent_id,
            };

            match Item::from_ty(
                &applicable_cursor.cur_type(),
                applicable_cursor,
                Some(relevant_parent_id),
                ctx,
            ) {
                Ok(ty) => return Ok(ty.into()),
                Err(parse::Error::Recurse) => return Err(parse::Error::Recurse),
                Err(parse::Error::Continue) => {},
            }
        }

        if cursor.kind() == CXCursor_UnexposedDecl {
            Err(parse::Error::Recurse)
        } else {
            match cursor.kind() {
                CXCursor_MacroDefinition
                | CXCursor_MacroExpansion
                | CXCursor_UsingDeclaration
                | CXCursor_UsingDirective
                | CXCursor_StaticAssert
                | CXCursor_FunctionTemplate => {
                    debug!("Unhandled cursor kind {:?}: {:?}", cursor.kind(), cursor);
                },
                CXCursor_InclusionDirective => {
                    let file = cursor.get_included_file_name();
                    match file {
                        None => {
                            warn!("Inclusion of a nameless file in {:?}", cursor);
                        },
                        Some(filename) => {
                            ctx.include_file(filename);
                        },
                    }
                },
                _ => {
                    let spelling = cursor.spelling();
                    if !spelling.starts_with("operator") {
                        warn!("Unhandled cursor kind {:?}: {:?}", cursor.kind(), cursor);
                    }
                },
            }

            Err(parse::Error::Continue)
        }
    }

    pub(crate) fn from_ty_or_ref(
        ty: clang::Type,
        location: clang::Cursor,
        parent_id: Option<ItemId>,
        ctx: &mut BindgenContext,
    ) -> TypeId {
        let id = ctx.next_item_id();
        Self::from_ty_or_ref_with_id(id, ty, location, parent_id, ctx)
    }

    pub(crate) fn from_ty_or_ref_with_id(
        potential_id: ItemId,
        ty: clang::Type,
        location: clang::Cursor,
        parent_id: Option<ItemId>,
        ctx: &mut BindgenContext,
    ) -> TypeId {
        debug!(
            "from_ty_or_ref_with_id: {:?} {:?}, {:?}, {:?}",
            potential_id, ty, location, parent_id
        );

        if ctx.collected_typerefs() {
            debug!("refs already collected, resolving directly");
            return Item::from_ty_with_id(potential_id, &ty, location, parent_id, ctx)
                .unwrap_or_else(|_| Item::new_opaque_type(potential_id, &ty, ctx));
        }

        if let Some(ty) = ctx.builtin_or_resolved_ty(potential_id, parent_id, &ty, Some(location)) {
            debug!("{:?} already resolved: {:?}", ty, location);
            return ty;
        }

        debug!("New unresolved type reference: {:?}, {:?}", ty, location);

        let is_const = ty.is_const();
        let kind = TyKind::UnresolvedTypeRef(ty, location, parent_id);
        let current_module = ctx.current_module();

        ctx.add_item(
            Item::new(
                potential_id,
                None,
                None,
                parent_id.unwrap_or_else(|| current_module.into()),
                ItemKind::Type(Type::new(None, None, kind, is_const)),
                Some(location.location()),
            ),
            None,
            None,
        );
        potential_id.as_type_id_unchecked()
    }

    pub(crate) fn from_ty(
        ty: &clang::Type,
        location: clang::Cursor,
        parent_id: Option<ItemId>,
        ctx: &mut BindgenContext,
    ) -> Result<TypeId, parse::Error> {
        let id = ctx.next_item_id();
        Item::from_ty_with_id(id, ty, location, parent_id, ctx)
    }

    pub(crate) fn from_ty_with_id(
        id: ItemId,
        ty: &clang::Type,
        location: clang::Cursor,
        parent_id: Option<ItemId>,
        ctx: &mut BindgenContext,
    ) -> Result<TypeId, parse::Error> {
        use clang_lib::*;

        debug!(
            "Item::from_ty_with_id: {:?}\n\
             \tty = {:?},\n\
             \tlocation = {:?}",
            id, ty, location
        );

        if ty.kind() == clang_lib::CXType_Unexposed || location.cur_type().kind() == clang_lib::CXType_Unexposed {
            if ty.is_associated_type() || location.cur_type().is_associated_type() {
                return Ok(Item::new_opaque_type(id, ty, ctx));
            }

            if let Some(param_id) = Item::type_param(None, location, ctx) {
                return Ok(ctx.build_ty_wrapper(id, param_id, None, ty));
            }
        }

        if let Some(ref parent) = ty.declaration().fallible_semantic_parent() {
            if FnKind::from_cursor(parent).is_some() {
                debug!("Skipping type declared inside function: {:?}", ty);
                return Ok(Item::new_opaque_type(id, ty, ctx));
            }
        }

        let decl = {
            let canonical_def = ty.canonical_type().declaration().definition();
            canonical_def.unwrap_or_else(|| ty.declaration())
        };

        let comment = location
            .raw_comment()
            .or_else(|| decl.raw_comment())
            .or_else(|| location.raw_comment());

        let annotations = Annotations::new(&decl).or_else(|| Annotations::new(&location));

        if let Some(ref annotations) = annotations {
            if let Some(replaced) = annotations.use_instead_of() {
                ctx.replace(replaced, id);
            }
        }

        if let Some(ty) = ctx.builtin_or_resolved_ty(id, parent_id, ty, Some(location)) {
            return Ok(ty);
        }

        let mut valid_decl = decl.kind() != CXCursor_NoDeclFound;
        let declaration_to_look_for = if valid_decl {
            decl.canonical()
        } else if location.kind() == CXCursor_ClassTemplate {
            valid_decl = true;
            location
        } else {
            decl
        };

        if valid_decl {
            if let Some(partial) = ctx
                .currently_parsed_types()
                .iter()
                .find(|ty| *ty.decl() == declaration_to_look_for)
            {
                debug!("Avoiding recursion parsing type: {:?}", ty);
                return Ok(partial.id().as_type_id_unchecked());
            }
        }

        let current_module = ctx.current_module().into();
        let partial_ty = PartialType::new(declaration_to_look_for, id);
        if valid_decl {
            ctx.begin_parsing(partial_ty);
        }

        let result = Type::from_clang_ty(id, ty, location, parent_id, ctx);
        let relevant_parent_id = parent_id.unwrap_or(current_module);
        let ret = match result {
            Ok(parse::Result::AlreadyResolved(ty)) => Ok(ty.as_type_id_unchecked()),
            Ok(parse::Result::New(item, declaration)) => {
                ctx.add_item(
                    Item::new(
                        id,
                        comment,
                        annotations,
                        relevant_parent_id,
                        ItemKind::Type(item),
                        Some(location.location()),
                    ),
                    declaration,
                    Some(location),
                );
                Ok(id.as_type_id_unchecked())
            },
            Err(parse::Error::Continue) => Err(parse::Error::Continue),
            Err(parse::Error::Recurse) => {
                debug!("Item::from_ty recursing in the ast");
                let mut result = Err(parse::Error::Recurse);

                if valid_decl {
                    let finished = ctx.finish_parsing();
                    assert_eq!(*finished.decl(), declaration_to_look_for);
                }

                location.visit(|cur| visit_child(cur, id, ty, parent_id, ctx, &mut result));

                if valid_decl {
                    let partial_ty = PartialType::new(declaration_to_look_for, id);
                    ctx.begin_parsing(partial_ty);
                }

                if let Err(parse::Error::Recurse) = result {
                    warn!(
                        "Unknown type, assuming named template type: \
                         id = {:?}; spelling = {}",
                        id,
                        ty.spelling()
                    );
                    Item::type_param(Some(id), location, ctx)
                        .map(Ok)
                        .unwrap_or(Err(parse::Error::Recurse))
                } else {
                    result
                }
            },
        };

        if valid_decl {
            let partial_ty = ctx.finish_parsing();
            assert_eq!(*partial_ty.decl(), declaration_to_look_for);
        }

        ret
    }

    pub(crate) fn type_param(
        with_id: Option<ItemId>,
        location: clang::Cursor,
        ctx: &mut BindgenContext,
    ) -> Option<TypeId> {
        let ty = location.cur_type();

        debug!(
            "Item::type_param:\n\
             \twith_id = {:?},\n\
             \tty = {} {:?},\n\
             \tlocation: {:?}",
            with_id,
            ty.spelling(),
            ty,
            location
        );

        if ty.kind() != clang_lib::CXType_Unexposed {
            return None;
        }

        let ty_spelling = ty.spelling();

        fn is_template_with_spelling(refd: &clang::Cursor, spelling: &str) -> bool {
            lazy_static! {
                static ref ANON_TYPE_PARAM_RE: regex::Regex =
                    regex::Regex::new(r"^type\-parameter\-\d+\-\d+$").unwrap();
            }

            if refd.kind() != clang_lib::CXCursor_TemplateTypeParameter {
                return false;
            }

            let refd_spelling = refd.spelling();
            refd_spelling == spelling || (refd_spelling.is_empty() && ANON_TYPE_PARAM_RE.is_match(spelling.as_ref()))
        }

        let definition = if is_template_with_spelling(&location, &ty_spelling) {
            location
        } else if location.kind() == clang_lib::CXCursor_TypeRef {
            match location.referenced() {
                Some(refd) if is_template_with_spelling(&refd, &ty_spelling) => refd,
                _ => return None,
            }
        } else {
            let mut definition = None;

            location.visit(|child| {
                let child_ty = child.cur_type();
                if child_ty.kind() == clang_lib::CXCursor_TypeRef && child_ty.spelling() == ty_spelling {
                    match child.referenced() {
                        Some(refd) if is_template_with_spelling(&refd, &ty_spelling) => {
                            definition = Some(refd);
                            return clang_lib::CXChildVisit_Break;
                        },
                        _ => {},
                    }
                }

                clang_lib::CXChildVisit_Continue
            });

            definition?
        };
        assert!(is_template_with_spelling(&definition, &ty_spelling));

        let parent = ctx.root_module().into();

        if let Some(id) = ctx.get_type_param(&definition) {
            if let Some(with_id) = with_id {
                return Some(ctx.build_ty_wrapper(with_id, id, Some(parent), &ty));
            } else {
                return Some(id);
            }
        }

        let name = ty_spelling.replace("const ", "").replace('.', "");

        let id = with_id.unwrap_or_else(|| ctx.next_item_id());
        let item = Item::new(
            id,
            None,
            None,
            parent,
            ItemKind::Type(Type::named(name)),
            Some(location.location()),
        );
        ctx.add_type_param(item, definition);
        Some(id.as_type_id_unchecked())
    }
}

impl CanonicalName for Item {
    fn canonical_name(&self, ctx: &BindgenContext) -> String {
        debug_assert!(ctx.in_codegen_phase(), "You're not supposed to call this yet");
        self.canonical_name
            .borrow_with(|| {
                let in_namespace = ctx.opts().enable_cxx_namespaces || ctx.opts().disable_name_namespacing;

                if in_namespace {
                    self.name(ctx).within_namespaces().get()
                } else {
                    self.name(ctx).get()
                }
            })
            .clone()
    }
}

impl CanonicalPath for Item {
    fn namespace_aware_canonical_path(&self, ctx: &BindgenContext) -> Vec<String> {
        let mut path = self.canonical_path(ctx);

        if ctx.opts().disable_name_namespacing {
            let split_idx = path.len() - 1;
            path = path.split_off(split_idx);
        } else if !ctx.opts().enable_cxx_namespaces {
            path = vec![path[1..].join("_")];
        }

        if self.is_constified_enum_module(ctx) {
            path.push(CONSTIFIED_ENUM_MODULE_REPR_NAME.into());
        }

        path
    }

    fn canonical_path(&self, ctx: &BindgenContext) -> Vec<String> {
        self.compute_path(ctx, UserMangled::Yes)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum UserMangled {
    No,
    Yes,
}

#[derive(Debug)]
pub(crate) struct NameOpts<'a> {
    item: &'a Item,
    ctx: &'a BindgenContext,
    within_namespaces: bool,
    user_mangled: UserMangled,
}

impl<'a> NameOpts<'a> {
    pub(crate) fn new(item: &'a Item, ctx: &'a BindgenContext) -> Self {
        NameOpts {
            item,
            ctx,
            within_namespaces: false,
            user_mangled: UserMangled::Yes,
        }
    }

    pub(crate) fn within_namespaces(&mut self) -> &mut Self {
        self.within_namespaces = true;
        self
    }

    fn user_mangled(&mut self, user_mangled: UserMangled) -> &mut Self {
        self.user_mangled = user_mangled;
        self
    }

    pub(crate) fn get(&self) -> String {
        self.item.real_canonical_name(self.ctx, self)
    }
}
