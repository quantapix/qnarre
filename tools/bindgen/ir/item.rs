use super::super::codegen::{utils::variation, CONSTIFIED_ENUM_MODULE_REPR_NAME};
use super::analysis::{HasVtable, Sizedness, *};
use super::annotations::Annotations;
use super::comp::{CompKind, MethodKind};
use super::context::PartialType;
use super::dot::DotAttrs;
use super::function::{FnKind, Function};
use super::item_kind::ItemKind;
use super::layout::Opaque;
use super::module::Module;
use super::template::{AsTemplParam, TemplParams};
use super::traversal::{EdgeKind, Trace, Tracer};
use super::typ::{Type, TypeKind};
use super::{Context, ItemId, TypeId};
use crate::clang;
use crate::parse::{self, SubItem};
use lazycell::LazyCell;
use std::cell::Cell;
use std::collections::BTreeSet;
use std::fmt::Write;
use std::io;
use std::iter;

#[derive(Copy, Clone, Debug, PartialEq)]
enum UserMangled {
    No,
    Yes,
}
#[derive(Debug)]
pub struct NameOpts<'a> {
    item: &'a Item,
    ctx: &'a Context,
    within_namespaces: bool,
    user_mangled: UserMangled,
}
impl<'a> NameOpts<'a> {
    pub fn new(item: &'a Item, ctx: &'a Context) -> Self {
        NameOpts {
            item,
            ctx,
            within_namespaces: false,
            user_mangled: UserMangled::Yes,
        }
    }
    pub fn within_namespaces(&mut self) -> &mut Self {
        self.within_namespaces = true;
        self
    }
    fn user_mangled(&mut self, user_mangled: UserMangled) -> &mut Self {
        self.user_mangled = user_mangled;
        self
    }
    pub fn get(&self) -> String {
        self.item.real_canonical_name(self.ctx, self)
    }
}

pub trait CanonName {
    fn canon_name(&self, ctx: &Context) -> String;
}
impl<T> CanonName for T
where
    T: Copy + Into<ItemId>,
{
    fn canon_name(&self, ctx: &Context) -> String {
        ctx.resolve_item(*self).canon_name(ctx)
    }
}
impl CanonName for Item {
    fn canon_name(&self, ctx: &Context) -> String {
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

pub trait CanonPath {
    fn canon_path(&self, ctx: &Context) -> Vec<String>;
    fn namespace_aware_canon_path(&self, ctx: &Context) -> Vec<String>;
}
impl<T> CanonPath for T
where
    T: Copy + Into<ItemId>,
{
    fn canon_path(&self, ctx: &Context) -> Vec<String> {
        ctx.resolve_item(*self).canon_path(ctx)
    }
    fn namespace_aware_canon_path(&self, ctx: &Context) -> Vec<String> {
        ctx.resolve_item(*self).namespace_aware_canon_path(ctx)
    }
}
impl CanonPath for Item {
    fn canon_path(&self, ctx: &Context) -> Vec<String> {
        self.compute_path(ctx, UserMangled::Yes)
    }
    fn namespace_aware_canon_path(&self, ctx: &Context) -> Vec<String> {
        let mut y = self.canon_path(ctx);
        if ctx.opts().disable_name_namespacing {
            let idx = y.len() - 1;
            y = y.split_off(idx);
        } else if !ctx.opts().enable_cxx_namespaces {
            y = vec![y[1..].join("_")];
        }
        if self.is_constified_enum_module(ctx) {
            y.push(CONSTIFIED_ENUM_MODULE_REPR_NAME.into());
        }
        y
    }
}

pub trait IsOpaque {
    type Extra;
    fn is_opaque(&self, ctx: &Context, extra: &Self::Extra) -> bool;
}
impl<T> IsOpaque for T
where
    T: Copy + Into<ItemId>,
{
    type Extra = ();
    fn is_opaque(&self, ctx: &Context, _: &()) -> bool {
        ctx.resolve_item((*self).into()).is_opaque(ctx, &())
    }
}
impl IsOpaque for Item {
    type Extra = ();
    fn is_opaque(&self, ctx: &Context, _: &()) -> bool {
        self.annotations.opaque()
            || self.as_type().map_or(false, |ty| ty.is_opaque(ctx, self))
            || ctx.opaque_by_name(self.path_for_allowlisting(ctx))
    }
}

pub trait HasFloat {
    fn has_float(&self, ctx: &Context) -> bool;
}
impl<T> HasFloat for T
where
    T: Copy + Into<ItemId>,
{
    fn has_float(&self, ctx: &Context) -> bool {
        ctx.lookup_has_float(*self)
    }
}
impl HasFloat for Item {
    fn has_float(&self, ctx: &Context) -> bool {
        ctx.lookup_has_float(self.id())
    }
}

pub trait HasTypeParam {
    fn has_type_param_in_array(&self, ctx: &Context) -> bool;
}
impl<T> HasTypeParam for T
where
    T: Copy + Into<ItemId>,
{
    fn has_type_param_in_array(&self, ctx: &Context) -> bool {
        ctx.lookup_has_type_param_in_array(*self)
    }
}
impl HasTypeParam for Item {
    fn has_type_param_in_array(&self, ctx: &Context) -> bool {
        ctx.lookup_has_type_param_in_array(self.id())
    }
}

impl<T> HasVtable for T
where
    T: Copy + Into<ItemId>,
{
    fn has_vtable(&self, ctx: &Context) -> bool {
        let id: ItemId = (*self).into();
        id.as_type_id(ctx)
            .map_or(false, |x| !matches!(ctx.lookup_has_vtable(x), has_vtable::Resolved::No))
    }
    fn has_vtable_ptr(&self, ctx: &Context) -> bool {
        let id: ItemId = (*self).into();
        id.as_type_id(ctx).map_or(false, |x| {
            matches!(ctx.lookup_has_vtable(x), has_vtable::Resolved::SelfHasVtable)
        })
    }
}
impl HasVtable for Item {
    fn has_vtable(&self, ctx: &Context) -> bool {
        self.id().has_vtable(ctx)
    }
    fn has_vtable_ptr(&self, ctx: &Context) -> bool {
        self.id().has_vtable_ptr(ctx)
    }
}

impl<T> Sizedness for T
where
    T: Copy + Into<ItemId>,
{
    fn sizedness(&self, ctx: &Context) -> sizedness::Resolved {
        let id: ItemId = (*self).into();
        id.as_type_id(ctx)
            .map_or(sizedness::Resolved::default(), |x| ctx.lookup_sizedness(x))
    }
}
impl Sizedness for Item {
    fn sizedness(&self, ctx: &Context) -> sizedness::Resolved {
        self.id().sizedness(ctx)
    }
}

pub trait Ancestors {
    fn ancestors<'a>(&self, ctx: &'a Context) -> AncestorsIter<'a>;
}
impl<T> Ancestors for T
where
    T: Copy + Into<ItemId>,
{
    fn ancestors<'a>(&self, ctx: &'a Context) -> AncestorsIter<'a> {
        AncestorsIter::new(ctx, *self)
    }
}
impl Ancestors for Item {
    fn ancestors<'a>(&self, ctx: &'a Context) -> AncestorsIter<'a> {
        self.id().ancestors(ctx)
    }
}

pub type ItemSet = BTreeSet<ItemId>;

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

pub struct AncestorsIter<'a> {
    item: ItemId,
    ctx: &'a Context,
    seen: DebugOnlyItemSet,
}
impl<'a> AncestorsIter<'a> {
    fn new<Id: Into<ItemId>>(ctx: &'a Context, id: Id) -> Self {
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

#[derive(Debug)]
pub struct Item {
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
impl Item {
    pub fn new(
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
    pub fn new_opaque_type(with_id: ItemId, ty: &clang::Type, ctx: &mut Context) -> TypeId {
        let location = ty.declaration().location();
        let ty = Opaque::from_clang_ty(ty, ctx);
        let kind = ItemKind::Type(ty);
        let parent = ctx.root_module().into();
        ctx.add_item(Item::new(with_id, None, None, parent, kind, Some(location)), None, None);
        with_id.as_type_id_unchecked()
    }
    pub fn id(&self) -> ItemId {
        self.id
    }
    pub fn parent_id(&self) -> ItemId {
        self.parent_id
    }
    pub fn set_parent_for_replacement<Id: Into<ItemId>>(&mut self, id: Id) {
        self.parent_id = id.into();
    }
    pub fn codegen_depth(&self, ctx: &Context) -> usize {
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
    pub fn comment(&self, ctx: &Context) -> Option<String> {
        if !ctx.opts().generate_comments {
            return None;
        }
        self.comment.as_ref().map(|comment| ctx.opts().process_comment(comment))
    }
    pub fn kind(&self) -> &ItemKind {
        &self.kind
    }
    pub fn kind_mut(&mut self) -> &mut ItemKind {
        &mut self.kind
    }
    pub fn location(&self) -> Option<&clang::SrcLoc> {
        self.location.as_ref()
    }
    pub fn local_id(&self, ctx: &Context) -> usize {
        *self.local_id.borrow_with(|| {
            let parent = ctx.resolve_item(self.parent_id);
            parent.next_child_local_id()
        })
    }
    pub fn next_child_local_id(&self) -> usize {
        let local_id = self.next_child_local_id.get();
        self.next_child_local_id.set(local_id + 1);
        local_id
    }
    pub fn is_toplevel(&self, ctx: &Context) -> bool {
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
    pub fn expect_type(&self) -> &Type {
        self.kind().expect_type()
    }
    pub fn as_type(&self) -> Option<&Type> {
        self.kind().as_type()
    }
    pub fn expect_function(&self) -> &Function {
        self.kind().expect_function()
    }
    pub fn is_module(&self) -> bool {
        matches!(self.kind, ItemKind::Module(..))
    }
    pub fn annotations(&self) -> &Annotations {
        &self.annotations
    }
    pub fn is_blocklisted(&self, ctx: &Context) -> bool {
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
    pub fn name<'a>(&'a self, ctx: &'a Context) -> NameOpts<'a> {
        NameOpts::new(self, ctx)
    }
    fn name_target(&self, ctx: &Context) -> ItemId {
        let mut targets_seen = DebugOnlyItemSet::new();
        let mut i = self;
        loop {
            targets_seen.insert(i.id());
            if self.annotations().use_instead_of().is_some() {
                return self.id();
            }
            match *i.kind() {
                ItemKind::Type(ref ty) => match *ty.kind() {
                    TypeKind::ResolvedTypeRef(inner) => {
                        i = ctx.resolve_item(inner);
                    },
                    TypeKind::TemplInstantiation(ref x) => {
                        i = ctx.resolve_item(x.templ_def());
                    },
                    _ => return i.id(),
                },
                _ => return i.id(),
            }
        }
    }
    pub fn full_disambiguated_name(&self, ctx: &Context) -> String {
        let mut s = String::new();
        let level = 0;
        self.push_disambiguated_name(ctx, &mut s, level);
        s
    }
    fn push_disambiguated_name(&self, ctx: &Context, to: &mut String, level: u8) {
        to.push_str(&self.canon_name(ctx));
        if let ItemKind::Type(ref ty) = *self.kind() {
            if let TypeKind::TemplInstantiation(ref x) = *ty.kind() {
                to.push_str(&format!("_open{}_", level));
                for arg in x.templ_args() {
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
    fn overload_index(&self, ctx: &Context) -> Option<usize> {
        self.func_name().and_then(|func_name| {
            let parent = ctx.resolve_item(self.parent_id());
            if let ItemKind::Type(ref ty) = *parent.kind() {
                if let TypeKind::Comp(ref ci) = *ty.kind() {
                    return ci.constructors().iter().position(|c| *c == self.id()).or_else(|| {
                        ci.methods()
                            .iter()
                            .filter(|m| {
                                let item = ctx.resolve_item(m.sig());
                                let func = item.expect_function();
                                func.name() == func_name
                            })
                            .position(|m| m.sig() == self.id())
                    });
                }
            }
            None
        })
    }
    fn base_name(&self, ctx: &Context) -> String {
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
    pub fn real_canonical_name(&self, ctx: &Context, opt: &NameOpts) -> String {
        let target = ctx.resolve_item(self.name_target(ctx));
        if let Some(path) = target.annotations.use_instead_of() {
            if ctx.opts().enable_cxx_namespaces {
                return path.last().unwrap().clone();
            }
            return path.join("_");
        }
        let base_name = target.base_name(ctx);
        if target.is_templ_param(ctx, &()) {
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
    pub fn exposed_id(&self, ctx: &Context) -> String {
        let ty_kind = self.kind().as_type().map(|t| t.kind());
        if let Some(ty_kind) = ty_kind {
            match *ty_kind {
                TypeKind::Comp(..) | TypeKind::TemplInstantiation(..) | TypeKind::Enum(..) => {
                    return self.local_id(ctx).to_string()
                },
                _ => {},
            }
        }
        format!("id_{}", self.id().as_usize())
    }
    pub fn as_module(&self) -> Option<&Module> {
        match self.kind {
            ItemKind::Module(ref module) => Some(module),
            _ => None,
        }
    }
    pub fn as_module_mut(&mut self) -> Option<&mut Module> {
        match self.kind {
            ItemKind::Module(ref mut module) => Some(module),
            _ => None,
        }
    }
    fn is_constified_enum_module(&self, ctx: &Context) -> bool {
        let item = self.id.into_resolver().through_type_refs().resolve(ctx);
        let type_ = match *item.kind() {
            ItemKind::Type(ref type_) => type_,
            _ => return false,
        };
        match *type_.kind() {
            TypeKind::Enum(ref enum_) => enum_.computed_enum_variation(ctx, self) == variation::Enum::ModuleConsts,
            TypeKind::Alias(inner_id) => {
                let inner_item = ctx.resolve_item(inner_id);
                let name = item.canon_name(ctx);
                if inner_item.canon_name(ctx) == name {
                    inner_item.is_constified_enum_module(ctx)
                } else {
                    false
                }
            },
            _ => false,
        }
    }
    pub fn is_enabled_for_codegen(&self, ctx: &Context) -> bool {
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
    pub fn path_for_allowlisting(&self, ctx: &Context) -> &Vec<String> {
        self.path_for_allowlisting
            .borrow_with(|| self.compute_path(ctx, UserMangled::No))
    }
    fn compute_path(&self, ctx: &Context, mangled: UserMangled) -> Vec<String> {
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
            TypeKind::Comp(ref ci) => match ci.kind() {
                CompKind::Struct => "struct",
                CompKind::Union => "union",
            },
            TypeKind::Enum(..) => "enum",
            _ => return None,
        })
    }
    pub fn must_use(&self, ctx: &Context) -> bool {
        self.annotations().must_use_type() || ctx.must_use_type_by_name(self)
    }
    pub fn builtin_type(kind: TypeKind, is_const: bool, ctx: &mut Context) -> TypeId {
        match kind {
            TypeKind::Void | TypeKind::Int(..) | TypeKind::Pointer(..) | TypeKind::Float(..) => {},
            _ => panic!("Unsupported builtin type"),
        }
        let ty = Type::new(None, None, kind, is_const);
        let id = ctx.next_item_id();
        let module = ctx.root_module().into();
        ctx.add_item(Item::new(id, None, None, module, ItemKind::Type(ty), None), None, None);
        id.as_type_id_unchecked()
    }
    pub fn parse(cur: clang::Cursor, parent_id: Option<ItemId>, ctx: &mut Context) -> Result<ItemId, parse::Error> {
        use crate::ir::var::Var;
        use clang_lib::*;
        if !cur.is_valid() {
            return Err(parse::Error::Continue);
        }
        let comment = cur.raw_comment();
        let annotations = Annotations::new(&cur);
        let current_module = ctx.current_module().into();
        let relevant_parent_id = parent_id.unwrap_or(current_module);
        macro_rules! try_parse {
            ($what:ident) => {
                match $what::parse(cur, ctx) {
                    Ok(parse::Resolved::New(item, decl)) => {
                        let id = ctx.next_item_id();
                        ctx.add_item(
                            Item::new(
                                id,
                                comment,
                                annotations,
                                relevant_parent_id,
                                ItemKind::$what(item),
                                Some(cur.location()),
                            ),
                            decl,
                            Some(cur),
                        );
                        return Ok(id);
                    },
                    Ok(parse::Resolved::AlreadyResolved(id)) => {
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
            let definition = cur.definition();
            let applicable_cursor = definition.unwrap_or(cur);
            let relevant_parent_id = match definition {
                Some(definition) => {
                    if definition != cur {
                        ctx.add_semantic_parent(definition, relevant_parent_id);
                        return Ok(Item::from_ty_or_ref(applicable_cursor.cur_type(), cur, parent_id, ctx).into());
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
        if cur.kind() == CXCursor_UnexposedDecl {
            Err(parse::Error::Recurse)
        } else {
            match cur.kind() {
                CXCursor_MacroDefinition
                | CXCursor_MacroExpansion
                | CXCursor_UsingDeclaration
                | CXCursor_UsingDirective
                | CXCursor_StaticAssert
                | CXCursor_FunctionTemplate => {
                    debug!("Unhandled cursor kind {:?}: {:?}", cur.kind(), cur);
                },
                CXCursor_InclusionDirective => {
                    let file = cur.get_included_file_name();
                    match file {
                        None => {
                            warn!("Inclusion of a nameless file in {:?}", cur);
                        },
                        Some(filename) => {
                            ctx.include_file(filename);
                        },
                    }
                },
                _ => {
                    let spelling = cur.spelling();
                    if !spelling.starts_with("operator") {
                        warn!("Unhandled cursor kind {:?}: {:?}", cur.kind(), cur);
                    }
                },
            }
            Err(parse::Error::Continue)
        }
    }
    pub fn from_ty_or_ref(
        ty: clang::Type,
        location: clang::Cursor,
        parent_id: Option<ItemId>,
        ctx: &mut Context,
    ) -> TypeId {
        let id = ctx.next_item_id();
        Self::from_ty_or_ref_with_id(id, ty, location, parent_id, ctx)
    }
    pub fn from_ty_or_ref_with_id(
        potential_id: ItemId,
        ty: clang::Type,
        location: clang::Cursor,
        parent_id: Option<ItemId>,
        ctx: &mut Context,
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
        let kind = TypeKind::UnresolvedTypeRef(ty, location, parent_id);
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
    pub fn from_ty(
        ty: &clang::Type,
        location: clang::Cursor,
        parent_id: Option<ItemId>,
        ctx: &mut Context,
    ) -> Result<TypeId, parse::Error> {
        let id = ctx.next_item_id();
        Item::from_ty_with_id(id, ty, location, parent_id, ctx)
    }
    pub fn from_ty_with_id(
        id: ItemId,
        ty: &clang::Type,
        location: clang::Cursor,
        parent_id: Option<ItemId>,
        ctx: &mut Context,
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
            Ok(parse::Resolved::AlreadyResolved(ty)) => Ok(ty.as_type_id_unchecked()),
            Ok(parse::Resolved::New(item, declaration)) => {
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
    pub fn type_param(with_id: Option<ItemId>, location: clang::Cursor, ctx: &mut Context) -> Option<TypeId> {
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
        fn is_templ_with_spelling(refd: &clang::Cursor, spelling: &str) -> bool {
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
        let definition = if is_templ_with_spelling(&location, &ty_spelling) {
            location
        } else if location.kind() == clang_lib::CXCursor_TypeRef {
            match location.referenced() {
                Some(refd) if is_templ_with_spelling(&refd, &ty_spelling) => refd,
                _ => return None,
            }
        } else {
            let mut definition = None;
            location.visit(|child| {
                let child_ty = child.cur_type();
                if child_ty.kind() == clang_lib::CXCursor_TypeRef && child_ty.spelling() == ty_spelling {
                    match child.referenced() {
                        Some(refd) if is_templ_with_spelling(&refd, &ty_spelling) => {
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
        assert!(is_templ_with_spelling(&definition, &ty_spelling));
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
impl DotAttrs for Item {
    fn dot_attrs<W>(&self, ctx: &Context, y: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        writeln!(
            y,
            "<tr><td>{:?}</td></tr>
                       <tr><td>name</td><td>{}</td></tr>",
            self.id,
            self.name(ctx).get()
        )?;
        if self.is_opaque(ctx, &()) {
            writeln!(y, "<tr><td>opaque</td><td>true</td></tr>")?;
        }
        self.kind.dot_attrs(ctx, y)
    }
}
impl AsRef<ItemId> for Item {
    fn as_ref(&self) -> &ItemId {
        &self.id
    }
}
impl TemplParams for Item {
    fn self_templ_params(&self, ctx: &Context) -> Vec<TypeId> {
        self.kind.self_templ_params(ctx)
    }
}
impl AsTemplParam for Item {
    type Extra = ();
    fn as_templ_param(&self, ctx: &Context, _: &()) -> Option<TypeId> {
        self.kind.as_templ_param(ctx, self)
    }
}
impl TemplParams for ItemKind {
    fn self_templ_params(&self, ctx: &Context) -> Vec<TypeId> {
        match *self {
            ItemKind::Type(ref x) => x.self_templ_params(ctx),
            ItemKind::Function(_) | ItemKind::Module(_) | ItemKind::Var(_) => {
                vec![]
            },
        }
    }
}
impl AsTemplParam for ItemKind {
    type Extra = Item;
    fn as_templ_param(&self, ctx: &Context, it: &Item) -> Option<TypeId> {
        match *self {
            ItemKind::Type(ref x) => x.as_templ_param(ctx, it),
            ItemKind::Module(..) | ItemKind::Function(..) | ItemKind::Var(..) => None,
        }
    }
}
impl Trace for Item {
    type Extra = ();
    fn trace<T>(&self, ctx: &Context, tracer: &mut T, _extra: &())
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
                tracer.visit(fun.sig().into());
            },
            ItemKind::Var(ref var) => {
                tracer.visit_kind(var.ty().into(), EdgeKind::VarType);
            },
            ItemKind::Module(_) => {},
        }
    }
}

fn visit_child(
    cur: clang::Cursor,
    id: ItemId,
    ty: &clang::Type,
    parent_id: Option<ItemId>,
    ctx: &mut Context,
    y: &mut Result<TypeId, parse::Error>,
) -> clang_lib::CXChildVisitResult {
    use clang_lib::*;
    if y.is_ok() {
        return CXChildVisit_Break;
    }
    *y = Item::from_ty_with_id(id, ty, cur, parent_id, ctx);
    match *y {
        Ok(..) => CXChildVisit_Break,
        Err(parse::Error::Recurse) => {
            cur.visit(|c| visit_child(c, id, ty, parent_id, ctx, y));
            CXChildVisit_Continue
        },
        Err(parse::Error::Continue) => CXChildVisit_Continue,
    }
}
