use super::super::codegen::{utils::variation, CONSTIFIED_ENUM_MODULE_REPR_NAME};
use super::analysis::{HasVtable, Sizedness, *};
use super::annos::Annotations;
use super::comp::{CompKind, MethKind};
use super::ctx::PartialType;
use super::dot::DotAttrs;
use super::func::{FnKind, Func};
use super::layout::Opaque;
use super::module::Mod;
use super::template::{AsTemplParam, TemplParams};
use super::typ::{Type, TypeKind};
use super::{Context, EdgeKind, ItemId, ItemKind, Trace, Tracer, TypeId};
use crate::clang;
use crate::parse::{self, SubItem};
use lazycell::LazyCell;
use std::cell::Cell;
use std::collections::BTreeSet;
use std::fmt::Write;
use std::io;
use std::iter;

#[derive(Debug)]
pub struct NameOpts<'a> {
    item: &'a Item,
    ctx: &'a Context,
    namespaces: bool,
    mangled: bool,
}
impl<'a> NameOpts<'a> {
    pub fn new(item: &'a Item, ctx: &'a Context) -> Self {
        NameOpts {
            item,
            ctx,
            namespaces: false,
            mangled: true,
        }
    }
    pub fn within_namespaces(&mut self) -> &mut Self {
        self.namespaces = true;
        self
    }
    fn mangled(&mut self, x: bool) -> &mut Self {
        self.mangled = x;
        self
    }
    pub fn get(&self) -> String {
        self.item.real_canon_name(self.ctx, self)
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
        self.canon_name
            .borrow_with(|| {
                let x = ctx.opts().enable_cxx_namespaces || ctx.opts().disable_name_namespacing;
                if x {
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
        self.compute_path(ctx, true)
    }
    fn namespace_aware_canon_path(&self, ctx: &Context) -> Vec<String> {
        let mut y = self.canon_path(ctx);
        if ctx.opts().disable_name_namespacing {
            let idx = y.len() - 1;
            y = y.split_off(idx);
        } else if !ctx.opts().enable_cxx_namespaces {
            y = vec![y[1..].join("_")];
        }
        if self.is_constified_enum_mod(ctx) {
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
        self.annos.opaque()
            || self.as_type().map_or(false, |x| x.is_opaque(ctx, self))
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
    fn contains(&self, _: &ItemId) -> bool {
        false
    }
    fn insert(&mut self, _: ItemId) {}
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
        if item.parent() == self.item {
            None
        } else {
            self.item = item.parent();
            self.seen.insert(item.id());
            Some(item.id())
        }
    }
}

#[derive(Debug)]
pub struct Item {
    id: ItemId,
    local: LazyCell<usize>,
    next_child_local: Cell<usize>,
    canon_name: LazyCell<String>,
    path_for_allowlisting: LazyCell<Vec<String>>,
    comment: Option<String>,
    annos: Annotations,
    parent: ItemId,
    kind: ItemKind,
    loc: Option<clang::SrcLoc>,
}
impl Item {
    pub fn new(
        id: ItemId,
        comment: Option<String>,
        annos: Option<Annotations>,
        parent: ItemId,
        kind: ItemKind,
        loc: Option<clang::SrcLoc>,
    ) -> Self {
        Item {
            id,
            local: LazyCell::new(),
            next_child_local: Cell::new(1),
            canon_name: LazyCell::new(),
            path_for_allowlisting: LazyCell::new(),
            parent,
            comment,
            annos: annos.unwrap_or_default(),
            kind,
            loc,
        }
    }
    pub fn new_opaque_type(id: ItemId, ty: &clang::Type, ctx: &mut Context) -> TypeId {
        let loc = ty.decl().location();
        let ty = Opaque::from_clang_ty(ty, ctx);
        let kind = ItemKind::Type(ty);
        let parent = ctx.root_mod().into();
        ctx.add_item(Item::new(id, None, None, parent, kind, Some(loc)), None, None);
        id.as_type_id_unchecked()
    }
    pub fn id(&self) -> ItemId {
        self.id
    }
    pub fn parent(&self) -> ItemId {
        self.parent
    }
    pub fn set_parent_for_replacement<Id: Into<ItemId>>(&mut self, id: Id) {
        self.parent = id.into();
    }
    pub fn codegen_depth(&self, ctx: &Context) -> usize {
        if !ctx.opts().enable_cxx_namespaces {
            return 0;
        }
        self.ancestors(ctx)
            .filter(|x| {
                ctx.resolve_item(*x)
                    .as_mod()
                    .map_or(false, |x| !x.is_inline() || ctx.opts().conservative_inline_namespaces)
            })
            .count()
            + 1
    }
    pub fn comment(&self, ctx: &Context) -> Option<String> {
        if !ctx.opts().generate_comments {
            return None;
        }
        self.comment.as_ref().map(|x| ctx.opts().process_comment(x))
    }
    pub fn kind(&self) -> &ItemKind {
        &self.kind
    }
    pub fn kind_mut(&mut self) -> &mut ItemKind {
        &mut self.kind
    }
    pub fn loc(&self) -> Option<&clang::SrcLoc> {
        self.loc.as_ref()
    }
    pub fn local(&self, ctx: &Context) -> usize {
        *self.local.borrow_with(|| {
            let y = ctx.resolve_item(self.parent);
            y.next_child_local()
        })
    }
    pub fn next_child_local(&self) -> usize {
        let y = self.next_child_local.get();
        self.next_child_local.set(y + 1);
        y
    }
    pub fn is_toplevel(&self, ctx: &Context) -> bool {
        if ctx.opts().enable_cxx_namespaces && self.kind().is_mod() && self.id() != ctx.root_mod() {
            return false;
        }
        let mut y = self.parent;
        loop {
            let x = match ctx.resolve_item_fallible(y) {
                Some(x) => x,
                None => return false,
            };
            if x.id() == ctx.root_mod() {
                return true;
            } else if ctx.opts().enable_cxx_namespaces || !x.kind().is_mod() {
                return false;
            }
            y = x.parent();
        }
    }
    pub fn expect_type(&self) -> &Type {
        self.kind().expect_type()
    }
    pub fn as_type(&self) -> Option<&Type> {
        self.kind().as_type()
    }
    pub fn expect_fn(&self) -> &Func {
        self.kind().expect_fn()
    }
    pub fn is_mod(&self) -> bool {
        matches!(self.kind, ItemKind::Mod(..))
    }
    pub fn annos(&self) -> &Annotations {
        &self.annos
    }
    pub fn is_blocklisted(&self, ctx: &Context) -> bool {
        if self.annos.hide() {
            return true;
        }
        if !ctx.opts().blocklisted_files.is_empty() {
            if let Some(x) = &self.loc {
                let (x, _, _, _) = x.location();
                if let Some(x) = x.name() {
                    if ctx.opts().blocklisted_files.matches(x) {
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
                ItemKind::Func(..) => ctx.opts().blocklisted_fns.matches(&name),
                ItemKind::Var(..) | ItemKind::Mod(..) => false,
            }
    }
    pub fn name<'a>(&'a self, ctx: &'a Context) -> NameOpts<'a> {
        NameOpts::new(self, ctx)
    }
    fn name_target(&self, ctx: &Context) -> ItemId {
        let mut i = self;
        loop {
            if self.annos().use_instead_of().is_some() {
                return self.id();
            }
            match *i.kind() {
                ItemKind::Type(ref x) => match *x.kind() {
                    TypeKind::ResolvedRef(x) => {
                        i = ctx.resolve_item(x);
                    },
                    TypeKind::TemplInst(ref x) => {
                        i = ctx.resolve_item(x.templ_def());
                    },
                    _ => return i.id(),
                },
                _ => return i.id(),
            }
        }
    }
    pub fn full_disambiguated_name(&self, ctx: &Context) -> String {
        let mut y = String::new();
        self.push_disambiguated_name(ctx, &mut y, 0);
        y
    }
    fn push_disambiguated_name(&self, ctx: &Context, to: &mut String, level: u8) {
        to.push_str(&self.canon_name(ctx));
        if let ItemKind::Type(ref x) = *self.kind() {
            if let TypeKind::TemplInst(ref x) = *x.kind() {
                to.push_str(&format!("_open{}_", level));
                for y in x.templ_args() {
                    y.into_resolver()
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
            ItemKind::Func(ref x) => Some(x.name()),
            _ => None,
        }
    }
    fn overload_idx(&self, ctx: &Context) -> Option<usize> {
        self.func_name().and_then(|name| {
            let parent = ctx.resolve_item(self.parent());
            if let ItemKind::Type(ref x) = *parent.kind() {
                if let TypeKind::Comp(ref x) = *x.kind() {
                    return x.constrs().iter().position(|x| *x == self.id()).or_else(|| {
                        x.methods()
                            .iter()
                            .filter(|x| {
                                let x = ctx.resolve_item(x.sig());
                                let x = x.expect_fn();
                                x.name() == name
                            })
                            .position(|x| x.sig() == self.id())
                    });
                }
            }
            None
        })
    }
    fn base_name(&self, ctx: &Context) -> String {
        if let Some(x) = self.annos().use_instead_of() {
            return x.last().unwrap().clone();
        }
        match *self.kind() {
            ItemKind::Var(ref x) => x.name().to_owned(),
            ItemKind::Mod(ref x) => x
                .name()
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("_bindgen_mod_{}", self.exposed_id(ctx))),
            ItemKind::Type(ref x) => x
                .sanitized_name(ctx)
                .map(Into::into)
                .unwrap_or_else(|| format!("_bindgen_ty_{}", self.exposed_id(ctx))),
            ItemKind::Func(ref x) => {
                let mut name = x.name().to_owned();
                if let Some(x) = self.overload_idx(ctx) {
                    if x > 0 {
                        write!(&mut name, "{}", x).unwrap();
                    }
                }
                name
            },
        }
    }
    fn is_anon(&self) -> bool {
        match self.kind() {
            ItemKind::Mod(x) => x.name().is_none(),
            ItemKind::Type(x) => x.name().is_none(),
            ItemKind::Func(_) => false,
            ItemKind::Var(_) => false,
        }
    }
    pub fn real_canon_name(&self, ctx: &Context, opt: &NameOpts) -> String {
        let target = ctx.resolve_item(self.name_target(ctx));
        if let Some(x) = target.annos.use_instead_of() {
            if ctx.opts().enable_cxx_namespaces {
                return x.last().unwrap().clone();
            }
            return x.join("_");
        }
        let base_name = target.base_name(ctx);
        if target.is_templ_param(ctx, &()) {
            return base_name;
        }
        let mut ids_iter = target
            .parent()
            .ancestors(ctx)
            .filter(|x| *x != ctx.root_mod())
            .take_while(|x| !opt.namespaces || !ctx.resolve_item(*x).is_mod())
            .filter(|x| {
                if !ctx.opts().conservative_inline_namespaces {
                    if let ItemKind::Mod(ref x) = *ctx.resolve_item(*x).kind() {
                        return !x.is_inline();
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
            .map(|x| {
                let x = ctx.resolve_item(x);
                let x = ctx.resolve_item(x.name_target(ctx));
                x.base_name(ctx)
            })
            .filter(|x| !x.is_empty())
            .collect();
        names.reverse();
        if !base_name.is_empty() {
            names.push(base_name);
        }
        if ctx.opts().c_naming {
            if let Some(x) = self.c_naming_prefix() {
                names.insert(0, x.to_string());
            }
        }
        let name = names.join("_");
        let name = if opt.mangled {
            ctx.opts().last_callback(|x| x.item_name(&name)).unwrap_or(name)
        } else {
            name
        };
        ctx.rust_mangle(&name).into_owned()
    }
    pub fn exposed_id(&self, ctx: &Context) -> String {
        let y = self.kind().as_type().map(|x| x.kind());
        if let Some(x) = y {
            match *x {
                TypeKind::Comp(..) | TypeKind::TemplInst(..) | TypeKind::Enum(..) => {
                    return self.local(ctx).to_string()
                },
                _ => {},
            }
        }
        format!("id_{}", self.id().as_usize())
    }
    pub fn as_mod(&self) -> Option<&Mod> {
        match self.kind {
            ItemKind::Mod(ref x) => Some(x),
            _ => None,
        }
    }
    pub fn as_mod_mut(&mut self) -> Option<&mut Mod> {
        match self.kind {
            ItemKind::Mod(ref mut x) => Some(x),
            _ => None,
        }
    }
    fn is_constified_enum_mod(&self, ctx: &Context) -> bool {
        let item = self.id.into_resolver().through_type_refs().resolve(ctx);
        let y = match *item.kind() {
            ItemKind::Type(ref x) => x,
            _ => return false,
        };
        match *y.kind() {
            TypeKind::Enum(ref x) => x.computed_enum_variation(ctx, self) == variation::Enum::ModConsts,
            TypeKind::Alias(x) => {
                let x = ctx.resolve_item(x);
                let name = item.canon_name(ctx);
                if x.canon_name(ctx) == name {
                    x.is_constified_enum_mod(ctx)
                } else {
                    false
                }
            },
            _ => false,
        }
    }
    pub fn is_enabled_for_gen(&self, ctx: &Context) -> bool {
        let y = &ctx.opts().config;
        match *self.kind() {
            ItemKind::Mod(..) => true,
            ItemKind::Var(_) => y.vars(),
            ItemKind::Type(_) => y.typs(),
            ItemKind::Func(ref f) => match f.kind() {
                FnKind::Func => y.fns(),
                FnKind::Method(MethKind::Constr) => y.constrs(),
                FnKind::Method(MethKind::Destr) | FnKind::Method(MethKind::VirtDestr { .. }) => y.destrs(),
                FnKind::Method(MethKind::Static)
                | FnKind::Method(MethKind::Normal)
                | FnKind::Method(MethKind::Virt { .. }) => y.methods(),
            },
        }
    }
    pub fn path_for_allowlisting(&self, ctx: &Context) -> &Vec<String> {
        self.path_for_allowlisting.borrow_with(|| self.compute_path(ctx, false))
    }
    fn compute_path(&self, ctx: &Context, mangled: bool) -> Vec<String> {
        if let Some(x) = self.annos().use_instead_of() {
            let mut y = vec![ctx.resolve_item(ctx.root_mod()).name(ctx).get()];
            y.extend_from_slice(x);
            return y;
        }
        let target = ctx.resolve_item(self.name_target(ctx));
        let mut y: Vec<_> = target
            .ancestors(ctx)
            .chain(iter::once(ctx.root_mod().into()))
            .map(|x| ctx.resolve_item(x))
            .filter(|x| {
                x.id() == target.id()
                    || x.as_mod()
                        .map_or(false, |x| !x.is_inline() || ctx.opts().conservative_inline_namespaces)
            })
            .map(|x| {
                ctx.resolve_item(x.name_target(ctx))
                    .name(ctx)
                    .within_namespaces()
                    .mangled(mangled)
                    .get()
            })
            .collect();
        y.reverse();
        y
    }
    fn c_naming_prefix(&self) -> Option<&str> {
        let ty = match self.kind {
            ItemKind::Type(ref x) => x,
            _ => return None,
        };
        Some(match ty.kind() {
            TypeKind::Comp(ref x) => match x.kind() {
                CompKind::Struct => "struct",
                CompKind::Union => "union",
            },
            TypeKind::Enum(..) => "enum",
            _ => return None,
        })
    }
    pub fn must_use(&self, ctx: &Context) -> bool {
        self.annos().must_use_type() || ctx.must_use_type_by_name(self)
    }
    pub fn builtin_type(kind: TypeKind, is_const: bool, ctx: &mut Context) -> TypeId {
        match kind {
            TypeKind::Void | TypeKind::Int(..) | TypeKind::Pointer(..) | TypeKind::Float(..) => {},
            _ => panic!("Unsupported builtin type"),
        }
        let ty = Type::new(None, None, kind, is_const);
        let id = ctx.next_item_id();
        let module = ctx.root_mod().into();
        ctx.add_item(Item::new(id, None, None, module, ItemKind::Type(ty), None), None, None);
        id.as_type_id_unchecked()
    }
    pub fn parse(cur: clang::Cursor, parent: Option<ItemId>, ctx: &mut Context) -> Result<ItemId, parse::Error> {
        use crate::ir::var::Var;
        use clang_lib::*;
        if !cur.is_valid() {
            return Err(parse::Error::Continue);
        }
        let comment = cur.raw_comment();
        let annos = Annotations::new(&cur);
        let current_mod = ctx.current_mod().into();
        let relevant_parent_id = parent.unwrap_or(current_mod);
        macro_rules! try_parse {
            ($what:ident) => {
                match $what::parse(cur, ctx) {
                    Ok(parse::Resolved::New(item, decl)) => {
                        let id = ctx.next_item_id();
                        ctx.add_item(
                            Item::new(
                                id,
                                comment,
                                annos,
                                relevant_parent_id,
                                ItemKind::$what(item),
                                Some(cur.location()),
                            ),
                            decl,
                            Some(cur),
                        );
                        return Ok(id);
                    },
                    Ok(parse::Resolved::AlreadyDone(id)) => {
                        return Ok(id);
                    },
                    Err(parse::Error::Recurse) => return Err(parse::Error::Recurse),
                    Err(parse::Error::Continue) => {},
                }
            };
        }
        try_parse!(Mod);
        try_parse!(Func);
        try_parse!(Var);
        {
            let definition = cur.definition();
            let applicable_cursor = definition.unwrap_or(cur);
            let relevant_parent_id = match definition {
                Some(definition) => {
                    if definition != cur {
                        ctx.add_semantic_parent(definition, relevant_parent_id);
                        return Ok(Item::from_ty_or_ref(applicable_cursor.cur_type(), cur, parent, ctx).into());
                    }
                    ctx.known_semantic_parent(definition)
                        .or(parent)
                        .unwrap_or_else(|| ctx.current_mod().into())
                },
                None => relevant_parent_id,
            };
            match Item::from_ty(
                &applicable_cursor.cur_type(),
                applicable_cursor,
                Some(relevant_parent_id),
                ctx,
            ) {
                Ok(x) => return Ok(x.into()),
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
                    warn!("Unhandled cursor kind {:?}: {:?}", cur.kind(), cur);
                },
                CXCursor_InclusionDirective => {
                    let file = cur.get_included_file_name();
                    match file {
                        None => {
                            warn!("Inclusion of a nameless file in {:?}", cur);
                        },
                        Some(x) => {
                            ctx.include_file(x);
                        },
                    }
                },
                _ => {
                    let x = cur.spelling();
                    if !x.starts_with("operator") {
                        warn!("Unhandled cursor kind {:?}: {:?}", cur.kind(), cur);
                    }
                },
            }
            Err(parse::Error::Continue)
        }
    }
    pub fn from_ty_or_ref(ty: clang::Type, cur: clang::Cursor, parent: Option<ItemId>, ctx: &mut Context) -> TypeId {
        let id = ctx.next_item_id();
        Self::from_ty_or_ref_with_id(id, ty, cur, parent, ctx)
    }
    pub fn from_ty_or_ref_with_id(
        id: ItemId,
        ty: clang::Type,
        cur: clang::Cursor,
        parent: Option<ItemId>,
        ctx: &mut Context,
    ) -> TypeId {
        if ctx.collected_typerefs() {
            return Item::from_ty_with_id(id, &ty, cur, parent, ctx)
                .unwrap_or_else(|_| Item::new_opaque_type(id, &ty, ctx));
        }
        if let Some(x) = ctx.builtin_or_resolved_ty(id, parent, &ty, Some(cur)) {
            return x;
        }
        let is_const = ty.is_const();
        let kind = TypeKind::UnresolvedRef(ty, cur, parent);
        let x = ctx.current_mod();
        ctx.add_item(
            Item::new(
                id,
                None,
                None,
                parent.unwrap_or_else(|| x.into()),
                ItemKind::Type(Type::new(None, None, kind, is_const)),
                Some(cur.location()),
            ),
            None,
            None,
        );
        id.as_type_id_unchecked()
    }
    pub fn from_ty(
        ty: &clang::Type,
        cur: clang::Cursor,
        parent: Option<ItemId>,
        ctx: &mut Context,
    ) -> Result<TypeId, parse::Error> {
        let id = ctx.next_item_id();
        Item::from_ty_with_id(id, ty, cur, parent, ctx)
    }
    pub fn from_ty_with_id(
        id: ItemId,
        ty: &clang::Type,
        cur: clang::Cursor,
        parent: Option<ItemId>,
        ctx: &mut Context,
    ) -> Result<TypeId, parse::Error> {
        use clang_lib::*;
        if ty.kind() == clang_lib::CXType_Unexposed || cur.cur_type().kind() == clang_lib::CXType_Unexposed {
            if ty.is_associated_type() || cur.cur_type().is_associated_type() {
                return Ok(Item::new_opaque_type(id, ty, ctx));
            }
            if let Some(x) = Item::type_param(None, cur, ctx) {
                return Ok(ctx.build_ty_wrapper(id, x, None, ty));
            }
        }
        if let Some(ref x) = ty.decl().fallible_semantic_parent() {
            if FnKind::from_cursor(x).is_some() {
                return Ok(Item::new_opaque_type(id, ty, ctx));
            }
        }
        let decl = {
            let x = ty.canon_type().decl().definition();
            x.unwrap_or_else(|| ty.decl())
        };
        let comment = cur
            .raw_comment()
            .or_else(|| decl.raw_comment())
            .or_else(|| cur.raw_comment());
        let annos = Annotations::new(&decl).or_else(|| Annotations::new(&cur));
        if let Some(ref x) = annos {
            if let Some(x) = x.use_instead_of() {
                ctx.replace(x, id);
            }
        }
        if let Some(x) = ctx.builtin_or_resolved_ty(id, parent, ty, Some(cur)) {
            return Ok(x);
        }
        let mut valid = decl.kind() != CXCursor_NoDeclFound;
        let declaration_to_look_for = if valid {
            decl.canonical()
        } else if cur.kind() == CXCursor_ClassTemplate {
            valid = true;
            cur
        } else {
            decl
        };
        if valid {
            if let Some(x) = ctx.parsed_types().iter().find(|x| *x.decl() == declaration_to_look_for) {
                return Ok(x.id().as_type_id_unchecked());
            }
        }
        let current_mod = ctx.current_mod().into();
        let partial_ty = PartialType::new(declaration_to_look_for, id);
        if valid {
            ctx.begin_parsing(partial_ty);
        }
        let result = Type::from_clang_ty(id, ty, cur, parent, ctx);
        let relevant_parent_id = parent.unwrap_or(current_mod);
        let y = match result {
            Ok(parse::Resolved::AlreadyDone(x)) => Ok(x.as_type_id_unchecked()),
            Ok(parse::Resolved::New(item, decl)) => {
                ctx.add_item(
                    Item::new(
                        id,
                        comment,
                        annos,
                        relevant_parent_id,
                        ItemKind::Type(item),
                        Some(cur.location()),
                    ),
                    decl,
                    Some(cur),
                );
                Ok(id.as_type_id_unchecked())
            },
            Err(parse::Error::Continue) => Err(parse::Error::Continue),
            Err(parse::Error::Recurse) => {
                let mut y = Err(parse::Error::Recurse);
                if valid {
                    let x = ctx.finish_parsing();
                    assert_eq!(*x.decl(), declaration_to_look_for);
                }
                cur.visit(|x| visit_child(x, id, ty, parent, ctx, &mut y));
                if valid {
                    let x = PartialType::new(declaration_to_look_for, id);
                    ctx.begin_parsing(x);
                }
                if let Err(parse::Error::Recurse) = y {
                    warn!(
                        "Unknown type, assuming named template type: \
                         id = {:?}; spelling = {}",
                        id,
                        ty.spelling()
                    );
                    Item::type_param(Some(id), cur, ctx)
                        .map(Ok)
                        .unwrap_or(Err(parse::Error::Recurse))
                } else {
                    y
                }
            },
        };
        if valid {
            let x = ctx.finish_parsing();
            assert_eq!(*x.decl(), declaration_to_look_for);
        }
        y
    }
    pub fn type_param(id: Option<ItemId>, cur: clang::Cursor, ctx: &mut Context) -> Option<TypeId> {
        let ty = cur.cur_type();
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
        let definition = if is_templ_with_spelling(&cur, &ty_spelling) {
            cur
        } else if cur.kind() == clang_lib::CXCursor_TypeRef {
            match cur.referenced() {
                Some(x) if is_templ_with_spelling(&x, &ty_spelling) => x,
                _ => return None,
            }
        } else {
            let mut definition = None;
            cur.visit(|x| {
                let child_ty = x.cur_type();
                if child_ty.kind() == clang_lib::CXCursor_TypeRef && child_ty.spelling() == ty_spelling {
                    match x.referenced() {
                        Some(x) if is_templ_with_spelling(&x, &ty_spelling) => {
                            definition = Some(x);
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
        let parent = ctx.root_mod().into();
        if let Some(x) = ctx.get_type_param(&definition) {
            if let Some(x2) = id {
                return Some(ctx.build_ty_wrapper(x2, x, Some(parent), &ty));
            } else {
                return Some(x);
            }
        }
        let name = ty_spelling.replace("const ", "").replace('.', "");
        let id = id.unwrap_or_else(|| ctx.next_item_id());
        let item = Item::new(
            id,
            None,
            None,
            parent,
            ItemKind::Type(Type::named(name)),
            Some(cur.location()),
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
    fn self_templ_ps(&self, ctx: &Context) -> Vec<TypeId> {
        self.kind.self_templ_ps(ctx)
    }
}
impl AsTemplParam for Item {
    type Extra = ();
    fn as_templ_param(&self, ctx: &Context, _: &Self::Extra) -> Option<TypeId> {
        self.kind.as_templ_param(ctx, self)
    }
}
impl TemplParams for ItemKind {
    fn self_templ_ps(&self, ctx: &Context) -> Vec<TypeId> {
        match *self {
            ItemKind::Type(ref x) => x.self_templ_ps(ctx),
            ItemKind::Func(_) | ItemKind::Mod(_) | ItemKind::Var(_) => {
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
            ItemKind::Mod(..) | ItemKind::Func(..) | ItemKind::Var(..) => None,
        }
    }
}
impl Trace for Item {
    type Extra = ();
    fn trace<T>(&self, ctx: &Context, tracer: &mut T, _: &Self::Extra)
    where
        T: Tracer,
    {
        match *self.kind() {
            ItemKind::Type(ref x) => {
                if x.should_trace() || !self.is_opaque(ctx, &()) {
                    x.trace(ctx, tracer, self);
                }
            },
            ItemKind::Func(ref x) => {
                tracer.visit(x.sig().into());
            },
            ItemKind::Var(ref x) => {
                tracer.visit_kind(x.ty().into(), EdgeKind::VarType);
            },
            ItemKind::Mod(_) => {},
        }
    }
}

fn visit_child(
    cur: clang::Cursor,
    id: ItemId,
    ty: &clang::Type,
    parent: Option<ItemId>,
    ctx: &mut Context,
    y: &mut Result<TypeId, parse::Error>,
) -> clang_lib::CXChildVisitResult {
    use clang_lib::*;
    if y.is_ok() {
        return CXChildVisit_Break;
    }
    *y = Item::from_ty_with_id(id, ty, cur, parent, ctx);
    match *y {
        Ok(..) => CXChildVisit_Break,
        Err(parse::Error::Recurse) => {
            cur.visit(|x| visit_child(x, id, ty, parent, ctx, y));
            CXChildVisit_Continue
        },
        Err(parse::Error::Continue) => CXChildVisit_Continue,
    }
}
