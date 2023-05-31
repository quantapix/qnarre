use super::super::timer::Timer;
use super::analysis::{analyze, as_cannot_derive_set, DeriveTrait, *};
use super::derive::Resolved;
use super::function::Function;
use super::int::IntKind;
use super::item::{Ancestors, IsOpaque, Item, ItemSet};
use super::item_kind::ItemKind;
use super::module::{ModKind, Module};
use super::template::{TemplInstantiation, TemplParams};
use super::traversal::{self, Edge, ItemTraversal};
use super::typ::{FloatKind, Type, TypeKind};
use crate::clang::{self, Cursor};
use crate::codegen::GenError;
use crate::Opts;
use crate::{Entry, HashMap, HashSet};
use proc_macro2::{Ident, Span, TokenStream};
use quote::ToTokens;
use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap as StdHashMap};
use std::iter::IntoIterator;
use std::mem;

#[derive(Debug, Copy, Clone, Eq, PartialOrd, Ord, Hash)]
pub struct ItemId(usize);
impl ItemId {
    pub fn as_usize(&self) -> usize {
        (*self).into()
    }
}
impl From<ItemId> for usize {
    fn from(id: ItemId) -> usize {
        id.0
    }
}
impl<T> ::std::cmp::PartialEq<T> for ItemId
where
    T: Copy + Into<ItemId>,
{
    fn eq(&self, rhs: &T) -> bool {
        let rhs: ItemId = (*rhs).into();
        self.0 == rhs.0
    }
}

macro_rules! item_id_newtype {
    (
        $( #[$attr:meta] )*
        pub struct $name:ident(ItemId)
        where
            $( #[$checked_attr:meta] )*
            checked = $checked:ident with $check_method:ident,
            $( #[$expected_attr:meta] )*
            expected = $expected:ident,
            $( #[$unchecked_attr:meta] )*
            unchecked = $unchecked:ident;
    ) => {
        $( #[$attr] )*
        #[derive(Debug, Copy, Clone, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(ItemId);
        impl $name {
            #[allow(dead_code)]
            pub fn into_resolver(self) -> ItemResolver {
                let id: ItemId = self.into();
                id.into()
            }
        }
        impl From<$name> for ItemId {
            fn from(id: $name) -> ItemId {
                id.0
            }
        }
        impl<'a> From<&'a $name> for ItemId {
            fn from(id: &'a $name) -> ItemId {
                id.0
            }
        }
        impl<T> ::std::cmp::PartialEq<T> for $name
        where
            T: Copy + Into<ItemId>
        {
            fn eq(&self, rhs: &T) -> bool {
                let rhs: ItemId = (*rhs).into();
                self.0 == rhs
            }
        }
        #[allow(dead_code)]
        impl ItemId {
            $( #[$checked_attr] )*
            pub fn $checked(&self, ctx: &Context) -> Option<$name> {
                if ctx.resolve_item(*self).kind().$check_method() {
                    Some($name(*self))
                } else {
                    None
                }
            }
            $( #[$expected_attr] )*
            pub fn $expected(&self, ctx: &Context) -> $name {
                self.$checked(ctx)
                    .expect(concat!(
                        stringify!($expected),
                        " called with ItemId that points to the wrong ItemKind"
                    ))
            }
            $( #[$unchecked_attr] )*
            pub fn $unchecked(&self) -> $name {
                $name(*self)
            }
        }
    }
}
item_id_newtype! {
    pub struct TypeId(ItemId)
    where
        checked = as_type_id with is_type,
        expected = expect_type_id,
        unchecked = as_type_id_unchecked;
}
item_id_newtype! {
    pub struct ModuleId(ItemId)
    where
        checked = as_module_id with is_module,
        expected = expect_module_id,
        unchecked = as_module_id_unchecked;
}
item_id_newtype! {
    pub struct VarId(ItemId)
    where
        checked = as_var_id with is_var,
        expected = expect_var_id,
        unchecked = as_var_id_unchecked;
}
item_id_newtype! {
    pub struct FnId(ItemId)
    where
        checked = as_function_id with is_function,
        expected = expect_function_id,
        unchecked = as_function_id_unchecked;
}

struct AllowedItemsTraversal<'ctx> {
    ctx: &'ctx Context,
    traversal: ItemTraversal<'ctx, ItemSet, Vec<ItemId>>,
}
impl<'ctx> AllowedItemsTraversal<'ctx> {
    pub fn new<R>(ctx: &'ctx Context, roots: R, predicate: for<'a> fn(&'a Context, Edge) -> bool) -> Self
    where
        R: IntoIterator<Item = ItemId>,
    {
        AllowedItemsTraversal {
            ctx,
            traversal: ItemTraversal::new(ctx, roots, predicate),
        }
    }
}
impl<'ctx> Iterator for AllowedItemsTraversal<'ctx> {
    type Item = ItemId;
    fn next(&mut self) -> Option<ItemId> {
        loop {
            let id = self.traversal.next()?;
            if self.ctx.resolve_item(id).is_blocklisted(self.ctx) {
                continue;
            }
            return Some(id);
        }
    }
}

#[derive(Eq, PartialEq, Hash, Debug)]
enum TypeKey {
    Usr(String),
    Declaration(Cursor),
}

#[derive(Debug)]
pub struct Context {
    allowed: Option<ItemSet>,
    blocklisted_types_implement_traits: RefCell<HashMap<DeriveTrait, HashMap<ItemId, Resolved>>>,
    cannot_derive_copy: Option<HashSet<ItemId>>,
    cannot_derive_debug: Option<HashSet<ItemId>>,
    cannot_derive_default: Option<HashSet<ItemId>>,
    cannot_derive_hash: Option<HashSet<ItemId>>,
    cannot_derive_partialeq_or_partialord: Option<HashMap<ItemId, Resolved>>,
    codegen_items: Option<ItemSet>,
    collected_typerefs: bool,
    current_module: ModuleId,
    currently_parsed_types: Vec<PartialType>,
    deps: BTreeSet<String>,
    enum_typedef_combos: Option<HashSet<ItemId>>,
    generated_bindgen_complex: Cell<bool>,
    has_float: Option<HashSet<ItemId>>,
    has_type_param_in_array: Option<HashSet<ItemId>>,
    have_destructor: Option<HashSet<ItemId>>,
    have_vtable: Option<HashMap<ItemId, has_vtable::Resolved>>,
    in_codegen: bool,
    items: Vec<Option<Item>>,
    modules: HashMap<Cursor, ModuleId>,
    need_bitfield_allocation: Vec<ItemId>,
    opts: Opts,
    parsed_macros: StdHashMap<Vec<u8>, cexpr::expr::EvalResult>,
    replacements: HashMap<Vec<String>, ItemId>,
    root_module: ModuleId,
    semantic_parents: HashMap<clang::Cursor, ItemId>,
    sizedness: Option<HashMap<TypeId, sizedness::Resolved>>,
    target_info: clang::TargetInfo,
    translation_unit: clang::TranslationUnit,
    type_params: HashMap<clang::Cursor, TypeId>,
    types: HashMap<TypeKey, TypeId>,
    used_templ_params: Option<HashMap<ItemId, ItemSet>>,
}
impl Context {
    pub fn new(opts: Opts, input_unsaved_files: &[clang::UnsavedFile]) -> Self {
        let index = clang::Index::new(false, true);
        let parse_options = clang_lib::CXTranslationUnit_DetailedPreprocessingRecord;
        let translation_unit = {
            let _t = Timer::new("translation_unit").with_output(opts.time_phases);
            clang::TranslationUnit::parse(&index, "", &opts.clang_args, input_unsaved_files, parse_options).expect(
                "libclang error; possible causes include:
- Invalid flag syntax
- Unrecognized flags
- Invalid flag arguments
- File I/O errors
- Host vs. target architecture mismatch
If you encounter an error missing from this list, please file an issue or a PR!",
            )
        };
        let target_info = clang::TargetInfo::new(&translation_unit);
        let root_module = Self::build_root_module(ItemId(0));
        let root_module_id = root_module.id().as_module_id_unchecked();
        let deps = opts.input_headers.iter().cloned().collect();
        Context {
            allowed: None,
            blocklisted_types_implement_traits: Default::default(),
            cannot_derive_copy: None,
            cannot_derive_debug: None,
            cannot_derive_default: None,
            cannot_derive_hash: None,
            cannot_derive_partialeq_or_partialord: None,
            codegen_items: None,
            collected_typerefs: false,
            current_module: root_module_id,
            currently_parsed_types: vec![],
            deps,
            enum_typedef_combos: None,
            generated_bindgen_complex: Cell::new(false),
            has_float: None,
            has_type_param_in_array: None,
            have_destructor: None,
            have_vtable: None,
            in_codegen: false,
            items: vec![Some(root_module)],
            modules: Default::default(),
            need_bitfield_allocation: Default::default(),
            opts,
            parsed_macros: Default::default(),
            replacements: Default::default(),
            root_module: root_module_id,
            semantic_parents: Default::default(),
            sizedness: None,
            target_info,
            translation_unit,
            type_params: Default::default(),
            types: Default::default(),
            used_templ_params: None,
        }
    }
    pub fn is_target_wasm32(&self) -> bool {
        self.target_info.triple.starts_with("wasm32-")
    }
    pub fn timer<'a>(&self, name: &'a str) -> Timer<'a> {
        Timer::new(name).with_output(self.opts.time_phases)
    }
    pub fn target_pointer_size(&self) -> usize {
        self.target_info.pointer_width / 8
    }
    pub fn currently_parsed_types(&self) -> &[PartialType] {
        &self.currently_parsed_types[..]
    }
    pub fn begin_parsing(&mut self, x: PartialType) {
        self.currently_parsed_types.push(x);
    }
    pub fn finish_parsing(&mut self) -> PartialType {
        self.currently_parsed_types
            .pop()
            .expect("should have been parsing a type, if we finished parsing a type")
    }
    pub fn include_file(&mut self, x: String) {
        for cb in &self.opts().parse_callbacks {
            cb.include_file(&x);
        }
        self.deps.insert(x);
    }
    pub fn deps(&self) -> &BTreeSet<String> {
        &self.deps
    }
    pub fn add_item(&mut self, it: Item, decl: Option<Cursor>, loc: Option<Cursor>) {
        debug!("Context::add_item({:?}, declaration: {:?}, loc: {:?}", it, decl, loc);
        debug_assert!(
            decl.is_some()
                || !it.kind().is_type()
                || it.kind().expect_type().is_builtin_or_type_param()
                || it.kind().expect_type().is_opaque(self, &it)
                || it.kind().expect_type().is_unresolved_ref(),
            "Adding a type without declaration?"
        );
        let id = it.id();
        let is_type = it.kind().is_type();
        let is_unnamed = is_type && it.expect_type().name().is_none();
        let is_template_instantiation = is_type && it.expect_type().is_templ_instantiation();
        if it.id() != self.root_module {
            self.add_item_to_module(&it);
        }
        if is_type && it.expect_type().is_comp() {
            self.need_bitfield_allocation.push(id);
        }
        let old_item = mem::replace(&mut self.items[id.0], Some(it));
        assert!(
            old_item.is_none(),
            "should not have already associated an item with the given id"
        );
        if !is_type || is_template_instantiation {
            return;
        }
        if let Some(mut decl) = decl {
            if !decl.is_valid() {
                if let Some(loc) = loc {
                    if loc.is_templ_like() {
                        decl = loc;
                    }
                }
            }
            decl = decl.canonical();
            if !decl.is_valid() {
                debug!(
                    "Invalid declaration {:?} found for type {:?}",
                    decl,
                    self.resolve_item_fallible(id).unwrap().kind().expect_type()
                );
                return;
            }
            let key = if is_unnamed {
                TypeKey::Declaration(decl)
            } else if let Some(usr) = decl.usr() {
                TypeKey::Usr(usr)
            } else {
                warn!("Valid declaration with no USR: {:?}, {:?}", decl, loc);
                TypeKey::Declaration(decl)
            };
            let old = self.types.insert(key, id.as_type_id_unchecked());
            debug_assert_eq!(old, None);
        }
    }
    fn add_item_to_module(&mut self, it: &Item) {
        assert!(it.id() != self.root_module);
        assert!(self.resolve_item_fallible(it.id()).is_none());
        if let Some(ref mut parent) = self.items[it.parent_id().0] {
            if let Some(module) = parent.as_module_mut() {
                debug!(
                    "add_item_to_module: adding {:?} as child of parent module {:?}",
                    it.id(),
                    it.parent_id()
                );
                module.children_mut().insert(it.id());
                return;
            }
        }
        debug!(
            "add_item_to_module: adding {:?} as child of current module {:?}",
            it.id(),
            self.current_module
        );
        self.items[(self.current_module.0).0]
            .as_mut()
            .expect("Should always have an item for self.current_module")
            .as_module_mut()
            .expect("self.current_module should always be a module")
            .children_mut()
            .insert(it.id());
    }
    pub fn add_type_param(&mut self, it: Item, definition: clang::Cursor) {
        debug!(
            "Context::add_type_param: item = {:?}; definition = {:?}",
            it, definition
        );
        assert!(
            it.expect_type().is_type_param(),
            "Should directly be a named type, not a resolved reference or anything"
        );
        assert_eq!(definition.kind(), clang_lib::CXCursor_TemplateTypeParameter);
        self.add_item_to_module(&it);
        let id = it.id();
        let old_item = mem::replace(&mut self.items[id.0], Some(it));
        assert!(
            old_item.is_none(),
            "should not have already associated an item with the given id"
        );
        let old_named_ty = self.type_params.insert(definition, id.as_type_id_unchecked());
        assert!(
            old_named_ty.is_none(),
            "should not have already associated a named type with this id"
        );
    }
    pub fn get_type_param(&self, definition: &clang::Cursor) -> Option<TypeId> {
        assert_eq!(definition.kind(), clang_lib::CXCursor_TemplateTypeParameter);
        self.type_params.get(definition).cloned()
    }
    #[rustfmt::skip]
    pub fn rust_mangle<'a>(&self, name: &'a str) -> Cow<'a, str> {
        if name.contains('@') ||
            name.contains('?') ||
            name.contains('$') ||
            matches!(
                name,
                "abstract" | "alignof" | "as" | "async" | "await" | "become" |
                    "box" | "break" | "const" | "continue" | "crate" | "do" |
                    "dyn" | "else" | "enum" | "extern" | "false" | "final" |
                    "fn" | "for" | "if" | "impl" | "in" | "let" | "loop" |
                    "macro" | "match" | "mod" | "move" | "mut" | "offsetof" |
                    "override" | "priv" | "proc" | "pub" | "pure" | "ref" |
                    "return" | "Self" | "self" | "sizeof" | "static" |
                    "struct" | "super" | "trait" | "true" | "try" | "type" | "typeof" |
                    "unsafe" | "unsized" | "use" | "virtual" | "where" |
                    "while" | "yield" | "str" | "bool" | "f32" | "f64" |
                    "usize" | "isize" | "u128" | "i128" | "u64" | "i64" |
                    "u32" | "i32" | "u16" | "i16" | "u8" | "i8" | "_"
            )
        {
            let mut s = name.to_owned();
            s = s.replace('@', "_");
            s = s.replace('?', "_");
            s = s.replace('$', "_");
            s.push('_');
            return Cow::Owned(s);
        }
        Cow::Borrowed(name)
    }
    pub fn rust_ident<S>(&self, name: S) -> Ident
    where
        S: AsRef<str>,
    {
        self.rust_ident_raw(self.rust_mangle(name.as_ref()))
    }
    pub fn rust_ident_raw<T>(&self, name: T) -> Ident
    where
        T: AsRef<str>,
    {
        Ident::new(name.as_ref(), Span::call_site())
    }
    pub fn items(&self) -> impl Iterator<Item = (ItemId, &Item)> {
        self.items.iter().enumerate().filter_map(|(index, item)| {
            let item = item.as_ref()?;
            Some((ItemId(index), item))
        })
    }
    pub fn collected_typerefs(&self) -> bool {
        self.collected_typerefs
    }
    fn collect_typerefs(&mut self) -> Vec<(ItemId, clang::Type, clang::Cursor, Option<ItemId>)> {
        debug_assert!(!self.collected_typerefs);
        self.collected_typerefs = true;
        let mut typerefs = vec![];
        for (id, item) in self.items() {
            let kind = item.kind();
            let ty = match kind.as_type() {
                Some(ty) => ty,
                None => continue,
            };
            if let TypeKind::UnresolvedTypeRef(ref ty, loc, parent_id) = *ty.kind() {
                typerefs.push((id, *ty, loc, parent_id));
            };
        }
        typerefs
    }
    fn resolve_typerefs(&mut self) {
        let _t = self.timer("resolve_typerefs");
        let typerefs = self.collect_typerefs();
        for (id, ty, loc, parent_id) in typerefs {
            let _resolved = {
                let resolved = Item::from_ty(&ty, loc, parent_id, self).unwrap_or_else(|_| {
                    warn!(
                        "Could not resolve type reference, falling back \
                               to opaque blob"
                    );
                    Item::new_opaque_type(self.next_item_id(), &ty, self)
                });
                let item = self.items[id.0].as_mut().unwrap();
                *item.kind_mut().as_type_mut().unwrap().kind_mut() = TypeKind::ResolvedTypeRef(resolved);
                resolved
            };
        }
    }
    fn with_loaned_item<F, T>(&mut self, id: ItemId, f: F) -> T
    where
        F: (FnOnce(&Context, &mut Item) -> T),
    {
        let mut item = self.items[id.0].take().unwrap();
        let result = f(self, &mut item);
        let existing = mem::replace(&mut self.items[id.0], Some(item));
        assert!(existing.is_none());
        result
    }
    fn compute_bitfield_units(&mut self) {
        let _t = self.timer("compute_bitfield_units");
        assert!(self.collected_typerefs());
        let need_bitfield_allocation = mem::take(&mut self.need_bitfield_allocation);
        for id in need_bitfield_allocation {
            self.with_loaned_item(id, |ctx, item| {
                let ty = item.kind_mut().as_type_mut().unwrap();
                let layout = ty.layout(ctx);
                ty.as_comp_mut().unwrap().compute_bitfield_units(ctx, layout.as_ref());
            });
        }
    }
    fn deanonymize_fields(&mut self) {
        let _t = self.timer("deanonymize_fields");
        let comp_item_ids: Vec<ItemId> = self
            .items()
            .filter_map(|(id, item)| {
                if item.kind().as_type()?.is_comp() {
                    return Some(id);
                }
                None
            })
            .collect();
        for id in comp_item_ids {
            self.with_loaned_item(id, |ctx, item| {
                item.kind_mut()
                    .as_type_mut()
                    .unwrap()
                    .as_comp_mut()
                    .unwrap()
                    .deanonymize_fields(ctx);
            });
        }
    }
    fn process_replacements(&mut self) {
        let _t = self.timer("process_replacements");
        if self.replacements.is_empty() {
            debug!("No replacements to process");
            return;
        }
        let mut replacements = vec![];
        for (id, item) in self.items() {
            if item.annotations().use_instead_of().is_some() {
                continue;
            }
            let ty = match item.kind().as_type() {
                Some(ty) => ty,
                None => continue,
            };
            match *ty.kind() {
                TypeKind::Comp(..) | TypeKind::TemplAlias(..) | TypeKind::Enum(..) | TypeKind::Alias(..) => {},
                _ => continue,
            }
            let path = item.path_for_allowlisting(self);
            let replacement = self.replacements.get(&path[1..]);
            if let Some(replacement) = replacement {
                if *replacement != id && self.resolve_item_fallible(*replacement).is_some() {
                    replacements.push((id.expect_type_id(self), replacement.expect_type_id(self)));
                }
            }
        }
        for (id, replacement_id) in replacements {
            debug!("Replacing {:?} with {:?}", id, replacement_id);
            let new_parent = {
                let item_id: ItemId = id.into();
                let item = self.items[item_id.0].as_mut().unwrap();
                *item.kind_mut().as_type_mut().unwrap().kind_mut() = TypeKind::ResolvedTypeRef(replacement_id);
                item.parent_id()
            };
            let old_parent = self.resolve_item(replacement_id).parent_id();
            if new_parent == old_parent {
                continue;
            }
            let replacement_item_id: ItemId = replacement_id.into();
            self.items[replacement_item_id.0]
                .as_mut()
                .unwrap()
                .set_parent_for_replacement(new_parent);
            let old_module = {
                let immut_self = &*self;
                old_parent
                    .ancestors(immut_self)
                    .chain(Some(immut_self.root_module.into()))
                    .find(|id| {
                        let item = immut_self.resolve_item(*id);
                        item.as_module()
                            .map_or(false, |m| m.children().contains(&replacement_id.into()))
                    })
            };
            let old_module = old_module.expect("Every replacement item should be in a module");
            let new_module = {
                let immut_self = &*self;
                new_parent
                    .ancestors(immut_self)
                    .find(|id| immut_self.resolve_item(*id).is_module())
            };
            let new_module = new_module.unwrap_or_else(|| self.root_module.into());
            if new_module == old_module {
                continue;
            }
            self.items[old_module.0]
                .as_mut()
                .unwrap()
                .as_module_mut()
                .unwrap()
                .children_mut()
                .remove(&replacement_id.into());
            self.items[new_module.0]
                .as_mut()
                .unwrap()
                .as_module_mut()
                .unwrap()
                .children_mut()
                .insert(replacement_id.into());
        }
    }
    pub fn gen<F, Out>(mut self, cb: F) -> Result<(Out, Opts), GenError>
    where
        F: FnOnce(&Self) -> Result<Out, GenError>,
    {
        self.in_codegen = true;
        self.resolve_typerefs();
        self.compute_bitfield_units();
        self.process_replacements();
        self.deanonymize_fields();
        self.assert_no_dangling_references();
        self.compute_allowed_and_codegen_items();
        self.assert_every_item_in_a_module();
        self.compute_has_vtable();
        self.compute_sizedness();
        self.compute_has_destructor();
        self.find_used_templ_params();
        self.compute_enum_typedef_combos();
        self.compute_cannot_derive_debug();
        self.compute_cannot_derive_default();
        self.compute_cannot_derive_copy();
        self.compute_has_type_param_in_array();
        self.compute_has_float();
        self.compute_cannot_derive_hash();
        self.compute_cannot_derive_partialord_partialeq_or_eq();
        let ret = cb(&self)?;
        Ok((ret, self.opts))
    }
    fn assert_no_dangling_references(&self) {}
    fn assert_no_dangling_item_traversal(&self) -> traversal::AssertNoDanglingItemsTraversal {
        assert!(self.in_codegen_phase());
        assert!(self.current_module == self.root_module);
        let roots = self.items().map(|(id, _)| id);
        traversal::AssertNoDanglingItemsTraversal::new(self, roots, traversal::all_edges)
    }
    fn assert_every_item_in_a_module(&self) {}
    fn compute_sizedness(&mut self) {
        let _t = self.timer("compute_sizedness");
        assert!(self.sizedness.is_none());
        self.sizedness = Some(analyze::<sizedness::Analysis>(self));
    }
    pub fn lookup_sizedness(&self, id: TypeId) -> sizedness::Resolved {
        assert!(self.in_codegen_phase());
        self.sizedness
            .as_ref()
            .unwrap()
            .get(&id)
            .cloned()
            .unwrap_or(sizedness::Resolved::ZeroSized)
    }
    fn compute_has_vtable(&mut self) {
        let _t = self.timer("compute_has_vtable");
        assert!(self.have_vtable.is_none());
        self.have_vtable = Some(analyze::<has_vtable::Analysis>(self));
    }
    pub fn lookup_has_vtable(&self, id: TypeId) -> has_vtable::Resolved {
        assert!(self.in_codegen_phase());
        self.have_vtable
            .as_ref()
            .unwrap()
            .get(&id.into())
            .cloned()
            .unwrap_or(has_vtable::Resolved::No)
    }
    fn compute_has_destructor(&mut self) {
        let _t = self.timer("compute_has_destructor");
        assert!(self.have_destructor.is_none());
        self.have_destructor = Some(analyze::<has_destructor::Analysis>(self));
    }
    pub fn lookup_has_destructor(&self, id: TypeId) -> bool {
        assert!(self.in_codegen_phase());
        self.have_destructor.as_ref().unwrap().contains(&id.into())
    }
    fn find_used_templ_params(&mut self) {
        let _t = self.timer("find_used_templ_params");
        if self.opts.allowlist_recursively {
            let used_params = analyze::<used_templ_param::Analysis>(self);
            self.used_templ_params = Some(used_params);
        } else {
            let mut used_params = HashMap::default();
            for &id in self.allowed_items() {
                used_params
                    .entry(id)
                    .or_insert_with(|| id.self_templ_params(self).into_iter().map(|p| p.into()).collect());
            }
            self.used_templ_params = Some(used_params);
        }
    }
    pub fn uses_templ_param(&self, id: ItemId, templ_param: TypeId) -> bool {
        assert!(self.in_codegen_phase());
        if self.resolve_item(id).is_blocklisted(self) {
            return true;
        }
        let templ_param = templ_param
            .into_resolver()
            .through_type_refs()
            .through_type_aliases()
            .resolve(self)
            .id();
        self.used_templ_params
            .as_ref()
            .expect("should have found template parameter usage if we're in codegen")
            .get(&id)
            .map_or(false, |x| x.contains(&templ_param))
    }
    pub fn uses_any_templ_params(&self, id: ItemId) -> bool {
        assert!(
            self.in_codegen_phase(),
            "We only compute template parameter usage as we enter codegen"
        );
        self.used_templ_params
            .as_ref()
            .expect("should have template parameter usage info in codegen phase")
            .get(&id)
            .map_or(false, |x| !x.is_empty())
    }
    fn add_builtin_item(&mut self, it: Item) {
        debug!("add_builtin_item: item = {:?}", it);
        debug_assert!(it.kind().is_type());
        self.add_item_to_module(&it);
        let id = it.id();
        let old_item = mem::replace(&mut self.items[id.0], Some(it));
        assert!(old_item.is_none(), "Inserted type twice?");
    }
    fn build_root_module(id: ItemId) -> Item {
        let module = Module::new(Some("root".into()), ModKind::Normal);
        Item::new(id, None, None, id, ItemKind::Module(module), None)
    }
    pub fn root_module(&self) -> ModuleId {
        self.root_module
    }
    pub fn resolve_type(&self, type_id: TypeId) -> &Type {
        self.resolve_item(type_id).kind().expect_type()
    }
    pub fn resolve_func(&self, func_id: FnId) -> &Function {
        self.resolve_item(func_id).kind().expect_function()
    }
    pub fn safe_resolve_type(&self, type_id: TypeId) -> Option<&Type> {
        self.resolve_item_fallible(type_id).map(|t| t.kind().expect_type())
    }
    pub fn resolve_item_fallible<Id: Into<ItemId>>(&self, id: Id) -> Option<&Item> {
        self.items.get(id.into().0)?.as_ref()
    }
    pub fn resolve_item<Id: Into<ItemId>>(&self, item_id: Id) -> &Item {
        let item_id = item_id.into();
        match self.resolve_item_fallible(item_id) {
            Some(item) => item,
            None => panic!("Not an item: {:?}", item_id),
        }
    }
    pub fn current_module(&self) -> ModuleId {
        self.current_module
    }
    pub fn add_semantic_parent(&mut self, definition: clang::Cursor, parent_id: ItemId) {
        self.semantic_parents.insert(definition, parent_id);
    }
    pub fn known_semantic_parent(&self, definition: clang::Cursor) -> Option<ItemId> {
        self.semantic_parents.get(&definition).cloned()
    }
    fn get_declaration_info_for_templ_instantiation(&self, cur: &Cursor) -> Option<(Cursor, ItemId, usize)> {
        cur.cur_type()
            .canonical_declaration(Some(cur))
            .and_then(|x| {
                self.get_resolved_type(&x).and_then(|x2| {
                    let n = x2.num_self_templ_params(self);
                    if n == 0 {
                        None
                    } else {
                        Some((*x.cursor(), x2.into(), n))
                    }
                })
            })
            .or_else(|| {
                cur.referenced()
                    .and_then(|x| self.currently_parsed_types().iter().find(|x2| *x2.decl() == x).cloned())
                    .and_then(|x| {
                        let n = x.num_self_templ_params(self);
                        if n == 0 {
                            None
                        } else {
                            Some((*x.decl(), x.id(), n))
                        }
                    })
            })
    }
    fn instantiate_templ(
        &mut self,
        with_id: ItemId,
        template: TypeId,
        ty: &clang::Type,
        location: clang::Cursor,
    ) -> Option<TypeId> {
        let num_expected_args = self.resolve_type(template).num_self_templ_params(self);
        if num_expected_args == 0 {
            warn!(
                "Tried to instantiate a template for which we could not \
                 determine any template parameters"
            );
            return None;
        }
        let mut args = vec![];
        let mut found_const_arg = false;
        let mut children = location.collect_children();
        if children.iter().all(|c| !c.has_children()) {
            let idx = children
                .iter()
                .position(|c| c.kind() == clang_lib::CXCursor_TemplateRef);
            if let Some(idx) = idx {
                if children
                    .iter()
                    .take(idx)
                    .all(|c| c.kind() == clang_lib::CXCursor_NamespaceRef)
                {
                    children = children.into_iter().skip(idx + 1).collect();
                }
            }
        }
        for child in children.iter().rev() {
            match child.kind() {
                clang_lib::CXCursor_TypeRef | clang_lib::CXCursor_TypedefDecl | clang_lib::CXCursor_TypeAliasDecl => {
                    let ty = Item::from_ty_or_ref(child.cur_type(), *child, Some(template.into()), self);
                    args.push(ty);
                },
                clang_lib::CXCursor_TemplateRef => {
                    let (template_decl_cursor, template_decl_id, num_expected_template_args) =
                        self.get_declaration_info_for_templ_instantiation(child)?;
                    if num_expected_template_args == 0 || child.has_at_least_num_children(num_expected_template_args) {
                        let ty = Item::from_ty_or_ref(child.cur_type(), *child, Some(template.into()), self);
                        args.push(ty);
                    } else {
                        let args_len = args.len();
                        if args_len < num_expected_template_args {
                            warn!(
                                "Found a template instantiation without \
                                 enough template arguments"
                            );
                            return None;
                        }
                        let mut sub_args: Vec<_> = args.drain(args_len - num_expected_template_args..).collect();
                        sub_args.reverse();
                        let sub_name = Some(template_decl_cursor.spelling());
                        let sub_inst = TemplInstantiation::new(template_decl_id.as_type_id_unchecked(), sub_args);
                        let sub_kind = TypeKind::TemplInstantiation(sub_inst);
                        let sub_ty = Type::new(
                            sub_name,
                            template_decl_cursor.cur_type().fallible_layout(self).ok(),
                            sub_kind,
                            false,
                        );
                        let sub_id = self.next_item_id();
                        let sub_item = Item::new(
                            sub_id,
                            None,
                            None,
                            self.current_module.into(),
                            ItemKind::Type(sub_ty),
                            Some(child.location()),
                        );
                        debug!(
                            "instantiate_template: inserting nested \
                             instantiation item: {:?}",
                            sub_item
                        );
                        self.add_item_to_module(&sub_item);
                        debug_assert_eq!(sub_id, sub_item.id());
                        self.items[sub_id.0] = Some(sub_item);
                        args.push(sub_id.as_type_id_unchecked());
                    }
                },
                _ => {
                    warn!("Found template arg cursor we can't handle: {:?}", child);
                    found_const_arg = true;
                },
            }
        }
        if found_const_arg {
            warn!(
                "Found template instantiated with a const value; \
                 bindgen can't handle this kind of template instantiation!"
            );
            return None;
        }
        if args.len() != num_expected_args {
            warn!(
                "Found a template with an unexpected number of template \
                 arguments"
            );
            return None;
        }
        args.reverse();
        let type_kind = TypeKind::TemplInstantiation(TemplInstantiation::new(template, args));
        let name = ty.spelling();
        let name = if name.is_empty() { None } else { Some(name) };
        let ty = Type::new(name, ty.fallible_layout(self).ok(), type_kind, ty.is_const());
        let item = Item::new(
            with_id,
            None,
            None,
            self.current_module.into(),
            ItemKind::Type(ty),
            Some(location.location()),
        );
        debug!("instantiate_template: inserting item: {:?}", item);
        self.add_item_to_module(&item);
        debug_assert_eq!(with_id, item.id());
        self.items[with_id.0] = Some(item);
        Some(with_id.as_type_id_unchecked())
    }
    pub fn get_resolved_type(&self, decl: &clang::CanonTyDecl) -> Option<TypeId> {
        self.types
            .get(&TypeKey::Declaration(*decl.cursor()))
            .or_else(|| decl.cursor().usr().and_then(|usr| self.types.get(&TypeKey::Usr(usr))))
            .cloned()
    }
    pub fn builtin_or_resolved_ty(
        &mut self,
        with_id: ItemId,
        parent_id: Option<ItemId>,
        ty: &clang::Type,
        location: Option<clang::Cursor>,
    ) -> Option<TypeId> {
        use clang_lib::{CXCursor_TypeAliasTemplateDecl, CXCursor_TypeRef};
        debug!(
            "builtin_or_resolved_ty: {:?}, {:?}, {:?}, {:?}",
            ty, location, with_id, parent_id
        );
        if let Some(decl) = ty.canonical_declaration(location.as_ref()) {
            if let Some(id) = self.get_resolved_type(&decl) {
                debug!("Already resolved ty {:?}, {:?}, {:?} {:?}", id, decl, ty, location);
                if let Some(location) = location {
                    if decl.cursor().is_templ_like() && *ty != decl.cursor().cur_type() {
                        if decl.cursor().kind() == CXCursor_TypeAliasTemplateDecl
                            && !location.contains_cursor(CXCursor_TypeRef)
                            && ty.canonical_type().is_valid_and_exposed()
                        {
                            return None;
                        }
                        return self.instantiate_templ(with_id, id, ty, location).or(Some(id));
                    }
                }
                return Some(self.build_ty_wrapper(with_id, id, parent_id, ty));
            }
        }
        debug!("Not resolved, maybe builtin?");
        self.build_builtin_ty(ty)
    }
    pub fn build_ty_wrapper(
        &mut self,
        with_id: ItemId,
        wrapped_id: TypeId,
        parent_id: Option<ItemId>,
        ty: &clang::Type,
    ) -> TypeId {
        self.build_wrapper(with_id, wrapped_id, parent_id, ty, ty.is_const())
    }
    pub fn build_const_wrapper(
        &mut self,
        with_id: ItemId,
        wrapped_id: TypeId,
        parent_id: Option<ItemId>,
        ty: &clang::Type,
    ) -> TypeId {
        self.build_wrapper(with_id, wrapped_id, parent_id, ty, /* is_const = */ true)
    }
    fn build_wrapper(
        &mut self,
        with_id: ItemId,
        wrapped_id: TypeId,
        parent_id: Option<ItemId>,
        ty: &clang::Type,
        is_const: bool,
    ) -> TypeId {
        let spelling = ty.spelling();
        let layout = ty.fallible_layout(self).ok();
        let location = ty.declaration().location();
        let type_kind = TypeKind::ResolvedTypeRef(wrapped_id);
        let ty = Type::new(Some(spelling), layout, type_kind, is_const);
        let item = Item::new(
            with_id,
            None,
            None,
            parent_id.unwrap_or_else(|| self.current_module.into()),
            ItemKind::Type(ty),
            Some(location),
        );
        self.add_builtin_item(item);
        with_id.as_type_id_unchecked()
    }
    pub fn next_item_id(&mut self) -> ItemId {
        let ret = ItemId(self.items.len());
        self.items.push(None);
        ret
    }
    fn build_builtin_ty(&mut self, ty: &clang::Type) -> Option<TypeId> {
        use clang_lib::*;
        let type_kind = match ty.kind() {
            CXType_NullPtr => TypeKind::NullPtr,
            CXType_Void => TypeKind::Void,
            CXType_Bool => TypeKind::Int(IntKind::Bool),
            CXType_Int => TypeKind::Int(IntKind::Int),
            CXType_UInt => TypeKind::Int(IntKind::UInt),
            CXType_Char_S => TypeKind::Int(IntKind::Char { is_signed: true }),
            CXType_Char_U => TypeKind::Int(IntKind::Char { is_signed: false }),
            CXType_SChar => TypeKind::Int(IntKind::SChar),
            CXType_UChar => TypeKind::Int(IntKind::UChar),
            CXType_Short => TypeKind::Int(IntKind::Short),
            CXType_UShort => TypeKind::Int(IntKind::UShort),
            CXType_WChar => TypeKind::Int(IntKind::WChar),
            CXType_Char16 => TypeKind::Int(IntKind::U16),
            CXType_Char32 => TypeKind::Int(IntKind::U32),
            CXType_Long => TypeKind::Int(IntKind::Long),
            CXType_ULong => TypeKind::Int(IntKind::ULong),
            CXType_LongLong => TypeKind::Int(IntKind::LongLong),
            CXType_ULongLong => TypeKind::Int(IntKind::ULongLong),
            CXType_Int128 => TypeKind::Int(IntKind::I128),
            CXType_UInt128 => TypeKind::Int(IntKind::U128),
            CXType_Float => TypeKind::Float(FloatKind::Float),
            CXType_Double => TypeKind::Float(FloatKind::Double),
            CXType_LongDouble => TypeKind::Float(FloatKind::LongDouble),
            CXType_Float128 => TypeKind::Float(FloatKind::Float128),
            CXType_Complex => {
                let float_type = ty.elem_type().expect("Not able to resolve complex type?");
                let float_kind = match float_type.kind() {
                    CXType_Float => FloatKind::Float,
                    CXType_Double => FloatKind::Double,
                    CXType_LongDouble => FloatKind::LongDouble,
                    CXType_Float128 => FloatKind::Float128,
                    _ => panic!("Non floating-type complex? {:?}, {:?}", ty, float_type,),
                };
                TypeKind::Complex(float_kind)
            },
            _ => return None,
        };
        let spelling = ty.spelling();
        let is_const = ty.is_const();
        let layout = ty.fallible_layout(self).ok();
        let location = ty.declaration().location();
        let ty = Type::new(Some(spelling), layout, type_kind, is_const);
        let id = self.next_item_id();
        let item = Item::new(
            id,
            None,
            None,
            self.root_module.into(),
            ItemKind::Type(ty),
            Some(location),
        );
        self.add_builtin_item(item);
        Some(id.as_type_id_unchecked())
    }
    pub fn translation_unit(&self) -> &clang::TranslationUnit {
        &self.translation_unit
    }
    pub fn parsed_macro(&self, macro_name: &[u8]) -> bool {
        self.parsed_macros.contains_key(macro_name)
    }
    pub fn parsed_macros(&self) -> &StdHashMap<Vec<u8>, cexpr::expr::EvalResult> {
        debug_assert!(!self.in_codegen_phase());
        &self.parsed_macros
    }
    pub fn note_parsed_macro(&mut self, id: Vec<u8>, value: cexpr::expr::EvalResult) {
        self.parsed_macros.insert(id, value);
    }
    pub fn in_codegen_phase(&self) -> bool {
        self.in_codegen
    }
    pub fn replace(&mut self, name: &[String], potential_ty: ItemId) {
        match self.replacements.entry(name.into()) {
            Entry::Vacant(entry) => {
                debug!("Defining replacement for {:?} as {:?}", name, potential_ty);
                entry.insert(potential_ty);
            },
            Entry::Occupied(occupied) => {
                warn!(
                    "Replacement for {:?} already defined as {:?}; \
                     ignoring duplicate replacement definition as {:?}",
                    name,
                    occupied.get(),
                    potential_ty
                );
            },
        }
    }
    pub fn is_replaced_type<Id: Into<ItemId>>(&self, path: &[String], id: Id) -> bool {
        let id = id.into();
        matches!(self.replacements.get(path), Some(replaced_by) if *replaced_by != id)
    }
    pub fn opaque_by_name(&self, path: &[String]) -> bool {
        debug_assert!(self.in_codegen_phase(), "You're not supposed to call this yet");
        self.opts.opaque_types.matches(path[1..].join("::"))
    }
    pub fn opts(&self) -> &Opts {
        &self.opts
    }
    fn tokenize_namespace(&self, cursor: &clang::Cursor) -> (Option<String>, ModKind) {
        assert_eq!(cursor.kind(), ::clang_lib::CXCursor_Namespace, "Be a nice person");
        let mut module_name = None;
        let spelling = cursor.spelling();
        if !spelling.is_empty() {
            module_name = Some(spelling)
        }
        let mut kind = ModKind::Normal;
        let mut looking_for_name = false;
        for token in cursor.tokens().iter() {
            match token.spelling() {
                b"inline" => {
                    debug_assert!(kind != ModKind::Inline, "Multiple inline keywords?");
                    kind = ModKind::Inline;
                    looking_for_name = true;
                },
                b"namespace" | b"::" => {
                    looking_for_name = true;
                },
                b"{" => {
                    assert!(looking_for_name);
                    break;
                },
                name => {
                    if looking_for_name {
                        if module_name.is_none() {
                            module_name = Some(String::from_utf8_lossy(name).into_owned());
                        }
                        break;
                    } else {
                        warn!(
                            "Ignored unknown namespace prefix '{}' at {:?} in {:?}",
                            String::from_utf8_lossy(name),
                            token,
                            cursor
                        );
                    }
                },
            }
        }
        (module_name, kind)
    }
    pub fn module(&mut self, cursor: clang::Cursor) -> ModuleId {
        use clang_lib::*;
        assert_eq!(cursor.kind(), CXCursor_Namespace, "Be a nice person");
        let cursor = cursor.canonical();
        if let Some(id) = self.modules.get(&cursor) {
            return *id;
        }
        let (module_name, kind) = self.tokenize_namespace(&cursor);
        let module_id = self.next_item_id();
        let module = Module::new(module_name, kind);
        let module = Item::new(
            module_id,
            None,
            None,
            self.current_module.into(),
            ItemKind::Module(module),
            Some(cursor.location()),
        );
        let module_id = module.id().as_module_id_unchecked();
        self.modules.insert(cursor, module_id);
        self.add_item(module, None, None);
        module_id
    }
    pub fn with_module<F>(&mut self, module_id: ModuleId, cb: F)
    where
        F: FnOnce(&mut Self),
    {
        debug_assert!(self.resolve_item(module_id).kind().is_module(), "Wat");
        let previous_id = self.current_module;
        self.current_module = module_id;
        cb(self);
        self.current_module = previous_id;
    }
    pub fn allowed_items(&self) -> &ItemSet {
        assert!(self.in_codegen_phase());
        assert!(self.current_module == self.root_module);
        self.allowed.as_ref().unwrap()
    }
    pub fn blocklisted_type_implements_trait(&self, it: &Item, derive_trait: DeriveTrait) -> Resolved {
        assert!(self.in_codegen_phase());
        assert!(self.current_module == self.root_module);
        *self
            .blocklisted_types_implement_traits
            .borrow_mut()
            .entry(derive_trait)
            .or_default()
            .entry(it.id())
            .or_insert_with(|| {
                it.expect_type()
                    .name()
                    .and_then(|name| {
                        if self.opts.parse_callbacks.is_empty() {
                            if self.is_stdint_type(name) {
                                Some(Resolved::Yes)
                            } else {
                                Some(Resolved::No)
                            }
                        } else {
                            self.opts
                                .last_callback(|cb| cb.blocklisted_type_implements_trait(name, derive_trait))
                        }
                    })
                    .unwrap_or(Resolved::No)
            })
    }
    pub fn is_stdint_type(&self, name: &str) -> bool {
        match name {
            "int8_t" | "uint8_t" | "int16_t" | "uint16_t" | "int32_t" | "uint32_t" | "int64_t" | "uint64_t"
            | "uintptr_t" | "intptr_t" | "ptrdiff_t" => true,
            "size_t" | "ssize_t" => self.opts.size_t_is_usize,
            _ => false,
        }
    }
    pub fn codegen_items(&self) -> &ItemSet {
        assert!(self.in_codegen_phase());
        assert!(self.current_module == self.root_module);
        self.codegen_items.as_ref().unwrap()
    }
    fn compute_allowed_and_codegen_items(&mut self) {
        assert!(self.in_codegen_phase());
        assert!(self.current_module == self.root_module);
        assert!(self.allowed.is_none());
        let _t = self.timer("compute_allowed_and_codegen_items");
        let roots = {
            let mut roots = self
                .items()
                .filter(|&(_, item)| item.is_enabled_for_codegen(self))
                .filter(|&(_, item)| {
                    if self.opts().allowed_types.is_empty()
                        && self.opts().allowed_fns.is_empty()
                        && self.opts().allowed_vars.is_empty()
                        && self.opts().allowed_files.is_empty()
                    {
                        return true;
                    }
                    if item.annotations().use_instead_of().is_some() {
                        return true;
                    }
                    if !self.opts().allowed_files.is_empty() {
                        if let Some(location) = item.location() {
                            let (file, _, _, _) = location.location();
                            if let Some(filename) = file.name() {
                                if self.opts().allowed_files.matches(filename) {
                                    return true;
                                }
                            }
                        }
                    }
                    let name = item.path_for_allowlisting(self)[1..].join("::");
                    debug!("allowed_items: testing {:?}", name);
                    match *item.kind() {
                        ItemKind::Module(..) => true,
                        ItemKind::Function(_) => self.opts().allowed_fns.matches(&name),
                        ItemKind::Var(_) => self.opts().allowed_vars.matches(&name),
                        ItemKind::Type(ref ty) => {
                            if self.opts().allowed_types.matches(&name) {
                                return true;
                            }
                            if !self.opts().allowlist_recursively {
                                match *ty.kind() {
                                    TypeKind::Void
                                    | TypeKind::NullPtr
                                    | TypeKind::Int(..)
                                    | TypeKind::Float(..)
                                    | TypeKind::Complex(..)
                                    | TypeKind::Array(..)
                                    | TypeKind::Vector(..)
                                    | TypeKind::Pointer(..)
                                    | TypeKind::Reference(..)
                                    | TypeKind::Function(..)
                                    | TypeKind::ResolvedTypeRef(..)
                                    | TypeKind::Opaque
                                    | TypeKind::TypeParam => return true,
                                    _ => {},
                                }
                                if self.is_stdint_type(&name) {
                                    return true;
                                }
                            }
                            let parent = self.resolve_item(item.parent_id());
                            if !parent.is_module() {
                                return false;
                            }
                            let enum_ = match *ty.kind() {
                                TypeKind::Enum(ref e) => e,
                                _ => return false,
                            };
                            if ty.name().is_some() {
                                return false;
                            }
                            let mut prefix_path = parent.path_for_allowlisting(self).clone();
                            enum_.variants().iter().any(|variant| {
                                prefix_path.push(variant.name_for_allowlisting().into());
                                let name = prefix_path[1..].join("::");
                                prefix_path.pop().unwrap();
                                self.opts().allowed_vars.matches(name)
                            })
                        },
                    }
                })
                .map(|(id, _)| id)
                .collect::<Vec<_>>();
            roots.reverse();
            roots
        };
        let allowed_items_predicate = if self.opts().allowlist_recursively {
            traversal::all_edges
        } else {
            traversal::only_inner_type_edges
        };
        let allowed = AllowedItemsTraversal::new(self, roots.clone(), allowed_items_predicate).collect::<ItemSet>();
        let codegen_items = if self.opts().allowlist_recursively {
            AllowedItemsTraversal::new(self, roots, traversal::codegen_edges).collect::<ItemSet>()
        } else {
            allowed.clone()
        };
        self.allowed = Some(allowed);
        self.codegen_items = Some(codegen_items);
    }
    pub fn trait_prefix(&self) -> Ident {
        if self.opts().use_core {
            self.rust_ident_raw("core")
        } else {
            self.rust_ident_raw("std")
        }
    }
    pub fn generated_bindgen_complex(&self) {
        self.generated_bindgen_complex.set(true)
    }
    pub fn need_bindgen_complex_type(&self) -> bool {
        self.generated_bindgen_complex.get()
    }
    fn compute_enum_typedef_combos(&mut self) {
        let _t = self.timer("compute_enum_typedef_combos");
        assert!(self.enum_typedef_combos.is_none());
        let mut enum_typedef_combos = HashSet::default();
        for item in &self.items {
            if let Some(ItemKind::Module(module)) = item.as_ref().map(Item::kind) {
                let mut names_of_typedefs = HashSet::default();
                for child_id in module.children() {
                    if let Some(ItemKind::Type(ty)) = self.items[child_id.0].as_ref().map(Item::kind) {
                        if let (Some(name), TypeKind::Alias(type_id)) = (ty.name(), ty.kind()) {
                            if type_id
                                .into_resolver()
                                .through_type_refs()
                                .through_type_aliases()
                                .resolve(self)
                                .expect_type()
                                .is_int()
                            {
                                names_of_typedefs.insert(name);
                            }
                        }
                    }
                }
                for child_id in module.children() {
                    if let Some(ItemKind::Type(ty)) = self.items[child_id.0].as_ref().map(Item::kind) {
                        if let (Some(name), true) = (ty.name(), ty.is_enum()) {
                            if names_of_typedefs.contains(name) {
                                enum_typedef_combos.insert(*child_id);
                            }
                        }
                    }
                }
            }
        }
        self.enum_typedef_combos = Some(enum_typedef_combos);
    }
    pub fn is_enum_typedef_combo(&self, id: ItemId) -> bool {
        assert!(
            self.in_codegen_phase(),
            "We only compute enum_typedef_combos when we enter codegen",
        );
        self.enum_typedef_combos.as_ref().unwrap().contains(&id)
    }
    fn compute_cannot_derive_debug(&mut self) {
        let _t = self.timer("compute_cannot_derive_debug");
        assert!(self.cannot_derive_debug.is_none());
        if self.opts.derive_debug {
            self.cannot_derive_debug = Some(as_cannot_derive_set(analyze::<derive::Analysis>((
                self,
                DeriveTrait::Debug,
            ))));
        }
    }
    pub fn lookup_can_derive_debug<Id: Into<ItemId>>(&self, id: Id) -> bool {
        let id = id.into();
        assert!(
            self.in_codegen_phase(),
            "We only compute can_derive_debug when we enter codegen"
        );
        !self.cannot_derive_debug.as_ref().unwrap().contains(&id)
    }
    fn compute_cannot_derive_default(&mut self) {
        let _t = self.timer("compute_cannot_derive_default");
        assert!(self.cannot_derive_default.is_none());
        if self.opts.derive_default {
            self.cannot_derive_default = Some(as_cannot_derive_set(analyze::<derive::Analysis>((
                self,
                DeriveTrait::Default,
            ))));
        }
    }
    pub fn lookup_can_derive_default<Id: Into<ItemId>>(&self, id: Id) -> bool {
        let id = id.into();
        assert!(
            self.in_codegen_phase(),
            "We only compute can_derive_default when we enter codegen"
        );
        !self.cannot_derive_default.as_ref().unwrap().contains(&id)
    }
    fn compute_cannot_derive_copy(&mut self) {
        let _t = self.timer("compute_cannot_derive_copy");
        assert!(self.cannot_derive_copy.is_none());
        self.cannot_derive_copy = Some(as_cannot_derive_set(analyze::<derive::Analysis>((
            self,
            DeriveTrait::Copy,
        ))));
    }
    fn compute_cannot_derive_hash(&mut self) {
        let _t = self.timer("compute_cannot_derive_hash");
        assert!(self.cannot_derive_hash.is_none());
        if self.opts.derive_hash {
            self.cannot_derive_hash = Some(as_cannot_derive_set(analyze::<derive::Analysis>((
                self,
                DeriveTrait::Hash,
            ))));
        }
    }
    pub fn lookup_can_derive_hash<Id: Into<ItemId>>(&self, id: Id) -> bool {
        let id = id.into();
        assert!(
            self.in_codegen_phase(),
            "We only compute can_derive_debug when we enter codegen"
        );
        !self.cannot_derive_hash.as_ref().unwrap().contains(&id)
    }
    fn compute_cannot_derive_partialord_partialeq_or_eq(&mut self) {
        let _t = self.timer("compute_cannot_derive_partialord_partialeq_or_eq");
        assert!(self.cannot_derive_partialeq_or_partialord.is_none());
        if self.opts.derive_partialord || self.opts.derive_partialeq || self.opts.derive_eq {
            self.cannot_derive_partialeq_or_partialord =
                Some(analyze::<derive::Analysis>((self, DeriveTrait::PartialEqOrPartialOrd)));
        }
    }
    pub fn lookup_can_derive_partialeq_or_partialord<Id: Into<ItemId>>(&self, id: Id) -> Resolved {
        let id = id.into();
        assert!(
            self.in_codegen_phase(),
            "We only compute can_derive_partialeq_or_partialord when we enter codegen"
        );
        self.cannot_derive_partialeq_or_partialord
            .as_ref()
            .unwrap()
            .get(&id)
            .cloned()
            .unwrap_or(Resolved::Yes)
    }
    pub fn lookup_can_derive_copy<Id: Into<ItemId>>(&self, id: Id) -> bool {
        assert!(
            self.in_codegen_phase(),
            "We only compute can_derive_debug when we enter codegen"
        );
        let id = id.into();
        !self.lookup_has_type_param_in_array(id) && !self.cannot_derive_copy.as_ref().unwrap().contains(&id)
    }
    fn compute_has_type_param_in_array(&mut self) {
        let _t = self.timer("compute_has_type_param_in_array");
        assert!(self.has_type_param_in_array.is_none());
        self.has_type_param_in_array = Some(analyze::<has_type_param::Analysis>(self));
    }
    pub fn lookup_has_type_param_in_array<Id: Into<ItemId>>(&self, id: Id) -> bool {
        assert!(
            self.in_codegen_phase(),
            "We only compute has array when we enter codegen"
        );
        self.has_type_param_in_array.as_ref().unwrap().contains(&id.into())
    }
    fn compute_has_float(&mut self) {
        let _t = self.timer("compute_has_float");
        assert!(self.has_float.is_none());
        if self.opts.derive_eq || self.opts.derive_ord {
            self.has_float = Some(analyze::<has_float::Analysis>(self));
        }
    }
    pub fn lookup_has_float<Id: Into<ItemId>>(&self, id: Id) -> bool {
        assert!(
            self.in_codegen_phase(),
            "We only compute has float when we enter codegen"
        );
        self.has_float.as_ref().unwrap().contains(&id.into())
    }
    pub fn no_partialeq_by_name(&self, it: &Item) -> bool {
        let name = it.path_for_allowlisting(self)[1..].join("::");
        self.opts().no_partialeq_types.matches(name)
    }
    pub fn no_copy_by_name(&self, it: &Item) -> bool {
        let name = it.path_for_allowlisting(self)[1..].join("::");
        self.opts().no_copy_types.matches(name)
    }
    pub fn no_debug_by_name(&self, it: &Item) -> bool {
        let name = it.path_for_allowlisting(self)[1..].join("::");
        self.opts().no_debug_types.matches(name)
    }
    pub fn no_default_by_name(&self, it: &Item) -> bool {
        let name = it.path_for_allowlisting(self)[1..].join("::");
        self.opts().no_default_types.matches(name)
    }
    pub fn no_hash_by_name(&self, it: &Item) -> bool {
        let name = it.path_for_allowlisting(self)[1..].join("::");
        self.opts().no_hash_types.matches(name)
    }
    pub fn must_use_type_by_name(&self, it: &Item) -> bool {
        let name = it.path_for_allowlisting(self)[1..].join("::");
        self.opts().must_use_types.matches(name)
    }
    pub fn wrap_unsafe_ops(&self, tokens: impl ToTokens) -> TokenStream {
        if self.opts.wrap_unsafe_ops {
            quote!(unsafe { #tokens })
        } else {
            tokens.into_token_stream()
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ItemResolver {
    id: ItemId,
    through_type_refs: bool,
    through_type_aliases: bool,
}
impl ItemResolver {
    pub fn new<Id: Into<ItemId>>(id: Id) -> ItemResolver {
        let id = id.into();
        ItemResolver {
            id,
            through_type_refs: false,
            through_type_aliases: false,
        }
    }
    pub fn through_type_refs(mut self) -> ItemResolver {
        self.through_type_refs = true;
        self
    }
    pub fn through_type_aliases(mut self) -> ItemResolver {
        self.through_type_aliases = true;
        self
    }
    pub fn resolve(self, ctx: &Context) -> &Item {
        assert!(ctx.collected_typerefs());
        let mut id = self.id;
        let mut seen_ids = HashSet::default();
        loop {
            let item = ctx.resolve_item(id);
            if !seen_ids.insert(id) {
                return item;
            }
            let ty_kind = item.as_type().map(|t| t.kind());
            match ty_kind {
                Some(&TypeKind::ResolvedTypeRef(next_id)) if self.through_type_refs => {
                    id = next_id.into();
                },
                Some(&TypeKind::Alias(next_id)) if self.through_type_aliases => {
                    id = next_id.into();
                },
                _ => return item,
            }
        }
    }
}
impl ItemId {
    pub fn into_resolver(self) -> ItemResolver {
        self.into()
    }
}
impl<T> From<T> for ItemResolver
where
    T: Into<ItemId>,
{
    fn from(id: T) -> ItemResolver {
        ItemResolver::new(id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartialType {
    decl: Cursor,
    id: ItemId,
}
impl PartialType {
    pub fn new(decl: Cursor, id: ItemId) -> PartialType {
        PartialType { decl, id }
    }
    pub fn decl(&self) -> &Cursor {
        &self.decl
    }
    pub fn id(&self) -> ItemId {
        self.id
    }
}
impl TemplParams for PartialType {
    fn self_templ_params(&self, _ctx: &Context) -> Vec<TypeId> {
        vec![]
    }
    fn num_self_templ_params(&self, _ctx: &Context) -> usize {
        match self.decl().kind() {
            clang_lib::CXCursor_ClassTemplate
            | clang_lib::CXCursor_FunctionTemplate
            | clang_lib::CXCursor_TypeAliasTemplateDecl => {
                let mut num_params = 0;
                self.decl().visit(|c| {
                    match c.kind() {
                        clang_lib::CXCursor_TemplateTypeParameter
                        | clang_lib::CXCursor_TemplateTemplateParameter
                        | clang_lib::CXCursor_NonTypeTemplateParameter => {
                            num_params += 1;
                        },
                        _ => {},
                    };
                    clang_lib::CXChildVisit_Continue
                });
                num_params
            },
            _ => 0,
        }
    }
}
