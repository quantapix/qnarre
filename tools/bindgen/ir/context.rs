use super::super::time::Timer;
use super::analysis::{
    analyze, as_cannot_derive_set, CannotDerive, DeriveTrait, HasDestructorAnalysis, HasFloat, HasTypeParameterInArray,
    HasVtableAnalysis, HasVtableResult, SizednessAnalysis, SizednessResult, UsedTemplateParameters,
};
use super::derive::{
    CanDerive, CanDeriveCopy, CanDeriveDebug, CanDeriveDefault, CanDeriveEq, CanDeriveHash, CanDeriveOrd,
    CanDerivePartialEq, CanDerivePartialOrd,
};
use super::function::Function;
use super::int::IntKind;
use super::item::{IsOpaque, Item, ItemAncestors, ItemSet};
use super::item_kind::ItemKind;
use super::module::{Module, ModuleKind};
use super::template::{TemplateInstantiation, TemplateParameters};
use super::traversal::{self, Edge, ItemTraversal};
use super::ty::{FloatKind, Type, TypeKind};
use crate::clang::{self, Cursor};
use crate::codegen::CodegenError;
use crate::BindgenOptions;
use crate::{Entry, HashMap, HashSet};

use proc_macro2::{Ident, Span, TokenStream};
use quote::ToTokens;
use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap as StdHashMap};
use std::iter::IntoIterator;
use std::mem;

#[derive(Debug, Copy, Clone, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ItemId(usize);

macro_rules! item_id_newtype {
    (
        $( #[$attr:meta] )*
        pub(crate) struct $name:ident(ItemId)
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
        pub(crate) struct $name(ItemId);

        impl $name {
            #[allow(dead_code)]
            pub(crate) fn into_resolver(self) -> ItemResolver {
                let id: ItemId = self.into();
                id.into()
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

        #[allow(dead_code)]
        impl ItemId {
            $( #[$checked_attr] )*
            pub(crate) fn $checked(&self, ctx: &BindgenContext) -> Option<$name> {
                if ctx.resolve_item(*self).kind().$check_method() {
                    Some($name(*self))
                } else {
                    None
                }
            }

            $( #[$expected_attr] )*
            pub(crate) fn $expected(&self, ctx: &BindgenContext) -> $name {
                self.$checked(ctx)
                    .expect(concat!(
                        stringify!($expected),
                        " called with ItemId that points to the wrong ItemKind"
                    ))
            }

            $( #[$unchecked_attr] )*
            pub(crate) fn $unchecked(&self) -> $name {
                $name(*self)
            }
        }
    }
}

item_id_newtype! {
    pub(crate) struct TypeId(ItemId)
    where
        checked = as_type_id with is_type,

        expected = expect_type_id,

        unchecked = as_type_id_unchecked;
}

item_id_newtype! {
    pub(crate) struct ModuleId(ItemId)
    where
        checked = as_module_id with is_module,

        expected = expect_module_id,

        unchecked = as_module_id_unchecked;
}

item_id_newtype! {
    pub(crate) struct VarId(ItemId)
    where
        checked = as_var_id with is_var,

        expected = expect_var_id,

        unchecked = as_var_id_unchecked;
}

item_id_newtype! {
    pub(crate) struct FunctionId(ItemId)
    where
        checked = as_function_id with is_function,

        expected = expect_function_id,

        unchecked = as_function_id_unchecked;
}

impl From<ItemId> for usize {
    fn from(id: ItemId) -> usize {
        id.0
    }
}

impl ItemId {
    pub(crate) fn as_usize(&self) -> usize {
        (*self).into()
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

impl<T> CanDeriveDebug for T
where
    T: Copy + Into<ItemId>,
{
    fn can_derive_debug(&self, ctx: &BindgenContext) -> bool {
        ctx.options().derive_debug && ctx.lookup_can_derive_debug(*self)
    }
}

impl<T> CanDeriveDefault for T
where
    T: Copy + Into<ItemId>,
{
    fn can_derive_default(&self, ctx: &BindgenContext) -> bool {
        ctx.options().derive_default && ctx.lookup_can_derive_default(*self)
    }
}

impl<T> CanDeriveCopy for T
where
    T: Copy + Into<ItemId>,
{
    fn can_derive_copy(&self, ctx: &BindgenContext) -> bool {
        ctx.options().derive_copy && ctx.lookup_can_derive_copy(*self)
    }
}

impl<T> CanDeriveHash for T
where
    T: Copy + Into<ItemId>,
{
    fn can_derive_hash(&self, ctx: &BindgenContext) -> bool {
        ctx.options().derive_hash && ctx.lookup_can_derive_hash(*self)
    }
}

impl<T> CanDerivePartialOrd for T
where
    T: Copy + Into<ItemId>,
{
    fn can_derive_partialord(&self, ctx: &BindgenContext) -> bool {
        ctx.options().derive_partialord && ctx.lookup_can_derive_partialeq_or_partialord(*self) == CanDerive::Yes
    }
}

impl<T> CanDerivePartialEq for T
where
    T: Copy + Into<ItemId>,
{
    fn can_derive_partialeq(&self, ctx: &BindgenContext) -> bool {
        ctx.options().derive_partialeq && ctx.lookup_can_derive_partialeq_or_partialord(*self) == CanDerive::Yes
    }
}

impl<T> CanDeriveEq for T
where
    T: Copy + Into<ItemId>,
{
    fn can_derive_eq(&self, ctx: &BindgenContext) -> bool {
        ctx.options().derive_eq
            && ctx.lookup_can_derive_partialeq_or_partialord(*self) == CanDerive::Yes
            && !ctx.lookup_has_float(*self)
    }
}

impl<T> CanDeriveOrd for T
where
    T: Copy + Into<ItemId>,
{
    fn can_derive_ord(&self, ctx: &BindgenContext) -> bool {
        ctx.options().derive_ord
            && ctx.lookup_can_derive_partialeq_or_partialord(*self) == CanDerive::Yes
            && !ctx.lookup_has_float(*self)
    }
}

#[derive(Eq, PartialEq, Hash, Debug)]
enum TypeKey {
    Usr(String),
    Declaration(Cursor),
}

#[derive(Debug)]
pub(crate) struct BindgenContext {
    items: Vec<Option<Item>>,

    types: HashMap<TypeKey, TypeId>,

    type_params: HashMap<clang::Cursor, TypeId>,

    modules: HashMap<Cursor, ModuleId>,

    root_module: ModuleId,

    current_module: ModuleId,

    semantic_parents: HashMap<clang::Cursor, ItemId>,

    currently_parsed_types: Vec<PartialType>,

    parsed_macros: StdHashMap<Vec<u8>, cexpr::expr::EvalResult>,

    deps: BTreeSet<String>,

    replacements: HashMap<Vec<String>, ItemId>,

    collected_typerefs: bool,

    in_codegen: bool,

    translation_unit: clang::TranslationUnit,

    target_info: clang::TargetInfo,

    options: BindgenOptions,

    generated_bindgen_complex: Cell<bool>,

    allowlisted: Option<ItemSet>,

    blocklisted_types_implement_traits: RefCell<HashMap<DeriveTrait, HashMap<ItemId, CanDerive>>>,

    codegen_items: Option<ItemSet>,

    used_template_parameters: Option<HashMap<ItemId, ItemSet>>,

    need_bitfield_allocation: Vec<ItemId>,

    enum_typedef_combos: Option<HashSet<ItemId>>,

    cannot_derive_debug: Option<HashSet<ItemId>>,

    cannot_derive_default: Option<HashSet<ItemId>>,

    cannot_derive_copy: Option<HashSet<ItemId>>,

    cannot_derive_hash: Option<HashSet<ItemId>>,

    cannot_derive_partialeq_or_partialord: Option<HashMap<ItemId, CanDerive>>,

    sizedness: Option<HashMap<TypeId, SizednessResult>>,

    have_vtable: Option<HashMap<ItemId, HasVtableResult>>,

    have_destructor: Option<HashSet<ItemId>>,

    has_type_param_in_array: Option<HashSet<ItemId>>,

    has_float: Option<HashSet<ItemId>>,
}

struct AllowlistedItemsTraversal<'ctx> {
    ctx: &'ctx BindgenContext,
    traversal: ItemTraversal<'ctx, ItemSet, Vec<ItemId>>,
}

impl<'ctx> Iterator for AllowlistedItemsTraversal<'ctx> {
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

impl<'ctx> AllowlistedItemsTraversal<'ctx> {
    pub(crate) fn new<R>(
        ctx: &'ctx BindgenContext,
        roots: R,
        predicate: for<'a> fn(&'a BindgenContext, Edge) -> bool,
    ) -> Self
    where
        R: IntoIterator<Item = ItemId>,
    {
        AllowlistedItemsTraversal {
            ctx,
            traversal: ItemTraversal::new(ctx, roots, predicate),
        }
    }
}

impl BindgenContext {
    pub(crate) fn new(options: BindgenOptions, input_unsaved_files: &[clang::UnsavedFile]) -> Self {
        let index = clang::Index::new(false, true);

        let parse_options = clang::CXTranslationUnit_DetailedPreprocessingRecord;

        let translation_unit = {
            let _t = Timer::new("translation_unit").with_output(options.time_phases);

            clang::TranslationUnit::parse(&index, "", &options.clang_args, input_unsaved_files, parse_options).expect(
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

        let deps = options.input_headers.iter().cloned().collect();

        BindgenContext {
            items: vec![Some(root_module)],
            deps,
            types: Default::default(),
            type_params: Default::default(),
            modules: Default::default(),
            root_module: root_module_id,
            current_module: root_module_id,
            semantic_parents: Default::default(),
            currently_parsed_types: vec![],
            parsed_macros: Default::default(),
            replacements: Default::default(),
            collected_typerefs: false,
            in_codegen: false,
            translation_unit,
            target_info,
            options,
            generated_bindgen_complex: Cell::new(false),
            allowlisted: None,
            blocklisted_types_implement_traits: Default::default(),
            codegen_items: None,
            used_template_parameters: None,
            need_bitfield_allocation: Default::default(),
            enum_typedef_combos: None,
            cannot_derive_debug: None,
            cannot_derive_default: None,
            cannot_derive_copy: None,
            cannot_derive_hash: None,
            cannot_derive_partialeq_or_partialord: None,
            sizedness: None,
            have_vtable: None,
            have_destructor: None,
            has_type_param_in_array: None,
            has_float: None,
        }
    }

    pub(crate) fn is_target_wasm32(&self) -> bool {
        self.target_info.triple.starts_with("wasm32-")
    }

    pub(crate) fn timer<'a>(&self, name: &'a str) -> Timer<'a> {
        Timer::new(name).with_output(self.options.time_phases)
    }

    pub(crate) fn target_pointer_size(&self) -> usize {
        self.target_info.pointer_width / 8
    }

    pub(crate) fn currently_parsed_types(&self) -> &[PartialType] {
        &self.currently_parsed_types[..]
    }

    pub(crate) fn begin_parsing(&mut self, partial_ty: PartialType) {
        self.currently_parsed_types.push(partial_ty);
    }

    pub(crate) fn finish_parsing(&mut self) -> PartialType {
        self.currently_parsed_types
            .pop()
            .expect("should have been parsing a type, if we finished parsing a type")
    }

    pub(crate) fn include_file(&mut self, filename: String) {
        for cb in &self.options().parse_callbacks {
            cb.include_file(&filename);
        }
        self.deps.insert(filename);
    }

    pub(crate) fn deps(&self) -> &BTreeSet<String> {
        &self.deps
    }

    pub(crate) fn add_item(&mut self, item: Item, declaration: Option<Cursor>, location: Option<Cursor>) {
        debug!(
            "BindgenContext::add_item({:?}, declaration: {:?}, loc: {:?}",
            item, declaration, location
        );
        debug_assert!(
            declaration.is_some()
                || !item.kind().is_type()
                || item.kind().expect_type().is_builtin_or_type_param()
                || item.kind().expect_type().is_opaque(self, &item)
                || item.kind().expect_type().is_unresolved_ref(),
            "Adding a type without declaration?"
        );

        let id = item.id();
        let is_type = item.kind().is_type();
        let is_unnamed = is_type && item.expect_type().name().is_none();
        let is_template_instantiation = is_type && item.expect_type().is_template_instantiation();

        if item.id() != self.root_module {
            self.add_item_to_module(&item);
        }

        if is_type && item.expect_type().is_comp() {
            self.need_bitfield_allocation.push(id);
        }

        let old_item = mem::replace(&mut self.items[id.0], Some(item));
        assert!(
            old_item.is_none(),
            "should not have already associated an item with the given id"
        );

        if !is_type || is_template_instantiation {
            return;
        }
        if let Some(mut declaration) = declaration {
            if !declaration.is_valid() {
                if let Some(location) = location {
                    if location.is_template_like() {
                        declaration = location;
                    }
                }
            }
            declaration = declaration.canonical();
            if !declaration.is_valid() {
                // This could happen, for example, with types like `int*` or
                // similar.
                //
                // Fortunately, we don't care about those types being
                // duplicated, so we can just ignore them.
                debug!(
                    "Invalid declaration {:?} found for type {:?}",
                    declaration,
                    self.resolve_item_fallible(id).unwrap().kind().expect_type()
                );
                return;
            }

            let key = if is_unnamed {
                TypeKey::Declaration(declaration)
            } else if let Some(usr) = declaration.usr() {
                TypeKey::Usr(usr)
            } else {
                warn!("Valid declaration with no USR: {:?}, {:?}", declaration, location);
                TypeKey::Declaration(declaration)
            };

            let old = self.types.insert(key, id.as_type_id_unchecked());
            debug_assert_eq!(old, None);
        }
    }

    fn add_item_to_module(&mut self, item: &Item) {
        assert!(item.id() != self.root_module);
        assert!(self.resolve_item_fallible(item.id()).is_none());

        if let Some(ref mut parent) = self.items[item.parent_id().0] {
            if let Some(module) = parent.as_module_mut() {
                debug!(
                    "add_item_to_module: adding {:?} as child of parent module {:?}",
                    item.id(),
                    item.parent_id()
                );

                module.children_mut().insert(item.id());
                return;
            }
        }

        debug!(
            "add_item_to_module: adding {:?} as child of current module {:?}",
            item.id(),
            self.current_module
        );

        self.items[(self.current_module.0).0]
            .as_mut()
            .expect("Should always have an item for self.current_module")
            .as_module_mut()
            .expect("self.current_module should always be a module")
            .children_mut()
            .insert(item.id());
    }

    pub(crate) fn add_type_param(&mut self, item: Item, definition: clang::Cursor) {
        debug!(
            "BindgenContext::add_type_param: item = {:?}; definition = {:?}",
            item, definition
        );

        assert!(
            item.expect_type().is_type_param(),
            "Should directly be a named type, not a resolved reference or anything"
        );
        assert_eq!(definition.kind(), clang::CXCursor_TemplateTypeParameter);

        self.add_item_to_module(&item);

        let id = item.id();
        let old_item = mem::replace(&mut self.items[id.0], Some(item));
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

    pub(crate) fn get_type_param(&self, definition: &clang::Cursor) -> Option<TypeId> {
        assert_eq!(definition.kind(), clang::CXCursor_TemplateTypeParameter);
        self.type_params.get(definition).cloned()
    }


    #[rustfmt::skip]
    pub(crate) fn rust_mangle<'a>(&self, name: &'a str) -> Cow<'a, str> {
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

    pub(crate) fn rust_ident<S>(&self, name: S) -> Ident
    where
        S: AsRef<str>,
    {
        self.rust_ident_raw(self.rust_mangle(name.as_ref()))
    }

    pub(crate) fn rust_ident_raw<T>(&self, name: T) -> Ident
    where
        T: AsRef<str>,
    {
        Ident::new(name.as_ref(), Span::call_site())
    }

    pub(crate) fn items(&self) -> impl Iterator<Item = (ItemId, &Item)> {
        self.items.iter().enumerate().filter_map(|(index, item)| {
            let item = item.as_ref()?;
            Some((ItemId(index), item))
        })
    }

    pub(crate) fn collected_typerefs(&self) -> bool {
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
        F: (FnOnce(&BindgenContext, &mut Item) -> T),
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
                TypeKind::Comp(..) | TypeKind::TemplateAlias(..) | TypeKind::Enum(..) | TypeKind::Alias(..) => {},
                _ => continue,
            }

            let path = item.path_for_allowlisting(self);
            let replacement = self.replacements.get(&path[1..]);

            if let Some(replacement) = replacement {
                if *replacement != id {
                    // We set this just after parsing the annotation. It's
                    // very unlikely, but this can happen.
                    if self.resolve_item_fallible(*replacement).is_some() {
                        replacements.push((id.expect_type_id(self), replacement.expect_type_id(self)));
                    }
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
                // Same parent and therefore also same containing
                // module. Nothing to do here.
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
                // Already in the correct module.
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

    pub(crate) fn gen<F, Out>(mut self, cb: F) -> Result<(Out, BindgenOptions), CodegenError>
    where
        F: FnOnce(&Self) -> Result<Out, CodegenError>,
    {
        self.in_codegen = true;

        self.resolve_typerefs();
        self.compute_bitfield_units();
        self.process_replacements();

        self.deanonymize_fields();

        self.assert_no_dangling_references();

        self.compute_allowlisted_and_codegen_items();

        self.assert_every_item_in_a_module();

        self.compute_has_vtable();
        self.compute_sizedness();
        self.compute_has_destructor();
        self.find_used_template_parameters();
        self.compute_enum_typedef_combos();
        self.compute_cannot_derive_debug();
        self.compute_cannot_derive_default();
        self.compute_cannot_derive_copy();
        self.compute_has_type_param_in_array();
        self.compute_has_float();
        self.compute_cannot_derive_hash();
        self.compute_cannot_derive_partialord_partialeq_or_eq();

        let ret = cb(&self)?;
        Ok((ret, self.options))
    }

    fn assert_no_dangling_references(&self) {
        if cfg!(feature = "__testing_only_extra_assertions") {
            for _ in self.assert_no_dangling_item_traversal() {
                // The iterator's next method does the asserting for us.
            }
        }
    }

    fn assert_no_dangling_item_traversal(&self) -> traversal::AssertNoDanglingItemsTraversal {
        assert!(self.in_codegen_phase());
        assert!(self.current_module == self.root_module);

        let roots = self.items().map(|(id, _)| id);
        traversal::AssertNoDanglingItemsTraversal::new(self, roots, traversal::all_edges)
    }

    fn assert_every_item_in_a_module(&self) {
        if cfg!(feature = "__testing_only_extra_assertions") {
            assert!(self.in_codegen_phase());
            assert!(self.current_module == self.root_module);

            for (id, _item) in self.items() {
                if id == self.root_module {
                    continue;
                }

                assert!(
                    {
                        let id = id
                            .into_resolver()
                            .through_type_refs()
                            .through_type_aliases()
                            .resolve(self)
                            .id();
                        id.ancestors(self).chain(Some(self.root_module.into())).any(|ancestor| {
                            debug!("Checking if {:?} is a child of {:?}", id, ancestor);
                            self.resolve_item(ancestor)
                                .as_module()
                                .map_or(false, |m| m.children().contains(&id))
                        })
                    },
                    "{:?} should be in some ancestor module's children set",
                    id
                );
            }
        }
    }

    fn compute_sizedness(&mut self) {
        let _t = self.timer("compute_sizedness");
        assert!(self.sizedness.is_none());
        self.sizedness = Some(analyze::<SizednessAnalysis>(self));
    }

    pub(crate) fn lookup_sizedness(&self, id: TypeId) -> SizednessResult {
        assert!(
            self.in_codegen_phase(),
            "We only compute sizedness after we've entered codegen"
        );

        self.sizedness
            .as_ref()
            .unwrap()
            .get(&id)
            .cloned()
            .unwrap_or(SizednessResult::ZeroSized)
    }

    fn compute_has_vtable(&mut self) {
        let _t = self.timer("compute_has_vtable");
        assert!(self.have_vtable.is_none());
        self.have_vtable = Some(analyze::<HasVtableAnalysis>(self));
    }

    pub(crate) fn lookup_has_vtable(&self, id: TypeId) -> HasVtableResult {
        assert!(self.in_codegen_phase(), "We only compute vtables when we enter codegen");

        self.have_vtable
            .as_ref()
            .unwrap()
            .get(&id.into())
            .cloned()
            .unwrap_or(HasVtableResult::No)
    }

    fn compute_has_destructor(&mut self) {
        let _t = self.timer("compute_has_destructor");
        assert!(self.have_destructor.is_none());
        self.have_destructor = Some(analyze::<HasDestructorAnalysis>(self));
    }

    pub(crate) fn lookup_has_destructor(&self, id: TypeId) -> bool {
        assert!(
            self.in_codegen_phase(),
            "We only compute destructors when we enter codegen"
        );

        self.have_destructor.as_ref().unwrap().contains(&id.into())
    }

    fn find_used_template_parameters(&mut self) {
        let _t = self.timer("find_used_template_parameters");
        if self.options.allowlist_recursively {
            let used_params = analyze::<UsedTemplateParameters>(self);
            self.used_template_parameters = Some(used_params);
        } else {
            let mut used_params = HashMap::default();
            for &id in self.allowlisted_items() {
                used_params
                    .entry(id)
                    .or_insert_with(|| id.self_template_params(self).into_iter().map(|p| p.into()).collect());
            }
            self.used_template_parameters = Some(used_params);
        }
    }

    pub(crate) fn uses_template_parameter(&self, item: ItemId, template_param: TypeId) -> bool {
        assert!(
            self.in_codegen_phase(),
            "We only compute template parameter usage as we enter codegen"
        );

        if self.resolve_item(item).is_blocklisted(self) {
            return true;
        }

        let template_param = template_param
            .into_resolver()
            .through_type_refs()
            .through_type_aliases()
            .resolve(self)
            .id();

        self.used_template_parameters
            .as_ref()
            .expect("should have found template parameter usage if we're in codegen")
            .get(&item)
            .map_or(false, |items_used_params| items_used_params.contains(&template_param))
    }

    pub(crate) fn uses_any_template_parameters(&self, item: ItemId) -> bool {
        assert!(
            self.in_codegen_phase(),
            "We only compute template parameter usage as we enter codegen"
        );

        self.used_template_parameters
            .as_ref()
            .expect("should have template parameter usage info in codegen phase")
            .get(&item)
            .map_or(false, |used| !used.is_empty())
    }

    fn add_builtin_item(&mut self, item: Item) {
        debug!("add_builtin_item: item = {:?}", item);
        debug_assert!(item.kind().is_type());
        self.add_item_to_module(&item);
        let id = item.id();
        let old_item = mem::replace(&mut self.items[id.0], Some(item));
        assert!(old_item.is_none(), "Inserted type twice?");
    }

    fn build_root_module(id: ItemId) -> Item {
        let module = Module::new(Some("root".into()), ModuleKind::Normal);
        Item::new(id, None, None, id, ItemKind::Module(module), None)
    }

    pub(crate) fn root_module(&self) -> ModuleId {
        self.root_module
    }

    pub(crate) fn resolve_type(&self, type_id: TypeId) -> &Type {
        self.resolve_item(type_id).kind().expect_type()
    }

    pub(crate) fn resolve_func(&self, func_id: FunctionId) -> &Function {
        self.resolve_item(func_id).kind().expect_function()
    }

    pub(crate) fn safe_resolve_type(&self, type_id: TypeId) -> Option<&Type> {
        self.resolve_item_fallible(type_id).map(|t| t.kind().expect_type())
    }

    pub(crate) fn resolve_item_fallible<Id: Into<ItemId>>(&self, id: Id) -> Option<&Item> {
        self.items.get(id.into().0)?.as_ref()
    }

    pub(crate) fn resolve_item<Id: Into<ItemId>>(&self, item_id: Id) -> &Item {
        let item_id = item_id.into();
        match self.resolve_item_fallible(item_id) {
            Some(item) => item,
            None => panic!("Not an item: {:?}", item_id),
        }
    }

    pub(crate) fn current_module(&self) -> ModuleId {
        self.current_module
    }

    pub(crate) fn add_semantic_parent(&mut self, definition: clang::Cursor, parent_id: ItemId) {
        self.semantic_parents.insert(definition, parent_id);
    }

    pub(crate) fn known_semantic_parent(&self, definition: clang::Cursor) -> Option<ItemId> {
        self.semantic_parents.get(&definition).cloned()
    }

    fn get_declaration_info_for_template_instantiation(
        &self,
        instantiation: &Cursor,
    ) -> Option<(Cursor, ItemId, usize)> {
        instantiation
            .cur_type()
            .canonical_declaration(Some(instantiation))
            .and_then(|canon_decl| {
                self.get_resolved_type(&canon_decl).and_then(|template_decl_id| {
                    let num_params = template_decl_id.num_self_template_params(self);
                    if num_params == 0 {
                        None
                    } else {
                        Some((*canon_decl.cursor(), template_decl_id.into(), num_params))
                    }
                })
            })
            .or_else(|| {
                // If we haven't already parsed the declaration of
                // the template being instantiated, then it *must*
                // be on the stack of types we are currently
                // parsing. If it wasn't then clang would have
                // already errored out before we started
                // constructing our IR because you can't instantiate
                // a template until it is fully defined.
                instantiation
                    .referenced()
                    .and_then(|referenced| {
                        self.currently_parsed_types()
                            .iter()
                            .find(|partial_ty| *partial_ty.decl() == referenced)
                            .cloned()
                    })
                    .and_then(|template_decl| {
                        let num_template_params = template_decl.num_self_template_params(self);
                        if num_template_params == 0 {
                            None
                        } else {
                            Some((*template_decl.decl(), template_decl.id(), num_template_params))
                        }
                    })
            })
    }

    fn instantiate_template(
        &mut self,
        with_id: ItemId,
        template: TypeId,
        ty: &clang::Type,
        location: clang::Cursor,
    ) -> Option<TypeId> {
        let num_expected_args = self.resolve_type(template).num_self_template_params(self);
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
            let idx = children.iter().position(|c| c.kind() == clang::CXCursor_TemplateRef);
            if let Some(idx) = idx {
                if children
                    .iter()
                    .take(idx)
                    .all(|c| c.kind() == clang::CXCursor_NamespaceRef)
                {
                    children = children.into_iter().skip(idx + 1).collect();
                }
            }
        }

        for child in children.iter().rev() {
            match child.kind() {
                clang::CXCursor_TypeRef | clang::CXCursor_TypedefDecl | clang::CXCursor_TypeAliasDecl => {
                    // The `with_id` ID will potentially end up unused if we give up
                    // on this type (for example, because it has const value
                    // template args), so if we pass `with_id` as the parent, it is
                    // potentially a dangling reference. Instead, use the canonical
                    // template declaration as the parent. It is already parsed and
                    // has a known-resolvable `ItemId`.
                    let ty = Item::from_ty_or_ref(child.cur_type(), *child, Some(template.into()), self);
                    args.push(ty);
                },
                clang::CXCursor_TemplateRef => {
                    let (template_decl_cursor, template_decl_id, num_expected_template_args) =
                        self.get_declaration_info_for_template_instantiation(child)?;

                    if num_expected_template_args == 0 || child.has_at_least_num_children(num_expected_template_args) {
                        // Do a happy little parse. See comment in the TypeRef
                        // match arm about parent IDs.
                        let ty = Item::from_ty_or_ref(child.cur_type(), *child, Some(template.into()), self);
                        args.push(ty);
                    } else {
                        // This is the case mentioned in the doc comment where
                        // clang gives us a flattened AST and we have to
                        // reconstruct which template arguments go to which
                        // instantiation :(
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
                        let sub_inst = TemplateInstantiation::new(
                            // This isn't guaranteed to be a type that we've
                            // already finished parsing yet.
                            template_decl_id.as_type_id_unchecked(),
                            sub_args,
                        );
                        let sub_kind = TypeKind::TemplateInstantiation(sub_inst);
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

                        // Bypass all the validations in add_item explicitly.
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
        let type_kind = TypeKind::TemplateInstantiation(TemplateInstantiation::new(template, args));
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

    pub(crate) fn get_resolved_type(&self, decl: &clang::CanonicalTypeDeclaration) -> Option<TypeId> {
        self.types
            .get(&TypeKey::Declaration(*decl.cursor()))
            .or_else(|| decl.cursor().usr().and_then(|usr| self.types.get(&TypeKey::Usr(usr))))
            .cloned()
    }

    pub(crate) fn builtin_or_resolved_ty(
        &mut self,
        with_id: ItemId,
        parent_id: Option<ItemId>,
        ty: &clang::Type,
        location: Option<clang::Cursor>,
    ) -> Option<TypeId> {
        use clang::{CXCursor_TypeAliasTemplateDecl, CXCursor_TypeRef};
        debug!(
            "builtin_or_resolved_ty: {:?}, {:?}, {:?}, {:?}",
            ty, location, with_id, parent_id
        );

        if let Some(decl) = ty.canonical_declaration(location.as_ref()) {
            if let Some(id) = self.get_resolved_type(&decl) {
                debug!("Already resolved ty {:?}, {:?}, {:?} {:?}", id, decl, ty, location);
                // If the declaration already exists, then either:
                //
                //   * the declaration is a template declaration of some sort,
                //     and we are looking at an instantiation or specialization
                //     of it, or
                //   * we have already parsed and resolved this type, and
                //     there's nothing left to do.
                if let Some(location) = location {
                    if decl.cursor().is_template_like() && *ty != decl.cursor().cur_type() {
                        // For specialized type aliases, there's no way to get the
                        // template parameters as of this writing (for a struct
                        // specialization we wouldn't be in this branch anyway).
                        //
                        // Explicitly return `None` if there aren't any
                        // unspecialized parameters (contains any `TypeRef`) so we
                        // resolve the canonical type if there is one and it's
                        // exposed.
                        //
                        // This is _tricky_, I know :(
                        if decl.cursor().kind() == CXCursor_TypeAliasTemplateDecl
                            && !location.contains_cursor(CXCursor_TypeRef)
                            && ty.canonical_type().is_valid_and_exposed()
                        {
                            return None;
                        }

                        return self.instantiate_template(with_id, id, ty, location).or(Some(id));
                    }
                }

                return Some(self.build_ty_wrapper(with_id, id, parent_id, ty));
            }
        }

        debug!("Not resolved, maybe builtin?");
        self.build_builtin_ty(ty)
    }

    pub(crate) fn build_ty_wrapper(
        &mut self,
        with_id: ItemId,
        wrapped_id: TypeId,
        parent_id: Option<ItemId>,
        ty: &clang::Type,
    ) -> TypeId {
        self.build_wrapper(with_id, wrapped_id, parent_id, ty, ty.is_const())
    }

    pub(crate) fn build_const_wrapper(
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

    pub(crate) fn next_item_id(&mut self) -> ItemId {
        let ret = ItemId(self.items.len());
        self.items.push(None);
        ret
    }

    fn build_builtin_ty(&mut self, ty: &clang::Type) -> Option<TypeId> {
        use clang::*;
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

    pub(crate) fn translation_unit(&self) -> &clang::TranslationUnit {
        &self.translation_unit
    }

    pub(crate) fn parsed_macro(&self, macro_name: &[u8]) -> bool {
        self.parsed_macros.contains_key(macro_name)
    }

    pub(crate) fn parsed_macros(&self) -> &StdHashMap<Vec<u8>, cexpr::expr::EvalResult> {
        debug_assert!(!self.in_codegen_phase());
        &self.parsed_macros
    }

    pub(crate) fn note_parsed_macro(&mut self, id: Vec<u8>, value: cexpr::expr::EvalResult) {
        self.parsed_macros.insert(id, value);
    }

    pub(crate) fn in_codegen_phase(&self) -> bool {
        self.in_codegen
    }

    pub(crate) fn replace(&mut self, name: &[String], potential_ty: ItemId) {
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

    pub(crate) fn is_replaced_type<Id: Into<ItemId>>(&self, path: &[String], id: Id) -> bool {
        let id = id.into();
        matches!(self.replacements.get(path), Some(replaced_by) if *replaced_by != id)
    }

    pub(crate) fn opaque_by_name(&self, path: &[String]) -> bool {
        debug_assert!(self.in_codegen_phase(), "You're not supposed to call this yet");
        self.options.opaque_types.matches(path[1..].join("::"))
    }

    pub(crate) fn options(&self) -> &BindgenOptions {
        &self.options
    }

    fn tokenize_namespace(&self, cursor: &clang::Cursor) -> (Option<String>, ModuleKind) {
        assert_eq!(cursor.kind(), ::clang::CXCursor_Namespace, "Be a nice person");

        let mut module_name = None;
        let spelling = cursor.spelling();
        if !spelling.is_empty() {
            module_name = Some(spelling)
        }

        let mut kind = ModuleKind::Normal;
        let mut looking_for_name = false;
        for token in cursor.tokens().iter() {
            match token.spelling() {
                b"inline" => {
                    debug_assert!(kind != ModuleKind::Inline, "Multiple inline keywords?");
                    kind = ModuleKind::Inline;
                    // When hitting a nested inline namespace we get a spelling
                    // that looks like ["inline", "foo"]. Deal with it properly.
                    looking_for_name = true;
                },
                // The double colon allows us to handle nested namespaces like
                // namespace foo::bar { }
                //
                // libclang still gives us two namespace cursors, which is cool,
                // but the tokenization of the second begins with the double
                // colon. That's ok, so we only need to handle the weird
                // tokenization here.
                b"namespace" | b"::" => {
                    looking_for_name = true;
                },
                b"{" => {
                    // This should be an anonymous namespace.
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
                        // This is _likely_, but not certainly, a macro that's
                        // been placed just before the namespace keyword.
                        // Unfortunately, clang tokens don't let us easily see
                        // through the ifdef tokens, so we don't know what this
                        // token should really be. Instead of panicking though,
                        // we warn the user that we assumed the token was blank,
                        // and then move on.
                        //
                        // See also https://github.com/rust-lang/rust-bindgen/issues/1676.
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

    pub(crate) fn module(&mut self, cursor: clang::Cursor) -> ModuleId {
        use clang::*;
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

    pub(crate) fn with_module<F>(&mut self, module_id: ModuleId, cb: F)
    where
        F: FnOnce(&mut Self),
    {
        debug_assert!(self.resolve_item(module_id).kind().is_module(), "Wat");

        let previous_id = self.current_module;
        self.current_module = module_id;

        cb(self);

        self.current_module = previous_id;
    }

    pub(crate) fn allowlisted_items(&self) -> &ItemSet {
        assert!(self.in_codegen_phase());
        assert!(self.current_module == self.root_module);

        self.allowlisted.as_ref().unwrap()
    }

    pub(crate) fn blocklisted_type_implements_trait(&self, item: &Item, derive_trait: DeriveTrait) -> CanDerive {
        assert!(self.in_codegen_phase());
        assert!(self.current_module == self.root_module);

        *self
            .blocklisted_types_implement_traits
            .borrow_mut()
            .entry(derive_trait)
            .or_default()
            .entry(item.id())
            .or_insert_with(|| {
                item.expect_type()
                    .name()
                    .and_then(|name| {
                        if self.options.parse_callbacks.is_empty() {
                            // Sized integer types from <stdint.h> get mapped to Rust primitive
                            // types regardless of whether they are blocklisted, so ensure that
                            // standard traits are considered derivable for them too.
                            if self.is_stdint_type(name) {
                                Some(CanDerive::Yes)
                            } else {
                                Some(CanDerive::No)
                            }
                        } else {
                            self.options
                                .last_callback(|cb| cb.blocklisted_type_implements_trait(name, derive_trait))
                        }
                    })
                    .unwrap_or(CanDerive::No)
            })
    }

    pub(crate) fn is_stdint_type(&self, name: &str) -> bool {
        match name {
            "int8_t" | "uint8_t" | "int16_t" | "uint16_t" | "int32_t" | "uint32_t" | "int64_t" | "uint64_t"
            | "uintptr_t" | "intptr_t" | "ptrdiff_t" => true,
            "size_t" | "ssize_t" => self.options.size_t_is_usize,
            _ => false,
        }
    }

    pub(crate) fn codegen_items(&self) -> &ItemSet {
        assert!(self.in_codegen_phase());
        assert!(self.current_module == self.root_module);
        self.codegen_items.as_ref().unwrap()
    }

    fn compute_allowlisted_and_codegen_items(&mut self) {
        assert!(self.in_codegen_phase());
        assert!(self.current_module == self.root_module);
        assert!(self.allowlisted.is_none());
        let _t = self.timer("compute_allowlisted_and_codegen_items");

        let roots = {
            let mut roots = self
                .items()
                // Only consider roots that are enabled for codegen.
                .filter(|&(_, item)| item.is_enabled_for_codegen(self))
                .filter(|&(_, item)| {
                    // If nothing is explicitly allowlisted, then everything is fair
                    // game.
                    if self.options().allowlisted_types.is_empty() &&
                        self.options().allowlisted_functions.is_empty() &&
                        self.options().allowlisted_vars.is_empty() &&
                        self.options().allowlisted_files.is_empty()
                    {
                        return true;
                    }

                    // If this is a type that explicitly replaces another, we assume
                    // you know what you're doing.
                    if item.annotations().use_instead_of().is_some() {
                        return true;
                    }

                    // Items with a source location in an explicitly allowlisted file
                    // are always included.
                    if !self.options().allowlisted_files.is_empty() {
                        if let Some(location) = item.location() {
                            let (file, _, _, _) = location.location();
                            if let Some(filename) = file.name() {
                                if self
                                    .options()
                                    .allowlisted_files
                                    .matches(filename)
                                {
                                    return true;
                                }
                            }
                        }
                    }

                    let name = item.path_for_allowlisting(self)[1..].join("::");
                    debug!("allowlisted_items: testing {:?}", name);
                    match *item.kind() {
                        ItemKind::Module(..) => true,
                        ItemKind::Function(_) => {
                            self.options().allowlisted_functions.matches(&name)
                        }
                        ItemKind::Var(_) => {
                            self.options().allowlisted_vars.matches(&name)
                        }
                        ItemKind::Type(ref ty) => {
                            if self.options().allowlisted_types.matches(&name) {
                                return true;
                            }

                            // Auto-allowlist types that don't need code
                            // generation if not allowlisting recursively, to
                            // make the #[derive] analysis not be lame.
                            if !self.options().allowlist_recursively {
                                match *ty.kind() {
                                    TypeKind::Void |
                                    TypeKind::NullPtr |
                                    TypeKind::Int(..) |
                                    TypeKind::Float(..) |
                                    TypeKind::Complex(..) |
                                    TypeKind::Array(..) |
                                    TypeKind::Vector(..) |
                                    TypeKind::Pointer(..) |
                                    TypeKind::Reference(..) |
                                    TypeKind::Function(..) |
                                    TypeKind::ResolvedTypeRef(..) |
                                    TypeKind::Opaque |
                                    TypeKind::TypeParam => return true,
                                    _ => {}
                                }
                                if self.is_stdint_type(&name) {
                                    return true;
                                }
                            }

                            // Unnamed top-level enums are special and we
                            // allowlist them via the `allowlisted_vars` filter,
                            // since they're effectively top-level constants,
                            // and there's no way for them to be referenced
                            // consistently.
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

                            let mut prefix_path =
                                parent.path_for_allowlisting(self).clone();
                            enum_.variants().iter().any(|variant| {
                                prefix_path.push(
                                    variant.name_for_allowlisting().into(),
                                );
                                let name = prefix_path[1..].join("::");
                                prefix_path.pop().unwrap();
                                self.options().allowlisted_vars.matches(name)
                            })
                        }
                    }
                })
                .map(|(id, _)| id)
                .collect::<Vec<_>>();

            roots.reverse();
            roots
        };

        let allowlisted_items_predicate = if self.options().allowlist_recursively {
            traversal::all_edges
        } else {
            traversal::only_inner_type_edges
        };

        let allowlisted =
            AllowlistedItemsTraversal::new(self, roots.clone(), allowlisted_items_predicate).collect::<ItemSet>();

        let codegen_items = if self.options().allowlist_recursively {
            AllowlistedItemsTraversal::new(self, roots, traversal::codegen_edges).collect::<ItemSet>()
        } else {
            allowlisted.clone()
        };

        self.allowlisted = Some(allowlisted);
        self.codegen_items = Some(codegen_items);

        for item in self.options().allowlisted_functions.unmatched_items() {
            unused_regex_diagnostic(item, "--allowlist-function", self);
        }

        for item in self.options().allowlisted_vars.unmatched_items() {
            unused_regex_diagnostic(item, "--allowlist-var", self);
        }

        for item in self.options().allowlisted_types.unmatched_items() {
            unused_regex_diagnostic(item, "--allowlist-type", self);
        }
    }

    pub(crate) fn trait_prefix(&self) -> Ident {
        if self.options().use_core {
            self.rust_ident_raw("core")
        } else {
            self.rust_ident_raw("std")
        }
    }

    pub(crate) fn generated_bindgen_complex(&self) {
        self.generated_bindgen_complex.set(true)
    }

    pub(crate) fn need_bindgen_complex_type(&self) -> bool {
        self.generated_bindgen_complex.get()
    }

    fn compute_enum_typedef_combos(&mut self) {
        let _t = self.timer("compute_enum_typedef_combos");
        assert!(self.enum_typedef_combos.is_none());

        let mut enum_typedef_combos = HashSet::default();
        for item in &self.items {
            if let Some(ItemKind::Module(module)) = item.as_ref().map(Item::kind) {
                // Find typedefs in this module, and build set of their names.
                let mut names_of_typedefs = HashSet::default();
                for child_id in module.children() {
                    if let Some(ItemKind::Type(ty)) = self.items[child_id.0].as_ref().map(Item::kind) {
                        if let (Some(name), TypeKind::Alias(type_id)) = (ty.name(), ty.kind()) {
                            // We disregard aliases that refer to the enum
                            // itself, such as in `typedef enum { ... } Enum;`.
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

                // Find enums in this module, and record the ID of each one that
                // has a typedef.
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

    pub(crate) fn is_enum_typedef_combo(&self, id: ItemId) -> bool {
        assert!(
            self.in_codegen_phase(),
            "We only compute enum_typedef_combos when we enter codegen",
        );
        self.enum_typedef_combos.as_ref().unwrap().contains(&id)
    }

    fn compute_cannot_derive_debug(&mut self) {
        let _t = self.timer("compute_cannot_derive_debug");
        assert!(self.cannot_derive_debug.is_none());
        if self.options.derive_debug {
            self.cannot_derive_debug = Some(as_cannot_derive_set(analyze::<CannotDerive>((
                self,
                DeriveTrait::Debug,
            ))));
        }
    }

    pub(crate) fn lookup_can_derive_debug<Id: Into<ItemId>>(&self, id: Id) -> bool {
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
        if self.options.derive_default {
            self.cannot_derive_default = Some(as_cannot_derive_set(analyze::<CannotDerive>((
                self,
                DeriveTrait::Default,
            ))));
        }
    }

    pub(crate) fn lookup_can_derive_default<Id: Into<ItemId>>(&self, id: Id) -> bool {
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
        self.cannot_derive_copy = Some(as_cannot_derive_set(analyze::<CannotDerive>((self, DeriveTrait::Copy))));
    }

    fn compute_cannot_derive_hash(&mut self) {
        let _t = self.timer("compute_cannot_derive_hash");
        assert!(self.cannot_derive_hash.is_none());
        if self.options.derive_hash {
            self.cannot_derive_hash = Some(as_cannot_derive_set(analyze::<CannotDerive>((self, DeriveTrait::Hash))));
        }
    }

    pub(crate) fn lookup_can_derive_hash<Id: Into<ItemId>>(&self, id: Id) -> bool {
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
        if self.options.derive_partialord || self.options.derive_partialeq || self.options.derive_eq {
            self.cannot_derive_partialeq_or_partialord =
                Some(analyze::<CannotDerive>((self, DeriveTrait::PartialEqOrPartialOrd)));
        }
    }

    pub(crate) fn lookup_can_derive_partialeq_or_partialord<Id: Into<ItemId>>(&self, id: Id) -> CanDerive {
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
            .unwrap_or(CanDerive::Yes)
    }

    pub(crate) fn lookup_can_derive_copy<Id: Into<ItemId>>(&self, id: Id) -> bool {
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
        self.has_type_param_in_array = Some(analyze::<HasTypeParameterInArray>(self));
    }

    pub(crate) fn lookup_has_type_param_in_array<Id: Into<ItemId>>(&self, id: Id) -> bool {
        assert!(
            self.in_codegen_phase(),
            "We only compute has array when we enter codegen"
        );

        self.has_type_param_in_array.as_ref().unwrap().contains(&id.into())
    }

    fn compute_has_float(&mut self) {
        let _t = self.timer("compute_has_float");
        assert!(self.has_float.is_none());
        if self.options.derive_eq || self.options.derive_ord {
            self.has_float = Some(analyze::<HasFloat>(self));
        }
    }

    pub(crate) fn lookup_has_float<Id: Into<ItemId>>(&self, id: Id) -> bool {
        assert!(
            self.in_codegen_phase(),
            "We only compute has float when we enter codegen"
        );

        self.has_float.as_ref().unwrap().contains(&id.into())
    }

    pub(crate) fn no_partialeq_by_name(&self, item: &Item) -> bool {
        let name = item.path_for_allowlisting(self)[1..].join("::");
        self.options().no_partialeq_types.matches(name)
    }

    pub(crate) fn no_copy_by_name(&self, item: &Item) -> bool {
        let name = item.path_for_allowlisting(self)[1..].join("::");
        self.options().no_copy_types.matches(name)
    }

    pub(crate) fn no_debug_by_name(&self, item: &Item) -> bool {
        let name = item.path_for_allowlisting(self)[1..].join("::");
        self.options().no_debug_types.matches(name)
    }

    pub(crate) fn no_default_by_name(&self, item: &Item) -> bool {
        let name = item.path_for_allowlisting(self)[1..].join("::");
        self.options().no_default_types.matches(name)
    }

    pub(crate) fn no_hash_by_name(&self, item: &Item) -> bool {
        let name = item.path_for_allowlisting(self)[1..].join("::");
        self.options().no_hash_types.matches(name)
    }

    pub(crate) fn must_use_type_by_name(&self, item: &Item) -> bool {
        let name = item.path_for_allowlisting(self)[1..].join("::");
        self.options().must_use_types.matches(name)
    }

    pub(crate) fn wrap_unsafe_ops(&self, tokens: impl ToTokens) -> TokenStream {
        if self.options.wrap_unsafe_ops {
            quote!(unsafe { #tokens })
        } else {
            tokens.into_token_stream()
        }
    }

    pub(crate) fn wrap_static_fns_suffix(&self) -> &str {
        self.options()
            .wrap_static_fns_suffix
            .as_deref()
            .unwrap_or(crate::DEFAULT_NON_EXTERN_FNS_SUFFIX)
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct ItemResolver {
    id: ItemId,
    through_type_refs: bool,
    through_type_aliases: bool,
}

impl ItemId {
    pub(crate) fn into_resolver(self) -> ItemResolver {
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

impl ItemResolver {
    pub(crate) fn new<Id: Into<ItemId>>(id: Id) -> ItemResolver {
        let id = id.into();
        ItemResolver {
            id,
            through_type_refs: false,
            through_type_aliases: false,
        }
    }

    pub(crate) fn through_type_refs(mut self) -> ItemResolver {
        self.through_type_refs = true;
        self
    }

    pub(crate) fn through_type_aliases(mut self) -> ItemResolver {
        self.through_type_aliases = true;
        self
    }

    pub(crate) fn resolve(self, ctx: &BindgenContext) -> &Item {
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
                // We intentionally ignore template aliases here, as they are
                // more complicated, and don't represent a simple renaming of
                // some type.
                Some(&TypeKind::Alias(next_id)) if self.through_type_aliases => {
                    id = next_id.into();
                },
                _ => return item,
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PartialType {
    decl: Cursor,
    id: ItemId,
}

impl PartialType {
    pub(crate) fn new(decl: Cursor, id: ItemId) -> PartialType {
        PartialType { decl, id }
    }

    pub(crate) fn decl(&self) -> &Cursor {
        &self.decl
    }

    pub(crate) fn id(&self) -> ItemId {
        self.id
    }
}

impl TemplateParameters for PartialType {
    fn self_template_params(&self, _ctx: &BindgenContext) -> Vec<TypeId> {
        vec![]
    }

    fn num_self_template_params(&self, _ctx: &BindgenContext) -> usize {
        match self.decl().kind() {
            clang::CXCursor_ClassTemplate
            | clang::CXCursor_FunctionTemplate
            | clang::CXCursor_TypeAliasTemplateDecl => {
                let mut num_params = 0;
                self.decl().visit(|c| {
                    match c.kind() {
                        clang::CXCursor_TemplateTypeParameter
                        | clang::CXCursor_TemplateTemplateParameter
                        | clang::CXCursor_NonTypeTemplateParameter => {
                            num_params += 1;
                        },
                        _ => {},
                    };
                    clang::CXChildVisit_Continue
                });
                num_params
            },
            _ => 0,
        }
    }
}

fn unused_regex_diagnostic(item: &str, name: &str, _ctx: &BindgenContext) {
    warn!("unused option: {} {}", name, item);

    #[cfg(feature = "experimental")]
    if _ctx.options().emit_diagnostics {
        use crate::diagnostics::{Diagnostic, Level};

        Diagnostic::default()
            .with_title(format!("Unused regular expression: `{}`.", item), Level::Warn)
            .add_annotation(
                format!("This regular expression was passed to `{}`.", name),
                Level::Note,
            )
            .display();
    }
}
