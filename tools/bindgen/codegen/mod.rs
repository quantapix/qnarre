use self::struct_layout::StructLayoutTracker;
use super::Opts;
use crate::callbacks::{DeriveInfo, TypeKind as DeriveTypeKind};
use crate::ir::analysis::{HasVtable, Sizedness};
use crate::ir::annotations::{Annotations, FieldAccessorKind, FieldVisibilityKind};
use crate::ir::comp::{Bitfield, BitfieldUnit, CompInfo, CompKind, Field, FieldData, FieldMethods, Method, MethodKind};
use crate::ir::derive::{
    CanDeriveCopy, CanDeriveDebug, CanDeriveDefault, CanDeriveEq, CanDeriveHash, CanDeriveOrd, CanDerivePartialEq,
    CanDerivePartialOrd, Resolved,
};
use crate::ir::dot;
use crate::ir::enum_ty::{Enum, EnumVariant, EnumVariantValue};
use crate::ir::function::{Abi, ClangAbi, FnKind, FnSig, Function, Linkage};
use crate::ir::int::IntKind;
use crate::ir::item::{CanonName, CanonPath, IsOpaque, Item};
use crate::ir::item_kind::ItemKind;
use crate::ir::layout::Layout;
use crate::ir::module::Module;
use crate::ir::template::{AsTemplParam, TemplInstantiation, TemplParams};
use crate::ir::typ::{Type, TypeKind};
use crate::ir::var::Var;
use crate::ir::{Context, ItemId};
use crate::{Entry, HashMap, HashSet};
use proc_macro2::{self, Ident, Span};
use quote::TokenStreamExt;
use std::borrow::Cow;
use std::cell::Cell;
use std::collections::VecDeque;
use std::ffi::CStr;
use std::fmt::{self, Write};
use std::ops;
use std::str::{self, FromStr};

mod dyngen {
    use crate::codegen;
    use crate::ir::function::ClangAbi;
    use crate::ir::Context;
    use proc_macro2::Ident;
    #[derive(Default)]
    pub struct DynItems {
        struct_members: Vec<proc_macro2::TokenStream>,
        struct_implementation: Vec<proc_macro2::TokenStream>,
        constructor_inits: Vec<proc_macro2::TokenStream>,
        init_fields: Vec<proc_macro2::TokenStream>,
    }
    impl DynItems {
        pub fn new() -> Self {
            Self::default()
        }
        pub fn get_tokens(&self, lib_ident: Ident, ctx: &Context) -> proc_macro2::TokenStream {
            let struct_members = &self.struct_members;
            let constructor_inits = &self.constructor_inits;
            let init_fields = &self.init_fields;
            let struct_implementation = &self.struct_implementation;
            let from_library = if ctx.opts().wrap_unsafe_ops {
                quote!(unsafe { Self::from_library(library) })
            } else {
                quote!(Self::from_library(library))
            };
            quote! {
                extern crate libloading;
                pub struct #lib_ident {
                    __library: ::libloading::Library,
                    #(#struct_members)*
                }
                impl #lib_ident {
                    pub unsafe fn new<P>(
                        path: P
                    ) -> Result<Self, ::libloading::Error>
                    where P: AsRef<::std::ffi::OsStr> {
                        let library = ::libloading::Library::new(path)?;
                        #from_library
                    }
                    pub unsafe fn from_library<L>(
                        library: L
                    ) -> Result<Self, ::libloading::Error>
                    where L: Into<::libloading::Library> {
                        let __library = library.into();
                        #( #constructor_inits )*
                        Ok(#lib_ident {
                            __library,
                            #( #init_fields ),*
                        })
                    }
                    #( #struct_implementation )*
                }
            }
        }
        #[allow(clippy::too_many_arguments)]
        pub fn push(
            &mut self,
            ident: Ident,
            abi: ClangAbi,
            is_variadic: bool,
            is_required: bool,
            args: Vec<proc_macro2::TokenStream>,
            args_identifiers: Vec<proc_macro2::TokenStream>,
            ret: proc_macro2::TokenStream,
            ret_ty: proc_macro2::TokenStream,
            attributes: Vec<proc_macro2::TokenStream>,
            ctx: &Context,
        ) {
            if !is_variadic {
                assert_eq!(args.len(), args_identifiers.len());
            }
            let signature = quote! { unsafe extern #abi fn ( #( #args),* ) #ret };
            let member = if is_required {
                signature
            } else {
                quote! { Result<#signature, ::libloading::Error> }
            };
            self.struct_members.push(quote! {
                pub #ident: #member,
            });
            let fn_ = if is_required {
                quote! { self.#ident }
            } else {
                quote! { self.#ident.as_ref().expect("Expected function, got error.") }
            };
            let call_body = if ctx.opts().wrap_unsafe_ops {
                quote!(unsafe { (#fn_)(#( #args_identifiers ),*) })
            } else {
                quote!((#fn_)(#( #args_identifiers ),*) )
            };
            if !is_variadic {
                self.struct_implementation.push(quote! {
                    #(#attributes)*
                    pub unsafe fn #ident ( &self, #( #args ),* ) #ret_ty {
                        #call_body
                    }
                });
            }
            let ident_str = codegen::utils::ast_ty::cstr_expr(ident.to_string());
            let library_get = if ctx.opts().wrap_unsafe_ops {
                quote!(unsafe { __library.get(#ident_str) })
            } else {
                quote!(__library.get(#ident_str))
            };
            self.constructor_inits.push(if is_required {
                quote! {
                    let #ident = #library_get.map(|sym| *sym)?;
                }
            } else {
                quote! {
                    let #ident = #library_get.map(|sym| *sym);
                }
            });
            self.init_fields.push(quote! {
                #ident
            });
        }
    }
}
mod error {
    use std::error;
    use std::fmt;
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum Error {
        NoLayoutForOpaqueBlob,
        InstantiationOfOpaqueType,
    }
    impl fmt::Display for Error {
        fn fmt(&self, x: &mut fmt::Formatter) -> fmt::Result {
            x.write_str(match *self {
                Error::NoLayoutForOpaqueBlob => "Tried to generate an opaque blob, but had no layout",
                Error::InstantiationOfOpaqueType => {
                    "Instantiation of opaque template type or partial template specialization"
                },
            })
        }
    }
    impl error::Error for Error {}
    pub type Result<T> = ::std::result::Result<T, Error>;
}
mod impl_debug;
mod postproc;
pub mod utils;
use utils::variation;

use self::dyngen::DynItems;
use self::utils::attributes;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CodegenError {
    Serialize { msg: String, loc: String },
    Io(String),
}
impl From<std::io::Error> for CodegenError {
    fn from(x: std::io::Error) -> Self {
        Self::Io(x.to_string())
    }
}
impl fmt::Display for CodegenError {
    fn fmt(&self, x: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Serialize { msg, loc } => {
                write!(x, "serialization error at {}: {}", loc, msg)
            },
            Self::Io(x2) => x2.fmt(x),
        }
    }
}

pub static CONSTIFIED_ENUM_MODULE_REPR_NAME: &str = "Type";
fn top_level_path(ctx: &Context, it: &Item) -> Vec<proc_macro2::TokenStream> {
    let mut path = vec![quote! { self }];
    if ctx.opts().enable_cxx_namespaces {
        for _ in 0..it.codegen_depth(ctx) {
            path.push(quote! { super });
        }
    }
    path
}
fn root_import(ctx: &Context, it: &Item) -> proc_macro2::TokenStream {
    assert!(ctx.opts().enable_cxx_namespaces, "Somebody messed it up");
    assert!(it.is_module());
    let mut path = top_level_path(ctx, it);
    let root = ctx.root_module().canon_name(ctx);
    let root_ident = ctx.rust_ident(root);
    path.push(quote! { #root_ident });
    let mut tokens = quote! {};
    tokens.append_separated(path, quote!(::));
    quote! {
        #[allow(unused_imports)]
        use #tokens ;
    }
}

bitflags! {
    #[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    struct DerivableTraits: u16 {
        const DEBUG       = 1 << 0;
        const DEFAULT     = 1 << 1;
        const COPY        = 1 << 2;
        const CLONE       = 1 << 3;
        const HASH        = 1 << 4;
        const PARTIAL_ORD = 1 << 5;
        const ORD         = 1 << 6;
        const PARTIAL_EQ  = 1 << 7;
        const EQ          = 1 << 8;
    }
}
fn derives_of_item(it: &Item, ctx: &Context, packed: bool) -> DerivableTraits {
    let mut ys = DerivableTraits::empty();
    let ps = it.all_templ_params(ctx);
    if it.can_derive_copy(ctx) && !it.annotations().disallow_copy() {
        ys |= DerivableTraits::COPY;
        ys |= DerivableTraits::CLONE;
    } else if packed {
        return ys;
    }
    if it.can_derive_debug(ctx) && !it.annotations().disallow_debug() {
        ys |= DerivableTraits::DEBUG;
    }
    if it.can_derive_default(ctx) && !it.annotations().disallow_default() {
        ys |= DerivableTraits::DEFAULT;
    }
    if it.can_derive_hash(ctx) {
        ys |= DerivableTraits::HASH;
    }
    if it.can_derive_partialord(ctx) {
        ys |= DerivableTraits::PARTIAL_ORD;
    }
    if it.can_derive_ord(ctx) {
        ys |= DerivableTraits::ORD;
    }
    if it.can_derive_partialeq(ctx) {
        ys |= DerivableTraits::PARTIAL_EQ;
    }
    if it.can_derive_eq(ctx) {
        ys |= DerivableTraits::EQ;
    }
    ys
}

impl From<DerivableTraits> for Vec<&'static str> {
    fn from(derivable_traits: DerivableTraits) -> Vec<&'static str> {
        [
            (DerivableTraits::DEBUG, "Debug"),
            (DerivableTraits::DEFAULT, "Default"),
            (DerivableTraits::COPY, "Copy"),
            (DerivableTraits::CLONE, "Clone"),
            (DerivableTraits::HASH, "Hash"),
            (DerivableTraits::PARTIAL_ORD, "PartialOrd"),
            (DerivableTraits::ORD, "Ord"),
            (DerivableTraits::PARTIAL_EQ, "PartialEq"),
            (DerivableTraits::EQ, "Eq"),
        ]
        .iter()
        .filter_map(|&(flag, derive)| Some(derive).filter(|_| derivable_traits.contains(flag)))
        .collect()
    }
}

struct CodegenResult<'a> {
    items: Vec<proc_macro2::TokenStream>,
    dynamic_items: DynItems,
    codegen_id: &'a Cell<usize>,
    saw_bindgen_union: bool,
    saw_incomplete_array: bool,
    saw_block: bool,
    saw_bitfield_unit: bool,
    items_seen: HashSet<ItemId>,
    functions_seen: HashSet<String>,
    vars_seen: HashSet<String>,
    overload_counters: HashMap<String, u32>,
    items_to_serialize: Vec<ItemId>,
}
impl<'a> CodegenResult<'a> {
    fn new(codegen_id: &'a Cell<usize>) -> Self {
        CodegenResult {
            items: vec![],
            dynamic_items: DynItems::new(),
            saw_bindgen_union: false,
            saw_incomplete_array: false,
            saw_block: false,
            saw_bitfield_unit: false,
            codegen_id,
            items_seen: Default::default(),
            functions_seen: Default::default(),
            vars_seen: Default::default(),
            overload_counters: Default::default(),
            items_to_serialize: Default::default(),
        }
    }
    fn dynamic_items(&mut self) -> &mut DynItems {
        &mut self.dynamic_items
    }
    fn saw_bindgen_union(&mut self) {
        self.saw_bindgen_union = true;
    }
    fn saw_incomplete_array(&mut self) {
        self.saw_incomplete_array = true;
    }
    fn saw_block(&mut self) {
        self.saw_block = true;
    }
    fn saw_bitfield_unit(&mut self) {
        self.saw_bitfield_unit = true;
    }
    fn seen<Id: Into<ItemId>>(&self, item: Id) -> bool {
        self.items_seen.contains(&item.into())
    }
    fn set_seen<Id: Into<ItemId>>(&mut self, item: Id) {
        self.items_seen.insert(item.into());
    }
    fn seen_function(&self, name: &str) -> bool {
        self.functions_seen.contains(name)
    }
    fn saw_function(&mut self, name: &str) {
        self.functions_seen.insert(name.into());
    }
    fn overload_number(&mut self, name: &str) -> u32 {
        let counter = self.overload_counters.entry(name.into()).or_insert(0);
        let number = *counter;
        *counter += 1;
        number
    }
    fn seen_var(&self, name: &str) -> bool {
        self.vars_seen.contains(name)
    }
    fn saw_var(&mut self, name: &str) {
        self.vars_seen.insert(name.into());
    }
    fn inner<F>(&mut self, cb: F) -> Vec<proc_macro2::TokenStream>
    where
        F: FnOnce(&mut Self),
    {
        let mut new = Self::new(self.codegen_id);
        cb(&mut new);
        self.saw_incomplete_array |= new.saw_incomplete_array;
        self.saw_block |= new.saw_block;
        self.saw_bitfield_unit |= new.saw_bitfield_unit;
        self.saw_bindgen_union |= new.saw_bindgen_union;
        new.items
    }
}
impl<'a> ops::Deref for CodegenResult<'a> {
    type Target = Vec<proc_macro2::TokenStream>;
    fn deref(&self) -> &Self::Target {
        &self.items
    }
}
impl<'a> ops::DerefMut for CodegenResult<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.items
    }
}
trait ToPtr {
    fn to_ptr(self, is_const: bool) -> proc_macro2::TokenStream;
}
impl ToPtr for proc_macro2::TokenStream {
    fn to_ptr(self, is_const: bool) -> proc_macro2::TokenStream {
        if is_const {
            quote! { *const #self }
        } else {
            quote! { *mut #self }
        }
    }
}
trait AppendImplicitTemplParams {
    fn append_implicit_templ_params(&mut self, ctx: &Context, it: &Item);
}
impl AppendImplicitTemplParams for proc_macro2::TokenStream {
    fn append_implicit_templ_params(&mut self, ctx: &Context, it: &Item) {
        let it = it.id().into_resolver().through_type_refs().resolve(ctx);
        match *it.expect_type().kind() {
            TypeKind::UnresolvedTypeRef(..) => {
                unreachable!("already resolved unresolved type refs")
            },
            TypeKind::ResolvedTypeRef(..) => {
                unreachable!("we resolved item through type refs")
            },
            TypeKind::Void
            | TypeKind::NullPtr
            | TypeKind::Pointer(..)
            | TypeKind::Reference(..)
            | TypeKind::Int(..)
            | TypeKind::Float(..)
            | TypeKind::Complex(..)
            | TypeKind::Array(..)
            | TypeKind::TypeParam
            | TypeKind::Opaque
            | TypeKind::Function(..)
            | TypeKind::Enum(..)
            | TypeKind::TemplInstantiation(..) => return,
            _ => {},
        }
        let ys: Vec<_> = it
            .used_templ_params(ctx)
            .iter()
            .map(|p| {
                p.try_to_rust(ctx, &())
                    .expect("template params cannot fail to be a rust type")
            })
            .collect();
        if !ys.is_empty() {
            self.append_all(quote! {
                < #( #ys ),* >
            });
        }
    }
}

trait Generator {
    type Extra;
    type Return;
    fn codegen(&self, ctx: &Context, y: &mut CodegenResult<'_>, extra: &Self::Extra) -> Self::Return;
}
impl Item {
    fn process_before_codegen(&self, ctx: &Context, y: &mut CodegenResult) -> bool {
        if !self.is_enabled_for_codegen(ctx) {
            return false;
        }
        if self.is_blocklisted(ctx) || y.seen(self.id()) {
            debug!(
                "<Item as CodeGenerator>::process_before_codegen: Ignoring hidden or seen: \
                 self = {:?}",
                self
            );
            return false;
        }
        if !ctx.codegen_items().contains(&self.id()) {
            warn!("Found non-allowed item in code generation: {:?}", self);
        }
        y.set_seen(self.id());
        true
    }
}
impl Generator for Item {
    type Extra = ();
    type Return = ();
    fn codegen(&self, ctx: &Context, y: &mut CodegenResult<'_>, _extra: &()) {
        debug!("<Item as CodeGenerator>::codegen: self = {:?}", self);
        if !self.process_before_codegen(ctx, y) {
            return;
        }
        match *self.kind() {
            ItemKind::Module(ref x) => {
                x.codegen(ctx, y, self);
            },
            ItemKind::Function(ref x) => {
                x.codegen(ctx, y, self);
            },
            ItemKind::Var(ref x) => {
                x.codegen(ctx, y, self);
            },
            ItemKind::Type(ref x) => {
                x.codegen(ctx, y, self);
            },
        }
    }
}
impl Generator for Module {
    type Extra = Item;
    type Return = ();
    fn codegen(&self, ctx: &Context, y: &mut CodegenResult<'_>, it: &Item) {
        debug!("<Module as CodeGenerator>::codegen: item = {:?}", it);
        let f = |y: &mut CodegenResult, done: &mut bool| {
            for x in self.children() {
                if ctx.codegen_items().contains(x) {
                    *done = true;
                    ctx.resolve_item(*x).codegen(ctx, y, &());
                }
            }
            if it.id() == ctx.root_module() {
                if y.saw_block {
                    utils::prepend_block_header(ctx, &mut *y);
                }
                if y.saw_bindgen_union {
                    utils::prepend_union_types(ctx, &mut *y);
                }
                if y.saw_incomplete_array {
                    utils::prepend_incomplete_array_types(ctx, &mut *y);
                }
                if ctx.need_bindgen_complex_type() {
                    utils::prepend_complex_type(&mut *y);
                }
                if y.saw_bitfield_unit {
                    utils::prepend_bitfield_unit_type(ctx, &mut *y);
                }
            }
        };
        if !ctx.opts().enable_cxx_namespaces || (self.is_inline() && !ctx.opts().conservative_inline_namespaces) {
            f(y, &mut false);
            return;
        }
        let mut done = false;
        let ys = y.inner(|y| {
            y.push(root_import(ctx, it));
            let path = it.namespace_aware_canon_path(ctx).join("::");
            if let Some(xs) = ctx.opts().module_lines.get(&path) {
                for x in xs {
                    done = true;
                    y.push(proc_macro2::TokenStream::from_str(x).unwrap());
                }
            }
            f(y, &mut done);
        });
        if !done {
            return;
        }
        let ident = ctx.rust_ident(it.canon_name(ctx));
        y.push(if it.id() == ctx.root_module() {
            quote! {
                #[allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
                pub mod #ident {
                    #( #ys )*
                }
            }
        } else {
            quote! {
                pub mod #ident {
                    #( #ys )*
                }
            }
        });
    }
}
impl Generator for Var {
    type Extra = Item;
    type Return = ();
    fn codegen(&self, ctx: &Context, y: &mut CodegenResult<'_>, it: &Item) {
        use crate::ir::var::VarType;
        debug!("<Var as CodeGenerator>::codegen: item = {:?}", it);
        debug_assert!(it.is_enabled_for_codegen(ctx));
        let name = it.canon_name(ctx);
        if y.seen_var(&name) {
            return;
        }
        y.saw_var(&name);
        let ident = ctx.rust_ident(&name);
        if !it.all_templ_params(ctx).is_empty() {
            return;
        }
        let mut attrs = vec![];
        if let Some(x) = it.comment(ctx) {
            attrs.push(attributes::doc(x));
        }
        let ty = self.ty().to_rust_or_opaque(ctx, &());
        if let Some(x) = self.val() {
            match *x {
                VarType::Bool(x) => {
                    y.push(quote! {
                        #(#attrs)*
                        pub const #ident : #ty = #x ;
                    });
                },
                VarType::Int(x) => {
                    let kind = self
                        .ty()
                        .into_resolver()
                        .through_type_aliases()
                        .through_type_refs()
                        .resolve(ctx)
                        .expect_type()
                        .as_integer()
                        .unwrap();
                    let x = if kind.is_signed() {
                        utils::ast_ty::int_expr(x)
                    } else {
                        utils::ast_ty::uint_expr(x as _)
                    };
                    y.push(quote! {
                        #(#attrs)*
                        pub const #ident : #ty = #x ;
                    });
                },
                VarType::String(ref x) => {
                    let prefix = ctx.trait_prefix();
                    let opts = ctx.opts();
                    let mut x = x.clone();
                    x.push(0);
                    let len = proc_macro2::Literal::usize_unsuffixed(x.len());
                    let x = CStr::from_bytes_with_nul(&x).unwrap();
                    let x = proc_macro2::Literal::byte_string(x.to_bytes_with_nul());
                    if opts.generate_cstr {
                        let ty = quote! { ::#prefix::ffi::CStr };
                        y.push(quote! {
                            #(#attrs)*
                            #[allow(unsafe_code)]
                            pub const #ident: &#ty = unsafe {
                                #ty::from_bytes_with_nul_unchecked(#x)
                            };
                        });
                    } else {
                        let lifetime = None.into_iter();
                        let ty = quote! { [u8; #len] };
                        y.push(quote! {
                            #(#attrs)*
                            pub const #ident: &#(#lifetime )*#ty = #x ;
                        });
                    }
                },
                VarType::Float(x) => {
                    if let Ok(x) = utils::ast_ty::float_expr(ctx, x) {
                        y.push(quote! {
                            #(#attrs)*
                            pub const #ident : #ty = #x ;
                        });
                    }
                },
                VarType::Char(x) => {
                    y.push(quote! {
                        #(#attrs)*
                        pub const #ident : #ty = #x ;
                    });
                },
            }
        } else {
            if let Some(x) = self.link() {
                attrs.push(attributes::link_name::<false>(x));
            } else {
                let n = self.mangled().unwrap_or_else(|| self.name());
                if !utils::names_will_be_identical_after_mangling(&name, n, None) {
                    attrs.push(attributes::link_name::<false>(n));
                }
            }
            let maybe_mut = if self.is_const() {
                quote! {}
            } else {
                quote! { mut }
            };
            let toks = quote!(
                extern "C" {
                    #(#attrs)*
                    pub static #maybe_mut #ident: #ty;
                }
            );
            y.push(toks);
        }
    }
}
impl Generator for Type {
    type Extra = Item;
    type Return = ();
    fn codegen(&self, ctx: &Context, y: &mut CodegenResult<'_>, it: &Item) {
        debug!("<Type as CodeGenerator>::codegen: item = {:?}", it);
        debug_assert!(it.is_enabled_for_codegen(ctx));
        match *self.kind() {
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
            | TypeKind::TypeParam => {},
            TypeKind::TemplInstantiation(ref x) => x.codegen(ctx, y, it),
            TypeKind::BlockPointer(inner) => {
                if !ctx.opts().generate_block {
                    return;
                }
                let inner_item = inner.into_resolver().through_type_refs().resolve(ctx);
                let name = it.canon_name(ctx);
                let inner_rust_type = {
                    if let TypeKind::Function(fnsig) = inner_item.kind().expect_type().kind() {
                        utils::fnsig_block(ctx, fnsig)
                    } else {
                        panic!("invalid block typedef: {:?}", inner_item)
                    }
                };
                let rust_name = ctx.rust_ident(name);
                let mut tokens = if let Some(comment) = it.comment(ctx) {
                    attributes::doc(comment)
                } else {
                    quote! {}
                };
                tokens.append_all(quote! {
                    pub type #rust_name = #inner_rust_type ;
                });
                y.push(tokens);
                y.saw_block();
            },
            TypeKind::Comp(ref x) => x.codegen(ctx, y, it),
            TypeKind::TemplAlias(inner, _) | TypeKind::Alias(inner) => {
                let inner_item = inner.into_resolver().through_type_refs().resolve(ctx);
                let name = it.canon_name(ctx);
                let path = it.canon_path(ctx);
                {
                    let through_type_aliases = inner
                        .into_resolver()
                        .through_type_refs()
                        .through_type_aliases()
                        .resolve(ctx);
                    if through_type_aliases.canon_path(ctx) == path {
                        return;
                    }
                }
                let spelling = self.name().expect("Unnamed alias?");
                if utils::type_from_named(ctx, spelling).is_some() {
                    if let "size_t" | "ssize_t" = spelling {
                        let layout = inner_item.kind().expect_type().layout(ctx).expect("No layout?");
                        assert_eq!(
                            layout.size,
                            ctx.target_pointer_size(),
                            "Target platform requires `--no-size_t-is-usize`. The size of `{}` ({}) does not match the target pointer size ({})",
                            spelling,
                            layout.size,
                            ctx.target_pointer_size(),
                            );
                        assert_eq!(
                            layout.align,
                            ctx.target_pointer_size(),
                            "Target platform requires `--no-size_t-is-usize`. The alignment of `{}` ({}) does not match the target pointer size ({})",
                            spelling,
                            layout.align,
                            ctx.target_pointer_size(),
                        );
                    }
                    return;
                }
                let mut outer_params = it.used_templ_params(ctx);
                let is_opaque = it.is_opaque(ctx, &());
                let inner_rust_type = if is_opaque {
                    outer_params = vec![];
                    self.to_opaque(ctx, it)
                } else {
                    let mut inner_ty = inner_item
                        .try_to_rust_or_opaque(ctx, &())
                        .unwrap_or_else(|_| self.to_opaque(ctx, it));
                    inner_ty.append_implicit_templ_params(ctx, inner_item);
                    inner_ty
                };
                {
                    let inner_canon_type = inner_item.expect_type().canonical_type(ctx);
                    if inner_canon_type.is_invalid_type_param() {
                        warn!(
                            "Item contained invalid named type, skipping: \
                             {:?}, {:?}",
                            it, inner_item
                        );
                        return;
                    }
                }
                let rust_name = ctx.rust_ident(&name);
                let mut tokens = if let Some(comment) = it.comment(ctx) {
                    attributes::doc(comment)
                } else {
                    quote! {}
                };
                let alias_style = if ctx.opts().type_alias.matches(&name) {
                    variation::Alias::TypeAlias
                } else if ctx.opts().new_type_alias.matches(&name) {
                    variation::Alias::NewType
                } else if ctx.opts().new_type_alias_deref.matches(&name) {
                    variation::Alias::NewTypeDeref
                } else {
                    ctx.opts().default_alias_style
                };
                if inner_rust_type
                    .to_string()
                    .chars()
                    .all(|c| matches!(c, 'A'..='Z' | 'a'..='z' | '0'..='9' | ':' | '_' | ' '))
                    && outer_params.is_empty()
                    && !is_opaque
                    && alias_style == variation::Alias::TypeAlias
                    && inner_item.expect_type().canonical_type(ctx).is_enum()
                {
                    tokens.append_all(quote! {
                        pub use
                    });
                    let path = top_level_path(ctx, it);
                    tokens.append_separated(path, quote!(::));
                    tokens.append_all(quote! {
                        :: #inner_rust_type  as #rust_name ;
                    });
                    y.push(tokens);
                    return;
                }
                tokens.append_all(match alias_style {
                    variation::Alias::TypeAlias => quote! {
                        pub type #rust_name
                    },
                    variation::Alias::NewType | variation::Alias::NewTypeDeref => {
                        let mut attributes = vec![attributes::repr("transparent")];
                        let packed = false; // Types can't be packed in Rust.
                        let derivable_traits = derives_of_item(it, ctx, packed);
                        if !derivable_traits.is_empty() {
                            let derives: Vec<_> = derivable_traits.into();
                            attributes.push(attributes::derives(&derives))
                        }
                        quote! {
                            #( #attributes )*
                            pub struct #rust_name
                        }
                    },
                });
                let params: Vec<_> = outer_params
                    .into_iter()
                    .filter_map(|p| p.as_templ_param(ctx, &()))
                    .collect();
                if params.iter().any(|p| ctx.resolve_type(*p).is_invalid_type_param()) {
                    warn!(
                        "Item contained invalid template \
                         parameter: {:?}",
                        it
                    );
                    return;
                }
                let params: Vec<_> = params
                    .iter()
                    .map(|p| {
                        p.try_to_rust(ctx, &())
                            .expect("type parameters can always convert to rust ty OK")
                    })
                    .collect();
                if !params.is_empty() {
                    tokens.append_all(quote! {
                        < #( #params ),* >
                    });
                }
                let access_spec = access_specifier(ctx.opts().default_visibility);
                tokens.append_all(match alias_style {
                    variation::Alias::TypeAlias => quote! {
                        = #inner_rust_type ;
                    },
                    variation::Alias::NewType | variation::Alias::NewTypeDeref => {
                        quote! {
                            (#access_spec #inner_rust_type) ;
                        }
                    },
                });
                if alias_style == variation::Alias::NewTypeDeref {
                    let prefix = ctx.trait_prefix();
                    tokens.append_all(quote! {
                        impl ::#prefix::ops::Deref for #rust_name {
                            type Target = #inner_rust_type;
                            #[inline]
                            fn deref(&self) -> &Self::Target {
                                &self.0
                            }
                        }
                        impl ::#prefix::ops::DerefMut for #rust_name {
                            #[inline]
                            fn deref_mut(&mut self) -> &mut Self::Target {
                                &mut self.0
                            }
                        }
                    });
                }
                y.push(tokens);
            },
            TypeKind::Enum(ref x) => x.codegen(ctx, y, it),
            ref u @ TypeKind::UnresolvedTypeRef(..) => {
                unreachable!("Should have been resolved after parsing {:?}!", u)
            },
        }
    }
}

struct Vtable<'a> {
    id: ItemId,
    #[allow(dead_code)]
    info: &'a CompInfo,
}
impl<'a> Vtable<'a> {
    fn new(id: ItemId, info: &'a CompInfo) -> Self {
        Vtable { id, info }
    }
}
impl<'a> Generator for Vtable<'a> {
    type Extra = Item;
    type Return = ();
    fn codegen(&self, ctx: &Context, y: &mut CodegenResult<'_>, it: &Item) {
        assert_eq!(it.id(), self.id);
        debug_assert!(it.is_enabled_for_codegen(ctx));
        let name = ctx.rust_ident(self.canon_name(ctx));
        if ctx.opts().vtable_generation && self.info.base_members().is_empty() && self.info.destructor().is_none() {
            let class_ident = ctx.rust_ident(self.id.canon_name(ctx));
            let methods = self
                .info
                .methods()
                .iter()
                .filter_map(|m| {
                    if !m.is_virtual() {
                        return None;
                    }
                    let function_item = ctx.resolve_item(m.sig());
                    let function = function_item.expect_function();
                    let signature_item = ctx.resolve_item(function.sig());
                    let signature = match signature_item.expect_type().kind() {
                        TypeKind::Function(ref sig) => sig,
                        _ => panic!("Function signature type mismatch"),
                    };
                    let function_name = function_item.canon_name(ctx);
                    let function_name = ctx.rust_ident(function_name);
                    let mut args = utils::fnsig_arguments(ctx, signature);
                    let ret = utils::fnsig_return_ty(ctx, signature);
                    args[0] = if m.is_const() {
                        quote! { this: *const #class_ident }
                    } else {
                        quote! { this: *mut #class_ident }
                    };
                    Some(quote! {
                        pub #function_name : unsafe extern "C" fn( #( #args ),* ) #ret
                    })
                })
                .collect::<Vec<_>>();
            y.push(quote! {
                #[repr(C)]
                pub struct #name {
                    #( #methods ),*
                }
            })
        } else {
            let void = utils::ast_ty::c_void(ctx);
            y.push(quote! {
                #[repr(C)]
                pub struct #name ( #void );
            });
        }
    }
}
impl<'a> CanonName for Vtable<'a> {
    fn canon_name(&self, ctx: &Context) -> String {
        format!("{}__bindgen_vtable", self.id.canon_name(ctx))
    }
}

impl Generator for TemplInstantiation {
    type Extra = Item;
    type Return = ();
    fn codegen(&self, ctx: &Context, y: &mut CodegenResult<'_>, it: &Item) {
        debug_assert!(it.is_enabled_for_codegen(ctx));
        if !ctx.opts().layout_tests || self.is_opaque(ctx, it) {
            return;
        }
        if ctx.uses_any_templ_params(it.id()) {
            return;
        }
        let layout = it.kind().expect_type().layout(ctx);
        if let Some(x) = layout {
            let size = x.size;
            let align = x.align;
            let name = it.full_disambiguated_name(ctx);
            let mut fn_name = format!("__bindgen_test_layout_{}_instantiation", name);
            let times_seen = y.overload_number(&fn_name);
            if times_seen > 0 {
                write!(&mut fn_name, "_{}", times_seen).unwrap();
            }
            let fn_name = ctx.rust_ident_raw(fn_name);
            let prefix = ctx.trait_prefix();
            let ident = it.to_rust_or_opaque(ctx, &());
            let size_of_expr = quote! {
                ::#prefix::mem::size_of::<#ident>()
            };
            let align_of_expr = quote! {
                ::#prefix::mem::align_of::<#ident>()
            };
            let item = quote! {
                #[test]
                fn #fn_name() {
                    assert_eq!(#size_of_expr, #size,
                               concat!("Size of template specialization: ",
                                       stringify!(#ident)));
                    assert_eq!(#align_of_expr, #align,
                               concat!("Alignment of template specialization: ",
                                       stringify!(#ident)));
                }
            };
            y.push(item);
        }
    }
}

trait FieldCodegen<'a> {
    type Extra;
    #[allow(clippy::too_many_arguments)]
    fn codegen<F, M>(
        &self,
        ctx: &Context,
        visibility_kind: FieldVisibilityKind,
        accessor_kind: FieldAccessorKind,
        parent: &CompInfo,
        result: &mut CodegenResult,
        struct_layout: &mut StructLayoutTracker,
        fields: &mut F,
        methods: &mut M,
        extra: Self::Extra,
    ) where
        F: Extend<proc_macro2::TokenStream>,
        M: Extend<proc_macro2::TokenStream>;
}
impl<'a> FieldCodegen<'a> for Field {
    type Extra = ();
    fn codegen<F, M>(
        &self,
        ctx: &Context,
        visibility_kind: FieldVisibilityKind,
        accessor_kind: FieldAccessorKind,
        parent: &CompInfo,
        result: &mut CodegenResult,
        struct_layout: &mut StructLayoutTracker,
        fields: &mut F,
        methods: &mut M,
        _: (),
    ) where
        F: Extend<proc_macro2::TokenStream>,
        M: Extend<proc_macro2::TokenStream>,
    {
        match *self {
            Field::DataMember(ref data) => {
                data.codegen(
                    ctx,
                    visibility_kind,
                    accessor_kind,
                    parent,
                    result,
                    struct_layout,
                    fields,
                    methods,
                    (),
                );
            },
            Field::Bitfields(ref unit) => {
                unit.codegen(
                    ctx,
                    visibility_kind,
                    accessor_kind,
                    parent,
                    result,
                    struct_layout,
                    fields,
                    methods,
                    (),
                );
            },
        }
    }
}
fn wrap_union_field_if_needed(
    ctx: &Context,
    struct_layout: &StructLayoutTracker,
    ty: proc_macro2::TokenStream,
    result: &mut CodegenResult,
) -> proc_macro2::TokenStream {
    if struct_layout.is_rust_union() {
        if struct_layout.can_copy_union_fields() {
            ty
        } else {
            let prefix = ctx.trait_prefix();
            quote! {
                ::#prefix::mem::ManuallyDrop<#ty>
            }
        }
    } else {
        result.saw_bindgen_union();
        if ctx.opts().enable_cxx_namespaces {
            quote! {
                root::__BindgenUnionField<#ty>
            }
        } else {
            quote! {
                __BindgenUnionField<#ty>
            }
        }
    }
}
impl<'a> FieldCodegen<'a> for FieldData {
    type Extra = ();
    fn codegen<F, M>(
        &self,
        ctx: &Context,
        parent_visibility_kind: FieldVisibilityKind,
        accessor_kind: FieldAccessorKind,
        parent: &CompInfo,
        result: &mut CodegenResult,
        struct_layout: &mut StructLayoutTracker,
        fields: &mut F,
        methods: &mut M,
        _: (),
    ) where
        F: Extend<proc_macro2::TokenStream>,
        M: Extend<proc_macro2::TokenStream>,
    {
        assert!(self.bitfield_width().is_none());
        let field_item = self.ty().into_resolver().through_type_refs().resolve(ctx);
        let field_ty = field_item.expect_type();
        let mut ty = self.ty().to_rust_or_opaque(ctx, &());
        ty.append_implicit_templ_params(ctx, field_item);
        let ty = if parent.is_union() {
            wrap_union_field_if_needed(ctx, struct_layout, ty, result)
        } else if let Some(item) = field_ty.is_incomplete_array(ctx) {
            result.saw_incomplete_array();
            let inner = item.to_rust_or_opaque(ctx, &());
            if ctx.opts().enable_cxx_namespaces {
                quote! {
                    root::__IncompleteArrayField<#inner>
                }
            } else {
                quote! {
                    __IncompleteArrayField<#inner>
                }
            }
        } else {
            ty
        };
        let mut field = quote! {};
        if ctx.opts().generate_comments {
            if let Some(raw_comment) = self.comment() {
                let comment = ctx.opts().process_comment(raw_comment);
                field = attributes::doc(comment);
            }
        }
        let field_name = self
            .name()
            .map(|name| ctx.rust_mangle(name).into_owned())
            .expect("Each field should have a name in codegen!");
        let field_ident = ctx.rust_ident_raw(field_name.as_str());
        if let Some(padding_field) = struct_layout.saw_field(&field_name, field_ty, self.offset()) {
            fields.extend(Some(padding_field));
        }
        let visibility = compute_visibility(ctx, self.is_public(), Some(self.annotations()), parent_visibility_kind);
        let accessor_kind = self.annotations().accessor_kind().unwrap_or(accessor_kind);
        match visibility {
            FieldVisibilityKind::Private => {
                field.append_all(quote! {
                    #field_ident : #ty ,
                });
            },
            FieldVisibilityKind::PublicCrate => {
                field.append_all(quote! {
                    pub #field_ident : #ty ,
                });
            },
            FieldVisibilityKind::Public => {
                field.append_all(quote! {
                    pub #field_ident : #ty ,
                });
            },
        }
        fields.extend(Some(field));
        if accessor_kind == FieldAccessorKind::None {
            return;
        }
        let getter_name = ctx.rust_ident_raw(format!("get_{}", field_name));
        let mutable_getter_name = ctx.rust_ident_raw(format!("get_{}_mut", field_name));
        let field_name = ctx.rust_ident_raw(field_name);
        methods.extend(Some(match accessor_kind {
            FieldAccessorKind::None => unreachable!(),
            FieldAccessorKind::Regular => {
                quote! {
                    #[inline]
                    pub fn #getter_name(&self) -> & #ty {
                        &self.#field_name
                    }
                    #[inline]
                    pub fn #mutable_getter_name(&mut self) -> &mut #ty {
                        &mut self.#field_name
                    }
                }
            },
            FieldAccessorKind::Unsafe => {
                quote! {
                    #[inline]
                    pub unsafe fn #getter_name(&self) -> & #ty {
                        &self.#field_name
                    }
                    #[inline]
                    pub unsafe fn #mutable_getter_name(&mut self) -> &mut #ty {
                        &mut self.#field_name
                    }
                }
            },
            FieldAccessorKind::Immutable => {
                quote! {
                    #[inline]
                    pub fn #getter_name(&self) -> & #ty {
                        &self.#field_name
                    }
                }
            },
        }));
    }
}
impl BitfieldUnit {
    fn ctor_name(&self) -> proc_macro2::TokenStream {
        let ctor_name = Ident::new(&format!("new_bitfield_{}", self.nth()), Span::call_site());
        quote! {
            #ctor_name
        }
    }
}
impl Bitfield {
    fn extend_ctor_impl(
        &self,
        ctx: &Context,
        param_name: proc_macro2::TokenStream,
        mut ctor_impl: proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        let bitfield_ty = ctx.resolve_type(self.ty());
        let bitfield_ty_layout = bitfield_ty.layout(ctx).expect("Bitfield without layout? Gah!");
        let bitfield_int_ty = utils::integer_type(ctx, bitfield_ty_layout).expect(
            "Should already have verified that the bitfield is \
                 representable as an int",
        );
        let offset = self.offset_into_unit();
        let width = self.width() as u8;
        let prefix = ctx.trait_prefix();
        ctor_impl.append_all(quote! {
            __bindgen_bitfield_unit.set(
                #offset,
                #width,
                {
                    let #param_name: #bitfield_int_ty = unsafe {
                        ::#prefix::mem::transmute(#param_name)
                    };
                    #param_name as u64
                }
            );
        });
        ctor_impl
    }
}
fn access_specifier(visibility: FieldVisibilityKind) -> proc_macro2::TokenStream {
    match visibility {
        FieldVisibilityKind::Private => quote! {},
        FieldVisibilityKind::PublicCrate => quote! { pub },
        FieldVisibilityKind::Public => quote! { pub },
    }
}
fn compute_visibility(
    ctx: &Context,
    is_declared_public: bool,
    annotations: Option<&Annotations>,
    default_kind: FieldVisibilityKind,
) -> FieldVisibilityKind {
    match (
        is_declared_public,
        ctx.opts().respect_cxx_access_specs,
        annotations.and_then(|e| e.visibility_kind()),
    ) {
        (true, true, annotated_visibility) => annotated_visibility.unwrap_or(FieldVisibilityKind::Public),
        (false, true, annotated_visibility) => annotated_visibility.unwrap_or(FieldVisibilityKind::Private),
        (_, false, annotated_visibility) => annotated_visibility.unwrap_or(default_kind),
    }
}
impl<'a> FieldCodegen<'a> for BitfieldUnit {
    type Extra = ();
    fn codegen<F, M>(
        &self,
        ctx: &Context,
        visibility_kind: FieldVisibilityKind,
        accessor_kind: FieldAccessorKind,
        parent: &CompInfo,
        result: &mut CodegenResult,
        struct_layout: &mut StructLayoutTracker,
        fields: &mut F,
        methods: &mut M,
        _: (),
    ) where
        F: Extend<proc_macro2::TokenStream>,
        M: Extend<proc_macro2::TokenStream>,
    {
        use crate::ir::RUST_DERIVE_IN_ARRAY_LIMIT;
        result.saw_bitfield_unit();
        let layout = self.layout();
        let unit_field_ty = utils::bitfield_unit(ctx, layout);
        let field_ty = {
            let unit_field_ty = unit_field_ty.clone();
            if parent.is_union() {
                wrap_union_field_if_needed(ctx, struct_layout, unit_field_ty, result)
            } else {
                unit_field_ty
            }
        };
        {
            let align_field_name = format!("_bitfield_align_{}", self.nth());
            let align_field_ident = ctx.rust_ident(align_field_name);
            let align_ty = match self.layout().align {
                n if n >= 8 => quote! { u64 },
                4 => quote! { u32 },
                2 => quote! { u16 },
                _ => quote! { u8  },
            };
            let access_spec = access_specifier(visibility_kind);
            let align_field = quote! {
                #access_spec #align_field_ident: [#align_ty; 0],
            };
            fields.extend(Some(align_field));
        }
        let unit_field_name = format!("_bitfield_{}", self.nth());
        let unit_field_ident = ctx.rust_ident(&unit_field_name);
        let ctor_name = self.ctor_name();
        let mut ctor_params = vec![];
        let mut ctor_impl = quote! {};
        let mut generate_ctor = layout.size <= RUST_DERIVE_IN_ARRAY_LIMIT;
        let mut all_fields_declared_as_public = true;
        for bf in self.bitfields() {
            if bf.name().is_none() {
                continue;
            }
            all_fields_declared_as_public &= bf.is_public();
            let mut bitfield_representable_as_int = true;
            bf.codegen(
                ctx,
                visibility_kind,
                accessor_kind,
                parent,
                result,
                struct_layout,
                fields,
                methods,
                (&unit_field_name, &mut bitfield_representable_as_int),
            );
            if !bitfield_representable_as_int {
                generate_ctor = false;
                continue;
            }
            let param_name = bitfield_getter_name(ctx, bf);
            let bitfield_ty_item = ctx.resolve_item(bf.ty());
            let bitfield_ty = bitfield_ty_item.expect_type();
            let bitfield_ty = bitfield_ty.to_rust_or_opaque(ctx, bitfield_ty_item);
            ctor_params.push(quote! {
                #param_name : #bitfield_ty
            });
            ctor_impl = bf.extend_ctor_impl(ctx, param_name, ctor_impl);
        }
        let visibility_kind = compute_visibility(ctx, all_fields_declared_as_public, None, visibility_kind);
        let access_spec = access_specifier(visibility_kind);
        let field = quote! {
            #access_spec #unit_field_ident : #field_ty ,
        };
        fields.extend(Some(field));
        if generate_ctor {
            methods.extend(Some(quote! {
                #[inline]
                #access_spec fn #ctor_name ( #( #ctor_params ),* ) -> #unit_field_ty {
                    let mut __bindgen_bitfield_unit: #unit_field_ty = Default::default();
                    #ctor_impl
                    __bindgen_bitfield_unit
                }
            }));
        }
        struct_layout.saw_bitfield_unit(layout);
    }
}
fn bitfield_getter_name(ctx: &Context, bitfield: &Bitfield) -> proc_macro2::TokenStream {
    let name = bitfield.getter();
    let name = ctx.rust_ident_raw(name);
    quote! { #name }
}
fn bitfield_setter_name(ctx: &Context, bitfield: &Bitfield) -> proc_macro2::TokenStream {
    let setter = bitfield.setter();
    let setter = ctx.rust_ident_raw(setter);
    quote! { #setter }
}
impl<'a> FieldCodegen<'a> for Bitfield {
    type Extra = (&'a str, &'a mut bool);
    fn codegen<F, M>(
        &self,
        ctx: &Context,
        visibility_kind: FieldVisibilityKind,
        _accessor_kind: FieldAccessorKind,
        parent: &CompInfo,
        _result: &mut CodegenResult,
        struct_layout: &mut StructLayoutTracker,
        _fields: &mut F,
        methods: &mut M,
        (unit_field_name, bitfield_representable_as_int): (&'a str, &mut bool),
    ) where
        F: Extend<proc_macro2::TokenStream>,
        M: Extend<proc_macro2::TokenStream>,
    {
        let prefix = ctx.trait_prefix();
        let getter_name = bitfield_getter_name(ctx, self);
        let setter_name = bitfield_setter_name(ctx, self);
        let unit_field_ident = Ident::new(unit_field_name, Span::call_site());
        let bitfield_ty_item = ctx.resolve_item(self.ty());
        let bitfield_ty = bitfield_ty_item.expect_type();
        let bitfield_ty_layout = bitfield_ty.layout(ctx).expect("Bitfield without layout? Gah!");
        let bitfield_int_ty = match utils::integer_type(ctx, bitfield_ty_layout) {
            Some(int_ty) => {
                *bitfield_representable_as_int = true;
                int_ty
            },
            None => {
                *bitfield_representable_as_int = false;
                return;
            },
        };
        let bitfield_ty = bitfield_ty.to_rust_or_opaque(ctx, bitfield_ty_item);
        let offset = self.offset_into_unit();
        let width = self.width() as u8;
        let visibility_kind = compute_visibility(ctx, self.is_public(), Some(self.annotations()), visibility_kind);
        let access_spec = access_specifier(visibility_kind);
        if parent.is_union() && !struct_layout.is_rust_union() {
            methods.extend(Some(quote! {
                #[inline]
                #access_spec fn #getter_name(&self) -> #bitfield_ty {
                    unsafe {
                        ::#prefix::mem::transmute(
                            self.#unit_field_ident.as_ref().get(#offset, #width)
                                as #bitfield_int_ty
                        )
                    }
                }
                #[inline]
                #access_spec fn #setter_name(&mut self, val: #bitfield_ty) {
                    unsafe {
                        let val: #bitfield_int_ty = ::#prefix::mem::transmute(val);
                        self.#unit_field_ident.as_mut().set(
                            #offset,
                            #width,
                            val as u64
                        )
                    }
                }
            }));
        } else {
            methods.extend(Some(quote! {
                #[inline]
                #access_spec fn #getter_name(&self) -> #bitfield_ty {
                    unsafe {
                        ::#prefix::mem::transmute(
                            self.#unit_field_ident.get(#offset, #width)
                                as #bitfield_int_ty
                        )
                    }
                }
                #[inline]
                #access_spec fn #setter_name(&mut self, val: #bitfield_ty) {
                    unsafe {
                        let val: #bitfield_int_ty = ::#prefix::mem::transmute(val);
                        self.#unit_field_ident.set(
                            #offset,
                            #width,
                            val as u64
                        )
                    }
                }
            }));
        }
    }
}
impl Generator for CompInfo {
    type Extra = Item;
    type Return = ();
    fn codegen(&self, ctx: &Context, y: &mut CodegenResult<'_>, it: &Item) {
        debug!("<CompInfo as CodeGenerator>::codegen: item = {:?}", it);
        debug_assert!(it.is_enabled_for_codegen(ctx));
        if self.has_non_type_templ_params() {
            return;
        }
        let ty = it.expect_type();
        let layout = ty.layout(ctx);
        let mut packed = self.is_packed(ctx, layout.as_ref());
        let canonical_name = it.canon_name(ctx);
        let canonical_ident = ctx.rust_ident(&canonical_name);
        let is_opaque = it.is_opaque(ctx, &());
        let mut fields = vec![];
        let mut struct_layout = StructLayoutTracker::new(ctx, self, ty, &canonical_name);
        if !is_opaque {
            if it.has_vtable_ptr(ctx) {
                let vtable = Vtable::new(it.id(), self);
                vtable.codegen(ctx, y, it);
                let vtable_type = vtable
                    .try_to_rust(ctx, &())
                    .expect("vtable to Rust type conversion is infallible")
                    .to_ptr(true);
                fields.push(quote! {
                    pub vtable_: #vtable_type ,
                });
                struct_layout.saw_vtable();
            }
            for base in self.base_members() {
                if !base.requires_storage(ctx) {
                    continue;
                }
                let inner_item = ctx.resolve_item(base.ty);
                let mut inner = inner_item.to_rust_or_opaque(ctx, &());
                inner.append_implicit_templ_params(ctx, inner_item);
                let field_name = ctx.rust_ident(&base.field_name);
                struct_layout.saw_base(inner_item.expect_type());
                let visibility = match (base.is_public(), ctx.opts().respect_cxx_access_specs) {
                    (true, true) => FieldVisibilityKind::Public,
                    (false, true) => FieldVisibilityKind::Private,
                    _ => ctx.opts().default_visibility,
                };
                let access_spec = access_specifier(visibility);
                fields.push(quote! {
                    #access_spec #field_name: #inner,
                });
            }
        }
        let mut methods = vec![];
        if !is_opaque {
            let visibility = it
                .annotations()
                .visibility_kind()
                .unwrap_or(ctx.opts().default_visibility);
            let struct_accessor_kind = it.annotations().accessor_kind().unwrap_or(FieldAccessorKind::None);
            for field in self.fields() {
                field.codegen(
                    ctx,
                    visibility,
                    struct_accessor_kind,
                    self,
                    y,
                    &mut struct_layout,
                    &mut fields,
                    &mut methods,
                    (),
                );
            }
            if let Some(comp_layout) = layout {
                fields.extend(struct_layout.add_tail_padding(&canonical_name, comp_layout));
            }
        }
        if is_opaque {
            debug_assert!(fields.is_empty());
            debug_assert!(methods.is_empty());
        }
        let is_union = self.kind() == CompKind::Union;
        let layout = it.kind().expect_type().layout(ctx);
        let zero_sized = it.is_zero_sized(ctx);
        let forward_decl = self.is_forward_declaration();
        let mut explicit_align = None;
        if !forward_decl && zero_sized {
            let has_address = if is_opaque {
                layout.is_none()
            } else {
                layout.map_or(true, |l| l.size != 0)
            };
            if has_address {
                let layout = Layout::new(1, 1);
                let ty = utils::blob(ctx, Layout::new(1, 1));
                struct_layout.saw_field_with_layout("_address", layout, /* offset = */ Some(0));
                fields.push(quote! {
                    pub _address: #ty,
                });
            }
        }
        if is_opaque {
            match layout {
                Some(l) => {
                    explicit_align = Some(l.align);
                    let ty = utils::blob(ctx, l);
                    fields.push(quote! {
                        pub _bindgen_opaque_blob: #ty ,
                    });
                },
                None => {
                    warn!("Opaque type without layout! Expect dragons!");
                },
            }
        } else if !is_union && !zero_sized {
            if let Some(padding_field) = layout.and_then(|layout| struct_layout.pad_struct(layout)) {
                fields.push(padding_field);
            }
            if let Some(layout) = layout {
                if struct_layout.requires_explicit_align(layout) {
                    if layout.align == 1 {
                        packed = true;
                    } else {
                        explicit_align = Some(layout.align);
                    }
                }
            }
        } else if is_union && !forward_decl {
            let layout = layout.expect("Unable to get layout information?");
            if struct_layout.requires_explicit_align(layout) {
                explicit_align = Some(layout.align);
            }
            if !struct_layout.is_rust_union() {
                let ty = utils::blob(ctx, layout);
                fields.push(quote! {
                    pub bindgen_union_field: #ty ,
                })
            }
        }
        if forward_decl {
            fields.push(quote! {
                _unused: [u8; 0],
            });
        }
        let mut generic_param_names = vec![];
        for (idx, ty) in it.used_templ_params(ctx).iter().enumerate() {
            let param = ctx.resolve_type(*ty);
            let name = param.name().unwrap();
            let ident = ctx.rust_ident(name);
            generic_param_names.push(ident.clone());
            let prefix = ctx.trait_prefix();
            let field_name = ctx.rust_ident(format!("_phantom_{}", idx));
            fields.push(quote! {
                pub #field_name : ::#prefix::marker::PhantomData<
                    ::#prefix::cell::UnsafeCell<#ident>
                > ,
            });
        }
        let generics = if !generic_param_names.is_empty() {
            let generic_param_names = generic_param_names.clone();
            quote! {
                < #( #generic_param_names ),* >
            }
        } else {
            quote! {}
        };
        let mut attributes = vec![];
        let mut needs_clone_impl = false;
        let mut needs_default_impl = false;
        let mut needs_debug_impl = false;
        let mut needs_partialeq_impl = false;
        if let Some(comment) = it.comment(ctx) {
            attributes.push(attributes::doc(comment));
        }
        if packed && !is_opaque {
            let n = layout.map_or(1, |l| l.align);
            let packed_repr = if n == 1 {
                "packed".to_string()
            } else {
                format!("packed({})", n)
            };
            attributes.push(attributes::repr_list(&["C", &packed_repr]));
        } else {
            attributes.push(attributes::repr("C"));
        }
        if let Some(explicit) = explicit_align {
            let explicit = utils::ast_ty::int_expr(explicit as i64);
            attributes.push(quote! {
                #[repr(align(#explicit))]
            });
        }
        let derivable_traits = derives_of_item(it, ctx, packed);
        if !derivable_traits.contains(DerivableTraits::DEBUG) {
            needs_debug_impl = ctx.opts().derive_debug
                && ctx.opts().impl_debug
                && !ctx.no_debug_by_name(it)
                && !it.annotations().disallow_debug();
        }
        if !derivable_traits.contains(DerivableTraits::DEFAULT) {
            needs_default_impl = ctx.opts().derive_default
                && !self.is_forward_declaration()
                && !ctx.no_default_by_name(it)
                && !it.annotations().disallow_default();
        }
        let all_templ_params = it.all_templ_params(ctx);
        if derivable_traits.contains(DerivableTraits::COPY) && !derivable_traits.contains(DerivableTraits::CLONE) {
            needs_clone_impl = true;
        }
        if !derivable_traits.contains(DerivableTraits::PARTIAL_EQ) {
            needs_partialeq_impl = ctx.opts().derive_partialeq
                && ctx.opts().impl_partialeq
                && ctx.lookup_can_derive_partialeq_or_partialord(it.id()) == Resolved::Manually;
        }
        let mut derives: Vec<_> = derivable_traits.into();
        derives.extend(it.annotations().derives().iter().map(String::as_str));
        let is_rust_union = is_union && struct_layout.is_rust_union();
        let custom_derives = ctx.opts().all_callbacks(|cb| {
            cb.add_derives(&DeriveInfo {
                name: &canonical_name,
                kind: if is_rust_union {
                    DeriveTypeKind::Union
                } else {
                    DeriveTypeKind::Struct
                },
            })
        });
        derives.extend(custom_derives.iter().map(|s| s.as_str()));
        if !derives.is_empty() {
            attributes.push(attributes::derives(&derives))
        }
        if it.must_use(ctx) {
            attributes.push(attributes::must_use());
        }
        let mut tokens = if is_rust_union {
            quote! {
                #( #attributes )*
                pub union #canonical_ident
            }
        } else {
            quote! {
                #( #attributes )*
                pub struct #canonical_ident
            }
        };
        tokens.append_all(quote! {
            #generics {
                #( #fields )*
            }
        });
        y.push(tokens);
        for ty in self.inner_types() {
            let child_item = ctx.resolve_item(*ty);
            child_item.codegen(ctx, y, &());
        }
        if self.found_unknown_attr() {
            warn!(
                "Type {} has an unknown attribute that may affect layout",
                canonical_ident
            );
        }
        if all_templ_params.is_empty() {
            if !is_opaque {
                for var in self.inner_vars() {
                    ctx.resolve_item(*var).codegen(ctx, y, &());
                }
            }
            if ctx.opts().layout_tests && !self.is_forward_declaration() {
                if let Some(layout) = layout {
                    let fn_name = format!("bindgen_test_layout_{}", canonical_ident);
                    let fn_name = ctx.rust_ident_raw(fn_name);
                    let prefix = ctx.trait_prefix();
                    let size_of_expr = quote! {
                        ::#prefix::mem::size_of::<#canonical_ident>()
                    };
                    let align_of_expr = quote! {
                        ::#prefix::mem::align_of::<#canonical_ident>()
                    };
                    let size = layout.size;
                    let align = layout.align;
                    let check_struct_align = Some(quote! {
                        assert_eq!(#align_of_expr,
                               #align,
                               concat!("Alignment of ", stringify!(#canonical_ident)));
                    });
                    let should_skip_field_offset_checks = is_opaque;
                    let check_field_offset = if should_skip_field_offset_checks {
                        vec![]
                    } else {
                        self.fields()
                            .iter()
                            .filter_map(|field| match *field {
                                Field::DataMember(ref f) if f.name().is_some() => Some(f),
                                _ => None,
                            })
                            .flat_map(|field| {
                                let name = field.name().unwrap();
                                field.offset().map(|offset| {
                                    let field_offset = offset / 8;
                                    let field_name = ctx.rust_ident(name);
                                    quote! {
                                        assert_eq!(
                                            unsafe {
                                                ::#prefix::ptr::addr_of!((*ptr).#field_name) as usize - ptr as usize
                                            },
                                            #field_offset,
                                            concat!("Offset of field: ", stringify!(#canonical_ident), "::", stringify!(#field_name))
                                        );
                                    }
                                })
                            })
                            .collect()
                    };
                    let uninit_decl = if !check_field_offset.is_empty() {
                        Some(quote! {
                            const UNINIT: ::#prefix::mem::MaybeUninit<#canonical_ident> = ::#prefix::mem::MaybeUninit::uninit();
                            let ptr = UNINIT.as_ptr();
                        })
                    } else {
                        None
                    };
                    let item = quote! {
                        #[test]
                        fn #fn_name() {
                            #uninit_decl
                            assert_eq!(#size_of_expr,
                                       #size,
                                       concat!("Size of: ", stringify!(#canonical_ident)));
                            #check_struct_align
                            #( #check_field_offset )*
                        }
                    };
                    y.push(item);
                }
            }
            let mut method_names = Default::default();
            if ctx.opts().codegen_config.methods() {
                for method in self.methods() {
                    assert!(method.kind() != MethodKind::Constructor);
                    method.codegen_method(ctx, &mut methods, &mut method_names, y, self);
                }
            }
            if ctx.opts().codegen_config.constructors() {
                for sig in self.constructors() {
                    Method::new(MethodKind::Constructor, *sig, /* const */ false).codegen_method(
                        ctx,
                        &mut methods,
                        &mut method_names,
                        y,
                        self,
                    );
                }
            }
            if ctx.opts().codegen_config.destructors() {
                if let Some((kind, destructor)) = self.destructor() {
                    debug_assert!(kind.is_destructor());
                    Method::new(kind, destructor, false).codegen_method(ctx, &mut methods, &mut method_names, y, self);
                }
            }
        }
        let ty_for_impl = quote! {
            #canonical_ident #generics
        };
        if needs_clone_impl {
            y.push(quote! {
                impl #generics Clone for #ty_for_impl {
                    fn clone(&self) -> Self { *self }
                }
            });
        }
        if needs_default_impl {
            let prefix = ctx.trait_prefix();
            let body = quote! {
                let mut s = ::#prefix::mem::MaybeUninit::<Self>::uninit();
                unsafe {
                    ::#prefix::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
                    s.assume_init()
                }
            };
            y.push(quote! {
                impl #generics Default for #ty_for_impl {
                    fn default() -> Self {
                        #body
                    }
                }
            });
        }
        if needs_debug_impl {
            let impl_ = impl_debug::gen_debug_impl(ctx, self.fields(), it, self.kind());
            let prefix = ctx.trait_prefix();
            y.push(quote! {
                impl #generics ::#prefix::fmt::Debug for #ty_for_impl {
                    #impl_
                }
            });
        }
        if needs_partialeq_impl {
            if let Some(impl_) = utils::gen_partialeq_impl(ctx, self, it, &ty_for_impl) {
                let partialeq_bounds = if !generic_param_names.is_empty() {
                    let bounds = generic_param_names.iter().map(|t| {
                        quote! { #t: PartialEq }
                    });
                    quote! { where #( #bounds ),* }
                } else {
                    quote! {}
                };
                let prefix = ctx.trait_prefix();
                y.push(quote! {
                    impl #generics ::#prefix::cmp::PartialEq for #ty_for_impl #partialeq_bounds {
                        #impl_
                    }
                });
            }
        }
        if !methods.is_empty() {
            y.push(quote! {
                impl #generics #ty_for_impl {
                    #( #methods )*
                }
            });
        }
    }
}
impl Method {
    fn codegen_method(
        &self,
        ctx: &Context,
        methods: &mut Vec<proc_macro2::TokenStream>,
        method_names: &mut HashSet<String>,
        result: &mut CodegenResult<'_>,
        _parent: &CompInfo,
    ) {
        assert!({
            let cc = &ctx.opts().codegen_config;
            match self.kind() {
                MethodKind::Constructor => cc.constructors(),
                MethodKind::Destructor => cc.destructors(),
                MethodKind::VirtualDestructor { .. } => cc.destructors(),
                MethodKind::Static | MethodKind::Normal | MethodKind::Virtual { .. } => cc.methods(),
            }
        });
        if self.is_virtual() {
            return; // FIXME
        }
        let function_item = ctx.resolve_item(self.sig());
        if !function_item.process_before_codegen(ctx, result) {
            return;
        }
        let function = function_item.expect_function();
        let times_seen = function.codegen(ctx, result, function_item);
        let times_seen = match times_seen {
            Some(seen) => seen,
            None => return,
        };
        let signature_item = ctx.resolve_item(function.sig());
        let mut name = match self.kind() {
            MethodKind::Constructor => "new".into(),
            MethodKind::Destructor => "destruct".into(),
            _ => function.name().to_owned(),
        };
        let signature = match *signature_item.expect_type().kind() {
            TypeKind::Function(ref sig) => sig,
            _ => panic!("How in the world?"),
        };
        let supported_abi = match signature.abi(ctx, Some(&*name)) {
            ClangAbi::Known(Abi::ThisCall) => true,
            ClangAbi::Known(Abi::Vectorcall) => true,
            ClangAbi::Known(Abi::CUnwind) => true,
            _ => true,
        };
        if !supported_abi {
            return;
        }
        if signature.is_variadic() {
            return;
        }
        if method_names.contains(&name) {
            let mut count = 1;
            let mut new_name;
            while {
                new_name = format!("{}{}", name, count);
                method_names.contains(&new_name)
            } {
                count += 1;
            }
            name = new_name;
        }
        method_names.insert(name.clone());
        let mut function_name = function_item.canon_name(ctx);
        if times_seen > 0 {
            write!(&mut function_name, "{}", times_seen).unwrap();
        }
        let function_name = ctx.rust_ident(function_name);
        let mut args = utils::fnsig_arguments(ctx, signature);
        let mut ret = utils::fnsig_return_ty(ctx, signature);
        if !self.is_static() && !self.is_constructor() {
            args[0] = if self.is_const() {
                quote! { &self }
            } else {
                quote! { &mut self }
            };
        }
        if self.is_constructor() {
            args.remove(0);
            ret = quote! { -> Self };
        }
        let mut exprs = utils::ast_ty::arguments_from_signature(signature, ctx);
        let mut stmts = vec![];
        if self.is_constructor() {
            let prefix = ctx.trait_prefix();
            let tmp_variable_decl = {
                exprs[0] = quote! {
                    __bindgen_tmp.as_mut_ptr()
                };
                quote! {
                    let mut __bindgen_tmp = ::#prefix::mem::MaybeUninit::uninit()
                }
            };
            stmts.push(tmp_variable_decl);
        } else if !self.is_static() {
            assert!(!exprs.is_empty());
            exprs[0] = quote! {
                self
            };
        };
        let call = quote! {
            #function_name (#( #exprs ),* )
        };
        stmts.push(call);
        if self.is_constructor() {
            stmts.push(quote! {
                __bindgen_tmp.assume_init()
            })
        }
        let block = ctx.wrap_unsafe_ops(quote! ( #( #stmts );*));
        let mut attrs = vec![attributes::inline()];
        if signature.must_use() {
            attrs.push(attributes::must_use());
        }
        let name = ctx.rust_ident(&name);
        methods.push(quote! {
            #(#attrs)*
            pub unsafe fn #name ( #( #args ),* ) #ret {
                #block
            }
        });
    }
}

enum EnumBuilder<'a> {
    Rust {
        attrs: Vec<proc_macro2::TokenStream>,
        ident: Ident,
        tokens: proc_macro2::TokenStream,
        emitted_any_variants: bool,
    },
    NewType {
        canonical_name: &'a str,
        tokens: proc_macro2::TokenStream,
        is_bitfield: bool,
        is_global: bool,
    },
    Consts {
        variants: Vec<proc_macro2::TokenStream>,
    },
    ModuleConsts {
        module_name: &'a str,
        module_items: Vec<proc_macro2::TokenStream>,
    },
}
impl<'a> EnumBuilder<'a> {
    fn is_rust_enum(&self) -> bool {
        matches!(*self, EnumBuilder::Rust { .. })
    }
    fn new(
        name: &'a str,
        mut attrs: Vec<proc_macro2::TokenStream>,
        repr: proc_macro2::TokenStream,
        enum_variation: variation::Enum,
        has_typedef: bool,
    ) -> Self {
        let ident = Ident::new(name, Span::call_site());
        match enum_variation {
            variation::Enum::NewType { is_bitfield, is_global } => EnumBuilder::NewType {
                canonical_name: name,
                tokens: quote! {
                    #( #attrs )*
                    pub struct #ident (pub #repr);
                },
                is_bitfield,
                is_global,
            },
            variation::Enum::Rust { .. } => {
                attrs.insert(0, quote! { #[repr( #repr )] });
                let tokens = quote!();
                EnumBuilder::Rust {
                    attrs,
                    ident,
                    tokens,
                    emitted_any_variants: false,
                }
            },
            variation::Enum::Consts => {
                let mut variants = Vec::new();
                if !has_typedef {
                    variants.push(quote! {
                        #( #attrs )*
                        pub type #ident = #repr;
                    });
                }
                EnumBuilder::Consts { variants }
            },
            variation::Enum::ModuleConsts => {
                let ident = Ident::new(CONSTIFIED_ENUM_MODULE_REPR_NAME, Span::call_site());
                let type_definition = quote! {
                    #( #attrs )*
                    pub type #ident = #repr;
                };
                EnumBuilder::ModuleConsts {
                    module_name: name,
                    module_items: vec![type_definition],
                }
            },
        }
    }
    fn with_variant(
        self,
        ctx: &Context,
        variant: &EnumVariant,
        mangling_prefix: Option<&str>,
        rust_ty: proc_macro2::TokenStream,
        result: &mut CodegenResult<'_>,
        is_ty_named: bool,
    ) -> Self {
        let variant_name = ctx.rust_mangle(variant.name());
        let is_rust_enum = self.is_rust_enum();
        let expr = match variant.val() {
            EnumVariantValue::Boolean(v) if is_rust_enum => utils::ast_ty::uint_expr(v as u64),
            EnumVariantValue::Boolean(v) => quote!(#v),
            EnumVariantValue::Signed(v) => utils::ast_ty::int_expr(v),
            EnumVariantValue::Unsigned(v) => utils::ast_ty::uint_expr(v),
        };
        let mut doc = quote! {};
        if ctx.opts().generate_comments {
            if let Some(raw_comment) = variant.comment() {
                let comment = ctx.opts().process_comment(raw_comment);
                doc = attributes::doc(comment);
            }
        }
        match self {
            EnumBuilder::Rust {
                attrs,
                ident,
                tokens,
                emitted_any_variants: _,
            } => {
                let name = ctx.rust_ident(variant_name);
                EnumBuilder::Rust {
                    attrs,
                    ident,
                    tokens: quote! {
                        #tokens
                        #doc
                        #name = #expr,
                    },
                    emitted_any_variants: true,
                }
            },
            EnumBuilder::NewType {
                canonical_name,
                is_global,
                ..
            } => {
                if is_ty_named && !is_global {
                    let enum_ident = ctx.rust_ident(canonical_name);
                    let variant_ident = ctx.rust_ident(variant_name);
                    result.push(quote! {
                        impl #enum_ident {
                            #doc
                            pub const #variant_ident : #rust_ty = #rust_ty ( #expr );
                        }
                    });
                } else {
                    let ident = ctx.rust_ident(match mangling_prefix {
                        Some(prefix) => Cow::Owned(format!("{}_{}", prefix, variant_name)),
                        None => variant_name,
                    });
                    result.push(quote! {
                        #doc
                        pub const #ident : #rust_ty = #rust_ty ( #expr );
                    });
                }
                self
            },
            EnumBuilder::Consts { .. } => {
                let constant_name = match mangling_prefix {
                    Some(prefix) => Cow::Owned(format!("{}_{}", prefix, variant_name)),
                    None => variant_name,
                };
                let ident = ctx.rust_ident(constant_name);
                result.push(quote! {
                    #doc
                    pub const #ident : #rust_ty = #expr ;
                });
                self
            },
            EnumBuilder::ModuleConsts {
                module_name,
                mut module_items,
            } => {
                let name = ctx.rust_ident(variant_name);
                let ty = ctx.rust_ident(CONSTIFIED_ENUM_MODULE_REPR_NAME);
                module_items.push(quote! {
                    #doc
                    pub const #name : #ty = #expr ;
                });
                EnumBuilder::ModuleConsts {
                    module_name,
                    module_items,
                }
            },
        }
    }
    fn build(
        self,
        ctx: &Context,
        rust_ty: proc_macro2::TokenStream,
        result: &mut CodegenResult<'_>,
    ) -> proc_macro2::TokenStream {
        match self {
            EnumBuilder::Rust {
                attrs,
                ident,
                tokens,
                emitted_any_variants,
                ..
            } => {
                let variants = if !emitted_any_variants {
                    quote!(__bindgen_cannot_repr_c_on_empty_enum = 0)
                } else {
                    tokens
                };
                quote! {
                    #( #attrs )*
                    pub enum #ident {
                        #variants
                    }
                }
            },
            EnumBuilder::NewType {
                canonical_name,
                tokens,
                is_bitfield,
                ..
            } => {
                if !is_bitfield {
                    return tokens;
                }
                let rust_ty_name = ctx.rust_ident_raw(canonical_name);
                let prefix = ctx.trait_prefix();
                result.push(quote! {
                    impl ::#prefix::ops::BitOr<#rust_ty> for #rust_ty {
                        type Output = Self;
                        #[inline]
                        fn bitor(self, other: Self) -> Self {
                            #rust_ty_name(self.0 | other.0)
                        }
                    }
                });
                result.push(quote! {
                    impl ::#prefix::ops::BitOrAssign for #rust_ty {
                        #[inline]
                        fn bitor_assign(&mut self, rhs: #rust_ty) {
                            self.0 |= rhs.0;
                        }
                    }
                });
                result.push(quote! {
                    impl ::#prefix::ops::BitAnd<#rust_ty> for #rust_ty {
                        type Output = Self;
                        #[inline]
                        fn bitand(self, other: Self) -> Self {
                            #rust_ty_name(self.0 & other.0)
                        }
                    }
                });
                result.push(quote! {
                    impl ::#prefix::ops::BitAndAssign for #rust_ty {
                        #[inline]
                        fn bitand_assign(&mut self, rhs: #rust_ty) {
                            self.0 &= rhs.0;
                        }
                    }
                });
                tokens
            },
            EnumBuilder::Consts { variants, .. } => quote! { #( #variants )* },
            EnumBuilder::ModuleConsts {
                module_items,
                module_name,
                ..
            } => {
                let ident = ctx.rust_ident(module_name);
                quote! {
                    pub mod #ident {
                        #( #module_items )*
                    }
                }
            },
        }
    }
}
impl Generator for Enum {
    type Extra = Item;
    type Return = ();
    fn codegen(&self, ctx: &Context, y: &mut CodegenResult<'_>, it: &Item) {
        debug!("<Enum as CodeGenerator>::codegen: item = {:?}", it);
        debug_assert!(it.is_enabled_for_codegen(ctx));
        let name = it.canon_name(ctx);
        let ident = ctx.rust_ident(&name);
        let enum_ty = it.expect_type();
        let layout = enum_ty.layout(ctx);
        let variation = self.computed_enum_variation(ctx, it);
        let repr_translated;
        let repr = match self.repr().map(|repr| ctx.resolve_type(repr)) {
            Some(repr) if !ctx.opts().translate_enum_integer_types && !variation.is_rust() => repr,
            repr => {
                let kind = match repr {
                    Some(repr) => match *repr.canonical_type(ctx).kind() {
                        TypeKind::Int(int_kind) => int_kind,
                        _ => panic!("Unexpected type as enum repr"),
                    },
                    None => {
                        warn!(
                            "Guessing type of enum! Forward declarations of enums \
                             shouldn't be legal!"
                        );
                        IntKind::Int
                    },
                };
                let signed = kind.is_signed();
                let size = layout.map(|l| l.size).or_else(|| kind.known_size()).unwrap_or(0);
                let translated = match (signed, size) {
                    (true, 1) => IntKind::I8,
                    (false, 1) => IntKind::U8,
                    (true, 2) => IntKind::I16,
                    (false, 2) => IntKind::U16,
                    (true, 4) => IntKind::I32,
                    (false, 4) => IntKind::U32,
                    (true, 8) => IntKind::I64,
                    (false, 8) => IntKind::U64,
                    _ => {
                        warn!("invalid enum decl: signed: {}, size: {}", signed, size);
                        IntKind::I32
                    },
                };
                repr_translated = Type::new(None, None, TypeKind::Int(translated), false);
                &repr_translated
            },
        };
        let mut attrs = vec![];
        match variation {
            variation::Enum::Rust { non_exhaustive } => {
                if non_exhaustive {
                    attrs.push(attributes::non_exhaustive());
                }
            },
            variation::Enum::NewType { .. } => {
                attrs.push(attributes::repr("transparent"));
            },
            _ => {},
        };
        if let Some(comment) = it.comment(ctx) {
            attrs.push(attributes::doc(comment));
        }
        if it.must_use(ctx) {
            attrs.push(attributes::must_use());
        }
        if !variation.is_const() {
            let packed = false; // Enums can't be packed in Rust.
            let mut derives = derives_of_item(it, ctx, packed);
            derives.insert(
                DerivableTraits::CLONE | DerivableTraits::HASH | DerivableTraits::PARTIAL_EQ | DerivableTraits::EQ,
            );
            let mut derives: Vec<_> = derives.into();
            for derive in it.annotations().derives().iter() {
                if !derives.contains(&derive.as_str()) {
                    derives.push(derive);
                }
            }
            let custom_derives = ctx.opts().all_callbacks(|cb| {
                cb.add_derives(&DeriveInfo {
                    name: &name,
                    kind: DeriveTypeKind::Enum,
                })
            });
            derives.extend(custom_derives.iter().map(|s| s.as_str()));
            attrs.push(attributes::derives(&derives));
        }
        fn add_constant(
            ctx: &Context,
            enum_: &Type,
            enum_canonical_name: &Ident,
            variant_name: &Ident,
            referenced_name: &Ident,
            enum_rust_ty: proc_macro2::TokenStream,
            result: &mut CodegenResult<'_>,
        ) {
            let constant_name = if enum_.name().is_some() {
                if ctx.opts().prepend_enum_name {
                    format!("{}_{}", enum_canonical_name, variant_name)
                } else {
                    format!("{}", variant_name)
                }
            } else {
                format!("{}", variant_name)
            };
            let constant_name = ctx.rust_ident(constant_name);
            result.push(quote! {
                pub const #constant_name : #enum_rust_ty =
                    #enum_canonical_name :: #referenced_name ;
            });
        }
        let repr = repr.to_rust_or_opaque(ctx, it);
        let has_typedef = ctx.is_enum_typedef_combo(it.id());
        let mut builder = EnumBuilder::new(&name, attrs, repr, variation, has_typedef);
        let mut seen_values = HashMap::<_, Ident>::default();
        let enum_rust_ty = it.to_rust_or_opaque(ctx, &());
        let is_toplevel = it.is_toplevel(ctx);
        let parent_canonical_name = if is_toplevel {
            None
        } else {
            Some(it.parent_id().canon_name(ctx))
        };
        let constant_mangling_prefix = if ctx.opts().prepend_enum_name {
            if enum_ty.name().is_none() {
                parent_canonical_name.as_deref()
            } else {
                Some(&*name)
            }
        } else {
            None
        };
        let mut constified_variants = VecDeque::new();
        let mut iter = self.variants().iter().peekable();
        while let Some(variant) = iter.next().or_else(|| constified_variants.pop_front()) {
            if variant.hidden() {
                continue;
            }
            if variant.force_constification() && iter.peek().is_some() {
                constified_variants.push_back(variant);
                continue;
            }
            match seen_values.entry(variant.val()) {
                Entry::Occupied(ref entry) => {
                    if variation.is_rust() {
                        let variant_name = ctx.rust_mangle(variant.name());
                        let mangled_name = if is_toplevel || enum_ty.name().is_some() {
                            variant_name
                        } else {
                            let parent_name = parent_canonical_name.as_ref().unwrap();
                            Cow::Owned(format!("{}_{}", parent_name, variant_name))
                        };
                        let existing_variant_name = entry.get();
                        if enum_ty.name().is_some() {
                            let enum_canonical_name = &ident;
                            let variant_name = ctx.rust_ident_raw(&*mangled_name);
                            y.push(quote! {
                                impl #enum_rust_ty {
                                    pub const #variant_name : #enum_rust_ty =
                                        #enum_canonical_name :: #existing_variant_name ;
                                }
                            });
                        } else {
                            add_constant(
                                ctx,
                                enum_ty,
                                &ident,
                                &Ident::new(&mangled_name, Span::call_site()),
                                existing_variant_name,
                                enum_rust_ty.clone(),
                                y,
                            );
                        }
                    } else {
                        builder = builder.with_variant(
                            ctx,
                            variant,
                            constant_mangling_prefix,
                            enum_rust_ty.clone(),
                            y,
                            enum_ty.name().is_some(),
                        );
                    }
                },
                Entry::Vacant(entry) => {
                    builder = builder.with_variant(
                        ctx,
                        variant,
                        constant_mangling_prefix,
                        enum_rust_ty.clone(),
                        y,
                        enum_ty.name().is_some(),
                    );
                    let variant_name = ctx.rust_ident(variant.name());
                    if (variation.is_rust() && enum_ty.name().is_none()) || variant.force_constification() {
                        let mangled_name = if is_toplevel {
                            variant_name.clone()
                        } else {
                            let parent_name = parent_canonical_name.as_ref().unwrap();
                            Ident::new(&format!("{}_{}", parent_name, variant_name), Span::call_site())
                        };
                        add_constant(
                            ctx,
                            enum_ty,
                            &ident,
                            &mangled_name,
                            &variant_name,
                            enum_rust_ty.clone(),
                            y,
                        );
                    }
                    entry.insert(variant_name);
                },
            }
        }
        let item = builder.build(ctx, enum_rust_ty, y);
        y.push(item);
    }
}

impl Generator for Function {
    type Extra = Item;
    type Return = Option<u32>;
    fn codegen(&self, ctx: &Context, y: &mut CodegenResult<'_>, it: &Item) -> Self::Return {
        debug!("<Function as CodeGenerator>::codegen: item = {:?}", it);
        debug_assert!(it.is_enabled_for_codegen(ctx));
        let is_internal = matches!(self.linkage(), Linkage::Internal);
        let signature_item = ctx.resolve_item(self.sig());
        let signature = signature_item.kind().expect_type().canonical_type(ctx);
        let signature = match *signature.kind() {
            TypeKind::Function(ref sig) => sig,
            _ => panic!("Signature kind is not a Function: {:?}", signature),
        };
        if is_internal {
            return None;
        }
        let is_dynamic_function = match self.kind() {
            FnKind::Method(ref method_kind) if method_kind.is_pure_virtual() => {
                return None;
            },
            FnKind::Function => ctx.opts().dynamic_library_name.is_some(),
            _ => false,
        };
        if !it.all_templ_params(ctx).is_empty() {
            return None;
        }
        let name = self.name();
        let mut canonical_name = it.canon_name(ctx);
        let mangled_name = self.mangled_name();
        {
            let seen_symbol_name = mangled_name.unwrap_or(&canonical_name);
            if y.seen_function(seen_symbol_name) {
                return None;
            }
            y.saw_function(seen_symbol_name);
        }
        let args = utils::fnsig_arguments(ctx, signature);
        let ret = utils::fnsig_return_ty(ctx, signature);
        let mut attributes = vec![];
        let must_use = signature.must_use() || {
            let ret_ty = signature.ret_type().into_resolver().through_type_refs().resolve(ctx);
            ret_ty.must_use(ctx)
        };
        if must_use {
            attributes.push(attributes::must_use());
        }
        if let Some(comment) = it.comment(ctx) {
            attributes.push(attributes::doc(comment));
        }
        let abi = match signature.abi(ctx, Some(name)) {
            abi => abi,
        };
        let times_seen = y.overload_number(&canonical_name);
        if times_seen > 0 {
            write!(&mut canonical_name, "{}", times_seen).unwrap();
        }
        let mut has_link_name_attr = false;
        if let Some(link_name) = self.link_name() {
            attributes.push(attributes::link_name::<false>(link_name));
            has_link_name_attr = true;
        } else {
            let link_name = mangled_name.unwrap_or(name);
            if !is_dynamic_function
                && !utils::names_will_be_identical_after_mangling(&canonical_name, link_name, Some(abi))
            {
                attributes.push(attributes::link_name::<false>(link_name));
                has_link_name_attr = true;
            }
        }
        let wasm_link_attribute = ctx.opts().wasm_import_module_name.as_ref().map(|name| {
            quote! { #[link(wasm_import_module = #name)] }
        });
        let ident = ctx.rust_ident(canonical_name);
        let tokens = quote! {
            #wasm_link_attribute
            extern #abi {
                #(#attributes)*
                pub fn #ident ( #( #args ),* ) #ret;
            }
        };
        if is_dynamic_function {
            let args_identifiers = utils::fnsig_argument_identifiers(ctx, signature);
            let ret_ty = utils::fnsig_return_ty(ctx, signature);
            y.dynamic_items().push(
                ident,
                abi,
                signature.is_variadic(),
                ctx.opts().dynamic_link_require_all,
                args,
                args_identifiers,
                ret,
                ret_ty,
                attributes,
                ctx,
            );
        } else {
            y.push(tokens);
        }
        Some(times_seen)
    }
}
pub fn codegen(ctx: Context) -> Result<(proc_macro2::TokenStream, Opts), CodegenError> {
    ctx.gen(|ctx2| {
        let _t = ctx2.timer("codegen");
        let counter = Cell::new(0);
        let mut y = CodegenResult::new(&counter);
        debug!("codegen: {:?}", ctx2.opts());
        if ctx2.opts().emit_ir {
            let codegen_items = ctx2.codegen_items();
            for (id, item) in ctx2.items() {
                if codegen_items.contains(&id) {
                    println!("ir: {:?} = {:#?}", id, item);
                }
            }
        }
        if let Some(path) = ctx2.opts().emit_ir_graphviz.as_ref() {
            match dot::write_dot_file(ctx2, path) {
                Ok(()) => info!("Your dot file was generated successfully into: {}", path),
                Err(e) => warn!("{}", e),
            }
        }
        if let Some(spec) = ctx2.opts().depfile.as_ref() {
            match spec.write(ctx2.deps()) {
                Ok(()) => info!(
                    "Your depfile was generated successfully into: {}",
                    spec.depfile_path.display()
                ),
                Err(e) => warn!("{}", e),
            }
        }
        ctx2.resolve_item(ctx2.root_module()).codegen(ctx2, &mut y, &());
        if let Some(ref lib_name) = ctx2.opts().dynamic_library_name {
            let lib_ident = ctx2.rust_ident(lib_name);
            let dynamic_items_tokens = y.dynamic_items().get_tokens(lib_ident, ctx2);
            y.push(dynamic_items_tokens);
        }
        utils::serialize_items(&y, ctx2)?;
        Ok(postproc::postproc(y.items, ctx2.opts()))
    })
}

#[cfg(test)]
#[allow(warnings)]
pub mod bitfield_unit;

mod serialize;
pub mod struct_layout;

trait TryToRust {
    type Extra;
    fn try_to_rust(&self, ctx: &Context, x: &Self::Extra) -> error::Result<proc_macro2::TokenStream>;
}
impl<T> TryToRust for T
where
    T: Copy + Into<ItemId>,
{
    type Extra = ();
    fn try_to_rust(&self, ctx: &Context, _: &()) -> error::Result<proc_macro2::TokenStream> {
        ctx.resolve_item((*self).into()).try_to_rust(ctx, &())
    }
}
impl TryToRust for FnSig {
    type Extra = ();
    fn try_to_rust(&self, ctx: &Context, _: &()) -> error::Result<proc_macro2::TokenStream> {
        let ty = utils::fnsig_return_ty(ctx, self);
        let args = utils::fnsig_arguments(ctx, self);
        match self.abi(ctx, None) {
            abi => Ok(quote! {
                unsafe extern #abi fn ( #( #args ),* ) #ty
            }),
        }
    }
}
impl TryToRust for Item {
    type Extra = ();
    fn try_to_rust(&self, ctx: &Context, _: &()) -> error::Result<proc_macro2::TokenStream> {
        self.kind().expect_type().try_to_rust(ctx, self)
    }
}
impl TryToRust for TemplInstantiation {
    type Extra = Item;
    fn try_to_rust(&self, ctx: &Context, it: &Item) -> error::Result<proc_macro2::TokenStream> {
        if self.is_opaque(ctx, it) {
            return Err(error::Error::InstantiationOfOpaqueType);
        }
        let def = self.templ_def().into_resolver().through_type_refs().resolve(ctx);
        let mut ty = quote! {};
        let path = def.namespace_aware_canon_path(ctx);
        ty.append_separated(path.into_iter().map(|x| ctx.rust_ident(x)), quote!(::));
        let ps = def.self_templ_params(ctx);
        if ps.is_empty() {
            return Err(error::Error::InstantiationOfOpaqueType);
        }
        let args = self
            .templ_args()
            .iter()
            .zip(ps.iter())
            .filter(|&(_, x)| ctx.uses_templ_param(def.id(), *x))
            .map(|(x, _)| {
                let x = x.into_resolver().through_type_refs().resolve(ctx);
                let mut ty = x.try_to_rust(ctx, &())?;
                ty.append_implicit_templ_params(ctx, x);
                Ok(ty)
            })
            .collect::<error::Result<Vec<_>>>()?;
        if args.is_empty() {
            return Ok(ty);
        }
        Ok(quote! {
            #ty < #( #args ),* >
        })
    }
}
impl TryToRust for Type {
    type Extra = Item;
    fn try_to_rust(&self, ctx: &Context, it: &Item) -> error::Result<proc_macro2::TokenStream> {
        use self::utils::ast_ty::*;
        match *self.kind() {
            TypeKind::Void => Ok(c_void(ctx)),
            TypeKind::NullPtr => Ok(c_void(ctx).to_ptr(true)),
            TypeKind::Int(ik) => match ik {
                IntKind::Bool => Ok(quote! { bool }),
                IntKind::Char { .. } => Ok(raw_type(ctx, "c_char")),
                IntKind::SChar => Ok(raw_type(ctx, "c_schar")),
                IntKind::UChar => Ok(raw_type(ctx, "c_uchar")),
                IntKind::Short => Ok(raw_type(ctx, "c_short")),
                IntKind::UShort => Ok(raw_type(ctx, "c_ushort")),
                IntKind::Int => Ok(raw_type(ctx, "c_int")),
                IntKind::UInt => Ok(raw_type(ctx, "c_uint")),
                IntKind::Long => Ok(raw_type(ctx, "c_long")),
                IntKind::ULong => Ok(raw_type(ctx, "c_ulong")),
                IntKind::LongLong => Ok(raw_type(ctx, "c_longlong")),
                IntKind::ULongLong => Ok(raw_type(ctx, "c_ulonglong")),
                IntKind::WChar => {
                    let x = self.layout(ctx).expect("Couldn't compute wchar_t's layout?");
                    let ty = Layout::known_type_for_size(ctx, x.size).expect("Non-representable wchar_t?");
                    let ty = ctx.rust_ident_raw(ty);
                    Ok(quote! { #ty })
                },
                IntKind::I8 => Ok(quote! { i8 }),
                IntKind::U8 => Ok(quote! { u8 }),
                IntKind::I16 => Ok(quote! { i16 }),
                IntKind::U16 => Ok(quote! { u16 }),
                IntKind::I32 => Ok(quote! { i32 }),
                IntKind::U32 => Ok(quote! { u32 }),
                IntKind::I64 => Ok(quote! { i64 }),
                IntKind::U64 => Ok(quote! { u64 }),
                IntKind::Custom { name, .. } => Ok(proc_macro2::TokenStream::from_str(name).unwrap()),
                IntKind::U128 => Ok(quote! { u128 }),
                IntKind::I128 => Ok(quote! { i128 }),
            },
            TypeKind::Float(x) => Ok(float_kind_rust_type(ctx, x, self.layout(ctx))),
            TypeKind::Complex(x) => {
                let ty = float_kind_rust_type(ctx, x, self.layout(ctx));
                ctx.generated_bindgen_complex();
                Ok(if ctx.opts().enable_cxx_namespaces {
                    quote! {
                        root::__BindgenComplex<#ty>
                    }
                } else {
                    quote! {
                        __BindgenComplex<#ty>
                    }
                })
            },
            TypeKind::Function(ref x) => {
                let ty = x.try_to_rust(ctx, &())?;
                let prefix = ctx.trait_prefix();
                Ok(quote! {
                    ::#prefix::option::Option<#ty>
                })
            },
            TypeKind::Array(item, len) | TypeKind::Vector(item, len) => {
                let ty = item.try_to_rust(ctx, &())?;
                Ok(quote! {
                    [ #ty ; #len ]
                })
            },
            TypeKind::Enum(..) => {
                let y = it.namespace_aware_canon_path(ctx);
                let y = proc_macro2::TokenStream::from_str(&y.join("::")).unwrap();
                Ok(quote!(#y))
            },
            TypeKind::TemplInstantiation(ref x) => x.try_to_rust(ctx, it),
            TypeKind::ResolvedTypeRef(x) => x.try_to_rust(ctx, &()),
            TypeKind::TemplAlias(..) | TypeKind::Alias(..) | TypeKind::BlockPointer(..) => {
                if self.is_block_pointer() && !ctx.opts().generate_block {
                    let y = c_void(ctx);
                    return Ok(y.to_ptr(/* is_const = */ false));
                }
                if it.is_opaque(ctx, &())
                    && it
                        .used_templ_params(ctx)
                        .into_iter()
                        .any(|x| x.is_templ_param(ctx, &()))
                {
                    self.try_to_opaque(ctx, it)
                } else if let Some(ty) = self.name().and_then(|x| utils::type_from_named(ctx, x)) {
                    Ok(ty)
                } else {
                    utils::build_path(it, ctx)
                }
            },
            TypeKind::Comp(ref x) => {
                let ps = it.all_templ_params(ctx);
                if x.has_non_type_templ_params() || (it.is_opaque(ctx, &()) && !ps.is_empty()) {
                    return self.try_to_opaque(ctx, it);
                }
                utils::build_path(it, ctx)
            },
            TypeKind::Opaque => self.try_to_opaque(ctx, it),
            TypeKind::Pointer(x) | TypeKind::Reference(x) => {
                let is_const = ctx.resolve_type(x).is_const();
                let x = x.into_resolver().through_type_refs().resolve(ctx);
                let mut ty = x.to_rust_or_opaque(ctx, &());
                ty.append_implicit_templ_params(ctx, x);
                if x.expect_type().canonical_type(ctx).is_function() {
                    Ok(ty)
                } else {
                    Ok(ty.to_ptr(is_const))
                }
            },
            TypeKind::TypeParam => {
                let y = it.canon_name(ctx);
                let y = ctx.rust_ident(y);
                Ok(quote! {
                    #y
                })
            },
            ref u @ TypeKind::UnresolvedTypeRef(..) => {
                unreachable!("Should have been resolved after parsing {:?}!", u)
            },
        }
    }
}
impl<'a> TryToRust for Vtable<'a> {
    type Extra = ();
    fn try_to_rust(&self, ctx: &Context, _: &()) -> error::Result<proc_macro2::TokenStream> {
        let y = ctx.rust_ident(self.canon_name(ctx));
        Ok(quote! {
            #y
        })
    }
}

trait TryToOpaque {
    type Extra;
    fn try_get_layout(&self, ctx: &Context, x: &Self::Extra) -> error::Result<Layout>;
    fn try_to_opaque(&self, ctx: &Context, x: &Self::Extra) -> error::Result<proc_macro2::TokenStream> {
        self.try_get_layout(ctx, x).map(|x| utils::blob(ctx, x))
    }
}
impl<T> TryToOpaque for T
where
    T: Copy + Into<ItemId>,
{
    type Extra = ();
    fn try_get_layout(&self, ctx: &Context, _: &()) -> error::Result<Layout> {
        ctx.resolve_item((*self).into()).try_get_layout(ctx, &())
    }
}
impl TryToOpaque for Item {
    type Extra = ();
    fn try_get_layout(&self, ctx: &Context, _: &()) -> error::Result<Layout> {
        self.kind().expect_type().try_get_layout(ctx, self)
    }
}
impl TryToOpaque for TemplInstantiation {
    type Extra = Item;
    fn try_get_layout(&self, ctx: &Context, it: &Item) -> error::Result<Layout> {
        it.expect_type().layout(ctx).ok_or(error::Error::NoLayoutForOpaqueBlob)
    }
}
impl TryToOpaque for Type {
    type Extra = Item;
    fn try_get_layout(&self, ctx: &Context, _: &Item) -> error::Result<Layout> {
        self.layout(ctx).ok_or(error::Error::NoLayoutForOpaqueBlob)
    }
}

trait TryToRustOrOpaque: TryToRust + TryToOpaque {
    type Extra;
    fn try_to_rust_or_opaque(
        &self,
        ctx: &Context,
        x: &<Self as TryToRustOrOpaque>::Extra,
    ) -> error::Result<proc_macro2::TokenStream>;
}
impl<E, T> TryToRustOrOpaque for T
where
    T: TryToRust<Extra = E> + TryToOpaque<Extra = E>,
{
    type Extra = E;
    fn try_to_rust_or_opaque(&self, ctx: &Context, x: &E) -> error::Result<proc_macro2::TokenStream> {
        self.try_to_rust(ctx, x).or_else(|_| {
            if let Ok(x) = self.try_get_layout(ctx, x) {
                Ok(utils::blob(ctx, x))
            } else {
                Err(error::Error::NoLayoutForOpaqueBlob)
            }
        })
    }
}

trait ToOpaque: TryToOpaque {
    fn get_layout(&self, ctx: &Context, x: &Self::Extra) -> Layout {
        self.try_get_layout(ctx, x).unwrap_or_else(|_| Layout::for_size(ctx, 1))
    }
    fn to_opaque(&self, ctx: &Context, x: &Self::Extra) -> proc_macro2::TokenStream {
        utils::blob(ctx, self.get_layout(ctx, x))
    }
}
impl<T> ToOpaque for T where T: TryToOpaque {}
trait ToRustOrOpaque: TryToRust + ToOpaque {
    type Extra;
    fn to_rust_or_opaque(&self, ctx: &Context, x: &<Self as ToRustOrOpaque>::Extra) -> proc_macro2::TokenStream;
}
impl<E, T> ToRustOrOpaque for T
where
    T: TryToRust<Extra = E> + ToOpaque<Extra = E>,
{
    type Extra = E;
    fn to_rust_or_opaque(&self, ctx: &Context, x: &E) -> proc_macro2::TokenStream {
        self.try_to_rust(ctx, x).unwrap_or_else(|_| self.to_opaque(ctx, x))
    }
}
