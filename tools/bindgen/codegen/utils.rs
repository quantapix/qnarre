use super::serialize::CSerialize;
use super::{error, GenError, GenResult, ToRustOrOpaque};
use crate::ir::comp::{CompInfo, CompKind, Field, FieldMethods};
use crate::ir::function::{Abi, ClangAbi, FnSig};
use crate::ir::item::{CanonPath, IsOpaque, Item};
use crate::ir::layout::Layout;
use crate::ir::typ::TypeKind;
use crate::ir::Context;
use crate::{args_are_cpp, file_is_cpp};

use proc_macro2::{Ident, Span, TokenStream};
use quote::TokenStreamExt;
use std::borrow::Cow;
use std::io::Write;
use std::mem;
use std::path::PathBuf;
use std::str::FromStr;

pub(super) fn serialize_items(result: &GenResult, ctx: &Context) -> Result<(), GenError> {
    if result.to_serialize.is_empty() {
        return Ok(());
    }
    let path = ctx
        .opts()
        .wrap_static_fns_path
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("bindgen").join("extern"));
    let dir = path.parent().unwrap();
    if !dir.exists() {
        std::fs::create_dir_all(dir)?;
    }
    let is_cpp = args_are_cpp(&ctx.opts().clang_args) || ctx.opts().input_headers.iter().any(|h| file_is_cpp(h));
    let source_path = path.with_extension(if is_cpp { "cpp" } else { "c" });
    let mut code = Vec::new();
    if !ctx.opts().input_headers.is_empty() {
        for header in &ctx.opts().input_headers {
            writeln!(code, "#include \"{}\"", header)?;
        }
        writeln!(code)?;
    }
    if !ctx.opts().input_header_contents.is_empty() {
        for (name, contents) in &ctx.opts().input_header_contents {
            writeln!(code, "// {}\n{}", name, contents)?;
        }
        writeln!(code)?;
    }
    writeln!(code, "// Static wrappers\n")?;
    for &id in &result.to_serialize {
        let item = ctx.resolve_item(id);
        item.serialize(ctx, (), &mut vec![], &mut code)?;
    }
    std::fs::write(source_path, code)?;
    Ok(())
}
pub fn prepend_bitfield_unit_type(ctx: &Context, y: &mut Vec<proc_macro2::TokenStream>) {
    let bitfield_unit_src = include_str!("./bitfield_unit.rs");
    let bitfield_unit_src = Cow::Borrowed(bitfield_unit_src);
    let bitfield_unit_type = proc_macro2::TokenStream::from_str(&bitfield_unit_src).unwrap();
    let bitfield_unit_type = quote!(#bitfield_unit_type);
    let items = vec![bitfield_unit_type];
    let old_items = mem::replace(y, items);
    y.extend(old_items);
}
pub fn prepend_block_header(ctx: &Context, y: &mut Vec<proc_macro2::TokenStream>) {
    let use_block = if ctx.opts().block_extern_crate {
        quote! {
            extern crate block;
        }
    } else {
        quote! {
            use block;
        }
    };
    let items = vec![use_block];
    let old_items = mem::replace(y, items);
    y.extend(old_items.into_iter());
}
pub fn prepend_union_types(ctx: &Context, y: &mut Vec<proc_macro2::TokenStream>) {
    let prefix = ctx.trait_prefix();
    let const_fn = quote! { const fn };
    let union_field_decl = quote! {
        #[repr(C)]
        pub struct __BindgenUnionField<T>(::#prefix::marker::PhantomData<T>);
    };
    let transmute = ctx.wrap_unsafe_ops(quote!(::#prefix::mem::transmute(self)));
    let union_field_impl = quote! {
        impl<T> __BindgenUnionField<T> {
            #[inline]
            pub #const_fn new() -> Self {
                __BindgenUnionField(::#prefix::marker::PhantomData)
            }
            #[inline]
            pub unsafe fn as_ref(&self) -> &T {
                #transmute
            }
            #[inline]
            pub unsafe fn as_mut(&mut self) -> &mut T {
                #transmute
            }
        }
    };
    let union_field_default_impl = quote! {
        impl<T> ::#prefix::default::Default for __BindgenUnionField<T> {
            #[inline]
            fn default() -> Self {
                Self::new()
            }
        }
    };
    let union_field_clone_impl = quote! {
        impl<T> ::#prefix::clone::Clone for __BindgenUnionField<T> {
            #[inline]
            fn clone(&self) -> Self {
                Self::new()
            }
        }
    };
    let union_field_copy_impl = quote! {
        impl<T> ::#prefix::marker::Copy for __BindgenUnionField<T> {}
    };
    let union_field_debug_impl = quote! {
        impl<T> ::#prefix::fmt::Debug for __BindgenUnionField<T> {
            fn fmt(&self, fmt: &mut ::#prefix::fmt::Formatter<'_>)
                   -> ::#prefix::fmt::Result {
                fmt.write_str("__BindgenUnionField")
            }
        }
    };
    let union_field_hash_impl = quote! {
        impl<T> ::#prefix::hash::Hash for __BindgenUnionField<T> {
            fn hash<H: ::#prefix::hash::Hasher>(&self, _state: &mut H) {
            }
        }
    };
    let union_field_partialeq_impl = quote! {
        impl<T> ::#prefix::cmp::PartialEq for __BindgenUnionField<T> {
           fn eq(&self, _other: &__BindgenUnionField<T>) -> bool {
               true
           }
       }
    };
    let union_field_eq_impl = quote! {
       impl<T> ::#prefix::cmp::Eq for __BindgenUnionField<T> {
       }
    };
    let items = vec![
        union_field_decl,
        union_field_impl,
        union_field_default_impl,
        union_field_clone_impl,
        union_field_copy_impl,
        union_field_debug_impl,
        union_field_hash_impl,
        union_field_partialeq_impl,
        union_field_eq_impl,
    ];
    let old_items = mem::replace(y, items);
    y.extend(old_items.into_iter());
}
pub fn prepend_incomplete_array_types(ctx: &Context, y: &mut Vec<proc_macro2::TokenStream>) {
    let prefix = ctx.trait_prefix();
    let const_fn = quote! { const fn };
    let incomplete_array_decl = quote! {
        #[repr(C)]
        #[derive(Default)]
        pub struct __IncompleteArrayField<T>(
            ::#prefix::marker::PhantomData<T>, [T; 0]);
    };
    let from_raw_parts = ctx.wrap_unsafe_ops(quote! (
        ::#prefix::slice::from_raw_parts(self.as_ptr(), len)
    ));
    let from_raw_parts_mut = ctx.wrap_unsafe_ops(quote! (
        ::#prefix::slice::from_raw_parts_mut(self.as_mut_ptr(), len)
    ));
    let incomplete_array_impl = quote! {
        impl<T> __IncompleteArrayField<T> {
            #[inline]
            pub #const_fn new() -> Self {
                __IncompleteArrayField(::#prefix::marker::PhantomData, [])
            }
            #[inline]
            pub fn as_ptr(&self) -> *const T {
                self as *const _ as *const T
            }
            #[inline]
            pub fn as_mut_ptr(&mut self) -> *mut T {
                self as *mut _ as *mut T
            }
            #[inline]
            pub unsafe fn as_slice(&self, len: usize) -> &[T] {
                #from_raw_parts
            }
            #[inline]
            pub unsafe fn as_mut_slice(&mut self, len: usize) -> &mut [T] {
                #from_raw_parts_mut
            }
        }
    };
    let incomplete_array_debug_impl = quote! {
        impl<T> ::#prefix::fmt::Debug for __IncompleteArrayField<T> {
            fn fmt(&self, fmt: &mut ::#prefix::fmt::Formatter<'_>)
                   -> ::#prefix::fmt::Result {
                fmt.write_str("__IncompleteArrayField")
            }
        }
    };
    let items = vec![
        incomplete_array_decl,
        incomplete_array_impl,
        incomplete_array_debug_impl,
    ];
    let old_items = mem::replace(y, items);
    y.extend(old_items.into_iter());
}
pub fn prepend_complex_type(y: &mut Vec<proc_macro2::TokenStream>) {
    let complex_type = quote! {
        #[derive(PartialEq, Copy, Clone, Hash, Debug, Default)]
        #[repr(C)]
        pub struct __BindgenComplex<T> {
            pub re: T,
            pub im: T
        }
    };
    let items = vec![complex_type];
    let old_items = mem::replace(y, items);
    y.extend(old_items.into_iter());
}
pub fn build_path(it: &Item, ctx: &Context) -> error::Result<proc_macro2::TokenStream> {
    let path = it.namespace_aware_canon_path(ctx);
    let ys = proc_macro2::TokenStream::from_str(&path.join("::")).unwrap();
    Ok(ys)
}
fn primitive_ty(ctx: &Context, name: &str) -> proc_macro2::TokenStream {
    let y = ctx.rust_ident_raw(name);
    quote! {
        #y
    }
}
pub fn type_from_named(ctx: &Context, name: &str) -> Option<proc_macro2::TokenStream> {
    Some(match name {
        "int8_t" => primitive_ty(ctx, "i8"),
        "uint8_t" => primitive_ty(ctx, "u8"),
        "int16_t" => primitive_ty(ctx, "i16"),
        "uint16_t" => primitive_ty(ctx, "u16"),
        "int32_t" => primitive_ty(ctx, "i32"),
        "uint32_t" => primitive_ty(ctx, "u32"),
        "int64_t" => primitive_ty(ctx, "i64"),
        "uint64_t" => primitive_ty(ctx, "u64"),
        "size_t" if ctx.opts().size_t_is_usize => primitive_ty(ctx, "usize"),
        "uintptr_t" => primitive_ty(ctx, "usize"),
        "ssize_t" if ctx.opts().size_t_is_usize => primitive_ty(ctx, "isize"),
        "intptr_t" | "ptrdiff_t" => primitive_ty(ctx, "isize"),
        _ => return None,
    })
}
fn fnsig_return_ty_internal(ctx: &Context, sig: &FnSig, include_arrow: bool) -> proc_macro2::TokenStream {
    if sig.is_divergent() {
        return if include_arrow {
            quote! { -> ! }
        } else {
            quote! { ! }
        };
    }
    let canonical_type_kind = sig
        .ret_type()
        .into_resolver()
        .through_type_refs()
        .through_type_aliases()
        .resolve(ctx)
        .kind()
        .expect_type()
        .kind();
    if let TypeKind::Void = canonical_type_kind {
        return if include_arrow {
            quote! {}
        } else {
            quote! { () }
        };
    }
    let ret_ty = sig.ret_type().to_rust_or_opaque(ctx, &());
    if include_arrow {
        quote! { -> #ret_ty }
    } else {
        ret_ty
    }
}
pub fn fnsig_return_ty(ctx: &Context, sig: &FnSig) -> proc_macro2::TokenStream {
    fnsig_return_ty_internal(ctx, sig, /* include_arrow = */ true)
}
pub fn fnsig_arguments(ctx: &Context, sig: &FnSig) -> Vec<proc_macro2::TokenStream> {
    use super::ToPtr;
    let mut unnamed_arguments = 0;
    let mut args = sig
        .arg_types()
        .iter()
        .map(|&(ref name, ty)| {
            let arg_item = ctx.resolve_item(ty);
            let arg_ty = arg_item.kind().expect_type();
            let arg_ty = match *arg_ty.canonical_type(ctx).kind() {
                TypeKind::Array(t, _) => {
                    let stream = if ctx.opts().array_pointers_in_arguments {
                        arg_ty.to_rust_or_opaque(ctx, arg_item)
                    } else {
                        t.to_rust_or_opaque(ctx, &())
                    };
                    stream.to_ptr(ctx.resolve_type(t).is_const())
                },
                TypeKind::Pointer(inner) => {
                    let inner = ctx.resolve_item(inner);
                    let inner_ty = inner.expect_type();
                    arg_item.to_rust_or_opaque(ctx, &())
                },
                _ => arg_item.to_rust_or_opaque(ctx, &()),
            };
            let arg_name = match *name {
                Some(ref name) => ctx.rust_mangle(name).into_owned(),
                None => {
                    unnamed_arguments += 1;
                    format!("arg{}", unnamed_arguments)
                },
            };
            assert!(!arg_name.is_empty());
            let arg_name = ctx.rust_ident(arg_name);
            quote! {
                #arg_name : #arg_ty
            }
        })
        .collect::<Vec<_>>();
    if sig.is_variadic() {
        args.push(quote! { ... })
    }
    args
}
pub fn fnsig_argument_identifiers(ctx: &Context, sig: &FnSig) -> Vec<proc_macro2::TokenStream> {
    let mut unnamed_arguments = 0;
    let args = sig
        .arg_types()
        .iter()
        .map(|&(ref name, _ty)| {
            let arg_name = match *name {
                Some(ref name) => ctx.rust_mangle(name).into_owned(),
                None => {
                    unnamed_arguments += 1;
                    format!("arg{}", unnamed_arguments)
                },
            };
            assert!(!arg_name.is_empty());
            let arg_name = ctx.rust_ident(arg_name);
            quote! {
                #arg_name
            }
        })
        .collect::<Vec<_>>();
    args
}
pub fn fnsig_block(ctx: &Context, sig: &FnSig) -> proc_macro2::TokenStream {
    let args = sig.arg_types().iter().map(|&(_, ty)| {
        let arg_item = ctx.resolve_item(ty);
        arg_item.to_rust_or_opaque(ctx, &())
    });
    let ret_ty = fnsig_return_ty_internal(ctx, sig, /* include_arrow = */ false);
    quote! {
        *const ::block::Block<(#(#args,)*), #ret_ty>
    }
}
pub fn names_will_be_identical_after_mangling(canonical: &str, mangled: &str, call_conv: Option<ClangAbi>) -> bool {
    if canonical == mangled {
        return true;
    }
    let canonical = canonical.as_bytes();
    let mangled = mangled.as_bytes();
    let (prefix, expect_suffix) = match call_conv {
        Some(ClangAbi::Known(Abi::C)) | None => (b'_', false),
        Some(ClangAbi::Known(Abi::Stdcall)) => (b'_', true),
        Some(ClangAbi::Known(Abi::Fastcall)) => (b'@', true),
        Some(_) => return false,
    };
    if mangled.len() < canonical.len() + 1 {
        return false;
    }
    if mangled[0] != prefix {
        return false;
    }
    if &mangled[1..canonical.len() + 1] != canonical {
        return false;
    }
    if expect_suffix {
        let suffix = &mangled[canonical.len() + 1..];
        if suffix.len() < 2 {
            return false;
        }
        if suffix[0] != b'@' || !suffix[1..].iter().all(u8::is_ascii_digit) {
            return false;
        }
    } else if mangled.len() != canonical.len() + 1 {
        return false;
    }
    true
}

pub mod attrs {
    use proc_macro2::{Ident, Span, TokenStream};
    use std::{borrow::Cow, str::FromStr};
    pub fn repr(which: &str) -> TokenStream {
        let which = Ident::new(which, Span::call_site());
        quote! {
            #[repr( #which )]
        }
    }
    pub fn repr_list(which_ones: &[&str]) -> TokenStream {
        let which_ones = which_ones
            .iter()
            .cloned()
            .map(|one| TokenStream::from_str(one).expect("repr to be valid"));
        quote! {
            #[repr( #( #which_ones ),* )]
        }
    }
    pub fn derives(which_ones: &[&str]) -> TokenStream {
        let which_ones = which_ones
            .iter()
            .cloned()
            .map(|one| TokenStream::from_str(one).expect("derive to be valid"));
        quote! {
            #[derive( #( #which_ones ),* )]
        }
    }
    pub fn inline() -> TokenStream {
        quote! {
            #[inline]
        }
    }
    pub fn must_use() -> TokenStream {
        quote! {
            #[must_use]
        }
    }
    pub fn non_exhaustive() -> TokenStream {
        quote! {
            #[non_exhaustive]
        }
    }
    pub fn doc(comment: String) -> TokenStream {
        if comment.is_empty() {
            quote!()
        } else {
            quote!(#[doc = #comment])
        }
    }
    pub fn link_name<const MANGLE: bool>(name: &str) -> TokenStream {
        let name: Cow<'_, str> = if MANGLE {
            name.into()
        } else {
            format!("\u{1}{}", name).into()
        };
        quote! {
            #[link_name = #name]
        }
    }
}
pub fn blob(ctx: &Context, layout: Layout) -> TokenStream {
    let opaque = layout.opaque();
    let ty_name = match opaque.known_rust_type_for_array(ctx) {
        Some(ty) => ty,
        None => {
            warn!("Found unknown alignment on code generation!");
            "u8"
        },
    };
    let ty_name = Ident::new(ty_name, Span::call_site());
    let data_len = opaque.array_size(ctx).unwrap_or(layout.size);
    if data_len == 1 {
        quote! {
            #ty_name
        }
    } else {
        quote! {
            [ #ty_name ; #data_len ]
        }
    }
}
pub fn integer_type(ctx: &Context, layout: Layout) -> Option<TokenStream> {
    let name = Layout::known_type_for_size(ctx, layout.size)?;
    let name = Ident::new(name, Span::call_site());
    Some(quote! { #name })
}
pub fn bitfield_unit(ctx: &Context, layout: Layout) -> TokenStream {
    let mut tokens = quote! {};
    if ctx.opts().enable_cxx_namespaces {
        tokens.append_all(quote! { root:: });
    }
    let size = layout.size;
    tokens.append_all(quote! {
        __BitfieldUnit<[u8; #size]>
    });
    tokens
}
pub mod ast_ty {
    use crate::ir::function::FnSig;
    use crate::ir::layout::Layout;
    use crate::ir::typ::FloatKind;
    use crate::ir::Context;
    use proc_macro2::{self, TokenStream};
    use std::str::FromStr;
    pub fn c_void(ctx: &Context) -> TokenStream {
        match ctx.opts().ctypes_prefix {
            Some(ref prefix) => {
                let prefix = TokenStream::from_str(prefix.as_str()).unwrap();
                quote! {
                    #prefix::c_void
                }
            },
            None => {
                quote! { ::core::ffi::c_void }
            },
        }
    }
    pub fn raw_type(ctx: &Context, name: &str) -> TokenStream {
        let ident = ctx.rust_ident_raw(name);
        match ctx.opts().ctypes_prefix {
            Some(ref prefix) => {
                let prefix = TokenStream::from_str(prefix.as_str()).unwrap();
                quote! {
                    #prefix::#ident
                }
            },
            None => {
                if ctx.opts().use_core {
                    quote! {
                        ::core::ffi::#ident
                    }
                } else {
                    quote! {
                        ::std::os::raw::#ident
                    }
                }
            },
        }
    }
    pub fn float_kind_rust_type(ctx: &Context, fk: FloatKind, layout: Option<Layout>) -> TokenStream {
        match (fk, ctx.opts().convert_floats) {
            (FloatKind::Float, true) => quote! { f32 },
            (FloatKind::Double, true) => quote! { f64 },
            (FloatKind::Float, false) => raw_type(ctx, "c_float"),
            (FloatKind::Double, false) => raw_type(ctx, "c_double"),
            (FloatKind::LongDouble, _) => match layout {
                Some(layout) => match layout.size {
                    4 => quote! { f32 },
                    8 => quote! { f64 },
                    _ => super::integer_type(ctx, layout).unwrap_or(quote! { f64 }),
                },
                None => {
                    debug_assert!(false, "How didn't we know the layout for a primitive type?");
                    quote! { f64 }
                },
            },
            (FloatKind::Float128, _) => {
                quote! { u128 }
            },
        }
    }
    pub fn int_expr(val: i64) -> TokenStream {
        let val = proc_macro2::Literal::i64_unsuffixed(val);
        quote!(#val)
    }
    pub fn uint_expr(val: u64) -> TokenStream {
        let val = proc_macro2::Literal::u64_unsuffixed(val);
        quote!(#val)
    }
    pub fn cstr_expr(mut string: String) -> TokenStream {
        string.push('\0');
        let b = proc_macro2::Literal::byte_string(string.as_bytes());
        quote! {
            #b
        }
    }
    pub fn float_expr(ctx: &Context, f: f64) -> Result<TokenStream, ()> {
        if f.is_finite() {
            let val = proc_macro2::Literal::f64_unsuffixed(f);
            return Ok(quote!(#val));
        }
        let prefix = ctx.trait_prefix();
        if f.is_nan() {
            return Ok(quote! {
                ::#prefix::f64::NAN
            });
        }
        if f.is_infinite() {
            return Ok(if f.is_sign_positive() {
                quote! {
                    ::#prefix::f64::INFINITY
                }
            } else {
                quote! {
                    ::#prefix::f64::NEG_INFINITY
                }
            });
        }
        warn!("Unknown non-finite float number: {:?}", f);
        Err(())
    }
    pub fn arguments_from_signature(signature: &FnSig, ctx: &Context) -> Vec<TokenStream> {
        let mut unnamed_arguments = 0;
        signature
            .arg_types()
            .iter()
            .map(|&(ref name, _ty)| match *name {
                Some(ref name) => {
                    let name = ctx.rust_ident(name);
                    quote! { #name }
                },
                None => {
                    unnamed_arguments += 1;
                    let name = ctx.rust_ident(format!("arg{}", unnamed_arguments));
                    quote! { #name }
                },
            })
            .collect()
    }
}
pub mod variation {
    use std::fmt;

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub enum Enum {
        Rust { non_exhaustive: bool },
        NewType { is_bitfield: bool, is_global: bool },
        Consts,
        ModuleConsts,
    }
    impl Enum {
        pub fn is_rust(&self) -> bool {
            matches!(*self, Enum::Rust { .. })
        }
        pub fn is_const(&self) -> bool {
            matches!(*self, Enum::Consts | Enum::ModuleConsts)
        }
    }
    impl Default for Enum {
        fn default() -> Enum {
            Enum::Consts
        }
    }
    impl fmt::Display for Enum {
        fn fmt(&self, x: &mut fmt::Formatter<'_>) -> fmt::Result {
            let y = match self {
                Self::Rust { non_exhaustive: false } => "rust",
                Self::Rust { non_exhaustive: true } => "rust_non_exhaustive",
                Self::NewType { is_bitfield: true, .. } => "bitfield",
                Self::NewType {
                    is_bitfield: false,
                    is_global,
                } => {
                    if *is_global {
                        "newtype_global"
                    } else {
                        "newtype"
                    }
                },
                Self::Consts => "consts",
                Self::ModuleConsts => "moduleconsts",
            };
            y.fmt(x)
        }
    }
    impl std::str::FromStr for Enum {
        type Err = std::io::Error;
        fn from_str(x: &str) -> Result<Self, Self::Err> {
            match x {
                "rust" => Ok(Enum::Rust { non_exhaustive: false }),
                "rust_non_exhaustive" => Ok(Enum::Rust { non_exhaustive: true }),
                "bitfield" => Ok(Enum::NewType {
                    is_bitfield: true,
                    is_global: false,
                }),
                "consts" => Ok(Enum::Consts),
                "moduleconsts" => Ok(Enum::ModuleConsts),
                "newtype" => Ok(Enum::NewType {
                    is_bitfield: false,
                    is_global: false,
                }),
                "newtype_global" => Ok(Enum::NewType {
                    is_bitfield: false,
                    is_global: true,
                }),
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    concat!(
                        "Got an invalid variation::Enum. Accepted values ",
                        "are 'rust', 'rust_non_exhaustive', 'bitfield', 'consts',",
                        "'moduleconsts', 'newtype' and 'newtype_global'."
                    ),
                )),
            }
        }
    }

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub enum MacroType {
        Signed,
        Unsigned,
    }
    impl fmt::Display for MacroType {
        fn fmt(&self, x: &mut fmt::Formatter<'_>) -> fmt::Result {
            let y = match self {
                Self::Signed => "signed",
                Self::Unsigned => "unsigned",
            };
            y.fmt(x)
        }
    }
    impl Default for MacroType {
        fn default() -> MacroType {
            MacroType::Unsigned
        }
    }
    impl std::str::FromStr for MacroType {
        type Err = std::io::Error;
        fn from_str(x: &str) -> Result<Self, Self::Err> {
            match x {
                "signed" => Ok(MacroType::Signed),
                "unsigned" => Ok(MacroType::Unsigned),
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    concat!(
                        "Got an invalid variation::MacroType. Accepted values ",
                        "are 'signed' and 'unsigned'"
                    ),
                )),
            }
        }
    }

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub enum Alias {
        TypeAlias,
        NewType,
        NewTypeDeref,
    }
    impl fmt::Display for Alias {
        fn fmt(&self, x: &mut fmt::Formatter<'_>) -> fmt::Result {
            let y = match self {
                Self::TypeAlias => "type_alias",
                Self::NewType => "new_type",
                Self::NewTypeDeref => "new_type_deref",
            };
            y.fmt(x)
        }
    }
    impl Default for Alias {
        fn default() -> Alias {
            Alias::TypeAlias
        }
    }
    impl std::str::FromStr for Alias {
        type Err = std::io::Error;
        fn from_str(x: &str) -> Result<Self, Self::Err> {
            match x {
                "type_alias" => Ok(Alias::TypeAlias),
                "new_type" => Ok(Alias::NewType),
                "new_type_deref" => Ok(Alias::NewTypeDeref),
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    concat!(
                        "Got an invalid variation::Alias. Accepted values ",
                        "are 'type_alias', 'new_type', and 'new_type_deref'"
                    ),
                )),
            }
        }
    }

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub enum NonCopyUnion {
        BindgenWrapper,
        ManuallyDrop,
    }
    impl fmt::Display for NonCopyUnion {
        fn fmt(&self, x: &mut fmt::Formatter<'_>) -> fmt::Result {
            let s = match self {
                Self::BindgenWrapper => "bindgen_wrapper",
                Self::ManuallyDrop => "manually_drop",
            };
            s.fmt(x)
        }
    }
    impl Default for NonCopyUnion {
        fn default() -> Self {
            Self::BindgenWrapper
        }
    }
    impl std::str::FromStr for NonCopyUnion {
        type Err = std::io::Error;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s {
                "bindgen_wrapper" => Ok(Self::BindgenWrapper),
                "manually_drop" => Ok(Self::ManuallyDrop),
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    concat!(
                        "Got an invalid variation::NonCopyUnion. Accepted values ",
                        "are 'bindgen_wrapper' and 'manually_drop'"
                    ),
                )),
            }
        }
    }
}

pub fn gen_partialeq_impl(
    ctx: &Context,
    comp: &CompInfo,
    it: &Item,
    ty: &proc_macro2::TokenStream,
) -> Option<proc_macro2::TokenStream> {
    let mut ys = vec![];
    if it.is_opaque(ctx, &()) {
        ys.push(quote! {
            &self._bindgen_opaque_blob[..] == &other._bindgen_opaque_blob[..]
        });
    } else if comp.kind() == CompKind::Union {
        assert!(!ctx.opts().untagged_union);
        ys.push(quote! {
            &self.bindgen_union_field[..] == &other.bindgen_union_field[..]
        });
    } else {
        for x in comp.base_members().iter() {
            if !x.requires_storage(ctx) {
                continue;
            }
            let ty = ctx.resolve_item(x.ty);
            let x = &x.field_name;
            if ty.is_opaque(ctx, &()) {
                let x = ctx.rust_ident(x);
                ys.push(quote! {
                    &self. #x [..] == &other. #x [..]
                });
            } else {
                ys.push(gen_field(ctx, ty, x));
            }
        }
        for f in comp.fields() {
            match *f {
                Field::DataMember(ref x) => {
                    let ty = ctx.resolve_item(x.ty());
                    let x = x.name().unwrap();
                    ys.push(gen_field(ctx, ty, x));
                },
                Field::Bitfields(ref x) => {
                    for x in x.bitfields() {
                        if x.name().is_some() {
                            let x = x.getter();
                            let x = ctx.rust_ident_raw(x);
                            ys.push(quote! {
                                self.#x () == other.#x ()
                            });
                        }
                    }
                },
            }
        }
    }
    Some(quote! {
        fn eq(&self, other: & #ty) -> bool {
            #( #ys )&&*
        }
    })
}
fn gen_field(ctx: &Context, it: &Item, name: &str) -> proc_macro2::TokenStream {
    fn quote_equals(x: proc_macro2::Ident) -> proc_macro2::TokenStream {
        quote! { self.#x == other.#x }
    }
    let y = ctx.rust_ident(name);
    let ty = it.expect_type();
    match *ty.kind() {
        TypeKind::Void
        | TypeKind::NullPtr
        | TypeKind::Int(..)
        | TypeKind::Complex(..)
        | TypeKind::Float(..)
        | TypeKind::Enum(..)
        | TypeKind::TypeParam
        | TypeKind::UnresolvedTypeRef(..)
        | TypeKind::Reference(..)
        | TypeKind::Comp(..)
        | TypeKind::Pointer(_)
        | TypeKind::Function(..)
        | TypeKind::Opaque => quote_equals(y),
        TypeKind::TemplInstantiation(ref x) => {
            if x.is_opaque(ctx, it) {
                quote! {
                    &self. #y [..] == &other. #y [..]
                }
            } else {
                quote_equals(y)
            }
        },
        TypeKind::Array(_, len) => quote_equals(y),
        TypeKind::Vector(_, len) => {
            let self_ids = 0..len;
            let other_ids = 0..len;
            quote! {
                #(self.#self_ids == other.#other_ids &&)* true
            }
        },
        TypeKind::ResolvedTypeRef(x) | TypeKind::TemplAlias(x, _) | TypeKind::Alias(x) | TypeKind::BlockPointer(x) => {
            let it2 = ctx.resolve_item(x);
            gen_field(ctx, it2, name)
        },
    }
}

#[cfg(test)]
#[allow(warnings)]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct __BitfieldUnit<Storage> {
    storage: Storage,
}
#[cfg(test)]
#[allow(warnings)]
impl<Storage> __BitfieldUnit<Storage> {
    #[inline]
    pub const fn new(storage: Storage) -> Self {
        Self { storage }
    }
}
#[cfg(test)]
#[allow(warnings)]
impl<Storage> __BitfieldUnit<Storage>
where
    Storage: AsRef<[u8]> + AsMut<[u8]>,
{
    #[inline]
    pub fn get_bit(&self, index: usize) -> bool {
        debug_assert!(index / 8 < self.storage.as_ref().len());
        let byte_index = index / 8;
        let byte = self.storage.as_ref()[byte_index];
        let bit_index = if cfg!(target_endian = "big") {
            7 - (index % 8)
        } else {
            index % 8
        };
        let mask = 1 << bit_index;
        byte & mask == mask
    }
    #[inline]
    pub fn set_bit(&mut self, index: usize, val: bool) {
        debug_assert!(index / 8 < self.storage.as_ref().len());
        let byte_index = index / 8;
        let byte = &mut self.storage.as_mut()[byte_index];
        let bit_index = if cfg!(target_endian = "big") {
            7 - (index % 8)
        } else {
            index % 8
        };
        let mask = 1 << bit_index;
        if val {
            *byte |= mask;
        } else {
            *byte &= !mask;
        }
    }
    #[inline]
    pub fn get(&self, bit_offset: usize, bit_width: u8) -> u64 {
        debug_assert!(bit_width <= 64);
        debug_assert!(bit_offset / 8 < self.storage.as_ref().len());
        debug_assert!((bit_offset + (bit_width as usize)) / 8 <= self.storage.as_ref().len());
        let mut val = 0;
        for i in 0..(bit_width as usize) {
            if self.get_bit(i + bit_offset) {
                let index = if cfg!(target_endian = "big") {
                    bit_width as usize - 1 - i
                } else {
                    i
                };
                val |= 1 << index;
            }
        }
        val
    }
    #[inline]
    pub fn set(&mut self, bit_offset: usize, bit_width: u8, val: u64) {
        debug_assert!(bit_width <= 64);
        debug_assert!(bit_offset / 8 < self.storage.as_ref().len());
        debug_assert!((bit_offset + (bit_width as usize)) / 8 <= self.storage.as_ref().len());
        for i in 0..(bit_width as usize) {
            let mask = 1 << i;
            let val_bit_is_set = val & mask == mask;
            let index = if cfg!(target_endian = "big") {
                bit_width as usize - 1 - i
            } else {
                i
            };
            self.set_bit(index + bit_offset, val_bit_is_set);
        }
    }
}

//#[cfg(all(test, target_endian = "little"))]
#[test]
fn bitfield_unit_get_bit() {
    let unit = __BitfieldUnit::<[u8; 2]>::new([0b10011101, 0b00011101]);
    let mut bits = vec![];
    for i in 0..16 {
        bits.push(unit.get_bit(i));
    }
    println!();
    println!("bits = {:?}", bits);
    assert_eq!(
        bits,
        &[
            true, false, true, true, true, false, false, true, // 0b00011101
            true, false, true, true, true, false, false, false
        ]
    );
}
#[test]
fn bitfield_unit_set_bit() {
    let mut unit = __BitfieldUnit::<[u8; 2]>::new([0b00000000, 0b00000000]);
    for i in 0..16 {
        if i % 3 == 0 {
            unit.set_bit(i, true);
        }
    }
    for i in 0..16 {
        assert_eq!(unit.get_bit(i), i % 3 == 0);
    }
    let mut unit = __BitfieldUnit::<[u8; 2]>::new([0b11111111, 0b11111111]);
    for i in 0..16 {
        if i % 3 == 0 {
            unit.set_bit(i, false);
        }
    }
    for i in 0..16 {
        assert_eq!(unit.get_bit(i), i % 3 != 0);
    }
}

macro_rules! bitfield_unit_get {
    (
        $(
            With $storage:expr , then get($start:expr, $len:expr) is $expected:expr;
        )*
    ) => {
        #[test]
        fn bitfield_unit_get() {
            $({
                let expected = $expected;
                let unit = __BitfieldUnit::<_>::new($storage);
                let actual = unit.get($start, $len);
                println!();
                println!("expected = {:064b}", expected);
                println!("actual   = {:064b}", actual);
                assert_eq!(expected, actual);
            })*
        }
   }
}
bitfield_unit_get! {
    With [0b11100010], then get(0, 1) is 0;
    With [0b11100010], then get(1, 1) is 1;
    With [0b11100010], then get(2, 1) is 0;
    With [0b11100010], then get(3, 1) is 0;
    With [0b11100010], then get(4, 1) is 0;
    With [0b11100010], then get(5, 1) is 1;
    With [0b11100010], then get(6, 1) is 1;
    With [0b11100010], then get(7, 1) is 1;
    With [0b11100010], then get(0, 2) is 0b10;
    With [0b11100010], then get(1, 2) is 0b01;
    With [0b11100010], then get(2, 2) is 0b00;
    With [0b11100010], then get(3, 2) is 0b00;
    With [0b11100010], then get(4, 2) is 0b10;
    With [0b11100010], then get(5, 2) is 0b11;
    With [0b11100010], then get(6, 2) is 0b11;
    With [0b11100010], then get(0, 3) is 0b010;
    With [0b11100010], then get(1, 3) is 0b001;
    With [0b11100010], then get(2, 3) is 0b000;
    With [0b11100010], then get(3, 3) is 0b100;
    With [0b11100010], then get(4, 3) is 0b110;
    With [0b11100010], then get(5, 3) is 0b111;
    With [0b11100010], then get(0, 4) is 0b0010;
    With [0b11100010], then get(1, 4) is 0b0001;
    With [0b11100010], then get(2, 4) is 0b1000;
    With [0b11100010], then get(3, 4) is 0b1100;
    With [0b11100010], then get(4, 4) is 0b1110;
    With [0b11100010], then get(0, 5) is 0b00010;
    With [0b11100010], then get(1, 5) is 0b10001;
    With [0b11100010], then get(2, 5) is 0b11000;
    With [0b11100010], then get(3, 5) is 0b11100;
    With [0b11100010], then get(0, 6) is 0b100010;
    With [0b11100010], then get(1, 6) is 0b110001;
    With [0b11100010], then get(2, 6) is 0b111000;
    With [0b11100010], then get(0, 7) is 0b1100010;
    With [0b11100010], then get(1, 7) is 0b1110001;
    With [0b11100010], then get(0, 8) is 0b11100010;

    With [0b01010101, 0b11111111, 0b00000000, 0b11111111],
    then get(0, 16) is 0b1111111101010101;
    With [0b01010101, 0b11111111, 0b00000000, 0b11111111],
    then get(1, 16) is 0b0111111110101010;
    With [0b01010101, 0b11111111, 0b00000000, 0b11111111],
    then get(2, 16) is 0b0011111111010101;
    With [0b01010101, 0b11111111, 0b00000000, 0b11111111],
    then get(3, 16) is 0b0001111111101010;
    With [0b01010101, 0b11111111, 0b00000000, 0b11111111],
    then get(4, 16) is 0b0000111111110101;
    With [0b01010101, 0b11111111, 0b00000000, 0b11111111],
    then get(5, 16) is 0b0000011111111010;
    With [0b01010101, 0b11111111, 0b00000000, 0b11111111],
    then get(6, 16) is 0b0000001111111101;
    With [0b01010101, 0b11111111, 0b00000000, 0b11111111],
    then get(7, 16) is 0b0000000111111110;
    With [0b01010101, 0b11111111, 0b00000000, 0b11111111],
    then get(8, 16) is 0b0000000011111111;
}
macro_rules! bitfield_unit_set {
    (
        $(
            set($start:expr, $len:expr, $val:expr) is $expected:expr;
        )*
    ) => {
        #[test]
        fn bitfield_unit_set() {
            $(
                let mut unit = __BitfieldUnit::<[u8; 4]>::new([0, 0, 0, 0]);
                unit.set($start, $len, $val);
                let actual = unit.get(0, 32);
                println!();
                println!("set({}, {}, {:032b}", $start, $len, $val);
                println!("expected = {:064b}", $expected);
                println!("actual   = {:064b}", actual);
                assert_eq!($expected, actual);
            )*
        }
    }
}
bitfield_unit_set! {
    set(0, 1, 0b11111111) is 0b00000001;
    set(1, 1, 0b11111111) is 0b00000010;
    set(2, 1, 0b11111111) is 0b00000100;
    set(3, 1, 0b11111111) is 0b00001000;
    set(4, 1, 0b11111111) is 0b00010000;
    set(5, 1, 0b11111111) is 0b00100000;
    set(6, 1, 0b11111111) is 0b01000000;
    set(7, 1, 0b11111111) is 0b10000000;
    set(0, 2, 0b11111111) is 0b00000011;
    set(1, 2, 0b11111111) is 0b00000110;
    set(2, 2, 0b11111111) is 0b00001100;
    set(3, 2, 0b11111111) is 0b00011000;
    set(4, 2, 0b11111111) is 0b00110000;
    set(5, 2, 0b11111111) is 0b01100000;
    set(6, 2, 0b11111111) is 0b11000000;
    set(0, 3, 0b11111111) is 0b00000111;
    set(1, 3, 0b11111111) is 0b00001110;
    set(2, 3, 0b11111111) is 0b00011100;
    set(3, 3, 0b11111111) is 0b00111000;
    set(4, 3, 0b11111111) is 0b01110000;
    set(5, 3, 0b11111111) is 0b11100000;
    set(0, 4, 0b11111111) is 0b00001111;
    set(1, 4, 0b11111111) is 0b00011110;
    set(2, 4, 0b11111111) is 0b00111100;
    set(3, 4, 0b11111111) is 0b01111000;
    set(4, 4, 0b11111111) is 0b11110000;
    set(0, 5, 0b11111111) is 0b00011111;
    set(1, 5, 0b11111111) is 0b00111110;
    set(2, 5, 0b11111111) is 0b01111100;
    set(3, 5, 0b11111111) is 0b11111000;
    set(0, 6, 0b11111111) is 0b00111111;
    set(1, 6, 0b11111111) is 0b01111110;
    set(2, 6, 0b11111111) is 0b11111100;
    set(0, 7, 0b11111111) is 0b01111111;
    set(1, 7, 0b11111111) is 0b11111110;
    set(0, 8, 0b11111111) is 0b11111111;

    set(0, 16, 0b1111111111111111) is 0b00000000000000001111111111111111;
    set(1, 16, 0b1111111111111111) is 0b00000000000000011111111111111110;
    set(2, 16, 0b1111111111111111) is 0b00000000000000111111111111111100;
    set(3, 16, 0b1111111111111111) is 0b00000000000001111111111111111000;
    set(4, 16, 0b1111111111111111) is 0b00000000000011111111111111110000;
    set(5, 16, 0b1111111111111111) is 0b00000000000111111111111111100000;
    set(6, 16, 0b1111111111111111) is 0b00000000001111111111111111000000;
    set(7, 16, 0b1111111111111111) is 0b00000000011111111111111110000000;
    set(8, 16, 0b1111111111111111) is 0b00000000111111111111111100000000;
}