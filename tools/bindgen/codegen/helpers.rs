use crate::ir::context::Context;
use crate::ir::layout::Layout;
use proc_macro2::{Ident, Span, TokenStream};
use quote::TokenStreamExt;

pub(crate) mod attributes {
    use proc_macro2::{Ident, Span, TokenStream};
    use std::{borrow::Cow, str::FromStr};

    pub(crate) fn repr(which: &str) -> TokenStream {
        let which = Ident::new(which, Span::call_site());
        quote! {
            #[repr( #which )]
        }
    }

    pub(crate) fn repr_list(which_ones: &[&str]) -> TokenStream {
        let which_ones = which_ones
            .iter()
            .cloned()
            .map(|one| TokenStream::from_str(one).expect("repr to be valid"));
        quote! {
            #[repr( #( #which_ones ),* )]
        }
    }

    pub(crate) fn derives(which_ones: &[&str]) -> TokenStream {
        let which_ones = which_ones
            .iter()
            .cloned()
            .map(|one| TokenStream::from_str(one).expect("derive to be valid"));
        quote! {
            #[derive( #( #which_ones ),* )]
        }
    }

    pub(crate) fn inline() -> TokenStream {
        quote! {
            #[inline]
        }
    }

    pub(crate) fn must_use() -> TokenStream {
        quote! {
            #[must_use]
        }
    }

    pub(crate) fn non_exhaustive() -> TokenStream {
        quote! {
            #[non_exhaustive]
        }
    }

    pub(crate) fn doc(comment: String) -> TokenStream {
        if comment.is_empty() {
            quote!()
        } else {
            quote!(#[doc = #comment])
        }
    }

    pub(crate) fn link_name<const MANGLE: bool>(name: &str) -> TokenStream {
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

pub(crate) fn blob(ctx: &Context, layout: Layout) -> TokenStream {
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

pub(crate) fn integer_type(ctx: &Context, layout: Layout) -> Option<TokenStream> {
    let name = Layout::known_type_for_size(ctx, layout.size)?;
    let name = Ident::new(name, Span::call_site());
    Some(quote! { #name })
}

pub(crate) fn bitfield_unit(ctx: &Context, layout: Layout) -> TokenStream {
    let mut tokens = quote! {};

    if ctx.opts().enable_cxx_namespaces {
        tokens.append_all(quote! { root:: });
    }

    let size = layout.size;
    tokens.append_all(quote! {
        __BindgenBitfieldUnit<[u8; #size]>
    });

    tokens
}

pub(crate) mod ast_ty {
    use crate::ir::context::Context;
    use crate::ir::function::FnSig;
    use crate::ir::layout::Layout;
    use crate::ir::ty::FloatKind;
    use proc_macro2::{self, TokenStream};
    use std::str::FromStr;

    pub(crate) fn c_void(ctx: &Context) -> TokenStream {
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

    pub(crate) fn raw_type(ctx: &Context, name: &str) -> TokenStream {
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

    pub(crate) fn float_kind_rust_type(ctx: &Context, fk: FloatKind, layout: Option<Layout>) -> TokenStream {
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

    pub(crate) fn int_expr(val: i64) -> TokenStream {
        let val = proc_macro2::Literal::i64_unsuffixed(val);
        quote!(#val)
    }

    pub(crate) fn uint_expr(val: u64) -> TokenStream {
        let val = proc_macro2::Literal::u64_unsuffixed(val);
        quote!(#val)
    }

    pub(crate) fn cstr_expr(mut string: String) -> TokenStream {
        string.push('\0');
        let b = proc_macro2::Literal::byte_string(string.as_bytes());
        quote! {
            #b
        }
    }

    pub(crate) fn float_expr(ctx: &Context, f: f64) -> Result<TokenStream, ()> {
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

    pub(crate) fn arguments_from_signature(signature: &FnSig, ctx: &Context) -> Vec<TokenStream> {
        let mut unnamed_arguments = 0;
        signature
            .argument_types()
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
