#![feature(allow_internal_unstable)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(never_type)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_span)]
#![allow(rustc::default_hash_types)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
#![recursion_limit = "128"]

use proc_macro::TokenStream;
use synstructure::decl_derive;
mod diag;
mod hash_stable {
    use proc_macro2::{self, Ident};
    use quote::quote;
    use syn::{self, parse_quote};
    struct Attributes {
        ignore: bool,
        project: Option<Ident>,
    }
    fn parse_attributes(field: &syn::Field) -> Attributes {
        let mut attrs = Attributes {
            ignore: false,
            project: None,
        };
        for attr in &field.attrs {
            let meta = &attr.meta;
            if !meta.path().is_ident("stable_hasher") {
                continue;
            }
            let mut any_attr = false;
            let _ = attr.parse_nested_meta(|nested| {
                if nested.path.is_ident("ignore") {
                    attrs.ignore = true;
                    any_attr = true;
                }
                if nested.path.is_ident("project") {
                    let _ = nested.parse_nested_meta(|meta| {
                        if attrs.project.is_none() {
                            attrs.project = meta.path.get_ident().cloned();
                        }
                        any_attr = true;
                        Ok(())
                    });
                }
                Ok(())
            });
            if !any_attr {
                panic!("error parsing stable_hasher");
            }
        }
        attrs
    }
    pub fn hash_stable_generic_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
        let generic: syn::GenericParam = parse_quote!(__CTX);
        s.add_bounds(synstructure::AddBounds::Generics);
        s.add_impl_generic(generic);
        s.add_where_predicate(parse_quote! { __CTX: crate::HashStableContext });
        let body = s.each(|bi| {
            let attrs = parse_attributes(bi.ast());
            if attrs.ignore {
                quote! {}
            } else if let Some(project) = attrs.project {
                quote! {
                    (&#bi.#project).hash_stable(__hcx, __hasher);
                }
            } else {
                quote! {
                    #bi.hash_stable(__hcx, __hasher);
                }
            }
        });
        let discriminant = match s.ast().data {
            syn::Data::Enum(_) => quote! {
                ::std::mem::discriminant(self).hash_stable(__hcx, __hasher);
            },
            syn::Data::Struct(_) => quote! {},
            syn::Data::Union(_) => panic!("cannot derive on union"),
        };
        s.bound_impl(
            quote!(::rustc_data_structures::stable_hasher::HashStable<__CTX>),
            quote! {
                #[inline]
                fn hash_stable(
                    &self,
                    __hcx: &mut __CTX,
                    __hasher: &mut ::rustc_data_structures::stable_hasher::StableHasher) {
                    #discriminant
                    match *self { #body }
                }
            },
        )
    }
    pub fn hash_stable_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
        let generic: syn::GenericParam = parse_quote!('__ctx);
        s.add_bounds(synstructure::AddBounds::Generics);
        s.add_impl_generic(generic);
        let body = s.each(|bi| {
            let attrs = parse_attributes(bi.ast());
            if attrs.ignore {
                quote! {}
            } else if let Some(project) = attrs.project {
                quote! {
                    (&#bi.#project).hash_stable(__hcx, __hasher);
                }
            } else {
                quote! {
                    #bi.hash_stable(__hcx, __hasher);
                }
            }
        });
        let discriminant = match s.ast().data {
            syn::Data::Enum(_) => quote! {
                ::std::mem::discriminant(self).hash_stable(__hcx, __hasher);
            },
            syn::Data::Struct(_) => quote! {},
            syn::Data::Union(_) => panic!("cannot derive on union"),
        };
        s.bound_impl(
            quote!(
                ::rustc_data_structures::stable_hasher::HashStable<
                    ::rustc_query_system::ich::StableHashingContext<'__ctx>,
                >
            ),
            quote! {
                #[inline]
                fn hash_stable(
                    &self,
                    __hcx: &mut ::rustc_query_system::ich::StableHashingContext<'__ctx>,
                    __hasher: &mut ::rustc_data_structures::stable_hasher::StableHasher) {
                    #discriminant
                    match *self { #body }
                }
            },
        )
    }
}
mod lift {
    use quote::quote;
    use syn::{self, parse_quote};
    pub fn lift_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
        s.add_bounds(synstructure::AddBounds::Generics);
        s.bind_with(|_| synstructure::BindStyle::Move);
        let tcx: syn::Lifetime = parse_quote!('tcx);
        let newtcx: syn::GenericParam = parse_quote!('__lifted);
        let lifted = {
            let ast = s.ast();
            let ident = &ast.ident;
            // Replace `'tcx` lifetime by the `'__lifted` lifetime
            let (_, generics, _) = ast.generics.split_for_impl();
            let mut generics: syn::AngleBracketedGenericArguments = syn::parse_quote! { #generics };
            for arg in generics.args.iter_mut() {
                match arg {
                    syn::GenericArgument::Lifetime(l) if *l == tcx => {
                        *arg = parse_quote!('__lifted);
                    },
                    syn::GenericArgument::Type(t) => {
                        *arg = syn::parse_quote! { #t::Lifted };
                    },
                    _ => {},
                }
            }
            quote! { #ident #generics }
        };
        let body = s.each_variant(|vi| {
            let bindings = &vi.bindings();
            vi.construct(|_, index| {
                let bi = &bindings[index];
                quote! { __tcx.lift(#bi)?  }
            })
        });
        s.add_impl_generic(newtcx);
        s.bound_impl(
            quote!(::rustc_middle::ty::Lift<'__lifted>),
            quote! {
                type Lifted = #lifted;
                fn lift_to_tcx(self, __tcx: ::rustc_middle::ty::TyCtxt<'__lifted>) -> Option<#lifted> {
                    Some(match self { #body })
                }
            },
        )
    }
}
mod newtype;
mod query;
mod serialize;
mod symbols;
mod type_foldable {
    use quote::{quote, ToTokens};
    use syn::parse_quote;
    pub fn type_foldable_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
        if let syn::Data::Union(_) = s.ast().data {
            panic!("cannot derive on union")
        }
        if !s.ast().generics.lifetimes().any(|lt| lt.lifetime.ident == "tcx") {
            s.add_impl_generic(parse_quote! { 'tcx });
        }
        s.add_bounds(synstructure::AddBounds::Generics);
        s.bind_with(|_| synstructure::BindStyle::Move);
        let body_fold = s.each_variant(|vi| {
            let bindings = vi.bindings();
            vi.construct(|_, index| {
                let bind = &bindings[index];
                let mut fixed = false;
                bind.ast().attrs.iter().for_each(|x| {
                    if !x.path().is_ident("type_foldable") {
                        return;
                    }
                    let _ = x.parse_nested_meta(|nested| {
                        if nested.path.is_ident("identity") {
                            fixed = true;
                        }
                        Ok(())
                    });
                });
                if fixed {
                    bind.to_token_stream()
                } else {
                    quote! {
                        ::rustc_middle::ty::fold::TypeFoldable::try_fold_with(#bind, __folder)?
                    }
                }
            })
        });
        s.bound_impl(
            quote!(::rustc_middle::ty::fold::TypeFoldable<::rustc_middle::ty::TyCtxt<'tcx>>),
            quote! {
                fn try_fold_with<__F: ::rustc_middle::ty::fold::FallibleTypeFolder<::rustc_middle::ty::TyCtxt<'tcx>>>(
                    self,
                    __folder: &mut __F
                ) -> Result<Self, __F::Error> {
                    Ok(match self { #body_fold })
                }
            },
        )
    }
}
mod type_visitable {
    use quote::quote;
    use syn::parse_quote;
    pub fn type_visitable_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
        if let syn::Data::Union(_) = s.ast().data {
            panic!("cannot derive on union")
        }
        s.filter(|bi| {
            let mut ignored = false;
            bi.ast().attrs.iter().for_each(|attr| {
                if !attr.path().is_ident("type_visitable") {
                    return;
                }
                let _ = attr.parse_nested_meta(|nested| {
                    if nested.path.is_ident("ignore") {
                        ignored = true;
                    }
                    Ok(())
                });
            });
            !ignored
        });
        if !s.ast().generics.lifetimes().any(|lt| lt.lifetime.ident == "tcx") {
            s.add_impl_generic(parse_quote! { 'tcx });
        }
        s.add_bounds(synstructure::AddBounds::Generics);
        let body_visit = s.each(|bind| {
            quote! {
                ::rustc_middle::ty::visit::TypeVisitable::visit_with(#bind, __visitor)?;
            }
        });
        s.bind_with(|_| synstructure::BindStyle::Move);
        s.bound_impl(
            quote!(::rustc_middle::ty::visit::TypeVisitable<::rustc_middle::ty::TyCtxt<'tcx>>),
            quote! {
                fn visit_with<__V: ::rustc_middle::ty::visit::TypeVisitor<::rustc_middle::ty::TyCtxt<'tcx>>>(
                    &self,
                    __visitor: &mut __V
                ) -> ::std::ops::ControlFlow<__V::BreakTy> {
                    match *self { #body_visit }
                    ::std::ops::ControlFlow::Continue(())
                }
            },
        )
    }
}
#[proc_macro]
pub fn rustc_queries(input: TokenStream) -> TokenStream {
    query::rustc_queries(input)
}
#[proc_macro]
pub fn symbols(input: TokenStream) -> TokenStream {
    symbols::symbols(input.into()).into()
}
/// Creates a struct type `S` that can be used as an index with
/// `IndexVec` and so on.
///
/// There are two ways of interacting with these indices:
///
/// - The `From` impls are the preferred way. So you can do
///   `S::from(v)` with a `usize` or `u32`. And you can convert back
///   to an integer with `u32::from(s)`.
///
/// - Alternatively, you can use the methods `S::new(v)` and `s.index()`
///   to create/return a value.
///
/// Internally, the index uses a u32, so the index must not exceed
/// `u32::MAX`. You can also customize things like the `Debug` impl,
/// what traits are derived, and so forth via the macro.
#[proc_macro]
#[allow_internal_unstable(step_trait, rustc_attrs, trusted_step, spec_option_partial_eq)]
pub fn newtype_index(input: TokenStream) -> TokenStream {
    newtype::newtype(input)
}
decl_derive!([HashStable, attributes(stable_hasher)] => hash_stable::hash_stable_derive);
decl_derive!(
    [HashStable_Generic, attributes(stable_hasher)] =>
    hash_stable::hash_stable_generic_derive
);
decl_derive!([Decodable] => serialize::decodable_derive);
decl_derive!([Encodable] => serialize::encodable_derive);
decl_derive!([TyDecodable] => serialize::type_decodable_derive);
decl_derive!([TyEncodable] => serialize::type_encodable_derive);
decl_derive!([MetadataDecodable] => serialize::meta_decodable_derive);
decl_derive!([MetadataEncodable] => serialize::meta_encodable_derive);
decl_derive!(
    [TypeFoldable, attributes(type_foldable)] =>
    /// Derives `TypeFoldable` for the annotated `struct` or `enum` (`union` is not supported).
    ///
    /// The fold will produce a value of the same struct or enum variant as the input, with
    /// each field respectively folded using the `TypeFoldable` implementation for its type.
    /// However, if a field of a struct or an enum variant is annotated with
    /// `#[type_foldable(identity)]` then that field will retain its incumbent value (and its
    /// type is not required to implement `TypeFoldable`).
    type_foldable::type_foldable_derive
);
decl_derive!(
    [TypeVisitable, attributes(type_visitable)] =>
    /// Derives `TypeVisitable` for the annotated `struct` or `enum` (`union` is not supported).
    ///
    /// Each field of the struct or enum variant will be visited in definition order, using the
    /// `TypeVisitable` implementation for its type. However, if a field of a struct or an enum
    /// variant is annotated with `#[type_visitable(ignore)]` then that field will not be
    /// visited (and its type is not required to implement `TypeVisitable`).
    type_visitable::type_visitable_derive
);
decl_derive!([Lift, attributes(lift)] => lift::lift_derive);
decl_derive!(
    [Diagnostic, attributes(
        // struct attributes
        diag,
        help,
        note,
        warning,
        // field attributes
        skip_arg,
        primary_span,
        label,
        subdiagnostic,
        suggestion,
        suggestion_short,
        suggestion_hidden,
        suggestion_verbose)] => diag::session_diagnostic_derive
);
decl_derive!(
    [LintDiagnostic, attributes(
        // struct attributes
        diag,
        help,
        note,
        warning,
        // field attributes
        skip_arg,
        primary_span,
        label,
        subdiagnostic,
        suggestion,
        suggestion_short,
        suggestion_hidden,
        suggestion_verbose)] => diag::lint_diagnostic_derive
);
decl_derive!(
    [Subdiagnostic, attributes(
        // struct/variant attributes
        label,
        help,
        note,
        warning,
        suggestion,
        suggestion_short,
        suggestion_hidden,
        suggestion_verbose,
        multipart_suggestion,
        multipart_suggestion_short,
        multipart_suggestion_hidden,
        multipart_suggestion_verbose,
        // field attributes
        skip_arg,
        primary_span,
        suggestion_part,
        applicability)] => diag::session_subdiagnostic_derive
);
