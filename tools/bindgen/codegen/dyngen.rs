use crate::codegen;
use crate::ir::context::BindgenContext;
use crate::ir::function::ClangAbi;
use proc_macro2::Ident;

#[derive(Default)]
pub(crate) struct DynItems {
    struct_members: Vec<proc_macro2::TokenStream>,
    struct_implementation: Vec<proc_macro2::TokenStream>,
    constructor_inits: Vec<proc_macro2::TokenStream>,
    init_fields: Vec<proc_macro2::TokenStream>,
}

impl DynItems {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn get_tokens(&self, lib_ident: Ident, ctx: &BindgenContext) -> proc_macro2::TokenStream {
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
    pub(crate) fn push(
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
        ctx: &BindgenContext,
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

        let ident_str = codegen::helpers::ast_ty::cstr_expr(ident.to_string());
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
