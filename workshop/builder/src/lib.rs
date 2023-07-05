use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::{parse_macro_input, parse_quote, spanned::Spanned, Data, DeriveInput, Fields, GenericParam, Generics, Index};

#[proc_macro_derive(Builder)]
pub fn derive(x: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let x = parse_macro_input!(x as DeriveInput);
    let n = x.ident;
    let y = quote! {
        impl #n {
            pub fn builder() {}
        }
    };
    proc_macro::TokenStream::from(y)
}

#[proc_macro_derive(HeapSize)]
pub fn derive_heap_size(x: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let x = parse_macro_input!(x as DeriveInput);
    let n = x.ident;
    let g = add_trait_bounds(x.generics);
    let (impl_g, ty_g, where_clause) = g.split_for_impl();
    let s = heap_size_sum(&x.data);
    let y = quote! {
        impl #impl_g utils::HeapSize for #n #ty_g #where_clause {
            fn heap_size_of_children(&self) -> usize {
                #s
            }
        }
    };
    proc_macro::TokenStream::from(y)
}

fn add_trait_bounds(mut y: Generics) -> Generics {
    for x in &mut y.params {
        if let GenericParam::Type(ref mut x) = *x {
            x.bounds.push(parse_quote!(utils::HeapSize));
        }
    }
    y
}

fn heap_size_sum(x: &Data) -> TokenStream {
    match *x {
        Data::Struct(ref x) => match x.fields {
            Fields::Named(ref xs) => {
                let y = xs.named.iter().map(|x| {
                    let n = &x.ident;
                    quote_spanned! {x.span()=>
                        utils::HeapSize::heap_size_of_children(&self.#n)
                    }
                });
                quote! {
                    0 #(+ #y)*
                }
            },
            Fields::Unnamed(ref xs) => {
                let y = xs.unnamed.iter().enumerate().map(|(i, x)| {
                    let i = Index::from(i);
                    quote_spanned! {x.span()=>
                        utils::HeapSize::heap_size_of_children(&self.#i)
                    }
                });
                quote! {
                    0 #(+ #y)*
                }
            },
            Fields::Unit => {
                quote!(0)
            },
        },
        Data::Enum(_) | Data::Union(_) => unimplemented!(),
    }
}
