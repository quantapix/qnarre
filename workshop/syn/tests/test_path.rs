#![allow(clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use proc_macro2::{Delimiter, Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};
use quote::{quote, ToTokens};
use syn::{parse_quote, ty::Path, Expr, Ty};
#[test]
fn parse_interpolated_leading_component() {
    let tokens = TokenStream::from_iter(vec![
        TokenTree::Group(Group::new(Delimiter::None, quote! { first })),
        TokenTree::Punct(Punct::new(':', Spacing::Joint)),
        TokenTree::Punct(Punct::new(':', Spacing::Alone)),
        TokenTree::Ident(Ident::new("rest", Span::call_site())),
    ]);
    snapshot!(tokens.clone() as Expr, @r###"
    Expr::Path {
        path: Path {
            segments: [
                path::Segment {
                    ident: "first",
                },
                path::Segment {
                    ident: "rest",
                },
            ],
        },
    }
    "###);
    snapshot!(tokens as Ty, @r###"
    Type::Path {
        path: Path {
            segments: [
                path::Segment {
                    ident: "first",
                },
                path::Segment {
                    ident: "rest",
                },
            ],
        },
    }
    "###);
}
#[test]
fn print_incomplete_qpath() {
    let mut ty: ty::Path = parse_quote!(<Self as A>::Q);
    snapshot!(ty.to_token_stream(), @r###"
    TokenStream(`< Self as A > :: Q`)
    "###);
    assert!(ty.path.segs.pop().is_some());
    snapshot!(ty.to_token_stream(), @r###"
    TokenStream(`< Self as A > ::`)
    "###);
    assert!(ty.path.segs.pop().is_some());
    snapshot!(ty.to_token_stream(), @r###"
    TokenStream(`< Self >`)
    "###);
    assert!(ty.path.segs.pop().is_none());
    let mut ty: ty::Path = parse_quote!(<Self>::A::B);
    snapshot!(ty.to_token_stream(), @r###"
    TokenStream(`< Self > :: A :: B`)
    "###);
    assert!(ty.path.segs.pop().is_some());
    snapshot!(ty.to_token_stream(), @r###"
    TokenStream(`< Self > :: A ::`)
    "###);
    assert!(ty.path.segs.pop().is_some());
    snapshot!(ty.to_token_stream(), @r###"
    TokenStream(`< Self > ::`)
    "###);
    assert!(ty.path.segs.pop().is_none());
    let mut ty: ty::Path = parse_quote!(Self::A::B);
    snapshot!(ty.to_token_stream(), @r###"
    TokenStream(`Self :: A :: B`)
    "###);
    assert!(ty.path.segs.pop().is_some());
    snapshot!(ty.to_token_stream(), @r###"
    TokenStream(`Self :: A ::`)
    "###);
    assert!(ty.path.segs.pop().is_some());
    snapshot!(ty.to_token_stream(), @r###"
    TokenStream(`Self ::`)
    "###);
    assert!(ty.path.segs.pop().is_some());
    snapshot!(ty.to_token_stream(), @r###"
    TokenStream(``)
    "###);
    assert!(ty.path.segs.pop().is_none());
}
#[test]
fn parse_parenthesized_path_arguments_with_disambiguator() {
    #[rustfmt::skip]
    let tokens = quote!(dyn FnOnce::() -> !);
    snapshot!(tokens as Ty, @r###"
    Type::TraitObject {
        dyn_: Some,
        bounds: [
            TypeParamBound::Trait(TraitBound {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "FnOnce",
                            arguments: path::Args::Parenthesized {
                                ret: ty::Ret::Type(
                                    Type::Never,
                                ),
                            },
                        },
                    ],
                },
            }),
        ],
    }
    "###);
}
