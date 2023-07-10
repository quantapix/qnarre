#![allow(clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use proc_macro2::{Delimiter, Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};
use quote::{quote, ToTokens};
use syn::{parse_quote, ty::Path, ty::Type, Expr};
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
    snapshot!(tokens as ty::Type, @r###"
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
    let mut typ: ty::Path = parse_quote!(<Self as A>::Q);
    snapshot!(typ.to_token_stream(), @r###"
    TokenStream(`< Self as A > :: Q`)
    "###);
    assert!(typ.path.segs.pop().is_some());
    snapshot!(typ.to_token_stream(), @r###"
    TokenStream(`< Self as A > ::`)
    "###);
    assert!(typ.path.segs.pop().is_some());
    snapshot!(typ.to_token_stream(), @r###"
    TokenStream(`< Self >`)
    "###);
    assert!(typ.path.segs.pop().is_none());
    let mut typ: ty::Path = parse_quote!(<Self>::A::B);
    snapshot!(typ.to_token_stream(), @r###"
    TokenStream(`< Self > :: A :: B`)
    "###);
    assert!(typ.path.segs.pop().is_some());
    snapshot!(typ.to_token_stream(), @r###"
    TokenStream(`< Self > :: A ::`)
    "###);
    assert!(typ.path.segs.pop().is_some());
    snapshot!(typ.to_token_stream(), @r###"
    TokenStream(`< Self > ::`)
    "###);
    assert!(typ.path.segs.pop().is_none());
    let mut typ: ty::Path = parse_quote!(Self::A::B);
    snapshot!(typ.to_token_stream(), @r###"
    TokenStream(`Self :: A :: B`)
    "###);
    assert!(typ.path.segs.pop().is_some());
    snapshot!(typ.to_token_stream(), @r###"
    TokenStream(`Self :: A ::`)
    "###);
    assert!(typ.path.segs.pop().is_some());
    snapshot!(typ.to_token_stream(), @r###"
    TokenStream(`Self ::`)
    "###);
    assert!(typ.path.segs.pop().is_some());
    snapshot!(typ.to_token_stream(), @r###"
    TokenStream(``)
    "###);
    assert!(typ.path.segs.pop().is_none());
}
#[test]
fn parse_parenthesized_path_arguments_with_disambiguator() {
    #[rustfmt::skip]
    let tokens = quote!(dyn FnOnce::() -> !);
    snapshot!(tokens as ty::Type, @r###"
    Type::TraitObject {
        dyn_: Some,
        bounds: [
            gen::bound::Type::Trait(gen::bound::Trait {
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
