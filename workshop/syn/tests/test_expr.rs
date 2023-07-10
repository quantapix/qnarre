#![allow(clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use proc_macro2::{Delimiter, Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};
use quote::quote;
use syn::{expr::Range, Expr};
#[test]
fn test_expr_parse() {
    let tokens = quote!(..100u32);
    snapshot!(tokens as Expr, @r###"
    Expr::Range {
        limits: RangeLimits::HalfOpen,
        end: Some(Expr::Lit {
            lit: 100u32,
        }),
    }
    "###);
    let tokens = quote!(..100u32);
    snapshot!(tokens as expr::Range, @r###"
    expr::Range {
        limits: RangeLimits::HalfOpen,
        end: Some(Expr::Lit {
            lit: 100u32,
        }),
    }
    "###);
}
#[test]
fn test_await() {
    let tokens = quote!(fut.await);
    snapshot!(tokens as Expr, @r###"
    Expr::Await {
        base: Expr::Path {
            path: Path {
                segments: [
                    path::Segment {
                        ident: "fut",
                    },
                ],
            },
        },
    }
    "###);
}
#[rustfmt::skip]
#[test]
fn test_tuple_multi_index() {
    let expected = snapshot!("tuple.0.0" as Expr, @r###"
    Expr::Field {
        base: Expr::Field {
            base: Expr::Path {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "tuple",
                        },
                    ],
                },
            },
            member: Member::Unnamed(Index {
                index: 0,
            }),
        },
        member: Member::Unnamed(Index {
            index: 0,
        }),
    }
    "###);
    for &input in &[
        "tuple .0.0",
        "tuple. 0.0",
        "tuple.0 .0",
        "tuple.0. 0",
        "tuple . 0 . 0",
    ] {
        assert_eq!(expected, syn::parse_str(input).unwrap());
    }
    for tokens in [
        quote!(tuple.0.0),
        quote!(tuple .0.0),
        quote!(tuple. 0.0),
        quote!(tuple.0 .0),
        quote!(tuple.0. 0),
        quote!(tuple . 0 . 0),
    ] {
        assert_eq!(expected, syn::parse2(tokens).unwrap());
    }
}
#[test]
fn test_macro_variable_func() {
    let tokens = TokenStream::from_iter(vec![
        TokenTree::Group(Group::new(Delimiter::None, quote! { f })),
        TokenTree::Group(Group::new(Delimiter::Parenthesis, TokenStream::new())),
    ]);
    snapshot!(tokens as Expr, @r###"
    Expr::Call {
        func: Expr::Group {
            expr: Expr::Path {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "f",
                        },
                    ],
                },
            },
        },
    }
    "###);
    let tokens = TokenStream::from_iter(vec![
        TokenTree::Punct(Punct::new('#', Spacing::Alone)),
        TokenTree::Group(Group::new(Delimiter::Bracket, quote! { outside })),
        TokenTree::Group(Group::new(Delimiter::None, quote! { #[inside] f })),
        TokenTree::Group(Group::new(Delimiter::Parenthesis, TokenStream::new())),
    ]);
    snapshot!(tokens as Expr, @r###"
    Expr::Call {
        attrs: [
            attr::Attr {
                style: attr::Style::Outer,
                meta: meta::Meta::Path {
                    segments: [
                        path::Segment {
                            ident: "outside",
                        },
                    ],
                },
            },
        ],
        func: Expr::Group {
            expr: Expr::Path {
                attrs: [
                    attr::Attr {
                        style: attr::Style::Outer,
                        meta: meta::Meta::Path {
                            segments: [
                                path::Segment {
                                    ident: "inside",
                                },
                            ],
                        },
                    },
                ],
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "f",
                        },
                    ],
                },
            },
        },
    }
    "###);
}
#[test]
fn test_macro_variable_macro() {
    let tokens = TokenStream::from_iter(vec![
        TokenTree::Group(Group::new(Delimiter::None, quote! { m })),
        TokenTree::Punct(Punct::new('!', Spacing::Alone)),
        TokenTree::Group(Group::new(Delimiter::Parenthesis, TokenStream::new())),
    ]);
    snapshot!(tokens as Expr, @r###"
    Expr::Macro {
        mac: Macro {
            path: Path {
                segments: [
                    path::Segment {
                        ident: "m",
                    },
                ],
            },
            delimiter: MacroDelimiter::Paren,
            tokens: TokenStream(``),
        },
    }
    "###);
}
#[test]
fn test_macro_variable_struct() {
    let tokens = TokenStream::from_iter(vec![
        TokenTree::Group(Group::new(Delimiter::None, quote! { S })),
        TokenTree::Group(Group::new(Delimiter::Brace, TokenStream::new())),
    ]);
    snapshot!(tokens as Expr, @r###"
    Expr::Struct {
        path: Path {
            segments: [
                path::Segment {
                    ident: "S",
                },
            ],
        },
    }
    "###);
}
#[test]
fn test_macro_variable_match_arm() {
    let tokens = TokenStream::from_iter(vec![
        TokenTree::Ident(Ident::new("match", Span::call_site())),
        TokenTree::Ident(Ident::new("v", Span::call_site())),
        TokenTree::Group(Group::new(
            Delimiter::Brace,
            TokenStream::from_iter(vec![
                TokenTree::Punct(Punct::new('_', Spacing::Alone)),
                TokenTree::Punct(Punct::new('=', Spacing::Joint)),
                TokenTree::Punct(Punct::new('>', Spacing::Alone)),
                TokenTree::Group(Group::new(Delimiter::None, quote! { #[a] () })),
            ]),
        )),
    ]);
    snapshot!(tokens as Expr, @r###"
    Expr::Match {
        expr: Expr::Path {
            path: Path {
                segments: [
                    path::Segment {
                        ident: "v",
                    },
                ],
            },
        },
        arms: [
            Arm {
                pat: patt::Patt::Wild,
                body: Expr::Group {
                    expr: Expr::Tuple {
                        attrs: [
                            attr::Attr {
                                style: attr::Style::Outer,
                                meta: meta::Meta::Path {
                                    segments: [
                                        path::Segment {
                                            ident: "a",
                                        },
                                    ],
                                },
                            },
                        ],
                    },
                },
            },
        ],
    }
    "###);
}
#[test]
fn test_closure_vs_rangefull() {
    #[rustfmt::skip] // rustfmt bug: https://github.com/rust-lang/rustfmt/issues/4808
    let tokens = quote!(|| .. .method());
    snapshot!(tokens as Expr, @r###"
    Expr::MethodCall {
        receiver: Expr::Closure {
            ret: ty::Ret::Default,
            body: Expr::Range {
                limits: RangeLimits::HalfOpen,
            },
        },
        method: "method",
    }
    "###);
}
#[test]
fn test_postfix_operator_after_cast() {
    syn::parse_str::<Expr>("|| &x as T[0]").unwrap_err();
    syn::parse_str::<Expr>("|| () as ()()").unwrap_err();
}
#[test]
fn test_ranges() {
    syn::parse_str::<Expr>("..").unwrap();
    syn::parse_str::<Expr>("..hi").unwrap();
    syn::parse_str::<Expr>("lo..").unwrap();
    syn::parse_str::<Expr>("lo..hi").unwrap();
    syn::parse_str::<Expr>("..=").unwrap_err();
    syn::parse_str::<Expr>("..=hi").unwrap();
    syn::parse_str::<Expr>("lo..=").unwrap_err();
    syn::parse_str::<Expr>("lo..=hi").unwrap();
    syn::parse_str::<Expr>("...").unwrap_err();
    syn::parse_str::<Expr>("...hi").unwrap_err();
    syn::parse_str::<Expr>("lo...").unwrap_err();
    syn::parse_str::<Expr>("lo...hi").unwrap_err();
}