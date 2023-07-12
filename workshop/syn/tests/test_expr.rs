#![allow(clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use syn::*;
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
    let tokens = pm2::Stream::from_iter(vec![
        pm2::Tree::Group(Group::new(pm2::Delim::None, quote! { f })),
        pm2::Tree::Group(Group::new(pm2::Delim::Parenthesis, pm2::Stream::new())),
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
    let tokens = pm2::Stream::from_iter(vec![
        pm2::Tree::Punct(Punct::new('#', pm2::Spacing::Alone)),
        pm2::Tree::Group(Group::new(pm2::Delim::Bracket, quote! { outside })),
        pm2::Tree::Group(Group::new(pm2::Delim::None, quote! { #[inside] f })),
        pm2::Tree::Group(Group::new(pm2::Delim::Parenthesis, pm2::Stream::new())),
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
    let tokens = pm2::Stream::from_iter(vec![
        pm2::Tree::Group(Group::new(pm2::Delim::None, quote! { m })),
        pm2::Tree::Punct(Punct::new('!', pm2::Spacing::Alone)),
        pm2::Tree::Group(Group::new(pm2::Delim::Parenthesis, pm2::Stream::new())),
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
            tokens: pm2::Stream(``),
        },
    }
    "###);
}
#[test]
fn test_macro_variable_struct() {
    let tokens = pm2::Stream::from_iter(vec![
        pm2::Tree::Group(Group::new(pm2::Delim::None, quote! { S })),
        pm2::Tree::Group(Group::new(pm2::Delim::Brace, pm2::Stream::new())),
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
    let tokens = pm2::Stream::from_iter(vec![
        pm2::Tree::Ident(Ident::new("match", pm2::Span::call_site())),
        pm2::Tree::Ident(Ident::new("v", pm2::Span::call_site())),
        pm2::Tree::Group(Group::new(
            pm2::Delim::Brace,
            pm2::Stream::from_iter(vec![
                pm2::Tree::Punct(Punct::new('_', pm2::Spacing::Alone)),
                pm2::Tree::Punct(Punct::new('=', pm2::Spacing::Joint)),
                pm2::Tree::Punct(Punct::new('>', pm2::Spacing::Alone)),
                pm2::Tree::Group(Group::new(pm2::Delim::None, quote! { #[a] () })),
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
            ret: typ::Ret::Default,
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
