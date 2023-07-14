#![allow(
    clippy::assertions_on_result_states,
    clippy::non_ascii_literal,
    clippy::uninlined_format_args
)]
#[macro_use]
mod macros;
use syn::*;
#[test]
fn test_raw_operator() {
    let stmt = syn::parse::parse_str::<stmt::Stmt>("let _ = &raw const x;").unwrap();
    snapshot!(stmt, @r###"
    stmt::Stmt::stmt::Local {
        pat: pat::Pat::Wild,
        init: Some(stmt::Init {
            expr: Expr::Stream(`& raw const x`),
        }),
    }
    "###);
}
#[test]
fn test_raw_variable() {
    let stmt = syn::parse::parse_str::<stmt::Stmt>("let _ = &raw;").unwrap();
    snapshot!(stmt, @r###"
    stmt::Stmt::stmt::Local {
        pat: pat::Pat::Wild,
        init: Some(stmt::Init {
            expr: Expr::Reference {
                expr: Expr::Path {
                    path: Path {
                        segments: [
                            path::Segment {
                                ident: "raw",
                            },
                        ],
                    },
                },
            },
        }),
    }
    "###);
}
#[test]
fn test_raw_invalid() {
    assert!(syn::parse::parse_str::<stmt::Stmt>("let _ = &raw x;").is_err());
}
#[test]
fn test_none_group() {
    let tokens = pm2::Stream::from_iter(vec![pm2::Tree::Group(Group::new(
        pm2::Delim::None,
        pm2::Stream::from_iter(vec![
            pm2::Tree::Ident(Ident::new("async", pm2::Span::call_site())),
            pm2::Tree::Ident(Ident::new("fn", pm2::Span::call_site())),
            pm2::Tree::Ident(Ident::new("f", pm2::Span::call_site())),
            pm2::Tree::Group(Group::new(pm2::Delim::Parenthesis, pm2::Stream::new())),
            pm2::Tree::Group(Group::new(pm2::Delim::Brace, pm2::Stream::new())),
        ]),
    ))]);
    snapshot!(tokens as stmt::Stmt, @r###"
    stmt::Stmt::Item(Item::Fn {
        vis: data::Visibility::Inherited,
        sig: item::Sig {
            asyncness: Some,
            ident: "f",
            gens: gen::Gens,
            ret: typ::Ret::Default,
        },
        block: Block,
    })
    "###);
}
#[test]
fn test_let_dot_dot() {
    let tokens = quote! {
        let .. = 10;
    };
    snapshot!(tokens as stmt::Stmt, @r###"
    stmt::Stmt::stmt::Local {
        pat: pat::Pat::Rest,
        init: Some(stmt::Init {
            expr: Expr::Lit {
                lit: 10,
            },
        }),
    }
    "###);
}
#[test]
fn test_let_else() {
    let tokens = quote! {
        let Some(x) = None else { return 0; };
    };
    snapshot!(tokens as stmt::Stmt, @r###"
    stmt::Stmt::stmt::Local {
        pat: pat::Pat::TupleStruct {
            path: Path {
                segments: [
                    path::Segment {
                        ident: "Some",
                    },
                ],
            },
            elems: [
                pat::Pat::Ident {
                    ident: "x",
                },
            ],
        },
        init: Some(stmt::Init {
            expr: Expr::Path {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "None",
                        },
                    ],
                },
            },
            diverge: Some(Expr::Block {
                block: Block {
                    stmts: [
                        stmt::Stmt::Expr(
                            Expr::Return {
                                expr: Some(Expr::Lit {
                                    lit: 0,
                                }),
                            },
                            Some,
                        ),
                    ],
                },
            }),
        }),
    }
    "###);
}
#[test]
fn test_macros() {
    let tokens = quote! {
        fn main() {
            macro_rules! mac {}
            thread_local! { static FOO }
            println!("");
            vec![]
        }
    };
    snapshot!(tokens as stmt::Stmt, @r###"
    stmt::Stmt::Item(Item::Fn {
        vis: data::Visibility::Inherited,
        sig: item::Sig {
            ident: "main",
            gens: gen::Gens,
            ret: typ::Ret::Default,
        },
        block: Block {
            stmts: [
                stmt::Stmt::Item(Item::Macro {
                    ident: Some("mac"),
                    mac: Macro {
                        path: Path {
                            segments: [
                                path::Segment {
                                    ident: "macro_rules",
                                },
                            ],
                        },
                        delimiter: MacroDelimiter::Brace,
                        tokens: pm2::Stream(``),
                    },
                }),
                stmt::Stmt::Macro {
                    mac: Macro {
                        path: Path {
                            segments: [
                                path::Segment {
                                    ident: "thread_local",
                                },
                            ],
                        },
                        delimiter: MacroDelimiter::Brace,
                        tokens: pm2::Stream(`static FOO`),
                    },
                },
                stmt::Stmt::Macro {
                    mac: Macro {
                        path: Path {
                            segments: [
                                path::Segment {
                                    ident: "println",
                                },
                            ],
                        },
                        delimiter: MacroDelimiter::Paren,
                        tokens: pm2::Stream(`""`),
                    },
                    semi: Some,
                },
                stmt::Stmt::Expr(
                    Expr::Macro {
                        mac: Macro {
                            path: Path {
                                segments: [
                                    path::Segment {
                                        ident: "vec",
                                    },
                                ],
                            },
                            delimiter: MacroDelimiter::Bracket,
                            tokens: pm2::Stream(``),
                        },
                    },
                    None,
                ),
            ],
        },
    })
    "###);
}
