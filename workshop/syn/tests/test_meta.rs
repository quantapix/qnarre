#![allow(clippy::shadow_unrelated, clippy::too_many_lines, clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use syn::{meta::List, meta::Meta, meta::NameValue};
#[test]
fn test_parse_meta_item_word() {
    let input = "hello";
    snapshot!(input as meta::Meta, @r###"
    meta::Meta::Path {
        segments: [
            path::Segment {
                ident: "hello",
            },
        ],
    }
    "###);
}
#[test]
fn test_parse_meta_name_value() {
    let input = "foo = 5";
    let (inner, meta) = (input, input);
    snapshot!(inner as meta::NameValue, @r###"
    meta::NameValue {
        path: Path {
            segments: [
                path::Segment {
                    ident: "foo",
                },
            ],
        },
        value: Expr::Lit {
            lit: 5,
        },
    }
    "###);
    snapshot!(meta as meta::Meta, @r###"
    meta::Meta::NameValue {
        path: Path {
            segments: [
                path::Segment {
                    ident: "foo",
                },
            ],
        },
        value: Expr::Lit {
            lit: 5,
        },
    }
    "###);
    assert_eq!(meta, inner.into());
}
#[test]
fn test_parse_meta_item_list_lit() {
    let input = "foo(5)";
    let (inner, meta) = (input, input);
    snapshot!(inner as meta::List, @r###"
    meta::List {
        path: Path {
            segments: [
                path::Segment {
                    ident: "foo",
                },
            ],
        },
        delimiter: MacroDelimiter::Paren,
        tokens: pm2::Stream(`5`),
    }
    "###);
    snapshot!(meta as meta::Meta, @r###"
    meta::Meta::List {
        path: Path {
            segments: [
                path::Segment {
                    ident: "foo",
                },
            ],
        },
        delimiter: MacroDelimiter::Paren,
        tokens: pm2::Stream(`5`),
    }
    "###);
    assert_eq!(meta, inner.into());
}
#[test]
fn test_parse_meta_item_multiple() {
    let input = "foo(word, name = 5, list(name2 = 6), word2)";
    let (inner, meta) = (input, input);
    snapshot!(inner as meta::List, @r###"
    meta::List {
        path: Path {
            segments: [
                path::Segment {
                    ident: "foo",
                },
            ],
        },
        delimiter: MacroDelimiter::Paren,
        tokens: pm2::Stream(`word , name = 5 , list (name2 = 6) , word2`),
    }
    "###);
    snapshot!(meta as meta::Meta, @r###"
    meta::Meta::List {
        path: Path {
            segments: [
                path::Segment {
                    ident: "foo",
                },
            ],
        },
        delimiter: MacroDelimiter::Paren,
        tokens: pm2::Stream(`word , name = 5 , list (name2 = 6) , word2`),
    }
    "###);
    assert_eq!(meta, inner.into());
}
#[test]
fn test_parse_path() {
    let input = "::serde::Serialize";
    snapshot!(input as meta::Meta, @r###"
    meta::Meta::Path {
        leading_colon: Some,
        segments: [
            path::Segment {
                ident: "serde",
            },
            path::Segment {
                ident: "Serialize",
            },
        ],
    }
    "###);
}
