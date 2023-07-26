#![allow(
    clippy::assertions_on_result_states,
    clippy::manual_let_else,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]
#[macro_use]
mod macros;
use quote::quote;
use syn::{Data, Input};
#[test]
fn test_unit() {
    let input = quote! {
        struct Unit;
    };
    snapshot!(input as DeriveInput, @r###"
    DeriveInput {
        vis: data::Visibility::Inherited,
        ident: "Unit",
        gens: gen::Gens,
        data: Data::Struct {
            fields: Fields::Unit,
            semi: Some,
        },
    }
    "###);
}
#[test]
fn test_struct() {
    let input = quote! {
        #[derive(Debug, Clone)]
        pub struct Item {
            pub ident: Ident,
            pub attrs: Vec<attr::Attr>
        }
    };
    snapshot!(input as DeriveInput, @r###"
    DeriveInput {
        attrs: [
            attr::Attr {
                style: attr::Style::Outer,
                meta: meta::Meta::List {
                    path: Path {
                        segments: [
                            path::Segment {
                                ident: "derive",
                            },
                        ],
                    },
                    delimiter: MacroDelimiter::Paren,
                    tokens: pm2::Stream(`Debug , Clone`),
                },
            },
        ],
        vis: data::Visibility::Public,
        ident: "Item",
        gens: gen::Gens,
        data: Data::Struct {
            fields: Fields::Named {
                fields: [
                    Field {
                        vis: data::Visibility::Public,
                        ident: Some("ident"),
                        colon: Some,
                        typ: Type::Path {
                            path: Path {
                                segments: [
                                    path::Segment {
                                        ident: "Ident",
                                    },
                                ],
                            },
                        },
                    },
                    Field {
                        vis: data::Visibility::Public,
                        ident: Some("attrs"),
                        colon: Some,
                        typ: Type::Path {
                            path: Path {
                                segments: [
                                    path::Segment {
                                        ident: "Vec",
                                        arguments: path::Args::AngleBracketed {
                                            args: [
                                                path::Arg::Type(Type::Path {
                                                    path: Path {
                                                        segments: [
                                                            path::Segment {
                                                                ident: "attr::Attr",
                                                            },
                                                        ],
                                                    },
                                                }),
                                            ],
                                        },
                                    },
                                ],
                            },
                        },
                    },
                ],
            },
        },
    }
    "###);
    snapshot!(&input.attrs[0].meta, @r###"
    meta::Meta::List {
        path: Path {
            segments: [
                path::Segment {
                    ident: "derive",
                },
            ],
        },
        delimiter: MacroDelimiter::Paren,
        tokens: pm2::Stream(`Debug , Clone`),
    }
    "###);
}
#[test]
fn test_union() {
    let input = quote! {
        union MaybeUninit<T> {
            uninit: (),
            value: T
        }
    };
    snapshot!(input as DeriveInput, @r###"
    DeriveInput {
        vis: data::Visibility::Inherited,
        ident: "MaybeUninit",
        gens: gen::Gens {
            lt: Some,
            params: [
                gen::Param::Type(gen::param::Type {
                    ident: "T",
                }),
            ],
            gt: Some,
        },
        data: Data::Union {
            fields: Named {
                fields: [
                    Field {
                        vis: data::Visibility::Inherited,
                        ident: Some("uninit"),
                        colon: Some,
                        typ: Type::Tuple,
                    },
                    Field {
                        vis: data::Visibility::Inherited,
                        ident: Some("value"),
                        colon: Some,
                        typ: Type::Path {
                            path: Path {
                                segments: [
                                    path::Segment {
                                        ident: "T",
                                    },
                                ],
                            },
                        },
                    },
                ],
            },
        },
    }
    "###);
}
#[test]
fn test_enum() {
    let input = quote! {
        #[must_use]
        pub enum Result<T, E> {
            Ok(T),
            Err(E),
            Surprise = 0isize,
            ProcMacroHack = (0, "data").0
        }
    };
    snapshot!(input as DeriveInput, @r###"
    DeriveInput {
        attrs: [
            attr::Attr {
                style: attr::Style::Outer,
                meta: meta::Meta::NameValue {
                    path: Path {
                        segments: [
                            path::Segment {
                                ident: "doc",
                            },
                        ],
                    },
                    value: Expr::Lit {
                        lit: " See the std::result module documentation for details.",
                    },
                },
            },
            attr::Attr {
                style: attr::Style::Outer,
                meta: meta::Meta::Path {
                    segments: [
                        path::Segment {
                            ident: "must_use",
                        },
                    ],
                },
            },
        ],
        vis: data::Visibility::Public,
        ident: "Result",
        gens: gen::Gens {
            lt: Some,
            params: [
                gen::Param::Type(gen::param::Type {
                    ident: "T",
                }),
                gen::Param::Type(gen::param::Type {
                    ident: "E",
                }),
            ],
            gt: Some,
        },
        data: Data::Enum {
            variants: [
                Variant {
                    ident: "Ok",
                    fields: Fields::Unnamed {
                        fields: [
                            Field {
                                vis: data::Visibility::Inherited,
                                typ: Type::Path {
                                    path: Path {
                                        segments: [
                                            path::Segment {
                                                ident: "T",
                                            },
                                        ],
                                    },
                                },
                            },
                        ],
                    },
                },
                Variant {
                    ident: "Err",
                    fields: Fields::Unnamed {
                        fields: [
                            Field {
                                vis: data::Visibility::Inherited,
                                typ: Type::Path {
                                    path: Path {
                                        segments: [
                                            path::Segment {
                                                ident: "E",
                                            },
                                        ],
                                    },
                                },
                            },
                        ],
                    },
                },
                Variant {
                    ident: "Surprise",
                    fields: Fields::Unit,
                    discriminant: Some(Expr::Lit {
                        lit: 0isize,
                    }),
                },
                Variant {
                    ident: "ProcMacroHack",
                    fields: Fields::Unit,
                    discriminant: Some(Expr::Field {
                        base: Expr::Tuple {
                            elems: [
                                Expr::Lit {
                                    lit: 0,
                                },
                                Expr::Lit {
                                    lit: "data",
                                },
                            ],
                        },
                        member: Member::Unnamed(Index {
                            index: 0,
                        }),
                    }),
                },
            ],
        },
    }
    "###);
    let meta_items: Vec<_> = input.attrs.into_iter().map(|attr| attr.meta).collect();
    snapshot!(meta_items, @r###"
    [
        meta::Meta::NameValue {
            path: Path {
                segments: [
                    path::Segment {
                        ident: "doc",
                    },
                ],
            },
            value: Expr::Lit {
                lit: " See the std::result module documentation for details.",
            },
        },
        meta::Meta::Path {
            segments: [
                path::Segment {
                    ident: "must_use",
                },
            ],
        },
    ]
    "###);
}
#[test]
fn test_attr_with_non_mod_style_path() {
    let input = quote! {
        #[inert <T>]
        struct S;
    };
    syn::parse2::<DeriveInput>(input).unwrap_err();
}
#[test]
fn test_attr_with_mod_style_path_with_self() {
    let input = quote! {
        #[foo::self]
        struct S;
    };
    snapshot!(input as DeriveInput, @r###"
    DeriveInput {
        attrs: [
            attr::Attr {
                style: attr::Style::Outer,
                meta: meta::Meta::Path {
                    segments: [
                        path::Segment {
                            ident: "foo",
                        },
                        path::Segment {
                            ident: "self",
                        },
                    ],
                },
            },
        ],
        vis: data::Visibility::Inherited,
        ident: "S",
        gens: gen::Gens,
        data: Data::Struct {
            fields: Fields::Unit,
            semi: Some,
        },
    }
    "###);
    snapshot!(&input.attrs[0].meta, @r###"
    meta::Meta::Path {
        segments: [
            path::Segment {
                ident: "foo",
            },
            path::Segment {
                ident: "self",
            },
        ],
    }
    "###);
}
#[test]
fn test_pub_restricted() {
    let input = quote! {
        pub(in m) struct Z(pub(in m::n) u8);
    };
    snapshot!(input as DeriveInput, @r###"
    DeriveInput {
        vis: data::Visibility::Restricted {
            in_: Some,
            path: Path {
                segments: [
                    path::Segment {
                        ident: "m",
                    },
                ],
            },
        },
        ident: "Z",
        gens: gen::Gens,
        data: Data::Struct {
            fields: Fields::Unnamed {
                fields: [
                    Field {
                        vis: data::Visibility::Restricted {
                            in_: Some,
                            path: Path {
                                segments: [
                                    path::Segment {
                                        ident: "m",
                                    },
                                    path::Segment {
                                        ident: "n",
                                    },
                                ],
                            },
                        },
                        typ: Type::Path {
                            path: Path {
                                segments: [
                                    path::Segment {
                                        ident: "u8",
                                    },
                                ],
                            },
                        },
                    },
                ],
            },
            semi: Some,
        },
    }
    "###);
}
#[test]
fn test_pub_restricted_crate() {
    let input = quote! {
        pub struct S;
    };
    snapshot!(input as DeriveInput, @r###"
    DeriveInput {
        vis: data::Visibility::Restricted {
            path: Path {
                segments: [
                    path::Segment {
                        ident: "crate",
                    },
                ],
            },
        },
        ident: "S",
        gens: gen::Gens,
        data: Data::Struct {
            fields: Fields::Unit,
            semi: Some,
        },
    }
    "###);
}
#[test]
fn test_pub_restricted_super() {
    let input = quote! {
        pub(super) struct S;
    };
    snapshot!(input as DeriveInput, @r###"
    DeriveInput {
        vis: data::Visibility::Restricted {
            path: Path {
                segments: [
                    path::Segment {
                        ident: "super",
                    },
                ],
            },
        },
        ident: "S",
        gens: gen::Gens,
        data: Data::Struct {
            fields: Fields::Unit,
            semi: Some,
        },
    }
    "###);
}
#[test]
fn test_pub_restricted_in_super() {
    let input = quote! {
        pub(in super) struct S;
    };
    snapshot!(input as DeriveInput, @r###"
    DeriveInput {
        vis: data::Visibility::Restricted {
            in_: Some,
            path: Path {
                segments: [
                    path::Segment {
                        ident: "super",
                    },
                ],
            },
        },
        ident: "S",
        gens: gen::Gens,
        data: Data::Struct {
            fields: Fields::Unit,
            semi: Some,
        },
    }
    "###);
}
#[test]
fn test_fields_on_unit_struct() {
    let input = quote! {
        struct S;
    };
    snapshot!(input as DeriveInput, @r###"
    DeriveInput {
        vis: data::Visibility::Inherited,
        ident: "S",
        gens: gen::Gens,
        data: Data::Struct {
            fields: Fields::Unit,
            semi: Some,
        },
    }
    "###);
    let data = match input.data {
        Data::Struct(data) => data,
        _ => panic!("expected a struct"),
    };
    assert_eq!(0, data.fields.iter().count());
}
#[test]
fn test_fields_on_named_struct() {
    let input = quote! {
        struct S {
            foo: i32,
            pub bar: String,
        }
    };
    snapshot!(input as DeriveInput, @r###"
    DeriveInput {
        vis: data::Visibility::Inherited,
        ident: "S",
        gens: gen::Gens,
        data: Data::Struct {
            fields: Named {
                fields: [
                    Field {
                        vis: data::Visibility::Inherited,
                        ident: Some("foo"),
                        colon: Some,
                        typ: Type::Path {
                            path: Path {
                                segments: [
                                    path::Segment {
                                        ident: "i32",
                                    },
                                ],
                            },
                        },
                    },
                    Field {
                        vis: data::Visibility::Public,
                        ident: Some("bar"),
                        colon: Some,
                        typ: Type::Path {
                            path: Path {
                                segments: [
                                    path::Segment {
                                        ident: "String",
                                    },
                                ],
                            },
                        },
                    },
                ],
            },
        },
    }
    "###);
    let data = match input.data {
        Data::Struct(data) => data,
        _ => panic!("expected a struct"),
    };
    snapshot!(data.fields.into_iter().collect::<Vec<_>>(), @r###"
    [
        Field {
            vis: data::Visibility::Inherited,
            ident: Some("foo"),
            colon: Some,
            typ: Type::Path {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "i32",
                        },
                    ],
                },
            },
        },
        Field {
            vis: data::Visibility::Public,
            ident: Some("bar"),
            colon: Some,
            typ: Type::Path {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "String",
                        },
                    ],
                },
            },
        },
    ]
    "###);
}
#[test]
fn test_fields_on_tuple_struct() {
    let input = quote! {
        struct S(i32, pub String);
    };
    snapshot!(input as DeriveInput, @r###"
    DeriveInput {
        vis: data::Visibility::Inherited,
        ident: "S",
        gens: gen::Gens,
        data: Data::Struct {
            fields: Fields::Unnamed {
                fields: [
                    Field {
                        vis: data::Visibility::Inherited,
                        typ: Type::Path {
                            path: Path {
                                segments: [
                                    path::Segment {
                                        ident: "i32",
                                    },
                                ],
                            },
                        },
                    },
                    Field {
                        vis: data::Visibility::Public,
                        typ: Type::Path {
                            path: Path {
                                segments: [
                                    path::Segment {
                                        ident: "String",
                                    },
                                ],
                            },
                        },
                    },
                ],
            },
            semi: Some,
        },
    }
    "###);
    let data = match input.data {
        Data::Struct(data) => data,
        _ => panic!("expected a struct"),
    };
    snapshot!(data.fields.iter().collect::<Vec<_>>(), @r###"
    [
        Field {
            vis: data::Visibility::Inherited,
            typ: Type::Path {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "i32",
                        },
                    ],
                },
            },
        },
        Field {
            vis: data::Visibility::Public,
            typ: Type::Path {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "String",
                        },
                    ],
                },
            },
        },
    ]
    "###);
}
#[test]
fn test_ambiguous_crate() {
    let input = quote! {
        struct S(crate::X);
    };
    snapshot!(input as DeriveInput, @r###"
    DeriveInput {
        vis: data::Visibility::Inherited,
        ident: "S",
        gens: gen::Gens,
        data: Data::Struct {
            fields: Fields::Unnamed {
                fields: [
                    Field {
                        vis: data::Visibility::Inherited,
                        typ: Type::Path {
                            path: Path {
                                segments: [
                                    path::Segment {
                                        ident: "crate",
                                    },
                                    path::Segment {
                                        ident: "X",
                                    },
                                ],
                            },
                        },
                    },
                ],
            },
            semi: Some,
        },
    }
    "###);
}
