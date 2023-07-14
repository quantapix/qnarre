#![allow(clippy::manual_let_else, clippy::too_many_lines, clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use quote::quote;
use syn::{gen::bound::Type, gen::Where, gen::Where::Pred, item::Fn, DeriveInput};
#[test]
fn test_split_for_impl() {
    let input = quote! {
        struct S<'a, 'b: 'a, #[may_dangle] T: 'a = ()> where T: Debug;
    };
    snapshot!(input as DeriveInput, @r###"
    DeriveInput {
        vis: data::Visibility::Inherited,
        ident: "S",
        gens: gen::Gens {
            lt: Some,
            params: [
                gen::Param::Life(gen::param::Life {
                    life: Life {
                        ident: "a",
                    },
                }),
                gen::Param::Life(gen::param::Life {
                    life: Life {
                        ident: "b",
                    },
                    colon: Some,
                    bounds: [
                        Life {
                            ident: "a",
                        },
                    ],
                }),
                gen::Param::Type(gen::param::Type {
                    attrs: [
                        attr::Attr {
                            style: attr::Style::Outer,
                            meta: meta::Meta::Path {
                                segments: [
                                    path::Segment {
                                        ident: "may_dangle",
                                    },
                                ],
                            },
                        },
                    ],
                    ident: "T",
                    colon: Some,
                    bounds: [
                        gen::bound::Type::Life {
                            ident: "a",
                        },
                    ],
                    eq: Some,
                    default: Some(Type::Tuple),
                }),
            ],
            gt: Some,
            where_clause: Some(gen::Where {
                predicates: [
                    WherePredicate::Type(PredicateType {
                        bounded: Type::Path {
                            path: Path {
                                segments: [
                                    path::Segment {
                                        ident: "T",
                                    },
                                ],
                            },
                        },
                        bounds: [
                            gen::bound::Type::Trait(gen::bound::Trait {
                                path: Path {
                                    segments: [
                                        path::Segment {
                                            ident: "Debug",
                                        },
                                    ],
                                },
                            }),
                        ],
                    }),
                ],
            }),
        },
        data: Data::Struct {
            fields: Fields::Unit,
            semi: Some,
        },
    }
    "###);
    let gens = input.gens;
    let (impl_generics, ty_generics, where_clause) = gens.split_for_impl();
    let generated = quote! {
        impl #impl_generics MyTrait for Test #ty_generics #where_clause {}
    };
    let expected = quote! {
        impl<'a, 'b: 'a, #[may_dangle] T: 'a> MyTrait
        for Test<'a, 'b, T>
        where
            T: Debug
        {}
    };
    assert_eq!(generated.to_string(), expected.to_string());
    let turbofish = ty_generics.as_turbofish();
    let generated = quote! {
        Test #turbofish
    };
    let expected = quote! {
        Test::<'a, 'b, T>
    };
    assert_eq!(generated.to_string(), expected.to_string());
}
#[test]
fn test_ty_param_bound() {
    let tokens = quote!('a);
    snapshot!(tokens as gen::bound::Type, @r###"
    gen::bound::Type::Life {
        ident: "a",
    }
    "###);
    let tokens = quote!('_);
    snapshot!(tokens as gen::bound::Type, @r###"
    gen::bound::Type::Life {
        ident: "_",
    }
    "###);
    let tokens = quote!(Debug);
    snapshot!(tokens as gen::bound::Type, @r###"
    gen::bound::Type::Trait(gen::bound::Trait {
        path: Path {
            segments: [
                path::Segment {
                    ident: "Debug",
                },
            ],
        },
    })
    "###);
    let tokens = quote!(?Sized);
    snapshot!(tokens as gen::bound::Type, @r###"
    gen::bound::Type::Trait(gen::bound::Trait {
        modifier: gen::bound::Modifier::Maybe,
        path: Path {
            segments: [
                path::Segment {
                    ident: "Sized",
                },
            ],
        },
    })
    "###);
}
#[test]
fn test_fn_precedence_in_where_clause() {
    let input = quote! {
        fn f<G>()
        where
            G: FnOnce() -> i32 + Send,
        {
        }
    };
    snapshot!(input as item::Fn, @r###"
    item::Fn {
        vis: data::Visibility::Inherited,
        sig: item::Sig {
            ident: "f",
            gens: gen::Gens {
                lt: Some,
                params: [
                    gen::Param::Type(gen::param::Type {
                        ident: "G",
                    }),
                ],
                gt: Some,
                where_clause: Some(gen::Where {
                    predicates: [
                        WherePredicate::Type(PredicateType {
                            bounded: Type::Path {
                                path: Path {
                                    segments: [
                                        path::Segment {
                                            ident: "G",
                                        },
                                    ],
                                },
                            },
                            bounds: [
                                gen::bound::Type::Trait(gen::bound::Trait {
                                    path: Path {
                                        segments: [
                                            path::Segment {
                                                ident: "FnOnce",
                                                arguments: path::Args::Parenthesized {
                                                    ret: typ::Ret::Type(
                                                        Type::Path {
                                                            path: Path {
                                                                segments: [
                                                                    path::Segment {
                                                                        ident: "i32",
                                                                    },
                                                                ],
                                                            },
                                                        },
                                                    ),
                                                },
                                            },
                                        ],
                                    },
                                }),
                                gen::bound::Type::Trait(gen::bound::Trait {
                                    path: Path {
                                        segments: [
                                            path::Segment {
                                                ident: "Send",
                                            },
                                        ],
                                    },
                                }),
                            ],
                        }),
                    ],
                }),
            },
            ret: typ::Ret::Default,
        },
        block: Block,
    }
    "###);
    let where_clause = input.sig.gens.where_clause.as_ref().unwrap();
    assert_eq!(where_clause.predicates.len(), 1);
    let predicate = match &where_clause.predicates[0] {
        gen::Where::Pred::Type(pred) => pred,
        _ => panic!("wrong predicate kind"),
    };
    assert_eq!(predicate.bounds.len(), 2, "{:#?}", predicate.bounds);
    let first_bound = &predicate.bounds[0];
    assert_eq!(quote!(#first_bound).to_string(), "FnOnce () -> i32");
    let second_bound = &predicate.bounds[1];
    assert_eq!(quote!(#second_bound).to_string(), "Send");
}
#[test]
fn test_where_clause_at_end_of_input() {
    let input = quote! {
        where
    };
    snapshot!(input as gen::Where, @"gen::Where");
    assert_eq!(input.predicates.len(), 0);
}
