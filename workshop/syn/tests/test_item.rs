#![allow(clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use syn::*;
#[test]
fn test_macro_variable_attr() {
    let tokens = pm2::Stream::from_iter(vec![
        pm2::Tree::Group(Group::new(pm2::Delim::None, quote! { #[test] })),
        pm2::Tree::Ident(Ident::new("fn", pm2::Span::call_site())),
        pm2::Tree::Ident(Ident::new("f", pm2::Span::call_site())),
        pm2::Tree::Group(Group::new(pm2::Delim::Parenthesis, pm2::Stream::new())),
        pm2::Tree::Group(Group::new(pm2::Delim::Brace, pm2::Stream::new())),
    ]);
    snapshot!(tokens as Item, @r###"
    Item::Fn {
        attrs: [
            attr::Attr {
                style: attr::Style::Outer,
                meta: meta::Meta::Path {
                    segments: [
                        path::Segment {
                            ident: "test",
                        },
                    ],
                },
            },
        ],
        vis: Visibility::Inherited,
        sig: item::Sig {
            ident: "f",
            gens: gen::Gens,
            ret: typ::Ret::Default,
        },
        block: Block,
    }
    "###);
}
#[test]
fn test_negative_impl() {
    #[cfg(any())]
    impl ! {}
    let tokens = quote! {
        impl ! {}
    };
    snapshot!(tokens as Item, @r###"
    Item::Impl {
        gens: gen::Gens,
        self_ty: Type::Never,
    }
    "###);
    #[cfg(any())]
    #[rustfmt::skip]
    impl !Trait {}
    let tokens = quote! {
        impl !Trait {}
    };
    snapshot!(tokens as Item, @r###"
    Item::Impl {
        gens: gen::Gens,
        self_ty: Type::Verbatim(`! Trait`),
    }
    "###);
    #[cfg(any())]
    impl !Trait for T {}
    let tokens = quote! {
        impl !Trait for T {}
    };
    snapshot!(tokens as Item, @r###"
    Item::Impl {
        gens: gen::Gens,
        trait_: Some((
            Some,
            Path {
                segments: [
                    path::Segment {
                        ident: "Trait",
                    },
                ],
            },
        )),
        self_ty: Type::Path {
            path: Path {
                segments: [
                    path::Segment {
                        ident: "T",
                    },
                ],
            },
        },
    }
    "###);
    #[cfg(any())]
    #[rustfmt::skip]
    impl !! {}
    let tokens = quote! {
        impl !! {}
    };
    snapshot!(tokens as Item, @r###"
    Item::Impl {
        gens: gen::Gens,
        self_ty: Type::Verbatim(`! !`),
    }
    "###);
}
#[test]
fn test_macro_variable_impl() {
    let tokens = pm2::Stream::from_iter(vec![
        pm2::Tree::Ident(Ident::new("impl", pm2::Span::call_site())),
        pm2::Tree::Group(Group::new(pm2::Delim::None, quote!(Trait))),
        pm2::Tree::Ident(Ident::new("for", pm2::Span::call_site())),
        pm2::Tree::Group(Group::new(pm2::Delim::None, quote!(Type))),
        pm2::Tree::Group(Group::new(pm2::Delim::Brace, pm2::Stream::new())),
    ]);
    snapshot!(tokens as Item, @r###"
    Item::Impl {
        gens: gen::Gens,
        trait_: Some((
            None,
            Path {
                segments: [
                    path::Segment {
                        ident: "Trait",
                    },
                ],
            },
        )),
        self_ty: Type::Group {
            elem: Type::Path {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "Type",
                        },
                    ],
                },
            },
        },
    }
    "###);
}
#[test]
fn test_supertraits() {
    #[rustfmt::skip]
    let tokens = quote!(trait Trait where {});
    snapshot!(tokens as item::Trait, @r###"
    item::Trait {
        vis: Visibility::Inherited,
        ident: "Trait",
        gens: gen::Gens {
            where_clause: Some(gen::Where),
        },
    }
    "###);
    #[rustfmt::skip]
    let tokens = quote!(trait Trait: where {});
    snapshot!(tokens as item::Trait, @r###"
    item::Trait {
        vis: Visibility::Inherited,
        ident: "Trait",
        gens: gen::Gens {
            where_clause: Some(gen::Where),
        },
        colon: Some,
    }
    "###);
    #[rustfmt::skip]
    let tokens = quote!(trait Trait: Sized where {});
    snapshot!(tokens as item::Trait, @r###"
    item::Trait {
        vis: Visibility::Inherited,
        ident: "Trait",
        gens: gen::Gens {
            where_clause: Some(gen::Where),
        },
        colon: Some,
        supertraits: [
            gen::bound::Type::Trait(gen::bound::Trait {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "Sized",
                        },
                    ],
                },
            }),
        ],
    }
    "###);
    #[rustfmt::skip]
    let tokens = quote!(trait Trait: Sized + where {});
    snapshot!(tokens as item::Trait, @r###"
    item::Trait {
        vis: Visibility::Inherited,
        ident: "Trait",
        gens: gen::Gens {
            where_clause: Some(gen::Where),
        },
        colon: Some,
        supertraits: [
            gen::bound::Type::Trait(gen::bound::Trait {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "Sized",
                        },
                    ],
                },
            }),
        ],
    }
    "###);
}
#[test]
fn test_type_empty_bounds() {
    #[rustfmt::skip]
    let tokens = quote! {
        trait Foo {
            type Bar: ;
        }
    };
    snapshot!(tokens as item::Trait, @r###"
    item::Trait {
        vis: Visibility::Inherited,
        ident: "Foo",
        gens: gen::Gens,
        items: [
            item::Trait::Item::Type {
                ident: "Bar",
                gens: gen::Gens,
                colon: Some,
            },
        ],
    }
    "###);
}
#[test]
fn test_impl_visibility() {
    let tokens = quote! {
        pub default unsafe impl union {}
    };
    snapshot!(tokens as Item, @"Item::Verbatim(`pub default unsafe impl union { }`)");
}
#[test]
fn test_impl_type_parameter_defaults() {
    #[cfg(any())]
    impl<T = ()> () {}
    let tokens = quote! {
        impl<T = ()> () {}
    };
    snapshot!(tokens as Item, @r###"
    Item::Impl {
        gens: gen::Gens {
            lt: Some,
            params: [
                gen::Param::Type(gen::param::Type {
                    ident: "T",
                    eq: Some,
                    default: Some(Type::Tuple),
                }),
            ],
            gt: Some,
        },
        self_ty: Type::Tuple,
    }
    "###);
}
#[test]
fn test_impl_trait_trailing_plus() {
    let tokens = quote! {
        fn f() -> impl Sized + {}
    };
    snapshot!(tokens as Item, @r###"
    Item::Fn {
        vis: Visibility::Inherited,
        sig: item::Sig {
            ident: "f",
            gens: gen::Gens,
            ret: typ::Ret::Type(
                Type::ImplTrait {
                    bounds: [
                        gen::bound::Type::Trait(gen::bound::Trait {
                            path: Path {
                                segments: [
                                    path::Segment {
                                        ident: "Sized",
                                    },
                                ],
                            },
                        }),
                    ],
                },
            ),
        },
        block: Block,
    }
    "###);
}
