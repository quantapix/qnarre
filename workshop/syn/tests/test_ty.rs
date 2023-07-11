#![allow(clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use syn::*;
#[test]
fn test_mut_self() {
    syn::parse_str::<ty::Type>("fn(mut self)").unwrap();
    syn::parse_str::<ty::Type>("fn(mut self,)").unwrap();
    syn::parse_str::<ty::Type>("fn(mut self: ())").unwrap();
    syn::parse_str::<ty::Type>("fn(mut self: ...)").unwrap_err();
    syn::parse_str::<ty::Type>("fn(mut self: mut self)").unwrap_err();
    syn::parse_str::<ty::Type>("fn(mut self::T)").unwrap_err();
}
#[test]
fn test_macro_variable_type() {
    let tokens = pm2::Stream::from_iter(vec![
        pm2::Tree::Group(Group::new(pm2::Delim::None, quote! { ty })),
        pm2::Tree::Punct(Punct::new('<', pm2::Spacing::Alone)),
        pm2::Tree::Ident(Ident::new("T", pm2::Span::call_site())),
        pm2::Tree::Punct(Punct::new('>', pm2::Spacing::Alone)),
    ]);
    snapshot!(tokens as ty::Type, @r###"
    Type::Path {
        path: Path {
            segments: [
                path::Segment {
                    ident: "ty",
                    arguments: path::Args::AngleBracketed {
                        args: [
                            path::Arg::Type(Type::Path {
                                path: Path {
                                    segments: [
                                        path::Segment {
                                            ident: "T",
                                        },
                                    ],
                                },
                            }),
                        ],
                    },
                },
            ],
        },
    }
    "###);
    let tokens = pm2::Stream::from_iter(vec![
        pm2::Tree::Group(Group::new(pm2::Delim::None, quote! { ty })),
        pm2::Tree::Punct(Punct::new(':', pm2::Spacing::Joint)),
        pm2::Tree::Punct(Punct::new(':', pm2::Spacing::Alone)),
        pm2::Tree::Punct(Punct::new('<', pm2::Spacing::Alone)),
        pm2::Tree::Ident(Ident::new("T", pm2::Span::call_site())),
        pm2::Tree::Punct(Punct::new('>', pm2::Spacing::Alone)),
    ]);
    snapshot!(tokens as ty::Type, @r###"
    Type::Path {
        path: Path {
            segments: [
                path::Segment {
                    ident: "ty",
                    arguments: path::Args::AngleBracketed {
                        colon2: Some,
                        args: [
                            path::Arg::Type(Type::Path {
                                path: Path {
                                    segments: [
                                        path::Segment {
                                            ident: "T",
                                        },
                                    ],
                                },
                            }),
                        ],
                    },
                },
            ],
        },
    }
    "###);
}
#[test]
fn test_group_angle_brackets() {
    let tokens = pm2::Stream::from_iter(vec![
        pm2::Tree::Ident(Ident::new("Option", pm2::Span::call_site())),
        pm2::Tree::Punct(Punct::new('<', pm2::Spacing::Alone)),
        pm2::Tree::Group(Group::new(pm2::Delim::None, quote! { Vec<u8> })),
        pm2::Tree::Punct(Punct::new('>', pm2::Spacing::Alone)),
    ]);
    snapshot!(tokens as ty::Type, @r###"
    Type::Path {
        path: Path {
            segments: [
                path::Segment {
                    ident: "Option",
                    arguments: path::Args::AngleBracketed {
                        args: [
                            path::Arg::Type(Type::Group {
                                elem: Type::Path {
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
                                                                        ident: "u8",
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
                            }),
                        ],
                    },
                },
            ],
        },
    }
    "###);
}
#[test]
fn test_group_colons() {
    let tokens = pm2::Stream::from_iter(vec![
        pm2::Tree::Group(Group::new(pm2::Delim::None, quote! { Vec<u8> })),
        pm2::Tree::Punct(Punct::new(':', pm2::Spacing::Joint)),
        pm2::Tree::Punct(Punct::new(':', pm2::Spacing::Alone)),
        pm2::Tree::Ident(Ident::new("Item", pm2::Span::call_site())),
    ]);
    snapshot!(tokens as ty::Type, @r###"
    Type::Path {
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
                                            ident: "u8",
                                        },
                                    ],
                                },
                            }),
                        ],
                    },
                },
                path::Segment {
                    ident: "Item",
                },
            ],
        },
    }
    "###);
    let tokens = pm2::Stream::from_iter(vec![
        pm2::Tree::Group(Group::new(pm2::Delim::None, quote! { [T] })),
        pm2::Tree::Punct(Punct::new(':', pm2::Spacing::Joint)),
        pm2::Tree::Punct(Punct::new(':', pm2::Spacing::Alone)),
        pm2::Tree::Ident(Ident::new("Element", pm2::Span::call_site())),
    ]);
    snapshot!(tokens as ty::Type, @r###"
    Type::Path {
        qself: Some(QSelf {
            ty: Type::Slice {
                elem: Type::Path {
                    path: Path {
                        segments: [
                            path::Segment {
                                ident: "T",
                            },
                        ],
                    },
                },
            },
            position: 0,
        }),
        path: Path {
            leading_colon: Some,
            segments: [
                path::Segment {
                    ident: "Element",
                },
            ],
        },
    }
    "###);
}
#[test]
fn test_trait_object() {
    let tokens = quote!(dyn for<'a> Trait<'a> + 'static);
    snapshot!(tokens as ty::Type, @r###"
    Type::TraitObject {
        dyn_: Some,
        bounds: [
            gen::bound::Type::Trait(gen::bound::Trait {
                lifetimes: Some(Bgen::bound::Lifes {
                    lifetimes: [
                        gen::Param::Life(gen::param::Life {
                            lifetime: Lifetime {
                                ident: "a",
                            },
                        }),
                    ],
                }),
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "Trait",
                            arguments: path::Args::AngleBracketed {
                                args: [
                                    path::Arg::Lifetime(Lifetime {
                                        ident: "a",
                                    }),
                                ],
                            },
                        },
                    ],
                },
            }),
            gen::bound::Type::Lifetime {
                ident: "static",
            },
        ],
    }
    "###);
    let tokens = quote!(dyn 'a + Trait);
    snapshot!(tokens as ty::Type, @r###"
    Type::TraitObject {
        dyn_: Some,
        bounds: [
            gen::bound::Type::Lifetime {
                ident: "a",
            },
            gen::bound::Type::Trait(gen::bound::Trait {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "Trait",
                        },
                    ],
                },
            }),
        ],
    }
    "###);
    syn::parse_str::<ty::Type>("for<'a> dyn Trait<'a>").unwrap_err();
    syn::parse_str::<ty::Type>("dyn for<'a> 'a + Trait").unwrap_err();
}
#[test]
fn test_trailing_plus() {
    #[rustfmt::skip]
    let tokens = quote!(impl Trait +);
    snapshot!(tokens as ty::Type, @r###"
    Type::ImplTrait {
        bounds: [
            gen::bound::Type::Trait(gen::bound::Trait {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "Trait",
                        },
                    ],
                },
            }),
        ],
    }
    "###);
    #[rustfmt::skip]
    let tokens = quote!(dyn Trait +);
    snapshot!(tokens as ty::Type, @r###"
    Type::TraitObject {
        dyn_: Some,
        bounds: [
            gen::bound::Type::Trait(gen::bound::Trait {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "Trait",
                        },
                    ],
                },
            }),
        ],
    }
    "###);
    #[rustfmt::skip]
    let tokens = quote!(Trait +);
    snapshot!(tokens as ty::Type, @r###"
    Type::TraitObject {
        bounds: [
            gen::bound::Type::Trait(gen::bound::Trait {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "Trait",
                        },
                    ],
                },
            }),
        ],
    }
    "###);
}
