#![allow(clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use proc_macro2::{Delimiter, Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};
use quote::quote;
use syn::ty::Type;
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
    let tokens = TokenStream::from_iter(vec![
        TokenTree::Group(Group::new(Delimiter::None, quote! { ty })),
        TokenTree::Punct(Punct::new('<', Spacing::Alone)),
        TokenTree::Ident(Ident::new("T", Span::call_site())),
        TokenTree::Punct(Punct::new('>', Spacing::Alone)),
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
    let tokens = TokenStream::from_iter(vec![
        TokenTree::Group(Group::new(Delimiter::None, quote! { ty })),
        TokenTree::Punct(Punct::new(':', Spacing::Joint)),
        TokenTree::Punct(Punct::new(':', Spacing::Alone)),
        TokenTree::Punct(Punct::new('<', Spacing::Alone)),
        TokenTree::Ident(Ident::new("T", Span::call_site())),
        TokenTree::Punct(Punct::new('>', Spacing::Alone)),
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
    let tokens = TokenStream::from_iter(vec![
        TokenTree::Ident(Ident::new("Option", Span::call_site())),
        TokenTree::Punct(Punct::new('<', Spacing::Alone)),
        TokenTree::Group(Group::new(Delimiter::None, quote! { Vec<u8> })),
        TokenTree::Punct(Punct::new('>', Spacing::Alone)),
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
    let tokens = TokenStream::from_iter(vec![
        TokenTree::Group(Group::new(Delimiter::None, quote! { Vec<u8> })),
        TokenTree::Punct(Punct::new(':', Spacing::Joint)),
        TokenTree::Punct(Punct::new(':', Spacing::Alone)),
        TokenTree::Ident(Ident::new("Item", Span::call_site())),
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
    let tokens = TokenStream::from_iter(vec![
        TokenTree::Group(Group::new(Delimiter::None, quote! { [T] })),
        TokenTree::Punct(Punct::new(':', Spacing::Joint)),
        TokenTree::Punct(Punct::new(':', Spacing::Alone)),
        TokenTree::Ident(Ident::new("Element", Span::call_site())),
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
            TypeParamBound::Trait(TraitBound {
                lifetimes: Some(BoundLifetimes {
                    lifetimes: [
                        GenericParam::Lifetime(LifetimeParam {
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
            TypeParamBound::Lifetime {
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
            TypeParamBound::Lifetime {
                ident: "a",
            },
            TypeParamBound::Trait(TraitBound {
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
            TypeParamBound::Trait(TraitBound {
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
            TypeParamBound::Trait(TraitBound {
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
            TypeParamBound::Trait(TraitBound {
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
