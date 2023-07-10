#![allow(clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use syn::{item::Trait::Fn, parse_quote};
#[test]
fn test_by_value() {
    let item::Trait::Fn { sig, .. } = parse_quote! {
        fn by_value(self: Self);
    };
    snapshot!(&sig.args[0], @r###"
    item::FnArg::Receiver(item::Receiver {
        colon: Some,
        ty: Type::Path {
            path: Path {
                segments: [
                    path::Segment {
                        ident: "Self",
                    },
                ],
            },
        },
    })
    "###);
}
#[test]
fn test_by_mut_value() {
    let item::Trait::Fn { sig, .. } = parse_quote! {
        fn by_mut(mut self: Self);
    };
    snapshot!(&sig.args[0], @r###"
    item::FnArg::Receiver(item::Receiver {
        mutability: Some,
        colon: Some,
        ty: Type::Path {
            path: Path {
                segments: [
                    path::Segment {
                        ident: "Self",
                    },
                ],
            },
        },
    })
    "###);
}
#[test]
fn test_by_ref() {
    let item::Trait::Fn { sig, .. } = parse_quote! {
        fn by_ref(self: &Self);
    };
    snapshot!(&sig.args[0], @r###"
    item::FnArg::Receiver(item::Receiver {
        colon: Some,
        ty: Type::Reference {
            elem: Type::Path {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "Self",
                        },
                    ],
                },
            },
        },
    })
    "###);
}
#[test]
fn test_by_box() {
    let item::Trait::Fn { sig, .. } = parse_quote! {
        fn by_box(self: Box<Self>);
    };
    snapshot!(&sig.args[0], @r###"
    item::FnArg::Receiver(item::Receiver {
        colon: Some,
        ty: Type::Path {
            path: Path {
                segments: [
                    path::Segment {
                        ident: "Box",
                        arguments: path::Args::AngleBracketed {
                            args: [
                                path::Arg::Type(Type::Path {
                                    path: Path {
                                        segments: [
                                            path::Segment {
                                                ident: "Self",
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
    })
    "###);
}
#[test]
fn test_by_pin() {
    let item::Trait::Fn { sig, .. } = parse_quote! {
        fn by_pin(self: Pin<Self>);
    };
    snapshot!(&sig.args[0], @r###"
    item::FnArg::Receiver(item::Receiver {
        colon: Some,
        ty: Type::Path {
            path: Path {
                segments: [
                    path::Segment {
                        ident: "Pin",
                        arguments: path::Args::AngleBracketed {
                            args: [
                                path::Arg::Type(Type::Path {
                                    path: Path {
                                        segments: [
                                            path::Segment {
                                                ident: "Self",
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
    })
    "###);
}
#[test]
fn test_explicit_type() {
    let item::Trait::Fn { sig, .. } = parse_quote! {
        fn explicit_type(self: Pin<MyType>);
    };
    snapshot!(&sig.args[0], @r###"
    item::FnArg::Receiver(item::Receiver {
        colon: Some,
        ty: Type::Path {
            path: Path {
                segments: [
                    path::Segment {
                        ident: "Pin",
                        arguments: path::Args::AngleBracketed {
                            args: [
                                path::Arg::Type(Type::Path {
                                    path: Path {
                                        segments: [
                                            path::Segment {
                                                ident: "MyType",
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
    })
    "###);
}
#[test]
fn test_value_shorthand() {
    let item::Trait::Fn { sig, .. } = parse_quote! {
        fn value_shorthand(self);
    };
    snapshot!(&sig.args[0], @r###"
    item::FnArg::Receiver(item::Receiver {
        ty: Type::Path {
            path: Path {
                segments: [
                    path::Segment {
                        ident: "Self",
                    },
                ],
            },
        },
    })
    "###);
}
#[test]
fn test_mut_value_shorthand() {
    let item::Trait::Fn { sig, .. } = parse_quote! {
        fn mut_value_shorthand(mut self);
    };
    snapshot!(&sig.args[0], @r###"
    item::FnArg::Receiver(item::Receiver {
        mutability: Some,
        ty: Type::Path {
            path: Path {
                segments: [
                    path::Segment {
                        ident: "Self",
                    },
                ],
            },
        },
    })
    "###);
}
#[test]
fn test_ref_shorthand() {
    let item::Trait::Fn { sig, .. } = parse_quote! {
        fn ref_shorthand(&self);
    };
    snapshot!(&sig.args[0], @r###"
    item::FnArg::Receiver(item::Receiver {
        reference: Some(None),
        ty: Type::Reference {
            elem: Type::Path {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "Self",
                        },
                    ],
                },
            },
        },
    })
    "###);
}
#[test]
fn test_ref_shorthand_with_lifetime() {
    let item::Trait::Fn { sig, .. } = parse_quote! {
        fn ref_shorthand(&'a self);
    };
    snapshot!(&sig.args[0], @r###"
    item::FnArg::Receiver(item::Receiver {
        reference: Some(Some(Lifetime {
            ident: "a",
        })),
        ty: Type::Reference {
            lifetime: Some(Lifetime {
                ident: "a",
            }),
            elem: Type::Path {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "Self",
                        },
                    ],
                },
            },
        },
    })
    "###);
}
#[test]
fn test_ref_mut_shorthand() {
    let item::Trait::Fn { sig, .. } = parse_quote! {
        fn ref_mut_shorthand(&mut self);
    };
    snapshot!(&sig.args[0], @r###"
    item::FnArg::Receiver(item::Receiver {
        reference: Some(None),
        mutability: Some,
        ty: Type::Reference {
            mutability: Some,
            elem: Type::Path {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "Self",
                        },
                    ],
                },
            },
        },
    })
    "###);
}
#[test]
fn test_ref_mut_shorthand_with_lifetime() {
    let item::Trait::Fn { sig, .. } = parse_quote! {
        fn ref_mut_shorthand(&'a mut self);
    };
    snapshot!(&sig.args[0], @r###"
    item::FnArg::Receiver(item::Receiver {
        reference: Some(Some(Lifetime {
            ident: "a",
        })),
        mutability: Some,
        ty: Type::Reference {
            lifetime: Some(Lifetime {
                ident: "a",
            }),
            mutability: Some,
            elem: Type::Path {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "Self",
                        },
                    ],
                },
            },
        },
    })
    "###);
}
