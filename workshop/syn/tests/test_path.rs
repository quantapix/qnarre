#![allow(clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use syn::*;
#[test]
fn parse_interpolated_leading_component() {
    let tokens = pm2::Stream::from_iter(vec![
        pm2::Tree::Group(Group::new(pm2::Delim::None, quote! { first })),
        pm2::Tree::Punct(Punct::new(':', pm2::Spacing::Joint)),
        pm2::Tree::Punct(Punct::new(':', pm2::Spacing::Alone)),
        pm2::Tree::Ident(Ident::new("rest", pm2::Span::call_site())),
    ]);
    snapshot!(tokens.clone() as Expr, @r###"
    Expr::Path {
        path: Path {
            segments: [
                path::Segment {
                    ident: "first",
                },
                path::Segment {
                    ident: "rest",
                },
            ],
        },
    }
    "###);
    snapshot!(tokens as ty::Type, @r###"
    Type::Path {
        path: Path {
            segments: [
                path::Segment {
                    ident: "first",
                },
                path::Segment {
                    ident: "rest",
                },
            ],
        },
    }
    "###);
}
#[test]
fn print_incomplete_qpath() {
    let mut typ: ty::Path = parse_quote!(<Self as A>::Q);
    snapshot!(typ.to_token_stream(), @r###"
    pm2::Stream(`< Self as A > :: Q`)
    "###);
    assert!(typ.path.segs.pop().is_some());
    snapshot!(typ.to_token_stream(), @r###"
    pm2::Stream(`< Self as A > ::`)
    "###);
    assert!(typ.path.segs.pop().is_some());
    snapshot!(typ.to_token_stream(), @r###"
    pm2::Stream(`< Self >`)
    "###);
    assert!(typ.path.segs.pop().is_none());
    let mut typ: ty::Path = parse_quote!(<Self>::A::B);
    snapshot!(typ.to_token_stream(), @r###"
    pm2::Stream(`< Self > :: A :: B`)
    "###);
    assert!(typ.path.segs.pop().is_some());
    snapshot!(typ.to_token_stream(), @r###"
    pm2::Stream(`< Self > :: A ::`)
    "###);
    assert!(typ.path.segs.pop().is_some());
    snapshot!(typ.to_token_stream(), @r###"
    pm2::Stream(`< Self > ::`)
    "###);
    assert!(typ.path.segs.pop().is_none());
    let mut typ: ty::Path = parse_quote!(Self::A::B);
    snapshot!(typ.to_token_stream(), @r###"
    pm2::Stream(`Self :: A :: B`)
    "###);
    assert!(typ.path.segs.pop().is_some());
    snapshot!(typ.to_token_stream(), @r###"
    pm2::Stream(`Self :: A ::`)
    "###);
    assert!(typ.path.segs.pop().is_some());
    snapshot!(typ.to_token_stream(), @r###"
    pm2::Stream(`Self ::`)
    "###);
    assert!(typ.path.segs.pop().is_some());
    snapshot!(typ.to_token_stream(), @r###"
    pm2::Stream(``)
    "###);
    assert!(typ.path.segs.pop().is_none());
}
#[test]
fn parse_parenthesized_path_arguments_with_disambiguator() {
    #[rustfmt::skip]
    let tokens = quote!(dyn FnOnce::() -> !);
    snapshot!(tokens as ty::Type, @r###"
    Type::TraitObject {
        dyn_: Some,
        bounds: [
            gen::bound::Type::Trait(gen::bound::Trait {
                path: Path {
                    segments: [
                        path::Segment {
                            ident: "FnOnce",
                            arguments: path::Args::Parenthesized {
                                ret: ty::Ret::Type(
                                    Type::Never,
                                ),
                            },
                        },
                    ],
                },
            }),
        ],
    }
    "###);
}
