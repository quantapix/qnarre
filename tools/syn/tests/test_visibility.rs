#![allow(clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use syn::*;
#[derive(Debug)]
struct VisRest {
    vis: data::Visibility,
    rest: pm2::Stream,
}
impl Parse for VisRest {
    fn parse(x: Stream) -> Res<Self> {
        Ok(VisRest {
            vis: x.parse()?,
            rest: x.parse()?,
        })
    }
}
macro_rules! assert_vis_parse {
    ($input:expr, Ok($p:pat)) => {
        assert_vis_parse!($input, Ok($p) + "");
    };
    ($input:expr, Ok($p:pat) + $rest:expr) => {
        let expected = $rest.parse::<pm2::Stream>().unwrap();
        let parse: VisRest = syn::parse::parse_str($input).unwrap();
        match parse.vis {
            $p => {},
            _ => panic!("Expected {}, got {:?}", stringify!($p), parse.vis),
        }
        assert_eq!(parse.rest.to_string(), expected.to_string());
    };
    ($input:expr, Err) => {
        syn::parse2::<VisRest>($input.parse().unwrap()).unwrap_err();
    };
}
#[test]
fn test_pub() {
    assert_vis_parse!("pub", Ok(data::Visibility::Public(_)));
}
#[test]
fn test_inherited() {
    assert_vis_parse!("", Ok(data::Visibility::Inherited));
}
#[test]
fn test_in() {
    assert_vis_parse!("pub(in foo::bar)", Ok(data::Visibility::Restricted(_)));
}
#[test]
fn test_pub_crate() {
    assert_vis_parse!("pub(crate)", Ok(data::Visibility::Restricted(_)));
}
#[test]
fn test_pub_self() {
    assert_vis_parse!("pub(self)", Ok(data::Visibility::Restricted(_)));
}
#[test]
fn test_pub_super() {
    assert_vis_parse!("pub(super)", Ok(data::Visibility::Restricted(_)));
}
#[test]
fn test_missing_in() {
    assert_vis_parse!("pub(foo::bar)", Ok(data::Visibility::Public(_)) + "(foo::bar)");
}
#[test]
fn test_missing_in_path() {
    assert_vis_parse!("pub(in)", Err);
}
#[test]
fn test_crate_path() {
    assert_vis_parse!(
        "pub(crate::A, crate::B)",
        Ok(data::Visibility::Public(_)) + "(crate::A, crate::B)"
    );
}
#[test]
fn test_junk_after_in() {
    assert_vis_parse!("pub(in some::path @@garbage)", Err);
}
#[test]
fn test_empty_group_vis() {
    let tokens = pm2::Stream::from_iter(vec![
        pm2::Tree::Ident(Ident::new("struct", pm2::Span::call_site())),
        pm2::Tree::Ident(Ident::new("S", pm2::Span::call_site())),
        pm2::Tree::Group(Group::new(
            pm2::Delim::Brace,
            pm2::Stream::from_iter(vec![
                pm2::Tree::Group(Group::new(pm2::Delim::None, pm2::Stream::new())),
                pm2::Tree::Group(Group::new(
                    pm2::Delim::None,
                    pm2::Stream::from_iter(vec![pm2::Tree::Ident(Ident::new("f", pm2::Span::call_site()))]),
                )),
                pm2::Tree::Punct(Punct::new(':', pm2::Spacing::Alone)),
                pm2::Tree::Group(Group::new(pm2::Delim::Parenthesis, pm2::Stream::new())),
            ]),
        )),
    ]);
    snapshot!(tokens as DeriveInput, @r###"
    DeriveInput {
        vis: data::Visibility::Inherited,
        ident: "S",
        gens: gen::Gens,
        data: Data::Struct {
            fields: Named {
                fields: [
                    Field {
                        vis: data::Visibility::Inherited,
                        ident: Some("f"),
                        colon: Some,
                        typ: Type::Tuple,
                    },
                ],
            },
        },
    }
    "###);
}