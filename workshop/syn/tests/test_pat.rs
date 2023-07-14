#![allow(clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use syn::*;
#[test]
fn test_pat_ident() {
    match pat::Pat::parse_single.parse2(quote!(self)).unwrap() {
        pat::Pat::Ident(_) => (),
        value => panic!("expected patt::Ident, got {:?}", value),
    }
}
#[test]
fn test_pat_path() {
    match pat::Pat::parse_single.parse2(quote!(self::CONST)).unwrap() {
        pat::Pat::Path(_) => (),
        value => panic!("expected PatPath, got {:?}", value),
    }
}
#[test]
fn test_leading_vert() {
    syn::parse::parse_str::<Item>("fn f() {}").unwrap();
    syn::parse::parse_str::<Item>("fn fun1(| A: E) {}").unwrap_err();
    syn::parse::parse_str::<Item>("fn fun2(|| A: E) {}").unwrap_err();
    syn::parse::parse_str::<stmt::Stmt>("let | () = ();").unwrap_err();
    syn::parse::parse_str::<stmt::Stmt>("let (| A): E;").unwrap();
    syn::parse::parse_str::<stmt::Stmt>("let (|| A): (E);").unwrap_err();
    syn::parse::parse_str::<stmt::Stmt>("let (| A,): (E,);").unwrap();
    syn::parse::parse_str::<stmt::Stmt>("let [| A]: [E; 1];").unwrap();
    syn::parse::parse_str::<stmt::Stmt>("let [|| A]: [E; 1];").unwrap_err();
    syn::parse::parse_str::<stmt::Stmt>("let TS(| A): TS;").unwrap();
    syn::parse::parse_str::<stmt::Stmt>("let TS(|| A): TS;").unwrap_err();
    syn::parse::parse_str::<stmt::Stmt>("let NS { f: | A }: NS;").unwrap();
    syn::parse::parse_str::<stmt::Stmt>("let NS { f: || A }: NS;").unwrap_err();
}
#[test]
fn test_group() {
    let group = Group::new(pm2::Delim::None, quote!(Some(_)));
    let tokens = pm2::Stream::from_iter(vec![pm2::Tree::Group(group)]);
    let pat = pat::Pat::parse_single.parse2(tokens).unwrap();
    snapshot!(pat, @r###"
    pat::Pat::TupleStruct {
        path: Path {
            segments: [
                path::Segment {
                    ident: "Some",
                },
            ],
        },
        elems: [
            pat::Pat::Wild,
        ],
    }
    "###);
}
#[test]
fn test_ranges() {
    pat::Pat::parse_single.parse_str("..").unwrap();
    pat::Pat::parse_single.parse_str("..hi").unwrap();
    pat::Pat::parse_single.parse_str("lo..").unwrap();
    pat::Pat::parse_single.parse_str("lo..hi").unwrap();
    pat::Pat::parse_single.parse_str("..=").unwrap_err();
    pat::Pat::parse_single.parse_str("..=hi").unwrap();
    pat::Pat::parse_single.parse_str("lo..=").unwrap_err();
    pat::Pat::parse_single.parse_str("lo..=hi").unwrap();
    pat::Pat::parse_single.parse_str("...").unwrap_err();
    pat::Pat::parse_single.parse_str("...hi").unwrap_err();
    pat::Pat::parse_single.parse_str("lo...").unwrap_err();
    pat::Pat::parse_single.parse_str("lo...hi").unwrap();
    pat::Pat::parse_single.parse_str("[lo..]").unwrap_err();
    pat::Pat::parse_single.parse_str("[..=hi]").unwrap_err();
    pat::Pat::parse_single.parse_str("[(lo..)]").unwrap();
    pat::Pat::parse_single.parse_str("[(..=hi)]").unwrap();
    pat::Pat::parse_single.parse_str("[lo..=hi]").unwrap();
    pat::Pat::parse_single.parse_str("[_, lo.., _]").unwrap_err();
    pat::Pat::parse_single.parse_str("[_, ..=hi, _]").unwrap_err();
    pat::Pat::parse_single.parse_str("[_, (lo..), _]").unwrap();
    pat::Pat::parse_single.parse_str("[_, (..=hi), _]").unwrap();
    pat::Pat::parse_single.parse_str("[_, lo..=hi, _]").unwrap();
}
