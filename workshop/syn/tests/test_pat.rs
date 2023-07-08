#![allow(clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use quote::quote;
use syn::parse::Parser;
use syn::{Item, Pat, Stmt};
#[test]
fn test_pat_ident() {
    match Pat::parse_single.parse2(quote!(self)).unwrap() {
        Pat::Ident(_) => (),
        value => panic!("expected PatIdent, got {:?}", value),
    }
}
#[test]
fn test_pat_path() {
    match Pat::parse_single.parse2(quote!(self::CONST)).unwrap() {
        Pat::Path(_) => (),
        value => panic!("expected PatPath, got {:?}", value),
    }
}
#[test]
fn test_leading_vert() {
    syn::parse_str::<Item>("fn f() {}").unwrap();
    syn::parse_str::<Item>("fn fun1(| A: E) {}").unwrap_err();
    syn::parse_str::<Item>("fn fun2(|| A: E) {}").unwrap_err();
    syn::parse_str::<Stmt>("let | () = ();").unwrap_err();
    syn::parse_str::<Stmt>("let (| A): E;").unwrap();
    syn::parse_str::<Stmt>("let (|| A): (E);").unwrap_err();
    syn::parse_str::<Stmt>("let (| A,): (E,);").unwrap();
    syn::parse_str::<Stmt>("let [| A]: [E; 1];").unwrap();
    syn::parse_str::<Stmt>("let [|| A]: [E; 1];").unwrap_err();
    syn::parse_str::<Stmt>("let TS(| A): TS;").unwrap();
    syn::parse_str::<Stmt>("let TS(|| A): TS;").unwrap_err();
    syn::parse_str::<Stmt>("let NS { f: | A }: NS;").unwrap();
    syn::parse_str::<Stmt>("let NS { f: || A }: NS;").unwrap_err();
}
#[test]
fn test_group() {
    let group = Group::new(Delimiter::None, quote!(Some(_)));
    let tokens = TokenStream::from_iter(vec![TokenTree::Group(group)]);
    let pat = Pat::parse_single.parse2(tokens).unwrap();
    snapshot!(pat, @r###"
    Pat::TupleStruct {
        path: Path {
            segments: [
                path::Segment {
                    ident: "Some",
                },
            ],
        },
        elems: [
            Pat::Wild,
        ],
    }
    "###);
}
#[test]
fn test_ranges() {
    Pat::parse_single.parse_str("..").unwrap();
    Pat::parse_single.parse_str("..hi").unwrap();
    Pat::parse_single.parse_str("lo..").unwrap();
    Pat::parse_single.parse_str("lo..hi").unwrap();
    Pat::parse_single.parse_str("..=").unwrap_err();
    Pat::parse_single.parse_str("..=hi").unwrap();
    Pat::parse_single.parse_str("lo..=").unwrap_err();
    Pat::parse_single.parse_str("lo..=hi").unwrap();
    Pat::parse_single.parse_str("...").unwrap_err();
    Pat::parse_single.parse_str("...hi").unwrap_err();
    Pat::parse_single.parse_str("lo...").unwrap_err();
    Pat::parse_single.parse_str("lo...hi").unwrap();
    Pat::parse_single.parse_str("[lo..]").unwrap_err();
    Pat::parse_single.parse_str("[..=hi]").unwrap_err();
    Pat::parse_single.parse_str("[(lo..)]").unwrap();
    Pat::parse_single.parse_str("[(..=hi)]").unwrap();
    Pat::parse_single.parse_str("[lo..=hi]").unwrap();
    Pat::parse_single.parse_str("[_, lo.., _]").unwrap_err();
    Pat::parse_single.parse_str("[_, ..=hi, _]").unwrap_err();
    Pat::parse_single.parse_str("[_, (lo..), _]").unwrap();
    Pat::parse_single.parse_str("[_, (..=hi), _]").unwrap();
    Pat::parse_single.parse_str("[_, lo..=hi, _]").unwrap();
}
