#![cfg(target_pointer_width = "64")]
use std::mem;
use syn::{patt::Patt, ty::Type, Expr, Item, Lit};
#[rustversion::attr(before(2022-11-24), ignore)]
#[test]
fn test_expr_size() {
    assert_eq!(mem::size_of::<Expr>(), 176);
}
#[rustversion::attr(before(2022-09-09), ignore)]
#[test]
fn test_item_size() {
    assert_eq!(mem::size_of::<Item>(), 360);
}
#[rustversion::attr(before(2023-04-29), ignore)]
#[test]
fn test_type_size() {
    assert_eq!(mem::size_of::<ty::Type>(), 232);
}
#[rustversion::attr(before(2023-04-29), ignore)]
#[test]
fn test_pat_size() {
    assert_eq!(mem::size_of::<patt::Patt>(), 184);
}
#[rustversion::attr(before(2022-09-09), ignore)]
#[test]
fn test_lit_size() {
    assert_eq!(mem::size_of::<Lit>(), 32);
}
