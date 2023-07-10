#![cfg(target_pointer_width = "64")]
use std::mem;
use syn::{patt::Patt, ty::Type, Expr, Item, Lit};
#[test]
fn test_expr_size() {
    assert_eq!(mem::size_of::<Expr>(), 176);
}
#[test]
fn test_item_size() {
    assert_eq!(mem::size_of::<Item>(), 360);
}
#[test]
fn test_type_size() {
    assert_eq!(mem::size_of::<ty::Type>(), 232);
}
#[test]
fn test_pat_size() {
    assert_eq!(mem::size_of::<patt::Patt>(), 184);
}
#[test]
fn test_lit_size() {
    assert_eq!(mem::size_of::<Lit>(), 32);
}
