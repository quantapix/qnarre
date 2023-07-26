#![cfg(target_pointer_width = "64")]
use std::mem;
use syn::{pat::Pat, typ::Type, Expr, Item, Lit};
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
    assert_eq!(mem::size_of::<typ::Type>(), 232);
}
#[test]
fn test_pat_size() {
    assert_eq!(mem::size_of::<pat::Pat>(), 184);
}
#[test]
fn test_lit_size() {
    assert_eq!(mem::size_of::<Lit>(), 32);
}
