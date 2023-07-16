extern crate proc_macro;

use std::mem;

#[rustversion::attr(before(1.32), ignore)]
#[test]
fn test_proc_macro_span_size() {
    assert_eq!(mem::size_of::<proc_macro::Span>(), 4);
    assert_eq!(mem::size_of::<Option<proc_macro::Span>>(), 4);
}

#[test]
fn test_proc_macro2_fallback_span_size_without_locations() {
    assert_eq!(mem::size_of::<proc_macro2::Span>(), 0);
    assert_eq!(mem::size_of::<Option<proc_macro2::Span>>(), 1);
}

#[test]
fn test_proc_macro2_fallback_span_size_with_locations() {
    assert_eq!(mem::size_of::<proc_macro2::Span>(), 8);
    assert_eq!(mem::size_of::<Option<proc_macro2::Span>>(), 12);
}

#[test]
fn test_proc_macro2_wrapper_span_size_without_locations() {
    assert_eq!(mem::size_of::<proc_macro2::Span>(), 4);
    assert_eq!(mem::size_of::<Option<proc_macro2::Span>>(), 8);
}

#[test]
fn test_proc_macro2_wrapper_span_size_with_locations() {
    assert_eq!(mem::size_of::<proc_macro2::Span>(), 12);
    assert_eq!(mem::size_of::<Option<proc_macro2::Span>>(), 12);
}
