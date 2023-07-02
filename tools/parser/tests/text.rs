use serde_test::*;
use static_assertions::*;
use std::{
    fmt::Debug,
    hash::Hash,
    marker::{Send, Sync},
    ops,
    panic::{RefUnwindSafe, UnwindSafe},
};
use syntax::{TextRange, TextSize};

fn size(x: u32) -> TextSize {
    TextSize::from(x)
}
#[test]
fn sum() {
    let xs: Vec<TextSize> = vec![size(0), size(1), size(2)];
    assert_eq!(xs.iter().sum::<TextSize>(), size(3));
    assert_eq!(xs.into_iter().sum::<TextSize>(), size(3));
}
#[test]
fn math() {
    assert_eq!(size(10) + size(5), size(15));
    assert_eq!(size(10) - size(5), size(5));
}
#[test]
fn checked_math() {
    assert_eq!(size(1).checked_add(size(1)), Some(size(2)));
    assert_eq!(size(1).checked_sub(size(1)), Some(size(0)));
    assert_eq!(size(1).checked_sub(size(2)), None);
    assert_eq!(size(!0).checked_add(size(1)), None);
}

fn range(x: ops::Range<u32>) -> TextRange {
    TextRange::new(x.start.into(), x.end.into())
}
#[test]
#[rustfmt::skip]
fn contains() {
    assert!(   range(2..4).contains_range(range(2..3)));
    assert!( ! range(2..4).contains_range(range(1..3)));
}
#[test]
fn intersect() {
    assert_eq!(range(1..2).intersect(range(2..3)), Some(range(2..2)));
    assert_eq!(range(1..5).intersect(range(2..3)), Some(range(2..3)));
    assert_eq!(range(1..2).intersect(range(3..4)), None);
}
#[test]
fn cover() {
    assert_eq!(range(1..2).cover(range(2..3)), range(1..3));
    assert_eq!(range(1..5).cover(range(2..3)), range(1..5));
    assert_eq!(range(1..2).cover(range(4..5)), range(1..5));
}
#[test]
fn cover_offset() {
    assert_eq!(range(1..3).cover_offset(size(0)), range(0..3));
    assert_eq!(range(1..3).cover_offset(size(1)), range(1..3));
    assert_eq!(range(1..3).cover_offset(size(2)), range(1..3));
    assert_eq!(range(1..3).cover_offset(size(3)), range(1..3));
    assert_eq!(range(1..3).cover_offset(size(4)), range(1..4));
}
#[test]
#[rustfmt::skip]
fn contains_point() {
    assert!( ! range(1..3).contains(size(0)));
    assert!(   range(1..3).contains(size(1)));
    assert!(   range(1..3).contains(size(2)));
    assert!( ! range(1..3).contains(size(3)));
    assert!( ! range(1..3).contains(size(4)));
    assert!( ! range(1..3).contains_inclusive(size(0)));
    assert!(   range(1..3).contains_inclusive(size(1)));
    assert!(   range(1..3).contains_inclusive(size(2)));
    assert!(   range(1..3).contains_inclusive(size(3)));
    assert!( ! range(1..3).contains_inclusive(size(4)));
}

#[derive(Copy, Clone)]
struct BadRope<'a>(&'a [&'a str]);
impl BadRope<'_> {
    fn text_len(self) -> TextSize {
        self.0.iter().copied().map(TextSize::of).sum()
    }
}
#[test]
fn bad_rope() {
    let x: char = 'c';
    let _ = TextSize::of(x);
    let x: &str = "hello";
    let _ = TextSize::of(x);
    let x: &String = &"hello".into();
    let _ = TextSize::of(x);
    let _ = BadRope(&[""]).text_len();
}

#[test]
fn indexing() {
    let range = TextRange::default();
    &""[range];
    &String::new()[range];
}

assert_impl_all!(TextSize: Send, Sync, Unpin, UnwindSafe, RefUnwindSafe);
assert_impl_all!(TextRange: Send, Sync, Unpin, UnwindSafe, RefUnwindSafe);

assert_impl_all!(TextSize: Copy, Debug, Default, Hash, Ord);
assert_impl_all!(TextRange: Copy, Debug, Default, Hash, Eq);

#[test]
fn size_serialization() {
    assert_tokens(&size(00), &[Token::U32(00)]);
    assert_tokens(&size(10), &[Token::U32(10)]);
    assert_tokens(&size(20), &[Token::U32(20)]);
    assert_tokens(&size(30), &[Token::U32(30)]);
}
#[test]
fn range_serialization() {
    assert_tokens(
        &range(00..10),
        &[Token::Tuple { len: 2 }, Token::U32(00), Token::U32(10), Token::TupleEnd],
    );
    assert_tokens(
        &range(10..20),
        &[Token::Tuple { len: 2 }, Token::U32(10), Token::U32(20), Token::TupleEnd],
    );
    assert_tokens(
        &range(20..30),
        &[Token::Tuple { len: 2 }, Token::U32(20), Token::U32(30), Token::TupleEnd],
    );
    assert_tokens(
        &range(30..40),
        &[Token::Tuple { len: 2 }, Token::U32(30), Token::U32(40), Token::TupleEnd],
    );
}
#[test]
fn invalid_range_deserialization() {
    assert_tokens::<TextRange>(
        &range(62..92),
        &[Token::Tuple { len: 2 }, Token::U32(62), Token::U32(92), Token::TupleEnd],
    );
    assert_de_tokens_error::<TextRange>(
        &[Token::Tuple { len: 2 }, Token::U32(92), Token::U32(62), Token::TupleEnd],
        "invalid range: 92..62",
    );
}
