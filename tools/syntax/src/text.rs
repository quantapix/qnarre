use cmp::Ordering;
use std::{
    cmp,
    convert::{TryFrom, TryInto},
    fmt, iter,
    num::TryFromIntError,
    ops::{Add, AddAssign, Bound, Index, IndexMut, Range, RangeBounds, Sub, SubAssign},
    u32,
};

#[derive(Default, Copy, Clone, Eq, PartialEq, Hash)]
pub struct TextRange {
    start: TextSize,
    end: TextSize,
}
impl TextRange {
    #[inline]
    pub fn new(start: TextSize, end: TextSize) -> TextRange {
        assert!(start <= end);
        TextRange { start, end }
    }
    #[inline]
    pub fn at(offset: TextSize, len: TextSize) -> TextRange {
        TextRange::new(offset, offset + len)
    }
    #[inline]
    pub fn empty(offset: TextSize) -> TextRange {
        TextRange {
            start: offset,
            end: offset,
        }
    }
    #[inline]
    pub fn up_to(end: TextSize) -> TextRange {
        TextRange { start: 0.into(), end }
    }
    #[inline]
    pub const fn start(self) -> TextSize {
        self.start
    }
    #[inline]
    pub const fn end(self) -> TextSize {
        self.end
    }
    #[inline]
    pub const fn len(self) -> TextSize {
        TextSize {
            raw: self.end().raw - self.start().raw,
        }
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.start().raw == self.end().raw
    }
    #[inline]
    pub fn contains(self, offset: TextSize) -> bool {
        self.start() <= offset && offset < self.end()
    }
    #[inline]
    pub fn contains_inclusive(self, offset: TextSize) -> bool {
        self.start() <= offset && offset <= self.end()
    }
    #[inline]
    pub fn contains_range(self, other: TextRange) -> bool {
        self.start() <= other.start() && other.end() <= self.end()
    }
    #[inline]
    pub fn intersect(self, other: TextRange) -> Option<TextRange> {
        let start = cmp::max(self.start(), other.start());
        let end = cmp::min(self.end(), other.end());
        if end < start {
            return None;
        }
        Some(TextRange::new(start, end))
    }
    #[inline]
    pub fn cover(self, other: TextRange) -> TextRange {
        let start = cmp::min(self.start(), other.start());
        let end = cmp::max(self.end(), other.end());
        TextRange::new(start, end)
    }
    #[inline]
    pub fn cover_offset(self, offset: TextSize) -> TextRange {
        self.cover(TextRange::empty(offset))
    }
    #[inline]
    pub fn checked_add(self, offset: TextSize) -> Option<TextRange> {
        Some(TextRange {
            start: self.start.checked_add(offset)?,
            end: self.end.checked_add(offset)?,
        })
    }
    #[inline]
    pub fn checked_sub(self, offset: TextSize) -> Option<TextRange> {
        Some(TextRange {
            start: self.start.checked_sub(offset)?,
            end: self.end.checked_sub(offset)?,
        })
    }
    #[inline]
    pub fn ordering(self, other: TextRange) -> Ordering {
        if self.end() <= other.start() {
            Ordering::Less
        } else if other.end() <= self.start() {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}
impl Add<TextSize> for TextRange {
    type Output = TextRange;
    #[inline]
    fn add(self, offset: TextSize) -> TextRange {
        self.checked_add(offset).expect("TextRange +offset overflowed")
    }
}
impl Sub<TextSize> for TextRange {
    type Output = TextRange;
    #[inline]
    fn sub(self, offset: TextSize) -> TextRange {
        self.checked_sub(offset).expect("TextRange -offset overflowed")
    }
}
impl RangeBounds<TextSize> for TextRange {
    fn start_bound(&self) -> Bound<&TextSize> {
        Bound::Included(&self.start)
    }
    fn end_bound(&self) -> Bound<&TextSize> {
        Bound::Excluded(&self.end)
    }
}
impl fmt::Debug for TextRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start().raw, self.end().raw)
    }
}

impl Index<TextRange> for str {
    type Output = str;
    #[inline]
    fn index(&self, index: TextRange) -> &str {
        &self[Range::<usize>::from(index)]
    }
}
impl Index<TextRange> for String {
    type Output = str;
    #[inline]
    fn index(&self, index: TextRange) -> &str {
        &self[Range::<usize>::from(index)]
    }
}
impl IndexMut<TextRange> for str {
    #[inline]
    fn index_mut(&mut self, index: TextRange) -> &mut str {
        &mut self[Range::<usize>::from(index)]
    }
}
impl IndexMut<TextRange> for String {
    #[inline]
    fn index_mut(&mut self, index: TextRange) -> &mut str {
        &mut self[Range::<usize>::from(index)]
    }
}

impl<T> From<TextRange> for Range<T>
where
    T: From<TextSize>,
{
    #[inline]
    fn from(r: TextRange) -> Self {
        r.start().into()..r.end().into()
    }
}

macro_rules! ops {
    (impl $Op:ident for TextRange by fn $f:ident = $op:tt) => {
        impl $Op<&TextSize> for TextRange {
            type Output = TextRange;
            #[inline]
            fn $f(self, other: &TextSize) -> TextRange {
                self $op *other
            }
        }
        impl<T> $Op<T> for &TextRange
        where
            TextRange: $Op<T, Output=TextRange>,
        {
            type Output = TextRange;
            #[inline]
            fn $f(self, other: T) -> TextRange {
                *self $op other
            }
        }
    };
}
ops!(impl Add for TextRange by fn add = +);
ops!(impl Sub for TextRange by fn sub = -);

impl<A> AddAssign<A> for TextRange
where
    TextRange: Add<A, Output = TextRange>,
{
    #[inline]
    fn add_assign(&mut self, rhs: A) {
        *self = *self + rhs
    }
}
impl<S> SubAssign<S> for TextRange
where
    TextRange: Sub<S, Output = TextRange>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: S) {
        *self = *self - rhs
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextSize {
    pub raw: u32,
}
impl TextSize {
    #[inline]
    pub fn of<T: TextLen>(text: T) -> TextSize {
        text.text_len()
    }
    #[inline]
    pub fn checked_add(self, rhs: TextSize) -> Option<TextSize> {
        self.raw.checked_add(rhs.raw).map(|raw| TextSize { raw })
    }
    #[inline]
    pub fn checked_sub(self, rhs: TextSize) -> Option<TextSize> {
        self.raw.checked_sub(rhs.raw).map(|raw| TextSize { raw })
    }
}
impl fmt::Debug for TextSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.raw)
    }
}
impl From<u32> for TextSize {
    #[inline]
    fn from(raw: u32) -> Self {
        TextSize { raw }
    }
}
impl From<TextSize> for u32 {
    #[inline]
    fn from(value: TextSize) -> Self {
        value.raw
    }
}
impl TryFrom<usize> for TextSize {
    type Error = TryFromIntError;
    #[inline]
    fn try_from(value: usize) -> Result<Self, TryFromIntError> {
        Ok(u32::try_from(value)?.into())
    }
}
impl From<TextSize> for usize {
    #[inline]
    fn from(value: TextSize) -> Self {
        value.raw as usize
    }
}

macro_rules! ops2 {
        (impl $Op:ident for TextSize by fn $f:ident = $op:tt) => {
            impl $Op<TextSize> for TextSize {
                type Output = TextSize;
                #[inline]
                fn $f(self, other: TextSize) -> TextSize {
                    TextSize { raw: self.raw $op other.raw }
                }
            }
            impl $Op<&TextSize> for TextSize {
                type Output = TextSize;
                #[inline]
                fn $f(self, other: &TextSize) -> TextSize {
                    self $op *other
                }
            }
            impl<T> $Op<T> for &TextSize
            where
                TextSize: $Op<T, Output=TextSize>,
            {
                type Output = TextSize;
                #[inline]
                fn $f(self, other: T) -> TextSize {
                    *self $op other
                }
            }
        };
    }
ops2!(impl Add for TextSize by fn add = +);
ops2!(impl Sub for TextSize by fn sub = -);

impl<A> AddAssign<A> for TextSize
where
    TextSize: Add<A, Output = TextSize>,
{
    #[inline]
    fn add_assign(&mut self, rhs: A) {
        *self = *self + rhs
    }
}
impl<S> SubAssign<S> for TextSize
where
    TextSize: Sub<S, Output = TextSize>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: S) {
        *self = *self - rhs
    }
}
impl<A> iter::Sum<A> for TextSize
where
    TextSize: Add<A, Output = TextSize>,
{
    #[inline]
    fn sum<I: Iterator<Item = A>>(iter: I) -> TextSize {
        iter.fold(0.into(), Add::add)
    }
}

use priv_in_pub::Sealed;

mod priv_in_pub {
    pub trait Sealed {}
}

pub trait TextLen: Copy + Sealed {
    fn text_len(self) -> TextSize;
}
impl Sealed for &'_ str {}
impl TextLen for &'_ str {
    #[inline]
    fn text_len(self) -> TextSize {
        self.len().try_into().unwrap()
    }
}
impl Sealed for &'_ String {}
impl TextLen for &'_ String {
    #[inline]
    fn text_len(self) -> TextSize {
        self.as_str().text_len()
    }
}
impl Sealed for char {}
impl TextLen for char {
    #[inline]
    fn text_len(self) -> TextSize {
        (self.len_utf8() as u32).into()
    }
}

#[cfg(feature = "serde")]
mod serde_impls {
    use super::*;
    use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
    impl Serialize for TextSize {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            self.raw.serialize(serializer)
        }
    }
    impl<'de> Deserialize<'de> for TextSize {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            u32::deserialize(deserializer).map(TextSize::from)
        }
    }
    impl Serialize for TextRange {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            (self.start(), self.end()).serialize(serializer)
        }
    }
    impl<'de> Deserialize<'de> for TextRange {
        #[allow(clippy::nonminimal_bool)]
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let (start, end) = Deserialize::deserialize(deserializer)?;
            if !(start <= end) {
                return Err(de::Error::custom(format!("invalid range: {:?}..{:?}", start, end)));
            }
            Ok(TextRange::new(start, end))
        }
    }
}

#[cfg(target_pointer_width = "16")]
compile_error!("text-size assumes usize >= u32 and does not work on 16-bit targets");
