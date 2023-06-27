// https://github.com/rust-analyzer/text-size.git (e4d0f2b)
use cmp::Ordering;
use itertools::Itertools;
use std::{
    cmp::{self, max},
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Indel {
    pub insert: String,
    pub delete: TextRange,
}
impl Indel {
    pub fn insert(offset: TextSize, text: String) -> Indel {
        Indel::replace(TextRange::empty(offset), text)
    }
    pub fn delete(range: TextRange) -> Indel {
        Indel::replace(range, String::new())
    }
    pub fn replace(range: TextRange, replace_with: String) -> Indel {
        Indel {
            delete: range,
            insert: replace_with,
        }
    }
    pub fn apply(&self, text: &mut String) {
        let start: usize = self.delete.start().into();
        let end: usize = self.delete.end().into();
        text.replace_range(start..end, &self.insert);
    }
}

#[derive(Default, Debug, Clone)]
pub struct Edit {
    indels: Vec<Indel>,
}
impl Edit {
    pub fn builder() -> EditBuilder {
        EditBuilder::default()
    }
    pub fn insert(offset: TextSize, text: String) -> Edit {
        let mut builder = Edit::builder();
        builder.insert(offset, text);
        builder.finish()
    }
    pub fn delete(range: TextRange) -> Edit {
        let mut builder = Edit::builder();
        builder.delete(range);
        builder.finish()
    }
    pub fn replace(range: TextRange, replace_with: String) -> Edit {
        let mut builder = Edit::builder();
        builder.replace(range, replace_with);
        builder.finish()
    }
    pub fn len(&self) -> usize {
        self.indels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.indels.is_empty()
    }
    pub fn iter(&self) -> std::slice::Iter<'_, Indel> {
        self.into_iter()
    }
    pub fn apply(&self, text: &mut String) {
        match self.len() {
            0 => return,
            1 => {
                self.indels[0].apply(text);
                return;
            },
            _ => (),
        }
        let text_size = TextSize::of(&*text);
        let mut total_len = text_size;
        let mut max_total_len = text_size;
        for indel in &self.indels {
            total_len += TextSize::of(&indel.insert);
            total_len -= indel.delete.len();
            max_total_len = max(max_total_len, total_len);
        }
        if let Some(additional) = max_total_len.checked_sub(text_size) {
            text.reserve(additional.into());
        }
        for indel in self.indels.iter().rev() {
            indel.apply(text);
        }
        assert_eq!(TextSize::of(&*text), total_len);
    }
    pub fn union(&mut self, other: Edit) -> Result<(), Edit> {
        let iter_merge = self
            .iter()
            .merge_by(other.iter(), |l, r| l.delete.start() <= r.delete.start());
        if !check_disjoint(&mut iter_merge.clone()) {
            return Err(other);
        }
        // Only dedup deletions and replacements, keep all insertions
        self.indels = iter_merge
            .dedup_by(|a, b| a == b && !a.delete.is_empty())
            .cloned()
            .collect();
        Ok(())
    }
    pub fn apply_to_offset(&self, offset: TextSize) -> Option<TextSize> {
        let mut res = offset;
        for indel in &self.indels {
            if indel.delete.start() >= offset {
                break;
            }
            if offset < indel.delete.end() {
                return None;
            }
            res += TextSize::of(&indel.insert);
            res -= indel.delete.len();
        }
        Some(res)
    }
}
impl IntoIterator for Edit {
    type Item = Indel;
    type IntoIter = std::vec::IntoIter<Indel>;
    fn into_iter(self) -> Self::IntoIter {
        self.indels.into_iter()
    }
}
impl<'a> IntoIterator for &'a Edit {
    type Item = &'a Indel;
    type IntoIter = std::slice::Iter<'a, Indel>;
    fn into_iter(self) -> Self::IntoIter {
        self.indels.iter()
    }
}

#[derive(Debug, Default, Clone)]
pub struct EditBuilder {
    indels: Vec<Indel>,
}
impl EditBuilder {
    pub fn is_empty(&self) -> bool {
        self.indels.is_empty()
    }
    pub fn replace(&mut self, range: TextRange, replace_with: String) {
        self.indel(Indel::replace(range, replace_with));
    }
    pub fn delete(&mut self, range: TextRange) {
        self.indel(Indel::delete(range));
    }
    pub fn insert(&mut self, offset: TextSize, text: String) {
        self.indel(Indel::insert(offset, text));
    }
    pub fn finish(self) -> Edit {
        let mut indels = self.indels;
        assert_disjoint_or_equal(&mut indels);
        indels = coalesce_indels(indels);
        Edit { indels }
    }
    pub fn invalidates_offset(&self, offset: TextSize) -> bool {
        self.indels.iter().any(|indel| indel.delete.contains_inclusive(offset))
    }
    fn indel(&mut self, indel: Indel) {
        self.indels.push(indel);
        if self.indels.len() <= 16 {
            assert_disjoint_or_equal(&mut self.indels);
        }
    }
}

fn assert_disjoint_or_equal(indels: &mut [Indel]) {
    assert!(check_disjoint_and_sort(indels));
}
fn check_disjoint_and_sort(indels: &mut [Indel]) -> bool {
    indels.sort_by_key(|indel| (indel.delete.start(), indel.delete.end()));
    check_disjoint(&mut indels.iter())
}
fn check_disjoint<'a, I>(indels: &mut I) -> bool
where
    I: std::iter::Iterator<Item = &'a Indel> + Clone,
{
    indels
        .clone()
        .zip(indels.skip(1))
        .all(|(l, r)| l.delete.end() <= r.delete.start() || l == r)
}
fn coalesce_indels(indels: Vec<Indel>) -> Vec<Indel> {
    indels
        .into_iter()
        .coalesce(|mut a, b| {
            if a.delete.end() == b.delete.start() {
                a.insert.push_str(&b.insert);
                a.delete = TextRange::new(a.delete.start(), b.delete.end());
                Ok(a)
            } else {
                Err((a, b))
            }
        })
        .collect_vec()
}

#[cfg(test)]
mod tests {
    use super::{Edit, EditBuilder, TextRange};

    fn range(start: u32, end: u32) -> TextRange {
        TextRange::new(start.into(), end.into())
    }
    #[test]
    fn test_apply() {
        let mut text = "_11h1_2222_xx3333_4444_6666".to_string();
        let mut builder = EditBuilder::default();
        builder.replace(range(3, 4), "1".to_string());
        builder.delete(range(11, 13));
        builder.insert(22.into(), "_5555".to_string());
        let text_edit = builder.finish();
        text_edit.apply(&mut text);
        assert_eq!(text, "_1111_2222_3333_4444_5555_6666")
    }
    #[test]
    fn test_union() {
        let mut edit1 = Edit::delete(range(7, 11));
        let mut builder = EditBuilder::default();
        builder.delete(range(1, 5));
        builder.delete(range(13, 17));
        let edit2 = builder.finish();
        assert!(edit1.union(edit2).is_ok());
        assert_eq!(edit1.indels.len(), 3);
    }
    #[test]
    fn test_union_with_duplicates() {
        let mut builder1 = EditBuilder::default();
        builder1.delete(range(7, 11));
        builder1.delete(range(13, 17));
        let mut builder2 = EditBuilder::default();
        builder2.delete(range(1, 5));
        builder2.delete(range(13, 17));
        let mut edit1 = builder1.finish();
        let edit2 = builder2.finish();
        assert!(edit1.union(edit2).is_ok());
        assert_eq!(edit1.indels.len(), 3);
    }
    #[test]
    fn test_union_panics() {
        let mut edit1 = Edit::delete(range(7, 11));
        let edit2 = Edit::delete(range(9, 13));
        assert!(edit1.union(edit2).is_err());
    }
    #[test]
    fn test_coalesce_disjoint() {
        let mut builder = EditBuilder::default();
        builder.replace(range(1, 3), "aa".into());
        builder.replace(range(5, 7), "bb".into());
        let edit = builder.finish();
        assert_eq!(edit.indels.len(), 2);
    }
    #[test]
    fn test_coalesce_adjacent() {
        let mut builder = EditBuilder::default();
        builder.replace(range(1, 3), "aa".into());
        builder.replace(range(3, 5), "bb".into());
        let edit = builder.finish();
        assert_eq!(edit.indels.len(), 1);
        assert_eq!(edit.indels[0].insert, "aabb");
        assert_eq!(edit.indels[0].delete, range(1, 5));
    }
    #[test]
    fn test_coalesce_adjacent_series() {
        let mut builder = EditBuilder::default();
        builder.replace(range(1, 3), "au".into());
        builder.replace(range(3, 5), "www".into());
        builder.replace(range(5, 8), "".into());
        builder.replace(range(8, 9), "ub".into());
        let edit = builder.finish();
        assert_eq!(edit.indels.len(), 1);
        assert_eq!(edit.indels[0].insert, "auwwwub");
        assert_eq!(edit.indels[0].delete, range(1, 9));
    }
}
