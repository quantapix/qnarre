use super::*;
use std::{
    mem::ManuallyDrop,
    ops::{Deref, DerefMut, Index, IndexMut},
    option, slice, vec,
};

mod iter {
    use super::*;
    pub trait Trait<'a, T: 'a>: Iterator<Item = &'a T> + DoubleEndedIterator + ExactSizeIterator {
        fn clone_box(&self) -> Box<NoDrop<dyn Trait<'a, T> + 'a>>;
    }
}

pub struct Puncted<T, P> {
    inner: Vec<(T, P)>,
    last: Option<Box<T>>,
}
impl<T, P> Puncted<T, P> {
    pub const fn new() -> Self {
        Puncted {
            inner: Vec::new(),
            last: None,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0 && self.last.is_none()
    }
    pub fn len(&self) -> usize {
        self.inner.len() + if self.last.is_some() { 1 } else { 0 }
    }
    pub fn first(&self) -> Option<&T> {
        self.iter().next()
    }
    pub fn first_mut(&mut self) -> Option<&mut T> {
        self.iter_mut().next()
    }
    pub fn last(&self) -> Option<&T> {
        self.iter().next_back()
    }
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.iter_mut().next_back()
    }
    pub fn iter(&self) -> Iter<T> {
        Iter {
            inner: Box::new(NoDrop::new(PrivateIter {
                inner: self.inner.iter(),
                last: self.last.as_ref().map(Box::as_ref).into_iter(),
            })),
        }
    }
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            inner: Box::new(NoDrop::new(PrivateIterMut {
                inner: self.inner.iter_mut(),
                last: self.last.as_mut().map(Box::as_mut).into_iter(),
            })),
        }
    }
    pub fn pairs(&self) -> Pairs<T, P> {
        Pairs {
            inner: self.inner.iter(),
            last: self.last.as_ref().map(Box::as_ref).into_iter(),
        }
    }
    pub fn pairs_mut(&mut self) -> PairsMut<T, P> {
        PairsMut {
            inner: self.inner.iter_mut(),
            last: self.last.as_mut().map(Box::as_mut).into_iter(),
        }
    }
    pub fn into_pairs(self) -> IntoPairs<T, P> {
        IntoPairs {
            inner: self.inner.into_iter(),
            last: self.last.map(|t| *t).into_iter(),
        }
    }
    pub fn push_value(&mut self, value: T) {
        assert!(
            self.empty_or_trailing(),
            "Puncted::push_value: cannot push value if Puncted is missing trailing punctuation",
        );
        self.last = Some(Box::new(value));
    }
    pub fn push_punct(&mut self, x: P) {
        assert!(
            self.last.is_some(),
            "Puncted::push_punct: cannot push punctuation if Puncted is empty or already has trailing punctuation",
        );
        let last = self.last.take().unwrap();
        self.inner.push((*last, x));
    }
    pub fn pop(&mut self) -> Option<Pair<T, P>> {
        if self.last.is_some() {
            self.last.take().map(|t| Pair::End(*t))
        } else {
            self.inner.pop().map(|(t, p)| Pair::Puncted(t, p))
        }
    }
    pub fn pop_punct(&mut self) -> Option<P> {
        if self.last.is_some() {
            None
        } else {
            let (t, p) = self.inner.pop()?;
            self.last = Some(Box::new(t));
            Some(p)
        }
    }
    pub fn trailing_punct(&self) -> bool {
        self.last.is_none() && !self.is_empty()
    }
    pub fn empty_or_trailing(&self) -> bool {
        self.last.is_none()
    }
    pub fn push(&mut self, value: T)
    where
        P: Default,
    {
        if !self.empty_or_trailing() {
            self.push_punct(Default::default());
        }
        self.push_value(value);
    }
    pub fn insert(&mut self, index: usize, value: T)
    where
        P: Default,
    {
        assert!(index <= self.len(), "Puncted::insert: index out of range",);
        if index == self.len() {
            self.push(value);
        } else {
            self.inner.insert(index, (value, Default::default()));
        }
    }
    pub fn clear(&mut self) {
        self.inner.clear();
        self.last = None;
    }

    pub fn parse_terminated(x: Stream) -> Res<Self>
    where
        T: Parse,
        P: Parse,
    {
        Self::parse_terminated_with(x, T::parse)
    }

    pub fn parse_terminated_with(x: Stream, f: fn(Stream) -> Res<T>) -> Res<Self>
    where
        P: Parse,
    {
        let mut ys = Puncted::new();
        loop {
            if x.is_empty() {
                break;
            }
            let y = f(x)?;
            ys.push_value(y);
            if x.is_empty() {
                break;
            }
            let y = x.parse()?;
            ys.push_punct(y);
        }
        Ok(ys)
    }

    pub fn parse_separated_nonempty(x: Stream) -> Res<Self>
    where
        T: Parse,
        P: Tok + Parse,
    {
        Self::parse_separated_nonempty_with(x, T::parse)
    }

    pub fn parse_separated_nonempty_with(x: Stream, f: fn(Stream) -> Res<T>) -> Res<Self>
    where
        P: Tok + Parse,
    {
        let mut ys = Puncted::new();
        loop {
            let y = f(x)?;
            ys.push_value(y);
            if !P::peek(x.cursor()) {
                break;
            }
            let y = x.parse()?;
            ys.push_punct(y);
        }
        Ok(ys)
    }
}
impl<T, P> Clone for Puncted<T, P>
where
    T: Clone,
    P: Clone,
{
    fn clone(&self) -> Self {
        Puncted {
            inner: self.inner.clone(),
            last: self.last.clone(),
        }
    }
}
impl<T: Debug, P: Debug> Debug for Puncted<T, P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut list = f.debug_list();
        for (t, p) in &self.inner {
            list.entry(t);
            list.entry(p);
        }
        if let Some(last) = &self.last {
            list.entry(last);
        }
        list.finish()
    }
}
impl<T, P> Eq for Puncted<T, P>
where
    T: Eq,
    P: Eq,
{
}
impl<T, P> PartialEq for Puncted<T, P>
where
    T: PartialEq,
    P: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        let Puncted { inner, last } = self;
        *inner == other.inner && *last == other.last
    }
}
impl<T, P> Hash for Puncted<T, P>
where
    T: Hash,
    P: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        let Puncted { inner, last } = self;
        inner.hash(state);
        last.hash(state);
    }
}
impl<T, P> FromIterator<T> for Puncted<T, P>
where
    P: Default,
{
    fn from_iter<I: IntoIterator<Item = T>>(i: I) -> Self {
        let mut ret = Puncted::new();
        ret.extend(i);
        ret
    }
}
impl<T, P> Extend<T> for Puncted<T, P>
where
    P: Default,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, i: I) {
        for value in i {
            self.push(value);
        }
    }
}
impl<T, P> FromIterator<Pair<T, P>> for Puncted<T, P> {
    fn from_iter<I: IntoIterator<Item = Pair<T, P>>>(i: I) -> Self {
        let mut y = Puncted::new();
        do_extend(&mut y, i.into_iter());
        y
    }
}
impl<T, P> Extend<Pair<T, P>> for Puncted<T, P>
where
    P: Default,
{
    fn extend<I: IntoIterator<Item = Pair<T, P>>>(&mut self, i: I) {
        if !self.empty_or_trailing() {
            self.push_punct(P::default());
        }
        do_extend(self, i.into_iter());
    }
}
impl<T, P> IntoIterator for Puncted<T, P> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        let mut ys = Vec::with_capacity(self.len());
        ys.extend(self.inner.into_iter().map(|x| x.0));
        ys.extend(self.last.map(|t| *t));
        IntoIter { inner: ys.into_iter() }
    }
}
impl<'a, T, P> IntoIterator for &'a Puncted<T, P> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        Puncted::iter(self)
    }
}
impl<'a, T, P> IntoIterator for &'a mut Puncted<T, P> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        Puncted::iter_mut(self)
    }
}
impl<T, P> Default for Puncted<T, P> {
    fn default() -> Self {
        Puncted::new()
    }
}
impl<T, P> Index<usize> for Puncted<T, P> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        if index == self.len() - 1 {
            match &self.last {
                Some(t) => t,
                None => &self.inner[index].0,
            }
        } else {
            &self.inner[index].0
        }
    }
}
impl<T, P> IndexMut<usize> for Puncted<T, P> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index == self.len() - 1 {
            match &mut self.last {
                Some(t) => t,
                None => &mut self.inner[index].0,
            }
        } else {
            &mut self.inner[index].0
        }
    }
}
impl<T, P> Lower for Puncted<T, P>
where
    T: Lower,
    P: Lower,
{
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.pairs());
    }
}

fn do_extend<T, P, I>(punctuated: &mut Puncted<T, P>, i: I)
where
    I: Iterator<Item = Pair<T, P>>,
{
    let mut nomore = false;
    for pair in i {
        if nomore {
            panic!("Puncted extended with items after a Pair::End");
        }
        match pair {
            Pair::Puncted(a, b) => punctuated.inner.push((a, b)),
            Pair::End(a) => {
                punctuated.last = Some(Box::new(a));
                nomore = true;
            },
        }
    }
}

pub struct Pairs<'a, T: 'a, P: 'a> {
    inner: slice::Iter<'a, (T, P)>,
    last: option::IntoIter<&'a T>,
}
impl<'a, T, P> Iterator for Pairs<'a, T, P> {
    type Item = Pair<&'a T, &'a P>;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(t, p)| Pair::Puncted(t, p))
            .or_else(|| self.last.next().map(Pair::End))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}
impl<'a, T, P> DoubleEndedIterator for Pairs<'a, T, P> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.last
            .next()
            .map(Pair::End)
            .or_else(|| self.inner.next_back().map(|(t, p)| Pair::Puncted(t, p)))
    }
}
impl<'a, T, P> ExactSizeIterator for Pairs<'a, T, P> {
    fn len(&self) -> usize {
        self.inner.len() + self.last.len()
    }
}
impl<'a, T, P> Clone for Pairs<'a, T, P> {
    fn clone(&self) -> Self {
        Pairs {
            inner: self.inner.clone(),
            last: self.last.clone(),
        }
    }
}

pub struct PairsMut<'a, T: 'a, P: 'a> {
    inner: slice::IterMut<'a, (T, P)>,
    last: option::IntoIter<&'a mut T>,
}
impl<'a, T, P> Iterator for PairsMut<'a, T, P> {
    type Item = Pair<&'a mut T, &'a mut P>;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(t, p)| Pair::Puncted(t, p))
            .or_else(|| self.last.next().map(Pair::End))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}
impl<'a, T, P> DoubleEndedIterator for PairsMut<'a, T, P> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.last
            .next()
            .map(Pair::End)
            .or_else(|| self.inner.next_back().map(|(t, p)| Pair::Puncted(t, p)))
    }
}
impl<'a, T, P> ExactSizeIterator for PairsMut<'a, T, P> {
    fn len(&self) -> usize {
        self.inner.len() + self.last.len()
    }
}

pub struct IntoPairs<T, P> {
    inner: vec::IntoIter<(T, P)>,
    last: option::IntoIter<T>,
}
impl<T, P> Iterator for IntoPairs<T, P> {
    type Item = Pair<T, P>;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(t, p)| Pair::Puncted(t, p))
            .or_else(|| self.last.next().map(Pair::End))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}
impl<T, P> DoubleEndedIterator for IntoPairs<T, P> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.last
            .next()
            .map(Pair::End)
            .or_else(|| self.inner.next_back().map(|(t, p)| Pair::Puncted(t, p)))
    }
}
impl<T, P> ExactSizeIterator for IntoPairs<T, P> {
    fn len(&self) -> usize {
        self.inner.len() + self.last.len()
    }
}
impl<T, P> Clone for IntoPairs<T, P>
where
    T: Clone,
    P: Clone,
{
    fn clone(&self) -> Self {
        IntoPairs {
            inner: self.inner.clone(),
            last: self.last.clone(),
        }
    }
}

pub struct IntoIter<T> {
    inner: vec::IntoIter<T>,
}
impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}
impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}
impl<T> ExactSizeIterator for IntoIter<T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}
impl<T> Clone for IntoIter<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        IntoIter {
            inner: self.inner.clone(),
        }
    }
}

pub struct Iter<'a, T: 'a> {
    inner: Box<NoDrop<dyn iter::Trait<'a, T> + 'a>>,
}
impl<'a, T> Clone for Iter<'a, T> {
    fn clone(&self) -> Self {
        Iter {
            inner: self.inner.clone_box(),
        }
    }
}
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}
impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}
impl<'a, T> ExactSizeIterator for Iter<'a, T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

struct PrivateIter<'a, T: 'a, P: 'a> {
    inner: slice::Iter<'a, (T, P)>,
    last: option::IntoIter<&'a T>,
}
impl<'a, T, P> TrivialDrop for PrivateIter<'a, T, P>
where
    slice::Iter<'a, (T, P)>: TrivialDrop,
    option::IntoIter<&'a T>: TrivialDrop,
{
}
impl<'a, T, P> Iterator for PrivateIter<'a, T, P> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|pair| &pair.0).or_else(|| self.last.next())
    }
}
impl<'a, T, P> DoubleEndedIterator for PrivateIter<'a, T, P> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.last.next().or_else(|| self.inner.next_back().map(|pair| &pair.0))
    }
}
impl<'a, T, P> ExactSizeIterator for PrivateIter<'a, T, P> {
    fn len(&self) -> usize {
        self.inner.len() + self.last.len()
    }
}
impl<'a, T, P> Clone for PrivateIter<'a, T, P> {
    fn clone(&self) -> Self {
        PrivateIter {
            inner: self.inner.clone(),
            last: self.last.clone(),
        }
    }
}
impl<'a, T, I> iter::Trait<'a, T> for I
where
    T: 'a,
    I: DoubleEndedIterator<Item = &'a T> + ExactSizeIterator<Item = &'a T> + Clone + TrivialDrop + 'a,
{
    fn clone_box(&self) -> Box<NoDrop<dyn iter::Trait<'a, T> + 'a>> {
        Box::new(NoDrop::new(self.clone()))
    }
}

pub fn empty_punctuated_iter<'a, T>() -> Iter<'a, T> {
    Iter {
        inner: Box::new(NoDrop::new(std::iter::empty())),
    }
}

pub struct IterMut<'a, T: 'a> {
    inner: Box<NoDrop<dyn IterMutTrait<'a, T, Item = &'a mut T> + 'a>>,
}
trait IterMutTrait<'a, T: 'a>: DoubleEndedIterator<Item = &'a mut T> + ExactSizeIterator<Item = &'a mut T> {}
impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}
impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}
impl<'a, T> ExactSizeIterator for IterMut<'a, T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

struct PrivateIterMut<'a, T: 'a, P: 'a> {
    inner: slice::IterMut<'a, (T, P)>,
    last: option::IntoIter<&'a mut T>,
}
impl<'a, T, P> TrivialDrop for PrivateIterMut<'a, T, P>
where
    slice::IterMut<'a, (T, P)>: TrivialDrop,
    option::IntoIter<&'a mut T>: TrivialDrop,
{
}
impl<'a, T, P> Iterator for PrivateIterMut<'a, T, P> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|pair| &mut pair.0).or_else(|| self.last.next())
    }
}
impl<'a, T, P> DoubleEndedIterator for PrivateIterMut<'a, T, P> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.last
            .next()
            .or_else(|| self.inner.next_back().map(|pair| &mut pair.0))
    }
}
impl<'a, T, P> ExactSizeIterator for PrivateIterMut<'a, T, P> {
    fn len(&self) -> usize {
        self.inner.len() + self.last.len()
    }
}
impl<'a, T, I> IterMutTrait<'a, T> for I
where
    T: 'a,
    I: DoubleEndedIterator<Item = &'a mut T> + ExactSizeIterator<Item = &'a mut T> + 'a,
{
}

pub fn empty_punctuated_iter_mut<'a, T>() -> IterMut<'a, T> {
    IterMut {
        inner: Box::new(NoDrop::new(std::iter::empty())),
    }
}

pub enum Pair<T, P> {
    Puncted(T, P),
    End(T),
}
impl<T, P> Pair<T, P> {
    pub fn into_value(self) -> T {
        match self {
            Pair::Puncted(t, _) | Pair::End(t) => t,
        }
    }
    pub fn value(&self) -> &T {
        match self {
            Pair::Puncted(t, _) | Pair::End(t) => t,
        }
    }
    pub fn value_mut(&mut self) -> &mut T {
        match self {
            Pair::Puncted(t, _) | Pair::End(t) => t,
        }
    }
    pub fn punct(&self) -> Option<&P> {
        match self {
            Pair::Puncted(_, p) => Some(p),
            Pair::End(_) => None,
        }
    }
    pub fn punct_mut(&mut self) -> Option<&mut P> {
        match self {
            Pair::Puncted(_, p) => Some(p),
            Pair::End(_) => None,
        }
    }
    pub fn new(t: T, p: Option<P>) -> Self {
        match p {
            Some(p) => Pair::Puncted(t, p),
            None => Pair::End(t),
        }
    }
    pub fn into_tuple(self) -> (T, Option<P>) {
        match self {
            Pair::Puncted(t, p) => (t, Some(p)),
            Pair::End(t) => (t, None),
        }
    }
}
impl<T, P> Pair<&T, &P> {
    pub fn cloned(self) -> Pair<T, P>
    where
        T: Clone,
        P: Clone,
    {
        match self {
            Pair::Puncted(t, p) => Pair::Puncted(t.clone(), p.clone()),
            Pair::End(t) => Pair::End(t.clone()),
        }
    }
}
impl<T, P> Clone for Pair<T, P>
where
    T: Clone,
    P: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Pair::Puncted(t, p) => Pair::Puncted(t.clone(), p.clone()),
            Pair::End(t) => Pair::End(t.clone()),
        }
    }
}
impl<T, P> Copy for Pair<T, P>
where
    T: Copy,
    P: Copy,
{
}
impl<T, P> Lower for Pair<T, P>
where
    T: Lower,
    P: Lower,
{
    fn lower(&self, s: &mut Stream) {
        match self {
            Pair::Puncted(a, b) => {
                a.lower(s);
                b.lower(s);
            },
            Pair::End(a) => a.lower(s),
        }
    }
}

#[repr(transparent)]
pub struct NoDrop<T: ?Sized>(ManuallyDrop<T>);
impl<T> NoDrop<T> {
    pub fn new(x: T) -> Self
    where
        T: TrivialDrop,
    {
        NoDrop(ManuallyDrop::new(x))
    }
}
impl<T: ?Sized> Deref for NoDrop<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T: ?Sized> DerefMut for NoDrop<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub trait TrivialDrop {}
impl<T> TrivialDrop for std::iter::Empty<T> {}
impl<'a, T> TrivialDrop for slice::Iter<'a, T> {}
impl<'a, T> TrivialDrop for slice::IterMut<'a, T> {}
impl<'a, T> TrivialDrop for option::IntoIter<&'a T> {}
impl<'a, T> TrivialDrop for option::IntoIter<&'a mut T> {}

#[test]
fn test_needs_drop() {
    use std::mem::needs_drop;
    struct NeedsDrop;
    impl Drop for NeedsDrop {
        fn drop(&mut self) {}
    }
    assert!(needs_drop::<NeedsDrop>());
    assert!(!needs_drop::<std::iter::Empty<NeedsDrop>>());
    assert!(!needs_drop::<slice::Iter<NeedsDrop>>());
    assert!(!needs_drop::<slice::IterMut<NeedsDrop>>());
    assert!(!needs_drop::<option::IntoIter<&NeedsDrop>>());
    assert!(!needs_drop::<option::IntoIter<&mut NeedsDrop>>());
}
