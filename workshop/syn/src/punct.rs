use super::{
    parse::{Parse, Stream},
    tok::Tok,
};
use std::{
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    iter,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut, Index, IndexMut},
    option, slice, vec,
};

pub struct Punctuated<T, P> {
    inner: Vec<(T, P)>,
    last: Option<Box<T>>,
}
impl<T, P> Punctuated<T, P> {
    pub const fn new() -> Self {
        Punctuated {
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
            "Punctuated::push_value: cannot push value if Punctuated is missing trailing punctuation",
        );
        self.last = Some(Box::new(value));
    }
    pub fn push_punct(&mut self, punctuation: P) {
        assert!(
            self.last.is_some(),
            "Punctuated::push_punct: cannot push punctuation if Punctuated is empty or already has trailing punctuation",
        );
        let last = self.last.take().unwrap();
        self.inner.push((*last, punctuation));
    }
    pub fn pop(&mut self) -> Option<Pair<T, P>> {
        if self.last.is_some() {
            self.last.take().map(|t| Pair::End(*t))
        } else {
            self.inner.pop().map(|(t, p)| Pair::Punctuated(t, p))
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
        assert!(index <= self.len(), "Punctuated::insert: index out of range",);
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
        let mut ys = Punctuated::new();
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
        let mut ys = Punctuated::new();
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
impl<T, P> Clone for Punctuated<T, P>
where
    T: Clone,
    P: Clone,
{
    fn clone(&self) -> Self {
        Punctuated {
            inner: self.inner.clone(),
            last: self.last.clone(),
        }
    }
}
impl<T, P> Eq for Punctuated<T, P>
where
    T: Eq,
    P: Eq,
{
}
impl<T, P> PartialEq for Punctuated<T, P>
where
    T: PartialEq,
    P: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        let Punctuated { inner, last } = self;
        *inner == other.inner && *last == other.last
    }
}
impl<T, P> Hash for Punctuated<T, P>
where
    T: Hash,
    P: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        let Punctuated { inner, last } = self;
        inner.hash(state);
        last.hash(state);
    }
}
impl<T: Debug, P: Debug> Debug for Punctuated<T, P> {
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
impl<T, P> FromIterator<T> for Punctuated<T, P>
where
    P: Default,
{
    fn from_iter<I: IntoIterator<Item = T>>(i: I) -> Self {
        let mut ret = Punctuated::new();
        ret.extend(i);
        ret
    }
}
impl<T, P> Extend<T> for Punctuated<T, P>
where
    P: Default,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, i: I) {
        for value in i {
            self.push(value);
        }
    }
}
impl<T, P> FromIterator<Pair<T, P>> for Punctuated<T, P> {
    fn from_iter<I: IntoIterator<Item = Pair<T, P>>>(i: I) -> Self {
        let mut ret = Punctuated::new();
        do_extend(&mut ret, i.into_iter());
        ret
    }
}
impl<T, P> Extend<Pair<T, P>> for Punctuated<T, P>
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
impl<T, P> IntoIterator for Punctuated<T, P> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        let mut elements = Vec::with_capacity(self.len());
        elements.extend(self.inner.into_iter().map(|pair| pair.0));
        elements.extend(self.last.map(|t| *t));
        IntoIter {
            inner: elements.into_iter(),
        }
    }
}
impl<'a, T, P> IntoIterator for &'a Punctuated<T, P> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        Punctuated::iter(self)
    }
}
impl<'a, T, P> IntoIterator for &'a mut Punctuated<T, P> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        Punctuated::iter_mut(self)
    }
}
impl<T, P> Default for Punctuated<T, P> {
    fn default() -> Self {
        Punctuated::new()
    }
}
impl<T, P> Index<usize> for Punctuated<T, P> {
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
impl<T, P> IndexMut<usize> for Punctuated<T, P> {
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
impl<T, P> ToTokens for Punctuated<T, P>
where
    T: ToTokens,
    P: ToTokens,
{
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        tokens.append_all(self.pairs());
    }
}

fn do_extend<T, P, I>(punctuated: &mut Punctuated<T, P>, i: I)
where
    I: Iterator<Item = Pair<T, P>>,
{
    let mut nomore = false;
    for pair in i {
        if nomore {
            panic!("Punctuated extended with items after a Pair::End");
        }
        match pair {
            Pair::Punctuated(a, b) => punctuated.inner.push((a, b)),
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
            .map(|(t, p)| Pair::Punctuated(t, p))
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
            .or_else(|| self.inner.next_back().map(|(t, p)| Pair::Punctuated(t, p)))
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
            .map(|(t, p)| Pair::Punctuated(t, p))
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
            .or_else(|| self.inner.next_back().map(|(t, p)| Pair::Punctuated(t, p)))
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
            .map(|(t, p)| Pair::Punctuated(t, p))
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
            .or_else(|| self.inner.next_back().map(|(t, p)| Pair::Punctuated(t, p)))
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
    inner: Box<NoDrop<dyn IterTrait<'a, T> + 'a>>,
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

trait IterTrait<'a, T: 'a>: Iterator<Item = &'a T> + DoubleEndedIterator + ExactSizeIterator {
    fn clone_box(&self) -> Box<NoDrop<dyn IterTrait<'a, T> + 'a>>;
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
impl<'a, T, I> IterTrait<'a, T> for I
where
    T: 'a,
    I: DoubleEndedIterator<Item = &'a T> + ExactSizeIterator<Item = &'a T> + Clone + TrivialDrop + 'a,
{
    fn clone_box(&self) -> Box<NoDrop<dyn IterTrait<'a, T> + 'a>> {
        Box::new(NoDrop::new(self.clone()))
    }
}

pub(crate) fn empty_punctuated_iter<'a, T>() -> Iter<'a, T> {
    Iter {
        inner: Box::new(NoDrop::new(iter::empty())),
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

pub(crate) fn empty_punctuated_iter_mut<'a, T>() -> IterMut<'a, T> {
    IterMut {
        inner: Box::new(NoDrop::new(iter::empty())),
    }
}

pub enum Pair<T, P> {
    Punctuated(T, P),
    End(T),
}
impl<T, P> Pair<T, P> {
    pub fn into_value(self) -> T {
        match self {
            Pair::Punctuated(t, _) | Pair::End(t) => t,
        }
    }
    pub fn value(&self) -> &T {
        match self {
            Pair::Punctuated(t, _) | Pair::End(t) => t,
        }
    }
    pub fn value_mut(&mut self) -> &mut T {
        match self {
            Pair::Punctuated(t, _) | Pair::End(t) => t,
        }
    }
    pub fn punct(&self) -> Option<&P> {
        match self {
            Pair::Punctuated(_, p) => Some(p),
            Pair::End(_) => None,
        }
    }
    pub fn punct_mut(&mut self) -> Option<&mut P> {
        match self {
            Pair::Punctuated(_, p) => Some(p),
            Pair::End(_) => None,
        }
    }
    pub fn new(t: T, p: Option<P>) -> Self {
        match p {
            Some(p) => Pair::Punctuated(t, p),
            None => Pair::End(t),
        }
    }
    pub fn into_tuple(self) -> (T, Option<P>) {
        match self {
            Pair::Punctuated(t, p) => (t, Some(p)),
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
            Pair::Punctuated(t, p) => Pair::Punctuated(t.clone(), p.clone()),
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
            Pair::Punctuated(t, p) => Pair::Punctuated(t.clone(), p.clone()),
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
impl<T, P> ToTokens for Pair<T, P>
where
    T: ToTokens,
    P: ToTokens,
{
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        match self {
            Pair::Punctuated(a, b) => {
                a.to_tokens(tokens);
                b.to_tokens(tokens);
            },
            Pair::End(a) => a.to_tokens(tokens),
        }
    }
}

#[repr(transparent)]
pub(crate) struct NoDrop<T: ?Sized>(ManuallyDrop<T>);
impl<T> NoDrop<T> {
    pub(crate) fn new(value: T) -> Self
    where
        T: TrivialDrop,
    {
        NoDrop(ManuallyDrop::new(value))
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

pub(crate) trait TrivialDrop {}
impl<T> TrivialDrop for iter::Empty<T> {}
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
    assert!(!needs_drop::<iter::Empty<NeedsDrop>>());
    assert!(!needs_drop::<slice::Iter<NeedsDrop>>());
    assert!(!needs_drop::<slice::IterMut<NeedsDrop>>());
    assert!(!needs_drop::<option::IntoIter<&NeedsDrop>>());
    assert!(!needs_drop::<option::IntoIter<&mut NeedsDrop>>());
}
