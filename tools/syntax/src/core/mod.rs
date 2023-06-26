#![allow(unsafe_code)]

use memoffset::offset_of;
use std::{
    alloc::{self, Layout},
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    iter::successors,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    ops::{self, AddAssign, Deref},
    ptr,
    sync::atomic::{
        self,
        Ordering::{Acquire, Relaxed, Release},
    },
};
pub use text_size::{TextLen, TextRange, TextSize};

pub mod api;
#[allow(unsafe_code)]
pub mod cursor;
#[allow(unsafe_code)]
pub mod green;

pub mod ast {
    use super::api;

    pub trait Node {
        type Lang: api::Lang;
        fn can_cast(x: <Self::Lang as api::Lang>::Kind) -> bool
        where
            Self: Sized;
        fn cast(x: api::Node<Self::Lang>) -> Option<Self>
        where
            Self: Sized;
        fn syntax(&self) -> &api::Node<Self::Lang>;
        fn clone_for_update(&self) -> Self
        where
            Self: Sized,
        {
            Self::cast(self.syntax().clone_for_update()).unwrap()
        }
        fn clone_subtree(&self) -> Self
        where
            Self: Sized,
        {
            Self::cast(self.syntax().clone_subtree()).unwrap()
        }
    }

    pub struct NodePtr<N: Node> {
        raw: NodePtr<N::Lang>,
    }
    impl<N: Node> NodePtr<N> {
        pub fn new(x: &N) -> Self {
            Self {
                raw: NodePtr::new(x.syntax()),
            }
        }
        pub fn to_node(&self, x: &api::Node<N::Lang>) -> N {
            N::cast(self.raw.to_node(x)).unwrap()
        }
        pub fn syntax_node_ptr(&self) -> NodePtr<N::Lang> {
            self.raw.clone()
        }
        pub fn cast<U: Node<Lang = N::Lang>>(self) -> Option<NodePtr<U>> {
            if !U::can_cast(self.raw.kind) {
                return None;
            }
            Some(NodePtr { raw: self.raw })
        }
    }
    impl<N: Node> Clone for NodePtr<N> {
        fn clone(&self) -> Self {
            Self { raw: self.raw.clone() }
        }
    }
    impl<N: Node> Eq for NodePtr<N> {}
    impl<N: Node> PartialEq for NodePtr<N> {
        fn eq(&self, other: &NodePtr<N>) -> bool {
            self.raw == other.raw
        }
    }
    impl<N: Node> Hash for NodePtr<N> {
        fn hash<H: Hasher>(&self, x: &mut H) {
            self.raw.hash(x)
        }
    }
    impl<N: Node> fmt::Debug for NodePtr<N> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("ast::NodePtr").field("raw", &self.raw).finish()
        }
    }
    impl<N: Node> From<NodePtr<N>> for NodePtr<N::Lang> {
        fn from(x: NodePtr<N>) -> NodePtr<N::Lang> {
            x.raw
        }
    }

    #[derive(Debug, Clone)]
    pub struct Children<N: Node> {
        inner: api::NodeChildren<N::Lang>,
        ph: PhantomData<N>,
    }
    impl<N: Node> Children<N> {
        pub fn new(x: &api::Node<N::Lang>) -> Self {
            Children {
                inner: x.children(),
                ph: PhantomData,
            }
        }
    }
    impl<N: Node> Iterator for Children<N> {
        type Item = N;
        fn next(&mut self) -> Option<N> {
            self.inner.find_map(N::cast)
        }
    }
}

pub mod support {
    use super::{api, ast};
    pub fn child<N: ast::Node>(x: &api::Node<N::Lang>) -> Option<N> {
        x.children().find_map(N::cast)
    }
    pub fn children<N: ast::Node>(x: &api::Node<N::Lang>) -> ast::Children<N> {
        ast::Children::new(x)
    }
    pub fn token<L: api::Lang>(x: &api::Node<L>, kind: L::Kind) -> Option<api::Token<L>> {
        x.children_with_tokens()
            .filter_map(|x| x.into_token())
            .find(|x| x.kind() == kind)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodePtr<L: api::Lang> {
    kind: L::Kind,
    range: TextRange,
}
impl<L: api::Lang> NodePtr<L> {
    pub fn new(x: &api::Node<L>) -> Self {
        Self {
            kind: x.kind(),
            range: x.text_range(),
        }
    }
    pub fn to_node(&self, x: &api::Node<L>) -> api::Node<L> {
        assert!(x.parent().is_none());
        successors(Some(x.clone()), |x| {
            x.child_or_token_at_range(self.range).and_then(|x| x.into_node())
        })
        .find(|x| x.text_range() == self.range && x.kind() == self.kind)
        .unwrap_or_else(|| panic!("can't resolve local ptr to api::Node: {:?}", self))
    }
    pub fn cast<N: ast::Node<Lang = L>>(self) -> Option<ast::NodePtr<N>> {
        if !N::can_cast(self.kind) {
            return None;
        }
        Some(ast::NodePtr { raw: self })
    }
    pub fn kind(&self) -> L::Kind {
        self.kind
    }
    pub fn text_range(&self) -> TextRange {
        self.range
    }
}

#[derive(Clone)]
pub struct SyntaxText {
    node: crate::Node,
    range: TextRange,
}
impl SyntaxText {
    pub fn new(node: crate::Node) -> SyntaxText {
        let range = node.text_range();
        SyntaxText { node, range }
    }
    pub fn len(&self) -> TextSize {
        self.range.len()
    }
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }
    pub fn contains_char(&self, c: char) -> bool {
        self.try_for_each_chunk(|x| if x.contains(c) { Err(()) } else { Ok(()) })
            .is_err()
    }
    pub fn find_char(&self, c: char) -> Option<TextSize> {
        let mut acc: TextSize = 0.into();
        let y = self.try_for_each_chunk(|x| {
            if let Some(x) = x.find(c) {
                let x: TextSize = (x as u32).into();
                return Err(acc + x);
            }
            acc += TextSize::of(x);
            Ok(())
        });
        found(y)
    }
    pub fn char_at(&self, off: TextSize) -> Option<char> {
        let off = off.into();
        let mut start: TextSize = 0.into();
        let y = self.try_for_each_chunk(|chunk| {
            let end = start + TextSize::of(chunk);
            if start <= off && off < end {
                let off: usize = u32::from(off - start) as usize;
                return Err(chunk[off..].chars().next().unwrap());
            }
            start = end;
            Ok(())
        });
        found(y)
    }
    pub fn slice<R: SyntaxTextRange>(&self, range: R) -> SyntaxText {
        let start = range.start().unwrap_or_default();
        let end = range.end().unwrap_or(self.len());
        assert!(start <= end);
        let len = end - start;
        let start = self.range.start() + start;
        let end = start + len;
        assert!(
            start <= end,
            "invalid slice, range: {:?}, slice: {:?}",
            self.range,
            (range.start(), range.end()),
        );
        let range = TextRange::new(start, end);
        assert!(
            self.range.contains_range(range),
            "invalid slice, range: {:?}, slice: {:?}",
            self.range,
            range,
        );
        SyntaxText {
            node: self.node.clone(),
            range,
        }
    }
    pub fn try_fold_chunks<T, F, E>(&self, init: T, mut f: F) -> Result<T, E>
    where
        F: FnMut(T, &str) -> Result<T, E>,
    {
        self.tokens_with_ranges()
            .try_fold(init, move |acc, (token, range)| f(acc, &token.text()[range]))
    }
    pub fn try_for_each_chunk<F: FnMut(&str) -> Result<(), E>, E>(&self, mut f: F) -> Result<(), E> {
        self.try_fold_chunks((), move |(), chunk| f(chunk))
    }
    pub fn for_each_chunk<F: FnMut(&str)>(&self, mut f: F) {
        enum Void {}
        match self.try_for_each_chunk(|chunk| Ok::<(), Void>(f(chunk))) {
            Ok(()) => (),
            Err(void) => match void {},
        }
    }
    fn tokens_with_ranges(&self) -> impl Iterator<Item = (crate::Token, TextRange)> {
        let text_range = self.range;
        self.node
            .descendants_with_tokens()
            .filter_map(|element| element.into_token())
            .filter_map(move |token| {
                let token_range = token.text_range();
                let range = text_range.intersect(token_range)?;
                Some((token, range - token_range.start()))
            })
    }
}
impl Eq for SyntaxText {}
impl PartialEq<str> for SyntaxText {
    fn eq(&self, mut rhs: &str) -> bool {
        self.try_for_each_chunk(|chunk| {
            if !rhs.starts_with(chunk) {
                return Err(());
            }
            rhs = &rhs[chunk.len()..];
            Ok(())
        })
        .is_ok()
            && rhs.is_empty()
    }
}
impl PartialEq<&'_ str> for SyntaxText {
    fn eq(&self, rhs: &&str) -> bool {
        self == *rhs
    }
}
impl PartialEq for SyntaxText {
    fn eq(&self, other: &SyntaxText) -> bool {
        if self.range.len() != other.range.len() {
            return false;
        }
        let mut lhs = self.tokens_with_ranges();
        let mut rhs = other.tokens_with_ranges();
        zip_texts(&mut lhs, &mut rhs).is_none() && lhs.all(|it| it.1.is_empty()) && rhs.all(|it| it.1.is_empty())
    }
}
impl fmt::Debug for SyntaxText {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.to_string(), f)
    }
}
impl fmt::Display for SyntaxText {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.try_for_each_chunk(|chunk| fmt::Display::fmt(chunk, f))
    }
}

fn found<T>(res: Result<(), T>) -> Option<T> {
    match res {
        Ok(()) => None,
        Err(it) => Some(it),
    }
}
fn zip_texts<I: Iterator<Item = (crate::Token, TextRange)>>(xs: &mut I, ys: &mut I) -> Option<()> {
    let mut x = xs.next()?;
    let mut y = ys.next()?;
    loop {
        while x.1.is_empty() {
            x = xs.next()?;
        }
        while y.1.is_empty() {
            y = ys.next()?;
        }
        let x_text = &x.0.text()[x.1];
        let y_text = &y.0.text()[y.1];
        if !(x_text.starts_with(y_text) || y_text.starts_with(x_text)) {
            return Some(());
        }
        let advance = std::cmp::min(x.1.len(), y.1.len());
        x.1 = TextRange::new(x.1.start() + advance, x.1.end());
        y.1 = TextRange::new(y.1.start() + advance, y.1.end());
    }
}

impl From<SyntaxText> for String {
    fn from(text: SyntaxText) -> String {
        text.to_string()
    }
}
impl PartialEq<SyntaxText> for str {
    fn eq(&self, rhs: &SyntaxText) -> bool {
        rhs == self
    }
}
impl PartialEq<SyntaxText> for &'_ str {
    fn eq(&self, rhs: &SyntaxText) -> bool {
        rhs == self
    }
}

trait SyntaxTextRange {
    fn start(&self) -> Option<TextSize>;
    fn end(&self) -> Option<TextSize>;
}
impl SyntaxTextRange for TextRange {
    fn start(&self) -> Option<TextSize> {
        Some(TextRange::start(*self))
    }
    fn end(&self) -> Option<TextSize> {
        Some(TextRange::end(*self))
    }
}
impl SyntaxTextRange for ops::Range<TextSize> {
    fn start(&self) -> Option<TextSize> {
        Some(self.start)
    }
    fn end(&self) -> Option<TextSize> {
        Some(self.end)
    }
}
impl SyntaxTextRange for ops::RangeFrom<TextSize> {
    fn start(&self) -> Option<TextSize> {
        Some(self.start)
    }
    fn end(&self) -> Option<TextSize> {
        None
    }
}
impl SyntaxTextRange for ops::RangeTo<TextSize> {
    fn start(&self) -> Option<TextSize> {
        None
    }
    fn end(&self) -> Option<TextSize> {
        Some(self.end)
    }
}
impl SyntaxTextRange for ops::RangeFull {
    fn start(&self) -> Option<TextSize> {
        None
    }
    fn end(&self) -> Option<TextSize> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeOrToken<N, T> {
    Node(N),
    Token(T),
}
impl<N, T> NodeOrToken<N, T> {
    pub fn into_node(self) -> Option<N> {
        match self {
            NodeOrToken::Node(node) => Some(node),
            NodeOrToken::Token(_) => None,
        }
    }
    pub fn into_token(self) -> Option<T> {
        match self {
            NodeOrToken::Node(_) => None,
            NodeOrToken::Token(token) => Some(token),
        }
    }
    pub fn as_node(&self) -> Option<&N> {
        match self {
            NodeOrToken::Node(node) => Some(node),
            NodeOrToken::Token(_) => None,
        }
    }
    pub fn as_token(&self) -> Option<&T> {
        match self {
            NodeOrToken::Node(_) => None,
            NodeOrToken::Token(token) => Some(token),
        }
    }
}
impl<N: Deref, T: Deref> NodeOrToken<N, T> {
    pub fn as_deref(&self) -> NodeOrToken<&N::Target, &T::Target> {
        match self {
            NodeOrToken::Node(node) => NodeOrToken::Node(&*node),
            NodeOrToken::Token(token) => NodeOrToken::Token(&*token),
        }
    }
}
impl<N: fmt::Display, T: fmt::Display> fmt::Display for NodeOrToken<N, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeOrToken::Node(node) => fmt::Display::fmt(node, f),
            NodeOrToken::Token(token) => fmt::Display::fmt(token, f),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Direction {
    Next,
    Prev,
}

#[derive(Debug, Copy, Clone)]
pub enum WalkEvent<T> {
    Enter(T),
    Leave(T),
}
impl<T> WalkEvent<T> {
    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> WalkEvent<U> {
        match self {
            WalkEvent::Enter(it) => WalkEvent::Enter(f(it)),
            WalkEvent::Leave(it) => WalkEvent::Leave(f(it)),
        }
    }
}

#[derive(Clone, Debug)]
pub enum TokenAtOffset<T> {
    None,
    Single(T),
    Between(T, T),
}
impl<T> TokenAtOffset<T> {
    pub fn map<F: Fn(T) -> U, U>(self, f: F) -> TokenAtOffset<U> {
        match self {
            TokenAtOffset::None => TokenAtOffset::None,
            TokenAtOffset::Single(it) => TokenAtOffset::Single(f(it)),
            TokenAtOffset::Between(l, r) => TokenAtOffset::Between(f(l), f(r)),
        }
    }
    pub fn right_biased(self) -> Option<T> {
        match self {
            TokenAtOffset::None => None,
            TokenAtOffset::Single(node) => Some(node),
            TokenAtOffset::Between(_, right) => Some(right),
        }
    }
    pub fn left_biased(self) -> Option<T> {
        match self {
            TokenAtOffset::None => None,
            TokenAtOffset::Single(node) => Some(node),
            TokenAtOffset::Between(left, _) => Some(left),
        }
    }
}
impl<T> Iterator for TokenAtOffset<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        match std::mem::replace(self, TokenAtOffset::None) {
            TokenAtOffset::None => None,
            TokenAtOffset::Single(node) => {
                *self = TokenAtOffset::None;
                Some(node)
            },
            TokenAtOffset::Between(left, right) => {
                *self = TokenAtOffset::Single(right);
                Some(left)
            },
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            TokenAtOffset::None => (0, Some(0)),
            TokenAtOffset::Single(_) => (1, Some(1)),
            TokenAtOffset::Between(_, _) => (2, Some(2)),
        }
    }
}
impl<T> ExactSizeIterator for TokenAtOffset<T> {}

macro_rules! _static_assert {
    ($e:expr) => {
        const _: i32 = 0 / $e as i32;
    };
}
pub use _static_assert as static_assert;

#[derive(Copy, Clone, Debug)]
pub enum Delta<T> {
    Add(T),
    Sub(T),
}

macro_rules! impls {
        ($($ty:ident)*) => {$(
            impl AddAssign<Delta<$ty>> for $ty {
                fn add_assign(&mut self, rhs: Delta<$ty>) {
                    match rhs {
                        Delta::Add(x) => *self += x,
                        Delta::Sub(x) => *self -= x,
                    }
                }
            }
        )*};
    }
impls!(u32 TextSize);

#[derive(Debug)]
pub enum CowMut<'a, T> {
    Owned(T),
    Borrowed(&'a mut T),
}
impl<T> std::ops::Deref for CowMut<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            CowMut::Owned(it) => it,
            CowMut::Borrowed(it) => *it,
        }
    }
}
impl<T> std::ops::DerefMut for CowMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        match self {
            CowMut::Owned(it) => it,
            CowMut::Borrowed(it) => *it,
        }
    }
}
impl<T: Default> Default for CowMut<'_, T> {
    fn default() -> Self {
        CowMut::Owned(T::default())
    }
}

const MAX_REFCOUNT: usize = (isize::MAX) as usize;

#[repr(C)]
pub struct ArcInner<T: ?Sized> {
    pub count: atomic::AtomicUsize,
    pub data: T,
}
unsafe impl<T: ?Sized + Sync + Send> Send for ArcInner<T> {}
unsafe impl<T: ?Sized + Sync + Send> Sync for ArcInner<T> {}

#[repr(transparent)]
pub struct Arc<T: ?Sized> {
    pub p: ptr::NonNull<ArcInner<T>>,
    pub phantom: PhantomData<T>,
}
impl<T> Arc<T> {
    #[inline]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        let ptr = (ptr as *const u8).sub(offset_of!(ArcInner<T>, data));
        Arc {
            p: ptr::NonNull::new_unchecked(ptr as *mut ArcInner<T>),
            phantom: PhantomData,
        }
    }
}
impl<T: ?Sized> Arc<T> {
    #[inline]
    fn inner(&self) -> &ArcInner<T> {
        unsafe { &*self.ptr() }
    }
    #[inline(never)]
    unsafe fn drop_slow(&mut self) {
        let _ = Box::from_raw(self.ptr());
    }
    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr() == other.ptr()
    }
    pub fn ptr(&self) -> *mut ArcInner<T> {
        self.p.as_ptr()
    }
}
impl<T: ?Sized> Arc<T> {
    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if this.is_unique() {
            unsafe { Some(&mut (*this.ptr()).data) }
        } else {
            None
        }
    }
    pub fn is_unique(&self) -> bool {
        self.inner().count.load(Acquire) == 1
    }
}
impl<H, T> Arc<HeaderSlice<H, [T]>> {
    #[inline]
    pub fn into_thin(a: Self) -> ThinArc<H, T> {
        assert_eq!(
            a.length,
            a.slice.len(),
            "Length needs to be correct for ThinArc to work"
        );
        let fat_ptr: *mut ArcInner<HeaderSlice<H, [T]>> = a.ptr();
        mem::forget(a);
        let thin_ptr = fat_ptr as *mut [usize] as *mut usize;
        ThinArc {
            ptr: unsafe { ptr::NonNull::new_unchecked(thin_ptr as *mut ArcInner<HeaderSlice<H, [T; 0]>>) },
            phantom: PhantomData,
        }
    }
    #[inline]
    pub fn from_thin(a: ThinArc<H, T>) -> Self {
        let ptr = thin_to_thick(a.ptr.as_ptr());
        mem::forget(a);
        unsafe {
            Arc {
                p: ptr::NonNull::new_unchecked(ptr),
                phantom: PhantomData,
            }
        }
    }
}
impl<T: ?Sized> Clone for Arc<T> {
    #[inline]
    fn clone(&self) -> Self {
        let old_size = self.inner().count.fetch_add(1, Relaxed);
        if old_size > MAX_REFCOUNT {
            std::process::abort();
        }
        unsafe {
            Arc {
                p: ptr::NonNull::new_unchecked(self.ptr()),
                phantom: PhantomData,
            }
        }
    }
}
impl<T: ?Sized> Deref for Arc<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.inner().data
    }
}
impl<T: ?Sized> Drop for Arc<T> {
    #[inline]
    fn drop(&mut self) {
        if self.inner().count.fetch_sub(1, Release) != 1 {
            return;
        }
        self.inner().count.load(Acquire);
        unsafe {
            self.drop_slow();
        }
    }
}
impl<T: ?Sized + PartialEq> PartialEq for Arc<T> {
    fn eq(&self, other: &Arc<T>) -> bool {
        Self::ptr_eq(self, other) || *(*self) == *(*other)
    }
    fn ne(&self, other: &Arc<T>) -> bool {
        !Self::ptr_eq(self, other) && *(*self) != *(*other)
    }
}
impl<T: ?Sized + PartialOrd> PartialOrd for Arc<T> {
    fn partial_cmp(&self, other: &Arc<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
    fn lt(&self, other: &Arc<T>) -> bool {
        *(*self) < *(*other)
    }
    fn le(&self, other: &Arc<T>) -> bool {
        *(*self) <= *(*other)
    }
    fn gt(&self, other: &Arc<T>) -> bool {
        *(*self) > *(*other)
    }
    fn ge(&self, other: &Arc<T>) -> bool {
        *(*self) >= *(*other)
    }
}
impl<T: ?Sized + Ord> Ord for Arc<T> {
    fn cmp(&self, other: &Arc<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}
impl<T: ?Sized + Eq> Eq for Arc<T> {}
impl<T: ?Sized + Hash> Hash for Arc<T> {
    fn hash<H: Hasher>(&self, x: &mut H) {
        (**self).hash(x)
    }
}
unsafe impl<T: ?Sized + Sync + Send> Send for Arc<T> {}
unsafe impl<T: ?Sized + Sync + Send> Sync for Arc<T> {}

#[derive(Debug, Eq, PartialEq, Hash, PartialOrd)]
#[repr(C)]
pub struct HeaderSlice<H, T: ?Sized> {
    pub header: H,
    length: usize,
    slice: T,
}
impl<H, T> HeaderSlice<H, [T]> {
    pub fn slice(&self) -> &[T] {
        &self.slice
    }
}
impl<H, T> Deref for HeaderSlice<H, [T; 0]> {
    type Target = HeaderSlice<H, [T]>;
    fn deref(&self) -> &Self::Target {
        unsafe {
            let len = self.length;
            let y: *const [T] = ptr::slice_from_raw_parts(self as *const _ as *const T, len);
            &*(y as *const HeaderSlice<H, [T]>)
        }
    }
}

#[repr(transparent)]
pub struct ThinArc<H, T> {
    ptr: ptr::NonNull<ArcInner<HeaderSlice<H, [T; 0]>>>,
    phantom: PhantomData<(H, T)>,
}
impl<H, T> ThinArc<H, T> {
    #[inline]
    pub fn with_arc<F, U>(&self, f: F) -> U
    where
        F: FnOnce(&Arc<HeaderSlice<H, [T]>>) -> U,
    {
        let transient = unsafe {
            ManuallyDrop::new(Arc {
                p: ptr::NonNull::new_unchecked(thin_to_thick(self.ptr.as_ptr())),
                phantom: PhantomData,
            })
        };
        let result = f(&transient);
        result
    }
    pub fn from_header_and_iter<I>(header: H, mut items: I) -> Self
    where
        I: Iterator<Item = T> + ExactSizeIterator,
    {
        assert_ne!(mem::size_of::<T>(), 0, "Need to think about ZST");
        let num_items = items.len();
        let inner_to_data_offset = offset_of!(ArcInner<HeaderSlice<H, [T; 0]>>, data);
        let data_to_slice_offset = offset_of!(HeaderSlice<H, [T; 0]>, slice);
        let slice_offset = inner_to_data_offset + data_to_slice_offset;
        let slice_size = mem::size_of::<T>().checked_mul(num_items).expect("size overflows");
        let usable_size = slice_offset.checked_add(slice_size).expect("size overflows");
        let align = mem::align_of::<ArcInner<HeaderSlice<H, [T; 0]>>>();
        let size = usable_size.wrapping_add(align - 1) & !(align - 1);
        assert!(size >= usable_size, "size overflows");
        let layout = Layout::from_size_align(size, align).expect("invalid layout");
        let ptr: *mut ArcInner<HeaderSlice<H, [T; 0]>>;
        unsafe {
            let buffer = alloc::alloc(layout);
            if buffer.is_null() {
                alloc::handle_alloc_error(layout);
            }
            ptr = buffer as *mut _;
            let count = atomic::AtomicUsize::new(1);
            ptr::write(ptr::addr_of_mut!((*ptr).count), count);
            ptr::write(ptr::addr_of_mut!((*ptr).data.header), header);
            ptr::write(ptr::addr_of_mut!((*ptr).data.length), num_items);
            if num_items != 0 {
                let mut current = ptr::addr_of_mut!((*ptr).data.slice) as *mut T;
                debug_assert_eq!(current as usize - buffer as usize, slice_offset);
                for _ in 0..num_items {
                    ptr::write(current, items.next().expect("ExactSizeIterator over-reported length"));
                    current = current.offset(1);
                }
                assert!(items.next().is_none(), "ExactSizeIterator under-reported length");
                debug_assert_eq!(current as *mut u8, buffer.add(usable_size));
            }
            assert!(items.next().is_none(), "ExactSizeIterator under-reported length");
        }
        ThinArc {
            ptr: unsafe { ptr::NonNull::new_unchecked(ptr) },
            phantom: PhantomData,
        }
    }
}
impl<H, T> Clone for ThinArc<H, T> {
    #[inline]
    fn clone(&self) -> Self {
        ThinArc::with_arc(self, |a| Arc::into_thin(a.clone()))
    }
}
impl<H, T> Deref for ThinArc<H, T> {
    type Target = HeaderSlice<H, [T]>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &(*thin_to_thick(self.ptr.as_ptr())).data }
    }
}
impl<H, T> Drop for ThinArc<H, T> {
    #[inline]
    fn drop(&mut self) {
        let _ = Arc::from_thin(ThinArc {
            ptr: self.ptr,
            phantom: PhantomData,
        });
    }
}
impl<H: PartialEq, T: PartialEq> PartialEq for ThinArc<H, T> {
    #[inline]
    fn eq(&self, other: &ThinArc<H, T>) -> bool {
        **self == **other
    }
}
impl<H: Eq, T: Eq> Eq for ThinArc<H, T> {}
impl<H: Hash, T: Hash> Hash for ThinArc<H, T> {
    fn hash<HSR: Hasher>(&self, state: &mut HSR) {
        (**self).hash(state)
    }
}
unsafe impl<H: Sync + Send, T: Sync + Send> Send for ThinArc<H, T> {}
unsafe impl<H: Sync + Send, T: Sync + Send> Sync for ThinArc<H, T> {}

fn thin_to_thick<H, T>(thin: *mut ArcInner<HeaderSlice<H, [T; 0]>>) -> *mut ArcInner<HeaderSlice<H, [T]>> {
    let len = unsafe { (*thin).data.length };
    let fake_slice: *mut [T] = ptr::slice_from_raw_parts_mut(thin as *mut T, len);
    fake_slice as *mut ArcInner<HeaderSlice<H, [T]>>
}

//#[cfg(feature = "serde1")]
mod serde_impls {
    use super::{api, NodeOrToken};
    use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
    use std::fmt;

    struct SerDisplay<T>(T);
    impl<T: fmt::Display> Serialize for SerDisplay<T> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serializer.collect_str(&self.0)
        }
    }

    struct DisplayDebug<T>(T);
    impl<T: fmt::Debug> fmt::Display for DisplayDebug<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Debug::fmt(&self.0, f)
        }
    }
    impl<L: api::Lang> Serialize for api::Node<L> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut state = serializer.serialize_map(Some(3))?;
            state.serialize_entry("kind", &SerDisplay(DisplayDebug(self.kind())))?;
            state.serialize_entry("text_range", &self.text_range())?;
            state.serialize_entry("children", &Children(self))?;
            state.end()
        }
    }
    impl<L: api::Lang> Serialize for api::Token<L> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut state = serializer.serialize_map(Some(3))?;
            state.serialize_entry("kind", &SerDisplay(DisplayDebug(self.kind())))?;
            state.serialize_entry("text_range", &self.text_range())?;
            state.serialize_entry("text", &self.text())?;
            state.end()
        }
    }

    struct Children<T>(T);
    impl<L: api::Lang> Serialize for Children<&'_ api::Node<L>> {
        fn serialize<S>(&self, x: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut y = x.serialize_seq(None)?;
            self.0.children_with_tokens().try_for_each(|x| match x {
                NodeOrToken::Node(x) => y.serialize_element(&x),
                NodeOrToken::Token(x) => y.serialize_element(&x),
            })?;
            y.end()
        }
    }
}

#[allow(unsafe_code)]
mod sll {
    use super::Delta;
    use std::{cell::Cell, cmp::Ordering, ptr};

    pub unsafe trait Elem {
        fn prev(&self) -> &Cell<*const Self>;
        fn next(&self) -> &Cell<*const Self>;
        fn key(&self) -> &Cell<u32>;
    }
    pub enum AddToSllResult<'a, E: Elem> {
        NoHead,
        EmptyHead(&'a Cell<*const E>),
        SmallerThanHead(&'a Cell<*const E>),
        SmallerThanNotHead(*const E),
        AlreadyInSll(*const E),
    }
    impl<'a, E: Elem> AddToSllResult<'a, E> {
        pub fn add_to_sll(&self, e: *const E) {
            unsafe {
                (*e).prev().set(e);
                (*e).next().set(e);
                match self {
                    AddToSllResult::EmptyHead(x) => x.set(e),
                    AddToSllResult::SmallerThanHead(x) => {
                        let old = x.get();
                        let prev = (*old).prev().replace(e);
                        (*prev).next().set(e);
                        (*e).next().set(old);
                        (*e).prev().set(prev);
                        x.set(e);
                    },
                    AddToSllResult::SmallerThanNotHead(x) => {
                        let next = (**x).next().replace(e);
                        (*next).prev().set(e);
                        (*e).prev().set(*x);
                        (*e).next().set(next);
                    },
                    AddToSllResult::NoHead | AddToSllResult::AlreadyInSll(_) => (),
                }
            }
        }
    }
    #[cold]
    pub fn init<'a, E: Elem>(head: Option<&'a Cell<*const E>>, elem: &E) -> AddToSllResult<'a, E> {
        if let Some(head) = head {
            link(head, elem)
        } else {
            AddToSllResult::NoHead
        }
    }
    #[cold]
    pub fn unlink<E: Elem>(head: &Cell<*const E>, elem: &E) {
        debug_assert!(!head.get().is_null(), "invalid linked list head");
        let elem_ptr: *const E = elem;
        let prev = elem.prev().replace(elem_ptr);
        let next = elem.next().replace(elem_ptr);
        unsafe {
            debug_assert_eq!((*prev).next().get(), elem_ptr, "invalid linked list links");
            debug_assert_eq!((*next).prev().get(), elem_ptr, "invalid linked list links");
            (*prev).next().set(next);
            (*next).prev().set(prev);
        }
        if head.get() == elem_ptr {
            head.set(if next == elem_ptr { ptr::null() } else { next })
        }
    }
    #[cold]
    pub fn link<'a, E: Elem>(head: &'a Cell<*const E>, elem: &E) -> AddToSllResult<'a, E> {
        unsafe {
            let old_head = head.get();
            if old_head.is_null() {
                return AddToSllResult::EmptyHead(head);
            }
            if elem.key() < (*old_head).key() {
                return AddToSllResult::SmallerThanHead(head);
            }
            let mut curr = (*old_head).prev().get();
            loop {
                match (*curr).key().cmp(elem.key()) {
                    Ordering::Less => return AddToSllResult::SmallerThanNotHead(curr),
                    Ordering::Equal => return AddToSllResult::AlreadyInSll(curr),
                    Ordering::Greater => curr = (*curr).prev().get(),
                }
            }
        }
    }
    pub fn adjust<E: Elem>(elem: &E, from: u32, by: Delta<u32>) {
        let elem_ptr: *const E = elem;
        unsafe {
            let mut curr = elem_ptr;
            loop {
                let mut key = (*curr).key().get();
                if key >= from {
                    key += by;
                    (*curr).key().set(key);
                }
                curr = (*curr).next().get();
                if curr == elem_ptr {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{api, green};
    fn build_tree(xs: &[&str]) -> api::Node {
        let mut y = green::NodeBuilder::new();
        y.start_node(green::Kind(62));
        for &x in xs.iter() {
            y.token(green::Kind(92), x.into())
        }
        y.finish_node();
        api::Node::new_root(y.finish())
    }
    #[test]
    fn test_text_equality() {
        fn do_check(t1: &[&str], t2: &[&str]) {
            let t1 = build_tree(t1).text();
            let t2 = build_tree(t2).text();
            let expected = t1.to_string() == t2.to_string();
            let actual = t1 == t2;
            assert_eq!(expected, actual, "`{}` (SyntaxText) `{}` (SyntaxText)", t1, t2);
            let actual = t1 == &*t2.to_string();
            assert_eq!(expected, actual, "`{}` (SyntaxText) `{}` (&str)", t1, t2);
        }
        fn check(t1: &[&str], t2: &[&str]) {
            do_check(t1, t2);
            do_check(t2, t1)
        }
        check(&[""], &[""]);
        check(&["a"], &[""]);
        check(&["a"], &["a"]);
        check(&["abc"], &["def"]);
        check(&["hello", "world"], &["hello", "world"]);
        check(&["hellowo", "rld"], &["hell", "oworld"]);
        check(&["hel", "lowo", "rld"], &["helloworld"]);
        check(&["{", "abc", "}"], &["{", "123", "}"]);
        check(&["{", "abc", "}", "{"], &["{", "123", "}"]);
        check(&["{", "abc", "}"], &["{", "123", "}", "{"]);
        check(&["{", "abc", "}ab"], &["{", "abc", "}", "ab"]);
    }
}
