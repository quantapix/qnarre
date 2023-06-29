use super::{static_assert, Arc, CowMut, HeaderSlice, NodeOrToken, ThinArc};
use crate::syntax::{TextRange, TextSize};
use countme::Count;
use hashbrown::hash_map::RawEntryMut;
use rustc_hash::FxHasher;
use std::{
    borrow::{Borrow, Cow},
    fmt,
    hash::{BuildHasherDefault, Hash, Hasher},
    iter::{self, FusedIterator},
    mem::{self, ManuallyDrop},
    ops, ptr, slice,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Kind(pub u16);

#[derive(Clone, Copy, Debug)]
pub struct Checkpoint(usize);

#[derive(Default, Debug)]
pub struct NodeBuilder<'cache> {
    cache: CowMut<'cache, NodeCache>,
    parents: Vec<(Kind, usize)>,
    children: Vec<(u64, Elem)>,
}
impl NodeBuilder<'_> {
    pub fn new() -> NodeBuilder<'static> {
        NodeBuilder::default()
    }
    pub fn with_cache(cache: &mut NodeCache) -> NodeBuilder<'_> {
        NodeBuilder {
            cache: CowMut::Borrowed(cache),
            parents: Vec::new(),
            children: Vec::new(),
        }
    }
    #[inline]
    pub fn token(&mut self, kind: Kind, text: &str) {
        let (hash, token) = self.cache.token(kind, text);
        self.children.push((hash, token.into()));
    }
    #[inline]
    pub fn start_node(&mut self, kind: Kind) {
        let len = self.children.len();
        self.parents.push((kind, len));
    }
    #[inline]
    pub fn finish_node(&mut self) {
        let (kind, first_child) = self.parents.pop().unwrap();
        let (hash, node) = self.cache.node(kind, &mut self.children, first_child);
        self.children.push((hash, node.into()));
    }
    #[inline]
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint(self.children.len())
    }
    #[inline]
    pub fn start_node_at(&mut self, checkpoint: Checkpoint, kind: Kind) {
        let Checkpoint(checkpoint) = checkpoint;
        assert!(
            checkpoint <= self.children.len(),
            "checkpoint no longer valid, was finish_node called early?"
        );
        if let Some(&(_, first_child)) = self.parents.last() {
            assert!(
                checkpoint >= first_child,
                "checkpoint no longer valid, was an unmatched start_node_at called?"
            );
        }
        self.parents.push((kind, checkpoint));
    }
    #[inline]
    pub fn finish(mut self) -> Node {
        assert_eq!(self.children.len(), 1);
        match self.children.pop().unwrap().1 {
            NodeOrToken::Node(node) => node,
            NodeOrToken::Token(_) => panic!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NodeHead {
    kind: Kind,
    text_len: TextSize,
    _c: Count<Node>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Child {
    Node { rel_offset: TextSize, node: Node },
    Token { rel_offset: TextSize, token: Token },
}
impl Child {
    #[inline]
    pub fn as_ref(&self) -> ElemRef {
        match self {
            Child::Node { node, .. } => NodeOrToken::Node(node),
            Child::Token { token, .. } => NodeOrToken::Token(token),
        }
    }
    #[inline]
    pub fn rel_offset(&self) -> TextSize {
        match self {
            Child::Node { rel_offset, .. } | Child::Token { rel_offset, .. } => *rel_offset,
        }
    }
    #[inline]
    fn rel_range(&self) -> TextRange {
        let len = self.as_ref().text_len();
        TextRange::at(self.rel_offset(), len)
    }
}

#[cfg(target_pointer_width = "64")]
static_assert!(mem::size_of::<Child>() == mem::size_of::<usize>() * 2);

type NodeRepr = HeaderSlice<NodeHead, [Child]>;
type NodeReprThin = HeaderSlice<NodeHead, [Child; 0]>;

#[repr(transparent)]
pub struct NodeData {
    data: NodeReprThin,
}
impl NodeData {
    #[inline]
    fn header(&self) -> &NodeHead {
        &self.data.header
    }
    #[inline]
    fn slice(&self) -> &[Child] {
        self.data.slice()
    }
    #[inline]
    pub fn kind(&self) -> Kind {
        self.header().kind
    }
    #[inline]
    pub fn text_len(&self) -> TextSize {
        self.header().text_len
    }
    #[inline]
    pub fn children(&self) -> Children<'_> {
        Children {
            raw: self.slice().iter(),
        }
    }
    pub fn child_at_range(&self, rel_range: TextRange) -> Option<(usize, TextSize, ElemRef<'_>)> {
        let idx = self
            .slice()
            .binary_search_by(|x| {
                let child_range = x.rel_range();
                TextRange::ordering(child_range, rel_range)
            })
            .unwrap_or_else(|x| x.saturating_sub(1));
        let child = &self
            .slice()
            .get(idx)
            .filter(|x| x.rel_range().contains_range(rel_range))?;
        Some((idx, child.rel_offset(), child.as_ref()))
    }
    #[must_use]
    pub fn replace_child(&self, idx: usize, x: Elem) -> Node {
        let mut x = Some(x);
        let children = self
            .children()
            .enumerate()
            .map(|(i, child)| if i == idx { x.take().unwrap() } else { child.to_owned() });
        Node::new(self.kind(), children)
    }
    #[must_use]
    pub fn insert_child(&self, idx: usize, x: Elem) -> Node {
        self.splice_children(idx..idx, iter::once(x))
    }
    #[must_use]
    pub fn remove_child(&self, idx: usize) -> Node {
        self.splice_children(idx..=idx, iter::empty())
    }
    #[must_use]
    pub fn splice_children<R, I>(&self, range: R, replace_with: I) -> Node
    where
        R: ops::RangeBounds<usize>,
        I: IntoIterator<Item = Elem>,
    {
        let mut children: Vec<_> = self.children().map(|x| x.to_owned()).collect();
        children.splice(range, replace_with);
        Node::new(self.kind(), children)
    }
}
impl PartialEq for NodeData {
    fn eq(&self, other: &Self) -> bool {
        self.header() == other.header() && self.slice() == other.slice()
    }
}
impl ToOwned for NodeData {
    type Owned = Node;
    #[inline]
    fn to_owned(&self) -> Node {
        unsafe {
            let green = Node::from_raw(ptr::NonNull::from(self));
            let green = ManuallyDrop::new(green);
            Node::clone(&green)
        }
    }
}
impl fmt::Debug for NodeData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("kind", &self.kind())
            .field("text_len", &self.text_len())
            .field("n_children", &self.children().len())
            .finish()
    }
}
impl fmt::Display for NodeData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for child in self.children() {
            write!(f, "{}", child)?;
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Node {
    ptr: ThinArc<NodeHead, Child>,
}
impl Node {
    #[inline]
    pub fn new<I>(kind: Kind, children: I) -> Node
    where
        I: IntoIterator<Item = Elem>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut text_len: TextSize = 0.into();
        let children = children.into_iter().map(|el| {
            let rel_offset = text_len;
            text_len += el.text_len();
            match el {
                NodeOrToken::Node(node) => Child::Node { rel_offset, node },
                NodeOrToken::Token(token) => Child::Token { rel_offset, token },
            }
        });
        let data = ThinArc::from_header_and_iter(
            NodeHead {
                kind,
                text_len: 0.into(),
                _c: Count::new(),
            },
            children,
        );
        let data = {
            let mut data = Arc::from_thin(data);
            Arc::get_mut(&mut data).unwrap().header.text_len = text_len;
            Arc::into_thin(data)
        };
        Node { ptr: data }
    }
    #[inline]
    pub fn into_raw(this: Node) -> ptr::NonNull<NodeData> {
        let green = ManuallyDrop::new(this);
        let green: &NodeData = &*green;
        ptr::NonNull::from(&*green)
    }
    #[inline]
    pub unsafe fn from_raw(ptr: ptr::NonNull<NodeData>) -> Node {
        let arc = Arc::from_raw(&ptr.as_ref().data as *const NodeReprThin);
        let arc = mem::transmute::<Arc<NodeReprThin>, ThinArc<NodeHead, Child>>(arc);
        Node { ptr: arc }
    }
}
impl Borrow<NodeData> for Node {
    #[inline]
    fn borrow(&self) -> &NodeData {
        &*self
    }
}
impl From<Cow<'_, NodeData>> for Node {
    #[inline]
    fn from(cow: Cow<'_, NodeData>) -> Self {
        cow.into_owned()
    }
}
impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data: &NodeData = &*self;
        fmt::Debug::fmt(data, f)
    }
}
impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data: &NodeData = &*self;
        fmt::Display::fmt(data, f)
    }
}
impl ops::Deref for Node {
    type Target = NodeData;
    #[inline]
    fn deref(&self) -> &NodeData {
        unsafe {
            let repr: &NodeRepr = &self.ptr;
            let repr: &NodeReprThin = &*(repr as *const NodeRepr as *const NodeReprThin);
            mem::transmute::<&NodeReprThin, &NodeData>(repr)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Children<'a> {
    pub raw: slice::Iter<'a, Child>,
}
impl<'a> Iterator for Children<'a> {
    type Item = ElemRef<'a>;
    #[inline]
    fn next(&mut self) -> Option<ElemRef<'a>> {
        self.raw.next().map(Child::as_ref)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.raw.size_hint()
    }
    #[inline]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.raw.count()
    }
    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.raw.nth(n).map(Child::as_ref)
    }
    #[inline]
    fn last(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.next_back()
    }
    #[inline]
    fn fold<Acc, Fold>(mut self, init: Acc, mut f: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut accum = init;
        while let Some(x) = self.next() {
            accum = f(accum, x);
        }
        accum
    }
}
impl<'a> DoubleEndedIterator for Children<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.raw.next_back().map(Child::as_ref)
    }
    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.raw.nth_back(n).map(Child::as_ref)
    }
    #[inline]
    fn rfold<Acc, Fold>(mut self, init: Acc, mut f: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut accum = init;
        while let Some(x) = self.next_back() {
            accum = f(accum, x);
        }
        accum
    }
}
impl ExactSizeIterator for Children<'_> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.raw.len()
    }
}
impl FusedIterator for Children<'_> {}

#[derive(PartialEq, Eq, Hash)]
struct TokHead {
    kind: Kind,
    _c: Count<Token>,
}

type TokRepr = HeaderSlice<TokHead, [u8]>;
type TokReprThin = HeaderSlice<TokHead, [u8; 0]>;

#[repr(transparent)]
pub struct TokData {
    data: TokReprThin,
}
impl TokData {
    #[inline]
    pub fn kind(&self) -> Kind {
        self.data.header.kind
    }
    #[inline]
    pub fn text(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(self.data.slice()) }
    }
    #[inline]
    pub fn text_len(&self) -> TextSize {
        TextSize::of(self.text())
    }
}
impl PartialEq for TokData {
    fn eq(&self, other: &Self) -> bool {
        self.kind() == other.kind() && self.text() == other.text()
    }
}
impl ToOwned for TokData {
    type Owned = Token;
    #[inline]
    fn to_owned(&self) -> Token {
        unsafe {
            let green = Token::from_raw(ptr::NonNull::from(self));
            let green = ManuallyDrop::new(green);
            Token::clone(&green)
        }
    }
}
impl fmt::Debug for TokData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Token")
            .field("kind", &self.kind())
            .field("text", &self.text())
            .finish()
    }
}
impl fmt::Display for TokData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text())
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
#[repr(transparent)]
pub struct Token {
    ptr: ThinArc<TokHead, u8>,
}
impl Token {
    #[inline]
    pub fn new(kind: Kind, text: &str) -> Token {
        let head = TokHead { kind, _c: Count::new() };
        let ptr = ThinArc::from_header_and_iter(head, text.bytes());
        Token { ptr }
    }
    #[inline]
    pub fn into_raw(this: Token) -> ptr::NonNull<TokData> {
        let green = ManuallyDrop::new(this);
        let green: &TokData = &*green;
        ptr::NonNull::from(&*green)
    }
    #[inline]
    pub unsafe fn from_raw(ptr: ptr::NonNull<TokData>) -> Token {
        let arc = Arc::from_raw(&ptr.as_ref().data as *const TokReprThin);
        let arc = mem::transmute::<Arc<TokReprThin>, ThinArc<TokHead, u8>>(arc);
        Token { ptr: arc }
    }
}
impl Borrow<TokData> for Token {
    #[inline]
    fn borrow(&self) -> &TokData {
        &*self
    }
}
impl fmt::Debug for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data: &TokData = &*self;
        fmt::Debug::fmt(data, f)
    }
}
impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data: &TokData = &*self;
        fmt::Display::fmt(data, f)
    }
}
impl ops::Deref for Token {
    type Target = TokData;
    #[inline]
    fn deref(&self) -> &TokData {
        unsafe {
            let repr: &TokRepr = &self.ptr;
            let repr: &TokReprThin = &*(repr as *const TokRepr as *const TokReprThin);
            mem::transmute::<&TokReprThin, &TokData>(repr)
        }
    }
}

type Elem = NodeOrToken<Node, Token>;
impl Elem {
    #[inline]
    pub fn kind(&self) -> Kind {
        self.as_deref().kind()
    }
    #[inline]
    pub fn text_len(&self) -> TextSize {
        self.as_deref().text_len()
    }
}
impl From<Node> for Elem {
    #[inline]
    fn from(node: Node) -> Elem {
        NodeOrToken::Node(node)
    }
}
impl From<Token> for Elem {
    #[inline]
    fn from(token: Token) -> Elem {
        NodeOrToken::Token(token)
    }
}
impl From<Cow<'_, NodeData>> for Elem {
    #[inline]
    fn from(cow: Cow<'_, NodeData>) -> Self {
        NodeOrToken::Node(cow.into_owned())
    }
}

pub type ElemRef<'a> = NodeOrToken<&'a NodeData, &'a TokData>;
impl ElemRef<'_> {
    pub fn to_owned(self) -> Elem {
        match self {
            NodeOrToken::Node(it) => NodeOrToken::Node(it.to_owned()),
            NodeOrToken::Token(it) => NodeOrToken::Token(it.to_owned()),
        }
    }
    #[inline]
    pub fn kind(&self) -> Kind {
        match self {
            NodeOrToken::Node(it) => it.kind(),
            NodeOrToken::Token(it) => it.kind(),
        }
    }
    #[inline]
    pub fn text_len(self) -> TextSize {
        match self {
            NodeOrToken::Node(it) => it.text_len(),
            NodeOrToken::Token(it) => it.text_len(),
        }
    }
}
impl<'a> From<&'a Node> for ElemRef<'a> {
    #[inline]
    fn from(node: &'a Node) -> ElemRef<'a> {
        NodeOrToken::Node(node)
    }
}
impl<'a> From<&'a Token> for ElemRef<'a> {
    #[inline]
    fn from(token: &'a Token) -> ElemRef<'a> {
        NodeOrToken::Token(token)
    }
}

type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasherDefault<FxHasher>>;

#[derive(Debug)]
struct NoHash<T>(T);

#[derive(Default, Debug)]
pub struct NodeCache {
    nodes: HashMap<NoHash<Node>, ()>,
    tokens: HashMap<NoHash<Token>, ()>,
}
impl NodeCache {
    pub fn node(&mut self, kind: Kind, children: &mut Vec<(u64, Elem)>, first_child: usize) -> (u64, Node) {
        let build_node =
            move |children: &mut Vec<(u64, Elem)>| Node::new(kind, children.drain(first_child..).map(|(_, it)| it));
        let children_ref = &children[first_child..];
        if children_ref.len() > 3 {
            let node = build_node(children);
            return (0, node);
        }
        let hash = {
            let mut h = FxHasher::default();
            kind.hash(&mut h);
            for &(hash, _) in children_ref {
                if hash == 0 {
                    let node = build_node(children);
                    return (0, node);
                }
                hash.hash(&mut h);
            }
            h.finish()
        };
        let entry = self.nodes.raw_entry_mut().from_hash(hash, |node| {
            node.0.kind() == kind && node.0.children().len() == children_ref.len() && {
                let lhs = node.0.children();
                let rhs = children_ref.iter().map(|(_, it)| it.as_deref());
                let lhs = lhs.map(element_id);
                let rhs = rhs.map(element_id);
                lhs.eq(rhs)
            }
        });
        let node = match entry {
            RawEntryMut::Occupied(entry) => {
                drop(children.drain(first_child..));
                entry.key().0.clone()
            },
            RawEntryMut::Vacant(entry) => {
                let node = build_node(children);
                entry.insert_with_hasher(hash, NoHash(node.clone()), (), |n| node_hash(&n.0));
                node
            },
        };
        (hash, node)
    }
    pub fn token(&mut self, kind: Kind, text: &str) -> (u64, Token) {
        let hash = {
            let mut h = FxHasher::default();
            kind.hash(&mut h);
            text.hash(&mut h);
            h.finish()
        };
        let entry = self
            .tokens
            .raw_entry_mut()
            .from_hash(hash, |token| token.0.kind() == kind && token.0.text() == text);
        let token = match entry {
            RawEntryMut::Occupied(entry) => entry.key().0.clone(),
            RawEntryMut::Vacant(entry) => {
                let token = Token::new(kind, text);
                entry.insert_with_hasher(hash, NoHash(token.clone()), (), |t| token_hash(&t.0));
                token
            },
        };
        (hash, token)
    }
}

fn token_hash(token: &TokData) -> u64 {
    let mut h = FxHasher::default();
    token.kind().hash(&mut h);
    token.text().hash(&mut h);
    h.finish()
}
fn node_hash(node: &NodeData) -> u64 {
    let mut h = FxHasher::default();
    node.kind().hash(&mut h);
    for child in node.children() {
        match child {
            NodeOrToken::Node(it) => node_hash(it),
            NodeOrToken::Token(it) => token_hash(it),
        }
        .hash(&mut h)
    }
    h.finish()
}
fn element_id(elem: ElemRef<'_>) -> *const () {
    match elem {
        NodeOrToken::Node(it) => it as *const NodeData as *const (),
        NodeOrToken::Token(it) => it as *const TokData as *const (),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn assert_send_sync() {
        fn f<T: Send + Sync>() {}
        f::<Node>();
        f::<Token>();
        f::<Elem>();
    }
    #[test]
    fn test_size_of() {
        use std::mem::size_of;
        eprintln!("Node          {}", size_of::<Node>());
        eprintln!("Token         {}", size_of::<Token>());
        eprintln!("Elem       {}", size_of::<Elem>());
    }
}
