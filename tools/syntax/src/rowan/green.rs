mod node {
    use crate::{
        arc::{Arc, HeaderSlice, ThinArc},
        green::{GreenElement, GreenElementRef, SyntaxKind},
        utility_types::static_assert,
        GreenToken, NodeOrToken, TextRange, TextSize,
    };
    use countme::Count;
    use std::{
        borrow::{Borrow, Cow},
        fmt,
        iter::{self, FusedIterator},
        mem::{self, ManuallyDrop},
        ops, ptr, slice,
    };

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub(super) struct GreenNodeHead {
        kind: SyntaxKind,
        text_len: TextSize,
        _c: Count<GreenNode>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub(crate) enum GreenChild {
        Node { rel_offset: TextSize, node: GreenNode },
        Token { rel_offset: TextSize, token: GreenToken },
    }

    #[cfg(target_pointer_width = "64")]
    static_assert!(mem::size_of::<GreenChild>() == mem::size_of::<usize>() * 2);

    type Repr = HeaderSlice<GreenNodeHead, [GreenChild]>;
    type ReprThin = HeaderSlice<GreenNodeHead, [GreenChild; 0]>;

    #[repr(transparent)]
    pub struct GreenNodeData {
        data: ReprThin,
    }
    impl PartialEq for GreenNodeData {
        fn eq(&self, other: &Self) -> bool {
            self.header() == other.header() && self.slice() == other.slice()
        }
    }

    #[derive(Clone, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct GreenNode {
        ptr: ThinArc<GreenNodeHead, GreenChild>,
    }
    impl ToOwned for GreenNodeData {
        type Owned = GreenNode;
        #[inline]
        fn to_owned(&self) -> GreenNode {
            unsafe {
                let green = GreenNode::from_raw(ptr::NonNull::from(self));
                let green = ManuallyDrop::new(green);
                GreenNode::clone(&green)
            }
        }
    }
    impl Borrow<GreenNodeData> for GreenNode {
        #[inline]
        fn borrow(&self) -> &GreenNodeData {
            &*self
        }
    }
    impl From<Cow<'_, GreenNodeData>> for GreenNode {
        #[inline]
        fn from(cow: Cow<'_, GreenNodeData>) -> Self {
            cow.into_owned()
        }
    }
    impl fmt::Debug for GreenNodeData {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("GreenNode")
                .field("kind", &self.kind())
                .field("text_len", &self.text_len())
                .field("n_children", &self.children().len())
                .finish()
        }
    }
    impl fmt::Debug for GreenNode {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let data: &GreenNodeData = &*self;
            fmt::Debug::fmt(data, f)
        }
    }
    impl fmt::Display for GreenNode {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let data: &GreenNodeData = &*self;
            fmt::Display::fmt(data, f)
        }
    }
    impl fmt::Display for GreenNodeData {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            for child in self.children() {
                write!(f, "{}", child)?;
            }
            Ok(())
        }
    }
    impl GreenNodeData {
        #[inline]
        fn header(&self) -> &GreenNodeHead {
            &self.data.header
        }
        #[inline]
        fn slice(&self) -> &[GreenChild] {
            self.data.slice()
        }
        #[inline]
        pub fn kind(&self) -> SyntaxKind {
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
        pub(crate) fn child_at_range(&self, rel_range: TextRange) -> Option<(usize, TextSize, GreenElementRef<'_>)> {
            let idx = self
                .slice()
                .binary_search_by(|it| {
                    let child_range = it.rel_range();
                    TextRange::ordering(child_range, rel_range)
                })
                .unwrap_or_else(|it| it.saturating_sub(1));
            let child = &self
                .slice()
                .get(idx)
                .filter(|it| it.rel_range().contains_range(rel_range))?;
            Some((idx, child.rel_offset(), child.as_ref()))
        }
        #[must_use]
        pub fn replace_child(&self, index: usize, new_child: GreenElement) -> GreenNode {
            let mut replacement = Some(new_child);
            let children = self.children().enumerate().map(|(i, child)| {
                if i == index {
                    replacement.take().unwrap()
                } else {
                    child.to_owned()
                }
            });
            GreenNode::new(self.kind(), children)
        }
        #[must_use]
        pub fn insert_child(&self, index: usize, new_child: GreenElement) -> GreenNode {
            self.splice_children(index..index, iter::once(new_child))
        }
        #[must_use]
        pub fn remove_child(&self, index: usize) -> GreenNode {
            self.splice_children(index..=index, iter::empty())
        }
        #[must_use]
        pub fn splice_children<R, I>(&self, range: R, replace_with: I) -> GreenNode
        where
            R: ops::RangeBounds<usize>,
            I: IntoIterator<Item = GreenElement>,
        {
            let mut children: Vec<_> = self.children().map(|it| it.to_owned()).collect();
            children.splice(range, replace_with);
            GreenNode::new(self.kind(), children)
        }
    }
    impl ops::Deref for GreenNode {
        type Target = GreenNodeData;
        #[inline]
        fn deref(&self) -> &GreenNodeData {
            unsafe {
                let repr: &Repr = &self.ptr;
                let repr: &ReprThin = &*(repr as *const Repr as *const ReprThin);
                mem::transmute::<&ReprThin, &GreenNodeData>(repr)
            }
        }
    }
    impl GreenNode {
        #[inline]
        pub fn new<I>(kind: SyntaxKind, children: I) -> GreenNode
        where
            I: IntoIterator<Item = GreenElement>,
            I::IntoIter: ExactSizeIterator,
        {
            let mut text_len: TextSize = 0.into();
            let children = children.into_iter().map(|el| {
                let rel_offset = text_len;
                text_len += el.text_len();
                match el {
                    NodeOrToken::Node(node) => GreenChild::Node { rel_offset, node },
                    NodeOrToken::Token(token) => GreenChild::Token { rel_offset, token },
                }
            });
            let data = ThinArc::from_header_and_iter(
                GreenNodeHead {
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
            GreenNode { ptr: data }
        }
        #[inline]
        pub(crate) fn into_raw(this: GreenNode) -> ptr::NonNull<GreenNodeData> {
            let green = ManuallyDrop::new(this);
            let green: &GreenNodeData = &*green;
            ptr::NonNull::from(&*green)
        }
        #[inline]
        pub(crate) unsafe fn from_raw(ptr: ptr::NonNull<GreenNodeData>) -> GreenNode {
            let arc = Arc::from_raw(&ptr.as_ref().data as *const ReprThin);
            let arc = mem::transmute::<Arc<ReprThin>, ThinArc<GreenNodeHead, GreenChild>>(arc);
            GreenNode { ptr: arc }
        }
    }
    impl GreenChild {
        #[inline]
        pub(crate) fn as_ref(&self) -> GreenElementRef {
            match self {
                GreenChild::Node { node, .. } => NodeOrToken::Node(node),
                GreenChild::Token { token, .. } => NodeOrToken::Token(token),
            }
        }
        #[inline]
        pub(crate) fn rel_offset(&self) -> TextSize {
            match self {
                GreenChild::Node { rel_offset, .. } | GreenChild::Token { rel_offset, .. } => *rel_offset,
            }
        }
        #[inline]
        fn rel_range(&self) -> TextRange {
            let len = self.as_ref().text_len();
            TextRange::at(self.rel_offset(), len)
        }
    }
    #[derive(Debug, Clone)]
    pub struct Children<'a> {
        pub(crate) raw: slice::Iter<'a, GreenChild>,
    }
    impl ExactSizeIterator for Children<'_> {
        #[inline(always)]
        fn len(&self) -> usize {
            self.raw.len()
        }
    }
    impl<'a> Iterator for Children<'a> {
        type Item = GreenElementRef<'a>;
        #[inline]
        fn next(&mut self) -> Option<GreenElementRef<'a>> {
            self.raw.next().map(GreenChild::as_ref)
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
            self.raw.nth(n).map(GreenChild::as_ref)
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
            self.raw.next_back().map(GreenChild::as_ref)
        }
        #[inline]
        fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
            self.raw.nth_back(n).map(GreenChild::as_ref)
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
    impl FusedIterator for Children<'_> {}
}
mod token {
    use crate::{
        arc::{Arc, HeaderSlice, ThinArc},
        green::SyntaxKind,
        TextSize,
    };
    use countme::Count;
    use std::{
        borrow::Borrow,
        fmt,
        mem::{self, ManuallyDrop},
        ops, ptr,
    };
    #[derive(PartialEq, Eq, Hash)]
    struct GreenTokenHead {
        kind: SyntaxKind,
        _c: Count<GreenToken>,
    }
    type Repr = HeaderSlice<GreenTokenHead, [u8]>;
    type ReprThin = HeaderSlice<GreenTokenHead, [u8; 0]>;
    #[repr(transparent)]
    pub struct GreenTokenData {
        data: ReprThin,
    }
    impl PartialEq for GreenTokenData {
        fn eq(&self, other: &Self) -> bool {
            self.kind() == other.kind() && self.text() == other.text()
        }
    }
    #[derive(PartialEq, Eq, Hash, Clone)]
    #[repr(transparent)]
    pub struct GreenToken {
        ptr: ThinArc<GreenTokenHead, u8>,
    }
    impl ToOwned for GreenTokenData {
        type Owned = GreenToken;
        #[inline]
        fn to_owned(&self) -> GreenToken {
            unsafe {
                let green = GreenToken::from_raw(ptr::NonNull::from(self));
                let green = ManuallyDrop::new(green);
                GreenToken::clone(&green)
            }
        }
    }
    impl Borrow<GreenTokenData> for GreenToken {
        #[inline]
        fn borrow(&self) -> &GreenTokenData {
            &*self
        }
    }
    impl fmt::Debug for GreenTokenData {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("GreenToken")
                .field("kind", &self.kind())
                .field("text", &self.text())
                .finish()
        }
    }
    impl fmt::Debug for GreenToken {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let data: &GreenTokenData = &*self;
            fmt::Debug::fmt(data, f)
        }
    }
    impl fmt::Display for GreenToken {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let data: &GreenTokenData = &*self;
            fmt::Display::fmt(data, f)
        }
    }
    impl fmt::Display for GreenTokenData {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.text())
        }
    }
    impl GreenTokenData {
        #[inline]
        pub fn kind(&self) -> SyntaxKind {
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
    impl GreenToken {
        #[inline]
        pub fn new(kind: SyntaxKind, text: &str) -> GreenToken {
            let head = GreenTokenHead { kind, _c: Count::new() };
            let ptr = ThinArc::from_header_and_iter(head, text.bytes());
            GreenToken { ptr }
        }
        #[inline]
        pub(crate) fn into_raw(this: GreenToken) -> ptr::NonNull<GreenTokenData> {
            let green = ManuallyDrop::new(this);
            let green: &GreenTokenData = &*green;
            ptr::NonNull::from(&*green)
        }
        #[inline]
        pub(crate) unsafe fn from_raw(ptr: ptr::NonNull<GreenTokenData>) -> GreenToken {
            let arc = Arc::from_raw(&ptr.as_ref().data as *const ReprThin);
            let arc = mem::transmute::<Arc<ReprThin>, ThinArc<GreenTokenHead, u8>>(arc);
            GreenToken { ptr: arc }
        }
    }
    impl ops::Deref for GreenToken {
        type Target = GreenTokenData;
        #[inline]
        fn deref(&self) -> &GreenTokenData {
            unsafe {
                let repr: &Repr = &self.ptr;
                let repr: &ReprThin = &*(repr as *const Repr as *const ReprThin);
                mem::transmute::<&ReprThin, &GreenTokenData>(repr)
            }
        }
    }
}
mod builder {
    use crate::{
        cow_mut::CowMut,
        green::{node_cache::NodeCache, GreenElement, GreenNode, SyntaxKind},
        NodeOrToken,
    };
    #[derive(Clone, Copy, Debug)]
    pub struct Checkpoint(usize);
    #[derive(Default, Debug)]
    pub struct GreenNodeBuilder<'cache> {
        cache: CowMut<'cache, NodeCache>,
        parents: Vec<(SyntaxKind, usize)>,
        children: Vec<(u64, GreenElement)>,
    }
    impl GreenNodeBuilder<'_> {
        pub fn new() -> GreenNodeBuilder<'static> {
            GreenNodeBuilder::default()
        }
        pub fn with_cache(cache: &mut NodeCache) -> GreenNodeBuilder<'_> {
            GreenNodeBuilder {
                cache: CowMut::Borrowed(cache),
                parents: Vec::new(),
                children: Vec::new(),
            }
        }
        #[inline]
        pub fn token(&mut self, kind: SyntaxKind, text: &str) {
            let (hash, token) = self.cache.token(kind, text);
            self.children.push((hash, token.into()));
        }
        #[inline]
        pub fn start_node(&mut self, kind: SyntaxKind) {
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
        pub fn start_node_at(&mut self, checkpoint: Checkpoint, kind: SyntaxKind) {
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
        pub fn finish(mut self) -> GreenNode {
            assert_eq!(self.children.len(), 1);
            match self.children.pop().unwrap().1 {
                NodeOrToken::Node(node) => node,
                NodeOrToken::Token(_) => panic!(),
            }
        }
    }
}
mod element {
    use super::GreenTokenData;
    use crate::{
        green::{GreenNode, GreenToken, SyntaxKind},
        GreenNodeData, NodeOrToken, TextSize,
    };
    use std::borrow::Cow;
    pub(super) type GreenElement = NodeOrToken<GreenNode, GreenToken>;
    pub(crate) type GreenElementRef<'a> = NodeOrToken<&'a GreenNodeData, &'a GreenTokenData>;
    impl From<GreenNode> for GreenElement {
        #[inline]
        fn from(node: GreenNode) -> GreenElement {
            NodeOrToken::Node(node)
        }
    }
    impl<'a> From<&'a GreenNode> for GreenElementRef<'a> {
        #[inline]
        fn from(node: &'a GreenNode) -> GreenElementRef<'a> {
            NodeOrToken::Node(node)
        }
    }
    impl From<GreenToken> for GreenElement {
        #[inline]
        fn from(token: GreenToken) -> GreenElement {
            NodeOrToken::Token(token)
        }
    }
    impl From<Cow<'_, GreenNodeData>> for GreenElement {
        #[inline]
        fn from(cow: Cow<'_, GreenNodeData>) -> Self {
            NodeOrToken::Node(cow.into_owned())
        }
    }
    impl<'a> From<&'a GreenToken> for GreenElementRef<'a> {
        #[inline]
        fn from(token: &'a GreenToken) -> GreenElementRef<'a> {
            NodeOrToken::Token(token)
        }
    }
    impl GreenElementRef<'_> {
        pub fn to_owned(self) -> GreenElement {
            match self {
                NodeOrToken::Node(it) => NodeOrToken::Node(it.to_owned()),
                NodeOrToken::Token(it) => NodeOrToken::Token(it.to_owned()),
            }
        }
    }
    impl GreenElement {
        #[inline]
        pub fn kind(&self) -> SyntaxKind {
            self.as_deref().kind()
        }
        #[inline]
        pub fn text_len(&self) -> TextSize {
            self.as_deref().text_len()
        }
    }
    impl GreenElementRef<'_> {
        #[inline]
        pub fn kind(&self) -> SyntaxKind {
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
}
mod node_cache {
    use super::element::GreenElement;
    use crate::{
        green::GreenElementRef, GreenNode, GreenNodeData, GreenToken, GreenTokenData, NodeOrToken, SyntaxKind,
    };
    use hashbrown::hash_map::RawEntryMut;
    use rustc_hash::FxHasher;
    use std::hash::{BuildHasherDefault, Hash, Hasher};
    type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasherDefault<FxHasher>>;
    #[derive(Debug)]
    struct NoHash<T>(T);
    #[derive(Default, Debug)]
    pub struct NodeCache {
        nodes: HashMap<NoHash<GreenNode>, ()>,
        tokens: HashMap<NoHash<GreenToken>, ()>,
    }
    fn token_hash(token: &GreenTokenData) -> u64 {
        let mut h = FxHasher::default();
        token.kind().hash(&mut h);
        token.text().hash(&mut h);
        h.finish()
    }
    fn node_hash(node: &GreenNodeData) -> u64 {
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
    fn element_id(elem: GreenElementRef<'_>) -> *const () {
        match elem {
            NodeOrToken::Node(it) => it as *const GreenNodeData as *const (),
            NodeOrToken::Token(it) => it as *const GreenTokenData as *const (),
        }
    }
    impl NodeCache {
        pub(crate) fn node(
            &mut self,
            kind: SyntaxKind,
            children: &mut Vec<(u64, GreenElement)>,
            first_child: usize,
        ) -> (u64, GreenNode) {
            let build_node = move |children: &mut Vec<(u64, GreenElement)>| {
                GreenNode::new(kind, children.drain(first_child..).map(|(_, it)| it))
            };
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
        pub(crate) fn token(&mut self, kind: SyntaxKind, text: &str) -> (u64, GreenToken) {
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
                    let token = GreenToken::new(kind, text);
                    entry.insert_with_hasher(hash, NoHash(token.clone()), (), |t| token_hash(&t.0));
                    token
                },
            };
            (hash, token)
        }
    }
}

use self::element::GreenElement;
pub use self::{
    builder::{Checkpoint, GreenNodeBuilder},
    node::{Children, GreenNode, GreenNodeData},
    node_cache::NodeCache,
    token::{GreenToken, GreenTokenData},
};
pub(crate) use self::{element::GreenElementRef, node::GreenChild};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SyntaxKind(pub u16);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn assert_send_sync() {
        fn f<T: Send + Sync>() {}
        f::<GreenNode>();
        f::<GreenToken>();
        f::<GreenElement>();
    }
    #[test]
    fn test_size_of() {
        use std::mem::size_of;
        eprintln!("GreenNode          {}", size_of::<GreenNode>());
        eprintln!("GreenToken         {}", size_of::<GreenToken>());
        eprintln!("GreenElement       {}", size_of::<GreenElement>());
    }
}
