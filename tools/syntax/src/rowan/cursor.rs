use super::{
    green::{GreenChild, GreenElementRef, GreenNode, GreenNodeData, GreenToken, GreenTokenData, SyntaxKind},
    sll, Delta, Direction, NodeOrToken, SyntaxText, TextRange, TextSize, TokenAtOffset, WalkEvent,
};
use countme::Count;
use std::{
    borrow::Cow,
    cell::Cell,
    fmt,
    hash::{Hash, Hasher},
    iter,
    mem::{self, ManuallyDrop},
    ops::Range,
    ptr, slice,
};

enum Green {
    Node { ptr: Cell<ptr::NonNull<GreenNodeData>> },
    Token { ptr: ptr::NonNull<GreenTokenData> },
}

struct _SyntaxElement;

struct NodeData {
    _c: Count<_SyntaxElement>,
    rc: Cell<u32>,
    parent: Cell<Option<ptr::NonNull<NodeData>>>,
    index: Cell<u32>,
    green: Green,
    mutable: bool,
    offset: TextSize,
    first: Cell<*const NodeData>,
    next: Cell<*const NodeData>,
    prev: Cell<*const NodeData>,
}
impl NodeData {
    #[inline]
    fn new(
        parent: Option<SyntaxNode>,
        index: u32,
        offset: TextSize,
        green: Green,
        mutable: bool,
    ) -> ptr::NonNull<NodeData> {
        let parent = ManuallyDrop::new(parent);
        let res = NodeData {
            _c: Count::new(),
            rc: Cell::new(1),
            parent: Cell::new(parent.as_ref().map(|it| it.ptr)),
            index: Cell::new(index),
            green,
            mutable,
            offset,
            first: Cell::new(ptr::null()),
            next: Cell::new(ptr::null()),
            prev: Cell::new(ptr::null()),
        };
        unsafe {
            if mutable {
                let res_ptr: *const NodeData = &res;
                match sll::init((*res_ptr).parent().map(|it| &it.first), res_ptr.as_ref().unwrap()) {
                    sll::AddToSllResult::AlreadyInSll(node) => {
                        if cfg!(debug_assertions) {
                            assert_eq!((*node).index(), (*res_ptr).index());
                            match ((*node).green(), (*res_ptr).green()) {
                                (NodeOrToken::Node(lhs), NodeOrToken::Node(rhs)) => {
                                    assert!(ptr::eq(lhs, rhs))
                                },
                                (NodeOrToken::Token(lhs), NodeOrToken::Token(rhs)) => {
                                    assert!(ptr::eq(lhs, rhs))
                                },
                                it => {
                                    panic!("node/token confusion: {:?}", it)
                                },
                            }
                        }
                        ManuallyDrop::into_inner(parent);
                        let res = node as *mut NodeData;
                        (*res).inc_rc();
                        return ptr::NonNull::new_unchecked(res);
                    },
                    it => {
                        let res = Box::into_raw(Box::new(res));
                        it.add_to_sll(res);
                        return ptr::NonNull::new_unchecked(res);
                    },
                }
            }
            ptr::NonNull::new_unchecked(Box::into_raw(Box::new(res)))
        }
    }
    #[inline]
    fn inc_rc(&self) {
        let rc = match self.rc.get().checked_add(1) {
            Some(it) => it,
            None => std::process::abort(),
        };
        self.rc.set(rc)
    }
    #[inline]
    fn dec_rc(&self) -> bool {
        let rc = self.rc.get() - 1;
        self.rc.set(rc);
        rc == 0
    }
    #[inline]
    fn key(&self) -> (ptr::NonNull<()>, TextSize) {
        let ptr = match &self.green {
            Green::Node { ptr } => ptr.get().cast(),
            Green::Token { ptr } => ptr.cast(),
        };
        (ptr, self.offset())
    }
    #[inline]
    fn parent_node(&self) -> Option<SyntaxNode> {
        let parent = self.parent()?;
        debug_assert!(matches!(parent.green, Green::Node { .. }));
        parent.inc_rc();
        Some(SyntaxNode {
            ptr: ptr::NonNull::from(parent),
        })
    }
    #[inline]
    fn parent(&self) -> Option<&NodeData> {
        self.parent.get().map(|it| unsafe { &*it.as_ptr() })
    }
    #[inline]
    fn green(&self) -> GreenElementRef<'_> {
        match &self.green {
            Green::Node { ptr } => GreenElementRef::Node(unsafe { &*ptr.get().as_ptr() }),
            Green::Token { ptr } => GreenElementRef::Token(unsafe { &*ptr.as_ref() }),
        }
    }
    #[inline]
    fn green_siblings(&self) -> slice::Iter<GreenChild> {
        match &self.parent().map(|it| &it.green) {
            Some(Green::Node { ptr }) => unsafe { &*ptr.get().as_ptr() }.children().raw,
            Some(Green::Token { .. }) => {
                debug_assert!(false);
                [].iter()
            },
            None => [].iter(),
        }
    }
    #[inline]
    fn index(&self) -> u32 {
        self.index.get()
    }
    #[inline]
    fn offset(&self) -> TextSize {
        if self.mutable {
            self.offset_mut()
        } else {
            self.offset
        }
    }
    #[cold]
    fn offset_mut(&self) -> TextSize {
        let mut res = TextSize::from(0);
        let mut node = self;
        while let Some(parent) = node.parent() {
            let green = parent.green().into_node().unwrap();
            res += green.children().raw.nth(node.index() as usize).unwrap().rel_offset();
            node = parent;
        }
        res
    }
    #[inline]
    fn text_range(&self) -> TextRange {
        let offset = self.offset();
        let len = self.green().text_len();
        TextRange::at(offset, len)
    }
    #[inline]
    fn kind(&self) -> SyntaxKind {
        self.green().kind()
    }
    fn next_sibling(&self) -> Option<SyntaxNode> {
        let mut siblings = self.green_siblings().enumerate();
        let index = self.index() as usize;
        siblings.nth(index);
        siblings.find_map(|(index, child)| {
            child.as_ref().into_node().and_then(|green| {
                let parent = self.parent_node()?;
                let offset = parent.offset() + child.rel_offset();
                Some(SyntaxNode::new_child(green, parent, index as u32, offset))
            })
        })
    }
    fn prev_sibling(&self) -> Option<SyntaxNode> {
        let mut rev_siblings = self.green_siblings().enumerate().rev();
        let index = rev_siblings.len().checked_sub(self.index() as usize + 1)?;
        rev_siblings.nth(index);
        rev_siblings.find_map(|(index, child)| {
            child.as_ref().into_node().and_then(|green| {
                let parent = self.parent_node()?;
                let offset = parent.offset() + child.rel_offset();
                Some(SyntaxNode::new_child(green, parent, index as u32, offset))
            })
        })
    }
    fn next_sibling_or_token(&self) -> Option<SyntaxElement> {
        let mut siblings = self.green_siblings().enumerate();
        let index = self.index() as usize + 1;
        siblings.nth(index).and_then(|(index, child)| {
            let parent = self.parent_node()?;
            let offset = parent.offset() + child.rel_offset();
            Some(SyntaxElement::new(child.as_ref(), parent, index as u32, offset))
        })
    }
    fn prev_sibling_or_token(&self) -> Option<SyntaxElement> {
        let mut siblings = self.green_siblings().enumerate();
        let index = self.index().checked_sub(1)? as usize;
        siblings.nth(index).and_then(|(index, child)| {
            let parent = self.parent_node()?;
            let offset = parent.offset() + child.rel_offset();
            Some(SyntaxElement::new(child.as_ref(), parent, index as u32, offset))
        })
    }
    fn detach(&self) {
        assert!(self.mutable);
        assert!(self.rc.get() > 0);
        let parent_ptr = match self.parent.take() {
            Some(parent) => parent,
            None => return,
        };
        unsafe {
            sll::adjust(self, self.index() + 1, Delta::Sub(1));
            let parent = parent_ptr.as_ref();
            sll::unlink(&parent.first, self);
            match self.green().to_owned() {
                NodeOrToken::Node(it) => {
                    GreenNode::into_raw(it);
                },
                NodeOrToken::Token(it) => {
                    GreenToken::into_raw(it);
                },
            }
            match parent.green() {
                NodeOrToken::Node(green) => {
                    let green = green.remove_child(self.index() as usize);
                    parent.respine(green)
                },
                NodeOrToken::Token(_) => unreachable!(),
            }
            if parent.dec_rc() {
                free(parent_ptr)
            }
        }
    }
    fn attach_child(&self, index: usize, child: &NodeData) {
        assert!(self.mutable && child.mutable && child.parent().is_none());
        assert!(self.rc.get() > 0 && child.rc.get() > 0);
        unsafe {
            child.index.set(index as u32);
            child.parent.set(Some(self.into()));
            self.inc_rc();
            if !self.first.get().is_null() {
                sll::adjust(&*self.first.get(), index as u32, Delta::Add(1));
            }
            match sll::link(&self.first, child) {
                sll::AddToSllResult::AlreadyInSll(_) => {
                    panic!("Child already in sorted linked list")
                },
                it => it.add_to_sll(child),
            }
            match self.green() {
                NodeOrToken::Node(green) => {
                    let child_green = match &child.green {
                        Green::Node { ptr } => GreenNode::from_raw(ptr.get()).into(),
                        Green::Token { ptr } => GreenToken::from_raw(*ptr).into(),
                    };
                    let green = green.insert_child(index, child_green);
                    self.respine(green);
                },
                NodeOrToken::Token(_) => unreachable!(),
            }
        }
    }
    unsafe fn respine(&self, mut new_green: GreenNode) {
        let mut node = self;
        loop {
            let old_green = match &node.green {
                Green::Node { ptr } => ptr.replace(ptr::NonNull::from(&*new_green)),
                Green::Token { .. } => unreachable!(),
            };
            match node.parent() {
                Some(parent) => match parent.green() {
                    NodeOrToken::Node(parent_green) => {
                        new_green = parent_green.replace_child(node.index() as usize, new_green.into());
                        node = parent;
                    },
                    _ => unreachable!(),
                },
                None => {
                    mem::forget(new_green);
                    let _ = GreenNode::from_raw(old_green);
                    break;
                },
            }
        }
    }
}
unsafe impl sll::Elem for NodeData {
    fn prev(&self) -> &Cell<*const Self> {
        &self.prev
    }
    fn next(&self) -> &Cell<*const Self> {
        &self.next
    }
    fn key(&self) -> &Cell<u32> {
        &self.index
    }
}

pub struct SyntaxNode {
    ptr: ptr::NonNull<NodeData>,
}
impl SyntaxNode {
    pub fn new_root(green: GreenNode) -> SyntaxNode {
        let green = GreenNode::into_raw(green);
        let green = Green::Node { ptr: Cell::new(green) };
        SyntaxNode {
            ptr: NodeData::new(None, 0, 0.into(), green, false),
        }
    }
    pub fn new_root_mut(green: GreenNode) -> SyntaxNode {
        let green = GreenNode::into_raw(green);
        let green = Green::Node { ptr: Cell::new(green) };
        SyntaxNode {
            ptr: NodeData::new(None, 0, 0.into(), green, true),
        }
    }
    fn new_child(green: &GreenNodeData, parent: SyntaxNode, index: u32, offset: TextSize) -> SyntaxNode {
        let mutable = parent.data().mutable;
        let green = Green::Node {
            ptr: Cell::new(green.into()),
        };
        SyntaxNode {
            ptr: NodeData::new(Some(parent), index, offset, green, mutable),
        }
    }
    pub fn clone_for_update(&self) -> SyntaxNode {
        assert!(!self.data().mutable);
        match self.parent() {
            Some(parent) => {
                let parent = parent.clone_for_update();
                SyntaxNode::new_child(self.green_ref(), parent, self.data().index(), self.offset())
            },
            None => SyntaxNode::new_root_mut(self.green_ref().to_owned()),
        }
    }
    pub fn clone_subtree(&self) -> SyntaxNode {
        SyntaxNode::new_root(self.green().into())
    }
    #[inline]
    fn data(&self) -> &NodeData {
        unsafe { self.ptr.as_ref() }
    }
    pub fn replace_with(&self, replacement: GreenNode) -> GreenNode {
        assert_eq!(self.kind(), replacement.kind());
        match &self.parent() {
            None => replacement,
            Some(parent) => {
                let new_parent = parent
                    .green_ref()
                    .replace_child(self.data().index() as usize, replacement.into());
                parent.replace_with(new_parent)
            },
        }
    }
    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        self.data().kind()
    }
    #[inline]
    fn offset(&self) -> TextSize {
        self.data().offset()
    }
    #[inline]
    pub fn text_range(&self) -> TextRange {
        self.data().text_range()
    }
    #[inline]
    pub fn index(&self) -> usize {
        self.data().index() as usize
    }
    #[inline]
    pub fn text(&self) -> SyntaxText {
        SyntaxText::new(self.clone())
    }
    #[inline]
    pub fn green(&self) -> Cow<'_, GreenNodeData> {
        let green_ref = self.green_ref();
        match self.data().mutable {
            false => Cow::Borrowed(green_ref),
            true => Cow::Owned(green_ref.to_owned()),
        }
    }
    #[inline]
    fn green_ref(&self) -> &GreenNodeData {
        self.data().green().into_node().unwrap()
    }
    #[inline]
    pub fn parent(&self) -> Option<SyntaxNode> {
        self.data().parent_node()
    }
    #[inline]
    pub fn ancestors(&self) -> impl Iterator<Item = SyntaxNode> {
        iter::successors(Some(self.clone()), SyntaxNode::parent)
    }
    #[inline]
    pub fn children(&self) -> SyntaxNodeChildren {
        SyntaxNodeChildren::new(self.clone())
    }
    #[inline]
    pub fn children_with_tokens(&self) -> SyntaxElementChildren {
        SyntaxElementChildren::new(self.clone())
    }
    pub fn first_child(&self) -> Option<SyntaxNode> {
        self.green_ref().children().raw.enumerate().find_map(|(index, child)| {
            child.as_ref().into_node().map(|green| {
                SyntaxNode::new_child(green, self.clone(), index as u32, self.offset() + child.rel_offset())
            })
        })
    }
    pub fn last_child(&self) -> Option<SyntaxNode> {
        self.green_ref()
            .children()
            .raw
            .enumerate()
            .rev()
            .find_map(|(index, child)| {
                child.as_ref().into_node().map(|green| {
                    SyntaxNode::new_child(green, self.clone(), index as u32, self.offset() + child.rel_offset())
                })
            })
    }
    pub fn first_child_or_token(&self) -> Option<SyntaxElement> {
        self.green_ref()
            .children()
            .raw
            .next()
            .map(|child| SyntaxElement::new(child.as_ref(), self.clone(), 0, self.offset() + child.rel_offset()))
    }
    pub fn last_child_or_token(&self) -> Option<SyntaxElement> {
        self.green_ref()
            .children()
            .raw
            .enumerate()
            .next_back()
            .map(|(index, child)| {
                SyntaxElement::new(
                    child.as_ref(),
                    self.clone(),
                    index as u32,
                    self.offset() + child.rel_offset(),
                )
            })
    }
    pub fn next_sibling(&self) -> Option<SyntaxNode> {
        self.data().next_sibling()
    }
    pub fn prev_sibling(&self) -> Option<SyntaxNode> {
        self.data().prev_sibling()
    }
    pub fn next_sibling_or_token(&self) -> Option<SyntaxElement> {
        self.data().next_sibling_or_token()
    }
    pub fn prev_sibling_or_token(&self) -> Option<SyntaxElement> {
        self.data().prev_sibling_or_token()
    }
    pub fn first_token(&self) -> Option<SyntaxToken> {
        self.first_child_or_token()?.first_token()
    }
    pub fn last_token(&self) -> Option<SyntaxToken> {
        self.last_child_or_token()?.last_token()
    }
    #[inline]
    pub fn siblings(&self, direction: Direction) -> impl Iterator<Item = SyntaxNode> {
        iter::successors(Some(self.clone()), move |node| match direction {
            Direction::Next => node.next_sibling(),
            Direction::Prev => node.prev_sibling(),
        })
    }
    #[inline]
    pub fn siblings_with_tokens(&self, direction: Direction) -> impl Iterator<Item = SyntaxElement> {
        let me: SyntaxElement = self.clone().into();
        iter::successors(Some(me), move |el| match direction {
            Direction::Next => el.next_sibling_or_token(),
            Direction::Prev => el.prev_sibling_or_token(),
        })
    }
    #[inline]
    pub fn descendants(&self) -> impl Iterator<Item = SyntaxNode> {
        self.preorder().filter_map(|event| match event {
            WalkEvent::Enter(node) => Some(node),
            WalkEvent::Leave(_) => None,
        })
    }
    #[inline]
    pub fn descendants_with_tokens(&self) -> impl Iterator<Item = SyntaxElement> {
        self.preorder_with_tokens().filter_map(|event| match event {
            WalkEvent::Enter(it) => Some(it),
            WalkEvent::Leave(_) => None,
        })
    }
    #[inline]
    pub fn preorder(&self) -> Preorder {
        Preorder::new(self.clone())
    }
    #[inline]
    pub fn preorder_with_tokens(&self) -> PreorderWithTokens {
        PreorderWithTokens::new(self.clone())
    }
    pub fn token_at_offset(&self, offset: TextSize) -> TokenAtOffset<SyntaxToken> {
        let range = self.text_range();
        assert!(
            range.start() <= offset && offset <= range.end(),
            "Bad offset: range {:?} offset {:?}",
            range,
            offset
        );
        if range.is_empty() {
            return TokenAtOffset::None;
        }
        let mut children = self.children_with_tokens().filter(|child| {
            let child_range = child.text_range();
            !child_range.is_empty() && (child_range.start() <= offset && offset <= child_range.end())
        });
        let left = children.next().unwrap();
        let right = children.next();
        assert!(children.next().is_none());
        if let Some(right) = right {
            match (left.token_at_offset(offset), right.token_at_offset(offset)) {
                (TokenAtOffset::Single(left), TokenAtOffset::Single(right)) => TokenAtOffset::Between(left, right),
                _ => unreachable!(),
            }
        } else {
            left.token_at_offset(offset)
        }
    }
    pub fn covering_element(&self, range: TextRange) -> SyntaxElement {
        let mut res: SyntaxElement = self.clone().into();
        loop {
            assert!(
                res.text_range().contains_range(range),
                "Bad range: node range {:?}, range {:?}",
                res.text_range(),
                range,
            );
            res = match &res {
                NodeOrToken::Token(_) => return res,
                NodeOrToken::Node(node) => match node.child_or_token_at_range(range) {
                    Some(it) => it,
                    None => return res,
                },
            };
        }
    }
    pub fn child_or_token_at_range(&self, range: TextRange) -> Option<SyntaxElement> {
        let rel_range = range - self.offset();
        self.green_ref()
            .child_at_range(rel_range)
            .map(|(index, rel_offset, green)| {
                SyntaxElement::new(green, self.clone(), index as u32, self.offset() + rel_offset)
            })
    }
    pub fn splice_children(&self, to_delete: Range<usize>, to_insert: Vec<SyntaxElement>) {
        assert!(self.data().mutable, "immutable tree: {}", self);
        for (i, child) in self.children_with_tokens().enumerate() {
            if to_delete.contains(&i) {
                child.detach();
            }
        }
        let mut index = to_delete.start;
        for child in to_insert {
            self.attach_child(index, child);
            index += 1;
        }
    }
    pub fn detach(&self) {
        assert!(self.data().mutable, "immutable tree: {}", self);
        self.data().detach()
    }
    fn attach_child(&self, index: usize, child: SyntaxElement) {
        assert!(self.data().mutable, "immutable tree: {}", self);
        child.detach();
        let data = match &child {
            NodeOrToken::Node(it) => it.data(),
            NodeOrToken::Token(it) => it.data(),
        };
        self.data().attach_child(index, data)
    }
}
impl Clone for SyntaxNode {
    #[inline]
    fn clone(&self) -> Self {
        self.data().inc_rc();
        SyntaxNode { ptr: self.ptr }
    }
}
impl Drop for SyntaxNode {
    #[inline]
    fn drop(&mut self) {
        if self.data().dec_rc() {
            unsafe { free(self.ptr) }
        }
    }
}
impl Eq for SyntaxNode {}
impl PartialEq for SyntaxNode {
    #[inline]
    fn eq(&self, other: &SyntaxNode) -> bool {
        self.data().key() == other.data().key()
    }
}
impl Hash for SyntaxNode {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data().key().hash(state);
    }
}
impl fmt::Debug for SyntaxNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SyntaxNode")
            .field("kind", &self.kind())
            .field("text_range", &self.text_range())
            .finish()
    }
}
impl fmt::Display for SyntaxNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.preorder_with_tokens()
            .filter_map(|event| match event {
                WalkEvent::Enter(NodeOrToken::Token(token)) => Some(token),
                _ => None,
            })
            .try_for_each(|it| fmt::Display::fmt(&it, f))
    }
}

#[derive(Debug)]
pub struct SyntaxToken {
    ptr: ptr::NonNull<NodeData>,
}
impl SyntaxToken {
    fn new(green: &GreenTokenData, parent: SyntaxNode, index: u32, offset: TextSize) -> SyntaxToken {
        let mutable = parent.data().mutable;
        let green = Green::Token { ptr: green.into() };
        SyntaxToken {
            ptr: NodeData::new(Some(parent), index, offset, green, mutable),
        }
    }
    #[inline]
    fn data(&self) -> &NodeData {
        unsafe { self.ptr.as_ref() }
    }
    pub fn replace_with(&self, replacement: GreenToken) -> GreenNode {
        assert_eq!(self.kind(), replacement.kind());
        let parent = self.parent().unwrap();
        let me: u32 = self.data().index();
        let new_parent = parent.green_ref().replace_child(me as usize, replacement.into());
        parent.replace_with(new_parent)
    }
    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        self.data().kind()
    }
    #[inline]
    pub fn text_range(&self) -> TextRange {
        self.data().text_range()
    }
    #[inline]
    pub fn index(&self) -> usize {
        self.data().index() as usize
    }
    #[inline]
    pub fn text(&self) -> &str {
        match self.data().green().as_token() {
            Some(it) => it.text(),
            None => {
                debug_assert!(
                    false,
                    "corrupted tree: a node thinks it is a token: {:?}",
                    self.data().green().as_node().unwrap().to_string()
                );
                ""
            },
        }
    }
    #[inline]
    pub fn green(&self) -> &GreenTokenData {
        self.data().green().into_token().unwrap()
    }
    #[inline]
    pub fn parent(&self) -> Option<SyntaxNode> {
        self.data().parent_node()
    }
    #[inline]
    pub fn ancestors(&self) -> impl Iterator<Item = SyntaxNode> {
        std::iter::successors(self.parent(), SyntaxNode::parent)
    }
    pub fn next_sibling_or_token(&self) -> Option<SyntaxElement> {
        self.data().next_sibling_or_token()
    }
    pub fn prev_sibling_or_token(&self) -> Option<SyntaxElement> {
        self.data().prev_sibling_or_token()
    }
    #[inline]
    pub fn siblings_with_tokens(&self, direction: Direction) -> impl Iterator<Item = SyntaxElement> {
        let me: SyntaxElement = self.clone().into();
        iter::successors(Some(me), move |el| match direction {
            Direction::Next => el.next_sibling_or_token(),
            Direction::Prev => el.prev_sibling_or_token(),
        })
    }
    pub fn next_token(&self) -> Option<SyntaxToken> {
        match self.next_sibling_or_token() {
            Some(element) => element.first_token(),
            None => self
                .ancestors()
                .find_map(|it| it.next_sibling_or_token())
                .and_then(|element| element.first_token()),
        }
    }
    pub fn prev_token(&self) -> Option<SyntaxToken> {
        match self.prev_sibling_or_token() {
            Some(element) => element.last_token(),
            None => self
                .ancestors()
                .find_map(|it| it.prev_sibling_or_token())
                .and_then(|element| element.last_token()),
        }
    }
    pub fn detach(&self) {
        assert!(self.data().mutable, "immutable tree: {}", self);
        self.data().detach()
    }
}
impl Clone for SyntaxToken {
    #[inline]
    fn clone(&self) -> Self {
        self.data().inc_rc();
        SyntaxToken { ptr: self.ptr }
    }
}
impl Drop for SyntaxToken {
    #[inline]
    fn drop(&mut self) {
        if self.data().dec_rc() {
            unsafe { free(self.ptr) }
        }
    }
}
impl Eq for SyntaxToken {}
impl PartialEq for SyntaxToken {
    #[inline]
    fn eq(&self, other: &SyntaxToken) -> bool {
        self.data().key() == other.data().key()
    }
}
impl Hash for SyntaxToken {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data().key().hash(state);
    }
}
impl fmt::Display for SyntaxToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.text(), f)
    }
}

pub type SyntaxElement = NodeOrToken<SyntaxNode, SyntaxToken>;
impl SyntaxElement {
    fn new(element: GreenElementRef<'_>, parent: SyntaxNode, index: u32, offset: TextSize) -> SyntaxElement {
        match element {
            NodeOrToken::Node(node) => SyntaxNode::new_child(node, parent, index as u32, offset).into(),
            NodeOrToken::Token(token) => SyntaxToken::new(token, parent, index as u32, offset).into(),
        }
    }
    #[inline]
    pub fn text_range(&self) -> TextRange {
        match self {
            NodeOrToken::Node(it) => it.text_range(),
            NodeOrToken::Token(it) => it.text_range(),
        }
    }
    #[inline]
    pub fn index(&self) -> usize {
        match self {
            NodeOrToken::Node(it) => it.index(),
            NodeOrToken::Token(it) => it.index(),
        }
    }
    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        match self {
            NodeOrToken::Node(it) => it.kind(),
            NodeOrToken::Token(it) => it.kind(),
        }
    }
    #[inline]
    pub fn parent(&self) -> Option<SyntaxNode> {
        match self {
            NodeOrToken::Node(it) => it.parent(),
            NodeOrToken::Token(it) => it.parent(),
        }
    }
    #[inline]
    pub fn ancestors(&self) -> impl Iterator<Item = SyntaxNode> {
        let first = match self {
            NodeOrToken::Node(it) => Some(it.clone()),
            NodeOrToken::Token(it) => it.parent(),
        };
        iter::successors(first, SyntaxNode::parent)
    }
    pub fn first_token(&self) -> Option<SyntaxToken> {
        match self {
            NodeOrToken::Node(it) => it.first_token(),
            NodeOrToken::Token(it) => Some(it.clone()),
        }
    }
    pub fn last_token(&self) -> Option<SyntaxToken> {
        match self {
            NodeOrToken::Node(it) => it.last_token(),
            NodeOrToken::Token(it) => Some(it.clone()),
        }
    }
    pub fn next_sibling_or_token(&self) -> Option<SyntaxElement> {
        match self {
            NodeOrToken::Node(it) => it.next_sibling_or_token(),
            NodeOrToken::Token(it) => it.next_sibling_or_token(),
        }
    }
    pub fn prev_sibling_or_token(&self) -> Option<SyntaxElement> {
        match self {
            NodeOrToken::Node(it) => it.prev_sibling_or_token(),
            NodeOrToken::Token(it) => it.prev_sibling_or_token(),
        }
    }
    fn token_at_offset(&self, offset: TextSize) -> TokenAtOffset<SyntaxToken> {
        assert!(self.text_range().start() <= offset && offset <= self.text_range().end());
        match self {
            NodeOrToken::Token(token) => TokenAtOffset::Single(token.clone()),
            NodeOrToken::Node(node) => node.token_at_offset(offset),
        }
    }
    pub fn detach(&self) {
        match self {
            NodeOrToken::Node(it) => it.detach(),
            NodeOrToken::Token(it) => it.detach(),
        }
    }
}
impl From<SyntaxNode> for SyntaxElement {
    #[inline]
    fn from(node: SyntaxNode) -> SyntaxElement {
        NodeOrToken::Node(node)
    }
}
impl From<SyntaxToken> for SyntaxElement {
    #[inline]
    fn from(token: SyntaxToken) -> SyntaxElement {
        NodeOrToken::Token(token)
    }
}

#[derive(Clone, Debug)]
pub struct SyntaxNodeChildren {
    next: Option<SyntaxNode>,
}
impl SyntaxNodeChildren {
    fn new(parent: SyntaxNode) -> SyntaxNodeChildren {
        SyntaxNodeChildren {
            next: parent.first_child(),
        }
    }
}
impl Iterator for SyntaxNodeChildren {
    type Item = SyntaxNode;
    fn next(&mut self) -> Option<SyntaxNode> {
        self.next.take().map(|next| {
            self.next = next.next_sibling();
            next
        })
    }
}

#[derive(Clone, Debug)]
pub struct SyntaxElementChildren {
    next: Option<SyntaxElement>,
}
impl SyntaxElementChildren {
    fn new(parent: SyntaxNode) -> SyntaxElementChildren {
        SyntaxElementChildren {
            next: parent.first_child_or_token(),
        }
    }
}
impl Iterator for SyntaxElementChildren {
    type Item = SyntaxElement;
    fn next(&mut self) -> Option<SyntaxElement> {
        self.next.take().map(|next| {
            self.next = next.next_sibling_or_token();
            next
        })
    }
}

pub struct Preorder {
    start: SyntaxNode,
    next: Option<WalkEvent<SyntaxNode>>,
    skip_subtree: bool,
}
impl Preorder {
    fn new(start: SyntaxNode) -> Preorder {
        let next = Some(WalkEvent::Enter(start.clone()));
        Preorder {
            start,
            next,
            skip_subtree: false,
        }
    }
    pub fn skip_subtree(&mut self) {
        self.skip_subtree = true;
    }
    #[cold]
    fn do_skip(&mut self) {
        self.next = self.next.take().map(|next| match next {
            WalkEvent::Enter(first_child) => WalkEvent::Leave(first_child.parent().unwrap()),
            WalkEvent::Leave(parent) => WalkEvent::Leave(parent),
        })
    }
}
impl Iterator for Preorder {
    type Item = WalkEvent<SyntaxNode>;
    fn next(&mut self) -> Option<WalkEvent<SyntaxNode>> {
        if self.skip_subtree {
            self.do_skip();
            self.skip_subtree = false;
        }
        let next = self.next.take();
        self.next = next.as_ref().and_then(|next| {
            Some(match next {
                WalkEvent::Enter(node) => match node.first_child() {
                    Some(child) => WalkEvent::Enter(child),
                    None => WalkEvent::Leave(node.clone()),
                },
                WalkEvent::Leave(node) => {
                    if node == &self.start {
                        return None;
                    }
                    match node.next_sibling() {
                        Some(sibling) => WalkEvent::Enter(sibling),
                        None => WalkEvent::Leave(node.parent()?),
                    }
                },
            })
        });
        next
    }
}

pub struct PreorderWithTokens {
    start: SyntaxElement,
    next: Option<WalkEvent<SyntaxElement>>,
    skip_subtree: bool,
}
impl PreorderWithTokens {
    fn new(start: SyntaxNode) -> PreorderWithTokens {
        let next = Some(WalkEvent::Enter(start.clone().into()));
        PreorderWithTokens {
            start: start.into(),
            next,
            skip_subtree: false,
        }
    }
    pub fn skip_subtree(&mut self) {
        self.skip_subtree = true;
    }
    #[cold]
    fn do_skip(&mut self) {
        self.next = self.next.take().map(|next| match next {
            WalkEvent::Enter(first_child) => WalkEvent::Leave(first_child.parent().unwrap().into()),
            WalkEvent::Leave(parent) => WalkEvent::Leave(parent),
        })
    }
}
impl Iterator for PreorderWithTokens {
    type Item = WalkEvent<SyntaxElement>;
    fn next(&mut self) -> Option<WalkEvent<SyntaxElement>> {
        if self.skip_subtree {
            self.do_skip();
            self.skip_subtree = false;
        }
        let next = self.next.take();
        self.next = next.as_ref().and_then(|next| {
            Some(match next {
                WalkEvent::Enter(el) => match el {
                    NodeOrToken::Node(node) => match node.first_child_or_token() {
                        Some(child) => WalkEvent::Enter(child),
                        None => WalkEvent::Leave(node.clone().into()),
                    },
                    NodeOrToken::Token(token) => WalkEvent::Leave(token.clone().into()),
                },
                WalkEvent::Leave(el) if el == &self.start => return None,
                WalkEvent::Leave(el) => match el.next_sibling_or_token() {
                    Some(sibling) => WalkEvent::Enter(sibling),
                    None => WalkEvent::Leave(el.parent()?.into()),
                },
            })
        });
        next
    }
}

#[inline(never)]
unsafe fn free(mut data: ptr::NonNull<NodeData>) {
    loop {
        debug_assert_eq!(data.as_ref().rc.get(), 0);
        debug_assert!(data.as_ref().first.get().is_null());
        let node = Box::from_raw(data.as_ptr());
        match node.parent.take() {
            Some(parent) => {
                debug_assert!(parent.as_ref().rc.get() > 0);
                if node.mutable {
                    sll::unlink(&parent.as_ref().first, &*node)
                }
                if parent.as_ref().dec_rc() {
                    data = parent;
                } else {
                    break;
                }
            },
            None => {
                match &node.green {
                    Green::Node { ptr } => {
                        let _ = GreenNode::from_raw(ptr.get());
                    },
                    Green::Token { ptr } => {
                        let _ = GreenToken::from_raw(*ptr);
                    },
                }
                break;
            },
        }
    }
}
