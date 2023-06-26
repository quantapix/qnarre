use super::*;
use std::{borrow::Cow, fmt, iter, marker::PhantomData, ops::Range};

pub trait Lang: Sized + Copy + fmt::Debug + Eq + Ord + std::hash::Hash {
    type Kind: Sized + Copy + fmt::Debug + Eq + Ord + std::hash::Hash;
    fn kind_from_raw(x: green::Kind) -> Self::Kind;
    fn kind_to_raw(x: Self::Kind) -> green::Kind;
}

pub struct Preorder<L: Lang> {
    raw: cursor::Preorder,
    _p: PhantomData<L>,
}
impl<L: Lang> Preorder<L> {
    pub fn skip_subtree(&mut self) {
        self.raw.skip_subtree()
    }
}
impl<L: Lang> Iterator for Preorder<L> {
    type Item = WalkEvent<Node<L>>;
    fn next(&mut self) -> Option<Self::Item> {
        self.raw.next().map(|it| it.map(Node::from))
    }
}

pub struct PreorderWithToks<L: Lang> {
    raw: cursor::PreorderWithToks,
    _p: PhantomData<L>,
}
impl<L: Lang> PreorderWithToks<L> {
    pub fn skip_subtree(&mut self) {
        self.raw.skip_subtree()
    }
}
impl<L: Lang> Iterator for PreorderWithToks<L> {
    type Item = WalkEvent<Elem<L>>;
    fn next(&mut self) -> Option<Self::Item> {
        self.raw.next().map(|it| it.map(Elem::from))
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Node<L: Lang> {
    raw: cursor::Node,
    _p: PhantomData<L>,
}
impl<L: Lang> Node<L> {
    pub fn new_root(x: green::Node) -> Node<L> {
        Node::from(cursor::Node::new_root(x))
    }
    pub fn replace_with(&self, x: green::Node) -> green::Node {
        self.raw.replace_with(x)
    }
    pub fn kind(&self) -> L::Kind {
        L::kind_from_raw(self.raw.kind())
    }
    pub fn text_range(&self) -> crate::TextRange {
        self.raw.text_range()
    }
    pub fn index(&self) -> usize {
        self.raw.index()
    }
    pub fn text(&self) -> Text {
        self.raw.text()
    }
    pub fn green(&self) -> Cow<'_, green::NodeData> {
        self.raw.green()
    }
    pub fn parent(&self) -> Option<Node<L>> {
        self.raw.parent().map(Self::from)
    }
    pub fn ancestors(&self) -> impl Iterator<Item = Node<L>> {
        self.raw.ancestors().map(Node::from)
    }
    pub fn children(&self) -> NodeChildren<L> {
        NodeChildren {
            raw: self.raw.children(),
            _p: PhantomData,
        }
    }
    pub fn children_with_tokens(&self) -> ElemChildren<L> {
        ElemChildren {
            raw: self.raw.children_with_tokens(),
            _p: PhantomData,
        }
    }
    pub fn first_child(&self) -> Option<Node<L>> {
        self.raw.first_child().map(Self::from)
    }
    pub fn last_child(&self) -> Option<Node<L>> {
        self.raw.last_child().map(Self::from)
    }
    pub fn first_child_or_token(&self) -> Option<Elem<L>> {
        self.raw.first_child_or_token().map(NodeOrToken::from)
    }
    pub fn last_child_or_token(&self) -> Option<Elem<L>> {
        self.raw.last_child_or_token().map(NodeOrToken::from)
    }
    pub fn next_sibling(&self) -> Option<Node<L>> {
        self.raw.next_sibling().map(Self::from)
    }
    pub fn prev_sibling(&self) -> Option<Node<L>> {
        self.raw.prev_sibling().map(Self::from)
    }
    pub fn next_sibling_or_token(&self) -> Option<Elem<L>> {
        self.raw.next_sibling_or_token().map(NodeOrToken::from)
    }
    pub fn prev_sibling_or_token(&self) -> Option<Elem<L>> {
        self.raw.prev_sibling_or_token().map(NodeOrToken::from)
    }
    pub fn first_token(&self) -> Option<Token<L>> {
        self.raw.first_token().map(Token::from)
    }
    pub fn last_token(&self) -> Option<Token<L>> {
        self.raw.last_token().map(Token::from)
    }
    pub fn siblings(&self, direction: Direction) -> impl Iterator<Item = Node<L>> {
        self.raw.siblings(direction).map(Node::from)
    }
    pub fn siblings_with_tokens(&self, direction: Direction) -> impl Iterator<Item = Elem<L>> {
        self.raw.siblings_with_tokens(direction).map(Elem::from)
    }
    pub fn descendants(&self) -> impl Iterator<Item = Node<L>> {
        self.raw.descendants().map(Node::from)
    }
    pub fn descendants_with_tokens(&self) -> impl Iterator<Item = Elem<L>> {
        self.raw.descendants_with_tokens().map(NodeOrToken::from)
    }
    pub fn preorder(&self) -> Preorder<L> {
        Preorder {
            raw: self.raw.preorder(),
            _p: PhantomData,
        }
    }
    pub fn preorder_with_tokens(&self) -> PreorderWithToks<L> {
        PreorderWithToks {
            raw: self.raw.preorder_with_tokens(),
            _p: PhantomData,
        }
    }
    pub fn token_at_offset(&self, offset: crate::TextSize) -> TokAtOffset<Token<L>> {
        self.raw.token_at_offset(offset).map(Token::from)
    }
    pub fn covering_element(&self, range: crate::TextRange) -> Elem<L> {
        NodeOrToken::from(self.raw.covering_element(range))
    }
    pub fn child_or_token_at_range(&self, range: crate::TextRange) -> Option<Elem<L>> {
        self.raw.child_or_token_at_range(range).map(Elem::from)
    }
    pub fn clone_subtree(&self) -> Node<L> {
        Node::from(self.raw.clone_subtree())
    }
    pub fn clone_for_update(&self) -> Node<L> {
        Node::from(self.raw.clone_for_update())
    }
    pub fn detach(&self) {
        self.raw.detach()
    }
    pub fn splice_children(&self, to_delete: Range<usize>, to_insert: Vec<Elem<L>>) {
        let to_insert = to_insert.into_iter().map(cursor::Elem::from).collect::<Vec<_>>();
        self.raw.splice_children(to_delete, to_insert)
    }
}
impl<L: Lang> fmt::Debug for Node<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            let mut level = 0;
            for event in self.preorder_with_tokens() {
                match event {
                    WalkEvent::Enter(element) => {
                        for _ in 0..level {
                            write!(f, "  ")?;
                        }
                        match element {
                            NodeOrToken::Node(node) => writeln!(f, "{:?}", node)?,
                            NodeOrToken::Token(token) => writeln!(f, "{:?}", token)?,
                        }
                        level += 1;
                    },
                    WalkEvent::Leave(_) => level -= 1,
                }
            }
            assert_eq!(level, 0);
            Ok(())
        } else {
            write!(f, "{:?}@{:?}", self.kind(), self.text_range())
        }
    }
}
impl<L: Lang> fmt::Display for Node<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.raw, f)
    }
}
impl<L: Lang> From<cursor::Node> for Node<L> {
    fn from(raw: cursor::Node) -> Node<L> {
        Node { raw, _p: PhantomData }
    }
}
impl<L: Lang> From<Node<L>> for cursor::Node {
    fn from(node: Node<L>) -> cursor::Node {
        node.raw
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Token<L: Lang> {
    raw: cursor::Token,
    _p: PhantomData<L>,
}
impl<L: Lang> Token<L> {
    pub fn replace_with(&self, new_token: green::Token) -> green::Node {
        self.raw.replace_with(new_token)
    }
    pub fn kind(&self) -> L::Kind {
        L::kind_from_raw(self.raw.kind())
    }
    pub fn text_range(&self) -> crate::TextRange {
        self.raw.text_range()
    }
    pub fn index(&self) -> usize {
        self.raw.index()
    }
    pub fn text(&self) -> &str {
        self.raw.text()
    }
    pub fn green(&self) -> &green::TokData {
        self.raw.green()
    }
    pub fn parent(&self) -> Option<Node<L>> {
        self.raw.parent().map(Node::from)
    }
    #[deprecated = "use `Token::parent_ancestors` instead"]
    pub fn ancestors(&self) -> impl Iterator<Item = Node<L>> {
        self.parent_ancestors()
    }
    pub fn parent_ancestors(&self) -> impl Iterator<Item = Node<L>> {
        self.raw.ancestors().map(Node::from)
    }
    pub fn next_sibling_or_token(&self) -> Option<Elem<L>> {
        self.raw.next_sibling_or_token().map(NodeOrToken::from)
    }
    pub fn prev_sibling_or_token(&self) -> Option<Elem<L>> {
        self.raw.prev_sibling_or_token().map(NodeOrToken::from)
    }
    pub fn siblings_with_tokens(&self, direction: Direction) -> impl Iterator<Item = Elem<L>> {
        self.raw.siblings_with_tokens(direction).map(Elem::from)
    }
    pub fn next_token(&self) -> Option<Token<L>> {
        self.raw.next_token().map(Token::from)
    }
    pub fn prev_token(&self) -> Option<Token<L>> {
        self.raw.prev_token().map(Token::from)
    }
    pub fn detach(&self) {
        self.raw.detach()
    }
}
impl<L: Lang> fmt::Debug for Token<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}@{:?}", self.kind(), self.text_range())?;
        if self.text().len() < 25 {
            return write!(f, " {:?}", self.text());
        }
        let text = self.text();
        for idx in 21..25 {
            if text.is_char_boundary(idx) {
                let text = format!("{} ...", &text[..idx]);
                return write!(f, " {:?}", text);
            }
        }
        unreachable!()
    }
}
impl<L: Lang> fmt::Display for Token<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.raw, f)
    }
}
impl<L: Lang> From<cursor::Token> for Token<L> {
    fn from(raw: cursor::Token) -> Token<L> {
        Token { raw, _p: PhantomData }
    }
}
impl<L: Lang> From<Token<L>> for cursor::Token {
    fn from(token: Token<L>) -> cursor::Token {
        token.raw
    }
}

pub type Elem<L> = NodeOrToken<Node<L>, Token<L>>;
impl<L: Lang> Elem<L> {
    pub fn text_range(&self) -> crate::TextRange {
        match self {
            NodeOrToken::Node(it) => it.text_range(),
            NodeOrToken::Token(it) => it.text_range(),
        }
    }
    pub fn index(&self) -> usize {
        match self {
            NodeOrToken::Node(it) => it.index(),
            NodeOrToken::Token(it) => it.index(),
        }
    }
    pub fn kind(&self) -> L::Kind {
        match self {
            NodeOrToken::Node(it) => it.kind(),
            NodeOrToken::Token(it) => it.kind(),
        }
    }
    pub fn parent(&self) -> Option<Node<L>> {
        match self {
            NodeOrToken::Node(it) => it.parent(),
            NodeOrToken::Token(it) => it.parent(),
        }
    }
    pub fn ancestors(&self) -> impl Iterator<Item = Node<L>> {
        let first = match self {
            NodeOrToken::Node(it) => Some(it.clone()),
            NodeOrToken::Token(it) => it.parent(),
        };
        iter::successors(first, Node::parent)
    }
    pub fn next_sibling_or_token(&self) -> Option<Elem<L>> {
        match self {
            NodeOrToken::Node(it) => it.next_sibling_or_token(),
            NodeOrToken::Token(it) => it.next_sibling_or_token(),
        }
    }
    pub fn prev_sibling_or_token(&self) -> Option<Elem<L>> {
        match self {
            NodeOrToken::Node(it) => it.prev_sibling_or_token(),
            NodeOrToken::Token(it) => it.prev_sibling_or_token(),
        }
    }
    pub fn detach(&self) {
        match self {
            NodeOrToken::Node(it) => it.detach(),
            NodeOrToken::Token(it) => it.detach(),
        }
    }
}
impl<L: Lang> From<Node<L>> for Elem<L> {
    fn from(node: Node<L>) -> Elem<L> {
        NodeOrToken::Node(node)
    }
}
impl<L: Lang> From<Token<L>> for Elem<L> {
    fn from(token: Token<L>) -> Elem<L> {
        NodeOrToken::Token(token)
    }
}
impl<L: Lang> From<cursor::Elem> for Elem<L> {
    fn from(raw: cursor::Elem) -> Elem<L> {
        match raw {
            NodeOrToken::Node(it) => NodeOrToken::Node(it.into()),
            NodeOrToken::Token(it) => NodeOrToken::Token(it.into()),
        }
    }
}
impl<L: Lang> From<Elem<L>> for cursor::Elem {
    fn from(element: Elem<L>) -> cursor::Elem {
        match element {
            NodeOrToken::Node(it) => NodeOrToken::Node(it.into()),
            NodeOrToken::Token(it) => NodeOrToken::Token(it.into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NodeChildren<L: Lang> {
    raw: cursor::NodeChildren,
    _p: PhantomData<L>,
}
impl<L: Lang> Iterator for NodeChildren<L> {
    type Item = Node<L>;
    fn next(&mut self) -> Option<Self::Item> {
        self.raw.next().map(Node::from)
    }
}

#[derive(Debug, Clone)]
pub struct ElemChildren<L: Lang> {
    raw: cursor::ElemChildren,
    _p: PhantomData<L>,
}
impl<L: Lang> Iterator for ElemChildren<L> {
    type Item = Elem<L>;
    fn next(&mut self) -> Option<Self::Item> {
        self.raw.next().map(NodeOrToken::from)
    }
}
