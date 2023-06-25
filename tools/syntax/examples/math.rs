//! Example that takes the input
//! 1 + 2 * 3 + 4
//! and builds the tree
//! - Marker(Root)
//!   - Marker(Operation)
//!     - Marker(Operation)
//!       - "1" Token(Number)
//!       - "+" Token(Add)
//!       - Marker(Operation)
//!         - "2" Token(Number)
//!         - "*" Token(Mul)
//!         - "3" Token(Number)
//!     - "+" Token(Add)
//!     - "4" Token(Number)

use std::iter::Peekable;
use syntax::core::{api, green, NodeOrToken};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
#[repr(u16)]
enum SyntaxKind {
    WHITESPACE = 0,
    ADD,
    SUB,
    MUL,
    DIV,
    NUMBER,
    ERROR,
    OPERATION,
    ROOT,
}
use SyntaxKind::*;

impl From<SyntaxKind> for core::SyntaxKind {
    fn from(kind: SyntaxKind) -> Self {
        Self(kind as u16)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Lang {}
impl core::Language for Lang {
    type Kind = SyntaxKind;
    fn kind_from_raw(raw: core::SyntaxKind) -> Self::Kind {
        assert!(raw.0 <= ROOT as u16);
        unsafe { std::mem::transmute::<u16, SyntaxKind>(raw.0) }
    }
    fn kind_to_raw(kind: Self::Kind) -> core::SyntaxKind {
        kind.into()
    }
}

type Node = core::api::Node<Lang>;
#[allow(unused)]
type Token = core::api::Token<Lang>;
#[allow(unused)]
type Elem = core::NodeOrToken<api::Node, api::Token>;

struct Parser<I: Iterator<Item = (SyntaxKind, String)>> {
    builder: green::NodeBuilder<'static>,
    iter: Peekable<I>,
}
impl<I: Iterator<Item = (SyntaxKind, String)>> Parser<I> {
    fn peek(&mut self) -> Option<SyntaxKind> {
        while self.iter.peek().map(|&(t, _)| t == WHITESPACE).unwrap_or(false) {
            self.bump();
        }
        self.iter.peek().map(|&(t, _)| t)
    }
    fn bump(&mut self) {
        if let Some((token, string)) = self.iter.next() {
            self.builder.token(token.into(), string.as_str());
        }
    }
    fn parse_val(&mut self) {
        match self.peek() {
            Some(NUMBER) => self.bump(),
            _ => {
                self.builder.start_node(ERROR.into());
                self.bump();
                self.builder.finish_node();
            },
        }
    }
    fn handle_operation(&mut self, tokens: &[SyntaxKind], next: fn(&mut Self)) {
        let checkpoint = self.builder.checkpoint();
        next(self);
        while self.peek().map(|t| tokens.contains(&t)).unwrap_or(false) {
            self.builder.start_node_at(checkpoint, OPERATION.into());
            self.bump();
            next(self);
            self.builder.finish_node();
        }
    }
    fn parse_mul(&mut self) {
        self.handle_operation(&[MUL, DIV], Self::parse_val)
    }
    fn parse_add(&mut self) {
        self.handle_operation(&[ADD, SUB], Self::parse_mul)
    }
    fn parse(mut self) -> api::Node {
        self.builder.start_node(ROOT.into());
        self.parse_add();
        self.builder.finish_node();

        api::Node::new_root(self.builder.finish())
    }
}

fn print(indent: usize, element: api::Elem) {
    let kind: SyntaxKind = element.kind().into();
    print!("{:indent$}", "", indent = indent);
    match element {
        NodeOrToken::Node(node) => {
            println!("- {:?}", kind);
            for child in node.children_with_tokens() {
                print(indent + 2, child);
            }
        },

        NodeOrToken::Token(token) => println!("- {:?} {:?}", token.text(), kind),
    }
}

fn main() {
    let ast = Parser {
        builder: green::NodeBuilder::new(),
        iter: vec![
            // 1 + 2 * 3 + 4
            (NUMBER, "1".into()),
            (WHITESPACE, " ".into()),
            (ADD, "+".into()),
            (WHITESPACE, " ".into()),
            (NUMBER, "2".into()),
            (WHITESPACE, " ".into()),
            (MUL, "*".into()),
            (WHITESPACE, " ".into()),
            (NUMBER, "3".into()),
            (WHITESPACE, " ".into()),
            (ADD, "+".into()),
            (WHITESPACE, " ".into()),
            (NUMBER, "4".into()),
        ]
        .into_iter()
        .peekable(),
    }
    .parse();
    print(0, ast.into());
}
