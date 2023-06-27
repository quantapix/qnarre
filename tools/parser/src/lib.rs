#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]
#![allow(rustdoc::private_intra_doc_links)]

pub use crate::{
    input::Input,
    lexed_str::LexedStr,
    output::{Output, Step},
    shortcuts::StrStep,
    syntax_kind::SyntaxKind,
};
pub use token_set::TokenSet;

#[derive(Debug)]
pub enum TopEntryPoint {
    SourceFile,
    MacroStmts,
    MacroItems,
    Pattern,
    Type,
    Expr,
    MetaItem,
}
impl TopEntryPoint {
    pub fn parse(&self, input: &Input) -> Output {
        let entry_point: fn(&'_ mut parser::Parser<'_>) = match self {
            TopEntryPoint::SourceFile => grammar::top::source_file,
            TopEntryPoint::MacroStmts => grammar::top::macro_stmts,
            TopEntryPoint::MacroItems => grammar::top::macro_items,
            TopEntryPoint::Pattern => grammar::top::pattern,
            TopEntryPoint::Type => grammar::top::type_,
            TopEntryPoint::Expr => grammar::top::expr,
            TopEntryPoint::MetaItem => grammar::top::meta_item,
        };
        let mut p = parser::Parser::new(input);
        entry_point(&mut p);
        let events = p.finish();
        let res = event::process(events);
        if cfg!(debug_assertions) {
            let mut depth = 0;
            let mut first = true;
            for step in res.iter() {
                assert!(depth > 0 || first);
                first = false;
                match step {
                    Step::Enter { .. } => depth += 1,
                    Step::Exit => depth -= 1,
                    Step::FloatSplit {
                        ends_in_dot: has_pseudo_dot,
                    } => depth -= 1 + !has_pseudo_dot as usize,
                    Step::Token { .. } | Step::Error { .. } => (),
                }
            }
            assert!(!first, "no tree at all");
            assert_eq!(depth, 0, "unbalanced tree");
        }
        res
    }
}

#[derive(Debug)]
pub enum PrefixEntryPoint {
    Vis,
    Block,
    Stmt,
    Pat,
    PatTop,
    Ty,
    Expr,
    Path,
    Item,
    MetaItem,
}
impl PrefixEntryPoint {
    pub fn parse(&self, input: &Input) -> Output {
        let entry_point: fn(&'_ mut parser::Parser<'_>) = match self {
            PrefixEntryPoint::Vis => grammar::prefix::vis,
            PrefixEntryPoint::Block => grammar::prefix::block,
            PrefixEntryPoint::Stmt => grammar::prefix::stmt,
            PrefixEntryPoint::Pat => grammar::prefix::pat,
            PrefixEntryPoint::PatTop => grammar::prefix::pat_top,
            PrefixEntryPoint::Ty => grammar::prefix::ty,
            PrefixEntryPoint::Expr => grammar::prefix::expr,
            PrefixEntryPoint::Path => grammar::prefix::path,
            PrefixEntryPoint::Item => grammar::prefix::item,
            PrefixEntryPoint::MetaItem => grammar::prefix::meta_item,
        };
        let mut p = parser::Parser::new(input);
        entry_point(&mut p);
        let events = p.finish();
        event::process(events)
    }
}

pub struct Reparser(fn(&mut parser::Parser<'_>));
impl Reparser {
    pub fn for_node(node: SyntaxKind, first_child: Option<SyntaxKind>, parent: Option<SyntaxKind>) -> Option<Reparser> {
        grammar::reparser(node, first_child, parent).map(Reparser)
    }
    pub fn parse(self, tokens: &Input) -> Output {
        let Reparser(r) = self;
        let mut p = parser::Parser::new(tokens);
        r(&mut p);
        let events = p.finish();
        event::process(events)
    }
}

mod event {
    use crate::{
        output::Output,
        SyntaxKind::{self, *},
    };
    use std::mem;
    #[derive(Debug)]
    pub enum Event {
        Start {
            kind: SyntaxKind,
            forward_parent: Option<u32>,
        },
        Finish,
        Token {
            kind: SyntaxKind,
            n_raw_tokens: u8,
        },
        FloatSplitHack {
            ends_in_dot: bool,
        },
        Error {
            msg: String,
        },
    }
    impl Event {
        pub fn tombstone() -> Self {
            Event::Start {
                kind: TOMBSTONE,
                forward_parent: None,
            }
        }
    }
    pub fn process(mut events: Vec<Event>) -> Output {
        let mut res = Output::default();
        let mut forward_parents = Vec::new();
        for i in 0..events.len() {
            match mem::replace(&mut events[i], Event::tombstone()) {
                Event::Start { kind, forward_parent } => {
                    forward_parents.push(kind);
                    let mut idx = i;
                    let mut fp = forward_parent;
                    while let Some(fwd) = fp {
                        idx += fwd as usize;
                        fp = match mem::replace(&mut events[idx], Event::tombstone()) {
                            Event::Start { kind, forward_parent } => {
                                forward_parents.push(kind);
                                forward_parent
                            },
                            _ => unreachable!(),
                        };
                    }
                    for kind in forward_parents.drain(..).rev() {
                        if kind != TOMBSTONE {
                            res.enter_node(kind);
                        }
                    }
                },
                Event::Finish => res.leave_node(),
                Event::Token { kind, n_raw_tokens } => {
                    res.token(kind, n_raw_tokens);
                },
                Event::FloatSplitHack { ends_in_dot } => {
                    res.float_split_hack(ends_in_dot);
                    let ev = mem::replace(&mut events[i + 1], Event::tombstone());
                    assert!(matches!(ev, Event::Finish), "{ev:?}");
                },
                Event::Error { msg } => res.error(msg),
            }
        }
        res
    }
}
mod grammar;
mod input {
    use crate::SyntaxKind;
    #[allow(non_camel_case_types)]
    type bits = u64;
    #[derive(Default)]
    pub struct Input {
        kind: Vec<SyntaxKind>,
        joint: Vec<bits>,
        contextual_kind: Vec<SyntaxKind>,
    }
    impl Input {
        #[inline]
        pub fn push(&mut self, kind: SyntaxKind) {
            self.push_impl(kind, SyntaxKind::EOF)
        }
        #[inline]
        pub fn push_ident(&mut self, contextual_kind: SyntaxKind) {
            self.push_impl(SyntaxKind::IDENT, contextual_kind)
        }
        #[inline]
        pub fn was_joint(&mut self) {
            let n = self.len() - 1;
            let (idx, b_idx) = self.bit_index(n);
            self.joint[idx] |= 1 << b_idx;
        }
        #[inline]
        fn push_impl(&mut self, kind: SyntaxKind, contextual_kind: SyntaxKind) {
            let idx = self.len();
            if idx % (bits::BITS as usize) == 0 {
                self.joint.push(0);
            }
            self.kind.push(kind);
            self.contextual_kind.push(contextual_kind);
        }
    }
    impl Input {
        pub fn kind(&self, idx: usize) -> SyntaxKind {
            self.kind.get(idx).copied().unwrap_or(SyntaxKind::EOF)
        }
        pub fn contextual_kind(&self, idx: usize) -> SyntaxKind {
            self.contextual_kind.get(idx).copied().unwrap_or(SyntaxKind::EOF)
        }
        pub fn is_joint(&self, n: usize) -> bool {
            let (idx, b_idx) = self.bit_index(n);
            self.joint[idx] & 1 << b_idx != 0
        }
    }
    impl Input {
        fn bit_index(&self, n: usize) -> (usize, usize) {
            let idx = n / (bits::BITS as usize);
            let b_idx = n % (bits::BITS as usize);
            (idx, b_idx)
        }
        fn len(&self) -> usize {
            self.kind.len()
        }
    }
}
mod lexed_str;
mod output {
    use crate::SyntaxKind;
    #[derive(Default)]
    pub struct Output {
        event: Vec<u32>,
        error: Vec<String>,
    }
    #[derive(Debug)]
    pub enum Step<'a> {
        Token { kind: SyntaxKind, n_input_tokens: u8 },
        FloatSplit { ends_in_dot: bool },
        Enter { kind: SyntaxKind },
        Exit,
        Error { msg: &'a str },
    }
    impl Output {
        const EVENT_MASK: u32 = 0b1;
        const TAG_MASK: u32 = 0x0000_00F0;
        const N_INPUT_TOKEN_MASK: u32 = 0x0000_FF00;
        const KIND_MASK: u32 = 0xFFFF_0000;
        const ERROR_SHIFT: u32 = Self::EVENT_MASK.trailing_ones();
        const TAG_SHIFT: u32 = Self::TAG_MASK.trailing_zeros();
        const N_INPUT_TOKEN_SHIFT: u32 = Self::N_INPUT_TOKEN_MASK.trailing_zeros();
        const KIND_SHIFT: u32 = Self::KIND_MASK.trailing_zeros();
        const TOKEN_EVENT: u8 = 0;
        const ENTER_EVENT: u8 = 1;
        const EXIT_EVENT: u8 = 2;
        const SPLIT_EVENT: u8 = 3;
        pub fn iter(&self) -> impl Iterator<Item = Step<'_>> {
            self.event.iter().map(|&event| {
                if event & Self::EVENT_MASK == 0 {
                    return Step::Error {
                        msg: self.error[(event as usize) >> Self::ERROR_SHIFT].as_str(),
                    };
                }
                let tag = ((event & Self::TAG_MASK) >> Self::TAG_SHIFT) as u8;
                match tag {
                    Self::TOKEN_EVENT => {
                        let kind: SyntaxKind = (((event & Self::KIND_MASK) >> Self::KIND_SHIFT) as u16).into();
                        let n_input_tokens = ((event & Self::N_INPUT_TOKEN_MASK) >> Self::N_INPUT_TOKEN_SHIFT) as u8;
                        Step::Token { kind, n_input_tokens }
                    },
                    Self::ENTER_EVENT => {
                        let kind: SyntaxKind = (((event & Self::KIND_MASK) >> Self::KIND_SHIFT) as u16).into();
                        Step::Enter { kind }
                    },
                    Self::EXIT_EVENT => Step::Exit,
                    Self::SPLIT_EVENT => Step::FloatSplit {
                        ends_in_dot: event & Self::N_INPUT_TOKEN_MASK != 0,
                    },
                    _ => unreachable!(),
                }
            })
        }
        pub fn token(&mut self, kind: SyntaxKind, n_tokens: u8) {
            let e = ((kind as u16 as u32) << Self::KIND_SHIFT)
                | ((n_tokens as u32) << Self::N_INPUT_TOKEN_SHIFT)
                | Self::EVENT_MASK;
            self.event.push(e)
        }
        pub fn float_split_hack(&mut self, ends_in_dot: bool) {
            let e = (Self::SPLIT_EVENT as u32) << Self::TAG_SHIFT
                | ((ends_in_dot as u32) << Self::N_INPUT_TOKEN_SHIFT)
                | Self::EVENT_MASK;
            self.event.push(e);
        }
        pub fn enter_node(&mut self, kind: SyntaxKind) {
            let e = ((kind as u16 as u32) << Self::KIND_SHIFT)
                | ((Self::ENTER_EVENT as u32) << Self::TAG_SHIFT)
                | Self::EVENT_MASK;
            self.event.push(e)
        }
        pub fn leave_node(&mut self) {
            let e = (Self::EXIT_EVENT as u32) << Self::TAG_SHIFT | Self::EVENT_MASK;
            self.event.push(e)
        }
        pub fn error(&mut self, error: String) {
            let idx = self.error.len();
            self.error.push(error);
            let e = (idx as u32) << Self::ERROR_SHIFT;
            self.event.push(e);
        }
    }
}
mod parser;
mod shortcuts;
mod syntax_kind {
    mod generated;
    #[allow(unreachable_pub)]
    pub use self::generated::{SyntaxKind, T};
    impl From<u16> for SyntaxKind {
        #[inline]
        fn from(d: u16) -> SyntaxKind {
            assert!(d <= (SyntaxKind::__LAST as u16));
            unsafe { std::mem::transmute::<u16, SyntaxKind>(d) }
        }
    }
    impl From<SyntaxKind> for u16 {
        #[inline]
        fn from(k: SyntaxKind) -> u16 {
            k as u16
        }
    }
    impl SyntaxKind {
        #[inline]
        pub fn is_trivia(self) -> bool {
            matches!(self, SyntaxKind::WHITESPACE | SyntaxKind::COMMENT)
        }
    }
}
mod token_set {
    use crate::SyntaxKind;
    #[derive(Clone, Copy)]
    pub struct TokenSet(u128);
    impl TokenSet {
        pub const EMPTY: TokenSet = TokenSet(0);
        pub const fn new(kinds: &[SyntaxKind]) -> TokenSet {
            let mut res = 0u128;
            let mut i = 0;
            while i < kinds.len() {
                res |= mask(kinds[i]);
                i += 1;
            }
            TokenSet(res)
        }
        pub const fn union(self, other: TokenSet) -> TokenSet {
            TokenSet(self.0 | other.0)
        }
        pub const fn contains(&self, kind: SyntaxKind) -> bool {
            self.0 & mask(kind) != 0
        }
    }
    const fn mask(kind: SyntaxKind) -> u128 {
        1u128 << (kind as usize)
    }
    #[test]
    fn token_set_works_for_tokens() {
        use crate::SyntaxKind::*;
        let ts = TokenSet::new(&[EOF, SHEBANG]);
        assert!(ts.contains(EOF));
        assert!(ts.contains(SHEBANG));
        assert!(!ts.contains(PLUS));
    }
}

#[cfg(test)]
mod tests;
