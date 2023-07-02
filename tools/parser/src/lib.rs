#![allow(dead_code)]
#![forbid(
    // missing_debug_implementations,
    unconditional_recursion,
    future_incompatible,
    // missing_docs,
)]
#![warn(unused_lifetimes)]

pub use crate::{
    input::Input,
    lexed::Lexed,
    output::{Output, Step},
    shortcuts::StrStep,
};
use std::mem;
pub use token_set::TokenSet;

#[derive(Debug)]
pub enum TopEntry {
    SourceFile,
    MacroStmts,
    MacroItems,
    Pattern,
    Type,
    Expr,
    MetaItem,
}
impl TopEntry {
    pub fn parse(&self, x: &Input) -> Output {
        let mut p = parser::Parser::new(x);
        let entry: fn(&'_ mut parser::Parser<'_>) = match self {
            TopEntry::SourceFile => grammar::top::src_file,
            TopEntry::MacroStmts => grammar::top::mac_stmts,
            TopEntry::MacroItems => grammar::top::mac_items,
            TopEntry::Pattern => grammar::top::pattern,
            TopEntry::Type => grammar::top::ty,
            TopEntry::Expr => grammar::top::expr,
            TopEntry::MetaItem => grammar::top::meta,
        };
        entry(&mut p);
        let y = p.finish();
        let y = process(y);
        if cfg!(debug_assertions) {
            let mut depth = 0;
            let mut first = true;
            for x in y.iter() {
                assert!(depth > 0 || first);
                first = false;
                use Step::*;
                match x {
                    Enter { .. } => depth += 1,
                    Exit => depth -= 1,
                    FloatSplit {
                        ends_in_dot: has_pseudo_dot,
                    } => depth -= 1 + !has_pseudo_dot as usize,
                    Token { .. } | Step::Error { .. } => (),
                }
            }
            assert!(!first, "no tree at all");
            assert_eq!(depth, 0, "unbalanced tree");
        }
        y
    }
}

#[derive(Debug)]
pub enum PreEntry {
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
impl PreEntry {
    pub fn parse(&self, x: &Input) -> Output {
        let mut p = parser::Parser::new(x);
        let entry: fn(&'_ mut parser::Parser<'_>) = match self {
            PreEntry::Vis => grammar::pre::vis,
            PreEntry::Block => grammar::pre::block,
            PreEntry::Stmt => grammar::pre::stmt,
            PreEntry::Pat => grammar::pre::pat,
            PreEntry::PatTop => grammar::pre::pat_top,
            PreEntry::Ty => grammar::pre::ty,
            PreEntry::Expr => grammar::pre::expr,
            PreEntry::Path => grammar::pre::path,
            PreEntry::Item => grammar::pre::item,
            PreEntry::MetaItem => grammar::pre::meta,
        };
        entry(&mut p);
        let y = p.finish();
        process(y)
    }
}

pub struct Reparser(fn(&mut parser::Parser<'_>));
impl Reparser {
    pub fn for_node(x: SyntaxKind, first_child: Option<SyntaxKind>, parent: Option<SyntaxKind>) -> Option<Reparser> {
        grammar::reparser(x, first_child, parent).map(Reparser)
    }
    pub fn parse(self, x: &Input) -> Output {
        let mut p = parser::Parser::new(x);
        let Reparser(r) = self;
        r(&mut p);
        let y = p.finish();
        process(y)
    }
}

#[derive(Debug)]
pub enum Event {
    Start { kind: SyntaxKind, fwd_parent: Option<u32> },
    Finish,
    Token { kind: SyntaxKind, n_raw_toks: u8 },
    FloatSplitHack { ends_in_dot: bool },
    Error { msg: String },
}
impl Event {
    pub fn tombstone() -> Self {
        Event::Start {
            kind: SyntaxKind::TOMBSTONE,
            fwd_parent: None,
        }
    }
}

pub fn process(mut xs: Vec<Event>) -> Output {
    let mut y = Output::default();
    let mut fwd_parents = Vec::new();
    for i in 0..xs.len() {
        use Event::*;
        match mem::replace(&mut xs[i], Event::tombstone()) {
            Start { kind, fwd_parent } => {
                fwd_parents.push(kind);
                let mut idx = i;
                let mut fp = fwd_parent;
                while let Some(fwd) = fp {
                    idx += fwd as usize;
                    fp = match mem::replace(&mut xs[idx], Event::tombstone()) {
                        Start { kind, fwd_parent } => {
                            fwd_parents.push(kind);
                            fwd_parent
                        },
                        _ => unreachable!(),
                    };
                }
                for kind in fwd_parents.drain(..).rev() {
                    if kind != SyntaxKind::TOMBSTONE {
                        y.enter_node(kind);
                    }
                }
            },
            Finish => y.leave_node(),
            Token { kind, n_raw_toks } => {
                y.token(kind, n_raw_toks);
            },
            FloatSplitHack { ends_in_dot } => {
                y.float_split_hack(ends_in_dot);
                let ev = mem::replace(&mut xs[i + 1], Event::tombstone());
                assert!(matches!(ev, Event::Finish), "{ev:?}");
            },
            Error { msg } => y.error(msg),
        }
    }
    y
}

mod grammar;
mod input {
    use crate::SyntaxKind;
    #[allow(non_camel_case_types)]
    type bits = u64;
    #[derive(Default)]
    pub struct Input {
        kinds: Vec<SyntaxKind>,
        joint: Vec<bits>,
        contextuals: Vec<SyntaxKind>,
    }
    impl Input {
        #[inline]
        pub fn push(&mut self, x: SyntaxKind) {
            self.push_impl(x, SyntaxKind::EOF)
        }
        #[inline]
        pub fn push_ident(&mut self, contextual: SyntaxKind) {
            self.push_impl(SyntaxKind::IDENT, contextual)
        }
        #[inline]
        pub fn was_joint(&mut self) {
            let n = self.len() - 1;
            let (idx, b_idx) = self.bit_index(n);
            self.joint[idx] |= 1 << b_idx;
        }
        #[inline]
        fn push_impl(&mut self, x: SyntaxKind, contextual: SyntaxKind) {
            let idx = self.len();
            if idx % (bits::BITS as usize) == 0 {
                self.joint.push(0);
            }
            self.kinds.push(x);
            self.contextuals.push(contextual);
        }
    }
    impl Input {
        pub fn kind(&self, i: usize) -> SyntaxKind {
            self.kinds.get(i).copied().unwrap_or(SyntaxKind::EOF)
        }
        pub fn contextual_kind(&self, i: usize) -> SyntaxKind {
            self.contextuals.get(i).copied().unwrap_or(SyntaxKind::EOF)
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
            self.kinds.len()
        }
    }
}
mod lexed;
mod lexer;
mod output {
    use crate::SyntaxKind;
    #[derive(Default)]
    pub struct Output {
        event: Vec<u32>,
        errs: Vec<String>,
    }
    #[derive(Debug)]
    pub enum Step<'a> {
        Token { kind: SyntaxKind, n_input_toks: u8 },
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
            self.event.iter().map(|&x| {
                if x & Self::EVENT_MASK == 0 {
                    return Step::Error {
                        msg: self.errs[(x as usize) >> Self::ERROR_SHIFT].as_str(),
                    };
                }
                let tag = ((x & Self::TAG_MASK) >> Self::TAG_SHIFT) as u8;
                match tag {
                    Self::TOKEN_EVENT => {
                        let kind: SyntaxKind = (((x & Self::KIND_MASK) >> Self::KIND_SHIFT) as u16).into();
                        let n_input_toks = ((x & Self::N_INPUT_TOKEN_MASK) >> Self::N_INPUT_TOKEN_SHIFT) as u8;
                        Step::Token { kind, n_input_toks }
                    },
                    Self::ENTER_EVENT => {
                        let kind: SyntaxKind = (((x & Self::KIND_MASK) >> Self::KIND_SHIFT) as u16).into();
                        Step::Enter { kind }
                    },
                    Self::EXIT_EVENT => Step::Exit,
                    Self::SPLIT_EVENT => Step::FloatSplit {
                        ends_in_dot: x & Self::N_INPUT_TOKEN_MASK != 0,
                    },
                    _ => unreachable!(),
                }
            })
        }
        pub fn token(&mut self, x: SyntaxKind, n: u8) {
            let y =
                ((x as u16 as u32) << Self::KIND_SHIFT) | ((n as u32) << Self::N_INPUT_TOKEN_SHIFT) | Self::EVENT_MASK;
            self.event.push(y)
        }
        pub fn float_split_hack(&mut self, ends_in_dot: bool) {
            let y = (Self::SPLIT_EVENT as u32) << Self::TAG_SHIFT
                | ((ends_in_dot as u32) << Self::N_INPUT_TOKEN_SHIFT)
                | Self::EVENT_MASK;
            self.event.push(y);
        }
        pub fn enter_node(&mut self, x: SyntaxKind) {
            let y = ((x as u16 as u32) << Self::KIND_SHIFT)
                | ((Self::ENTER_EVENT as u32) << Self::TAG_SHIFT)
                | Self::EVENT_MASK;
            self.event.push(y)
        }
        pub fn leave_node(&mut self) {
            let y = (Self::EXIT_EVENT as u32) << Self::TAG_SHIFT | Self::EVENT_MASK;
            self.event.push(y)
        }
        pub fn error(&mut self, err: String) {
            let i = self.errs.len();
            self.errs.push(err);
            let y = (i as u32) << Self::ERROR_SHIFT;
            self.event.push(y);
        }
    }
}
mod kind_gen;
mod parser;
mod shortcuts {
    use std::mem;

    use crate::{
        Lexed, Step,
        SyntaxKind::{self, *},
    };

    #[derive(Debug)]
    pub enum StrStep<'a> {
        Token { kind: SyntaxKind, text: &'a str },
        Enter { kind: SyntaxKind },
        Exit,
        Error { msg: &'a str, pos: usize },
    }

    impl<'a> Lexed<'a> {
        pub fn to_input(&self) -> crate::Input {
            let mut res = crate::Input::default();
            let mut was_joint = false;
            for i in 0..self.len() {
                let kind = self.kind(i);
                if kind.is_trivia() {
                    was_joint = false
                } else {
                    if kind == SyntaxKind::IDENT {
                        let token_text = self.text(i);
                        let contextual_kw =
                            SyntaxKind::from_contextual_keyword(token_text).unwrap_or(SyntaxKind::IDENT);
                        res.push_ident(contextual_kw);
                    } else {
                        if was_joint {
                            res.was_joint();
                        }
                        res.push(kind);
                        if kind == SyntaxKind::FLOAT_NUMBER && !self.text(i).ends_with('.') {
                            res.was_joint();
                        }
                    }

                    was_joint = true;
                }
            }
            res
        }

        pub fn intersperse_trivia(&self, output: &crate::Output, sink: &mut dyn FnMut(StrStep<'_>)) -> bool {
            let mut builder = Builder {
                lexed: self,
                pos: 0,
                state: State::PendingEnter,
                sink,
            };

            for event in output.iter() {
                match event {
                    Step::Token {
                        kind,
                        n_input_toks: n_raw_tokens,
                    } => builder.token(kind, n_raw_tokens),
                    Step::FloatSplit {
                        ends_in_dot: has_pseudo_dot,
                    } => builder.float_split(has_pseudo_dot),
                    Step::Enter { kind } => builder.enter(kind),
                    Step::Exit => builder.exit(),
                    Step::Error { msg } => {
                        let text_pos = builder.lexed.text_start(builder.pos);
                        (builder.sink)(StrStep::Error { msg, pos: text_pos });
                    },
                }
            }

            match mem::replace(&mut builder.state, State::Normal) {
                State::PendingExit => {
                    builder.eat_trivias();
                    (builder.sink)(StrStep::Exit);
                },
                State::PendingEnter | State::Normal => unreachable!(),
            }

            builder.pos == builder.lexed.len()
        }
    }

    struct Builder<'a, 'b> {
        lexed: &'a Lexed<'a>,
        pos: usize,
        state: State,
        sink: &'b mut dyn FnMut(StrStep<'_>),
    }

    enum State {
        PendingEnter,
        Normal,
        PendingExit,
    }

    impl Builder<'_, '_> {
        fn token(&mut self, kind: SyntaxKind, n_tokens: u8) {
            match mem::replace(&mut self.state, State::Normal) {
                State::PendingEnter => unreachable!(),
                State::PendingExit => (self.sink)(StrStep::Exit),
                State::Normal => (),
            }
            self.eat_trivias();
            self.do_token(kind, n_tokens as usize);
        }

        fn float_split(&mut self, has_pseudo_dot: bool) {
            match mem::replace(&mut self.state, State::Normal) {
                State::PendingEnter => unreachable!(),
                State::PendingExit => (self.sink)(StrStep::Exit),
                State::Normal => (),
            }
            self.eat_trivias();
            self.do_float_split(has_pseudo_dot);
        }

        fn enter(&mut self, kind: SyntaxKind) {
            match mem::replace(&mut self.state, State::Normal) {
                State::PendingEnter => {
                    (self.sink)(StrStep::Enter { kind });
                    return;
                },
                State::PendingExit => (self.sink)(StrStep::Exit),
                State::Normal => (),
            }

            let n_trivias = (self.pos..self.lexed.len())
                .take_while(|&x| self.lexed.kind(x).is_trivia())
                .count();
            let leading_trivias = self.pos..self.pos + n_trivias;
            let n_attached_trivias = n_attached_trivias(
                kind,
                leading_trivias.rev().map(|x| (self.lexed.kind(x), self.lexed.text(x))),
            );
            self.eat_n_trivias(n_trivias - n_attached_trivias);
            (self.sink)(StrStep::Enter { kind });
            self.eat_n_trivias(n_attached_trivias);
        }

        fn exit(&mut self) {
            match mem::replace(&mut self.state, State::PendingExit) {
                State::PendingEnter => unreachable!(),
                State::PendingExit => (self.sink)(StrStep::Exit),
                State::Normal => (),
            }
        }

        fn eat_trivias(&mut self) {
            while self.pos < self.lexed.len() {
                let kind = self.lexed.kind(self.pos);
                if !kind.is_trivia() {
                    break;
                }
                self.do_token(kind, 1);
            }
        }

        fn eat_n_trivias(&mut self, n: usize) {
            for _ in 0..n {
                let kind = self.lexed.kind(self.pos);
                assert!(kind.is_trivia());
                self.do_token(kind, 1);
            }
        }

        fn do_token(&mut self, kind: SyntaxKind, n_tokens: usize) {
            let text = &self.lexed.range_text(self.pos..self.pos + n_tokens);
            self.pos += n_tokens;
            (self.sink)(StrStep::Token { kind, text });
        }

        fn do_float_split(&mut self, has_pseudo_dot: bool) {
            let text = &self.lexed.range_text(self.pos..self.pos + 1);
            self.pos += 1;
            match text.split_once('.') {
                Some((left, right)) => {
                    assert!(!left.is_empty());
                    (self.sink)(StrStep::Enter {
                        kind: SyntaxKind::NAME_REF,
                    });
                    (self.sink)(StrStep::Token {
                        kind: SyntaxKind::INT_NUMBER,
                        text: left,
                    });
                    (self.sink)(StrStep::Exit);

                    (self.sink)(StrStep::Exit);

                    (self.sink)(StrStep::Token {
                        kind: SyntaxKind::DOT,
                        text: ".",
                    });

                    if has_pseudo_dot {
                        assert!(right.is_empty(), "{left}.{right}");
                        self.state = State::Normal;
                    } else {
                        (self.sink)(StrStep::Enter {
                            kind: SyntaxKind::NAME_REF,
                        });
                        (self.sink)(StrStep::Token {
                            kind: SyntaxKind::INT_NUMBER,
                            text: right,
                        });
                        (self.sink)(StrStep::Exit);

                        self.state = State::PendingExit;
                    }
                },
                None => unreachable!(),
            }
        }
    }

    fn n_attached_trivias<'a>(kind: SyntaxKind, trivias: impl Iterator<Item = (SyntaxKind, &'a str)>) -> usize {
        match kind {
            CONST | ENUM | FN | IMPL | MACRO_CALL | MACRO_DEF | MACRO_RULES | MODULE | RECORD_FIELD | STATIC
            | STRUCT | TRAIT | TUPLE_FIELD | TYPE_ALIAS | UNION | USE | VARIANT => {
                let mut res = 0;
                let mut trivias = trivias.enumerate().peekable();

                while let Some((i, (kind, text))) = trivias.next() {
                    match kind {
                        WHITESPACE if text.contains("\n\n") => {
                            if let Some((COMMENT, peek_text)) = trivias.peek().map(|(_, pair)| pair) {
                                if is_outer(peek_text) {
                                    continue;
                                }
                            }
                            break;
                        },
                        COMMENT => {
                            if is_inner(text) {
                                break;
                            }
                            res = i + 1;
                        },
                        _ => (),
                    }
                }
                res
            },
            _ => 0,
        }
    }

    fn is_outer(text: &str) -> bool {
        if text.starts_with("////") || text.starts_with("/***") {
            return false;
        }
        text.starts_with("///") || text.starts_with("/**")
    }

    fn is_inner(text: &str) -> bool {
        text.starts_with("//!") || text.starts_with("/*!")
    }
}

pub use kind_gen::SyntaxKind;
impl From<u16> for SyntaxKind {
    #[inline]
    fn from(x: u16) -> SyntaxKind {
        assert!(x <= (SyntaxKind::__LAST as u16));
        unsafe { std::mem::transmute::<u16, SyntaxKind>(x) }
    }
}
impl From<SyntaxKind> for u16 {
    #[inline]
    fn from(x: SyntaxKind) -> u16 {
        x as u16
    }
}
impl SyntaxKind {
    #[inline]
    pub fn is_trivia(self) -> bool {
        matches!(self, SyntaxKind::WHITESPACE | SyntaxKind::COMMENT)
    }
}

mod token_set {
    use crate::SyntaxKind;
    #[derive(Clone, Copy)]
    pub struct TokenSet(u128);
    impl TokenSet {
        pub const EMPTY: TokenSet = TokenSet(0);
        pub const fn new(xs: &[SyntaxKind]) -> TokenSet {
            let mut y = 0u128;
            let mut i = 0;
            while i < xs.len() {
                y |= mask(xs[i]);
                i += 1;
            }
            TokenSet(y)
        }
        pub const fn union(self, other: TokenSet) -> TokenSet {
            TokenSet(self.0 | other.0)
        }
        pub const fn contains(&self, x: SyntaxKind) -> bool {
            self.0 & mask(x) != 0
        }
    }
    const fn mask(x: SyntaxKind) -> u128 {
        1u128 << (x as usize)
    }
    #[test]
    fn token_set_works_for_tokens() {
        use crate::SyntaxKind::*;
        let y = TokenSet::new(&[EOF, SHEBANG]);
        assert!(y.contains(EOF));
        assert!(y.contains(SHEBANG));
        assert!(!y.contains(PLUS));
    }
}
mod limit {
    #[cfg(feature = "tracking")]
    use std::sync::atomic::AtomicUsize;
    #[derive(Debug)]
    pub struct Limit {
        upper_bound: usize,
        #[cfg(feature = "tracking")]
        max: AtomicUsize,
    }
    impl Limit {
        #[inline]
        pub const fn new(upper_bound: usize) -> Self {
            Self {
                upper_bound,
                #[cfg(feature = "tracking")]
                max: AtomicUsize::new(0),
            }
        }
        #[inline]
        #[cfg(feature = "tracking")]
        pub const fn new_tracking(upper_bound: usize) -> Self {
            Self {
                upper_bound,
                #[cfg(feature = "tracking")]
                max: AtomicUsize::new(1),
            }
        }
        #[inline]
        pub const fn _inner(&self) -> usize {
            self.upper_bound
        }
        #[inline]
        pub fn check(&self, other: usize) -> Result<(), ()> {
            if other > self.upper_bound {
                Err(())
            } else {
                #[cfg(feature = "tracking")]
                loop {
                    use std::sync::atomic::Ordering;
                    let old_max = self.max.load(Ordering::Relaxed);
                    if other <= old_max || old_max == 0 {
                        break;
                    }
                    if self
                        .max
                        .compare_exchange_weak(old_max, other, Ordering::Relaxed, Ordering::Relaxed)
                        .is_ok()
                    {
                        eprintln!("new max: {other}");
                    }
                }

                Ok(())
            }
        }
    }
}
mod srcgen {
    use std::{
        fmt, fs, mem,
        path::{Path, PathBuf},
    };
    use xshell::{cmd, Shell};

    #[derive(Clone)]
    pub struct CommentBlock {
        pub id: String,
        pub line: usize,
        pub texts: Vec<String>,
        is_doc: bool,
    }
    impl CommentBlock {
        pub fn extract(tag: &str, x: &str) -> Vec<CommentBlock> {
            assert!(tag.starts_with(char::is_uppercase));
            let tag = format!("{tag}:");
            let mut ys = CommentBlock::extract_untagged(x);
            ys.retain_mut(|x| {
                let first = x.texts.remove(0);
                let Some(id) = first.strip_prefix(&tag) else {
                    return false;
                };
                if x.is_doc {
                    panic!("Use plain (non-doc) comments with tags like {tag}:\n    {first}");
                }
                x.id = id.trim().to_string();
                true
            });
            ys
        }
        pub fn extract_untagged(x: &str) -> Vec<CommentBlock> {
            let mut ys = Vec::new();
            let xs = x.lines().map(str::trim_start);
            let dummy = CommentBlock {
                id: String::new(),
                line: 0,
                texts: Vec::new(),
                is_doc: false,
            };
            let mut y = dummy.clone();
            for (i, x) in xs.enumerate() {
                match x.strip_prefix("//") {
                    Some(mut x) => {
                        if let Some('/' | '!') = x.chars().next() {
                            x = &x[1..];
                            y.is_doc = true;
                        }
                        if let Some(' ') = x.chars().next() {
                            x = &x[1..];
                        }
                        y.texts.push(x.to_string());
                    },
                    None => {
                        if !y.texts.is_empty() {
                            let y = mem::replace(&mut y, dummy.clone());
                            ys.push(y);
                        }
                        y.line = i + 2;
                    },
                }
            }
            if !y.texts.is_empty() {
                ys.push(y);
            }
            ys
        }
    }

    #[derive(Debug)]
    pub struct Location {
        pub file: PathBuf,
        pub line: usize,
    }
    impl fmt::Display for Location {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let p = self.file.strip_prefix(project_root()).unwrap().display().to_string();
            let p = p.replace('\\', "/");
            let n = self.file.file_name().unwrap();
            write!(
                f,
                "https://github.com/rust-lang/rust-analyzer/blob/master/{}#L{}[{}]",
                p,
                self.line,
                n.to_str().unwrap()
            )
        }
    }

    pub fn project_root() -> PathBuf {
        let y = env!("CARGO_MANIFEST_DIR");
        let y = PathBuf::from(y).parent().unwrap().parent().unwrap().to_owned();
        assert!(y.join("triagebot.toml").exists());
        y
    }

    pub fn list_rust_files(x: &Path) -> Vec<PathBuf> {
        let mut ys = list_files(x);
        ys.retain(|x| {
            x.file_name()
                .unwrap_or_default()
                .to_str()
                .unwrap_or_default()
                .ends_with(".rs")
        });
        ys
    }
    fn list_files(x: &Path) -> Vec<PathBuf> {
        let mut ys = Vec::new();
        let mut xs = vec![x.to_path_buf()];
        while let Some(x) = xs.pop() {
            for e in x.read_dir().unwrap() {
                let e = e.unwrap();
                let p = e.path();
                let is_hidden = p
                    .file_name()
                    .unwrap_or_default()
                    .to_str()
                    .unwrap_or_default()
                    .starts_with('.');
                if !is_hidden {
                    let t = e.file_type().unwrap();
                    if t.is_dir() {
                        xs.push(p);
                    } else if t.is_file() {
                        ys.push(p);
                    }
                }
            }
        }
        ys
    }

    pub fn reformat(x: String) -> String {
        let sh = Shell::new().unwrap();
        ensure_rustfmt(&sh);
        let rustfmt_toml = project_root().join("rustfmt.toml");
        let mut y = cmd!(
            sh,
            "rustup run stable rustfmt --config-path {rustfmt_toml} --config fn_single_line=true"
        )
        .stdin(x)
        .read()
        .unwrap();
        if !y.ends_with('\n') {
            y.push('\n');
        }
        y
    }
    fn ensure_rustfmt(x: &Shell) {
        let y = cmd!(x, "rustup run stable rustfmt --version")
            .read()
            .unwrap_or_default();
        if !y.contains("stable") {
            panic!("Failed to run rustfmt from toolchain 'stable'.",);
        }
    }

    pub fn add_preamble(gen: &'static str, mut y: String) -> String {
        let x = format!("//! Generated by `{gen}`, do not edit by hand.\n\n");
        y.insert_str(0, &x);
        y
    }

    pub fn ensure_file_contents(file: &Path, x: &str) {
        if let Ok(y) = fs::read_to_string(file) {
            if normalize_newlines(&y) == normalize_newlines(x) {
                return;
            }
        }
        let display_path = file.strip_prefix(project_root()).unwrap_or(file);
        eprintln!(
            "\n\x1b[31;1merror\x1b[0m: {} was not up-to-date, updating\n",
            display_path.display()
        );
        if std::env::var("CI").is_ok() {
            eprintln!("    NOTE: run `cargo test` locally and commit the updated files\n");
        }
        if let Some(parent) = file.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(file, x).unwrap();
        panic!("some file was not up to date and has been updated, simply re-run the tests");
    }

    fn normalize_newlines(x: &str) -> String {
        x.replace("\r\n", "\n")
    }
}
pub mod syntax;

#[cfg(test)]
mod test;
