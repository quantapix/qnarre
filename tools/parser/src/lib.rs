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
    kind::SyntaxKind,
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
mod parser;
mod shortcuts;
mod kind {
    mod generated;
    #[allow(unreachable_pub)]
    pub use self::generated::{SyntaxKind, T};
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

    pub fn list_rust_files(dir: &Path) -> Vec<PathBuf> {
        let mut res = list_files(dir);
        res.retain(|it| {
            it.file_name()
                .unwrap_or_default()
                .to_str()
                .unwrap_or_default()
                .ends_with(".rs")
        });
        res
    }
    pub fn list_files(dir: &Path) -> Vec<PathBuf> {
        let mut res = Vec::new();
        let mut work = vec![dir.to_path_buf()];
        while let Some(dir) = work.pop() {
            for entry in dir.read_dir().unwrap() {
                let entry = entry.unwrap();
                let file_type = entry.file_type().unwrap();
                let path = entry.path();
                let is_hidden = path
                    .file_name()
                    .unwrap_or_default()
                    .to_str()
                    .unwrap_or_default()
                    .starts_with('.');
                if !is_hidden {
                    if file_type.is_dir() {
                        work.push(path);
                    } else if file_type.is_file() {
                        res.push(path);
                    }
                }
            }
        }
        res
    }

    #[derive(Clone)]
    pub struct CommentBlock {
        pub id: String,
        pub line: usize,
        pub contents: Vec<String>,
        is_doc: bool,
    }
    impl CommentBlock {
        pub fn extract(tag: &str, text: &str) -> Vec<CommentBlock> {
            assert!(tag.starts_with(char::is_uppercase));
            let tag = format!("{tag}:");
            let mut blocks = CommentBlock::extract_untagged(text);
            blocks.retain_mut(|block| {
                let first = block.contents.remove(0);
                let Some(id) = first.strip_prefix(&tag) else { return false; };
                if block.is_doc {
                    panic!("Use plain (non-doc) comments with tags like {tag}:\n    {first}");
                }
                block.id = id.trim().to_string();
                true
            });
            blocks
        }
        pub fn extract_untagged(text: &str) -> Vec<CommentBlock> {
            let mut res = Vec::new();
            let lines = text.lines().map(str::trim_start);
            let dummy_block = CommentBlock {
                id: String::new(),
                line: 0,
                contents: Vec::new(),
                is_doc: false,
            };
            let mut block = dummy_block.clone();
            for (line_num, line) in lines.enumerate() {
                match line.strip_prefix("//") {
                    Some(mut contents) => {
                        if let Some('/' | '!') = contents.chars().next() {
                            contents = &contents[1..];
                            block.is_doc = true;
                        }
                        if let Some(' ') = contents.chars().next() {
                            contents = &contents[1..];
                        }
                        block.contents.push(contents.to_string());
                    },
                    None => {
                        if !block.contents.is_empty() {
                            let block = mem::replace(&mut block, dummy_block.clone());
                            res.push(block);
                        }
                        block.line = line_num + 2;
                    },
                }
            }
            if !block.contents.is_empty() {
                res.push(block);
            }
            res
        }
    }

    #[derive(Debug)]
    pub struct Location {
        pub file: PathBuf,
        pub line: usize,
    }
    impl fmt::Display for Location {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let path = self.file.strip_prefix(project_root()).unwrap().display().to_string();
            let path = path.replace('\\', "/");
            let name = self.file.file_name().unwrap();
            write!(
                f,
                "https://github.com/rust-lang/rust-analyzer/blob/master/{}#L{}[{}]",
                path,
                self.line,
                name.to_str().unwrap()
            )
        }
    }

    fn ensure_rustfmt(sh: &Shell) {
        let version = cmd!(sh, "rustup run stable rustfmt --version")
            .read()
            .unwrap_or_default();
        if !version.contains("stable") {
            panic!(
                "Failed to run rustfmt from toolchain 'stable'. \
                 Please run `rustup component add rustfmt --toolchain stable` to install it.",
            );
        }
    }
    pub fn reformat(text: String) -> String {
        let sh = Shell::new().unwrap();
        ensure_rustfmt(&sh);
        let rustfmt_toml = project_root().join("rustfmt.toml");
        let mut stdout = cmd!(
            sh,
            "rustup run stable rustfmt --config-path {rustfmt_toml} --config fn_single_line=true"
        )
        .stdin(text)
        .read()
        .unwrap();
        if !stdout.ends_with('\n') {
            stdout.push('\n');
        }
        stdout
    }
    pub fn add_preamble(generator: &'static str, mut text: String) -> String {
        let preamble = format!("//! Generated by `{generator}`, do not edit by hand.\n\n");
        text.insert_str(0, &preamble);
        text
    }
    pub fn ensure_file_contents(file: &Path, contents: &str) {
        if let Ok(old_contents) = fs::read_to_string(file) {
            if normalize_newlines(&old_contents) == normalize_newlines(contents) {
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
        fs::write(file, contents).unwrap();
        panic!("some file was not up to date and has been updated, simply re-run the tests");
    }

    fn normalize_newlines(s: &str) -> String {
        s.replace("\r\n", "\n")
    }

    pub fn project_root() -> PathBuf {
        let dir = env!("CARGO_MANIFEST_DIR");
        let res = PathBuf::from(dir).parent().unwrap().parent().unwrap().to_owned();
        assert!(res.join("triagebot.toml").exists());
        res
    }
}
pub mod syntax;

#[cfg(test)]
mod tests;
