use super::*;
use std::{
    error::Error,
    iter::FromIterator,
    marker::PhantomData,
    ops::RangeBounds,
    panic::{RefUnwindSafe, UnwindSafe},
    rc::Rc,
    str::FromStr,
};

pub type Marker = PhantomData<AutoTraits>;
pub use self::value::*;
mod value {
    pub use std::marker::PhantomData as Marker;
}
pub struct AutoTraits(Rc<()>);
impl UnwindSafe for AutoTraits {}
impl RefUnwindSafe for AutoTraits {}

mod detection {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Once,
    };
    static FLAG: AtomicUsize = AtomicUsize::new(0);
    static INIT: Once = Once::new();
    pub fn inside_proc_macro() -> bool {
        match FLAG.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {},
        }
        INIT.call_once(init);
        inside_proc_macro()
    }
    pub fn force_fallback() {
        FLAG.store(1, Ordering::Relaxed);
    }
    pub fn unforce_fallback() {
        init();
    }
    fn init() {
        let y = proc_macro::is_available();
        FLAG.store(y as usize + 1, Ordering::Relaxed);
    }
}
use detection::inside_proc_macro;

fn mismatch() -> ! {
    panic!("compiler/fallback mismatch")
}

#[derive(Clone)]
pub struct DeferredTokenStream {
    stream: proc_macro::TokenStream,
    extra: Vec<proc_macro::TokenTree>,
}
impl DeferredTokenStream {
    fn new(stream: proc_macro::TokenStream) -> Self {
        DeferredTokenStream {
            stream,
            extra: Vec::new(),
        }
    }
    fn is_empty(&self) -> bool {
        self.stream.is_empty() && self.extra.is_empty()
    }
    fn evaluate_now(&mut self) {
        if !self.extra.is_empty() {
            self.stream.extend(self.extra.drain(..));
        }
    }
    fn into_token_stream(mut self) -> proc_macro::TokenStream {
        self.evaluate_now();
        self.stream
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Delim {
    Paren,
    Brace,
    Bracket,
    None,
}

#[derive(Clone)]
pub enum Group {
    Compiler(proc_macro::Group),
    Fallback(fallback::Group),
}
impl Group {
    pub fn new(x: Delim, s: TokenStream) -> Self {
        match s {
            TokenStream::Compiler(y) => {
                let x = match x {
                    Delim::Paren => proc_macro::Delimiter::Parenthesis,
                    Delim::Bracket => proc_macro::Delimiter::Bracket,
                    Delim::Brace => proc_macro::Delimiter::Brace,
                    Delim::None => proc_macro::Delimiter::None,
                };
                Group::Compiler(proc_macro::Group::new(x, y.into_token_stream()))
            },
            TokenStream::Fallback(y) => Group::Fallback(fallback::Group::new(x, y)),
        }
    }
    pub fn delim(&self) -> Delim {
        match self {
            Group::Compiler(x) => match x.delimiter() {
                proc_macro::Delimiter::Parenthesis => Delim::Paren,
                proc_macro::Delimiter::Bracket => Delim::Bracket,
                proc_macro::Delimiter::Brace => Delim::Brace,
                proc_macro::Delimiter::None => Delim::None,
            },
            Group::Fallback(x) => x.delim(),
        }
    }
    pub fn stream(&self) -> TokenStream {
        match self {
            Group::Compiler(x) => TokenStream::Compiler(DeferredTokenStream::new(x.stream())),
            Group::Fallback(x) => TokenStream::Fallback(x.stream()),
        }
    }
    pub fn span(&self) -> Span {
        match self {
            Group::Compiler(x) => Span::Compiler(x.span()),
            Group::Fallback(x) => Span::Fallback(x.span()),
        }
    }
    pub fn set_span(&mut self, s: Span) {
        match (self, s) {
            (Group::Compiler(x), Span::Compiler(s)) => x.set_span(s),
            (Group::Fallback(x), Span::Fallback(s)) => x.set_span(s),
            _ => mismatch(),
        }
    }
    pub fn span_open(&self) -> Span {
        match self {
            Group::Compiler(x) => Span::Compiler(x.span_open()),
            Group::Fallback(x) => Span::Fallback(x.span_open()),
        }
    }
    pub fn span_close(&self) -> Span {
        match self {
            Group::Compiler(x) => Span::Compiler(x.span_close()),
            Group::Fallback(x) => Span::Fallback(x.span_close()),
        }
    }
    pub fn delim_span(&self) -> DelimSpan {
        DelimSpan::new(&self)
    }
    fn unwrap_nightly(self) -> proc_macro::Group {
        match self {
            Group::Compiler(x) => x,
            Group::Fallback(_) => mismatch(),
        }
    }
}
impl From<fallback::Group> for Group {
    fn from(x: fallback::Group) -> Self {
        Group::Fallback(x)
    }
}
impl Display for Group {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Group::Compiler(x) => Display::fmt(x, f),
            Group::Fallback(x) => Display::fmt(x, f),
        }
    }
}
impl Debug for Group {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Group::Compiler(x) => Debug::fmt(x, f),
            Group::Fallback(x) => Debug::fmt(x, f),
        }
    }
}

#[derive(Clone)]
pub enum Lit {
    Compiler(proc_macro::Literal),
    Fallback(fallback::Lit),
}
macro_rules! suffixed_nums {
    ($($n:ident => $kind:ident,)*) => ($(
        pub fn $n(n: $kind) -> Lit {
            if inside_proc_macro() {
                Lit::Compiler(proc_macro::Literal::$n(n))
            } else {
                Lit::Fallback(fallback::Lit::$n(n))
            }
        }
    )*)
}
macro_rules! unsuffixed_nums {
    ($($n:ident => $kind:ident,)*) => ($(
        pub fn $n(n: $kind) -> Lit {
            if inside_proc_macro() {
                Lit::Compiler(proc_macro::Literal::$n(n))
            } else {
                Lit::Fallback(fallback::Lit::$n(n))
            }
        }
    )*)
}
impl Lit {
    pub unsafe fn from_str_unchecked(x: &str) -> Self {
        if inside_proc_macro() {
            Lit::Compiler(compiler_lit_from_str(x).expect("invalid literal"))
        } else {
            Lit::Fallback(fallback::Lit::from_str_unchecked(x))
        }
    }
    suffixed_nums! {
        f32_suffixed => f32,
        f64_suffixed => f64,
        i128_suffixed => i128,
        i16_suffixed => i16,
        i32_suffixed => i32,
        i64_suffixed => i64,
        i8_suffixed => i8,
        isize_suffixed => isize,
        u128_suffixed => u128,
        u16_suffixed => u16,
        u32_suffixed => u32,
        u64_suffixed => u64,
        u8_suffixed => u8,
        usize_suffixed => usize,
    }
    unsuffixed_nums! {
        i128_unsuffixed => i128,
        i16_unsuffixed => i16,
        i32_unsuffixed => i32,
        i64_unsuffixed => i64,
        i8_unsuffixed => i8,
        isize_unsuffixed => isize,
        u128_unsuffixed => u128,
        u16_unsuffixed => u16,
        u32_unsuffixed => u32,
        u64_unsuffixed => u64,
        u8_unsuffixed => u8,
        usize_unsuffixed => usize,
    }
    pub fn f32_suffixed(x: f32) -> Lit {
        assert!(x.is_finite());
        Lit::Fallback(fallback::Lit::f32_suffixed(x))
    }
    pub fn f64_suffixed(x: f64) -> Lit {
        assert!(x.is_finite());
        Lit::Fallback(fallback::Lit::f64_suffixed(x))
    }
    pub fn f32_unsuffixed(x: f32) -> Lit {
        assert!(x.is_finite());
        if inside_proc_macro() {
            Lit::Compiler(proc_macro::Literal::f32_unsuffixed(x))
        } else {
            Lit::Fallback(fallback::Lit::f32_unsuffixed(x))
        }
    }
    pub fn f64_unsuffixed(x: f64) -> Lit {
        assert!(x.is_finite());
        if inside_proc_macro() {
            Lit::Compiler(proc_macro::Literal::f64_unsuffixed(x))
        } else {
            Lit::Fallback(fallback::Lit::f64_unsuffixed(x))
        }
    }
    pub fn string(x: &str) -> Lit {
        if inside_proc_macro() {
            Lit::Compiler(proc_macro::Literal::string(x))
        } else {
            Lit::Fallback(fallback::Lit::string(x))
        }
    }
    pub fn character(x: char) -> Lit {
        if inside_proc_macro() {
            Lit::Compiler(proc_macro::Literal::character(x))
        } else {
            Lit::Fallback(fallback::Lit::character(x))
        }
    }
    pub fn byte_string(xs: &[u8]) -> Lit {
        if inside_proc_macro() {
            Lit::Compiler(proc_macro::Literal::byte_string(xs))
        } else {
            Lit::Fallback(fallback::Lit::byte_string(xs))
        }
    }
    pub fn span(&self) -> Span {
        match self {
            Lit::Compiler(x) => Span::Compiler(x.span()),
            Lit::Fallback(x) => Span::Fallback(x.span()),
        }
    }
    pub fn set_span(&mut self, x: Span) {
        match (self, x) {
            (Lit::Compiler(x), Span::Compiler(s)) => x.set_span(s),
            (Lit::Fallback(x), Span::Fallback(s)) => x.set_span(s),
            _ => mismatch(),
        }
    }
    pub fn subspan<R: RangeBounds<usize>>(&self, range: R) -> Option<Span> {
        match self {
            Lit::Compiler(x) => x.subspan(range).map(Span::Compiler),
            Lit::Fallback(x) => x.subspan(range).map(Span::Fallback),
        }
    }
    fn unwrap_nightly(self) -> proc_macro::Literal {
        match self {
            Lit::Compiler(x) => x,
            Lit::Fallback(_) => mismatch(),
        }
    }
}
impl From<fallback::Lit> for Lit {
    fn from(x: fallback::Lit) -> Self {
        Lit::Fallback(x)
    }
}
impl FromStr for Lit {
    type Err = LexError;
    fn from_str(x: &str) -> Result<Self, Self::Err> {
        if inside_proc_macro() {
            compiler_lit_from_str(x).map(Lit::Compiler)
        } else {
            let y = fallback::Lit::from_str(x)?;
            Ok(Lit::Fallback(y))
        }
    }
}
impl Display for Lit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Lit::Compiler(x) => Display::fmt(x, f),
            Lit::Fallback(x) => Display::fmt(x, f),
        }
    }
}
impl Debug for Lit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Lit::Compiler(x) => Debug::fmt(x, f),
            Lit::Fallback(x) => Debug::fmt(x, f),
        }
    }
}
fn compiler_lit_from_str(x: &str) -> Result<proc_macro::Literal, LexError> {
    proc_macro::Literal::from_str(x).map_err(LexError::Compiler)
}

#[derive(Copy, Clone)]
pub enum Span {
    Compiler(proc_macro::Span),
    Fallback(fallback::Span),
}
impl Span {
    pub fn call_site() -> Self {
        if inside_proc_macro() {
            Span::Compiler(proc_macro::Span::call_site())
        } else {
            Span::Fallback(fallback::Span::call_site())
        }
    }
    pub fn mixed_site() -> Self {
        if inside_proc_macro() {
            Span::Compiler(proc_macro::Span::mixed_site())
        } else {
            Span::Fallback(fallback::Span::mixed_site())
        }
    }
    pub fn resolved_at(&self, x: Span) -> Span {
        match (self, x) {
            (Span::Compiler(a), Span::Compiler(b)) => Span::Compiler(a.resolved_at(b)),
            (Span::Fallback(a), Span::Fallback(b)) => Span::Fallback(a.resolved_at(b)),
            _ => mismatch(),
        }
    }
    pub fn located_at(&self, x: Span) -> Span {
        match (self, x) {
            (Span::Compiler(a), Span::Compiler(b)) => Span::Compiler(a.located_at(b)),
            (Span::Fallback(a), Span::Fallback(b)) => Span::Fallback(a.located_at(b)),
            _ => mismatch(),
        }
    }
    pub fn start(&self) -> LineColumn {
        match self {
            Span::Compiler(_) => LineColumn { line: 0, column: 0 },
            Span::Fallback(x) => x.start(),
        }
    }
    pub fn end(&self) -> LineColumn {
        match self {
            Span::Compiler(_) => LineColumn { line: 0, column: 0 },
            Span::Fallback(x) => x.end(),
        }
    }
    pub fn join(&self, x: Span) -> Option<Span> {
        let y = match (self, x) {
            (Span::Compiler(a), Span::Compiler(b)) => Span::Compiler(a.join(b)?),
            (Span::Fallback(a), Span::Fallback(b)) => Span::Fallback(a.join(b)?),
            _ => return None,
        };
        Some(y)
    }
    pub fn source_text(&self) -> Option<String> {
        match self {
            Span::Compiler(x) => x.source_text(),
            Span::Fallback(x) => x.source_text(),
        }
    }
    pub fn unstable(self) -> proc_macro::Span {
        self.unwrap()
    }
    pub fn unwrap(self) -> proc_macro::Span {
        match self {
            Span::Compiler(x) => x,
            Span::Fallback(_) => mismatch(),
        }
    }
    fn unwrap_nightly(self) -> proc_macro::Span {
        match self {
            Span::Compiler(x) => x,
            Span::Fallback(_) => mismatch(),
        }
    }
}
impl From<proc_macro::Span> for Span {
    fn from(x: proc_macro::Span) -> Self {
        Span::Compiler(x)
    }
}
impl From<fallback::Span> for Span {
    fn from(x: fallback::Span) -> Self {
        Span::Fallback(x)
    }
}
impl Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Span::Compiler(x) => Debug::fmt(x, f),
            Span::Fallback(x) => Debug::fmt(x, f),
        }
    }
}
pub fn debug_span_field(x: &mut fmt::DebugStruct, s: Span) {
    match s {
        Span::Compiler(s) => {
            x.field("span", &s);
        },
        Span::Fallback(s) => fallback::debug_span_field(x, s),
    }
}

mod parse {
    use super::fallback::{
        is_ident_continue, is_ident_start, Group, LexError, Lit, Span, TokenStream, TokenStreamBuilder,
    };
    use super::{Delim, Punct, Spacing, Tree};
    use std::{
        char,
        str::{Bytes, CharIndices, Chars},
    };
    #[derive(Copy, Clone, Eq, PartialEq)]
    pub struct Cursor<'a> {
        pub rest: &'a str,
        pub off: u32,
    }
    impl<'a> Cursor<'a> {
        pub fn advance(&self, bytes: usize) -> Cursor<'a> {
            let (_front, rest) = self.rest.split_at(bytes);
            Cursor {
                rest,
                off: self.off + _front.chars().count() as u32,
            }
        }
        pub fn starts_with(&self, s: &str) -> bool {
            self.rest.starts_with(s)
        }
        pub fn starts_with_char(&self, ch: char) -> bool {
            self.rest.starts_with(ch)
        }
        pub fn starts_with_fn<Pattern>(&self, f: Pattern) -> bool
        where
            Pattern: FnMut(char) -> bool,
        {
            self.rest.starts_with(f)
        }
        pub fn is_empty(&self) -> bool {
            self.rest.is_empty()
        }
        fn len(&self) -> usize {
            self.rest.len()
        }
        fn as_bytes(&self) -> &'a [u8] {
            self.rest.as_bytes()
        }
        fn bytes(&self) -> Bytes<'a> {
            self.rest.bytes()
        }
        fn chars(&self) -> Chars<'a> {
            self.rest.chars()
        }
        fn char_indices(&self) -> CharIndices<'a> {
            self.rest.char_indices()
        }
        fn parse(&self, tag: &str) -> Result<Cursor<'a>, Reject> {
            if self.starts_with(tag) {
                Ok(self.advance(tag.len()))
            } else {
                Err(Reject)
            }
        }
    }
    pub struct Reject;
    type PResult<'a, O> = Result<(Cursor<'a>, O), Reject>;
    fn skip_whitespace(input: Cursor) -> Cursor {
        let mut s = input;
        while !s.is_empty() {
            let byte = s.as_bytes()[0];
            if byte == b'/' {
                if s.starts_with("//") && (!s.starts_with("///") || s.starts_with("////")) && !s.starts_with("//!") {
                    let (cursor, _) = take_until_newline_or_eof(s);
                    s = cursor;
                    continue;
                } else if s.starts_with("/**/") {
                    s = s.advance(4);
                    continue;
                } else if s.starts_with("/*")
                    && (!s.starts_with("/**") || s.starts_with("/***"))
                    && !s.starts_with("/*!")
                {
                    match block_comment(s) {
                        Ok((rest, _)) => {
                            s = rest;
                            continue;
                        },
                        Err(Reject) => return s,
                    }
                }
            }
            match byte {
                b' ' | 0x09..=0x0d => {
                    s = s.advance(1);
                    continue;
                },
                b if b.is_ascii() => {},
                _ => {
                    let ch = s.chars().next().unwrap();
                    if is_whitespace(ch) {
                        s = s.advance(ch.len_utf8());
                        continue;
                    }
                },
            }
            return s;
        }
        s
    }
    fn block_comment(input: Cursor) -> PResult<&str> {
        if !input.starts_with("/*") {
            return Err(Reject);
        }
        let mut depth = 0usize;
        let bytes = input.as_bytes();
        let mut i = 0usize;
        let upper = bytes.len() - 1;
        while i < upper {
            if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                depth += 1;
                i += 1; // eat '*'
            } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                depth -= 1;
                if depth == 0 {
                    return Ok((input.advance(i + 2), &input.rest[..i + 2]));
                }
                i += 1; // eat '/'
            }
            i += 1;
        }
        Err(Reject)
    }
    fn is_whitespace(ch: char) -> bool {
        ch.is_whitespace() || ch == '\u{200e}' || ch == '\u{200f}'
    }
    fn word_break(input: Cursor) -> Result<Cursor, Reject> {
        match input.chars().next() {
            Some(ch) if is_ident_continue(ch) => Err(Reject),
            Some(_) | None => Ok(input),
        }
    }
    pub fn token_stream(mut input: Cursor) -> Result<TokenStream, LexError> {
        let mut trees = TokenStreamBuilder::new();
        let mut stack = Vec::new();
        loop {
            input = skip_whitespace(input);
            if let Ok((rest, ())) = doc_comment(input, &mut trees) {
                input = rest;
                continue;
            }
            let lo = input.off;
            let first = match input.bytes().next() {
                Some(first) => first,
                None => match stack.last() {
                    None => return Ok(trees.build()),
                    Some((lo, _frame)) => {
                        return Err(LexError {
                            span: Span { lo: *lo, hi: *lo },
                        })
                    },
                },
            };
            if let Some(open_delimiter) = match first {
                b'(' => Some(Delim::Paren),
                b'[' => Some(Delim::Bracket),
                b'{' => Some(Delim::Brace),
                _ => None,
            } {
                input = input.advance(1);
                let frame = (open_delimiter, trees);
                let frame = (lo, frame);
                stack.push(frame);
                trees = TokenStreamBuilder::new();
            } else if let Some(close_delimiter) = match first {
                b')' => Some(Delim::Paren),
                b']' => Some(Delim::Bracket),
                b'}' => Some(Delim::Brace),
                _ => None,
            } {
                let frame = match stack.pop() {
                    Some(frame) => frame,
                    None => return Err(lex_error(input)),
                };
                let (lo, frame) = frame;
                let (open_delimiter, outer) = frame;
                if open_delimiter != close_delimiter {
                    return Err(lex_error(input));
                }
                input = input.advance(1);
                let mut g = Group::new(open_delimiter, trees.build());
                g.set_span(Span { lo, hi: input.off });
                trees = outer;
                trees.push_token_from_parser(Tree::Group(super::Group::_new_fallback(g)));
            } else {
                let (rest, mut tt) = match leaf_token(input) {
                    Ok((rest, tt)) => (rest, tt),
                    Err(Reject) => return Err(lex_error(input)),
                };
                tt.set_span(super::Span::_new_fallback(Span { lo, hi: rest.off }));
                trees.push_token_from_parser(tt);
                input = rest;
            }
        }
    }
    fn lex_error(cursor: Cursor) -> LexError {
        LexError {
            span: Span {
                lo: cursor.off,
                hi: cursor.off,
            },
        }
    }
    fn leaf_token(input: Cursor) -> PResult<Tree> {
        if let Ok((input, l)) = literal(input) {
            Ok((input, Tree::Lit(super::Lit::_new_fallback(l))))
        } else if let Ok((input, p)) = punct(input) {
            Ok((input, Tree::Punct(p)))
        } else if let Ok((input, i)) = ident(input) {
            Ok((input, Tree::Ident(i)))
        } else {
            Err(Reject)
        }
    }
    fn ident(input: Cursor) -> PResult<super::Ident> {
        if ["r\"", "r#\"", "r##", "b\"", "b\'", "br\"", "br#", "c\"", "cr\"", "cr#"]
            .iter()
            .any(|prefix| input.starts_with(prefix))
        {
            Err(Reject)
        } else {
            ident_any(input)
        }
    }
    fn ident_any(input: Cursor) -> PResult<super::Ident> {
        let raw = input.starts_with("r#");
        let rest = input.advance((raw as usize) << 1);
        let (rest, sym) = ident_not_raw(rest)?;
        if !raw {
            let ident = super::Ident::new(sym, super::Span::call_site());
            return Ok((rest, ident));
        }
        match sym {
            "_" | "super" | "self" | "Self" | "crate" => return Err(Reject),
            _ => {},
        }
        let ident = super::Ident::_new_raw(sym, super::Span::call_site());
        Ok((rest, ident))
    }
    fn ident_not_raw(input: Cursor) -> PResult<&str> {
        let mut chars = input.char_indices();
        match chars.next() {
            Some((_, ch)) if is_ident_start(ch) => {},
            _ => return Err(Reject),
        }
        let mut end = input.len();
        for (i, ch) in chars {
            if !is_ident_continue(ch) {
                end = i;
                break;
            }
        }
        Ok((input.advance(end), &input.rest[..end]))
    }
    pub fn literal(input: Cursor) -> PResult<Lit> {
        let rest = literal_nocapture(input)?;
        let end = input.len() - rest.len();
        Ok((rest, Lit::_new(input.rest[..end].to_string())))
    }
    fn literal_nocapture(input: Cursor) -> Result<Cursor, Reject> {
        if let Ok(ok) = string(input) {
            Ok(ok)
        } else if let Ok(ok) = byte_string(input) {
            Ok(ok)
        } else if let Ok(ok) = c_string(input) {
            Ok(ok)
        } else if let Ok(ok) = byte(input) {
            Ok(ok)
        } else if let Ok(ok) = character(input) {
            Ok(ok)
        } else if let Ok(ok) = float(input) {
            Ok(ok)
        } else if let Ok(ok) = int(input) {
            Ok(ok)
        } else {
            Err(Reject)
        }
    }
    fn literal_suffix(input: Cursor) -> Cursor {
        match ident_not_raw(input) {
            Ok((input, _)) => input,
            Err(Reject) => input,
        }
    }
    fn string(input: Cursor) -> Result<Cursor, Reject> {
        if let Ok(input) = input.parse("\"") {
            cooked_string(input)
        } else if let Ok(input) = input.parse("r") {
            raw_string(input)
        } else {
            Err(Reject)
        }
    }
    fn cooked_string(mut input: Cursor) -> Result<Cursor, Reject> {
        let mut chars = input.char_indices();
        while let Some((i, ch)) = chars.next() {
            match ch {
                '"' => {
                    let input = input.advance(i + 1);
                    return Ok(literal_suffix(input));
                },
                '\r' => match chars.next() {
                    Some((_, '\n')) => {},
                    _ => break,
                },
                '\\' => match chars.next() {
                    Some((_, 'x')) => {
                        backslash_x_char(&mut chars)?;
                    },
                    Some((_, 'n')) | Some((_, 'r')) | Some((_, 't')) | Some((_, '\\')) | Some((_, '\''))
                    | Some((_, '"')) | Some((_, '0')) => {},
                    Some((_, 'u')) => {
                        backslash_u(&mut chars)?;
                    },
                    Some((newline, ch @ '\n')) | Some((newline, ch @ '\r')) => {
                        input = input.advance(newline + 1);
                        trailing_backslash(&mut input, ch as u8)?;
                        chars = input.char_indices();
                    },
                    _ => break,
                },
                _ch => {},
            }
        }
        Err(Reject)
    }
    fn raw_string(input: Cursor) -> Result<Cursor, Reject> {
        let (input, delimiter) = delimiter_of_raw_string(input)?;
        let mut bytes = input.bytes().enumerate();
        while let Some((i, byte)) = bytes.next() {
            match byte {
                b'"' if input.rest[i + 1..].starts_with(delimiter) => {
                    let rest = input.advance(i + 1 + delimiter.len());
                    return Ok(literal_suffix(rest));
                },
                b'\r' => match bytes.next() {
                    Some((_, b'\n')) => {},
                    _ => break,
                },
                _ => {},
            }
        }
        Err(Reject)
    }
    fn byte_string(input: Cursor) -> Result<Cursor, Reject> {
        if let Ok(input) = input.parse("b\"") {
            cooked_byte_string(input)
        } else if let Ok(input) = input.parse("br") {
            raw_byte_string(input)
        } else {
            Err(Reject)
        }
    }
    fn cooked_byte_string(mut input: Cursor) -> Result<Cursor, Reject> {
        let mut bytes = input.bytes().enumerate();
        while let Some((offset, b)) = bytes.next() {
            match b {
                b'"' => {
                    let input = input.advance(offset + 1);
                    return Ok(literal_suffix(input));
                },
                b'\r' => match bytes.next() {
                    Some((_, b'\n')) => {},
                    _ => break,
                },
                b'\\' => match bytes.next() {
                    Some((_, b'x')) => {
                        backslash_x_byte(&mut bytes)?;
                    },
                    Some((_, b'n')) | Some((_, b'r')) | Some((_, b't')) | Some((_, b'\\')) | Some((_, b'0'))
                    | Some((_, b'\'')) | Some((_, b'"')) => {},
                    Some((newline, b @ b'\n')) | Some((newline, b @ b'\r')) => {
                        input = input.advance(newline + 1);
                        trailing_backslash(&mut input, b)?;
                        bytes = input.bytes().enumerate();
                    },
                    _ => break,
                },
                b if b.is_ascii() => {},
                _ => break,
            }
        }
        Err(Reject)
    }
    fn delimiter_of_raw_string(input: Cursor) -> PResult<&str> {
        for (i, byte) in input.bytes().enumerate() {
            match byte {
                b'"' => {
                    if i > 255 {
                        return Err(Reject);
                    }
                    return Ok((input.advance(i + 1), &input.rest[..i]));
                },
                b'#' => {},
                _ => break,
            }
        }
        Err(Reject)
    }
    fn raw_byte_string(input: Cursor) -> Result<Cursor, Reject> {
        let (input, delimiter) = delimiter_of_raw_string(input)?;
        let mut bytes = input.bytes().enumerate();
        while let Some((i, byte)) = bytes.next() {
            match byte {
                b'"' if input.rest[i + 1..].starts_with(delimiter) => {
                    let rest = input.advance(i + 1 + delimiter.len());
                    return Ok(literal_suffix(rest));
                },
                b'\r' => match bytes.next() {
                    Some((_, b'\n')) => {},
                    _ => break,
                },
                other => {
                    if !other.is_ascii() {
                        break;
                    }
                },
            }
        }
        Err(Reject)
    }
    fn c_string(input: Cursor) -> Result<Cursor, Reject> {
        if let Ok(input) = input.parse("c\"") {
            cooked_c_string(input)
        } else if let Ok(input) = input.parse("cr") {
            raw_c_string(input)
        } else {
            Err(Reject)
        }
    }
    fn raw_c_string(input: Cursor) -> Result<Cursor, Reject> {
        let (input, delimiter) = delimiter_of_raw_string(input)?;
        let mut bytes = input.bytes().enumerate();
        while let Some((i, byte)) = bytes.next() {
            match byte {
                b'"' if input.rest[i + 1..].starts_with(delimiter) => {
                    let rest = input.advance(i + 1 + delimiter.len());
                    return Ok(literal_suffix(rest));
                },
                b'\r' => match bytes.next() {
                    Some((_, b'\n')) => {},
                    _ => break,
                },
                b'\0' => break,
                _ => {},
            }
        }
        Err(Reject)
    }
    fn cooked_c_string(mut input: Cursor) -> Result<Cursor, Reject> {
        let mut chars = input.char_indices();
        while let Some((i, ch)) = chars.next() {
            match ch {
                '"' => {
                    let input = input.advance(i + 1);
                    return Ok(literal_suffix(input));
                },
                '\r' => match chars.next() {
                    Some((_, '\n')) => {},
                    _ => break,
                },
                '\\' => match chars.next() {
                    Some((_, 'x')) => {
                        backslash_x_nonzero(&mut chars)?;
                    },
                    Some((_, 'n')) | Some((_, 'r')) | Some((_, 't')) | Some((_, '\\')) | Some((_, '\''))
                    | Some((_, '"')) => {},
                    Some((_, 'u')) => {
                        if backslash_u(&mut chars)? == '\0' {
                            break;
                        }
                    },
                    Some((newline, ch @ '\n')) | Some((newline, ch @ '\r')) => {
                        input = input.advance(newline + 1);
                        trailing_backslash(&mut input, ch as u8)?;
                        chars = input.char_indices();
                    },
                    _ => break,
                },
                '\0' => break,
                _ch => {},
            }
        }
        Err(Reject)
    }
    fn byte(input: Cursor) -> Result<Cursor, Reject> {
        let input = input.parse("b'")?;
        let mut bytes = input.bytes().enumerate();
        let ok = match bytes.next().map(|(_, b)| b) {
            Some(b'\\') => match bytes.next().map(|(_, b)| b) {
                Some(b'x') => backslash_x_byte(&mut bytes).is_ok(),
                Some(b'n') | Some(b'r') | Some(b't') | Some(b'\\') | Some(b'0') | Some(b'\'') | Some(b'"') => true,
                _ => false,
            },
            b => b.is_some(),
        };
        if !ok {
            return Err(Reject);
        }
        let (offset, _) = bytes.next().ok_or(Reject)?;
        if !input.chars().as_str().is_char_boundary(offset) {
            return Err(Reject);
        }
        let input = input.advance(offset).parse("'")?;
        Ok(literal_suffix(input))
    }
    fn character(input: Cursor) -> Result<Cursor, Reject> {
        let input = input.parse("'")?;
        let mut chars = input.char_indices();
        let ok = match chars.next().map(|(_, ch)| ch) {
            Some('\\') => match chars.next().map(|(_, ch)| ch) {
                Some('x') => backslash_x_char(&mut chars).is_ok(),
                Some('u') => backslash_u(&mut chars).is_ok(),
                Some('n') | Some('r') | Some('t') | Some('\\') | Some('0') | Some('\'') | Some('"') => true,
                _ => false,
            },
            ch => ch.is_some(),
        };
        if !ok {
            return Err(Reject);
        }
        let (idx, _) = chars.next().ok_or(Reject)?;
        let input = input.advance(idx).parse("'")?;
        Ok(literal_suffix(input))
    }
    macro_rules! next_ch {
        ($chars:ident @ $pat:pat_param $(| $rest:pat)*) => {
            match $chars.next() {
                Some((_, ch)) => match ch {
                    $pat $(| $rest)* => ch,
                    _ => return Err(Reject),
                },
                None => return Err(Reject),
            }
        };
    }
    fn backslash_x_char<I>(chars: &mut I) -> Result<(), Reject>
    where
        I: Iterator<Item = (usize, char)>,
    {
        next_ch!(chars @ '0'..='7');
        next_ch!(chars @ '0'..='9' | 'a'..='f' | 'A'..='F');
        Ok(())
    }
    fn backslash_x_byte<I>(chars: &mut I) -> Result<(), Reject>
    where
        I: Iterator<Item = (usize, u8)>,
    {
        next_ch!(chars @ b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F');
        next_ch!(chars @ b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F');
        Ok(())
    }
    fn backslash_x_nonzero<I>(chars: &mut I) -> Result<(), Reject>
    where
        I: Iterator<Item = (usize, char)>,
    {
        let first = next_ch!(chars @ '0'..='9' | 'a'..='f' | 'A'..='F');
        let second = next_ch!(chars @ '0'..='9' | 'a'..='f' | 'A'..='F');
        if first == '0' && second == '0' {
            Err(Reject)
        } else {
            Ok(())
        }
    }
    fn backslash_u<I>(chars: &mut I) -> Result<char, Reject>
    where
        I: Iterator<Item = (usize, char)>,
    {
        next_ch!(chars @ '{');
        let mut value = 0;
        let mut len = 0;
        for (_, ch) in chars {
            let digit = match ch {
                '0'..='9' => ch as u8 - b'0',
                'a'..='f' => 10 + ch as u8 - b'a',
                'A'..='F' => 10 + ch as u8 - b'A',
                '_' if len > 0 => continue,
                '}' if len > 0 => return char::from_u32(value).ok_or(Reject),
                _ => break,
            };
            if len == 6 {
                break;
            }
            value *= 0x10;
            value += u32::from(digit);
            len += 1;
        }
        Err(Reject)
    }
    fn trailing_backslash(input: &mut Cursor, mut last: u8) -> Result<(), Reject> {
        let mut whitespace = input.bytes().enumerate();
        loop {
            if last == b'\r' && whitespace.next().map_or(true, |(_, b)| b != b'\n') {
                return Err(Reject);
            }
            match whitespace.next() {
                Some((_, b @ b' ')) | Some((_, b @ b'\t')) | Some((_, b @ b'\n')) | Some((_, b @ b'\r')) => {
                    last = b;
                },
                Some((offset, _)) => {
                    *input = input.advance(offset);
                    return Ok(());
                },
                None => return Err(Reject),
            }
        }
    }
    fn float(input: Cursor) -> Result<Cursor, Reject> {
        let mut rest = float_digits(input)?;
        if let Some(ch) = rest.chars().next() {
            if is_ident_start(ch) {
                rest = ident_not_raw(rest)?.0;
            }
        }
        word_break(rest)
    }
    fn float_digits(input: Cursor) -> Result<Cursor, Reject> {
        let mut chars = input.chars().peekable();
        match chars.next() {
            Some(ch) if ch >= '0' && ch <= '9' => {},
            _ => return Err(Reject),
        }
        let mut len = 1;
        let mut has_dot = false;
        let mut has_exp = false;
        while let Some(&ch) = chars.peek() {
            match ch {
                '0'..='9' | '_' => {
                    chars.next();
                    len += 1;
                },
                '.' => {
                    if has_dot {
                        break;
                    }
                    chars.next();
                    if chars.peek().map_or(false, |&ch| ch == '.' || is_ident_start(ch)) {
                        return Err(Reject);
                    }
                    len += 1;
                    has_dot = true;
                },
                'e' | 'E' => {
                    chars.next();
                    len += 1;
                    has_exp = true;
                    break;
                },
                _ => break,
            }
        }
        if !(has_dot || has_exp) {
            return Err(Reject);
        }
        if has_exp {
            let token_before_exp = if has_dot {
                Ok(input.advance(len - 1))
            } else {
                Err(Reject)
            };
            let mut has_sign = false;
            let mut has_exp_value = false;
            while let Some(&ch) = chars.peek() {
                match ch {
                    '+' | '-' => {
                        if has_exp_value {
                            break;
                        }
                        if has_sign {
                            return token_before_exp;
                        }
                        chars.next();
                        len += 1;
                        has_sign = true;
                    },
                    '0'..='9' => {
                        chars.next();
                        len += 1;
                        has_exp_value = true;
                    },
                    '_' => {
                        chars.next();
                        len += 1;
                    },
                    _ => break,
                }
            }
            if !has_exp_value {
                return token_before_exp;
            }
        }
        Ok(input.advance(len))
    }
    fn int(input: Cursor) -> Result<Cursor, Reject> {
        let mut rest = digits(input)?;
        if let Some(ch) = rest.chars().next() {
            if is_ident_start(ch) {
                rest = ident_not_raw(rest)?.0;
            }
        }
        word_break(rest)
    }
    fn digits(mut input: Cursor) -> Result<Cursor, Reject> {
        let base = if input.starts_with("0x") {
            input = input.advance(2);
            16
        } else if input.starts_with("0o") {
            input = input.advance(2);
            8
        } else if input.starts_with("0b") {
            input = input.advance(2);
            2
        } else {
            10
        };
        let mut len = 0;
        let mut empty = true;
        for b in input.bytes() {
            match b {
                b'0'..=b'9' => {
                    let digit = (b - b'0') as u64;
                    if digit >= base {
                        return Err(Reject);
                    }
                },
                b'a'..=b'f' => {
                    let digit = 10 + (b - b'a') as u64;
                    if digit >= base {
                        break;
                    }
                },
                b'A'..=b'F' => {
                    let digit = 10 + (b - b'A') as u64;
                    if digit >= base {
                        break;
                    }
                },
                b'_' => {
                    if empty && base == 10 {
                        return Err(Reject);
                    }
                    len += 1;
                    continue;
                },
                _ => break,
            };
            len += 1;
            empty = false;
        }
        if empty {
            Err(Reject)
        } else {
            Ok(input.advance(len))
        }
    }
    fn punct(input: Cursor) -> PResult<Punct> {
        let (rest, ch) = punct_char(input)?;
        if ch == '\'' {
            if ident_any(rest)?.0.starts_with_char('\'') {
                Err(Reject)
            } else {
                Ok((rest, Punct::new('\'', Spacing::Joint)))
            }
        } else {
            let kind = match punct_char(rest) {
                Ok(_) => Spacing::Joint,
                Err(Reject) => Spacing::Alone,
            };
            Ok((rest, Punct::new(ch, kind)))
        }
    }
    fn punct_char(input: Cursor) -> PResult<char> {
        if input.starts_with("//") || input.starts_with("/*") {
            return Err(Reject);
        }
        let mut chars = input.chars();
        let first = match chars.next() {
            Some(ch) => ch,
            None => {
                return Err(Reject);
            },
        };
        let recognized = "~!@#$%^&*-=+|;:,<.>/?'";
        if recognized.contains(first) {
            Ok((input.advance(first.len_utf8()), first))
        } else {
            Err(Reject)
        }
    }
    fn doc_comment<'a>(input: Cursor<'a>, trees: &mut TokenStreamBuilder) -> PResult<'a, ()> {
        let lo = input.off;
        let (rest, (comment, inner)) = doc_comment_contents(input)?;
        let span = super::Span::_new_fallback(Span { lo, hi: rest.off });
        let mut scan_for_bare_cr = comment;
        while let Some(cr) = scan_for_bare_cr.find('\r') {
            let rest = &scan_for_bare_cr[cr + 1..];
            if !rest.starts_with('\n') {
                return Err(Reject);
            }
            scan_for_bare_cr = rest;
        }
        let mut pound = Punct::new('#', Spacing::Alone);
        pound.set_span(span);
        trees.push_token_from_parser(Tree::Punct(pound));
        if inner {
            let mut bang = Punct::new('!', Spacing::Alone);
            bang.set_span(span);
            trees.push_token_from_parser(Tree::Punct(bang));
        }
        let doc_ident = super::Ident::new("doc", span);
        let mut equal = Punct::new('=', Spacing::Alone);
        equal.set_span(span);
        let mut literal = super::Lit::string(comment);
        literal.set_span(span);
        let mut bracketed = TokenStreamBuilder::with_capacity(3);
        bracketed.push_token_from_parser(Tree::Ident(doc_ident));
        bracketed.push_token_from_parser(Tree::Punct(equal));
        bracketed.push_token_from_parser(Tree::Lit(literal));
        let group = Group::new(Delim::Bracket, bracketed.build());
        let mut group = super::Group::_new_fallback(group);
        group.set_span(span);
        trees.push_token_from_parser(Tree::Group(group));
        Ok((rest, ()))
    }
    fn doc_comment_contents(input: Cursor) -> PResult<(&str, bool)> {
        if input.starts_with("//!") {
            let input = input.advance(3);
            let (input, s) = take_until_newline_or_eof(input);
            Ok((input, (s, true)))
        } else if input.starts_with("/*!") {
            let (input, s) = block_comment(input)?;
            Ok((input, (&s[3..s.len() - 2], true)))
        } else if input.starts_with("///") {
            let input = input.advance(3);
            if input.starts_with_char('/') {
                return Err(Reject);
            }
            let (input, s) = take_until_newline_or_eof(input);
            Ok((input, (s, false)))
        } else if input.starts_with("/**") && !input.rest[3..].starts_with('*') {
            let (input, s) = block_comment(input)?;
            Ok((input, (&s[3..s.len() - 2], false)))
        } else {
            Err(Reject)
        }
    }
    fn take_until_newline_or_eof(input: Cursor) -> (Cursor, &str) {
        let chars = input.char_indices();
        for (i, ch) in chars {
            if ch == '\n' {
                return (input.advance(i), &input.rest[..i]);
            } else if ch == '\r' && input.rest[i + 1..].starts_with('\n') {
                return (input.advance(i + 1), &input.rest[..i]);
            }
        }
        (input.advance(input.len()), input.rest)
    }
}
mod rcvec {
    use std::mem;
    use std::panic::RefUnwindSafe;
    use std::rc::Rc;
    use std::slice;
    use std::vec;
    pub struct RcVec<T> {
        inner: Rc<Vec<T>>,
    }
    pub struct RcVecBuilder<T> {
        inner: Vec<T>,
    }
    pub struct RcVecMut<'a, T> {
        inner: &'a mut Vec<T>,
    }
    #[derive(Clone)]
    pub struct RcVecIntoIter<T> {
        inner: vec::IntoIter<T>,
    }
    impl<T> RcVec<T> {
        pub fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }
        pub fn len(&self) -> usize {
            self.inner.len()
        }
        pub fn iter(&self) -> slice::Iter<T> {
            self.inner.iter()
        }
        pub fn make_mut(&mut self) -> RcVecMut<T>
        where
            T: Clone,
        {
            RcVecMut {
                inner: Rc::make_mut(&mut self.inner),
            }
        }
        pub fn get_mut(&mut self) -> Option<RcVecMut<T>> {
            let inner = Rc::get_mut(&mut self.inner)?;
            Some(RcVecMut { inner })
        }
        pub fn make_owned(mut self) -> RcVecBuilder<T>
        where
            T: Clone,
        {
            let vec = if let Some(owned) = Rc::get_mut(&mut self.inner) {
                mem::replace(owned, Vec::new())
            } else {
                Vec::clone(&self.inner)
            };
            RcVecBuilder { inner: vec }
        }
    }
    impl<T> RcVecBuilder<T> {
        pub fn new() -> Self {
            RcVecBuilder { inner: Vec::new() }
        }
        pub fn with_capacity(cap: usize) -> Self {
            RcVecBuilder {
                inner: Vec::with_capacity(cap),
            }
        }
        pub fn push(&mut self, element: T) {
            self.inner.push(element);
        }
        pub fn extend(&mut self, iter: impl IntoIterator<Item = T>) {
            self.inner.extend(iter);
        }
        pub fn as_mut(&mut self) -> RcVecMut<T> {
            RcVecMut { inner: &mut self.inner }
        }
        pub fn build(self) -> RcVec<T> {
            RcVec {
                inner: Rc::new(self.inner),
            }
        }
    }
    impl<'a, T> RcVecMut<'a, T> {
        pub fn push(&mut self, element: T) {
            self.inner.push(element);
        }
        pub fn extend(&mut self, iter: impl IntoIterator<Item = T>) {
            self.inner.extend(iter);
        }
        pub fn pop(&mut self) -> Option<T> {
            self.inner.pop()
        }
        pub fn as_mut(&mut self) -> RcVecMut<T> {
            RcVecMut { inner: self.inner }
        }
    }
    impl<T> Clone for RcVec<T> {
        fn clone(&self) -> Self {
            RcVec {
                inner: Rc::clone(&self.inner),
            }
        }
    }
    impl<T> IntoIterator for RcVecBuilder<T> {
        type Item = T;
        type IntoIter = RcVecIntoIter<T>;
        fn into_iter(self) -> Self::IntoIter {
            RcVecIntoIter {
                inner: self.inner.into_iter(),
            }
        }
    }
    impl<T> Iterator for RcVecIntoIter<T> {
        type Item = T;
        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.inner.size_hint()
        }
    }
    impl<T> RefUnwindSafe for RcVec<T> where T: RefUnwindSafe {}
}
pub mod fallback {
    use super::{
        location::LineColumn,
        parse::{self, Cursor},
        rcvec::{RcVec, RcVecBuilder, RcVecIntoIter, RcVecMut},
        *,
    };
    use std::{
        cell::RefCell,
        cmp,
        fmt::{self, Debug, Display, Write},
        iter::FromIterator,
        mem::ManuallyDrop,
        ops::RangeBounds,
        path::PathBuf,
        ptr,
        str::FromStr,
    };
    pub fn force() {
        super::detection::force_fallback();
    }
    pub fn unforce() {
        super::detection::unforce_fallback();
    }
    #[derive(Clone)]
    pub struct TokenStream {
        inner: RcVec<Tree>,
    }
    #[derive(Debug)]
    pub struct LexError {
        pub span: Span,
    }
    impl LexError {
        pub fn span(&self) -> Span {
            self.span
        }
        fn call_site() -> Self {
            LexError {
                span: Span::call_site(),
            }
        }
    }
    impl TokenStream {
        pub fn new() -> Self {
            TokenStream {
                inner: RcVecBuilder::new().build(),
            }
        }
        pub fn is_empty(&self) -> bool {
            self.inner.len() == 0
        }
        fn take_inner(self) -> RcVecBuilder<Tree> {
            let nodrop = ManuallyDrop::new(self);
            unsafe { ptr::read(&nodrop.inner) }.make_owned()
        }
    }
    fn push_token_from_proc_macro(mut vec: RcVecMut<Tree>, token: Tree) {
        match token {
            Tree::Lit(super::Lit {
                inner: super::imp::Lit::Fallback(literal),
                ..
            }) if literal.repr.starts_with('-') => {
                push_negative_literal(vec, literal);
            },
            _ => vec.push(token),
        }
        #[cold]
        fn push_negative_literal(mut vec: RcVecMut<Tree>, mut literal: Lit) {
            literal.repr.remove(0);
            let mut punct = super::Punct::new('-', Spacing::Alone);
            punct.set_span(super::Span::_new_fallback(literal.span));
            vec.push(Tree::Punct(punct));
            vec.push(Tree::Lit(super::Lit::_new_fallback(literal)));
        }
    }
    impl Drop for TokenStream {
        fn drop(&mut self) {
            let mut inner = match self.inner.get_mut() {
                Some(inner) => inner,
                None => return,
            };
            while let Some(token) = inner.pop() {
                let group = match token {
                    Tree::Group(group) => group.inner,
                    _ => continue,
                };
                let group = match group {
                    super::imp::Group::Fallback(group) => group,
                    super::imp::Group::Compiler(_) => continue,
                };
                inner.extend(group.stream.take_inner());
            }
        }
    }
    pub struct TokenStreamBuilder {
        inner: RcVecBuilder<Tree>,
    }
    impl TokenStreamBuilder {
        pub fn new() -> Self {
            TokenStreamBuilder {
                inner: RcVecBuilder::new(),
            }
        }
        pub fn with_capacity(cap: usize) -> Self {
            TokenStreamBuilder {
                inner: RcVecBuilder::with_capacity(cap),
            }
        }
        pub fn push_token_from_parser(&mut self, tt: Tree) {
            self.inner.push(tt);
        }
        pub fn build(self) -> TokenStream {
            TokenStream {
                inner: self.inner.build(),
            }
        }
    }
    fn get_cursor(src: &str) -> Cursor {
        SRC_MAP.with(|cm| {
            let mut cm = cm.borrow_mut();
            let span = cm.add_file(src);
            Cursor {
                rest: src,
                off: span.lo,
            }
        })
    }
    impl FromStr for TokenStream {
        type Err = LexError;
        fn from_str(src: &str) -> Result<TokenStream, LexError> {
            let mut cursor = get_cursor(src);
            const BYTE_ORDER_MARK: &str = "\u{feff}";
            if cursor.starts_with(BYTE_ORDER_MARK) {
                cursor = cursor.advance(BYTE_ORDER_MARK.len());
            }
            parse::token_stream(cursor)
        }
    }
    impl Display for LexError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("cannot parse string into token stream")
        }
    }
    impl Display for TokenStream {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut joint = false;
            for (i, tt) in self.inner.iter().enumerate() {
                if i != 0 && !joint {
                    write!(f, " ")?;
                }
                joint = false;
                match tt {
                    Tree::Group(tt) => Display::fmt(tt, f),
                    Tree::Ident(tt) => Display::fmt(tt, f),
                    Tree::Punct(tt) => {
                        joint = tt.spacing() == Spacing::Joint;
                        Display::fmt(tt, f)
                    },
                    Tree::Lit(tt) => Display::fmt(tt, f),
                }?;
            }
            Ok(())
        }
    }
    impl Debug for TokenStream {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("TokenStream ")?;
            f.debug_list().entries(self.clone()).finish()
        }
    }
    impl From<proc_macro::TokenStream> for TokenStream {
        fn from(inner: proc_macro::TokenStream) -> Self {
            inner.to_string().parse().expect("compiler token stream parse failed")
        }
    }
    impl From<TokenStream> for proc_macro::TokenStream {
        fn from(inner: TokenStream) -> Self {
            inner.to_string().parse().expect("failed to parse to compiler tokens")
        }
    }
    impl From<Tree> for TokenStream {
        fn from(tree: Tree) -> Self {
            let mut stream = RcVecBuilder::new();
            push_token_from_proc_macro(stream.as_mut(), tree);
            TokenStream { inner: stream.build() }
        }
    }
    impl FromIterator<Tree> for TokenStream {
        fn from_iter<I: IntoIterator<Item = Tree>>(tokens: I) -> Self {
            let mut stream = TokenStream::new();
            stream.extend(tokens);
            stream
        }
    }
    impl FromIterator<TokenStream> for TokenStream {
        fn from_iter<I: IntoIterator<Item = TokenStream>>(streams: I) -> Self {
            let mut v = RcVecBuilder::new();
            for stream in streams {
                v.extend(stream.take_inner());
            }
            TokenStream { inner: v.build() }
        }
    }
    impl Extend<Tree> for TokenStream {
        fn extend<I: IntoIterator<Item = Tree>>(&mut self, tokens: I) {
            let mut vec = self.inner.make_mut();
            tokens
                .into_iter()
                .for_each(|token| push_token_from_proc_macro(vec.as_mut(), token));
        }
    }
    impl Extend<TokenStream> for TokenStream {
        fn extend<I: IntoIterator<Item = TokenStream>>(&mut self, streams: I) {
            self.inner.make_mut().extend(streams.into_iter().flatten());
        }
    }
    pub type TreeIter = RcVecIntoIter<Tree>;
    impl IntoIterator for TokenStream {
        type Item = Tree;
        type IntoIter = TreeIter;
        fn into_iter(self) -> TreeIter {
            self.take_inner().into_iter()
        }
    }
    #[derive(Clone, PartialEq, Eq)]
    pub struct SourceFile {
        path: PathBuf,
    }
    impl SourceFile {
        pub fn path(&self) -> PathBuf {
            self.path.clone()
        }
        pub fn is_real(&self) -> bool {
            false
        }
    }
    impl Debug for SourceFile {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.debug_struct("SourceFile")
                .field("path", &self.path())
                .field("is_real", &self.is_real())
                .finish()
        }
    }
    thread_local! {
        static SRC_MAP: RefCell<SourceMap> = RefCell::new(SourceMap {
            files: vec![FileInfo {
                text: String::new(),
                span: Span { lo: 0, hi: 0 },
                lines: vec![0],
            }],
        });
    }
    struct FileInfo {
        text: String,
        span: Span,
        lines: Vec<usize>,
    }
    impl FileInfo {
        fn offset_line_column(&self, offset: usize) -> LineColumn {
            assert!(self.span_within(Span {
                lo: offset as u32,
                hi: offset as u32
            }));
            let offset = offset - self.span.lo as usize;
            match self.lines.binary_search(&offset) {
                Ok(found) => LineColumn {
                    line: found + 1,
                    column: 0,
                },
                Err(idx) => LineColumn {
                    line: idx,
                    column: offset - self.lines[idx - 1],
                },
            }
        }
        fn span_within(&self, span: Span) -> bool {
            span.lo >= self.span.lo && span.hi <= self.span.hi
        }
        fn source_text(&self, span: Span) -> String {
            let lo = (span.lo - self.span.lo) as usize;
            let hi = (span.hi - self.span.lo) as usize;
            self.text[lo..hi].to_owned()
        }
    }
    fn lines_offsets(s: &str) -> (usize, Vec<usize>) {
        let mut lines = vec![0];
        let mut total = 0;
        for ch in s.chars() {
            total += 1;
            if ch == '\n' {
                lines.push(total);
            }
        }
        (total, lines)
    }
    struct SourceMap {
        files: Vec<FileInfo>,
    }
    impl SourceMap {
        fn next_start_pos(&self) -> u32 {
            self.files.last().unwrap().span.hi + 1
        }
        fn add_file(&mut self, src: &str) -> Span {
            let (len, lines) = lines_offsets(src);
            let lo = self.next_start_pos();
            let span = Span {
                lo,
                hi: lo + (len as u32),
            };
            self.files.push(FileInfo {
                text: src.to_owned(),
                span,
                lines,
            });
            span
        }
        fn fileinfo(&self, span: Span) -> &FileInfo {
            for file in &self.files {
                if file.span_within(span) {
                    return file;
                }
            }
            unreachable!("Invalid span with no related FileInfo!");
        }
    }

    #[derive(Clone)]
    pub struct Ident {
        sym: String,
        span: Span,
        raw: bool,
    }
    impl Ident {
        fn _new(string: &str, raw: bool, span: Span) -> Self {
            validate_ident(string, raw);
            Ident {
                sym: string.to_owned(),
                span,
                raw,
            }
        }
        pub fn new(string: &str, span: Span) -> Self {
            Ident::_new(string, false, span)
        }
        pub fn new_raw(string: &str, span: Span) -> Self {
            Ident::_new(string, true, span)
        }
        pub fn span(&self) -> Span {
            self.span
        }
        pub fn set_span(&mut self, span: Span) {
            self.span = span;
        }
    }
    pub fn is_ident_start(c: char) -> bool {
        c == '_' || unicode_ident::is_xid_start(c)
    }
    pub fn is_ident_continue(c: char) -> bool {
        unicode_ident::is_xid_continue(c)
    }
    fn validate_ident(string: &str, raw: bool) {
        if string.is_empty() {
            panic!("Ident is not allowed to be empty; use Option<Ident>");
        }
        if string.bytes().all(|digit| digit >= b'0' && digit <= b'9') {
            panic!("Ident cannot be a number; use Literal instead");
        }
        fn ident_ok(string: &str) -> bool {
            let mut chars = string.chars();
            let first = chars.next().unwrap();
            if !is_ident_start(first) {
                return false;
            }
            for ch in chars {
                if !is_ident_continue(ch) {
                    return false;
                }
            }
            true
        }
        if !ident_ok(string) {
            panic!("{:?} is not a valid Ident", string);
        }
        if raw {
            match string {
                "_" | "super" | "self" | "Self" | "crate" => {
                    panic!("`r#{}` cannot be a raw identifier", string);
                },
                _ => {},
            }
        }
    }
    impl PartialEq for Ident {
        fn eq(&self, other: &Ident) -> bool {
            self.sym == other.sym && self.raw == other.raw
        }
    }
    impl<T> PartialEq<T> for Ident
    where
        T: ?Sized + AsRef<str>,
    {
        fn eq(&self, other: &T) -> bool {
            let other = other.as_ref();
            if self.raw {
                other.starts_with("r#") && self.sym == other[2..]
            } else {
                self.sym == other
            }
        }
    }
    impl Display for Ident {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            if self.raw {
                f.write_str("r#")?;
            }
            Display::fmt(&self.sym, f)
        }
    }
    #[allow(clippy::missing_fields_in_debug)]
    impl Debug for Ident {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut y = f.debug_struct("Ident");
            y.field("sym", &format_args!("{}", self));
            debug_span_field(&mut y, self.span);
            y.finish()
        }
    }

    #[derive(Clone)]
    pub struct Group {
        delim: Delim,
        stream: TokenStream,
        span: Span,
    }
    impl Group {
        pub fn new(delim: Delim, stream: TokenStream) -> Self {
            Group {
                delim,
                stream,
                span: Span::call_site(),
            }
        }
        pub fn delim(&self) -> Delim {
            self.delim
        }
        pub fn stream(&self) -> TokenStream {
            self.stream.clone()
        }
        pub fn span(&self) -> Span {
            self.span
        }
        pub fn set_span(&mut self, x: Span) {
            self.span = x;
        }
        pub fn span_open(&self) -> Span {
            self.span.first_byte()
        }
        pub fn span_close(&self) -> Span {
            self.span.last_byte()
        }
    }
    impl Display for Group {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let (open, close) = match self.delim {
                Delim::Paren => ("(", ")"),
                Delim::Brace => ("{ ", "}"),
                Delim::Bracket => ("[", "]"),
                Delim::None => ("", ""),
            };
            f.write_str(open)?;
            Display::fmt(&self.stream, f)?;
            if self.delim == Delim::Brace && !self.stream.inner.is_empty() {
                f.write_str(" ")?;
            }
            f.write_str(close)?;
            Ok(())
        }
    }
    impl Debug for Group {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut y = f.debug_struct("Group");
            y.field("delimiter", &self.delim);
            y.field("stream", &self.stream);
            debug_span_field(&mut y, self.span);
            y.finish()
        }
    }

    #[derive(Clone)]
    pub struct Lit {
        repr: String,
        span: Span,
        _marker: Marker,
    }
    macro_rules! suffixed_nums {
        ($($n:ident => $kind:ident,)*) => ($(
            pub fn $n(n: $kind) -> Lit {
                Lit::_new(format!(concat!("{}", stringify!($kind)), n))
            }
        )*)
    }
    macro_rules! unsuffixed_nums {
        ($($n:ident => $kind:ident,)*) => ($(
            pub fn $n(n: $kind) -> Lit {
                Lit::_new(n.to_string())
            }
        )*)
    }
    impl Lit {
        pub fn _new(repr: String) -> Self {
            Lit {
                repr,
                span: Span::call_site(),
                _marker: Marker,
            }
        }
        pub unsafe fn from_str_unchecked(x: &str) -> Self {
            Lit::_new(x.to_owned())
        }
        suffixed_nums! {
            f32_suffixed => f32,
            f64_suffixed => f64,
            i128_suffixed => i128,
            i16_suffixed => i16,
            i32_suffixed => i32,
            i64_suffixed => i64,
            i8_suffixed => i8,
            isize_suffixed => isize,
            u128_suffixed => u128,
            u16_suffixed => u16,
            u32_suffixed => u32,
            u64_suffixed => u64,
            u8_suffixed => u8,
            usize_suffixed => usize,
        }
        unsuffixed_nums! {
            i128_unsuffixed => i128,
            i16_unsuffixed => i16,
            i32_unsuffixed => i32,
            i64_unsuffixed => i64,
            i8_unsuffixed => i8,
            isize_unsuffixed => isize,
            u128_unsuffixed => u128,
            u16_unsuffixed => u16,
            u32_unsuffixed => u32,
            u64_unsuffixed => u64,
            u8_unsuffixed => u8,
            usize_unsuffixed => usize,
        }
        pub fn f32_unsuffixed(x: f32) -> Lit {
            let mut y = x.to_string();
            if !y.contains('.') {
                y.push_str(".0");
            }
            Lit::_new(y)
        }
        pub fn f64_unsuffixed(x: f64) -> Lit {
            let mut y = x.to_string();
            if !y.contains('.') {
                y.push_str(".0");
            }
            Lit::_new(y)
        }
        pub fn string(x: &str) -> Lit {
            let mut y = String::with_capacity(x.len() + 2);
            y.push('"');
            let mut xs = x.chars();
            while let Some(x) = xs.next() {
                if x == '\0' {
                    y.push_str(if xs.as_str().starts_with(|x| '0' <= x && x <= '7') {
                        "\\x00"
                    } else {
                        "\\0"
                    });
                } else if x == '\'' {
                    y.push(x);
                } else {
                    y.extend(x.escape_debug());
                }
            }
            y.push('"');
            Lit::_new(y)
        }
        pub fn character(x: char) -> Lit {
            let mut y = String::new();
            y.push('\'');
            if x == '"' {
                y.push(x);
            } else {
                y.extend(x.escape_debug());
            }
            y.push('\'');
            Lit::_new(y)
        }
        pub fn byte_string(xs: &[u8]) -> Lit {
            let mut y = "b\"".to_string();
            let mut xs = xs.iter();
            while let Some(&x) = xs.next() {
                #[allow(clippy::match_overlapping_arm)]
                match x {
                    b'\0' => y.push_str(match xs.as_slice().first() {
                        Some(b'0'..=b'7') => r"\x00",
                        _ => r"\0",
                    }),
                    b'\t' => y.push_str(r"\t"),
                    b'\n' => y.push_str(r"\n"),
                    b'\r' => y.push_str(r"\r"),
                    b'"' => y.push_str("\\\""),
                    b'\\' => y.push_str("\\\\"),
                    b'\x20'..=b'\x7E' => y.push(x as char),
                    _ => {
                        let _ = write!(y, "\\x{:02X}", x);
                    },
                }
            }
            y.push('"');
            Lit::_new(y)
        }
        pub fn span(&self) -> Span {
            self.span
        }
        pub fn set_span(&mut self, x: Span) {
            self.span = x;
        }
        pub fn subspan<R: RangeBounds<usize>>(&self, range: R) -> Option<Span> {
            use super::convert::usize_to_u32;
            use std::ops::Bound::*;
            let lo = match range.start_bound() {
                Included(x) => {
                    let x = usize_to_u32(*x)?;
                    self.span.lo.checked_add(x)?
                },
                Excluded(x) => {
                    let x = usize_to_u32(*x)?;
                    self.span.lo.checked_add(x)?.checked_add(1)?
                },
                Unbounded => self.span.lo,
            };
            let hi = match range.end_bound() {
                Included(x) => {
                    let x = usize_to_u32(*x)?;
                    self.span.lo.checked_add(x)?.checked_add(1)?
                },
                Excluded(x) => {
                    let x = usize_to_u32(*x)?;
                    self.span.lo.checked_add(x)?
                },
                Unbounded => self.span.hi,
            };
            if lo <= hi && hi <= self.span.hi {
                Some(Span { lo, hi })
            } else {
                None
            }
        }
    }
    impl FromStr for Lit {
        type Err = LexError;
        fn from_str(repr: &str) -> Result<Self, Self::Err> {
            let mut cursor = get_cursor(repr);
            let lo = cursor.off;
            let negative = cursor.starts_with_char('-');
            if negative {
                cursor = cursor.advance(1);
                if !cursor.starts_with_fn(|ch| ch.is_ascii_digit()) {
                    return Err(LexError::call_site());
                }
            }
            if let Ok((rest, mut literal)) = parse::literal(cursor) {
                if rest.is_empty() {
                    if negative {
                        literal.repr.insert(0, '-');
                    }
                    literal.span = Span { lo, hi: rest.off };
                    return Ok(literal);
                }
            }
            Err(LexError::call_site())
        }
    }
    impl Display for Lit {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            Display::fmt(&self.repr, f)
        }
    }
    impl Debug for Lit {
        fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
            let mut debug = fmt.debug_struct("Literal");
            debug.field("lit", &format_args!("{}", self.repr));
            debug_span_field(&mut debug, self.span);
            debug.finish()
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct Span {
        pub lo: u32,
        pub hi: u32,
        _marker: Marker,
    }
    impl Span {
        pub fn call_site() -> Self {
            Span {
                lo: 0,
                hi: 0,
                _marker: Marker,
            }
        }
        pub fn mixed_site() -> Self {
            Span::call_site()
        }
        pub fn resolved_at(&self, _: Span) -> Span {
            *self
        }
        pub fn located_at(&self, x: Span) -> Span {
            x
        }
        pub fn start(&self) -> LineColumn {
            SRC_MAP.with(|x| {
                let x = x.borrow();
                let y = x.fileinfo(*self);
                y.offset_line_column(self.lo as usize)
            })
        }
        pub fn end(&self) -> LineColumn {
            SRC_MAP.with(|x| {
                let x = x.borrow();
                let y = x.fileinfo(*self);
                y.offset_line_column(self.hi as usize)
            })
        }
        pub fn join(&self, s: Span) -> Option<Span> {
            SRC_MAP.with(|x| {
                let x = x.borrow();
                if !x.fileinfo(*self).span_within(s) {
                    return None;
                }
                Some(Span {
                    lo: cmp::min(self.lo, s.lo),
                    hi: cmp::max(self.hi, s.hi),
                    _marker: Marker,
                })
            })
        }
        pub fn source_text(&self) -> Option<String> {
            {
                if self.is_call_site() {
                    None
                } else {
                    Some(SRC_MAP.with(|x| x.borrow().fileinfo(*self).source_text(*self)))
                }
            }
        }
        pub fn first_byte(self) -> Self {
            Span {
                lo: self.lo,
                hi: cmp::min(self.lo.saturating_add(1), self.hi),
                _marker: Marker,
            }
        }
        pub fn last_byte(self) -> Self {
            Span {
                lo: cmp::max(self.hi.saturating_sub(1), self.lo),
                hi: self.hi,
                _marker: Marker,
            }
        }
        fn is_call_site(&self) -> bool {
            self.lo == 0 && self.hi == 0
        }
    }
    impl Debug for Span {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            return write!(f, "bytes({}..{})", self.lo, self.hi);
        }
    }
    pub fn debug_span_field(x: &mut fmt::DebugStruct, s: Span) {
        {
            if s.is_call_site() {
                return;
            }
        }
        x.field("span", &s);
    }
}
pub mod extra {
    use super::{fallback, imp, Marker, Span};
    use std::fmt::{self, Debug};
    #[derive(Copy, Clone)]
    pub struct DelimSpan {
        inner: DelimSpanEnum,
        _marker: Marker,
    }
    #[derive(Copy, Clone)]
    enum DelimSpanEnum {
        Compiler {
            join: proc_macro::Span,
            open: proc_macro::Span,
            close: proc_macro::Span,
        },
        Fallback(fallback::Span),
    }
    impl DelimSpan {
        pub fn new(group: &imp::Group) -> Self {
            let inner = match group {
                imp::Group::Compiler(group) => DelimSpanEnum::Compiler {
                    join: group.span(),
                    open: group.span_open(),
                    close: group.span_close(),
                },
                imp::Group::Fallback(group) => DelimSpanEnum::Fallback(group.span()),
            };
            DelimSpan { inner, _marker: Marker }
        }
        pub fn join(&self) -> Span {
            match &self.inner {
                DelimSpanEnum::Compiler { join, .. } => Span::_new(imp::Span::Compiler(*join)),
                DelimSpanEnum::Fallback(span) => Span::_new_fallback(*span),
            }
        }
        pub fn open(&self) -> Span {
            match &self.inner {
                DelimSpanEnum::Compiler { open, .. } => Span::_new(imp::Span::Compiler(*open)),
                DelimSpanEnum::Fallback(span) => Span::_new_fallback(span.first_byte()),
            }
        }
        pub fn close(&self) -> Span {
            match &self.inner {
                DelimSpanEnum::Compiler { close, .. } => Span::_new(imp::Span::Compiler(*close)),
                DelimSpanEnum::Fallback(span) => Span::_new_fallback(span.last_byte()),
            }
        }
    }
    impl Debug for DelimSpan {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            Debug::fmt(&self.join(), f)
        }
    }
}
mod imp {
    use super::{location::LineColumn, *};
    use std::fmt::{self, Debug, Display};
    use std::iter::FromIterator;
    use std::ops::RangeBounds;
    use std::panic;
    use std::str::FromStr;
    #[derive(Clone)]
    pub enum TokenStream {
        Compiler(DeferredTokenStream),
        Fallback(fallback::TokenStream),
    }
    pub enum LexError {
        Compiler(proc_macro::LexError),
        Fallback(fallback::LexError),
    }
    impl LexError {
        fn call_site() -> Self {
            LexError::Fallback(fallback::LexError {
                span: fallback::Span::call_site(),
            })
        }
    }
    impl TokenStream {
        pub fn new() -> Self {
            if inside_proc_macro() {
                TokenStream::Compiler(DeferredTokenStream::new(proc_macro::TokenStream::new()))
            } else {
                TokenStream::Fallback(fallback::TokenStream::new())
            }
        }
        pub fn is_empty(&self) -> bool {
            match self {
                TokenStream::Compiler(tts) => tts.is_empty(),
                TokenStream::Fallback(tts) => tts.is_empty(),
            }
        }
        fn unwrap_nightly(self) -> proc_macro::TokenStream {
            match self {
                TokenStream::Compiler(s) => s.into_token_stream(),
                TokenStream::Fallback(_) => mismatch(),
            }
        }
        fn unwrap_stable(self) -> fallback::TokenStream {
            match self {
                TokenStream::Compiler(_) => mismatch(),
                TokenStream::Fallback(s) => s,
            }
        }
    }
    impl FromStr for TokenStream {
        type Err = LexError;
        fn from_str(src: &str) -> Result<TokenStream, LexError> {
            if inside_proc_macro() {
                Ok(TokenStream::Compiler(DeferredTokenStream::new(proc_macro_parse(src)?)))
            } else {
                Ok(TokenStream::Fallback(src.parse()?))
            }
        }
    }
    fn proc_macro_parse(src: &str) -> Result<proc_macro::TokenStream, LexError> {
        let result = panic::catch_unwind(|| src.parse().map_err(LexError::Compiler));
        result.unwrap_or_else(|_| Err(LexError::call_site()))
    }
    impl Display for TokenStream {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                TokenStream::Compiler(tts) => Display::fmt(&tts.clone().into_token_stream(), f),
                TokenStream::Fallback(tts) => Display::fmt(tts, f),
            }
        }
    }
    impl From<proc_macro::TokenStream> for TokenStream {
        fn from(inner: proc_macro::TokenStream) -> Self {
            TokenStream::Compiler(DeferredTokenStream::new(inner))
        }
    }
    impl From<TokenStream> for proc_macro::TokenStream {
        fn from(inner: TokenStream) -> Self {
            match inner {
                TokenStream::Compiler(inner) => inner.into_token_stream(),
                TokenStream::Fallback(inner) => inner.to_string().parse().unwrap(),
            }
        }
    }
    impl From<fallback::TokenStream> for TokenStream {
        fn from(inner: fallback::TokenStream) -> Self {
            TokenStream::Fallback(inner)
        }
    }
    fn into_compiler_token(token: Tree) -> proc_macro::TokenTree {
        match token {
            Tree::Group(tt) => tt.inner.unwrap_nightly().into(),
            Tree::Punct(tt) => {
                let spacing = match tt.spacing() {
                    Spacing::Joint => proc_macro::Spacing::Joint,
                    Spacing::Alone => proc_macro::Spacing::Alone,
                };
                let mut punct = proc_macro::Punct::new(tt.as_char(), spacing);
                punct.set_span(tt.span().inner.unwrap_nightly());
                punct.into()
            },
            Tree::Ident(tt) => tt.inner.unwrap_nightly().into(),
            Tree::Lit(tt) => tt.inner.unwrap_nightly().into(),
        }
    }
    impl From<Tree> for TokenStream {
        fn from(token: Tree) -> Self {
            if inside_proc_macro() {
                TokenStream::Compiler(DeferredTokenStream::new(into_compiler_token(token).into()))
            } else {
                TokenStream::Fallback(token.into())
            }
        }
    }
    impl FromIterator<Tree> for TokenStream {
        fn from_iter<I: IntoIterator<Item = Tree>>(trees: I) -> Self {
            if inside_proc_macro() {
                TokenStream::Compiler(DeferredTokenStream::new(
                    trees.into_iter().map(into_compiler_token).collect(),
                ))
            } else {
                TokenStream::Fallback(trees.into_iter().collect())
            }
        }
    }
    impl FromIterator<TokenStream> for TokenStream {
        fn from_iter<I: IntoIterator<Item = TokenStream>>(streams: I) -> Self {
            let mut streams = streams.into_iter();
            match streams.next() {
                Some(TokenStream::Compiler(mut first)) => {
                    first.evaluate_now();
                    first.stream.extend(streams.map(|s| match s {
                        TokenStream::Compiler(s) => s.into_token_stream(),
                        TokenStream::Fallback(_) => mismatch(),
                    }));
                    TokenStream::Compiler(first)
                },
                Some(TokenStream::Fallback(mut first)) => {
                    first.extend(streams.map(|s| match s {
                        TokenStream::Fallback(s) => s,
                        TokenStream::Compiler(_) => mismatch(),
                    }));
                    TokenStream::Fallback(first)
                },
                None => TokenStream::new(),
            }
        }
    }
    impl Extend<Tree> for TokenStream {
        fn extend<I: IntoIterator<Item = Tree>>(&mut self, stream: I) {
            match self {
                TokenStream::Compiler(tts) => {
                    // Here is the reason for DeferredTokenStream.
                    for token in stream {
                        tts.extra.push(into_compiler_token(token));
                    }
                },
                TokenStream::Fallback(tts) => tts.extend(stream),
            }
        }
    }
    impl Extend<TokenStream> for TokenStream {
        fn extend<I: IntoIterator<Item = TokenStream>>(&mut self, streams: I) {
            match self {
                TokenStream::Compiler(tts) => {
                    tts.evaluate_now();
                    tts.stream.extend(streams.into_iter().map(TokenStream::unwrap_nightly));
                },
                TokenStream::Fallback(tts) => {
                    tts.extend(streams.into_iter().map(TokenStream::unwrap_stable));
                },
            }
        }
    }
    impl Debug for TokenStream {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                TokenStream::Compiler(tts) => Debug::fmt(&tts.clone().into_token_stream(), f),
                TokenStream::Fallback(tts) => Debug::fmt(tts, f),
            }
        }
    }
    impl LexError {
        pub fn span(&self) -> Span {
            match self {
                LexError::Compiler(_) => Span::call_site(),
                LexError::Fallback(e) => Span::Fallback(e.span()),
            }
        }
    }
    impl From<proc_macro::LexError> for LexError {
        fn from(e: proc_macro::LexError) -> Self {
            LexError::Compiler(e)
        }
    }
    impl From<fallback::LexError> for LexError {
        fn from(e: fallback::LexError) -> Self {
            LexError::Fallback(e)
        }
    }
    impl Debug for LexError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                LexError::Compiler(e) => Debug::fmt(e, f),
                LexError::Fallback(e) => Debug::fmt(e, f),
            }
        }
    }
    impl Display for LexError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                LexError::Compiler(e) => Display::fmt(e, f),
                LexError::Fallback(e) => Display::fmt(e, f),
            }
        }
    }
    #[derive(Clone)]
    pub enum TreeIter {
        Compiler(proc_macro::token_stream::IntoIter),
        Fallback(fallback::TreeIter),
    }
    impl IntoIterator for TokenStream {
        type Item = Tree;
        type IntoIter = TreeIter;
        fn into_iter(self) -> TreeIter {
            match self {
                TokenStream::Compiler(tts) => TreeIter::Compiler(tts.into_token_stream().into_iter()),
                TokenStream::Fallback(tts) => TreeIter::Fallback(tts.into_iter()),
            }
        }
    }
    impl Iterator for TreeIter {
        type Item = Tree;
        fn next(&mut self) -> Option<Tree> {
            let token = match self {
                TreeIter::Compiler(iter) => iter.next()?,
                TreeIter::Fallback(iter) => return iter.next(),
            };
            Some(match token {
                proc_macro::TokenTree::Group(tt) => super::Group::_new(Group::Compiler(tt)).into(),
                proc_macro::TokenTree::Punct(tt) => {
                    let spacing = match tt.spacing() {
                        proc_macro::Spacing::Joint => Spacing::Joint,
                        proc_macro::Spacing::Alone => Spacing::Alone,
                    };
                    let mut o = Punct::new(tt.as_char(), spacing);
                    o.set_span(super::Span::_new(Span::Compiler(tt.span())));
                    o.into()
                },
                proc_macro::TokenTree::Ident(s) => super::Ident::_new(Ident::Compiler(s)).into(),
                proc_macro::TokenTree::Literal(l) => super::Lit::_new(Lit::Compiler(l)).into(),
            })
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            match self {
                TreeIter::Compiler(tts) => tts.size_hint(),
                TreeIter::Fallback(tts) => tts.size_hint(),
            }
        }
    }
    #[derive(Clone)]
    pub enum Ident {
        Compiler(proc_macro::Ident),
        Fallback(fallback::Ident),
    }
    impl Ident {
        pub fn new(string: &str, span: Span) -> Self {
            match span {
                Span::Compiler(s) => Ident::Compiler(proc_macro::Ident::new(string, s)),
                Span::Fallback(s) => Ident::Fallback(fallback::Ident::new(string, s)),
            }
        }
        pub fn new_raw(string: &str, span: Span) -> Self {
            match span {
                Span::Compiler(s) => Ident::Compiler(proc_macro::Ident::new_raw(string, s)),
                Span::Fallback(s) => Ident::Fallback(fallback::Ident::new_raw(string, s)),
            }
        }
        pub fn span(&self) -> Span {
            match self {
                Ident::Compiler(t) => Span::Compiler(t.span()),
                Ident::Fallback(t) => Span::Fallback(t.span()),
            }
        }
        pub fn set_span(&mut self, span: Span) {
            match (self, span) {
                (Ident::Compiler(t), Span::Compiler(s)) => t.set_span(s),
                (Ident::Fallback(t), Span::Fallback(s)) => t.set_span(s),
                _ => mismatch(),
            }
        }
        fn unwrap_nightly(self) -> proc_macro::Ident {
            match self {
                Ident::Compiler(s) => s,
                Ident::Fallback(_) => mismatch(),
            }
        }
    }
    impl PartialEq for Ident {
        fn eq(&self, other: &Ident) -> bool {
            match (self, other) {
                (Ident::Compiler(t), Ident::Compiler(o)) => t.to_string() == o.to_string(),
                (Ident::Fallback(t), Ident::Fallback(o)) => t == o,
                _ => mismatch(),
            }
        }
    }
    impl<T> PartialEq<T> for Ident
    where
        T: ?Sized + AsRef<str>,
    {
        fn eq(&self, other: &T) -> bool {
            let other = other.as_ref();
            match self {
                Ident::Compiler(t) => t.to_string() == other,
                Ident::Fallback(t) => t == other,
            }
        }
    }
    impl Display for Ident {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Ident::Compiler(t) => Display::fmt(t, f),
                Ident::Fallback(t) => Display::fmt(t, f),
            }
        }
    }
    impl Debug for Ident {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Ident::Compiler(t) => Debug::fmt(t, f),
                Ident::Fallback(t) => Debug::fmt(t, f),
            }
        }
    }
}
mod convert {
    pub fn usize_to_u32(u: usize) -> Option<u32> {
        use std::convert::TryFrom;
        u32::try_from(u).ok()
    }
}
mod location {
    use std::cmp::Ordering;
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct LineColumn {
        pub line: usize,
        pub column: usize,
    }
    impl Ord for LineColumn {
        fn cmp(&self, other: &Self) -> Ordering {
            self.line.cmp(&other.line).then(self.column.cmp(&other.column))
        }
    }
    impl PartialOrd for LineColumn {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
}
pub use location::LineColumn;

#[derive(Clone)]
pub struct TokenStream {
    inner: imp::TokenStream,
    _marker: Marker,
}
pub struct LexError {
    inner: imp::LexError,
    _marker: Marker,
}
impl TokenStream {
    fn _new(inner: imp::TokenStream) -> Self {
        TokenStream { inner, _marker: Marker }
    }
    fn _new_fallback(inner: fallback::TokenStream) -> Self {
        TokenStream {
            inner: inner.into(),
            _marker: Marker,
        }
    }
    pub fn new() -> Self {
        TokenStream::_new(imp::TokenStream::new())
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
impl Default for TokenStream {
    fn default() -> Self {
        TokenStream::new()
    }
}
impl FromStr for TokenStream {
    type Err = LexError;
    fn from_str(src: &str) -> Result<TokenStream, LexError> {
        let e = src.parse().map_err(|e| LexError {
            inner: e,
            _marker: Marker,
        })?;
        Ok(TokenStream::_new(e))
    }
}
impl From<proc_macro::TokenStream> for TokenStream {
    fn from(inner: proc_macro::TokenStream) -> Self {
        TokenStream::_new(inner.into())
    }
}
impl From<TokenStream> for proc_macro::TokenStream {
    fn from(inner: TokenStream) -> Self {
        inner.inner.into()
    }
}
impl From<Tree> for TokenStream {
    fn from(token: Tree) -> Self {
        TokenStream::_new(imp::TokenStream::from(token))
    }
}
impl Extend<Tree> for TokenStream {
    fn extend<I: IntoIterator<Item = Tree>>(&mut self, streams: I) {
        self.inner.extend(streams);
    }
}
impl Extend<TokenStream> for TokenStream {
    fn extend<I: IntoIterator<Item = TokenStream>>(&mut self, streams: I) {
        self.inner.extend(streams.into_iter().map(|stream| stream.inner));
    }
}
impl FromIterator<Tree> for TokenStream {
    fn from_iter<I: IntoIterator<Item = Tree>>(streams: I) -> Self {
        TokenStream::_new(streams.into_iter().collect())
    }
}
impl FromIterator<TokenStream> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenStream>>(streams: I) -> Self {
        TokenStream::_new(streams.into_iter().map(|i| i.inner).collect())
    }
}
impl Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.inner, f)
    }
}
impl Debug for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.inner, f)
    }
}
impl LexError {
    pub fn span(&self) -> Span {
        Span::_new(self.inner.span())
    }
}
impl Debug for LexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.inner, f)
    }
}
impl Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.inner, f)
    }
}
impl Error for LexError {}
#[derive(Clone)]
pub enum Tree {
    Group(Group),
    Ident(Ident),
    Punct(Punct),
    Lit(Lit),
}
impl Tree {
    pub fn span(&self) -> Span {
        match self {
            Tree::Group(t) => t.span(),
            Tree::Ident(t) => t.span(),
            Tree::Punct(t) => t.span(),
            Tree::Lit(t) => t.span(),
        }
    }
    pub fn set_span(&mut self, span: Span) {
        match self {
            Tree::Group(t) => t.set_span(span),
            Tree::Ident(t) => t.set_span(span),
            Tree::Punct(t) => t.set_span(span),
            Tree::Lit(t) => t.set_span(span),
        }
    }
}
impl From<Group> for Tree {
    fn from(g: Group) -> Self {
        Tree::Group(g)
    }
}
impl From<Ident> for Tree {
    fn from(g: Ident) -> Self {
        Tree::Ident(g)
    }
}
impl From<Punct> for Tree {
    fn from(g: Punct) -> Self {
        Tree::Punct(g)
    }
}
impl From<Lit> for Tree {
    fn from(g: Lit) -> Self {
        Tree::Lit(g)
    }
}
impl Display for Tree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Tree::Group(t) => Display::fmt(t, f),
            Tree::Ident(t) => Display::fmt(t, f),
            Tree::Punct(t) => Display::fmt(t, f),
            Tree::Lit(t) => Display::fmt(t, f),
        }
    }
}
impl Debug for Tree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Tree::Group(t) => Debug::fmt(t, f),
            Tree::Ident(t) => {
                let mut debug = f.debug_struct("Ident");
                debug.field("sym", &format_args!("{}", t));
                debug_span_field(&mut debug, t.span().inner);
                debug.finish()
            },
            Tree::Punct(t) => Debug::fmt(t, f),
            Tree::Lit(t) => Debug::fmt(t, f),
        }
    }
}
#[derive(Clone)]
pub struct Punct {
    ch: char,
    spacing: Spacing,
    span: Span,
}
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Spacing {
    Alone,
    Joint,
}
impl Punct {
    pub fn new(ch: char, spacing: Spacing) -> Self {
        Punct {
            ch,
            spacing,
            span: Span::call_site(),
        }
    }
    pub fn as_char(&self) -> char {
        self.ch
    }
    pub fn spacing(&self) -> Spacing {
        self.spacing
    }
    pub fn span(&self) -> Span {
        self.span
    }
    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }
}
impl Display for Punct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.ch, f)
    }
}
impl Debug for Punct {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut debug = fmt.debug_struct("Punct");
        debug.field("char", &self.ch);
        debug.field("spacing", &self.spacing);
        debug_span_field(&mut debug, self.span.inner);
        debug.finish()
    }
}

#[derive(Clone)]
pub struct Ident {
    inner: imp::Ident,
    _marker: Marker,
}
impl Ident {
    fn _new(inner: imp::Ident) -> Self {
        Ident { inner, _marker: Marker }
    }
    pub fn new(string: &str, span: Span) -> Self {
        Ident::_new(imp::Ident::new(string, span.inner))
    }
    pub fn new_raw(string: &str, span: Span) -> Self {
        Ident::_new_raw(string, span)
    }
    fn _new_raw(string: &str, span: Span) -> Self {
        Ident::_new(imp::Ident::new_raw(string, span.inner))
    }
    pub fn span(&self) -> Span {
        Span::_new(self.inner.span())
    }
    pub fn set_span(&mut self, span: Span) {
        self.inner.set_span(span.inner);
    }
}
impl PartialEq for Ident {
    fn eq(&self, other: &Ident) -> bool {
        self.inner == other.inner
    }
}
impl<T> PartialEq<T> for Ident
where
    T: ?Sized + AsRef<str>,
{
    fn eq(&self, other: &T) -> bool {
        self.inner == other
    }
}
impl Eq for Ident {}
impl PartialOrd for Ident {
    fn partial_cmp(&self, other: &Ident) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ident {
    fn cmp(&self, other: &Ident) -> Ordering {
        self.to_string().cmp(&other.to_string())
    }
}
impl Hash for Ident {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.to_string().hash(hasher);
    }
}
impl Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.inner, f)
    }
}
impl Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.inner, f)
    }
}

pub mod token_stream {
    use super::*;

    #[derive(Clone)]
    pub struct IntoIter {
        inner: imp::TreeIter,
        _marker: Marker,
    }
    impl Iterator for IntoIter {
        type Item = Tree;
        fn next(&mut self) -> Option<Tree> {
            self.inner.next()
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.inner.size_hint()
        }
    }
    impl Debug for IntoIter {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("TokenStream ")?;
            f.debug_list().entries(self.clone()).finish()
        }
    }
    impl IntoIterator for TokenStream {
        type Item = Tree;
        type IntoIter = IntoIter;
        fn into_iter(self) -> IntoIter {
            IntoIter {
                inner: self.inner.into_iter(),
                _marker: Marker,
            }
        }
    }
}

pub trait IntoSpans<S> {
    fn into_spans(self) -> S;
}
impl IntoSpans<Span> for Span {
    fn into_spans(self) -> Span {
        self
    }
}
impl IntoSpans<[Span; 1]> for Span {
    fn into_spans(self) -> [Span; 1] {
        [self]
    }
}
impl IntoSpans<[Span; 2]> for Span {
    fn into_spans(self) -> [Span; 2] {
        [self, self]
    }
}
impl IntoSpans<[Span; 3]> for Span {
    fn into_spans(self) -> [Span; 3] {
        [self, self, self]
    }
}
impl IntoSpans<[Span; 1]> for [Span; 1] {
    fn into_spans(self) -> [Span; 1] {
        self
    }
}
impl IntoSpans<[Span; 2]> for [Span; 2] {
    fn into_spans(self) -> [Span; 2] {
        self
    }
}
impl IntoSpans<[Span; 3]> for [Span; 3] {
    fn into_spans(self) -> [Span; 3] {
        self
    }
}
impl IntoSpans<DelimSpan> for Span {
    fn into_spans(self) -> DelimSpan {
        let mut y = Group::new(Delim::None, Stream::new());
        y.set_span(self);
        y.delim_span()
    }
}
impl IntoSpans<DelimSpan> for DelimSpan {
    fn into_spans(self) -> DelimSpan {
        self
    }
}
pub use self::{extra::DelimSpan, Delim, TokenStream as Stream};
