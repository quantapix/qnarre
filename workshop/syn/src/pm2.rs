use super::*;
use std::{error::Error, iter::FromIterator, ops::RangeBounds, panic, str::FromStr};

#[derive(Clone)]
pub struct DeferredStream {
    stream: pm::TokenStream,
    extra: Vec<pm::TokenTree>,
}
impl DeferredStream {
    fn new(stream: pm::TokenStream) -> Self {
        DeferredStream {
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
    fn into_pm_stream(mut self) -> pm::TokenStream {
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

#[derive(Copy, Clone)]
pub enum DelimSpan {
    Compiler {
        join: pm::Span,
        open: pm::Span,
        close: pm::Span,
    },
    Fallback(fallback::Span),
}
impl DelimSpan {
    pub fn new(x: &Group) -> Self {
        match x {
            Group::Compiler(x) => DelimSpan::Compiler {
                join: x.span(),
                open: x.span_open(),
                close: x.span_close(),
            },
            Group::Fallback(x) => DelimSpan::Fallback(x.span()),
        }
    }
    pub fn join(&self) -> Span {
        match &self {
            DelimSpan::Compiler { join, .. } => Span::Compiler(*join),
            DelimSpan::Fallback(x) => Span::Fallback(*x),
        }
    }
    pub fn open(&self) -> Span {
        match &self {
            DelimSpan::Compiler { open, .. } => Span::Compiler(*open),
            DelimSpan::Fallback(x) => Span::_new_fallback(x.first_byte()),
        }
    }
    pub fn close(&self) -> Span {
        match &self {
            DelimSpan::Compiler { close, .. } => Span::Compiler(*close),
            DelimSpan::Fallback(x) => Span::_new_fallback(x.last_byte()),
        }
    }
}
impl Debug for DelimSpan {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.join(), f)
    }
}

#[derive(Clone)]
pub enum Group {
    Compiler(pm::Group),
    Fallback(fallback::Group),
}
impl Group {
    pub fn new(x: Delim, s: Stream) -> Self {
        match s {
            Stream::Compiler(y) => {
                let x = match x {
                    Delim::Paren => pm::Delimiter::Parenthesis,
                    Delim::Bracket => pm::Delimiter::Bracket,
                    Delim::Brace => pm::Delimiter::Brace,
                    Delim::None => pm::Delimiter::None,
                };
                Group::Compiler(pm::Group::new(x, y.into_pm_stream()))
            },
            Stream::Fallback(y) => Group::Fallback(fallback::Group::new(x, y)),
        }
    }
    pub fn delim(&self) -> Delim {
        match self {
            Group::Compiler(x) => match x.delimiter() {
                pm::Delimiter::Parenthesis => Delim::Paren,
                pm::Delimiter::Bracket => Delim::Bracket,
                pm::Delimiter::Brace => Delim::Brace,
                pm::Delimiter::None => Delim::None,
            },
            Group::Fallback(x) => x.delim(),
        }
    }
    pub fn stream(&self) -> Stream {
        match self {
            Group::Compiler(x) => Stream::Compiler(DeferredStream::new(x.stream())),
            Group::Fallback(x) => Stream::Fallback(x.stream()),
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
    fn unwrap_nightly(self) -> pm::Group {
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
fn mismatch() -> ! {
    panic!("compiler/fallback mismatch")
}

#[derive(Clone)]
pub enum Ident {
    Compiler(pm::Ident),
    Fallback(fallback::Ident),
}
impl Ident {
    pub fn new(x: &str, s: Span) -> Self {
        match s {
            Span::Compiler(s) => Ident::Compiler(pm::Ident::new(x, s)),
            Span::Fallback(s) => Ident::Fallback(fallback::Ident::new(x, s)),
        }
    }
    pub fn new_raw(x: &str, s: Span) -> Self {
        match s {
            Span::Compiler(s) => Ident::Compiler(pm::Ident::new_raw(x, s)),
            Span::Fallback(s) => Ident::Fallback(fallback::Ident::new_raw(x, s)),
        }
    }
    pub fn span(&self) -> Span {
        match self {
            Ident::Compiler(x) => Span::Compiler(x.span()),
            Ident::Fallback(x) => Span::Fallback(x.span()),
        }
    }
    pub fn set_span(&mut self, s: Span) {
        match (self, s) {
            (Ident::Compiler(x), Span::Compiler(s)) => x.set_span(s),
            (Ident::Fallback(x), Span::Fallback(s)) => x.set_span(s),
            _ => mismatch(),
        }
    }
    fn unwrap_nightly(self) -> pm::Ident {
        match self {
            Ident::Compiler(x) => x,
            Ident::Fallback(_) => mismatch(),
        }
    }
}
impl PartialEq for Ident {
    fn eq(&self, i: &Ident) -> bool {
        match (self, i) {
            (Ident::Compiler(x), Ident::Compiler(i)) => x.to_string() == i.to_string(),
            (Ident::Fallback(x), Ident::Fallback(i)) => x == i,
            _ => mismatch(),
        }
    }
}
impl<T> PartialEq<T> for Ident
where
    T: ?Sized + AsRef<str>,
{
    fn eq(&self, t: &T) -> bool {
        let t = t.as_ref();
        match self {
            Ident::Compiler(x) => x.to_string() == t,
            Ident::Fallback(x) => x == t,
        }
    }
}
impl Eq for Ident {}
impl PartialOrd for Ident {
    fn partial_cmp(&self, x: &Ident) -> Option<Ordering> {
        Some(self.cmp(x))
    }
}
impl Ord for Ident {
    fn cmp(&self, x: &Ident) -> Ordering {
        self.to_string().cmp(&x.to_string())
    }
}
impl Hash for Ident {
    fn hash<H: Hasher>(&self, x: &mut H) {
        self.to_string().hash(x);
    }
}
impl Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Ident::Compiler(x) => Display::fmt(x, f),
            Ident::Fallback(x) => Display::fmt(x, f),
        }
    }
}
impl Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Ident::Compiler(x) => Debug::fmt(x, f),
            Ident::Fallback(x) => Debug::fmt(x, f),
        }
    }
}

pub enum LexErr {
    Compiler(pm::LexError),
    Fallback(fallback::LexErr),
}
impl LexErr {
    pub fn span(&self) -> Span {
        match self {
            LexErr::Compiler(_) => Span::call_site(),
            LexErr::Fallback(x) => Span::Fallback(x.span()),
        }
    }
    fn call_site() -> Self {
        LexErr::Fallback(fallback::LexErr {
            span: fallback::Span::call_site(),
            _marker: util::Marker,
        })
    }
}
impl Error for LexErr {}
impl From<pm::LexError> for LexErr {
    fn from(x: pm::LexError) -> Self {
        LexErr::Compiler(x)
    }
}
impl From<fallback::LexErr> for LexErr {
    fn from(x: fallback::LexErr) -> Self {
        LexErr::Fallback(x)
    }
}
impl Debug for LexErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LexErr::Compiler(x) => Debug::fmt(x, f),
            LexErr::Fallback(x) => Debug::fmt(x, f),
        }
    }
}
impl Display for LexErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LexErr::Compiler(x) => Display::fmt(x, f),
            LexErr::Fallback(x) => Display::fmt(x, f),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LineCol {
    pub line: usize,
    pub col: usize,
}
impl Ord for LineCol {
    fn cmp(&self, x: &Self) -> Ordering {
        self.line.cmp(&x.line).then(self.col.cmp(&x.col))
    }
}
impl PartialOrd for LineCol {
    fn partial_cmp(&self, x: &Self) -> Option<Ordering> {
        Some(self.cmp(x))
    }
}

#[derive(Clone)]
pub enum Lit {
    Compiler(pm::Literal),
    Fallback(fallback::Lit),
}
macro_rules! suffixed_nums {
    ($($n:ident => $kind:ident,)*) => ($(
        pub fn $n(n: $kind) -> Lit {
            if util::inside_pm() {
                Lit::Compiler(pm::Literal::$n(n))
            } else {
                Lit::Fallback(fallback::Lit::$n(n))
            }
        }
    )*)
}
macro_rules! unsuffixed_nums {
    ($($n:ident => $kind:ident,)*) => ($(
        pub fn $n(n: $kind) -> Lit {
            if util::inside_pm() {
                Lit::Compiler(pm::Literal::$n(n))
            } else {
                Lit::Fallback(fallback::Lit::$n(n))
            }
        }
    )*)
}
impl Lit {
    pub unsafe fn from_str_unchecked(x: &str) -> Self {
        if util::inside_pm() {
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
        if util::inside_pm() {
            Lit::Compiler(pm::Literal::f32_unsuffixed(x))
        } else {
            Lit::Fallback(fallback::Lit::f32_unsuffixed(x))
        }
    }
    pub fn f64_unsuffixed(x: f64) -> Lit {
        assert!(x.is_finite());
        if util::inside_pm() {
            Lit::Compiler(pm::Literal::f64_unsuffixed(x))
        } else {
            Lit::Fallback(fallback::Lit::f64_unsuffixed(x))
        }
    }
    pub fn string(x: &str) -> Lit {
        if util::inside_pm() {
            Lit::Compiler(pm::Literal::string(x))
        } else {
            Lit::Fallback(fallback::Lit::string(x))
        }
    }
    pub fn character(x: char) -> Lit {
        if util::inside_pm() {
            Lit::Compiler(pm::Literal::character(x))
        } else {
            Lit::Fallback(fallback::Lit::character(x))
        }
    }
    pub fn byte_string(xs: &[u8]) -> Lit {
        if util::inside_pm() {
            Lit::Compiler(pm::Literal::byte_string(xs))
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
    fn unwrap_nightly(self) -> pm::Literal {
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
    type Err = LexErr;
    fn from_str(x: &str) -> Result<Self, Self::Err> {
        if util::inside_pm() {
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
fn compiler_lit_from_str(x: &str) -> Result<pm::Literal, LexErr> {
    pm::Literal::from_str(x).map_err(LexErr::Compiler)
}

#[derive(Clone)]
pub struct Punct {
    ch: char,
    spacing: Spacing,
    span: Span,
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
    pub fn set_span(&mut self, s: Span) {
        self.span = s;
    }
}
impl Display for Punct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.ch, f)
    }
}
impl Debug for Punct {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut y = fmt.debug_struct("Punct");
        y.field("char", &self.ch);
        y.field("spacing", &self.spacing);
        debug_span_field(&mut y, self.span.inner);
        y.finish()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Spacing {
    Alone,
    Joint,
}

#[derive(Copy, Clone)]
pub enum Span {
    Compiler(pm::Span),
    Fallback(fallback::Span),
}
impl Span {
    pub fn call_site() -> Self {
        if util::inside_pm() {
            Span::Compiler(pm::Span::call_site())
        } else {
            Span::Fallback(fallback::Span::call_site())
        }
    }
    pub fn mixed_site() -> Self {
        if util::inside_pm() {
            Span::Compiler(pm::Span::mixed_site())
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
    pub fn start(&self) -> LineCol {
        match self {
            Span::Compiler(_) => LineCol { line: 0, col: 0 },
            Span::Fallback(x) => x.start(),
        }
    }
    pub fn end(&self) -> LineCol {
        match self {
            Span::Compiler(_) => LineCol { line: 0, col: 0 },
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
    pub fn unstable(self) -> pm::Span {
        self.unwrap()
    }
    pub fn unwrap(self) -> pm::Span {
        match self {
            Span::Compiler(x) => x,
            Span::Fallback(_) => mismatch(),
        }
    }
    fn unwrap_nightly(self) -> pm::Span {
        match self {
            Span::Compiler(x) => x,
            Span::Fallback(_) => mismatch(),
        }
    }
}
impl From<pm::Span> for Span {
    fn from(x: pm::Span) -> Self {
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

#[derive(Clone)]
pub enum Stream {
    Compiler(DeferredStream),
    Fallback(fallback::Stream),
}
impl Stream {
    pub fn new() -> Self {
        if util::inside_pm() {
            Stream::Compiler(DeferredStream::new(pm::TokenStream::new()))
        } else {
            Stream::Fallback(fallback::Stream::new())
        }
    }
    pub fn is_empty(&self) -> bool {
        match self {
            Stream::Compiler(x) => x.is_empty(),
            Stream::Fallback(x) => x.is_empty(),
        }
    }
    fn unwrap_nightly(self) -> pm::TokenStream {
        match self {
            Stream::Compiler(x) => x.into_pm_stream(),
            Stream::Fallback(_) => mismatch(),
        }
    }
    fn unwrap_stable(self) -> fallback::Stream {
        match self {
            Stream::Compiler(_) => mismatch(),
            Stream::Fallback(x) => x,
        }
    }
}
impl Default for Stream {
    fn default() -> Self {
        Stream::new()
    }
}
impl FromStr for Stream {
    type Err = LexErr;
    fn from_str(x: &str) -> Result<Stream, LexErr> {
        if util::inside_pm() {
            Ok(Stream::Compiler(DeferredStream::new(pm_parse(x)?)))
        } else {
            Ok(Stream::Fallback(x.parse()?))
        }
    }
}
impl From<pm::TokenStream> for Stream {
    fn from(x: pm::TokenStream) -> Self {
        Stream::Compiler(DeferredStream::new(x))
    }
}
impl From<fallback::Stream> for Stream {
    fn from(x: fallback::Stream) -> Self {
        Stream::Fallback(x)
    }
}
impl From<Tree> for Stream {
    fn from(x: Tree) -> Self {
        if util::inside_pm() {
            Stream::Compiler(DeferredStream::new(into_pm_tok(x).into()))
        } else {
            Stream::Fallback(x.into())
        }
    }
}
impl FromIterator<Stream> for Stream {
    fn from_iter<I: IntoIterator<Item = Stream>>(xs: I) -> Self {
        let mut xs = xs.into_iter();
        match xs.next() {
            Some(Stream::Compiler(mut first)) => {
                first.evaluate_now();
                first.stream.extend(xs.map(|x| match x {
                    Stream::Compiler(x) => x.into_pm_stream(),
                    Stream::Fallback(_) => mismatch(),
                }));
                Stream::Compiler(first)
            },
            Some(Stream::Fallback(mut first)) => {
                first.extend(xs.map(|x| match x {
                    Stream::Fallback(x) => x,
                    Stream::Compiler(_) => mismatch(),
                }));
                Stream::Fallback(first)
            },
            None => Stream::new(),
        }
    }
}
impl FromIterator<Tree> for Stream {
    fn from_iter<I: IntoIterator<Item = Tree>>(xs: I) -> Self {
        if util::inside_pm() {
            Stream::Compiler(DeferredStream::new(xs.into_iter().map(into_pm_tok).collect()))
        } else {
            Stream::Fallback(xs.into_iter().collect())
        }
    }
}
impl Extend<Stream> for Stream {
    fn extend<I: IntoIterator<Item = Stream>>(&mut self, xs: I) {
        match self {
            Stream::Compiler(x) => {
                x.evaluate_now();
                x.stream.extend(xs.into_iter().map(Stream::unwrap_nightly));
            },
            Stream::Fallback(x) => {
                x.extend(xs.into_iter().map(Stream::unwrap_stable));
            },
        }
    }
}
impl Extend<Tree> for Stream {
    fn extend<I: IntoIterator<Item = Tree>>(&mut self, xs: I) {
        match self {
            Stream::Compiler(y) => {
                for x in xs {
                    y.extra.push(into_pm_tok(x));
                }
            },
            Stream::Fallback(x) => x.extend(xs),
        }
    }
}
impl IntoIterator for Stream {
    type Item = Tree;
    type IntoIter = TreeIter;
    fn into_iter(self) -> TreeIter {
        match self {
            Stream::Compiler(x) => TreeIter::Compiler(x.into_pm_stream().into_iter()),
            Stream::Fallback(x) => TreeIter::Fallback(x.into_iter()),
        }
    }
}
impl Display for Stream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Stream::Compiler(x) => Display::fmt(&x.clone().into_pm_stream(), f),
            Stream::Fallback(x) => Display::fmt(x, f),
        }
    }
}
impl Debug for Stream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Stream::Compiler(x) => Debug::fmt(&x.clone().into_pm_stream(), f),
            Stream::Fallback(x) => Debug::fmt(x, f),
        }
    }
}
impl From<Stream> for pm::TokenStream {
    fn from(x: Stream) -> Self {
        match x {
            Stream::Compiler(x) => x.into_pm_stream(),
            Stream::Fallback(x) => x.to_string().parse().unwrap(),
        }
    }
}
fn pm_parse(x: &str) -> Result<pm::TokenStream, LexErr> {
    let y = panic::catch_unwind(|| x.parse().map_err(LexErr::Compiler));
    y.unwrap_or_else(|_| Err(LexErr::call_site()))
}
fn into_pm_tok(x: Tree) -> pm::TokenTree {
    match x {
        Tree::Group(x) => x.inner.unwrap_nightly().into(),
        Tree::Punct(x) => {
            let spacing = match x.spacing() {
                Spacing::Joint => pm::Spacing::Joint,
                Spacing::Alone => pm::Spacing::Alone,
            };
            let mut y = pm::Punct::new(x.as_char(), spacing);
            y.set_span(x.span().inner.unwrap_nightly());
            y.into()
        },
        Tree::Ident(x) => x.inner.unwrap_nightly().into(),
        Tree::Lit(x) => x.inner.unwrap_nightly().into(),
    }
}

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
            Tree::Group(x) => x.span(),
            Tree::Ident(x) => x.span(),
            Tree::Punct(x) => x.span(),
            Tree::Lit(x) => x.span(),
        }
    }
    pub fn set_span(&mut self, s: Span) {
        match self {
            Tree::Group(x) => x.set_span(s),
            Tree::Ident(x) => x.set_span(s),
            Tree::Punct(x) => x.set_span(s),
            Tree::Lit(x) => x.set_span(s),
        }
    }
}
impl From<Group> for Tree {
    fn from(x: Group) -> Self {
        Tree::Group(x)
    }
}
impl From<Ident> for Tree {
    fn from(x: Ident) -> Self {
        Tree::Ident(x)
    }
}
impl From<Punct> for Tree {
    fn from(x: Punct) -> Self {
        Tree::Punct(x)
    }
}
impl From<Lit> for Tree {
    fn from(x: Lit) -> Self {
        Tree::Lit(x)
    }
}
impl Display for Tree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Tree::Group(x) => Display::fmt(x, f),
            Tree::Ident(x) => Display::fmt(x, f),
            Tree::Punct(x) => Display::fmt(x, f),
            Tree::Lit(x) => Display::fmt(x, f),
        }
    }
}
impl Debug for Tree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Tree::Group(x) => Debug::fmt(x, f),
            Tree::Ident(t) => {
                let mut y = f.debug_struct("Ident");
                y.field("sym", &format_args!("{}", t));
                debug_span_field(&mut y, t.span().inner);
                y.finish()
            },
            Tree::Punct(x) => Debug::fmt(x, f),
            Tree::Lit(x) => Debug::fmt(x, f),
        }
    }
}

#[derive(Clone)]
pub enum TreeIter {
    Compiler(pm::token_stream::IntoIter),
    Fallback(fallback::TreeIter),
}
impl Iterator for TreeIter {
    type Item = Tree;
    fn next(&mut self) -> Option<Tree> {
        let y = match self {
            TreeIter::Compiler(x) => x.next()?,
            TreeIter::Fallback(x) => return x.next(),
        };
        Some(match y {
            pm::TokenTree::Group(x) => Group::Compiler(x).into(),
            pm::TokenTree::Punct(x) => {
                let spacing = match x.spacing() {
                    pm::Spacing::Joint => Spacing::Joint,
                    pm::Spacing::Alone => Spacing::Alone,
                };
                let mut y = Punct::new(x.as_char(), spacing);
                y.set_span(Span::Compiler(x.span()));
                y.into()
            },
            pm::TokenTree::Ident(x) => Ident::Compiler(x).into(),
            pm::TokenTree::Literal(x) => Lit::Compiler(x).into(),
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            TreeIter::Compiler(x) => x.size_hint(),
            TreeIter::Fallback(x) => x.size_hint(),
        }
    }
}

#[derive(Clone)]
pub struct IntoIter {
    inner: TreeIter,
    _marker: util::Marker,
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
impl IntoIterator for Stream {
    type Item = Tree;
    type IntoIter = IntoIter;
    fn into_iter(self) -> IntoIter {
        let inner = match self {
            Stream::Compiler(x) => x.into_pm_stream().into_iter(),
            Stream::Fallback(x) => x.into_iter(),
        };
        IntoIter {
            inner,
            _marker: util::Marker,
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

pub mod fallback {
    use super::{
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
        util::force_fallback();
    }
    pub fn unforce() {
        util::unforce_fallback();
    }

    pub struct StreamBuilder {
        inner: RcVecBuilder<Tree>,
    }
    impl StreamBuilder {
        pub fn new() -> Self {
            StreamBuilder {
                inner: RcVecBuilder::new(),
            }
        }
        pub fn with_capacity(x: usize) -> Self {
            StreamBuilder {
                inner: RcVecBuilder::with_capacity(x),
            }
        }
        pub fn push_tok_from_parser(&mut self, x: Tree) {
            self.inner.push(x);
        }
        pub fn build(self) -> Stream {
            Stream {
                inner: self.inner.build(),
                _marker: util::Marker,
            }
        }
    }

    pub type TreeIter = RcVecIntoIter<Tree>;

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
        fn offset_line_column(&self, offset: usize) -> LineCol {
            assert!(self.span_within(Span {
                lo: offset as u32,
                hi: offset as u32,
                _marker: util::Marker
            }));
            let offset = offset - self.span.lo as usize;
            match self.lines.binary_search(&offset) {
                Ok(found) => LineCol {
                    line: found + 1,
                    col: 0,
                },
                Err(idx) => LineCol {
                    line: idx,
                    col: offset - self.lines[idx - 1],
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
                _marker: util::Marker,
            };
            self.files.push(FileInfo {
                text: src.to_owned(),
                span,
                lines,
            });
            span
        }
        fn fileinfo(&self, s: Span) -> &FileInfo {
            for x in &self.files {
                if x.span_within(s) {
                    return x;
                }
            }
            unreachable!("Invalid span with no related FileInfo!");
        }
    }

    #[derive(Clone)]
    pub struct Group {
        delim: Delim,
        stream: Stream,
        span: Span,
    }
    impl Group {
        pub fn new(delim: Delim, stream: Stream) -> Self {
            Group {
                delim,
                stream,
                span: Span::call_site(),
            }
        }
        pub fn delim(&self) -> Delim {
            self.delim
        }
        pub fn stream(&self) -> Stream {
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
    pub struct Ident {
        sym: String,
        span: Span,
        raw: bool,
        _marker: util::Marker,
    }
    impl Ident {
        pub fn new(x: &str, s: Span) -> Self {
            Ident::_new(x, false, s)
        }
        pub fn new_raw(x: &str, s: Span) -> Self {
            Ident::_new(x, true, s)
        }
        fn _new(x: &str, raw: bool, span: Span) -> Self {
            validate_ident(x, raw);
            Ident {
                sym: x.to_owned(),
                span,
                raw,
                _marker: util::Marker,
            }
        }
        pub fn span(&self) -> Span {
            self.span
        }
        pub fn set_span(&mut self, x: Span) {
            self.span = x;
        }
    }
    impl PartialEq for Ident {
        fn eq(&self, x: &Ident) -> bool {
            self.sym == x.sym && self.raw == x.raw
        }
    }
    impl<T> PartialEq<T> for Ident
    where
        T: ?Sized + AsRef<str>,
    {
        fn eq(&self, t: &T) -> bool {
            let t = t.as_ref();
            if self.raw {
                t.starts_with("r#") && self.sym == t[2..]
            } else {
                self.sym == t
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
    pub fn is_ident_start(x: char) -> bool {
        x == '_' || unicode_ident::is_xid_start(x)
    }
    pub fn is_ident_cont(x: char) -> bool {
        unicode_ident::is_xid_continue(x)
    }
    fn validate_ident(x: &str, raw: bool) {
        if x.is_empty() {
            panic!("Ident is not allowed to be empty; use Option<Ident>");
        }
        if x.bytes().all(|x| x >= b'0' && x <= b'9') {
            panic!("Ident cannot be a number; use Literal instead");
        }
        fn ident_ok(x: &str) -> bool {
            let mut xs = x.chars();
            let first = xs.next().unwrap();
            if !is_ident_start(first) {
                return false;
            }
            for x in xs {
                if !is_ident_cont(x) {
                    return false;
                }
            }
            true
        }
        if !ident_ok(x) {
            panic!("{:?} is not a valid Ident", x);
        }
        if raw {
            match x {
                "_" | "super" | "self" | "Self" | "crate" => {
                    panic!("`r#{}` cannot be a raw identifier", x);
                },
                _ => {},
            }
        }
    }

    #[derive(Debug)]
    pub struct LexErr {
        pub span: Span,
        _marker: util::Marker,
    }
    impl LexErr {
        pub fn span(&self) -> Span {
            self.span
        }
        fn call_site() -> Self {
            LexErr {
                span: Span::call_site(),
                _marker: util::Marker,
            }
        }
    }
    impl Display for LexErr {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("cannot parse string into token stream")
        }
    }

    #[derive(Clone)]
    pub struct Lit {
        repr: String,
        span: Span,
        _marker: util::Marker,
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
                _marker: util::Marker,
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
            use super::util::usize_to_u32;
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
                Some(Span {
                    lo,
                    hi,
                    _marker: util::Marker,
                })
            } else {
                None
            }
        }
    }
    impl FromStr for Lit {
        type Err = LexErr;
        fn from_str(repr: &str) -> Result<Self, Self::Err> {
            let mut cursor = get_cursor(repr);
            let lo = cursor.off;
            let negative = cursor.starts_with_char('-');
            if negative {
                cursor = cursor.advance(1);
                if !cursor.starts_with_fn(|ch| ch.is_ascii_digit()) {
                    return Err(LexErr::call_site());
                }
            }
            if let Ok((rest, mut literal)) = parse::literal(cursor) {
                if rest.is_empty() {
                    if negative {
                        literal.repr.insert(0, '-');
                    }
                    literal.span = Span {
                        lo,
                        hi: rest.off,
                        _marker: util::Marker,
                    };
                    return Ok(literal);
                }
            }
            Err(LexErr::call_site())
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
        _marker: util::Marker,
    }
    impl Span {
        pub fn call_site() -> Self {
            Span {
                lo: 0,
                hi: 0,
                _marker: util::Marker,
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
        pub fn start(&self) -> LineCol {
            SRC_MAP.with(|x| {
                let x = x.borrow();
                let y = x.fileinfo(*self);
                y.offset_line_column(self.lo as usize)
            })
        }
        pub fn end(&self) -> LineCol {
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
                    _marker: util::Marker,
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
                _marker: util::Marker,
            }
        }
        pub fn last_byte(self) -> Self {
            Span {
                lo: cmp::max(self.hi.saturating_sub(1), self.lo),
                hi: self.hi,
                _marker: util::Marker,
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

    #[derive(Clone)]
    pub struct Stream {
        inner: RcVec<Tree>,
        _marker: util::Marker,
    }
    impl Stream {
        pub fn new() -> Self {
            Stream {
                inner: RcVecBuilder::new().build(),
                _marker: util::Marker,
            }
        }
        pub fn is_empty(&self) -> bool {
            self.inner.len() == 0
        }
        fn take_inner(self) -> RcVecBuilder<Tree> {
            let y = ManuallyDrop::new(self);
            unsafe { ptr::read(&y.inner) }.make_owned()
        }
    }
    impl Drop for Stream {
        fn drop(&mut self) {
            let mut ys = match self.inner.get_mut() {
                Some(x) => x,
                None => return,
            };
            while let Some(x) = ys.pop() {
                let y = match x {
                    Tree::Group(x) => x.inner,
                    _ => continue,
                };
                let y = match y {
                    super::Group::Compiler(_) => continue,
                    super::Group::Fallback(x) => x,
                };
                ys.extend(y.stream.take_inner());
            }
        }
    }
    impl FromStr for Stream {
        type Err = LexErr;
        fn from_str(x: &str) -> Result<Stream, LexErr> {
            let mut y = get_cursor(x);
            const BYTE_ORDER_MARK: &str = "\u{feff}";
            if y.starts_with(BYTE_ORDER_MARK) {
                y = y.advance(BYTE_ORDER_MARK.len());
            }
            parse::token_stream(y)
        }
    }
    impl From<pm::TokenStream> for Stream {
        fn from(x: pm::TokenStream) -> Self {
            x.to_string().parse().expect("compiler token stream parse failed")
        }
    }
    impl From<Tree> for Stream {
        fn from(x: Tree) -> Self {
            let mut y = RcVecBuilder::new();
            push_tok_from_pm(y.as_mut(), x);
            Stream {
                inner: y.build(),
                _marker: util::Marker,
            }
        }
    }
    impl FromIterator<Stream> for Stream {
        fn from_iter<I: IntoIterator<Item = Stream>>(xs: I) -> Self {
            let mut y = RcVecBuilder::new();
            for x in xs {
                y.extend(x.take_inner());
            }
            Stream {
                inner: y.build(),
                _marker: util::Marker,
            }
        }
    }
    impl FromIterator<Tree> for Stream {
        fn from_iter<I: IntoIterator<Item = Tree>>(xs: I) -> Self {
            let mut ys = Stream::new();
            ys.extend(xs);
            ys
        }
    }
    impl Extend<Stream> for Stream {
        fn extend<I: IntoIterator<Item = Stream>>(&mut self, xs: I) {
            self.inner.make_mut().extend(xs.into_iter().flatten());
        }
    }
    impl Extend<Tree> for Stream {
        fn extend<I: IntoIterator<Item = Tree>>(&mut self, xs: I) {
            let mut y = self.inner.make_mut();
            xs.into_iter().for_each(|x| push_tok_from_pm(y.as_mut(), x));
        }
    }
    impl IntoIterator for Stream {
        type Item = Tree;
        type IntoIter = TreeIter;
        fn into_iter(self) -> TreeIter {
            self.take_inner().into_iter()
        }
    }
    impl Display for Stream {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut joint = false;
            for (i, x) in self.inner.iter().enumerate() {
                if i != 0 && !joint {
                    write!(f, " ")?;
                }
                joint = false;
                match x {
                    Tree::Group(x) => Display::fmt(x, f),
                    Tree::Ident(x) => Display::fmt(x, f),
                    Tree::Punct(x) => {
                        joint = x.spacing() == Spacing::Joint;
                        Display::fmt(x, f)
                    },
                    Tree::Lit(x) => Display::fmt(x, f),
                }?;
            }
            Ok(())
        }
    }
    impl Debug for Stream {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("TokenStream ")?;
            f.debug_list().entries(self.clone()).finish()
        }
    }
    impl From<Stream> for pm::TokenStream {
        fn from(x: Stream) -> Self {
            x.to_string().parse().expect("failed to parse to compiler tokens")
        }
    }
    fn push_tok_from_pm(mut ys: RcVecMut<Tree>, x: Tree) {
        #[cold]
        fn push_neg_lit(mut ys: RcVecMut<Tree>, mut x: Lit) {
            x.repr.remove(0);
            let mut y = super::Punct::new('-', Spacing::Alone);
            y.set_span(super::Span::_new_fallback(x.span));
            ys.push(Tree::Punct(y));
            ys.push(Tree::Lit(super::Lit::_new_fallback(x)));
        }
        match x {
            Tree::Lit(super::fallback::Lit {
                inner: super::Lit::Fallback(x),
                ..
            }) if x.repr.starts_with('-') => {
                push_neg_lit(ys, x);
            },
            _ => ys.push(x),
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
}
mod parse {
    use super::{
        fallback::{is_ident_cont, is_ident_start, Group, LexErr, Lit, Span, Stream, StreamBuilder},
        *,
    };
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
            Some(ch) if is_ident_cont(ch) => Err(Reject),
            Some(_) | None => Ok(input),
        }
    }
    pub fn token_stream(mut input: Cursor) -> Result<Stream, LexErr> {
        let mut trees = StreamBuilder::new();
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
                        return Err(LexErr {
                            span: Span {
                                lo: *lo,
                                hi: *lo,
                                _marker: util::Marker,
                            },
                            _marker: util::Marker,
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
                trees = StreamBuilder::new();
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
                g.set_span(Span {
                    lo,
                    hi: input.off,
                    _marker: util::Marker,
                });
                trees = outer;
                trees.push_tok_from_parser(Tree::Group(super::Group::_new_fallback(g)));
            } else {
                let (rest, mut tt) = match leaf_token(input) {
                    Ok((rest, tt)) => (rest, tt),
                    Err(Reject) => return Err(lex_error(input)),
                };
                tt.set_span(super::Span::Fallback(Span {
                    lo,
                    hi: rest.off,
                    _marker: util::Marker,
                }));
                trees.push_tok_from_parser(tt);
                input = rest;
            }
        }
    }
    fn lex_error(x: Cursor) -> LexErr {
        LexErr {
            span: Span {
                lo: x.off,
                hi: x.off,
                _marker: util::Marker,
            },
            _marker: util::Marker,
        }
    }
    fn leaf_token(x: Cursor) -> PResult<Tree> {
        if let Ok((x, l)) = literal(x) {
            Ok((x, Tree::Lit(super::Lit::_new_fallback(l))))
        } else if let Ok((x, p)) = punct(x) {
            Ok((x, Tree::Punct(p)))
        } else if let Ok((x, i)) = ident(x) {
            Ok((x, Tree::Ident(i)))
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
            if !is_ident_cont(ch) {
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
    fn doc_comment<'a>(input: Cursor<'a>, trees: &mut StreamBuilder) -> PResult<'a, ()> {
        let lo = input.off;
        let (rest, (comment, inner)) = doc_comment_contents(input)?;
        let span = super::Span::Fallback(Span {
            lo,
            hi: rest.off,
            _marker: util::Marker,
        });
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
        trees.push_tok_from_parser(Tree::Punct(pound));
        if inner {
            let mut bang = Punct::new('!', Spacing::Alone);
            bang.set_span(span);
            trees.push_tok_from_parser(Tree::Punct(bang));
        }
        let doc_ident = super::Ident::new("doc", span);
        let mut equal = Punct::new('=', Spacing::Alone);
        equal.set_span(span);
        let mut literal = super::Lit::string(comment);
        literal.set_span(span);
        let mut bracketed = StreamBuilder::with_capacity(3);
        bracketed.push_tok_from_parser(Tree::Ident(doc_ident));
        bracketed.push_tok_from_parser(Tree::Punct(equal));
        bracketed.push_tok_from_parser(Tree::Lit(literal));
        let group = Group::new(Delim::Bracket, bracketed.build());
        let mut group = super::Group::_new_fallback(group);
        group.set_span(span);
        trees.push_tok_from_parser(Tree::Group(group));
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
mod util {
    use std::{
        marker::PhantomData,
        panic::{RefUnwindSafe, UnwindSafe},
        rc::Rc,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Once,
        },
    };

    pub struct AutoTraits(Rc<()>);
    impl UnwindSafe for AutoTraits {}
    impl RefUnwindSafe for AutoTraits {}

    pub type Marker = PhantomData<AutoTraits>;
    mod value {
        pub use std::marker::PhantomData as Marker;
    }
    pub use value::*;

    static FLAG: AtomicUsize = AtomicUsize::new(0);
    static INIT: Once = Once::new();
    pub fn inside_pm() -> bool {
        match FLAG.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {},
        }
        INIT.call_once(init);
        inside_pm()
    }
    pub fn force_fallback() {
        FLAG.store(1, Ordering::Relaxed);
    }
    pub fn unforce_fallback() {
        init();
    }
    fn init() {
        let y = pm::is_available();
        FLAG.store(y as usize + 1, Ordering::Relaxed);
    }

    pub fn usize_to_u32(x: usize) -> Option<u32> {
        u32::try_from(x).ok()
    }
}
