use super::*;
use std::{iter::FromIterator, ops::RangeBounds, panic, str::FromStr};

#[derive(Clone)]
pub struct Deferred {
    stream: pm::TokenStream,
    trees: Vec<pm::TokenTree>,
}
impl Deferred {
    fn new(stream: pm::TokenStream) -> Self {
        Deferred {
            stream,
            trees: Vec::new(),
        }
    }
    fn is_empty(&self) -> bool {
        self.stream.is_empty() && self.trees.is_empty()
    }
    fn evaluate(&mut self) {
        if !self.trees.is_empty() {
            self.stream.extend(self.trees.drain(..));
        }
    }
    fn into_stream(mut self) -> pm::TokenStream {
        self.evaluate();
        self.stream
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Delim {
    Brace,
    Bracket,
    Parenth,
    None,
}

#[derive(Copy, Clone, Debug)]
pub enum DelimSpan {
    Compiler {
        join: pm::Span,
        open: pm::Span,
        close: pm::Span,
    },
    Fallback(imp::Span),
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
            DelimSpan::Fallback(x) => Span::Fallback(x.first_byte()),
        }
    }
    pub fn close(&self) -> Span {
        match &self {
            DelimSpan::Compiler { close, .. } => Span::Compiler(*close),
            DelimSpan::Fallback(x) => Span::Fallback(x.last_byte()),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Group {
    Compiler(pm::Group),
    Fallback(imp::Group),
}
impl Group {
    pub fn new(x: Delim, s: Stream) -> Self {
        match s {
            Stream::Compiler(y) => {
                let x = match x {
                    Delim::Parenth => pm::Delimiter::Parenthesis,
                    Delim::Bracket => pm::Delimiter::Bracket,
                    Delim::Brace => pm::Delimiter::Brace,
                    Delim::None => pm::Delimiter::None,
                };
                Group::Compiler(pm::Group::new(x, y.into_stream()))
            },
            Stream::Fallback(y) => Group::Fallback(imp::Group::new(x, y)),
        }
    }
    pub fn delim(&self) -> Delim {
        match self {
            Group::Compiler(x) => match x.delimiter() {
                pm::Delimiter::Parenthesis => Delim::Parenth,
                pm::Delimiter::Bracket => Delim::Bracket,
                pm::Delimiter::Brace => Delim::Brace,
                pm::Delimiter::None => Delim::None,
            },
            Group::Fallback(x) => x.delim(),
        }
    }
    pub fn stream(&self) -> Stream {
        match self {
            Group::Compiler(x) => Stream::Compiler(Deferred::new(x.stream())),
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
impl From<imp::Group> for Group {
    fn from(x: imp::Group) -> Self {
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
fn mismatch() -> ! {
    panic!("compiler/fallback mismatch")
}

#[derive(Clone, Debug, Eq, Hash)]
pub enum Ident {
    Compiler(pm::Ident),
    Fallback(imp::Ident),
}
impl Ident {
    pub fn new(x: &str, s: Span) -> Self {
        match s {
            Span::Compiler(s) => Ident::Compiler(pm::Ident::new(x, s)),
            Span::Fallback(s) => Ident::Fallback(imp::Ident::new(x, s)),
        }
    }
    pub fn new_raw(x: &str, s: Span) -> Self {
        match s {
            Span::Compiler(s) => Ident::Compiler(pm::Ident::new_raw(x, s)),
            Span::Fallback(s) => Ident::Fallback(imp::Ident::new_raw(x, s)),
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
impl Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Ident::Compiler(x) => Display::fmt(x, f),
            Ident::Fallback(x) => Display::fmt(x, f),
        }
    }
}

#[derive(Debug)]
pub enum LexErr {
    Compiler(pm::LexError),
    Fallback(imp::LexErr),
}
impl LexErr {
    pub fn span(&self) -> Span {
        match self {
            LexErr::Compiler(_) => Span::call_site(),
            LexErr::Fallback(x) => Span::Fallback(x.span()),
        }
    }
    fn call_site() -> Self {
        LexErr::Fallback(imp::LexErr {
            span: imp::Span::call_site(),
            _marker: util::Marker,
        })
    }
}
impl std::error::Error for LexErr {}
impl From<pm::LexError> for LexErr {
    fn from(x: pm::LexError) -> Self {
        LexErr::Compiler(x)
    }
}
impl From<imp::LexErr> for LexErr {
    fn from(x: imp::LexErr) -> Self {
        LexErr::Fallback(x)
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

#[derive(Clone, Debug)]
pub enum Lit {
    Compiler(pm::Literal),
    Fallback(imp::Lit),
}
macro_rules! suffixed_nums {
    ($($n:ident => $kind:ident,)*) => ($(
        pub fn $n(n: $kind) -> Lit {
            if util::inside_pm() {
                Lit::Compiler(pm::Literal::$n(n))
            } else {
                Lit::Fallback(imp::Lit::$n(n))
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
                Lit::Fallback(imp::Lit::$n(n))
            }
        }
    )*)
}
impl Lit {
    pub unsafe fn from_str_unchecked(x: &str) -> Self {
        if util::inside_pm() {
            Lit::Compiler(compiler_lit_from_str(x).expect("invalid literal"))
        } else {
            Lit::Fallback(imp::Lit::from_str_unchecked(x))
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
        Lit::Fallback(imp::Lit::f32_suffixed(x))
    }
    pub fn f64_suffixed(x: f64) -> Lit {
        assert!(x.is_finite());
        Lit::Fallback(imp::Lit::f64_suffixed(x))
    }
    pub fn f32_unsuffixed(x: f32) -> Lit {
        assert!(x.is_finite());
        if util::inside_pm() {
            Lit::Compiler(pm::Literal::f32_unsuffixed(x))
        } else {
            Lit::Fallback(imp::Lit::f32_unsuffixed(x))
        }
    }
    pub fn f64_unsuffixed(x: f64) -> Lit {
        assert!(x.is_finite());
        if util::inside_pm() {
            Lit::Compiler(pm::Literal::f64_unsuffixed(x))
        } else {
            Lit::Fallback(imp::Lit::f64_unsuffixed(x))
        }
    }
    pub fn char(x: char) -> Lit {
        if util::inside_pm() {
            Lit::Compiler(pm::Literal::character(x))
        } else {
            Lit::Fallback(imp::Lit::char(x))
        }
    }
    pub fn byte_string(xs: &[u8]) -> Lit {
        if util::inside_pm() {
            Lit::Compiler(pm::Literal::byte_string(xs))
        } else {
            Lit::Fallback(imp::Lit::byte_string(xs))
        }
    }
    pub fn string(x: &str) -> Lit {
        if util::inside_pm() {
            Lit::Compiler(pm::Literal::string(x))
        } else {
            Lit::Fallback(imp::Lit::string(x))
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
impl From<imp::Lit> for Lit {
    fn from(x: imp::Lit) -> Self {
        Lit::Fallback(x)
    }
}
impl FromStr for Lit {
    type Err = LexErr;
    fn from_str(x: &str) -> Result<Self, Self::Err> {
        if util::inside_pm() {
            compiler_lit_from_str(x).map(Lit::Compiler)
        } else {
            let y = imp::Lit::from_str(x)?;
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
impl Pretty for Lit {
    fn pretty(&self, p: &mut Print) {
        p.word(self.to_string());
    }
}
fn compiler_lit_from_str(x: &str) -> Result<pm::Literal, LexErr> {
    pm::Literal::from_str(x).map_err(LexErr::Compiler)
}

#[derive(Clone, Debug)]
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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Spacing {
    Alone,
    Joint,
}

#[derive(Copy, Clone, Debug)]
pub enum Span {
    Compiler(pm::Span),
    Fallback(imp::Span),
}
impl Span {
    pub fn call_site() -> Self {
        if util::inside_pm() {
            Span::Compiler(pm::Span::call_site())
        } else {
            Span::Fallback(imp::Span::call_site())
        }
    }
    pub fn mixed_site() -> Self {
        if util::inside_pm() {
            Span::Compiler(pm::Span::mixed_site())
        } else {
            Span::Fallback(imp::Span::mixed_site())
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
impl From<imp::Span> for Span {
    fn from(x: imp::Span) -> Self {
        Span::Fallback(x)
    }
}
impl<F: Folder + ?Sized> Fold for Span {
    fn fold(&self, f: &mut F) {
        self
    }
}
impl<V: Visitor + ?Sized> Visit for Span {
    fn visit(&self, v: &mut V) {}
    fn visit_mut(&mut self, v: &mut V) {}
}

#[derive(Clone, Debug, Eq, Hash)]
pub enum Stream {
    Compiler(Deferred),
    Fallback(imp::Stream),
}
impl Stream {
    pub fn new() -> Self {
        if util::inside_pm() {
            Stream::Compiler(Deferred::new(pm::TokenStream::new()))
        } else {
            Stream::Fallback(imp::Stream::new())
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
            Stream::Compiler(x) => x.into_stream(),
            Stream::Fallback(_) => mismatch(),
        }
    }
    fn unwrap_stable(self) -> imp::Stream {
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
            Ok(Stream::Compiler(Deferred::new(pm_parse(x)?)))
        } else {
            Ok(Stream::Fallback(x.parse()?))
        }
    }
}
impl From<pm::TokenStream> for Stream {
    fn from(x: pm::TokenStream) -> Self {
        Stream::Compiler(Deferred::new(x))
    }
}
impl From<imp::Stream> for Stream {
    fn from(x: imp::Stream) -> Self {
        Stream::Fallback(x)
    }
}
impl From<Tree> for Stream {
    fn from(x: Tree) -> Self {
        if util::inside_pm() {
            Stream::Compiler(Deferred::new(into_pm_tok(x).into()))
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
                first.evaluate();
                first.stream.extend(xs.map(|x| match x {
                    Stream::Compiler(x) => x.into_stream(),
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
            Stream::Compiler(Deferred::new(xs.into_iter().map(into_pm_tok).collect()))
        } else {
            Stream::Fallback(xs.into_iter().collect())
        }
    }
}
impl Extend<Stream> for Stream {
    fn extend<I: IntoIterator<Item = Stream>>(&mut self, xs: I) {
        match self {
            Stream::Compiler(x) => {
                x.evaluate();
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
                    y.trees.push(into_pm_tok(x));
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
            Stream::Compiler(x) => TreeIter::Compiler(x.into_stream().into_iter()),
            Stream::Fallback(x) => TreeIter::Fallback(x.into_iter()),
        }
    }
}
impl Display for Stream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Stream::Compiler(x) => Display::fmt(&x.clone().into_stream(), f),
            Stream::Fallback(x) => Display::fmt(x, f),
        }
    }
}
impl From<Stream> for pm::TokenStream {
    fn from(x: Stream) -> Self {
        match x {
            Stream::Compiler(x) => x.into_stream(),
            Stream::Fallback(x) => x.to_string().parse().unwrap(),
        }
    }
}
impl PartialEq for Stream {
    fn eq(&self, x: &Self) -> bool {
        let left = self.0.clone().into_iter().collect::<Vec<_>>();
        let right = x.0.clone().into_iter().collect::<Vec<_>>();
        if left.len() != right.len() {
            return false;
        }
        for (a, b) in left.into_iter().zip(right) {
            if TreeHelper(&a) != TreeHelper(&b) {
                return false;
            }
        }
        true
    }
}

fn pm_parse(x: &str) -> Result<pm::TokenStream, LexErr> {
    let y = panic::catch_unwind(|| x.parse().map_err(LexErr::Compiler));
    y.unwrap_or_else(|_| Err(LexErr::call_site()))
}
fn into_pm_tok(x: Tree) -> pm::TokenTree {
    match x {
        Tree::Group(x) => x.unwrap_nightly().into(),
        Tree::Punct(x) => {
            let spacing = match x.spacing() {
                Spacing::Joint => pm::Spacing::Joint,
                Spacing::Alone => pm::Spacing::Alone,
            };
            let mut y = pm::Punct::new(x.as_char(), spacing);
            y.set_span(x.span().unwrap_nightly());
            y.into()
        },
        Tree::Ident(x) => x.unwrap_nightly().into(),
        Tree::Lit(x) => x.unwrap_nightly().into(),
    }
}

#[derive(Clone, Debug)]
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

#[derive(Clone)]
pub enum TreeIter {
    Compiler(pm::token_stream::IntoIter),
    Fallback(imp::TreeIter),
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

#[derive(Clone, Debug)]
pub struct IntoIter {
    iter: TreeIter,
    _marker: util::Marker,
}
impl Iterator for IntoIter {
    type Item = Tree;
    fn next(&mut self) -> Option<Tree> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
impl IntoIterator for Stream {
    type Item = Tree;
    type IntoIter = IntoIter;
    fn into_iter(self) -> IntoIter {
        let iter = match self {
            Stream::Compiler(x) => x.into_stream().into_iter(),
            Stream::Fallback(x) => x.into_iter(),
        };
        IntoIter {
            iter,
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

mod imp {
    use super::{
        rcvec::{RcVec, RcVecMut},
        *,
    };
    use std::{
        cell::RefCell,
        char, cmp,
        fmt::{self, Debug, Display, Write},
        iter::FromIterator,
        mem::ManuallyDrop,
        ops::RangeBounds,
        path::PathBuf,
        str::{Bytes, CharIndices, Chars},
    };

    pub struct Builder {
        inner: rcvec::Builder<Tree>,
    }
    impl Builder {
        pub fn new() -> Self {
            Builder {
                inner: rcvec::Builder::new(),
            }
        }
        pub fn with_capacity(x: usize) -> Self {
            Builder {
                inner: rcvec::Builder::with_capacity(x),
            }
        }
        pub fn push_from_parser(&mut self, x: Tree) {
            self.inner.push(x);
        }
        pub fn build(self) -> Stream {
            Stream {
                trees: self.inner.build(),
                _marker: util::Marker,
            }
        }
    }

    #[derive(Copy, Clone, Eq, PartialEq)]
    pub struct Cursor<'a> {
        pub rest: &'a str,
        pub off: u32,
    }
    impl<'a> Cursor<'a> {
        pub fn is_empty(&self) -> bool {
            self.rest.is_empty()
        }
        pub fn starts_with(&self, x: &str) -> bool {
            self.rest.starts_with(x)
        }
        pub fn starts_with_char(&self, x: char) -> bool {
            self.rest.starts_with(x)
        }
        pub fn starts_with_fn<T>(&self, f: T) -> bool
        where
            T: FnMut(char) -> bool,
        {
            self.rest.starts_with(f)
        }
        pub fn advance(&self, off: usize) -> Cursor<'a> {
            let (y, rest) = self.rest.split_at(off);
            Cursor {
                rest,
                off: self.off + y.chars().count() as u32,
            }
        }
        pub fn to_nl_or_eof(&self) -> (Cursor<'a>, &str) {
            let xs = self.char_indices();
            for (i, x) in xs {
                if x == '\n' {
                    return (self.advance(i), &self.rest[..i]);
                } else if x == '\r' && self.rest[i + 1..].starts_with('\n') {
                    return (self.advance(i + 1), &self.rest[..i]);
                }
            }
            (self.advance(self.len()), self.rest)
        }
        pub fn comment(&self) -> Res<&str> {
            if !self.starts_with("/*") {
                return Err(Rej);
            }
            let mut depth = 0usize;
            let xs = self.as_bytes();
            let mut i = 0usize;
            let upper = xs.len() - 1;
            while i < upper {
                if xs[i] == b'/' && xs[i + 1] == b'*' {
                    depth += 1;
                    i += 1;
                } else if xs[i] == b'*' && xs[i + 1] == b'/' {
                    depth -= 1;
                    if depth == 0 {
                        return Ok((self.advance(i + 2), &self.rest[..i + 2]));
                    }
                    i += 1;
                }
                i += 1;
            }
            Err(Rej)
        }
        fn ident(&self) -> Res<super::Ident> {
            if ["r\"", "r#\"", "r##", "b\"", "b\'", "br\"", "br#", "c\"", "cr\"", "cr#"]
                .iter()
                .any(|x| self.starts_with(x))
            {
                Err(Rej)
            } else {
                self.ident_any()
            }
        }
        fn ident_any(&self) -> Res<super::Ident> {
            let raw = self.starts_with("r#");
            let x = self.advance((raw as usize) << 1);
            let (x, sym) = x.ident_not_raw()?;
            if !raw {
                let y = super::Ident::new(sym, super::Span::call_site());
                return Ok((x, y));
            }
            match sym {
                "_" | "super" | "self" | "Self" | "crate" => return Err(Rej),
                _ => {},
            }
            let y = super::Ident::new_raw(sym, super::Span::call_site());
            Ok((x, y))
        }
        pub fn lit(&self) -> Res<Lit> {
            fn parse(c: Cursor) -> Result<Cursor, Rej> {
                if let Ok(x) = c.string() {
                    Ok(x)
                } else if let Ok(x) = c.byte_string() {
                    Ok(x)
                } else if let Ok(x) = c.c_string() {
                    Ok(x)
                } else if let Ok(x) = c.byte() {
                    Ok(x)
                } else if let Ok(x) = c.char() {
                    Ok(x)
                } else if let Ok(x) = c.float() {
                    Ok(x)
                } else if let Ok(x) = c.int() {
                    Ok(x)
                } else {
                    Err(Rej)
                }
            }
            let y = parse(self)?;
            let end = self.len() - y.len();
            Ok((y, Lit::new(self.rest[..end].to_string())))
        }
        pub fn lex_err(&self) -> LexErr {
            LexErr {
                span: Span {
                    lo: self.off,
                    hi: self.off,
                    _marker: util::Marker,
                },
                _marker: util::Marker,
            }
        }
        fn float(&self) -> Result<Cursor<'a>, Rej> {
            fn parse(c: Cursor) -> Result<Cursor, Rej> {
                let mut xs = c.chars().peekable();
                match xs.next() {
                    Some(x) if x >= '0' && x <= '9' => {},
                    _ => return Err(Rej),
                }
                let mut len = 1;
                let mut has_dot = false;
                let mut has_exp = false;
                while let Some(&x) = xs.peek() {
                    match x {
                        '0'..='9' | '_' => {
                            xs.next();
                            len += 1;
                        },
                        '.' => {
                            if has_dot {
                                break;
                            }
                            xs.next();
                            if xs.peek().map_or(false, |&x| x == '.' || util::is_ident_start(x)) {
                                return Err(Rej);
                            }
                            len += 1;
                            has_dot = true;
                        },
                        'e' | 'E' => {
                            xs.next();
                            len += 1;
                            has_exp = true;
                            break;
                        },
                        _ => break,
                    }
                }
                if !(has_dot || has_exp) {
                    return Err(Rej);
                }
                if has_exp {
                    let pre = if has_dot { Ok(c.advance(len - 1)) } else { Err(Rej) };
                    let mut has_sign = false;
                    let mut has_val = false;
                    while let Some(&x) = xs.peek() {
                        match x {
                            '+' | '-' => {
                                if has_val {
                                    break;
                                }
                                if has_sign {
                                    return pre;
                                }
                                xs.next();
                                len += 1;
                                has_sign = true;
                            },
                            '0'..='9' => {
                                xs.next();
                                len += 1;
                                has_val = true;
                            },
                            '_' => {
                                xs.next();
                                len += 1;
                            },
                            _ => break,
                        }
                    }
                    if !has_val {
                        return pre;
                    }
                }
                Ok(c.advance(len))
            }
            let mut y = parse(self)?;
            if let Some(x) = y.chars().next() {
                if util::is_ident_start(x) {
                    y = y.ident_not_raw()?.0;
                }
            }
            y.word_break()
        }
        fn int(&self) -> Result<Cursor<'a>, Rej> {
            fn parse(mut c: Cursor) -> Result<Cursor, Rej> {
                let base = if c.starts_with("0x") {
                    c = &c.advance(2);
                    16
                } else if c.starts_with("0o") {
                    c = &c.advance(2);
                    8
                } else if c.starts_with("0b") {
                    c = &c.advance(2);
                    2
                } else {
                    10
                };
                let mut len = 0;
                let mut empty = true;
                for x in c.bytes() {
                    match x {
                        b'0'..=b'9' => {
                            let y = (x - b'0') as u64;
                            if y >= base {
                                return Err(Rej);
                            }
                        },
                        b'a'..=b'f' => {
                            let y = 10 + (x - b'a') as u64;
                            if y >= base {
                                break;
                            }
                        },
                        b'A'..=b'F' => {
                            let y = 10 + (x - b'A') as u64;
                            if y >= base {
                                break;
                            }
                        },
                        b'_' => {
                            if empty && base == 10 {
                                return Err(Rej);
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
                    Err(Rej)
                } else {
                    Ok(c.advance(len))
                }
            }
            let mut y = parse(self)?;
            if let Some(x) = y.chars().next() {
                if util::is_ident_start(x) {
                    y = y.ident_not_raw()?.0;
                }
            }
            y.word_break()
        }
        fn punct(&self) -> Res<Punct> {
            fn parse(c: Cursor) -> Res<char> {
                if c.starts_with("//") || c.starts_with("/*") {
                    return Err(Rej);
                }
                let mut xs = c.chars();
                let first = match xs.next() {
                    Some(x) => x,
                    None => {
                        return Err(Rej);
                    },
                };
                let set = "~!@#$%^&*-=+|;:,<.>/?'";
                if set.contains(first) {
                    Ok((c.advance(first.len_utf8()), first))
                } else {
                    Err(Rej)
                }
            }
            let (y, x) = parse(self)?;
            if x == '\'' {
                if y.ident_any()?.0.starts_with_char('\'') {
                    Err(Rej)
                } else {
                    Ok((y, Punct::new('\'', Spacing::Joint)))
                }
            } else {
                let kind = match parse(y) {
                    Ok(_) => Spacing::Joint,
                    Err(Rej) => Spacing::Alone,
                };
                Ok((y, Punct::new(x, kind)))
            }
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
        fn parse(&self, x: &str) -> Result<Cursor<'a>, Rej> {
            if self.starts_with(x) {
                Ok(self.advance(x.len()))
            } else {
                Err(Rej)
            }
        }
        fn word_break(&self) -> Result<Cursor<'a>, Rej> {
            match self.chars().next() {
                Some(x) if util::is_ident_cont(x) => Err(Rej),
                Some(_) | None => Ok(self),
            }
        }
        fn byte(&self) -> Result<Cursor<'a>, Rej> {
            let y = self.parse("b'")?;
            let mut xs = y.bytes().enumerate();
            let ok = match xs.next().map(|(_, x)| x) {
                Some(b'\\') => match xs.next().map(|(_, x)| x) {
                    Some(b'x') => bslash_x_byte(&mut xs).is_ok(),
                    Some(b'n') | Some(b'r') | Some(b't') | Some(b'\\') | Some(b'0') | Some(b'\'') | Some(b'"') => true,
                    _ => false,
                },
                x => x.is_some(),
            };
            if !ok {
                return Err(Rej);
            }
            let (i, _) = xs.next().ok_or(Rej)?;
            if !y.chars().as_str().is_char_boundary(i) {
                return Err(Rej);
            }
            let y = y.advance(i).parse("'")?;
            Ok(y.lit_suff())
        }
        fn char(&self) -> Result<Cursor<'a>, Rej> {
            let y = self.parse("'")?;
            let mut xs = y.char_indices();
            let ok = match xs.next().map(|(_, x)| x) {
                Some('\\') => match xs.next().map(|(_, x)| x) {
                    Some('x') => bslash_x_char(&mut xs).is_ok(),
                    Some('u') => bslash_u(&mut xs).is_ok(),
                    Some('n') | Some('r') | Some('t') | Some('\\') | Some('0') | Some('\'') | Some('"') => true,
                    _ => false,
                },
                x => x.is_some(),
            };
            if !ok {
                return Err(Rej);
            }
            let (i, _) = xs.next().ok_or(Rej)?;
            let y = y.advance(i).parse("'")?;
            Ok(y.lit_suff())
        }
        fn byte_string(&self) -> Result<Cursor<'a>, Rej> {
            fn parse(mut c: Cursor) -> Result<Cursor, Rej> {
                let mut xs = c.bytes().enumerate();
                while let Some((i, x)) = xs.next() {
                    match x {
                        b'"' => {
                            let y = c.advance(i + 1);
                            return Ok(y.lit_suff());
                        },
                        b'\r' => match xs.next() {
                            Some((_, b'\n')) => {},
                            _ => break,
                        },
                        b'\\' => match xs.next() {
                            Some((_, b'x')) => {
                                bslash_x_byte(&mut xs)?;
                            },
                            Some((_, b'n')) | Some((_, b'r')) | Some((_, b't')) | Some((_, b'\\'))
                            | Some((_, b'0')) | Some((_, b'\'')) | Some((_, b'"')) => {},
                            Some((newline, x @ b'\n')) | Some((newline, x @ b'\r')) => {
                                c = c.advance(newline + 1);
                                trailing_bslash(&mut c, x)?;
                                xs = c.bytes().enumerate();
                            },
                            _ => break,
                        },
                        x if x.is_ascii() => {},
                        _ => break,
                    }
                }
                Err(Rej)
            }
            fn parse_raw(c: Cursor) -> Result<Cursor, Rej> {
                let (y, delim) = c.delim_of_raw()?;
                let mut xs = y.bytes().enumerate();
                while let Some((i, x)) = xs.next() {
                    match x {
                        b'"' if y.rest[i + 1..].starts_with(delim) => {
                            let y = y.advance(i + 1 + delim.len());
                            return Ok(y.lit_suff());
                        },
                        b'\r' => match xs.next() {
                            Some((_, b'\n')) => {},
                            _ => break,
                        },
                        x => {
                            if !x.is_ascii() {
                                break;
                            }
                        },
                    }
                }
                Err(Rej)
            }
            if let Ok(y) = self.parse("b\"") {
                parse(y)
            } else if let Ok(y) = self.parse("br") {
                parse_raw(y)
            } else {
                Err(Rej)
            }
        }
        fn c_string(&self) -> Result<Cursor<'a>, Rej> {
            fn parse(mut c: Cursor) -> Result<Cursor, Rej> {
                let mut xs = c.char_indices();
                while let Some((i, x)) = xs.next() {
                    match x {
                        '"' => {
                            let y = c.advance(i + 1);
                            return Ok(y.lit_suff());
                        },
                        '\r' => match xs.next() {
                            Some((_, '\n')) => {},
                            _ => break,
                        },
                        '\\' => match xs.next() {
                            Some((_, 'x')) => {
                                bslash_x_nonzero(&mut xs)?;
                            },
                            Some((_, 'n')) | Some((_, 'r')) | Some((_, 't')) | Some((_, '\\')) | Some((_, '\''))
                            | Some((_, '"')) => {},
                            Some((_, 'u')) => {
                                if bslash_u(&mut xs)? == '\0' {
                                    break;
                                }
                            },
                            Some((newline, x @ '\n')) | Some((newline, x @ '\r')) => {
                                c = c.advance(newline + 1);
                                trailing_bslash(&mut c, x as u8)?;
                                xs = c.char_indices();
                            },
                            _ => break,
                        },
                        '\0' => break,
                        _ => {},
                    }
                }
                Err(Rej)
            }
            fn parse_raw(c: Cursor) -> Result<Cursor, Rej> {
                let (y, delim) = c.delim_of_raw()?;
                let mut xs = y.bytes().enumerate();
                while let Some((i, x)) = xs.next() {
                    match x {
                        b'"' if y.rest[i + 1..].starts_with(delim) => {
                            let y = y.advance(i + 1 + delim.len());
                            return Ok(y.lit_suff());
                        },
                        b'\r' => match xs.next() {
                            Some((_, b'\n')) => {},
                            _ => break,
                        },
                        b'\0' => break,
                        _ => {},
                    }
                }
                Err(Rej)
            }
            if let Ok(y) = self.parse("c\"") {
                parse(y)
            } else if let Ok(y) = self.parse("cr") {
                parse_raw(y)
            } else {
                Err(Rej)
            }
        }
        fn string(&self) -> Result<Cursor<'a>, Rej> {
            fn parse(mut c: Cursor) -> Result<Cursor, Rej> {
                let mut xs = c.char_indices();
                while let Some((i, x)) = xs.next() {
                    match x {
                        '"' => {
                            let y = c.advance(i + 1);
                            return Ok(y.lit_suff());
                        },
                        '\r' => match xs.next() {
                            Some((_, '\n')) => {},
                            _ => break,
                        },
                        '\\' => match xs.next() {
                            Some((_, 'x')) => {
                                bslash_x_char(&mut xs)?;
                            },
                            Some((_, 'n')) | Some((_, 'r')) | Some((_, 't')) | Some((_, '\\')) | Some((_, '\''))
                            | Some((_, '"')) | Some((_, '0')) => {},
                            Some((_, 'u')) => {
                                bslash_u(&mut xs)?;
                            },
                            Some((newline, x @ '\n')) | Some((newline, x @ '\r')) => {
                                c = c.advance(newline + 1);
                                trailing_bslash(&mut c, x as u8)?;
                                xs = c.char_indices();
                            },
                            _ => break,
                        },
                        _ => {},
                    }
                }
                Err(Rej)
            }
            fn parse_raw(c: Cursor) -> Result<Cursor, Rej> {
                let (y, delim) = c.delim_of_raw()?;
                let mut xs = y.bytes().enumerate();
                while let Some((i, x)) = xs.next() {
                    match x {
                        b'"' if y.rest[i + 1..].starts_with(delim) => {
                            let y = y.advance(i + 1 + delim.len());
                            return Ok(y.lit_suff());
                        },
                        b'\r' => match xs.next() {
                            Some((_, b'\n')) => {},
                            _ => break,
                        },
                        _ => {},
                    }
                }
                Err(Rej)
            }
            if let Ok(y) = self.parse("\"") {
                parse(y)
            } else if let Ok(y) = self.parse("r") {
                parse_raw(y)
            } else {
                Err(Rej)
            }
        }
        fn lit_suff(&self) -> Cursor {
            match self.ident_not_raw() {
                Ok((x, _)) => x,
                Err(Rej) => self,
            }
        }
        fn ident_not_raw(&self) -> Res<&str> {
            let mut xs = self.char_indices();
            match xs.next() {
                Some((_, x)) if util::is_ident_start(x) => {},
                _ => return Err(Rej),
            }
            let mut end = self.len();
            for (i, x) in xs {
                if !util::is_ident_cont(x) {
                    end = i;
                    break;
                }
            }
            Ok((self.advance(end), &self.rest[..end]))
        }
        fn delim_of_raw(&self) -> Res<&str> {
            for (i, x) in self.bytes().enumerate() {
                match x {
                    b'"' => {
                        if i > 255 {
                            return Err(Rej);
                        }
                        return Ok((self.advance(i + 1), &self.rest[..i]));
                    },
                    b'#' => {},
                    _ => break,
                }
            }
            Err(Rej)
        }
    }

    struct FileInfo {
        text: String,
        span: Span,
        lines: Vec<usize>,
    }
    impl FileInfo {
        fn offset_line_col(&self, x: usize) -> LineCol {
            assert!(self.span_within(Span {
                lo: x as u32,
                hi: x as u32,
                _marker: util::Marker
            }));
            let y = x - self.span.lo as usize;
            match self.lines.binary_search(&y) {
                Ok(x) => LineCol { line: x + 1, col: 0 },
                Err(line) => LineCol {
                    line,
                    col: y - self.lines[line - 1],
                },
            }
        }
        fn span_within(&self, x: Span) -> bool {
            x.lo >= self.span.lo && x.hi <= self.span.hi
        }
        fn source_text(&self, x: Span) -> String {
            let lo = (x.lo - self.span.lo) as usize;
            let hi = (x.hi - self.span.lo) as usize;
            self.text[lo..hi].to_owned()
        }
    }

    #[derive(Clone, Debug)]
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
                Delim::Parenth => ("(", ")"),
                Delim::Brace => ("{ ", "}"),
                Delim::Bracket => ("[", "]"),
                Delim::None => ("", ""),
            };
            f.write_str(open)?;
            Display::fmt(&self.stream, f)?;
            if self.delim == Delim::Brace && !self.stream.trees.is_empty() {
                f.write_str(" ")?;
            }
            f.write_str(close)?;
            Ok(())
        }
    }

    #[derive(Clone, Debug)]
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
            util::validate_ident(x, raw);
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

    #[derive(Clone, Debug)]
    pub struct Lit {
        repr: String,
        span: Span,
        _marker: util::Marker,
    }
    macro_rules! suffixed_nums {
        ($($n:ident => $kind:ident,)*) => ($(
            pub fn $n(n: $kind) -> Lit {
                Lit::new(format!(concat!("{}", stringify!($kind)), n))
            }
        )*)
    }
    macro_rules! unsuffixed_nums {
        ($($n:ident => $kind:ident,)*) => ($(
            pub fn $n(n: $kind) -> Lit {
                Lit::new(n.to_string())
            }
        )*)
    }
    impl Lit {
        pub fn new(repr: String) -> Self {
            Lit {
                repr,
                span: Span::call_site(),
                _marker: util::Marker,
            }
        }
        pub unsafe fn from_str_unchecked(x: &str) -> Self {
            Lit::new(x.to_owned())
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
            Lit::new(y)
        }
        pub fn f64_unsuffixed(x: f64) -> Lit {
            let mut y = x.to_string();
            if !y.contains('.') {
                y.push_str(".0");
            }
            Lit::new(y)
        }
        pub fn char(x: char) -> Lit {
            let mut y = String::new();
            y.push('\'');
            if x == '"' {
                y.push(x);
            } else {
                y.extend(x.escape_debug());
            }
            y.push('\'');
            Lit::new(y)
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
            Lit::new(y)
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
            Lit::new(y)
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
        fn from_str(x: &str) -> Result<Self, Self::Err> {
            let mut x = get_cursor(x);
            let lo = x.off;
            let neg = x.starts_with_char('-');
            if neg {
                x = x.advance(1);
                if !x.starts_with_fn(|x| x.is_ascii_digit()) {
                    return Err(LexErr::call_site());
                }
            }
            if let Ok((x, mut y)) = x.lit() {
                if x.is_empty() {
                    if neg {
                        y.repr.insert(0, '-');
                    }
                    y.span = Span {
                        lo,
                        hi: x.off,
                        _marker: util::Marker,
                    };
                    return Ok(y);
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

    pub struct Rej;
    pub type Res<'a, T> = Result<(Cursor<'a>, T), Rej>;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
                y.offset_line_col(self.lo as usize)
            })
        }
        pub fn end(&self) -> LineCol {
            SRC_MAP.with(|x| {
                let x = x.borrow();
                let y = x.fileinfo(*self);
                y.offset_line_col(self.hi as usize)
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

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct SrcFile {
        path: PathBuf,
    }
    impl SrcFile {
        pub fn path(&self) -> PathBuf {
            self.path.clone()
        }
        pub fn is_real(&self) -> bool {
            false
        }
    }

    struct SrcMap {
        files: Vec<FileInfo>,
    }
    impl SrcMap {
        fn next_start_pos(&self) -> u32 {
            self.files.last().unwrap().span.hi + 1
        }
        fn add_file(&mut self, x: &str) -> Span {
            let (len, lines) = lines_offsets(x);
            let lo = self.next_start_pos();
            let span = Span {
                lo,
                hi: lo + (len as u32),
                _marker: util::Marker,
            };
            self.files.push(FileInfo {
                text: x.to_owned(),
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
    fn lines_offsets(xs: &str) -> (usize, Vec<usize>) {
        let mut lines = vec![0];
        let mut total = 0;
        for x in xs.chars() {
            total += 1;
            if x == '\n' {
                lines.push(total);
            }
        }
        (total, lines)
    }

    thread_local! {
        static SRC_MAP: RefCell<SrcMap> = RefCell::new(SrcMap {
            files: vec![FileInfo {
                text: String::new(),
                span: Span { lo: 0, hi: 0 },
                lines: vec![0],
            }],
        });
    }

    #[derive(Clone, Debug)]
    pub struct Stream {
        trees: RcVec<Tree>,
        _marker: util::Marker,
    }
    impl Stream {
        pub fn new() -> Self {
            Stream {
                trees: rcvec::Builder::new().build(),
                _marker: util::Marker,
            }
        }
        pub fn is_empty(&self) -> bool {
            self.trees.len() == 0
        }
        fn take_trees(self) -> rcvec::Builder<Tree> {
            let y = ManuallyDrop::new(self);
            unsafe { std::ptr::read(&y.trees) }.make_owned()
        }
    }
    impl Drop for Stream {
        fn drop(&mut self) {
            let mut ys = match self.trees.get_mut() {
                Some(x) => x,
                None => return,
            };
            while let Some(x) = ys.pop() {
                let y = match x {
                    Tree::Group(x) => x,
                    _ => continue,
                };
                let y = match y {
                    super::Group::Compiler(_) => continue,
                    super::Group::Fallback(x) => x,
                };
                ys.extend(y.stream.take_trees());
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
            token_stream(y)
        }
    }
    impl From<pm::TokenStream> for Stream {
        fn from(x: pm::TokenStream) -> Self {
            x.to_string().parse().expect("compiler token stream parse failed")
        }
    }
    impl From<Tree> for Stream {
        fn from(x: Tree) -> Self {
            let mut y = rcvec::Builder::new();
            push_from_pm(y.as_mut(), x);
            Stream {
                trees: y.build(),
                _marker: util::Marker,
            }
        }
    }
    impl FromIterator<Stream> for Stream {
        fn from_iter<I: IntoIterator<Item = Stream>>(xs: I) -> Self {
            let mut y = rcvec::Builder::new();
            for x in xs {
                y.extend(x.take_trees());
            }
            Stream {
                trees: y.build(),
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
            self.trees.make_mut().extend(xs.into_iter().flatten());
        }
    }
    impl Extend<Tree> for Stream {
        fn extend<I: IntoIterator<Item = Tree>>(&mut self, xs: I) {
            let mut y = self.trees.make_mut();
            xs.into_iter().for_each(|x| push_from_pm(y.as_mut(), x));
        }
    }
    impl IntoIterator for Stream {
        type Item = Tree;
        type IntoIter = TreeIter;
        fn into_iter(self) -> TreeIter {
            self.take_trees().into_iter()
        }
    }
    impl Display for Stream {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut joint = false;
            for (i, x) in self.trees.iter().enumerate() {
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
    impl From<Stream> for pm::TokenStream {
        fn from(x: Stream) -> Self {
            x.to_string().parse().expect("failed to parse to compiler tokens")
        }
    }
    fn push_from_pm(mut ys: RcVecMut<Tree>, x: Tree) {
        #[cold]
        fn push_neg_lit(mut ys: RcVecMut<Tree>, mut x: Lit) {
            x.repr.remove(0);
            let mut y = super::Punct::new('-', Spacing::Alone);
            y.set_span(super::Span::Fallback(x.span));
            ys.push(Tree::Punct(y));
            ys.push(Tree::Lit(super::Lit::Fallback(x)));
        }
        match x {
            Tree::Lit(super::imp::Lit {
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

    pub type TreeIter = rcvec::IntoIter<Tree>;

    pub fn force() {
        util::force_fallback();
    }
    pub fn unforce() {
        util::unforce_fallback();
    }

    pub fn token_stream(mut c: Cursor) -> Result<Stream, LexErr> {
        fn skip_ws(c: Cursor) -> Cursor {
            fn is_ws(x: char) -> bool {
                x.is_whitespace() || x == '\u{200e}' || x == '\u{200f}'
            }
            let mut y = c;
            while !y.is_empty() {
                let x = y.as_bytes()[0];
                if x == b'/' {
                    if y.starts_with("//") && (!y.starts_with("///") || y.starts_with("////")) && !y.starts_with("//!")
                    {
                        let (x, _) = y.to_nl_or_eof();
                        y = &x;
                        continue;
                    } else if y.starts_with("/**/") {
                        y = &y.advance(4);
                        continue;
                    } else if y.starts_with("/*")
                        && (!y.starts_with("/**") || y.starts_with("/***"))
                        && !y.starts_with("/*!")
                    {
                        match y.comment() {
                            Ok((x, _)) => {
                                y = &x;
                                continue;
                            },
                            Err(Rej) => return y,
                        }
                    }
                }
                match x {
                    b' ' | 0x09..=0x0d => {
                        y = &y.advance(1);
                        continue;
                    },
                    x if x.is_ascii() => {},
                    _ => {
                        let x = y.chars().next().unwrap();
                        if is_ws(x) {
                            y = &y.advance(x.len_utf8());
                            continue;
                        }
                    },
                }
                return y;
            }
            y
        }
        fn doc<'a>(c: Cursor<'a>, ys: &mut Builder) -> Res<'a, ()> {
            fn parse(c: Cursor) -> Res<(&str, bool)> {
                if c.starts_with("//!") {
                    let x = c.advance(3);
                    let (x, y) = x.to_nl_or_eof();
                    Ok((x, (y, true)))
                } else if c.starts_with("/*!") {
                    let (x, y) = c.comment()?;
                    Ok((x, (&y[3..y.len() - 2], true)))
                } else if c.starts_with("///") {
                    let x = c.advance(3);
                    if x.starts_with_char('/') {
                        return Err(Rej);
                    }
                    let (x, y) = x.to_nl_or_eof();
                    Ok((x, (y, false)))
                } else if c.starts_with("/**") && !c.rest[3..].starts_with('*') {
                    let (x, y) = c.comment()?;
                    Ok((x, (&y[3..y.len() - 2], false)))
                } else {
                    Err(Rej)
                }
            }
            let lo = c.off;
            let (y, (comment, inner)) = parse(c)?;
            let s = super::Span::Fallback(Span {
                lo,
                hi: y.off,
                _marker: util::Marker,
            });
            let mut cr = comment;
            while let Some(cr) = cr.find('\r') {
                let y = &cr[cr + 1..];
                if !y.starts_with('\n') {
                    return Err(Rej);
                }
                cr = y;
            }
            let mut pound = Punct::new('#', Spacing::Alone);
            pound.set_span(s);
            ys.push_from_parser(Tree::Punct(pound));
            if inner {
                let mut bang = Punct::new('!', Spacing::Alone);
                bang.set_span(s);
                ys.push_from_parser(Tree::Punct(bang));
            }
            let ident = super::Ident::new("doc", s);
            let mut eq = Punct::new('=', Spacing::Alone);
            eq.set_span(s);
            let mut lit = super::Lit::string(comment);
            lit.set_span(s);
            let mut bracketed = Builder::with_capacity(3);
            bracketed.push_from_parser(Tree::Ident(ident));
            bracketed.push_from_parser(Tree::Punct(eq));
            bracketed.push_from_parser(Tree::Lit(lit));
            let g = Group::new(Delim::Bracket, bracketed.build());
            let mut g = super::Group::Fallback(g);
            g.set_span(s);
            ys.push_from_parser(Tree::Group(g));
            Ok((y, ()))
        }
        fn parse(c: Cursor) -> Res<Tree> {
            if let Ok((x, y)) = c.lit() {
                Ok((x, Tree::Lit(super::Lit::Fallback(y))))
            } else if let Ok((x, y)) = c.punct() {
                Ok((x, Tree::Punct(y)))
            } else if let Ok((x, y)) = c.ident() {
                Ok((x, Tree::Ident(y)))
            } else {
                Err(Rej)
            }
        }
        let mut trees = Builder::new();
        let mut stack = Vec::new();
        loop {
            c = skip_ws(c);
            if let Ok((y, ())) = doc(c, &mut trees) {
                c = y;
                continue;
            }
            let lo = c.off;
            let first = match c.bytes().next() {
                Some(x) => x,
                None => match stack.last() {
                    None => return Ok(trees.build()),
                    Some((lo, _)) => {
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
            if let Some(open) = match first {
                b'(' => Some(Delim::Parenth),
                b'[' => Some(Delim::Bracket),
                b'{' => Some(Delim::Brace),
                _ => None,
            } {
                c = c.advance(1);
                let frame = (open, trees);
                let frame = (lo, frame);
                stack.push(frame);
                trees = Builder::new();
            } else if let Some(close) = match first {
                b')' => Some(Delim::Parenth),
                b']' => Some(Delim::Bracket),
                b'}' => Some(Delim::Brace),
                _ => None,
            } {
                let frame = match stack.pop() {
                    Some(x) => x,
                    None => return Err(c.lex_err()),
                };
                let (lo, frame) = frame;
                let (open, outer) = frame;
                if open != close {
                    return Err(c.lex_err());
                }
                c = c.advance(1);
                let mut g = Group::new(open, trees.build());
                g.set_span(Span {
                    lo,
                    hi: c.off,
                    _marker: util::Marker,
                });
                trees = outer;
                trees.push_from_parser(Tree::Group(super::Group::Fallback(g)));
            } else {
                let (rest, mut tt) = match parse(c) {
                    Ok((rest, tt)) => (rest, tt),
                    Err(Rej) => return Err(c.lex_err()),
                };
                tt.set_span(super::Span::Fallback(Span {
                    lo,
                    hi: rest.off,
                    _marker: util::Marker,
                }));
                trees.push_from_parser(tt);
                c = rest;
            }
        }
    }

    macro_rules! next_char {
        ($xs:ident @ $p:pat_param $(| $rest:pat)*) => {
            match $xs.next() {
                Some((_, x)) => match x {
                    $p $(| $rest)* => x,
                    _ => return Err(Rej),
                },
                None => return Err(Rej),
            }
        };
    }
    fn bslash_x_char<I>(xs: &mut I) -> Result<(), Rej>
    where
        I: Iterator<Item = (usize, char)>,
    {
        next_char!(xs @ '0'..='7');
        next_char!(xs @ '0'..='9' | 'a'..='f' | 'A'..='F');
        Ok(())
    }
    fn bslash_x_byte<I>(xs: &mut I) -> Result<(), Rej>
    where
        I: Iterator<Item = (usize, u8)>,
    {
        next_char!(xs @ b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F');
        next_char!(xs @ b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F');
        Ok(())
    }
    fn bslash_x_nonzero<I>(xs: &mut I) -> Result<(), Rej>
    where
        I: Iterator<Item = (usize, char)>,
    {
        let first = next_char!(xs @ '0'..='9' | 'a'..='f' | 'A'..='F');
        let second = next_char!(xs @ '0'..='9' | 'a'..='f' | 'A'..='F');
        if first == '0' && second == '0' {
            Err(Rej)
        } else {
            Ok(())
        }
    }
    fn bslash_u<I>(xs: &mut I) -> Result<char, Rej>
    where
        I: Iterator<Item = (usize, char)>,
    {
        next_char!(xs @ '{');
        let mut y = 0;
        let mut len = 0;
        for (_, x) in xs {
            let digit = match x {
                '0'..='9' => x as u8 - b'0',
                'a'..='f' => 10 + x as u8 - b'a',
                'A'..='F' => 10 + x as u8 - b'A',
                '_' if len > 0 => continue,
                '}' if len > 0 => return char::from_u32(y).ok_or(Rej),
                _ => break,
            };
            if len == 6 {
                break;
            }
            y *= 0x10;
            y += u32::from(digit);
            len += 1;
        }
        Err(Rej)
    }

    fn trailing_bslash(c: &mut Cursor, mut last: u8) -> Result<(), Rej> {
        let mut ws = c.bytes().enumerate();
        loop {
            if last == b'\r' && ws.next().map_or(true, |(_, x)| x != b'\n') {
                return Err(Rej);
            }
            match ws.next() {
                Some((_, x @ b' ')) | Some((_, x @ b'\t')) | Some((_, x @ b'\n')) | Some((_, x @ b'\r')) => {
                    last = x;
                },
                Some((i, _)) => {
                    *c = c.advance(i);
                    return Ok(());
                },
                None => return Err(Rej),
            }
        }
    }
}
mod rcvec {
    use std::{panic::RefUnwindSafe, rc::Rc, slice, vec};
    pub struct Builder<T> {
        vec: Vec<T>,
    }
    impl<T> Builder<T> {
        pub fn new() -> Self {
            Builder { vec: Vec::new() }
        }
        pub fn with_capacity(x: usize) -> Self {
            Builder {
                vec: Vec::with_capacity(x),
            }
        }
        pub fn push(&mut self, x: T) {
            self.vec.push(x);
        }
        pub fn extend(&mut self, xs: impl IntoIterator<Item = T>) {
            self.vec.extend(xs);
        }
        pub fn as_mut(&mut self) -> RcVecMut<T> {
            RcVecMut { vec: &mut self.vec }
        }
        pub fn build(self) -> RcVec<T> {
            RcVec { vec: Rc::new(self.vec) }
        }
    }

    pub struct RcVec<T> {
        vec: Rc<Vec<T>>,
    }
    impl<T> RcVec<T> {
        pub fn is_empty(&self) -> bool {
            self.vec.is_empty()
        }
        pub fn len(&self) -> usize {
            self.vec.len()
        }
        pub fn iter(&self) -> slice::Iter<T> {
            self.vec.iter()
        }
        pub fn make_mut(&mut self) -> RcVecMut<T>
        where
            T: Clone,
        {
            RcVecMut {
                vec: Rc::make_mut(&mut self.vec),
            }
        }
        pub fn get_mut(&mut self) -> Option<RcVecMut<T>> {
            let vec = Rc::get_mut(&mut self.vec)?;
            Some(RcVecMut { vec })
        }
        pub fn make_owned(mut self) -> Builder<T>
        where
            T: Clone,
        {
            let vec = if let Some(x) = Rc::get_mut(&mut self.vec) {
                std::mem::replace(x, Vec::new())
            } else {
                Vec::clone(&self.vec)
            };
            Builder { vec }
        }
    }
    impl<T> Clone for RcVec<T> {
        fn clone(&self) -> Self {
            RcVec {
                vec: Rc::clone(&self.vec),
            }
        }
    }
    impl<T> RefUnwindSafe for RcVec<T> where T: RefUnwindSafe {}

    pub struct RcVecMut<'a, T> {
        vec: &'a mut Vec<T>,
    }
    impl<'a, T> RcVecMut<'a, T> {
        pub fn push(&mut self, x: T) {
            self.vec.push(x);
        }
        pub fn extend(&mut self, xs: impl IntoIterator<Item = T>) {
            self.vec.extend(xs);
        }
        pub fn pop(&mut self) -> Option<T> {
            self.vec.pop()
        }
        pub fn as_mut(&mut self) -> RcVecMut<T> {
            RcVecMut { vec: self.vec }
        }
    }

    #[derive(Clone)]
    pub struct IntoIter<T> {
        iter: vec::IntoIter<T>,
    }
    impl<T> IntoIterator for Builder<T> {
        type Item = T;
        type IntoIter = IntoIter<T>;
        fn into_iter(self) -> Self::IntoIter {
            IntoIter {
                iter: self.vec.into_iter(),
            }
        }
    }
    impl<T> Iterator for IntoIter<T> {
        type Item = T;
        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next()
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.iter.size_hint()
        }
    }
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

    pub fn is_ident_start(x: char) -> bool {
        x == '_' || unicode_ident::is_xid_start(x)
    }
    pub fn is_ident_cont(x: char) -> bool {
        unicode_ident::is_xid_continue(x)
    }
    pub fn validate_ident(x: &str, raw: bool) {
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
}
