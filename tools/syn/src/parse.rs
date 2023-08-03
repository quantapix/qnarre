use super::{
    pm2::{Delim, DelimSpan, Span},
    *,
};
use std::{cell::Cell, rc::Rc};

pub struct Parenths<'a> {
    pub tok: tok::Parenth,
    pub buf: Buffer<'a>,
}
pub fn parse_parenths<'a>(x: &Buffer<'a>) -> Res<Parenths<'a>> {
    parse_delimited(x, Delim::Parenth).map(|(x, buf)| Parenths {
        tok: tok::Parenth(x),
        buf,
    })
}

pub struct Braces<'a> {
    pub tok: tok::Brace,
    pub buf: Buffer<'a>,
}
pub fn parse_braces<'a>(x: &Buffer<'a>) -> Res<Braces<'a>> {
    parse_delimited(x, Delim::Brace).map(|(x, buf)| Braces {
        tok: tok::Brace(x),
        buf,
    })
}

pub struct Brackets<'a> {
    pub tok: tok::Bracket,
    pub buf: Buffer<'a>,
}
pub fn parse_brackets<'a>(x: &Buffer<'a>) -> Res<Brackets<'a>> {
    parse_delimited(x, Delim::Bracket).map(|(x, buf)| Brackets {
        tok: tok::Bracket(x),
        buf,
    })
}

pub struct Group<'a> {
    pub tok: tok::Group,
    pub buf: Buffer<'a>,
}
pub fn parse_group<'a>(x: &Buffer<'a>) -> Res<Group<'a>> {
    parse_delimited(x, Delim::None).map(|(span, buf)| Group {
        tok: tok::Group(span.join()),
        buf,
    })
}
fn parse_delimited<'a>(b: &Buffer<'a>, d: Delim) -> Res<(DelimSpan, Buffer<'a>)> {
    b.step(|x| {
        if let Some((y, span, rest)) = x.group(d) {
            let scope = cur::close_span_of_group(*x);
            let nested = advance_step_cursor(x, y);
            let unexpected = get_unexpected(b);
            let y = new_parse_buffer(scope, nested, unexpected);
            Ok(((span, y), rest))
        } else {
            let y = match d {
                Delim::Parenth => "expected parentheses",
                Delim::Brace => "expected braces",
                Delim::Bracket => "expected brackets",
                Delim::None => "expected group",
            };
            Err(x.err(y))
        }
    })
}

pub struct Buffer<'a> {
    scope: Span,
    cur: Cell<Cursor<'static>>,
    unexp: Cell<Option<Rc<Cell<Unexpected>>>>,
    _marker: PhantomData<Cursor<'a>>,
}
impl<'a> Buffer<'a> {
    pub fn parse<T: Parse>(&self) -> Res<T> {
        T::parse(self)
    }
    pub fn call<T>(&self, x: fn(Stream) -> Res<T>) -> Res<T> {
        x(self)
    }
    pub fn peek<T: Peek>(&self, _: T) -> bool {
        T::Tok::peek(self.cursor())
    }
    pub fn peek2<T: Peek>(&self, _: T) -> bool {
        fn doit(x: &Buffer, f: fn(Cursor) -> bool) -> bool {
            if let Some(x) = x.cursor().group(Delim::None) {
                if x.0.skip().map_or(false, f) {
                    return true;
                }
            }
            x.cursor().skip().map_or(false, f)
        }
        doit(self, T::Tok::peek)
    }
    pub fn peek3<T: Peek>(&self, _: T) -> bool {
        fn doit(x: &Buffer, f: fn(Cursor) -> bool) -> bool {
            if let Some(x) = x.cursor().group(Delim::None) {
                if x.0.skip().and_then(Cursor::skip).map_or(false, f) {
                    return true;
                }
            }
            x.cursor().skip().and_then(Cursor::skip).map_or(false, f)
        }
        doit(self, T::Tok::peek)
    }
    pub fn parse_terminated<T, P>(&self, f: fn(Stream) -> Res<T>, sep: P) -> Res<Puncted<T, P::Tok>>
    where
        P: Peek,
        P::Tok: Parse,
    {
        let _ = sep;
        Puncted::parse_terminated_with(self, f)
    }
    pub fn is_empty(&self) -> bool {
        self.cursor().eof()
    }
    pub fn look1(&self) -> Look1<'a> {
        look::new(self.scope, self.cursor())
    }
    pub fn fork(&self) -> Self {
        Buffer {
            scope: self.scope,
            cur: self.cur.clone(),
            _marker: PhantomData,
            unexp: Cell::new(Some(Rc::new(Cell::new(Unexpected::None)))),
        }
    }
    pub fn error<T: Display>(&self, x: T) -> Err {
        err::new_at(self.scope, self.cursor(), x)
    }
    pub fn step<F, R>(&self, f: F) -> Res<R>
    where
        F: for<'c> FnOnce(Step<'c, 'a>) -> Res<(R, Cursor<'c>)>,
    {
        let (y, rest) = f(Step {
            scope: self.scope,
            cur: self.cur.get(),
            _marker: PhantomData,
        })?;
        self.cur.set(rest);
        Ok(y)
    }
    pub fn span(&self) -> Span {
        let y = self.cursor();
        if y.eof() {
            self.scope
        } else {
            cur::open_span_of_group(y)
        }
    }
    pub fn cursor(&self) -> Cursor<'a> {
        self.cur.get()
    }
    fn check_unexpected(&self) -> Res<()> {
        match inner_unexpected(self).1 {
            Some(x) => Err(Err::new(x, "unexpected token")),
            None => Ok(()),
        }
    }
}
impl<'a> Drop for Buffer<'a> {
    fn drop(&mut self) {
        if let Some(x) = span_of_unexpected_ignoring_nones(self.cursor()) {
            let (inner, old) = inner_unexpected(self);
            if old.is_none() {
                inner.set(Unexpected::Some(x));
            }
        }
    }
}
impl<'a> Display for Buffer<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.cursor().token_stream(), f)
    }
}
impl<'a> Debug for Buffer<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.cursor().token_stream(), f)
    }
}

pub enum Unexpected {
    None,
    Some(Span),
    Chain(Rc<Cell<Unexpected>>),
}
impl Default for Unexpected {
    fn default() -> Self {
        Unexpected::None
    }
}
impl Clone for Unexpected {
    fn clone(&self) -> Self {
        use Unexpected::*;
        match self {
            None => None,
            Some(x) => Some(*x),
            Chain(x) => Chain(x.clone()),
        }
    }
}

pub struct Step<'c, 'a> {
    scope: Span,
    cur: Cursor<'c>,
    _marker: PhantomData<fn(Cursor<'c>) -> Cursor<'a>>,
}
impl<'c, 'a> Deref for Step<'c, 'a> {
    type Target = Cursor<'c>;
    fn deref(&self) -> &Self::Target {
        &self.cur
    }
}
impl<'c, 'a> Copy for Step<'c, 'a> {}
impl<'c, 'a> Clone for Step<'c, 'a> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'c, 'a> Step<'c, 'a> {
    pub fn err<T: Display>(self, x: T) -> Err {
        err::new_at(self.scope, self.cur, x)
    }
}

pub type Stream<'a> = &'a Buffer<'a>;

pub trait Quote: Sized {
    fn parse(s: Stream) -> Res<Self>;
}
impl<T: Parse> Quote for T {
    fn parse(s: Stream) -> Res<Self> {
        <T as Parse>::parse(s)
    }
}
impl Quote for attr::Attr {
    fn parse(s: Stream) -> Res<Self> {
        if s.peek(Token![#]) && s.peek2(Token![!]) {
            attr::parse_one_inner(s)
        } else {
            attr::parse_one_outer(s)
        }
    }
}
impl Quote for pat::Pat {
    fn parse(s: Stream) -> Res<Self> {
        pat::Pat::parse_with_vert(s)
    }
}
impl Quote for Box<pat::Pat> {
    fn parse(s: Stream) -> Res<Self> {
        <pat::Pat as Quote>::parse(s).map(Box::new)
    }
}
impl<T: Parse, P: Parse> Quote for Puncted<T, P> {
    fn parse(s: Stream) -> Res<Self> {
        Self::parse_terminated(s)
    }
}
impl Quote for Vec<stmt::Stmt> {
    fn parse(s: Stream) -> Res<Self> {
        stmt::Block::parse_within(s)
    }
}
pub fn parse_quote_fn<T: Quote>(s: Stream) -> T {
    let y = T::parse;
    match y.parse2(s) {
        Ok(x) => x,
        Err(x) => panic!("{}", x),
    }
}

pub trait Parse: Sized {
    fn parse(s: Stream) -> Res<Self>;
}
impl<T: Parse> Parse for Box<T> {
    fn parse(s: Stream) -> Res<Self> {
        s.parse().map(Box::new)
    }
}
impl<T: Parse + Tok> Parse for Option<T> {
    fn parse(s: Stream) -> Res<Self> {
        if T::peek(s.cursor()) {
            Ok(Some(s.parse()?))
        } else {
            Ok(None)
        }
    }
}
impl Parse for pm2::Stream {
    fn parse(s: Stream) -> Res<Self> {
        s.step(|x| Ok((x.token_stream(), Cursor::empty())))
    }
}
impl Parse for pm2::Tree {
    fn parse(s: Stream) -> Res<Self> {
        s.step(|x| match x.token_tree() {
            Some((y, c)) => Ok((y, c)),
            None => Err(x.err("expected token tree")),
        })
    }
}
impl Parse for pm2::Group {
    fn parse(s: Stream) -> Res<Self> {
        s.step(|x| {
            if let Some((y, c)) = x.any_group_token() {
                if y.delim() != Delim::None {
                    return Ok((y, c));
                }
            }
            Err(x.err("expected group token"))
        })
    }
}
impl Parse for Punct {
    fn parse(s: Stream) -> Res<Self> {
        s.step(|x| match x.punct() {
            Some((y, c)) => Ok((y, c)),
            None => Err(x.err("expected punct token")),
        })
    }
}
impl Parse for pm2::Lit {
    fn parse(s: Stream) -> Res<Self> {
        s.step(|x| match x.literal() {
            Some((y, c)) => Ok((y, c)),
            None => Err(x.err("expected lit token")),
        })
    }
}

pub trait Parser: Sized {
    type Output;
    fn parse2(self, s: pm2::Stream) -> Res<Self::Output>;
    fn parse(self, s: pm2::Stream) -> Res<Self::Output> {
        self.parse2(pm2::Stream::from(s))
    }
    fn parse_str(self, x: &str) -> Res<Self::Output> {
        self.parse2(pm2::Stream::from_str(x)?)
    }
    fn parse_scoped(self, scope: Span, s: pm2::Stream) -> Res<Self::Output> {
        let _ = scope;
        self.parse2(s)
    }
}
impl<F, T> Parser for F
where
    F: FnOnce(Stream) -> Res<T>,
{
    type Output = T;
    fn parse2(self, s: pm2::Stream) -> Res<T> {
        let buf = cur::Buffer::new2(s);
        let state = tokens_to_parse_buffer(&buf);
        let node = self(&state)?;
        state.check_unexpected()?;
        if let Some(x) = span_of_unexpected_ignoring_nones(state.cursor()) {
            Err(Err::new(x, "unexpected token"))
        } else {
            Ok(node)
        }
    }
    fn parse_scoped(self, scope: Span, s: pm2::Stream) -> Res<Self::Output> {
        let buf = cur::Buffer::new2(s);
        let cursor = buf.begin();
        let unexpected = Rc::new(Cell::new(Unexpected::None));
        let state = new_parse_buffer(scope, cursor, unexpected);
        let node = self(&state)?;
        state.check_unexpected()?;
        if let Some(x) = span_of_unexpected_ignoring_nones(state.cursor()) {
            Err(Err::new(x, "unexpected token"))
        } else {
            Ok(node)
        }
    }
}

pub struct Nothing;
impl Parse for Nothing {
    fn parse(_: Stream) -> Res<Self> {
        Ok(Nothing)
    }
}
impl Debug for Nothing {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Nothing")
    }
}
impl Eq for Nothing {}
impl PartialEq for Nothing {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}
impl Hash for Nothing {
    fn hash<H: Hasher>(&self, _: &mut H) {}
}

pub mod discouraged {
    use super::*;
    pub trait Speculative {
        fn advance_to(&self, fork: &Self);
    }
    impl<'a> Speculative for Buffer<'a> {
        fn advance_to(&self, fork: &Self) {
            if !cur::same_scope(self.cursor(), fork.cursor()) {
                panic!("Fork was not derived from the advancing parse stream");
            }
            let (self_unexp, self_sp) = inner_unexpected(self);
            let (fork_unexp, fork_sp) = inner_unexpected(fork);
            if !Rc::ptr_eq(&self_unexp, &fork_unexp) {
                match (fork_sp, self_sp) {
                    (Some(x), None) => {
                        self_unexp.set(Unexpected::Some(x));
                    },
                    (None, None) => {
                        fork_unexp.set(Unexpected::Chain(self_unexp));
                        fork.unexp.set(Some(Rc::new(Cell::new(Unexpected::None))));
                    },
                    (_, Some(_)) => {},
                }
            }
            self.cell
                .set(unsafe { std::mem::transmute::<Cursor, Cursor<'static>>(fork.cursor()) });
        }
    }
    pub trait AnyDelim {
        fn parse_any_delim(&self) -> Res<(Delim, DelimSpan, Buffer)>;
    }
    impl<'a> AnyDelim for Buffer<'a> {
        fn parse_any_delim(&self) -> Res<(Delim, DelimSpan, Buffer)> {
            self.step(|c| {
                if let Some((y, delim, span, rest)) = c.any_group() {
                    let scope = cur::close_span_of_group(*c);
                    let nested = advance_step_cursor(c, y);
                    let unexp = get_unexpected(self);
                    let y = new_parse_buffer(scope, nested, unexp);
                    Ok(((delim, span, y), rest))
                } else {
                    Err(c.error("expected any delimiter"))
                }
            })
        }
    }
}

pub fn advance_step_cursor<'c, 'a>(_: Step<'c, 'a>, to: Cursor<'c>) -> Cursor<'a> {
    unsafe { std::mem::transmute::<Cursor<'c>, Cursor<'a>>(to) }
}
pub fn new_parse_buffer(scope: Span, cur: Cursor, unexp: Rc<Cell<Unexpected>>) -> Buffer {
    Buffer {
        scope,
        cur: Cell::new(unsafe { std::mem::transmute::<Cursor, Cursor<'static>>(cur) }),
        _marker: PhantomData,
        unexp: Cell::new(Some(unexp)),
    }
}

fn cell_clone<T: Default + Clone>(x: &Cell<T>) -> T {
    let prev = x.take();
    let y = prev.clone();
    x.set(prev);
    y
}
fn inner_unexpected(x: &Buffer) -> (Rc<Cell<Unexpected>>, Option<Span>) {
    let mut y = get_unexpected(x);
    loop {
        use Unexpected::*;
        match cell_clone(&y) {
            None => return (y, None),
            Some(x) => return (y, Some(x)),
            Chain(x) => y = x,
        }
    }
}
pub fn get_unexpected(x: &Buffer) -> Rc<Cell<Unexpected>> {
    cell_clone(&x.unexp).unwrap()
}
fn span_of_unexpected_ignoring_nones(mut x: Cursor) -> Option<Span> {
    if x.eof() {
        return None;
    }
    while let Some((inner, _span, rest)) = x.group(Delim::None) {
        if let Some(x) = span_of_unexpected_ignoring_nones(inner) {
            return Some(x);
        }
        x = rest;
    }
    if x.eof() {
        None
    } else {
        Some(x.span())
    }
}

fn tokens_to_parse_buffer(x: &cur::Buffer) -> Buffer {
    let scope = Span::call_site();
    let cur = x.begin();
    let unexp = Rc::new(Cell::new(Unexpected::None));
    new_parse_buffer(scope, cur, unexp)
}

pub fn parse_scoped<F: Parser>(f: F, scope: Span, s: pm2::Stream) -> Res<F::Output> {
    f.parse_scoped(scope, s)
}

pub fn parse_verbatim<'a>(beg: Stream<'a>, end: Stream<'a>) -> Stream<'a> {
    let end = end.cursor();
    let mut cur = beg.cursor();
    assert!(cur::same_buff(end, cur));
    let mut ys = Stream::new();
    while cur != end {
        let (tt, next) = cur.token_tree().unwrap();
        if cur::cmp_assuming_same_buffer(end, next) == Ordering::Less {
            if let Some((inside, _span, after)) = cur.group(pm2::Delim::None) {
                assert!(next == after);
                cur = inside;
                continue;
            } else {
                panic!("verbatim end must not be inside a delimited group");
            }
        }
        ys.extend(std::iter::once(tt));
        cur = next;
    }
    ys
}

pub fn parse_file(mut x: &str) -> Res<item::File> {
    const BOM: &str = "\u{feff}";
    if x.starts_with(BOM) {
        x = &x[BOM.len()..];
    }
    let mut shebang = None;
    if x.starts_with("#!") {
        let rest = skip_ws(&x[2..]);
        if !rest.starts_with('[') {
            if let Some(i) = x.find('\n') {
                shebang = Some(x[..i].to_string());
                x = &x[i..];
            } else {
                shebang = Some(x.to_string());
                x = "";
            }
        }
    }
    let mut y: item::File = parse_str(x)?;
    y.shebang = shebang;
    Ok(y)
}

pub fn parse_str<T: Parse>(x: &str) -> Res<T> {
    Parser::parse_str(T::parse, x)
}

fn skip_ws(mut x: &str) -> &str {
    fn is_ws(x: char) -> bool {
        x.is_whitespace() || x == '\u{200e}' || x == '\u{200f}'
    }
    'skip: while !x.is_empty() {
        let byte = x.as_bytes()[0];
        if byte == b'/' {
            if x.starts_with("//") && (!x.starts_with("///") || x.starts_with("////")) && !x.starts_with("//!") {
                if let Some(i) = x.find('\n') {
                    x = &x[i + 1..];
                    continue;
                } else {
                    return "";
                }
            } else if x.starts_with("/**/") {
                x = &x[4..];
                continue;
            } else if x.starts_with("/*") && (!x.starts_with("/**") || x.starts_with("/***")) && !x.starts_with("/*!") {
                let mut depth = 0;
                let bytes = x.as_bytes();
                let mut i = 0;
                let upper = bytes.len() - 1;
                while i < upper {
                    if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                        depth += 1;
                        i += 1;
                    } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        depth -= 1;
                        if depth == 0 {
                            x = &x[i + 2..];
                            continue 'skip;
                        }
                        i += 1;
                    }
                    i += 1;
                }
                return x;
            }
        }
        match byte {
            b' ' | 0x09..=0x0d => {
                x = &x[1..];
                continue;
            },
            b if b <= 0x7f => {},
            _ => {
                let ch = x.chars().next().unwrap();
                if is_ws(ch) {
                    x = &x[ch.len_utf8()..];
                    continue;
                }
            },
        }
        return x;
    }
    x
}
