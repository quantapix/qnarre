use super::{
    look::{self, Lookahead1, Peek},
    pm2::{Delim, DelimSpan, Span},
    punct::Punctuated,
    tok::Tok,
    *,
};
use std::{
    cell::Cell,
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem,
    ops::Deref,
    rc::Rc,
    str::FromStr,
};

pub struct Parens<'a> {
    pub tok: tok::Paren,
    pub buf: Buffer<'a>,
}
pub fn parse_parens<'a>(x: &Buffer<'a>) -> Res<Parens<'a>> {
    parse_delimited(x, Delim::Parenthesis).map(|(x, buf)| Parens {
        tok: tok::Paren(x),
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
    parse_delimited(x, Delim::None).map(|(x, buf)| Group {
        tok: tok::Group(x.join()),
        buf,
    })
}
fn parse_delimited<'a>(b: &Buffer<'a>, d: Delim) -> Res<(DelimSpan, Buffer<'a>)> {
    b.step(|c| {
        if let Some((y, span, rest)) = c.group(d) {
            let scope = close_span_of_group(*c);
            let nested = advance_step_cursor(c, y);
            let unexpected = get_unexpected(b);
            let y = new_parse_buffer(scope, nested, unexpected);
            Ok(((span, y), rest))
        } else {
            let y = match d {
                Delim::Parenthesis => "expected parentheses",
                Delim::Brace => "expected braces",
                Delim::Bracket => "expected brackets",
                Delim::None => "expected group",
            };
            Err(c.error(y))
        }
    })
}

pub struct Buffer<'a> {
    scope: Span,
    cell: Cell<Cursor<'static>>,
    marker: PhantomData<Cursor<'a>>,
    unexpected: Cell<Option<Rc<Cell<Unexpected>>>>,
}
impl<'a> Buffer<'a> {
    pub fn parse<T: Parse>(&self) -> Res<T> {
        T::parse(self)
    }
    pub fn call<T>(&self, x: fn(Stream) -> Res<T>) -> Res<T> {
        x(self)
    }
    pub fn peek<T: Peek>(&self, x: T) -> bool {
        let _ = x;
        T::Token::peek(self.cursor())
    }
    pub fn peek2<T: Peek>(&self, x: T) -> bool {
        fn peek2(x: &Buffer, f: fn(Cursor) -> bool) -> bool {
            if let Some(x) = x.cursor().group(Delim::None) {
                if x.0.skip().map_or(false, f) {
                    return true;
                }
            }
            x.cursor().skip().map_or(false, f)
        }
        let _ = x;
        peek2(self, T::Token::peek)
    }
    pub fn peek3<T: Peek>(&self, x: T) -> bool {
        fn peek3(x: &Buffer, f: fn(Cursor) -> bool) -> bool {
            if let Some(x) = x.cursor().group(Delim::None) {
                if x.0.skip().and_then(Cursor::skip).map_or(false, f) {
                    return true;
                }
            }
            x.cursor().skip().and_then(Cursor::skip).map_or(false, f)
        }
        let _ = x;
        peek3(self, T::Token::peek)
    }
    pub fn parse_terminated<T, P>(&self, f: fn(Stream) -> Res<T>, sep: P) -> Res<Punctuated<T, P::Token>>
    where
        P: Peek,
        P::Token: Parse,
    {
        let _ = sep;
        Punctuated::parse_terminated_with(self, f)
    }
    pub fn is_empty(&self) -> bool {
        self.cursor().eof()
    }
    pub fn lookahead1(&self) -> Lookahead1<'a> {
        look::new(self.scope, self.cursor())
    }
    pub fn fork(&self) -> Self {
        Buffer {
            scope: self.scope,
            cell: self.cell.clone(),
            marker: PhantomData,
            unexpected: Cell::new(Some(Rc::new(Cell::new(Unexpected::None)))),
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
            cursor: self.cell.get(),
            marker: PhantomData,
        })?;
        self.cell.set(rest);
        Ok(y)
    }
    pub fn span(&self) -> Span {
        let y = self.cursor();
        if y.eof() {
            self.scope
        } else {
            super::open_span_of_group(y)
        }
    }
    pub fn cursor(&self) -> Cursor<'a> {
        self.cell.get()
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
    cursor: Cursor<'c>,
    marker: PhantomData<fn(Cursor<'c>) -> Cursor<'a>>,
}
impl<'c, 'a> Deref for Step<'c, 'a> {
    type Target = Cursor<'c>;
    fn deref(&self) -> &Self::Target {
        &self.cursor
    }
}
impl<'c, 'a> Copy for Step<'c, 'a> {}
impl<'c, 'a> Clone for Step<'c, 'a> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'c, 'a> Step<'c, 'a> {
    pub fn error<T: Display>(self, x: T) -> Err {
        err::new_at(self.scope, self.cursor, x)
    }
}

pub type Stream<'a> = &'a Buffer<'a>;

pub trait Parse: Sized {
    fn parse(x: Stream) -> Res<Self>;
}
impl<T: Parse> Parse for Box<T> {
    fn parse(x: Stream) -> Res<Self> {
        x.parse().map(Box::new)
    }
}
impl<T: Parse + Tok> Parse for Option<T> {
    fn parse(x: Stream) -> Res<Self> {
        if T::peek(x.cursor()) {
            Ok(Some(x.parse()?))
        } else {
            Ok(None)
        }
    }
}
impl Parse for pm2::Stream {
    fn parse(x: Stream) -> Res<Self> {
        x.step(|x| Ok((x.token_stream(), Cursor::empty())))
    }
}
impl Parse for pm2::Tree {
    fn parse(x: Stream) -> Res<Self> {
        x.step(|x| match x.token_tree() {
            Some((tt, rest)) => Ok((tt, rest)),
            None => Err(x.error("expected token tree")),
        })
    }
}
impl Parse for Group {
    fn parse(x: Stream) -> Res<Self> {
        x.step(|x| {
            if let Some((y, rest)) = x.any_group_token() {
                if y.delimiter() != Delim::None {
                    return Ok((y, rest));
                }
            }
            Err(x.error("expected group token"))
        })
    }
}
impl Parse for Punct {
    fn parse(x: Stream) -> Res<Self> {
        x.step(|x| match x.punct() {
            Some((y, rest)) => Ok((y, rest)),
            None => Err(x.error("expected punctuation token")),
        })
    }
}
impl Parse for pm2::Lit {
    fn parse(x: Stream) -> Res<Self> {
        x.step(|x| match x.literal() {
            Some((y, rest)) => Ok((y, rest)),
            None => Err(x.error("expected literal token")),
        })
    }
}

pub trait Parser: Sized {
    type Output;
    fn parse2(self, tokens: pm2::Stream) -> Res<Self::Output>;
    fn parse(self, tokens: pm2::Stream) -> Res<Self::Output> {
        self.parse2(proc_macro2::pm2::Stream::from(tokens))
    }
    fn parse_str(self, s: &str) -> Res<Self::Output> {
        self.parse2(proc_macro2::pm2::Stream::from_str(s)?)
    }
    fn __parse_scoped(self, scope: Span, tokens: pm2::Stream) -> Res<Self::Output> {
        let _ = scope;
        self.parse2(tokens)
    }
}
impl<F, T> Parser for F
where
    F: FnOnce(Stream) -> Res<T>,
{
    type Output = T;
    fn parse2(self, tokens: pm2::Stream) -> Res<T> {
        let buf = cur::Buffer::new2(tokens);
        let state = tokens_to_parse_buffer(&buf);
        let node = self(&state)?;
        state.check_unexpected()?;
        if let Some(unexpected_span) = span_of_unexpected_ignoring_nones(state.cursor()) {
            Err(Err::new(unexpected_span, "unexpected token"))
        } else {
            Ok(node)
        }
    }
    fn __parse_scoped(self, scope: Span, tokens: pm2::Stream) -> Res<Self::Output> {
        let buf = cur::Buffer::new2(tokens);
        let cursor = buf.begin();
        let unexpected = Rc::new(Cell::new(Unexpected::None));
        let state = new_parse_buffer(scope, cursor, unexpected);
        let node = self(&state)?;
        state.check_unexpected()?;
        if let Some(unexpected_span) = span_of_unexpected_ignoring_nones(state.cursor()) {
            Err(Err::new(unexpected_span, "unexpected token"))
        } else {
            Ok(node)
        }
    }
}

pub fn advance_step_cursor<'c, 'a>(proof: Step<'c, 'a>, to: Cursor<'c>) -> Cursor<'a> {
    let _ = proof;
    unsafe { mem::transmute::<Cursor<'c>, Cursor<'a>>(to) }
}
pub fn new_parse_buffer(scope: Span, cursor: Cursor, unexpected: Rc<Cell<Unexpected>>) -> Buffer {
    Buffer {
        scope,
        cell: Cell::new(unsafe { mem::transmute::<Cursor, Cursor<'static>>(cursor) }),
        marker: PhantomData,
        unexpected: Cell::new(Some(unexpected)),
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
    cell_clone(&x.unexpected).unwrap()
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
    let cursor = x.begin();
    let unexpected = Rc::new(Cell::new(Unexpected::None));
    new_parse_buffer(scope, cursor, unexpected)
}

pub fn parse_scoped<F: Parser>(f: F, s: Span, xs: pm2::Stream) -> Res<F::Output> {
    f.__parse_scoped(s, xs)
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
    pub trait Speculative {
        fn advance_to(&self, fork: &Self);
    }
    impl<'a> Speculative for Buffer<'a> {
        fn advance_to(&self, fork: &Self) {
            if !crate::same_scope(self.cursor(), fork.cursor()) {
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
                        fork.unexpected.set(Some(Rc::new(Cell::new(Unexpected::None))));
                    },
                    (_, Some(_)) => {},
                }
            }
            self.cell
                .set(unsafe { mem::transmute::<Cursor, Cursor<'static>>(fork.cursor()) });
        }
    }
    pub trait AnyDelim {
        fn parse_any_delim(&self) -> Res<(Delim, DelimSpan, Buffer)>;
    }
    impl<'a> AnyDelim for Buffer<'a> {
        fn parse_any_delim(&self) -> Res<(Delim, DelimSpan, Buffer)> {
            self.step(|c| {
                if let Some((content, delim, span, rest)) = c.any_group() {
                    let scope = crate::close_span_of_group(*c);
                    let nested = advance_step_cursor(c, content);
                    let unexpected = get_unexpected(self);
                    let content = new_parse_buffer(scope, nested, unexpected);
                    Ok(((delim, span, content), rest))
                } else {
                    Err(c.error("expected any delimiter"))
                }
            })
        }
    }
}
