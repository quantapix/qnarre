use super::{
    err::{self, Err, Result},
    lookahead::{self, Lookahead1, Peek},
    proc_macro,
    punctuated::Punctuated,
    tok::Tok,
    Cursor, TokBuff,
};
use proc_macro2::{self, Delimiter, Group, Literal, Punct, Span, TokenStream, TokenTree};
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

pub mod discouraged {
    use super::*;
    use proc_macro2::extra::DelimSpan;
    pub trait Speculative {
        fn advance_to(&self, fork: &Self);
    }
    impl<'a> Speculative for ParseBuffer<'a> {
        fn advance_to(&self, fork: &Self) {
            if !crate::same_scope(self.cursor(), fork.cursor()) {
                panic!("Fork was not derived from the advancing parse stream");
            }
            let (self_unexp, self_sp) = inner_unexpected(self);
            let (fork_unexp, fork_sp) = inner_unexpected(fork);
            if !Rc::ptr_eq(&self_unexp, &fork_unexp) {
                match (fork_sp, self_sp) {
                    (Some(span), None) => {
                        self_unexp.set(Unexpected::Some(span));
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
    pub trait AnyDelimiter {
        fn parse_any_delimiter(&self) -> Result<(Delimiter, DelimSpan, ParseBuffer)>;
    }
    impl<'a> AnyDelimiter for ParseBuffer<'a> {
        fn parse_any_delimiter(&self) -> Result<(Delimiter, DelimSpan, ParseBuffer)> {
            self.step(|cursor| {
                if let Some((content, delimiter, span, rest)) = cursor.any_group() {
                    let scope = crate::close_span_of_group(*cursor);
                    let nested = crate::parse::advance_step_cursor(cursor, content);
                    let unexpected = crate::parse::get_unexpected(self);
                    let content = crate::parse::new_parse_buffer(scope, nested, unexpected);
                    Ok(((delimiter, span, content), rest))
                } else {
                    Err(cursor.error("expected any delimiter"))
                }
            })
        }
    }
}

pub trait Parse: Sized {
    fn parse(input: ParseStream) -> Result<Self>;
}
pub type ParseStream<'a> = &'a ParseBuffer<'a>;
pub struct ParseBuffer<'a> {
    scope: Span,
    cell: Cell<Cursor<'static>>,
    marker: PhantomData<Cursor<'a>>,
    unexpected: Cell<Option<Rc<Cell<Unexpected>>>>,
}
impl<'a> Drop for ParseBuffer<'a> {
    fn drop(&mut self) {
        if let Some(unexpected_span) = span_of_unexpected_ignoring_nones(self.cursor()) {
            let (inner, old_span) = inner_unexpected(self);
            if old_span.is_none() {
                inner.set(Unexpected::Some(unexpected_span));
            }
        }
    }
}
impl<'a> Display for ParseBuffer<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.cursor().token_stream(), f)
    }
}
impl<'a> Debug for ParseBuffer<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.cursor().token_stream(), f)
    }
}
pub struct StepCursor<'c, 'a> {
    scope: Span,
    cursor: Cursor<'c>,
    marker: PhantomData<fn(Cursor<'c>) -> Cursor<'a>>,
}
impl<'c, 'a> Deref for StepCursor<'c, 'a> {
    type Target = Cursor<'c>;
    fn deref(&self) -> &Self::Target {
        &self.cursor
    }
}
impl<'c, 'a> Copy for StepCursor<'c, 'a> {}
impl<'c, 'a> Clone for StepCursor<'c, 'a> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'c, 'a> StepCursor<'c, 'a> {
    pub fn error<T: Display>(self, message: T) -> Err {
        err::new_at(self.scope, self.cursor, message)
    }
}
pub(crate) fn advance_step_cursor<'c, 'a>(proof: StepCursor<'c, 'a>, to: Cursor<'c>) -> Cursor<'a> {
    let _ = proof;
    unsafe { mem::transmute::<Cursor<'c>, Cursor<'a>>(to) }
}
pub(crate) fn new_parse_buffer(scope: Span, cursor: Cursor, unexpected: Rc<Cell<Unexpected>>) -> ParseBuffer {
    ParseBuffer {
        scope,
        cell: Cell::new(unsafe { mem::transmute::<Cursor, Cursor<'static>>(cursor) }),
        marker: PhantomData,
        unexpected: Cell::new(Some(unexpected)),
    }
}
pub(crate) enum Unexpected {
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
        match self {
            Unexpected::None => Unexpected::None,
            Unexpected::Some(span) => Unexpected::Some(*span),
            Unexpected::Chain(next) => Unexpected::Chain(next.clone()),
        }
    }
}
fn cell_clone<T: Default + Clone>(cell: &Cell<T>) -> T {
    let prev = cell.take();
    let ret = prev.clone();
    cell.set(prev);
    ret
}
fn inner_unexpected(buffer: &ParseBuffer) -> (Rc<Cell<Unexpected>>, Option<Span>) {
    let mut unexpected = get_unexpected(buffer);
    loop {
        match cell_clone(&unexpected) {
            Unexpected::None => return (unexpected, None),
            Unexpected::Some(span) => return (unexpected, Some(span)),
            Unexpected::Chain(next) => unexpected = next,
        }
    }
}
pub(crate) fn get_unexpected(buffer: &ParseBuffer) -> Rc<Cell<Unexpected>> {
    cell_clone(&buffer.unexpected).unwrap()
}
fn span_of_unexpected_ignoring_nones(mut cursor: Cursor) -> Option<Span> {
    if cursor.eof() {
        return None;
    }
    while let Some((inner, _span, rest)) = cursor.group(Delimiter::None) {
        if let Some(unexpected) = span_of_unexpected_ignoring_nones(inner) {
            return Some(unexpected);
        }
        cursor = rest;
    }
    if cursor.eof() {
        None
    } else {
        Some(cursor.span())
    }
}
impl<'a> ParseBuffer<'a> {
    pub fn parse<T: Parse>(&self) -> Result<T> {
        T::parse(self)
    }
    pub fn call<T>(&self, function: fn(ParseStream) -> Result<T>) -> Result<T> {
        function(self)
    }
    pub fn peek<T: Peek>(&self, token: T) -> bool {
        let _ = token;
        T::Token::peek(self.cursor())
    }
    pub fn peek2<T: Peek>(&self, token: T) -> bool {
        fn peek2(buffer: &ParseBuffer, peek: fn(Cursor) -> bool) -> bool {
            if let Some(group) = buffer.cursor().group(Delimiter::None) {
                if group.0.skip().map_or(false, peek) {
                    return true;
                }
            }
            buffer.cursor().skip().map_or(false, peek)
        }
        let _ = token;
        peek2(self, T::Token::peek)
    }
    pub fn peek3<T: Peek>(&self, token: T) -> bool {
        fn peek3(buffer: &ParseBuffer, peek: fn(Cursor) -> bool) -> bool {
            if let Some(group) = buffer.cursor().group(Delimiter::None) {
                if group.0.skip().and_then(Cursor::skip).map_or(false, peek) {
                    return true;
                }
            }
            buffer.cursor().skip().and_then(Cursor::skip).map_or(false, peek)
        }
        let _ = token;
        peek3(self, T::Token::peek)
    }
    pub fn parse_terminated<T, P>(
        &self,
        parser: fn(ParseStream) -> Result<T>,
        separator: P,
    ) -> Result<Punctuated<T, P::Token>>
    where
        P: Peek,
        P::Token: Parse,
    {
        let _ = separator;
        Punctuated::parse_terminated_with(self, parser)
    }
    pub fn is_empty(&self) -> bool {
        self.cursor().eof()
    }
    pub fn lookahead1(&self) -> Lookahead1<'a> {
        lookahead::new(self.scope, self.cursor())
    }
    pub fn fork(&self) -> Self {
        ParseBuffer {
            scope: self.scope,
            cell: self.cell.clone(),
            marker: PhantomData,
            unexpected: Cell::new(Some(Rc::new(Cell::new(Unexpected::None)))),
        }
    }
    pub fn error<T: Display>(&self, message: T) -> Err {
        err::new_at(self.scope, self.cursor(), message)
    }
    pub fn step<F, R>(&self, function: F) -> Result<R>
    where
        F: for<'c> FnOnce(StepCursor<'c, 'a>) -> Result<(R, Cursor<'c>)>,
    {
        let (node, rest) = function(StepCursor {
            scope: self.scope,
            cursor: self.cell.get(),
            marker: PhantomData,
        })?;
        self.cell.set(rest);
        Ok(node)
    }
    pub fn span(&self) -> Span {
        let cursor = self.cursor();
        if cursor.eof() {
            self.scope
        } else {
            super::open_span_of_group(cursor)
        }
    }
    pub fn cursor(&self) -> Cursor<'a> {
        self.cell.get()
    }
    fn check_unexpected(&self) -> Result<()> {
        match inner_unexpected(self).1 {
            Some(span) => Err(Err::new(span, "unexpected token")),
            None => Ok(()),
        }
    }
}

impl<T: Parse> Parse for Box<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        input.parse().map(Box::new)
    }
}

impl<T: Parse + Tok> Parse for Option<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        if T::peek(input.cursor()) {
            Ok(Some(input.parse()?))
        } else {
            Ok(None)
        }
    }
}

impl Parse for TokenStream {
    fn parse(input: ParseStream) -> Result<Self> {
        input.step(|cursor| Ok((cursor.token_stream(), Cursor::empty())))
    }
}

impl Parse for TokenTree {
    fn parse(input: ParseStream) -> Result<Self> {
        input.step(|cursor| match cursor.token_tree() {
            Some((tt, rest)) => Ok((tt, rest)),
            None => Err(cursor.error("expected token tree")),
        })
    }
}

impl Parse for Group {
    fn parse(input: ParseStream) -> Result<Self> {
        input.step(|cursor| {
            if let Some((group, rest)) = cursor.any_group_token() {
                if group.delimiter() != Delimiter::None {
                    return Ok((group, rest));
                }
            }
            Err(cursor.error("expected group token"))
        })
    }
}

impl Parse for Punct {
    fn parse(input: ParseStream) -> Result<Self> {
        input.step(|cursor| match cursor.punct() {
            Some((punct, rest)) => Ok((punct, rest)),
            None => Err(cursor.error("expected punctuation token")),
        })
    }
}

impl Parse for Literal {
    fn parse(input: ParseStream) -> Result<Self> {
        input.step(|cursor| match cursor.literal() {
            Some((literal, rest)) => Ok((literal, rest)),
            None => Err(cursor.error("expected literal token")),
        })
    }
}
pub trait Parser: Sized {
    type Output;
    fn parse2(self, tokens: TokenStream) -> Result<Self::Output>;
    fn parse(self, tokens: proc_macro::TokenStream) -> Result<Self::Output> {
        self.parse2(proc_macro2::TokenStream::from(tokens))
    }
    fn parse_str(self, s: &str) -> Result<Self::Output> {
        self.parse2(proc_macro2::TokenStream::from_str(s)?)
    }
    fn __parse_scoped(self, scope: Span, tokens: TokenStream) -> Result<Self::Output> {
        let _ = scope;
        self.parse2(tokens)
    }
}
fn tokens_to_parse_buffer(tokens: &TokBuff) -> ParseBuffer {
    let scope = Span::call_site();
    let cursor = tokens.begin();
    let unexpected = Rc::new(Cell::new(Unexpected::None));
    new_parse_buffer(scope, cursor, unexpected)
}
impl<F, T> Parser for F
where
    F: FnOnce(ParseStream) -> Result<T>,
{
    type Output = T;
    fn parse2(self, tokens: TokenStream) -> Result<T> {
        let buf = TokBuff::new2(tokens);
        let state = tokens_to_parse_buffer(&buf);
        let node = self(&state)?;
        state.check_unexpected()?;
        if let Some(unexpected_span) = span_of_unexpected_ignoring_nones(state.cursor()) {
            Err(Err::new(unexpected_span, "unexpected token"))
        } else {
            Ok(node)
        }
    }
    fn __parse_scoped(self, scope: Span, tokens: TokenStream) -> Result<Self::Output> {
        let buf = TokBuff::new2(tokens);
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

pub(crate) fn parse_scoped<F: Parser>(f: F, scope: Span, tokens: TokenStream) -> Result<F::Output> {
    f.__parse_scoped(scope, tokens)
}
pub struct Nothing;
impl Parse for Nothing {
    fn parse(_input: ParseStream) -> Result<Self> {
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
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Hash for Nothing {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}
