use crate::buffer::Cursor;
use crate::error::{self, Error};
use crate::sealed::lookahead::Sealed;
use crate::span::IntoSpans;
use crate::token::Token;
use proc_macro2::{Delimiter, Span};
use std::cell::RefCell;
pub struct Lookahead1<'a> {
    scope: Span,
    cursor: Cursor<'a>,
    comparisons: RefCell<Vec<&'static str>>,
}
pub(crate) fn new(scope: Span, cursor: Cursor) -> Lookahead1 {
    Lookahead1 {
        scope,
        cursor,
        comparisons: RefCell::new(Vec::new()),
    }
}
fn peek_impl(lookahead: &Lookahead1, peek: fn(Cursor) -> bool, display: fn() -> &'static str) -> bool {
    if peek(lookahead.cursor) {
        return true;
    }
    lookahead.comparisons.borrow_mut().push(display());
    false
}
impl<'a> Lookahead1<'a> {
    pub fn peek<T: Peek>(&self, token: T) -> bool {
        let _ = token;
        peek_impl(self, T::Token::peek, T::Token::display)
    }
    pub fn error(self) -> Error {
        let comparisons = self.comparisons.borrow();
        match comparisons.len() {
            0 => {
                if self.cursor.eof() {
                    Error::new(self.scope, "unexpected end of input")
                } else {
                    Error::new(self.cursor.span(), "unexpected token")
                }
            },
            1 => {
                let message = format!("expected {}", comparisons[0]);
                error::new_at(self.scope, self.cursor, message)
            },
            2 => {
                let message = format!("expected {} or {}", comparisons[0], comparisons[1]);
                error::new_at(self.scope, self.cursor, message)
            },
            _ => {
                let join = comparisons.join(", ");
                let message = format!("expected one of: {}", join);
                error::new_at(self.scope, self.cursor, message)
            },
        }
    }
}
pub trait Peek: Sealed {
    type Token: Token;
}
impl<F: Copy + FnOnce(TokenMarker) -> T, T: Token> Peek for F {
    type Token = T;
}
pub enum TokenMarker {}
impl<S> IntoSpans<S> for TokenMarker {
    fn into_spans(self) -> S {
        match self {}
    }
}
pub(crate) fn is_delimiter(cursor: Cursor, delimiter: Delimiter) -> bool {
    cursor.group(delimiter).is_some()
}
impl<F: Copy + FnOnce(TokenMarker) -> T, T: Token> Sealed for F {}
