use crate::buffer::Cursor;
use crate::thread::ThreadBound;
use proc_macro2::{Delimiter, Group, Ident, LexError, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
use quote::ToTokens;
use std::fmt::{self, Debug, Display};
use std::slice;
use std::vec;
pub type Result<T> = std::result::Result<T, Error>;
pub struct Error {
    messages: Vec<ErrorMessage>,
}
struct ErrorMessage {
    span: ThreadBound<SpanRange>,
    message: String,
}
struct SpanRange {
    start: Span,
    end: Span,
}
#[cfg(test)]
struct _Test
where
    Error: Send + Sync;
impl Error {
    pub fn new<T: Display>(span: Span, message: T) -> Self {
        return new(span, message.to_string());
        fn new(span: Span, message: String) -> Error {
            Error {
                messages: vec![ErrorMessage {
                    span: ThreadBound::new(SpanRange { start: span, end: span }),
                    message,
                }],
            }
        }
    }
    pub fn new_spanned<T: ToTokens, U: Display>(tokens: T, message: U) -> Self {
        return new_spanned(tokens.into_token_stream(), message.to_string());
        fn new_spanned(tokens: TokenStream, message: String) -> Error {
            let mut iter = tokens.into_iter();
            let start = iter.next().map_or_else(Span::call_site, |t| t.span());
            let end = iter.last().map_or(start, |t| t.span());
            Error {
                messages: vec![ErrorMessage {
                    span: ThreadBound::new(SpanRange { start, end }),
                    message,
                }],
            }
        }
    }
    pub fn span(&self) -> Span {
        let SpanRange { start, end } = match self.messages[0].span.get() {
            Some(span) => *span,
            None => return Span::call_site(),
        };
        start.join(end).unwrap_or(start)
    }
    pub fn to_compile_error(&self) -> TokenStream {
        self.messages.iter().map(ErrorMessage::to_compile_error).collect()
    }
    pub fn into_compile_error(self) -> TokenStream {
        self.to_compile_error()
    }
    pub fn combine(&mut self, another: Error) {
        self.messages.extend(another.messages);
    }
}
impl ErrorMessage {
    fn to_compile_error(&self) -> TokenStream {
        let (start, end) = match self.span.get() {
            Some(range) => (range.start, range.end),
            None => (Span::call_site(), Span::call_site()),
        };
        TokenStream::from_iter(vec![
            TokenTree::Punct({
                let mut punct = Punct::new(':', Spacing::Joint);
                punct.set_span(start);
                punct
            }),
            TokenTree::Punct({
                let mut punct = Punct::new(':', Spacing::Alone);
                punct.set_span(start);
                punct
            }),
            TokenTree::Ident(Ident::new("core", start)),
            TokenTree::Punct({
                let mut punct = Punct::new(':', Spacing::Joint);
                punct.set_span(start);
                punct
            }),
            TokenTree::Punct({
                let mut punct = Punct::new(':', Spacing::Alone);
                punct.set_span(start);
                punct
            }),
            TokenTree::Ident(Ident::new("compile_error", start)),
            TokenTree::Punct({
                let mut punct = Punct::new('!', Spacing::Alone);
                punct.set_span(start);
                punct
            }),
            TokenTree::Group({
                let mut group = Group::new(Delimiter::Brace, {
                    TokenStream::from_iter(vec![TokenTree::Literal({
                        let mut string = Literal::string(&self.message);
                        string.set_span(end);
                        string
                    })])
                });
                group.set_span(end);
                group
            }),
        ])
    }
}
pub(crate) fn new_at<T: Display>(scope: Span, cursor: Cursor, message: T) -> Error {
    if cursor.eof() {
        Error::new(scope, format!("unexpected end of input, {}", message))
    } else {
        let span = crate::buffer::open_span_of_group(cursor);
        Error::new(span, message)
    }
}
pub(crate) fn new2<T: Display>(start: Span, end: Span, message: T) -> Error {
    return new2(start, end, message.to_string());
    fn new2(start: Span, end: Span, message: String) -> Error {
        Error {
            messages: vec![ErrorMessage {
                span: ThreadBound::new(SpanRange { start, end }),
                message,
            }],
        }
    }
}
impl Debug for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        if self.messages.len() == 1 {
            formatter.debug_tuple("Error").field(&self.messages[0]).finish()
        } else {
            formatter.debug_tuple("Error").field(&self.messages).finish()
        }
    }
}
impl Debug for ErrorMessage {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.message, formatter)
    }
}
impl Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(&self.messages[0].message)
    }
}
impl Clone for Error {
    fn clone(&self) -> Self {
        Error {
            messages: self.messages.clone(),
        }
    }
}
impl Clone for ErrorMessage {
    fn clone(&self) -> Self {
        ErrorMessage {
            span: self.span,
            message: self.message.clone(),
        }
    }
}
impl Clone for SpanRange {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for SpanRange {}
impl std::error::Error for Error {}
impl From<LexError> for Error {
    fn from(err: LexError) -> Self {
        Error::new(err.span(), "lex error")
    }
}
impl IntoIterator for Error {
    type Item = Error;
    type IntoIter = IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            messages: self.messages.into_iter(),
        }
    }
}
pub struct IntoIter {
    messages: vec::IntoIter<ErrorMessage>,
}
impl Iterator for IntoIter {
    type Item = Error;
    fn next(&mut self) -> Option<Self::Item> {
        Some(Error {
            messages: vec![self.messages.next()?],
        })
    }
}
impl<'a> IntoIterator for &'a Error {
    type Item = Error;
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        Iter {
            messages: self.messages.iter(),
        }
    }
}
pub struct Iter<'a> {
    messages: slice::Iter<'a, ErrorMessage>,
}
impl<'a> Iterator for Iter<'a> {
    type Item = Error;
    fn next(&mut self) -> Option<Self::Item> {
        Some(Error {
            messages: vec![self.messages.next()?.clone()],
        })
    }
}
impl Extend<Error> for Error {
    fn extend<T: IntoIterator<Item = Error>>(&mut self, iter: T) {
        for err in iter {
            self.combine(err);
        }
    }
}
