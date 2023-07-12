use proc_macro2::{LexError, Punct};
use std::{
    fmt::{self, Debug, Display},
    slice, vec,
};

pub struct Err {
    msgs: Vec<ErrMsg>,
}
pub type Res<T> = std::result::Result<T, Err>;
struct ErrMsg {
    span: ThreadBound<SpanRange>,
    message: String,
}
struct SpanRange {
    start: pm2::Span,
    end: pm2::Span,
}
#[cfg(test)]
struct _Test
where
    Err: Send + Sync;
impl Err {
    pub fn new<T: Display>(span: pm2::Span, message: T) -> Self {
        return new(span, message.to_string());
        fn new(span: pm2::Span, message: String) -> Err {
            Err {
                msgs: vec![ErrMsg {
                    span: ThreadBound::new(SpanRange { start: span, end: span }),
                    message,
                }],
            }
        }
    }
    pub fn new_spanned<T: ToTokens, U: Display>(tokens: T, message: U) -> Self {
        return new_spanned(tokens.into_token_stream(), message.to_string());
        fn new_spanned(tokens: pm2::Stream, message: String) -> Err {
            let mut iter = tokens.into_iter();
            let start = iter.next().map_or_else(pm2::Span::call_site, |t| t.span());
            let end = iter.last().map_or(start, |t| t.span());
            Err {
                msgs: vec![ErrMsg {
                    span: ThreadBound::new(SpanRange { start, end }),
                    message,
                }],
            }
        }
    }
    pub fn span(&self) -> pm2::Span {
        let SpanRange { start, end } = match self.msgs[0].span.get() {
            Some(span) => *span,
            None => return pm2::Span::call_site(),
        };
        start.join(end).unwrap_or(start)
    }
    pub fn to_compile_error(&self) -> pm2::Stream {
        self.msgs.iter().map(ErrMsg::to_compile_error).collect()
    }
    pub fn into_compile_error(self) -> pm2::Stream {
        self.to_compile_error()
    }
    pub fn combine(&mut self, another: Err) {
        self.msgs.extend(another.msgs);
    }
}
impl ErrMsg {
    fn to_compile_error(&self) -> pm2::Stream {
        let (start, end) = match self.span.get() {
            Some(range) => (range.start, range.end),
            None => (pm2::Span::call_site(), pm2::Span::call_site()),
        };
        use pm2::{Spacing::*, Tree::*};
        pm2::Stream::from_iter(vec![
            pm2::Tree::Punct({
                let y = Punct::new(':', Joint);
                y.set_span(start);
                y
            }),
            pm2::Tree::Punct({
                let y = Punct::new(':', Alone);
                y.set_span(start);
                y
            }),
            pm2::Tree::Ident(Ident::new("core", start)),
            pm2::Tree::Punct({
                let y = Punct::new(':', Joint);
                y.set_span(start);
                y
            }),
            pm2::Tree::Punct({
                let y = Punct::new(':', Alone);
                y.set_span(start);
                y
            }),
            pm2::Tree::Ident(Ident::new("compile_error", start)),
            pm2::Tree::Punct({
                let y = Punct::new('!', Alone);
                y.set_span(start);
                y
            }),
            pm2::Tree::Group({
                let y = Group::new(pm2::Delim::Brace, {
                    pm2::Stream::from_iter(vec![pm2::Tree::Literal({
                        let y = pm2::Lit::string(&self.message);
                        y.set_span(end);
                        y
                    })])
                });
                y.set_span(end);
                y
            }),
        ])
    }
}
pub fn new_at<T: Display>(scope: pm2::Span, cursor: Cursor, message: T) -> Err {
    if cursor.eof() {
        Err::new(scope, format!("unexpected end of input, {}", message))
    } else {
        let span = super::open_span_of_group(cursor);
        Err::new(span, message)
    }
}
pub fn new2<T: Display>(start: pm2::Span, end: pm2::Span, message: T) -> Err {
    return new2(start, end, message.to_string());
    fn new2(start: pm2::Span, end: pm2::Span, message: String) -> Err {
        Err {
            msgs: vec![ErrMsg {
                span: ThreadBound::new(SpanRange { start, end }),
                message,
            }],
        }
    }
}
impl Debug for Err {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.msgs.len() == 1 {
            f.debug_tuple("Error").field(&self.msgs[0]).finish()
        } else {
            f.debug_tuple("Error").field(&self.msgs).finish()
        }
    }
}
impl Debug for ErrMsg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.message, f)
    }
}
impl Display for Err {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.msgs[0].message)
    }
}
impl Clone for Err {
    fn clone(&self) -> Self {
        Err {
            msgs: self.msgs.clone(),
        }
    }
}
impl Clone for ErrMsg {
    fn clone(&self) -> Self {
        ErrMsg {
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
impl std::error::Error for Err {}
impl From<LexError> for Err {
    fn from(x: LexError) -> Self {
        Err::new(x.span(), "lex error")
    }
}
impl IntoIterator for Err {
    type Item = Err;
    type IntoIter = IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            messages: self.msgs.into_iter(),
        }
    }
}
pub struct IntoIter {
    messages: vec::IntoIter<ErrMsg>,
}
impl Iterator for IntoIter {
    type Item = Err;
    fn next(&mut self) -> Option<Self::Item> {
        Some(Err {
            msgs: vec![self.messages.next()?],
        })
    }
}
impl<'a> IntoIterator for &'a Err {
    type Item = Err;
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        Iter {
            messages: self.msgs.iter(),
        }
    }
}
pub struct Iter<'a> {
    messages: slice::Iter<'a, ErrMsg>,
}
impl<'a> Iterator for Iter<'a> {
    type Item = Err;
    fn next(&mut self) -> Option<Self::Item> {
        Some(Err {
            msgs: vec![self.messages.next()?.clone()],
        })
    }
}
impl Extend<Err> for Err {
    fn extend<T: IntoIterator<Item = Err>>(&mut self, iter: T) {
        for err in iter {
            self.combine(err);
        }
    }
}
