use super::*;
use std::{
    slice,
    thread::{self, ThreadId},
    vec,
};

pub type Res<T> = std::result::Result<T, Err>;

pub struct Err {
    msgs: Vec<ErrMsg>,
}
impl Err {
    pub fn new<T: Display>(span: pm2::Span, msg: T) -> Self {
        return new(span, msg.to_string());
        fn new(span: pm2::Span, msg: String) -> Err {
            Err {
                msgs: vec![ErrMsg {
                    span: ThreadBound::new(SpanRange { beg: span, end: span }),
                    msg,
                }],
            }
        }
    }
    pub fn new_spanned<T: ToStream, U: Display>(tokens: T, msg: U) -> Self {
        return new_spanned(tokens.into_stream(), msg.to_string());
        fn new_spanned(tokens: pm2::Stream, msg: String) -> Err {
            let mut iter = tokens.into_iter();
            let beg = iter.next().map_or_else(pm2::Span::call_site, |t| t.span());
            let end = iter.last().map_or(beg, |t| t.span());
            Err {
                msgs: vec![ErrMsg {
                    span: ThreadBound::new(SpanRange { beg, end }),
                    msg,
                }],
            }
        }
    }
    pub fn span(&self) -> pm2::Span {
        let SpanRange { beg: start, end } = match self.msgs[0].span.get() {
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
impl Clone for Err {
    fn clone(&self) -> Self {
        Err {
            msgs: self.msgs.clone(),
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
impl Display for Err {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.msgs[0].msg)
    }
}
impl std::error::Error for Err {}
impl From<pm2::LexError> for Err {
    fn from(x: pm2::LexError) -> Self {
        Err::new(x.span(), "lex error")
    }
}
impl IntoIterator for Err {
    type Item = Err;
    type IntoIter = IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            msgs: self.msgs.into_iter(),
        }
    }
}
impl Extend<Err> for Err {
    fn extend<T: IntoIterator<Item = Err>>(&mut self, iter: T) {
        for err in iter {
            self.combine(err);
        }
    }
}

pub struct Iter<'a> {
    msgs: slice::Iter<'a, ErrMsg>,
}
impl<'a> Iterator for Iter<'a> {
    type Item = Err;
    fn next(&mut self) -> Option<Self::Item> {
        Some(Err {
            msgs: vec![self.msgs.next()?.clone()],
        })
    }
}

pub struct IntoIter {
    msgs: vec::IntoIter<ErrMsg>,
}
impl Iterator for IntoIter {
    type Item = Err;
    fn next(&mut self) -> Option<Self::Item> {
        Some(Err {
            msgs: vec![self.msgs.next()?],
        })
    }
}
impl<'a> IntoIterator for &'a Err {
    type Item = Err;
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        Iter { msgs: self.msgs.iter() }
    }
}

struct ErrMsg {
    span: ThreadBound<SpanRange>,
    msg: String,
}
impl ErrMsg {
    fn to_compile_error(&self) -> pm2::Stream {
        let (start, end) = match self.span.get() {
            Some(x) => (x.beg, x.end),
            None => (pm2::Span::call_site(), pm2::Span::call_site()),
        };
        use pm2::{Spacing::*, Tree::*};
        pm2::Stream::from_iter(vec![
            pm2::Tree::Punct({
                let mut y = pm2::Punct::new(':', Joint);
                y.set_span(start);
                y
            }),
            pm2::Tree::Punct({
                let mut y = pm2::Punct::new(':', Alone);
                y.set_span(start);
                y
            }),
            pm2::Tree::Ident(pm2::Ident::new("core", start)),
            pm2::Tree::Punct({
                let mut y = pm2::Punct::new(':', Joint);
                y.set_span(start);
                y
            }),
            pm2::Tree::Punct({
                let mut y = pm2::Punct::new(':', Alone);
                y.set_span(start);
                y
            }),
            pm2::Tree::Ident(pm2::Ident::new("compile_error", start)),
            pm2::Tree::Punct({
                let mut y = pm2::Punct::new('!', Alone);
                y.set_span(start);
                y
            }),
            pm2::Tree::Group({
                let mut y = pm2::Group::new(pm2::Delim::Brace, {
                    pm2::Stream::from_iter(vec![pm2::Tree::Lit({
                        let mut y = pm2::Lit::string(&self.msg);
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
impl Clone for ErrMsg {
    fn clone(&self) -> Self {
        ErrMsg {
            span: self.span,
            msg: self.msg.clone(),
        }
    }
}
impl Debug for ErrMsg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.msg, f)
    }
}

struct ThreadBound<T> {
    val: T,
    id: ThreadId,
}
impl<T> ThreadBound<T> {
    pub fn new(val: T) -> Self {
        ThreadBound {
            val,
            id: thread::current().id(),
        }
    }
    pub fn get(&self) -> Option<&T> {
        if thread::current().id() == self.id {
            Some(&self.val)
        } else {
            None
        }
    }
}
impl<T: Debug> Debug for ThreadBound<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.get() {
            Some(x) => Debug::fmt(x, f),
            None => f.write_str("unknown"),
        }
    }
}
impl<T: Copy> Copy for ThreadBound<T> {}
impl<T: Copy> Clone for ThreadBound<T> {
    fn clone(&self) -> Self {
        *self
    }
}
unsafe impl<T> Sync for ThreadBound<T> {}
unsafe impl<T: Copy> Send for ThreadBound<T> {}

struct SpanRange {
    beg: pm2::Span,
    end: pm2::Span,
}
impl Clone for SpanRange {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for SpanRange {}

pub fn new_at<T: Display>(scope: pm2::Span, cursor: Cursor, msg: T) -> Err {
    if cursor.eof() {
        Err::new(scope, format!("unexpected end of input, {}", msg))
    } else {
        let span = cur::open_span_of_group(cursor);
        Err::new(span, msg)
    }
}
pub fn new2<T: Display>(beg: pm2::Span, end: pm2::Span, msg: T) -> Err {
    fn doit(beg: pm2::Span, end: pm2::Span, msg: String) -> Err {
        Err {
            msgs: vec![ErrMsg {
                span: ThreadBound::new(SpanRange { beg, end }),
                msg,
            }],
        }
    }
    return doit(beg, end, msg.to_string());
}

#[cfg(test)]
struct _Test
where
    Err: Send + Sync;
