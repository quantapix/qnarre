extern crate proc_macro;

pub use quote::ToTokens;
use quote::{quote, spanned, TokenStreamExt};
pub use std::{
    cmp::{self, Ordering},
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{self, Deref, DerefMut},
};

mod pm2 {
    pub use proc_macro2::{
        extra::DelimSpan, Delimiter as Delim, Group, Ident, LexError, Literal as Lit, Punct, Spacing, Span,
        TokenStream as Stream, TokenTree as Tree,
    };
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
}

#[macro_use]
mod mac;

mod attr;
mod cur;
mod data;
mod err;
mod expr;
mod gen;
mod ident;
mod item;
mod lit;
mod meta;
mod parse;
mod pat;
mod path;
mod punct;
mod stmt;
mod tok;
mod typ;

use cur::Cursor;
use data::DeriveInput;
use err::{Err, Res};
use ident::Life;
use parse::{Parse, Parser, Stream};
use path::Path;
use pm2::{Ident, IntoSpans, Punct};
use punct::Puncted;
use tok::Tok;

mod look {
    use super::{sealed::look::Sealed, *};
    use std::cell::RefCell;
    pub struct Look1<'a> {
        scope: pm2::Span,
        cur: Cursor<'a>,
        comps: RefCell<Vec<&'static str>>,
    }
    impl<'a> Look1<'a> {
        pub fn peek<T: Peek>(&self, _: T) -> bool {
            fn doit(x: &Look1, f: fn(Cursor) -> bool, d: fn() -> &'static str) -> bool {
                if f(x.cur) {
                    return true;
                }
                x.comps.borrow_mut().push(d());
                false
            }
            doit(self, T::Token::peek, T::Token::display)
        }
        pub fn err(self) -> Err {
            let ys = self.comps.borrow();
            match ys.len() {
                0 => {
                    if self.cur.eof() {
                        Err::new(self.scope, "unexpected end of input")
                    } else {
                        Err::new(self.cur.span(), "unexpected token")
                    }
                },
                1 => {
                    let y = format!("expected {}", ys[0]);
                    err::new_at(self.scope, self.cur, y)
                },
                2 => {
                    let y = format!("expected {} or {}", ys[0], ys[1]);
                    err::new_at(self.scope, self.cur, y)
                },
                _ => {
                    let y = ys.join(", ");
                    let y = format!("expected one of: {}", y);
                    err::new_at(self.scope, self.cur, y)
                },
            }
        }
    }

    pub fn new(scope: pm2::Span, cur: Cursor) -> Look1 {
        Look1 {
            scope,
            cur,
            comps: RefCell::new(Vec::new()),
        }
    }

    pub trait Peek: Sealed {
        type Token: Tok;
    }
    impl<F: Copy + FnOnce(Marker) -> T, T: Tok> Peek for F {
        type Token = T;
    }

    pub enum Marker {}
    impl<S> IntoSpans<S> for Marker {
        fn into_spans(self) -> S {
            match self {}
        }
    }

    pub fn is_delim(x: Cursor, d: pm2::Delim) -> bool {
        x.group(d).is_some()
    }

    impl<F: Copy + FnOnce(Marker) -> T, T: Tok> Sealed for F {}
}
use look::{Look1, Peek};

struct TokensOrDefault<'a, T: 'a>(pub &'a Option<T>);
impl<'a, T> ToTokens for TokensOrDefault<'a, T>
where
    T: ToTokens + Default,
{
    fn to_tokens(&self, ys: &mut Stream) {
        match self.0 {
            Some(x) => x.to_tokens(ys),
            None => T::default().to_tokens(ys),
        }
    }
}

mod sealed {
    pub mod look {
        pub trait Sealed: Copy {}
    }
}

pub trait Spanned: private::Sealed {
    fn span(&self) -> pm2::Span;
}
impl<T: ?Sized + spanned::Spanned> Spanned for T {
    fn span(&self) -> pm2::Span {
        self.__span()
    }
}
mod private {
    pub trait Sealed {}
    impl<T: ?Sized + spanned::Spanned> Sealed for T {}
}

struct TokenTreeHelper<'a>(pub &'a pm2::Tree);
impl<'a> PartialEq for TokenTreeHelper<'a> {
    fn eq(&self, other: &Self) -> bool {
        use pm2::{Delim::*, Spacing::*};
        match (self.0, other.0) {
            (pm2::Tree::Group(g1), pm2::Tree::Group(g2)) => {
                match (g1.delimiter(), g2.delimiter()) {
                    (Parenthesis, Parenthesis) | (Brace, Brace) | (Bracket, Bracket) | (None, None) => {},
                    _ => return false,
                }
                let s1 = g1.stream().into_iter();
                let mut s2 = g2.stream().into_iter();
                for item1 in s1 {
                    let item2 = match s2.next() {
                        Some(x) => x,
                        None => return false,
                    };
                    if TokenTreeHelper(&item1) != TokenTreeHelper(&item2) {
                        return false;
                    }
                }
                s2.next().is_none()
            },
            (pm2::Tree::Punct(o1), pm2::Tree::Punct(o2)) => {
                o1.as_char() == o2.as_char()
                    && match (o1.spacing(), o2.spacing()) {
                        (Alone, Alone) | (Joint, Joint) => true,
                        _ => false,
                    }
            },
            (pm2::Tree::Literal(l1), pm2::Tree::Literal(l2)) => l1.to_string() == l2.to_string(),
            (pm2::Tree::Ident(s1), pm2::Tree::Ident(s2)) => s1 == s2,
            _ => false,
        }
    }
}
impl<'a> Hash for TokenTreeHelper<'a> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        match self.0 {
            pm2::Tree::Group(g) => {
                0u8.hash(h);
                use pm2::Delim::*;
                match g.delimiter() {
                    Parenthesis => 0u8.hash(h),
                    Brace => 1u8.hash(h),
                    Bracket => 2u8.hash(h),
                    None => 3u8.hash(h),
                }
                for item in g.stream() {
                    TokenTreeHelper(&item).hash(h);
                }
                0xffu8.hash(h); // terminator w/ a variant we don't normally hash
            },
            pm2::Tree::Punct(op) => {
                1u8.hash(h);
                op.as_char().hash(h);
                use pm2::Spacing::*;
                match op.spacing() {
                    Alone => 0u8.hash(h),
                    Joint => 1u8.hash(h),
                }
            },
            pm2::Tree::Literal(x) => (2u8, x.to_string()).hash(h),
            pm2::Tree::Ident(x) => (3u8, x).hash(h),
        }
    }
}

struct TokenStreamHelper<'a>(pub &'a pm2::Stream);
impl<'a> PartialEq for TokenStreamHelper<'a> {
    fn eq(&self, other: &Self) -> bool {
        let left = self.0.clone().into_iter().collect::<Vec<_>>();
        let right = other.0.clone().into_iter().collect::<Vec<_>>();
        if left.len() != right.len() {
            return false;
        }
        for (a, b) in left.into_iter().zip(right) {
            if TokenTreeHelper(&a) != TokenTreeHelper(&b) {
                return false;
            }
        }
        true
    }
}
impl<'a> Hash for TokenStreamHelper<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let tts = self.0.clone().into_iter().collect::<Vec<_>>();
        tts.len().hash(state);
        for tt in tts {
            TokenTreeHelper(&tt).hash(state);
        }
    }
}

mod fab {
    #[rustfmt::skip]
    pub mod fold;
    #[rustfmt::skip]
    pub mod visit;
    #[rustfmt::skip]
    pub mod visit_mut;
        #[rustfmt::skip]
    mod clone;
        #[rustfmt::skip]
    mod debug;
        #[rustfmt::skip]
    mod eq;
        #[rustfmt::skip]
    mod hash;
    mod helper {
        pub mod fold {
            use crate::punct::{Pair, Puncted};
            pub trait FoldHelper {
                type Item;
                fn lift<F>(self, f: F) -> Self
                where
                    F: FnMut(Self::Item) -> Self::Item;
            }
            impl<T> FoldHelper for Vec<T> {
                type Item = T;
                fn lift<F>(self, f: F) -> Self
                where
                    F: FnMut(Self::Item) -> Self::Item,
                {
                    self.into_iter().map(f).collect()
                }
            }
            impl<T, U> FoldHelper for Puncted<T, U> {
                type Item = T;
                fn lift<F>(self, mut f: F) -> Self
                where
                    F: FnMut(Self::Item) -> Self::Item,
                {
                    self.into_pairs()
                        .map(Pair::into_tuple)
                        .map(|(t, u)| Pair::new(f(t), u))
                        .collect()
                }
            }
        }
    }
}
pub use fab::*;
pub mod __private {
    pub use super::{
        lower::punct as print_punct,
        parse::parse_quote_fn,
        parsing::{peek_punct, punct as parse_punct},
    };
    pub use proc_macro::TokenStream;
    pub use proc_macro2::TokenStream as TokenStream2;
    pub use quote::{self, ToTokens, TokenStreamExt};
    pub use std::{
        clone::Clone,
        cmp::{Eq, PartialEq},
        concat,
        default::Default,
        fmt::{self, Debug, Formatter},
        hash::{Hash, Hasher},
        marker::Copy,
        option::Option::{None, Some},
        result::Result::{Err, Ok},
        stringify,
    };
    pub type bool = help::Bool;
    pub type str = help::Str;
    mod help {
        pub type Bool = bool;
        pub type Str = str;
    }
    pub struct private(pub ());
}

pub struct File {
    pub shebang: Option<String>,
    pub attrs: Vec<attr::Attr>,
    pub items: Vec<Item>,
}
impl Parse for File {
    fn parse(x: Stream) -> Res<Self> {
        Ok(File {
            shebang: None,
            attrs: x.call(attr::Attr::parse_inner)?,
            items: {
                let mut ys = Vec::new();
                while !x.is_empty() {
                    ys.push(x.parse()?);
                }
                ys
            },
        })
    }
}
impl ToTokens for File {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.inner());
        ys.append_all(&self.items);
    }
}

pub fn parse_file(mut x: &str) -> Res<File> {
    const BOM: &str = "\u{feff}";
    if x.starts_with(BOM) {
        x = &x[BOM.len()..];
    }
    let mut shebang = None;
    if x.starts_with("#!") {
        let rest = whitespace::ws_skip(&x[2..]);
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
    let mut y: File = parse_str(x)?;
    y.shebang = shebang;
    Ok(y)
}
pub fn parse_str<T: parse::Parse>(s: &str) -> Res<T> {
    Parser::parse_str(T::parse, s)
}

pub fn parse<T: parse::Parse>(s: Stream) -> Res<T> {
    Parser::parse(T::parse, s)
}
pub fn parse2<T: parse::Parse>(s: Stream) -> Res<T> {
    Parser::parse2(T::parse, s)
}

fn wrap_bare_struct(ys: &mut Stream, e: &Expr) {
    if let Expr::Struct(_) = *e {
        tok::Paren::default().surround(ys, |ys| {
            e.to_tokens(ys);
        });
    } else {
        e.to_tokens(ys);
    }
}
pub fn verbatim_between<'a>(begin: parse::Stream<'a>, end: parse::Stream<'a>) -> pm2::Stream {
    let end = end.cursor();
    let mut cursor = begin.cursor();
    assert!(same_buffer(end, cursor));
    let mut tokens = pm2::Stream::new();
    while cursor != end {
        let (tt, next) = cursor.token_tree().unwrap();
        if cmp_assuming_same_buffer(end, next) == Ordering::Less {
            if let Some((inside, _span, after)) = cursor.group(pm2::Delim::None) {
                assert!(next == after);
                cursor = inside;
                continue;
            } else {
                panic!("verbatim end must not be inside a delimited group");
            }
        }
        tokens.extend(iter::once(tt));
        cursor = next;
    }
    tokens
}

pub fn ws_skip(mut s: &str) -> &str {
    'skip: while !s.is_empty() {
        let byte = s.as_bytes()[0];
        if byte == b'/' {
            if s.starts_with("//") && (!s.starts_with("///") || s.starts_with("////")) && !s.starts_with("//!") {
                if let Some(i) = s.find('\n') {
                    s = &s[i + 1..];
                    continue;
                } else {
                    return "";
                }
            } else if s.starts_with("/**/") {
                s = &s[4..];
                continue;
            } else if s.starts_with("/*") && (!s.starts_with("/**") || s.starts_with("/***")) && !s.starts_with("/*!") {
                let mut depth = 0;
                let bytes = s.as_bytes();
                let mut i = 0;
                let upper = bytes.len() - 1;
                while i < upper {
                    if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                        depth += 1;
                        i += 1; // eat '*'
                    } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        depth -= 1;
                        if depth == 0 {
                            s = &s[i + 2..];
                            continue 'skip;
                        }
                        i += 1; // eat '/'
                    }
                    i += 1;
                }
                return s;
            }
        }
        match byte {
            b' ' | 0x09..=0x0d => {
                s = &s[1..];
                continue;
            },
            b if b <= 0x7f => {},
            _ => {
                let ch = s.chars().next().unwrap();
                if is_whitespace(ch) {
                    s = &s[ch.len_utf8()..];
                    continue;
                }
            },
        }
        return s;
    }
    s
}
fn is_whitespace(x: char) -> bool {
    x.is_whitespace() || x == '\u{200e}' || x == '\u{200f}'
}

pub fn punct(s: &str, spans: &[pm2::Span], ys: &mut Stream) {
    assert_eq!(s.len(), spans.len());
    let mut chars = s.chars();
    let mut spans = spans.iter();
    let ch = chars.next_back().unwrap();
    let span = spans.next_back().unwrap();
    for (ch, span) in chars.zip(spans) {
        let mut op = Punct::new(ch, pm2::Spacing::Joint);
        op.set_span(*span);
        ys.append(op);
    }
    let mut op = Punct::new(ch, pm2::Spacing::Alone);
    op.set_span(*span);
    ys.append(op);
}
pub fn keyword(x: &str, s: pm2::Span, ys: &mut Stream) {
    ys.append(Ident::new(x, s));
}
pub fn delim(d: pm2::Delim, s: pm2::Span, ys: &mut Stream, inner: pm2::Stream) {
    let mut g = Group::new(d, inner);
    g.set_span(s);
    ys.append(g);
}
