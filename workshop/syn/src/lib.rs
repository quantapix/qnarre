#![allow(non_camel_case_types)]
#![allow(
    clippy::bool_to_int_with_if,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_ptr_alignment,
    clippy::default_trait_access,
    clippy::derivable_impls,
    clippy::doc_markdown,
    clippy::expl_impl_clone_on_copy,
    clippy::explicit_auto_deref,
    clippy::if_not_else,
    clippy::inherent_to_string,
    clippy::items_after_statements,
    clippy::large_enum_variant,
    clippy::let_underscore_untyped, // https://github.com/rust-lang/rust-clippy/issues/10410
    clippy::manual_assert,
    clippy::manual_let_else,
    clippy::match_like_matches_macro,
    clippy::match_on_vec_items,
    clippy::match_same_arms,
    clippy::match_wildcard_for_single_variants, // clippy bug: https://github.com/rust-lang/rust-clippy/issues/6984
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_doctest_main,
    clippy::needless_pass_by_value,
    clippy::never_loop,
    clippy::range_plus_one,
    clippy::redundant_else,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::single_match_else,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref,
    clippy::uninlined_format_args,
    clippy::unnecessary_box_returns,
    clippy::unnecessary_unwrap,
    clippy::used_underscore_binding,
    clippy::wildcard_imports,
)]

extern crate proc_macro;

pub use quote::ToTokens;
pub use std::{
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{self, Deref, DerefMut},
};

use quote::{quote, spanned, TokenStreamExt};
use std::{
    cmp::{self, Ordering},
    mem,
    thread::{self, ThreadId},
};

mod pm2 {
    pub use proc_macro2::{
        extra::DelimSpan, Delimiter as Delim, Group, Ident, Literal as Lit, Punct, Spacing, Span,
        TokenStream as Stream, TokenTree as Tree,
    };
}
pub use pm2::{Ident, Punct};

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

pub enum Visibility {
    Public(Token![pub]),
    Restricted(VisRestricted),
    Inherited,
}
impl Visibility {
    fn is_inherited(&self) -> bool {
        match self {
            Visibility::Inherited => true,
            _ => false,
        }
    }
    fn parse_pub(x: Stream) -> Res<Self> {
        let pub_ = x.parse::<Token![pub]>()?;
        if x.peek(tok::Paren) {
            let ahead = x.fork();
            let y;
            let paren = parenthesized!(y in ahead);
            if y.peek(Token![crate]) || y.peek(Token![self]) || y.peek(Token![super]) {
                let path = y.call(Ident::parse_any)?;
                if y.is_empty() {
                    x.advance_to(&ahead);
                    return Ok(Visibility::Restricted(VisRestricted {
                        pub_,
                        paren,
                        in_: None,
                        path: Box::new(Path::from(path)),
                    }));
                }
            } else if y.peek(Token![in]) {
                let in_: Token![in] = y.parse()?;
                let path = y.call(Path::parse_mod_style)?;
                x.advance_to(&ahead);
                return Ok(Visibility::Restricted(VisRestricted {
                    pub_,
                    paren,
                    in_: Some(in_),
                    path: Box::new(path),
                }));
            }
        }
        Ok(Visibility::Public(pub_))
    }
    pub fn is_some(&self) -> bool {
        match self {
            Visibility::Inherited => false,
            _ => true,
        }
    }
}
impl Parse for Visibility {
    fn parse(x: Stream) -> Res<Self> {
        if x.peek(tok::Group) {
            let ahead = x.fork();
            let y = parse::parse_group(&ahead)?;
            if y.buf.is_empty() {
                x.advance_to(&ahead);
                return Ok(Visibility::Inherited);
            }
        }
        if x.peek(Token![pub]) {
            Self::parse_pub(x)
        } else {
            Ok(Visibility::Inherited)
        }
    }
}
impl ToTokens for Visibility {
    fn to_tokens(&self, ys: &mut Stream) {
        match self {
            Visibility::Public(x) => x.to_tokens(ys),
            Visibility::Restricted(x) => x.to_tokens(ys),
            Visibility::Inherited => {},
        }
    }
}

pub struct VisRestricted {
    pub pub_: Token![pub],
    pub paren: tok::Paren,
    pub in_: Option<Token![in]>,
    pub path: Box<Path>,
}
impl ToTokens for VisRestricted {
    fn to_tokens(&self, ys: &mut Stream) {
        self.pub_.to_tokens(ys);
        self.paren.surround(ys, |ys| {
            self.in_.to_tokens(ys);
            self.path.to_tokens(ys);
        });
    }
}

mod sealed {
    pub mod look {
        pub trait Sealed: Copy {}
    }
}

pub trait IntoSpans<S> {
    fn into_spans(self) -> S;
}
impl IntoSpans<pm2::Span> for pm2::Span {
    fn into_spans(self) -> pm2::Span {
        self
    }
}
impl IntoSpans<[pm2::Span; 1]> for pm2::Span {
    fn into_spans(self) -> [pm2::Span; 1] {
        [self]
    }
}
impl IntoSpans<[pm2::Span; 2]> for pm2::Span {
    fn into_spans(self) -> [pm2::Span; 2] {
        [self, self]
    }
}
impl IntoSpans<[pm2::Span; 3]> for pm2::Span {
    fn into_spans(self) -> [pm2::Span; 3] {
        [self, self, self]
    }
}
impl IntoSpans<[pm2::Span; 1]> for [pm2::Span; 1] {
    fn into_spans(self) -> [pm2::Span; 1] {
        self
    }
}
impl IntoSpans<[pm2::Span; 2]> for [pm2::Span; 2] {
    fn into_spans(self) -> [pm2::Span; 2] {
        self
    }
}
impl IntoSpans<[pm2::Span; 3]> for [pm2::Span; 3] {
    fn into_spans(self) -> [pm2::Span; 3] {
        self
    }
}
impl IntoSpans<pm2::DelimSpan> for pm2::Span {
    fn into_spans(self) -> pm2::DelimSpan {
        let mut y = Group::new(pm2::Delim::None, pm2::Stream::new());
        y.set_span(self);
        y.delim_span()
    }
}
impl IntoSpans<pm2::DelimSpan> for pm2::DelimSpan {
    fn into_spans(self) -> pm2::DelimSpan {
        self
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

struct ThreadBound<T> {
    val: T,
    id: ThreadId,
}
unsafe impl<T> Sync for ThreadBound<T> {}
unsafe impl<T: Copy> Send for ThreadBound<T> {}
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
