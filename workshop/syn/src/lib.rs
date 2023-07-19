extern crate proc_macro as pm;

use std::{
    cmp::{self, Ordering},
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem,
    ops::{self, Deref, DerefMut},
};

#[macro_use]
mod quote;
use quote::{StreamExt, ToStream};

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
mod pm2;
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
    use super::*;
    use std::cell::RefCell;
    pub struct Look1<'a> {
        cur: Cursor<'a>,
        scope: pm2::Span,
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
            cur,
            scope,
            comps: RefCell::new(Vec::new()),
        }
    }

    pub trait Peek {
        type Tok: tok::Tok;
    }
    impl<F: Copy + FnOnce(Marker) -> T, T: Tok> Peek for F {
        type Tok = T;
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
}
use look::{Look1, Peek};

struct ToksOrDefault<'a, T: 'a>(pub &'a Option<T>);
impl<'a, T> ToStream for ToksOrDefault<'a, T>
where
    T: ToStream + Default,
{
    fn to_tokens(&self, ys: &mut Stream) {
        match self.0 {
            Some(x) => x.to_tokens(ys),
            None => T::default().to_tokens(ys),
        }
    }
}

pub trait Spanned {
    fn span(&self) -> pm2::Span;
}
impl<T: ?Sized + quote::Spanned> Spanned for T {
    fn span(&self) -> pm2::Span {
        self.__span()
    }
}

struct TreeHelper<'a>(pub &'a pm2::Tree);
impl<'a> PartialEq for TreeHelper<'a> {
    fn eq(&self, x: &Self) -> bool {
        use pm2::{Delim::*, Spacing::*};
        match (self.0, x.0) {
            (pm2::Tree::Group(g1), pm2::Tree::Group(g2)) => {
                match (g1.delim(), g2.delim()) {
                    (Paren, Paren) | (Brace, Brace) | (Bracket, Bracket) | (None, None) => {},
                    _ => return false,
                }
                let s1 = g1.stream().into_iter();
                let mut s2 = g2.stream().into_iter();
                for x1 in s1 {
                    let x2 = match s2.next() {
                        Some(x) => x,
                        None => return false,
                    };
                    if TreeHelper(&x1) != TreeHelper(&x2) {
                        return false;
                    }
                }
                s2.next().is_none()
            },
            (pm2::Tree::Punct(x1), pm2::Tree::Punct(x2)) => {
                x1.as_char() == x2.as_char()
                    && match (x1.spacing(), x2.spacing()) {
                        (Alone, Alone) | (Joint, Joint) => true,
                        _ => false,
                    }
            },
            (pm2::Tree::Lit(x1), pm2::Tree::Lit(x2)) => x1.to_string() == x2.to_string(),
            (pm2::Tree::Ident(x1), pm2::Tree::Ident(x2)) => x1 == x2,
            _ => false,
        }
    }
}
impl<'a> Hash for TreeHelper<'a> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        match self.0 {
            pm2::Tree::Group(g) => {
                0u8.hash(h);
                use pm2::Delim::*;
                match g.delim() {
                    Paren => 0u8.hash(h),
                    Brace => 1u8.hash(h),
                    Bracket => 2u8.hash(h),
                    None => 3u8.hash(h),
                }
                for x in g.stream() {
                    TreeHelper(&x).hash(h);
                }
                0xffu8.hash(h);
            },
            pm2::Tree::Punct(x) => {
                1u8.hash(h);
                x.as_char().hash(h);
                use pm2::Spacing::*;
                match x.spacing() {
                    Alone => 0u8.hash(h),
                    Joint => 1u8.hash(h),
                }
            },
            pm2::Tree::Lit(x) => (2u8, x.to_string()).hash(h),
            pm2::Tree::Ident(x) => (3u8, x).hash(h),
        }
    }
}

struct StreamHelper<'a>(pub &'a pm2::Stream);
impl<'a> PartialEq for StreamHelper<'a> {
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
impl<'a> Hash for StreamHelper<'a> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        let xs = self.0.clone().into_iter().collect::<Vec<_>>();
        xs.len().hash(h);
        for x in xs {
            TreeHelper(&x).hash(h);
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

pub fn parse<T: parse::Parse>(s: Stream) -> Res<T> {
    Parser::parse(T::parse, s)
}
pub fn parse2<T: parse::Parse>(s: Stream) -> Res<T> {
    Parser::parse2(T::parse, s)
}
