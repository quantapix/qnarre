#![allow(non_camel_case_types, non_snake_case, unused_macros)]

extern crate proc_macro as pm;

use std::{
    cmp::{self, Ordering},
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{self, Deref, DerefMut},
};

#[macro_use]
mod lower;
use lower::Lower;

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
mod parse;
mod pat;
mod path;
mod pm2;
mod pretty;
mod punct;
mod stmt;
mod structure;
mod tok;
mod typ;

use cur::Cursor;
use data::Input;
use err::{Err, Res};
use ident::Life;
use parse::{Parse, Parser, Stream};
use path::Path;
use pm2::{Ident, Punct, Span};
use pretty::{Pretty, Print};
use punct::{Pair, Puncted};
use tok::Tok;

mod look {
    use super::*;
    use std::cell::RefCell;
    pub struct Look1<'a> {
        cur: Cursor<'a>,
        scope: Span,
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

    pub fn new(scope: Span, cur: Cursor) -> Look1 {
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
    impl<S> pm2::IntoSpans<S> for Marker {
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
impl<'a, T> Lower for ToksOrDefault<'a, T>
where
    T: Lower + Default,
{
    fn lower(&self, s: &mut Stream) {
        match self.0 {
            Some(x) => x.lower(s),
            None => T::default().lower(s),
        }
    }
}

pub trait Spanned {
    fn span(&self) -> Span;
}
impl<T: ?Sized + lower::Spanned> Spanned for T {
    fn span(&self) -> Span {
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
                    (Parenth, Parenth) | (Brace, Brace) | (Bracket, Bracket) | (None, None) => {},
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
                    Parenth => 0u8.hash(h),
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

pub fn parse<T: parse::Parse>(s: Stream) -> Res<T> {
    Parser::parse(T::parse, s)
}
pub fn parse2<T: parse::Parse>(s: Stream) -> Res<T> {
    Parser::parse2(T::parse, s)
}

trait Folder {}

trait Fold {
    fn fold<F>(&self, f: &mut F);
}

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

trait Visitor {}

trait Visit {
    fn visit<V>(&self, v: &mut V);
    fn visit_mut<V>(&mut self, v: &mut V);
}

pub const MARGIN: isize = 89;
pub const INDENT: isize = 4;
pub const MIN_SPACE: isize = 60;
