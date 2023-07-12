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

use proc_macro2::Punct;
use quote::{quote, spanned, ToTokens, TokenStreamExt};
use std::{
    cmp::{self, Ordering},
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem,
    ops::{self, Deref, DerefMut},
    thread::{self, ThreadId},
};

mod pm2 {
    pub use proc_macro2::{
        extra::DelimSpan, Delimiter as Delim, Group, Ident, Literal as Lit, Spacing, Span, TokenStream as Stream,
        TokenTree as Tree,
    };
}
pub use pm2::Ident;

#[macro_use]
mod mac;

mod attr;
mod cur;
mod expr;
mod gen;
mod item;
mod lit;
mod meta;
mod parse;
mod pat;
mod path;
mod tok;
mod typ;

use cur::Cursor;
use parse::{Parse, Stream};
use path::Path;

mod ident {
    macro_rules! ident_from_tok {
        ($x:ident) => {
            impl From<Token![$x]> for Ident {
                fn from(x: Token![$x]) -> Ident {
                    Ident::new(stringify!($x), x.span)
                }
            }
        };
    }
    ident_from_tok!(self);
    ident_from_tok!(Self);
    ident_from_tok!(super);
    ident_from_tok!(crate);
    ident_from_tok!(extern);
    impl From<Token![_]> for Ident {
        fn from(x: Token![_]) -> Ident {
            Ident::new("_", x.span)
        }
    }
    pub fn xid_ok(x: &str) -> bool {
        let mut ys = x.chars();
        let first = ys.next().unwrap();
        if !(first == '_' || unicode_ident::is_xid_start(first)) {
            return false;
        }
        for y in ys {
            if !unicode_ident::is_xid_continue(y) {
                return false;
            }
        }
        true
    }

    pub struct Lifetime {
        pub apos: pm2::Span,
        pub ident: Ident,
    }
    impl Lifetime {
        pub fn new(x: &str, s: pm2::Span) -> Self {
            if !x.starts_with('\'') {
                panic!("lifetime name must start with apostrophe as in \"'a\", got {:?}", x);
            }
            if x == "'" {
                panic!("lifetime name must not be empty");
            }
            if !ident::xid_ok(&x[1..]) {
                panic!("{:?} is not a valid lifetime name", x);
            }
            Lifetime {
                apos: s,
                ident: Ident::new(&x[1..], s),
            }
        }
        pub fn span(&self) -> pm2::Span {
            self.apos.join(self.ident.span()).unwrap_or(self.apos)
        }
        pub fn set_span(&mut self, s: pm2::Span) {
            self.apos = s;
            self.ident.set_span(s);
        }
    }
    impl Display for Lifetime {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            "'".fmt(f)?;
            self.ident.fmt(f)
        }
    }
    impl Clone for Lifetime {
        fn clone(&self) -> Self {
            Lifetime {
                apos: self.apos,
                ident: self.ident.clone(),
            }
        }
    }
    impl PartialEq for Lifetime {
        fn eq(&self, x: &Lifetime) -> bool {
            self.ident.eq(&x.ident)
        }
    }
    impl Eq for Lifetime {}
    impl PartialOrd for Lifetime {
        fn partial_cmp(&self, x: &Lifetime) -> Option<Ordering> {
            Some(self.cmp(x))
        }
    }
    impl Ord for Lifetime {
        fn cmp(&self, x: &Lifetime) -> Ordering {
            self.ident.cmp(&x.ident)
        }
    }
    impl Hash for Lifetime {
        fn hash<H: Hasher>(&self, x: &mut H) {
            self.ident.hash(x);
        }
    }

    #[allow(non_snake_case)]
    pub fn Ident(x: look::TokenMarker) -> Ident {
        match x {}
    }
    #[allow(non_snake_case)]
    pub fn Lifetime(x: look::TokenMarker) -> Lifetime {
        match x {}
    }
}
use ident::Lifetime;
pub mod ext {
    use super::{
        parse::{Peek, Stream},
        sealed::look,
        tok::Custom,
    };
    pub trait IdentExt: Sized + private::Sealed {
        #[allow(non_upper_case_globals)]
        const peek_any: private::PeekFn = private::PeekFn;
        fn parse_any(x: Stream) -> Res<Self>;
        fn unraw(&self) -> Ident;
    }
    impl IdentExt for Ident {
        fn parse_any(x: Stream) -> Res<Self> {
            x.step(|c| match c.ident() {
                Some((ident, rest)) => Ok((ident, rest)),
                None => Err(c.error("expected ident")),
            })
        }
        fn unraw(&self) -> Ident {
            let y = self.to_string();
            if let Some(x) = y.strip_prefix("r#") {
                Ident::new(x, self.span())
            } else {
                self.clone()
            }
        }
    }
    impl Peek for private::PeekFn {
        type Token = private::IdentAny;
    }
    impl Custom for private::IdentAny {
        fn peek(x: Cursor) -> bool {
            x.ident().is_some()
        }
        fn display() -> &'static str {
            "identifier"
        }
    }
    impl look::Sealed for private::PeekFn {}
    mod private {
        pub trait Sealed {}
        impl Sealed for Ident {}
        pub struct PeekFn;
        pub struct IdentAny;
        impl Copy for PeekFn {}
        impl Clone for PeekFn {
            fn clone(&self) -> Self {
                *self
            }
        }
    }
}

pub mod punct;
use punct::Punctuated;

mod stmt {
    pub struct Block {
        pub brace: tok::Brace,
        pub stmts: Vec<Stmt>,
    }
    pub enum Stmt {
        Local(Local),
        Item(Item),
        Expr(Expr, Option<Token![;]>),
        Mac(Mac),
    }
    pub struct Local {
        pub attrs: Vec<attr::Attr>,
        pub let_: Token![let],
        pub pat: pat::Pat,
        pub init: Option<LocalInit>,
        pub semi: Token![;],
    }
    pub struct LocalInit {
        pub eq: Token![=],
        pub expr: Box<Expr>,
        pub diverge: Option<(Token![else], Box<Expr>)>,
    }
    pub struct Mac {
        pub attrs: Vec<attr::Attr>,
        pub mac: Macro,
        pub semi: Option<Token![;]>,
    }
}

pub mod data {
    pub struct DeriveInput {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub ident: Ident,
        pub gens: gen::Gens,
        pub data: Data,
    }
    pub enum Data {
        Enum(Enum),
        Struct(Struct),
        Union(Union),
    }
    pub struct Enum {
        pub enum_: Token![enum],
        pub brace: tok::Brace,
        pub variants: Punctuated<Variant, Token![,]>,
    }
    pub struct Struct {
        pub struct_: Token![struct],
        pub fields: Fields,
        pub semi: Option<Token![;]>,
    }
    pub struct Union {
        pub union_: Token![union],
        pub named: Named,
    }
    pub struct Variant {
        pub attrs: Vec<attr::Attr>,
        pub ident: Ident,
        pub fields: Fields,
        pub discriminant: Option<(Token![=], Expr)>,
    }
    ast_enum_of_structs! {
        pub enum Fields {
            Named(Named),
            Unnamed(Unnamed),
            Unit,
        }
    }
    impl Fields {
        pub fn iter(&self) -> punct::Iter<Field> {
            use Fields::*;
            match self {
                Named(x) => x.field.iter(),
                Unnamed(x) => x.field.iter(),
                Unit => punct::empty_punctuated_iter(),
            }
        }
        pub fn iter_mut(&mut self) -> punct::IterMut<Field> {
            use Fields::*;
            match self {
                Named(x) => x.field.iter_mut(),
                Unnamed(x) => x.field.iter_mut(),
                Unit => punct::empty_punctuated_iter_mut(),
            }
        }
        pub fn len(&self) -> usize {
            use Fields::*;
            match self {
                Named(x) => x.field.len(),
                Unnamed(x) => x.field.len(),
                Unit => 0,
            }
        }
        pub fn is_empty(&self) -> bool {
            use Fields::*;
            match self {
                Named(x) => x.field.is_empty(),
                Unnamed(x) => x.field.is_empty(),
                Unit => true,
            }
        }
    }
    impl IntoIterator for Fields {
        type Item = Field;
        type IntoIter = punct::IntoIter<Field>;
        fn into_iter(self) -> Self::IntoIter {
            use Fields::*;
            match self {
                Named(x) => x.field.into_iter(),
                Unnamed(x) => x.field.into_iter(),
                Unit => Punctuated::<Field, ()>::new().into_iter(),
            }
        }
    }
    impl<'a> IntoIterator for &'a Fields {
        type Item = &'a Field;
        type IntoIter = punct::Iter<'a, Field>;
        fn into_iter(self) -> Self::IntoIter {
            self.iter()
        }
    }
    impl<'a> IntoIterator for &'a mut Fields {
        type Item = &'a mut Field;
        type IntoIter = punct::IterMut<'a, Field>;
        fn into_iter(self) -> Self::IntoIter {
            self.iter_mut()
        }
    }
    pub struct Named {
        pub brace: tok::Brace,
        pub field: Punctuated<Field, Token![,]>,
    }
    pub struct Unnamed {
        pub paren: tok::Paren,
        pub field: Punctuated<Field, Token![,]>,
    }
    pub struct Field {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub mut_: Mut,
        pub ident: Option<Ident>,
        pub colon: Option<Token![:]>,
        pub typ: typ::Type,
    }
    pub enum Mut {
        None,
    }
}
use data::DeriveInput;

mod err {
    use proc_macro2::{LexError, Punct};
    use quote::ToTokens;
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
        fn from(err: LexError) -> Self {
            Err::new(err.span(), "lex error")
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
}
pub use err::{Err, Res};

mod look {
    use super::{
        err::{self, Err},
        sealed::look::Sealed,
        tok::Tok,
    };
    use std::cell::RefCell;
    pub struct Lookahead1<'a> {
        scope: pm2::Span,
        cursor: Cursor<'a>,
        comparisons: RefCell<Vec<&'static str>>,
    }
    impl<'a> Lookahead1<'a> {
        pub fn peek<T: Peek>(&self, token: T) -> bool {
            let _ = token;
            peek_impl(self, T::Token::peek, T::Token::display)
        }
        pub fn error(self) -> Err {
            let comparisons = self.comparisons.borrow();
            match comparisons.len() {
                0 => {
                    if self.cursor.eof() {
                        Err::new(self.scope, "unexpected end of input")
                    } else {
                        Err::new(self.cursor.span(), "unexpected token")
                    }
                },
                1 => {
                    let message = format!("expected {}", comparisons[0]);
                    err::new_at(self.scope, self.cursor, message)
                },
                2 => {
                    let message = format!("expected {} or {}", comparisons[0], comparisons[1]);
                    err::new_at(self.scope, self.cursor, message)
                },
                _ => {
                    let join = comparisons.join(", ");
                    let message = format!("expected one of: {}", join);
                    err::new_at(self.scope, self.cursor, message)
                },
            }
        }
    }

    pub fn new(scope: pm2::Span, cursor: Cursor) -> Lookahead1 {
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

    pub trait Peek: Sealed {
        type Token: Tok;
    }
    impl<F: Copy + FnOnce(TokenMarker) -> T, T: Tok> Peek for F {
        type Token = T;
    }

    pub enum TokenMarker {}
    impl<S> IntoSpans<S> for TokenMarker {
        fn into_spans(self) -> S {
            match self {}
        }
    }

    pub fn is_delimiter(x: Cursor, d: pm2::Delim) -> bool {
        x.group(d).is_some()
    }

    impl<F: Copy + FnOnce(TokenMarker) -> T, T: Tok> Sealed for F {}
}

pub struct Parens<'a> {
    pub tok: tok::Paren,
    pub buf: parse::Buffer<'a>,
}
pub fn parse_parens<'a>(x: &parse::Buffer<'a>) -> Res<Parens<'a>> {
    parse_delimited(x, pm2::Delim::Parenthesis).map(|(span, buf)| Parens {
        tok: tok::Paren(span),
        buf,
    })
}
pub struct Braces<'a> {
    pub token: tok::Brace,
    pub buf: parse::Buffer<'a>,
}
pub fn parse_braces<'a>(x: &parse::Buffer<'a>) -> Res<Braces<'a>> {
    parse_delimited(x, pm2::Delim::Brace).map(|(span, buf)| Braces {
        token: tok::Brace(span),
        buf,
    })
}
pub struct Brackets<'a> {
    pub token: tok::Bracket,
    pub buf: parse::Buffer<'a>,
}
pub fn parse_brackets<'a>(x: &parse::Buffer<'a>) -> Res<Brackets<'a>> {
    parse_delimited(x, pm2::Delim::Bracket).map(|(span, buf)| Brackets {
        token: tok::Bracket(span),
        buf,
    })
}
pub struct Group<'a> {
    pub token: tok::Group,
    pub buf: parse::Buffer<'a>,
}
pub fn parse_group<'a>(x: &parse::Buffer<'a>) -> Res<Group<'a>> {
    parse_delimited(x, pm2::Delim::None).map(|(span, buf)| Group {
        token: tok::Group(span.join()),
        buf,
    })
}
fn parse_delimited<'a>(x: &parse::Buffer<'a>, d: pm2::Delim) -> Res<(pm2::DelimSpan, parse::Buffer<'a>)> {
    x.step(|c| {
        if let Some((gist, span, rest)) = c.group(d) {
            let scope = close_span_of_group(*c);
            let nested = parse::advance_step_cursor(c, gist);
            let unexpected = parse::get_unexpected(x);
            let gist = parse::new_parse_buffer(scope, nested, unexpected);
            Ok(((span, gist), rest))
        } else {
            use pm2::Delim::*;
            let y = match d {
                Parenthesis => "expected parentheses",
                Brace => "expected braces",
                Bracket => "expected brackets",
                None => "expected group",
            };
            Err(c.error(y))
        }
    })
}

pub trait ParseQuote: Sized {
    fn parse(x: parse::Stream) -> Res<Self>;
}
impl<T: Parse> ParseQuote for T {
    fn parse(x: parse::Stream) -> Res<Self> {
        <T as Parse>::parse(x)
    }
}
impl ParseQuote for attr::Attr {
    fn parse(x: parse::Stream) -> Res<Self> {
        if x.peek(Token![#]) && x.peek2(Token![!]) {
            parsing::attr::single_inner(x)
        } else {
            parsing::attr::single_outer(x)
        }
    }
}
impl ParseQuote for pat::Pat {
    fn parse(x: parse::Stream) -> Res<Self> {
        pat::Pat::parse_multi_with_leading_vert(x)
    }
}
impl ParseQuote for Box<pat::Pat> {
    fn parse(x: parse::Stream) -> Res<Self> {
        <pat::Pat as ParseQuote>::parse(x).map(Box::new)
    }
}
impl<T: Parse, P: Parse> ParseQuote for Punctuated<T, P> {
    fn parse(x: parse::Stream) -> Res<Self> {
        Self::parse_terminated(x)
    }
}
impl ParseQuote for Vec<Stmt> {
    fn parse(x: parse::Stream) -> Res<Self> {
        Block::parse_within(x)
    }
}

pub fn parse_quote_fn<T: ParseQuote>(x: pm2::Stream) -> T {
    let y = T::parse;
    match y.parse2(x) {
        Ok(x) => x,
        Err(x) => panic!("{}", x),
    }
}

struct TokensOrDefault<'a, T: 'a>(pub &'a Option<T>);
impl<'a, T> ToTokens for TokensOrDefault<'a, T>
where
    T: ToTokens + Default,
{
    fn to_tokens(&self, ys: &mut pm2::Stream) {
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

pub struct VisRestricted {
    pub pub_: Token![pub],
    pub paren: tok::Paren,
    pub in_: Option<Token![in]>,
    pub path: Box<Path>,
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
            use crate::punct::{Pair, Punctuated};
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
            impl<T, U> FoldHelper for Punctuated<T, U> {
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
        parse_quote_fn,
        parsing::{peek_punct, punct as parse_punct},
    };
    pub use proc_macro::pm2::Stream;
    pub use proc_macro2::pm2::Stream as TokenStream2;
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
pub fn parse<T: parse::Parse>(tokens: proc_macro::pm2::Stream) -> Res<T> {
    parse::Parser::parse(T::parse, tokens)
}
pub fn parse2<T: parse::Parse>(tokens: proc_macro2::pm2::Stream) -> Res<T> {
    parse::Parser::parse2(T::parse, tokens)
}

pub struct File {
    pub shebang: Option<String>,
    pub attrs: Vec<attr::Attr>,
    pub items: Vec<Item>,
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
    parse::Parser::parse_str(T::parse, s)
}
