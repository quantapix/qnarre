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

use crate::{
    lookahead,
    parse::{Parse, ParseStream, Parser, Result},
    tok::{Brace, Bracket, Paren},
};
use proc_macro2::{extra::DelimSpan, Delimiter, Group, Ident, Span, TokenStream, TokenTree};
use quote::{spanned, ToTokens};
use std::{
    cmp::Ordering,
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    ops,
    thread::{self, ThreadId},
};

extern crate proc_macro;
#[macro_use]
mod mac;
#[macro_use]
mod group {
    use crate::{err::Result, parse::ParseBuffer, tok};
    use proc_macro2::{extra::DelimSpan, Delimiter};
    pub struct Parens<'a> {
        pub token: tok::Paren,
        pub content: ParseBuffer<'a>,
    }
    pub fn parse_parens<'a>(x: &ParseBuffer<'a>) -> Result<Parens<'a>> {
        parse_delimited(x, Delimiter::Parenthesis).map(|(span, content)| Parens {
            token: tok::Paren(span),
            content,
        })
    }
    #[macro_export]
    macro_rules! parenthesized {
        ($content:ident in $cur:expr) => {
            match $crate::group::parse_parens(&$cur) {
                $crate::__private::Ok(x) => {
                    $content = x.content;
                    x.token
                },
                $crate::__private::Err(x) => {
                    return $crate::__private::Err(x);
                },
            }
        };
    }
    pub struct Braces<'a> {
        pub token: tok::Brace,
        pub content: ParseBuffer<'a>,
    }
    pub fn parse_braces<'a>(x: &ParseBuffer<'a>) -> Result<Braces<'a>> {
        parse_delimited(x, Delimiter::Brace).map(|(span, content)| Braces {
            token: tok::Brace(span),
            content,
        })
    }
    #[macro_export]
    macro_rules! braced {
        ($content:ident in $cur:expr) => {
            match $crate::group::parse_braces(&$cur) {
                $crate::__private::Ok(x) => {
                    $content = x.content;
                    x.token
                },
                $crate::__private::Err(x) => {
                    return $crate::__private::Err(x);
                },
            }
        };
    }
    pub struct Brackets<'a> {
        pub token: tok::Bracket,
        pub content: ParseBuffer<'a>,
    }
    pub fn parse_brackets<'a>(x: &ParseBuffer<'a>) -> Result<Brackets<'a>> {
        parse_delimited(x, Delimiter::Bracket).map(|(span, content)| Brackets {
            token: tok::Bracket(span),
            content,
        })
    }
    #[macro_export]
    macro_rules! bracketed {
        ($content:ident in $cur:expr) => {
            match $crate::group::parse_brackets(&$cur) {
                $crate::__private::Ok(x) => {
                    $content = x.content;
                    x.token
                },
                $crate::__private::Err(x) => {
                    return $crate::__private::Err(x);
                },
            }
        };
    }
    pub struct Group<'a> {
        pub token: tok::Group,
        pub content: ParseBuffer<'a>,
    }
    pub fn parse_group<'a>(x: &ParseBuffer<'a>) -> Result<Group<'a>> {
        parse_delimited(x, Delimiter::None).map(|(span, content)| Group {
            token: tok::Group(span.join()),
            content,
        })
    }
    fn parse_delimited<'a>(x: &ParseBuffer<'a>, delim: Delimiter) -> Result<(DelimSpan, ParseBuffer<'a>)> {
        x.step(|cursor| {
            if let Some((content, span, rest)) = cursor.group(delim) {
                let scope = crate::buffer::close_span_of_group(*cursor);
                let nested = crate::parse::advance_step_cursor(cursor, content);
                let unexpected = crate::parse::get_unexpected(x);
                let content = crate::parse::new_parse_buffer(scope, nested, unexpected);
                Ok(((span, content), rest))
            } else {
                let y = match delim {
                    Delimiter::Parenthesis => "expected parentheses",
                    Delimiter::Brace => "expected curly braces",
                    Delimiter::Bracket => "expected square brackets",
                    Delimiter::None => "expected invisible group",
                };
                Err(cursor.error(y))
            }
        })
    }
}
#[macro_use]
pub mod tok;

mod attr {
    use super::{
        ext::IdentExt,
        lit::Lit,
        parse::{Err, Parse, ParseStream, Parser, Result},
        parsing,
        path::{Path, PathSegment},
        punctuated::Punctuated,
        *,
    };
    use proc_macro2::{Ident, TokenStream};
    use std::{fmt::Display, iter, slice};

    ast_struct! {
        pub struct Attribute {
            pub pound_token: Token![#],
            pub style: AttrStyle,
            pub bracket_token: tok::Bracket,
            pub meta: Meta,
        }
    }
    impl Attribute {
        pub fn path(&self) -> &Path {
            self.meta.path()
        }
        pub fn parse_args<T: Parse>(&self) -> Result<T> {
            self.parse_args_with(T::parse)
        }
        pub fn parse_args_with<T: Parser>(&self, parser: T) -> Result<T::Output> {
            match &self.meta {
                Meta::Path(x) => Err(crate::err::new2(
                    x.segments.first().unwrap().ident.span(),
                    x.segments.last().unwrap().ident.span(),
                    format!(
                        "expected attribute arguments in parentheses: {}[{}(...)]",
                        parsing::DisplayAttrStyle(&self.style),
                        parsing::DisplayPath(x),
                    ),
                )),
                Meta::NameValue(x) => Err(Err::new(
                    x.eq_token.span,
                    format_args!(
                        "expected parentheses: {}[{}(...)]",
                        parsing::DisplayAttrStyle(&self.style),
                        parsing::DisplayPath(&meta.path),
                    ),
                )),
                Meta::List(x) => x.parse_args_with(parser),
            }
        }
        pub fn parse_nested_meta(&self, x: impl FnMut(ParseNestedMeta) -> Result<()>) -> Result<()> {
            self.parse_args_with(meta_parser(x))
        }
        pub fn parse_outer(x: ParseStream) -> Result<Vec<Self>> {
            let mut y = Vec::new();
            while x.peek(Token![#]) {
                y.push(x.call(parsing::single_parse_outer)?);
            }
            Ok(y)
        }
        pub fn parse_inner(x: ParseStream) -> Result<Vec<Self>> {
            let mut y = Vec::new();
            parsing::parse_inner(x, &mut y)?;
            Ok(y)
        }
    }
    ast_enum! {
        pub enum AttrStyle {
            Outer,
            Inner(Token![!]),
        }
    }
    ast_enum_of_structs! {
        pub enum Meta {
            Path(Path),
            List(MetaList),
            NameValue(MetaNameValue),
        }
    }
    ast_struct! {
        pub struct MetaList {
            pub path: Path,
            pub delimiter: MacroDelimiter,
            pub tokens: TokenStream,
        }
    }
    ast_struct! {
        pub struct MetaNameValue {
            pub path: Path,
            pub eq_token: Token![=],
            pub value: Expr,
        }
    }
    impl Meta {
        pub fn path(&self) -> &Path {
            match self {
                Meta::Path(x) => x,
                Meta::List(x) => &x.path,
                Meta::NameValue(x) => &x.path,
            }
        }
        pub fn require_path_only(&self) -> Result<&Path> {
            let y = match self {
                Meta::Path(x) => return Ok(x),
                Meta::List(x) => x.delimiter.span().open(),
                Meta::NameValue(x) => x.eq_token.span,
            };
            Err(Err::new(y, "unexpected token in attribute"))
        }
        pub fn require_list(&self) -> Result<&MetaList> {
            match self {
                Meta::List(x) => Ok(x),
                Meta::Path(x) => Err(crate::err::new2(
                    x.segments.first().unwrap().ident.span(),
                    x.segments.last().unwrap().ident.span(),
                    format!(
                        "expected attribute arguments in parentheses: `{}(...)`",
                        parsing::DisplayPath(x),
                    ),
                )),
                Meta::NameValue(x) => Err(Err::new(x.eq_token.span, "expected `(`")),
            }
        }
        pub fn require_name_value(&self) -> Result<&MetaNameValue> {
            match self {
                Meta::NameValue(x) => Ok(x),
                Meta::Path(x) => Err(crate::err::new2(
                    x.segments.first().unwrap().ident.span(),
                    x.segments.last().unwrap().ident.span(),
                    format!(
                        "expected a value for this attribute: `{} = ...`",
                        parsing::DisplayPath(x),
                    ),
                )),
                Meta::List(x) => Err(Err::new(x.delimiter.span().open(), "expected `=`")),
            }
        }
    }
    impl MetaList {
        pub fn parse_args<T: Parse>(&self) -> Result<T> {
            self.parse_args_with(T::parse)
        }
        pub fn parse_args_with<T: Parser>(&self, parser: T) -> Result<T::Output> {
            let y = self.delimiter.span().close();
            crate::parse::parse_scoped(parser, y, self.tokens.clone())
        }
        pub fn parse_nested_meta(&self, x: impl FnMut(ParseNestedMeta) -> Result<()>) -> Result<()> {
            self.parse_args_with(meta_parser(x))
        }
    }

    pub trait FilterAttrs<'a> {
        type Ret: Iterator<Item = &'a Attribute>;
        fn outer(self) -> Self::Ret;
        fn inner(self) -> Self::Ret;
    }
    impl<'a> FilterAttrs<'a> for &'a [Attribute] {
        type Ret = iter::Filter<slice::Iter<'a, Attribute>, fn(&&Attribute) -> bool>;
        fn outer(self) -> Self::Ret {
            fn is_outer(x: &&Attribute) -> bool {
                match x.style {
                    AttrStyle::Outer => true,
                    AttrStyle::Inner(_) => false,
                }
            }
            self.iter().filter(is_outer)
        }
        fn inner(self) -> Self::Ret {
            fn is_inner(x: &&Attribute) -> bool {
                match x.style {
                    AttrStyle::Inner(_) => true,
                    AttrStyle::Outer => false,
                }
            }
            self.iter().filter(is_inner)
        }
    }
    pub fn meta_parser(logic: impl FnMut(ParseNestedMeta) -> Result<()>) -> impl Parser<Output = ()> {
        |input: ParseStream| {
            if input.is_empty() {
                Ok(())
            } else {
                parse_nested_meta(input, logic)
            }
        }
    }

    pub struct ParseNestedMeta<'a> {
        pub path: Path,
        pub input: ParseStream<'a>,
    }
    impl<'a> ParseNestedMeta<'a> {
        pub fn value(&self) -> Result<ParseStream<'a>> {
            self.input.parse::<Token![=]>()?;
            Ok(self.input)
        }
        pub fn parse_nested_meta(&self, logic: impl FnMut(ParseNestedMeta) -> Result<()>) -> Result<()> {
            let content;
            parenthesized!(content in self.input);
            parse_nested_meta(&content, logic)
        }
        pub fn error(&self, msg: impl Display) -> Err {
            let start_span = self.path.segments[0].ident.span();
            let end_span = self.input.cursor().prev_span();
            crate::err::new2(start_span, end_span, msg)
        }
    }

    pub fn parse_nested_meta(input: ParseStream, mut logic: impl FnMut(ParseNestedMeta) -> Result<()>) -> Result<()> {
        loop {
            let path = input.call(parse_meta_path)?;
            logic(ParseNestedMeta { path, input })?;
            if input.is_empty() {
                return Ok(());
            }
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                return Ok(());
            }
        }
    }

    fn parse_meta_path(input: ParseStream) -> Result<Path> {
        Ok(Path {
            leading_colon: input.parse()?,
            segments: {
                let mut segments = Punctuated::new();
                if input.peek(Ident::peek_any) {
                    let ident = Ident::parse_any(input)?;
                    segments.push_value(PathSegment::from(ident));
                } else if input.is_empty() {
                    return Err(input.error("expected nested attribute"));
                } else if input.peek(Lit) {
                    return Err(input.error("unexpected literal in nested attribute, expected ident"));
                } else {
                    return Err(input.error("unexpected token in nested attribute, expected ident"));
                }
                while input.peek(Token![::]) {
                    let punct = input.parse()?;
                    segments.push_punct(punct);
                    let ident = Ident::parse_any(input)?;
                    segments.push_value(PathSegment::from(ident));
                }
                segments
            },
        })
    }
}
pub use crate::attr::{AttrStyle, Attribute, Meta, MetaList, MetaNameValue};
pub mod buffer {
    use crate::Lifetime;
    use proc_macro2::{
        extra::DelimSpan, Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree,
    };
    use std::{cmp::Ordering, marker::PhantomData};

    enum Entry {
        Group(Group, usize),
        Ident(Ident),
        Punct(Punct),
        Literal(Literal),
        End(isize),
    }
    pub struct TokenBuffer {
        entries: Box<[Entry]>,
    }
    impl TokenBuffer {
        fn recursive_new(ys: &mut Vec<Entry>, xs: TokenStream) {
            for tt in xs {
                match tt {
                    TokenTree::Ident(x) => ys.push(Entry::Ident(x)),
                    TokenTree::Punct(x) => ys.push(Entry::Punct(x)),
                    TokenTree::Literal(x) => ys.push(Entry::Literal(x)),
                    TokenTree::Group(x) => {
                        let beg = ys.len();
                        ys.push(Entry::End(0));
                        Self::recursive_new(ys, x.stream());
                        let end = ys.len();
                        ys.push(Entry::End(-(end as isize)));
                        let off = end - beg;
                        ys[beg] = Entry::Group(x, off);
                    },
                }
            }
        }
        pub fn new(x: proc_macro::TokenStream) -> Self {
            Self::new2(x.into())
        }
        pub fn new2(x: TokenStream) -> Self {
            let mut ys = Vec::new();
            Self::recursive_new(&mut ys, x);
            ys.push(Entry::End(-(ys.len() as isize)));
            Self {
                entries: ys.into_boxed_slice(),
            }
        }
        pub fn begin(&self) -> Cursor {
            let ptr = self.entries.as_ptr();
            unsafe { Cursor::create(ptr, ptr.add(self.entries.len() - 1)) }
        }
    }
    pub struct Cursor<'a> {
        ptr: *const Entry,
        scope: *const Entry,
        marker: PhantomData<&'a Entry>,
    }
    impl<'a> Cursor<'a> {
        pub fn empty() -> Self {
            struct UnsafeSyncEntry(Entry);
            unsafe impl Sync for UnsafeSyncEntry {}
            static EMPTY_ENTRY: UnsafeSyncEntry = UnsafeSyncEntry(Entry::End(0));
            Cursor {
                ptr: &EMPTY_ENTRY.0,
                scope: &EMPTY_ENTRY.0,
                marker: PhantomData,
            }
        }
        unsafe fn create(mut ptr: *const Entry, scope: *const Entry) -> Self {
            while let Entry::End(_) = *ptr {
                if ptr == scope {
                    break;
                }
                ptr = ptr.add(1);
            }
            Cursor {
                ptr,
                scope,
                marker: PhantomData,
            }
        }
        fn entry(self) -> &'a Entry {
            unsafe { &*self.ptr }
        }
        unsafe fn bump_ignore_group(self) -> Cursor<'a> {
            Cursor::create(self.ptr.offset(1), self.scope)
        }
        fn ignore_none(&mut self) {
            while let Entry::Group(x, _) = self.entry() {
                if x.delimiter() == Delimiter::None {
                    unsafe { *self = self.bump_ignore_group() };
                } else {
                    break;
                }
            }
        }
        pub fn eof(self) -> bool {
            self.ptr == self.scope
        }
        pub fn group(mut self, delim: Delimiter) -> Option<(Cursor<'a>, DelimSpan, Cursor<'a>)> {
            if delim != Delimiter::None {
                self.ignore_none();
            }
            if let Entry::Group(x, end) = self.entry() {
                if x.delimiter() == delim {
                    let span = x.delim_span();
                    let end_of_group = unsafe { self.ptr.add(*end) };
                    let inside_of_group = unsafe { Cursor::create(self.ptr.add(1), end_of_group) };
                    let after_group = unsafe { Cursor::create(end_of_group, self.scope) };
                    return Some((inside_of_group, span, after_group));
                }
            }
            None
        }
        pub(crate) fn any_group(self) -> Option<(Cursor<'a>, Delimiter, DelimSpan, Cursor<'a>)> {
            if let Entry::Group(x, end) = self.entry() {
                let delimiter = x.delimiter();
                let span = x.delim_span();
                let end_of_group = unsafe { self.ptr.add(*end) };
                let inside_of_group = unsafe { Cursor::create(self.ptr.add(1), end_of_group) };
                let after_group = unsafe { Cursor::create(end_of_group, self.scope) };
                return Some((inside_of_group, delimiter, span, after_group));
            }
            None
        }
        pub(crate) fn any_group_token(self) -> Option<(Group, Cursor<'a>)> {
            if let Entry::Group(x, end) = self.entry() {
                let end_of_group = unsafe { self.ptr.add(*end) };
                let after_group = unsafe { Cursor::create(end_of_group, self.scope) };
                return Some((x.clone(), after_group));
            }
            None
        }
        pub fn ident(mut self) -> Option<(Ident, Cursor<'a>)> {
            self.ignore_none();
            match self.entry() {
                Entry::Ident(x) => Some((x.clone(), unsafe { self.bump_ignore_group() })),
                _ => None,
            }
        }
        pub fn punct(mut self) -> Option<(Punct, Cursor<'a>)> {
            self.ignore_none();
            match self.entry() {
                Entry::Punct(x) if x.as_char() != '\'' => Some((x.clone(), unsafe { self.bump_ignore_group() })),
                _ => None,
            }
        }
        pub fn literal(mut self) -> Option<(Literal, Cursor<'a>)> {
            self.ignore_none();
            match self.entry() {
                Entry::Literal(x) => Some((x.clone(), unsafe { self.bump_ignore_group() })),
                _ => None,
            }
        }
        pub fn lifetime(mut self) -> Option<(Lifetime, Cursor<'a>)> {
            self.ignore_none();
            match self.entry() {
                Entry::Punct(x) if x.as_char() == '\'' && x.spacing() == Spacing::Joint => {
                    let next = unsafe { self.bump_ignore_group() };
                    let (ident, rest) = next.ident()?;
                    let lifetime = Lifetime {
                        apostrophe: x.span(),
                        ident,
                    };
                    Some((lifetime, rest))
                },
                _ => None,
            }
        }
        pub fn token_stream(self) -> TokenStream {
            let mut ys = Vec::new();
            let mut cur = self;
            while let Some((x, rest)) = cur.token_tree() {
                ys.push(x);
                cur = rest;
            }
            ys.into_iter().collect()
        }
        pub fn token_tree(self) -> Option<(TokenTree, Cursor<'a>)> {
            let (tree, len) = match self.entry() {
                Entry::Group(x, end) => (x.clone().into(), *end),
                Entry::Literal(x) => (x.clone().into(), 1),
                Entry::Ident(x) => (x.clone().into(), 1),
                Entry::Punct(x) => (x.clone().into(), 1),
                Entry::End(_) => return None,
            };
            let rest = unsafe { Cursor::create(self.ptr.add(len), self.scope) };
            Some((tree, rest))
        }
        pub fn span(self) -> Span {
            match self.entry() {
                Entry::Group(x, _) => x.span(),
                Entry::Literal(x) => x.span(),
                Entry::Ident(x) => x.span(),
                Entry::Punct(x) => x.span(),
                Entry::End(_) => Span::call_site(),
            }
        }
        pub(crate) fn prev_span(mut self) -> Span {
            if start_of_buffer(self) < self.ptr {
                self.ptr = unsafe { self.ptr.offset(-1) };
                if let Entry::End(_) = self.entry() {
                    let mut depth = 1;
                    loop {
                        self.ptr = unsafe { self.ptr.offset(-1) };
                        match self.entry() {
                            Entry::Group(x, _) => {
                                depth -= 1;
                                if depth == 0 {
                                    return x.span();
                                }
                            },
                            Entry::End(_) => depth += 1,
                            Entry::Literal(_) | Entry::Ident(_) | Entry::Punct(_) => {},
                        }
                    }
                }
            }
            self.span()
        }
        pub(crate) fn skip(self) -> Option<Cursor<'a>> {
            let y = match self.entry() {
                Entry::End(_) => return None,
                Entry::Punct(x) if x.as_char() == '\'' && x.spacing() == Spacing::Joint => {
                    match unsafe { &*self.ptr.add(1) } {
                        Entry::Ident(_) => 2,
                        _ => 1,
                    }
                },
                Entry::Group(_, x) => *x,
                _ => 1,
            };
            Some(unsafe { Cursor::create(self.ptr.add(y), self.scope) })
        }
    }
    impl<'a> Copy for Cursor<'a> {}
    impl<'a> Clone for Cursor<'a> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<'a> Eq for Cursor<'a> {}
    impl<'a> PartialEq for Cursor<'a> {
        fn eq(&self, x: &Self) -> bool {
            self.ptr == x.ptr
        }
    }
    impl<'a> PartialOrd for Cursor<'a> {
        fn partial_cmp(&self, x: &Self) -> Option<Ordering> {
            if same_buffer(*self, *x) {
                Some(self.ptr.cmp(&x.ptr))
            } else {
                None
            }
        }
    }
    pub(crate) fn same_scope(a: Cursor, b: Cursor) -> bool {
        a.scope == b.scope
    }
    pub(crate) fn same_buffer(a: Cursor, b: Cursor) -> bool {
        start_of_buffer(a) == start_of_buffer(b)
    }
    fn start_of_buffer(c: Cursor) -> *const Entry {
        unsafe {
            match &*c.scope {
                Entry::End(x) => c.scope.offset(*x),
                _ => unreachable!(),
            }
        }
    }

    pub(crate) fn cmp_assuming_same_buffer(a: Cursor, b: Cursor) -> Ordering {
        a.ptr.cmp(&b.ptr)
    }
    pub(crate) fn open_span_of_group(c: Cursor) -> Span {
        match c.entry() {
            Entry::Group(x, _) => x.span_open(),
            _ => c.span(),
        }
    }
    pub(crate) fn close_span_of_group(c: Cursor) -> Span {
        match c.entry() {
            Entry::Group(x, _) => x.span_close(),
            _ => c.span(),
        }
    }
}
mod expr;
pub use crate::expr::{
    Arm, Expr, ExprArray, ExprAssign, ExprAsync, ExprAwait, ExprBinary, ExprBlock, ExprBreak, ExprCall, ExprCast,
    ExprClosure, ExprConst, ExprContinue, ExprField, ExprForLoop, ExprGroup, ExprIf, ExprIndex, ExprInfer, ExprLet,
    ExprLit, ExprLoop, ExprMacro, ExprMatch, ExprMethodCall, ExprParen, ExprPath, ExprRange, ExprReference, ExprRepeat,
    ExprReturn, ExprStruct, ExprTry, ExprTryBlock, ExprTuple, ExprUnary, ExprUnsafe, ExprWhile, ExprYield, FieldValue,
    Index, Label, Member, RangeLimits,
};
pub use crate::expr::{
    ExprConst as PatConst, ExprLit as PatLit, ExprMacro as PatMacro, ExprPath as PatPath, ExprRange as PatRange,
};
mod generic {
    use super::*;
    use crate::punctuated::{Iter, IterMut, Punctuated};
    use proc_macro2::TokenStream;
    use std::fmt::{self, Debug};
    use std::hash::{Hash, Hasher};
    ast_struct! {
        pub struct Generics {
            pub lt_token: Option<Token![<]>,
            pub params: Punctuated<GenericParam, Token![,]>,
            pub gt_token: Option<Token![>]>,
            pub where_clause: Option<WhereClause>,
        }
    }
    ast_enum_of_structs! {
        pub enum GenericParam {
            Lifetime(LifetimeParam),
            Type(TypeParam),
            Const(ConstParam),
        }
    }
    ast_struct! {
        pub struct LifetimeParam {
            pub attrs: Vec<Attribute>,
            pub lifetime: Lifetime,
            pub colon_token: Option<Token![:]>,
            pub bounds: Punctuated<Lifetime, Token![+]>,
        }
    }
    ast_struct! {
        pub struct TypeParam {
            pub attrs: Vec<Attribute>,
            pub ident: Ident,
            pub colon_token: Option<Token![:]>,
            pub bounds: Punctuated<TypeParamBound, Token![+]>,
            pub eq_token: Option<Token![=]>,
            pub default: Option<Type>,
        }
    }
    ast_struct! {
        pub struct ConstParam {
            pub attrs: Vec<Attribute>,
            pub const_token: Token![const],
            pub ident: Ident,
            pub colon_token: Token![:],
            pub ty: Type,
            pub eq_token: Option<Token![=]>,
            pub default: Option<Expr>,
        }
    }
    impl Default for Generics {
        fn default() -> Self {
            Generics {
                lt_token: None,
                params: Punctuated::new(),
                gt_token: None,
                where_clause: None,
            }
        }
    }
    impl Generics {
        pub fn lifetimes(&self) -> Lifetimes {
            Lifetimes(self.params.iter())
        }
        pub fn lifetimes_mut(&mut self) -> LifetimesMut {
            LifetimesMut(self.params.iter_mut())
        }
        pub fn type_params(&self) -> TypeParams {
            TypeParams(self.params.iter())
        }
        pub fn type_params_mut(&mut self) -> TypeParamsMut {
            TypeParamsMut(self.params.iter_mut())
        }
        pub fn const_params(&self) -> ConstParams {
            ConstParams(self.params.iter())
        }
        pub fn const_params_mut(&mut self) -> ConstParamsMut {
            ConstParamsMut(self.params.iter_mut())
        }
        pub fn make_where_clause(&mut self) -> &mut WhereClause {
            self.where_clause.get_or_insert_with(|| WhereClause {
                where_token: <Token![where]>::default(),
                predicates: Punctuated::new(),
            })
        }
    }
    pub struct Lifetimes<'a>(Iter<'a, GenericParam>);
    impl<'a> Iterator for Lifetimes<'a> {
        type Item = &'a LifetimeParam;
        fn next(&mut self) -> Option<Self::Item> {
            let next = match self.0.next() {
                Some(item) => item,
                None => return None,
            };
            if let GenericParam::Lifetime(lifetime) = next {
                Some(lifetime)
            } else {
                self.next()
            }
        }
    }
    pub struct LifetimesMut<'a>(IterMut<'a, GenericParam>);
    impl<'a> Iterator for LifetimesMut<'a> {
        type Item = &'a mut LifetimeParam;
        fn next(&mut self) -> Option<Self::Item> {
            let next = match self.0.next() {
                Some(item) => item,
                None => return None,
            };
            if let GenericParam::Lifetime(lifetime) = next {
                Some(lifetime)
            } else {
                self.next()
            }
        }
    }
    pub struct TypeParams<'a>(Iter<'a, GenericParam>);
    impl<'a> Iterator for TypeParams<'a> {
        type Item = &'a TypeParam;
        fn next(&mut self) -> Option<Self::Item> {
            let next = match self.0.next() {
                Some(item) => item,
                None => return None,
            };
            if let GenericParam::Type(type_param) = next {
                Some(type_param)
            } else {
                self.next()
            }
        }
    }
    pub struct TypeParamsMut<'a>(IterMut<'a, GenericParam>);
    impl<'a> Iterator for TypeParamsMut<'a> {
        type Item = &'a mut TypeParam;
        fn next(&mut self) -> Option<Self::Item> {
            let next = match self.0.next() {
                Some(item) => item,
                None => return None,
            };
            if let GenericParam::Type(type_param) = next {
                Some(type_param)
            } else {
                self.next()
            }
        }
    }
    pub struct ConstParams<'a>(Iter<'a, GenericParam>);
    impl<'a> Iterator for ConstParams<'a> {
        type Item = &'a ConstParam;
        fn next(&mut self) -> Option<Self::Item> {
            let next = match self.0.next() {
                Some(item) => item,
                None => return None,
            };
            if let GenericParam::Const(const_param) = next {
                Some(const_param)
            } else {
                self.next()
            }
        }
    }
    pub struct ConstParamsMut<'a>(IterMut<'a, GenericParam>);
    impl<'a> Iterator for ConstParamsMut<'a> {
        type Item = &'a mut ConstParam;
        fn next(&mut self) -> Option<Self::Item> {
            let next = match self.0.next() {
                Some(item) => item,
                None => return None,
            };
            if let GenericParam::Const(const_param) = next {
                Some(const_param)
            } else {
                self.next()
            }
        }
    }
    pub struct ImplGenerics<'a>(pub &'a Generics);
    pub struct TypeGenerics<'a>(pub &'a Generics);
    pub struct Turbofish<'a>(pub &'a Generics);
    impl Generics {
        pub fn split_for_impl(&self) -> (ImplGenerics, TypeGenerics, Option<&WhereClause>) {
            (ImplGenerics(self), TypeGenerics(self), self.where_clause.as_ref())
        }
    }
    macro_rules! generics_wrapper_impls {
        ($ty:ident) => {
            impl<'a> Clone for $ty<'a> {
                fn clone(&self) -> Self {
                    $ty(self.0)
                }
            }
            impl<'a> Debug for $ty<'a> {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.debug_tuple(stringify!($ty)).field(self.0).finish()
                }
            }
            impl<'a> Eq for $ty<'a> {}
            impl<'a> PartialEq for $ty<'a> {
                fn eq(&self, other: &Self) -> bool {
                    self.0 == other.0
                }
            }
            impl<'a> Hash for $ty<'a> {
                fn hash<H: Hasher>(&self, state: &mut H) {
                    self.0.hash(state);
                }
            }
        };
    }
    generics_wrapper_impls!(ImplGenerics);
    generics_wrapper_impls!(TypeGenerics);
    generics_wrapper_impls!(Turbofish);
    impl<'a> TypeGenerics<'a> {
        pub fn as_turbofish(&self) -> Turbofish {
            Turbofish(self.0)
        }
    }
    ast_struct! {
        pub struct BoundLifetimes {
            pub for_token: Token![for],
            pub lt_token: Token![<],
            pub lifetimes: Punctuated<GenericParam, Token![,]>,
            pub gt_token: Token![>],
        }
    }
    impl Default for BoundLifetimes {
        fn default() -> Self {
            BoundLifetimes {
                for_token: Default::default(),
                lt_token: Default::default(),
                lifetimes: Punctuated::new(),
                gt_token: Default::default(),
            }
        }
    }
    impl LifetimeParam {
        pub fn new(lifetime: Lifetime) -> Self {
            LifetimeParam {
                attrs: Vec::new(),
                lifetime,
                colon_token: None,
                bounds: Punctuated::new(),
            }
        }
    }
    impl From<Ident> for TypeParam {
        fn from(ident: Ident) -> Self {
            TypeParam {
                attrs: vec![],
                ident,
                colon_token: None,
                bounds: Punctuated::new(),
                eq_token: None,
                default: None,
            }
        }
    }
    ast_enum_of_structs! {
        pub enum TypeParamBound {
            Trait(TraitBound),
            Lifetime(Lifetime),
            Verbatim(TokenStream),
        }
    }
    ast_struct! {
        pub struct TraitBound {
            pub paren_token: Option<tok::Paren>,
            pub modifier: TraitBoundModifier,
            pub lifetimes: Option<BoundLifetimes>,
            pub path: Path,
        }
    }
    ast_enum! {
        pub enum TraitBoundModifier {
            None,
            Maybe(Token![?]),
        }
    }
    ast_struct! {
        pub struct WhereClause {
            pub where_token: Token![where],
            pub predicates: Punctuated<WherePredicate, Token![,]>,
        }
    }
    ast_enum_of_structs! {
        pub enum WherePredicate {
            Lifetime(PredicateLifetime),
            Type(PredicateType),
        }
    }
    ast_struct! {
        pub struct PredicateLifetime {
            pub lifetime: Lifetime,
            pub colon_token: Token![:],
            pub bounds: Punctuated<Lifetime, Token![+]>,
        }
    }
    ast_struct! {
        pub struct PredicateType {
            pub lifetimes: Option<BoundLifetimes>,
            pub bounded_ty: Type,
            pub colon_token: Token![:],
            pub bounds: Punctuated<TypeParamBound, Token![+]>,
        }
    }
}
pub use crate::generic::{
    BoundLifetimes, ConstParam, GenericParam, Generics, ImplGenerics, LifetimeParam, PredicateLifetime, PredicateType,
    TraitBound, TraitBoundModifier, Turbofish, TypeGenerics, TypeParam, TypeParamBound, WhereClause, WherePredicate,
};
mod item;
pub use crate::item::{
    FnArg, ForeignItem, ForeignItemFn, ForeignItemMacro, ForeignItemStatic, ForeignItemType, ImplItem, ImplItemConst,
    ImplItemFn, ImplItemMacro, ImplItemType, ImplRestriction, Item, ItemConst, ItemEnum, ItemExternCrate, ItemFn,
    ItemForeignMod, ItemImpl, ItemMacro, ItemMod, ItemStatic, ItemStruct, ItemTrait, ItemTraitAlias, ItemType,
    ItemUnion, ItemUse, Receiver, Signature, StaticMutability, TraitItem, TraitItemConst, TraitItemFn, TraitItemMacro,
    TraitItemType, UseGlob, UseGroup, UseName, UsePath, UseRename, UseTree, Variadic,
};
pub mod punctuated;
use punctuated::Punctuated;
mod lit;
pub use crate::lit::{Lit, LitBool, LitByte, LitByteStr, LitChar, LitFloat, LitInt, LitStr, StrStyle};
mod pat {
    use super::*;
    use crate::punctuated::Punctuated;
    use proc_macro2::TokenStream;
    ast_enum_of_structs! {
        pub enum Pat {
            Const(PatConst),
            Ident(PatIdent),
            Lit(PatLit),
            Macro(PatMacro),
            Or(PatOr),
            Paren(PatParen),
            Path(PatPath),
            Range(PatRange),
            Reference(PatReference),
            Rest(PatRest),
            Slice(PatSlice),
            Struct(PatStruct),
            Tuple(PatTuple),
            TupleStruct(PatTupleStruct),
            Type(PatType),
            Verbatim(TokenStream),
            Wild(PatWild),
        }
    }
    ast_struct! {
        pub struct PatIdent {
            pub attrs: Vec<Attribute>,
            pub by_ref: Option<Token![ref]>,
            pub mutability: Option<Token![mut]>,
            pub ident: Ident,
            pub subpat: Option<(Token![@], Box<Pat>)>,
        }
    }
    ast_struct! {
        pub struct PatOr {
            pub attrs: Vec<Attribute>,
            pub leading_vert: Option<Token![|]>,
            pub cases: Punctuated<Pat, Token![|]>,
        }
    }
    ast_struct! {
        pub struct PatParen {
            pub attrs: Vec<Attribute>,
            pub paren_token: tok::Paren,
            pub pat: Box<Pat>,
        }
    }
    ast_struct! {
        pub struct PatReference {
            pub attrs: Vec<Attribute>,
            pub and_token: Token![&],
            pub mutability: Option<Token![mut]>,
            pub pat: Box<Pat>,
        }
    }
    ast_struct! {
        pub struct PatRest {
            pub attrs: Vec<Attribute>,
            pub dot2_token: Token![..],
        }
    }
    ast_struct! {
        pub struct PatSlice {
            pub attrs: Vec<Attribute>,
            pub bracket_token: tok::Bracket,
            pub elems: Punctuated<Pat, Token![,]>,
        }
    }
    ast_struct! {
        pub struct PatStruct {
            pub attrs: Vec<Attribute>,
            pub qself: Option<QSelf>,
            pub path: Path,
            pub brace_token: tok::Brace,
            pub fields: Punctuated<FieldPat, Token![,]>,
            pub rest: Option<PatRest>,
        }
    }
    ast_struct! {
        pub struct PatTuple {
            pub attrs: Vec<Attribute>,
            pub paren_token: tok::Paren,
            pub elems: Punctuated<Pat, Token![,]>,
        }
    }
    ast_struct! {
        pub struct PatTupleStruct {
            pub attrs: Vec<Attribute>,
            pub qself: Option<QSelf>,
            pub path: Path,
            pub paren_token: tok::Paren,
            pub elems: Punctuated<Pat, Token![,]>,
        }
    }
    ast_struct! {
        pub struct PatType {
            pub attrs: Vec<Attribute>,
            pub pat: Box<Pat>,
            pub colon_token: Token![:],
            pub ty: Box<Type>,
        }
    }
    ast_struct! {
        pub struct PatWild {
            pub attrs: Vec<Attribute>,
            pub underscore_token: Token![_],
        }
    }
    ast_struct! {
        pub struct FieldPat {
            pub attrs: Vec<Attribute>,
            pub member: Member,
            pub colon_token: Option<Token![:]>,
            pub pat: Box<Pat>,
        }
    }
}
pub use crate::pat::{
    FieldPat, Pat, PatIdent, PatOr, PatParen, PatReference, PatRest, PatSlice, PatStruct, PatTuple, PatTupleStruct,
    PatType, PatWild,
};
mod path {
    use super::*;
    use crate::punctuated::Punctuated;
    ast_struct! {
        pub struct Path {
            pub leading_colon: Option<Token![::]>,
            pub segments: Punctuated<PathSegment, Token![::]>,
        }
    }
    impl<T> From<T> for Path
    where
        T: Into<PathSegment>,
    {
        fn from(segment: T) -> Self {
            let mut path = Path {
                leading_colon: None,
                segments: Punctuated::new(),
            };
            path.segments.push_value(segment.into());
            path
        }
    }
    impl Path {
        pub fn is_ident<I: ?Sized>(&self, ident: &I) -> bool
        where
            Ident: PartialEq<I>,
        {
            match self.get_ident() {
                Some(id) => id == ident,
                None => false,
            }
        }
        pub fn get_ident(&self) -> Option<&Ident> {
            if self.leading_colon.is_none() && self.segments.len() == 1 && self.segments[0].arguments.is_none() {
                Some(&self.segments[0].ident)
            } else {
                None
            }
        }
    }
    ast_struct! {
        pub struct PathSegment {
            pub ident: Ident,
            pub arguments: PathArguments,
        }
    }
    impl<T> From<T> for PathSegment
    where
        T: Into<Ident>,
    {
        fn from(ident: T) -> Self {
            PathSegment {
                ident: ident.into(),
                arguments: PathArguments::None,
            }
        }
    }
    ast_enum! {
        pub enum PathArguments {
            None,
            AngleBracketed(AngleBracketedGenericArguments),
            Parenthesized(ParenthesizedGenericArguments),
        }
    }
    impl Default for PathArguments {
        fn default() -> Self {
            PathArguments::None
        }
    }
    impl PathArguments {
        pub fn is_empty(&self) -> bool {
            match self {
                PathArguments::None => true,
                PathArguments::AngleBracketed(bracketed) => bracketed.args.is_empty(),
                PathArguments::Parenthesized(_) => false,
            }
        }
        pub fn is_none(&self) -> bool {
            match self {
                PathArguments::None => true,
                PathArguments::AngleBracketed(_) | PathArguments::Parenthesized(_) => false,
            }
        }
    }
    ast_enum! {
        pub enum GenericArgument {
            Lifetime(Lifetime),
            Type(Type),
            Const(Expr),
            AssocType(AssocType),
            AssocConst(AssocConst),
            Constraint(Constraint),
        }
    }
    ast_struct! {
        pub struct AngleBracketedGenericArguments {
            pub colon2_token: Option<Token![::]>,
            pub lt_token: Token![<],
            pub args: Punctuated<GenericArgument, Token![,]>,
            pub gt_token: Token![>],
        }
    }
    ast_struct! {
        pub struct AssocType {
            pub ident: Ident,
            pub generics: Option<AngleBracketedGenericArguments>,
            pub eq_token: Token![=],
            pub ty: Type,
        }
    }
    ast_struct! {
        pub struct AssocConst {
            pub ident: Ident,
            pub generics: Option<AngleBracketedGenericArguments>,
            pub eq_token: Token![=],
            pub value: Expr,
        }
    }
    ast_struct! {
        pub struct Constraint {
            pub ident: Ident,
            pub generics: Option<AngleBracketedGenericArguments>,
            pub colon_token: Token![:],
            pub bounds: Punctuated<TypeParamBound, Token![+]>,
        }
    }
    ast_struct! {
        pub struct ParenthesizedGenericArguments {
            pub paren_token: tok::Paren,
            pub inputs: Punctuated<Type, Token![,]>,
            pub output: ReturnType,
        }
    }
    ast_struct! {
        pub struct QSelf {
            pub lt_token: Token![<],
            pub ty: Box<Type>,
            pub position: usize,
            pub as_token: Option<Token![as]>,
            pub gt_token: Token![>],
        }
    }
}
pub use crate::path::{
    AngleBracketedGenericArguments, AssocConst, AssocType, Constraint, GenericArgument, ParenthesizedGenericArguments,
    Path, PathArguments, PathSegment, QSelf,
};

ast_struct! {
    pub struct Block {
        pub brace_token: tok::Brace,
        pub stmts: Vec<Stmt>,
    }
}
ast_enum! {
    pub enum Stmt {
        Local(Local),
        Item(Item),
        Expr(Expr, Option<Token![;]>),
        Macro(StmtMacro),
    }
}
ast_struct! {
    pub struct Local {
        pub attrs: Vec<Attribute>,
        pub let_token: Token![let],
        pub pat: Pat,
        pub init: Option<LocalInit>,
        pub semi_token: Token![;],
    }
}
ast_struct! {
    pub struct LocalInit {
        pub eq_token: Token![=],
        pub expr: Box<Expr>,
        pub diverge: Option<(Token![else], Box<Expr>)>,
    }
}
ast_struct! {
    pub struct StmtMacro {
        pub attrs: Vec<Attribute>,
        pub mac: Macro,
        pub semi_token: Option<Token![;]>,
    }
}

mod ty {
    use super::{punctuated::Punctuated, *};
    use proc_macro2::TokenStream;

    ast_enum_of_structs! {
        pub enum Type {
            Array(TypeArray),
            BareFn(TypeBareFn),
            Group(TypeGroup),
            ImplTrait(TypeImplTrait),
            Infer(TypeInfer),
            Macro(TypeMacro),
            Never(TypeNever),
            Paren(TypeParen),
            Path(TypePath),
            Ptr(TypePtr),
            Reference(TypeReference),
            Slice(TypeSlice),
            TraitObject(TypeTraitObject),
            Tuple(TypeTuple),
            Verbatim(TokenStream),
        }
    }
    ast_struct! {
        pub struct TypeArray {
            pub bracket_token: tok::Bracket,
            pub elem: Box<Type>,
            pub semi_token: Token![;],
            pub len: Expr,
        }
    }
    ast_struct! {
        pub struct TypeBareFn {
            pub lifetimes: Option<BoundLifetimes>,
            pub unsafety: Option<Token![unsafe]>,
            pub abi: Option<Abi>,
            pub fn_token: Token![fn],
            pub paren_token: tok::Paren,
            pub inputs: Punctuated<BareFnArg, Token![,]>,
            pub variadic: Option<BareVariadic>,
            pub output: ReturnType,
        }
    }
    ast_struct! {
        pub struct TypeGroup {
            pub group_token: tok::Group,
            pub elem: Box<Type>,
        }
    }
    ast_struct! {
        pub struct TypeImplTrait {
            pub impl_token: Token![impl],
            pub bounds: Punctuated<TypeParamBound, Token![+]>,
        }
    }
    ast_struct! {
        pub struct TypeInfer {
            pub underscore_token: Token![_],
        }
    }
    ast_struct! {
        pub struct TypeMacro {
            pub mac: Macro,
        }
    }
    ast_struct! {
        pub struct TypeNever {
            pub bang_token: Token![!],
        }
    }
    ast_struct! {
        pub struct TypeParen {
            pub paren_token: tok::Paren,
            pub elem: Box<Type>,
        }
    }
    ast_struct! {
        pub struct TypePath {
            pub qself: Option<QSelf>,
            pub path: Path,
        }
    }
    ast_struct! {
        pub struct TypePtr {
            pub star_token: Token![*],
            pub const_token: Option<Token![const]>,
            pub mutability: Option<Token![mut]>,
            pub elem: Box<Type>,
        }
    }
    ast_struct! {
        pub struct TypeReference {
            pub and_token: Token![&],
            pub lifetime: Option<Lifetime>,
            pub mutability: Option<Token![mut]>,
            pub elem: Box<Type>,
        }
    }
    ast_struct! {
        pub struct TypeSlice {
            pub bracket_token: tok::Bracket,
            pub elem: Box<Type>,
        }
    }
    ast_struct! {
        pub struct TypeTraitObject {
            pub dyn_token: Option<Token![dyn]>,
            pub bounds: Punctuated<TypeParamBound, Token![+]>,
        }
    }
    ast_struct! {
        pub struct TypeTuple {
            pub paren_token: tok::Paren,
            pub elems: Punctuated<Type, Token![,]>,
        }
    }
    ast_struct! {
        pub struct Abi {
            pub extern_token: Token![extern],
            pub name: Option<LitStr>,
        }
    }
    ast_struct! {
        pub struct BareFnArg {
            pub attrs: Vec<Attribute>,
            pub name: Option<(Ident, Token![:])>,
            pub ty: Type,
        }
    }
    ast_struct! {
        pub struct BareVariadic {
            pub attrs: Vec<Attribute>,
            pub name: Option<(Ident, Token![:])>,
            pub dots: Token![...],
            pub comma: Option<Token![,]>,
        }
    }
    ast_enum! {
        pub enum ReturnType {
            Default,
            Type(Token![->], Box<Type>),
        }
    }
}
pub use crate::ty::{
    Abi, BareFnArg, BareVariadic, ReturnType, Type, TypeArray, TypeBareFn, TypeGroup, TypeImplTrait, TypeInfer,
    TypeMacro, TypeNever, TypeParen, TypePath, TypePtr, TypeReference, TypeSlice, TypeTraitObject, TypeTuple,
};

pub struct BigInt {
    digits: Vec<u8>,
}
impl BigInt {
    pub fn new() -> Self {
        BigInt { digits: Vec::new() }
    }
    pub fn to_string(&self) -> String {
        let mut y = String::with_capacity(self.digits.len());
        let mut has_nonzero = false;
        for x in self.digits.iter().rev() {
            has_nonzero |= *x != 0;
            if has_nonzero {
                y.push((*x + b'0') as char);
            }
        }
        if y.is_empty() {
            y.push('0');
        }
        y
    }
    fn reserve_two_digits(&mut self) {
        let len = self.digits.len();
        let desired = len + !self.digits.ends_with(&[0, 0]) as usize + !self.digits.ends_with(&[0]) as usize;
        self.digits.resize(desired, 0);
    }
}
impl ops::AddAssign<u8> for BigInt {
    fn add_assign(&mut self, mut increment: u8) {
        self.reserve_two_digits();
        let mut i = 0;
        while increment > 0 {
            let sum = self.digits[i] + increment;
            self.digits[i] = sum % 10;
            increment = sum / 10;
            i += 1;
        }
    }
}
impl ops::MulAssign<u8> for BigInt {
    fn mul_assign(&mut self, base: u8) {
        self.reserve_two_digits();
        let mut carry = 0;
        for digit in &mut self.digits {
            let prod = *digit * base + carry;
            *digit = prod % 10;
            carry = prod / 10;
        }
    }
}

ast_struct! {
    pub struct Variant {
        pub attrs: Vec<Attribute>,
        pub ident: Ident,
        pub fields: Fields,
        pub discriminant: Option<(Token![=], Expr)>,
    }
}
ast_enum_of_structs! {
    pub enum Fields {
        Named(FieldsNamed),
        Unnamed(FieldsUnnamed),
        Unit,
    }
}
ast_struct! {
    pub struct FieldsNamed {
        pub brace_token: tok::Brace,
        pub named: Punctuated<Field, Token![,]>,
    }
}
ast_struct! {
    pub struct FieldsUnnamed {
        pub paren_token: tok::Paren,
        pub unnamed: Punctuated<Field, Token![,]>,
    }
}
impl Fields {
    pub fn iter(&self) -> punctuated::Iter<Field> {
        match self {
            Fields::Unit => crate::punctuated::empty_punctuated_iter(),
            Fields::Named(f) => f.named.iter(),
            Fields::Unnamed(f) => f.unnamed.iter(),
        }
    }
    pub fn iter_mut(&mut self) -> punctuated::IterMut<Field> {
        match self {
            Fields::Unit => crate::punctuated::empty_punctuated_iter_mut(),
            Fields::Named(f) => f.named.iter_mut(),
            Fields::Unnamed(f) => f.unnamed.iter_mut(),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Fields::Unit => 0,
            Fields::Named(f) => f.named.len(),
            Fields::Unnamed(f) => f.unnamed.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        match self {
            Fields::Unit => true,
            Fields::Named(f) => f.named.is_empty(),
            Fields::Unnamed(f) => f.unnamed.is_empty(),
        }
    }
}
impl IntoIterator for Fields {
    type Item = Field;
    type IntoIter = punctuated::IntoIter<Field>;
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Fields::Unit => Punctuated::<Field, ()>::new().into_iter(),
            Fields::Named(f) => f.named.into_iter(),
            Fields::Unnamed(f) => f.unnamed.into_iter(),
        }
    }
}
impl<'a> IntoIterator for &'a Fields {
    type Item = &'a Field;
    type IntoIter = punctuated::Iter<'a, Field>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a> IntoIterator for &'a mut Fields {
    type Item = &'a mut Field;
    type IntoIter = punctuated::IterMut<'a, Field>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
ast_struct! {
    pub struct Field {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub mutability: FieldMutability,
        pub ident: Option<Ident>,
        pub colon_token: Option<Token![:]>,
        pub ty: Type,
    }
}
ast_struct! {
    pub struct DeriveInput {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub ident: Ident,
        pub generics: Generics,
        pub data: Data,
    }
}
ast_enum! {
    pub enum Data {
        Struct(DataStruct),
        Enum(DataEnum),
        Union(DataUnion),
    }
}
ast_struct! {
    pub struct DataStruct {
        pub struct_token: Token![struct],
        pub fields: Fields,
        pub semi_token: Option<Token![;]>,
    }
}
ast_struct! {
    pub struct DataEnum {
        pub enum_token: Token![enum],
        pub brace_token: tok::Brace,
        pub variants: Punctuated<Variant, Token![,]>,
    }
}
ast_struct! {
    pub struct DataUnion {
        pub union_token: Token![union],
        pub fields: FieldsNamed,
    }
}

mod err {
    use crate::{buffer::Cursor, ThreadBound};
    use proc_macro2::{Delimiter, Group, Ident, LexError, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
    use quote::ToTokens;
    use std::fmt::{self, Debug, Display};
    use std::slice;
    use std::vec;
    pub type Result<T> = std::result::Result<T, Err>;
    pub struct Err {
        messages: Vec<ErrMsg>,
    }
    struct ErrMsg {
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
        Err: Send + Sync;
    impl Err {
        pub fn new<T: Display>(span: Span, message: T) -> Self {
            return new(span, message.to_string());
            fn new(span: Span, message: String) -> Err {
                Err {
                    messages: vec![ErrMsg {
                        span: ThreadBound::new(SpanRange { start: span, end: span }),
                        message,
                    }],
                }
            }
        }
        pub fn new_spanned<T: ToTokens, U: Display>(tokens: T, message: U) -> Self {
            return new_spanned(tokens.into_token_stream(), message.to_string());
            fn new_spanned(tokens: TokenStream, message: String) -> Err {
                let mut iter = tokens.into_iter();
                let start = iter.next().map_or_else(Span::call_site, |t| t.span());
                let end = iter.last().map_or(start, |t| t.span());
                Err {
                    messages: vec![ErrMsg {
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
            self.messages.iter().map(ErrMsg::to_compile_error).collect()
        }
        pub fn into_compile_error(self) -> TokenStream {
            self.to_compile_error()
        }
        pub fn combine(&mut self, another: Err) {
            self.messages.extend(another.messages);
        }
    }
    impl ErrMsg {
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
    pub fn new_at<T: Display>(scope: Span, cursor: Cursor, message: T) -> Err {
        if cursor.eof() {
            Err::new(scope, format!("unexpected end of input, {}", message))
        } else {
            let span = crate::buffer::open_span_of_group(cursor);
            Err::new(span, message)
        }
    }
    pub fn new2<T: Display>(start: Span, end: Span, message: T) -> Err {
        return new2(start, end, message.to_string());
        fn new2(start: Span, end: Span, message: String) -> Err {
            Err {
                messages: vec![ErrMsg {
                    span: ThreadBound::new(SpanRange { start, end }),
                    message,
                }],
            }
        }
    }
    impl Debug for Err {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            if self.messages.len() == 1 {
                formatter.debug_tuple("Error").field(&self.messages[0]).finish()
            } else {
                formatter.debug_tuple("Error").field(&self.messages).finish()
            }
        }
    }
    impl Debug for ErrMsg {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            Debug::fmt(&self.message, formatter)
        }
    }
    impl Display for Err {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str(&self.messages[0].message)
        }
    }
    impl Clone for Err {
        fn clone(&self) -> Self {
            Err {
                messages: self.messages.clone(),
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
                messages: self.messages.into_iter(),
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
                messages: vec![self.messages.next()?],
            })
        }
    }
    impl<'a> IntoIterator for &'a Err {
        type Item = Err;
        type IntoIter = Iter<'a>;
        fn into_iter(self) -> Self::IntoIter {
            Iter {
                messages: self.messages.iter(),
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
                messages: vec![self.messages.next()?.clone()],
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
pub use crate::err::{Err, Result};

pub mod ext {
    use crate::buffer::Cursor;
    use crate::parse::Peek;
    use crate::parse::{ParseStream, Result};
    use crate::sealed::lookahead;
    use crate::tok::CustomToken;
    use proc_macro2::Ident;
    pub trait IdentExt: Sized + private::Sealed {
        fn parse_any(input: ParseStream) -> Result<Self>;
        #[allow(non_upper_case_globals)]
        const peek_any: private::PeekFn = private::PeekFn;
        fn unraw(&self) -> Ident;
    }
    impl IdentExt for Ident {
        fn parse_any(input: ParseStream) -> Result<Self> {
            input.step(|cursor| match cursor.ident() {
                Some((ident, rest)) => Ok((ident, rest)),
                None => Err(cursor.error("expected ident")),
            })
        }
        fn unraw(&self) -> Ident {
            let string = self.to_string();
            if let Some(string) = string.strip_prefix("r#") {
                Ident::new(string, self.span())
            } else {
                self.clone()
            }
        }
    }
    impl Peek for private::PeekFn {
        type Token = private::IdentAny;
    }
    impl CustomToken for private::IdentAny {
        fn peek(cursor: Cursor) -> bool {
            cursor.ident().is_some()
        }
        fn display() -> &'static str {
            "identifier"
        }
    }
    impl lookahead::Sealed for private::PeekFn {}
    mod private {
        use proc_macro2::Ident;
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

ast_struct! {
    pub struct File {
        pub shebang: Option<String>,
        pub attrs: Vec<Attribute>,
        pub items: Vec<Item>,
    }
}

mod ident {
    use crate::lookahead;
    pub use proc_macro2::Ident;
    #[allow(non_snake_case)]
    pub fn Ident(x: lookahead::TokenMarker) -> Ident {
        match x {}
    }
    macro_rules! ident_from_token {
        ($token:ident) => {
            impl From<Token![$token]> for Ident {
                fn from(token: Token![$token]) -> Ident {
                    Ident::new(stringify!($token), token.span)
                }
            }
        };
    }
    ident_from_token!(self);
    ident_from_token!(Self);
    ident_from_token!(super);
    ident_from_token!(crate);
    ident_from_token!(extern);
    impl From<Token![_]> for Ident {
        fn from(token: Token![_]) -> Ident {
            Ident::new("_", token.span)
        }
    }
    pub fn xid_ok(symbol: &str) -> bool {
        let mut chars = symbol.chars();
        let first = chars.next().unwrap();
        if !(first == '_' || unicode_ident::is_xid_start(first)) {
            return false;
        }
        for ch in chars {
            if !unicode_ident::is_xid_continue(ch) {
                return false;
            }
        }
        true
    }
}
pub use crate::ident::Ident;

pub struct Lifetime {
    pub apostrophe: Span,
    pub ident: Ident,
}
impl Lifetime {
    pub fn new(symbol: &str, span: Span) -> Self {
        if !symbol.starts_with('\'') {
            panic!(
                "lifetime name must start with apostrophe as in \"'a\", got {:?}",
                symbol
            );
        }
        if symbol == "'" {
            panic!("lifetime name must not be empty");
        }
        if !crate::ident::xid_ok(&symbol[1..]) {
            panic!("{:?} is not a valid lifetime name", symbol);
        }
        Lifetime {
            apostrophe: span,
            ident: Ident::new(&symbol[1..], span),
        }
    }
    pub fn span(&self) -> Span {
        self.apostrophe.join(self.ident.span()).unwrap_or(self.apostrophe)
    }
    pub fn set_span(&mut self, span: Span) {
        self.apostrophe = span;
        self.ident.set_span(span);
    }
}
impl Display for Lifetime {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        "'".fmt(formatter)?;
        self.ident.fmt(formatter)
    }
}
impl Clone for Lifetime {
    fn clone(&self) -> Self {
        Lifetime {
            apostrophe: self.apostrophe,
            ident: self.ident.clone(),
        }
    }
}
impl PartialEq for Lifetime {
    fn eq(&self, other: &Lifetime) -> bool {
        self.ident.eq(&other.ident)
    }
}
impl Eq for Lifetime {}
impl PartialOrd for Lifetime {
    fn partial_cmp(&self, other: &Lifetime) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Lifetime {
    fn cmp(&self, other: &Lifetime) -> Ordering {
        self.ident.cmp(&other.ident)
    }
}
impl Hash for Lifetime {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.ident.hash(h);
    }
}
#[allow(non_snake_case)]
pub fn Lifetime(marker: lookahead::TokenMarker) -> Lifetime {
    match marker {}
}

mod lookahead {
    use crate::buffer::Cursor;
    use crate::err::{self, Err};
    use crate::sealed::lookahead::Sealed;
    use crate::tok::Token;
    use crate::IntoSpans;
    use proc_macro2::{Delimiter, Span};
    use std::cell::RefCell;
    pub struct Lookahead1<'a> {
        scope: Span,
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

    pub fn new(scope: Span, cursor: Cursor) -> Lookahead1 {
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

    pub fn is_delimiter(cursor: Cursor, delimiter: Delimiter) -> bool {
        cursor.group(delimiter).is_some()
    }

    impl<F: Copy + FnOnce(TokenMarker) -> T, T: Token> Sealed for F {}
}

ast_struct! {
    pub struct Macro {
        pub path: Path,
        pub bang_token: Token![!],
        pub delimiter: MacroDelimiter,
        pub tokens: TokenStream,
    }
}
ast_enum! {
    pub enum MacroDelimiter {
        Paren(Paren),
        Brace(Brace),
        Bracket(Bracket),
    }
}
impl MacroDelimiter {
    pub fn span(&self) -> &DelimSpan {
        match self {
            MacroDelimiter::Paren(token) => &token.span,
            MacroDelimiter::Brace(token) => &token.span,
            MacroDelimiter::Bracket(token) => &token.span,
        }
    }
}
impl Macro {
    pub fn parse_body<T: Parse>(&self) -> Result<T> {
        self.parse_body_with(T::parse)
    }

    pub fn parse_body_with<F: Parser>(&self, parser: F) -> Result<F::Output> {
        let scope = self.delimiter.span().close();
        crate::parse::parse_scoped(parser, scope, self.tokens.clone())
    }
}
fn mac_parse_delimiter(input: ParseStream) -> Result<(MacroDelimiter, TokenStream)> {
    input.step(|cursor| {
        if let Some((TokenTree::Group(g), rest)) = cursor.token_tree() {
            let span = g.delim_span();
            let delimiter = match g.delimiter() {
                Delimiter::Parenthesis => MacroDelimiter::Paren(Paren(span)),
                Delimiter::Brace => MacroDelimiter::Brace(Brace(span)),
                Delimiter::Bracket => MacroDelimiter::Bracket(Bracket(span)),
                Delimiter::None => {
                    return Err(cursor.error("expected delimiter"));
                },
            };
            Ok(((delimiter, g.stream()), rest))
        } else {
            Err(cursor.error("expected delimiter"))
        }
    })
}

ast_enum! {
    pub enum BinOp {
        Add(Token![+]),
        Sub(Token![-]),
        Mul(Token![*]),
        Div(Token![/]),
        Rem(Token![%]),
        And(Token![&&]),
        Or(Token![||]),
        BitXor(Token![^]),
        BitAnd(Token![&]),
        BitOr(Token![|]),
        Shl(Token![<<]),
        Shr(Token![>>]),
        Eq(Token![==]),
        Lt(Token![<]),
        Le(Token![<=]),
        Ne(Token![!=]),
        Ge(Token![>=]),
        Gt(Token![>]),
        AddAssign(Token![+=]),
        SubAssign(Token![-=]),
        MulAssign(Token![*=]),
        DivAssign(Token![/=]),
        RemAssign(Token![%=]),
        BitXorAssign(Token![^=]),
        BitAndAssign(Token![&=]),
        BitOrAssign(Token![|=]),
        ShlAssign(Token![<<=]),
        ShrAssign(Token![>>=]),
    }
}
ast_enum! {
    pub enum UnOp {
        Deref(Token![*]),
        Not(Token![!]),
        Neg(Token![-]),
    }
}

pub mod parse;
mod parse_macro_input {
    #[macro_export]
    macro_rules! parse_macro_input {
        ($tokenstream:ident as $ty:ty) => {
            match $crate::parse::<$ty>($tokenstream) {
                $crate::__private::Ok(data) => data,
                $crate::__private::Err(err) => {
                    return $crate::__private::TokenStream::from(err.to_compile_error());
                },
            }
        };
        ($tokenstream:ident with $parser:path) => {
            match $crate::parse::Parser::parse($parser, $tokenstream) {
                $crate::__private::Ok(data) => data,
                $crate::__private::Err(err) => {
                    return $crate::__private::TokenStream::from(err.to_compile_error());
                },
            }
        };
        ($tokenstream:ident) => {
            $crate::parse_macro_input!($tokenstream as _)
        };
    }
}
mod parse_quote {
    #[macro_export]
    macro_rules! parse_quote {
    ($($tt:tt)*) => {
        $crate::__private::parse_quote($crate::__private::quote::quote!($($tt)*))
    };
}
    #[macro_export]
    macro_rules! parse_quote_spanned {
    ($span:expr=> $($tt:tt)*) => {
        $crate::__private::parse_quote($crate::__private::quote::quote_spanned!($span=> $($tt)*))
    };
}
    use crate::parse::{Parse, ParseStream, Parser, Result};
    use proc_macro2::TokenStream;
    pub fn parse<T: ParseQuote>(token_stream: TokenStream) -> T {
        let parser = T::parse;
        match parser.parse2(token_stream) {
            Ok(t) => t,
            Err(err) => panic!("{}", err),
        }
    }
    pub trait ParseQuote: Sized {
        fn parse(input: ParseStream) -> Result<Self>;
    }
    impl<T: Parse> ParseQuote for T {
        fn parse(input: ParseStream) -> Result<Self> {
            <T as Parse>::parse(input)
        }
    }
    use crate::punctuated::Punctuated;

    use crate::{attr, Attribute};
    use crate::{Block, Pat, Stmt};

    impl ParseQuote for Attribute {
        fn parse(input: ParseStream) -> Result<Self> {
            if input.peek(Token![#]) && input.peek2(Token![!]) {
                parsing::single_parse_inner(input)
            } else {
                parsing::single_parse_outer(input)
            }
        }
    }
    impl ParseQuote for Pat {
        fn parse(input: ParseStream) -> Result<Self> {
            Pat::parse_multi_with_leading_vert(input)
        }
    }
    impl ParseQuote for Box<Pat> {
        fn parse(input: ParseStream) -> Result<Self> {
            <Pat as ParseQuote>::parse(input).map(Box::new)
        }
    }
    impl<T: Parse, P: Parse> ParseQuote for Punctuated<T, P> {
        fn parse(input: ParseStream) -> Result<Self> {
            Self::parse_terminated(input)
        }
    }
    impl ParseQuote for Vec<Stmt> {
        fn parse(input: ParseStream) -> Result<Self> {
            Block::parse_within(input)
        }
    }
}

struct TokensOrDefault<'a, T: 'a>(pub &'a Option<T>);
impl<'a, T> ToTokens for TokensOrDefault<'a, T>
where
    T: ToTokens + Default,
{
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self.0 {
            Some(t) => t.to_tokens(tokens),
            None => T::default().to_tokens(tokens),
        }
    }
}

mod restriction {
    use super::*;
    ast_enum! {
        pub enum Visibility {
            Public(Token![pub]),
            Restricted(VisRestricted),
            Inherited,
        }
    }
    ast_struct! {
        pub struct VisRestricted {
            pub pub_token: Token![pub],
            pub paren_token: tok::Paren,
            pub in_token: Option<Token![in]>,
            pub path: Box<Path>,
        }
    }
    ast_enum! {
        pub enum FieldMutability {
            None,
        }
    }
}
pub use crate::restriction::{FieldMutability, VisRestricted, Visibility};
mod sealed {
    pub mod lookahead {
        pub trait Sealed: Copy {}
    }
}

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
        let mut group = Group::new(Delimiter::None, TokenStream::new());
        group.set_span(self);
        group.delim_span()
    }
}
impl IntoSpans<DelimSpan> for DelimSpan {
    fn into_spans(self) -> DelimSpan {
        self
    }
}

pub trait Spanned: private::Sealed {
    fn span(&self) -> Span;
}
impl<T: ?Sized + spanned::Spanned> Spanned for T {
    fn span(&self) -> Span {
        self.__span()
    }
}
mod private {
    use super::*;
    pub trait Sealed {}
    impl<T: ?Sized + spanned::Spanned> Sealed for T {}
}

struct ThreadBound<T> {
    value: T,
    thread_id: ThreadId,
}
unsafe impl<T> Sync for ThreadBound<T> {}
unsafe impl<T: Copy> Send for ThreadBound<T> {}
impl<T> ThreadBound<T> {
    pub fn new(value: T) -> Self {
        ThreadBound {
            value,
            thread_id: thread::current().id(),
        }
    }
    pub fn get(&self) -> Option<&T> {
        if thread::current().id() == self.thread_id {
            Some(&self.value)
        } else {
            None
        }
    }
}
impl<T: Debug> Debug for ThreadBound<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self.get() {
            Some(value) => Debug::fmt(value, formatter),
            None => formatter.write_str("unknown"),
        }
    }
}
impl<T: Copy> Copy for ThreadBound<T> {}
impl<T: Copy> Clone for ThreadBound<T> {
    fn clone(&self) -> Self {
        *self
    }
}

struct TokenTreeHelper<'a>(pub &'a TokenTree);
impl<'a> PartialEq for TokenTreeHelper<'a> {
    fn eq(&self, other: &Self) -> bool {
        use proc_macro2::Spacing;
        match (self.0, other.0) {
            (TokenTree::Group(g1), TokenTree::Group(g2)) => {
                match (g1.delimiter(), g2.delimiter()) {
                    (Delimiter::Parenthesis, Delimiter::Parenthesis)
                    | (Delimiter::Brace, Delimiter::Brace)
                    | (Delimiter::Bracket, Delimiter::Bracket)
                    | (Delimiter::None, Delimiter::None) => {},
                    _ => return false,
                }
                let s1 = g1.stream().into_iter();
                let mut s2 = g2.stream().into_iter();
                for item1 in s1 {
                    let item2 = match s2.next() {
                        Some(item) => item,
                        None => return false,
                    };
                    if TokenTreeHelper(&item1) != TokenTreeHelper(&item2) {
                        return false;
                    }
                }
                s2.next().is_none()
            },
            (TokenTree::Punct(o1), TokenTree::Punct(o2)) => {
                o1.as_char() == o2.as_char()
                    && match (o1.spacing(), o2.spacing()) {
                        (Spacing::Alone, Spacing::Alone) | (Spacing::Joint, Spacing::Joint) => true,
                        _ => false,
                    }
            },
            (TokenTree::Literal(l1), TokenTree::Literal(l2)) => l1.to_string() == l2.to_string(),
            (TokenTree::Ident(s1), TokenTree::Ident(s2)) => s1 == s2,
            _ => false,
        }
    }
}
impl<'a> Hash for TokenTreeHelper<'a> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        use proc_macro2::Spacing;
        match self.0 {
            TokenTree::Group(g) => {
                0u8.hash(h);
                match g.delimiter() {
                    Delimiter::Parenthesis => 0u8.hash(h),
                    Delimiter::Brace => 1u8.hash(h),
                    Delimiter::Bracket => 2u8.hash(h),
                    Delimiter::None => 3u8.hash(h),
                }
                for item in g.stream() {
                    TokenTreeHelper(&item).hash(h);
                }
                0xffu8.hash(h); // terminator w/ a variant we don't normally hash
            },
            TokenTree::Punct(op) => {
                1u8.hash(h);
                op.as_char().hash(h);
                match op.spacing() {
                    Spacing::Alone => 0u8.hash(h),
                    Spacing::Joint => 1u8.hash(h),
                }
            },
            TokenTree::Literal(lit) => (2u8, lit.to_string()).hash(h),
            TokenTree::Ident(word) => (3u8, word).hash(h),
        }
    }
}

struct TokenStreamHelper<'a>(pub &'a TokenStream);
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

pub fn verbatim_between<'a>(begin: ParseStream<'a>, end: ParseStream<'a>) -> TokenStream {
    let end = end.cursor();
    let mut cursor = begin.cursor();
    assert!(crate::buffer::same_buffer(end, cursor));
    let mut tokens = TokenStream::new();
    while cursor != end {
        let (tt, next) = cursor.token_tree().unwrap();
        if crate::buffer::cmp_assuming_same_buffer(end, next) == Ordering::Less {
            if let Some((inside, _span, after)) = cursor.group(Delimiter::None) {
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

mod gen {
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
            use crate::punctuated::{Pair, Punctuated};
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
pub use crate::gen::*;
pub mod __private {
    pub use crate::parse_quote::parse as parse_quote;
    pub use crate::parsing::{peek_punct, punct as parse_punct};
    pub use crate::printing::punct as print_punct;
    pub use proc_macro::TokenStream;
    pub use proc_macro2::{Span, TokenStream as TokenStream2};
    pub use quote;
    pub use quote::{ToTokens, TokenStreamExt};
    pub use std::clone::Clone;
    pub use std::cmp::{Eq, PartialEq};
    pub use std::concat;
    pub use std::default::Default;
    pub use std::fmt::{self, Debug, Formatter};
    pub use std::hash::{Hash, Hasher};
    pub use std::marker::Copy;
    pub use std::option::Option::{None, Some};
    pub use std::result::Result::{Err, Ok};
    pub use std::stringify;
    pub type bool = help::Bool;
    pub type str = help::Str;
    mod help {
        pub type Bool = bool;
        pub type Str = str;
    }
    pub struct private(pub(crate) ());
}
pub fn parse<T: parse::Parse>(tokens: proc_macro::TokenStream) -> Result<T> {
    parse::Parser::parse(T::parse, tokens)
}
pub fn parse2<T: parse::Parse>(tokens: proc_macro2::TokenStream) -> Result<T> {
    parse::Parser::parse2(T::parse, tokens)
}
pub fn parse_str<T: parse::Parse>(s: &str) -> Result<T> {
    parse::Parser::parse_str(T::parse, s)
}
pub fn parse_file(mut content: &str) -> Result<File> {
    const BOM: &str = "\u{feff}";
    if content.starts_with(BOM) {
        content = &content[BOM.len()..];
    }
    let mut shebang = None;
    if content.starts_with("#!") {
        let rest = whitespace::ws_skip(&content[2..]);
        if !rest.starts_with('[') {
            if let Some(idx) = content.find('\n') {
                shebang = Some(content[..idx].to_string());
                content = &content[idx..];
            } else {
                shebang = Some(content.to_string());
                content = "";
            }
        }
    }
    let mut file: File = parse_str(content)?;
    file.shebang = shebang;
    Ok(file)
}

pub mod parsing;
pub mod printing;
