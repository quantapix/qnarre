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

use proc_macro2::{extra::DelimSpan, Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
use quote::{spanned, ToTokens};
use std::{
    cmp::Ordering,
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops,
    thread::{self, ThreadId},
};

#[macro_use]
mod mac;

pub struct Parens<'a> {
    pub tok: tok::Paren,
    pub gist: ParseBuffer<'a>,
}
pub fn parse_parens<'a>(x: &ParseBuffer<'a>) -> Result<Parens<'a>> {
    parse_delimited(x, Delimiter::Parenthesis).map(|(span, gist)| Parens {
        tok: tok::Paren(span),
        gist,
    })
}
pub struct Braces<'a> {
    pub token: tok::Brace,
    pub gist: ParseBuffer<'a>,
}
pub fn parse_braces<'a>(x: &ParseBuffer<'a>) -> Result<Braces<'a>> {
    parse_delimited(x, Delimiter::Brace).map(|(span, gist)| Braces {
        token: tok::Brace(span),
        gist,
    })
}
pub struct Brackets<'a> {
    pub token: tok::Bracket,
    pub gist: ParseBuffer<'a>,
}
pub fn parse_brackets<'a>(x: &ParseBuffer<'a>) -> Result<Brackets<'a>> {
    parse_delimited(x, Delimiter::Bracket).map(|(span, gist)| Brackets {
        token: tok::Bracket(span),
        gist,
    })
}
pub struct Group<'a> {
    pub token: tok::Group,
    pub gist: ParseBuffer<'a>,
}
pub fn parse_group<'a>(x: &ParseBuffer<'a>) -> Result<Group<'a>> {
    parse_delimited(x, Delimiter::None).map(|(span, gist)| Group {
        token: tok::Group(span.join()),
        gist,
    })
}
fn parse_delimited<'a>(x: &ParseBuffer<'a>, d: Delimiter) -> Result<(DelimSpan, ParseBuffer<'a>)> {
    x.step(|c| {
        if let Some((gist, span, rest)) = c.group(d) {
            let scope = close_span_of_group(*c);
            let nested = parse::advance_step_cursor(c, gist);
            let unexpected = parse::get_unexpected(x);
            let gist = parse::new_parse_buffer(scope, nested, unexpected);
            Ok(((span, gist), rest))
        } else {
            use Delimiter::*;
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

ast_enum! {
    pub enum AttrStyle {
        Outer,
        Inner(Token![!]),
    }
}
ast_struct! {
    pub struct Attribute {
        pub pound: Token![#],
        pub style: AttrStyle,
        pub bracket: tok::Bracket,
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
            Meta::Path(x) => Err(err::new2(
                x.segs.first().unwrap().ident.span(),
                x.segs.last().unwrap().ident.span(),
                format!(
                    "expected attribute arguments in parentheses: {}[{}(...)]",
                    parsing::DisplayAttrStyle(&self.style),
                    parsing::DisplayPath(x),
                ),
            )),
            Meta::NameValue(x) => Err(Err::new(
                x.eq.span,
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

ast_enum_of_structs! {
    pub enum Meta {
        Path(Path),
        List(MetaList),
        NameValue(MetaNameValue),
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
            Meta::List(x) => x.delim.span().open(),
            Meta::NameValue(x) => x.eq.span,
        };
        Err(Err::new(y, "unexpected token in attribute"))
    }
    pub fn require_list(&self) -> Result<&MetaList> {
        match self {
            Meta::List(x) => Ok(x),
            Meta::Path(x) => Err(err::new2(
                x.segs.first().unwrap().ident.span(),
                x.segs.last().unwrap().ident.span(),
                format!(
                    "expected attribute arguments in parentheses: `{}(...)`",
                    parsing::DisplayPath(x),
                ),
            )),
            Meta::NameValue(x) => Err(Err::new(x.eq.span, "expected `(`")),
        }
    }
    pub fn require_name_value(&self) -> Result<&MetaNameValue> {
        match self {
            Meta::NameValue(x) => Ok(x),
            Meta::Path(x) => Err(err::new2(
                x.segs.first().unwrap().ident.span(),
                x.segs.last().unwrap().ident.span(),
                format!(
                    "expected a value for this attribute: `{} = ...`",
                    parsing::DisplayPath(x),
                ),
            )),
            Meta::List(x) => Err(Err::new(x.delim.span().open(), "expected `=`")),
        }
    }
}
ast_struct! {
    pub struct MetaList {
        pub path: Path,
        pub delim: MacroDelimiter,
        pub toks: TokenStream,
    }
}
impl MetaList {
    pub fn parse_args<T: Parse>(&self) -> Result<T> {
        self.parse_args_with(T::parse)
    }
    pub fn parse_args_with<T: Parser>(&self, parser: T) -> Result<T::Output> {
        let y = self.delim.span().close();
        parse::parse_scoped(parser, y, self.toks.clone())
    }
    pub fn parse_nested_meta(&self, x: impl FnMut(ParseNestedMeta) -> Result<()>) -> Result<()> {
        self.parse_args_with(meta_parser(x))
    }
}
ast_struct! {
    pub struct MetaNameValue {
        pub path: Path,
        pub eq: Token![=],
        pub val: Expr,
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
    pub fn parse_nested_meta(&self, cb: impl FnMut(ParseNestedMeta) -> Result<()>) -> Result<()> {
        let content;
        parenthesized!(content in self.input);
        parse_nested_meta(&content, cb)
    }
    pub fn error(&self, x: impl Display) -> Err {
        let beg = self.path.segs[0].ident.span();
        let end = self.input.cursor().prev_span();
        err::new2(beg, end, x)
    }
}

pub fn meta_parser(cb: impl FnMut(ParseNestedMeta) -> Result<()>) -> impl Parser<Output = ()> {
    |x: ParseStream| {
        if x.is_empty() {
            Ok(())
        } else {
            parse_nested_meta(x, cb)
        }
    }
}
pub fn parse_nested_meta(x: ParseStream, mut cb: impl FnMut(ParseNestedMeta) -> Result<()>) -> Result<()> {
    loop {
        let path = x.call(parse_meta_path)?;
        cb(ParseNestedMeta { path, input: x })?;
        if x.is_empty() {
            return Ok(());
        }
        x.parse::<Token![,]>()?;
        if x.is_empty() {
            return Ok(());
        }
    }
}
fn parse_meta_path(x: ParseStream) -> Result<Path> {
    Ok(Path {
        colon: x.parse()?,
        segs: {
            let mut ys = Punctuated::new();
            if x.peek(Ident::peek_any) {
                let y = Ident::parse_any(x)?;
                ys.push_value(Segment::from(y));
            } else if x.is_empty() {
                return Err(x.error("expected nested attribute"));
            } else if x.peek(lit::Lit) {
                return Err(x.error("unexpected literal in nested attribute, expected ident"));
            } else {
                return Err(x.error("unexpected token in nested attribute, expected ident"));
            }
            while x.peek(Token![::]) {
                let y = x.parse()?;
                ys.push_punct(y);
                let y = Ident::parse_any(x)?;
                ys.push_value(Segment::from(y));
            }
            ys
        },
    })
}

enum Entry {
    Group(Group, usize),
    Ident(Ident),
    Punct(Punct),
    Literal(Literal),
    End(isize),
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
    pub fn group(mut self, d: Delimiter) -> Option<(Cursor<'a>, DelimSpan, Cursor<'a>)> {
        if d != Delimiter::None {
            self.ignore_none();
        }
        if let Entry::Group(x, end) = self.entry() {
            if x.delimiter() == d {
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
        if buff_start(self) < self.ptr {
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
        if same_buff(*self, *x) {
            Some(self.ptr.cmp(&x.ptr))
        } else {
            None
        }
    }
}

fn same_scope(a: Cursor, b: Cursor) -> bool {
    a.scope == b.scope
}
fn same_buff(a: Cursor, b: Cursor) -> bool {
    buff_start(a) == buff_start(b)
}
fn buff_start(c: Cursor) -> *const Entry {
    unsafe {
        match &*c.scope {
            Entry::End(x) => c.scope.offset(*x),
            _ => unreachable!(),
        }
    }
}
fn cmp_assuming_same_buffer(a: Cursor, b: Cursor) -> Ordering {
    a.ptr.cmp(&b.ptr)
}
fn open_span_of_group(c: Cursor) -> Span {
    match c.entry() {
        Entry::Group(x, _) => x.span_open(),
        _ => c.span(),
    }
}
fn close_span_of_group(c: Cursor) -> Span {
    match c.entry() {
        Entry::Group(x, _) => x.span_close(),
        _ => c.span(),
    }
}

pub struct TokBuff {
    entries: Box<[Entry]>,
}
impl TokBuff {
    fn recursive_new(ys: &mut Vec<Entry>, xs: TokenStream) {
        for x in xs {
            match x {
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
        let y = self.entries.as_ptr();
        unsafe { Cursor::create(y, y.add(self.entries.len() - 1)) }
    }
}

mod expr;
pub use expr::{
    Arm, Expr, ExprArray, ExprAssign, ExprAsync, ExprAwait, ExprBinary, ExprBlock, ExprBreak, ExprCall, ExprCast,
    ExprClosure, ExprConst, ExprConst as PatConst, ExprContinue, ExprField, ExprForLoop, ExprGroup, ExprIf, ExprIndex,
    ExprInfer, ExprLet, ExprLit, ExprLit as PatLit, ExprLoop, ExprMacro, ExprMacro as PatMacro, ExprMatch,
    ExprMethodCall, ExprParen, ExprPath, ExprPath as PatPath, ExprRange, ExprRange as PatRange, ExprReference,
    ExprRepeat, ExprReturn, ExprStruct, ExprTry, ExprTryBlock, ExprTuple, ExprUnary, ExprUnsafe, ExprWhile, ExprYield,
    FieldValue, Index, Label, Member, RangeLimits,
};

ast_struct! {
    pub struct LifetimeParam {
        pub attrs: Vec<Attribute>,
        pub lifetime: Lifetime,
        pub colon: Option<Token![:]>,
        pub bounds: Punctuated<Lifetime, Token![+]>,
    }
}
ast_struct! {
    pub struct TypeParam {
        pub attrs: Vec<Attribute>,
        pub ident: Ident,
        pub colon: Option<Token![:]>,
        pub bounds: Punctuated<TypeParamBound, Token![+]>,
        pub eq: Option<Token![=]>,
        pub default: Option<Ty>,
    }
}
ast_struct! {
    pub struct ConstParam {
        pub attrs: Vec<Attribute>,
        pub const_: Token![const],
        pub ident: Ident,
        pub colon: Token![:],
        pub ty: Ty,
        pub eq: Option<Token![=]>,
        pub default: Option<Expr>,
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
    pub struct Generics {
        pub lt: Option<Token![<]>,
        pub params: Punctuated<GenericParam, Token![,]>,
        pub gt: Option<Token![>]>,
        pub clause: Option<WhereClause>,
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
        self.clause.get_or_insert_with(|| WhereClause {
            where_: <Token![where]>::default(),
            preds: Punctuated::new(),
        })
    }
    pub fn split_for_impl(&self) -> (ImplGenerics, TypeGenerics, Option<&WhereClause>) {
        (ImplGenerics(self), TypeGenerics(self), self.clause.as_ref())
    }
}
impl Default for Generics {
    fn default() -> Self {
        Generics {
            lt: None,
            params: Punctuated::new(),
            gt: None,
            clause: None,
        }
    }
}

pub struct Lifetimes<'a>(Iter<'a, GenericParam>);
impl<'a> Iterator for Lifetimes<'a> {
    type Item = &'a LifetimeParam;
    fn next(&mut self) -> Option<Self::Item> {
        let next = match self.0.next() {
            Some(x) => x,
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
            Some(x) => x,
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
            Some(x) => x,
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
            Some(x) => x,
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
            Some(x) => x,
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
            Some(x) => x,
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

macro_rules! generics_wrapper_impls {
    ($ty:ident) => {
        impl<'a> Clone for $ty<'a> {
            fn clone(&self) -> Self {
                $ty(self.0)
            }
        }
        impl<'a> Debug for $ty<'a> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.debug_tuple(stringify!($ty)).field(self.0).finish()
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
        pub for_: Token![for],
        pub lt: Token![<],
        pub lifetimes: Punctuated<GenericParam, Token![,]>,
        pub gt: Token![>],
    }
}
impl Default for BoundLifetimes {
    fn default() -> Self {
        BoundLifetimes {
            for_: Default::default(),
            lt: Default::default(),
            lifetimes: Punctuated::new(),
            gt: Default::default(),
        }
    }
}
impl LifetimeParam {
    pub fn new(lifetime: Lifetime) -> Self {
        LifetimeParam {
            attrs: Vec::new(),
            lifetime,
            colon: None,
            bounds: Punctuated::new(),
        }
    }
}
impl From<Ident> for TypeParam {
    fn from(ident: Ident) -> Self {
        TypeParam {
            attrs: vec![],
            ident,
            colon: None,
            bounds: Punctuated::new(),
            eq: None,
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
        pub paren: Option<tok::Paren>,
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
        pub where_: Token![where],
        pub preds: Punctuated<WherePred, Token![,]>,
    }
}
ast_enum_of_structs! {
    pub enum WherePred {
        Lifetime(PredLifetime),
        Type(PredType),
    }
}
ast_struct! {
    pub struct PredLifetime {
        pub lifetime: Lifetime,
        pub colon: Token![:],
        pub bounds: Punctuated<Lifetime, Token![+]>,
    }
}
ast_struct! {
    pub struct PredType {
        pub lifetimes: Option<BoundLifetimes>,
        pub bounded_ty: Ty,
        pub colon: Token![:],
        pub bounds: Punctuated<TypeParamBound, Token![+]>,
    }
}

mod item;
pub use item::{
    FnArg, ForeignItem, ForeignItemFn, ForeignItemMacro, ForeignItemStatic, ForeignItemType, ImplItem, ImplItemConst,
    ImplItemFn, ImplItemMacro, ImplItemType, ImplRestriction, Item, ItemConst, ItemEnum, ItemExternCrate, ItemFn,
    ItemForeignMod, ItemImpl, ItemMacro, ItemMod, ItemStatic, ItemStruct, ItemTrait, ItemTraitAlias, ItemType,
    ItemUnion, ItemUse, Receiver, Signature, StaticMutability, TraitItem, TraitItemConst, TraitItemFn, TraitItemMacro,
    TraitItemType, UseGlob, UseGroup, UseName, UsePath, UseRename, UseTree, Variadic,
};
pub mod punct;
use punct::Punctuated;
pub mod lit;
pub mod tok;

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
        pub ref_: Option<Token![ref]>,
        pub mut_: Option<Token![mut]>,
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
        pub paren: tok::Paren,
        pub pat: Box<Pat>,
    }
}
ast_struct! {
    pub struct PatReference {
        pub attrs: Vec<Attribute>,
        pub and_: Token![&],
        pub mutability: Option<Token![mut]>,
        pub pat: Box<Pat>,
    }
}
ast_struct! {
    pub struct PatRest {
        pub attrs: Vec<Attribute>,
        pub dot2: Token![..],
    }
}
ast_struct! {
    pub struct PatSlice {
        pub attrs: Vec<Attribute>,
        pub bracket: tok::Bracket,
        pub elems: Punctuated<Pat, Token![,]>,
    }
}
ast_struct! {
    pub struct PatStruct {
        pub attrs: Vec<Attribute>,
        pub qself: Option<QSelf>,
        pub path: Path,
        pub brace: tok::Brace,
        pub fields: Punctuated<FieldPat, Token![,]>,
        pub rest: Option<PatRest>,
    }
}
ast_struct! {
    pub struct PatTuple {
        pub attrs: Vec<Attribute>,
        pub paren: tok::Paren,
        pub elems: Punctuated<Pat, Token![,]>,
    }
}
ast_struct! {
    pub struct PatTupleStruct {
        pub attrs: Vec<Attribute>,
        pub qself: Option<QSelf>,
        pub path: Path,
        pub paren: tok::Paren,
        pub elems: Punctuated<Pat, Token![,]>,
    }
}
ast_struct! {
    pub struct PatType {
        pub attrs: Vec<Attribute>,
        pub pat: Box<Pat>,
        pub colon: Token![:],
        pub ty: Box<Ty>,
    }
}
ast_struct! {
    pub struct PatWild {
        pub attrs: Vec<Attribute>,
        pub underscore: Token![_],
    }
}
ast_struct! {
    pub struct FieldPat {
        pub attrs: Vec<Attribute>,
        pub member: Member,
        pub colon: Option<Token![:]>,
        pub pat: Box<Pat>,
    }
}

pub mod path {
    use super::{punct::Punctuated, *};
    ast_struct! {
        pub struct Path {
            pub colon: Option<Token![::]>,
            pub segs: Punctuated<Segment, Token![::]>,
        }
    }
    impl Path {
        pub fn is_ident<I: ?Sized>(&self, i: &I) -> bool
        where
            Ident: PartialEq<I>,
        {
            match self.get_ident() {
                Some(x) => x == i,
                None => false,
            }
        }
        pub fn get_ident(&self) -> Option<&Ident> {
            if self.colon.is_none() && self.segs.len() == 1 && self.segs[0].args.is_none() {
                Some(&self.segs[0].ident)
            } else {
                None
            }
        }
    }
    impl<T> From<T> for Path
    where
        T: Into<Segment>,
    {
        fn from(x: T) -> Self {
            let mut y = Path {
                colon: None,
                segs: Punctuated::new(),
            };
            y.segs.push_value(x.into());
            y
        }
    }

    ast_struct! {
        pub struct Segment {
            pub ident: Ident,
            pub args: Args,
        }
    }
    impl<T> From<T> for Segment
    where
        T: Into<Ident>,
    {
        fn from(x: T) -> Self {
            Segment {
                ident: x.into(),
                args: Args::None,
            }
        }
    }

    ast_enum! {
        pub enum Args {
            None,
            Angled(AngledArgs),
            Parenthesized(ParenthesizedArgs),
        }
    }
    impl Args {
        pub fn is_empty(&self) -> bool {
            use Args::*;
            match self {
                None => true,
                Angled(x) => x.args.is_empty(),
                Parenthesized(_) => false,
            }
        }
        pub fn is_none(&self) -> bool {
            use Args::*;
            match self {
                None => true,
                Angled(_) | Parenthesized(_) => false,
            }
        }
    }
    impl Default for Args {
        fn default() -> Self {
            Args::None
        }
    }

    ast_enum! {
        pub enum Arg {
            Lifetime(Lifetime),
            Type(Ty),
            Const(Expr),
            AssocType(AssocType),
            AssocConst(AssocConst),
            Constraint(Constraint),
        }
    }
    ast_struct! {
        pub struct AngledArgs {
            pub colon2: Option<Token![::]>,
            pub lt: Token![<],
            pub args: Punctuated<Arg, Token![,]>,
            pub gt: Token![>],
        }
    }
    ast_struct! {
        pub struct AssocType {
            pub ident: Ident,
            pub gnrs: Option<AngledArgs>,
            pub eq: Token![=],
            pub ty: Ty,
        }
    }
    ast_struct! {
        pub struct AssocConst {
            pub ident: Ident,
            pub gnrs: Option<AngledArgs>,
            pub eq: Token![=],
            pub val: Expr,
        }
    }
    ast_struct! {
        pub struct Constraint {
            pub ident: Ident,
            pub gnrs: Option<AngledArgs>,
            pub colon: Token![:],
            pub bounds: Punctuated<TypeParamBound, Token![+]>,
        }
    }
    ast_struct! {
        pub struct ParenthesizedArgs {
            pub paren: tok::Paren,
            pub args: Punctuated<Ty, Token![,]>,
            pub out: ReturnType,
        }
    }
    ast_struct! {
        pub struct QSelf {
            pub lt: Token![<],
            pub ty: Box<Ty>,
            pub pos: usize,
            pub as_: Option<Token![as]>,
            pub gt: Token![>],
        }
    }
}

ast_struct! {
    pub struct Block {
        pub brace: tok::Brace,
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
        pub let_: Token![let],
        pub pat: Pat,
        pub init: Option<LocalInit>,
        pub semi: Token![;],
    }
}
ast_struct! {
    pub struct LocalInit {
        pub eq: Token![=],
        pub expr: Box<Expr>,
        pub diverge: Option<(Token![else], Box<Expr>)>,
    }
}
ast_struct! {
    pub struct StmtMacro {
        pub attrs: Vec<Attribute>,
        pub mac: Macro,
        pub semi: Option<Token![;]>,
    }
}

mod ty {
    use super::{punct::Punctuated, *};
    use proc_macro2::TokenStream;

    ast_enum_of_structs! {
        pub enum Ty {
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
            pub bracket: tok::Bracket,
            pub elem: Box<Ty>,
            pub semi: Token![;],
            pub len: Expr,
        }
    }
    ast_struct! {
        pub struct TypeBareFn {
            pub lifetimes: Option<BoundLifetimes>,
            pub unsafe_: Option<Token![unsafe]>,
            pub abi: Option<Abi>,
            pub fn_: Token![fn],
            pub paren: tok::Paren,
            pub inputs: Punctuated<BareFnArg, Token![,]>,
            pub variadic: Option<BareVariadic>,
            pub output: ReturnType,
        }
    }
    ast_struct! {
        pub struct TypeGroup {
            pub group: tok::Group,
            pub elem: Box<Ty>,
        }
    }
    ast_struct! {
        pub struct TypeImplTrait {
            pub impl_: Token![impl],
            pub bounds: Punctuated<TypeParamBound, Token![+]>,
        }
    }
    ast_struct! {
        pub struct TypeInfer {
            pub underscore: Token![_],
        }
    }
    ast_struct! {
        pub struct TypeMacro {
            pub mac: Macro,
        }
    }
    ast_struct! {
        pub struct TypeNever {
            pub bang: Token![!],
        }
    }
    ast_struct! {
        pub struct TypeParen {
            pub paren_token: tok::Paren,
            pub elem: Box<Ty>,
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
            pub star: Token![*],
            pub const_: Option<Token![const]>,
            pub mut_: Option<Token![mut]>,
            pub elem: Box<Ty>,
        }
    }
    ast_struct! {
        pub struct TypeReference {
            pub and_: Token![&],
            pub lifetime: Option<Lifetime>,
            pub mut_: Option<Token![mut]>,
            pub elem: Box<Ty>,
        }
    }
    ast_struct! {
        pub struct TypeSlice {
            pub bracket: tok::Bracket,
            pub elem: Box<Ty>,
        }
    }
    ast_struct! {
        pub struct TypeTraitObject {
            pub dyn_: Option<Token![dyn]>,
            pub bounds: Punctuated<TypeParamBound, Token![+]>,
        }
    }
    ast_struct! {
        pub struct TypeTuple {
            pub paren: tok::Paren,
            pub elems: Punctuated<Ty, Token![,]>,
        }
    }
    ast_struct! {
        pub struct Abi {
            pub extern_: Token![extern],
            pub name: Option<lit::Str>,
        }
    }
    ast_struct! {
        pub struct BareFnArg {
            pub attrs: Vec<Attribute>,
            pub name: Option<(Ident, Token![:])>,
            pub ty: Ty,
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
            Type(Token![->], Box<Ty>),
        }
    }
}
pub use ty::{
    Abi, BareFnArg, BareVariadic, ReturnType, Ty, TypeArray, TypeBareFn, TypeGroup, TypeImplTrait, TypeInfer,
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
    pub fn iter(&self) -> punct::Iter<Field> {
        match self {
            Fields::Unit => punct::empty_punctuated_iter(),
            Fields::Named(f) => f.named.iter(),
            Fields::Unnamed(f) => f.unnamed.iter(),
        }
    }
    pub fn iter_mut(&mut self) -> punct::IterMut<Field> {
        match self {
            Fields::Unit => punct::empty_punctuated_iter_mut(),
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
    type IntoIter = punct::IntoIter<Field>;
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
ast_struct! {
    pub struct Field {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub mutability: FieldMut,
        pub ident: Option<Ident>,
        pub colon_token: Option<Token![:]>,
        pub ty: Ty,
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
        pub struct_: Token![struct],
        pub fields: Fields,
        pub semi: Option<Token![;]>,
    }
}
ast_struct! {
    pub struct DataEnum {
        pub enum_: Token![enum],
        pub brace: tok::Brace,
        pub variants: Punctuated<Variant, Token![,]>,
    }
}
ast_struct! {
    pub struct DataUnion {
        pub union_: Token![union],
        pub fields: FieldsNamed,
    }
}

mod err {
    use super::{Cursor, ThreadBound};
    use proc_macro2::{Delimiter, Group, Ident, LexError, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
    use quote::ToTokens;
    use std::{
        fmt::{self, Debug, Display},
        slice, vec,
    };
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
            let span = super::open_span_of_group(cursor);
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
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            if self.messages.len() == 1 {
                f.debug_tuple("Error").field(&self.messages[0]).finish()
            } else {
                f.debug_tuple("Error").field(&self.messages).finish()
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
            f.write_str(&self.messages[0].message)
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
pub use err::{Err, Result};

pub mod ext {
    use super::{
        parse::{ParseStream, Peek, Result},
        sealed::lookahead,
        tok::CustomTok,
        Cursor,
    };
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
    impl CustomTok for private::IdentAny {
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
    use super::lookahead;
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
pub use ident::Ident;

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
        if !ident::xid_ok(&symbol[1..]) {
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "'".fmt(f)?;
        self.ident.fmt(f)
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
    use super::{
        err::{self, Err},
        sealed::lookahead::Sealed,
        tok::Tok,
        Cursor, IntoSpans,
    };
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

    pub fn is_delimiter(x: Cursor, d: Delimiter) -> bool {
        x.group(d).is_some()
    }

    impl<F: Copy + FnOnce(TokenMarker) -> T, T: Tok> Sealed for F {}
}

ast_struct! {
    pub struct Macro {
        pub path: Path,
        pub bang: Token![!],
        pub delim: MacroDelimiter,
        pub toks: TokenStream,
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
        let scope = self.delim.span().close();
        parse::parse_scoped(parser, scope, self.toks.clone())
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

pub mod parse {
    use super::{
        err::{self, Err, Result},
        lookahead::{self, Lookahead1, Peek},
        proc_macro,
        punct::Punctuated,
        tok::Tok,
        Cursor, TokBuff,
    };
    use proc_macro2::{self, Delimiter, Group, Literal, Punct, Span, TokenStream, TokenTree};
    use std::{
        cell::Cell,
        fmt::{self, Debug, Display},
        hash::{Hash, Hasher},
        marker::PhantomData,
        mem,
        ops::Deref,
        rc::Rc,
        str::FromStr,
    };

    pub mod discouraged {
        use super::*;
        use proc_macro2::extra::DelimSpan;
        pub trait Speculative {
            fn advance_to(&self, fork: &Self);
        }
        impl<'a> Speculative for ParseBuffer<'a> {
            fn advance_to(&self, fork: &Self) {
                if !crate::same_scope(self.cursor(), fork.cursor()) {
                    panic!("Fork was not derived from the advancing parse stream");
                }
                let (self_unexp, self_sp) = inner_unexpected(self);
                let (fork_unexp, fork_sp) = inner_unexpected(fork);
                if !Rc::ptr_eq(&self_unexp, &fork_unexp) {
                    match (fork_sp, self_sp) {
                        (Some(span), None) => {
                            self_unexp.set(Unexpected::Some(span));
                        },
                        (None, None) => {
                            fork_unexp.set(Unexpected::Chain(self_unexp));
                            fork.unexpected.set(Some(Rc::new(Cell::new(Unexpected::None))));
                        },
                        (_, Some(_)) => {},
                    }
                }
                self.cell
                    .set(unsafe { mem::transmute::<Cursor, Cursor<'static>>(fork.cursor()) });
            }
        }
        pub trait AnyDelimiter {
            fn parse_any_delimiter(&self) -> Result<(Delimiter, DelimSpan, ParseBuffer)>;
        }
        impl<'a> AnyDelimiter for ParseBuffer<'a> {
            fn parse_any_delimiter(&self) -> Result<(Delimiter, DelimSpan, ParseBuffer)> {
                self.step(|cursor| {
                    if let Some((content, delimiter, span, rest)) = cursor.any_group() {
                        let scope = crate::close_span_of_group(*cursor);
                        let nested = crate::parse::advance_step_cursor(cursor, content);
                        let unexpected = crate::parse::get_unexpected(self);
                        let content = crate::parse::new_parse_buffer(scope, nested, unexpected);
                        Ok(((delimiter, span, content), rest))
                    } else {
                        Err(cursor.error("expected any delimiter"))
                    }
                })
            }
        }
    }

    pub trait Parse: Sized {
        fn parse(input: ParseStream) -> Result<Self>;
    }
    pub type ParseStream<'a> = &'a ParseBuffer<'a>;
    pub struct ParseBuffer<'a> {
        scope: Span,
        cell: Cell<Cursor<'static>>,
        marker: PhantomData<Cursor<'a>>,
        unexpected: Cell<Option<Rc<Cell<Unexpected>>>>,
    }
    impl<'a> Drop for ParseBuffer<'a> {
        fn drop(&mut self) {
            if let Some(unexpected_span) = span_of_unexpected_ignoring_nones(self.cursor()) {
                let (inner, old_span) = inner_unexpected(self);
                if old_span.is_none() {
                    inner.set(Unexpected::Some(unexpected_span));
                }
            }
        }
    }
    impl<'a> Display for ParseBuffer<'a> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            Display::fmt(&self.cursor().token_stream(), f)
        }
    }
    impl<'a> Debug for ParseBuffer<'a> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            Debug::fmt(&self.cursor().token_stream(), f)
        }
    }
    pub struct StepCursor<'c, 'a> {
        scope: Span,
        cursor: Cursor<'c>,
        marker: PhantomData<fn(Cursor<'c>) -> Cursor<'a>>,
    }
    impl<'c, 'a> Deref for StepCursor<'c, 'a> {
        type Target = Cursor<'c>;
        fn deref(&self) -> &Self::Target {
            &self.cursor
        }
    }
    impl<'c, 'a> Copy for StepCursor<'c, 'a> {}
    impl<'c, 'a> Clone for StepCursor<'c, 'a> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<'c, 'a> StepCursor<'c, 'a> {
        pub fn error<T: Display>(self, message: T) -> Err {
            err::new_at(self.scope, self.cursor, message)
        }
    }
    pub(crate) fn advance_step_cursor<'c, 'a>(proof: StepCursor<'c, 'a>, to: Cursor<'c>) -> Cursor<'a> {
        let _ = proof;
        unsafe { mem::transmute::<Cursor<'c>, Cursor<'a>>(to) }
    }
    pub(crate) fn new_parse_buffer(scope: Span, cursor: Cursor, unexpected: Rc<Cell<Unexpected>>) -> ParseBuffer {
        ParseBuffer {
            scope,
            cell: Cell::new(unsafe { mem::transmute::<Cursor, Cursor<'static>>(cursor) }),
            marker: PhantomData,
            unexpected: Cell::new(Some(unexpected)),
        }
    }
    pub(crate) enum Unexpected {
        None,
        Some(Span),
        Chain(Rc<Cell<Unexpected>>),
    }
    impl Default for Unexpected {
        fn default() -> Self {
            Unexpected::None
        }
    }
    impl Clone for Unexpected {
        fn clone(&self) -> Self {
            match self {
                Unexpected::None => Unexpected::None,
                Unexpected::Some(span) => Unexpected::Some(*span),
                Unexpected::Chain(next) => Unexpected::Chain(next.clone()),
            }
        }
    }
    fn cell_clone<T: Default + Clone>(cell: &Cell<T>) -> T {
        let prev = cell.take();
        let ret = prev.clone();
        cell.set(prev);
        ret
    }
    fn inner_unexpected(buffer: &ParseBuffer) -> (Rc<Cell<Unexpected>>, Option<Span>) {
        let mut unexpected = get_unexpected(buffer);
        loop {
            match cell_clone(&unexpected) {
                Unexpected::None => return (unexpected, None),
                Unexpected::Some(span) => return (unexpected, Some(span)),
                Unexpected::Chain(next) => unexpected = next,
            }
        }
    }
    pub(crate) fn get_unexpected(buffer: &ParseBuffer) -> Rc<Cell<Unexpected>> {
        cell_clone(&buffer.unexpected).unwrap()
    }
    fn span_of_unexpected_ignoring_nones(mut cursor: Cursor) -> Option<Span> {
        if cursor.eof() {
            return None;
        }
        while let Some((inner, _span, rest)) = cursor.group(Delimiter::None) {
            if let Some(unexpected) = span_of_unexpected_ignoring_nones(inner) {
                return Some(unexpected);
            }
            cursor = rest;
        }
        if cursor.eof() {
            None
        } else {
            Some(cursor.span())
        }
    }
    impl<'a> ParseBuffer<'a> {
        pub fn parse<T: Parse>(&self) -> Result<T> {
            T::parse(self)
        }
        pub fn call<T>(&self, function: fn(ParseStream) -> Result<T>) -> Result<T> {
            function(self)
        }
        pub fn peek<T: Peek>(&self, token: T) -> bool {
            let _ = token;
            T::Token::peek(self.cursor())
        }
        pub fn peek2<T: Peek>(&self, token: T) -> bool {
            fn peek2(buffer: &ParseBuffer, peek: fn(Cursor) -> bool) -> bool {
                if let Some(group) = buffer.cursor().group(Delimiter::None) {
                    if group.0.skip().map_or(false, peek) {
                        return true;
                    }
                }
                buffer.cursor().skip().map_or(false, peek)
            }
            let _ = token;
            peek2(self, T::Token::peek)
        }
        pub fn peek3<T: Peek>(&self, token: T) -> bool {
            fn peek3(buffer: &ParseBuffer, peek: fn(Cursor) -> bool) -> bool {
                if let Some(group) = buffer.cursor().group(Delimiter::None) {
                    if group.0.skip().and_then(Cursor::skip).map_or(false, peek) {
                        return true;
                    }
                }
                buffer.cursor().skip().and_then(Cursor::skip).map_or(false, peek)
            }
            let _ = token;
            peek3(self, T::Token::peek)
        }
        pub fn parse_terminated<T, P>(
            &self,
            parser: fn(ParseStream) -> Result<T>,
            separator: P,
        ) -> Result<Punctuated<T, P::Token>>
        where
            P: Peek,
            P::Token: Parse,
        {
            let _ = separator;
            Punctuated::parse_terminated_with(self, parser)
        }
        pub fn is_empty(&self) -> bool {
            self.cursor().eof()
        }
        pub fn lookahead1(&self) -> Lookahead1<'a> {
            lookahead::new(self.scope, self.cursor())
        }
        pub fn fork(&self) -> Self {
            ParseBuffer {
                scope: self.scope,
                cell: self.cell.clone(),
                marker: PhantomData,
                unexpected: Cell::new(Some(Rc::new(Cell::new(Unexpected::None)))),
            }
        }
        pub fn error<T: Display>(&self, message: T) -> Err {
            err::new_at(self.scope, self.cursor(), message)
        }
        pub fn step<F, R>(&self, function: F) -> Result<R>
        where
            F: for<'c> FnOnce(StepCursor<'c, 'a>) -> Result<(R, Cursor<'c>)>,
        {
            let (node, rest) = function(StepCursor {
                scope: self.scope,
                cursor: self.cell.get(),
                marker: PhantomData,
            })?;
            self.cell.set(rest);
            Ok(node)
        }
        pub fn span(&self) -> Span {
            let cursor = self.cursor();
            if cursor.eof() {
                self.scope
            } else {
                super::open_span_of_group(cursor)
            }
        }
        pub fn cursor(&self) -> Cursor<'a> {
            self.cell.get()
        }
        fn check_unexpected(&self) -> Result<()> {
            match inner_unexpected(self).1 {
                Some(span) => Err(Err::new(span, "unexpected token")),
                None => Ok(()),
            }
        }
    }

    impl<T: Parse> Parse for Box<T> {
        fn parse(input: ParseStream) -> Result<Self> {
            input.parse().map(Box::new)
        }
    }

    impl<T: Parse + Tok> Parse for Option<T> {
        fn parse(input: ParseStream) -> Result<Self> {
            if T::peek(input.cursor()) {
                Ok(Some(input.parse()?))
            } else {
                Ok(None)
            }
        }
    }

    impl Parse for TokenStream {
        fn parse(input: ParseStream) -> Result<Self> {
            input.step(|cursor| Ok((cursor.token_stream(), Cursor::empty())))
        }
    }

    impl Parse for TokenTree {
        fn parse(input: ParseStream) -> Result<Self> {
            input.step(|cursor| match cursor.token_tree() {
                Some((tt, rest)) => Ok((tt, rest)),
                None => Err(cursor.error("expected token tree")),
            })
        }
    }

    impl Parse for Group {
        fn parse(input: ParseStream) -> Result<Self> {
            input.step(|cursor| {
                if let Some((group, rest)) = cursor.any_group_token() {
                    if group.delimiter() != Delimiter::None {
                        return Ok((group, rest));
                    }
                }
                Err(cursor.error("expected group token"))
            })
        }
    }

    impl Parse for Punct {
        fn parse(input: ParseStream) -> Result<Self> {
            input.step(|cursor| match cursor.punct() {
                Some((punct, rest)) => Ok((punct, rest)),
                None => Err(cursor.error("expected punctuation token")),
            })
        }
    }

    impl Parse for Literal {
        fn parse(input: ParseStream) -> Result<Self> {
            input.step(|cursor| match cursor.literal() {
                Some((literal, rest)) => Ok((literal, rest)),
                None => Err(cursor.error("expected literal token")),
            })
        }
    }
    pub trait Parser: Sized {
        type Output;
        fn parse2(self, tokens: TokenStream) -> Result<Self::Output>;
        fn parse(self, tokens: proc_macro::TokenStream) -> Result<Self::Output> {
            self.parse2(proc_macro2::TokenStream::from(tokens))
        }
        fn parse_str(self, s: &str) -> Result<Self::Output> {
            self.parse2(proc_macro2::TokenStream::from_str(s)?)
        }
        fn __parse_scoped(self, scope: Span, tokens: TokenStream) -> Result<Self::Output> {
            let _ = scope;
            self.parse2(tokens)
        }
    }
    fn tokens_to_parse_buffer(tokens: &TokBuff) -> ParseBuffer {
        let scope = Span::call_site();
        let cursor = tokens.begin();
        let unexpected = Rc::new(Cell::new(Unexpected::None));
        new_parse_buffer(scope, cursor, unexpected)
    }
    impl<F, T> Parser for F
    where
        F: FnOnce(ParseStream) -> Result<T>,
    {
        type Output = T;
        fn parse2(self, tokens: TokenStream) -> Result<T> {
            let buf = TokBuff::new2(tokens);
            let state = tokens_to_parse_buffer(&buf);
            let node = self(&state)?;
            state.check_unexpected()?;
            if let Some(unexpected_span) = span_of_unexpected_ignoring_nones(state.cursor()) {
                Err(Err::new(unexpected_span, "unexpected token"))
            } else {
                Ok(node)
            }
        }
        fn __parse_scoped(self, scope: Span, tokens: TokenStream) -> Result<Self::Output> {
            let buf = TokBuff::new2(tokens);
            let cursor = buf.begin();
            let unexpected = Rc::new(Cell::new(Unexpected::None));
            let state = new_parse_buffer(scope, cursor, unexpected);
            let node = self(&state)?;
            state.check_unexpected()?;
            if let Some(unexpected_span) = span_of_unexpected_ignoring_nones(state.cursor()) {
                Err(Err::new(unexpected_span, "unexpected token"))
            } else {
                Ok(node)
            }
        }
    }

    pub(crate) fn parse_scoped<F: Parser>(f: F, scope: Span, tokens: TokenStream) -> Result<F::Output> {
        f.__parse_scoped(scope, tokens)
    }
    pub struct Nothing;
    impl Parse for Nothing {
        fn parse(_input: ParseStream) -> Result<Self> {
            Ok(Nothing)
        }
    }
    impl Debug for Nothing {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("Nothing")
        }
    }
    impl Eq for Nothing {}
    impl PartialEq for Nothing {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl Hash for Nothing {
        fn hash<H: Hasher>(&self, _state: &mut H) {}
    }
}

pub trait ParseQuote: Sized {
    fn parse(x: ParseStream) -> Result<Self>;
}
impl<T: Parse> ParseQuote for T {
    fn parse(x: ParseStream) -> Result<Self> {
        <T as Parse>::parse(x)
    }
}
impl ParseQuote for Attribute {
    fn parse(x: ParseStream) -> Result<Self> {
        if x.peek(Token![#]) && x.peek2(Token![!]) {
            parsing::single_parse_inner(x)
        } else {
            parsing::single_parse_outer(x)
        }
    }
}
impl ParseQuote for Pat {
    fn parse(x: ParseStream) -> Result<Self> {
        Pat::parse_multi_with_leading_vert(x)
    }
}
impl ParseQuote for Box<Pat> {
    fn parse(x: ParseStream) -> Result<Self> {
        <Pat as ParseQuote>::parse(x).map(Box::new)
    }
}
impl<T: Parse, P: Parse> ParseQuote for Punctuated<T, P> {
    fn parse(x: ParseStream) -> Result<Self> {
        Self::parse_terminated(x)
    }
}
impl ParseQuote for Vec<Stmt> {
    fn parse(x: ParseStream) -> Result<Self> {
        Block::parse_within(x)
    }
}

pub fn parse_quote_fn<T: ParseQuote>(x: TokenStream) -> T {
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
    fn to_tokens(&self, ys: &mut TokenStream) {
        match self.0 {
            Some(x) => x.to_tokens(ys),
            None => T::default().to_tokens(ys),
        }
    }
}

ast_enum! {
    pub enum Visibility {
        Public(Token![pub]),
        Restricted(VisRestricted),
        Inherited,
    }
}
ast_struct! {
    pub struct VisRestricted {
        pub pub_: Token![pub],
        pub paren: tok::Paren,
        pub in_: Option<Token![in]>,
        pub path: Box<Path>,
    }
}
ast_enum! {
    pub enum FieldMut {
        None,
    }
}

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
        let mut y = Group::new(Delimiter::None, TokenStream::new());
        y.set_span(self);
        y.delim_span()
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
                        Some(x) => x,
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
    assert!(same_buffer(end, cursor));
    let mut tokens = TokenStream::new();
    while cursor != end {
        let (tt, next) = cursor.token_tree().unwrap();
        if cmp_assuming_same_buffer(end, next) == Ordering::Less {
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
pub use gen::*;
pub mod __private {
    pub use super::{
        dump::punct as print_punct,
        parse_quote_fn,
        parsing::{peek_punct, punct as parse_punct},
    };
    pub use proc_macro::TokenStream;
    pub use proc_macro2::{Span, TokenStream as TokenStream2};
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

pub mod dump;
pub mod parsing;
