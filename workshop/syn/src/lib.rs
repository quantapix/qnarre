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

use proc_macro2::{extra::DelimSpan, Delimiter, Group, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
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

pub mod meta {
    use path::Path;
    ast_enum_of_structs! {
        pub enum Meta {
            List(List),
            NameValue(NameValue),
            Path(Path),
        }
    }
    impl Meta {
        pub fn path(&self) -> &Path {
            use Meta::*;
            match self {
                Path(x) => x,
                List(x) => &x.path,
                NameValue(x) => &x.path,
            }
        }
        pub fn require_path_only(&self) -> Result<&Path> {
            use Meta::*;
            let y = match self {
                Path(x) => return Ok(x),
                List(x) => x.delim.span().open(),
                NameValue(x) => x.eq.span,
            };
            Err(Err::new(y, "unexpected token in attribute"))
        }
        pub fn require_list(&self) -> Result<&List> {
            use Meta::*;
            match self {
                List(x) => Ok(x),
                Path(x) => Err(err::new2(
                    x.segs.first().unwrap().ident.span(),
                    x.segs.last().unwrap().ident.span(),
                    format!(
                        "expected attribute arguments in parentheses: `{}(...)`",
                        parsing::DisplayPath(x),
                    ),
                )),
                NameValue(x) => Err(Err::new(x.eq.span, "expected `(`")),
            }
        }
        pub fn require_name_value(&self) -> Result<&NameValue> {
            use Meta::*;
            match self {
                NameValue(x) => Ok(x),
                Path(x) => Err(err::new2(
                    x.segs.first().unwrap().ident.span(),
                    x.segs.last().unwrap().ident.span(),
                    format!(
                        "expected a value for this attribute: `{} = ...`",
                        parsing::DisplayPath(x),
                    ),
                )),
                List(x) => Err(Err::new(x.delim.span().open(), "expected `=`")),
            }
        }
    }
    pub struct List {
        pub path: Path,
        pub delim: tok::Delim,
        pub toks: TokenStream,
    }
    impl List {
        pub fn parse_args<T: Parse>(&self) -> Result<T> {
            self.parse_args_with(T::parse)
        }
        pub fn parse_args_with<T: Parser>(&self, x: T) -> Result<T::Output> {
            let y = self.delim.span().close();
            parse::parse_scoped(x, y, self.toks.clone())
        }
        pub fn parse_nested(&self, x: impl FnMut(ParseNested) -> Result<()>) -> Result<()> {
            self.parse_args_with(parser(x))
        }
    }

    pub struct NameValue {
        pub path: Path,
        pub eq: Token![=],
        pub expr: Expr,
    }

    pub struct ParseNested<'a> {
        pub path: Path,
        pub ins: ParseStream<'a>,
    }
    impl<'a> ParseNested<'a> {
        pub fn val(&self) -> Result<ParseStream<'a>> {
            self.ins.parse::<Token![=]>()?;
            Ok(self.ins)
        }
        pub fn parse(&self, cb: impl FnMut(ParseNested) -> Result<()>) -> Result<()> {
            let y;
            parenthesized!(y in self.ins);
            parse_nested(&y, cb)
        }
        pub fn err(&self, x: impl Display) -> Err {
            let beg = self.path.segs[0].ident.span();
            let end = self.ins.cursor().prev_span();
            err::new2(beg, end, x)
        }
    }

    pub fn parser(cb: impl FnMut(ParseNested) -> Result<()>) -> impl Parser<Output = ()> {
        |x: ParseStream| {
            if x.is_empty() {
                Ok(())
            } else {
                parse_nested(x, cb)
            }
        }
    }
    pub fn parse_nested(ins: ParseStream, mut cb: impl FnMut(ParseNested) -> Result<()>) -> Result<()> {
        loop {
            let path = ins.call(parse_path)?;
            cb(ParseNested { path, ins })?;
            if ins.is_empty() {
                return Ok(());
            }
            ins.parse::<Token![,]>()?;
            if ins.is_empty() {
                return Ok(());
            }
        }
    }
    fn parse_path(x: ParseStream) -> Result<Path> {
        Ok(Path {
            colon: x.parse()?,
            segs: {
                let mut ys = Punctuated::new();
                if x.peek(Ident::peek_any) {
                    let y = Ident::parse_any(x)?;
                    ys.push_value(path::Segment::from(y));
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
                    ys.push_value(path::Segment::from(y));
                }
                ys
            },
        })
    }
}
pub mod attr {
    pub enum Style {
        Outer,
        Inner(Token![!]),
    }
    pub struct Attr {
        pub pound: Token![#],
        pub style: Style,
        pub bracket: tok::Bracket,
        pub meta: meta::Meta,
    }
    impl Attr {
        pub fn path(&self) -> &Path {
            self.meta.path()
        }
        pub fn parse_args<T: Parse>(&self) -> Result<T> {
            self.parse_args_with(T::parse)
        }
        pub fn parse_args_with<T: Parser>(&self, parser: T) -> Result<T::Output> {
            use meta::Meta::*;
            match &self.meta {
                Path(x) => Err(err::new2(
                    x.segs.first().unwrap().ident.span(),
                    x.segs.last().unwrap().ident.span(),
                    format!(
                        "expected attribute arguments in parentheses: {}[{}(...)]",
                        parsing::DisplayAttrStyle(&self.style),
                        parsing::DisplayPath(x),
                    ),
                )),
                NameValue(x) => Err(Err::new(
                    x.eq.span,
                    format_args!(
                        "expected parentheses: {}[{}(...)]",
                        parsing::DisplayAttrStyle(&self.style),
                        parsing::DisplayPath(&meta.path),
                    ),
                )),
                List(x) => x.parse_args_with(parser),
            }
        }
        pub fn parse_nested_meta(&self, x: impl FnMut(meta::ParseNested) -> Result<()>) -> Result<()> {
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
    pub trait Filter<'a> {
        type Ret: Iterator<Item = &'a Attr>;
        fn outer(self) -> Self::Ret;
        fn inner(self) -> Self::Ret;
    }
    impl<'a> Filter<'a> for &'a [Attr] {
        type Ret = iter::Filter<slice::Iter<'a, Attr>, fn(&&Attr) -> bool>;
        fn outer(self) -> Self::Ret {
            fn is_outer(x: &&Attr) -> bool {
                use Style::*;
                match x.style {
                    Outer => true,
                    Inner(_) => false,
                }
            }
            self.iter().filter(is_outer)
        }
        fn inner(self) -> Self::Ret {
            fn is_inner(x: &&Attr) -> bool {
                use Style::*;
                match x.style {
                    Inner(_) => true,
                    Outer => false,
                }
            }
            self.iter().filter(is_inner)
        }
    }
}

mod cur {
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
            use Entry::*;
            let (tree, len) = match self.entry() {
                Group(x, end) => (x.clone().into(), *end),
                Literal(x) => (x.clone().into(), 1),
                Ident(x) => (x.clone().into(), 1),
                Punct(x) => (x.clone().into(), 1),
                End(_) => return None,
            };
            let rest = unsafe { Cursor::create(self.ptr.add(len), self.scope) };
            Some((tree, rest))
        }
        pub fn span(self) -> Span {
            use Entry::*;
            match self.entry() {
                Group(x, _) => x.span(),
                Literal(x) => x.span(),
                Ident(x) => x.span(),
                Punct(x) => x.span(),
                End(_) => Span::call_site(),
            }
        }
        pub(crate) fn prev_span(mut self) -> Span {
            if buff_start(self) < self.ptr {
                self.ptr = unsafe { self.ptr.offset(-1) };
                if let Entry::End(_) = self.entry() {
                    let mut depth = 1;
                    loop {
                        self.ptr = unsafe { self.ptr.offset(-1) };
                        use Entry::*;
                        match self.entry() {
                            Group(x, _) => {
                                depth -= 1;
                                if depth == 0 {
                                    return x.span();
                                }
                            },
                            End(_) => depth += 1,
                            Literal(_) | Ident(_) | Punct(_) => {},
                        }
                    }
                }
            }
            self.span()
        }
        pub(crate) fn skip(self) -> Option<Cursor<'a>> {
            use Entry::*;
            let y = match self.entry() {
                End(_) => return None,
                Punct(x) if x.as_char() == '\'' && x.spacing() == Spacing::Joint => {
                    match unsafe { &*self.ptr.add(1) } {
                        Ident(_) => 2,
                        _ => 1,
                    }
                },
                Group(_, x) => *x,
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
    pub struct Buffer {
        entries: Box<[Entry]>,
    }
    impl Buffer {
        fn recursive_new(ys: &mut Vec<Entry>, xs: TokenStream) {
            for x in xs {
                use Entry::*;
                match x {
                    TokenTree::Ident(x) => ys.push(Ident(x)),
                    TokenTree::Punct(x) => ys.push(Punct(x)),
                    TokenTree::Literal(x) => ys.push(Literal(x)),
                    TokenTree::Group(x) => {
                        let beg = ys.len();
                        ys.push(End(0));
                        Self::recursive_new(ys, x.stream());
                        let end = ys.len();
                        ys.push(End(-(end as isize)));
                        let off = end - beg;
                        ys[beg] = Group(x, off);
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
}
use cur::Cursor;
mod ident {
    pub use proc_macro2::Ident;

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
        pub apos: Span,
        pub ident: Ident,
    }
    impl Lifetime {
        pub fn new(x: &str, s: Span) -> Self {
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
        pub fn span(&self) -> Span {
            self.apos.join(self.ident.span()).unwrap_or(self.apos)
        }
        pub fn set_span(&mut self, s: Span) {
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
pub use ident::{Ident, Lifetime};
pub mod punct;
use punct::Punctuated;

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

pub mod gen {
    pub mod bound {
        ast_enum_of_structs! {
            pub enum Type {
                Trait(Trait),
                Lifetime(Lifetime),
                Verbatim(TokenStream),
            }
        }
        pub struct Trait {
            pub paren: Option<tok::Paren>,
            pub modifier: Modifier,
            pub lifes: Option<Lifes>,
            pub path: path::Path,
        }
        pub enum Modifier {
            None,
            Maybe(Token![?]),
        }
        pub struct Lifes {
            pub for_: Token![for],
            pub lt: Token![<],
            pub lifes: Punctuated<Param, Token![,]>,
            pub gt: Token![>],
        }
        impl Default for Lifes {
            fn default() -> Self {
                Lifes {
                    for_: Default::default(),
                    lt: Default::default(),
                    lifes: Punctuated::new(),
                    gt: Default::default(),
                }
            }
        }
    }
    pub mod param {
        ast_enum_of_structs! {
            pub enum Param {
                Life(Life),
                Type(Type),
                Const(Const),
            }
        }
        pub struct Life {
            pub attrs: Vec<attr::Attr>,
            pub life: Lifetime,
            pub colon: Option<Token![:]>,
            pub bounds: Punctuated<Lifetime, Token![+]>,
        }
        impl Life {
            pub fn new(life: Lifetime) -> Self {
                param::Life {
                    attrs: Vec::new(),
                    life,
                    colon: None,
                    bounds: Punctuated::new(),
                }
            }
        }
        pub struct Lifes<'a>(Iter<'a, Param>);
        impl<'a> Iterator for Lifes<'a> {
            type Item = &'a Life;
            fn next(&mut self) -> Option<Self::Item> {
                let y = match self.0.next() {
                    Some(x) => x,
                    None => return None,
                };
                if let Param::Life(x) = y {
                    Some(&x)
                } else {
                    self.next()
                }
            }
        }
        pub struct LifesMut<'a>(IterMut<'a, Param>);
        impl<'a> Iterator for LifesMut<'a> {
            type Item = &'a mut Life;
            fn next(&mut self) -> Option<Self::Item> {
                let y = match self.0.next() {
                    Some(x) => x,
                    None => return None,
                };
                if let Param::Life(x) = y {
                    Some(&mut x)
                } else {
                    self.next()
                }
            }
        }
        pub struct Type {
            pub attrs: Vec<attr::Attr>,
            pub ident: Ident,
            pub colon: Option<Token![:]>,
            pub bounds: Punctuated<bound::Type, Token![+]>,
            pub eq: Option<Token![=]>,
            pub default: Option<ty::Type>,
        }
        impl From<Ident> for param::Type {
            fn from(ident: Ident) -> Self {
                param::Type {
                    attrs: vec![],
                    ident,
                    colon: None,
                    bounds: Punctuated::new(),
                    eq: None,
                    default: None,
                }
            }
        }
        pub struct Types<'a>(Iter<'a, Param>);
        impl<'a> Iterator for Types<'a> {
            type Item = &'a Type;
            fn next(&mut self) -> Option<Self::Item> {
                let y = match self.0.next() {
                    Some(x) => x,
                    None => return None,
                };
                if let Param::Type(x) = y {
                    Some(&x)
                } else {
                    self.next()
                }
            }
        }
        pub struct TypesMut<'a>(IterMut<'a, Param>);
        impl<'a> Iterator for TypesMut<'a> {
            type Item = &'a mut Type;
            fn next(&mut self) -> Option<Self::Item> {
                let y = match self.0.next() {
                    Some(x) => x,
                    None => return None,
                };
                if let Param::Type(x) = y {
                    Some(&mut x)
                } else {
                    self.next()
                }
            }
        }
        pub struct Const {
            pub attrs: Vec<attr::Attr>,
            pub const_: Token![const],
            pub ident: Ident,
            pub colon: Token![:],
            pub typ: ty::Type,
            pub eq: Option<Token![=]>,
            pub default: Option<Expr>,
        }
        pub struct Consts<'a>(Iter<'a, Param>);
        impl<'a> Iterator for Consts<'a> {
            type Item = &'a Const;
            fn next(&mut self) -> Option<Self::Item> {
                let y = match self.0.next() {
                    Some(x) => x,
                    None => return None,
                };
                if let Param::Const(x) = y {
                    Some(&x)
                } else {
                    self.next()
                }
            }
        }
        pub struct ConstsMut<'a>(IterMut<'a, Param>);
        impl<'a> Iterator for ConstsMut<'a> {
            type Item = &'a mut Const;
            fn next(&mut self) -> Option<Self::Item> {
                let y = match self.0.next() {
                    Some(x) => x,
                    None => return None,
                };
                if let Param::Const(x) = y {
                    Some(&mut x)
                } else {
                    self.next()
                }
            }
        }
    }
    use param::Param;
    pub struct Where {
        pub where_: Token![where],
        pub preds: Punctuated<Where::Pred, Token![,]>,
    }
    pub mod Where {
        ast_enum_of_structs! {
            pub enum Pred {
                Life(Life),
                Type(Type),
            }
        }
        pub struct Life {
            pub life: Lifetime,
            pub colon: Token![:],
            pub bounds: Punctuated<Lifetime, Token![+]>,
        }
        pub struct Type {
            pub lifes: Option<bound::Lifes>,
            pub bounded: ty::Type,
            pub colon: Token![:],
            pub bounds: Punctuated<bound::Type, Token![+]>,
        }
    }
    pub struct Gens {
        pub lt: Option<Token![<]>,
        pub ps: Punctuated<Param, Token![,]>,
        pub gt: Option<Token![>]>,
        pub where_: Option<Where>,
    }
    impl Gens {
        pub fn life_ps(&self) -> param::Lifes {
            param::Lifes(self.ps.iter())
        }
        pub fn life_ps_mut(&mut self) -> param::LifesMut {
            param::LifesMut(self.ps.iter_mut())
        }
        pub fn type_ps(&self) -> param::Types {
            param::Types(self.ps.iter())
        }
        pub fn type_ps_mut(&mut self) -> param::TypesMut {
            param::TypesMut(self.ps.iter_mut())
        }
        pub fn const_ps(&self) -> param::Consts {
            param::Consts(self.ps.iter())
        }
        pub fn const_ps_mut(&mut self) -> param::ConstsMut {
            param::ConstsMut(self.ps.iter_mut())
        }
        pub fn make_where_clause(&mut self) -> &mut Where {
            self.where_.get_or_insert_with(|| Where {
                where_: <Token![where]>::default(),
                preds: Punctuated::new(),
            })
        }
        pub fn split_for_impl(&self) -> (Impl, Type, Option<&Where>) {
            (Impl(self), Type(self), self.where_.as_ref())
        }
    }
    impl Default for Gens {
        fn default() -> Self {
            Gens {
                lt: None,
                ps: Punctuated::new(),
                gt: None,
                where_: None,
            }
        }
    }
    pub struct Impl<'a>(pub &'a Gens);
    pub struct Type<'a>(pub &'a Gens);
    pub struct Turbofish<'a>(pub &'a Gens);
    macro_rules! gens_impls {
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
    gens_impls!(Impl);
    gens_impls!(Type);
    gens_impls!(Turbofish);
    impl<'a> Type<'a> {
        pub fn as_turbofish(&self) -> Turbofish {
            Turbofish(self.0)
        }
    }
}
pub mod expr;
pub mod item;
pub mod lit;
pub mod tok;

mod pat {
    ast_enum_of_structs! {
        pub enum Pat {
            Const(Const),
            Ident(Ident),
            Lit(Lit),
            Mac(Mac),
            Or(Or),
            Paren(Paren),
            Path(Path),
            Range(Range),
            Ref(Ref),
            Rest(Rest),
            Slice(Slice),
            Struct(Struct),
            Tuple(Tuple),
            TupleStruct(TupleStruct),
            Type(Type),
            Verbatim(TokenStream),
            Wild(Wild),
        }
    }
    use expr::Const;
    pub struct Ident {
        pub attrs: Vec<attr::Attr>,
        pub ref_: Option<Token![ref]>,
        pub mut_: Option<Token![mut]>,
        pub ident: Ident,
        pub sub: Option<(Token![@], Box<Pat>)>,
    }
    use expr::Lit;
    use expr::Macro as Mac;
    pub struct Or {
        pub attrs: Vec<attr::Attr>,
        pub vert: Option<Token![|]>,
        pub cases: Punctuated<Pat, Token![|]>,
    }
    pub struct Paren {
        pub attrs: Vec<attr::Attr>,
        pub paren: tok::Paren,
        pub pat: Box<Pat>,
    }
    use expr::Path;
    use expr::Range;
    pub struct Ref {
        pub attrs: Vec<attr::Attr>,
        pub and: Token![&],
        pub mut_: Option<Token![mut]>,
        pub pat: Box<Pat>,
    }
    pub struct Rest {
        pub attrs: Vec<attr::Attr>,
        pub dot2: Token![..],
    }
    pub struct Slice {
        pub attrs: Vec<attr::Attr>,
        pub bracket: tok::Bracket,
        pub elems: Punctuated<Pat, Token![,]>,
    }
    pub struct Struct {
        pub attrs: Vec<attr::Attr>,
        pub qself: Option<QSelf>,
        pub path: Path,
        pub brace: tok::Brace,
        pub fields: Punctuated<Field, Token![,]>,
        pub rest: Option<Rest>,
    }
    pub struct Tuple {
        pub attrs: Vec<attr::Attr>,
        pub paren: tok::Paren,
        pub elems: Punctuated<Pat, Token![,]>,
    }
    pub struct TupleStruct {
        pub attrs: Vec<attr::Attr>,
        pub qself: Option<QSelf>,
        pub path: Path,
        pub paren: tok::Paren,
        pub elems: Punctuated<Pat, Token![,]>,
    }
    pub struct Type {
        pub attrs: Vec<attr::Attr>,
        pub pat: Box<Pat>,
        pub colon: Token![:],
        pub typ: Box<ty::Type>,
    }
    pub struct Wild {
        pub attrs: Vec<attr::Attr>,
        pub underscore: Token![_],
    }
    pub struct Field {
        pub attrs: Vec<attr::Attr>,
        pub member: Member,
        pub colon: Option<Token![:]>,
        pub pat: Box<Pat>,
    }
}
mod path {
    pub struct Path {
        pub colon: Option<Token![::]>,
        pub segs: Punctuated<Segment, Token![::]>,
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

    pub struct Segment {
        pub ident: Ident,
        pub args: Args,
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

    pub enum Args {
        None,
        Angled(AngledArgs),
        Parenthesized(ParenthesizedArgs),
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

    pub enum Arg {
        Lifetime(Lifetime),
        Type(ty::Type),
        Const(Expr),
        AssocType(AssocType),
        AssocConst(AssocConst),
        Constraint(Constraint),
    }

    pub struct AngledArgs {
        pub colon2: Option<Token![::]>,
        pub lt: Token![<],
        pub args: Punctuated<Arg, Token![,]>,
        pub gt: Token![>],
    }
    pub struct AssocType {
        pub ident: Ident,
        pub args: Option<AngledArgs>,
        pub eq: Token![=],
        pub typ: ty::Type,
    }
    pub struct AssocConst {
        pub ident: Ident,
        pub args: Option<AngledArgs>,
        pub eq: Token![=],
        pub val: Expr,
    }
    pub struct Constraint {
        pub ident: Ident,
        pub args: Option<AngledArgs>,
        pub colon: Token![:],
        pub bounds: Punctuated<gen::bound::Type, Token![+]>,
    }
    pub struct ParenthesizedArgs {
        pub paren: tok::Paren,
        pub args: Punctuated<ty::Type, Token![,]>,
        pub ret: Ret,
    }
    pub struct QSelf {
        pub lt: Token![<],
        pub typ: Box<ty::Type>,
        pub pos: usize,
        pub as_: Option<Token![as]>,
        pub gt: Token![>],
    }
}
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
mod ty {
    ast_enum_of_structs! {
        pub enum Type {
            Array(Array),
            Fn(Fn),
            Group(Group),
            Impl(Impl),
            Infer(Infer),
            Mac(Mac),
            Never(Never),
            Paren(Paren),
            Path(Path),
            Ptr(Ptr),
            Ref(Ref),
            Slice(Slice),
            TraitObj(TraitObj),
            Tuple(Tuple),
            Verbatim(TokenStream),
        }
    }
    pub struct Array {
        pub bracket: tok::Bracket,
        pub elem: Box<Type>,
        pub semi: Token![;],
        pub len: Expr,
    }
    pub struct Fn {
        pub lifes: Option<gen::bound::Lifes>,
        pub unsafe_: Option<Token![unsafe]>,
        pub abi: Option<Abi>,
        pub fn_: Token![fn],
        pub paren: tok::Paren,
        pub args: Punctuated<FnArg, Token![,]>,
        pub vari: Option<Variadic>,
        pub ret: Ret,
    }
    pub struct Group {
        pub group: tok::Group,
        pub elem: Box<Type>,
    }
    pub struct Impl {
        pub impl_: Token![impl],
        pub bounds: Punctuated<gen::bound::Type, Token![+]>,
    }
    pub struct Infer {
        pub underscore: Token![_],
    }
    pub struct Mac {
        pub mac: Macro,
    }
    pub struct Never {
        pub bang: Token![!],
    }
    pub struct Paren {
        pub paren: tok::Paren,
        pub elem: Box<Type>,
    }
    pub struct Path {
        pub qself: Option<QSelf>,
        pub path: Path,
    }
    pub struct Ptr {
        pub star: Token![*],
        pub const_: Option<Token![const]>,
        pub mut_: Option<Token![mut]>,
        pub elem: Box<Type>,
    }
    pub struct Ref {
        pub and: Token![&],
        pub life: Option<Lifetime>,
        pub mut_: Option<Token![mut]>,
        pub elem: Box<Type>,
    }
    pub struct Slice {
        pub bracket: tok::Bracket,
        pub elem: Box<Type>,
    }
    pub struct TraitObj {
        pub dyn_: Option<Token![dyn]>,
        pub bounds: Punctuated<gen::bound::Type, Token![+]>,
    }
    pub struct Tuple {
        pub paren: tok::Paren,
        pub elems: Punctuated<Type, Token![,]>,
    }
    pub struct Abi {
        pub extern_: Token![extern],
        pub name: Option<lit::Str>,
    }
    pub struct FnArg {
        pub attrs: Vec<attr::Attr>,
        pub name: Option<(Ident, Token![:])>,
        pub ty: Type,
    }
    pub struct Variadic {
        pub attrs: Vec<attr::Attr>,
        pub name: Option<(Ident, Token![:])>,
        pub dots: Token![...],
        pub comma: Option<Token![,]>,
    }
    pub enum Ret {
        Default,
        Type(Token![->], Box<Type>),
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
        pub typ: ty::Type,
    }
    pub enum Mut {
        None,
    }
}
use data::DeriveInput;

mod err {
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
        sealed::look,
        tok::Custom,
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
    impl Custom for private::IdentAny {
        fn peek(cursor: Cursor) -> bool {
            cursor.ident().is_some()
        }
        fn display() -> &'static str {
            "identifier"
        }
    }
    impl look::Sealed for private::PeekFn {}
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

mod look {
    use super::{
        err::{self, Err},
        sealed::look::Sealed,
        tok::Tok,
    };
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

pub mod parse {
    use super::{
        err::{self, Err, Result},
        look::{self, Lookahead1, Peek},
        proc_macro,
        punct::Punctuated,
        tok::Tok,
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
            look::new(self.scope, self.cursor())
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
    fn tokens_to_parse_buffer(tokens: &cur::Buffer) -> ParseBuffer {
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
            let buf = cur::Buffer::new2(tokens);
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
            let buf = cur::Buffer::new2(tokens);
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
impl ParseQuote for attr::Attr {
    fn parse(x: ParseStream) -> Result<Self> {
        if x.peek(Token![#]) && x.peek2(Token![!]) {
            parsing::single_parse_inner(x)
        } else {
            parsing::single_parse_outer(x)
        }
    }
}
impl ParseQuote for pat::Pat {
    fn parse(x: ParseStream) -> Result<Self> {
        pat::Pat::parse_multi_with_leading_vert(x)
    }
}
impl ParseQuote for Box<pat::Pat> {
    fn parse(x: ParseStream) -> Result<Self> {
        <pat::Pat as ParseQuote>::parse(x).map(Box::new)
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

pub struct File {
    pub shebang: Option<String>,
    pub attrs: Vec<attr::Attr>,
    pub items: Vec<Item>,
}
pub fn parse_file(mut x: &str) -> Result<File> {
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
pub fn parse_str<T: parse::Parse>(s: &str) -> Result<T> {
    parse::Parser::parse_str(T::parse, s)
}

pub mod lower;
pub mod parsing;
