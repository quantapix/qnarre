use super::pm2::{extra::DelimSpan, Delim, Group, Ident, Lit, Punct, Spacing, Span, TokenStream, Tree};
use std::{
    borrow::Cow,
    collections::btree_set::{self, BTreeSet},
    fmt, format, iter,
    ops::{BitOr, Deref},
    option::Option,
    rc::Rc,
    slice,
};

pub trait StreamExt {
    fn append<U>(&mut self, x: U)
    where
        U: Into<Tree>;
    fn append_all<I>(&mut self, xs: I)
    where
        I: IntoIterator,
        I::Item: ToStream;
    fn append_sep<I, U>(&mut self, xs: I, op: U)
    where
        I: IntoIterator,
        I::Item: ToStream,
        U: ToStream;
    fn append_term<I, U>(&mut self, xs: I, term: U)
    where
        I: IntoIterator,
        I::Item: ToStream,
        U: ToStream;
}
impl StreamExt for TokenStream {
    fn append<U>(&mut self, x: U)
    where
        U: Into<Tree>,
    {
        self.extend(iter::once(x.into()));
    }
    fn append_all<I>(&mut self, xs: I)
    where
        I: IntoIterator,
        I::Item: ToStream,
    {
        for x in xs {
            x.to_tokens(self);
        }
    }
    fn append_sep<I, U>(&mut self, xs: I, op: U)
    where
        I: IntoIterator,
        I::Item: ToStream,
        U: ToStream,
    {
        for (i, x) in xs.into_iter().enumerate() {
            if i > 0 {
                op.to_tokens(self);
            }
            x.to_tokens(self);
        }
    }
    fn append_term<I, U>(&mut self, xs: I, term: U)
    where
        I: IntoIterator,
        I::Item: ToStream,
        U: ToStream,
    {
        for x in xs {
            x.to_tokens(self);
            term.to_tokens(self);
        }
    }
}

#[macro_export]
macro_rules! format_ident {
    ($fmt:expr) => {
        $crate::format_ident_impl!([
            $crate::Option::None,
            $fmt
        ])
    };
    ($fmt:expr, $($rest:tt)*) => {
        $crate::format_ident_impl!([
            $crate::Option::None,
            $fmt
        ] $($rest)*)
    };
}
#[macro_export]
macro_rules! format_ident_impl {
    ([$span:expr, $($fmt:tt)*]) => {
        $crate::mk_ident(
            &$crate::format!($($fmt)*),
            $span,
        )
    };
    ([$old:expr, $($fmt:tt)*] span = $span:expr) => {
        $crate::format_ident_impl!([$old, $($fmt)*] span = $span,)
    };
    ([$old:expr, $($fmt:tt)*] span = $span:expr, $($rest:tt)*) => {
        $crate::format_ident_impl!([
            $crate::Option::Some::<$crate::Span>($span),
            $($fmt)*
        ] $($rest)*)
    };
    ([$span:expr, $($fmt:tt)*] $name:ident = $arg:expr) => {
        $crate::format_ident_impl!([$span, $($fmt)*] $name = $arg,)
    };
    ([$span:expr, $($fmt:tt)*] $name:ident = $arg:expr, $($rest:tt)*) => {
        match $crate::IdentFragmentAdapter(&$arg) {
            arg => $crate::format_ident_impl!([$span.or(arg.span()), $($fmt)*, $name = arg] $($rest)*),
        }
    };
    ([$span:expr, $($fmt:tt)*] $arg:expr) => {
        $crate::format_ident_impl!([$span, $($fmt)*] $arg,)
    };
    ([$span:expr, $($fmt:tt)*] $arg:expr, $($rest:tt)*) => {
        match $crate::IdentFragmentAdapter(&$arg) {
            arg => $crate::format_ident_impl!([$span.or(arg.span()), $($fmt)*, arg] $($rest)*),
        }
    };
}

pub trait Fragment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result;
    fn span(&self) -> Option<Span> {
        None
    }
}
impl<T: Fragment + ?Sized> Fragment for &T {
    fn span(&self) -> Option<Span> {
        <T as Fragment>::span(*self)
    }
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Fragment::fmt(*self, f)
    }
}
impl<T: Fragment + ?Sized> Fragment for &mut T {
    fn span(&self) -> Option<Span> {
        <T as Fragment>::span(*self)
    }
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Fragment::fmt(*self, f)
    }
}
impl Fragment for Ident {
    fn span(&self) -> Option<Span> {
        Some(self.span())
    }
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let id = self.to_string();
        if id.starts_with("r#") {
            fmt::Display::fmt(&id[2..], f)
        } else {
            fmt::Display::fmt(&id[..], f)
        }
    }
}
impl<T> Fragment for Cow<'_, T>
where
    T: Fragment + ToOwned + ?Sized,
{
    fn span(&self) -> Option<Span> {
        T::span(self)
    }
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        T::fmt(self, f)
    }
}

macro_rules! ident_fragment_display {
    ($($T:ty),*) => {
        $(
            impl Fragment for $T {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    fmt::Display::fmt(self, f)
                }
            }
        )*
    };
}
ident_fragment_display!(bool, str, String, char);
ident_fragment_display!(u8, u16, u32, u64, u128, usize);

pub trait ToStream {
    fn to_tokens(&self, tokens: &mut TokenStream);
    fn to_stream(&self) -> TokenStream {
        let mut tokens = TokenStream::new();
        self.to_tokens(&mut tokens);
        tokens
    }
    fn into_stream(self) -> TokenStream
    where
        Self: Sized,
    {
        self.to_stream()
    }
}
impl<'a, T: ?Sized + ToStream> ToStream for &'a T {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        (**self).to_tokens(tokens);
    }
}
impl<'a, T: ?Sized + ToStream> ToStream for &'a mut T {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        (**self).to_tokens(tokens);
    }
}
impl<'a, T: ?Sized + ToOwned + ToStream> ToStream for Cow<'a, T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        (**self).to_tokens(tokens);
    }
}
impl<T: ?Sized + ToStream> ToStream for Box<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        (**self).to_tokens(tokens);
    }
}
impl<T: ?Sized + ToStream> ToStream for Rc<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        (**self).to_tokens(tokens);
    }
}
impl<T: ToStream> ToStream for Option<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if let Some(ref t) = *self {
            t.to_tokens(tokens);
        }
    }
}
impl ToStream for str {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append(Lit::string(self));
    }
}
impl ToStream for String {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.as_str().to_tokens(tokens);
    }
}

macro_rules! primitive {
    ($($t:ident => $name:ident)*) => {
        $(
            impl ToStream for $t {
                fn to_tokens(&self, tokens: &mut TokenStream) {
                    tokens.append(Literal::$name(*self));
                }
            }
        )*
    };
}
primitive! {
    i8 => i8_suffixed
    i16 => i16_suffixed
    i32 => i32_suffixed
    i64 => i64_suffixed
    i128 => i128_suffixed
    isize => isize_suffixed
    u8 => u8_suffixed
    u16 => u16_suffixed
    u32 => u32_suffixed
    u64 => u64_suffixed
    u128 => u128_suffixed
    usize => usize_suffixed
    f32 => f32_suffixed
    f64 => f64_suffixed
}
impl ToStream for char {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append(Lit::character(*self));
    }
}
impl ToStream for bool {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let word = if *self { "true" } else { "false" };
        tokens.append(Ident::new(word, Span::call_site()));
    }
}
impl ToStream for Group {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append(self.clone());
    }
}
impl ToStream for Ident {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append(self.clone());
    }
}
impl ToStream for Punct {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append(self.clone());
    }
}
impl ToStream for Lit {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append(self.clone());
    }
}
impl ToStream for Tree {
    fn to_tokens(&self, dst: &mut TokenStream) {
        dst.append(self.clone());
    }
}
impl ToStream for TokenStream {
    fn to_tokens(&self, dst: &mut TokenStream) {
        dst.extend(iter::once(self.clone()));
    }
    fn into_stream(self) -> TokenStream {
        self
    }
}

pub trait Spanned {
    fn __span(&self) -> Span;
}
impl Spanned for Span {
    fn __span(&self) -> Span {
        *self
    }
}
impl Spanned for DelimSpan {
    fn __span(&self) -> Span {
        self.join()
    }
}
impl<T: ?Sized + ToStream> Spanned for T {
    fn __span(&self) -> Span {
        join_spans(self.into_stream())
    }
}
fn join_spans(s: TokenStream) -> Span {
    let mut y = s.into_iter().map(|x| x.span());
    let first = match y.next() {
        Some(x) => x,
        None => return Span::call_site(),
    };
    y.fold(None, |_, x| Some(x))
        .and_then(|x| first.join(x))
        .unwrap_or(first)
}

pub struct HasIter;
impl BitOr<HasIter> for HasIter {
    type Output = HasIter;
    fn bitor(self, _: HasIter) -> HasIter {
        HasIter
    }
}
impl BitOr<HasNoIter> for HasIter {
    type Output = HasIter;
    fn bitor(self, _: HasNoIter) -> HasIter {
        HasIter
    }
}

pub struct HasNoIter;
impl BitOr<HasNoIter> for HasNoIter {
    type Output = HasNoIter;
    fn bitor(self, _: HasNoIter) -> HasNoIter {
        HasNoIter
    }
}
impl BitOr<HasIter> for HasNoIter {
    type Output = HasIter;
    fn bitor(self, _: HasIter) -> HasIter {
        HasIter
    }
}

pub trait RepIteratorExt: Iterator + Sized {
    fn quote_into_iter(self) -> (Self, HasIter) {
        (self, HasIter)
    }
}
impl<T: Iterator> RepIteratorExt for T {}

pub trait RepToTokensExt {
    fn next(&self) -> Option<&Self> {
        Some(self)
    }
    fn quote_into_iter(&self) -> (&Self, HasNoIter) {
        (self, HasNoIter)
    }
}
impl<T: ToStream + ?Sized> RepToTokensExt for T {}

pub trait RepAsIteratorExt<'q> {
    type Iter: Iterator;
    fn quote_into_iter(&'q self) -> (Self::Iter, HasIter);
}
impl<'q, 'a, T: RepAsIteratorExt<'q> + ?Sized> RepAsIteratorExt<'q> for &'a T {
    type Iter = T::Iter;
    fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
        <T as RepAsIteratorExt>::quote_into_iter(*self)
    }
}
impl<'q, 'a, T: RepAsIteratorExt<'q> + ?Sized> RepAsIteratorExt<'q> for &'a mut T {
    type Iter = T::Iter;
    fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
        <T as RepAsIteratorExt>::quote_into_iter(*self)
    }
}
impl<'q, T: 'q> RepAsIteratorExt<'q> for [T] {
    type Iter = slice::Iter<'q, T>;
    fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
        (self.iter(), HasIter)
    }
}
impl<'q, T: 'q> RepAsIteratorExt<'q> for Vec<T> {
    type Iter = slice::Iter<'q, T>;
    fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
        (self.iter(), HasIter)
    }
}
impl<'q, T: 'q> RepAsIteratorExt<'q> for BTreeSet<T> {
    type Iter = btree_set::Iter<'q, T>;
    fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
        (self.iter(), HasIter)
    }
}
impl<'q, T: RepAsIteratorExt<'q>> RepAsIteratorExt<'q> for RepInterp<T> {
    type Iter = T::Iter;
    fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
        self.0.quote_into_iter()
    }
}

#[derive(Copy, Clone)]
pub struct RepInterp<T>(pub T);
impl<T> RepInterp<T> {
    pub fn next(self) -> Option<T> {
        Some(self.0)
    }
}
impl<T: Iterator> Iterator for RepInterp<T> {
    type Item = T::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
impl<T: ToStream> ToStream for RepInterp<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.0.to_tokens(tokens);
    }
}

pub struct GetSpanBase<T>(pub T);
impl<T> GetSpanBase<T> {
    #[allow(clippy::unused_self)]
    pub fn __into_span(&self) -> T {
        unreachable!()
    }
}
pub struct GetSpanInner<T>(pub GetSpanBase<T>);
impl GetSpanInner<DelimSpan> {
    #[inline]
    pub fn __into_span(&self) -> Span {
        (self.0).0.join()
    }
}
impl<T> Deref for GetSpanInner<T> {
    type Target = GetSpanBase<T>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
pub struct GetSpan<T>(pub GetSpanInner<T>);
impl GetSpan<Span> {
    #[inline]
    pub fn __into_span(self) -> Span {
        ((self.0).0).0
    }
}
impl<T> Deref for GetSpan<T> {
    type Target = GetSpanInner<T>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[inline]
pub fn get_span<T>(x: T) -> GetSpan<T> {
    GetSpan(GetSpanInner(GetSpanBase(x)))
}

pub fn push_group(tokens: &mut TokenStream, delimiter: Delim, inner: TokenStream) {
    tokens.append(Group::new(delimiter, inner));
}
pub fn push_group_spanned(tokens: &mut TokenStream, span: Span, delimiter: Delim, inner: TokenStream) {
    let mut g = Group::new(delimiter, inner);
    g.set_span(span);
    tokens.append(g);
}
pub fn parse(tokens: &mut TokenStream, s: &str) {
    let s: TokenStream = s.parse().expect("invalid token stream");
    tokens.extend(iter::once(s));
}
pub fn parse_spanned(tokens: &mut TokenStream, span: Span, s: &str) {
    let s: TokenStream = s.parse().expect("invalid token stream");
    tokens.extend(s.into_iter().map(|t| respan_token_tree(t, span)));
}
fn respan_token_tree(mut token: Tree, span: Span) -> Tree {
    match &mut token {
        Tree::Group(g) => {
            let stream = g
                .stream()
                .into_iter()
                .map(|token| respan_token_tree(token, span))
                .collect();
            *g = Group::new(g.delim(), stream);
            g.set_span(span);
        },
        other => other.set_span(span),
    }
    token
}
pub fn push_ident(tokens: &mut TokenStream, s: &str) {
    let span = Span::call_site();
    push_ident_spanned(tokens, span, s);
}
pub fn push_ident_spanned(tokens: &mut TokenStream, span: Span, s: &str) {
    tokens.append(ident_maybe_raw(s, span));
}
pub fn push_lifetime(tokens: &mut TokenStream, lifetime: &str) {
    struct Lifetime<'a> {
        name: &'a str,
        state: u8,
    }
    impl<'a> Iterator for Lifetime<'a> {
        type Item = Tree;
        fn next(&mut self) -> Option<Self::Item> {
            match self.state {
                0 => {
                    self.state = 1;
                    Some(Tree::Punct(Punct::new('\'', Spacing::Joint)))
                },
                1 => {
                    self.state = 2;
                    Some(Tree::Ident(Ident::new(self.name, Span::call_site())))
                },
                _ => None,
            }
        }
    }
    tokens.extend(Lifetime {
        name: &lifetime[1..],
        state: 0,
    });
}
pub fn push_lifetime_spanned(tokens: &mut TokenStream, span: Span, lifetime: &str) {
    struct Lifetime<'a> {
        name: &'a str,
        span: Span,
        state: u8,
    }
    impl<'a> Iterator for Lifetime<'a> {
        type Item = Tree;
        fn next(&mut self) -> Option<Self::Item> {
            match self.state {
                0 => {
                    self.state = 1;
                    let mut apostrophe = Punct::new('\'', Spacing::Joint);
                    apostrophe.set_span(self.span);
                    Some(Tree::Punct(apostrophe))
                },
                1 => {
                    self.state = 2;
                    Some(Tree::Ident(Ident::new(self.name, self.span)))
                },
                _ => None,
            }
        }
    }
    tokens.extend(Lifetime {
        name: &lifetime[1..],
        span,
        state: 0,
    });
}

macro_rules! push_punct {
    ($name:ident $spanned:ident $char1:tt) => {
        pub fn $name(tokens: &mut TokenStream) {
            tokens.append(Punct::new($char1, Spacing::Alone));
        }
        pub fn $spanned(tokens: &mut TokenStream, span: Span) {
            let mut punct = Punct::new($char1, Spacing::Alone);
            punct.set_span(span);
            tokens.append(punct);
        }
    };
    ($name:ident $spanned:ident $char1:tt $char2:tt) => {
        pub fn $name(tokens: &mut TokenStream) {
            tokens.append(Punct::new($char1, Spacing::Joint));
            tokens.append(Punct::new($char2, Spacing::Alone));
        }
        pub fn $spanned(tokens: &mut TokenStream, span: Span) {
            let mut punct = Punct::new($char1, Spacing::Joint);
            punct.set_span(span);
            tokens.append(punct);
            let mut punct = Punct::new($char2, Spacing::Alone);
            punct.set_span(span);
            tokens.append(punct);
        }
    };
    ($name:ident $spanned:ident $char1:tt $char2:tt $char3:tt) => {
        pub fn $name(tokens: &mut TokenStream) {
            tokens.append(Punct::new($char1, Spacing::Joint));
            tokens.append(Punct::new($char2, Spacing::Joint));
            tokens.append(Punct::new($char3, Spacing::Alone));
        }
        pub fn $spanned(tokens: &mut TokenStream, span: Span) {
            let mut punct = Punct::new($char1, Spacing::Joint);
            punct.set_span(span);
            tokens.append(punct);
            let mut punct = Punct::new($char2, Spacing::Joint);
            punct.set_span(span);
            tokens.append(punct);
            let mut punct = Punct::new($char3, Spacing::Alone);
            punct.set_span(span);
            tokens.append(punct);
        }
    };
}
push_punct!(push_add push_add_spanned '+');
push_punct!(push_add_eq push_add_eq_spanned '+' '=');
push_punct!(push_and push_and_spanned '&');
push_punct!(push_and_and push_and_and_spanned '&' '&');
push_punct!(push_and_eq push_and_eq_spanned '&' '=');
push_punct!(push_at push_at_spanned '@');
push_punct!(push_bang push_bang_spanned '!');
push_punct!(push_caret push_caret_spanned '^');
push_punct!(push_caret_eq push_caret_eq_spanned '^' '=');
push_punct!(push_colon push_colon_spanned ':');
push_punct!(push_colon2 push_colon2_spanned ':' ':');
push_punct!(push_comma push_comma_spanned ',');
push_punct!(push_div push_div_spanned '/');
push_punct!(push_div_eq push_div_eq_spanned '/' '=');
push_punct!(push_dot push_dot_spanned '.');
push_punct!(push_dot2 push_dot2_spanned '.' '.');
push_punct!(push_dot3 push_dot3_spanned '.' '.' '.');
push_punct!(push_dot_dot_eq push_dot_dot_eq_spanned '.' '.' '=');
push_punct!(push_eq push_eq_spanned '=');
push_punct!(push_eq_eq push_eq_eq_spanned '=' '=');
push_punct!(push_ge push_ge_spanned '>' '=');
push_punct!(push_gt push_gt_spanned '>');
push_punct!(push_le push_le_spanned '<' '=');
push_punct!(push_lt push_lt_spanned '<');
push_punct!(push_mul_eq push_mul_eq_spanned '*' '=');
push_punct!(push_ne push_ne_spanned '!' '=');
push_punct!(push_or push_or_spanned '|');
push_punct!(push_or_eq push_or_eq_spanned '|' '=');
push_punct!(push_or_or push_or_or_spanned '|' '|');
push_punct!(push_pound push_pound_spanned '#');
push_punct!(push_question push_question_spanned '?');
push_punct!(push_rarrow push_rarrow_spanned '-' '>');
push_punct!(push_larrow push_larrow_spanned '<' '-');
push_punct!(push_rem push_rem_spanned '%');
push_punct!(push_rem_eq push_rem_eq_spanned '%' '=');
push_punct!(push_fat_arrow push_fat_arrow_spanned '=' '>');
push_punct!(push_semi push_semi_spanned ';');
push_punct!(push_shl push_shl_spanned '<' '<');
push_punct!(push_shl_eq push_shl_eq_spanned '<' '<' '=');
push_punct!(push_shr push_shr_spanned '>' '>');
push_punct!(push_shr_eq push_shr_eq_spanned '>' '>' '=');
push_punct!(push_star push_star_spanned '*');
push_punct!(push_sub push_sub_spanned '-');
push_punct!(push_sub_eq push_sub_eq_spanned '-' '=');
pub fn push_underscore(tokens: &mut TokenStream) {
    push_underscore_spanned(tokens, Span::call_site());
}
pub fn push_underscore_spanned(tokens: &mut TokenStream, span: Span) {
    tokens.append(Ident::new("_", span));
}
pub fn mk_ident(id: &str, span: Option<Span>) -> Ident {
    let span = span.unwrap_or_else(Span::call_site);
    ident_maybe_raw(id, span)
}
fn ident_maybe_raw(id: &str, span: Span) -> Ident {
    if id.starts_with("r#") {
        Ident::new_raw(&id[2..], span)
    } else {
        Ident::new(id, span)
    }
}

#[derive(Copy, Clone)]
pub struct IdentFragmentAdapter<T: Fragment>(pub T);
impl<T: Fragment> IdentFragmentAdapter<T> {
    pub fn span(&self) -> Option<Span> {
        self.0.span()
    }
}
impl<T: Fragment> fmt::Display for IdentFragmentAdapter<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Fragment::fmt(&self.0, f)
    }
}
impl<T: Fragment + fmt::Octal> fmt::Octal for IdentFragmentAdapter<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Octal::fmt(&self.0, f)
    }
}
impl<T: Fragment + fmt::LowerHex> fmt::LowerHex for IdentFragmentAdapter<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}
impl<T: Fragment + fmt::UpperHex> fmt::UpperHex for IdentFragmentAdapter<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::UpperHex::fmt(&self.0, f)
    }
}
impl<T: Fragment + fmt::Binary> fmt::Binary for IdentFragmentAdapter<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Binary::fmt(&self.0, f)
    }
}

#[macro_export]
macro_rules! quote {
    () => {
        $crate::TokenStream::new()
    };
    ($tt:tt) => {{
        let mut _s = $crate::TokenStream::new();
        $crate::quote_token!{$tt _s}
        _s
    }};
    (# $var:ident) => {{
        let mut _s = $crate::TokenStream::new();
        $crate::ToStream::to_tokens(&$var, &mut _s);
        _s
    }};
    ($tt1:tt $tt2:tt) => {{
        let mut _s = $crate::TokenStream::new();
        $crate::quote_token!{$tt1 _s}
        $crate::quote_token!{$tt2 _s}
        _s
    }};
    ($($tt:tt)*) => {{
        let mut _s = $crate::TokenStream::new();
        $crate::quote_each_token!{_s $($tt)*}
        _s
    }};
}
#[macro_export]
macro_rules! quote_spanned {
    ($span:expr=>) => {{
        let _: $crate::Span = $crate::get_span($span).__into_span();
        $crate::TokenStream::new()
    }};
    ($span:expr=> $tt:tt) => {{
        let mut _s = $crate::TokenStream::new();
        let _span: $crate::Span = $crate::get_span($span).__into_span();
        $crate::quote_token_spanned!{$tt _s _span}
        _s
    }};
    ($span:expr=> # $var:ident) => {{
        let mut _s = $crate::TokenStream::new();
        let _: $crate::Span = $crate::get_span($span).__into_span();
        $crate::ToStream::to_tokens(&$var, &mut _s);
        _s
    }};
    ($span:expr=> $tt1:tt $tt2:tt) => {{
        let mut _s = $crate::TokenStream::new();
        let _span: $crate::Span = $crate::get_span($span).__into_span();
        $crate::quote_token_spanned!{$tt1 _s _span}
        $crate::quote_token_spanned!{$tt2 _s _span}
        _s
    }};
    ($span:expr=> $($tt:tt)*) => {{
        let mut _s = $crate::TokenStream::new();
        let _span: $crate::Span = $crate::get_span($span).__into_span();
        $crate::quote_each_token_spanned!{_s _span $($tt)*}
        _s
    }};
}
#[macro_export]
macro_rules! pounded_var_names {
    ($call:ident! $extra:tt $($tts:tt)*) => {
        $crate::pounded_var_names_with_context!{$call! $extra
            (@ $($tts)*)
            ($($tts)* @)
        }
    };
}
#[macro_export]
macro_rules! pounded_var_names_with_context {
    ($call:ident! $extra:tt ($($b1:tt)*) ($($curr:tt)*)) => {
        $(
            $crate::pounded_var_with_context!{$call! $extra $b1 $curr}
        )*
    };
}
#[macro_export]
macro_rules! pounded_var_with_context {
    ($call:ident! $extra:tt $b1:tt ( $($inner:tt)* )) => {
        $crate::pounded_var_names!{$call! $extra $($inner)*}
    };
    ($call:ident! $extra:tt $b1:tt [ $($inner:tt)* ]) => {
        $crate::pounded_var_names!{$call! $extra $($inner)*}
    };
    ($call:ident! $extra:tt $b1:tt { $($inner:tt)* }) => {
        $crate::pounded_var_names!{$call! $extra $($inner)*}
    };
    ($call:ident!($($extra:tt)*) # $var:ident) => {
        $crate::$call!($($extra)* $var);
    };
    ($call:ident! $extra:tt $b1:tt $curr:tt) => {};
}
#[macro_export]
macro_rules! quote_bind_into_iter {
    ($has_iter:ident $var:ident) => {
        #[allow(unused_mut)]
        let (mut $var, i) = $var.quote_into_iter();
        let $has_iter = $has_iter | i;
    };
}
#[macro_export]
macro_rules! quote_bind_next_or_break {
    ($var:ident) => {
        let $var = match $var.next() {
            Some(_x) => $crate::RepInterp(_x),
            None => break,
        };
    };
}
#[macro_export]
macro_rules! quote_each_token {
    ($tokens:ident $($tts:tt)*) => {
        $crate::quote_tokens_with_context!{$tokens
            (@ @ @ @ @ @ $($tts)*)
            (@ @ @ @ @ $($tts)* @)
            (@ @ @ @ $($tts)* @ @)
            (@ @ @ $(($tts))* @ @ @)
            (@ @ $($tts)* @ @ @ @)
            (@ $($tts)* @ @ @ @ @)
            ($($tts)* @ @ @ @ @ @)
        }
    };
}
#[macro_export]
macro_rules! quote_each_token_spanned {
    ($tokens:ident $span:ident $($tts:tt)*) => {
        $crate::quote_tokens_with_context_spanned!{$tokens $span
            (@ @ @ @ @ @ $($tts)*)
            (@ @ @ @ @ $($tts)* @)
            (@ @ @ @ $($tts)* @ @)
            (@ @ @ $(($tts))* @ @ @)
            (@ @ $($tts)* @ @ @ @)
            (@ $($tts)* @ @ @ @ @)
            ($($tts)* @ @ @ @ @ @)
        }
    };
}
#[macro_export]
macro_rules! quote_tokens_with_context {
    ($tokens:ident
        ($($b3:tt)*) ($($b2:tt)*) ($($b1:tt)*)
        ($($curr:tt)*)
        ($($a1:tt)*) ($($a2:tt)*) ($($a3:tt)*)
    ) => {
        $(
            $crate::quote_token_with_context!{$tokens $b3 $b2 $b1 $curr $a1 $a2 $a3}
        )*
    };
}
#[macro_export]
macro_rules! quote_tokens_with_context_spanned {
    ($tokens:ident $span:ident
        ($($b3:tt)*) ($($b2:tt)*) ($($b1:tt)*)
        ($($curr:tt)*)
        ($($a1:tt)*) ($($a2:tt)*) ($($a3:tt)*)
    ) => {
        $(
            $crate::quote_token_with_context_spanned!{$tokens $span $b3 $b2 $b1 $curr $a1 $a2 $a3}
        )*
    };
}
#[macro_export]
macro_rules! quote_token_with_context {
    ($tokens:ident $b3:tt $b2:tt $b1:tt @ $a1:tt $a2:tt $a3:tt) => {};
    ($tokens:ident $b3:tt $b2:tt $b1:tt (#) ( $($inner:tt)* ) * $a3:tt) => {{
        let has_iter = $crate::ThereIsNoIteratorInRepetition;
        $crate::pounded_var_names!{quote_bind_into_iter!(has_iter) () $($inner)*}
        let _: $crate::HasIterator = has_iter;
        while true {
            $crate::pounded_var_names!{quote_bind_next_or_break!() () $($inner)*}
            $crate::quote_each_token!{$tokens $($inner)*}
        }
    }};
    ($tokens:ident $b3:tt $b2:tt # (( $($inner:tt)* )) * $a2:tt $a3:tt) => {};
    ($tokens:ident $b3:tt # ( $($inner:tt)* ) (*) $a1:tt $a2:tt $a3:tt) => {};
    ($tokens:ident $b3:tt $b2:tt $b1:tt (#) ( $($inner:tt)* ) $sep:tt *) => {{
        let mut _i = 0usize;
        let has_iter = $crate::ThereIsNoIteratorInRepetition;
        $crate::pounded_var_names!{quote_bind_into_iter!(has_iter) () $($inner)*}
        let _: $crate::HasIterator = has_iter;
        while true {
            $crate::pounded_var_names!{quote_bind_next_or_break!() () $($inner)*}
            if _i > 0 {
                $crate::quote_token!{$sep $tokens}
            }
            _i += 1;
            $crate::quote_each_token!{$tokens $($inner)*}
        }
    }};
    ($tokens:ident $b3:tt $b2:tt # (( $($inner:tt)* )) $sep:tt * $a3:tt) => {};
    ($tokens:ident $b3:tt # ( $($inner:tt)* ) ($sep:tt) * $a2:tt $a3:tt) => {};
    ($tokens:ident # ( $($inner:tt)* ) * (*) $a1:tt $a2:tt $a3:tt) => {
        $crate::quote_token!{* $tokens}
    };
    ($tokens:ident # ( $($inner:tt)* ) $sep:tt (*) $a1:tt $a2:tt $a3:tt) => {};
    ($tokens:ident $b3:tt $b2:tt $b1:tt (#) $var:ident $a2:tt $a3:tt) => {
        $crate::ToStream::to_tokens(&$var, &mut $tokens);
    };
    ($tokens:ident $b3:tt $b2:tt # ($var:ident) $a1:tt $a2:tt $a3:tt) => {};
    ($tokens:ident $b3:tt $b2:tt $b1:tt ($curr:tt) $a1:tt $a2:tt $a3:tt) => {
        $crate::quote_token!{$curr $tokens}
    };
}
#[macro_export]
macro_rules! quote_token_with_context_spanned {
    ($tokens:ident $span:ident $b3:tt $b2:tt $b1:tt @ $a1:tt $a2:tt $a3:tt) => {};
    ($tokens:ident $span:ident $b3:tt $b2:tt $b1:tt (#) ( $($inner:tt)* ) * $a3:tt) => {{
        let has_iter = $crate::ThereIsNoIteratorInRepetition;
        $crate::pounded_var_names!{quote_bind_into_iter!(has_iter) () $($inner)*}
        let _: $crate::HasIterator = has_iter;
        while true {
            $crate::pounded_var_names!{quote_bind_next_or_break!() () $($inner)*}
            $crate::quote_each_token_spanned!{$tokens $span $($inner)*}
        }
    }};
    ($tokens:ident $span:ident $b3:tt $b2:tt # (( $($inner:tt)* )) * $a2:tt $a3:tt) => {};
    ($tokens:ident $span:ident $b3:tt # ( $($inner:tt)* ) (*) $a1:tt $a2:tt $a3:tt) => {};
    ($tokens:ident $span:ident $b3:tt $b2:tt $b1:tt (#) ( $($inner:tt)* ) $sep:tt *) => {{
        let mut _i = 0usize;
        let has_iter = $crate::ThereIsNoIteratorInRepetition;
        $crate::pounded_var_names!{quote_bind_into_iter!(has_iter) () $($inner)*}
        let _: $crate::HasIterator = has_iter;
        while true {
            $crate::pounded_var_names!{quote_bind_next_or_break!() () $($inner)*}
            if _i > 0 {
                $crate::quote_token_spanned!{$sep $tokens $span}
            }
            _i += 1;
            $crate::quote_each_token_spanned!{$tokens $span $($inner)*}
        }
    }};
    ($tokens:ident $span:ident $b3:tt $b2:tt # (( $($inner:tt)* )) $sep:tt * $a3:tt) => {};
    ($tokens:ident $span:ident $b3:tt # ( $($inner:tt)* ) ($sep:tt) * $a2:tt $a3:tt) => {};
    ($tokens:ident $span:ident # ( $($inner:tt)* ) * (*) $a1:tt $a2:tt $a3:tt) => {
        $crate::quote_token_spanned!{* $tokens $span}
    };
    ($tokens:ident $span:ident # ( $($inner:tt)* ) $sep:tt (*) $a1:tt $a2:tt $a3:tt) => {};
    ($tokens:ident $span:ident $b3:tt $b2:tt $b1:tt (#) $var:ident $a2:tt $a3:tt) => {
        $crate::ToStream::to_tokens(&$var, &mut $tokens);
    };
    ($tokens:ident $span:ident $b3:tt $b2:tt # ($var:ident) $a1:tt $a2:tt $a3:tt) => {};
    ($tokens:ident $span:ident $b3:tt $b2:tt $b1:tt ($curr:tt) $a1:tt $a2:tt $a3:tt) => {
        $crate::quote_token_spanned!{$curr $tokens $span}
    };
}
#[macro_export]
macro_rules! quote_token {
    ($ident:ident $tokens:ident) => {
        $crate::push_ident(&mut $tokens, stringify!($ident));
    };
    (:: $tokens:ident) => {
        $crate::push_colon2(&mut $tokens);
    };
    (( $($inner:tt)* ) $tokens:ident) => {
        $crate::push_group(
            &mut $tokens,
            $crate::Delimiter::Parenthesis,
            $crate::quote!($($inner)*),
        );
    };
    ([ $($inner:tt)* ] $tokens:ident) => {
        $crate::push_group(
            &mut $tokens,
            $crate::Delimiter::Bracket,
            $crate::quote!($($inner)*),
        );
    };
    ({ $($inner:tt)* } $tokens:ident) => {
        $crate::push_group(
            &mut $tokens,
            $crate::Delimiter::Brace,
            $crate::quote!($($inner)*),
        );
    };
    (# $tokens:ident) => {
        $crate::push_pound(&mut $tokens);
    };
    (, $tokens:ident) => {
        $crate::push_comma(&mut $tokens);
    };
    (. $tokens:ident) => {
        $crate::push_dot(&mut $tokens);
    };
    (; $tokens:ident) => {
        $crate::push_semi(&mut $tokens);
    };
    (: $tokens:ident) => {
        $crate::push_colon(&mut $tokens);
    };
    (+ $tokens:ident) => {
        $crate::push_add(&mut $tokens);
    };
    (+= $tokens:ident) => {
        $crate::push_add_eq(&mut $tokens);
    };
    (& $tokens:ident) => {
        $crate::push_and(&mut $tokens);
    };
    (&& $tokens:ident) => {
        $crate::push_and_and(&mut $tokens);
    };
    (&= $tokens:ident) => {
        $crate::push_and_eq(&mut $tokens);
    };
    (@ $tokens:ident) => {
        $crate::push_at(&mut $tokens);
    };
    (! $tokens:ident) => {
        $crate::push_bang(&mut $tokens);
    };
    (^ $tokens:ident) => {
        $crate::push_caret(&mut $tokens);
    };
    (^= $tokens:ident) => {
        $crate::push_caret_eq(&mut $tokens);
    };
    (/ $tokens:ident) => {
        $crate::push_div(&mut $tokens);
    };
    (/= $tokens:ident) => {
        $crate::push_div_eq(&mut $tokens);
    };
    (.. $tokens:ident) => {
        $crate::push_dot2(&mut $tokens);
    };
    (... $tokens:ident) => {
        $crate::push_dot3(&mut $tokens);
    };
    (..= $tokens:ident) => {
        $crate::push_dot_dot_eq(&mut $tokens);
    };
    (= $tokens:ident) => {
        $crate::push_eq(&mut $tokens);
    };
    (== $tokens:ident) => {
        $crate::push_eq_eq(&mut $tokens);
    };
    (>= $tokens:ident) => {
        $crate::push_ge(&mut $tokens);
    };
    (> $tokens:ident) => {
        $crate::push_gt(&mut $tokens);
    };
    (<= $tokens:ident) => {
        $crate::push_le(&mut $tokens);
    };
    (< $tokens:ident) => {
        $crate::push_lt(&mut $tokens);
    };
    (*= $tokens:ident) => {
        $crate::push_mul_eq(&mut $tokens);
    };
    (!= $tokens:ident) => {
        $crate::push_ne(&mut $tokens);
    };
    (| $tokens:ident) => {
        $crate::push_or(&mut $tokens);
    };
    (|= $tokens:ident) => {
        $crate::push_or_eq(&mut $tokens);
    };
    (|| $tokens:ident) => {
        $crate::push_or_or(&mut $tokens);
    };
    (? $tokens:ident) => {
        $crate::push_question(&mut $tokens);
    };
    (-> $tokens:ident) => {
        $crate::push_rarrow(&mut $tokens);
    };
    (<- $tokens:ident) => {
        $crate::push_larrow(&mut $tokens);
    };
    (% $tokens:ident) => {
        $crate::push_rem(&mut $tokens);
    };
    (%= $tokens:ident) => {
        $crate::push_rem_eq(&mut $tokens);
    };
    (=> $tokens:ident) => {
        $crate::push_fat_arrow(&mut $tokens);
    };
    (<< $tokens:ident) => {
        $crate::push_shl(&mut $tokens);
    };
    (<<= $tokens:ident) => {
        $crate::push_shl_eq(&mut $tokens);
    };
    (>> $tokens:ident) => {
        $crate::push_shr(&mut $tokens);
    };
    (>>= $tokens:ident) => {
        $crate::push_shr_eq(&mut $tokens);
    };
    (* $tokens:ident) => {
        $crate::push_star(&mut $tokens);
    };
    (- $tokens:ident) => {
        $crate::push_sub(&mut $tokens);
    };
    (-= $tokens:ident) => {
        $crate::push_sub_eq(&mut $tokens);
    };
    ($lifetime:lifetime $tokens:ident) => {
        $crate::push_lifetime(&mut $tokens, stringify!($lifetime));
    };
    (_ $tokens:ident) => {
        $crate::push_underscore(&mut $tokens);
    };
    ($other:tt $tokens:ident) => {
        $crate::parse(&mut $tokens, stringify!($other));
    };
}
#[macro_export]
macro_rules! quote_token_spanned {
    ($ident:ident $tokens:ident $span:ident) => {
        $crate::push_ident_spanned(&mut $tokens, $span, stringify!($ident));
    };
    (:: $tokens:ident $span:ident) => {
        $crate::push_colon2_spanned(&mut $tokens, $span);
    };
    (( $($inner:tt)* ) $tokens:ident $span:ident) => {
        $crate::push_group_spanned(
            &mut $tokens,
            $span,
            $crate::Delimiter::Parenthesis,
            $crate::quote_spanned!($span=> $($inner)*),
        );
    };
    ([ $($inner:tt)* ] $tokens:ident $span:ident) => {
        $crate::push_group_spanned(
            &mut $tokens,
            $span,
            $crate::Delimiter::Bracket,
            $crate::quote_spanned!($span=> $($inner)*),
        );
    };
    ({ $($inner:tt)* } $tokens:ident $span:ident) => {
        $crate::push_group_spanned(
            &mut $tokens,
            $span,
            $crate::Delimiter::Brace,
            $crate::quote_spanned!($span=> $($inner)*),
        );
    };
    (# $tokens:ident $span:ident) => {
        $crate::push_pound_spanned(&mut $tokens, $span);
    };
    (, $tokens:ident $span:ident) => {
        $crate::push_comma_spanned(&mut $tokens, $span);
    };
    (. $tokens:ident $span:ident) => {
        $crate::push_dot_spanned(&mut $tokens, $span);
    };
    (; $tokens:ident $span:ident) => {
        $crate::push_semi_spanned(&mut $tokens, $span);
    };
    (: $tokens:ident $span:ident) => {
        $crate::push_colon_spanned(&mut $tokens, $span);
    };
    (+ $tokens:ident $span:ident) => {
        $crate::push_add_spanned(&mut $tokens, $span);
    };
    (+= $tokens:ident $span:ident) => {
        $crate::push_add_eq_spanned(&mut $tokens, $span);
    };
    (& $tokens:ident $span:ident) => {
        $crate::push_and_spanned(&mut $tokens, $span);
    };
    (&& $tokens:ident $span:ident) => {
        $crate::push_and_and_spanned(&mut $tokens, $span);
    };
    (&= $tokens:ident $span:ident) => {
        $crate::push_and_eq_spanned(&mut $tokens, $span);
    };
    (@ $tokens:ident $span:ident) => {
        $crate::push_at_spanned(&mut $tokens, $span);
    };
    (! $tokens:ident $span:ident) => {
        $crate::push_bang_spanned(&mut $tokens, $span);
    };
    (^ $tokens:ident $span:ident) => {
        $crate::push_caret_spanned(&mut $tokens, $span);
    };
    (^= $tokens:ident $span:ident) => {
        $crate::push_caret_eq_spanned(&mut $tokens, $span);
    };
    (/ $tokens:ident $span:ident) => {
        $crate::push_div_spanned(&mut $tokens, $span);
    };
    (/= $tokens:ident $span:ident) => {
        $crate::push_div_eq_spanned(&mut $tokens, $span);
    };
    (.. $tokens:ident $span:ident) => {
        $crate::push_dot2_spanned(&mut $tokens, $span);
    };
    (... $tokens:ident $span:ident) => {
        $crate::push_dot3_spanned(&mut $tokens, $span);
    };
    (..= $tokens:ident $span:ident) => {
        $crate::push_dot_dot_eq_spanned(&mut $tokens, $span);
    };
    (= $tokens:ident $span:ident) => {
        $crate::push_eq_spanned(&mut $tokens, $span);
    };
    (== $tokens:ident $span:ident) => {
        $crate::push_eq_eq_spanned(&mut $tokens, $span);
    };
    (>= $tokens:ident $span:ident) => {
        $crate::push_ge_spanned(&mut $tokens, $span);
    };
    (> $tokens:ident $span:ident) => {
        $crate::push_gt_spanned(&mut $tokens, $span);
    };
    (<= $tokens:ident $span:ident) => {
        $crate::push_le_spanned(&mut $tokens, $span);
    };
    (< $tokens:ident $span:ident) => {
        $crate::push_lt_spanned(&mut $tokens, $span);
    };
    (*= $tokens:ident $span:ident) => {
        $crate::push_mul_eq_spanned(&mut $tokens, $span);
    };
    (!= $tokens:ident $span:ident) => {
        $crate::push_ne_spanned(&mut $tokens, $span);
    };
    (| $tokens:ident $span:ident) => {
        $crate::push_or_spanned(&mut $tokens, $span);
    };
    (|= $tokens:ident $span:ident) => {
        $crate::push_or_eq_spanned(&mut $tokens, $span);
    };
    (|| $tokens:ident $span:ident) => {
        $crate::push_or_or_spanned(&mut $tokens, $span);
    };
    (? $tokens:ident $span:ident) => {
        $crate::push_question_spanned(&mut $tokens, $span);
    };
    (-> $tokens:ident $span:ident) => {
        $crate::push_rarrow_spanned(&mut $tokens, $span);
    };
    (<- $tokens:ident $span:ident) => {
        $crate::push_larrow_spanned(&mut $tokens, $span);
    };
    (% $tokens:ident $span:ident) => {
        $crate::push_rem_spanned(&mut $tokens, $span);
    };
    (%= $tokens:ident $span:ident) => {
        $crate::push_rem_eq_spanned(&mut $tokens, $span);
    };
    (=> $tokens:ident $span:ident) => {
        $crate::push_fat_arrow_spanned(&mut $tokens, $span);
    };
    (<< $tokens:ident $span:ident) => {
        $crate::push_shl_spanned(&mut $tokens, $span);
    };
    (<<= $tokens:ident $span:ident) => {
        $crate::push_shl_eq_spanned(&mut $tokens, $span);
    };
    (>> $tokens:ident $span:ident) => {
        $crate::push_shr_spanned(&mut $tokens, $span);
    };
    (>>= $tokens:ident $span:ident) => {
        $crate::push_shr_eq_spanned(&mut $tokens, $span);
    };
    (* $tokens:ident $span:ident) => {
        $crate::push_star_spanned(&mut $tokens, $span);
    };
    (- $tokens:ident $span:ident) => {
        $crate::push_sub_spanned(&mut $tokens, $span);
    };
    (-= $tokens:ident $span:ident) => {
        $crate::push_sub_eq_spanned(&mut $tokens, $span);
    };
    ($lifetime:lifetime $tokens:ident $span:ident) => {
        $crate::push_lifetime_spanned(&mut $tokens, $span, stringify!($lifetime));
    };
    (_ $tokens:ident $span:ident) => {
        $crate::push_underscore_spanned(&mut $tokens, $span);
    };
    ($other:tt $tokens:ident $span:ident) => {
        $crate::parse_spanned(&mut $tokens, $span, stringify!($other));
    };
}
