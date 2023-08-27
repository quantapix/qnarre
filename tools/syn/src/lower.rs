use super::{
    ident,
    pm2::{Delim, DelimSpan, Group, Ident, Lit, Punct, Spacing, Span, Stream, Tree},
};
use std::{borrow::Cow, fmt, iter, ops::Deref, option::Option, rc::Rc};

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

macro_rules! impl_Fragment {
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
impl_Fragment!(bool, str, String, char);
impl_Fragment!(u8, u16, u32, u64, u128, usize);

pub trait Lower {
    fn lower(&self, s: &mut Stream);
    fn to_stream(&self) -> Stream {
        let mut y = Stream::new();
        self.lower(&mut y);
        y
    }
    fn into_stream(self) -> Stream
    where
        Self: Sized,
    {
        self.to_stream()
    }
}
impl<'a, T: ?Sized + Lower> Lower for &'a T {
    fn lower(&self, s: &mut Stream) {
        (**self).lower(s);
    }
}
impl<'a, T: ?Sized + Lower> Lower for &'a mut T {
    fn lower(&self, s: &mut Stream) {
        (**self).lower(s);
    }
}
impl<'a, T: ?Sized + ToOwned + Lower> Lower for Cow<'a, T> {
    fn lower(&self, s: &mut Stream) {
        (**self).lower(s);
    }
}
impl<T: ?Sized + Lower> Lower for Box<T> {
    fn lower(&self, s: &mut Stream) {
        (**self).lower(s);
    }
}
impl<T: ?Sized + Lower> Lower for Rc<T> {
    fn lower(&self, s: &mut Stream) {
        (**self).lower(s);
    }
}
impl<T: Lower> Lower for Option<T> {
    fn lower(&self, s: &mut Stream) {
        if let Some(ref x) = *self {
            x.lower(s);
        }
    }
}
impl Lower for bool {
    fn lower(&self, s: &mut Stream) {
        let y = if *self { "true" } else { "false" };
        s.append(Ident::new(y, Span::call_site()));
    }
}
impl Lower for char {
    fn lower(&self, s: &mut Stream) {
        s.append(Lit::char(*self));
    }
}
impl Lower for Group {
    fn lower(&self, s: &mut Stream) {
        s.append(self.clone());
    }
}
impl Lower for Ident {
    fn lower(&self, s: &mut Stream) {
        s.append(self.clone());
    }
}
impl Lower for Lit {
    fn lower(&self, s: &mut Stream) {
        s.append(self.clone());
    }
}
impl Lower for Punct {
    fn lower(&self, s: &mut Stream) {
        s.append(self.clone());
    }
}
impl Lower for str {
    fn lower(&self, s: &mut Stream) {
        s.append(Lit::string(self));
    }
}
impl Lower for String {
    fn lower(&self, s: &mut Stream) {
        self.as_str().lower(s);
    }
}
impl Lower for Stream {
    fn lower(&self, s: &mut Stream) {
        s.extend(iter::once(self.clone()));
    }
    fn into_stream(self) -> Stream {
        self
    }
}
impl Lower for Tree {
    fn lower(&self, s: &mut Stream) {
        s.append(self.clone());
    }
}

macro_rules! impl_Lower {
    ($($x:ident => $y:ident)*) => {
        $(
            impl Lower for $x {
                fn lower(&self, s: &mut Stream) {
                    s.append(Lit::$y(*self));
                }
            }
        )*
    };
}
impl_Lower! {
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
impl<T: ?Sized + Lower> Spanned for T {
    fn __span(&self) -> Span {
        join_spans(self.into_stream())
    }
}

fn join_spans(s: Stream) -> Span {
    let mut y = s.into_iter().map(|x| x.span());
    let first = match y.next() {
        Some(x) => x,
        None => return Span::call_site(),
    };
    y.fold(None, |_, x| Some(x))
        .and_then(|x| first.join(x))
        .unwrap_or(first)
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

macro_rules! push_punct {
    ($name:ident $spanned:ident $char1:tt) => {
        pub fn $name(tokens: &mut Stream) {
            tokens.append(Punct::new($char1, Spacing::Alone));
        }
        pub fn $spanned(tokens: &mut Stream, span: Span) {
            let mut punct = Punct::new($char1, Spacing::Alone);
            punct.set_span(span);
            tokens.append(punct);
        }
    };
    ($name:ident $spanned:ident $char1:tt $char2:tt) => {
        pub fn $name(tokens: &mut Stream) {
            tokens.append(Punct::new($char1, Spacing::Joint));
            tokens.append(Punct::new($char2, Spacing::Alone));
        }
        pub fn $spanned(tokens: &mut Stream, span: Span) {
            let mut punct = Punct::new($char1, Spacing::Joint);
            punct.set_span(span);
            tokens.append(punct);
            let mut punct = Punct::new($char2, Spacing::Alone);
            punct.set_span(span);
            tokens.append(punct);
        }
    };
    ($name:ident $spanned:ident $char1:tt $char2:tt $char3:tt) => {
        pub fn $name(tokens: &mut Stream) {
            tokens.append(Punct::new($char1, Spacing::Joint));
            tokens.append(Punct::new($char2, Spacing::Joint));
            tokens.append(Punct::new($char3, Spacing::Alone));
        }
        pub fn $spanned(tokens: &mut Stream, span: Span) {
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

pub fn push_group(s: &mut Stream, delim: Delim, inner: Stream) {
    s.append(Group::new(delim, inner));
}
pub fn push_group_spanned(s: &mut Stream, span: Span, delim: Delim, inner: Stream) {
    let mut y = Group::new(delim, inner);
    y.set_span(span);
    s.append(y);
}
pub fn parse(s: &mut Stream, x: &str) {
    let y: Stream = x.parse().expect("invalid token stream");
    s.extend(iter::once(y));
}
pub fn parse_spanned(s: &mut Stream, span: Span, x: &str) {
    let y: Stream = x.parse().expect("invalid token stream");
    s.extend(y.into_iter().map(|x| respan_tree(x, span)));
}
fn respan_tree(mut x: Tree, span: Span) -> Tree {
    match &mut x {
        Tree::Group(x) => {
            let y = x.stream().into_iter().map(|x| respan_tree(x, span)).collect();
            *x = Group::new(x.delim(), y);
            x.set_span(span);
        },
        x => x.set_span(span),
    }
    x
}
pub fn push_ident(s: &mut Stream, x: &str) {
    let span = Span::call_site();
    push_ident_spanned(s, span, x);
}
pub fn push_ident_spanned(s: &mut Stream, span: Span, x: &str) {
    s.append(ident::maybe_raw(x, span));
}
pub fn push_life(s: &mut Stream, x: &str) {
    struct Life<'a> {
        name: &'a str,
        state: u8,
    }
    impl<'a> Iterator for Life<'a> {
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
    s.extend(Life {
        name: &x[1..],
        state: 0,
    });
}
pub fn push_life_spanned(s: &mut Stream, span: Span, x: &str) {
    struct Life<'a> {
        name: &'a str,
        span: Span,
        state: u8,
    }
    impl<'a> Iterator for Life<'a> {
        type Item = Tree;
        fn next(&mut self) -> Option<Self::Item> {
            match self.state {
                0 => {
                    self.state = 1;
                    let mut y = Punct::new('\'', Spacing::Joint);
                    y.set_span(self.span);
                    Some(Tree::Punct(y))
                },
                1 => {
                    self.state = 2;
                    Some(Tree::Ident(Ident::new(self.name, self.span)))
                },
                _ => None,
            }
        }
    }
    s.extend(Life {
        name: &x[1..],
        span,
        state: 0,
    });
}
pub fn push_underscore(s: &mut Stream) {
    push_underscore_spanned(s, Span::call_site());
}
pub fn push_underscore_spanned(s: &mut Stream, span: Span) {
    s.append(Ident::new("_", span));
}

#[derive(Copy, Clone)]
pub struct FragAdaptor<T: Fragment>(pub T);
impl<T: Fragment> FragAdaptor<T> {
    pub fn span(&self) -> Option<Span> {
        self.0.span()
    }
}
impl<T: Fragment> fmt::Display for FragAdaptor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Fragment::fmt(&self.0, f)
    }
}
impl<T: Fragment + fmt::Octal> fmt::Octal for FragAdaptor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Octal::fmt(&self.0, f)
    }
}
impl<T: Fragment + fmt::LowerHex> fmt::LowerHex for FragAdaptor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}
impl<T: Fragment + fmt::UpperHex> fmt::UpperHex for FragAdaptor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::UpperHex::fmt(&self.0, f)
    }
}
impl<T: Fragment + fmt::Binary> fmt::Binary for FragAdaptor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Binary::fmt(&self.0, f)
    }
}
