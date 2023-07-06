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
#[macro_use]
mod macros;
#[macro_use]
mod group {
    use crate::error::Result;
    use crate::parse::ParseBuffer;
    use crate::token;
    use proc_macro2::extra::DelimSpan;
    use proc_macro2::Delimiter;
    pub struct Parens<'a> {
        pub token: token::Paren,
        pub content: ParseBuffer<'a>,
    }
    pub struct Braces<'a> {
        pub token: token::Brace,
        pub content: ParseBuffer<'a>,
    }
    pub struct Brackets<'a> {
        pub token: token::Bracket,
        pub content: ParseBuffer<'a>,
    }

    pub struct Group<'a> {
        pub token: token::Group,
        pub content: ParseBuffer<'a>,
    }
    pub fn parse_parens<'a>(input: &ParseBuffer<'a>) -> Result<Parens<'a>> {
        parse_delimited(input, Delimiter::Parenthesis).map(|(span, content)| Parens {
            token: token::Paren(span),
            content,
        })
    }
    pub fn parse_braces<'a>(input: &ParseBuffer<'a>) -> Result<Braces<'a>> {
        parse_delimited(input, Delimiter::Brace).map(|(span, content)| Braces {
            token: token::Brace(span),
            content,
        })
    }
    pub fn parse_brackets<'a>(input: &ParseBuffer<'a>) -> Result<Brackets<'a>> {
        parse_delimited(input, Delimiter::Bracket).map(|(span, content)| Brackets {
            token: token::Bracket(span),
            content,
        })
    }

    pub(crate) fn parse_group<'a>(input: &ParseBuffer<'a>) -> Result<Group<'a>> {
        parse_delimited(input, Delimiter::None).map(|(span, content)| Group {
            token: token::Group(span.join()),
            content,
        })
    }
    fn parse_delimited<'a>(input: &ParseBuffer<'a>, delimiter: Delimiter) -> Result<(DelimSpan, ParseBuffer<'a>)> {
        input.step(|cursor| {
            if let Some((content, span, rest)) = cursor.group(delimiter) {
                let scope = crate::buffer::close_span_of_group(*cursor);
                let nested = crate::parse::advance_step_cursor(cursor, content);
                let unexpected = crate::parse::get_unexpected(input);
                let content = crate::parse::new_parse_buffer(scope, nested, unexpected);
                Ok(((span, content), rest))
            } else {
                let message = match delimiter {
                    Delimiter::Parenthesis => "expected parentheses",
                    Delimiter::Brace => "expected curly braces",
                    Delimiter::Bracket => "expected square brackets",
                    Delimiter::None => "expected invisible group",
                };
                Err(cursor.error(message))
            }
        })
    }
    #[macro_export]

    macro_rules! parenthesized {
        ($content:ident in $cursor:expr) => {
            match $crate::__private::parse_parens(&$cursor) {
                $crate::__private::Ok(parens) => {
                    $content = parens.content;
                    parens.token
                },
                $crate::__private::Err(error) => {
                    return $crate::__private::Err(error);
                },
            }
        };
    }
    #[macro_export]

    macro_rules! braced {
        ($content:ident in $cursor:expr) => {
            match $crate::__private::parse_braces(&$cursor) {
                $crate::__private::Ok(braces) => {
                    $content = braces.content;
                    braces.token
                },
                $crate::__private::Err(error) => {
                    return $crate::__private::Err(error);
                },
            }
        };
    }
    #[macro_export]

    macro_rules! bracketed {
        ($content:ident in $cursor:expr) => {
            match $crate::__private::parse_brackets(&$cursor) {
                $crate::__private::Ok(brackets) => {
                    $content = brackets.content;
                    brackets.token
                },
                $crate::__private::Err(error) => {
                    return $crate::__private::Err(error);
                },
            }
        };
    }
}
#[macro_use]
pub mod token;
mod attr;
pub use crate::attr::{AttrStyle, Attribute, Meta, MetaList, MetaNameValue};
mod bigint {
    use std::ops::{AddAssign, MulAssign};
    pub(crate) struct BigInt {
        digits: Vec<u8>,
    }
    impl BigInt {
        pub(crate) fn new() -> Self {
            BigInt { digits: Vec::new() }
        }
        pub(crate) fn to_string(&self) -> String {
            let mut repr = String::with_capacity(self.digits.len());
            let mut has_nonzero = false;
            for digit in self.digits.iter().rev() {
                has_nonzero |= *digit != 0;
                if has_nonzero {
                    repr.push((*digit + b'0') as char);
                }
            }
            if repr.is_empty() {
                repr.push('0');
            }
            repr
        }
        fn reserve_two_digits(&mut self) {
            let len = self.digits.len();
            let desired = len + !self.digits.ends_with(&[0, 0]) as usize + !self.digits.ends_with(&[0]) as usize;
            self.digits.resize(desired, 0);
        }
    }
    impl AddAssign<u8> for BigInt {
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
    impl MulAssign<u8> for BigInt {
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
}
pub mod buffer;
mod custom_keyword {
    #[macro_export]
    macro_rules! custom_keyword {
        ($ident:ident) => {
            #[allow(non_camel_case_types)]
            pub struct $ident {
                pub span: $crate::__private::Span,
            }
            #[allow(dead_code, non_snake_case)]
            pub fn $ident<__S: $crate::__private::IntoSpans<$crate::__private::Span>>(span: __S) -> $ident {
                $ident {
                    span: $crate::__private::IntoSpans::into_spans(span),
                }
            }
            const _: () = {
                impl $crate::__private::Default for $ident {
                    fn default() -> Self {
                        $ident {
                            span: $crate::__private::Span::call_site(),
                        }
                    }
                }
                $crate::impl_parse_for_custom_keyword!($ident);
                $crate::impl_to_tokens_for_custom_keyword!($ident);
                $crate::impl_clone_for_custom_keyword!($ident);
                $crate::impl_extra_traits_for_custom_keyword!($ident);
            };
        };
    }
    #[macro_export]
    macro_rules! impl_parse_for_custom_keyword {
        ($ident:ident) => {
            impl $crate::token::CustomToken for $ident {
                fn peek(cursor: $crate::buffer::Cursor) -> $crate::__private::bool {
                    if let $crate::__private::Some((ident, _rest)) = cursor.ident() {
                        ident == $crate::__private::stringify!($ident)
                    } else {
                        false
                    }
                }
                fn display() -> &'static $crate::__private::str {
                    $crate::__private::concat!("`", $crate::__private::stringify!($ident), "`")
                }
            }
            impl $crate::parse::Parse for $ident {
                fn parse(input: $crate::parse::ParseStream) -> $crate::parse::Result<$ident> {
                    input.step(|cursor| {
                        if let $crate::__private::Some((ident, rest)) = cursor.ident() {
                            if ident == $crate::__private::stringify!($ident) {
                                return $crate::__private::Ok(($ident { span: ident.span() }, rest));
                            }
                        }
                        $crate::__private::Err(cursor.error($crate::__private::concat!(
                            "expected `",
                            $crate::__private::stringify!($ident),
                            "`",
                        )))
                    })
                }
            }
        };
    }
    #[cfg(not(feature = "parsing"))]
    #[macro_export]
    macro_rules! impl_parse_for_custom_keyword {
        ($ident:ident) => {};
    }
    #[macro_export]
    macro_rules! impl_to_tokens_for_custom_keyword {
        ($ident:ident) => {
            impl $crate::__private::ToTokens for $ident {
                fn to_tokens(&self, tokens: &mut $crate::__private::TokenStream2) {
                    let ident = $crate::Ident::new($crate::__private::stringify!($ident), self.span);
                    $crate::__private::TokenStreamExt::append(tokens, ident);
                }
            }
        };
    }
    #[cfg(not(feature = "printing"))]
    #[macro_export]
    macro_rules! impl_to_tokens_for_custom_keyword {
        ($ident:ident) => {};
    }
    #[macro_export]
    macro_rules! impl_clone_for_custom_keyword {
        ($ident:ident) => {
            impl $crate::__private::Copy for $ident {}
            #[allow(clippy::expl_impl_clone_on_copy)]
            impl $crate::__private::Clone for $ident {
                fn clone(&self) -> Self {
                    *self
                }
            }
        };
    }
    #[cfg(not(feature = "clone-impls"))]
    #[macro_export]
    macro_rules! impl_clone_for_custom_keyword {
        ($ident:ident) => {};
    }
    #[macro_export]
    macro_rules! impl_extra_traits_for_custom_keyword {
        ($ident:ident) => {
            impl $crate::__private::Debug for $ident {
                fn fmt(&self, f: &mut $crate::__private::Formatter) -> $crate::__private::fmt::Result {
                    $crate::__private::Formatter::write_str(
                        f,
                        $crate::__private::concat!("Keyword [", $crate::__private::stringify!($ident), "]",),
                    )
                }
            }
            impl $crate::__private::Eq for $ident {}
            impl $crate::__private::PartialEq for $ident {
                fn eq(&self, _other: &Self) -> $crate::__private::bool {
                    true
                }
            }
            impl $crate::__private::Hash for $ident {
                fn hash<__H: $crate::__private::Hasher>(&self, _state: &mut __H) {}
            }
        };
    }
    #[cfg(not(feature = "extra-traits"))]
    #[macro_export]
    macro_rules! impl_extra_traits_for_custom_keyword {
        ($ident:ident) => {};
    }
}
mod custom_punctuation {
    #[macro_export]
    macro_rules! custom_punctuation {
        ($ident:ident, $($tt:tt)+) => {
            pub struct $ident {
                pub spans: $crate::custom_punctuation_repr!($($tt)+),
            }
                    #[allow(dead_code, non_snake_case)]
            pub fn $ident<__S: $crate::__private::IntoSpans<$crate::custom_punctuation_repr!($($tt)+)>>(
                spans: __S,
            ) -> $ident {
                let _validate_len = 0 $(+ $crate::custom_punctuation_len!(strict, $tt))*;
                $ident {
                    spans: $crate::__private::IntoSpans::into_spans(spans)
                }
            }
            const _: () = {
                impl $crate::__private::Default for $ident {
                    fn default() -> Self {
                        $ident($crate::__private::Span::call_site())
                    }
                }
                $crate::impl_parse_for_custom_punctuation!($ident, $($tt)+);
                $crate::impl_to_tokens_for_custom_punctuation!($ident, $($tt)+);
                $crate::impl_clone_for_custom_punctuation!($ident, $($tt)+);
                $crate::impl_extra_traits_for_custom_punctuation!($ident, $($tt)+);
            };
        };
    }
    #[macro_export]
    macro_rules! impl_parse_for_custom_punctuation {
        ($ident:ident, $($tt:tt)+) => {
            impl $crate::token::CustomToken for $ident {
                fn peek(cursor: $crate::buffer::Cursor) -> bool {
                    $crate::__private::peek_punct(cursor, $crate::stringify_punct!($($tt)+))
                }
                fn display() -> &'static $crate::__private::str {
                    $crate::__private::concat!("`", $crate::stringify_punct!($($tt)+), "`")
                }
            }
            impl $crate::parse::Parse for $ident {
                fn parse(input: $crate::parse::ParseStream) -> $crate::parse::Result<$ident> {
                    let spans: $crate::custom_punctuation_repr!($($tt)+) =
                        $crate::__private::parse_punct(input, $crate::stringify_punct!($($tt)+))?;
                    Ok($ident(spans))
                }
            }
        };
    }
    #[cfg(not(feature = "parsing"))]
    #[macro_export]
    macro_rules! impl_parse_for_custom_punctuation {
        ($ident:ident, $($tt:tt)+) => {};
    }
    #[macro_export]
    macro_rules! impl_to_tokens_for_custom_punctuation {
        ($ident:ident, $($tt:tt)+) => {
            impl $crate::__private::ToTokens for $ident {
                fn to_tokens(&self, tokens: &mut $crate::__private::TokenStream2) {
                    $crate::__private::print_punct($crate::stringify_punct!($($tt)+), &self.spans, tokens)
                }
            }
        };
    }
    #[cfg(not(feature = "printing"))]
    #[macro_export]
    macro_rules! impl_to_tokens_for_custom_punctuation {
        ($ident:ident, $($tt:tt)+) => {};
    }
    #[macro_export]
    macro_rules! impl_clone_for_custom_punctuation {
        ($ident:ident, $($tt:tt)+) => {
            impl $crate::__private::Copy for $ident {}
            #[allow(clippy::expl_impl_clone_on_copy)]
            impl $crate::__private::Clone for $ident {
                fn clone(&self) -> Self {
                    *self
                }
            }
        };
    }
    #[cfg(not(feature = "clone-impls"))]
    #[macro_export]
    macro_rules! impl_clone_for_custom_punctuation {
        ($ident:ident, $($tt:tt)+) => {};
    }
    #[macro_export]
    macro_rules! impl_extra_traits_for_custom_punctuation {
        ($ident:ident, $($tt:tt)+) => {
            impl $crate::__private::Debug for $ident {
                fn fmt(&self, f: &mut $crate::__private::Formatter) -> $crate::__private::fmt::Result {
                    $crate::__private::Formatter::write_str(f, $crate::__private::stringify!($ident))
                }
            }
            impl $crate::__private::Eq for $ident {}
            impl $crate::__private::PartialEq for $ident {
                fn eq(&self, _other: &Self) -> $crate::__private::bool {
                    true
                }
            }
            impl $crate::__private::Hash for $ident {
                fn hash<__H: $crate::__private::Hasher>(&self, _state: &mut __H) {}
            }
        };
    }
    #[cfg(not(feature = "extra-traits"))]
    #[macro_export]
    macro_rules! impl_extra_traits_for_custom_punctuation {
        ($ident:ident, $($tt:tt)+) => {};
    }
    #[macro_export]
    macro_rules! custom_punctuation_repr {
        ($($tt:tt)+) => {
            [$crate::__private::Span; 0 $(+ $crate::custom_punctuation_len!(lenient, $tt))+]
        };
    }
    #[macro_export]
    #[rustfmt::skip]
    macro_rules! custom_punctuation_len {
        ($mode:ident, +)     => { 1 };
        ($mode:ident, +=)    => { 2 };
        ($mode:ident, &)     => { 1 };
        ($mode:ident, &&)    => { 2 };
        ($mode:ident, &=)    => { 2 };
        ($mode:ident, @)     => { 1 };
        ($mode:ident, !)     => { 1 };
        ($mode:ident, ^)     => { 1 };
        ($mode:ident, ^=)    => { 2 };
        ($mode:ident, :)     => { 1 };
        ($mode:ident, ::)    => { 2 };
        ($mode:ident, ,)     => { 1 };
        ($mode:ident, /)     => { 1 };
        ($mode:ident, /=)    => { 2 };
        ($mode:ident, .)     => { 1 };
        ($mode:ident, ..)    => { 2 };
        ($mode:ident, ...)   => { 3 };
        ($mode:ident, ..=)   => { 3 };
        ($mode:ident, =)     => { 1 };
        ($mode:ident, ==)    => { 2 };
        ($mode:ident, >=)    => { 2 };
        ($mode:ident, >)     => { 1 };
        ($mode:ident, <=)    => { 2 };
        ($mode:ident, <)     => { 1 };
        ($mode:ident, *=)    => { 2 };
        ($mode:ident, !=)    => { 2 };
        ($mode:ident, |)     => { 1 };
        ($mode:ident, |=)    => { 2 };
        ($mode:ident, ||)    => { 2 };
        ($mode:ident, #)     => { 1 };
        ($mode:ident, ?)     => { 1 };
        ($mode:ident, ->)    => { 2 };
        ($mode:ident, <-)    => { 2 };
        ($mode:ident, %)     => { 1 };
        ($mode:ident, %=)    => { 2 };
        ($mode:ident, =>)    => { 2 };
        ($mode:ident, ;)     => { 1 };
        ($mode:ident, <<)    => { 2 };
        ($mode:ident, <<=)   => { 3 };
        ($mode:ident, >>)    => { 2 };
        ($mode:ident, >>=)   => { 3 };
        ($mode:ident, *)     => { 1 };
        ($mode:ident, -)     => { 1 };
        ($mode:ident, -=)    => { 2 };
        ($mode:ident, ~)     => { 1 };
        (lenient, $tt:tt)    => { 0 };
        (strict, $tt:tt)     => {{ $crate::custom_punctuation_unexpected!($tt); 0 }};
    }
    #[macro_export]
    macro_rules! custom_punctuation_unexpected {
        () => {};
    }
    #[macro_export]
    macro_rules! stringify_punct {
        ($($tt:tt)+) => {
            $crate::__private::concat!($($crate::__private::stringify!($tt)),+)
        };
    }
}
mod data {
    use super::*;
    use crate::punctuated::Punctuated;
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
            pub brace_token: token::Brace,
            pub named: Punctuated<Field, Token![,]>,
        }
    }
    ast_struct! {
        pub struct FieldsUnnamed {
            pub paren_token: token::Paren,
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
    pub(crate) mod parsing {
        use super::*;
        use crate::ext::IdentExt;
        use crate::parse::{Parse, ParseStream, Result};
        impl Parse for Variant {
            fn parse(input: ParseStream) -> Result<Self> {
                let attrs = input.call(Attribute::parse_outer)?;
                let _visibility: Visibility = input.parse()?;
                let ident: Ident = input.parse()?;
                let fields = if input.peek(token::Brace) {
                    Fields::Named(input.parse()?)
                } else if input.peek(token::Paren) {
                    Fields::Unnamed(input.parse()?)
                } else {
                    Fields::Unit
                };
                let discriminant = if input.peek(Token![=]) {
                    let eq_token: Token![=] = input.parse()?;
                    let discriminant: Expr = input.parse()?;
                    Some((eq_token, discriminant))
                } else {
                    None
                };
                Ok(Variant {
                    attrs,
                    ident,
                    fields,
                    discriminant,
                })
            }
        }
        impl Parse for FieldsNamed {
            fn parse(input: ParseStream) -> Result<Self> {
                let content;
                Ok(FieldsNamed {
                    brace_token: braced!(content in input),
                    named: content.parse_terminated(Field::parse_named, Token![,])?,
                })
            }
        }
        impl Parse for FieldsUnnamed {
            fn parse(input: ParseStream) -> Result<Self> {
                let content;
                Ok(FieldsUnnamed {
                    paren_token: parenthesized!(content in input),
                    unnamed: content.parse_terminated(Field::parse_unnamed, Token![,])?,
                })
            }
        }
        impl Field {
            pub fn parse_named(input: ParseStream) -> Result<Self> {
                Ok(Field {
                    attrs: input.call(Attribute::parse_outer)?,
                    vis: input.parse()?,
                    mutability: FieldMutability::None,
                    ident: Some(if input.peek(Token![_]) {
                        input.call(Ident::parse_any)
                    } else {
                        input.parse()
                    }?),
                    colon_token: Some(input.parse()?),
                    ty: input.parse()?,
                })
            }
            pub fn parse_unnamed(input: ParseStream) -> Result<Self> {
                Ok(Field {
                    attrs: input.call(Attribute::parse_outer)?,
                    vis: input.parse()?,
                    mutability: FieldMutability::None,
                    ident: None,
                    colon_token: None,
                    ty: input.parse()?,
                })
            }
        }
    }
    mod printing {
        use super::*;
        use crate::print::TokensOrDefault;
        use proc_macro2::TokenStream;
        use quote::{ToTokens, TokenStreamExt};
        impl ToTokens for Variant {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                tokens.append_all(&self.attrs);
                self.ident.to_tokens(tokens);
                self.fields.to_tokens(tokens);
                if let Some((eq_token, disc)) = &self.discriminant {
                    eq_token.to_tokens(tokens);
                    disc.to_tokens(tokens);
                }
            }
        }
        impl ToTokens for FieldsNamed {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                self.brace_token.surround(tokens, |tokens| {
                    self.named.to_tokens(tokens);
                });
            }
        }
        impl ToTokens for FieldsUnnamed {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                self.paren_token.surround(tokens, |tokens| {
                    self.unnamed.to_tokens(tokens);
                });
            }
        }
        impl ToTokens for Field {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                tokens.append_all(&self.attrs);
                self.vis.to_tokens(tokens);
                if let Some(ident) = &self.ident {
                    ident.to_tokens(tokens);
                    TokensOrDefault(&self.colon_token).to_tokens(tokens);
                }
                self.ty.to_tokens(tokens);
            }
        }
    }
}
pub use crate::data::{Field, Fields, FieldsNamed, FieldsUnnamed, Variant};
mod derive {
    use super::*;
    use crate::punctuated::Punctuated;
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
            pub brace_token: token::Brace,
            pub variants: Punctuated<Variant, Token![,]>,
        }
    }
    ast_struct! {
        pub struct DataUnion {
            pub union_token: Token![union],
            pub fields: FieldsNamed,
        }
    }
    pub(crate) mod parsing {
        use super::*;
        use crate::parse::{Parse, ParseStream, Result};
        impl Parse for DeriveInput {
            fn parse(input: ParseStream) -> Result<Self> {
                let attrs = input.call(Attribute::parse_outer)?;
                let vis = input.parse::<Visibility>()?;
                let lookahead = input.lookahead1();
                if lookahead.peek(Token![struct]) {
                    let struct_token = input.parse::<Token![struct]>()?;
                    let ident = input.parse::<Ident>()?;
                    let generics = input.parse::<Generics>()?;
                    let (where_clause, fields, semi) = data_struct(input)?;
                    Ok(DeriveInput {
                        attrs,
                        vis,
                        ident,
                        generics: Generics {
                            where_clause,
                            ..generics
                        },
                        data: Data::Struct(DataStruct {
                            struct_token,
                            fields,
                            semi_token: semi,
                        }),
                    })
                } else if lookahead.peek(Token![enum]) {
                    let enum_token = input.parse::<Token![enum]>()?;
                    let ident = input.parse::<Ident>()?;
                    let generics = input.parse::<Generics>()?;
                    let (where_clause, brace, variants) = data_enum(input)?;
                    Ok(DeriveInput {
                        attrs,
                        vis,
                        ident,
                        generics: Generics {
                            where_clause,
                            ..generics
                        },
                        data: Data::Enum(DataEnum {
                            enum_token,
                            brace_token: brace,
                            variants,
                        }),
                    })
                } else if lookahead.peek(Token![union]) {
                    let union_token = input.parse::<Token![union]>()?;
                    let ident = input.parse::<Ident>()?;
                    let generics = input.parse::<Generics>()?;
                    let (where_clause, fields) = data_union(input)?;
                    Ok(DeriveInput {
                        attrs,
                        vis,
                        ident,
                        generics: Generics {
                            where_clause,
                            ..generics
                        },
                        data: Data::Union(DataUnion { union_token, fields }),
                    })
                } else {
                    Err(lookahead.error())
                }
            }
        }
        pub(crate) fn data_struct(input: ParseStream) -> Result<(Option<WhereClause>, Fields, Option<Token![;]>)> {
            let mut lookahead = input.lookahead1();
            let mut where_clause = None;
            if lookahead.peek(Token![where]) {
                where_clause = Some(input.parse()?);
                lookahead = input.lookahead1();
            }
            if where_clause.is_none() && lookahead.peek(token::Paren) {
                let fields = input.parse()?;
                lookahead = input.lookahead1();
                if lookahead.peek(Token![where]) {
                    where_clause = Some(input.parse()?);
                    lookahead = input.lookahead1();
                }
                if lookahead.peek(Token![;]) {
                    let semi = input.parse()?;
                    Ok((where_clause, Fields::Unnamed(fields), Some(semi)))
                } else {
                    Err(lookahead.error())
                }
            } else if lookahead.peek(token::Brace) {
                let fields = input.parse()?;
                Ok((where_clause, Fields::Named(fields), None))
            } else if lookahead.peek(Token![;]) {
                let semi = input.parse()?;
                Ok((where_clause, Fields::Unit, Some(semi)))
            } else {
                Err(lookahead.error())
            }
        }
        pub(crate) fn data_enum(
            input: ParseStream,
        ) -> Result<(Option<WhereClause>, token::Brace, Punctuated<Variant, Token![,]>)> {
            let where_clause = input.parse()?;
            let content;
            let brace = braced!(content in input);
            let variants = content.parse_terminated(Variant::parse, Token![,])?;
            Ok((where_clause, brace, variants))
        }
        pub(crate) fn data_union(input: ParseStream) -> Result<(Option<WhereClause>, FieldsNamed)> {
            let where_clause = input.parse()?;
            let fields = input.parse()?;
            Ok((where_clause, fields))
        }
    }
    mod printing {
        use super::*;
        use crate::attr::FilterAttrs;
        use crate::print::TokensOrDefault;
        use proc_macro2::TokenStream;
        use quote::ToTokens;
        impl ToTokens for DeriveInput {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                for attr in self.attrs.outer() {
                    attr.to_tokens(tokens);
                }
                self.vis.to_tokens(tokens);
                match &self.data {
                    Data::Struct(d) => d.struct_token.to_tokens(tokens),
                    Data::Enum(d) => d.enum_token.to_tokens(tokens),
                    Data::Union(d) => d.union_token.to_tokens(tokens),
                }
                self.ident.to_tokens(tokens);
                self.generics.to_tokens(tokens);
                match &self.data {
                    Data::Struct(data) => match &data.fields {
                        Fields::Named(fields) => {
                            self.generics.where_clause.to_tokens(tokens);
                            fields.to_tokens(tokens);
                        },
                        Fields::Unnamed(fields) => {
                            fields.to_tokens(tokens);
                            self.generics.where_clause.to_tokens(tokens);
                            TokensOrDefault(&data.semi_token).to_tokens(tokens);
                        },
                        Fields::Unit => {
                            self.generics.where_clause.to_tokens(tokens);
                            TokensOrDefault(&data.semi_token).to_tokens(tokens);
                        },
                    },
                    Data::Enum(data) => {
                        self.generics.where_clause.to_tokens(tokens);
                        data.brace_token.surround(tokens, |tokens| {
                            data.variants.to_tokens(tokens);
                        });
                    },
                    Data::Union(data) => {
                        self.generics.where_clause.to_tokens(tokens);
                        data.fields.to_tokens(tokens);
                    },
                }
            }
        }
    }
}
pub use crate::derive::{Data, DataEnum, DataStruct, DataUnion, DeriveInput};
mod drops {
    use std::iter;
    use std::mem::ManuallyDrop;
    use std::ops::{Deref, DerefMut};
    use std::option;
    use std::slice;
    #[repr(transparent)]
    pub(crate) struct NoDrop<T: ?Sized>(ManuallyDrop<T>);
    impl<T> NoDrop<T> {
        pub(crate) fn new(value: T) -> Self
        where
            T: TrivialDrop,
        {
            NoDrop(ManuallyDrop::new(value))
        }
    }
    impl<T: ?Sized> Deref for NoDrop<T> {
        type Target = T;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<T: ?Sized> DerefMut for NoDrop<T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }
    pub(crate) trait TrivialDrop {}
    impl<T> TrivialDrop for iter::Empty<T> {}
    impl<'a, T> TrivialDrop for slice::Iter<'a, T> {}
    impl<'a, T> TrivialDrop for slice::IterMut<'a, T> {}
    impl<'a, T> TrivialDrop for option::IntoIter<&'a T> {}
    impl<'a, T> TrivialDrop for option::IntoIter<&'a mut T> {}
    #[test]
    fn test_needs_drop() {
        use std::mem::needs_drop;
        struct NeedsDrop;
        impl Drop for NeedsDrop {
            fn drop(&mut self) {}
        }
        assert!(needs_drop::<NeedsDrop>());
        assert!(!needs_drop::<iter::Empty<NeedsDrop>>());
        assert!(!needs_drop::<slice::Iter<NeedsDrop>>());
        assert!(!needs_drop::<slice::IterMut<NeedsDrop>>());
        assert!(!needs_drop::<option::IntoIter<&NeedsDrop>>());
        assert!(!needs_drop::<option::IntoIter<&mut NeedsDrop>>());
    }
}
mod error {
    use crate::buffer::Cursor;
    use crate::thread::ThreadBound;
    use proc_macro2::{Delimiter, Group, Ident, LexError, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
    use quote::ToTokens;
    use std::fmt::{self, Debug, Display};
    use std::slice;
    use std::vec;
    pub type Result<T> = std::result::Result<T, Error>;
    pub struct Error {
        messages: Vec<ErrorMessage>,
    }
    struct ErrorMessage {
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
        Error: Send + Sync;
    impl Error {
        pub fn new<T: Display>(span: Span, message: T) -> Self {
            return new(span, message.to_string());
            fn new(span: Span, message: String) -> Error {
                Error {
                    messages: vec![ErrorMessage {
                        span: ThreadBound::new(SpanRange { start: span, end: span }),
                        message,
                    }],
                }
            }
        }
        pub fn new_spanned<T: ToTokens, U: Display>(tokens: T, message: U) -> Self {
            return new_spanned(tokens.into_token_stream(), message.to_string());
            fn new_spanned(tokens: TokenStream, message: String) -> Error {
                let mut iter = tokens.into_iter();
                let start = iter.next().map_or_else(Span::call_site, |t| t.span());
                let end = iter.last().map_or(start, |t| t.span());
                Error {
                    messages: vec![ErrorMessage {
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
            self.messages.iter().map(ErrorMessage::to_compile_error).collect()
        }
        pub fn into_compile_error(self) -> TokenStream {
            self.to_compile_error()
        }
        pub fn combine(&mut self, another: Error) {
            self.messages.extend(another.messages);
        }
    }
    impl ErrorMessage {
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
    pub(crate) fn new_at<T: Display>(scope: Span, cursor: Cursor, message: T) -> Error {
        if cursor.eof() {
            Error::new(scope, format!("unexpected end of input, {}", message))
        } else {
            let span = crate::buffer::open_span_of_group(cursor);
            Error::new(span, message)
        }
    }
    pub(crate) fn new2<T: Display>(start: Span, end: Span, message: T) -> Error {
        return new2(start, end, message.to_string());
        fn new2(start: Span, end: Span, message: String) -> Error {
            Error {
                messages: vec![ErrorMessage {
                    span: ThreadBound::new(SpanRange { start, end }),
                    message,
                }],
            }
        }
    }
    impl Debug for Error {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            if self.messages.len() == 1 {
                formatter.debug_tuple("Error").field(&self.messages[0]).finish()
            } else {
                formatter.debug_tuple("Error").field(&self.messages).finish()
            }
        }
    }
    impl Debug for ErrorMessage {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            Debug::fmt(&self.message, formatter)
        }
    }
    impl Display for Error {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str(&self.messages[0].message)
        }
    }
    impl Clone for Error {
        fn clone(&self) -> Self {
            Error {
                messages: self.messages.clone(),
            }
        }
    }
    impl Clone for ErrorMessage {
        fn clone(&self) -> Self {
            ErrorMessage {
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
    impl std::error::Error for Error {}
    impl From<LexError> for Error {
        fn from(err: LexError) -> Self {
            Error::new(err.span(), "lex error")
        }
    }
    impl IntoIterator for Error {
        type Item = Error;
        type IntoIter = IntoIter;
        fn into_iter(self) -> Self::IntoIter {
            IntoIter {
                messages: self.messages.into_iter(),
            }
        }
    }
    pub struct IntoIter {
        messages: vec::IntoIter<ErrorMessage>,
    }
    impl Iterator for IntoIter {
        type Item = Error;
        fn next(&mut self) -> Option<Self::Item> {
            Some(Error {
                messages: vec![self.messages.next()?],
            })
        }
    }
    impl<'a> IntoIterator for &'a Error {
        type Item = Error;
        type IntoIter = Iter<'a>;
        fn into_iter(self) -> Self::IntoIter {
            Iter {
                messages: self.messages.iter(),
            }
        }
    }
    pub struct Iter<'a> {
        messages: slice::Iter<'a, ErrorMessage>,
    }
    impl<'a> Iterator for Iter<'a> {
        type Item = Error;
        fn next(&mut self) -> Option<Self::Item> {
            Some(Error {
                messages: vec![self.messages.next()?.clone()],
            })
        }
    }
    impl Extend<Error> for Error {
        fn extend<T: IntoIterator<Item = Error>>(&mut self, iter: T) {
            for err in iter {
                self.combine(err);
            }
        }
    }
}
pub use crate::error::{Error, Result};
mod expr;
pub use crate::expr::{Arm, FieldValue, Label, RangeLimits};
pub use crate::expr::{
    Expr, ExprArray, ExprAssign, ExprAsync, ExprAwait, ExprBinary, ExprBlock, ExprBreak, ExprCall, ExprCast,
    ExprClosure, ExprConst, ExprContinue, ExprField, ExprForLoop, ExprGroup, ExprIf, ExprIndex, ExprInfer, ExprLet,
    ExprLit, ExprLoop, ExprMacro, ExprMatch, ExprMethodCall, ExprParen, ExprPath, ExprRange, ExprReference, ExprRepeat,
    ExprReturn, ExprStruct, ExprTry, ExprTryBlock, ExprTuple, ExprUnary, ExprUnsafe, ExprWhile, ExprYield, Index,
    Member,
};
pub mod ext {
    use crate::buffer::Cursor;
    use crate::parse::Peek;
    use crate::parse::{ParseStream, Result};
    use crate::sealed::lookahead;
    use crate::token::CustomToken;
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
mod file {
    use super::*;
    ast_struct! {
        pub struct File {
            pub shebang: Option<String>,
            pub attrs: Vec<Attribute>,
            pub items: Vec<Item>,
        }
    }
    pub(crate) mod parsing {
        use super::*;
        use crate::parse::{Parse, ParseStream, Result};
        impl Parse for File {
            fn parse(input: ParseStream) -> Result<Self> {
                Ok(File {
                    shebang: None,
                    attrs: input.call(Attribute::parse_inner)?,
                    items: {
                        let mut items = Vec::new();
                        while !input.is_empty() {
                            items.push(input.parse()?);
                        }
                        items
                    },
                })
            }
        }
    }
    mod printing {
        use super::*;
        use crate::attr::FilterAttrs;
        use proc_macro2::TokenStream;
        use quote::{ToTokens, TokenStreamExt};
        impl ToTokens for File {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                tokens.append_all(self.attrs.inner());
                tokens.append_all(&self.items);
            }
        }
    }
}
pub use crate::file::File;
mod generics;
pub use crate::generics::{
    BoundLifetimes, ConstParam, GenericParam, Generics, LifetimeParam, PredicateLifetime, PredicateType, TraitBound,
    TraitBoundModifier, TypeParam, TypeParamBound, WhereClause, WherePredicate,
};
pub use crate::generics::{ImplGenerics, Turbofish, TypeGenerics};
mod ident {
    use crate::lookahead;
    pub use proc_macro2::Ident;
    #[cfg(not(doc))] // rustdoc bug: https://github.com/rust-lang/rust/issues/105735
    #[allow(non_snake_case)]
    pub fn Ident(marker: lookahead::TokenMarker) -> Ident {
        match marker {}
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
    pub(crate) fn xid_ok(symbol: &str) -> bool {
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
    mod parsing {
        use crate::buffer::Cursor;
        use crate::parse::{Parse, ParseStream, Result};
        use crate::token::Token;
        use proc_macro2::Ident;
        fn accept_as_ident(ident: &Ident) -> bool {
            match ident.to_string().as_str() {
                "_" | "abstract" | "as" | "async" | "await" | "become" | "box" | "break" | "const" | "continue"
                | "crate" | "do" | "dyn" | "else" | "enum" | "extern" | "false" | "final" | "fn" | "for" | "if"
                | "impl" | "in" | "let" | "loop" | "macro" | "match" | "mod" | "move" | "mut" | "override" | "priv"
                | "pub" | "ref" | "return" | "Self" | "self" | "static" | "struct" | "super" | "trait" | "true"
                | "try" | "type" | "typeof" | "unsafe" | "unsized" | "use" | "virtual" | "where" | "while"
                | "yield" => false,
                _ => true,
            }
        }

        impl Parse for Ident {
            fn parse(input: ParseStream) -> Result<Self> {
                input.step(|cursor| {
                    if let Some((ident, rest)) = cursor.ident() {
                        if accept_as_ident(&ident) {
                            Ok((ident, rest))
                        } else {
                            Err(cursor.error(format_args!("expected identifier, found keyword `{}`", ident,)))
                        }
                    } else {
                        Err(cursor.error("expected identifier"))
                    }
                })
            }
        }
        impl Token for Ident {
            fn peek(cursor: Cursor) -> bool {
                if let Some((ident, _rest)) = cursor.ident() {
                    accept_as_ident(&ident)
                } else {
                    false
                }
            }
            fn display() -> &'static str {
                "identifier"
            }
        }
    }
}
pub use crate::ident::Ident;
mod item;
pub use crate::item::{
    FnArg, ForeignItem, ForeignItemFn, ForeignItemMacro, ForeignItemStatic, ForeignItemType, ImplItem, ImplItemConst,
    ImplItemFn, ImplItemMacro, ImplItemType, ImplRestriction, Item, ItemConst, ItemEnum, ItemExternCrate, ItemFn,
    ItemForeignMod, ItemImpl, ItemMacro, ItemMod, ItemStatic, ItemStruct, ItemTrait, ItemTraitAlias, ItemType,
    ItemUnion, ItemUse, Receiver, Signature, StaticMutability, TraitItem, TraitItemConst, TraitItemFn, TraitItemMacro,
    TraitItemType, UseGlob, UseGroup, UseName, UsePath, UseRename, UseTree, Variadic,
};
mod lifetime {
    use crate::lookahead;
    use proc_macro2::{Ident, Span};
    use std::cmp::Ordering;
    use std::fmt::{self, Display};
    use std::hash::{Hash, Hasher};
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
    pub(crate) mod parsing {
        use super::*;
        use crate::parse::{Parse, ParseStream, Result};

        impl Parse for Lifetime {
            fn parse(input: ParseStream) -> Result<Self> {
                input.step(|cursor| cursor.lifetime().ok_or_else(|| cursor.error("expected lifetime")))
            }
        }
    }
    mod printing {
        use super::*;
        use proc_macro2::{Punct, Spacing, TokenStream};
        use quote::{ToTokens, TokenStreamExt};
        impl ToTokens for Lifetime {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                let mut apostrophe = Punct::new('\'', Spacing::Joint);
                apostrophe.set_span(self.apostrophe);
                tokens.append(apostrophe);
                self.ident.to_tokens(tokens);
            }
        }
    }
}
pub use crate::lifetime::Lifetime;
mod lit;
pub use crate::lit::{Lit, LitBool, LitByte, LitByteStr, LitChar, LitFloat, LitInt, LitStr, StrStyle};
mod lookahead {
    use crate::buffer::Cursor;
    use crate::error::{self, Error};
    use crate::sealed::lookahead::Sealed;
    use crate::span::IntoSpans;
    use crate::token::Token;
    use proc_macro2::{Delimiter, Span};
    use std::cell::RefCell;
    pub struct Lookahead1<'a> {
        scope: Span,
        cursor: Cursor<'a>,
        comparisons: RefCell<Vec<&'static str>>,
    }
    pub(crate) fn new(scope: Span, cursor: Cursor) -> Lookahead1 {
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
    impl<'a> Lookahead1<'a> {
        pub fn peek<T: Peek>(&self, token: T) -> bool {
            let _ = token;
            peek_impl(self, T::Token::peek, T::Token::display)
        }
        pub fn error(self) -> Error {
            let comparisons = self.comparisons.borrow();
            match comparisons.len() {
                0 => {
                    if self.cursor.eof() {
                        Error::new(self.scope, "unexpected end of input")
                    } else {
                        Error::new(self.cursor.span(), "unexpected token")
                    }
                },
                1 => {
                    let message = format!("expected {}", comparisons[0]);
                    error::new_at(self.scope, self.cursor, message)
                },
                2 => {
                    let message = format!("expected {} or {}", comparisons[0], comparisons[1]);
                    error::new_at(self.scope, self.cursor, message)
                },
                _ => {
                    let join = comparisons.join(", ");
                    let message = format!("expected one of: {}", join);
                    error::new_at(self.scope, self.cursor, message)
                },
            }
        }
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
    pub(crate) fn is_delimiter(cursor: Cursor, delimiter: Delimiter) -> bool {
        cursor.group(delimiter).is_some()
    }
    impl<F: Copy + FnOnce(TokenMarker) -> T, T: Token> Sealed for F {}
}
mod mac {
    use super::*;
    use crate::parse::{Parse, ParseStream, Parser, Result};
    use crate::token::{Brace, Bracket, Paren};
    use proc_macro2::extra::DelimSpan;
    use proc_macro2::Delimiter;
    use proc_macro2::TokenStream;
    use proc_macro2::TokenTree;
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
    pub(crate) fn parse_delimiter(input: ParseStream) -> Result<(MacroDelimiter, TokenStream)> {
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
    pub(crate) mod parsing {
        use super::*;
        use crate::parse::{Parse, ParseStream, Result};

        impl Parse for Macro {
            fn parse(input: ParseStream) -> Result<Self> {
                let tokens;
                Ok(Macro {
                    path: input.call(Path::parse_mod_style)?,
                    bang_token: input.parse()?,
                    delimiter: {
                        let (delimiter, content) = parse_delimiter(input)?;
                        tokens = content;
                        delimiter
                    },
                    tokens,
                })
            }
        }
    }
    mod printing {
        use super::*;
        use proc_macro2::TokenStream;
        use quote::ToTokens;
        impl MacroDelimiter {
            pub(crate) fn surround(&self, tokens: &mut TokenStream, inner: TokenStream) {
                let (delim, span) = match self {
                    MacroDelimiter::Paren(paren) => (Delimiter::Parenthesis, paren.span),
                    MacroDelimiter::Brace(brace) => (Delimiter::Brace, brace.span),
                    MacroDelimiter::Bracket(bracket) => (Delimiter::Bracket, bracket.span),
                };
                token::printing::delim(delim, span.join(), tokens, inner);
            }
        }
        impl ToTokens for Macro {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                self.path.to_tokens(tokens);
                self.bang_token.to_tokens(tokens);
                self.delimiter.surround(tokens, self.tokens.clone());
            }
        }
    }
}
pub use crate::mac::{Macro, MacroDelimiter};
pub mod meta {
    use crate::ext::IdentExt;
    use crate::lit::Lit;
    use crate::parse::{Error, ParseStream, Parser, Result};
    use crate::path::{Path, PathSegment};
    use crate::punctuated::Punctuated;
    use proc_macro2::Ident;
    use std::fmt::Display;
    pub fn parser(logic: impl FnMut(ParseNestedMeta) -> Result<()>) -> impl Parser<Output = ()> {
        |input: ParseStream| {
            if input.is_empty() {
                Ok(())
            } else {
                parse_nested_meta(input, logic)
            }
        }
    }
    #[non_exhaustive]
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
        pub fn error(&self, msg: impl Display) -> Error {
            let start_span = self.path.segments[0].ident.span();
            let end_span = self.input.cursor().prev_span();
            crate::error::new2(start_span, end_span, msg)
        }
    }
    pub(crate) fn parse_nested_meta(
        input: ParseStream,
        mut logic: impl FnMut(ParseNestedMeta) -> Result<()>,
    ) -> Result<()> {
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
mod op {
    ast_enum! {
        #[non_exhaustive]
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
        #[non_exhaustive]
        pub enum UnOp {
            Deref(Token![*]),
            Not(Token![!]),
            Neg(Token![-]),
        }
    }
    pub(crate) mod parsing {
        use super::*;
        use crate::parse::{Parse, ParseStream, Result};
        fn parse_binop(input: ParseStream) -> Result<BinOp> {
            if input.peek(Token![&&]) {
                input.parse().map(BinOp::And)
            } else if input.peek(Token![||]) {
                input.parse().map(BinOp::Or)
            } else if input.peek(Token![<<]) {
                input.parse().map(BinOp::Shl)
            } else if input.peek(Token![>>]) {
                input.parse().map(BinOp::Shr)
            } else if input.peek(Token![==]) {
                input.parse().map(BinOp::Eq)
            } else if input.peek(Token![<=]) {
                input.parse().map(BinOp::Le)
            } else if input.peek(Token![!=]) {
                input.parse().map(BinOp::Ne)
            } else if input.peek(Token![>=]) {
                input.parse().map(BinOp::Ge)
            } else if input.peek(Token![+]) {
                input.parse().map(BinOp::Add)
            } else if input.peek(Token![-]) {
                input.parse().map(BinOp::Sub)
            } else if input.peek(Token![*]) {
                input.parse().map(BinOp::Mul)
            } else if input.peek(Token![/]) {
                input.parse().map(BinOp::Div)
            } else if input.peek(Token![%]) {
                input.parse().map(BinOp::Rem)
            } else if input.peek(Token![^]) {
                input.parse().map(BinOp::BitXor)
            } else if input.peek(Token![&]) {
                input.parse().map(BinOp::BitAnd)
            } else if input.peek(Token![|]) {
                input.parse().map(BinOp::BitOr)
            } else if input.peek(Token![<]) {
                input.parse().map(BinOp::Lt)
            } else if input.peek(Token![>]) {
                input.parse().map(BinOp::Gt)
            } else {
                Err(input.error("expected binary operator"))
            }
        }

        impl Parse for BinOp {
            #[cfg(not(feature = "full"))]
            fn parse(input: ParseStream) -> Result<Self> {
                parse_binop(input)
            }
            fn parse(input: ParseStream) -> Result<Self> {
                if input.peek(Token![+=]) {
                    input.parse().map(BinOp::AddAssign)
                } else if input.peek(Token![-=]) {
                    input.parse().map(BinOp::SubAssign)
                } else if input.peek(Token![*=]) {
                    input.parse().map(BinOp::MulAssign)
                } else if input.peek(Token![/=]) {
                    input.parse().map(BinOp::DivAssign)
                } else if input.peek(Token![%=]) {
                    input.parse().map(BinOp::RemAssign)
                } else if input.peek(Token![^=]) {
                    input.parse().map(BinOp::BitXorAssign)
                } else if input.peek(Token![&=]) {
                    input.parse().map(BinOp::BitAndAssign)
                } else if input.peek(Token![|=]) {
                    input.parse().map(BinOp::BitOrAssign)
                } else if input.peek(Token![<<=]) {
                    input.parse().map(BinOp::ShlAssign)
                } else if input.peek(Token![>>=]) {
                    input.parse().map(BinOp::ShrAssign)
                } else {
                    parse_binop(input)
                }
            }
        }

        impl Parse for UnOp {
            fn parse(input: ParseStream) -> Result<Self> {
                let lookahead = input.lookahead1();
                if lookahead.peek(Token![*]) {
                    input.parse().map(UnOp::Deref)
                } else if lookahead.peek(Token![!]) {
                    input.parse().map(UnOp::Not)
                } else if lookahead.peek(Token![-]) {
                    input.parse().map(UnOp::Neg)
                } else {
                    Err(lookahead.error())
                }
            }
        }
    }
    mod printing {
        use super::*;
        use proc_macro2::TokenStream;
        use quote::ToTokens;
        impl ToTokens for BinOp {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                match self {
                    BinOp::Add(t) => t.to_tokens(tokens),
                    BinOp::Sub(t) => t.to_tokens(tokens),
                    BinOp::Mul(t) => t.to_tokens(tokens),
                    BinOp::Div(t) => t.to_tokens(tokens),
                    BinOp::Rem(t) => t.to_tokens(tokens),
                    BinOp::And(t) => t.to_tokens(tokens),
                    BinOp::Or(t) => t.to_tokens(tokens),
                    BinOp::BitXor(t) => t.to_tokens(tokens),
                    BinOp::BitAnd(t) => t.to_tokens(tokens),
                    BinOp::BitOr(t) => t.to_tokens(tokens),
                    BinOp::Shl(t) => t.to_tokens(tokens),
                    BinOp::Shr(t) => t.to_tokens(tokens),
                    BinOp::Eq(t) => t.to_tokens(tokens),
                    BinOp::Lt(t) => t.to_tokens(tokens),
                    BinOp::Le(t) => t.to_tokens(tokens),
                    BinOp::Ne(t) => t.to_tokens(tokens),
                    BinOp::Ge(t) => t.to_tokens(tokens),
                    BinOp::Gt(t) => t.to_tokens(tokens),
                    BinOp::AddAssign(t) => t.to_tokens(tokens),
                    BinOp::SubAssign(t) => t.to_tokens(tokens),
                    BinOp::MulAssign(t) => t.to_tokens(tokens),
                    BinOp::DivAssign(t) => t.to_tokens(tokens),
                    BinOp::RemAssign(t) => t.to_tokens(tokens),
                    BinOp::BitXorAssign(t) => t.to_tokens(tokens),
                    BinOp::BitAndAssign(t) => t.to_tokens(tokens),
                    BinOp::BitOrAssign(t) => t.to_tokens(tokens),
                    BinOp::ShlAssign(t) => t.to_tokens(tokens),
                    BinOp::ShrAssign(t) => t.to_tokens(tokens),
                }
            }
        }
        impl ToTokens for UnOp {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                match self {
                    UnOp::Deref(t) => t.to_tokens(tokens),
                    UnOp::Not(t) => t.to_tokens(tokens),
                    UnOp::Neg(t) => t.to_tokens(tokens),
                }
            }
        }
    }
}
pub use crate::op::{BinOp, UnOp};
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
                attr::parsing::single_parse_inner(input)
            } else {
                attr::parsing::single_parse_outer(input)
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
mod pat;
pub use crate::expr::{
    ExprConst as PatConst, ExprLit as PatLit, ExprMacro as PatMacro, ExprPath as PatPath, ExprRange as PatRange,
};
pub use crate::pat::{
    FieldPat, Pat, PatIdent, PatOr, PatParen, PatReference, PatRest, PatSlice, PatStruct, PatTuple, PatTupleStruct,
    PatType, PatWild,
};
mod path;
pub use crate::path::{
    AngleBracketedGenericArguments, AssocConst, AssocType, Constraint, GenericArgument, ParenthesizedGenericArguments,
    Path, PathArguments, PathSegment, QSelf,
};
mod print {
    use proc_macro2::TokenStream;
    use quote::ToTokens;
    pub(crate) struct TokensOrDefault<'a, T: 'a>(pub &'a Option<T>);
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
}
pub mod punctuated;
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
            pub paren_token: token::Paren,
            pub in_token: Option<Token![in]>,
            pub path: Box<Path>,
        }
    }
    ast_enum! {
        #[non_exhaustive]
        pub enum FieldMutability {
            None,
        }
    }
    pub(crate) mod parsing {
        use super::*;
        use crate::ext::IdentExt;
        use crate::parse::discouraged::Speculative;
        use crate::parse::{Parse, ParseStream, Result};

        impl Parse for Visibility {
            fn parse(input: ParseStream) -> Result<Self> {
                if input.peek(token::Group) {
                    let ahead = input.fork();
                    let group = crate::group::parse_group(&ahead)?;
                    if group.content.is_empty() {
                        input.advance_to(&ahead);
                        return Ok(Visibility::Inherited);
                    }
                }
                if input.peek(Token![pub]) {
                    Self::parse_pub(input)
                } else {
                    Ok(Visibility::Inherited)
                }
            }
        }
        impl Visibility {
            fn parse_pub(input: ParseStream) -> Result<Self> {
                let pub_token = input.parse::<Token![pub]>()?;
                if input.peek(token::Paren) {
                    let ahead = input.fork();
                    let content;
                    let paren_token = parenthesized!(content in ahead);
                    if content.peek(Token![crate]) || content.peek(Token![self]) || content.peek(Token![super]) {
                        let path = content.call(Ident::parse_any)?;
                        if content.is_empty() {
                            input.advance_to(&ahead);
                            return Ok(Visibility::Restricted(VisRestricted {
                                pub_token,
                                paren_token,
                                in_token: None,
                                path: Box::new(Path::from(path)),
                            }));
                        }
                    } else if content.peek(Token![in]) {
                        let in_token: Token![in] = content.parse()?;
                        let path = content.call(Path::parse_mod_style)?;
                        input.advance_to(&ahead);
                        return Ok(Visibility::Restricted(VisRestricted {
                            pub_token,
                            paren_token,
                            in_token: Some(in_token),
                            path: Box::new(path),
                        }));
                    }
                }
                Ok(Visibility::Public(pub_token))
            }
            pub(crate) fn is_some(&self) -> bool {
                match self {
                    Visibility::Inherited => false,
                    _ => true,
                }
            }
        }
    }
    mod printing {
        use super::*;
        use proc_macro2::TokenStream;
        use quote::ToTokens;
        impl ToTokens for Visibility {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                match self {
                    Visibility::Public(pub_token) => pub_token.to_tokens(tokens),
                    Visibility::Restricted(vis_restricted) => vis_restricted.to_tokens(tokens),
                    Visibility::Inherited => {},
                }
            }
        }
        impl ToTokens for VisRestricted {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                self.pub_token.to_tokens(tokens);
                self.paren_token.surround(tokens, |tokens| {
                    self.in_token.to_tokens(tokens);
                    self.path.to_tokens(tokens);
                });
            }
        }
    }
}
pub use crate::restriction::{FieldMutability, VisRestricted, Visibility};
mod sealed {
    pub(crate) mod lookahead {
        pub trait Sealed: Copy {}
    }
}
mod span {
    use proc_macro2::extra::DelimSpan;
    use proc_macro2::{Delimiter, Group, Span, TokenStream};
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
}
pub mod spanned {
    use proc_macro2::Span;
    use quote::spanned::Spanned as ToTokens;
    pub trait Spanned: private::Sealed {
        fn span(&self) -> Span;
    }
    impl<T: ?Sized + ToTokens> Spanned for T {
        fn span(&self) -> Span {
            self.__span()
        }
    }
    mod private {
        use super::*;
        pub trait Sealed {}
        impl<T: ?Sized + ToTokens> Sealed for T {}
    }
}
mod stmt;
pub use crate::stmt::{Block, Local, LocalInit, Stmt, StmtMacro};
mod thread {
    use std::fmt::{self, Debug};
    use std::thread::{self, ThreadId};
    pub(crate) struct ThreadBound<T> {
        value: T,
        thread_id: ThreadId,
    }
    unsafe impl<T> Sync for ThreadBound<T> {}
    unsafe impl<T: Copy> Send for ThreadBound<T> {}
    impl<T> ThreadBound<T> {
        pub(crate) fn new(value: T) -> Self {
            ThreadBound {
                value,
                thread_id: thread::current().id(),
            }
        }
        pub(crate) fn get(&self) -> Option<&T> {
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
}
mod tt {
    use proc_macro2::{Delimiter, TokenStream, TokenTree};
    use std::hash::{Hash, Hasher};
    pub(crate) struct TokenTreeHelper<'a>(pub &'a TokenTree);
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
    pub(crate) struct TokenStreamHelper<'a>(pub &'a TokenStream);
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
}
mod ty;
pub use crate::ty::{
    Abi, BareFnArg, BareVariadic, ReturnType, Type, TypeArray, TypeBareFn, TypeGroup, TypeImplTrait, TypeInfer,
    TypeMacro, TypeNever, TypeParen, TypePath, TypePtr, TypeReference, TypeSlice, TypeTraitObject, TypeTuple,
};
mod verbatim {
    use crate::parse::ParseStream;
    use proc_macro2::{Delimiter, TokenStream};
    use std::cmp::Ordering;
    use std::iter;
    pub(crate) fn between<'a>(begin: ParseStream<'a>, end: ParseStream<'a>) -> TokenStream {
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
}
mod whitespace {
    pub(crate) fn skip(mut s: &str) -> &str {
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
                } else if s.starts_with("/*")
                    && (!s.starts_with("/**") || s.starts_with("/***"))
                    && !s.starts_with("/*!")
                {
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
    fn is_whitespace(ch: char) -> bool {
        ch.is_whitespace() || ch == '\u{200e}' || ch == '\u{200f}'
    }
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
        pub(crate) mod fold {
            use crate::punctuated::{Pair, Punctuated};
            pub(crate) trait FoldHelper {
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
    pub use crate::group::{parse_braces, parse_brackets, parse_parens};
    pub use crate::parse_quote::parse as parse_quote;
    pub use crate::span::IntoSpans;
    pub use crate::token::parsing::{peek_punct, punct as parse_punct};
    pub use crate::token::printing::punct as print_punct;
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
    #[allow(non_camel_case_types)]
    pub type bool = help::Bool;
    #[allow(non_camel_case_types)]
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
        let rest = whitespace::skip(&content[2..]);
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
