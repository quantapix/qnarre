use crate::lookahead;
use crate::parse::{Parse, Parser};
use crate::{Error, Result};
use proc_macro2::{Ident, Literal, Span};
use proc_macro2::{TokenStream, TokenTree};
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};
use std::str::{self, FromStr};
ast_enum_of_structs! {
    #[non_exhaustive]
    pub enum Lit {
        Str(LitStr),
        ByteStr(LitByteStr),
        Byte(LitByte),
        Char(LitChar),
        Int(LitInt),
        Float(LitFloat),
        Bool(LitBool),
        Verbatim(Literal),
    }
}
ast_struct! {
    pub struct LitStr {
        repr: Box<LitRepr>,
    }
}
ast_struct! {
    pub struct LitByteStr {
        repr: Box<LitRepr>,
    }
}
ast_struct! {
    pub struct LitByte {
        repr: Box<LitRepr>,
    }
}
ast_struct! {
    pub struct LitChar {
        repr: Box<LitRepr>,
    }
}
struct LitRepr {
    token: Literal,
    suffix: Box<str>,
}
ast_struct! {
    pub struct LitInt {
        repr: Box<LitIntRepr>,
    }
}
struct LitIntRepr {
    token: Literal,
    digits: Box<str>,
    suffix: Box<str>,
}
ast_struct! {
    pub struct LitFloat {
        repr: Box<LitFloatRepr>,
    }
}
struct LitFloatRepr {
    token: Literal,
    digits: Box<str>,
    suffix: Box<str>,
}
ast_struct! {
    pub struct LitBool {
        pub value: bool,
        pub span: Span,
    }
}
impl LitStr {
    pub fn new(value: &str, span: Span) -> Self {
        let mut token = Literal::string(value);
        token.set_span(span);
        LitStr {
            repr: Box::new(LitRepr {
                token,
                suffix: Box::<str>::default(),
            }),
        }
    }
    pub fn value(&self) -> String {
        let repr = self.repr.token.to_string();
        let (value, _suffix) = value::parse_lit_str(&repr);
        String::from(value)
    }

    pub fn parse<T: Parse>(&self) -> Result<T> {
        self.parse_with(T::parse)
    }

    pub fn parse_with<F: Parser>(&self, parser: F) -> Result<F::Output> {
        use proc_macro2::Group;
        fn respan_token_stream(stream: TokenStream, span: Span) -> TokenStream {
            stream.into_iter().map(|token| respan_token_tree(token, span)).collect()
        }
        fn respan_token_tree(mut token: TokenTree, span: Span) -> TokenTree {
            match &mut token {
                TokenTree::Group(g) => {
                    let stream = respan_token_stream(g.stream(), span);
                    *g = Group::new(g.delimiter(), stream);
                    g.set_span(span);
                },
                other => other.set_span(span),
            }
            token
        }
        let mut tokens = TokenStream::from_str(&self.value())?;
        tokens = respan_token_stream(tokens, self.span());
        parser.parse2(tokens)
    }
    pub fn span(&self) -> Span {
        self.repr.token.span()
    }
    pub fn set_span(&mut self, span: Span) {
        self.repr.token.set_span(span);
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suffix
    }
    pub fn token(&self) -> Literal {
        self.repr.token.clone()
    }
}
impl LitByteStr {
    pub fn new(value: &[u8], span: Span) -> Self {
        let mut token = Literal::byte_string(value);
        token.set_span(span);
        LitByteStr {
            repr: Box::new(LitRepr {
                token,
                suffix: Box::<str>::default(),
            }),
        }
    }
    pub fn value(&self) -> Vec<u8> {
        let repr = self.repr.token.to_string();
        let (value, _suffix) = value::parse_lit_byte_str(&repr);
        value
    }
    pub fn span(&self) -> Span {
        self.repr.token.span()
    }
    pub fn set_span(&mut self, span: Span) {
        self.repr.token.set_span(span);
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suffix
    }
    pub fn token(&self) -> Literal {
        self.repr.token.clone()
    }
}
impl LitByte {
    pub fn new(value: u8, span: Span) -> Self {
        let mut token = Literal::u8_suffixed(value);
        token.set_span(span);
        LitByte {
            repr: Box::new(LitRepr {
                token,
                suffix: Box::<str>::default(),
            }),
        }
    }
    pub fn value(&self) -> u8 {
        let repr = self.repr.token.to_string();
        let (value, _suffix) = value::parse_lit_byte(&repr);
        value
    }
    pub fn span(&self) -> Span {
        self.repr.token.span()
    }
    pub fn set_span(&mut self, span: Span) {
        self.repr.token.set_span(span);
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suffix
    }
    pub fn token(&self) -> Literal {
        self.repr.token.clone()
    }
}
impl LitChar {
    pub fn new(value: char, span: Span) -> Self {
        let mut token = Literal::character(value);
        token.set_span(span);
        LitChar {
            repr: Box::new(LitRepr {
                token,
                suffix: Box::<str>::default(),
            }),
        }
    }
    pub fn value(&self) -> char {
        let repr = self.repr.token.to_string();
        let (value, _suffix) = value::parse_lit_char(&repr);
        value
    }
    pub fn span(&self) -> Span {
        self.repr.token.span()
    }
    pub fn set_span(&mut self, span: Span) {
        self.repr.token.set_span(span);
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suffix
    }
    pub fn token(&self) -> Literal {
        self.repr.token.clone()
    }
}
impl LitInt {
    pub fn new(repr: &str, span: Span) -> Self {
        let (digits, suffix) = match value::parse_lit_int(repr) {
            Some(parse) => parse,
            None => panic!("Not an integer literal: `{}`", repr),
        };
        let mut token: Literal = repr.parse().unwrap();
        token.set_span(span);
        LitInt {
            repr: Box::new(LitIntRepr { token, digits, suffix }),
        }
    }
    pub fn base10_digits(&self) -> &str {
        &self.repr.digits
    }
    pub fn base10_parse<N>(&self) -> Result<N>
    where
        N: FromStr,
        N::Err: Display,
    {
        self.base10_digits().parse().map_err(|err| Error::new(self.span(), err))
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suffix
    }
    pub fn span(&self) -> Span {
        self.repr.token.span()
    }
    pub fn set_span(&mut self, span: Span) {
        self.repr.token.set_span(span);
    }
    pub fn token(&self) -> Literal {
        self.repr.token.clone()
    }
}
impl From<Literal> for LitInt {
    fn from(token: Literal) -> Self {
        let repr = token.to_string();
        if let Some((digits, suffix)) = value::parse_lit_int(&repr) {
            LitInt {
                repr: Box::new(LitIntRepr { token, digits, suffix }),
            }
        } else {
            panic!("Not an integer literal: `{}`", repr);
        }
    }
}
impl Display for LitInt {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.repr.token.fmt(formatter)
    }
}
impl LitFloat {
    pub fn new(repr: &str, span: Span) -> Self {
        let (digits, suffix) = match value::parse_lit_float(repr) {
            Some(parse) => parse,
            None => panic!("Not a float literal: `{}`", repr),
        };
        let mut token: Literal = repr.parse().unwrap();
        token.set_span(span);
        LitFloat {
            repr: Box::new(LitFloatRepr { token, digits, suffix }),
        }
    }
    pub fn base10_digits(&self) -> &str {
        &self.repr.digits
    }
    pub fn base10_parse<N>(&self) -> Result<N>
    where
        N: FromStr,
        N::Err: Display,
    {
        self.base10_digits().parse().map_err(|err| Error::new(self.span(), err))
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suffix
    }
    pub fn span(&self) -> Span {
        self.repr.token.span()
    }
    pub fn set_span(&mut self, span: Span) {
        self.repr.token.set_span(span);
    }
    pub fn token(&self) -> Literal {
        self.repr.token.clone()
    }
}
impl From<Literal> for LitFloat {
    fn from(token: Literal) -> Self {
        let repr = token.to_string();
        if let Some((digits, suffix)) = value::parse_lit_float(&repr) {
            LitFloat {
                repr: Box::new(LitFloatRepr { token, digits, suffix }),
            }
        } else {
            panic!("Not a float literal: `{}`", repr);
        }
    }
}
impl Display for LitFloat {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.repr.token.fmt(formatter)
    }
}
impl LitBool {
    pub fn new(value: bool, span: Span) -> Self {
        LitBool { value, span }
    }
    pub fn value(&self) -> bool {
        self.value
    }
    pub fn span(&self) -> Span {
        self.span
    }
    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }
    pub fn token(&self) -> Ident {
        let s = if self.value { "true" } else { "false" };
        Ident::new(s, self.span)
    }
}
mod debug_impls {
    use super::*;
    use std::fmt::{self, Debug};
    #[cfg_attr(doc_cfg, doc(cfg(feature = "extra-traits")))]
    impl Debug for LitStr {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            impl LitStr {
                pub(crate) fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                    formatter
                        .debug_struct(name)
                        .field("token", &format_args!("{}", self.repr.token))
                        .finish()
                }
            }
            self.debug(formatter, "LitStr")
        }
    }
    #[cfg_attr(doc_cfg, doc(cfg(feature = "extra-traits")))]
    impl Debug for LitByteStr {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            impl LitByteStr {
                pub(crate) fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                    formatter
                        .debug_struct(name)
                        .field("token", &format_args!("{}", self.repr.token))
                        .finish()
                }
            }
            self.debug(formatter, "LitByteStr")
        }
    }
    #[cfg_attr(doc_cfg, doc(cfg(feature = "extra-traits")))]
    impl Debug for LitByte {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            impl LitByte {
                pub(crate) fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                    formatter
                        .debug_struct(name)
                        .field("token", &format_args!("{}", self.repr.token))
                        .finish()
                }
            }
            self.debug(formatter, "LitByte")
        }
    }
    #[cfg_attr(doc_cfg, doc(cfg(feature = "extra-traits")))]
    impl Debug for LitChar {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            impl LitChar {
                pub(crate) fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                    formatter
                        .debug_struct(name)
                        .field("token", &format_args!("{}", self.repr.token))
                        .finish()
                }
            }
            self.debug(formatter, "LitChar")
        }
    }
    #[cfg_attr(doc_cfg, doc(cfg(feature = "extra-traits")))]
    impl Debug for LitInt {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            impl LitInt {
                pub(crate) fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                    formatter
                        .debug_struct(name)
                        .field("token", &format_args!("{}", self.repr.token))
                        .finish()
                }
            }
            self.debug(formatter, "LitInt")
        }
    }
    #[cfg_attr(doc_cfg, doc(cfg(feature = "extra-traits")))]
    impl Debug for LitFloat {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            impl LitFloat {
                pub(crate) fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                    formatter
                        .debug_struct(name)
                        .field("token", &format_args!("{}", self.repr.token))
                        .finish()
                }
            }
            self.debug(formatter, "LitFloat")
        }
    }
    #[cfg_attr(doc_cfg, doc(cfg(feature = "extra-traits")))]
    impl Debug for LitBool {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            impl LitBool {
                pub(crate) fn debug(&self, formatter: &mut fmt::Formatter, name: &str) -> fmt::Result {
                    formatter.debug_struct(name).field("value", &self.value).finish()
                }
            }
            self.debug(formatter, "LitBool")
        }
    }
}
impl Clone for LitRepr {
    fn clone(&self) -> Self {
        LitRepr {
            token: self.token.clone(),
            suffix: self.suffix.clone(),
        }
    }
}
impl Clone for LitIntRepr {
    fn clone(&self) -> Self {
        LitIntRepr {
            token: self.token.clone(),
            digits: self.digits.clone(),
            suffix: self.suffix.clone(),
        }
    }
}
impl Clone for LitFloatRepr {
    fn clone(&self) -> Self {
        LitFloatRepr {
            token: self.token.clone(),
            digits: self.digits.clone(),
            suffix: self.suffix.clone(),
        }
    }
}
macro_rules! lit_extra_traits {
    ($ty:ident) => {
        #[cfg_attr(doc_cfg, doc(cfg(feature = "clone-impls")))]
        impl Clone for $ty {
            fn clone(&self) -> Self {
                $ty {
                    repr: self.repr.clone(),
                }
            }
        }
        #[cfg_attr(doc_cfg, doc(cfg(feature = "extra-traits")))]
        impl PartialEq for $ty {
            fn eq(&self, other: &Self) -> bool {
                self.repr.token.to_string() == other.repr.token.to_string()
            }
        }
        #[cfg_attr(doc_cfg, doc(cfg(feature = "extra-traits")))]
        impl Hash for $ty {
            fn hash<H>(&self, state: &mut H)
            where
                H: Hasher,
            {
                self.repr.token.to_string().hash(state);
            }
        }
        #[allow(non_snake_case)]
        pub fn $ty(marker: lookahead::TokenMarker) -> $ty {
            match marker {}
        }
    };
}
lit_extra_traits!(LitStr);
lit_extra_traits!(LitByteStr);
lit_extra_traits!(LitByte);
lit_extra_traits!(LitChar);
lit_extra_traits!(LitInt);
lit_extra_traits!(LitFloat);
#[allow(non_snake_case)]
pub fn LitBool(marker: lookahead::TokenMarker) -> LitBool {
    match marker {}
}
ast_enum! {
    pub enum StrStyle #no_visit {
        Cooked,
        Raw(usize),
    }
}
#[allow(non_snake_case)]
pub fn Lit(marker: lookahead::TokenMarker) -> Lit {
    match marker {}
}
pub(crate) mod parsing {
    use super::*;
    use crate::buffer::Cursor;
    use crate::parse::{Parse, ParseStream, Result};
    use proc_macro2::Punct;

    impl Parse for Lit {
        fn parse(input: ParseStream) -> Result<Self> {
            input.step(|cursor| {
                if let Some((lit, rest)) = cursor.literal() {
                    return Ok((Lit::new(lit), rest));
                }
                if let Some((ident, rest)) = cursor.ident() {
                    let value = ident == "true";
                    if value || ident == "false" {
                        let lit_bool = LitBool {
                            value,
                            span: ident.span(),
                        };
                        return Ok((Lit::Bool(lit_bool), rest));
                    }
                }
                if let Some((punct, rest)) = cursor.punct() {
                    if punct.as_char() == '-' {
                        if let Some((lit, rest)) = parse_negative_lit(punct, rest) {
                            return Ok((lit, rest));
                        }
                    }
                }
                Err(cursor.error("expected literal"))
            })
        }
    }
    fn parse_negative_lit(neg: Punct, cursor: Cursor) -> Option<(Lit, Cursor)> {
        let (lit, rest) = cursor.literal()?;
        let mut span = neg.span();
        span = span.join(lit.span()).unwrap_or(span);
        let mut repr = lit.to_string();
        repr.insert(0, '-');
        if let Some((digits, suffix)) = value::parse_lit_int(&repr) {
            let mut token: Literal = repr.parse().unwrap();
            token.set_span(span);
            return Some((
                Lit::Int(LitInt {
                    repr: Box::new(LitIntRepr { token, digits, suffix }),
                }),
                rest,
            ));
        }
        let (digits, suffix) = value::parse_lit_float(&repr)?;
        let mut token: Literal = repr.parse().unwrap();
        token.set_span(span);
        Some((
            Lit::Float(LitFloat {
                repr: Box::new(LitFloatRepr { token, digits, suffix }),
            }),
            rest,
        ))
    }

    impl Parse for LitStr {
        fn parse(input: ParseStream) -> Result<Self> {
            let head = input.fork();
            match input.parse() {
                Ok(Lit::Str(lit)) => Ok(lit),
                _ => Err(head.error("expected string literal")),
            }
        }
    }

    impl Parse for LitByteStr {
        fn parse(input: ParseStream) -> Result<Self> {
            let head = input.fork();
            match input.parse() {
                Ok(Lit::ByteStr(lit)) => Ok(lit),
                _ => Err(head.error("expected byte string literal")),
            }
        }
    }

    impl Parse for LitByte {
        fn parse(input: ParseStream) -> Result<Self> {
            let head = input.fork();
            match input.parse() {
                Ok(Lit::Byte(lit)) => Ok(lit),
                _ => Err(head.error("expected byte literal")),
            }
        }
    }

    impl Parse for LitChar {
        fn parse(input: ParseStream) -> Result<Self> {
            let head = input.fork();
            match input.parse() {
                Ok(Lit::Char(lit)) => Ok(lit),
                _ => Err(head.error("expected character literal")),
            }
        }
    }

    impl Parse for LitInt {
        fn parse(input: ParseStream) -> Result<Self> {
            let head = input.fork();
            match input.parse() {
                Ok(Lit::Int(lit)) => Ok(lit),
                _ => Err(head.error("expected integer literal")),
            }
        }
    }

    impl Parse for LitFloat {
        fn parse(input: ParseStream) -> Result<Self> {
            let head = input.fork();
            match input.parse() {
                Ok(Lit::Float(lit)) => Ok(lit),
                _ => Err(head.error("expected floating point literal")),
            }
        }
    }

    impl Parse for LitBool {
        fn parse(input: ParseStream) -> Result<Self> {
            let head = input.fork();
            match input.parse() {
                Ok(Lit::Bool(lit)) => Ok(lit),
                _ => Err(head.error("expected boolean literal")),
            }
        }
    }
}
mod printing {
    use super::*;
    use proc_macro2::TokenStream;
    use quote::{ToTokens, TokenStreamExt};
    #[cfg_attr(doc_cfg, doc(cfg(feature = "printing")))]
    impl ToTokens for LitStr {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.repr.token.to_tokens(tokens);
        }
    }
    #[cfg_attr(doc_cfg, doc(cfg(feature = "printing")))]
    impl ToTokens for LitByteStr {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.repr.token.to_tokens(tokens);
        }
    }
    #[cfg_attr(doc_cfg, doc(cfg(feature = "printing")))]
    impl ToTokens for LitByte {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.repr.token.to_tokens(tokens);
        }
    }
    #[cfg_attr(doc_cfg, doc(cfg(feature = "printing")))]
    impl ToTokens for LitChar {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.repr.token.to_tokens(tokens);
        }
    }
    #[cfg_attr(doc_cfg, doc(cfg(feature = "printing")))]
    impl ToTokens for LitInt {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.repr.token.to_tokens(tokens);
        }
    }
    #[cfg_attr(doc_cfg, doc(cfg(feature = "printing")))]
    impl ToTokens for LitFloat {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.repr.token.to_tokens(tokens);
        }
    }
    #[cfg_attr(doc_cfg, doc(cfg(feature = "printing")))]
    impl ToTokens for LitBool {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            tokens.append(self.token());
        }
    }
}
mod value {
    use super::*;
    use crate::bigint::BigInt;
    use std::char;
    use std::ops::{Index, RangeFrom};
    impl Lit {
        pub fn new(token: Literal) -> Self {
            let repr = token.to_string();
            match byte(&repr, 0) {
                b'"' | b'r' => {
                    let (_, suffix) = parse_lit_str(&repr);
                    return Lit::Str(LitStr {
                        repr: Box::new(LitRepr { token, suffix }),
                    });
                },
                b'b' => match byte(&repr, 1) {
                    b'"' | b'r' => {
                        let (_, suffix) = parse_lit_byte_str(&repr);
                        return Lit::ByteStr(LitByteStr {
                            repr: Box::new(LitRepr { token, suffix }),
                        });
                    },
                    b'\'' => {
                        let (_, suffix) = parse_lit_byte(&repr);
                        return Lit::Byte(LitByte {
                            repr: Box::new(LitRepr { token, suffix }),
                        });
                    },
                    _ => {},
                },
                b'\'' => {
                    let (_, suffix) = parse_lit_char(&repr);
                    return Lit::Char(LitChar {
                        repr: Box::new(LitRepr { token, suffix }),
                    });
                },
                b'0'..=b'9' | b'-' => {
                    if let Some((digits, suffix)) = parse_lit_int(&repr) {
                        return Lit::Int(LitInt {
                            repr: Box::new(LitIntRepr { token, digits, suffix }),
                        });
                    }
                    if let Some((digits, suffix)) = parse_lit_float(&repr) {
                        return Lit::Float(LitFloat {
                            repr: Box::new(LitFloatRepr { token, digits, suffix }),
                        });
                    }
                },
                b't' | b'f' => {
                    if repr == "true" || repr == "false" {
                        return Lit::Bool(LitBool {
                            value: repr == "true",
                            span: token.span(),
                        });
                    }
                },
                b'c' => return Lit::Verbatim(token),
                _ => {},
            }
            panic!("Unrecognized literal: `{}`", repr);
        }
        pub fn suffix(&self) -> &str {
            match self {
                Lit::Str(lit) => lit.suffix(),
                Lit::ByteStr(lit) => lit.suffix(),
                Lit::Byte(lit) => lit.suffix(),
                Lit::Char(lit) => lit.suffix(),
                Lit::Int(lit) => lit.suffix(),
                Lit::Float(lit) => lit.suffix(),
                Lit::Bool(_) | Lit::Verbatim(_) => "",
            }
        }
        pub fn span(&self) -> Span {
            match self {
                Lit::Str(lit) => lit.span(),
                Lit::ByteStr(lit) => lit.span(),
                Lit::Byte(lit) => lit.span(),
                Lit::Char(lit) => lit.span(),
                Lit::Int(lit) => lit.span(),
                Lit::Float(lit) => lit.span(),
                Lit::Bool(lit) => lit.span,
                Lit::Verbatim(lit) => lit.span(),
            }
        }
        pub fn set_span(&mut self, span: Span) {
            match self {
                Lit::Str(lit) => lit.set_span(span),
                Lit::ByteStr(lit) => lit.set_span(span),
                Lit::Byte(lit) => lit.set_span(span),
                Lit::Char(lit) => lit.set_span(span),
                Lit::Int(lit) => lit.set_span(span),
                Lit::Float(lit) => lit.set_span(span),
                Lit::Bool(lit) => lit.span = span,
                Lit::Verbatim(lit) => lit.set_span(span),
            }
        }
    }
    pub(crate) fn byte<S: AsRef<[u8]> + ?Sized>(s: &S, idx: usize) -> u8 {
        let s = s.as_ref();
        if idx < s.len() {
            s[idx]
        } else {
            0
        }
    }
    fn next_chr(s: &str) -> char {
        s.chars().next().unwrap_or('\0')
    }
    pub(crate) fn parse_lit_str(s: &str) -> (Box<str>, Box<str>) {
        match byte(s, 0) {
            b'"' => parse_lit_str_cooked(s),
            b'r' => parse_lit_str_raw(s),
            _ => unreachable!(),
        }
    }
    #[allow(clippy::needless_continue)]
    fn parse_lit_str_cooked(mut s: &str) -> (Box<str>, Box<str>) {
        assert_eq!(byte(s, 0), b'"');
        s = &s[1..];
        let mut content = String::new();
        'outer: loop {
            let ch = match byte(s, 0) {
                b'"' => break,
                b'\\' => {
                    let b = byte(s, 1);
                    s = &s[2..];
                    match b {
                        b'x' => {
                            let (byte, rest) = backslash_x(s);
                            s = rest;
                            assert!(byte <= 0x7F, "Invalid \\x byte in string literal");
                            char::from_u32(u32::from(byte)).unwrap()
                        },
                        b'u' => {
                            let (chr, rest) = backslash_u(s);
                            s = rest;
                            chr
                        },
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'\\' => '\\',
                        b'0' => '\0',
                        b'\'' => '\'',
                        b'"' => '"',
                        b'\r' | b'\n' => loop {
                            let b = byte(s, 0);
                            match b {
                                b' ' | b'\t' | b'\n' | b'\r' => s = &s[1..],
                                _ => continue 'outer,
                            }
                        },
                        b => panic!("unexpected byte {:?} after \\ character in byte literal", b),
                    }
                },
                b'\r' => {
                    assert_eq!(byte(s, 1), b'\n', "Bare CR not allowed in string");
                    s = &s[2..];
                    '\n'
                },
                _ => {
                    let ch = next_chr(s);
                    s = &s[ch.len_utf8()..];
                    ch
                },
            };
            content.push(ch);
        }
        assert!(s.starts_with('"'));
        let content = content.into_boxed_str();
        let suffix = s[1..].to_owned().into_boxed_str();
        (content, suffix)
    }
    fn parse_lit_str_raw(mut s: &str) -> (Box<str>, Box<str>) {
        assert_eq!(byte(s, 0), b'r');
        s = &s[1..];
        let mut pounds = 0;
        while byte(s, pounds) == b'#' {
            pounds += 1;
        }
        assert_eq!(byte(s, pounds), b'"');
        let close = s.rfind('"').unwrap();
        for end in s[close + 1..close + 1 + pounds].bytes() {
            assert_eq!(end, b'#');
        }
        let content = s[pounds + 1..close].to_owned().into_boxed_str();
        let suffix = s[close + 1 + pounds..].to_owned().into_boxed_str();
        (content, suffix)
    }
    pub(crate) fn parse_lit_byte_str(s: &str) -> (Vec<u8>, Box<str>) {
        assert_eq!(byte(s, 0), b'b');
        match byte(s, 1) {
            b'"' => parse_lit_byte_str_cooked(s),
            b'r' => parse_lit_byte_str_raw(s),
            _ => unreachable!(),
        }
    }
    #[allow(clippy::needless_continue)]
    fn parse_lit_byte_str_cooked(mut s: &str) -> (Vec<u8>, Box<str>) {
        assert_eq!(byte(s, 0), b'b');
        assert_eq!(byte(s, 1), b'"');
        s = &s[2..];
        let mut v = s.as_bytes();
        let mut out = Vec::new();
        'outer: loop {
            let byte = match byte(v, 0) {
                b'"' => break,
                b'\\' => {
                    let b = byte(v, 1);
                    v = &v[2..];
                    match b {
                        b'x' => {
                            let (b, rest) = backslash_x(v);
                            v = rest;
                            b
                        },
                        b'n' => b'\n',
                        b'r' => b'\r',
                        b't' => b'\t',
                        b'\\' => b'\\',
                        b'0' => b'\0',
                        b'\'' => b'\'',
                        b'"' => b'"',
                        b'\r' | b'\n' => loop {
                            let byte = byte(v, 0);
                            if matches!(byte, b' ' | b'\t' | b'\n' | b'\r') {
                                v = &v[1..];
                            } else {
                                continue 'outer;
                            }
                        },
                        b => panic!("unexpected byte {:?} after \\ character in byte literal", b),
                    }
                },
                b'\r' => {
                    assert_eq!(byte(v, 1), b'\n', "Bare CR not allowed in string");
                    v = &v[2..];
                    b'\n'
                },
                b => {
                    v = &v[1..];
                    b
                },
            };
            out.push(byte);
        }
        assert_eq!(byte(v, 0), b'"');
        let suffix = s[s.len() - v.len() + 1..].to_owned().into_boxed_str();
        (out, suffix)
    }
    fn parse_lit_byte_str_raw(s: &str) -> (Vec<u8>, Box<str>) {
        assert_eq!(byte(s, 0), b'b');
        let (value, suffix) = parse_lit_str_raw(&s[1..]);
        (String::from(value).into_bytes(), suffix)
    }
    pub(crate) fn parse_lit_byte(s: &str) -> (u8, Box<str>) {
        assert_eq!(byte(s, 0), b'b');
        assert_eq!(byte(s, 1), b'\'');
        let mut v = s[2..].as_bytes();
        let b = match byte(v, 0) {
            b'\\' => {
                let b = byte(v, 1);
                v = &v[2..];
                match b {
                    b'x' => {
                        let (b, rest) = backslash_x(v);
                        v = rest;
                        b
                    },
                    b'n' => b'\n',
                    b'r' => b'\r',
                    b't' => b'\t',
                    b'\\' => b'\\',
                    b'0' => b'\0',
                    b'\'' => b'\'',
                    b'"' => b'"',
                    b => panic!("unexpected byte {:?} after \\ character in byte literal", b),
                }
            },
            b => {
                v = &v[1..];
                b
            },
        };
        assert_eq!(byte(v, 0), b'\'');
        let suffix = s[s.len() - v.len() + 1..].to_owned().into_boxed_str();
        (b, suffix)
    }
    pub(crate) fn parse_lit_char(mut s: &str) -> (char, Box<str>) {
        assert_eq!(byte(s, 0), b'\'');
        s = &s[1..];
        let ch = match byte(s, 0) {
            b'\\' => {
                let b = byte(s, 1);
                s = &s[2..];
                match b {
                    b'x' => {
                        let (byte, rest) = backslash_x(s);
                        s = rest;
                        assert!(byte <= 0x80, "Invalid \\x byte in string literal");
                        char::from_u32(u32::from(byte)).unwrap()
                    },
                    b'u' => {
                        let (chr, rest) = backslash_u(s);
                        s = rest;
                        chr
                    },
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    b'\\' => '\\',
                    b'0' => '\0',
                    b'\'' => '\'',
                    b'"' => '"',
                    b => panic!("unexpected byte {:?} after \\ character in byte literal", b),
                }
            },
            _ => {
                let ch = next_chr(s);
                s = &s[ch.len_utf8()..];
                ch
            },
        };
        assert_eq!(byte(s, 0), b'\'');
        let suffix = s[1..].to_owned().into_boxed_str();
        (ch, suffix)
    }
    fn backslash_x<S>(s: &S) -> (u8, &S)
    where
        S: Index<RangeFrom<usize>, Output = S> + AsRef<[u8]> + ?Sized,
    {
        let mut ch = 0;
        let b0 = byte(s, 0);
        let b1 = byte(s, 1);
        ch += 0x10
            * match b0 {
                b'0'..=b'9' => b0 - b'0',
                b'a'..=b'f' => 10 + (b0 - b'a'),
                b'A'..=b'F' => 10 + (b0 - b'A'),
                _ => panic!("unexpected non-hex character after \\x"),
            };
        ch += match b1 {
            b'0'..=b'9' => b1 - b'0',
            b'a'..=b'f' => 10 + (b1 - b'a'),
            b'A'..=b'F' => 10 + (b1 - b'A'),
            _ => panic!("unexpected non-hex character after \\x"),
        };
        (ch, &s[2..])
    }
    fn backslash_u(mut s: &str) -> (char, &str) {
        if byte(s, 0) != b'{' {
            panic!("{}", "expected { after \\u");
        }
        s = &s[1..];
        let mut ch = 0;
        let mut digits = 0;
        loop {
            let b = byte(s, 0);
            let digit = match b {
                b'0'..=b'9' => b - b'0',
                b'a'..=b'f' => 10 + b - b'a',
                b'A'..=b'F' => 10 + b - b'A',
                b'_' if digits > 0 => {
                    s = &s[1..];
                    continue;
                },
                b'}' if digits == 0 => panic!("invalid empty unicode escape"),
                b'}' => break,
                _ => panic!("unexpected non-hex character after \\u"),
            };
            if digits == 6 {
                panic!("overlong unicode escape (must have at most 6 hex digits)");
            }
            ch *= 0x10;
            ch += u32::from(digit);
            digits += 1;
            s = &s[1..];
        }
        assert!(byte(s, 0) == b'}');
        s = &s[1..];
        if let Some(ch) = char::from_u32(ch) {
            (ch, s)
        } else {
            panic!("character code {:x} is not a valid unicode character", ch);
        }
    }
    pub(crate) fn parse_lit_int(mut s: &str) -> Option<(Box<str>, Box<str>)> {
        let negative = byte(s, 0) == b'-';
        if negative {
            s = &s[1..];
        }
        let base = match (byte(s, 0), byte(s, 1)) {
            (b'0', b'x') => {
                s = &s[2..];
                16
            },
            (b'0', b'o') => {
                s = &s[2..];
                8
            },
            (b'0', b'b') => {
                s = &s[2..];
                2
            },
            (b'0'..=b'9', _) => 10,
            _ => return None,
        };
        let mut value = BigInt::new();
        let mut has_digit = false;
        'outer: loop {
            let b = byte(s, 0);
            let digit = match b {
                b'0'..=b'9' => b - b'0',
                b'a'..=b'f' if base > 10 => b - b'a' + 10,
                b'A'..=b'F' if base > 10 => b - b'A' + 10,
                b'_' => {
                    s = &s[1..];
                    continue;
                },
                b'.' if base == 10 => return None,
                b'e' | b'E' if base == 10 => {
                    let mut has_exp = false;
                    for (i, b) in s[1..].bytes().enumerate() {
                        match b {
                            b'_' => {},
                            b'-' | b'+' => return None,
                            b'0'..=b'9' => has_exp = true,
                            _ => {
                                let suffix = &s[1 + i..];
                                if has_exp && crate::ident::xid_ok(suffix) {
                                    return None;
                                } else {
                                    break 'outer;
                                }
                            },
                        }
                    }
                    if has_exp {
                        return None;
                    } else {
                        break;
                    }
                },
                _ => break,
            };
            if digit >= base {
                return None;
            }
            has_digit = true;
            value *= base;
            value += digit;
            s = &s[1..];
        }
        if !has_digit {
            return None;
        }
        let suffix = s;
        if suffix.is_empty() || crate::ident::xid_ok(suffix) {
            let mut repr = value.to_string();
            if negative {
                repr.insert(0, '-');
            }
            Some((repr.into_boxed_str(), suffix.to_owned().into_boxed_str()))
        } else {
            None
        }
    }
    pub(crate) fn parse_lit_float(input: &str) -> Option<(Box<str>, Box<str>)> {
        let mut bytes = input.to_owned().into_bytes();
        let start = (*bytes.first()? == b'-') as usize;
        match bytes.get(start)? {
            b'0'..=b'9' => {},
            _ => return None,
        }
        let mut read = start;
        let mut write = start;
        let mut has_dot = false;
        let mut has_e = false;
        let mut has_sign = false;
        let mut has_exponent = false;
        while read < bytes.len() {
            match bytes[read] {
                b'_' => {
                    read += 1;
                    continue;
                },
                b'0'..=b'9' => {
                    if has_e {
                        has_exponent = true;
                    }
                    bytes[write] = bytes[read];
                },
                b'.' => {
                    if has_e || has_dot {
                        return None;
                    }
                    has_dot = true;
                    bytes[write] = b'.';
                },
                b'e' | b'E' => {
                    match bytes[read + 1..].iter().find(|b| **b != b'_').unwrap_or(&b'\0') {
                        b'-' | b'+' | b'0'..=b'9' => {},
                        _ => break,
                    }
                    if has_e {
                        if has_exponent {
                            break;
                        } else {
                            return None;
                        }
                    }
                    has_e = true;
                    bytes[write] = b'e';
                },
                b'-' | b'+' => {
                    if has_sign || has_exponent || !has_e {
                        return None;
                    }
                    has_sign = true;
                    if bytes[read] == b'-' {
                        bytes[write] = bytes[read];
                    } else {
                        read += 1;
                        continue;
                    }
                },
                _ => break,
            }
            read += 1;
            write += 1;
        }
        if has_e && !has_exponent {
            return None;
        }
        let mut digits = String::from_utf8(bytes).unwrap();
        let suffix = digits.split_off(read);
        digits.truncate(write);
        if suffix.is_empty() || crate::ident::xid_ok(&suffix) {
            Some((digits.into_boxed_str(), suffix.into_boxed_str()))
        } else {
            None
        }
    }
}
