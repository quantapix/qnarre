use super::{
    ident, lookahead,
    parse::{Parse, Parser},
    Err, Result,
};
use proc_macro2::{Ident, Literal, Span, TokenStream, TokenTree};
use std::{
    char,
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    ops::{Index, RangeFrom},
    str::{self, FromStr},
};

ast_enum_of_structs! {
    pub enum Lit {
        Str(Str),
        ByteStr(ByteStr),
        Byte(Byte),
        Char(Char),
        Int(Int),
        Float(Float),
        Bool(Bool),
        Verbatim(Literal),
    }
}
impl Lit {
    pub fn new(tok: Literal) -> Self {
        let repr = tok.to_string();
        match byte(&repr, 0) {
            b'"' | b'r' => {
                let (_, suff) = parse_str(&repr);
                return Lit::Str(Str {
                    repr: Box::new(Repr { tok, suff }),
                });
            },
            b'b' => match byte(&repr, 1) {
                b'"' | b'r' => {
                    let (_, suff) = parse_byte_str(&repr);
                    return Lit::ByteStr(ByteStr {
                        repr: Box::new(Repr { tok, suff }),
                    });
                },
                b'\'' => {
                    let (_, suff) = parse_byte(&repr);
                    return Lit::Byte(Byte {
                        repr: Box::new(Repr { tok, suff }),
                    });
                },
                _ => {},
            },
            b'\'' => {
                let (_, suff) = parse_char(&repr);
                return Lit::Char(Char {
                    repr: Box::new(Repr { tok, suff }),
                });
            },
            b'0'..=b'9' | b'-' => {
                if let Some((digits, suff)) = parse_int(&repr) {
                    return Lit::Int(Int {
                        repr: Box::new(IntRepr { tok, digits, suff }),
                    });
                }
                if let Some((digits, suff)) = parse_float(&repr) {
                    return Lit::Float(Float {
                        repr: Box::new(FloatRepr { tok, digits, suff }),
                    });
                }
            },
            b't' | b'f' => {
                if repr == "true" || repr == "false" {
                    return Lit::Bool(Bool {
                        val: repr == "true",
                        span: tok.span(),
                    });
                }
            },
            b'c' => return Lit::Verbatim(tok),
            _ => {},
        }
        panic!("Unrecognized literal: `{}`", repr);
    }
    pub fn suffix(&self) -> &str {
        use Lit::*;
        match self {
            Str(x) => x.suffix(),
            ByteStr(x) => x.suffix(),
            Byte(x) => x.suffix(),
            Char(x) => x.suffix(),
            Int(x) => x.suffix(),
            Float(x) => x.suffix(),
            Bool(_) | Verbatim(_) => "",
        }
    }
    pub fn span(&self) -> Span {
        use Lit::*;
        match self {
            Str(x) => x.span(),
            ByteStr(x) => x.span(),
            Byte(x) => x.span(),
            Char(x) => x.span(),
            Int(x) => x.span(),
            Float(x) => x.span(),
            Bool(x) => x.span,
            Verbatim(x) => x.span(),
        }
    }
    pub fn set_span(&mut self, s: Span) {
        use Lit::*;
        match self {
            Str(x) => x.set_span(s),
            ByteStr(x) => x.set_span(s),
            Byte(x) => x.set_span(s),
            Char(x) => x.set_span(s),
            Int(x) => x.set_span(s),
            Float(x) => x.set_span(s),
            Bool(x) => x.span = s,
            Verbatim(x) => x.set_span(s),
        }
    }
}
#[allow(non_snake_case)]
pub fn Lit(x: lookahead::TokenMarker) -> Lit {
    match x {}
}

struct Repr {
    pub tok: Literal,
    suff: Box<str>,
}
impl Clone for Repr {
    fn clone(&self) -> Self {
        Repr {
            tok: self.tok.clone(),
            suff: self.suff.clone(),
        }
    }
}

pub struct Str {
    pub repr: Box<Repr>,
}
impl Str {
    pub fn new(x: &str, s: Span) -> Self {
        let mut tok = Literal::string(x);
        tok.set_span(s);
        Str {
            repr: Box::new(Repr {
                tok,
                suff: Box::<str>::default(),
            }),
        }
    }
    pub fn value(&self) -> String {
        let repr = self.repr.tok.to_string();
        let (value, _) = parse_str(&repr);
        String::from(value)
    }
    pub fn parse<T: Parse>(&self) -> Result<T> {
        self.parse_with(T::parse)
    }
    pub fn parse_with<F: Parser>(&self, parser: F) -> Result<F::Output> {
        use proc_macro2::Group;
        fn respan_token_stream(x: TokenStream, s: Span) -> TokenStream {
            x.into_iter().map(|x| respan_token_tree(x, s)).collect()
        }
        fn respan_token_tree(mut token: TokenTree, s: Span) -> TokenTree {
            match &mut token {
                TokenTree::Group(g) => {
                    let stream = respan_token_stream(g.stream(), s);
                    *g = Group::new(g.delimiter(), stream);
                    g.set_span(s);
                },
                other => other.set_span(s),
            }
            token
        }
        let mut tokens = TokenStream::from_str(&self.value())?;
        tokens = respan_token_stream(tokens, self.span());
        parser.parse2(tokens)
    }
    pub fn span(&self) -> Span {
        self.repr.tok.span()
    }
    pub fn set_span(&mut self, s: Span) {
        self.repr.tok.set_span(s);
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suff
    }
    pub fn token(&self) -> Literal {
        self.repr.tok.clone()
    }
}
impl Debug for Str {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Str {
            pub fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                f.debug_struct(name)
                    .field("token", &format_args!("{}", self.repr.tok))
                    .finish()
            }
        }
        self.debug(f, "lit::Str")
    }
}
macro_rules! extra_traits {
    ($ty:ident) => {
        impl Clone for $ty {
            fn clone(&self) -> Self {
                $ty {
                    repr: self.repr.clone(),
                }
            }
        }
        impl PartialEq for $ty {
            fn eq(&self, x: &Self) -> bool {
                self.repr.tok.to_string() == x.repr.tok.to_string()
            }
        }
        impl Hash for $ty {
            fn hash<H>(&self, x: &mut H)
            where
                H: Hasher,
            {
                self.repr.tok.to_string().hash(x);
            }
        }
        #[allow(non_snake_case)]
        pub fn $ty(x: lookahead::TokenMarker) -> $ty {
            match x {}
        }
    };
}
extra_traits!(Str);

pub struct ByteStr {
    pub repr: Box<Repr>,
}
impl ByteStr {
    pub fn new(x: &[u8], s: Span) -> Self {
        let mut tok = Literal::byte_string(x);
        tok.set_span(s);
        ByteStr {
            repr: Box::new(Repr {
                tok,
                suff: Box::<str>::default(),
            }),
        }
    }
    pub fn value(&self) -> Vec<u8> {
        let repr = self.repr.tok.to_string();
        let (value, _) = parse_byte_str(&repr);
        value
    }
    pub fn span(&self) -> Span {
        self.repr.tok.span()
    }
    pub fn set_span(&mut self, s: Span) {
        self.repr.tok.set_span(s);
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suff
    }
    pub fn token(&self) -> Literal {
        self.repr.tok.clone()
    }
}
impl Debug for ByteStr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl ByteStr {
            pub fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                f.debug_struct(name)
                    .field("token", &format_args!("{}", self.repr.tok))
                    .finish()
            }
        }
        self.debug(f, "lit::ByteStr")
    }
}
extra_traits!(ByteStr);

pub struct Byte {
    pub repr: Box<Repr>,
}
impl Byte {
    pub fn new(x: u8, s: Span) -> Self {
        let mut tok = Literal::u8_suffixed(x);
        tok.set_span(s);
        Byte {
            repr: Box::new(Repr {
                tok,
                suff: Box::<str>::default(),
            }),
        }
    }
    pub fn value(&self) -> u8 {
        let repr = self.repr.tok.to_string();
        let (value, _) = parse_byte(&repr);
        value
    }
    pub fn span(&self) -> Span {
        self.repr.tok.span()
    }
    pub fn set_span(&mut self, x: Span) {
        self.repr.tok.set_span(x);
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suff
    }
    pub fn token(&self) -> Literal {
        self.repr.tok.clone()
    }
}
impl Debug for Byte {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Byte {
            pub fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                f.debug_struct(name)
                    .field("token", &format_args!("{}", self.repr.tok))
                    .finish()
            }
        }
        self.debug(f, "LitByte")
    }
}
extra_traits!(Byte);

pub struct Char {
    pub repr: Box<Repr>,
}
impl Char {
    pub fn new(x: char, s: Span) -> Self {
        let mut tok = Literal::character(x);
        tok.set_span(s);
        Char {
            repr: Box::new(Repr {
                tok,
                suff: Box::<str>::default(),
            }),
        }
    }
    pub fn value(&self) -> char {
        let repr = self.repr.tok.to_string();
        let (value, _) = parse_char(&repr);
        value
    }
    pub fn span(&self) -> Span {
        self.repr.tok.span()
    }
    pub fn set_span(&mut self, s: Span) {
        self.repr.tok.set_span(s);
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suff
    }
    pub fn token(&self) -> Literal {
        self.repr.tok.clone()
    }
}
impl Debug for Char {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Char {
            pub fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                f.debug_struct(name)
                    .field("token", &format_args!("{}", self.repr.tok))
                    .finish()
            }
        }
        self.debug(f, "lit::Char")
    }
}
extra_traits!(Char);

struct IntRepr {
    pub tok: Literal,
    digits: Box<str>,
    suff: Box<str>,
}
impl Clone for IntRepr {
    fn clone(&self) -> Self {
        IntRepr {
            tok: self.tok.clone(),
            digits: self.digits.clone(),
            suff: self.suff.clone(),
        }
    }
}

pub struct Int {
    pub repr: Box<IntRepr>,
}
impl Int {
    pub fn new(x: &str, s: Span) -> Self {
        let (digits, suff) = match parse_int(x) {
            Some(x) => x,
            None => panic!("Not an integer literal: `{}`", x),
        };
        let mut tok: Literal = x.parse().unwrap();
        tok.set_span(s);
        Int {
            repr: Box::new(IntRepr { tok, digits, suff }),
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
        self.base10_digits().parse().map_err(|x| Err::new(self.span(), x))
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suff
    }
    pub fn span(&self) -> Span {
        self.repr.tok.span()
    }
    pub fn set_span(&mut self, s: Span) {
        self.repr.tok.set_span(s);
    }
    pub fn token(&self) -> Literal {
        self.repr.tok.clone()
    }
}
impl From<Literal> for Int {
    fn from(tok: Literal) -> Self {
        let repr = tok.to_string();
        if let Some((digits, suff)) = parse_int(&repr) {
            Int {
                repr: Box::new(IntRepr { tok, digits, suff }),
            }
        } else {
            panic!("Not an integer literal: `{}`", repr);
        }
    }
}
impl Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.repr.tok.fmt(f)
    }
}
impl Debug for Int {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Int {
            pub fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                f.debug_struct(name)
                    .field("token", &format_args!("{}", self.repr.tok))
                    .finish()
            }
        }
        self.debug(f, "LitInt")
    }
}
extra_traits!(Int);

struct FloatRepr {
    pub tok: Literal,
    digits: Box<str>,
    suff: Box<str>,
}
impl Clone for FloatRepr {
    fn clone(&self) -> Self {
        FloatRepr {
            tok: self.tok.clone(),
            digits: self.digits.clone(),
            suff: self.suff.clone(),
        }
    }
}

pub struct Float {
    pub repr: Box<FloatRepr>,
}
impl Float {
    pub fn new(x: &str, s: Span) -> Self {
        let (digits, suff) = match parse_float(x) {
            Some(x) => x,
            None => panic!("Not a float literal: `{}`", x),
        };
        let mut tok: Literal = x.parse().unwrap();
        tok.set_span(s);
        Float {
            repr: Box::new(FloatRepr { tok, digits, suff }),
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
        self.base10_digits().parse().map_err(|err| Err::new(self.span(), err))
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suff
    }
    pub fn span(&self) -> Span {
        self.repr.tok.span()
    }
    pub fn set_span(&mut self, s: Span) {
        self.repr.tok.set_span(s);
    }
    pub fn token(&self) -> Literal {
        self.repr.tok.clone()
    }
}
impl From<Literal> for Float {
    fn from(tok: Literal) -> Self {
        let repr = tok.to_string();
        if let Some((digits, suff)) = parse_float(&repr) {
            Float {
                repr: Box::new(FloatRepr { tok, digits, suff }),
            }
        } else {
            panic!("Not a float literal: `{}`", repr);
        }
    }
}
impl Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.repr.tok.fmt(f)
    }
}
impl Debug for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Float {
            pub fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                f.debug_struct(name)
                    .field("token", &format_args!("{}", self.repr.tok))
                    .finish()
            }
        }
        self.debug(f, "lit::Float")
    }
}
extra_traits!(Float);

pub struct Bool {
    pub val: bool,
    pub span: Span,
}
impl Bool {
    pub fn new(val: bool, span: Span) -> Self {
        Bool { val, span }
    }
    pub fn value(&self) -> bool {
        self.val
    }
    pub fn span(&self) -> Span {
        self.span
    }
    pub fn set_span(&mut self, s: Span) {
        self.span = s;
    }
    pub fn token(&self) -> Ident {
        let s = if self.val { "true" } else { "false" };
        Ident::new(s, self.span)
    }
}
impl Debug for Bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Bool {
            pub fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                f.debug_struct(name).field("value", &self.val).finish()
            }
        }
        self.debug(f, "lit::Bool")
    }
}
#[allow(non_snake_case)]
pub fn Bool(x: lookahead::TokenMarker) -> Bool {
    match x {}
}

pub fn parse_str(x: &str) -> (Box<str>, Box<str>) {
    match byte(x, 0) {
        b'"' => parse_str_cooked(x),
        b'r' => parse_str_raw(x),
        _ => unreachable!(),
    }
}
#[allow(clippy::needless_continue)]
fn parse_str_cooked(mut x: &str) -> (Box<str>, Box<str>) {
    assert_eq!(byte(x, 0), b'"');
    x = &x[1..];
    let mut content = String::new();
    'outer: loop {
        let ch = match byte(x, 0) {
            b'"' => break,
            b'\\' => {
                let b = byte(x, 1);
                x = &x[2..];
                match b {
                    b'x' => {
                        let (byte, rest) = backslash_x(x);
                        x = rest;
                        assert!(byte <= 0x7F, "Invalid \\x byte in string literal");
                        char::from_u32(u32::from(byte)).unwrap()
                    },
                    b'u' => {
                        let (chr, rest) = backslash_u(x);
                        x = rest;
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
                        let b = byte(x, 0);
                        match b {
                            b' ' | b'\t' | b'\n' | b'\r' => x = &x[1..],
                            _ => continue 'outer,
                        }
                    },
                    b => panic!("unexpected byte {:?} after \\ character in byte literal", b),
                }
            },
            b'\r' => {
                assert_eq!(byte(x, 1), b'\n', "Bare CR not allowed in string");
                x = &x[2..];
                '\n'
            },
            _ => {
                let ch = next_chr(x);
                x = &x[ch.len_utf8()..];
                ch
            },
        };
        content.push(ch);
    }
    assert!(x.starts_with('"'));
    let content = content.into_boxed_str();
    let suffix = x[1..].to_owned().into_boxed_str();
    (content, suffix)
}
fn parse_str_raw(mut x: &str) -> (Box<str>, Box<str>) {
    assert_eq!(byte(x, 0), b'r');
    x = &x[1..];
    let mut pounds = 0;
    while byte(x, pounds) == b'#' {
        pounds += 1;
    }
    assert_eq!(byte(x, pounds), b'"');
    let close = x.rfind('"').unwrap();
    for end in x[close + 1..close + 1 + pounds].bytes() {
        assert_eq!(end, b'#');
    }
    let content = x[pounds + 1..close].to_owned().into_boxed_str();
    let suffix = x[close + 1 + pounds..].to_owned().into_boxed_str();
    (content, suffix)
}

pub fn parse_byte_str(x: &str) -> (Vec<u8>, Box<str>) {
    assert_eq!(byte(x, 0), b'b');
    match byte(x, 1) {
        b'"' => parse_byte_str_cooked(x),
        b'r' => parse_byte_str_raw(x),
        _ => unreachable!(),
    }
}
#[allow(clippy::needless_continue)]
fn parse_byte_str_cooked(mut x: &str) -> (Vec<u8>, Box<str>) {
    assert_eq!(byte(x, 0), b'b');
    assert_eq!(byte(x, 1), b'"');
    x = &x[2..];
    let mut v = x.as_bytes();
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
    let suffix = x[x.len() - v.len() + 1..].to_owned().into_boxed_str();
    (out, suffix)
}
fn parse_byte_str_raw(x: &str) -> (Vec<u8>, Box<str>) {
    assert_eq!(byte(x, 0), b'b');
    let (value, suffix) = parse_str_raw(&x[1..]);
    (String::from(value).into_bytes(), suffix)
}

pub fn parse_byte(x: &str) -> (u8, Box<str>) {
    assert_eq!(byte(x, 0), b'b');
    assert_eq!(byte(x, 1), b'\'');
    let mut v = x[2..].as_bytes();
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
    let suffix = x[x.len() - v.len() + 1..].to_owned().into_boxed_str();
    (b, suffix)
}
pub fn parse_char(mut x: &str) -> (char, Box<str>) {
    assert_eq!(byte(x, 0), b'\'');
    x = &x[1..];
    let ch = match byte(x, 0) {
        b'\\' => {
            let b = byte(x, 1);
            x = &x[2..];
            match b {
                b'x' => {
                    let (byte, rest) = backslash_x(x);
                    x = rest;
                    assert!(byte <= 0x80, "Invalid \\x byte in string literal");
                    char::from_u32(u32::from(byte)).unwrap()
                },
                b'u' => {
                    let (chr, rest) = backslash_u(x);
                    x = rest;
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
            let ch = next_chr(x);
            x = &x[ch.len_utf8()..];
            ch
        },
    };
    assert_eq!(byte(x, 0), b'\'');
    let suffix = x[1..].to_owned().into_boxed_str();
    (ch, suffix)
}

pub fn parse_int(mut x: &str) -> Option<(Box<str>, Box<str>)> {
    let negative = byte(x, 0) == b'-';
    if negative {
        x = &x[1..];
    }
    let base = match (byte(x, 0), byte(x, 1)) {
        (b'0', b'x') => {
            x = &x[2..];
            16
        },
        (b'0', b'o') => {
            x = &x[2..];
            8
        },
        (b'0', b'b') => {
            x = &x[2..];
            2
        },
        (b'0'..=b'9', _) => 10,
        _ => return None,
    };
    let mut value = BigInt::new();
    let mut has_digit = false;
    'outer: loop {
        let b = byte(x, 0);
        let digit = match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' if base > 10 => b - b'a' + 10,
            b'A'..=b'F' if base > 10 => b - b'A' + 10,
            b'_' => {
                x = &x[1..];
                continue;
            },
            b'.' if base == 10 => return None,
            b'e' | b'E' if base == 10 => {
                let mut has_exp = false;
                for (i, b) in x[1..].bytes().enumerate() {
                    match b {
                        b'_' => {},
                        b'-' | b'+' => return None,
                        b'0'..=b'9' => has_exp = true,
                        _ => {
                            let suffix = &x[1 + i..];
                            if has_exp && ident::xid_ok(suffix) {
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
        x = &x[1..];
    }
    if !has_digit {
        return None;
    }
    let suffix = x;
    if suffix.is_empty() || ident::xid_ok(suffix) {
        let mut repr = value.to_string();
        if negative {
            repr.insert(0, '-');
        }
        Some((repr.into_boxed_str(), suffix.to_owned().into_boxed_str()))
    } else {
        None
    }
}
pub fn parse_float(x: &str) -> Option<(Box<str>, Box<str>)> {
    let mut bytes = x.to_owned().into_bytes();
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
    if suffix.is_empty() || ident::xid_ok(&suffix) {
        Some((digits.into_boxed_str(), suffix.into_boxed_str()))
    } else {
        None
    }
}

fn byte<S: AsRef<[u8]> + ?Sized>(s: &S, idx: usize) -> u8 {
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