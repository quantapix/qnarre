use super::*;
use std::{
    char,
    ops::{Index, RangeFrom},
    str::{self, FromStr},
};

enum_of_structs! {
    pub enum Lit {
        Bool(Bool),
        Byte(Byte),
        ByteStr(ByteStr),
        Char(Char),
        Float(Float),
        Int(Int),
        Str(Str),
        Verbatim(pm2::Lit),
    }
}
impl Lit {
    pub fn new(tok: pm2::Lit) -> Self {
        let repr = tok.to_string();
        match byte(&repr, 0) {
            b'"' | b'r' => {
                let (_, suff) = parse::parse_str(&repr);
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
    pub fn span(&self) -> pm2::Span {
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
    pub fn set_span(&mut self, s: pm2::Span) {
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
impl Parse for Lit {
    fn parse(s: Stream) -> Res<Self> {
        s.step(|x| {
            if let Some((x, xs)) = x.literal() {
                return Ok((Lit::new(x), xs));
            }
            if let Some((x, xs)) = x.ident() {
                let val = x == "true";
                if val || x == "false" {
                    let y = Bool { val, span: x.span() };
                    return Ok((Lit::Bool(y), xs));
                }
            }
            if let Some((x, xs)) = x.punct() {
                if x.as_char() == '-' {
                    if let Some((x, xs)) = parse_negative(x, xs) {
                        return Ok((x, xs));
                    }
                }
            }
            Err(x.err("expected literal"))
        })
    }
}
impl Clone for Lit {
    fn clone(&self) -> Self {
        use Lit::*;
        match self {
            Bool(x) => Bool(x.clone()),
            Byte(x) => Byte(x.clone()),
            ByteStr(x) => ByteStr(x.clone()),
            Char(x) => Char(x.clone()),
            Float(x) => Float(x.clone()),
            Int(x) => Int(x.clone()),
            Str(x) => Str(x.clone()),
            Verbatim(x) => Verbatim(x.clone()),
        }
    }
}
impl Debug for Lit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Lit::")?;
        use Lit::*;
        match self {
            Bool(x) => x.debug(f, "Bool"),
            Byte(x) => x.debug(f, "Byte"),
            ByteStr(x) => x.debug(f, "ByteStr"),
            Char(x) => x.debug(f, "Char"),
            Float(x) => x.debug(f, "Float"),
            Int(x) => x.debug(f, "Int"),
            Str(x) => x.debug(f, "Str"),
            Verbatim(x) => {
                let mut f = f.debug_tuple("Verbatim");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Eq for Lit {}
impl PartialEq for Lit {
    fn eq(&self, x: &Self) -> bool {
        use Lit::*;
        match (self, x) {
            (Str(x), Str(y)) => x == y,
            (ByteStr(x), ByteStr(y)) => x == y,
            (Byte(x), Byte(y)) => x == y,
            (Char(x), Char(y)) => x == y,
            (Int(x), Int(y)) => x == y,
            (Float(x), Float(y)) => x == y,
            (Bool(x), Bool(y)) => x == y,
            (Verbatim(x), Verbatim(y)) => x.to_string() == y.to_string(),
            _ => false,
        }
    }
}
impl Pretty for Lit {
    fn pretty(&self, p: &mut Print) {
        use Lit::*;
        match self {
            Str(x) => x.pretty(p),
            ByteStr(x) => x.pretty(p),
            Byte(x) => x.pretty(p),
            Char(x) => x.pretty(p),
            Int(x) => x.pretty(p),
            Float(x) => x.pretty(p),
            Bool(x) => x.pretty(p),
            Verbatim(x) => x.pretty(p),
        }
    }
}
impl<F: Folder + ?Sized> Fold for Lit {
    fn fold(&self, f: &mut F) {
        use Lit::*;
        match self {
            Bool(x) => Bool(x.fold(f)),
            Byte(x) => Byte(x.fold(f)),
            ByteStr(x) => ByteStr(x.fold(f)),
            Char(x) => Char(x.fold(f)),
            Float(x) => Float(x.fold(f)),
            Int(x) => Int(x.fold(f)),
            Str(x) => Str(x.fold(f)),
            Verbatim(x) => Verbatim(x),
        }
    }
}
impl<H: Hasher> Hash for Lit {
    fn hash(&self, h: &mut H) {
        use Lit::*;
        match self {
            Str(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            ByteStr(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Byte(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Char(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Int(x) => {
                h.write_u8(4u8);
                x.hash(h);
            },
            Float(x) => {
                h.write_u8(5u8);
                x.hash(h);
            },
            Bool(x) => {
                h.write_u8(6u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(7u8);
                x.to_string().hash(h);
            },
        }
    }
}
impl<V: Visitor + ?Sized> Visit for Lit {
    fn visit(&self, v: &mut V) {
        use Lit::*;
        match self {
            Str(x) => {
                x.visit(v);
            },
            ByteStr(x) => {
                x.visit(v);
            },
            Byte(x) => {
                x.visit(v);
            },
            Char(x) => {
                x.visit(v);
            },
            Int(x) => {
                x.visit(v);
            },
            Float(x) => {
                x.visit(v);
            },
            Bool(x) => {
                x.visit(v);
            },
            Verbatim(_) => {},
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        use Lit::*;
        match self {
            Str(x) => {
                x.visit_mut(v);
            },
            ByteStr(x) => {
                x.visit_mut(v);
            },
            Byte(x) => {
                x.visit_mut(v);
            },
            Char(x) => {
                x.visit_mut(v);
            },
            Int(x) => {
                x.visit_mut(v);
            },
            Float(x) => {
                x.visit_mut(v);
            },
            Bool(x) => {
                x.visit_mut(v);
            },
            Verbatim(_) => {},
        }
    }
}

pub struct Bool {
    pub val: bool,
    pub span: pm2::Span,
}
impl Bool {
    pub fn new(val: bool, span: pm2::Span) -> Self {
        Bool { val, span }
    }
    pub fn value(&self) -> bool {
        self.val
    }
    pub fn span(&self) -> pm2::Span {
        self.span
    }
    pub fn set_span(&mut self, s: pm2::Span) {
        self.span = s;
    }
    pub fn token(&self) -> Ident {
        let s = if self.val { "true" } else { "false" };
        Ident::new(s, self.span)
    }
}
impl Parse for Bool {
    fn parse(s: Stream) -> Res<Self> {
        let x = s.fork();
        match s.parse() {
            Ok(Lit::Bool(x)) => Ok(x),
            _ => Err(x.error("expected boolean literal")),
        }
    }
}
impl Lower for Bool {
    fn lower(&self, s: &mut Stream) {
        s.append(self.token());
    }
}
impl Clone for Bool {
    fn clone(&self) -> Self {
        Bool {
            val: self.val.clone(),
            span: self.span.clone(),
        }
    }
}
impl Eq for Bool {}
impl PartialEq for Bool {
    fn eq(&self, x: &Self) -> bool {
        self.val == x.val
    }
}
impl Pretty for Bool {
    fn pretty(&self, p: &mut Print) {
        p.word(if self.val { "true" } else { "false" });
    }
}
impl<F: Folder + ?Sized> Fold for Bool {
    fn fold(&self, f: &mut F) {
        Bool {
            val: self.val,
            span: self.span.fold(f),
        }
    }
}
impl<H: Hasher> Hash for Bool {
    fn hash(&self, h: &mut H) {
        self.val.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Bool {
    fn visit(&self, v: &mut V) {
        &self.span.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut self.span.visit_mut(v);
    }
}
impl Debug for Bool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Bool {
            pub fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                f.debug_struct(name).field("value", &self.val).finish()
            }
        }
        self.debug(f, "lit::Bool")
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
        pub fn $ty(x: look::Marker) -> $ty {
            match x {}
        }
    };
}

pub struct Byte {
    pub repr: Box<Repr>,
}
impl Byte {
    pub fn new(x: u8, s: pm2::Span) -> Self {
        let mut tok = pm2::Lit::u8_suffixed(x);
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
    pub fn span(&self) -> pm2::Span {
        self.repr.tok.span()
    }
    pub fn set_span(&mut self, x: pm2::Span) {
        self.repr.tok.set_span(x);
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suff
    }
    pub fn token(&self) -> pm2::Lit {
        self.repr.tok.clone()
    }
}
extra_traits!(Byte);
impl Parse for Byte {
    fn parse(s: Stream) -> Res<Self> {
        let x = s.fork();
        match s.parse() {
            Ok(Lit::Byte(x)) => Ok(x),
            _ => Err(x.error("expected byte literal")),
        }
    }
}
impl Lower for Byte {
    fn lower(&self, s: &mut Stream) {
        self.repr.tok.lower(s);
    }
}
impl Debug for Byte {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Byte {
            pub fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                f.debug_struct(name)
                    .field("tok", &format_args!("{}", self.repr.tok))
                    .finish()
            }
        }
        self.debug(f, "lit::Byte")
    }
}
impl Eq for Byte {}
impl Pretty for Byte {
    fn pretty(&self, p: &mut Print) {
        p.word(self.token().to_string());
    }
}
impl<F: Folder + ?Sized> Fold for Byte {
    fn fold(&self, f: &mut F) {
        let span = self.span().fold(f);
        let mut y = self;
        self.set_span(span);
        y
    }
}
impl<V: Visitor + ?Sized> Visit for Byte {
    fn visit(&self, v: &mut V) {}
    fn visit_mut(&mut self, v: &mut V) {}
}

pub struct ByteStr {
    pub repr: Box<Repr>,
}
impl ByteStr {
    pub fn new(x: &[u8], s: pm2::Span) -> Self {
        let mut tok = pm2::Lit::byte_string(x);
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
    pub fn span(&self) -> pm2::Span {
        self.repr.tok.span()
    }
    pub fn set_span(&mut self, s: pm2::Span) {
        self.repr.tok.set_span(s);
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suff
    }
    pub fn token(&self) -> pm2::Lit {
        self.repr.tok.clone()
    }
}
extra_traits!(ByteStr);
impl Parse for ByteStr {
    fn parse(s: Stream) -> Res<Self> {
        let x = s.fork();
        match s.parse() {
            Ok(Lit::ByteStr(x)) => Ok(x),
            _ => Err(x.error("expected byte string literal")),
        }
    }
}
impl Lower for ByteStr {
    fn lower(&self, s: &mut Stream) {
        self.repr.tok.lower(s);
    }
}
impl Debug for ByteStr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl ByteStr {
            pub fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                f.debug_struct(x)
                    .field("tok", &format_args!("{}", self.repr.tok))
                    .finish()
            }
        }
        self.debug(f, "lit::ByteStr")
    }
}
impl Eq for ByteStr {}
impl Pretty for ByteStr {
    fn pretty(&self, p: &mut Print) {
        p.word(self.token().to_string());
    }
}
impl<F: Folder + ?Sized> Fold for ByteStr {
    fn fold(&self, f: &mut F) {
        let span = self.span().fold(f);
        let mut y = self;
        self.set_span(span);
        y
    }
}
impl<V: Visitor + ?Sized> Visit for ByteStr {
    fn visit(&self, v: &mut V) {}
    fn visit_mut(&mut self, v: &mut V) {}
}

pub struct Char {
    pub repr: Box<Repr>,
}
impl Char {
    pub fn new(x: char, s: pm2::Span) -> Self {
        let mut tok = pm2::Lit::char(x);
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
    pub fn span(&self) -> pm2::Span {
        self.repr.tok.span()
    }
    pub fn set_span(&mut self, s: pm2::Span) {
        self.repr.tok.set_span(s);
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suff
    }
    pub fn token(&self) -> pm2::Lit {
        self.repr.tok.clone()
    }
}
extra_traits!(Char);
impl Parse for Char {
    fn parse(s: Stream) -> Res<Self> {
        let x = s.fork();
        match s.parse() {
            Ok(Lit::Char(x)) => Ok(x),
            _ => Err(x.error("expected character literal")),
        }
    }
}
impl Lower for Char {
    fn lower(&self, s: &mut Stream) {
        self.repr.tok.lower(s);
    }
}
impl Debug for Char {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Char {
            pub fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                f.debug_struct(x)
                    .field("tok", &format_args!("{}", self.repr.tok))
                    .finish()
            }
        }
        self.debug(f, "lit::Char")
    }
}
impl Eq for Char {}
impl Pretty for Char {
    fn pretty(&self, p: &mut Print) {
        p.word(self.token().to_string());
    }
}
impl<F: Folder + ?Sized> Fold for Char {
    fn fold(&self, f: &mut F) {
        let span = self.span().fold(f);
        let mut y = self;
        self.set_span(span);
        y
    }
}
impl<V: Visitor + ?Sized> Visit for Char {
    fn visit(&self, v: &mut V) {}
    fn visit_mut(&mut self, v: &mut V) {}
}

pub struct Float {
    pub repr: Box<FloatRepr>,
}
impl Float {
    pub fn new(x: &str, s: pm2::Span) -> Self {
        let (digits, suff) = match parse_float(x) {
            Some(x) => x,
            None => panic!("Not a float literal: `{}`", x),
        };
        let mut tok: pm2::Lit = x.parse().unwrap();
        tok.set_span(s);
        Float {
            repr: Box::new(FloatRepr { tok, digits, suff }),
        }
    }
    pub fn base10_digits(&self) -> &str {
        &self.repr.digits
    }
    pub fn base10_parse<N>(&self) -> Res<N>
    where
        N: FromStr,
        N::Err: Display,
    {
        self.base10_digits().parse().map_err(|err| Err::new(self.span(), err))
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suff
    }
    pub fn span(&self) -> pm2::Span {
        self.repr.tok.span()
    }
    pub fn set_span(&mut self, s: pm2::Span) {
        self.repr.tok.set_span(s);
    }
    pub fn token(&self) -> pm2::Lit {
        self.repr.tok.clone()
    }
}
impl From<pm2::Lit> for Float {
    fn from(tok: pm2::Lit) -> Self {
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
extra_traits!(Float);
impl Parse for Float {
    fn parse(s: Stream) -> Res<Self> {
        let x = s.fork();
        match s.parse() {
            Ok(Lit::Float(x)) => Ok(x),
            _ => Err(x.error("expected floating point literal")),
        }
    }
}
impl Lower for Float {
    fn lower(&self, s: &mut Stream) {
        self.repr.tok.lower(s);
    }
}
impl Debug for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Float {
            pub fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                f.debug_struct(x)
                    .field("tok", &format_args!("{}", self.repr.tok))
                    .finish()
            }
        }
        self.debug(f, "lit::Float")
    }
}
impl Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.repr.tok.fmt(f)
    }
}
impl Eq for Float {}
impl Pretty for Float {
    fn pretty(&self, p: &mut Print) {
        p.word(self.token().to_string());
    }
}
impl<F: Folder + ?Sized> Fold for Float {
    fn fold(&self, f: &mut F) {
        let span = self.span().fold(f);
        let mut y = self;
        self.set_span(span);
        y
    }
}
impl<V: Visitor + ?Sized> Visit for Float {
    fn visit(&self, v: &mut V) {}
    fn visit_mut(&mut self, v: &mut V) {}
}

pub struct Int {
    pub repr: Box<IntRepr>,
}
impl Int {
    pub fn new(x: &str, s: pm2::Span) -> Self {
        let (digits, suff) = match parse_int(x) {
            Some(x) => x,
            None => panic!("Not an integer literal: `{}`", x),
        };
        let mut tok: pm2::Lit = x.parse().unwrap();
        tok.set_span(s);
        Int {
            repr: Box::new(IntRepr { tok, digits, suff }),
        }
    }
    pub fn base10_digits(&self) -> &str {
        &self.repr.digits
    }
    pub fn base10_parse<N>(&self) -> Res<N>
    where
        N: FromStr,
        N::Err: Display,
    {
        self.base10_digits().parse().map_err(|x| Err::new(self.span(), x))
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suff
    }
    pub fn span(&self) -> pm2::Span {
        self.repr.tok.span()
    }
    pub fn set_span(&mut self, s: pm2::Span) {
        self.repr.tok.set_span(s);
    }
    pub fn token(&self) -> pm2::Lit {
        self.repr.tok.clone()
    }
}
impl From<pm2::Lit> for Int {
    fn from(tok: pm2::Lit) -> Self {
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
extra_traits!(Int);
impl Parse for Int {
    fn parse(s: Stream) -> Res<Self> {
        let x = s.fork();
        match s.parse() {
            Ok(Lit::Int(x)) => Ok(x),
            _ => Err(x.error("expected integer literal")),
        }
    }
}
impl Lower for Int {
    fn lower(&self, s: &mut Stream) {
        self.repr.tok.lower(s);
    }
}
impl Debug for Int {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Int {
            pub fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                f.debug_struct(x)
                    .field("tok", &format_args!("{}", self.repr.tok))
                    .finish()
            }
        }
        self.debug(f, "LitInt")
    }
}
impl Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.repr.tok.fmt(f)
    }
}
impl Eq for Int {}
impl Pretty for Int {
    fn pretty(&self, p: &mut Print) {
        p.word(self.token().to_string());
    }
}
impl<F: Folder + ?Sized> Fold for Int {
    fn fold(&self, f: &mut F) {
        let span = self.span().fold(f);
        let mut y = self;
        self.set_span(span);
        y
    }
}
impl<V: Visitor + ?Sized> Visit for Int {
    fn visit(&self, v: &mut V) {}
    fn visit_mut(&mut self, v: &mut V) {}
}

pub struct Str {
    pub repr: Box<Repr>,
}
impl Str {
    pub fn new(x: &str, s: pm2::Span) -> Self {
        let mut tok = pm2::Lit::string(x);
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
        let (value, _) = parse::parse_str(&repr);
        String::from(value)
    }
    pub fn parse<T: parse::Parse>(&self) -> Res<T> {
        self.parse_with(T::parse)
    }
    pub fn parse_with<F: Parser>(&self, parser: F) -> Res<F::Output> {
        use pm2::Group;
        fn respan_token_stream(x: pm2::Stream, s: pm2::Span) -> pm2::Stream {
            x.into_iter().map(|x| respan_token_tree(x, s)).collect()
        }
        fn respan_token_tree(mut token: pm2::Tree, s: pm2::Span) -> pm2::Tree {
            match &mut token {
                pm2::Tree::Group(g) => {
                    let stream = respan_token_stream(g.stream(), s);
                    *g = Group::new(g.delim(), stream);
                    g.set_span(s);
                },
                other => other.set_span(s),
            }
            token
        }
        let mut tokens = pm2::Stream::from_str(&self.value())?;
        tokens = respan_token_stream(tokens, self.span());
        parser.parse2(tokens)
    }
    pub fn span(&self) -> pm2::Span {
        self.repr.tok.span()
    }
    pub fn set_span(&mut self, s: pm2::Span) {
        self.repr.tok.set_span(s);
    }
    pub fn suffix(&self) -> &str {
        &self.repr.suff
    }
    pub fn token(&self) -> pm2::Lit {
        self.repr.tok.clone()
    }
}
extra_traits!(Str);
impl Parse for Str {
    fn parse(s: Stream) -> Res<Self> {
        let x = s.fork();
        match s.parse() {
            Ok(Lit::Str(x)) => Ok(x),
            _ => Err(x.error("expected string literal")),
        }
    }
}
impl Lower for Str {
    fn lower(&self, s: &mut Stream) {
        self.repr.tok.lower(s);
    }
}
impl Debug for Str {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Str {
            pub fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                f.debug_struct(x)
                    .field("tok", &format_args!("{}", self.repr.tok))
                    .finish()
            }
        }
        self.debug(f, "lit::Str")
    }
}
impl Eq for Str {}
impl Pretty for Str {
    fn pretty(&self, p: &mut Print) {
        p.word(self.token().to_string());
    }
}
impl<F: Folder + ?Sized> Fold for Str {
    fn fold(&self, f: &mut F) {
        let span = self.span().fold(f);
        let mut y = self;
        self.set_span(span);
        y
    }
}
impl<V: Visitor + ?Sized> Visit for Str {
    fn visit(&self, v: &mut V) {}
    fn visit_mut(&mut self, v: &mut V) {}
}

struct Repr {
    pub tok: pm2::Lit,
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

struct IntRepr {
    pub tok: pm2::Lit,
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

struct FloatRepr {
    pub tok: pm2::Lit,
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

pub struct BigInt {
    digits: Vec<u8>,
}
impl BigInt {
    pub fn new() -> Self {
        BigInt { digits: Vec::new() }
    }
    pub fn to_string(&self) -> String {
        let mut y = String::with_capacity(self.digits.len());
        let mut nonzero = false;
        for x in self.digits.iter().rev() {
            nonzero |= *x != 0;
            if nonzero {
                y.push((*x + b'0') as char);
            }
        }
        if y.is_empty() {
            y.push('0');
        }
        y
    }
    fn reserve_two(&mut self) {
        let len = self.digits.len();
        let y = len + !self.digits.ends_with(&[0, 0]) as usize + !self.digits.ends_with(&[0]) as usize;
        self.digits.resize(y, 0);
    }
}
impl ops::AddAssign<u8> for BigInt {
    fn add_assign(&mut self, mut inc: u8) {
        self.reserve_two();
        let mut i = 0;
        while inc > 0 {
            let y = self.digits[i] + inc;
            self.digits[i] = y % 10;
            inc = y / 10;
            i += 1;
        }
    }
}
impl ops::MulAssign<u8> for BigInt {
    fn mul_assign(&mut self, base: u8) {
        self.reserve_two();
        let mut carry = 0;
        for x in &mut self.digits {
            let y = *x * base + carry;
            *x = y % 10;
            carry = y / 10;
        }
    }
}

#[allow(non_snake_case)]
pub fn Lit(x: look::Marker) -> Lit {
    match x {}
}

#[allow(non_snake_case)]
pub fn Bool(x: look::Marker) -> Bool {
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
fn parse_negative(neg: Punct, cursor: Cursor) -> Option<(Lit, Cursor)> {
    let (lit, rest) = cursor.literal()?;
    let mut span = neg.span();
    span = span.join(lit.span()).unwrap_or(span);
    let mut repr = lit.to_string();
    repr.insert(0, '-');
    if let Some((digits, suff)) = parse_int(&repr) {
        let mut tok: pm2::Lit = repr.parse().unwrap();
        tok.set_span(span);
        return Some((
            Lit::Int(Int {
                repr: Box::new(IntRepr { tok, digits, suff }),
            }),
            rest,
        ));
    }
    let (digits, suff) = parse_float(&repr)?;
    let mut tok: pm2::Lit = repr.parse().unwrap();
    tok.set_span(span);
    Some((
        Lit::Float(Float {
            repr: Box::new(FloatRepr { tok, digits, suff }),
        }),
        rest,
    ))
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
