use super::*;

pub trait Tok {
    fn peek(x: Cursor) -> bool;
    fn display() -> &'static str;
}

fn peek_impl(x: Cursor, f: fn(Stream) -> bool) -> bool {
    use crate::parse::Unexpected;
    use std::{cell::Cell, rc::Rc};
    let scope = pm2::Span::call_site();
    let unexpected = Rc::new(Cell::new(Unexpected::None));
    let y = crate::parse::new_parse_buffer(scope, x, unexpected);
    f(&y)
}
macro_rules! impl_tok {
    ($d:literal $n:ty) => {
        impl Tok for $n {
            fn peek(x: Cursor) -> bool {
                fn peek(x: parse::Stream) -> bool {
                    <$n as Parse>::parse(x).is_ok()
                }
                peek_impl(x, peek)
            }
            fn display() -> &'static str {
                $d
            }
        }
    };
}
impl_tok!("life" Life);
impl_tok!("literal" lit::Lit);
impl_tok!("string literal" lit::Str);
impl_tok!("byte string literal" lit::ByteStr);
impl_tok!("byte literal" lit::Byte);
impl_tok!("character literal" lit::Char);
impl_tok!("integer literal" lit::Int);
impl_tok!("floating point literal" lit::Float);
impl_tok!("boolean literal" lit::Bool);
impl_tok!("group" pm2::Group);

macro_rules! impl_low_level {
    ($d:literal $ty:ident $get:ident) => {
        impl Tok for $ty {
            fn peek(x: Cursor) -> bool {
                x.$get().is_some()
            }
            fn display() -> &'static str {
                $d
            }
        }
    };
}
impl_low_level!("punctuation token" Punct punct);
impl_low_level!("literal" pm2::Lit literal);
impl_low_level!("token" pm2::Tree token_tree);

impl Tok for Ident {
    fn peek(x: Cursor) -> bool {
        if let Some((ident, _)) = x.ident() {
            accept_as_ident(&ident)
        } else {
            false
        }
    }
    fn display() -> &'static str {
        "identifier"
    }
}

pub trait Custom {
    fn peek(x: Cursor) -> bool;
    fn display() -> &'static str;
}
impl<T: Custom> Tok for T {
    fn peek(x: Cursor) -> bool {
        <Self as Custom>::peek(x)
    }
    fn display() -> &'static str {
        <Self as Custom>::display()
    }
}

pub fn kw_to_tokens(x: &str, s: pm2::Span, ys: &mut Stream) {
    ys.append(Ident::new(x, s));
}

macro_rules! def_kws {
    ($($t:literal pub struct $n:ident)*) => {
        $(
            pub struct $n {
                pub span: pm2::Span,
            }
            #[allow(non_snake_case)]
            pub fn $n<S: IntoSpans<pm2::Span>>(s: S) -> $n {
                $n {
                    span: s.into_spans(),
                }
            }
            impl std::default::Default for $n {
                fn default() -> Self {
                    $n {
                        span: pm2::Span::call_site(),
                    }
                }
            }
            impl Copy for $n {}
            impl Clone for $n {
                fn clone(&self) -> Self {
                    *self
                }
            }
            impl Debug for $n {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str(stringify!($n))
                }
            }
            impl cmp::Eq for $n {}
            impl PartialEq for $n {
                fn eq(&self, _: &$n) -> bool {
                    true
                }
            }
            impl Hash for $n {
                fn hash<H: Hasher>(&self, _: &mut H) {}
            }
            impl Parse for $n {
                fn parse(x: parse::Stream) -> Res<Self> {
                    Ok($n {
                        span: crate::tok::kw_to_tokens(x, $t)?,
                    })
                }
            }
            impl ToTokens for $n {
                fn to_tokens(&self, ys: &mut pm2::Stream) {
                    crate::tok::kw_to_tokens($t, self.span, ys);
                }
            }
            impl Tok for $n {
                fn peek(x: Cursor) -> bool {
                    crate::tok::peek_kw(x, $t)
                }
                fn display() -> &'static str {
                    concat!("`", $t, "`")
                }
            }
        )*
    };
}
def_kws! {
    "abstract"    pub struct Abstract
    "as"          pub struct As
    "async"       pub struct Async
    "auto"        pub struct Auto
    "await"       pub struct Await
    "become"      pub struct Become
    "box"         pub struct Box
    "break"       pub struct Break
    "const"       pub struct Const
    "continue"    pub struct Continue
    "crate"       pub struct Crate
    "default"     pub struct Default
    "do"          pub struct Do
    "dyn"         pub struct Dyn
    "else"        pub struct Else
    "enum"        pub struct Enum
    "extern"      pub struct Extern
    "final"       pub struct Final
    "fn"          pub struct Fn
    "for"         pub struct For
    "if"          pub struct If
    "impl"        pub struct Impl
    "in"          pub struct In
    "let"         pub struct Let
    "loop"        pub struct Loop
    "macro"       pub struct Macro
    "match"       pub struct Match
    "mod"         pub struct Mod
    "move"        pub struct Move
    "mut"         pub struct Mut
    "override"    pub struct Override
    "priv"        pub struct Priv
    "pub"         pub struct Pub
    "ref"         pub struct Ref
    "return"      pub struct Return
    "Self"        pub struct SelfType
    "self"        pub struct SelfValue
    "static"      pub struct Static
    "struct"      pub struct Struct
    "super"       pub struct Super
    "trait"       pub struct Trait
    "try"         pub struct Try
    "type"        pub struct Type
    "typeof"      pub struct Typeof
    "union"       pub struct Union
    "unsafe"      pub struct Unsafe
    "unsized"     pub struct Unsized
    "use"         pub struct Use
    "virtual"     pub struct Virtual
    "where"       pub struct Where
    "while"       pub struct While
    "yield"       pub struct Yield
}

macro_rules! impl_deref_len_1 {
    ($n:ident/1) => {
        impl Deref for $n {
            type Target = WithSpan;
            fn deref(&self) -> &Self::Target {
                unsafe { &*(self as *const Self).cast::<WithSpan>() }
            }
        }
        impl DerefMut for $n {
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { &mut *(self as *mut Self).cast::<WithSpan>() }
            }
        }
    };
    ($n:ident/$x:literal) => {};
}

macro_rules! def_punct_structs {
    ($($t:literal pub struct $n:ident/$len:tt)*) => {
        $(
            pub struct $n {
                pub spans: [pm2::Span; $len],
            }
            #[allow(non_snake_case)]
            pub fn $n<S: IntoSpans<[pm2::Span; $len]>>(ss: S) -> $n {
                $n {
                    spans: ss.into_spans(),
                }
            }
            impl std::default::Default for $n {
                fn default() -> Self {
                    $n {
                        spans: [pm2::Span::call_site(); $len],
                    }
                }
            }
            impl Copy for $n {}
            impl Clone for $n {
                fn clone(&self) -> Self {
                    *self
                }
            }
            impl Debug for $n {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str(stringify!($n))
                }
            }
            impl cmp::Eq for $n {}
            impl PartialEq for $n {
                fn eq(&self, _: &$n) -> bool {
                    true
                }
            }
            impl Hash for $n {
                fn hash<H: Hasher>(&self, _: &mut H) {}
            }
            impl_deref_len_1!($n/$len);
        )*
    };
}
def_punct_structs! {
    "_" pub struct Underscore/1
}
impl ToTokens for Underscore {
    fn to_tokens(&self, ts: &mut pm2::Stream) {
        ts.append(Ident::new("_", self.span));
    }
}
impl Parse for Underscore {
    fn parse(x: Stream) -> Res<Self> {
        x.step(|x| {
            if let Some((x, rest)) = x.ident() {
                if x == "_" {
                    return Ok((Underscore(x.span()), rest));
                }
            }
            if let Some((x, rest)) = x.punct() {
                if x.as_char() == '_' {
                    return Ok((Underscore(x.span()), rest));
                }
            }
            Err(x.error("expected `_`"))
        })
    }
}
impl Tok for Underscore {
    fn peek(x: Cursor) -> bool {
        if let Some((x, _)) = x.ident() {
            return x == "_";
        }
        if let Some((x, _)) = x.punct() {
            return x.as_char() == '_';
        }
        false
    }
    fn display() -> &'static str {
        "`_`"
    }
}

pub fn punct_to_tokens(x: &str, spans: &[pm2::Span], ys: &mut Stream) {
    assert_eq!(x.len(), spans.len());
    let mut chars = x.chars();
    let mut spans = spans.iter();
    let ch = chars.next_back().unwrap();
    let span = spans.next_back().unwrap();
    for (ch, span) in chars.zip(spans) {
        let mut y = Punct::new(ch, pm2::Spacing::Joint);
        y.set_span(*span);
        ys.append(y);
    }
    let mut y = Punct::new(ch, pm2::Spacing::Alone);
    y.set_span(*span);
    ys.append(y);
}

macro_rules! def_punct {
    ($($t:literal pub struct $n:ident/$len:tt)*) => {
        $(
            def_punct_structs! {
                $t pub struct $n/$len
            }
            impl Tok for $n {
                fn peek(x: Cursor) -> bool {
                    crate::tok::peek_punct(x, $t)
                }
                fn display() -> &'static str {
                    concat!("`", $t, "`")
                }
            }
            impl Parse for $n {
                fn parse(x: Stream) -> Res<Self> {
                    Ok($n {
                        spans: crate::tok::parse_punct(x, $t)?,
                    })
                }
            }
            impl ToTokens for $n {
                fn to_tokens(&self, ts: &mut pm2::Stream) {
                    crate::tok::punct_to_tokens($t, &self.spans, ts);
                }
            }
        )*
    };
}
def_punct! {
    "&"           pub struct And/1
    "&&"          pub struct AndAnd/2
    "&="          pub struct AndEq/2
    "@"           pub struct At/1
    "^"           pub struct Caret/1
    "^="          pub struct CaretEq/2
    ":"           pub struct Colon/1
    ","           pub struct Comma/1
    "$"           pub struct Dollar/1
    "."           pub struct Dot/1
    ".."          pub struct DotDot/2
    "..."         pub struct DotDotDot/3
    "..="         pub struct DotDotEq/3
    "="           pub struct Eq/1
    "=="          pub struct EqEq/2
    "=>"          pub struct FatArrow/2
    ">="          pub struct Ge/2
    ">"           pub struct Gt/1
    "<-"          pub struct LArrow/2
    "<="          pub struct Le/2
    "<"           pub struct Lt/1
    "-"           pub struct Minus/1
    "-="          pub struct MinusEq/2
    "!="          pub struct Ne/2
    "!"           pub struct Not/1
    "|"           pub struct Or/1
    "|="          pub struct OrEq/2
    "||"          pub struct OrOr/2
    "::"          pub struct PathSep/2
    "%"           pub struct Percent/1
    "%="          pub struct PercentEq/2
    "+"           pub struct Plus/1
    "+="          pub struct PlusEq/2
    "#"           pub struct Pound/1
    "?"           pub struct Question/1
    "->"          pub struct RArrow/2
    ";"           pub struct Semi/1
    "<<"          pub struct Shl/2
    "<<="         pub struct ShlEq/3
    ">>"          pub struct Shr/2
    ">>="         pub struct ShrEq/3
    "/"           pub struct Slash/1
    "/="          pub struct SlashEq/2
    "*"           pub struct Star/1
    "*="          pub struct StarEq/2
    "~"           pub struct Tilde/1
}

pub fn delim_to_tokens(d: pm2::Delim, s: pm2::Span, ys: &mut Stream, inner: pm2::Stream) {
    let mut y = Group::new(d, inner);
    y.set_span(s);
    ys.append(y);
}

macro_rules! def_delims {
    ($($d:ident pub struct $n:ident)*) => {
        $(
            pub struct $n {
                pub span: pm2::DelimSpan,
            }
            impl $n {
                pub fn surround<F>(&self, ts: &mut pm2::Stream, f: F)
                where
                    F: FnOnce(&mut pm2::Stream),
                {
                    let mut inner = pm2::Stream::new();
                    f(&mut inner);
                    crate::tok::delim_to_tokens(pm2::Delim::$d, self.span.join(), ts, inner);
                }
            }
            #[allow(non_snake_case)]
            pub fn $n<S: IntoSpans<pm2::DelimSpan>>(s: S) -> $n {
                $n {
                    span: s.into_spans(),
                }
            }
            impl std::default::Default for $n {
                fn default() -> Self {
                    $n(pm2::Span::call_site())
                }
            }
            impl Copy for $n {}
            impl Clone for $n {
                fn clone(&self) -> Self {
                    *self
                }
            }
            impl Debug for $n {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str(stringify!($n))
                }
            }
            impl cmp::Eq for $n {}
            impl PartialEq for $n {
                fn eq(&self, _: &$n) -> bool {
                    true
                }
            }
            impl Hash for $n {
                fn hash<H: Hasher>(&self, _: &mut H) {}
            }
        )*
    };
}
def_delims! {
    Brace pub struct Brace
    Bracket pub struct Bracket
    Paren pub struct Paren
}
impl Tok for Brace {
    fn peek(x: Cursor) -> bool {
        look::is_delim(x, pm2::Delim::Brace)
    }
    fn display() -> &'static str {
        "curly braces"
    }
}
impl Tok for Bracket {
    fn peek(x: Cursor) -> bool {
        look::is_delim(x, pm2::Delim::Bracket)
    }
    fn display() -> &'static str {
        "square brackets"
    }
}
impl Tok for Paren {
    fn peek(x: Cursor) -> bool {
        look::is_delim(x, pm2::Delim::Parenthesis)
    }
    fn display() -> &'static str {
        "parentheses"
    }
}

pub enum Delim {
    Paren(Paren),
    Brace(Brace),
    Bracket(Bracket),
}
impl Delim {
    pub fn span(&self) -> &pm2::DelimSpan {
        use Delim::*;
        match self {
            Paren(x) => &x.span,
            Brace(x) => &x.span,
            Bracket(x) => &x.span,
        }
    }
    pub fn surround(&self, ys: &mut Stream, inner: pm2::Stream) {
        let (delim, span) = match self {
            tok::Delim::Paren(x) => (pm2::Delim::Parenthesis, x.span),
            tok::Delim::Brace(x) => (pm2::Delim::Brace, x.span),
            tok::Delim::Bracket(x) => (pm2::Delim::Bracket, x.span),
        };
        delim_to_tokens(delim, span.join(), ys, inner);
    }
    pub fn is_brace(&self) -> bool {
        match self {
            tok::Delim::Brace(_) => true,
            tok::Delim::Paren(_) | tok::Delim::Bracket(_) => false,
        }
    }
}

pub struct Group {
    pub span: pm2::Span,
}
impl Group {
    pub fn surround<F>(&self, ts: &mut pm2::Stream, f: F)
    where
        F: FnOnce(&mut pm2::Stream),
    {
        let mut inner = pm2::Stream::new();
        f(&mut inner);
        delim_to_tokens(pm2::Delim::None, self.span, ts, inner);
    }
}
impl std::default::Default for Group {
    fn default() -> Self {
        Group {
            span: pm2::Span::call_site(),
        }
    }
}
impl Copy for Group {}
impl Clone for Group {
    fn clone(&self) -> Self {
        *self
    }
}
impl Debug for Group {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Group")
    }
}
impl cmp::Eq for Group {}
impl PartialEq for Group {
    fn eq(&self, _: &Group) -> bool {
        true
    }
}
impl Hash for Group {
    fn hash<H: Hasher>(&self, _: &mut H) {}
}
impl Tok for Group {
    fn peek(x: Cursor) -> bool {
        look::is_delim(x, pm2::Delim::None)
    }
    fn display() -> &'static str {
        "invisible group"
    }
}

#[allow(non_snake_case)]
pub fn Group<S: IntoSpans<pm2::Span>>(s: S) -> Group {
    Group { span: s.into_spans() }
}

pub struct WithSpan {
    pub span: pm2::Span,
}

pub fn accept_as_ident(x: &Ident) -> bool {
    match x.to_string().as_str() {
        "_" | "abstract" | "as" | "async" | "await" | "become" | "box" | "break" | "const" | "continue" | "crate"
        | "do" | "dyn" | "else" | "enum" | "extern" | "false" | "final" | "fn" | "for" | "if" | "impl" | "in"
        | "let" | "loop" | "macro" | "match" | "mod" | "move" | "mut" | "override" | "priv" | "pub" | "ref"
        | "return" | "Self" | "self" | "static" | "struct" | "super" | "trait" | "true" | "try" | "type" | "typeof"
        | "unsafe" | "unsized" | "use" | "virtual" | "where" | "while" | "yield" => false,
        _ => true,
    }
}

pub fn parse_kw(s: Stream, token: &str) -> Res<pm2::Span> {
    s.step(|x| {
        if let Some((y, rest)) = x.ident() {
            if y == token {
                return Ok((y.span(), rest));
            }
        }
        Err(x.error(format!("expected `{}`", token)))
    })
}
pub fn peek_kw(c: Cursor, token: &str) -> bool {
    if let Some((x, _)) = c.ident() {
        x == token
    } else {
        false
    }
}

pub fn parse_punct<const N: usize>(s: Stream, token: &str) -> Res<[pm2::Span; N]> {
    fn doit(s: Stream, token: &str, spans: &mut [pm2::Span]) -> Res<()> {
        s.step(|c| {
            let mut c = *c;
            assert_eq!(token.len(), spans.len());
            for (i, x) in token.chars().enumerate() {
                match c.punct() {
                    Some((y, rest)) => {
                        spans[i] = y.span();
                        if y.as_char() != x {
                            break;
                        } else if i == token.len() - 1 {
                            return Ok(((), rest));
                        } else if y.spacing() != pm2::Spacing::Joint {
                            break;
                        }
                        c = rest;
                    },
                    None => break,
                }
            }
            Err(Err::new(spans[0], format!("expected `{}`", token)))
        })
    }
    let mut ys = [s.span(); N];
    doit(s, token, &mut ys)?;
    Ok(ys)
}
pub fn peek_punct(mut c: Cursor, token: &str) -> bool {
    for (i, x) in token.chars().enumerate() {
        match c.punct() {
            Some((y, rest)) => {
                if y.as_char() != x {
                    break;
                } else if i == token.len() - 1 {
                    return true;
                } else if y.spacing() != pm2::Spacing::Joint {
                    break;
                }
                c = rest;
            },
            None => break,
        }
    }
    false
}
