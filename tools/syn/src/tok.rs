use super::*;

pub trait Tok {
    fn peek(x: Cursor) -> bool;
    fn display() -> &'static str;
}
impl Tok for Ident {
    fn peek(c: Cursor) -> bool {
        if let Some((x, _)) = c.ident() {
            accept_as_ident(&x)
        } else {
            false
        }
    }
    fn display() -> &'static str {
        "identifier"
    }
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
impl_tok!("lifetime" Life);
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
    ($d:literal $get:ident $n:ty) => {
        impl Tok for $n {
            fn peek(x: Cursor) -> bool {
                x.$get().is_some()
            }
            fn display() -> &'static str {
                $d
            }
        }
    };
}
impl_low_level!("punct token" punct Punct);
impl_low_level!("literal" literal pm2::Lit);
impl_low_level!("token" token_tree pm2::Tree);

fn peek_impl(x: Cursor, f: fn(Stream) -> bool) -> bool {
    use parse::Unexpected;
    use std::{cell::Cell, rc::Rc};
    let scope = pm2::Span::call_site();
    let unexpected = Rc::new(Cell::new(Unexpected::None));
    let y = crate::parse::new_parse_buffer(scope, x, unexpected);
    f(&y)
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

pub fn kw_to_toks(txt: &str, span: pm2::Span, s: &mut Stream) {
    s.append(Ident::new(txt, span));
}

macro_rules! def_keywords {
    ($($t:literal $n:ident)*) => {
        $(
            pub struct $n {
                pub span: pm2::Span,
            }
            pub fn $n<S: pm2::IntoSpans<pm2::Span>>(s: S) -> $n {
                $n {                    span: s.into_spans()                }
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
            impl std::cmp::Eq for $n {}
            impl PartialEq for $n {
                fn eq(&self, _: &$n) -> bool {
                    true
                }
            }
            impl Hash for $n {
                fn hash<H: Hasher>(&self, _: &mut H) {}
            }
            impl Parse for $n {
                fn parse(s: parse::Stream) -> Res<Self> {
                    Ok($n {
                        span: crate::tok::parse_kw(s, $t)?,
                    })
                }
            }
            impl Lower for $n {
                fn lower(&self, s: &mut pm2::Stream) {
                    crate::tok::kw_to_toks($t, self.span, s);
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
def_keywords! {
    "abstract"    Abstract
    "as"          As
    "async"       Async
    "auto"        Auto
    "await"       Await
    "become"      Become
    "box"         Box
    "break"       Break
    "const"       Const
    "continue"    Continue
    "crate"       Crate
    "default"     Default
    "do"          Do
    "dyn"         Dyn
    "else"        Else
    "enum"        Enum
    "extern"      Extern
    "final"       Final
    "fn"          Fn
    "for"         For
    "if"          If
    "impl"        Impl
    "in"          In
    "let"         Let
    "loop"        Loop
    "macro"       Macro
    "match"       Match
    "mod"         Mod
    "move"        Move
    "mut"         Mut
    "override"    Override
    "priv"        Priv
    "pub"         Pub
    "ref"         Ref
    "return"      Return
    "Self"        SelfType
    "self"        SelfValue
    "static"      Static
    "struct"      Struct
    "super"       Super
    "trait"       Trait
    "try"         Try
    "type"        Type
    "typeof"      Typeof
    "union"       Union
    "unsafe"      Unsafe
    "unsized"     Unsized
    "use"         Use
    "virtual"     Virtual
    "where"       Where
    "while"       While
    "yield"       Yield
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
            pub fn $n<S: pm2::IntoSpans<[pm2::Span; $len]>>(ss: S) -> $n {
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
            impl std::cmp::Eq for $n {}
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
impl Lower for Underscore {
    fn lower(&self, ts: &mut pm2::Stream) {
        ts.append(Ident::new("_", self.span));
    }
}
impl Parse for Underscore {
    fn parse(s: Stream) -> Res<Self> {
        s.step(|x| {
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

pub fn punct_lower(txt: &str, spans: &[pm2::Span], s: &mut Stream) {
    assert_eq!(txt.len(), spans.len());
    let mut chars = txt.chars();
    let mut spans = spans.iter();
    let ch = chars.next_back().unwrap();
    let span = spans.next_back().unwrap();
    for (ch, span) in chars.zip(spans) {
        let mut y = Punct::new(ch, pm2::Spacing::Joint);
        y.set_span(*span);
        s.append(y);
    }
    let mut y = Punct::new(ch, pm2::Spacing::Alone);
    y.set_span(*span);
    s.append(y);
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
                fn parse(s: Stream) -> Res<Self> {
                    Ok($n {
                        spans: crate::tok::parse_punct(s, $t)?,
                    })
                }
            }
            impl Lower for $n {
                fn lower(&self, s: &mut pm2::Stream) {
                    crate::tok::punct_lower($t, &self.spans, s);
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

pub fn delim_lower(d: pm2::Delim, span: pm2::Span, s: &mut Stream, inner: pm2::Stream) {
    let mut y = Group::new(d, inner);
    y.set_span(span);
    s.append(y);
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
                    crate::tok::delim_lower(pm2::Delim::$d, self.span.join(), ts, inner);
                }
            }
            #[allow(non_snake_case)]
            pub fn $n<S: pm2::IntoSpans<pm2::DelimSpan>>(s: S) -> $n {
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
            impl std::cmp::Eq for $n {}
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
    Parenth pub struct Parenth
}
impl Tok for Brace {
    fn peek(x: Cursor) -> bool {
        look::is_delim(x, pm2::Delim::Brace)
    }
    fn display() -> &'static str {
        "braces"
    }
}
impl Tok for Bracket {
    fn peek(x: Cursor) -> bool {
        look::is_delim(x, pm2::Delim::Bracket)
    }
    fn display() -> &'static str {
        "brackets"
    }
}
impl Tok for Parenth {
    fn peek(x: Cursor) -> bool {
        look::is_delim(x, pm2::Delim::Parenth)
    }
    fn display() -> &'static str {
        "parentheses"
    }
}

pub enum Delim {
    Brace(Brace),
    Bracket(Bracket),
    Parenth(Parenth),
}
impl Delim {
    pub fn span(&self) -> &pm2::DelimSpan {
        use Delim::*;
        match self {
            Brace(x) => &x.span,
            Bracket(x) => &x.span,
            Parenth(x) => &x.span,
        }
    }
    pub fn surround(&self, s: &mut Stream, inner: pm2::Stream) {
        use Delim::*;
        let (delim, span) = match self {
            Brace(x) => (pm2::Delim::Brace, x.span),
            Bracket(x) => (pm2::Delim::Bracket, x.span),
            Parenth(x) => (pm2::Delim::Parenth, x.span),
        };
        delim_lower(delim, span.join(), s, inner);
    }
    pub fn is_brace(&self) -> bool {
        use Delim::*;
        match self {
            Brace(_) => true,
            Bracket(_) | Parenth(_) => false,
        }
    }
    pub fn pretty_open(&self, p: &mut Print) {
        use Delim::*;
        p.word(match self {
            Brace(_) => "{",
            Bracket(_) => "[",
            Parenth(_) => "(",
            None => return,
        });
    }
    pub fn pretty_close(&self, p: &mut Print) {
        use Delim::*;
        p.word(match self {
            Brace(_) => "}",
            Bracket(_) => "]",
            Parenth(_) => ")",
            None => return,
        });
    }
}
impl Clone for Delim {
    fn clone(&self) -> Self {
        use Delim::*;
        match self {
            Brace(x) => Brace(x.clone()),
            Bracket(x) => Bracket(x.clone()),
            Parenth(x) => Parenth(x.clone()),
        }
    }
}
impl Debug for tok::Delim {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Delim::")?;
        use Delim::*;
        match self {
            Brace(x) => {
                let mut f = f.debug_tuple("Brace");
                f.field(x);
                f.finish()
            },
            Bracket(x) => {
                let mut f = f.debug_tuple("Bracket");
                f.field(x);
                f.finish()
            },
            Parenth(x) => {
                let mut f = f.debug_tuple("Parenth");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl std::cmp::Eq for Delim {}
impl PartialEq for Delim {
    fn eq(&self, x: &Self) -> bool {
        use Delim::*;
        match (self, x) {
            (Brace(_), Brace(_)) => true,
            (Bracket(_), Bracket(_)) => true,
            (Parenth(_), Parenth(_)) => true,
            _ => false,
        }
    }
}
impl<F: Folder + ?Sized> Fold for tok::Delim {
    fn fold(&self, f: &mut F) {
        use Delim::*;
        match self {
            Brace(x) => Brace(x),
            Bracket(x) => Bracket(x),
            Parenth(x) => Parenth(x),
        }
    }
}
impl<H: Hasher> Hash for Delim {
    fn hash(&self, h: &mut H) {
        use Delim::*;
        match self {
            Parenth(_) => {
                h.write_u8(0u8);
            },
            Brace(_) => {
                h.write_u8(1u8);
            },
            Bracket(_) => {
                h.write_u8(2u8);
            },
        }
    }
}
impl<V: Visitor + ?Sized> Visit for Delim {
    fn visit(&self, v: &mut V) {
        use Delim::*;
        match self {
            Brace(_) => {},
            Bracket(_) => {},
            Parenth(_) => {},
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        use Delim::*;
        match self {
            Brace(_) => {},
            Bracket(_) => {},
            Parenth(_) => {},
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
        delim_lower(pm2::Delim::None, self.span, ts, inner);
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
impl std::cmp::Eq for Group {}
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
pub fn Group<S: pm2::IntoSpans<pm2::Span>>(s: S) -> Group {
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

pub fn parse_kw(s: Stream, txt: &str) -> Res<pm2::Span> {
    s.step(|x| {
        if let Some((y, rest)) = x.ident() {
            if y == txt {
                return Ok((y.span(), rest));
            }
        }
        Err(x.error(format!("expected `{}`", txt)))
    })
}
pub fn peek_kw(c: Cursor, txt: &str) -> bool {
    if let Some((x, _)) = c.ident() {
        x == txt
    } else {
        false
    }
}

pub fn parse_punct<const N: usize>(s: Stream, txt: &str) -> Res<[pm2::Span; N]> {
    fn doit(s: Stream, txt: &str, ys: &mut [pm2::Span]) -> Res<()> {
        s.step(|c| {
            let mut c = *c;
            assert_eq!(txt.len(), ys.len());
            for (i, x) in txt.chars().enumerate() {
                match c.punct() {
                    Some((y, rest)) => {
                        ys[i] = y.span();
                        if y.as_char() != x {
                            break;
                        } else if i == txt.len() - 1 {
                            return Ok(((), rest));
                        } else if y.spacing() != pm2::Spacing::Joint {
                            break;
                        }
                        c = rest;
                    },
                    None => break,
                }
            }
            Err(Err::new(ys[0], format!("expected `{}`", txt)))
        })
    }
    let mut ys = [s.span(); N];
    doit(s, txt, &mut ys)?;
    Ok(ys)
}
pub fn peek_punct(mut c: Cursor, txt: &str) -> bool {
    for (i, x) in txt.chars().enumerate() {
        match c.punct() {
            Some((y, rest)) => {
                if y.as_char() != x {
                    break;
                } else if i == txt.len() - 1 {
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
