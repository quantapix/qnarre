use super::{
    err::Result,
    lit,
    parse::{Parse, ParseStream},
    *,
};
use proc_macro2::{extra::DelimSpan, Delimiter, Ident, Literal, Punct, Span, TokenStream, TokenTree};
use quote::{ToTokens, TokenStreamExt};
use std::{
    cmp,
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
};

pub trait Tok: private::Sealed {
    fn peek(x: Cursor) -> bool;
    fn display() -> &'static str;
}

fn peek_impl(x: Cursor, peek: fn(ParseStream) -> bool) -> bool {
    use crate::parse::Unexpected;
    use std::{cell::Cell, rc::Rc};
    let scope = Span::call_site();
    let unexpected = Rc::new(Cell::new(Unexpected::None));
    let y = crate::parse::new_parse_buffer(scope, x, unexpected);
    peek(&y)
}
macro_rules! impl_tok {
    ($d:literal $n:ty) => {
        impl Tok for $n {
            fn peek(x: Cursor) -> bool {
                fn peek(x: ParseStream) -> bool {
                    <$n as Parse>::parse(x).is_ok()
                }
                peek_impl(x, peek)
            }
            fn display() -> &'static str {
                $d
            }
        }
        impl private::Sealed for $n {}
    };
}
impl_tok!("lifetime" Lifetime);
impl_tok!("literal" Lit);
impl_tok!("string literal" lit::Str);
impl_tok!("byte string literal" lit::ByteStr);
impl_tok!("byte literal" lit::Byte);
impl_tok!("character literal" lit::Char);
impl_tok!("integer literal" lit::Int);
impl_tok!("floating point literal" lit::Float);
impl_tok!("boolean literal" lit::Bool);
impl_tok!("group" proc_macro2::Group);

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
        impl private::Sealed for $ty {}
    };
}
impl_low_level!("punctuation token" Punct punct);
impl_low_level!("literal" Literal literal);
impl_low_level!("token" TokenTree token_tree);

pub trait CustomTok {
    fn peek(x: Cursor) -> bool;
    fn display() -> &'static str;
}
impl<T: CustomTok> private::Sealed for T {}
impl<T: CustomTok> Tok for T {
    fn peek(x: Cursor) -> bool {
        <Self as CustomTok>::peek(x)
    }
    fn display() -> &'static str {
        <Self as CustomTok>::display()
    }
}

macro_rules! def_keywords {
    ($($t:literal pub struct $n:ident)*) => {
        $(
            pub struct $n {
                pub span: Span,
            }
            #[allow(non_snake_case)]
            pub fn $n<S: IntoSpans<Span>>(s: S) -> $n {
                $n {
                    span: s.into_spans(),
                }
            }
            impl std::default::Default for $n {
                fn default() -> Self {
                    $n {
                        span: Span::call_site(),
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
            impl ToTokens for $n {
                fn to_tokens(&self, toks: &mut TokenStream) {
                    crate::dump::keyword($t, self.span, toks);
                }
            }
            impl Parse for $n {
                fn parse(x: ParseStream) -> Result<Self> {
                    Ok($n {
                        span: parsing::keyword(x, $t)?,
                    })
                }
            }
            impl Tok for $n {
                fn peek(x: Cursor) -> bool {
                    parsing::peek_keyword(x, $t)
                }
                fn display() -> &'static str {
                    concat!("`", $t, "`")
                }
            }
            impl private::Sealed for $n {}
        )*
    };
}
def_keywords! {
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
            type Target = private::WithSpan;
            fn deref(&self) -> &Self::Target {
                unsafe { &*(self as *const Self).cast::<private::WithSpan>() }
            }
        }
        impl DerefMut for $n {
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { &mut *(self as *mut Self).cast::<private::WithSpan>() }
            }
        }
    };
    ($n:ident/$x:literal) => {};
}

macro_rules! def_punct_structs {
    ($($t:literal pub struct $n:ident/$len:tt)*) => {
        $(
            pub struct $n {
                pub spans: [Span; $len],
            }
            #[allow(non_snake_case)]
            pub fn $n<S: IntoSpans<[Span; $len]>>(ss: S) -> $n {
                $n {
                    spans: ss.into_spans(),
                }
            }
            impl std::default::Default for $n {
                fn default() -> Self {
                    $n {
                        spans: [Span::call_site(); $len],
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
    fn to_tokens(&self, ts: &mut TokenStream) {
        ts.append(Ident::new("_", self.span));
    }
}
impl Parse for Underscore {
    fn parse(x: ParseStream) -> Result<Self> {
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
impl private::Sealed for Underscore {}

macro_rules! def_punct {
    ($($t:literal pub struct $n:ident/$len:tt)*) => {
        $(
            def_punct_structs! {
                $t pub struct $n/$len
            }
            impl ToTokens for $n {
                fn to_tokens(&self, ts: &mut TokenStream) {
                    crate::dump::punct($t, &self.spans, ts);
                }
            }
            impl Parse for $n {
                fn parse(x: ParseStream) -> Result<Self> {
                    Ok($n {
                        spans: parsing::punct(x, $t)?,
                    })
                }
            }
            impl Tok for $n {
                fn peek(x: Cursor) -> bool {
                    parsing::peek_punct(x, $t)
                }
                fn display() -> &'static str {
                    concat!("`", $t, "`")
                }
            }
            impl private::Sealed for $n {}
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

macro_rules! def_delims {
    ($($d:ident pub struct $n:ident)*) => {
        $(
            pub struct $n {
                pub span: DelimSpan,
            }
            #[allow(non_snake_case)]
            pub fn $n<S: IntoSpans<DelimSpan>>(s: S) -> $n {
                $n {
                    span: s.into_spans(),
                }
            }
            impl std::default::Default for $n {
                fn default() -> Self {
                    $n(Span::call_site())
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
            impl $n {
                pub fn surround<F>(&self, ts: &mut TokenStream, f: F)
                where
                    F: FnOnce(&mut TokenStream),
                {
                    let mut inner = TokenStream::new();
                    f(&mut inner);
                    crate::dump::delim(Delimiter::$d, self.span.join(), ts, inner);
                }
            }
            impl private::Sealed for $n {}
        )*
    };
}
def_delims! {
    Brace         pub struct Brace
    Bracket       pub struct Bracket
    Parenthesis   pub struct Paren
}
impl Tok for Paren {
    fn peek(x: Cursor) -> bool {
        lookahead::is_delimiter(x, Delimiter::Parenthesis)
    }
    fn display() -> &'static str {
        "parentheses"
    }
}
impl Tok for Brace {
    fn peek(x: Cursor) -> bool {
        lookahead::is_delimiter(x, Delimiter::Brace)
    }
    fn display() -> &'static str {
        "curly braces"
    }
}
impl Tok for Bracket {
    fn peek(x: Cursor) -> bool {
        lookahead::is_delimiter(x, Delimiter::Bracket)
    }
    fn display() -> &'static str {
        "square brackets"
    }
}

pub struct Group {
    pub span: Span,
}
impl Group {
    pub fn surround<F>(&self, ts: &mut TokenStream, f: F)
    where
        F: FnOnce(&mut TokenStream),
    {
        let mut inner = TokenStream::new();
        f(&mut inner);
        dump::delim(Delimiter::None, self.span, ts, inner);
    }
}
impl std::default::Default for Group {
    fn default() -> Self {
        Group {
            span: Span::call_site(),
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
impl private::Sealed for Group {}
impl Tok for Group {
    fn peek(x: Cursor) -> bool {
        lookahead::is_delimiter(x, Delimiter::None)
    }
    fn display() -> &'static str {
        "invisible group"
    }
}

#[allow(non_snake_case)]
pub fn Group<S: IntoSpans<Span>>(s: S) -> Group {
    Group { span: s.into_spans() }
}

mod private {
    use proc_macro2::Span;
    pub trait Sealed {}
    #[repr(transparent)]
    pub struct WithSpan {
        pub span: Span,
    }
}
impl private::Sealed for Ident {}
