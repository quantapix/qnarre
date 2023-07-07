use super::{
    err::Result,
    lit::{lit::Bool, lit::Byte, lit::ByteStr, lit::Char, lit::Float, lit::Int, lit::Str, Lit},
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

mod private {
    use proc_macro2::Span;
    pub trait Sealed {}
    #[repr(transparent)]
    pub struct WithSpan {
        pub span: Span,
    }
}
impl private::Sealed for Ident {}

fn peek_impl(cursor: Cursor, peek: fn(ParseStream) -> bool) -> bool {
    use crate::parse::Unexpected;
    use std::{cell::Cell, rc::Rc};
    let scope = Span::call_site();
    let unexpected = Rc::new(Cell::new(Unexpected::None));
    let buffer = crate::parse::new_parse_buffer(scope, cursor, unexpected);
    peek(&buffer)
}

pub trait Token: private::Sealed {
    fn peek(cursor: Cursor) -> bool;
    fn display() -> &'static str;
}

macro_rules! impl_token {
    ($display:literal $name:ty) => {
        impl Token for $name {
            fn peek(cursor: Cursor) -> bool {
                fn peek(input: ParseStream) -> bool {
                    <$name as Parse>::parse(input).is_ok()
                }
                peek_impl(cursor, peek)
            }
            fn display() -> &'static str {
                $display
            }
        }
        impl private::Sealed for $name {}
    };
}
impl_token!("lifetime" Lifetime);
impl_token!("literal" Lit);
impl_token!("string literal" lit::Str);
impl_token!("byte string literal" lit::ByteStr);
impl_token!("byte literal" lit::Byte);
impl_token!("character literal" lit::Char);
impl_token!("integer literal" lit::Int);
impl_token!("floating point literal" lit::Float);
impl_token!("boolean literal" lit::Bool);
impl_token!("group token" proc_macro2::Group);

macro_rules! impl_low_level_token {
    ($display:literal $ty:ident $get:ident) => {
        impl Token for $ty {
            fn peek(cursor: Cursor) -> bool {
                cursor.$get().is_some()
            }
            fn display() -> &'static str {
                $display
            }
        }
        impl private::Sealed for $ty {}
    };
}
impl_low_level_token!("punctuation token" Punct punct);
impl_low_level_token!("literal" Literal literal);
impl_low_level_token!("token" TokenTree token_tree);

pub trait CustomToken {
    fn peek(cursor: Cursor) -> bool;
    fn display() -> &'static str;
}
impl<T: CustomToken> private::Sealed for T {}
impl<T: CustomToken> Token for T {
    fn peek(cursor: Cursor) -> bool {
        <Self as CustomToken>::peek(cursor)
    }
    fn display() -> &'static str {
        <Self as CustomToken>::display()
    }
}

macro_rules! define_keywords {
    ($($token:literal pub struct $name:ident)*) => {
        $(
            #[doc = concat!('`', $token, '`')]
            pub struct $name {
                pub span: Span,
            }
            #[allow(non_snake_case)]
            pub fn $name<S: IntoSpans<Span>>(span: S) -> $name {
                $name {
                    span: span.into_spans(),
                }
            }
            impl std::default::Default for $name {
                fn default() -> Self {
                    $name {
                        span: Span::call_site(),
                    }
                }
            }
            impl Copy for $name {}
            impl Clone for $name {
                fn clone(&self) -> Self {
                    *self
                }
            }
            impl Debug for $name {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str(stringify!($name))
                }
            }
            impl cmp::Eq for $name {}
            impl PartialEq for $name {
                fn eq(&self, _other: &$name) -> bool {
                    true
                }
            }
            impl Hash for $name {
                fn hash<H: Hasher>(&self, _state: &mut H) {}
            }
            impl ToTokens for $name {
                fn to_tokens(&self, tokens: &mut TokenStream) {
                    printing::keyword($token, self.span, tokens);
                }
            }
            impl Parse for $name {
                fn parse(input: ParseStream) -> Result<Self> {
                    Ok($name {
                        span: parsing::keyword(input, $token)?,
                    })
                }
            }
            impl Token for $name {
                fn peek(cursor: Cursor) -> bool {
                    parsing::peek_keyword(cursor, $token)
                }
                fn display() -> &'static str {
                    concat!("`", $token, "`")
                }
            }
            impl private::Sealed for $name {}
        )*
    };
}
macro_rules! impl_deref_if_len_is_1 {
    ($name:ident/1) => {
        impl Deref for $name {
            type Target = private::WithSpan;
            fn deref(&self) -> &Self::Target {
                unsafe { &*(self as *const Self).cast::<private::WithSpan>() }
            }
        }
        impl DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { &mut *(self as *mut Self).cast::<private::WithSpan>() }
            }
        }
    };
    ($name:ident/$len:literal) => {};
}
macro_rules! define_punctuation_structs {
    ($($token:literal pub struct $name:ident/$len:tt #[doc = $usage:literal])*) => {
        $(
            #[cfg_attr(not(doc), repr(transparent))]
            #[doc = concat!('`', $token, '`')]
            #[doc = concat!($usage, '.')]
            pub struct $name {
                pub spans: [Span; $len],
            }
                        #[allow(non_snake_case)]
            pub fn $name<S: IntoSpans<[Span; $len]>>(spans: S) -> $name {
                $name {
                    spans: spans.into_spans(),
                }
            }
            impl std::default::Default for $name {
                fn default() -> Self {
                    $name {
                        spans: [Span::call_site(); $len],
                    }
                }
            }
            impl Copy for $name {}
            impl Clone for $name {
                fn clone(&self) -> Self {
                    *self
                }
            }
            impl Debug for $name {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str(stringify!($name))
                }
            }
            impl cmp::Eq for $name {}
            impl PartialEq for $name {
                fn eq(&self, _other: &$name) -> bool {
                    true
                }
            }
            impl Hash for $name {
                fn hash<H: Hasher>(&self, _state: &mut H) {}
            }
            impl_deref_if_len_is_1!($name/$len);
        )*
    };
}
macro_rules! define_punctuation {
    ($($token:literal pub struct $name:ident/$len:tt #[doc = $usage:literal])*) => {
        $(
            define_punctuation_structs! {
                $token pub struct $name/$len #[doc = $usage]
            }
            impl ToTokens for $name {
                fn to_tokens(&self, tokens: &mut TokenStream) {
                    printing::punct($token, &self.spans, tokens);
                }
            }
            impl Parse for $name {
                fn parse(input: ParseStream) -> Result<Self> {
                    Ok($name {
                        spans: parsing::punct(input, $token)?,
                    })
                }
            }
            impl Token for $name {
                fn peek(cursor: Cursor) -> bool {
                    parsing::peek_punct(cursor, $token)
                }
                fn display() -> &'static str {
                    concat!("`", $token, "`")
                }
            }
            impl private::Sealed for $name {}
        )*
    };
}
macro_rules! define_delimiters {
    ($($delim:ident pub struct $name:ident #[$doc:meta])*) => {
        $(
            #[$doc]
            pub struct $name {
                pub span: DelimSpan,
            }
                        #[allow(non_snake_case)]
            pub fn $name<S: IntoSpans<DelimSpan>>(span: S) -> $name {
                $name {
                    span: span.into_spans(),
                }
            }
            impl std::default::Default for $name {
                fn default() -> Self {
                    $name(Span::call_site())
                }
            }
            impl Copy for $name {}
            impl Clone for $name {
                fn clone(&self) -> Self {
                    *self
                }
            }
            impl Debug for $name {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str(stringify!($name))
                }
            }
            impl cmp::Eq for $name {}
            impl PartialEq for $name {
                fn eq(&self, _other: &$name) -> bool {
                    true
                }
            }
            impl Hash for $name {
                fn hash<H: Hasher>(&self, _state: &mut H) {}
            }
            impl $name {
                pub fn surround<F>(&self, tokens: &mut TokenStream, f: F)
                where
                    F: FnOnce(&mut TokenStream),
                {
                    let mut inner = TokenStream::new();
                    f(&mut inner);
                    printing::delim(Delimiter::$delim, self.span.join(), tokens, inner);
                }
            }
            impl private::Sealed for $name {}
        )*
    };
}

define_punctuation_structs! {
    "_" pub struct Underscore/1 /// wildcard patterns, inferred types, unnamed items in constants, extern crates, use declarations, and destructuring assignment
}

impl ToTokens for Underscore {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append(Ident::new("_", self.span));
    }
}
impl Parse for Underscore {
    fn parse(input: ParseStream) -> Result<Self> {
        input.step(|cursor| {
            if let Some((ident, rest)) = cursor.ident() {
                if ident == "_" {
                    return Ok((Underscore(ident.span()), rest));
                }
            }
            if let Some((punct, rest)) = cursor.punct() {
                if punct.as_char() == '_' {
                    return Ok((Underscore(punct.span()), rest));
                }
            }
            Err(cursor.error("expected `_`"))
        })
    }
}
impl Token for Underscore {
    fn peek(cursor: Cursor) -> bool {
        if let Some((ident, _rest)) = cursor.ident() {
            return ident == "_";
        }
        if let Some((punct, _rest)) = cursor.punct() {
            return punct.as_char() == '_';
        }
        false
    }
    fn display() -> &'static str {
        "`_`"
    }
}
impl private::Sealed for Underscore {}

pub struct Group {
    pub span: Span,
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
    fn eq(&self, _other: &Group) -> bool {
        true
    }
}
impl Hash for Group {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}
impl Group {
    pub fn surround<F>(&self, tokens: &mut TokenStream, f: F)
    where
        F: FnOnce(&mut TokenStream),
    {
        let mut inner = TokenStream::new();
        f(&mut inner);
        printing::delim(Delimiter::None, self.span, tokens, inner);
    }
}
impl private::Sealed for Group {}

#[allow(non_snake_case)]
pub fn Group<S: IntoSpans<Span>>(span: S) -> Group {
    Group {
        span: span.into_spans(),
    }
}

impl Token for Paren {
    fn peek(cursor: Cursor) -> bool {
        lookahead::is_delimiter(cursor, Delimiter::Parenthesis)
    }
    fn display() -> &'static str {
        "parentheses"
    }
}
impl Token for Brace {
    fn peek(cursor: Cursor) -> bool {
        lookahead::is_delimiter(cursor, Delimiter::Brace)
    }
    fn display() -> &'static str {
        "curly braces"
    }
}
impl Token for Bracket {
    fn peek(cursor: Cursor) -> bool {
        lookahead::is_delimiter(cursor, Delimiter::Bracket)
    }
    fn display() -> &'static str {
        "square brackets"
    }
}
impl Token for Group {
    fn peek(cursor: Cursor) -> bool {
        lookahead::is_delimiter(cursor, Delimiter::None)
    }
    fn display() -> &'static str {
        "invisible group"
    }
}

define_keywords! {
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
define_punctuation! {
    "&"           pub struct And/1        /// bitwise and logical AND, borrow, references, reference patterns
    "&&"          pub struct AndAnd/2     /// lazy AND, borrow, references, reference patterns
    "&="          pub struct AndEq/2      /// bitwise AND assignment
    "@"           pub struct At/1         /// subpattern binding
    "^"           pub struct Caret/1      /// bitwise and logical XOR
    "^="          pub struct CaretEq/2    /// bitwise XOR assignment
    ":"           pub struct Colon/1      /// various separators
    ","           pub struct Comma/1      /// various separators
    "$"           pub struct Dollar/1     /// macros
    "."           pub struct Dot/1        /// field access, tuple index
    ".."          pub struct DotDot/2     /// range, struct expressions, patterns, range patterns
    "..."         pub struct DotDotDot/3  /// variadic functions, range patterns
    "..="         pub struct DotDotEq/3   /// inclusive range, range patterns
    "="           pub struct Eq/1         /// assignment, attributes, various type definitions
    "=="          pub struct EqEq/2       /// equal
    "=>"          pub struct FatArrow/2   /// match arms, macros
    ">="          pub struct Ge/2         /// greater than or equal to, generics
    ">"           pub struct Gt/1         /// greater than, generics, paths
    "<-"          pub struct LArrow/2     /// unused
    "<="          pub struct Le/2         /// less than or equal to
    "<"           pub struct Lt/1         /// less than, generics, paths
    "-"           pub struct Minus/1      /// subtraction, negation
    "-="          pub struct MinusEq/2    /// subtraction assignment
    "!="          pub struct Ne/2         /// not equal
    "!"           pub struct Not/1        /// bitwise and logical NOT, macro calls, inner attributes, never type, negative impls
    "|"           pub struct Or/1         /// bitwise and logical OR, closures, patterns in match, if let, and while let
    "|="          pub struct OrEq/2       /// bitwise OR assignment
    "||"          pub struct OrOr/2       /// lazy OR, closures
    "::"          pub struct PathSep/2    /// path separator
    "%"           pub struct Percent/1    /// remainder
    "%="          pub struct PercentEq/2  /// remainder assignment
    "+"           pub struct Plus/1       /// addition, trait bounds, macro Kleene matcher
    "+="          pub struct PlusEq/2     /// addition assignment
    "#"           pub struct Pound/1      /// attributes
    "?"           pub struct Question/1   /// question mark operator, questionably sized, macro Kleene matcher
    "->"          pub struct RArrow/2     /// function return type, closure return type, function pointer type
    ";"           pub struct Semi/1       /// terminator for various items and statements, array types
    "<<"          pub struct Shl/2        /// shift left, nested generics
    "<<="         pub struct ShlEq/3      /// shift left assignment
    ">>"          pub struct Shr/2        /// shift right, nested generics
    ">>="         pub struct ShrEq/3      /// shift right assignment, nested generics
    "/"           pub struct Slash/1      /// division
    "/="          pub struct SlashEq/2    /// division assignment
    "*"           pub struct Star/1       /// multiplication, dereference, raw pointers, macro Kleene matcher, use wildcards
    "*="          pub struct StarEq/2     /// multiplication assignment
    "~"           pub struct Tilde/1      /// unused since before Rust 1.0
}
define_delimiters! {
    Brace         pub struct Brace        /// `{`&hellip;`}`
    Bracket       pub struct Bracket      /// `[`&hellip;`]`
    Parenthesis   pub struct Paren        /// `(`&hellip;`)`
}

#[macro_export]
macro_rules! Token {
    [abstract]    => { $crate::token::Abstract };
    [as]          => { $crate::token::As };
    [async]       => { $crate::token::Async };
    [auto]        => { $crate::token::Auto };
    [await]       => { $crate::token::Await };
    [become]      => { $crate::token::Become };
    [box]         => { $crate::token::Box };
    [break]       => { $crate::token::Break };
    [const]       => { $crate::token::Const };
    [continue]    => { $crate::token::Continue };
    [crate]       => { $crate::token::Crate };
    [default]     => { $crate::token::Default };
    [do]          => { $crate::token::Do };
    [dyn]         => { $crate::token::Dyn };
    [else]        => { $crate::token::Else };
    [enum]        => { $crate::token::Enum };
    [extern]      => { $crate::token::Extern };
    [final]       => { $crate::token::Final };
    [fn]          => { $crate::token::Fn };
    [for]         => { $crate::token::For };
    [if]          => { $crate::token::If };
    [impl]        => { $crate::token::Impl };
    [in]          => { $crate::token::In };
    [let]         => { $crate::token::Let };
    [loop]        => { $crate::token::Loop };
    [macro]       => { $crate::token::Macro };
    [match]       => { $crate::token::Match };
    [mod]         => { $crate::token::Mod };
    [move]        => { $crate::token::Move };
    [mut]         => { $crate::token::Mut };
    [override]    => { $crate::token::Override };
    [priv]        => { $crate::token::Priv };
    [pub]         => { $crate::token::Pub };
    [ref]         => { $crate::token::Ref };
    [return]      => { $crate::token::Return };
    [Self]        => { $crate::token::SelfType };
    [self]        => { $crate::token::SelfValue };
    [static]      => { $crate::token::Static };
    [struct]      => { $crate::token::Struct };
    [super]       => { $crate::token::Super };
    [trait]       => { $crate::token::Trait };
    [try]         => { $crate::token::Try };
    [type]        => { $crate::token::Type };
    [typeof]      => { $crate::token::Typeof };
    [union]       => { $crate::token::Union };
    [unsafe]      => { $crate::token::Unsafe };
    [unsized]     => { $crate::token::Unsized };
    [use]         => { $crate::token::Use };
    [virtual]     => { $crate::token::Virtual };
    [where]       => { $crate::token::Where };
    [while]       => { $crate::token::While };
    [yield]       => { $crate::token::Yield };
    [&]           => { $crate::token::And };
    [&&]          => { $crate::token::AndAnd };
    [&=]          => { $crate::token::AndEq };
    [@]           => { $crate::token::At };
    [^]           => { $crate::token::Caret };
    [^=]          => { $crate::token::CaretEq };
    [:]           => { $crate::token::Colon };
    [,]           => { $crate::token::Comma };
    [$]           => { $crate::token::Dollar };
    [.]           => { $crate::token::Dot };
    [..]          => { $crate::token::DotDot };
    [...]         => { $crate::token::DotDotDot };
    [..=]         => { $crate::token::DotDotEq };
    [=]           => { $crate::token::Eq };
    [==]          => { $crate::token::EqEq };
    [=>]          => { $crate::token::FatArrow };
    [>=]          => { $crate::token::Ge };
    [>]           => { $crate::token::Gt };
    [<-]          => { $crate::token::LArrow };
    [<=]          => { $crate::token::Le };
    [<]           => { $crate::token::Lt };
    [-]           => { $crate::token::Minus };
    [-=]          => { $crate::token::MinusEq };
    [!=]          => { $crate::token::Ne };
    [!]           => { $crate::token::Not };
    [|]           => { $crate::token::Or };
    [|=]          => { $crate::token::OrEq };
    [||]          => { $crate::token::OrOr };
    [::]          => { $crate::token::PathSep };
    [%]           => { $crate::token::Percent };
    [%=]          => { $crate::token::PercentEq };
    [+]           => { $crate::token::Plus };
    [+=]          => { $crate::token::PlusEq };
    [#]           => { $crate::token::Pound };
    [?]           => { $crate::token::Question };
    [->]          => { $crate::token::RArrow };
    [;]           => { $crate::token::Semi };
    [<<]          => { $crate::token::Shl };
    [<<=]         => { $crate::token::ShlEq };
    [>>]          => { $crate::token::Shr };
    [>>=]         => { $crate::token::ShrEq };
    [/]           => { $crate::token::Slash };
    [/=]          => { $crate::token::SlashEq };
    [*]           => { $crate::token::Star };
    [*=]          => { $crate::token::StarEq };
    [~]           => { $crate::token::Tilde };
    [_]           => { $crate::token::Underscore };
}
