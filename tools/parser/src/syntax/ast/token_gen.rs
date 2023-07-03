use crate::{
    syntax::{self, ast},
    SyntaxKind::{self, *},
};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Whitespace {
    pub syntax: syntax::Token,
}
impl fmt::Display for Whitespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::Token for Whitespace {
    fn can_cast(x: SyntaxKind) -> bool {
        x == WHITESPACE
    }
    fn cast(syntax: syntax::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> syntax::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Comment {
    pub syntax: syntax::Token,
}
impl fmt::Display for Comment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::Token for Comment {
    fn can_cast(x: SyntaxKind) -> bool {
        x == COMMENT
    }
    fn cast(syntax: syntax::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> syntax::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct String {
    pub syntax: syntax::Token,
}
impl fmt::Display for String {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::Token for String {
    fn can_cast(x: SyntaxKind) -> bool {
        x == STRING
    }
    fn cast(syntax: syntax::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> syntax::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ByteString {
    pub syntax: syntax::Token,
}
impl fmt::Display for ByteString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::Token for ByteString {
    fn can_cast(x: SyntaxKind) -> bool {
        x == BYTE_STRING
    }
    fn cast(syntax: syntax::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> syntax::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CString {
    pub syntax: syntax::Token,
}
impl fmt::Display for CString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::Token for CString {
    fn can_cast(x: SyntaxKind) -> bool {
        x == C_STRING
    }
    fn cast(syntax: syntax::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> syntax::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntNumber {
    pub syntax: syntax::Token,
}
impl fmt::Display for IntNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::Token for IntNumber {
    fn can_cast(x: SyntaxKind) -> bool {
        x == INT_NUMBER
    }
    fn cast(syntax: syntax::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> syntax::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FloatNumber {
    pub syntax: syntax::Token,
}
impl fmt::Display for FloatNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::Token for FloatNumber {
    fn can_cast(x: SyntaxKind) -> bool {
        x == FLOAT_NUMBER
    }
    fn cast(syntax: syntax::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> syntax::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Char {
    pub syntax: syntax::Token,
}
impl fmt::Display for Char {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::Token for Char {
    fn can_cast(x: SyntaxKind) -> bool {
        x == CHAR
    }
    fn cast(syntax: syntax::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> syntax::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Byte {
    pub syntax: syntax::Token,
}
impl fmt::Display for Byte {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::Token for Byte {
    fn can_cast(x: SyntaxKind) -> bool {
        x == BYTE
    }
    fn cast(syntax: syntax::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> syntax::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ident {
    pub syntax: syntax::Token,
}
impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::Token for Ident {
    fn can_cast(kind: SyntaxKind) -> bool {
        kind == IDENT
    }
    fn cast(syntax: syntax::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> syntax::Token {
        &self.syntax
    }
}
