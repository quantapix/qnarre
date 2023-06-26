use crate::{
    ast,
    SyntaxKind::{self, *},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Whitespace {
    pub syntax: crate::Token,
}
impl std::fmt::Display for Whitespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::AstToken for Whitespace {
    fn can_cast(kind: SyntaxKind) -> bool {
        kind == WHITESPACE
    }
    fn cast(syntax: crate::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> crate::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Comment {
    pub syntax: crate::Token,
}
impl std::fmt::Display for Comment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::AstToken for Comment {
    fn can_cast(kind: SyntaxKind) -> bool {
        kind == COMMENT
    }
    fn cast(syntax: crate::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> crate::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct String {
    pub syntax: crate::Token,
}
impl std::fmt::Display for String {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::AstToken for String {
    fn can_cast(kind: SyntaxKind) -> bool {
        kind == STRING
    }
    fn cast(syntax: crate::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> crate::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ByteString {
    pub syntax: crate::Token,
}
impl std::fmt::Display for ByteString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::AstToken for ByteString {
    fn can_cast(kind: SyntaxKind) -> bool {
        kind == BYTE_STRING
    }
    fn cast(syntax: crate::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> crate::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CString {
    pub syntax: crate::Token,
}
impl std::fmt::Display for CString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::AstToken for CString {
    fn can_cast(kind: SyntaxKind) -> bool {
        kind == C_STRING
    }
    fn cast(syntax: crate::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> crate::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntNumber {
    pub syntax: crate::Token,
}
impl std::fmt::Display for IntNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::AstToken for IntNumber {
    fn can_cast(kind: SyntaxKind) -> bool {
        kind == INT_NUMBER
    }
    fn cast(syntax: crate::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> crate::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FloatNumber {
    pub syntax: crate::Token,
}
impl std::fmt::Display for FloatNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::AstToken for FloatNumber {
    fn can_cast(kind: SyntaxKind) -> bool {
        kind == FLOAT_NUMBER
    }
    fn cast(syntax: crate::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> crate::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Char {
    pub syntax: crate::Token,
}
impl std::fmt::Display for Char {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::AstToken for Char {
    fn can_cast(kind: SyntaxKind) -> bool {
        kind == CHAR
    }
    fn cast(syntax: crate::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> crate::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Byte {
    pub syntax: crate::Token,
}
impl std::fmt::Display for Byte {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::AstToken for Byte {
    fn can_cast(kind: SyntaxKind) -> bool {
        kind == BYTE
    }
    fn cast(syntax: crate::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> crate::Token {
        &self.syntax
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ident {
    pub syntax: crate::Token,
}
impl std::fmt::Display for Ident {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl ast::AstToken for Ident {
    fn can_cast(kind: SyntaxKind) -> bool {
        kind == IDENT
    }
    fn cast(syntax: crate::Token) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> crate::Token {
        &self.syntax
    }
}
