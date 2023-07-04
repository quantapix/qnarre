use super::{ptr::P, util::case::Case, *};
use macros::HashStable_Generic;
use rustc_data_structures::{
    stable_hasher::{HashStable, StableHasher},
    sync::Lrc,
};
use rustc_span::{
    self,
    edition::Edition,
    symbol::{kw, sym, Ident, Symbol},
    Span, DUMMY_SP,
};
use std::{borrow::Cow, fmt};

#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum CommentKind {
    Line,
    Block,
}

#[derive(Clone, PartialEq, Encodable, Decodable, Hash, Debug, Copy, HashStable_Generic)]
pub enum BinOpToken {
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    And,
    Or,
    Shl,
    Shr,
}
pub use BinOpToken::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Encodable, Decodable, Hash, HashStable_Generic)]
pub enum Delimiter {
    Parenthesis,
    Brace,
    Bracket,
    Invisible,
}

#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum LiteralKind {
    Bool,
    Byte,
    Char,
    Integer,
    Float,
    Str,
    StrRaw(u8),
    ByteStr,
    ByteStrRaw(u8),
    CStr,
    CStrRaw(u8),
    Err,
}
impl LiteralKind {
    pub fn article(self) -> &'static str {
        match self {
            Integer | Err => "an",
            _ => "a",
        }
    }
    pub fn descr(self) -> &'static str {
        match self {
            Bool => panic!("literal token contains `Lit::Bool`"),
            Byte => "byte",
            Char => "char",
            Integer => "integer",
            Float => "float",
            Str | StrRaw(..) => "string",
            ByteStr | ByteStrRaw(..) => "byte string",
            CStr | CStrRaw(..) => "C string",
            Err => "error",
        }
    }
    pub(crate) fn may_have_suffix(self) -> bool {
        matches!(self, Integer | Float | Err)
    }
}
pub use LiteralKind::*;

#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct Lit {
    pub kind: LiteralKind,
    pub symbol: Symbol,
    pub suffix: Option<Symbol>,
}
impl Lit {
    pub fn new(kind: LiteralKind, symbol: Symbol, suffix: Option<Symbol>) -> Lit {
        Lit { kind, symbol, suffix }
    }
    pub fn is_semantic_float(&self) -> bool {
        match self.kind {
            LiteralKind::Float => true,
            LiteralKind::Integer => match self.suffix {
                Some(sym) => sym == sym::f32 || sym == sym::f64,
                None => false,
            },
            _ => false,
        }
    }
    pub fn from_token(token: &Token) -> Option<Lit> {
        match token.uninterpolate().kind {
            Ident(name, false) if name.is_bool_lit() => {
                Some(Lit::new(Bool, name, None))
            }
            Literal(token_lit) => Some(token_lit),
            Interpolated(ref nt)
                if let NtExpr(expr) | NtLiteral(expr) = &**nt
                && let ExprKind::Lit(token_lit) = expr.kind =>
            {
                Some(token_lit)
            }
            _ => None,
        }
    }
}
impl fmt::Display for Lit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Lit { kind, symbol, suffix } = *self;
        match kind {
            Byte => write!(f, "b'{symbol}'")?,
            Char => write!(f, "'{symbol}'")?,
            Str => write!(f, "\"{symbol}\"")?,
            StrRaw(n) => write!(
                f,
                "r{delim}\"{string}\"{delim}",
                delim = "#".repeat(n as usize),
                string = symbol
            )?,
            ByteStr => write!(f, "b\"{symbol}\"")?,
            ByteStrRaw(n) => write!(
                f,
                "br{delim}\"{string}\"{delim}",
                delim = "#".repeat(n as usize),
                string = symbol
            )?,
            CStr => write!(f, "c\"{symbol}\"")?,
            CStrRaw(n) => write!(f, "cr{delim}\"{symbol}\"{delim}", delim = "#".repeat(n as usize))?,
            Integer | Float | Bool | Err => write!(f, "{symbol}")?,
        }
        if let Some(suffix) = suffix {
            write!(f, "{suffix}")?;
        }
        Ok(())
    }
}

pub fn ident_can_begin_expr(name: Symbol, span: Span, is_raw: bool) -> bool {
    let ident_token = Token::new(Ident(name, is_raw), span);
    !ident_token.is_reserved_ident()
        || ident_token.is_path_segment_keyword()
        || [
            kw::Async,
            kw::Do,
            kw::Box,
            kw::Break,
            kw::Const,
            kw::Continue,
            kw::False,
            kw::For,
            kw::If,
            kw::Let,
            kw::Loop,
            kw::Match,
            kw::Move,
            kw::Return,
            kw::True,
            kw::Try,
            kw::Unsafe,
            kw::While,
            kw::Yield,
            kw::Static,
        ]
        .contains(&name)
}
fn ident_can_begin_type(name: Symbol, span: Span, is_raw: bool) -> bool {
    let ident_token = Token::new(Ident(name, is_raw), span);
    !ident_token.is_reserved_ident()
        || ident_token.is_path_segment_keyword()
        || [
            kw::Underscore,
            kw::For,
            kw::Impl,
            kw::Fn,
            kw::Unsafe,
            kw::Extern,
            kw::Typeof,
            kw::Dyn,
        ]
        .contains(&name)
}

#[derive(Clone, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum TokenKind {
    Eq,
    Lt,
    Le,
    EqEq,
    Ne,
    Ge,
    Gt,
    AndAnd,
    OrOr,
    Not,
    Tilde,
    BinOp(BinOpToken),
    BinOpEq(BinOpToken),
    At,
    Dot,
    DotDot,
    DotDotDot,
    DotDotEq,
    Comma,
    Semi,
    Colon,
    ModSep,
    RArrow,
    LArrow,
    FatArrow,
    Pound,
    Dollar,
    Question,
    SingleQuote,
    OpenDelim(Delimiter),
    CloseDelim(Delimiter),
    Literal(Lit),
    Ident(Symbol, /* is_raw */ bool),
    Lifetime(Symbol),
    Interpolated(Lrc<Nonterminal>),
    DocComment(CommentKind, AttrStyle, Symbol),
    Eof,
}
impl TokenKind {
    pub fn lit(kind: LiteralKind, symbol: Symbol, suffix: Option<Symbol>) -> TokenKind {
        Literal(Lit::new(kind, symbol, suffix))
    }
    pub fn break_two_token_op(&self) -> Option<(TokenKind, TokenKind)> {
        Some(match *self {
            Le => (Lt, Eq),
            EqEq => (Eq, Eq),
            Ne => (Not, Eq),
            Ge => (Gt, Eq),
            AndAnd => (BinOp(And), BinOp(And)),
            OrOr => (BinOp(Or), BinOp(Or)),
            BinOp(Shl) => (Lt, Lt),
            BinOp(Shr) => (Gt, Gt),
            BinOpEq(Plus) => (BinOp(Plus), Eq),
            BinOpEq(Minus) => (BinOp(Minus), Eq),
            BinOpEq(Star) => (BinOp(Star), Eq),
            BinOpEq(Slash) => (BinOp(Slash), Eq),
            BinOpEq(Percent) => (BinOp(Percent), Eq),
            BinOpEq(Caret) => (BinOp(Caret), Eq),
            BinOpEq(And) => (BinOp(And), Eq),
            BinOpEq(Or) => (BinOp(Or), Eq),
            BinOpEq(Shl) => (Lt, Le),
            BinOpEq(Shr) => (Gt, Ge),
            DotDot => (Dot, Dot),
            DotDotDot => (Dot, DotDot),
            ModSep => (Colon, Colon),
            RArrow => (BinOp(Minus), Gt),
            LArrow => (Lt, BinOp(Minus)),
            FatArrow => (Eq, Gt),
            _ => return None,
        })
    }
    pub fn similar_tokens(&self) -> Option<Vec<TokenKind>> {
        match *self {
            Comma => Some(vec![Dot, Lt, Semi]),
            Semi => Some(vec![Colon, Comma]),
            FatArrow => Some(vec![Eq, RArrow]),
            _ => None,
        }
    }
    pub fn should_end_const_arg(&self) -> bool {
        matches!(self, Gt | Ge | BinOp(Shr) | BinOpEq(Shr))
    }
}
pub use TokenKind::*;

#[derive(Clone, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}
impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Token { kind, span }
    }
    pub fn dummy() -> Self {
        Token::new(TokenKind::Question, DUMMY_SP)
    }
    pub fn from_ast_ident(ident: Ident) -> Self {
        Token::new(Ident(ident.name, ident.is_raw_guess()), ident.span)
    }
    pub fn uninterpolated_span(&self) -> Span {
        match &self.kind {
            Interpolated(nt) => nt.span(),
            _ => self.span,
        }
    }
    pub fn is_range_separator(&self) -> bool {
        [DotDot, DotDotDot, DotDotEq].contains(&self.kind)
    }
    pub fn is_op(&self) -> bool {
        match self.kind {
            Eq | Lt | Le | EqEq | Ne | Ge | Gt | AndAnd | OrOr | Not | Tilde | BinOp(_) | BinOpEq(_) | At | Dot
            | DotDot | DotDotDot | DotDotEq | Comma | Semi | Colon | ModSep | RArrow | LArrow | FatArrow | Pound
            | Dollar | Question | SingleQuote => true,
            OpenDelim(..) | CloseDelim(..) | Literal(..) | DocComment(..) | Ident(..) | Lifetime(..)
            | Interpolated(..) | Eof => false,
        }
    }
    pub fn is_like_plus(&self) -> bool {
        matches!(self.kind, BinOp(Plus) | BinOpEq(Plus))
    }
    pub fn can_begin_expr(&self) -> bool {
        match self.uninterpolate().kind {
            Ident(name, is_raw) => ident_can_begin_expr(name, self.span, is_raw),
            OpenDelim(..) | Literal(..) | Not | BinOp(Minus) | BinOp(Star) | BinOp(Or) | OrOr | BinOp(And) | AndAnd
            | DotDot | DotDotDot | DotDotEq | Lt | BinOp(Shl) | ModSep | Lifetime(..) | Pound => true,
            Interpolated(ref nt) => matches!(**nt, NtLiteral(..) | NtExpr(..) | NtBlock(..) | NtPath(..)),
            _ => false,
        }
    }
    pub fn can_begin_pattern(&self) -> bool {
        match self.uninterpolate().kind {
            Ident(name, is_raw) => ident_can_begin_expr(name, self.span, is_raw),
            OpenDelim(Delimiter::Bracket | Delimiter::Parenthesis)
            | Literal(..)
            | BinOp(Minus)
            | BinOp(And)
            | AndAnd
            | DotDot
            | DotDotDot
            | DotDotEq
            | Lt
            | BinOp(Shl)
            | ModSep => true,
            Interpolated(ref nt) => matches!(**nt, NtLiteral(..) | NtPat(..) | NtBlock(..) | NtPath(..)),
            _ => false,
        }
    }
    pub fn can_begin_type(&self) -> bool {
        match self.uninterpolate().kind {
            Ident(name, is_raw) => ident_can_begin_type(name, self.span, is_raw),
            OpenDelim(Delimiter::Parenthesis)
            | OpenDelim(Delimiter::Bracket)
            | Not
            | BinOp(Star)
            | BinOp(And)
            | AndAnd
            | Question
            | Lifetime(..)
            | Lt
            | BinOp(Shl)
            | ModSep => true,
            Interpolated(ref nt) => matches!(**nt, NtTy(..) | NtPath(..)),
            _ => false,
        }
    }
    pub fn can_begin_const_arg(&self) -> bool {
        match self.kind {
            OpenDelim(Delimiter::Brace) => true,
            Interpolated(ref nt) => matches!(**nt, NtExpr(..) | NtBlock(..) | NtLiteral(..)),
            _ => self.can_begin_literal_maybe_minus(),
        }
    }
    pub fn can_begin_bound(&self) -> bool {
        self.is_path_start()
            || self.is_lifetime()
            || self.is_keyword(kw::For)
            || self == &Question
            || self == &OpenDelim(Delimiter::Parenthesis)
    }
    pub fn can_begin_item(&self) -> bool {
        match self.kind {
            Ident(name, _) => [
                kw::Fn,
                kw::Use,
                kw::Struct,
                kw::Enum,
                kw::Pub,
                kw::Trait,
                kw::Extern,
                kw::Impl,
                kw::Unsafe,
                kw::Const,
                kw::Static,
                kw::Union,
                kw::Macro,
                kw::Mod,
                kw::Type,
            ]
            .contains(&name),
            _ => false,
        }
    }
    pub fn is_lit(&self) -> bool {
        matches!(self.kind, Literal(..))
    }
    pub fn can_begin_literal_maybe_minus(&self) -> bool {
        match self.uninterpolate().kind {
            Literal(..) | BinOp(Minus) => true,
            Ident(name, false) if name.is_bool_lit() => true,
            Interpolated(ref nt) => match &**nt {
                NtLiteral(_) => true,
                NtExpr(e) => match &e.kind {
                    ExprKind::Lit(_) => true,
                    ExprKind::Unary(UnOp::Neg, e) => {
                        matches!(&e.kind, ExprKind::Lit(_))
                    },
                    _ => false,
                },
                _ => false,
            },
            _ => false,
        }
    }
    pub fn uninterpolate(&self) -> Cow<'_, Token> {
        match &self.kind {
            Interpolated(nt) => match **nt {
                NtIdent(ident, is_raw) => Cow::Owned(Token::new(Ident(ident.name, is_raw), ident.span)),
                NtLifetime(ident) => Cow::Owned(Token::new(Lifetime(ident.name), ident.span)),
                _ => Cow::Borrowed(self),
            },
            _ => Cow::Borrowed(self),
        }
    }
    #[inline]
    pub fn ident(&self) -> Option<(Ident, /* is_raw */ bool)> {
        match &self.kind {
            &Ident(name, is_raw) => Some((Ident::new(name, self.span), is_raw)),
            Interpolated(nt) => match **nt {
                NtIdent(ident, is_raw) => Some((ident, is_raw)),
                _ => None,
            },
            _ => None,
        }
    }
    #[inline]
    pub fn lifetime(&self) -> Option<Ident> {
        match &self.kind {
            &Lifetime(name) => Some(Ident::new(name, self.span)),
            Interpolated(nt) => match **nt {
                NtLifetime(ident) => Some(ident),
                _ => None,
            },
            _ => None,
        }
    }
    pub fn is_ident(&self) -> bool {
        self.ident().is_some()
    }
    pub fn is_lifetime(&self) -> bool {
        self.lifetime().is_some()
    }
    pub fn is_ident_named(&self, name: Symbol) -> bool {
        self.ident().is_some_and(|(ident, _)| ident.name == name)
    }
    fn is_path(&self) -> bool {
        if let Interpolated(nt) = &self.kind && let NtPath(..) = **nt {
            return true;
        }
        false
    }
    pub fn is_whole_expr(&self) -> bool {
        if let Interpolated(nt) = &self.kind
            && let NtExpr(_) | NtLiteral(_) | NtPath(_) | NtBlock(_) = **nt
        {
            return true;
        }
        false
    }
    pub fn is_whole_block(&self) -> bool {
        if let Interpolated(nt) = &self.kind && let NtBlock(..) = **nt {
            return true;
        }
        false
    }
    pub fn is_mutability(&self) -> bool {
        self.is_keyword(kw::Mut) || self.is_keyword(kw::Const)
    }
    pub fn is_qpath_start(&self) -> bool {
        self == &Lt || self == &BinOp(Shl)
    }
    pub fn is_path_start(&self) -> bool {
        self == &ModSep
            || self.is_qpath_start()
            || self.is_path()
            || self.is_path_segment_keyword()
            || self.is_ident() && !self.is_reserved_ident()
    }
    pub fn is_keyword(&self, kw: Symbol) -> bool {
        self.is_non_raw_ident_where(|id| id.name == kw)
    }
    pub fn is_keyword_case(&self, kw: Symbol, case: Case) -> bool {
        self.is_keyword(kw)
            || (case == Case::Insensitive
                && self.is_non_raw_ident_where(|id| id.name.as_str().to_lowercase() == kw.as_str().to_lowercase()))
    }
    pub fn is_path_segment_keyword(&self) -> bool {
        self.is_non_raw_ident_where(Ident::is_path_segment_keyword)
    }
    pub fn is_special_ident(&self) -> bool {
        self.is_non_raw_ident_where(Ident::is_special)
    }
    pub fn is_used_keyword(&self) -> bool {
        self.is_non_raw_ident_where(Ident::is_used_keyword)
    }
    pub fn is_unused_keyword(&self) -> bool {
        self.is_non_raw_ident_where(Ident::is_unused_keyword)
    }
    pub fn is_reserved_ident(&self) -> bool {
        self.is_non_raw_ident_where(Ident::is_reserved)
    }
    pub fn is_bool_lit(&self) -> bool {
        self.is_non_raw_ident_where(|id| id.name.is_bool_lit())
    }
    pub fn is_numeric_lit(&self) -> bool {
        matches!(
            self.kind,
            Literal(Lit {
                kind: LiteralKind::Integer,
                ..
            }) | Literal(Lit {
                kind: LiteralKind::Float,
                ..
            })
        )
    }
    pub fn is_non_raw_ident_where(&self, pred: impl FnOnce(Ident) -> bool) -> bool {
        match self.ident() {
            Some((id, false)) => pred(id),
            _ => false,
        }
    }
    pub fn glue(&self, joint: &Token) -> Option<Token> {
        let kind = match self.kind {
            Eq => match joint.kind {
                Eq => EqEq,
                Gt => FatArrow,
                _ => return None,
            },
            Lt => match joint.kind {
                Eq => Le,
                Lt => BinOp(Shl),
                Le => BinOpEq(Shl),
                BinOp(Minus) => LArrow,
                _ => return None,
            },
            Gt => match joint.kind {
                Eq => Ge,
                Gt => BinOp(Shr),
                Ge => BinOpEq(Shr),
                _ => return None,
            },
            Not => match joint.kind {
                Eq => Ne,
                _ => return None,
            },
            BinOp(op) => match joint.kind {
                Eq => BinOpEq(op),
                BinOp(And) if op == And => AndAnd,
                BinOp(Or) if op == Or => OrOr,
                Gt if op == Minus => RArrow,
                _ => return None,
            },
            Dot => match joint.kind {
                Dot => DotDot,
                DotDot => DotDotDot,
                _ => return None,
            },
            DotDot => match joint.kind {
                Dot => DotDotDot,
                Eq => DotDotEq,
                _ => return None,
            },
            Colon => match joint.kind {
                Colon => ModSep,
                _ => return None,
            },
            SingleQuote => match joint.kind {
                Ident(name, false) => Lifetime(Symbol::intern(&format!("'{name}"))),
                _ => return None,
            },
            Le | EqEq | Ne | Ge | AndAnd | OrOr | Tilde | BinOpEq(..) | At | DotDotDot | DotDotEq | Comma | Semi
            | ModSep | RArrow | LArrow | FatArrow | Pound | Dollar | Question | OpenDelim(..) | CloseDelim(..)
            | Literal(..) | Ident(..) | Lifetime(..) | Interpolated(..) | DocComment(..) | Eof => return None,
        };
        Some(Token::new(kind, self.span.to(joint.span)))
    }
}
impl PartialEq<TokenKind> for Token {
    #[inline]
    fn eq(&self, rhs: &TokenKind) -> bool {
        self.kind == *rhs
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Encodable, Decodable)]
pub enum NonterminalKind {
    Item,
    Block,
    Stmt,
    PatParam { inferred: bool },
    PatWithOr,
    Expr,
    Ty,
    Ident,
    Lifetime,
    Literal,
    Meta,
    Path,
    Vis,
    TT,
}
impl NonterminalKind {
    pub fn from_symbol(symbol: Symbol, edition: impl FnOnce() -> Edition) -> Option<NonterminalKind> {
        Some(match symbol {
            sym::item => NonterminalKind::Item,
            sym::block => NonterminalKind::Block,
            sym::stmt => NonterminalKind::Stmt,
            sym::pat => match edition() {
                Edition::Edition2015 | Edition::Edition2018 => NonterminalKind::PatParam { inferred: true },
                Edition::Edition2021 | Edition::Edition2024 => NonterminalKind::PatWithOr,
            },
            sym::pat_param => NonterminalKind::PatParam { inferred: false },
            sym::expr => NonterminalKind::Expr,
            sym::ty => NonterminalKind::Ty,
            sym::ident => NonterminalKind::Ident,
            sym::lifetime => NonterminalKind::Lifetime,
            sym::literal => NonterminalKind::Literal,
            sym::meta => NonterminalKind::Meta,
            sym::path => NonterminalKind::Path,
            sym::vis => NonterminalKind::Vis,
            sym::tt => NonterminalKind::TT,
            _ => return None,
        })
    }
    fn symbol(self) -> Symbol {
        match self {
            NonterminalKind::Item => sym::item,
            NonterminalKind::Block => sym::block,
            NonterminalKind::Stmt => sym::stmt,
            NonterminalKind::PatParam { inferred: false } => sym::pat_param,
            NonterminalKind::PatParam { inferred: true } | NonterminalKind::PatWithOr => sym::pat,
            NonterminalKind::Expr => sym::expr,
            NonterminalKind::Ty => sym::ty,
            NonterminalKind::Ident => sym::ident,
            NonterminalKind::Lifetime => sym::lifetime,
            NonterminalKind::Literal => sym::literal,
            NonterminalKind::Meta => sym::meta,
            NonterminalKind::Path => sym::path,
            NonterminalKind::Vis => sym::vis,
            NonterminalKind::TT => sym::tt,
        }
    }
}
impl fmt::Display for NonterminalKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

#[derive(Clone, Encodable, Decodable)]
pub enum Nonterminal {
    NtItem(P<Item>),
    NtBlock(P<Block>),
    NtStmt(P<Stmt>),
    NtPat(P<Pat>),
    NtExpr(P<Expr>),
    NtTy(P<Ty>),
    NtIdent(Ident, /* is_raw */ bool),
    NtLifetime(Ident),
    NtLiteral(P<Expr>),
    NtMeta(P<AttrItem>),
    NtPath(P<Path>),
    NtVis(P<Visibility>),
}
impl Nonterminal {
    pub fn span(&self) -> Span {
        match self {
            NtItem(item) => item.span,
            NtBlock(block) => block.span,
            NtStmt(stmt) => stmt.span,
            NtPat(pat) => pat.span,
            NtExpr(expr) | NtLiteral(expr) => expr.span,
            NtTy(ty) => ty.span,
            NtIdent(ident, _) | NtLifetime(ident) => ident.span,
            NtMeta(attr_item) => attr_item.span(),
            NtPath(path) => path.span,
            NtVis(vis) => vis.span,
        }
    }
}
impl PartialEq for Nonterminal {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (NtIdent(ident_lhs, is_raw_lhs), NtIdent(ident_rhs, is_raw_rhs)) => {
                ident_lhs == ident_rhs && is_raw_lhs == is_raw_rhs
            },
            (NtLifetime(ident_lhs), NtLifetime(ident_rhs)) => ident_lhs == ident_rhs,
            _ => false,
        }
    }
}
impl fmt::Debug for Nonterminal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            NtItem(..) => f.pad("NtItem(..)"),
            NtBlock(..) => f.pad("NtBlock(..)"),
            NtStmt(..) => f.pad("NtStmt(..)"),
            NtPat(..) => f.pad("NtPat(..)"),
            NtExpr(..) => f.pad("NtExpr(..)"),
            NtTy(..) => f.pad("NtTy(..)"),
            NtIdent(..) => f.pad("NtIdent(..)"),
            NtLiteral(..) => f.pad("NtLiteral(..)"),
            NtMeta(..) => f.pad("NtMeta(..)"),
            NtPath(..) => f.pad("NtPath(..)"),
            NtVis(..) => f.pad("NtVis(..)"),
            NtLifetime(..) => f.pad("NtLifetime(..)"),
        }
    }
}
impl<CTX> HashStable<CTX> for Nonterminal
where
    CTX: crate::ast::HashStableContext,
{
    fn hash_stable(&self, _hcx: &mut CTX, _hasher: &mut StableHasher) {
        panic!("interpolated tokens should not be present in the HIR")
    }
}
pub use Nonterminal::*;

pub mod stream {
    use crate::ast::{
        token::{self, Delimiter, Nonterminal, Token, TokenKind},
        *,
    };
    use macros::HashStable_Generic;
    use rustc_data_structures::{
        stable_hasher::{HashStable, StableHasher},
        sync::{self, Lrc},
    };
    use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
    use rustc_span::{Span, DUMMY_SP};
    use smallvec::{smallvec, SmallVec};
    use std::{fmt, iter};

    #[cfg(parallel_compiler)]
    fn _dummy()
    where
        Token: sync::DynSend + sync::DynSync,
        Spacing: sync::DynSend + sync::DynSync,
        DelimSpan: sync::DynSend + sync::DynSync,
        Delimiter: sync::DynSend + sync::DynSync,
        TokenStream: sync::DynSend + sync::DynSync,
    {
    }

    #[derive(Debug, Clone, PartialEq, Encodable, Decodable, HashStable_Generic)]
    pub enum TokenTree {
        Token(Token, Spacing),
        Delimited(DelimSpan, Delimiter, TokenStream),
    }
    impl TokenTree {
        pub fn eq_unspanned(&self, other: &TokenTree) -> bool {
            match (self, other) {
                (TokenTree::Token(token, _), TokenTree::Token(token2, _)) => token.kind == token2.kind,
                (TokenTree::Delimited(_, delim, tts), TokenTree::Delimited(_, delim2, tts2)) => {
                    delim == delim2 && tts.eq_unspanned(tts2)
                },
                _ => false,
            }
        }
        pub fn span(&self) -> Span {
            match self {
                TokenTree::Token(token, _) => token.span,
                TokenTree::Delimited(sp, ..) => sp.entire(),
            }
        }
        pub fn set_span(&mut self, span: Span) {
            match self {
                TokenTree::Token(token, _) => token.span = span,
                TokenTree::Delimited(dspan, ..) => *dspan = DelimSpan::from_single(span),
            }
        }
        pub fn token_alone(kind: TokenKind, span: Span) -> TokenTree {
            TokenTree::Token(Token::new(kind, span), Spacing::Alone)
        }
        pub fn token_joint(kind: TokenKind, span: Span) -> TokenTree {
            TokenTree::Token(Token::new(kind, span), Spacing::Joint)
        }
        pub fn uninterpolate(self) -> TokenTree {
            match self {
                TokenTree::Token(token, spacing) => TokenTree::Token(token.uninterpolate().into_owned(), spacing),
                tt => tt,
            }
        }
    }

    #[derive(Clone)]
    pub struct LazyAttrTokenStream(Lrc<Box<dyn ToAttrTokenStream>>);
    impl LazyAttrTokenStream {
        pub fn new(inner: impl ToAttrTokenStream + 'static) -> LazyAttrTokenStream {
            LazyAttrTokenStream(Lrc::new(Box::new(inner)))
        }
        pub fn to_attr_token_stream(&self) -> AttrTokenStream {
            self.0.to_attr_token_stream()
        }
    }
    impl fmt::Debug for LazyAttrTokenStream {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "LazyAttrTokenStream({:?})", self.to_attr_token_stream())
        }
    }
    impl<S: Encoder> Encodable<S> for LazyAttrTokenStream {
        fn encode(&self, s: &mut S) {
            Encodable::encode(&self.to_attr_token_stream(), s);
        }
    }
    impl<D: Decoder> Decodable<D> for LazyAttrTokenStream {
        fn decode(_d: &mut D) -> Self {
            panic!("Attempted to decode LazyAttrTokenStream");
        }
    }
    impl<CTX> HashStable<CTX> for LazyAttrTokenStream {
        fn hash_stable(&self, _hcx: &mut CTX, _hasher: &mut StableHasher) {
            panic!("Attempted to compute stable hash for LazyAttrTokenStream");
        }
    }

    #[derive(Clone, Debug, Default, Encodable, Decodable)]
    pub struct AttrTokenStream(pub Lrc<Vec<AttrTokenTree>>);
    impl AttrTokenStream {
        pub fn new(tokens: Vec<AttrTokenTree>) -> AttrTokenStream {
            AttrTokenStream(Lrc::new(tokens))
        }
        pub fn to_stream(&self) -> TokenStream {
            let trees: Vec<_> = self
                .0
                .iter()
                .flat_map(|tree| match &tree {
                    AttrTokenTree::Token(inner, spacing) => {
                        smallvec![TokenTree::Token(inner.clone(), *spacing)].into_iter()
                    },
                    AttrTokenTree::Delimited(span, delim, stream) => {
                        smallvec![TokenTree::Delimited(*span, *delim, stream.to_stream()),].into_iter()
                    },
                    AttrTokenTree::Attributes(data) => {
                        let mut outer_attrs = Vec::new();
                        let mut inner_attrs = Vec::new();
                        for attr in &data.attrs {
                            match attr.style {
                                crate::ast::AttrStyle::Outer => outer_attrs.push(attr),
                                crate::ast::AttrStyle::Inner => inner_attrs.push(attr),
                            }
                        }
                        let mut target_tokens: Vec<_> = data
                            .tokens
                            .to_attr_token_stream()
                            .to_stream()
                            .0
                            .iter()
                            .cloned()
                            .collect();
                        if !inner_attrs.is_empty() {
                            let mut found = false;
                            for tree in target_tokens.iter_mut().rev().take(2) {
                                if let TokenTree::Delimited(span, delim, delim_tokens) = tree {
                                    let mut stream = TokenStream::default();
                                    for inner_attr in inner_attrs {
                                        stream.push_stream(inner_attr.tokens());
                                    }
                                    stream.push_stream(delim_tokens.clone());
                                    *tree = TokenTree::Delimited(*span, *delim, stream);
                                    found = true;
                                    break;
                                }
                            }
                            assert!(found, "Failed to find trailing delimited group in: {target_tokens:?}");
                        }
                        let mut flat: SmallVec<[_; 1]> = SmallVec::new();
                        for attr in outer_attrs {
                            flat.extend(attr.tokens().0.clone().iter().cloned());
                        }
                        flat.extend(target_tokens);
                        flat.into_iter()
                    },
                })
                .collect();
            TokenStream::new(trees)
        }
    }

    pub trait ToAttrTokenStream: sync::DynSend + sync::DynSync {
        fn to_attr_token_stream(&self) -> AttrTokenStream;
    }
    impl ToAttrTokenStream for AttrTokenStream {
        fn to_attr_token_stream(&self) -> AttrTokenStream {
            self.clone()
        }
    }

    #[derive(Clone, Debug, Encodable, Decodable)]
    pub enum AttrTokenTree {
        Token(Token, Spacing),
        Delimited(DelimSpan, Delimiter, AttrTokenStream),
        Attributes(AttributesData),
    }

    #[derive(Clone, Debug, Encodable, Decodable)]
    pub struct AttributesData {
        pub attrs: AttrVec,
        pub tokens: LazyAttrTokenStream,
    }

    #[derive(Clone, Debug, Default, Encodable, Decodable)]
    pub struct TokenStream(pub(crate) Lrc<Vec<TokenTree>>);
    impl TokenStream {
        pub fn new(streams: Vec<TokenTree>) -> TokenStream {
            TokenStream(Lrc::new(streams))
        }
        pub fn is_empty(&self) -> bool {
            self.0.is_empty()
        }
        pub fn len(&self) -> usize {
            self.0.len()
        }
        pub fn trees(&self) -> RefTokenTreeCursor<'_> {
            RefTokenTreeCursor::new(self)
        }
        pub fn into_trees(self) -> TokenTreeCursor {
            TokenTreeCursor::new(self)
        }
        pub fn eq_unspanned(&self, other: &TokenStream) -> bool {
            let mut t1 = self.trees();
            let mut t2 = other.trees();
            for (t1, t2) in iter::zip(&mut t1, &mut t2) {
                if !t1.eq_unspanned(t2) {
                    return false;
                }
            }
            t1.next().is_none() && t2.next().is_none()
        }
        pub fn map_enumerated<F: FnMut(usize, &TokenTree) -> TokenTree>(self, mut f: F) -> TokenStream {
            TokenStream(Lrc::new(
                self.0.iter().enumerate().map(|(i, tree)| f(i, tree)).collect(),
            ))
        }
        pub fn token_alone(kind: TokenKind, span: Span) -> TokenStream {
            TokenStream::new(vec![TokenTree::token_alone(kind, span)])
        }
        pub fn token_joint(kind: TokenKind, span: Span) -> TokenStream {
            TokenStream::new(vec![TokenTree::token_joint(kind, span)])
        }
        pub fn delimited(span: DelimSpan, delim: Delimiter, tts: TokenStream) -> TokenStream {
            TokenStream::new(vec![TokenTree::Delimited(span, delim, tts)])
        }
        pub fn from_ast(node: &(impl HasAttrs + HasSpan + HasTokens + fmt::Debug)) -> TokenStream {
            let Some(tokens) = node.tokens() else {
                panic!("missing tokens for node at {:?}: {:?}", node.span(), node);
            };
            let attrs = node.attrs();
            let attr_stream = if attrs.is_empty() {
                tokens.to_attr_token_stream()
            } else {
                let attr_data = AttributesData {
                    attrs: attrs.iter().cloned().collect(),
                    tokens: tokens.clone(),
                };
                AttrTokenStream::new(vec![AttrTokenTree::Attributes(attr_data)])
            };
            attr_stream.to_stream()
        }
        pub fn from_nonterminal_ast(nt: &Nonterminal) -> TokenStream {
            match nt {
                Nonterminal::NtIdent(ident, is_raw) => {
                    TokenStream::token_alone(token::Ident(ident.name, *is_raw), ident.span)
                }
                Nonterminal::NtLifetime(ident) => {
                    TokenStream::token_alone(token::Lifetime(ident.name), ident.span)
                }
                Nonterminal::NtItem(item) => TokenStream::from_ast(item),
                Nonterminal::NtBlock(block) => TokenStream::from_ast(block),
                Nonterminal::NtStmt(stmt) if let StmtKind::Empty = stmt.kind => {
                    TokenStream::token_alone(token::Semi, stmt.span)
                }
                Nonterminal::NtStmt(stmt) => TokenStream::from_ast(stmt),
                Nonterminal::NtPat(pat) => TokenStream::from_ast(pat),
                Nonterminal::NtTy(ty) => TokenStream::from_ast(ty),
                Nonterminal::NtMeta(attr) => TokenStream::from_ast(attr),
                Nonterminal::NtPath(path) => TokenStream::from_ast(path),
                Nonterminal::NtVis(vis) => TokenStream::from_ast(vis),
                Nonterminal::NtExpr(expr) | Nonterminal::NtLiteral(expr) => TokenStream::from_ast(expr),
            }
        }
        fn flatten_token(token: &Token, spacing: Spacing) -> TokenTree {
            match &token.kind {
                token::Interpolated(nt) if let token::NtIdent(ident, is_raw) = **nt => {
                    TokenTree::Token(Token::new(token::Ident(ident.name, is_raw), ident.span), spacing)
                }
                token::Interpolated(nt) => TokenTree::Delimited(
                    DelimSpan::from_single(token.span),
                    Delimiter::Invisible,
                    TokenStream::from_nonterminal_ast(nt).flattened(),
                ),
                _ => TokenTree::Token(token.clone(), spacing),
            }
        }
        fn flatten_token_tree(tree: &TokenTree) -> TokenTree {
            match tree {
                TokenTree::Token(token, spacing) => TokenStream::flatten_token(token, *spacing),
                TokenTree::Delimited(span, delim, tts) => TokenTree::Delimited(*span, *delim, tts.flattened()),
            }
        }
        #[must_use]
        pub fn flattened(&self) -> TokenStream {
            fn can_skip(stream: &TokenStream) -> bool {
                stream.trees().all(|tree| match tree {
                    TokenTree::Token(token, _) => !matches!(token.kind, token::Interpolated(_)),
                    TokenTree::Delimited(_, _, inner) => can_skip(inner),
                })
            }
            if can_skip(self) {
                return self.clone();
            }
            self.trees().map(|tree| TokenStream::flatten_token_tree(tree)).collect()
        }
        fn try_glue_to_last(vec: &mut Vec<TokenTree>, tt: &TokenTree) -> bool {
            if let Some(TokenTree::Token(last_tok, Spacing::Joint)) = vec.last()
                && let TokenTree::Token(tok, spacing) = tt
                && let Some(glued_tok) = last_tok.glue(tok)
            {
                *vec.last_mut().unwrap() = TokenTree::Token(glued_tok, *spacing);
                true
            } else {
                false
            }
        }
        pub fn add_comma(&self) -> Option<(TokenStream, Span)> {
            let mut suggestion = None;
            let mut iter = self.0.iter().enumerate().peekable();
            while let Some((pos, ts)) = iter.next() {
                if let Some((_, next)) = iter.peek() {
                    let sp = match (&ts, &next) {
                        (_, TokenTree::Token(Token { kind: token::Comma, .. }, _)) => continue,
                        (TokenTree::Token(token_left, Spacing::Alone), TokenTree::Token(token_right, _))
                            if ((token_left.is_ident() && !token_left.is_reserved_ident()) || token_left.is_lit())
                                && ((token_right.is_ident() && !token_right.is_reserved_ident())
                                    || token_right.is_lit()) =>
                        {
                            token_left.span
                        },
                        (TokenTree::Delimited(sp, ..), _) => sp.entire(),
                        _ => continue,
                    };
                    let sp = sp.shrink_to_hi();
                    let comma = TokenTree::token_alone(token::Comma, sp);
                    suggestion = Some((pos, comma, sp));
                }
            }
            if let Some((pos, comma, sp)) = suggestion {
                let mut new_stream = Vec::with_capacity(self.0.len() + 1);
                let parts = self.0.split_at(pos + 1);
                new_stream.extend_from_slice(parts.0);
                new_stream.push(comma);
                new_stream.extend_from_slice(parts.1);
                return Some((TokenStream::new(new_stream), sp));
            }
            None
        }
        pub fn push_tree(&mut self, tt: TokenTree) {
            let vec_mut = Lrc::make_mut(&mut self.0);
            if Self::try_glue_to_last(vec_mut, &tt) {
            } else {
                vec_mut.push(tt);
            }
        }
        pub fn push_stream(&mut self, stream: TokenStream) {
            let vec_mut = Lrc::make_mut(&mut self.0);
            let stream_iter = stream.0.iter().cloned();
            if let Some(first) = stream.0.first() && Self::try_glue_to_last(vec_mut, first) {
                vec_mut.extend(stream_iter.skip(1));
            } else {
                vec_mut.extend(stream_iter);
            }
        }
        pub fn chunks(&self, chunk_size: usize) -> core::slice::Chunks<'_, TokenTree> {
            self.0.chunks(chunk_size)
        }
    }
    impl FromIterator<TokenTree> for TokenStream {
        fn from_iter<I: IntoIterator<Item = TokenTree>>(iter: I) -> Self {
            TokenStream::new(iter.into_iter().collect::<Vec<TokenTree>>())
        }
    }
    impl Eq for TokenStream {}
    impl PartialEq<TokenStream> for TokenStream {
        fn eq(&self, other: &TokenStream) -> bool {
            self.trees().eq(other.trees())
        }
    }
    impl<CTX> HashStable<CTX> for TokenStream
    where
        CTX: crate::ast::HashStableContext,
    {
        fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
            for sub_tt in self.trees() {
                sub_tt.hash_stable(hcx, hasher);
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Encodable, Decodable, HashStable_Generic)]
    pub enum Spacing {
        Alone,
        Joint,
    }

    #[derive(Clone)]
    pub struct RefTokenTreeCursor<'t> {
        stream: &'t TokenStream,
        index: usize,
    }
    impl<'t> RefTokenTreeCursor<'t> {
        fn new(stream: &'t TokenStream) -> Self {
            RefTokenTreeCursor { stream, index: 0 }
        }
        pub fn look_ahead(&self, n: usize) -> Option<&TokenTree> {
            self.stream.0.get(self.index + n)
        }
    }
    impl<'t> Iterator for RefTokenTreeCursor<'t> {
        type Item = &'t TokenTree;
        fn next(&mut self) -> Option<&'t TokenTree> {
            self.stream.0.get(self.index).map(|tree| {
                self.index += 1;
                tree
            })
        }
    }

    #[derive(Clone)]
    pub struct TokenTreeCursor {
        pub stream: TokenStream,
        index: usize,
    }
    impl TokenTreeCursor {
        fn new(stream: TokenStream) -> Self {
            TokenTreeCursor { stream, index: 0 }
        }
        #[inline]
        pub fn next_ref(&mut self) -> Option<&TokenTree> {
            self.stream.0.get(self.index).map(|tree| {
                self.index += 1;
                tree
            })
        }
        pub fn look_ahead(&self, n: usize) -> Option<&TokenTree> {
            self.stream.0.get(self.index + n)
        }
        pub fn replace_prev_and_rewind(&mut self, tts: Vec<TokenTree>) {
            assert!(self.index > 0);
            self.index -= 1;
            let stream = Lrc::make_mut(&mut self.stream.0);
            stream.splice(self.index..self.index + 1, tts);
        }
    }
    impl Iterator for TokenTreeCursor {
        type Item = TokenTree;
        fn next(&mut self) -> Option<TokenTree> {
            self.stream.0.get(self.index).map(|tree| {
                self.index += 1;
                tree.clone()
            })
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq, Encodable, Decodable, HashStable_Generic)]
    pub struct DelimSpan {
        pub open: Span,
        pub close: Span,
    }
    impl DelimSpan {
        pub fn from_single(sp: Span) -> Self {
            DelimSpan { open: sp, close: sp }
        }
        pub fn from_pair(open: Span, close: Span) -> Self {
            DelimSpan { open, close }
        }
        pub fn dummy() -> Self {
            Self::from_single(DUMMY_SP)
        }
        pub fn entire(self) -> Span {
            self.open.with_hi(self.close.hi())
        }
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    mod size_asserts {
        use super::*;
        use rustc_data_structures::static_assert_size;
        static_assert_size!(AttrTokenStream, 8);
        static_assert_size!(AttrTokenTree, 32);
        static_assert_size!(LazyAttrTokenStream, 8);
        static_assert_size!(TokenStream, 8);
        static_assert_size!(TokenTree, 32);
    }
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod size_asserts {
    use super::*;
    use rustc_data_structures::static_assert_size;
    static_assert_size!(Lit, 12);
    static_assert_size!(LiteralKind, 2);
    static_assert_size!(Nonterminal, 16);
    static_assert_size!(Token, 24);
    static_assert_size!(TokenKind, 16);
}
