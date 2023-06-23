use crate::ast::{self, BinOpKind};
use crate::token::{self, BinOpToken, Token};
use rustc_span::symbol::kw;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum AssocOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulus,
    LAnd,
    LOr,
    BitXor,
    BitAnd,
    BitOr,
    ShiftLeft,
    ShiftRight,
    Equal,
    Less,
    LessEqual,
    NotEqual,
    Greater,
    GreaterEqual,
    Assign,
    AssignOp(BinOpToken),
    As,
    DotDot,
    DotDotEq,
}

#[derive(PartialEq, Debug)]
pub enum Fixity {
    Left,
    Right,
    None,
}

impl AssocOp {
    pub fn from_token(t: &Token) -> Option<AssocOp> {
        use AssocOp::*;
        match t.kind {
            token::BinOpEq(k) => Some(AssignOp(k)),
            token::Eq => Some(Assign),
            token::BinOp(BinOpToken::Star) => Some(Multiply),
            token::BinOp(BinOpToken::Slash) => Some(Divide),
            token::BinOp(BinOpToken::Percent) => Some(Modulus),
            token::BinOp(BinOpToken::Plus) => Some(Add),
            token::BinOp(BinOpToken::Minus) => Some(Subtract),
            token::BinOp(BinOpToken::Shl) => Some(ShiftLeft),
            token::BinOp(BinOpToken::Shr) => Some(ShiftRight),
            token::BinOp(BinOpToken::And) => Some(BitAnd),
            token::BinOp(BinOpToken::Caret) => Some(BitXor),
            token::BinOp(BinOpToken::Or) => Some(BitOr),
            token::Lt => Some(Less),
            token::Le => Some(LessEqual),
            token::Ge => Some(GreaterEqual),
            token::Gt => Some(Greater),
            token::EqEq => Some(Equal),
            token::Ne => Some(NotEqual),
            token::AndAnd => Some(LAnd),
            token::OrOr => Some(LOr),
            token::DotDot => Some(DotDot),
            token::DotDotEq => Some(DotDotEq),
            token::DotDotDot => Some(DotDotEq),
            token::LArrow => Some(Less),
            _ if t.is_keyword(kw::As) => Some(As),
            _ => None,
        }
    }

    pub fn from_ast_binop(op: BinOpKind) -> Self {
        use AssocOp::*;
        match op {
            BinOpKind::Lt => Less,
            BinOpKind::Gt => Greater,
            BinOpKind::Le => LessEqual,
            BinOpKind::Ge => GreaterEqual,
            BinOpKind::Eq => Equal,
            BinOpKind::Ne => NotEqual,
            BinOpKind::Mul => Multiply,
            BinOpKind::Div => Divide,
            BinOpKind::Rem => Modulus,
            BinOpKind::Add => Add,
            BinOpKind::Sub => Subtract,
            BinOpKind::Shl => ShiftLeft,
            BinOpKind::Shr => ShiftRight,
            BinOpKind::BitAnd => BitAnd,
            BinOpKind::BitXor => BitXor,
            BinOpKind::BitOr => BitOr,
            BinOpKind::And => LAnd,
            BinOpKind::Or => LOr,
        }
    }

    pub fn precedence(&self) -> usize {
        use AssocOp::*;
        match *self {
            As => 14,
            Multiply | Divide | Modulus => 13,
            Add | Subtract => 12,
            ShiftLeft | ShiftRight => 11,
            BitAnd => 10,
            BitXor => 9,
            BitOr => 8,
            Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual => 7,
            LAnd => 6,
            LOr => 5,
            DotDot | DotDotEq => 4,
            Assign | AssignOp(_) => 2,
        }
    }

    pub fn fixity(&self) -> Fixity {
        use AssocOp::*;
        match *self {
            Assign | AssignOp(_) => Fixity::Right,
            As | Multiply | Divide | Modulus | Add | Subtract | ShiftLeft | ShiftRight | BitAnd | BitXor | BitOr
            | Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual | LAnd | LOr => Fixity::Left,
            DotDot | DotDotEq => Fixity::None,
        }
    }

    pub fn is_comparison(&self) -> bool {
        use AssocOp::*;
        match *self {
            Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual => true,
            Assign | AssignOp(_) | As | Multiply | Divide | Modulus | Add | Subtract | ShiftLeft | ShiftRight
            | BitAnd | BitXor | BitOr | LAnd | LOr | DotDot | DotDotEq => false,
        }
    }

    pub fn is_assign_like(&self) -> bool {
        use AssocOp::*;
        match *self {
            Assign | AssignOp(_) => true,
            Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual | As | Multiply | Divide | Modulus | Add
            | Subtract | ShiftLeft | ShiftRight | BitAnd | BitXor | BitOr | LAnd | LOr | DotDot | DotDotEq => false,
        }
    }

    pub fn to_ast_binop(&self) -> Option<BinOpKind> {
        use AssocOp::*;
        match *self {
            Less => Some(BinOpKind::Lt),
            Greater => Some(BinOpKind::Gt),
            LessEqual => Some(BinOpKind::Le),
            GreaterEqual => Some(BinOpKind::Ge),
            Equal => Some(BinOpKind::Eq),
            NotEqual => Some(BinOpKind::Ne),
            Multiply => Some(BinOpKind::Mul),
            Divide => Some(BinOpKind::Div),
            Modulus => Some(BinOpKind::Rem),
            Add => Some(BinOpKind::Add),
            Subtract => Some(BinOpKind::Sub),
            ShiftLeft => Some(BinOpKind::Shl),
            ShiftRight => Some(BinOpKind::Shr),
            BitAnd => Some(BinOpKind::BitAnd),
            BitXor => Some(BinOpKind::BitXor),
            BitOr => Some(BinOpKind::BitOr),
            LAnd => Some(BinOpKind::And),
            LOr => Some(BinOpKind::Or),
            Assign | AssignOp(_) | As | DotDot | DotDotEq => None,
        }
    }

    pub fn can_continue_expr_unambiguously(&self) -> bool {
        use AssocOp::*;
        matches!(
            self,
            BitXor | // `{ 42 } ^ 3`
            Assign | // `{ 42 } = { 42 }`
            Divide | // `{ 42 } / 42`
            Modulus | // `{ 42 } % 2`
            ShiftRight | // `{ 42 } >> 2`
            LessEqual | // `{ 42 } <= 3`
            Greater | // `{ 42 } > 3`
            GreaterEqual | // `{ 42 } >= 3`
            AssignOp(_) | // `{ 42 } +=`
            As // `{ 42 } as usize`
        )
    }
}

pub const PREC_CLOSURE: i8 = -40;
pub const PREC_JUMP: i8 = -30;
pub const PREC_RANGE: i8 = -10;
pub const PREC_PREFIX: i8 = 50;
pub const PREC_POSTFIX: i8 = 60;
pub const PREC_PAREN: i8 = 99;
pub const PREC_FORCE_PAREN: i8 = 100;

#[derive(Debug, Clone, Copy)]
pub enum ExprPrecedence {
    Closure,
    Break,
    Continue,
    Ret,
    Yield,
    Yeet,
    Become,

    Range,

    Binary(BinOpKind),

    Cast,

    Assign,
    AssignOp,

    AddrOf,
    Let,
    Unary,

    Call,
    MethodCall,
    Field,
    Index,
    Try,
    InlineAsm,
    OffsetOf,
    Mac,
    FormatArgs,

    Array,
    Repeat,
    Tup,
    Lit,
    Path,
    Paren,
    If,
    While,
    ForLoop,
    Loop,
    Match,
    ConstBlock,
    Block,
    TryBlock,
    Struct,
    Async,
    Await,
    Err,
}

impl ExprPrecedence {
    pub fn order(self) -> i8 {
        match self {
            ExprPrecedence::Closure => PREC_CLOSURE,

            ExprPrecedence::Break
            | ExprPrecedence::Continue
            | ExprPrecedence::Ret
            | ExprPrecedence::Yield
            | ExprPrecedence::Yeet
            | ExprPrecedence::Become => PREC_JUMP,

            ExprPrecedence::Range => PREC_RANGE,

            ExprPrecedence::Binary(op) => AssocOp::from_ast_binop(op).precedence() as i8,
            ExprPrecedence::Cast => AssocOp::As.precedence() as i8,

            ExprPrecedence::Assign | ExprPrecedence::AssignOp => AssocOp::Assign.precedence() as i8,

            ExprPrecedence::AddrOf | ExprPrecedence::Let | ExprPrecedence::Unary => PREC_PREFIX,

            ExprPrecedence::Await
            | ExprPrecedence::Call
            | ExprPrecedence::MethodCall
            | ExprPrecedence::Field
            | ExprPrecedence::Index
            | ExprPrecedence::Try
            | ExprPrecedence::InlineAsm
            | ExprPrecedence::Mac
            | ExprPrecedence::FormatArgs
            | ExprPrecedence::OffsetOf => PREC_POSTFIX,

            ExprPrecedence::Array
            | ExprPrecedence::Repeat
            | ExprPrecedence::Tup
            | ExprPrecedence::Lit
            | ExprPrecedence::Path
            | ExprPrecedence::Paren
            | ExprPrecedence::If
            | ExprPrecedence::While
            | ExprPrecedence::ForLoop
            | ExprPrecedence::Loop
            | ExprPrecedence::Match
            | ExprPrecedence::ConstBlock
            | ExprPrecedence::Block
            | ExprPrecedence::TryBlock
            | ExprPrecedence::Async
            | ExprPrecedence::Struct
            | ExprPrecedence::Err => PREC_PAREN,
        }
    }
}

pub fn prec_let_scrutinee_needs_par() -> usize {
    AssocOp::LAnd.precedence()
}

pub fn needs_par_as_let_scrutinee(order: i8) -> bool {
    order <= prec_let_scrutinee_needs_par() as i8
}

pub fn contains_exterior_struct_lit(value: &ast::Expr) -> bool {
    match &value.kind {
        ast::ExprKind::Struct(..) => true,

        ast::ExprKind::Assign(lhs, rhs, _)
        | ast::ExprKind::AssignOp(_, lhs, rhs)
        | ast::ExprKind::Binary(_, lhs, rhs) => contains_exterior_struct_lit(lhs) || contains_exterior_struct_lit(rhs),
        ast::ExprKind::Await(x, _)
        | ast::ExprKind::Unary(_, x)
        | ast::ExprKind::Cast(x, _)
        | ast::ExprKind::Type(x, _)
        | ast::ExprKind::Field(x, _)
        | ast::ExprKind::Index(x, _) => contains_exterior_struct_lit(x),

        ast::ExprKind::MethodCall(box ast::MethodCall { receiver, .. }) => contains_exterior_struct_lit(receiver),

        _ => false,
    }
}