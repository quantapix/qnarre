use super::*;
use crate::punct::Punctuated;
use proc_macro2::{Span, TokenStream};
use quote::IdentFragment;
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};
use std::mem;
ast_enum_of_structs! {
    pub enum Expr {
        Array(ExprArray),
        Assign(ExprAssign),
        Async(ExprAsync),
        Await(ExprAwait),
        Binary(ExprBinary),
        Block(ExprBlock),
        Break(ExprBreak),
        Call(ExprCall),
        Cast(ExprCast),
        Closure(ExprClosure),
        Const(ExprConst),
        Continue(ExprContinue),
        Field(ExprField),
        ForLoop(ExprForLoop),
        Group(ExprGroup),
        If(ExprIf),
        Index(ExprIndex),
        Infer(ExprInfer),
        Let(ExprLet),
        Lit(ExprLit),
        Loop(ExprLoop),
        Macro(ExprMacro),
        Match(ExprMatch),
        MethodCall(ExprMethodCall),
        Paren(ExprParen),
        Path(ExprPath),
        Range(ExprRange),
        Reference(ExprReference),
        Repeat(ExprRepeat),
        Return(ExprReturn),
        Struct(ExprStruct),
        Try(ExprTry),
        TryBlock(ExprTryBlock),
        Tuple(ExprTuple),
        Unary(ExprUnary),
        Unsafe(ExprUnsafe),
        Verbatim(TokenStream),
        While(ExprWhile),
        Yield(ExprYield),
    }
}
ast_struct! {
    pub struct ExprArray #full {
        pub attrs: Vec<Attribute>,
        pub bracket_token: tok::Bracket,
        pub elems: Punctuated<Expr, Token![,]>,
    }
}
ast_struct! {
    pub struct ExprAssign #full {
        pub attrs: Vec<Attribute>,
        pub left: Box<Expr>,
        pub eq_token: Token![=],
        pub right: Box<Expr>,
    }
}
ast_struct! {
    pub struct ExprAsync #full {
        pub attrs: Vec<Attribute>,
        pub async_token: Token![async],
        pub capture: Option<Token![move]>,
        pub block: Block,
    }
}
ast_struct! {
    pub struct ExprAwait #full {
        pub attrs: Vec<Attribute>,
        pub base: Box<Expr>,
        pub dot_token: Token![.],
        pub await_token: Token![await],
    }
}
ast_struct! {
    pub struct ExprBinary {
        pub attrs: Vec<Attribute>,
        pub left: Box<Expr>,
        pub op: BinOp,
        pub right: Box<Expr>,
    }
}
ast_struct! {
    pub struct ExprBlock #full {
        pub attrs: Vec<Attribute>,
        pub label: Option<Label>,
        pub block: Block,
    }
}
ast_struct! {
    pub struct ExprBreak #full {
        pub attrs: Vec<Attribute>,
        pub break_token: Token![break],
        pub label: Option<Lifetime>,
        pub expr: Option<Box<Expr>>,
    }
}
ast_struct! {
    pub struct ExprCall {
        pub attrs: Vec<Attribute>,
        pub func: Box<Expr>,
        pub paren_token: tok::Paren,
        pub args: Punctuated<Expr, Token![,]>,
    }
}
ast_struct! {
    pub struct ExprCast {
        pub attrs: Vec<Attribute>,
        pub expr: Box<Expr>,
        pub as_token: Token![as],
        pub ty: Box<Type>,
    }
}
ast_struct! {
    pub struct ExprClosure #full {
        pub attrs: Vec<Attribute>,
        pub lifetimes: Option<BoundLifetimes>,
        pub constness: Option<Token![const]>,
        pub movability: Option<Token![static]>,
        pub asyncness: Option<Token![async]>,
        pub capture: Option<Token![move]>,
        pub or1_token: Token![|],
        pub inputs: Punctuated<Pat, Token![,]>,
        pub or2_token: Token![|],
        pub output: ReturnType,
        pub body: Box<Expr>,
    }
}
ast_struct! {
    pub struct ExprConst #full {
        pub attrs: Vec<Attribute>,
        pub const_token: Token![const],
        pub block: Block,
    }
}
ast_struct! {
    pub struct ExprContinue #full {
        pub attrs: Vec<Attribute>,
        pub continue_token: Token![continue],
        pub label: Option<Lifetime>,
    }
}
ast_struct! {
    pub struct ExprField {
        pub attrs: Vec<Attribute>,
        pub base: Box<Expr>,
        pub dot_token: Token![.],
        pub member: Member,
    }
}
ast_struct! {
    pub struct ExprForLoop #full {
        pub attrs: Vec<Attribute>,
        pub label: Option<Label>,
        pub for_token: Token![for],
        pub pat: Box<Pat>,
        pub in_token: Token![in],
        pub expr: Box<Expr>,
        pub body: Block,
    }
}
ast_struct! {
    pub struct ExprGroup {
        pub attrs: Vec<Attribute>,
        pub group_token: tok::Group,
        pub expr: Box<Expr>,
    }
}
ast_struct! {
    pub struct ExprIf #full {
        pub attrs: Vec<Attribute>,
        pub if_token: Token![if],
        pub cond: Box<Expr>,
        pub then_branch: Block,
        pub else_branch: Option<(Token![else], Box<Expr>)>,
    }
}
ast_struct! {
    pub struct ExprIndex {
        pub attrs: Vec<Attribute>,
        pub expr: Box<Expr>,
        pub bracket_token: tok::Bracket,
        pub index: Box<Expr>,
    }
}
ast_struct! {
    pub struct ExprInfer #full {
        pub attrs: Vec<Attribute>,
        pub underscore_token: Token![_],
    }
}
ast_struct! {
    pub struct ExprLet #full {
        pub attrs: Vec<Attribute>,
        pub let_token: Token![let],
        pub pat: Box<Pat>,
        pub eq_token: Token![=],
        pub expr: Box<Expr>,
    }
}
ast_struct! {
    pub struct ExprLit {
        pub attrs: Vec<Attribute>,
        pub lit: Lit,
    }
}
ast_struct! {
    pub struct ExprLoop #full {
        pub attrs: Vec<Attribute>,
        pub label: Option<Label>,
        pub loop_token: Token![loop],
        pub body: Block,
    }
}
ast_struct! {
    pub struct ExprMacro {
        pub attrs: Vec<Attribute>,
        pub mac: Macro,
    }
}
ast_struct! {
    pub struct ExprMatch #full {
        pub attrs: Vec<Attribute>,
        pub match_token: Token![match],
        pub expr: Box<Expr>,
        pub brace_token: tok::Brace,
        pub arms: Vec<Arm>,
    }
}
ast_struct! {
    pub struct ExprMethodCall #full {
        pub attrs: Vec<Attribute>,
        pub receiver: Box<Expr>,
        pub dot_token: Token![.],
        pub method: Ident,
        pub turbofish: Option<AngledArgs>,
        pub paren_token: tok::Paren,
        pub args: Punctuated<Expr, Token![,]>,
    }
}
ast_struct! {
    pub struct ExprParen {
        pub attrs: Vec<Attribute>,
        pub paren_token: tok::Paren,
        pub expr: Box<Expr>,
    }
}
ast_struct! {
    pub struct ExprPath {
        pub attrs: Vec<Attribute>,
        pub qself: Option<QSelf>,
        pub path: Path,
    }
}
ast_struct! {
    pub struct ExprRange #full {
        pub attrs: Vec<Attribute>,
        pub start: Option<Box<Expr>>,
        pub limits: RangeLimits,
        pub end: Option<Box<Expr>>,
    }
}
ast_struct! {
    pub struct ExprReference #full {
        pub attrs: Vec<Attribute>,
        pub and_token: Token![&],
        pub mutability: Option<Token![mut]>,
        pub expr: Box<Expr>,
    }
}
ast_struct! {
    pub struct ExprRepeat #full {
        pub attrs: Vec<Attribute>,
        pub bracket_token: tok::Bracket,
        pub expr: Box<Expr>,
        pub semi_token: Token![;],
        pub len: Box<Expr>,
    }
}
ast_struct! {
    pub struct ExprReturn #full {
        pub attrs: Vec<Attribute>,
        pub return_token: Token![return],
        pub expr: Option<Box<Expr>>,
    }
}
ast_struct! {
    pub struct ExprStruct #full {
        pub attrs: Vec<Attribute>,
        pub qself: Option<QSelf>,
        pub path: Path,
        pub brace_token: tok::Brace,
        pub fields: Punctuated<FieldValue, Token![,]>,
        pub dot2_token: Option<Token![..]>,
        pub rest: Option<Box<Expr>>,
    }
}
ast_struct! {
    pub struct ExprTry #full {
        pub attrs: Vec<Attribute>,
        pub expr: Box<Expr>,
        pub question_token: Token![?],
    }
}
ast_struct! {
    pub struct ExprTryBlock #full {
        pub attrs: Vec<Attribute>,
        pub try_token: Token![try],
        pub block: Block,
    }
}
ast_struct! {
    pub struct ExprTuple #full {
        pub attrs: Vec<Attribute>,
        pub paren_token: tok::Paren,
        pub elems: Punctuated<Expr, Token![,]>,
    }
}
ast_struct! {
    pub struct ExprUnary {
        pub attrs: Vec<Attribute>,
        pub op: UnOp,
        pub expr: Box<Expr>,
    }
}
ast_struct! {
    pub struct ExprUnsafe #full {
        pub attrs: Vec<Attribute>,
        pub unsafe_token: Token![unsafe],
        pub block: Block,
    }
}
ast_struct! {
    pub struct ExprWhile #full {
        pub attrs: Vec<Attribute>,
        pub label: Option<Label>,
        pub while_token: Token![while],
        pub cond: Box<Expr>,
        pub body: Block,
    }
}
ast_struct! {
    pub struct ExprYield #full {
        pub attrs: Vec<Attribute>,
        pub yield_token: Token![yield],
        pub expr: Option<Box<Expr>>,
    }
}
impl Expr {
    pub const DUMMY: Self = Expr::Path(ExprPath {
        attrs: Vec::new(),
        qself: None,
        path: Path {
            colon: None,
            segs: Punctuated::new(),
        },
    });
    pub(crate) fn replace_attrs(&mut self, new: Vec<Attribute>) -> Vec<Attribute> {
        match self {
            Expr::Array(ExprArray { attrs, .. })
            | Expr::Assign(ExprAssign { attrs, .. })
            | Expr::Async(ExprAsync { attrs, .. })
            | Expr::Await(ExprAwait { attrs, .. })
            | Expr::Binary(ExprBinary { attrs, .. })
            | Expr::Block(ExprBlock { attrs, .. })
            | Expr::Break(ExprBreak { attrs, .. })
            | Expr::Call(ExprCall { attrs, .. })
            | Expr::Cast(ExprCast { attrs, .. })
            | Expr::Closure(ExprClosure { attrs, .. })
            | Expr::Const(ExprConst { attrs, .. })
            | Expr::Continue(ExprContinue { attrs, .. })
            | Expr::Field(ExprField { attrs, .. })
            | Expr::ForLoop(ExprForLoop { attrs, .. })
            | Expr::Group(ExprGroup { attrs, .. })
            | Expr::If(ExprIf { attrs, .. })
            | Expr::Index(ExprIndex { attrs, .. })
            | Expr::Infer(ExprInfer { attrs, .. })
            | Expr::Let(ExprLet { attrs, .. })
            | Expr::Lit(ExprLit { attrs, .. })
            | Expr::Loop(ExprLoop { attrs, .. })
            | Expr::Macro(ExprMacro { attrs, .. })
            | Expr::Match(ExprMatch { attrs, .. })
            | Expr::MethodCall(ExprMethodCall { attrs, .. })
            | Expr::Paren(ExprParen { attrs, .. })
            | Expr::Path(ExprPath { attrs, .. })
            | Expr::Range(ExprRange { attrs, .. })
            | Expr::Reference(ExprReference { attrs, .. })
            | Expr::Repeat(ExprRepeat { attrs, .. })
            | Expr::Return(ExprReturn { attrs, .. })
            | Expr::Struct(ExprStruct { attrs, .. })
            | Expr::Try(ExprTry { attrs, .. })
            | Expr::TryBlock(ExprTryBlock { attrs, .. })
            | Expr::Tuple(ExprTuple { attrs, .. })
            | Expr::Unary(ExprUnary { attrs, .. })
            | Expr::Unsafe(ExprUnsafe { attrs, .. })
            | Expr::While(ExprWhile { attrs, .. })
            | Expr::Yield(ExprYield { attrs, .. }) => mem::replace(attrs, new),
            Expr::Verbatim(_) => Vec::new(),
        }
    }
}
ast_enum! {
    pub enum Member {
        Named(Ident),
        Unnamed(Index),
    }
}
impl From<Ident> for Member {
    fn from(ident: Ident) -> Member {
        Member::Named(ident)
    }
}
impl From<Index> for Member {
    fn from(index: Index) -> Member {
        Member::Unnamed(index)
    }
}
impl From<usize> for Member {
    fn from(index: usize) -> Member {
        Member::Unnamed(Index::from(index))
    }
}
impl Eq for Member {}
impl PartialEq for Member {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Member::Named(this), Member::Named(other)) => this == other,
            (Member::Unnamed(this), Member::Unnamed(other)) => this == other,
            _ => false,
        }
    }
}
impl Hash for Member {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Member::Named(m) => m.hash(state),
            Member::Unnamed(m) => m.hash(state),
        }
    }
}
impl IdentFragment for Member {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Member::Named(x) => Display::fmt(x, f),
            Member::Unnamed(x) => Display::fmt(&x.index, f),
        }
    }
    fn span(&self) -> Option<Span> {
        match self {
            Member::Named(m) => Some(m.span()),
            Member::Unnamed(m) => Some(m.span),
        }
    }
}
ast_struct! {
    pub struct Index {
        pub index: u32,
        pub span: Span,
    }
}
impl From<usize> for Index {
    fn from(index: usize) -> Index {
        assert!(index < u32::max_value() as usize);
        Index {
            index: index as u32,
            span: Span::call_site(),
        }
    }
}
impl Eq for Index {}
impl PartialEq for Index {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}
impl Hash for Index {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}
impl IdentFragment for Index {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.index, f)
    }
    fn span(&self) -> Option<Span> {
        Some(self.span)
    }
}
ast_struct! {
    pub struct FieldValue {
        pub attrs: Vec<Attribute>,
        pub member: Member,
        pub colon_token: Option<Token![:]>,
        pub expr: Expr,
    }
}
ast_struct! {
    pub struct Label {
        pub name: Lifetime,
        pub colon_token: Token![:],
    }
}
ast_struct! {
    pub struct Arm {
        pub attrs: Vec<Attribute>,
        pub pat: Pat,
        pub guard: Option<(Token![if], Box<Expr>)>,
        pub fat_arrow_token: Token![=>],
        pub body: Box<Expr>,
        pub comma: Option<Token![,]>,
    }
}
ast_enum! {
    pub enum RangeLimits {
        HalfOpen(Token![..]),
        Closed(Token![..=]),
    }
}
pub(crate) fn requires_terminator(expr: &Expr) -> bool {
    match expr {
        Expr::If(_)
        | Expr::Match(_)
        | Expr::Block(_) | Expr::Unsafe(_) // both under ExprKind::Block in rustc
        | Expr::While(_)
        | Expr::Loop(_)
        | Expr::ForLoop(_)
        | Expr::TryBlock(_)
        | Expr::Const(_) => false,
        Expr::Array(_)
        | Expr::Assign(_)
        | Expr::Async(_)
        | Expr::Await(_)
        | Expr::Binary(_)
        | Expr::Break(_)
        | Expr::Call(_)
        | Expr::Cast(_)
        | Expr::Closure(_)
        | Expr::Continue(_)
        | Expr::Field(_)
        | Expr::Group(_)
        | Expr::Index(_)
        | Expr::Infer(_)
        | Expr::Let(_)
        | Expr::Lit(_)
        | Expr::Macro(_)
        | Expr::MethodCall(_)
        | Expr::Paren(_)
        | Expr::Path(_)
        | Expr::Range(_)
        | Expr::Reference(_)
        | Expr::Repeat(_)
        | Expr::Return(_)
        | Expr::Struct(_)
        | Expr::Try(_)
        | Expr::Tuple(_)
        | Expr::Unary(_)
        | Expr::Yield(_)
        | Expr::Verbatim(_) => true
    }
}
