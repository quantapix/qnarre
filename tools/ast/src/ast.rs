pub use crate::format::*;
pub use crate::util::parser::ExprPrecedence;
pub use GenericArgs::*;
pub use UnsafeSource::*;

use crate::ptr::P;
use crate::token::{self, CommentKind, Delimiter};
use crate::tokenstream::{DelimSpan, LazyAttrTokenStream, TokenStream};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_data_structures::sync::Lrc;
use rustc_macros::HashStable_Generic;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_span::source_map::{respan, Spanned};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};
use std::fmt;
use std::mem;
use thin_vec::{thin_vec, ThinVec};

#[derive(Clone, Encodable, Decodable, Copy, HashStable_Generic, Eq, PartialEq)]
pub struct Label {
    pub ident: Ident,
}

impl fmt::Debug for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "label({:?})", self.ident)
    }
}

#[derive(Clone, Encodable, Decodable, Copy, PartialEq, Eq)]
pub struct Lifetime {
    pub id: NodeId,
    pub ident: Ident,
}

impl fmt::Debug for Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lifetime({}: {})", self.id, self)
    }
}

impl fmt::Display for Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ident.name)
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Path {
    pub span: Span,
    pub segments: ThinVec<PathSegment>,
    pub tokens: Option<LazyAttrTokenStream>,
}

impl PartialEq<Symbol> for Path {
    #[inline]
    fn eq(&self, symbol: &Symbol) -> bool {
        self.segments.len() == 1 && { self.segments[0].ident.name == *symbol }
    }
}

impl<CTX: rustc_span::HashStableContext> HashStable<CTX> for Path {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        self.segments.len().hash_stable(hcx, hasher);
        for segment in &self.segments {
            segment.ident.hash_stable(hcx, hasher);
        }
    }
}

impl Path {
    pub fn from_ident(ident: Ident) -> Path {
        Path {
            segments: thin_vec![PathSegment::from_ident(ident)],
            span: ident.span,
            tokens: None,
        }
    }

    pub fn is_global(&self) -> bool {
        !self.segments.is_empty() && self.segments[0].ident.name == kw::PathRoot
    }

    pub fn is_potential_trivial_const_arg(&self) -> bool {
        self.segments.len() == 1 && self.segments[0].args.is_none()
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct PathSegment {
    pub ident: Ident,

    pub id: NodeId,

    pub args: Option<P<GenericArgs>>,
}

impl PathSegment {
    pub fn from_ident(ident: Ident) -> Self {
        PathSegment {
            ident,
            id: DUMMY_NODE_ID,
            args: None,
        }
    }

    pub fn path_root(span: Span) -> Self {
        PathSegment::from_ident(Ident::new(kw::PathRoot, span))
    }

    pub fn span(&self) -> Span {
        match &self.args {
            Some(args) => self.ident.span.to(args.span()),
            None => self.ident.span,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum GenericArgs {
    AngleBracketed(AngleBracketedArgs),
    Parenthesized(ParenthesizedArgs),
}

impl GenericArgs {
    pub fn is_angle_bracketed(&self) -> bool {
        matches!(self, AngleBracketed(..))
    }

    pub fn span(&self) -> Span {
        match self {
            AngleBracketed(data) => data.span,
            Parenthesized(data) => data.span,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum GenericArg {
    Lifetime(Lifetime),
    Type(P<Ty>),
    Const(AnonConst),
}

impl GenericArg {
    pub fn span(&self) -> Span {
        match self {
            GenericArg::Lifetime(lt) => lt.ident.span,
            GenericArg::Type(ty) => ty.span,
            GenericArg::Const(ct) => ct.value.span,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug, Default)]
pub struct AngleBracketedArgs {
    pub span: Span,
    pub args: ThinVec<AngleBracketedArg>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum AngleBracketedArg {
    Arg(GenericArg),
    Constraint(AssocConstraint),
}

impl AngleBracketedArg {
    pub fn span(&self) -> Span {
        match self {
            AngleBracketedArg::Arg(arg) => arg.span(),
            AngleBracketedArg::Constraint(constraint) => constraint.span,
        }
    }
}

impl Into<P<GenericArgs>> for AngleBracketedArgs {
    fn into(self) -> P<GenericArgs> {
        P(GenericArgs::AngleBracketed(self))
    }
}

impl Into<P<GenericArgs>> for ParenthesizedArgs {
    fn into(self) -> P<GenericArgs> {
        P(GenericArgs::Parenthesized(self))
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct ParenthesizedArgs {
    pub span: Span,

    pub inputs: ThinVec<P<Ty>>,

    pub inputs_span: Span,

    pub output: FnRetTy,
}

impl ParenthesizedArgs {
    pub fn as_angle_bracketed_args(&self) -> AngleBracketedArgs {
        let args = self
            .inputs
            .iter()
            .cloned()
            .map(|input| AngleBracketedArg::Arg(GenericArg::Type(input)))
            .collect();
        AngleBracketedArgs {
            span: self.inputs_span,
            args,
        }
    }
}

pub use crate::node_id::{NodeId, CRATE_NODE_ID, DUMMY_NODE_ID};

#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Debug)]
pub enum TraitBoundModifier {
    None,

    Negative,

    Maybe,

    MaybeConst,

    MaybeConstNegative,

    MaybeConstMaybe,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum GenericBound {
    Trait(PolyTraitRef, TraitBoundModifier),
    Outlives(Lifetime),
}

impl GenericBound {
    pub fn span(&self) -> Span {
        match self {
            GenericBound::Trait(t, ..) => t.span,
            GenericBound::Outlives(l) => l.ident.span,
        }
    }
}

pub type GenericBounds = Vec<GenericBound>;

#[derive(Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ParamKindOrd {
    Lifetime,
    TypeOrConst,
}

impl fmt::Display for ParamKindOrd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamKindOrd::Lifetime => "lifetime".fmt(f),
            ParamKindOrd::TypeOrConst => "type and const".fmt(f),
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum GenericParamKind {
    Lifetime,
    Type {
        default: Option<P<Ty>>,
    },
    Const {
        ty: P<Ty>,
        kw_span: Span,
        default: Option<AnonConst>,
    },
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct GenericParam {
    pub id: NodeId,
    pub ident: Ident,
    pub attrs: AttrVec,
    pub bounds: GenericBounds,
    pub is_placeholder: bool,
    pub kind: GenericParamKind,
    pub colon_span: Option<Span>,
}

impl GenericParam {
    pub fn span(&self) -> Span {
        match &self.kind {
            GenericParamKind::Lifetime | GenericParamKind::Type { default: None } => self.ident.span,
            GenericParamKind::Type { default: Some(ty) } => self.ident.span.to(ty.span),
            GenericParamKind::Const {
                kw_span,
                default: Some(default),
                ..
            } => kw_span.to(default.value.span),
            GenericParamKind::Const {
                kw_span,
                default: None,
                ty,
            } => kw_span.to(ty.span),
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Generics {
    pub params: ThinVec<GenericParam>,
    pub where_clause: WhereClause,
    pub span: Span,
}

impl Default for Generics {
    fn default() -> Generics {
        Generics {
            params: ThinVec::new(),
            where_clause: Default::default(),
            span: DUMMY_SP,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct WhereClause {
    pub has_where_token: bool,
    pub predicates: ThinVec<WherePredicate>,
    pub span: Span,
}

impl Default for WhereClause {
    fn default() -> WhereClause {
        WhereClause {
            has_where_token: false,
            predicates: ThinVec::new(),
            span: DUMMY_SP,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum WherePredicate {
    BoundPredicate(WhereBoundPredicate),
    RegionPredicate(WhereRegionPredicate),
    EqPredicate(WhereEqPredicate),
}

impl WherePredicate {
    pub fn span(&self) -> Span {
        match self {
            WherePredicate::BoundPredicate(p) => p.span,
            WherePredicate::RegionPredicate(p) => p.span,
            WherePredicate::EqPredicate(p) => p.span,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct WhereBoundPredicate {
    pub span: Span,
    pub bound_generic_params: ThinVec<GenericParam>,
    pub bounded_ty: P<Ty>,
    pub bounds: GenericBounds,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct WhereRegionPredicate {
    pub span: Span,
    pub lifetime: Lifetime,
    pub bounds: GenericBounds,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct WhereEqPredicate {
    pub span: Span,
    pub lhs_ty: P<Ty>,
    pub rhs_ty: P<Ty>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Crate {
    pub attrs: AttrVec,
    pub items: ThinVec<P<Item>>,
    pub spans: ModSpans,
    pub id: NodeId,
    pub is_placeholder: bool,
}

#[derive(Clone, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct MetaItem {
    pub path: Path,
    pub kind: MetaItemKind,
    pub span: Span,
}

#[derive(Clone, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum MetaItemKind {
    Word,

    List(ThinVec<NestedMetaItem>),

    NameValue(MetaItemLit),
}

#[derive(Clone, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum NestedMetaItem {
    MetaItem(MetaItem),

    Lit(MetaItemLit),
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Block {
    pub stmts: ThinVec<Stmt>,
    pub id: NodeId,
    pub rules: BlockCheckMode,
    pub span: Span,
    pub tokens: Option<LazyAttrTokenStream>,
    pub could_be_bare_literal: bool,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Pat {
    pub id: NodeId,
    pub kind: PatKind,
    pub span: Span,
    pub tokens: Option<LazyAttrTokenStream>,
}

impl Pat {
    pub fn to_ty(&self) -> Option<P<Ty>> {
        let kind = match &self.kind {
            PatKind::Wild => TyKind::Infer,
            PatKind::Ident(BindingAnnotation::NONE, ident, None) => TyKind::Path(None, Path::from_ident(*ident)),
            PatKind::Path(qself, path) => TyKind::Path(qself.clone(), path.clone()),
            PatKind::MacCall(mac) => TyKind::MacCall(mac.clone()),
            PatKind::Ref(pat, mutbl) => pat.to_ty().map(|ty| TyKind::Ref(None, MutTy { ty, mutbl: *mutbl }))?,
            PatKind::Slice(pats) if pats.len() == 1 => pats[0].to_ty().map(TyKind::Slice)?,
            PatKind::Tuple(pats) => {
                let mut tys = ThinVec::with_capacity(pats.len());
                for pat in pats {
                    tys.push(pat.to_ty()?);
                }
                TyKind::Tup(tys)
            },
            _ => return None,
        };

        Some(P(Ty {
            kind,
            id: self.id,
            span: self.span,
            tokens: None,
        }))
    }

    pub fn walk(&self, it: &mut impl FnMut(&Pat) -> bool) {
        if !it(self) {
            return;
        }

        match &self.kind {
            PatKind::Ident(_, _, Some(p)) => p.walk(it),

            PatKind::Struct(_, _, fields, _) => fields.iter().for_each(|field| field.pat.walk(it)),

            PatKind::TupleStruct(_, _, s) | PatKind::Tuple(s) | PatKind::Slice(s) | PatKind::Or(s) => {
                s.iter().for_each(|p| p.walk(it))
            },

            PatKind::Box(s) | PatKind::Ref(s, _) | PatKind::Paren(s) => s.walk(it),

            PatKind::Wild
            | PatKind::Rest
            | PatKind::Lit(_)
            | PatKind::Range(..)
            | PatKind::Ident(..)
            | PatKind::Path(..)
            | PatKind::MacCall(_) => {},
        }
    }

    pub fn is_rest(&self) -> bool {
        matches!(self.kind, PatKind::Rest)
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct PatField {
    pub ident: Ident,
    pub pat: P<Pat>,
    pub is_shorthand: bool,
    pub attrs: AttrVec,
    pub id: NodeId,
    pub span: Span,
    pub is_placeholder: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub enum ByRef {
    Yes,
    No,
}

impl From<bool> for ByRef {
    fn from(b: bool) -> ByRef {
        match b {
            false => ByRef::No,
            true => ByRef::Yes,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub struct BindingAnnotation(pub ByRef, pub Mutability);

impl BindingAnnotation {
    pub const NONE: Self = Self(ByRef::No, Mutability::Not);
    pub const REF: Self = Self(ByRef::Yes, Mutability::Not);
    pub const MUT: Self = Self(ByRef::No, Mutability::Mut);
    pub const REF_MUT: Self = Self(ByRef::Yes, Mutability::Mut);

    pub fn prefix_str(self) -> &'static str {
        match self {
            Self::NONE => "",
            Self::REF => "ref ",
            Self::MUT => "mut ",
            Self::REF_MUT => "ref mut ",
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum RangeEnd {
    Included(RangeSyntax),
    Excluded,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum RangeSyntax {
    DotDotDot,
    DotDotEq,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum PatKind {
    Wild,

    Ident(BindingAnnotation, Ident, Option<P<Pat>>),

    Struct(Option<P<QSelf>>, Path, ThinVec<PatField>, /* recovered */ bool),

    TupleStruct(Option<P<QSelf>>, Path, ThinVec<P<Pat>>),

    Or(ThinVec<P<Pat>>),

    Path(Option<P<QSelf>>, Path),

    Tuple(ThinVec<P<Pat>>),

    Box(P<Pat>),

    Ref(P<Pat>, Mutability),

    Lit(P<Expr>),

    Range(Option<P<Expr>>, Option<P<Expr>>, Spanned<RangeEnd>),

    Slice(ThinVec<P<Pat>>),

    Rest,

    Paren(P<Pat>),

    MacCall(P<MacCall>),
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Copy, HashStable_Generic, Encodable, Decodable)]
pub enum Mutability {
    Not,
    Mut,
}

impl Mutability {
    pub fn invert(self) -> Self {
        match self {
            Mutability::Mut => Mutability::Not,
            Mutability::Not => Mutability::Mut,
        }
    }

    pub fn prefix_str(self) -> &'static str {
        match self {
            Mutability::Mut => "mut ",
            Mutability::Not => "",
        }
    }

    pub fn ref_prefix_str(self) -> &'static str {
        match self {
            Mutability::Not => "&",
            Mutability::Mut => "&mut ",
        }
    }

    pub fn mutably_str(self) -> &'static str {
        match self {
            Mutability::Not => "",
            Mutability::Mut => "mutably ",
        }
    }

    pub fn is_mut(self) -> bool {
        matches!(self, Self::Mut)
    }

    pub fn is_not(self) -> bool {
        matches!(self, Self::Not)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum BorrowKind {
    Ref,
    Raw,
}

#[derive(Clone, PartialEq, Encodable, Decodable, Debug, Copy)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    BitXor,
    BitAnd,
    BitOr,
    Shl,
    Shr,
    Eq,
    Lt,
    Le,
    Ne,
    Ge,
    Gt,
}

impl BinOpKind {
    pub fn to_string(&self) -> &'static str {
        use BinOpKind::*;
        match *self {
            Add => "+",
            Sub => "-",
            Mul => "*",
            Div => "/",
            Rem => "%",
            And => "&&",
            Or => "||",
            BitXor => "^",
            BitAnd => "&",
            BitOr => "|",
            Shl => "<<",
            Shr => ">>",
            Eq => "==",
            Lt => "<",
            Le => "<=",
            Ne => "!=",
            Ge => ">=",
            Gt => ">",
        }
    }
    pub fn lazy(&self) -> bool {
        matches!(self, BinOpKind::And | BinOpKind::Or)
    }

    pub fn is_comparison(&self) -> bool {
        use BinOpKind::*;
        match *self {
            Eq | Lt | Le | Ne | Gt | Ge => true,
            And | Or | Add | Sub | Mul | Div | Rem | BitXor | BitAnd | BitOr | Shl | Shr => false,
        }
    }
}

pub type BinOp = Spanned<BinOpKind>;

#[derive(Clone, Encodable, Decodable, Debug, Copy)]
pub enum UnOp {
    Deref,
    Not,
    Neg,
}

impl UnOp {
    pub fn to_string(op: UnOp) -> &'static str {
        match op {
            UnOp::Deref => "*",
            UnOp::Not => "!",
            UnOp::Neg => "-",
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Stmt {
    pub id: NodeId,
    pub kind: StmtKind,
    pub span: Span,
}

impl Stmt {
    pub fn has_trailing_semicolon(&self) -> bool {
        match &self.kind {
            StmtKind::Semi(_) => true,
            StmtKind::MacCall(mac) => matches!(mac.style, MacStmtStyle::Semicolon),
            _ => false,
        }
    }

    pub fn add_trailing_semicolon(mut self) -> Self {
        self.kind = match self.kind {
            StmtKind::Expr(expr) => StmtKind::Semi(expr),
            StmtKind::MacCall(mac) => StmtKind::MacCall(mac.map(
                |MacCallStmt {
                     mac,
                     style: _,
                     attrs,
                     tokens,
                 }| {
                    MacCallStmt {
                        mac,
                        style: MacStmtStyle::Semicolon,
                        attrs,
                        tokens,
                    }
                },
            )),
            kind => kind,
        };

        self
    }

    pub fn is_item(&self) -> bool {
        matches!(self.kind, StmtKind::Item(_))
    }

    pub fn is_expr(&self) -> bool {
        matches!(self.kind, StmtKind::Expr(_))
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum StmtKind {
    Local(P<Local>),
    Item(P<Item>),
    Expr(P<Expr>),
    Semi(P<Expr>),
    Empty,
    MacCall(P<MacCallStmt>),
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct MacCallStmt {
    pub mac: P<MacCall>,
    pub style: MacStmtStyle,
    pub attrs: AttrVec,
    pub tokens: Option<LazyAttrTokenStream>,
}

#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug)]
pub enum MacStmtStyle {
    Semicolon,
    Braces,
    NoBraces,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Local {
    pub id: NodeId,
    pub pat: P<Pat>,
    pub ty: Option<P<Ty>>,
    pub kind: LocalKind,
    pub span: Span,
    pub attrs: AttrVec,
    pub tokens: Option<LazyAttrTokenStream>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum LocalKind {
    Decl,
    Init(P<Expr>),
    InitElse(P<Expr>, P<Block>),
}

impl LocalKind {
    pub fn init(&self) -> Option<&Expr> {
        match self {
            Self::Decl => None,
            Self::Init(i) | Self::InitElse(i, _) => Some(i),
        }
    }

    pub fn init_else_opt(&self) -> Option<(&Expr, Option<&Block>)> {
        match self {
            Self::Decl => None,
            Self::Init(init) => Some((init, None)),
            Self::InitElse(init, els) => Some((init, Some(els))),
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Arm {
    pub attrs: AttrVec,
    pub pat: P<Pat>,
    pub guard: Option<P<Expr>>,
    pub body: P<Expr>,
    pub span: Span,
    pub id: NodeId,
    pub is_placeholder: bool,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct ExprField {
    pub attrs: AttrVec,
    pub id: NodeId,
    pub span: Span,
    pub ident: Ident,
    pub expr: P<Expr>,
    pub is_shorthand: bool,
    pub is_placeholder: bool,
}

#[derive(Clone, PartialEq, Encodable, Decodable, Debug, Copy)]
pub enum BlockCheckMode {
    Default,
    Unsafe(UnsafeSource),
}

#[derive(Clone, PartialEq, Encodable, Decodable, Debug, Copy)]
pub enum UnsafeSource {
    CompilerGenerated,
    UserProvided,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct AnonConst {
    pub id: NodeId,
    pub value: P<Expr>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Expr {
    pub id: NodeId,
    pub kind: ExprKind,
    pub span: Span,
    pub attrs: AttrVec,
    pub tokens: Option<LazyAttrTokenStream>,
}

impl Expr {
    pub fn is_potential_trivial_const_arg(&self) -> bool {
        let this = if let ExprKind::Block(block, None) = &self.kind
            && block.stmts.len() == 1
            && let StmtKind::Expr(expr) = &block.stmts[0].kind
        {
            expr
        } else {
            self
        };

        if let ExprKind::Path(None, path) = &this.kind
            && path.is_potential_trivial_const_arg()
        {
            true
        } else {
            false
        }
    }

    pub fn to_bound(&self) -> Option<GenericBound> {
        match &self.kind {
            ExprKind::Path(None, path) => Some(GenericBound::Trait(
                PolyTraitRef::new(ThinVec::new(), path.clone(), self.span),
                TraitBoundModifier::None,
            )),
            _ => None,
        }
    }

    pub fn peel_parens(&self) -> &Expr {
        let mut expr = self;
        while let ExprKind::Paren(inner) = &expr.kind {
            expr = inner;
        }
        expr
    }

    pub fn peel_parens_and_refs(&self) -> &Expr {
        let mut expr = self;
        while let ExprKind::Paren(inner) | ExprKind::AddrOf(BorrowKind::Ref, _, inner) = &expr.kind {
            expr = inner;
        }
        expr
    }

    pub fn to_ty(&self) -> Option<P<Ty>> {
        let kind = match &self.kind {
            ExprKind::Path(qself, path) => TyKind::Path(qself.clone(), path.clone()),
            ExprKind::MacCall(mac) => TyKind::MacCall(mac.clone()),

            ExprKind::Paren(expr) => expr.to_ty().map(TyKind::Paren)?,

            ExprKind::AddrOf(BorrowKind::Ref, mutbl, expr) => {
                expr.to_ty().map(|ty| TyKind::Ref(None, MutTy { ty, mutbl: *mutbl }))?
            },

            ExprKind::Repeat(expr, expr_len) => expr.to_ty().map(|ty| TyKind::Array(ty, expr_len.clone()))?,

            ExprKind::Array(exprs) if exprs.len() == 1 => exprs[0].to_ty().map(TyKind::Slice)?,

            ExprKind::Tup(exprs) => {
                let tys = exprs.iter().map(|expr| expr.to_ty()).collect::<Option<ThinVec<_>>>()?;
                TyKind::Tup(tys)
            },

            ExprKind::Binary(binop, lhs, rhs) if binop.node == BinOpKind::Add => {
                if let (Some(lhs), Some(rhs)) = (lhs.to_bound(), rhs.to_bound()) {
                    TyKind::TraitObject(vec![lhs, rhs], TraitObjectSyntax::None)
                } else {
                    return None;
                }
            },

            ExprKind::Underscore => TyKind::Infer,

            _ => return None,
        };

        Some(P(Ty {
            kind,
            id: self.id,
            span: self.span,
            tokens: None,
        }))
    }

    pub fn precedence(&self) -> ExprPrecedence {
        match self.kind {
            ExprKind::Array(_) => ExprPrecedence::Array,
            ExprKind::ConstBlock(_) => ExprPrecedence::ConstBlock,
            ExprKind::Call(..) => ExprPrecedence::Call,
            ExprKind::MethodCall(..) => ExprPrecedence::MethodCall,
            ExprKind::Tup(_) => ExprPrecedence::Tup,
            ExprKind::Binary(op, ..) => ExprPrecedence::Binary(op.node),
            ExprKind::Unary(..) => ExprPrecedence::Unary,
            ExprKind::Lit(_) | ExprKind::IncludedBytes(..) => ExprPrecedence::Lit,
            ExprKind::Type(..) | ExprKind::Cast(..) => ExprPrecedence::Cast,
            ExprKind::Let(..) => ExprPrecedence::Let,
            ExprKind::If(..) => ExprPrecedence::If,
            ExprKind::While(..) => ExprPrecedence::While,
            ExprKind::ForLoop(..) => ExprPrecedence::ForLoop,
            ExprKind::Loop(..) => ExprPrecedence::Loop,
            ExprKind::Match(..) => ExprPrecedence::Match,
            ExprKind::Closure(..) => ExprPrecedence::Closure,
            ExprKind::Block(..) => ExprPrecedence::Block,
            ExprKind::TryBlock(..) => ExprPrecedence::TryBlock,
            ExprKind::Async(..) => ExprPrecedence::Async,
            ExprKind::Await(..) => ExprPrecedence::Await,
            ExprKind::Assign(..) => ExprPrecedence::Assign,
            ExprKind::AssignOp(..) => ExprPrecedence::AssignOp,
            ExprKind::Field(..) => ExprPrecedence::Field,
            ExprKind::Index(..) => ExprPrecedence::Index,
            ExprKind::Range(..) => ExprPrecedence::Range,
            ExprKind::Underscore => ExprPrecedence::Path,
            ExprKind::Path(..) => ExprPrecedence::Path,
            ExprKind::AddrOf(..) => ExprPrecedence::AddrOf,
            ExprKind::Break(..) => ExprPrecedence::Break,
            ExprKind::Continue(..) => ExprPrecedence::Continue,
            ExprKind::Ret(..) => ExprPrecedence::Ret,
            ExprKind::InlineAsm(..) => ExprPrecedence::InlineAsm,
            ExprKind::OffsetOf(..) => ExprPrecedence::OffsetOf,
            ExprKind::MacCall(..) => ExprPrecedence::Mac,
            ExprKind::Struct(..) => ExprPrecedence::Struct,
            ExprKind::Repeat(..) => ExprPrecedence::Repeat,
            ExprKind::Paren(..) => ExprPrecedence::Paren,
            ExprKind::Try(..) => ExprPrecedence::Try,
            ExprKind::Yield(..) => ExprPrecedence::Yield,
            ExprKind::Yeet(..) => ExprPrecedence::Yeet,
            ExprKind::FormatArgs(..) => ExprPrecedence::FormatArgs,
            ExprKind::Become(..) => ExprPrecedence::Become,
            ExprKind::Err => ExprPrecedence::Err,
        }
    }

    pub fn take(&mut self) -> Self {
        mem::replace(
            self,
            Expr {
                id: DUMMY_NODE_ID,
                kind: ExprKind::Err,
                span: DUMMY_SP,
                attrs: AttrVec::new(),
                tokens: None,
            },
        )
    }

    pub fn is_approximately_pattern(&self) -> bool {
        matches!(
            &self.peel_parens().kind,
            ExprKind::Array(_)
                | ExprKind::Call(_, _)
                | ExprKind::Tup(_)
                | ExprKind::Lit(_)
                | ExprKind::Range(_, _, _)
                | ExprKind::Underscore
                | ExprKind::Path(_, _)
                | ExprKind::Struct(_)
        )
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Closure {
    pub binder: ClosureBinder,
    pub capture_clause: CaptureBy,
    pub constness: Const,
    pub asyncness: Async,
    pub movability: Movability,
    pub fn_decl: P<FnDecl>,
    pub body: P<Expr>,
    pub fn_decl_span: Span,
    pub fn_arg_span: Span,
}

#[derive(Copy, Clone, PartialEq, Encodable, Decodable, Debug)]
pub enum RangeLimits {
    HalfOpen,
    Closed,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct MethodCall {
    pub seg: PathSegment,
    pub receiver: P<Expr>,
    pub args: ThinVec<P<Expr>>,
    pub span: Span,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum StructRest {
    Base(P<Expr>),
    Rest(Span),
    None,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct StructExpr {
    pub qself: Option<P<QSelf>>,
    pub path: Path,
    pub fields: ThinVec<ExprField>,
    pub rest: StructRest,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum ExprKind {
    Array(ThinVec<P<Expr>>),
    ConstBlock(AnonConst),
    Call(P<Expr>, ThinVec<P<Expr>>),
    MethodCall(Box<MethodCall>),
    Tup(ThinVec<P<Expr>>),
    Binary(BinOp, P<Expr>, P<Expr>),
    Unary(UnOp, P<Expr>),
    Lit(token::Lit),
    Cast(P<Expr>, P<Ty>),
    Type(P<Expr>, P<Ty>),
    Let(P<Pat>, P<Expr>, Span),
    If(P<Expr>, P<Block>, Option<P<Expr>>),
    While(P<Expr>, P<Block>, Option<Label>),
    ForLoop(P<Pat>, P<Expr>, P<Block>, Option<Label>),
    Loop(P<Block>, Option<Label>, Span),
    Match(P<Expr>, ThinVec<Arm>),
    Closure(Box<Closure>),
    Block(P<Block>, Option<Label>),
    Async(CaptureBy, P<Block>),
    Await(P<Expr>, Span),

    TryBlock(P<Block>),

    Assign(P<Expr>, P<Expr>, Span),
    AssignOp(BinOp, P<Expr>, P<Expr>),
    Field(P<Expr>, Ident),
    Index(P<Expr>, P<Expr>),
    Range(Option<P<Expr>>, Option<P<Expr>>, RangeLimits),
    Underscore,

    Path(Option<P<QSelf>>, Path),

    AddrOf(BorrowKind, Mutability, P<Expr>),
    Break(Option<Label>, Option<P<Expr>>),
    Continue(Option<Label>),
    Ret(Option<P<Expr>>),

    InlineAsm(P<InlineAsm>),

    OffsetOf(P<Ty>, P<[Ident]>),

    MacCall(P<MacCall>),

    Struct(P<StructExpr>),

    Repeat(P<Expr>, AnonConst),

    Paren(P<Expr>),

    Try(P<Expr>),

    Yield(Option<P<Expr>>),

    Yeet(Option<P<Expr>>),

    Become(P<Expr>),

    IncludedBytes(Lrc<[u8]>),

    FormatArgs(P<FormatArgs>),

    Err,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct QSelf {
    pub ty: P<Ty>,

    pub path_span: Span,
    pub position: usize,
}

#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum CaptureBy {
    Value,
    Ref,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable, Debug, Copy, HashStable_Generic)]
pub enum Movability {
    Static,
    Movable,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum ClosureBinder {
    NotPresent,
    For {
        span: Span,

        generic_params: ThinVec<GenericParam>,
    },
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct MacCall {
    pub path: Path,
    pub args: P<DelimArgs>,
}

impl MacCall {
    pub fn span(&self) -> Span {
        self.path.span.to(self.args.dspan.entire())
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum AttrArgs {
    Empty,
    Delimited(DelimArgs),
    Eq(Span, AttrArgsEq),
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum AttrArgsEq {
    Ast(P<Expr>),
    Hir(MetaItemLit),
}

impl AttrArgs {
    pub fn span(&self) -> Option<Span> {
        match self {
            AttrArgs::Empty => None,
            AttrArgs::Delimited(args) => Some(args.dspan.entire()),
            AttrArgs::Eq(eq_span, AttrArgsEq::Ast(expr)) => Some(eq_span.to(expr.span)),
            AttrArgs::Eq(_, AttrArgsEq::Hir(lit)) => {
                unreachable!("in literal form when getting span: {:?}", lit);
            },
        }
    }

    pub fn inner_tokens(&self) -> TokenStream {
        match self {
            AttrArgs::Empty => TokenStream::default(),
            AttrArgs::Delimited(args) => args.tokens.clone(),
            AttrArgs::Eq(_, AttrArgsEq::Ast(expr)) => TokenStream::from_ast(expr),
            AttrArgs::Eq(_, AttrArgsEq::Hir(lit)) => {
                unreachable!("in literal form when getting inner tokens: {:?}", lit)
            },
        }
    }
}

impl<CTX> HashStable<CTX> for AttrArgs
where
    CTX: crate::HashStableContext,
{
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        mem::discriminant(self).hash_stable(ctx, hasher);
        match self {
            AttrArgs::Empty => {},
            AttrArgs::Delimited(args) => args.hash_stable(ctx, hasher),
            AttrArgs::Eq(_eq_span, AttrArgsEq::Ast(expr)) => {
                unreachable!("hash_stable {:?}", expr);
            },
            AttrArgs::Eq(eq_span, AttrArgsEq::Hir(lit)) => {
                eq_span.hash_stable(ctx, hasher);
                lit.hash_stable(ctx, hasher);
            },
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct DelimArgs {
    pub dspan: DelimSpan,
    pub delim: MacDelimiter,
    pub tokens: TokenStream,
}

impl DelimArgs {
    pub fn need_semicolon(&self) -> bool {
        !matches!(
            self,
            DelimArgs {
                delim: MacDelimiter::Brace,
                ..
            }
        )
    }
}

impl<CTX> HashStable<CTX> for DelimArgs
where
    CTX: crate::HashStableContext,
{
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        let DelimArgs { dspan, delim, tokens } = self;
        dspan.hash_stable(ctx, hasher);
        delim.hash_stable(ctx, hasher);
        tokens.hash_stable(ctx, hasher);
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum MacDelimiter {
    Parenthesis,
    Bracket,
    Brace,
}

impl MacDelimiter {
    pub fn to_token(self) -> Delimiter {
        match self {
            MacDelimiter::Parenthesis => Delimiter::Parenthesis,
            MacDelimiter::Bracket => Delimiter::Bracket,
            MacDelimiter::Brace => Delimiter::Brace,
        }
    }

    pub fn from_token(delim: Delimiter) -> Option<MacDelimiter> {
        match delim {
            Delimiter::Parenthesis => Some(MacDelimiter::Parenthesis),
            Delimiter::Bracket => Some(MacDelimiter::Bracket),
            Delimiter::Brace => Some(MacDelimiter::Brace),
            Delimiter::Invisible => None,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct MacroDef {
    pub body: P<DelimArgs>,
    pub macro_rules: bool,
}

#[derive(Clone, Encodable, Decodable, Debug, Copy, Hash, Eq, PartialEq, HashStable_Generic)]
pub enum StrStyle {
    Cooked,
    Raw(u8),
}

#[derive(Clone, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct MetaItemLit {
    pub symbol: Symbol,
    pub suffix: Option<Symbol>,
    pub kind: LitKind,
    pub span: Span,
}

#[derive(Clone, Copy, Encodable, Decodable, Debug)]
pub struct StrLit {
    pub symbol: Symbol,
    pub suffix: Option<Symbol>,
    pub symbol_unescaped: Symbol,
    pub style: StrStyle,
    pub span: Span,
}

impl StrLit {
    pub fn as_token_lit(&self) -> token::Lit {
        let token_kind = match self.style {
            StrStyle::Cooked => token::Str,
            StrStyle::Raw(n) => token::StrRaw(n),
        };
        token::Lit::new(token_kind, self.symbol, self.suffix)
    }
}

#[derive(Clone, Copy, Encodable, Decodable, Debug, Hash, Eq, PartialEq, HashStable_Generic)]
pub enum LitIntType {
    Signed(IntTy),
    Unsigned(UintTy),
    Unsuffixed,
}

#[derive(Clone, Copy, Encodable, Decodable, Debug, Hash, Eq, PartialEq, HashStable_Generic)]
pub enum LitFloatType {
    Suffixed(FloatTy),
    Unsuffixed,
}

#[derive(Clone, Encodable, Decodable, Debug, Hash, Eq, PartialEq, HashStable_Generic)]
pub enum LitKind {
    Str(Symbol, StrStyle),
    ByteStr(Lrc<[u8]>, StrStyle),
    CStr(Lrc<[u8]>, StrStyle),
    Byte(u8),
    Char(char),
    Int(u128, LitIntType),
    Float(Symbol, LitFloatType),
    Bool(bool),
    Err,
}

impl LitKind {
    pub fn str(&self) -> Option<Symbol> {
        match *self {
            LitKind::Str(s, _) => Some(s),
            _ => None,
        }
    }

    pub fn is_str(&self) -> bool {
        matches!(self, LitKind::Str(..))
    }

    pub fn is_bytestr(&self) -> bool {
        matches!(self, LitKind::ByteStr(..))
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self, LitKind::Int(..) | LitKind::Float(..))
    }

    pub fn is_unsuffixed(&self) -> bool {
        !self.is_suffixed()
    }

    pub fn is_suffixed(&self) -> bool {
        match *self {
            LitKind::Int(_, LitIntType::Signed(..) | LitIntType::Unsigned(..))
            | LitKind::Float(_, LitFloatType::Suffixed(..)) => true,
            LitKind::Str(..)
            | LitKind::ByteStr(..)
            | LitKind::CStr(..)
            | LitKind::Byte(..)
            | LitKind::Char(..)
            | LitKind::Int(_, LitIntType::Unsuffixed)
            | LitKind::Float(_, LitFloatType::Unsuffixed)
            | LitKind::Bool(..)
            | LitKind::Err => false,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct MutTy {
    pub ty: P<Ty>,
    pub mutbl: Mutability,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct FnSig {
    pub header: FnHeader,
    pub decl: P<FnDecl>,
    pub span: Span,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum FloatTy {
    F32,
    F64,
}

impl FloatTy {
    pub fn name_str(self) -> &'static str {
        match self {
            FloatTy::F32 => "f32",
            FloatTy::F64 => "f64",
        }
    }

    pub fn name(self) -> Symbol {
        match self {
            FloatTy::F32 => sym::f32,
            FloatTy::F64 => sym::f64,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum IntTy {
    Isize,
    I8,
    I16,
    I32,
    I64,
    I128,
}

impl IntTy {
    pub fn name_str(&self) -> &'static str {
        match *self {
            IntTy::Isize => "isize",
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
            IntTy::I128 => "i128",
        }
    }

    pub fn name(&self) -> Symbol {
        match *self {
            IntTy::Isize => sym::isize,
            IntTy::I8 => sym::i8,
            IntTy::I16 => sym::i16,
            IntTy::I32 => sym::i32,
            IntTy::I64 => sym::i64,
            IntTy::I128 => sym::i128,
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum UintTy {
    Usize,
    U8,
    U16,
    U32,
    U64,
    U128,
}

impl UintTy {
    pub fn name_str(&self) -> &'static str {
        match *self {
            UintTy::Usize => "usize",
            UintTy::U8 => "u8",
            UintTy::U16 => "u16",
            UintTy::U32 => "u32",
            UintTy::U64 => "u64",
            UintTy::U128 => "u128",
        }
    }

    pub fn name(&self) -> Symbol {
        match *self {
            UintTy::Usize => sym::usize,
            UintTy::U8 => sym::u8,
            UintTy::U16 => sym::u16,
            UintTy::U32 => sym::u32,
            UintTy::U64 => sym::u64,
            UintTy::U128 => sym::u128,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct AssocConstraint {
    pub id: NodeId,
    pub ident: Ident,
    pub gen_args: Option<GenericArgs>,
    pub kind: AssocConstraintKind,
    pub span: Span,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum Term {
    Ty(P<Ty>),
    Const(AnonConst),
}

impl From<P<Ty>> for Term {
    fn from(v: P<Ty>) -> Self {
        Term::Ty(v)
    }
}

impl From<AnonConst> for Term {
    fn from(v: AnonConst) -> Self {
        Term::Const(v)
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum AssocConstraintKind {
    Equality { term: Term },
    Bound { bounds: GenericBounds },
}

#[derive(Encodable, Decodable, Debug)]
pub struct Ty {
    pub id: NodeId,
    pub kind: TyKind,
    pub span: Span,
    pub tokens: Option<LazyAttrTokenStream>,
}

impl Clone for Ty {
    fn clone(&self) -> Self {
        ensure_sufficient_stack(|| Self {
            id: self.id,
            kind: self.kind.clone(),
            span: self.span,
            tokens: self.tokens.clone(),
        })
    }
}

impl Ty {
    pub fn peel_refs(&self) -> &Self {
        let mut final_ty = self;
        while let TyKind::Ref(_, MutTy { ty, .. }) | TyKind::Ptr(MutTy { ty, .. }) = &final_ty.kind {
            final_ty = ty;
        }
        final_ty
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct BareFnTy {
    pub unsafety: Unsafe,
    pub ext: Extern,
    pub generic_params: ThinVec<GenericParam>,
    pub decl: P<FnDecl>,
    pub decl_span: Span,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum TyKind {
    Slice(P<Ty>),
    Array(P<Ty>, AnonConst),
    Ptr(MutTy),
    Ref(Option<Lifetime>, MutTy),
    BareFn(P<BareFnTy>),
    Never,
    Tup(ThinVec<P<Ty>>),
    Path(Option<P<QSelf>>, Path),
    TraitObject(GenericBounds, TraitObjectSyntax),
    ImplTrait(NodeId, GenericBounds),
    Paren(P<Ty>),
    Typeof(AnonConst),
    Infer,
    ImplicitSelf,
    MacCall(P<MacCall>),
    Err,
    CVarArgs,
}

impl TyKind {
    pub fn is_implicit_self(&self) -> bool {
        matches!(self, TyKind::ImplicitSelf)
    }

    pub fn is_unit(&self) -> bool {
        matches!(self, TyKind::Tup(tys) if tys.is_empty())
    }

    pub fn is_simple_path(&self) -> Option<Symbol> {
        if let TyKind::Path(None, Path { segments, .. }) = &self
            && let [segment] = &segments[..]
            && segment.args.is_none()
        {
            Some(segment.ident.name)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum TraitObjectSyntax {
    Dyn,
    DynStar,
    None,
}

#[derive(Clone, Copy, Encodable, Decodable, Debug)]
pub enum InlineAsmRegOrRegClass {
    Reg(Symbol),
    RegClass(Symbol),
}

bitflags::bitflags! {
    #[derive(Encodable, Decodable, HashStable_Generic)]
    pub struct InlineAsmOptions: u16 {
        const PURE            = 1 << 0;
        const NOMEM           = 1 << 1;
        const READONLY        = 1 << 2;
        const PRESERVES_FLAGS = 1 << 3;
        const NORETURN        = 1 << 4;
        const NOSTACK         = 1 << 5;
        const ATT_SYNTAX      = 1 << 6;
        const RAW             = 1 << 7;
        const MAY_UNWIND      = 1 << 8;
    }
}

#[derive(Clone, PartialEq, Encodable, Decodable, Debug, Hash, HashStable_Generic)]
pub enum InlineAsmTemplatePiece {
    String(String),
    Placeholder {
        operand_idx: usize,
        modifier: Option<char>,
        span: Span,
    },
}

impl fmt::Display for InlineAsmTemplatePiece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(s) => {
                for c in s.chars() {
                    match c {
                        '{' => f.write_str("{{")?,
                        '}' => f.write_str("}}")?,
                        _ => c.fmt(f)?,
                    }
                }
                Ok(())
            },
            Self::Placeholder {
                operand_idx,
                modifier: Some(modifier),
                ..
            } => {
                write!(f, "{{{operand_idx}:{modifier}}}")
            },
            Self::Placeholder {
                operand_idx,
                modifier: None,
                ..
            } => {
                write!(f, "{{{operand_idx}}}")
            },
        }
    }
}

impl InlineAsmTemplatePiece {
    pub fn to_string(s: &[Self]) -> String {
        use fmt::Write;
        let mut out = String::new();
        for p in s.iter() {
            let _ = write!(out, "{p}");
        }
        out
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct InlineAsmSym {
    pub id: NodeId,
    pub qself: Option<P<QSelf>>,
    pub path: Path,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum InlineAsmOperand {
    In {
        reg: InlineAsmRegOrRegClass,
        expr: P<Expr>,
    },
    Out {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: Option<P<Expr>>,
    },
    InOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: P<Expr>,
    },
    SplitInOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        in_expr: P<Expr>,
        out_expr: Option<P<Expr>>,
    },
    Const {
        anon_const: AnonConst,
    },
    Sym {
        sym: InlineAsmSym,
    },
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct InlineAsm {
    pub template: Vec<InlineAsmTemplatePiece>,
    pub template_strs: Box<[(Symbol, Option<Symbol>, Span)]>,
    pub operands: Vec<(InlineAsmOperand, Span)>,
    pub clobber_abis: Vec<(Symbol, Span)>,
    pub options: InlineAsmOptions,
    pub line_spans: Vec<Span>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Param {
    pub attrs: AttrVec,
    pub ty: P<Ty>,
    pub pat: P<Pat>,
    pub id: NodeId,
    pub span: Span,
    pub is_placeholder: bool,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum SelfKind {
    Value(Mutability),
    Region(Option<Lifetime>, Mutability),
    Explicit(P<Ty>, Mutability),
}

pub type ExplicitSelf = Spanned<SelfKind>;

impl Param {
    pub fn to_self(&self) -> Option<ExplicitSelf> {
        if let PatKind::Ident(BindingAnnotation(ByRef::No, mutbl), ident, _) = self.pat.kind {
            if ident.name == kw::SelfLower {
                return match self.ty.kind {
                    TyKind::ImplicitSelf => Some(respan(self.pat.span, SelfKind::Value(mutbl))),
                    TyKind::Ref(lt, MutTy { ref ty, mutbl }) if ty.kind.is_implicit_self() => {
                        Some(respan(self.pat.span, SelfKind::Region(lt, mutbl)))
                    },
                    _ => Some(respan(
                        self.pat.span.to(self.ty.span),
                        SelfKind::Explicit(self.ty.clone(), mutbl),
                    )),
                };
            }
        }
        None
    }

    pub fn is_self(&self) -> bool {
        if let PatKind::Ident(_, ident, _) = self.pat.kind {
            ident.name == kw::SelfLower
        } else {
            false
        }
    }

    pub fn from_self(attrs: AttrVec, eself: ExplicitSelf, eself_ident: Ident) -> Param {
        let span = eself.span.to(eself_ident.span);
        let infer_ty = P(Ty {
            id: DUMMY_NODE_ID,
            kind: TyKind::ImplicitSelf,
            span,
            tokens: None,
        });
        let (mutbl, ty) = match eself.node {
            SelfKind::Explicit(ty, mutbl) => (mutbl, ty),
            SelfKind::Value(mutbl) => (mutbl, infer_ty),
            SelfKind::Region(lt, mutbl) => (
                Mutability::Not,
                P(Ty {
                    id: DUMMY_NODE_ID,
                    kind: TyKind::Ref(lt, MutTy { ty: infer_ty, mutbl }),
                    span,
                    tokens: None,
                }),
            ),
        };
        Param {
            attrs,
            pat: P(Pat {
                id: DUMMY_NODE_ID,
                kind: PatKind::Ident(BindingAnnotation(ByRef::No, mutbl), eself_ident, None),
                span,
                tokens: None,
            }),
            span,
            ty,
            id: DUMMY_NODE_ID,
            is_placeholder: false,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct FnDecl {
    pub inputs: ThinVec<Param>,
    pub output: FnRetTy,
}

impl FnDecl {
    pub fn has_self(&self) -> bool {
        self.inputs.get(0).is_some_and(Param::is_self)
    }
    pub fn c_variadic(&self) -> bool {
        self.inputs
            .last()
            .is_some_and(|arg| matches!(arg.ty.kind, TyKind::CVarArgs))
    }
}

#[derive(Copy, Clone, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum IsAuto {
    Yes,
    No,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum Unsafe {
    Yes(Span),
    No,
}

#[derive(Copy, Clone, Encodable, Decodable, Debug)]
pub enum Async {
    Yes {
        span: Span,
        closure_id: NodeId,
        return_impl_trait_id: NodeId,
    },
    No,
}

impl Async {
    pub fn is_async(self) -> bool {
        matches!(self, Async::Yes { .. })
    }

    pub fn opt_return_id(self) -> Option<(NodeId, Span)> {
        match self {
            Async::Yes {
                return_impl_trait_id,
                span,
                ..
            } => Some((return_impl_trait_id, span)),
            Async::No => None,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum Const {
    Yes(Span),
    No,
}

#[derive(Copy, Clone, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum Defaultness {
    Default(Span),
    Final,
}

#[derive(Copy, Clone, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub enum ImplPolarity {
    Positive,
    Negative(Span),
}

impl fmt::Debug for ImplPolarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ImplPolarity::Positive => "positive".fmt(f),
            ImplPolarity::Negative(_) => "negative".fmt(f),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub enum BoundPolarity {
    Positive,
    Negative(Span),
    Maybe(Span),
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum FnRetTy {
    Default(Span),
    Ty(P<Ty>),
}

impl FnRetTy {
    pub fn span(&self) -> Span {
        match self {
            &FnRetTy::Default(span) => span,
            FnRetTy::Ty(ty) => ty.span,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Encodable, Decodable, Debug)]
pub enum Inline {
    Yes,
    No,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum ModKind {
    Loaded(ThinVec<P<Item>>, Inline, ModSpans),
    Unloaded,
}

#[derive(Copy, Clone, Encodable, Decodable, Debug, Default)]
pub struct ModSpans {
    pub inner_span: Span,
    pub inject_use_span: Span,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct ForeignMod {
    pub unsafety: Unsafe,
    pub abi: Option<StrLit>,
    pub items: ThinVec<P<ForeignItem>>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct EnumDef {
    pub variants: ThinVec<Variant>,
}
#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Variant {
    pub attrs: AttrVec,
    pub id: NodeId,
    pub span: Span,
    pub vis: Visibility,
    pub ident: Ident,

    pub data: VariantData,
    pub disr_expr: Option<AnonConst>,
    pub is_placeholder: bool,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum UseTreeKind {
    Simple(Option<Ident>),
    Nested(ThinVec<(UseTree, NodeId)>),
    Glob,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct UseTree {
    pub prefix: Path,
    pub kind: UseTreeKind,
    pub span: Span,
}

impl UseTree {
    pub fn ident(&self) -> Ident {
        match self.kind {
            UseTreeKind::Simple(Some(rename)) => rename,
            UseTreeKind::Simple(None) => {
                self.prefix
                    .segments
                    .last()
                    .expect("empty prefix in a simple import")
                    .ident
            },
            _ => panic!("`UseTree::ident` can only be used on a simple import"),
        }
    }
}

#[derive(Clone, PartialEq, Encodable, Decodable, Debug, Copy, HashStable_Generic)]
pub enum AttrStyle {
    Outer,
    Inner,
}

rustc_index::newtype_index! {
    #[custom_encodable]
    #[debug_format = "AttrId({})"]
    pub struct AttrId {}
}

impl<S: Encoder> Encodable<S> for AttrId {
    fn encode(&self, _s: &mut S) {}
}

impl<D: Decoder> Decodable<D> for AttrId {
    default fn decode(_: &mut D) -> AttrId {
        panic!("cannot decode `AttrId` with `{}`", std::any::type_name::<D>());
    }
}

pub type AttrVec = ThinVec<Attribute>;

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Attribute {
    pub kind: AttrKind,
    pub id: AttrId,
    pub style: AttrStyle,
    pub span: Span,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum AttrKind {
    Normal(P<NormalAttr>),

    DocComment(CommentKind, Symbol),
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct NormalAttr {
    pub item: AttrItem,
    pub tokens: Option<LazyAttrTokenStream>,
}

#[derive(Clone, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct AttrItem {
    pub path: Path,
    pub args: AttrArgs,
    pub tokens: Option<LazyAttrTokenStream>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct TraitRef {
    pub path: Path,
    pub ref_id: NodeId,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct PolyTraitRef {
    pub bound_generic_params: ThinVec<GenericParam>,

    pub trait_ref: TraitRef,

    pub span: Span,
}

impl PolyTraitRef {
    pub fn new(generic_params: ThinVec<GenericParam>, path: Path, span: Span) -> Self {
        PolyTraitRef {
            bound_generic_params: generic_params,
            trait_ref: TraitRef {
                path,
                ref_id: DUMMY_NODE_ID,
            },
            span,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Visibility {
    pub kind: VisibilityKind,
    pub span: Span,
    pub tokens: Option<LazyAttrTokenStream>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum VisibilityKind {
    Public,
    Restricted { path: P<Path>, id: NodeId, shorthand: bool },
    Inherited,
}

impl VisibilityKind {
    pub fn is_pub(&self) -> bool {
        matches!(self, VisibilityKind::Public)
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct FieldDef {
    pub attrs: AttrVec,
    pub id: NodeId,
    pub span: Span,
    pub vis: Visibility,
    pub ident: Option<Ident>,

    pub ty: P<Ty>,
    pub is_placeholder: bool,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum VariantData {
    Struct(ThinVec<FieldDef>, bool),
    Tuple(ThinVec<FieldDef>, NodeId),
    Unit(NodeId),
}

impl VariantData {
    pub fn fields(&self) -> &[FieldDef] {
        match self {
            VariantData::Struct(fields, ..) | VariantData::Tuple(fields, _) => fields,
            _ => &[],
        }
    }

    pub fn ctor_node_id(&self) -> Option<NodeId> {
        match *self {
            VariantData::Struct(..) => None,
            VariantData::Tuple(_, id) | VariantData::Unit(id) => Some(id),
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Item<K = ItemKind> {
    pub attrs: AttrVec,
    pub id: NodeId,
    pub span: Span,
    pub vis: Visibility,
    pub ident: Ident,

    pub kind: K,

    pub tokens: Option<LazyAttrTokenStream>,
}

impl Item {
    pub fn span_with_attributes(&self) -> Span {
        self.attrs.iter().fold(self.span, |acc, attr| acc.to(attr.span))
    }
}

#[derive(Clone, Copy, Encodable, Decodable, Debug)]
pub enum Extern {
    None,
    Implicit(Span),
    Explicit(StrLit, Span),
}

impl Extern {
    pub fn from_abi(abi: Option<StrLit>, span: Span) -> Extern {
        match abi {
            Some(name) => Extern::Explicit(name, span),
            None => Extern::Implicit(span),
        }
    }
}

#[derive(Clone, Copy, Encodable, Decodable, Debug)]
pub struct FnHeader {
    pub unsafety: Unsafe,
    pub asyncness: Async,
    pub constness: Const,
    pub ext: Extern,
}

impl FnHeader {
    pub fn has_qualifiers(&self) -> bool {
        let Self {
            unsafety,
            asyncness,
            constness,
            ext,
        } = self;
        matches!(unsafety, Unsafe::Yes(_))
            || asyncness.is_async()
            || matches!(constness, Const::Yes(_))
            || !matches!(ext, Extern::None)
    }
}

impl Default for FnHeader {
    fn default() -> FnHeader {
        FnHeader {
            unsafety: Unsafe::No,
            asyncness: Async::No,
            constness: Const::No,
            ext: Extern::None,
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Trait {
    pub unsafety: Unsafe,
    pub is_auto: IsAuto,
    pub generics: Generics,
    pub bounds: GenericBounds,
    pub items: ThinVec<P<AssocItem>>,
}

#[derive(Copy, Clone, Encodable, Decodable, Debug, Default)]
pub struct TyAliasWhereClause(pub bool, pub Span);

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct TyAlias {
    pub defaultness: Defaultness,
    pub generics: Generics,
    pub where_clauses: (TyAliasWhereClause, TyAliasWhereClause),
    pub where_predicates_split: usize,
    pub bounds: GenericBounds,
    pub ty: Option<P<Ty>>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Impl {
    pub defaultness: Defaultness,
    pub unsafety: Unsafe,
    pub generics: Generics,
    pub constness: Const,
    pub polarity: ImplPolarity,
    pub of_trait: Option<TraitRef>,
    pub self_ty: P<Ty>,
    pub items: ThinVec<P<AssocItem>>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct Fn {
    pub defaultness: Defaultness,
    pub generics: Generics,
    pub sig: FnSig,
    pub body: Option<P<Block>>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct StaticItem {
    pub ty: P<Ty>,
    pub mutability: Mutability,
    pub expr: Option<P<Expr>>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct ConstItem {
    pub defaultness: Defaultness,
    pub ty: P<Ty>,
    pub expr: Option<P<Expr>>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum ItemKind {
    ExternCrate(Option<Symbol>),
    Use(UseTree),
    Static(Box<StaticItem>),
    Const(Box<ConstItem>),
    Fn(Box<Fn>),
    Mod(Unsafe, ModKind),
    ForeignMod(ForeignMod),
    GlobalAsm(Box<InlineAsm>),
    TyAlias(Box<TyAlias>),
    Enum(EnumDef, Generics),
    Struct(VariantData, Generics),
    Union(VariantData, Generics),
    Trait(Box<Trait>),
    TraitAlias(Generics, GenericBounds),
    Impl(Box<Impl>),
    MacCall(P<MacCall>),

    MacroDef(MacroDef),
}

impl ItemKind {
    pub fn article(&self) -> &'static str {
        use ItemKind::*;
        match self {
            Use(..) | Static(..) | Const(..) | Fn(..) | Mod(..) | GlobalAsm(..) | TyAlias(..) | Struct(..)
            | Union(..) | Trait(..) | TraitAlias(..) | MacroDef(..) => "a",
            ExternCrate(..) | ForeignMod(..) | MacCall(..) | Enum(..) | Impl { .. } => "an",
        }
    }

    pub fn descr(&self) -> &'static str {
        match self {
            ItemKind::ExternCrate(..) => "extern crate",
            ItemKind::Use(..) => "`use` import",
            ItemKind::Static(..) => "static item",
            ItemKind::Const(..) => "constant item",
            ItemKind::Fn(..) => "function",
            ItemKind::Mod(..) => "module",
            ItemKind::ForeignMod(..) => "extern block",
            ItemKind::GlobalAsm(..) => "global asm item",
            ItemKind::TyAlias(..) => "type alias",
            ItemKind::Enum(..) => "enum",
            ItemKind::Struct(..) => "struct",
            ItemKind::Union(..) => "union",
            ItemKind::Trait(..) => "trait",
            ItemKind::TraitAlias(..) => "trait alias",
            ItemKind::MacCall(..) => "item macro invocation",
            ItemKind::MacroDef(..) => "macro definition",
            ItemKind::Impl { .. } => "implementation",
        }
    }

    pub fn generics(&self) -> Option<&Generics> {
        match self {
            Self::Fn(box Fn { generics, .. })
            | Self::TyAlias(box TyAlias { generics, .. })
            | Self::Enum(_, generics)
            | Self::Struct(_, generics)
            | Self::Union(_, generics)
            | Self::Trait(box Trait { generics, .. })
            | Self::TraitAlias(generics, _)
            | Self::Impl(box Impl { generics, .. }) => Some(generics),
            _ => None,
        }
    }
}

pub type AssocItem = Item<AssocItemKind>;

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum AssocItemKind {
    Const(Box<ConstItem>),
    Fn(Box<Fn>),
    Type(Box<TyAlias>),
    MacCall(P<MacCall>),
}

impl AssocItemKind {
    pub fn defaultness(&self) -> Defaultness {
        match *self {
            Self::Const(box ConstItem { defaultness, .. })
            | Self::Fn(box Fn { defaultness, .. })
            | Self::Type(box TyAlias { defaultness, .. }) => defaultness,
            Self::MacCall(..) => Defaultness::Final,
        }
    }
}

impl From<AssocItemKind> for ItemKind {
    fn from(assoc_item_kind: AssocItemKind) -> ItemKind {
        match assoc_item_kind {
            AssocItemKind::Const(item) => ItemKind::Const(item),
            AssocItemKind::Fn(fn_kind) => ItemKind::Fn(fn_kind),
            AssocItemKind::Type(ty_alias_kind) => ItemKind::TyAlias(ty_alias_kind),
            AssocItemKind::MacCall(a) => ItemKind::MacCall(a),
        }
    }
}

impl TryFrom<ItemKind> for AssocItemKind {
    type Error = ItemKind;

    fn try_from(item_kind: ItemKind) -> Result<AssocItemKind, ItemKind> {
        Ok(match item_kind {
            ItemKind::Const(item) => AssocItemKind::Const(item),
            ItemKind::Fn(fn_kind) => AssocItemKind::Fn(fn_kind),
            ItemKind::TyAlias(ty_kind) => AssocItemKind::Type(ty_kind),
            ItemKind::MacCall(a) => AssocItemKind::MacCall(a),
            _ => return Err(item_kind),
        })
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum ForeignItemKind {
    Static(P<Ty>, Mutability, Option<P<Expr>>),
    Fn(Box<Fn>),
    TyAlias(Box<TyAlias>),
    MacCall(P<MacCall>),
}

impl From<ForeignItemKind> for ItemKind {
    fn from(foreign_item_kind: ForeignItemKind) -> ItemKind {
        match foreign_item_kind {
            ForeignItemKind::Static(a, b, c) => ItemKind::Static(
                StaticItem {
                    ty: a,
                    mutability: b,
                    expr: c,
                }
                .into(),
            ),
            ForeignItemKind::Fn(fn_kind) => ItemKind::Fn(fn_kind),
            ForeignItemKind::TyAlias(ty_alias_kind) => ItemKind::TyAlias(ty_alias_kind),
            ForeignItemKind::MacCall(a) => ItemKind::MacCall(a),
        }
    }
}

impl TryFrom<ItemKind> for ForeignItemKind {
    type Error = ItemKind;

    fn try_from(item_kind: ItemKind) -> Result<ForeignItemKind, ItemKind> {
        Ok(match item_kind {
            ItemKind::Static(box StaticItem {
                ty: a,
                mutability: b,
                expr: c,
            }) => ForeignItemKind::Static(a, b, c),
            ItemKind::Fn(fn_kind) => ForeignItemKind::Fn(fn_kind),
            ItemKind::TyAlias(ty_alias_kind) => ForeignItemKind::TyAlias(ty_alias_kind),
            ItemKind::MacCall(a) => ForeignItemKind::MacCall(a),
            _ => return Err(item_kind),
        })
    }
}

pub type ForeignItem = Item<ForeignItemKind>;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod size_asserts {
    use super::*;
    use rustc_data_structures::static_assert_size;
    static_assert_size!(AssocItem, 88);
    static_assert_size!(AssocItemKind, 16);
    static_assert_size!(Attribute, 32);
    static_assert_size!(Block, 32);
    static_assert_size!(Expr, 72);
    static_assert_size!(ExprKind, 40);
    static_assert_size!(Fn, 152);
    static_assert_size!(ForeignItem, 96);
    static_assert_size!(ForeignItemKind, 24);
    static_assert_size!(GenericArg, 24);
    static_assert_size!(GenericBound, 56);
    static_assert_size!(Generics, 40);
    static_assert_size!(Impl, 136);
    static_assert_size!(Item, 136);
    static_assert_size!(ItemKind, 64);
    static_assert_size!(LitKind, 24);
    static_assert_size!(Local, 72);
    static_assert_size!(MetaItemLit, 40);
    static_assert_size!(Param, 40);
    static_assert_size!(Pat, 72);
    static_assert_size!(Path, 24);
    static_assert_size!(PathSegment, 24);
    static_assert_size!(PatKind, 48);
    static_assert_size!(Stmt, 32);
    static_assert_size!(StmtKind, 16);
    static_assert_size!(Ty, 64);
    static_assert_size!(TyKind, 40);
}
