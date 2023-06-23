#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    test(attr(deny(warnings)))
)]
#![feature(associated_type_bounds)]
#![feature(box_patterns)]
#![feature(const_trait_impl)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(min_specialization)]
#![feature(negative_impls)]
#![feature(stmt_expr_attributes)]
#![recursion_limit = "256"]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate rustc_macros;

#[macro_use]
extern crate tracing;

use crate::ptr::P;
use crate::token::{self, CommentKind, Delimiter, Nonterminal};
use crate::tokenstream::{DelimSpan, LazyAttrTokenStream, TokenStream};
pub use crate::util::parser::ExprPrecedence;
use rustc_data_structures::{
    fx::FxHashMap,
    stable_hasher::{HashStable, StableHasher},
    stack::ensure_sufficient_stack,
    sync::Lrc,
};
use rustc_macros::HashStable_Generic;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_span::{
    source_map::{respan, Spanned},
    symbol::{kw, sym, Ident, Symbol},
    LocalExpnId, Span, DUMMY_SP,
};
use std::{fmt, marker::PhantomData, mem};
use thin_vec::{thin_vec, ThinVec};

pub mod util {
    pub mod case;
    pub mod classify;
    pub mod comments;
    pub mod literal;
    pub mod parser;
    pub mod unicode;
}
pub mod attr;
pub mod expand;
pub mod mut_visit;
pub mod ptr {
    use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
    use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
    use std::fmt::{self, Debug, Display};
    use std::ops::{Deref, DerefMut};
    use std::{slice, vec};
    pub struct P<T: ?Sized> {
        ptr: Box<T>,
    }
    #[allow(non_snake_case)]
    pub fn P<T: 'static>(value: T) -> P<T> {
        P { ptr: Box::new(value) }
    }
    impl<T: 'static> P<T> {
        pub fn and_then<U, F>(self, f: F) -> U
        where
            F: FnOnce(T) -> U,
        {
            f(*self.ptr)
        }
        pub fn into_inner(self) -> T {
            *self.ptr
        }
        pub fn map<F>(mut self, f: F) -> P<T>
        where
            F: FnOnce(T) -> T,
        {
            let x = f(*self.ptr);
            *self.ptr = x;
            self
        }
        pub fn filter_map<F>(mut self, f: F) -> Option<P<T>>
        where
            F: FnOnce(T) -> Option<T>,
        {
            *self.ptr = f(*self.ptr)?;
            Some(self)
        }
    }
    impl<T: ?Sized> Deref for P<T> {
        type Target = T;
        fn deref(&self) -> &T {
            &self.ptr
        }
    }
    impl<T: ?Sized> DerefMut for P<T> {
        fn deref_mut(&mut self) -> &mut T {
            &mut self.ptr
        }
    }
    impl<T: 'static + Clone> Clone for P<T> {
        fn clone(&self) -> P<T> {
            P((**self).clone())
        }
    }
    impl<T: ?Sized + Debug> Debug for P<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            Debug::fmt(&self.ptr, f)
        }
    }
    impl<T: Display> Display for P<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            Display::fmt(&**self, f)
        }
    }
    impl<T> fmt::Pointer for P<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Pointer::fmt(&self.ptr, f)
        }
    }
    impl<D: Decoder, T: 'static + Decodable<D>> Decodable<D> for P<T> {
        fn decode(d: &mut D) -> P<T> {
            P(Decodable::decode(d))
        }
    }
    impl<S: Encoder, T: Encodable<S>> Encodable<S> for P<T> {
        fn encode(&self, s: &mut S) {
            (**self).encode(s);
        }
    }
    impl<T> P<[T]> {
        pub fn new() -> P<[T]> {
            P { ptr: Box::default() }
        }
        #[inline(never)]
        pub fn from_vec(v: Vec<T>) -> P<[T]> {
            P {
                ptr: v.into_boxed_slice(),
            }
        }
        #[inline(never)]
        pub fn into_vec(self) -> Vec<T> {
            self.ptr.into_vec()
        }
    }
    impl<T> Default for P<[T]> {
        fn default() -> P<[T]> {
            P::new()
        }
    }
    impl<T: Clone> Clone for P<[T]> {
        fn clone(&self) -> P<[T]> {
            P::from_vec(self.to_vec())
        }
    }
    impl<T> From<Vec<T>> for P<[T]> {
        fn from(v: Vec<T>) -> Self {
            P::from_vec(v)
        }
    }
    impl<T> Into<Vec<T>> for P<[T]> {
        fn into(self) -> Vec<T> {
            self.into_vec()
        }
    }
    impl<T> FromIterator<T> for P<[T]> {
        fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> P<[T]> {
            P::from_vec(iter.into_iter().collect())
        }
    }
    impl<T> IntoIterator for P<[T]> {
        type Item = T;
        type IntoIter = vec::IntoIter<T>;
        fn into_iter(self) -> Self::IntoIter {
            self.into_vec().into_iter()
        }
    }
    impl<'a, T> IntoIterator for &'a P<[T]> {
        type Item = &'a T;
        type IntoIter = slice::Iter<'a, T>;
        fn into_iter(self) -> Self::IntoIter {
            self.ptr.into_iter()
        }
    }
    impl<S: Encoder, T: Encodable<S>> Encodable<S> for P<[T]> {
        fn encode(&self, s: &mut S) {
            Encodable::encode(&**self, s);
        }
    }
    impl<D: Decoder, T: Decodable<D>> Decodable<D> for P<[T]> {
        fn decode(d: &mut D) -> P<[T]> {
            P::from_vec(Decodable::decode(d))
        }
    }
    impl<CTX, T> HashStable<CTX> for P<T>
    where
        T: ?Sized + HashStable<CTX>,
    {
        fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
            (**self).hash_stable(hcx, hasher);
        }
    }
}
pub mod token;
pub mod tokenstream;
pub mod visit;

#[derive(Debug)]
pub enum EntryPointType {
    None,
    MainNamed,
    RustcMainAttr,
    Start,
    OtherMain,
}

rustc_index::newtype_index! {
    #[debug_format = "NodeId({})"]
    pub struct NodeId {
        const CRATE_NODE_ID = 0;
    }
}
rustc_data_structures::define_id_collections!(NodeMap, NodeSet, NodeMapEntry, NodeId);
impl NodeId {
    pub fn placeholder_from_expn_id(expn_id: LocalExpnId) -> Self {
        NodeId::from_u32(expn_id.as_u32())
    }
    pub fn placeholder_to_expn_id(self) -> LocalExpnId {
        LocalExpnId::from_u32(self.as_u32())
    }
}
impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.as_u32(), f)
    }
}

pub const DUMMY_NODE_ID: NodeId = NodeId::MAX;

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
pub use GenericArgs::*;

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
pub use UnsafeSource::*;

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

pub trait HashStableContext: rustc_span::HashStableContext {
    fn hash_attr(&mut self, _: &Attribute, hasher: &mut StableHasher);
}
impl<AstCtx: crate::HashStableContext> HashStable<AstCtx> for Attribute {
    fn hash_stable(&self, hcx: &mut AstCtx, hasher: &mut StableHasher) {
        hcx.hash_attr(self, hasher)
    }
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
pub enum SelfKind {
    Value(Mutability),
    Region(Option<Lifetime>, Mutability),
    Explicit(P<Ty>, Mutability),
}
pub type ExplicitSelf = Spanned<SelfKind>;

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
pub type AssocItem = Item<AssocItemKind>;

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

pub trait AstDeref {
    type Target;
    fn ast_deref(&self) -> &Self::Target;
    fn ast_deref_mut(&mut self) -> &mut Self::Target;
}
impl<T> AstDeref for P<T> {
    type Target = T;
    fn ast_deref(&self) -> &Self::Target {
        self
    }
    fn ast_deref_mut(&mut self) -> &mut Self::Target {
        self
    }
}

macro_rules! impl_not_ast_deref {
        ($($T:ty),+ $(,)?) => {
            $(
                impl !AstDeref for $T {}
            )+
        };
    }
impl_not_ast_deref!(AssocItem, Expr, ForeignItem, Item, Stmt);

pub trait HasNodeId {
    fn node_id(&self) -> NodeId;
    fn node_id_mut(&mut self) -> &mut NodeId;
}
impl<T: AstDeref<Target: HasNodeId>> HasNodeId for T {
    fn node_id(&self) -> NodeId {
        self.ast_deref().node_id()
    }
    fn node_id_mut(&mut self) -> &mut NodeId {
        self.ast_deref_mut().node_id_mut()
    }
}

macro_rules! impl_has_node_id {
    ($($T:ty),+ $(,)?) => {
        $(
            impl HasNodeId for $T {
                fn node_id(&self) -> NodeId {
                    self.id
                }
                fn node_id_mut(&mut self) -> &mut NodeId {
                    &mut self.id
                }
            }
        )+
    };
}
impl_has_node_id!(
    Arm,
    AssocItem,
    Crate,
    Expr,
    ExprField,
    FieldDef,
    ForeignItem,
    GenericParam,
    Item,
    Param,
    Pat,
    PatField,
    Stmt,
    Ty,
    Variant,
);

pub trait HasSpan {
    fn span(&self) -> Span;
}
impl<T: AstDeref<Target: HasSpan>> HasSpan for T {
    fn span(&self) -> Span {
        self.ast_deref().span()
    }
}
impl HasSpan for AttrItem {
    fn span(&self) -> Span {
        self.span()
    }
}

macro_rules! impl_has_span {
    ($($T:ty),+ $(,)?) => {
        $(
            impl HasSpan for $T {
                fn span(&self) -> Span {
                    self.span
                }
            }
        )+
    };
}
impl_has_span!(
    AssocItem,
    Block,
    Expr,
    ForeignItem,
    Item,
    Pat,
    Path,
    Stmt,
    Ty,
    Visibility
);

pub trait HasTokens {
    fn tokens(&self) -> Option<&LazyAttrTokenStream>;
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>>;
}
impl<T: AstDeref<Target: HasTokens>> HasTokens for T {
    fn tokens(&self) -> Option<&LazyAttrTokenStream> {
        self.ast_deref().tokens()
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>> {
        self.ast_deref_mut().tokens_mut()
    }
}
impl<T: HasTokens> HasTokens for Option<T> {
    fn tokens(&self) -> Option<&LazyAttrTokenStream> {
        self.as_ref().and_then(|inner| inner.tokens())
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>> {
        self.as_mut().and_then(|inner| inner.tokens_mut())
    }
}
impl HasTokens for StmtKind {
    fn tokens(&self) -> Option<&LazyAttrTokenStream> {
        match self {
            StmtKind::Local(local) => local.tokens.as_ref(),
            StmtKind::Item(item) => item.tokens(),
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr.tokens(),
            StmtKind::Empty => return None,
            StmtKind::MacCall(mac) => mac.tokens.as_ref(),
        }
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>> {
        match self {
            StmtKind::Local(local) => Some(&mut local.tokens),
            StmtKind::Item(item) => item.tokens_mut(),
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr.tokens_mut(),
            StmtKind::Empty => return None,
            StmtKind::MacCall(mac) => Some(&mut mac.tokens),
        }
    }
}
impl HasTokens for Stmt {
    fn tokens(&self) -> Option<&LazyAttrTokenStream> {
        self.kind.tokens()
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>> {
        self.kind.tokens_mut()
    }
}
impl HasTokens for Attribute {
    fn tokens(&self) -> Option<&LazyAttrTokenStream> {
        match &self.kind {
            AttrKind::Normal(normal) => normal.tokens.as_ref(),
            kind @ AttrKind::DocComment(..) => {
                panic!("Called tokens on doc comment attr {kind:?}")
            },
        }
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>> {
        Some(match &mut self.kind {
            AttrKind::Normal(normal) => &mut normal.tokens,
            kind @ AttrKind::DocComment(..) => {
                panic!("Called tokens_mut on doc comment attr {kind:?}")
            },
        })
    }
}
impl HasTokens for Nonterminal {
    fn tokens(&self) -> Option<&LazyAttrTokenStream> {
        match self {
            Nonterminal::NtItem(item) => item.tokens(),
            Nonterminal::NtStmt(stmt) => stmt.tokens(),
            Nonterminal::NtExpr(expr) | Nonterminal::NtLiteral(expr) => expr.tokens(),
            Nonterminal::NtPat(pat) => pat.tokens(),
            Nonterminal::NtTy(ty) => ty.tokens(),
            Nonterminal::NtMeta(attr_item) => attr_item.tokens(),
            Nonterminal::NtPath(path) => path.tokens(),
            Nonterminal::NtVis(vis) => vis.tokens(),
            Nonterminal::NtBlock(block) => block.tokens(),
            Nonterminal::NtIdent(..) | Nonterminal::NtLifetime(..) => None,
        }
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>> {
        match self {
            Nonterminal::NtItem(item) => item.tokens_mut(),
            Nonterminal::NtStmt(stmt) => stmt.tokens_mut(),
            Nonterminal::NtExpr(expr) | Nonterminal::NtLiteral(expr) => expr.tokens_mut(),
            Nonterminal::NtPat(pat) => pat.tokens_mut(),
            Nonterminal::NtTy(ty) => ty.tokens_mut(),
            Nonterminal::NtMeta(attr_item) => attr_item.tokens_mut(),
            Nonterminal::NtPath(path) => path.tokens_mut(),
            Nonterminal::NtVis(vis) => vis.tokens_mut(),
            Nonterminal::NtBlock(block) => block.tokens_mut(),
            Nonterminal::NtIdent(..) | Nonterminal::NtLifetime(..) => None,
        }
    }
}

macro_rules! impl_has_tokens {
    ($($T:ty),+ $(,)?) => {
        $(
            impl HasTokens for $T {
                fn tokens(&self) -> Option<&LazyAttrTokenStream> {
                    self.tokens.as_ref()
                }
                fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>> {
                    Some(&mut self.tokens)
                }
            }
        )+
    };
}
impl_has_tokens!(
    AssocItem,
    AttrItem,
    Block,
    Expr,
    ForeignItem,
    Item,
    Pat,
    Path,
    Ty,
    Visibility
);

macro_rules! impl_has_tokens_none {
    ($($T:ty),+ $(,)?) => {
        $(
            impl HasTokens for $T {
                fn tokens(&self) -> Option<&LazyAttrTokenStream> {
                    None
                }
                fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>> {
                    None
                }
            }
        )+
    };
}
impl_has_tokens_none!(Arm, ExprField, FieldDef, GenericParam, Param, PatField, Variant);

pub trait HasAttrs {
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool;
    fn attrs(&self) -> &[Attribute];
    fn visit_attrs(&mut self, f: impl FnOnce(&mut AttrVec));
}
impl<T: AstDeref<Target: HasAttrs>> HasAttrs for T {
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = T::Target::SUPPORTS_CUSTOM_INNER_ATTRS;
    fn attrs(&self) -> &[Attribute] {
        self.ast_deref().attrs()
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut AttrVec)) {
        self.ast_deref_mut().visit_attrs(f)
    }
}
impl<T: HasAttrs> HasAttrs for Option<T> {
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = T::SUPPORTS_CUSTOM_INNER_ATTRS;
    fn attrs(&self) -> &[Attribute] {
        self.as_ref().map(|inner| inner.attrs()).unwrap_or(&[])
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut AttrVec)) {
        if let Some(inner) = self.as_mut() {
            inner.visit_attrs(f);
        }
    }
}
impl HasAttrs for StmtKind {
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = true;
    fn attrs(&self) -> &[Attribute] {
        match self {
            StmtKind::Local(local) => &local.attrs,
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr.attrs(),
            StmtKind::Item(item) => item.attrs(),
            StmtKind::Empty => &[],
            StmtKind::MacCall(mac) => &mac.attrs,
        }
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut AttrVec)) {
        match self {
            StmtKind::Local(local) => f(&mut local.attrs),
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr.visit_attrs(f),
            StmtKind::Item(item) => item.visit_attrs(f),
            StmtKind::Empty => {},
            StmtKind::MacCall(mac) => f(&mut mac.attrs),
        }
    }
}
impl HasAttrs for Stmt {
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = StmtKind::SUPPORTS_CUSTOM_INNER_ATTRS;
    fn attrs(&self) -> &[Attribute] {
        self.kind.attrs()
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut AttrVec)) {
        self.kind.visit_attrs(f);
    }
}

macro_rules! impl_has_attrs {
    (const SUPPORTS_CUSTOM_INNER_ATTRS: bool = $inner:literal, $($T:ty),+ $(,)?) => {
        $(
            impl HasAttrs for $T {
                const SUPPORTS_CUSTOM_INNER_ATTRS: bool = $inner;
                #[inline]
                fn attrs(&self) -> &[Attribute] {
                    &self.attrs
                }
                fn visit_attrs(&mut self, f: impl FnOnce(&mut AttrVec)) {
                    f(&mut self.attrs)
                }
            }
        )+
    };
}
impl_has_attrs!(
const SUPPORTS_CUSTOM_INNER_ATTRS: bool = true,
AssocItem,
ForeignItem,
Item,
);
impl_has_attrs!(
const SUPPORTS_CUSTOM_INNER_ATTRS: bool = false,
Arm,
Crate,
Expr,
ExprField,
FieldDef,
GenericParam,
Param,
PatField,
Variant,
);

macro_rules! impl_has_attrs_none {
($($T:ty),+ $(,)?) => {
    $(
        impl HasAttrs for $T {
            const SUPPORTS_CUSTOM_INNER_ATTRS: bool = false;
            fn attrs(&self) -> &[Attribute] {
                &[]
            }
            fn visit_attrs(&mut self, _f: impl FnOnce(&mut AttrVec)) {}
        }
    )+
};
}
impl_has_attrs_none!(Attribute, AttrItem, Block, Pat, Path, Ty, Visibility);

pub struct AstNodeWrapper<Wrapped, Tag> {
    pub wrapped: Wrapped,
    pub tag: PhantomData<Tag>,
}
impl<Wrapped, Tag> AstNodeWrapper<Wrapped, Tag> {
    pub fn new(wrapped: Wrapped, _tag: Tag) -> AstNodeWrapper<Wrapped, Tag> {
        AstNodeWrapper {
            wrapped,
            tag: Default::default(),
        }
    }
}
impl<Wrapped, Tag> AstDeref for AstNodeWrapper<Wrapped, Tag> {
    type Target = Wrapped;
    fn ast_deref(&self) -> &Self::Target {
        &self.wrapped
    }
    fn ast_deref_mut(&mut self) -> &mut Self::Target {
        &mut self.wrapped
    }
}
impl<Wrapped: fmt::Debug, Tag> fmt::Debug for AstNodeWrapper<Wrapped, Tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AstNodeWrapper")
            .field("wrapped", &self.wrapped)
            .field("tag", &self.tag)
            .finish()
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct FormatArgs {
    pub span: Span,
    pub template: Vec<FormatArgsPiece>,
    pub arguments: FormatArguments,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum FormatArgsPiece {
    Literal(Symbol),
    Placeholder(FormatPlaceholder),
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct FormatArguments {
    arguments: Vec<FormatArgument>,
    num_unnamed_args: usize,
    num_explicit_args: usize,
    names: FxHashMap<Symbol, usize>,
}

#[cfg(parallel_compiler)]
unsafe impl Sync for FormatArguments {}
#[cfg(parallel_compiler)]
unsafe impl Send for FormatArguments {}

impl FormatArguments {
    pub fn new() -> Self {
        Self {
            arguments: Vec::new(),
            names: FxHashMap::default(),
            num_unnamed_args: 0,
            num_explicit_args: 0,
        }
    }
    pub fn add(&mut self, arg: FormatArgument) -> usize {
        let index = self.arguments.len();
        if let Some(name) = arg.kind.ident() {
            self.names.insert(name.name, index);
        } else if self.names.is_empty() {
            self.num_unnamed_args += 1;
        }
        if !matches!(arg.kind, FormatArgumentKind::Captured(..)) {
            assert_eq!(
                self.num_explicit_args,
                self.arguments.len(),
                "captured arguments must be added last"
            );
            self.num_explicit_args += 1;
        }
        self.arguments.push(arg);
        index
    }
    pub fn by_name(&self, name: Symbol) -> Option<(usize, &FormatArgument)> {
        let i = *self.names.get(&name)?;
        Some((i, &self.arguments[i]))
    }
    pub fn by_index(&self, i: usize) -> Option<&FormatArgument> {
        (i < self.num_explicit_args).then(|| &self.arguments[i])
    }
    pub fn unnamed_args(&self) -> &[FormatArgument] {
        &self.arguments[..self.num_unnamed_args]
    }
    pub fn named_args(&self) -> &[FormatArgument] {
        &self.arguments[self.num_unnamed_args..self.num_explicit_args]
    }
    pub fn explicit_args(&self) -> &[FormatArgument] {
        &self.arguments[..self.num_explicit_args]
    }
    pub fn all_args(&self) -> &[FormatArgument] {
        &self.arguments[..]
    }
    pub fn all_args_mut(&mut self) -> &mut Vec<FormatArgument> {
        &mut self.arguments
    }
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub struct FormatArgument {
    pub kind: FormatArgumentKind,
    pub expr: P<Expr>,
}

#[derive(Clone, Encodable, Decodable, Debug)]
pub enum FormatArgumentKind {
    Normal,
    Named(Ident),
    Captured(Ident),
}

impl FormatArgumentKind {
    pub fn ident(&self) -> Option<Ident> {
        match self {
            &Self::Normal => None,
            &Self::Named(id) => Some(id),
            &Self::Captured(id) => Some(id),
        }
    }
}

#[derive(Clone, Encodable, Decodable, Debug, PartialEq, Eq)]
pub struct FormatPlaceholder {
    pub argument: FormatArgPosition,
    pub span: Option<Span>,
    pub format_trait: FormatTrait,
    pub format_options: FormatOptions,
}

#[derive(Clone, Encodable, Decodable, Debug, PartialEq, Eq)]
pub struct FormatArgPosition {
    pub index: Result<usize, usize>,
    pub kind: FormatArgPositionKind,
    pub span: Option<Span>,
}

#[derive(Copy, Clone, Encodable, Decodable, Debug, PartialEq, Eq)]
pub enum FormatArgPositionKind {
    Implicit,
    Number,
    Named,
}

#[derive(Copy, Clone, Encodable, Decodable, Debug, PartialEq, Eq, Hash)]
pub enum FormatTrait {
    Display,
    Debug,
    LowerExp,
    UpperExp,
    Octal,
    Pointer,
    Binary,
    LowerHex,
    UpperHex,
}

#[derive(Clone, Encodable, Decodable, Default, Debug, PartialEq, Eq)]
pub struct FormatOptions {
    pub width: Option<FormatCount>,
    pub precision: Option<FormatCount>,
    pub alignment: Option<FormatAlignment>,
    pub fill: Option<char>,
    pub sign: Option<FormatSign>,
    pub alternate: bool,
    pub zero_pad: bool,
    pub debug_hex: Option<FormatDebugHex>,
}

#[derive(Copy, Clone, Encodable, Decodable, Debug, PartialEq, Eq)]
pub enum FormatSign {
    Plus,
    Minus,
}

#[derive(Copy, Clone, Encodable, Decodable, Debug, PartialEq, Eq)]
pub enum FormatDebugHex {
    Lower,
    Upper,
}

#[derive(Copy, Clone, Encodable, Decodable, Debug, PartialEq, Eq)]
pub enum FormatAlignment {
    Left,
    Right,
    Center,
}

#[derive(Clone, Encodable, Decodable, Debug, PartialEq, Eq)]
pub enum FormatCount {
    Literal(usize),
    Argument(FormatArgPosition),
}

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
