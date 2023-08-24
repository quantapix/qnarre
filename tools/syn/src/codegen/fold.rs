#![allow(unreachable_code, unused_variables)]
#![allow(clippy::match_wildcard_for_single_variants, clippy::needless_match)]

use crate::*;

macro_rules! full {
    ($e:expr) => {
        $e
    };
}

impl<F> Fold for path::Angle
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        path::Angle {
            colon2: self.colon2,
            lt: self.lt,
            args: FoldHelper::lift(self.args, |x| x.fold(f)),
            gt: self.gt,
        }
    }
}
impl<F> Fold for Arm
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        Arm {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            pat: self.pat.fold(f),
            guard: (self.guard).map(|x| ((x).0, Box::new(*(x).1.fold(f)))),
            fat_arrow: self.fat_arrow,
            body: Box::new(*self.body.fold(f)),
            comma: self.comma,
        }
    }
}
impl<F> Fold for path::AssocConst
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        path::AssocConst {
            ident: self.ident.fold(f),
            args: (self.args).map(|x| x.fold(f)),
            eq: self.eq,
            val: self.val.fold(f),
        }
    }
}
impl<F> Fold for path::AssocType
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        path::AssocType {
            ident: self.ident.fold(f),
            args: (self.args).map(|x| x.fold(f)),
            eq: self.eq,
            typ: self.typ.fold(f),
        }
    }
}
impl<F> Fold for typ::FnArg
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::FnArg {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            name: (self.name).map(|x| ((x).0.fold(f), (x).1)),
            typ: self.typ.fold(f),
        }
    }
}
impl<F> Fold for typ::Variadic
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::Variadic {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            name: (self.name).map(|x| ((x).0.fold(f), (x).1)),
            dots: self.dots,
            comma: self.comma,
        }
    }
}
impl<F> Fold for BinOp
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            BinOp::Add(x) => BinOp::Add(x),
            BinOp::Sub(x) => BinOp::Sub(x),
            BinOp::Mul(x) => BinOp::Mul(x),
            BinOp::Div(x) => BinOp::Div(x),
            BinOp::Rem(x) => BinOp::Rem(x),
            BinOp::And(x) => BinOp::And(x),
            BinOp::Or(x) => BinOp::Or(x),
            BinOp::BitXor(x) => BinOp::BitXor(x),
            BinOp::BitAnd(x) => BinOp::BitAnd(x),
            BinOp::BitOr(x) => BinOp::BitOr(x),
            BinOp::Shl(x) => BinOp::Shl(x),
            BinOp::Shr(x) => BinOp::Shr(x),
            BinOp::Eq(x) => BinOp::Eq(x),
            BinOp::Lt(x) => BinOp::Lt(x),
            BinOp::Le(x) => BinOp::Le(x),
            BinOp::Ne(x) => BinOp::Ne(x),
            BinOp::Ge(x) => BinOp::Ge(x),
            BinOp::Gt(x) => BinOp::Gt(x),
            BinOp::AddAssign(x) => BinOp::AddAssign(x),
            BinOp::SubAssign(x) => BinOp::SubAssign(x),
            BinOp::MulAssign(x) => BinOp::MulAssign(x),
            BinOp::DivAssign(x) => BinOp::DivAssign(x),
            BinOp::RemAssign(x) => BinOp::RemAssign(x),
            BinOp::BitXorAssign(x) => BinOp::BitXorAssign(x),
            BinOp::BitAndAssign(x) => BinOp::BitAndAssign(x),
            BinOp::BitOrAssign(x) => BinOp::BitOrAssign(x),
            BinOp::ShlAssign(x) => BinOp::ShlAssign(x),
            BinOp::ShrAssign(x) => BinOp::ShrAssign(x),
        }
    }
}
impl<F> Fold for Block
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        Block {
            brace: self.brace,
            stmts: FoldHelper::lift(self.stmts, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for Bgen::bound::Lifes
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        Bgen::bound::Lifes {
            for_: self.for_,
            lt: self.lt,
            lifes: FoldHelper::lift(self.lifes, |x| x.fold(f)),
            gt: self.gt,
        }
    }
}
impl<F> Fold for gen::param::Const
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::param::Const {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            const_: self.const_,
            ident: self.ident.fold(f),
            colon: self.colon,
            typ: self.typ.fold(f),
            eq: self.eq,
            default: (self.default).map(|x| x.fold(f)),
        }
    }
}
impl<F> Fold for Constraint
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        Constraint {
            ident: self.ident.fold(f),
            gnrs: (self.gnrs).map(|x| x.fold(f)),
            colon: self.colon,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for Expr
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            Expr::Array(x) => Expr::Array(full!(x.fold(f))),
            Expr::Assign(x) => Expr::Assign(full!(x.fold(f))),
            Expr::Async(x) => Expr::Async(full!(x.fold(f))),
            Expr::Await(x) => Expr::Await(full!(x.fold(f))),
            Expr::Binary(x) => Expr::Binary(x.fold(f)),
            Expr::Block(x) => Expr::Block(full!(x.fold(f))),
            Expr::Break(x) => Expr::Break(full!(x.fold(f))),
            Expr::Call(x) => Expr::Call(x.fold(f)),
            Expr::Cast(x) => Expr::Cast(x.fold(f)),
            Expr::Closure(x) => Expr::Closure(full!(x.fold(f))),
            Expr::Const(x) => Expr::Const(full!(x.fold(f))),
            Expr::Continue(x) => Expr::Continue(full!(x.fold(f))),
            Expr::Field(x) => Expr::Field(x.fold(f)),
            Expr::For(x) => Expr::For(full!(x.fold(f))),
            Expr::Group(x) => Expr::Group(x.fold(f)),
            Expr::If(x) => Expr::If(full!(x.fold(f))),
            Expr::Index(x) => Expr::Index(x.fold(f)),
            Expr::Infer(x) => Expr::Infer(full!(x.fold(f))),
            Expr::Let(x) => Expr::Let(full!(x.fold(f))),
            Expr::Lit(x) => Expr::Lit(x.fold(f)),
            Expr::Loop(x) => Expr::Loop(full!(x.fold(f))),
            Expr::Macro(x) => Expr::Macro(x.fold(f)),
            Expr::Match(x) => Expr::Match(full!(x.fold(f))),
            Expr::Method(x) => Expr::Method(full!(x.fold(f))),
            Expr::Parenth(x) => Expr::Parenth(x.fold(f)),
            Expr::Path(x) => Expr::Path(x.fold(f)),
            Expr::Range(x) => Expr::Range(full!(x.fold(f))),
            Expr::Reference(x) => Expr::Reference(full!(x.fold(f))),
            Expr::Repeat(x) => Expr::Repeat(full!(x.fold(f))),
            Expr::Return(x) => Expr::Return(full!(x.fold(f))),
            Expr::Struct(x) => Expr::Struct(full!(x.fold(f))),
            Expr::Try(x) => Expr::Try(full!(x.fold(f))),
            Expr::TryBlock(x) => Expr::TryBlock(full!(x.fold(f))),
            Expr::Tuple(x) => Expr::Tuple(full!(x.fold(f))),
            Expr::Unary(x) => Expr::Unary(x.fold(f)),
            Expr::Unsafe(x) => Expr::Unsafe(full!(x.fold(f))),
            Expr::Stream(x) => Expr::Stream(x),
            Expr::While(x) => Expr::While(full!(x.fold(f))),
            Expr::Yield(x) => Expr::Yield(full!(x.fold(f))),
        }
    }
}
impl<F> Fold for pat::Field
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        pat::Fieldeld {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            member: self.memb.fold(f),
            colon: self.colon,
            pat: Box::new(*self.pat.fold(f)),
        }
    }
}
impl<F> Fold for FieldValue
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        FieldValue {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            member: self.member.fold(f),
            colon: self.colon,
            expr: self.expr.fold(f),
        }
    }
}
impl<F> Fold for Arg
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            Arg::Life(x) => Arg::Life(x.fold(f)),
            Arg::Type(x) => Arg::Type(x.fold(f)),
            Arg::Const(x) => Arg::Const(x.fold(f)),
            Arg::AssocType(x) => Arg::AssocType(x.fold(f)),
            Arg::AssocConst(x) => Arg::AssocConst(x.fold(f)),
            Arg::Constraint(x) => Arg::Constraint(x.fold(f)),
        }
    }
}
impl<F> Fold for gen::Param
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            gen::Param::Life(x) => gen::Param::Life(x.fold(f)),
            gen::Param::Type(x) => gen::Param::Type(x.fold(f)),
            gen::Param::Const(x) => gen::Param::Const(x.fold(f)),
        }
    }
}
impl<F> Fold for gen::Gens
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::Gens {
            lt: self.lt,
            params: FoldHelper::lift(self.params, |x| x.fold(f)),
            gt: self.gt,
            where_: (self.where_).map(|x| x.fold(f)),
        }
    }
}
impl<F> Fold for Ident
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        let mut y = self;
        let span = self.span().fold(f);
        self.set_span(span);
        y
    }
}
impl<F> Fold for Index
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        Index {
            index: self.index,
            span: self.span.fold(f),
        }
    }
}
impl<F> Fold for Label
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        Label {
            name: self.name.fold(f),
            colon: self.colon,
        }
    }
}
impl<F> Fold for Life
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        Life {
            apos: self.apos.fold(f),
            ident: self.ident.fold(f),
        }
    }
}
impl<F> Fold for gen::param::Life
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::param::Life {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            life: self.life.fold(f),
            colon: self.colon,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for Lit
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            Lit::Str(x) => Lit::Str(x.fold(f)),
            Lit::ByteStr(x) => Lit::ByteStr(x.fold(f)),
            Lit::Byte(x) => Lit::Byte(x.fold(f)),
            Lit::Char(x) => Lit::Char(x.fold(f)),
            Lit::Int(x) => Lit::Int(x.fold(f)),
            Lit::Float(x) => Lit::Float(x.fold(f)),
            Lit::Bool(x) => Lit::Bool(x.fold(f)),
            Lit::Stream(x) => Lit::Stream(x),
        }
    }
}
impl<F> Fold for lit::Bool
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        lit::Bool {
            val: self.val,
            span: self.span.fold(f),
        }
    }
}
impl<F> Fold for lit::Byte
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        let span = self.span().fold(f);
        let mut y = self;
        self.set_span(span);
        y
    }
}
impl<F> Fold for lit::ByteStr
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        let span = self.span().fold(f);
        let mut y = self;
        self.set_span(span);
        y
    }
}
impl<F> Fold for lit::Char
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        let span = self.span().fold(f);
        let mut y = self;
        self.set_span(span);
        y
    }
}
impl<F> Fold for lit::Float
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        let span = self.span().fold(f);
        let mut y = self;
        self.set_span(span);
        y
    }
}
impl<F> Fold for lit::Int
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        let span = self.span().fold(f);
        let mut y = self;
        self.set_span(span);
        y
    }
}
impl<F> Fold for lit::Str
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        let span = self.span().fold(f);
        let mut y = self;
        self.set_span(span);
        y
    }
}
impl<F> Fold for stmt::Local
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        stmt::Local {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            let_: self.let_,
            pat: self.pat.fold(f),
            init: (self.init).map(|x| x.fold(f)),
            semi: self.semi,
        }
    }
}
impl<F> Fold for stmt::Init
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        stmt::Init {
            eq: self.eq,
            expr: Box::new(*self.expr.fold(f)),
            diverge: (self.diverge).map(|x| ((x).0, Box::new(*(x).1.fold(f)))),
        }
    }
}
impl<F> Fold for Macro
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        Macro {
            path: self.path.fold(f),
            bang: self.bang,
            delim: self.delim.fold(f),
            toks: self.toks,
        }
    }
}
impl<F> Fold for tok::Delim
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            tok::Delim::Parenth(x) => tok::Delim::Parenth(x),
            tok::Delim::Brace(x) => tok::Delim::Brace(x),
            tok::Delim::Bracket(x) => tok::Delim::Bracket(x),
        }
    }
}
impl<F> Fold for Member
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            Member::Named(x) => Member::Named(x.fold(f)),
            Member::Unnamed(x) => Member::Unnamed(x.fold(f)),
        }
    }
}
impl<F> Fold for path::Parenth
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        path::Parenth {
            parenth: self.parenth,
            args: FoldHelper::lift(self.args, |ixt| x.fold(f)),
            ret: self.ret.fold(f),
        }
    }
}
impl<F> Fold for pat::Pat
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            pat::Pat::Const(x) => pat::Pat::Const(x.fold(f)),
            pat::Pat::Ident(x) => pat::Pat::Ident(x.fold(f)),
            pat::Pat::Lit(x) => pat::Pat::Lit(x.fold(f)),
            pat::Pat::Mac(x) => pat::Pat::Mac(x.fold(f)),
            pat::Pat::Or(x) => pat::Pat::Or(x.fold(f)),
            pat::Pat::Parenth(x) => pat::Pat::Parenth(x.fold(f)),
            pat::Pat::Path(x) => pat::Pat::Path(x.fold(f)),
            pat::Pat::Range(x) => pat::Pat::Range(x.fold(f)),
            pat::Pat::Ref(x) => pat::Pat::Ref(x.fold(f)),
            pat::Pat::Rest(x) => pat::Pat::Rest(x.fold(f)),
            pat::Pat::Slice(x) => pat::Pat::Slice(x.fold(f)),
            pat::Pat::Struct(x) => pat::Pat::Struct(x.fold(f)),
            pat::Pat::Tuple(x) => pat::Pat::Tuple(x.fold(f)),
            pat::Pat::TupleStruct(x) => pat::Pat::TupleStruct(x.fold(f)),
            pat::Pat::Type(x) => pat::Pat::Type(x.fold(f)),
            pat::Pat::Verbatim(x) => pat::Pat::Verbatim(x),
            pat::Pat::Wild(x) => pat::Pat::Wild(x.fold(f)),
        }
    }
}
impl<F> Fold for pat::Ident
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        pat::Ident {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            ref_: self.ref_,
            mut_: self.mut_,
            ident: self.ident.fold(f),
            sub: (self.sub).map(|x| ((x).0, Box::new(*(x).1.fold(f)))),
        }
    }
}
impl<F> Fold for pat::Or
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        pat::Or {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vert: self.vert,
            cases: FoldHelper::lift(self.cases, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for pat::Parenth
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        pat::Parenth {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            parenth: self.parenth,
            pat: Box::new(*self.pat.fold(f)),
        }
    }
}
impl<F> Fold for pat::Ref
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        pat::Ref {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            and: self.and,
            mut_: self.mut_,
            pat: Box::new(*self.pat.fold(f)),
        }
    }
}
impl<F> Fold for pat::Rest
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        pat::Restest {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            dot2: self.dot2,
        }
    }
}
impl<F> Fold for pat::Slice
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        pat::Slice {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            bracket: self.bracket,
            pats: FoldHelper::lift(self.pats, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for pat::Struct
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        pat::Struct {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            qself: (self.qself).map(|x| x.fold(f)),
            path: self.path.fold(f),
            brace: self.brace,
            fields: FoldHelper::lift(self.fields, |x| x.fold(f)),
            rest: (self.rest).map(|x| x.fold(f)),
        }
    }
}
impl<F> Fold for pat::Tuple
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        pat::Tuple {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            parenth: self.parenth,
            pats: FoldHelper::lift(self.pats, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for pat::TupleStruct
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        pat::TupleStructuct {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            qself: (self.qself).map(|x| x.fold(f)),
            path: self.path.fold(f),
            parenth: self.parenth,
            elems: FoldHelper::lift(self.pats, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for pat::Type
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        pat::Type {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            pat: Box::new(*self.pat.fold(f)),
            colon: self.colon,
            typ: Box::new(*self.typ.fold(f)),
        }
    }
}
impl<F> Fold for pat::Wild
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        pat::Wild {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            underscore: self.underscore,
        }
    }
}
impl<F> Fold for Path
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        Path {
            colon: self.colon,
            segs: FoldHelper::lift(self.segs, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for path::Args
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        use path::Args::*;
        match self {
            None => Args::None,
            Angle(x) => Angle(x.fold(f)),
            Parenth(x) => arenthed(x.fold(f)),
        }
    }
}
impl<F> Fold for Segment
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        Segment {
            ident: self.ident.fold(f),
            args: self.args.fold(f),
        }
    }
}
impl<F> Fold for gen::Where::Life
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::Where::Life {
            life: self.life.fold(f),
            colon: self.colon,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for gen::Where::Type
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::Where::Type {
            lifes: (self.lifes).map(|x| x.fold(f)),
            bounded: self.bounded.fold(f),
            colon: self.colon,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for path::QSelf
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        path::QSelf {
            lt: self.lt,
            typ: Box::new(*self.typ.fold(f)),
            pos: self.pos,
            as_: self.as_,
            gt: self.gt,
        }
    }
}
impl<F> Fold for typ::Ret
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            typ::Ret::Default => typ::Ret::Default,
            typ::Ret::Type(x, y) => typ::Ret::Type(x, Box::new(*y.fold(f))),
        }
    }
}
impl<F> Fold for pm2::Span
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        node
    }
}
impl<F> Fold for StaticMut
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            StaticMut::Mut(x) => StaticMut::Mut(x),
            StaticMut::None => StaticMut::None,
        }
    }
}
impl<F> Fold for stmt::Stmt
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            stmt::Stmt::stmt::Local(x) => stmt::Stmt::stmt::Local(x.fold(f)),
            stmt::Stmt::Item(x) => stmt::Stmt::Item(x.fold(f)),
            stmt::Stmt::Expr(x, y) => stmt::Stmt::Expr(x.fold(f), y),
            stmt::Stmt::Mac(x) => stmt::Stmt::Mac(x.fold(f)),
        }
    }
}
impl<F> Fold for stmt::Mac
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        stmt::Mac {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            mac: self.mac.fold(f),
            semi: self.semi,
        }
    }
}
impl<F> Fold for gen::bound::Trait
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::bound::Trait {
            parenth: self.parenth,
            modif: self.modif.fold(f),
            lifes: (self.lifes).map(|x| x.fold(f)),
            path: self.path.fold(f),
        }
    }
}
impl<F> Fold for gen::bound::Modifier
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            gen::bound::Modifier::None => gen::bound::Modifier::None,
            gen::bound::Modifier::Maybe(x) => gen::bound::Modifier::Maybe(x),
        }
    }
}
impl<F> Fold for typ::Type
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            typ::Type::Array(x) => typ::Type::Array(x.fold(f)),
            typ::Type::Fn(x) => typ::Type::Fn(x.fold(f)),
            typ::Type::Group(x) => typ::Type::Group(x.fold(f)),
            typ::Type::Impl(x) => typ::Type::Impl(x.fold(f)),
            typ::Type::Infer(x) => typ::Type::Infer(x.fold(f)),
            typ::Type::Mac(x) => typ::Type::Mac(x.fold(f)),
            typ::Type::Never(x) => typ::Type::Never(x.fold(f)),
            typ::Type::Parenth(x) => typ::Type::Parenth(x.fold(f)),
            typ::Type::Path(x) => typ::Type::Path(x.fold(f)),
            typ::Type::Ptr(x) => typ::Type::Ptr(x.fold(f)),
            typ::Type::Ref(x) => typ::Type::Ref(x.fold(f)),
            typ::Type::Slice(x) => typ::Type::Slice(x.fold(f)),
            typ::Type::Trait(x) => typ::Type::Trait(x.fold(f)),
            typ::Type::Tuple(x) => typ::Type::Tuple(x.fold(f)),
            typ::Type::Stream(x) => typ::Type::Stream(x),
        }
    }
}
impl<F> Fold for typ::Array
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::Array {
            bracket: self.bracket,
            elem: Box::new(*self.elem.fold(f)),
            semi: self.semi,
            len: self.len.fold(f),
        }
    }
}
impl<F> Fold for typ::Fn
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::Fn {
            lifes: (self.lifes).map(|x| x.fold(f)),
            unsafe_: self.unsafe_,
            abi: (self.abi).map(|x| x.fold(f)),
            fn_: self.fn_,
            parenth: self.parenth,
            args: FoldHelper::lift(self.args, |x| x.fold(f)),
            vari: (self.vari).map(|x| x.fold(f)),
            ret: self.ret.fold(f),
        }
    }
}
impl<F> Fold for typ::Group
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::Group {
            group: self.group,
            elem: Box::new(*self.elem.fold(f)),
        }
    }
}
impl<F> Fold for typ::Impl
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::Impl {
            impl_: self.impl_,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for typ::Infer
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::Infer {
            underscore: self.underscore,
        }
    }
}
impl<F> Fold for typ::Mac
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::Mac { mac: self.mac.fold(f) }
    }
}
impl<F> Fold for typ::Never
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::Never { bang: self.bang }
    }
}
impl<F> Fold for gen::param::Type
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::param::Type {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            ident: self.ident.fold(f),
            colon: self.colon,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
            eq: self.eq,
            default: (self.default).map(|x| x.fold(f)),
        }
    }
}
impl<F> Fold for gen::bound::Type
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            gen::bound::Type::Trait(x) => gen::bound::Type::Trait(x.fold(f)),
            gen::bound::Type::Life(x) => gen::bound::Type::Life(x.fold(f)),
            gen::bound::Type::Verbatim(x) => gen::bound::Type::Verbatim(x),
        }
    }
}
impl<F> Fold for typ::Parenth
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::Parenth {
            parenth: self.parenth,
            elem: Box::new(*self.elem.fold(f)),
        }
    }
}
impl<F> Fold for typ::Path
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::Path {
            qself: (self.qself).map(|x| x.fold(f)),
            path: self.path.fold(f),
        }
    }
}
impl<F> Fold for typ::Ptr
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::Ptr {
            star: self.star,
            const_: self.const_,
            mut_: self.mut_,
            elem: Box::new(*self.elem.fold(f)),
        }
    }
}
impl<F> Fold for typ::Ref
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::Ref {
            and: self.and,
            life: (self.life).map(|x| x.fold(f)),
            mut_: self.mut_,
            elem: Box::new(*self.elem.fold(f)),
        }
    }
}
impl<F> Fold for typ::Slice
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::Slice {
            bracket: self.bracket,
            elem: Box::new(*self.elem.fold(f)),
        }
    }
}
impl<F> Fold for typ::Trait
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::Trait {
            dyn_: self.dyn_,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for typ::Tuple
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        typ::Tuple {
            parenth: self.parenth,
            elems: FoldHelper::lift(self.elems, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for UnOp
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            UnOp::Deref(x) => UnOp::Deref(x),
            UnOp::Not(x) => UnOp::Not(x),
            UnOp::Neg(x) => UnOp::Neg(x),
        }
    }
}
impl<F> Fold for gen::Where
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::Where {
            where_: self.where_,
            preds: FoldHelper::lift(self.preds, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for gen::Where::Pred
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            gen::Where::Pred::Life(x) => gen::Where::Pred::Life(x.fold(f)),
            gen::Where::Pred::Type(x) => gen::Where::Pred::Type(x.fold(f)),
        }
    }
}
