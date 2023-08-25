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
