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
impl<F> Fold for Data
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            Data::Struct(x) => Data::Struct(x.fold(f)),
            Data::Enum(x) => Data::Enum(x.fold(f)),
            Data::Union(x) => Data::Union(x.fold(f)),
        }
    }
}
impl<F> Fold for data::Enum
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        data::Enum {
            enum_: self.enum_,
            brace: self.brace,
            variants: FoldHelper::lift(self.variants, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for data::Struct
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        data::Struct {
            struct_: self.struct_,
            fields: self.fields.fold(f),
            semi: self.semi,
        }
    }
}
impl<F> Fold for data::Union
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        data::Union {
            union_: self.union_,
            fields: self.fields.fold(f),
        }
    }
}
impl<F> Fold for Input
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        Input {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            ident: self.ident.fold(f),
            gens: self.gens.fold(f),
            data: self.data.fold(f),
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
impl<F> Fold for expr::Array
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Array {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            bracket: self.bracket,
            elems: FoldHelper::lift(self.elems, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for expr::Assign
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Assign {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            left: Box::new(*self.left.fold(f)),
            eq: self.eq,
            right: Box::new(*self.right.fold(f)),
        }
    }
}
impl<F> Fold for expr::Async
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Async {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            async_: self.async_,
            move_: self.move_,
            block: self.block.fold(f),
        }
    }
}
impl<F> Fold for expr::Await
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Await {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            expr: Box::new(*self.expr.fold(f)),
            dot: self.dot,
            await_: self.await_,
        }
    }
}
impl<F> Fold for expr::Binary
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Binary {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            left: Box::new(*self.left.fold(f)),
            op: self.op.fold(f),
            right: Box::new(*self.right.fold(f)),
        }
    }
}
impl<F> Fold for expr::Block
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Block {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            label: (self.label).map(|x| x.fold(f)),
            block: self.block.fold(f),
        }
    }
}
impl<F> Fold for expr::Break
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Break {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            break_: self.break_,
            life: (self.life).map(|x| x.fold(f)),
            val: (self.val).map(|x| Box::new(*x.fold(f))),
        }
    }
}
impl<F> Fold for expr::Call
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Call {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            func: Box::new(*self.func.fold(f)),
            parenth: self.parenth,
            args: FoldHelper::lift(self.args, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for expr::Cast
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Cast {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            expr: Box::new(*self.expr.fold(f)),
            as_: self.as_,
            typ: Box::new(*self.typ.fold(f)),
        }
    }
}
impl<F> Fold for expr::Closure
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Closure {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            lifes: (self.lifes).map(|x| x.fold(f)),
            const_: self.const_,
            static_: self.static_,
            async_: self.async_,
            move_: self.move_,
            or1: self.or1,
            ins: FoldHelper::lift(self.inputs, |x| x.fold(f)),
            or2: self.or2,
            ret: self.ret.fold(f),
            body: Box::new(*self.body.fold(f)),
        }
    }
}
impl<F> Fold for expr::Const
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Const {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            const_: self.const_,
            block: self.block.fold(f),
        }
    }
}
impl<F> Fold for expr::Continue
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Continue {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            continue_: self.continue_,
            life: (self.life).map(|x| x.fold(f)),
        }
    }
}
impl<F> Fold for expr::Field
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Field {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            expr: Box::new(*self.expr.fold(f)),
            dot: self.dot,
            memb: self.memb.fold(f),
        }
    }
}
impl<F> Fold for expr::For
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::For {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            label: (self.label).map(|x| x.fold(f)),
            for_: self.for_,
            pat: Box::new(*self.pat.fold(f)),
            in_: self.in_,
            expr: Box::new(*self.expr.fold(f)),
            body: self.body.fold(f),
        }
    }
}
impl<F> Fold for expr::Group
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Group {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            group: self.group,
            expr: Box::new(*self.expr.fold(f)),
        }
    }
}
impl<F> Fold for expr::If
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::If {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            if_: self.if_,
            cond: Box::new(*self.cond.fold(f)),
            then_: self.then_.fold(f),
            else_: (self.else_).map(|x| ((x).0, Box::new(*x.fold(f)))),
        }
    }
}
impl<F> Fold for expr::Index
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Index {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            expr: Box::new(*self.expr.fold(f)),
            bracket: self.bracket,
            idx: Box::new(*self.idx.fold(f)),
        }
    }
}
impl<F> Fold for expr::Infer
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Infer {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            underscore: self.underscore,
        }
    }
}
impl<F> Fold for expr::Let
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Let {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            let_: self.let_,
            pat: Box::new(*self.pat.fold(f)),
            eq: self.eq,
            expr: Box::new(*self.expr.fold(f)),
        }
    }
}
impl<F> Fold for expr::Lit
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Lit {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            lit: self.lit.fold(f),
        }
    }
}
impl<F> Fold for expr::Loop
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Loop {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            label: (self.label).map(|x| x.fold(f)),
            loop_: self.loop_,
            body: self.body.fold(f),
        }
    }
}
impl<F> Fold for expr::Mac
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Mac {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            mac: self.mac.fold(f),
        }
    }
}
impl<F> Fold for expr::Match
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Match {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            match_: self.match_,
            expr: Box::new(*self.expr.fold(f)),
            brace: self.brace,
            arms: FoldHelper::lift(self.arms, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for expr::Method
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Method {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            expr: Box::new(*self.expr.fold(f)),
            dot: self.dot,
            method: self.method.fold(f),
            turbofish: (self.turbofish).map(|x| x.fold(f)),
            parenth: self.parenth,
            args: FoldHelper::lift(self.args, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for expr::Parenth
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Parenth {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            parenth: self.parenth,
            expr: Box::new(*self.expr.fold(f)),
        }
    }
}
impl<F> Fold for expr::Path
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Path {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            qself: (self.qself).map(|x| x.fold(f)),
            path: self.path.fold(f),
        }
    }
}
impl<F> Fold for expr::Range
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Range {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            beg: (self.beg).map(|x| Box::new(*x.fold(f))),
            limits: self.limits.fold(f),
            end: (self.end).map(|x| Box::new(*x.fold(f))),
        }
    }
}
impl<F> Fold for expr::Ref
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Ref {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            and: self.and,
            mut_: self.mut_,
            expr: Box::new(*self.expr.fold(f)),
        }
    }
}
impl<F> Fold for expr::Repeat
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Repeat {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            bracket: self.bracket,
            expr: Box::new(*self.expr.fold(f)),
            semi: self.semi,
            len: Box::new(*self.len.fold(f)),
        }
    }
}
impl<F> Fold for expr::Return
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Return {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            return_: self.return_,
            expr: (self.expr).map(|x| Box::new(*x.fold(f))),
        }
    }
}
impl<F> Fold for expr::Struct
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Struct {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            qself: (self.qself).map(|x| x.fold(f)),
            path: self.path.fold(f),
            brace: self.brace,
            fields: FoldHelper::lift(self.fields, |x| x.fold(f)),
            dot2: self.dot2,
            rest: (self.rest).map(|x| Box::new(*x.fold(f))),
        }
    }
}
impl<F> Fold for expr::Try
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Try {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            expr: Box::new(*self.expr.fold(f)),
            question: self.question,
        }
    }
}
impl<F> Fold for expr::TryBlock
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::TryBlock {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            try_: self.try_,
            block: self.block.fold(f),
        }
    }
}
impl<F> Fold for expr::Tuple
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Tuple {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            parenth: self.parenth,
            elems: FoldHelper::lift(self.elems, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for expr::Unary
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Unary {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            op: self.op.fold(f),
            expr: Box::new(*self.expr.fold(f)),
        }
    }
}
impl<F> Fold for expr::Unsafe
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Unsafe {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            unsafe_: self.unsafe_,
            block: self.block.fold(f),
        }
    }
}
impl<F> Fold for expr::While
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::While {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            label: (self.label).map(|x| x.fold(f)),
            while_: self.while_,
            cond: Box::new(*self.cond.fold(f)),
            block: self.block.fold(f),
        }
    }
}
impl<F> Fold for expr::Yield
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        expr::Yield {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            yield_: self.yield_,
            expr: (self.expr).map(|x| Box::new(*x.fold(f))),
        }
    }
}
impl<F> Fold for data::Field
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        data::Field {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            mut_: self.mut_.fold(f),
            ident: (self.ident).map(|x| x.fold(f)),
            colon: self.colon,
            typ: self.typ.fold(f),
        }
    }
}
impl<F> Fold for data::Mut
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            data::Mut::None => data::Mut::None,
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
impl<F> Fold for data::Fields
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            data::Fields::Named(x) => data::Fields::Named(x.fold(f)),
            data::Fields::Unnamed(x) => data::Fields::Unnamed(x.fold(f)),
            data::Fields::Unit => data::Fields::Unit,
        }
    }
}
impl<F> Fold for data::Named
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        data::Named {
            brace: self.brace,
            fields: FoldHelper::lift(self.fields, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for data::Unnamed
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        data::Unnamed {
            parenth: self.parenth,
            fields: FoldHelper::lift(self.fields, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for item::File
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::File {
            shebang: self.shebang,
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            items: FoldHelper::lift(self.items, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for item::FnArg
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            item::FnArg::Receiver(x) => item::FnArg::Receiver(x.fold(f)),
            item::FnArg::Type(x) => item::FnArg::Type(x.fold(f)),
        }
    }
}
impl<F> Fold for item::foreign::Item
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            item::foreign::Item::Fn(x) => item::foreign::Item::Fn(x.fold(f)),
            item::foreign::Item::Static(x) => item::foreign::Item::Static(x.fold(f)),
            item::foreign::Item::Type(x) => item::foreign::Item::Type(x.fold(f)),
            item::foreign::Item::Macro(x) => item::foreign::Item::Macro(x.fold(f)),
            item::foreign::Item::Verbatim(x) => item::foreign::Item::Verbatim(x),
        }
    }
}
impl<F> Fold for item::foreign::Fn
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::foreign::Fn {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            sig: self.sig.fold(f),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::foreign::Mac
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::foreign::Mac {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            mac: self.mac.fold(f),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::foreign::Static
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::foreign::Static {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            static_: self.static_,
            mut_: self.mut_.fold(f),
            ident: self.ident.fold(f),
            colon: self.colon,
            typ: Box::new(*self.typ.fold(f)),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::foreign::Type
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::foreign::Type {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            type_: self.type_,
            ident: self.ident.fold(f),
            gens: self.gens.fold(f),
            semi: self.semi,
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
impl<F> Fold for item::impl_::Item
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            item::impl_::Item::Const(x) => item::impl_::Item::Const(x.fold(f)),
            item::impl_::Item::Fn(x) => item::impl_::Item::Fn(x.fold(f)),
            item::impl_::Item::Type(x) => item::impl_::Item::Type(x.fold(f)),
            item::impl_::Item::Macro(x) => item::impl_::Item::Macro(x.fold(f)),
            item::impl_::Item::Verbatim(x) => item::impl_::Item::Verbatim(x),
        }
    }
}
impl<F> Fold for item::impl_::Const
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::impl_::Const {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            default_: self.default_,
            const_: self.const_,
            ident: self.ident.fold(f),
            gens: self.gens.fold(f),
            colon: self.colon,
            typ: self.typ.fold(f),
            eq: self.eq,
            expr: self.expr.fold(f),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::impl_::Fn
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::impl_::Fn {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            default_: self.default_,
            sig: self.sig.fold(f),
            block: self.block.fold(f),
        }
    }
}
impl<F> Fold for item::impl_::Mac
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::impl_::Mac {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            mac: self.mac.fold(f),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::impl_::Type
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::impl_::Type {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            default_: self.default_,
            type_: self.type_,
            ident: self.ident.fold(f),
            gens: self.gens.fold(f),
            eq: self.eq,
            typ: self.typ.fold(f),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::impl_::Restriction
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {}
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
impl<F> Fold for Item
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            Item::Const(x) => Item::Const(x.fold(f)),
            Item::Enum(x) => Item::Enum(x.fold(f)),
            Item::Extern(x) => Item::Extern(x.fold(f)),
            Item::Fn(x) => Item::Fn(x.fold(f)),
            Item::Foreign(x) => Item::Foreign(x.fold(f)),
            Item::Impl(x) => Item::Impl(x.fold(f)),
            Item::Macro(x) => Item::Macro(x.fold(f)),
            Item::Mod(x) => Item::Mod(x.fold(f)),
            Item::Static(x) => Item::Static(x.fold(f)),
            Item::Struct(x) => Item::Struct(x.fold(f)),
            Item::Trait(x) => Item::Trait(x.fold(f)),
            Item::Alias(x) => Item::Alias(x.fold(f)),
            Item::Type(x) => Item::Type(x.fold(f)),
            Item::Union(x) => Item::Union(x.fold(f)),
            Item::Use(x) => Item::Use(x.fold(f)),
            Item::Stream(x) => Item::Stream(x),
        }
    }
}
impl<F> Fold for item::Const
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Const {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            const_: self.const_,
            ident: self.ident.fold(f),
            gens: self.gens.fold(f),
            colon: self.colon,
            typ: Box::new(*self.typ.fold(f)),
            eq: self.eq,
            expr: Box::new(*self.expr.fold(f)),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::Enum
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Enum {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            enum_: self.enum_,
            ident: self.ident.fold(f),
            gens: self.gens.fold(f),
            brace: self.brace,
            variants: FoldHelper::lift(self.variants, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for item::Extern
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Extern {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            extern_: self.extern_,
            crate_: self.crate_,
            ident: self.ident.fold(f),
            rename: (self.rename).map(|x| ((x).0, (x).1.fold(f))),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::Fn
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Fn {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            sig: self.sig.fold(f),
            block: Box::new(*self.block.fold(f)),
        }
    }
}
impl<F> Fold for item::Foreign
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Foreign {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            unsafe_: self.unsafe_,
            abi: self.abi.fold(f),
            brace: self.brace,
            items: FoldHelper::lift(self.items, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for item::Impl
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Impl {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            default: self.default,
            unsafe_: self.unsafe_,
            impl_: self.impl_,
            gens: self.gens.fold(f),
            trait_: (self.trait_).map(|x| ((x).0, (x).1.fold(f), (x).2)),
            typ: Box::new(*self.typ.fold(f)),
            brace: self.brace,
            items: FoldHelper::lift(self.items, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for item::Mac
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Mac {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            ident: (self.ident).map(|x| x.fold(f)),
            mac: self.mac.fold(f),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::Mod
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Mod {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            unsafe_: self.unsafe_,
            mod_: self.mod_,
            ident: self.ident.fold(f),
            items: (self.items).map(|x| ((x).0, FoldHelper::lift((x).1, |x| x.fold(f)))),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::Static
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Static {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            static_: self.static_,
            mut_: self.mut_.fold(f),
            ident: self.ident.fold(f),
            colon: self.colon,
            typ: Box::new(*self.typ.fold(f)),
            eq: self.eq,
            expr: Box::new(*self.expr.fold(f)),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::Struct
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Struct {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            struct_: self.struct_,
            ident: self.ident.fold(f),
            gens: self.gens.fold(f),
            fields: self.fields.fold(f),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::Trait
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Trait {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            unsafe_: self.unsafe_,
            auto: self.auto,
            restriction: (self.restriction).map(|x| x.fold(f)),
            trait_: self.trait_,
            ident: self.ident.fold(f),
            gens: self.gens.fold(f),
            colon: self.colon,
            supers: FoldHelper::lift(self.supers, |x| x.fold(f)),
            brace: self.brace,
            items: FoldHelper::lift(self.items, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for item::Alias
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Alias {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            trait_: self.trait_,
            ident: self.ident.fold(f),
            gens: self.gens.fold(f),
            eq: self.eq,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::Type
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Type {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            type_: self.type_,
            ident: self.ident.fold(f),
            gens: self.gens.fold(f),
            eq: self.eq,
            typ: Box::new(*self.typ.fold(f)),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::Union
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Union {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            union_: self.union_,
            ident: self.ident.fold(f),
            gens: self.gens.fold(f),
            fields: self.fields.fold(f),
        }
    }
}
impl<F> Fold for item::Use
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Use {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            use_: self.use_,
            colon: self.colon,
            tree: self.tree.fold(f),
            semi: self.semi,
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
impl<F> Fold for expr::Limits
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            expr::Limits::HalfOpen(x) => expr::Limits::HalfOpen(x),
            expr::Limits::Closed(x) => expr::Limits::Closed(x),
        }
    }
}
impl<F> Fold for item::Receiver
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Receiver {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            ref_: (self.ref_).map(|x| ((x).0, ((x).1).map(|x| x.fold(f)))),
            mut_: self.mut_,
            self_: self.self_,
            colon: self.colon,
            typ: Box::new(*self.typ.fold(f)),
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
impl<F> Fold for item::Sig
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Sig {
            const_: self.const_,
            async_: self.async_,
            unsafe_: self.unsafe_,
            abi: (self.abi).map(|x| x.fold(f)),
            fn_: self.fn_,
            ident: self.ident.fold(f),
            gens: self.gens.fold(f),
            parenth: self.parenth,
            args: FoldHelper::lift(self.args, |x| x.fold(f)),
            vari: (self.vari).map(|x| x.fold(f)),
            ret: self.ret.fold(f),
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
impl<F> Fold for item::trait_::Item
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            item::trait_::Item::Const(x) => item::trait_::Item::Const(x.fold(f)),
            item::trait_::Item::Fn(x) => item::trait_::Item::Fn(x.fold(f)),
            item::trait_::Item::Type(x) => item::trait_::Item::Type(x.fold(f)),
            item::trait_::Item::Macro(x) => item::trait_::Item::Macro(x.fold(f)),
            item::trait_::Item::Verbatim(x) => item::trait_::Item::Verbatim(x),
        }
    }
}
impl<F> Fold for item::trait_::Const
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::trait_::Const {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            const_: self.const_,
            ident: self.ident.fold(f),
            gens: self.gens.fold(f),
            colon: self.colon,
            typ: self.typ.fold(f),
            default: (self.default).map(|x| ((x).0, (x).1.fold(f))),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::trait_::Fn
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::trait_::Fn {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            sig: self.sig.fold(f),
            default: (self.default).map(|x| x.fold(f)),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::trait_::Mac
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::trait_::Mac {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            mac: self.mac.fold(f),
            semi: self.semi,
        }
    }
}
impl<F> Fold for item::trait_::Type
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::trait_::Type {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            type_: self.type_,
            ident: self.ident.fold(f),
            gens: self.gens.fold(f),
            colon: self.colon,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
            default: (self.default).map(|x| ((x).0, (x).1.fold(f))),
            semi: self.semi,
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
impl<F> Fold for item::use_::Glob
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::use_::Glob { star: self.star }
    }
}
impl<F> Fold for item::use_::Group
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::use_::Group {
            brace: self.brace,
            trees: FoldHelper::lift(self.trees, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for item::use_::Name
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::use_::Name {
            ident: self.ident.fold(f),
        }
    }
}
impl<F> Fold for item::use_::Path
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::use_::Path {
            ident: self.ident.fold(f),
            colon2: self.colon2,
            tree: Box::new(*self.tree.fold(f)),
        }
    }
}
impl<F> Fold for item::use_::Rename
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::use_::Rename {
            ident: self.ident.fold(f),
            as_: self.as_,
            rename: self.rename.fold(f),
        }
    }
}
impl<F> Fold for item::use_::Tree
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            item::use_::Tree::Path(x) => item::use_::Tree::Path(x.fold(f)),
            item::use_::Tree::Name(x) => item::use_::Tree::Name(x.fold(f)),
            item::use_::Tree::Rename(x) => item::use_::Tree::Rename(x.fold(f)),
            item::use_::Tree::Glob(x) => item::use_::Tree::Glob(x.fold(f)),
            item::use_::Tree::Group(x) => item::use_::Tree::Group(x.fold(f)),
        }
    }
}
impl<F> Fold for item::Variadic
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        item::Variadic {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            pat: (self.pat).map(|x| (Box::new(*(x).0.fold(f), (x).1))),
            dots: self.dots,
            comma: self.comma,
        }
    }
}
impl<F> Fold for data::Variant
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        data::Variant {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            ident: self.ident.fold(f),
            fields: self.fields.fold(f),
            discrim: (self.discrim).map(|x| ((x).0, (x).1.fold(f))),
        }
    }
}
impl<F> Fold for data::Restricted
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        data::Restricted {
            pub_: self.pub_,
            parenth: self.parenth,
            in_: self.in_,
            path: Box::new(*self.path.fold(f)),
        }
    }
}
impl<F> Fold for data::Visibility
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            data::Visibility::Public(x) => data::Visibility::Public(x),
            data::Visibility::Restricted(x) => data::Visibility::Restricted(x.fold(f)),
            data::Visibility::Inherited => data::Visibility::Inherited,
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
