use crate::StreamHelper;
use crate::*;
use std::hash::{Hash, Hasher};
impl<H> Hash for path::Angled
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.colon2.hash(h);
        self.args.hash(h);
    }
}
impl<H> Hash for expr::Arm
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.guard.hash(h);
        self.body.hash(h);
        self.comma.hash(h);
    }
}
impl<H> Hash for path::AssocConst
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.ident.hash(h);
        self.args.hash(h);
        self.val.hash(h);
    }
}
impl<H> Hash for path::AssocType
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.ident.hash(h);
        self.args.hash(h);
        self.typ.hash(h);
    }
}
impl<H> Hash for attr::Style
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use attr::Style::*;
        match self {
            Outer => {
                h.write_u8(0u8);
            },
            Inner(_) => {
                h.write_u8(1u8);
            },
        }
    }
}
impl<H> Hash for attr::Attr
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.style.hash(h);
        self.meta.hash(h);
    }
}
impl<H> Hash for expr::BinOp
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use expr::BinOp::*;
        match self {
            Add(_) => {
                h.write_u8(0u8);
            },
            Sub(_) => {
                h.write_u8(1u8);
            },
            Mul(_) => {
                h.write_u8(2u8);
            },
            Div(_) => {
                h.write_u8(3u8);
            },
            Rem(_) => {
                h.write_u8(4u8);
            },
            And(_) => {
                h.write_u8(5u8);
            },
            Or(_) => {
                h.write_u8(6u8);
            },
            BitXor(_) => {
                h.write_u8(7u8);
            },
            BitAnd(_) => {
                h.write_u8(8u8);
            },
            BitOr(_) => {
                h.write_u8(9u8);
            },
            Shl(_) => {
                h.write_u8(10u8);
            },
            Shr(_) => {
                h.write_u8(11u8);
            },
            Eq(_) => {
                h.write_u8(12u8);
            },
            Lt(_) => {
                h.write_u8(13u8);
            },
            Le(_) => {
                h.write_u8(14u8);
            },
            Ne(_) => {
                h.write_u8(15u8);
            },
            Ge(_) => {
                h.write_u8(16u8);
            },
            Gt(_) => {
                h.write_u8(17u8);
            },
            AddAssign(_) => {
                h.write_u8(18u8);
            },
            SubAssign(_) => {
                h.write_u8(19u8);
            },
            MulAssign(_) => {
                h.write_u8(20u8);
            },
            DivAssign(_) => {
                h.write_u8(21u8);
            },
            RemAssign(_) => {
                h.write_u8(22u8);
            },
            BitXorAssign(_) => {
                h.write_u8(23u8);
            },
            BitAndAssign(_) => {
                h.write_u8(24u8);
            },
            BitOrAssign(_) => {
                h.write_u8(25u8);
            },
            ShlAssign(_) => {
                h.write_u8(26u8);
            },
            ShrAssign(_) => {
                h.write_u8(27u8);
            },
        }
    }
}
impl<H> Hash for stmt::Block
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.stmts.hash(h);
    }
}
impl<H> Hash for gen::bound::Lifes
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.lifes.hash(h);
    }
}
impl<H> Hash for gen::param::Const
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.typ.hash(h);
        self.eq.hash(h);
        self.default.hash(h);
    }
}
impl<H> Hash for path::Constraint
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.ident.hash(h);
        self.args.hash(h);
        self.bounds.hash(h);
    }
}
impl<H> Hash for data::Data
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use data::Data::*;
        match self {
            Struct(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Enum(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Union(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
        }
    }
}
impl<H> Hash for data::Enum
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.variants.hash(h);
    }
}
impl<H> Hash for data::Struct
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.fields.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for data::Union
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.fields.hash(h);
    }
}
impl<H> Hash for Input
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.data.hash(h);
    }
}
impl<H> Hash for expr::Expr
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use expr::Expr::*;
        match self {
            Array(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Assign(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Async(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Await(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Binary(x) => {
                h.write_u8(4u8);
                x.hash(h);
            },
            Block(x) => {
                h.write_u8(5u8);
                x.hash(h);
            },
            Break(x) => {
                h.write_u8(6u8);
                x.hash(h);
            },
            Call(x) => {
                h.write_u8(7u8);
                x.hash(h);
            },
            Cast(x) => {
                h.write_u8(8u8);
                x.hash(h);
            },
            Closure(x) => {
                h.write_u8(9u8);
                x.hash(h);
            },
            Const(x) => {
                h.write_u8(10u8);
                x.hash(h);
            },
            Continue(x) => {
                h.write_u8(11u8);
                x.hash(h);
            },
            Field(x) => {
                h.write_u8(12u8);
                x.hash(h);
            },
            ForLoop(x) => {
                h.write_u8(13u8);
                x.hash(h);
            },
            Group(x) => {
                h.write_u8(14u8);
                x.hash(h);
            },
            If(x) => {
                h.write_u8(15u8);
                x.hash(h);
            },
            Index(x) => {
                h.write_u8(16u8);
                x.hash(h);
            },
            Infer(x) => {
                h.write_u8(17u8);
                x.hash(h);
            },
            Let(x) => {
                h.write_u8(18u8);
                x.hash(h);
            },
            Lit(x) => {
                h.write_u8(19u8);
                x.hash(h);
            },
            Loop(x) => {
                h.write_u8(20u8);
                x.hash(h);
            },
            Mac(x) => {
                h.write_u8(21u8);
                x.hash(h);
            },
            Match(x) => {
                h.write_u8(22u8);
                x.hash(h);
            },
            MethodCall(x) => {
                h.write_u8(23u8);
                x.hash(h);
            },
            Parenth(x) => {
                h.write_u8(24u8);
                x.hash(h);
            },
            Path(x) => {
                h.write_u8(25u8);
                x.hash(h);
            },
            Range(x) => {
                h.write_u8(26u8);
                x.hash(h);
            },
            Ref(x) => {
                h.write_u8(27u8);
                x.hash(h);
            },
            Repeat(x) => {
                h.write_u8(28u8);
                x.hash(h);
            },
            Return(x) => {
                h.write_u8(29u8);
                x.hash(h);
            },
            Struct(x) => {
                h.write_u8(30u8);
                x.hash(h);
            },
            Try(x) => {
                h.write_u8(31u8);
                x.hash(h);
            },
            TryBlock(x) => {
                h.write_u8(32u8);
                x.hash(h);
            },
            Tuple(x) => {
                h.write_u8(33u8);
                x.hash(h);
            },
            Unary(x) => {
                h.write_u8(34u8);
                x.hash(h);
            },
            Unsafe(x) => {
                h.write_u8(35u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(36u8);
                StreamHelper(x).hash(h);
            },
            While(x) => {
                h.write_u8(37u8);
                x.hash(h);
            },
            Yield(x) => {
                h.write_u8(38u8);
                x.hash(h);
            },
        }
    }
}
impl<H> Hash for expr::Array
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.elems.hash(h);
    }
}
impl<H> Hash for expr::Assign
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.left.hash(h);
        self.right.hash(h);
    }
}
impl<H> Hash for expr::Async
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.move_.hash(h);
        self.block.hash(h);
    }
}
impl<H> Hash for expr::Await
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for expr::Binary
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.left.hash(h);
        self.op.hash(h);
        self.right.hash(h);
    }
}
impl<H> Hash for expr::Block
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.label.hash(h);
        self.block.hash(h);
    }
}
impl<H> Hash for expr::Break
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.life.hash(h);
        self.val.hash(h);
    }
}
impl<H> Hash for expr::Call
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.func.hash(h);
        self.args.hash(h);
    }
}
impl<H> Hash for expr::Cast
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.expr.hash(h);
        self.typ.hash(h);
    }
}
impl<H> Hash for expr::Closure
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.lifes.hash(h);
        self.const_.hash(h);
        self.static_.hash(h);
        self.async_.hash(h);
        self.move_.hash(h);
        self.ins.hash(h);
        self.ret.hash(h);
        self.body.hash(h);
    }
}
impl<H> Hash for expr::Const
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.block.hash(h);
    }
}
impl<H> Hash for expr::Continue
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.life.hash(h);
    }
}
impl<H> Hash for expr::Field
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.expr.hash(h);
        self.memb.hash(h);
    }
}
impl<H> Hash for expr::ForLoop
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.label.hash(h);
        self.pat.hash(h);
        self.expr.hash(h);
        self.body.hash(h);
    }
}
impl<H> Hash for expr::Group
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for expr::If
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.cond.hash(h);
        self.then_.hash(h);
        self.else_.hash(h);
    }
}
impl<H> Hash for expr::Index
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.expr.hash(h);
        self.idx.hash(h);
    }
}
impl<H> Hash for expr::Infer
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
    }
}
impl<H> Hash for expr::Let
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for expr::Lit
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.lit.hash(h);
    }
}
impl<H> Hash for expr::Loop
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.label.hash(h);
        self.body.hash(h);
    }
}
impl<H> Hash for expr::Mac
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.mac.hash(h);
    }
}
impl<H> Hash for expr::Match
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.expr.hash(h);
        self.arms.hash(h);
    }
}
impl<H> Hash for expr::MethodCall
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.expr.hash(h);
        self.method.hash(h);
        self.turbofish.hash(h);
        self.args.hash(h);
    }
}
impl<H> Hash for expr::Parenth
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for expr::Path
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.qself.hash(h);
        self.path.hash(h);
    }
}
impl<H> Hash for expr::Range
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.beg.hash(h);
        self.limits.hash(h);
        self.end.hash(h);
    }
}
impl<H> Hash for expr::Ref
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.mut_.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for expr::Repeat
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.expr.hash(h);
        self.len.hash(h);
    }
}
impl<H> Hash for expr::Return
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for expr::Struct
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.qself.hash(h);
        self.path.hash(h);
        self.fields.hash(h);
        self.dot2.hash(h);
        self.rest.hash(h);
    }
}
impl<H> Hash for expr::Try
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for expr::TryBlock
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.block.hash(h);
    }
}
impl<H> Hash for expr::Tuple
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.elems.hash(h);
    }
}
impl<H> Hash for expr::Unary
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.op.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for expr::Unsafe
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.block.hash(h);
    }
}
impl<H> Hash for expr::While
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.label.hash(h);
        self.cond.hash(h);
        self.block.hash(h);
    }
}
impl<H> Hash for expr::Yield
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for data::Field
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.mut_.hash(h);
        self.ident.hash(h);
        self.colon.hash(h);
        self.typ.hash(h);
    }
}
impl<H> Hash for data::Mut
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        match self {
            data::Mut::None => {
                h.write_u8(0u8);
            },
        }
    }
}
impl<H> Hash for pat::Field
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.memb.hash(h);
        self.colon.hash(h);
        self.pat.hash(h);
    }
}
impl<H> Hash for expr::FieldValue
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.memb.hash(h);
        self.colon.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for data::Fields
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use data::Fields::*;
        match self {
            Named(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Unnamed(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Unit => {
                h.write_u8(2u8);
            },
        }
    }
}
impl<H> Hash for data::Named
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.fields.hash(h);
    }
}
impl<H> Hash for data::Unnamed
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.fields.hash(h);
    }
}
impl<H> Hash for item::File
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.shebang.hash(h);
        self.attrs.hash(h);
        self.items.hash(h);
    }
}
impl<H> Hash for item::FnArg
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use item::FnArg::*;
        match self {
            Receiver(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
        }
    }
}
impl<H> Hash for item::foreign::Item
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use item::foreign::Item::*;
        match self {
            Fn(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Static(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Mac(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(4u8);
                StreamHelper(x).hash(h);
            },
        }
    }
}
impl<H> Hash for item::foreign::Fn
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.sig.hash(h);
    }
}
impl<H> Hash for item::foreign::Mac
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for item::foreign::Static
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.mut_.hash(h);
        self.ident.hash(h);
        self.typ.hash(h);
    }
}
impl<H> Hash for item::foreign::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
    }
}
impl<H> Hash for path::Arg
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use path::Arg::*;
        match self {
            Life(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Const(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            AssocType(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            AssocConst(x) => {
                h.write_u8(4u8);
                x.hash(h);
            },
            Constraint(x) => {
                h.write_u8(5u8);
                x.hash(h);
            },
        }
    }
}
impl<H> Hash for gen::Param
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use gen::Param::*;
        match self {
            Life(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Const(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
        }
    }
}
impl<H> Hash for gen::Gens
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.lt.hash(h);
        self.params.hash(h);
        self.gt.hash(h);
        self.where_.hash(h);
    }
}
impl<H> Hash for item::impl_::Item
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use item::impl_::Item::*;
        match self {
            Const(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Fn(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Mac(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(4u8);
                StreamHelper(x).hash(h);
            },
        }
    }
}
impl<H> Hash for item::impl_::Const
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.default.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.typ.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for item::impl_::Fn
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.default.hash(h);
        self.sig.hash(h);
        self.block.hash(h);
    }
}
impl<H> Hash for item::impl_::Mac
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for item::impl_::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.default.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.typ.hash(h);
    }
}
impl<H> Hash for item::impl_::Restriction
where
    H: Hasher,
{
    fn hash(&self, _h: &mut H) {
        match *self {}
    }
}
impl<H> Hash for item::Item
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use item::Item::*;
        match self {
            Const(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Enum(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Extern(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Fn(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Foreign(x) => {
                h.write_u8(4u8);
                x.hash(h);
            },
            Impl(x) => {
                h.write_u8(5u8);
                x.hash(h);
            },
            Mac(x) => {
                h.write_u8(6u8);
                x.hash(h);
            },
            Mod(x) => {
                h.write_u8(7u8);
                x.hash(h);
            },
            Static(x) => {
                h.write_u8(8u8);
                x.hash(h);
            },
            Struct(x) => {
                h.write_u8(9u8);
                x.hash(h);
            },
            Trait(x) => {
                h.write_u8(10u8);
                x.hash(h);
            },
            TraitAlias(x) => {
                h.write_u8(11u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(12u8);
                x.hash(h);
            },
            Union(x) => {
                h.write_u8(13u8);
                x.hash(h);
            },
            Use(x) => {
                h.write_u8(14u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(15u8);
                StreamHelper(x).hash(h);
            },
        }
    }
}
impl<H> Hash for item::Const
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.typ.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for item::Enum
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.variants.hash(h);
    }
}
impl<H> Hash for item::Extern
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.rename.hash(h);
    }
}
impl<H> Hash for item::Fn
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.sig.hash(h);
        self.block.hash(h);
    }
}
impl<H> Hash for item::Foreign
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.unsafe_.hash(h);
        self.abi.hash(h);
        self.items.hash(h);
    }
}
impl<H> Hash for item::Impl
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.default.hash(h);
        self.unsafe_.hash(h);
        self.gens.hash(h);
        self.trait_.hash(h);
        self.typ.hash(h);
        self.items.hash(h);
    }
}
impl<H> Hash for item::Mac
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for item::Mod
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.unsafe_.hash(h);
        self.ident.hash(h);
        self.items.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for item::Static
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.mut_.hash(h);
        self.ident.hash(h);
        self.typ.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for item::Struct
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.fields.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for item::Trait
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.unsafe_.hash(h);
        self.auto.hash(h);
        self.restriction.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.colon.hash(h);
        self.supers.hash(h);
        self.items.hash(h);
    }
}
impl<H> Hash for item::TraitAlias
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.bounds.hash(h);
    }
}
impl<H> Hash for item::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.typ.hash(h);
    }
}
impl<H> Hash for item::Union
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.fields.hash(h);
    }
}
impl<H> Hash for item::Use
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.colon.hash(h);
        self.tree.hash(h);
    }
}
impl<H> Hash for expr::Label
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.name.hash(h);
    }
}
impl<H> Hash for gen::param::Life
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.life.hash(h);
        self.colon.hash(h);
        self.bounds.hash(h);
    }
}
impl<H> Hash for lit::Lit
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use lit::Lit::*;
        match self {
            Str(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            ByteStr(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Byte(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Char(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Int(x) => {
                h.write_u8(4u8);
                x.hash(h);
            },
            Float(x) => {
                h.write_u8(5u8);
                x.hash(h);
            },
            Bool(x) => {
                h.write_u8(6u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(7u8);
                x.to_string().hash(h);
            },
        }
    }
}
impl<H> Hash for lit::Bool
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.val.hash(h);
    }
}
impl<H> Hash for stmt::Local
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.init.hash(h);
    }
}
impl<H> Hash for stmt::Init
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.expr.hash(h);
        self.diverge.hash(h);
    }
}
impl<H> Hash for mac::Mac
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.path.hash(h);
        self.delim.hash(h);
        StreamHelper(&self.toks).hash(h);
    }
}
impl<H> Hash for tok::Delim
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use tok::Delim::*;
        match self {
            Parenth(_) => {
                h.write_u8(0u8);
            },
            Brace(_) => {
                h.write_u8(1u8);
            },
            Bracket(_) => {
                h.write_u8(2u8);
            },
        }
    }
}
impl<H> Hash for attr::Meta
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use attr::Meta::*;
        match self {
            Path(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            List(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            NameValue(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
        }
    }
}
impl<H> Hash for attr::List
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.path.hash(h);
        self.delim.hash(h);
        StreamHelper(&self.toks).hash(h);
    }
}
impl<H> Hash for attr::NameValue
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.name.hash(h);
        self.val.hash(h);
    }
}
impl<H> Hash for path::Parenthed
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.args.hash(h);
        self.ret.hash(h);
    }
}
impl<H> Hash for pat::Pat
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use pat::Pat::*;
        match self {
            Const(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Ident(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Lit(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Mac(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Or(x) => {
                h.write_u8(4u8);
                x.hash(h);
            },
            Parenth(x) => {
                h.write_u8(5u8);
                x.hash(h);
            },
            Path(x) => {
                h.write_u8(6u8);
                x.hash(h);
            },
            Range(x) => {
                h.write_u8(7u8);
                x.hash(h);
            },
            Ref(x) => {
                h.write_u8(8u8);
                x.hash(h);
            },
            Rest(x) => {
                h.write_u8(9u8);
                x.hash(h);
            },
            Slice(x) => {
                h.write_u8(10u8);
                x.hash(h);
            },
            Struct(x) => {
                h.write_u8(11u8);
                x.hash(h);
            },
            Tuple(x) => {
                h.write_u8(12u8);
                x.hash(h);
            },
            TupleStruct(x) => {
                h.write_u8(13u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(14u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(15u8);
                StreamHelper(x).hash(h);
            },
            Wild(x) => {
                h.write_u8(16u8);
                x.hash(h);
            },
        }
    }
}
impl<H> Hash for pat::Ident
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ref_.hash(h);
        self.mut_.hash(h);
        self.ident.hash(h);
        self.sub.hash(h);
    }
}
impl<H> Hash for pat::Or
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vert.hash(h);
        self.cases.hash(h);
    }
}
impl<H> Hash for pat::Parenth
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pat.hash(h);
    }
}
impl<H> Hash for pat::Ref
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.mut_.hash(h);
        self.pat.hash(h);
    }
}
impl<H> Hash for pat::Rest
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
    }
}
impl<H> Hash for pat::Slice
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pats.hash(h);
    }
}
impl<H> Hash for pat::Struct
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.qself.hash(h);
        self.path.hash(h);
        self.fields.hash(h);
        self.rest.hash(h);
    }
}
impl<H> Hash for pat::Tuple
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pats.hash(h);
    }
}
impl<H> Hash for pat::TupleStruct
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.qself.hash(h);
        self.path.hash(h);
        self.pats.hash(h);
    }
}
impl<H> Hash for pat::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.typ.hash(h);
    }
}
impl<H> Hash for pat::Wild
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
    }
}
impl<H> Hash for Path
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.colon.hash(h);
        self.segs.hash(h);
    }
}
impl<H> Hash for path::Args
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use path::Args::*;
        match self {
            None => {
                h.write_u8(0u8);
            },
            Angled(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Parenthed(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
        }
    }
}
impl<H> Hash for path::Segment
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.ident.hash(h);
        self.args.hash(h);
    }
}
impl<H> Hash for gen::where_::Life
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.life.hash(h);
        self.bounds.hash(h);
    }
}
impl<H> Hash for gen::where_::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.lifes.hash(h);
        self.typ.hash(h);
        self.bounds.hash(h);
    }
}
impl<H> Hash for path::QSelf
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.typ.hash(h);
        self.pos.hash(h);
        self.as_.hash(h);
    }
}
impl<H> Hash for expr::Limits
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use expr::Limits::*;
        match self {
            HalfOpen(_) => {
                h.write_u8(0u8);
            },
            Closed(_) => {
                h.write_u8(1u8);
            },
        }
    }
}
impl<H> Hash for item::Receiver
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ref_.hash(h);
        self.mut_.hash(h);
        self.colon.hash(h);
        self.typ.hash(h);
    }
}
impl<H> Hash for item::Sig
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.const_.hash(h);
        self.async_.hash(h);
        self.unsafe_.hash(h);
        self.abi.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.args.hash(h);
        self.vari.hash(h);
        self.ret.hash(h);
    }
}
impl<H> Hash for item::StaticMut
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use item::StaticMut::*;
        match self {
            Mut(_) => {
                h.write_u8(0u8);
            },
            None => {
                h.write_u8(1u8);
            },
        }
    }
}
impl<H> Hash for stmt::Stmt
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use stmt::Stmt::*;
        match self {
            Local(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Item(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Expr(x, v1) => {
                h.write_u8(2u8);
                x.hash(h);
                v1.hash(h);
            },
            Mac(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
        }
    }
}
impl<H> Hash for stmt::Mac
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for gen::bound::Trait
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.parenth.hash(h);
        self.modif.hash(h);
        self.lifes.hash(h);
        self.path.hash(h);
    }
}
impl<H> Hash for gen::bound::Modifier
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use gen::bound::Modifier::*;
        match self {
            None => {
                h.write_u8(0u8);
            },
            Maybe(_) => {
                h.write_u8(1u8);
            },
        }
    }
}
impl<H> Hash for item::trait_::Item
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use item::trait_::Item::*;
        match self {
            Const(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Fn(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Mac(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(4u8);
                StreamHelper(x).hash(h);
            },
        }
    }
}
impl<H> Hash for item::trait_::Const
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.typ.hash(h);
        self.default.hash(h);
    }
}
impl<H> Hash for item::trait_::Fn
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.sig.hash(h);
        self.default.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for item::trait_::Mac
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for item::trait_::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.colon.hash(h);
        self.bounds.hash(h);
        self.default.hash(h);
    }
}
impl<H> Hash for gen::param::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.colon.hash(h);
        self.bounds.hash(h);
        self.eq.hash(h);
        self.default.hash(h);
    }
}
impl<H> Hash for gen::bound::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use gen::bound::Type::*;
        match self {
            Trait(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Life(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(2u8);
                StreamHelper(x).hash(h);
            },
        }
    }
}
impl<H> Hash for expr::UnOp
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use expr::UnOp::*;
        match self {
            Deref(_) => {
                h.write_u8(0u8);
            },
            Not(_) => {
                h.write_u8(1u8);
            },
            Neg(_) => {
                h.write_u8(2u8);
            },
        }
    }
}
impl<H> Hash for item::use_::Glob
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {}
}
impl<H> Hash for item::use_::Group
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.trees.hash(h);
    }
}
impl<H> Hash for item::use_::Name
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.ident.hash(h);
    }
}
impl<H> Hash for item::use_::Path
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.ident.hash(h);
        self.tree.hash(h);
    }
}
impl<H> Hash for item::use_::Rename
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.ident.hash(h);
        self.rename.hash(h);
    }
}
impl<H> Hash for item::use_::Tree
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use item::use_::Tree::*;
        match self {
            Path(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Name(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Rename(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Glob(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Group(x) => {
                h.write_u8(4u8);
                x.hash(h);
            },
        }
    }
}
impl<H> Hash for item::Variadic
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.comma.hash(h);
    }
}
impl<H> Hash for data::Variant
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.fields.hash(h);
        self.discrim.hash(h);
    }
}
impl<H> Hash for data::Restricted
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.in_.hash(h);
        self.path.hash(h);
    }
}
impl<H> Hash for data::Visibility
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use data::Visibility::*;
        match self {
            Public(_) => {
                h.write_u8(0u8);
            },
            Restricted(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Inherited => {
                h.write_u8(2u8);
            },
        }
    }
}
impl<H> Hash for gen::Where
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.preds.hash(h);
    }
}
impl<H> Hash for gen::where_::Pred
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use gen::where_::Pred::*;
        match self {
            Life(v0) => {
                h.write_u8(0u8);
                v0.hash(h);
            },
            Type(v0) => {
                h.write_u8(1u8);
                v0.hash(h);
            },
        }
    }
}
