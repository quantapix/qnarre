use crate::StreamHelper;
use crate::*;
use std::hash::{Hash, Hasher};
impl Hash for typ::Abi {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.name.hash(h);
    }
}
impl Hash for path::Angled {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.colon2.hash(h);
        self.args.hash(h);
    }
}
impl Hash for expr::Arm {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.guard.hash(h);
        self.body.hash(h);
        self.comma.hash(h);
    }
}
impl Hash for path::AssocConst {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(h);
        self.args.hash(h);
        self.val.hash(h);
    }
}
impl Hash for path::AssocType {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(h);
        self.args.hash(h);
        self.typ.hash(h);
    }
}
impl Hash for attr::Style {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for attr::Attr {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.style.hash(h);
        self.meta.hash(h);
    }
}
impl Hash for typ::FnArg {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.name.hash(h);
        self.typ.hash(h);
    }
}
impl Hash for typ::Variadic {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.name.hash(h);
        self.comma.hash(h);
    }
}
impl Hash for expr::BinOp {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for stmt::Block {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.stmts.hash(h);
    }
}
impl Hash for gen::bound::Lifes {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.lifes.hash(h);
    }
}
impl Hash for gen::param::Const {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.typ.hash(h);
        self.eq.hash(h);
        self.default.hash(h);
    }
}
impl Hash for path::Constraint {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(h);
        self.args.hash(h);
        self.bounds.hash(h);
    }
}
impl Hash for data::Data {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for data::Enum {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.variants.hash(h);
    }
}
impl Hash for data::Struct {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.fields.hash(h);
        self.semi.hash(h);
    }
}
impl Hash for data::Union {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.fields.hash(h);
    }
}
impl Hash for Input {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.data.hash(h);
    }
}
impl Hash for expr::Expr {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for expr::Array {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.elems.hash(h);
    }
}
impl Hash for expr::Assign {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.left.hash(h);
        self.right.hash(h);
    }
}
impl Hash for expr::Async {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.move_.hash(h);
        self.block.hash(h);
    }
}
impl Hash for expr::Await {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.expr.hash(h);
    }
}
impl Hash for expr::Binary {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.left.hash(h);
        self.op.hash(h);
        self.right.hash(h);
    }
}
impl Hash for expr::Block {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.label.hash(h);
        self.block.hash(h);
    }
}
impl Hash for expr::Break {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.life.hash(h);
        self.val.hash(h);
    }
}
impl Hash for expr::Call {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.func.hash(h);
        self.args.hash(h);
    }
}
impl Hash for expr::Cast {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.expr.hash(h);
        self.typ.hash(h);
    }
}
impl Hash for expr::Closure {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for expr::Const {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.block.hash(h);
    }
}
impl Hash for expr::Continue {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.life.hash(h);
    }
}
impl Hash for expr::Field {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.expr.hash(h);
        self.memb.hash(h);
    }
}
impl Hash for expr::ForLoop {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.label.hash(h);
        self.pat.hash(h);
        self.expr.hash(h);
        self.body.hash(h);
    }
}
impl Hash for expr::Group {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.expr.hash(h);
    }
}
impl Hash for expr::If {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.cond.hash(h);
        self.then_.hash(h);
        self.else_.hash(h);
    }
}
impl Hash for expr::Index {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.expr.hash(h);
        self.idx.hash(h);
    }
}
impl Hash for expr::Infer {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
    }
}
impl Hash for expr::Let {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.expr.hash(h);
    }
}
impl Hash for expr::Lit {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.lit.hash(h);
    }
}
impl Hash for expr::Loop {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.label.hash(h);
        self.body.hash(h);
    }
}
impl Hash for expr::Mac {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.mac.hash(h);
    }
}
impl Hash for expr::Match {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.expr.hash(h);
        self.arms.hash(h);
    }
}
impl Hash for expr::MethodCall {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.expr.hash(h);
        self.method.hash(h);
        self.turbofish.hash(h);
        self.args.hash(h);
    }
}
impl Hash for expr::Parenth {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.expr.hash(h);
    }
}
impl Hash for expr::Path {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.qself.hash(h);
        self.path.hash(h);
    }
}
impl Hash for expr::Range {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.beg.hash(h);
        self.limits.hash(h);
        self.end.hash(h);
    }
}
impl Hash for expr::Ref {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.mut_.hash(h);
        self.expr.hash(h);
    }
}
impl Hash for expr::Repeat {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.expr.hash(h);
        self.len.hash(h);
    }
}
impl Hash for expr::Return {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.expr.hash(h);
    }
}
impl Hash for expr::Struct {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.qself.hash(h);
        self.path.hash(h);
        self.fields.hash(h);
        self.dot2.hash(h);
        self.rest.hash(h);
    }
}
impl Hash for expr::Try {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.expr.hash(h);
    }
}
impl Hash for expr::TryBlock {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.block.hash(h);
    }
}
impl Hash for expr::Tuple {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.elems.hash(h);
    }
}
impl Hash for expr::Unary {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.op.hash(h);
        self.expr.hash(h);
    }
}
impl Hash for expr::Unsafe {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.block.hash(h);
    }
}
impl Hash for expr::While {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.label.hash(h);
        self.cond.hash(h);
        self.block.hash(h);
    }
}
impl Hash for expr::Yield {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.expr.hash(h);
    }
}
impl Hash for data::Field {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.mut_.hash(h);
        self.ident.hash(h);
        self.colon.hash(h);
        self.typ.hash(h);
    }
}
impl Hash for data::Mut {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        match self {
            data::Mut::None => {
                h.write_u8(0u8);
            },
        }
    }
}
impl Hash for pat::Field {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.memb.hash(h);
        self.colon.hash(h);
        self.pat.hash(h);
    }
}
impl Hash for expr::FieldValue {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.memb.hash(h);
        self.colon.hash(h);
        self.expr.hash(h);
    }
}
impl Hash for data::Fields {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for data::Named {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.fields.hash(h);
    }
}
impl Hash for data::Unnamed {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.fields.hash(h);
    }
}
impl Hash for item::File {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.shebang.hash(h);
        self.attrs.hash(h);
        self.items.hash(h);
    }
}
impl Hash for item::FnArg {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for item::foreign::Item {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for item::foreign::Fn {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.sig.hash(h);
    }
}
impl Hash for item::foreign::Mac {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
    }
}
impl Hash for item::foreign::Static {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.mut_.hash(h);
        self.ident.hash(h);
        self.typ.hash(h);
    }
}
impl Hash for item::foreign::Type {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
    }
}
impl Hash for path::Arg {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for gen::Param {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for gen::Gens {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.lt.hash(h);
        self.params.hash(h);
        self.gt.hash(h);
        self.where_.hash(h);
    }
}
impl Hash for item::impl_::Item {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for item::impl_::Const {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.default.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.typ.hash(h);
        self.expr.hash(h);
    }
}
impl Hash for item::impl_::Fn {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.default.hash(h);
        self.sig.hash(h);
        self.block.hash(h);
    }
}
impl Hash for item::impl_::Mac {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
    }
}
impl Hash for item::impl_::Type {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.default.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.typ.hash(h);
    }
}
impl Hash for item::impl_::Restriction {
    fn hash<H>(&self, _h: &mut H)
    where
        H: Hasher,
    {
        match *self {}
    }
}
impl Hash for item::Item {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for item::Const {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.typ.hash(h);
        self.expr.hash(h);
    }
}
impl Hash for item::Enum {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.variants.hash(h);
    }
}
impl Hash for item::Extern {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.rename.hash(h);
    }
}
impl Hash for item::Fn {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.sig.hash(h);
        self.block.hash(h);
    }
}
impl Hash for item::Foreign {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.unsafe_.hash(h);
        self.abi.hash(h);
        self.items.hash(h);
    }
}
impl Hash for item::Impl {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.default.hash(h);
        self.unsafe_.hash(h);
        self.gens.hash(h);
        self.trait_.hash(h);
        self.typ.hash(h);
        self.items.hash(h);
    }
}
impl Hash for item::Mac {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
    }
}
impl Hash for item::Mod {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.unsafe_.hash(h);
        self.ident.hash(h);
        self.items.hash(h);
        self.semi.hash(h);
    }
}
impl Hash for item::Static {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.mut_.hash(h);
        self.ident.hash(h);
        self.typ.hash(h);
        self.expr.hash(h);
    }
}
impl Hash for item::Struct {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.fields.hash(h);
        self.semi.hash(h);
    }
}
impl Hash for item::Trait {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for item::TraitAlias {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.bounds.hash(h);
    }
}
impl Hash for item::Type {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.typ.hash(h);
    }
}
impl Hash for item::Union {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.fields.hash(h);
    }
}
impl Hash for item::Use {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.colon.hash(h);
        self.tree.hash(h);
    }
}
impl Hash for expr::Label {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.name.hash(h);
    }
}
impl Hash for gen::param::Life {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.life.hash(h);
        self.colon.hash(h);
        self.bounds.hash(h);
    }
}
impl Hash for lit::Lit {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for lit::Bool {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.val.hash(h);
    }
}
impl Hash for stmt::Local {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.init.hash(h);
    }
}
impl Hash for stmt::Init {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.expr.hash(h);
        self.diverge.hash(h);
    }
}
impl Hash for mac::Mac {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.path.hash(h);
        self.delim.hash(h);
        StreamHelper(&self.toks).hash(h);
    }
}
impl Hash for tok::Delim {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for attr::Meta {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for attr::List {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.path.hash(h);
        self.delim.hash(h);
        StreamHelper(&self.toks).hash(h);
    }
}
impl Hash for attr::NameValue {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.name.hash(h);
        self.val.hash(h);
    }
}
impl Hash for path::Parenthed {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.args.hash(h);
        self.ret.hash(h);
    }
}
impl Hash for pat::Pat {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for pat::Ident {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.ref_.hash(h);
        self.mut_.hash(h);
        self.ident.hash(h);
        self.sub.hash(h);
    }
}
impl Hash for pat::Or {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.vert.hash(h);
        self.cases.hash(h);
    }
}
impl Hash for pat::Parenth {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.pat.hash(h);
    }
}
impl Hash for pat::Ref {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.mut_.hash(h);
        self.pat.hash(h);
    }
}
impl Hash for pat::Rest {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
    }
}
impl Hash for pat::Slice {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.pats.hash(h);
    }
}
impl Hash for pat::Struct {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.qself.hash(h);
        self.path.hash(h);
        self.fields.hash(h);
        self.rest.hash(h);
    }
}
impl Hash for pat::Tuple {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.pats.hash(h);
    }
}
impl Hash for pat::TupleStruct {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.qself.hash(h);
        self.path.hash(h);
        self.pats.hash(h);
    }
}
impl Hash for pat::Type {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.typ.hash(h);
    }
}
impl Hash for pat::Wild {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
    }
}
impl Hash for Path {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.colon.hash(h);
        self.segs.hash(h);
    }
}
impl Hash for path::Args {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for path::Segment {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(h);
        self.args.hash(h);
    }
}
impl Hash for gen::where_::Life {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.life.hash(h);
        self.bounds.hash(h);
    }
}
impl Hash for gen::where_::Type {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.lifes.hash(h);
        self.typ.hash(h);
        self.bounds.hash(h);
    }
}
impl Hash for path::QSelf {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.typ.hash(h);
        self.pos.hash(h);
        self.as_.hash(h);
    }
}
impl Hash for expr::Limits {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for item::Receiver {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.ref_.hash(h);
        self.mut_.hash(h);
        self.colon.hash(h);
        self.typ.hash(h);
    }
}
impl Hash for typ::Ret {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        use typ::Ret::*;
        match self {
            Default => {
                h.write_u8(0u8);
            },
            Type(_, v1) => {
                h.write_u8(1u8);
                v1.hash(h);
            },
        }
    }
}
impl Hash for item::Sig {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for item::StaticMut {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for stmt::Stmt {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for stmt::Mac {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
    }
}
impl Hash for gen::bound::Trait {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.parenth.hash(h);
        self.modif.hash(h);
        self.lifes.hash(h);
        self.path.hash(h);
    }
}
impl Hash for gen::bound::Modifier {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for item::trait_::Item {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for item::trait_::Const {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.typ.hash(h);
        self.default.hash(h);
    }
}
impl Hash for item::trait_::Fn {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.sig.hash(h);
        self.default.hash(h);
        self.semi.hash(h);
    }
}
impl Hash for item::trait_::Mac {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
    }
}
impl Hash for item::trait_::Type {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.colon.hash(h);
        self.bounds.hash(h);
        self.default.hash(h);
    }
}
impl Hash for typ::Type {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        use typ::Type::*;
        match self {
            Array(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Fn(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Group(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Impl(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Infer(x) => {
                h.write_u8(4u8);
                x.hash(h);
            },
            Mac(x) => {
                h.write_u8(5u8);
                x.hash(h);
            },
            Never(x) => {
                h.write_u8(6u8);
                x.hash(h);
            },
            Parenth(x) => {
                h.write_u8(7u8);
                x.hash(h);
            },
            Path(x) => {
                h.write_u8(8u8);
                x.hash(h);
            },
            Ptr(x) => {
                h.write_u8(9u8);
                x.hash(h);
            },
            Ref(x) => {
                h.write_u8(10u8);
                x.hash(h);
            },
            Slice(x) => {
                h.write_u8(11u8);
                x.hash(h);
            },
            Trait(x) => {
                h.write_u8(12u8);
                x.hash(h);
            },
            Tuple(x) => {
                h.write_u8(13u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(14u8);
                StreamHelper(x).hash(h);
            },
        }
    }
}
impl Hash for typ::Array {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(h);
        self.len.hash(h);
    }
}
impl Hash for typ::Fn {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.lifes.hash(h);
        self.unsafe_.hash(h);
        self.abi.hash(h);
        self.args.hash(h);
        self.vari.hash(h);
        self.ret.hash(h);
    }
}
impl Hash for typ::Group {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(h);
    }
}
impl Hash for typ::Impl {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.bounds.hash(h);
    }
}
impl Hash for typ::Infer {
    fn hash<H>(&self, _: &mut H)
    where
        H: Hasher,
    {
    }
}
impl Hash for typ::Mac {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.mac.hash(h);
    }
}
impl Hash for typ::Never {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
    }
}
impl Hash for gen::param::Type {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.colon.hash(h);
        self.bounds.hash(h);
        self.eq.hash(h);
        self.default.hash(h);
    }
}
impl Hash for gen::bound::Type {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for typ::Parenth {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(h);
    }
}
impl Hash for typ::Path {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.qself.hash(h);
        self.path.hash(h);
    }
}
impl Hash for typ::Ptr {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.const_.hash(h);
        self.mut_.hash(h);
        self.elem.hash(h);
    }
}
impl Hash for typ::Ref {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.life.hash(h);
        self.mut_.hash(h);
        self.elem.hash(h);
    }
}
impl Hash for typ::Slice {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(h);
    }
}
impl Hash for typ::Trait {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.dyn_.hash(h);
        self.bounds.hash(h);
    }
}
impl Hash for typ::Tuple {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.elems.hash(h);
    }
}
impl Hash for expr::UnOp {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for item::use_::Glob {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
    }
}
impl Hash for item::use_::Group {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.trees.hash(h);
    }
}
impl Hash for item::use_::Name {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(h);
    }
}
impl Hash for item::use_::Path {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(h);
        self.tree.hash(h);
    }
}
impl Hash for item::use_::Rename {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(h);
        self.rename.hash(h);
    }
}
impl Hash for item::use_::Tree {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for item::Variadic {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.comma.hash(h);
    }
}
impl Hash for data::Variant {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.fields.hash(h);
        self.discrim.hash(h);
    }
}
impl Hash for data::Restricted {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.in_.hash(h);
        self.path.hash(h);
    }
}
impl Hash for data::Visibility {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
impl Hash for gen::Where {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
        self.preds.hash(h);
    }
}
impl Hash for gen::where_::Pred {
    fn hash<H>(&self, h: &mut H)
    where
        H: Hasher,
    {
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
