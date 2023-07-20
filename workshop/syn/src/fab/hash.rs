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
impl Hash for path::AngledArgs {
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
impl Hash for DeriveInput {
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
            Macro(x) => {
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
            Paren(x) => {
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
            Reference(x) => {
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
            Stream(x) => {
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
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.elems.hash(state);
    }
}
impl Hash for expr::Assign {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.left.hash(state);
        self.right.hash(state);
    }
}
impl Hash for expr::Async {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.move_.hash(state);
        self.block.hash(state);
    }
}
impl Hash for expr::Await {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for expr::Binary {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.left.hash(state);
        self.op.hash(state);
        self.right.hash(state);
    }
}
impl Hash for expr::Block {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.label.hash(state);
        self.block.hash(state);
    }
}
impl Hash for expr::Break {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.label.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for expr::Call {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.func.hash(state);
        self.args.hash(state);
    }
}
impl Hash for expr::Cast {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
        self.typ.hash(state);
    }
}
impl Hash for expr::Closure {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.lifes.hash(state);
        self.const_.hash(state);
        self.static_.hash(state);
        self.async_.hash(state);
        self.move_.hash(state);
        self.inputs.hash(state);
        self.ret.hash(state);
        self.body.hash(state);
    }
}
impl Hash for expr::Const {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.block.hash(state);
    }
}
impl Hash for expr::Continue {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.label.hash(state);
    }
}
impl Hash for expr::Field {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.base.hash(state);
        self.memb.hash(state);
    }
}
impl Hash for expr::ForLoop {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.label.hash(state);
        self.pat.hash(state);
        self.expr.hash(state);
        self.body.hash(state);
    }
}
impl Hash for expr::Group {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for expr::If {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.cond.hash(state);
        self.then_branch.hash(state);
        self.else_branch.hash(state);
    }
}
impl Hash for expr::Index {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
        self.index.hash(state);
    }
}
impl Hash for expr::Infer {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
    }
}
impl Hash for expr::Let {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.pat.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for expr::Lit {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.lit.hash(state);
    }
}
impl Hash for expr::Loop {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.label.hash(state);
        self.body.hash(state);
    }
}
impl Hash for expr::Mac {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mac.hash(state);
    }
}
impl Hash for expr::Match {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
        self.arms.hash(state);
    }
}
impl Hash for expr::MethodCall {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
        self.method.hash(state);
        self.turbofish.hash(state);
        self.args.hash(state);
    }
}
impl Hash for expr::Paren {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for expr::Path {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.qself.hash(state);
        self.path.hash(state);
    }
}
impl Hash for expr::Range {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.beg.hash(state);
        self.limits.hash(state);
        self.end.hash(state);
    }
}
impl Hash for expr::Ref {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mut_.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for expr::Repeat {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
        self.len.hash(state);
    }
}
impl Hash for expr::Return {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for expr::Struct {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.qself.hash(state);
        self.path.hash(state);
        self.fields.hash(state);
        self.dot2.hash(state);
        self.rest.hash(state);
    }
}
impl Hash for expr::Try {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for expr::TryBlock {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.block.hash(state);
    }
}
impl Hash for expr::Tuple {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.elems.hash(state);
    }
}
impl Hash for expr::Unary {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.op.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for expr::Unsafe {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.block.hash(state);
    }
}
impl Hash for expr::While {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.label.hash(state);
        self.cond.hash(state);
        self.body.hash(state);
    }
}
impl Hash for expr::Yield {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for data::Field {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.mut_.hash(state);
        self.ident.hash(state);
        self.colon.hash(state);
        self.typ.hash(state);
    }
}
impl Hash for data::Mut {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            data::Mut::None => {
                state.write_u8(0u8);
            },
        }
    }
}
impl Hash for pat::Field {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.memb.hash(state);
        self.colon.hash(state);
        self.pat.hash(state);
    }
}
impl Hash for FieldValue {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.member.hash(state);
        self.colon.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for data::Fields {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            data::Fields::Named(x) => {
                state.write_u8(0u8);
                x.hash(state);
            },
            data::Fields::Unnamed(x) => {
                state.write_u8(1u8);
                x.hash(state);
            },
            data::Fields::Unit => {
                state.write_u8(2u8);
            },
        }
    }
}
impl Hash for data::Named {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.fields.hash(state);
    }
}
impl Hash for data::Unnamed {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.fields.hash(state);
    }
}
impl Hash for item::File {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.shebang.hash(state);
        self.attrs.hash(state);
        self.items.hash(state);
    }
}
impl Hash for item::FnArg {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            item::FnArg::Receiver(x) => {
                state.write_u8(0u8);
                x.hash(state);
            },
            item::FnArg::Type(x) => {
                state.write_u8(1u8);
                x.hash(state);
            },
        }
    }
}
impl Hash for item::foreign::Item {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            item::foreign::Item::Fn(x) => {
                state.write_u8(0u8);
                x.hash(state);
            },
            item::foreign::Item::Static(x) => {
                state.write_u8(1u8);
                x.hash(state);
            },
            item::foreign::Item::Type(x) => {
                state.write_u8(2u8);
                x.hash(state);
            },
            item::foreign::Item::Macro(x) => {
                state.write_u8(3u8);
                x.hash(state);
            },
            item::foreign::Item::Stream(x) => {
                state.write_u8(4u8);
                StreamHelper(x).hash(state);
            },
        }
    }
}
impl Hash for item::foreign::Fn {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.sig.hash(state);
    }
}
impl Hash for item::foreign::Mac {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mac.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for item::foreign::Static {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.mut_.hash(state);
        self.ident.hash(state);
        self.typ.hash(state);
    }
}
impl Hash for item::foreign::Type {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.ident.hash(state);
        self.gens.hash(state);
    }
}
impl Hash for Arg {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            Arg::Life(x) => {
                state.write_u8(0u8);
                x.hash(state);
            },
            Arg::Type(x) => {
                state.write_u8(1u8);
                x.hash(state);
            },
            Arg::Const(x) => {
                state.write_u8(2u8);
                x.hash(state);
            },
            Arg::AssocType(x) => {
                state.write_u8(3u8);
                x.hash(state);
            },
            Arg::AssocConst(x) => {
                state.write_u8(4u8);
                x.hash(state);
            },
            Arg::Constraint(x) => {
                state.write_u8(5u8);
                x.hash(state);
            },
        }
    }
}
impl Hash for gen::Param {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            gen::Param::Life(x) => {
                state.write_u8(0u8);
                x.hash(state);
            },
            gen::Param::Type(x) => {
                state.write_u8(1u8);
                x.hash(state);
            },
            gen::Param::Const(x) => {
                state.write_u8(2u8);
                x.hash(state);
            },
        }
    }
}
impl Hash for gen::Gens {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.lt.hash(state);
        self.params.hash(state);
        self.gt.hash(state);
        self.where_.hash(state);
    }
}
impl Hash for item::impl_::Item {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            item::impl_::Item::Const(x) => {
                state.write_u8(0u8);
                x.hash(state);
            },
            item::impl_::Item::Fn(x) => {
                state.write_u8(1u8);
                x.hash(state);
            },
            item::impl_::Item::Type(x) => {
                state.write_u8(2u8);
                x.hash(state);
            },
            item::impl_::Item::Macro(x) => {
                state.write_u8(3u8);
                x.hash(state);
            },
            item::impl_::Item::Stream(x) => {
                state.write_u8(4u8);
                StreamHelper(x).hash(state);
            },
        }
    }
}
impl Hash for item::impl_::Const {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.default_.hash(state);
        self.ident.hash(state);
        self.gens.hash(state);
        self.typ.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for item::impl_::Fn {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.default_.hash(state);
        self.sig.hash(state);
        self.block.hash(state);
    }
}
impl Hash for item::impl_::Mac {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mac.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for item::impl_::Type {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.default_.hash(state);
        self.ident.hash(state);
        self.gens.hash(state);
        self.typ.hash(state);
    }
}
impl Hash for item::impl_::Restriction {
    fn hash<H>(&self, _state: &mut H)
    where
        H: Hasher,
    {
        match *self {}
    }
}
impl Hash for Item {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            Item::Const(x) => {
                state.write_u8(0u8);
                x.hash(state);
            },
            Item::Enum(x) => {
                state.write_u8(1u8);
                x.hash(state);
            },
            Item::Extern(x) => {
                state.write_u8(2u8);
                x.hash(state);
            },
            Item::Fn(x) => {
                state.write_u8(3u8);
                x.hash(state);
            },
            Item::Foreign(x) => {
                state.write_u8(4u8);
                x.hash(state);
            },
            Item::Impl(x) => {
                state.write_u8(5u8);
                x.hash(state);
            },
            Item::Macro(x) => {
                state.write_u8(6u8);
                x.hash(state);
            },
            Item::Mod(x) => {
                state.write_u8(7u8);
                x.hash(state);
            },
            Item::Static(x) => {
                state.write_u8(8u8);
                x.hash(state);
            },
            Item::Struct(x) => {
                state.write_u8(9u8);
                x.hash(state);
            },
            Item::Trait(x) => {
                state.write_u8(10u8);
                x.hash(state);
            },
            Item::TraitAlias(x) => {
                state.write_u8(11u8);
                x.hash(state);
            },
            Item::Type(x) => {
                state.write_u8(12u8);
                x.hash(state);
            },
            Item::Union(x) => {
                state.write_u8(13u8);
                x.hash(state);
            },
            Item::Use(x) => {
                state.write_u8(14u8);
                x.hash(state);
            },
            Item::Stream(x) => {
                state.write_u8(15u8);
                StreamHelper(x).hash(state);
            },
        }
    }
}
impl Hash for item::Const {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.ident.hash(state);
        self.gens.hash(state);
        self.typ.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for item::Enum {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.ident.hash(state);
        self.gens.hash(state);
        self.variants.hash(state);
    }
}
impl Hash for item::Extern {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.ident.hash(state);
        self.rename.hash(state);
    }
}
impl Hash for item::Fn {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.sig.hash(state);
        self.block.hash(state);
    }
}
impl Hash for item::Foreign {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.unsafe_.hash(state);
        self.abi.hash(state);
        self.items.hash(state);
    }
}
impl Hash for item::Impl {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.default_.hash(state);
        self.unsafe_.hash(state);
        self.gens.hash(state);
        self.trait_.hash(state);
        self.typ.hash(state);
        self.items.hash(state);
    }
}
impl Hash for item::Mac {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.ident.hash(state);
        self.mac.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for item::Mod {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.unsafe_.hash(state);
        self.ident.hash(state);
        self.items.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for item::Static {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.mut_.hash(state);
        self.ident.hash(state);
        self.typ.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for item::Struct {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.ident.hash(state);
        self.gens.hash(state);
        self.fields.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for item::Trait {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.unsafe_.hash(state);
        self.auto_.hash(state);
        self.restriction.hash(state);
        self.ident.hash(state);
        self.gens.hash(state);
        self.colon.hash(state);
        self.supers.hash(state);
        self.items.hash(state);
    }
}
impl Hash for item::TraitAlias {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.ident.hash(state);
        self.gens.hash(state);
        self.bounds.hash(state);
    }
}
impl Hash for item::Type {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.ident.hash(state);
        self.gens.hash(state);
        self.typ.hash(state);
    }
}
impl Hash for item::Union {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.ident.hash(state);
        self.gens.hash(state);
        self.fields.hash(state);
    }
}
impl Hash for item::Use {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.colon.hash(state);
        self.tree.hash(state);
    }
}
impl Hash for Label {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.name.hash(state);
    }
}
impl Hash for gen::param::Life {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.life.hash(state);
        self.colon.hash(state);
        self.bounds.hash(state);
    }
}
impl Hash for Lit {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            Lit::Str(x) => {
                state.write_u8(0u8);
                x.hash(state);
            },
            Lit::ByteStr(x) => {
                state.write_u8(1u8);
                x.hash(state);
            },
            Lit::Byte(x) => {
                state.write_u8(2u8);
                x.hash(state);
            },
            Lit::Char(x) => {
                state.write_u8(3u8);
                x.hash(state);
            },
            Lit::Int(x) => {
                state.write_u8(4u8);
                x.hash(state);
            },
            Lit::Float(x) => {
                state.write_u8(5u8);
                x.hash(state);
            },
            Lit::Bool(x) => {
                state.write_u8(6u8);
                x.hash(state);
            },
            Lit::Stream(x) => {
                state.write_u8(7u8);
                x.to_string().hash(state);
            },
        }
    }
}
impl Hash for lit::Bool {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.val.hash(state);
    }
}
impl Hash for stmt::Local {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.pat.hash(state);
        self.init.hash(state);
    }
}
impl Hash for stmt::Init {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.expr.hash(state);
        self.diverge.hash(state);
    }
}
impl Hash for Macro {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.path.hash(state);
        self.delim.hash(state);
        StreamHelper(&self.toks).hash(state);
    }
}
impl Hash for tok::Delim {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            tok::Delim::Paren(_) => {
                state.write_u8(0u8);
            },
            tok::Delim::Brace(_) => {
                state.write_u8(1u8);
            },
            tok::Delim::Bracket(_) => {
                state.write_u8(2u8);
            },
        }
    }
}
impl Hash for meta::Meta {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            meta::Meta::Path(x) => {
                state.write_u8(0u8);
                x.hash(state);
            },
            meta::Meta::List(x) => {
                state.write_u8(1u8);
                x.hash(state);
            },
            meta::Meta::NameValue(x) => {
                state.write_u8(2u8);
                x.hash(state);
            },
        }
    }
}
impl Hash for meta::List {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.path.hash(state);
        self.delim.hash(state);
        StreamHelper(&self.toks).hash(state);
    }
}
impl Hash for meta::NameValue {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.path.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for ParenthesizedArgs {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ins.hash(state);
        self.out.hash(state);
    }
}
impl Hash for pat::Pat {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            pat::Pat::Const(x) => {
                state.write_u8(0u8);
                x.hash(state);
            },
            pat::Pat::Ident(x) => {
                state.write_u8(1u8);
                x.hash(state);
            },
            pat::Pat::Lit(x) => {
                state.write_u8(2u8);
                x.hash(state);
            },
            pat::Pat::Mac(x) => {
                state.write_u8(3u8);
                x.hash(state);
            },
            pat::Pat::Or(x) => {
                state.write_u8(4u8);
                x.hash(state);
            },
            pat::Pat::Paren(x) => {
                state.write_u8(5u8);
                x.hash(state);
            },
            pat::Pat::Path(x) => {
                state.write_u8(6u8);
                x.hash(state);
            },
            pat::Pat::Range(x) => {
                state.write_u8(7u8);
                x.hash(state);
            },
            pat::Pat::Ref(x) => {
                state.write_u8(8u8);
                x.hash(state);
            },
            pat::Pat::Rest(x) => {
                state.write_u8(9u8);
                x.hash(state);
            },
            pat::Pat::Slice(x) => {
                state.write_u8(10u8);
                x.hash(state);
            },
            pat::Pat::Struct(x) => {
                state.write_u8(11u8);
                x.hash(state);
            },
            pat::Pat::Tuple(x) => {
                state.write_u8(12u8);
                x.hash(state);
            },
            pat::Pat::TupleStruct(x) => {
                state.write_u8(13u8);
                x.hash(state);
            },
            pat::Pat::Type(x) => {
                state.write_u8(14u8);
                x.hash(state);
            },
            pat::Pat::Stream(x) => {
                state.write_u8(15u8);
                StreamHelper(x).hash(state);
            },
            pat::Pat::Wild(x) => {
                state.write_u8(16u8);
                x.hash(state);
            },
        }
    }
}
impl Hash for pat::Ident {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.ref_.hash(state);
        self.mut_.hash(state);
        self.ident.hash(state);
        self.sub.hash(state);
    }
}
impl Hash for pat::Or {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vert.hash(state);
        self.cases.hash(state);
    }
}
impl Hash for pat::Paren {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.pat.hash(state);
    }
}
impl Hash for pat::Ref {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mut_.hash(state);
        self.pat.hash(state);
    }
}
impl Hash for pat::Rest {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
    }
}
impl Hash for pat::Slice {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.elems.hash(state);
    }
}
impl Hash for pat::Struct {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.qself.hash(state);
        self.path.hash(state);
        self.fields.hash(state);
        self.rest.hash(state);
    }
}
impl Hash for pat::Tuple {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.elems.hash(state);
    }
}
impl Hash for pat::TupleStruct {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.qself.hash(state);
        self.path.hash(state);
        self.elems.hash(state);
    }
}
impl Hash for pat::Type {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.pat.hash(state);
        self.typ.hash(state);
    }
}
impl Hash for pat::Wild {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
    }
}
impl Hash for Path {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.colon.hash(state);
        self.segs.hash(state);
    }
}
impl Hash for Args {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            Args::None => {
                state.write_u8(0u8);
            },
            Args::Angled(x) => {
                state.write_u8(1u8);
                x.hash(state);
            },
            Args::Parenthesized(x) => {
                state.write_u8(2u8);
                x.hash(state);
            },
        }
    }
}
impl Hash for Segment {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(state);
        self.args.hash(state);
    }
}
impl Hash for gen::Where::Life {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.life.hash(state);
        self.bounds.hash(state);
    }
}
impl Hash for gen::Where::Type {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.lifes.hash(state);
        self.bounded.hash(state);
        self.bounds.hash(state);
    }
}
impl Hash for QSelf {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ty.hash(state);
        self.pos.hash(state);
        self.as_.hash(state);
    }
}
impl Hash for expr::Limits {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            expr::Limits::HalfOpen(_) => {
                state.write_u8(0u8);
            },
            expr::Limits::Closed(_) => {
                state.write_u8(1u8);
            },
        }
    }
}
impl Hash for item::Receiver {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.ref_.hash(state);
        self.mut_.hash(state);
        self.colon.hash(state);
        self.typ.hash(state);
    }
}
impl Hash for typ::Ret {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            typ::Ret::Default => {
                state.write_u8(0u8);
            },
            typ::Ret::Type(_, v1) => {
                state.write_u8(1u8);
                v1.hash(state);
            },
        }
    }
}
impl Hash for item::Sig {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.const_.hash(state);
        self.async_.hash(state);
        self.unsafe_.hash(state);
        self.abi.hash(state);
        self.ident.hash(state);
        self.gens.hash(state);
        self.args.hash(state);
        self.vari.hash(state);
        self.ret.hash(state);
    }
}
impl Hash for StaticMut {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            StaticMut::Mut(_) => {
                state.write_u8(0u8);
            },
            StaticMut::None => {
                state.write_u8(1u8);
            },
        }
    }
}
impl Hash for stmt::Stmt {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            stmt::Stmt::stmt::Local(x) => {
                state.write_u8(0u8);
                x.hash(state);
            },
            stmt::Stmt::Item(x) => {
                state.write_u8(1u8);
                x.hash(state);
            },
            stmt::Stmt::Expr(x, v1) => {
                state.write_u8(2u8);
                x.hash(state);
                v1.hash(state);
            },
            stmt::Stmt::Mac(x) => {
                state.write_u8(3u8);
                x.hash(state);
            },
        }
    }
}
impl Hash for stmt::Mac {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mac.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for gen::bound::Trait {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.paren.hash(state);
        self.modif.hash(state);
        self.lifes.hash(state);
        self.path.hash(state);
    }
}
impl Hash for gen::bound::Modifier {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            gen::bound::Modifier::None => {
                state.write_u8(0u8);
            },
            gen::bound::Modifier::Maybe(_) => {
                state.write_u8(1u8);
            },
        }
    }
}
impl Hash for item::trait_::Item {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            item::trait_::Item::Const(x) => {
                state.write_u8(0u8);
                x.hash(state);
            },
            item::trait_::Item::Fn(x) => {
                state.write_u8(1u8);
                x.hash(state);
            },
            item::trait_::Item::Type(x) => {
                state.write_u8(2u8);
                x.hash(state);
            },
            item::trait_::Item::Macro(x) => {
                state.write_u8(3u8);
                x.hash(state);
            },
            item::trait_::Item::Stream(x) => {
                state.write_u8(4u8);
                StreamHelper(x).hash(state);
            },
        }
    }
}
impl Hash for item::trait_::Const {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.ident.hash(state);
        self.gens.hash(state);
        self.typ.hash(state);
        self.default.hash(state);
    }
}
impl Hash for item::trait_::Fn {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.sig.hash(state);
        self.default.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for item::trait_::Mac {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mac.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for item::trait_::Type {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.ident.hash(state);
        self.gens.hash(state);
        self.colon.hash(state);
        self.bounds.hash(state);
        self.default.hash(state);
    }
}
impl Hash for typ::Type {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            typ::Type::Array(x) => {
                state.write_u8(0u8);
                x.hash(state);
            },
            typ::Type::Fn(x) => {
                state.write_u8(1u8);
                x.hash(state);
            },
            typ::Type::Group(x) => {
                state.write_u8(2u8);
                x.hash(state);
            },
            typ::Type::Impl(x) => {
                state.write_u8(3u8);
                x.hash(state);
            },
            typ::Type::Infer(x) => {
                state.write_u8(4u8);
                x.hash(state);
            },
            typ::Type::Mac(x) => {
                state.write_u8(5u8);
                x.hash(state);
            },
            typ::Type::Never(x) => {
                state.write_u8(6u8);
                x.hash(state);
            },
            typ::Type::Paren(x) => {
                state.write_u8(7u8);
                x.hash(state);
            },
            typ::Type::Path(x) => {
                state.write_u8(8u8);
                x.hash(state);
            },
            typ::Type::Ptr(x) => {
                state.write_u8(9u8);
                x.hash(state);
            },
            typ::Type::Ref(x) => {
                state.write_u8(10u8);
                x.hash(state);
            },
            typ::Type::Slice(x) => {
                state.write_u8(11u8);
                x.hash(state);
            },
            typ::Type::Trait(x) => {
                state.write_u8(12u8);
                x.hash(state);
            },
            typ::Type::Tuple(x) => {
                state.write_u8(13u8);
                x.hash(state);
            },
            typ::Type::Stream(x) => {
                state.write_u8(14u8);
                StreamHelper(x).hash(state);
            },
        }
    }
}
impl Hash for typ::Array {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(state);
        self.len.hash(state);
    }
}
impl Hash for typ::Fn {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.lifes.hash(state);
        self.unsafe_.hash(state);
        self.abi.hash(state);
        self.args.hash(state);
        self.vari.hash(state);
        self.ret.hash(state);
    }
}
impl Hash for typ::Group {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(state);
    }
}
impl Hash for typ::Impl {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.bounds.hash(state);
    }
}
impl Hash for typ::Infer {
    fn hash<H>(&self, _state: &mut H)
    where
        H: Hasher,
    {
    }
}
impl Hash for typ::Mac {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.mac.hash(state);
    }
}
impl Hash for typ::Never {
    fn hash<H>(&self, _state: &mut H)
    where
        H: Hasher,
    {
    }
}
impl Hash for gen::param::Type {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.ident.hash(state);
        self.colon.hash(state);
        self.bounds.hash(state);
        self.eq.hash(state);
        self.default.hash(state);
    }
}
impl Hash for gen::bound::Type {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            gen::bound::Type::Trait(x) => {
                state.write_u8(0u8);
                x.hash(state);
            },
            gen::bound::Type::Life(x) => {
                state.write_u8(1u8);
                x.hash(state);
            },
            gen::bound::Type::Stream(x) => {
                state.write_u8(2u8);
                StreamHelper(x).hash(state);
            },
        }
    }
}
impl Hash for typ::Paren {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(state);
    }
}
impl Hash for typ::Path {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.qself.hash(state);
        self.path.hash(state);
    }
}
impl Hash for typ::Ptr {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.const_.hash(state);
        self.mut_.hash(state);
        self.elem.hash(state);
    }
}
impl Hash for typ::Ref {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.life.hash(state);
        self.mut_.hash(state);
        self.elem.hash(state);
    }
}
impl Hash for typ::Slice {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(state);
    }
}
impl Hash for typ::Trait {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.dyn_.hash(state);
        self.bounds.hash(state);
    }
}
impl Hash for typ::Tuple {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.elems.hash(state);
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
        self.elems.hash(h);
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
        gen::Where::Pred::*;
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
