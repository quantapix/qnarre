use crate::TokenStreamHelper;
use crate::*;
use std::hash::{Hash, Hasher};
impl Hash for Abi {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.name.hash(state);
    }
}
impl Hash for AngledArgs {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.colon2.hash(state);
        self.args.hash(state);
    }
}
impl Hash for Arm {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.pat.hash(state);
        self.guard.hash(state);
        self.body.hash(state);
        self.comma.hash(state);
    }
}
impl Hash for AssocConst {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(state);
        self.gnrs.hash(state);
        self.val.hash(state);
    }
}
impl Hash for AssocType {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(state);
        self.gnrs.hash(state);
        self.ty.hash(state);
    }
}
impl Hash for attr::Style {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            attr::Style::Outer => {
                state.write_u8(0u8);
            },
            attr::Style::Inner(_) => {
                state.write_u8(1u8);
            },
        }
    }
}
impl Hash for attr::Attr {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.style.hash(state);
        self.meta.hash(state);
    }
}
impl Hash for ty::FnArg {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.name.hash(state);
        self.ty.hash(state);
    }
}
impl Hash for ty::Variadic {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.name.hash(state);
        self.comma.hash(state);
    }
}
impl Hash for BinOp {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            BinOp::Add(_) => {
                state.write_u8(0u8);
            },
            BinOp::Sub(_) => {
                state.write_u8(1u8);
            },
            BinOp::Mul(_) => {
                state.write_u8(2u8);
            },
            BinOp::Div(_) => {
                state.write_u8(3u8);
            },
            BinOp::Rem(_) => {
                state.write_u8(4u8);
            },
            BinOp::And(_) => {
                state.write_u8(5u8);
            },
            BinOp::Or(_) => {
                state.write_u8(6u8);
            },
            BinOp::BitXor(_) => {
                state.write_u8(7u8);
            },
            BinOp::BitAnd(_) => {
                state.write_u8(8u8);
            },
            BinOp::BitOr(_) => {
                state.write_u8(9u8);
            },
            BinOp::Shl(_) => {
                state.write_u8(10u8);
            },
            BinOp::Shr(_) => {
                state.write_u8(11u8);
            },
            BinOp::Eq(_) => {
                state.write_u8(12u8);
            },
            BinOp::Lt(_) => {
                state.write_u8(13u8);
            },
            BinOp::Le(_) => {
                state.write_u8(14u8);
            },
            BinOp::Ne(_) => {
                state.write_u8(15u8);
            },
            BinOp::Ge(_) => {
                state.write_u8(16u8);
            },
            BinOp::Gt(_) => {
                state.write_u8(17u8);
            },
            BinOp::AddAssign(_) => {
                state.write_u8(18u8);
            },
            BinOp::SubAssign(_) => {
                state.write_u8(19u8);
            },
            BinOp::MulAssign(_) => {
                state.write_u8(20u8);
            },
            BinOp::DivAssign(_) => {
                state.write_u8(21u8);
            },
            BinOp::RemAssign(_) => {
                state.write_u8(22u8);
            },
            BinOp::BitXorAssign(_) => {
                state.write_u8(23u8);
            },
            BinOp::BitAndAssign(_) => {
                state.write_u8(24u8);
            },
            BinOp::BitOrAssign(_) => {
                state.write_u8(25u8);
            },
            BinOp::ShlAssign(_) => {
                state.write_u8(26u8);
            },
            BinOp::ShrAssign(_) => {
                state.write_u8(27u8);
            },
        }
    }
}
impl Hash for Block {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.stmts.hash(state);
    }
}
impl Hash for Bgen::bound::Lifes {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.lifes.hash(state);
    }
}
impl Hash for gen::param::Const {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.ident.hash(state);
        self.typ.hash(state);
        self.eq.hash(state);
        self.default.hash(state);
    }
}
impl Hash for Constraint {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(state);
        self.gnrs.hash(state);
        self.bounds.hash(state);
    }
}
impl Hash for Data {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            Data::Struct(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            Data::Enum(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            Data::Union(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
        }
    }
}
impl Hash for data::Enum {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.variants.hash(state);
    }
}
impl Hash for data::Struct {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.fields.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for data::Union {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.named.hash(state);
    }
}
impl Hash for DeriveInput {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.ident.hash(state);
        self.gens.hash(state);
        self.data.hash(state);
    }
}
impl Hash for Expr {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            Expr::Array(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            Expr::Assign(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            Expr::Async(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            Expr::Await(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            Expr::Binary(v0) => {
                state.write_u8(4u8);
                v0.hash(state);
            },
            Expr::Block(v0) => {
                state.write_u8(5u8);
                v0.hash(state);
            },
            Expr::Break(v0) => {
                state.write_u8(6u8);
                v0.hash(state);
            },
            Expr::Call(v0) => {
                state.write_u8(7u8);
                v0.hash(state);
            },
            Expr::Cast(v0) => {
                state.write_u8(8u8);
                v0.hash(state);
            },
            Expr::Closure(v0) => {
                state.write_u8(9u8);
                v0.hash(state);
            },
            Expr::Const(v0) => {
                state.write_u8(10u8);
                v0.hash(state);
            },
            Expr::Continue(v0) => {
                state.write_u8(11u8);
                v0.hash(state);
            },
            Expr::Field(v0) => {
                state.write_u8(12u8);
                v0.hash(state);
            },
            Expr::ForLoop(v0) => {
                state.write_u8(13u8);
                v0.hash(state);
            },
            Expr::Group(v0) => {
                state.write_u8(14u8);
                v0.hash(state);
            },
            Expr::If(v0) => {
                state.write_u8(15u8);
                v0.hash(state);
            },
            Expr::Index(v0) => {
                state.write_u8(16u8);
                v0.hash(state);
            },
            Expr::Infer(v0) => {
                state.write_u8(17u8);
                v0.hash(state);
            },
            Expr::Let(v0) => {
                state.write_u8(18u8);
                v0.hash(state);
            },
            Expr::Lit(v0) => {
                state.write_u8(19u8);
                v0.hash(state);
            },
            Expr::Loop(v0) => {
                state.write_u8(20u8);
                v0.hash(state);
            },
            Expr::Macro(v0) => {
                state.write_u8(21u8);
                v0.hash(state);
            },
            Expr::Match(v0) => {
                state.write_u8(22u8);
                v0.hash(state);
            },
            Expr::MethodCall(v0) => {
                state.write_u8(23u8);
                v0.hash(state);
            },
            Expr::Paren(v0) => {
                state.write_u8(24u8);
                v0.hash(state);
            },
            Expr::Path(v0) => {
                state.write_u8(25u8);
                v0.hash(state);
            },
            Expr::Range(v0) => {
                state.write_u8(26u8);
                v0.hash(state);
            },
            Expr::Reference(v0) => {
                state.write_u8(27u8);
                v0.hash(state);
            },
            Expr::Repeat(v0) => {
                state.write_u8(28u8);
                v0.hash(state);
            },
            Expr::Return(v0) => {
                state.write_u8(29u8);
                v0.hash(state);
            },
            Expr::Struct(v0) => {
                state.write_u8(30u8);
                v0.hash(state);
            },
            Expr::Try(v0) => {
                state.write_u8(31u8);
                v0.hash(state);
            },
            Expr::TryBlock(v0) => {
                state.write_u8(32u8);
                v0.hash(state);
            },
            Expr::Tuple(v0) => {
                state.write_u8(33u8);
                v0.hash(state);
            },
            Expr::Unary(v0) => {
                state.write_u8(34u8);
                v0.hash(state);
            },
            Expr::Unsafe(v0) => {
                state.write_u8(35u8);
                v0.hash(state);
            },
            Expr::Verbatim(v0) => {
                state.write_u8(36u8);
                TokenStreamHelper(v0).hash(state);
            },
            Expr::While(v0) => {
                state.write_u8(37u8);
                v0.hash(state);
            },
            Expr::Yield(v0) => {
                state.write_u8(38u8);
                v0.hash(state);
            },
            #[cfg(not(feature = "full"))]
            _ => unreachable!(),
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
        self.mutability.hash(state);
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
        self.member.hash(state);
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
            data::Fields::Named(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            data::Fields::Unnamed(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
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
        self.field.hash(state);
    }
}
impl Hash for data::Unnamed {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.field.hash(state);
    }
}
impl Hash for File {
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
            item::FnArg::Receiver(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            item::FnArg::Typed(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
        }
    }
}
impl Hash for item::Foreign::Item {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            item::Foreign::Item::Fn(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            item::Foreign::Item::Static(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            item::Foreign::Item::Type(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            item::Foreign::Item::Macro(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            item::Foreign::Item::Verbatim(v0) => {
                state.write_u8(4u8);
                TokenStreamHelper(v0).hash(state);
            },
        }
    }
}
impl Hash for item::Foreign::Fn {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.sig.hash(state);
    }
}
impl Hash for item::Foreign::Mac {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mac.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for item::Foreign::Static {
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
impl Hash for item::Foreign::Type {
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
            Arg::Lifetime(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            Arg::Type(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            Arg::Const(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            Arg::AssocType(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            Arg::AssocConst(v0) => {
                state.write_u8(4u8);
                v0.hash(state);
            },
            Arg::Constraint(v0) => {
                state.write_u8(5u8);
                v0.hash(state);
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
            gen::Param::Life(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            gen::Param::Type(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            gen::Param::Const(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
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
impl Hash for item::Impl::Item {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            item::Impl::Item::Const(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            item::Impl::Item::Fn(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            item::Impl::Item::Type(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            item::Impl::Item::Macro(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            item::Impl::Item::Verbatim(v0) => {
                state.write_u8(4u8);
                TokenStreamHelper(v0).hash(state);
            },
        }
    }
}
impl Hash for item::Impl::Const {
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
impl Hash for item::Impl::Fn {
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
impl Hash for item::Impl::Mac {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mac.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for item::Impl::Type {
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
impl Hash for item::Impl::Restriction {
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
            Item::Const(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            Item::Enum(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            Item::ExternCrate(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            Item::Fn(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            Item::Foreign(v0) => {
                state.write_u8(4u8);
                v0.hash(state);
            },
            Item::Impl(v0) => {
                state.write_u8(5u8);
                v0.hash(state);
            },
            Item::Macro(v0) => {
                state.write_u8(6u8);
                v0.hash(state);
            },
            Item::Mod(v0) => {
                state.write_u8(7u8);
                v0.hash(state);
            },
            Item::Static(v0) => {
                state.write_u8(8u8);
                v0.hash(state);
            },
            Item::Struct(v0) => {
                state.write_u8(9u8);
                v0.hash(state);
            },
            Item::Trait(v0) => {
                state.write_u8(10u8);
                v0.hash(state);
            },
            Item::TraitAlias(v0) => {
                state.write_u8(11u8);
                v0.hash(state);
            },
            Item::Type(v0) => {
                state.write_u8(12u8);
                v0.hash(state);
            },
            Item::Union(v0) => {
                state.write_u8(13u8);
                v0.hash(state);
            },
            Item::Use(v0) => {
                state.write_u8(14u8);
                v0.hash(state);
            },
            Item::Verbatim(v0) => {
                state.write_u8(15u8);
                TokenStreamHelper(v0).hash(state);
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
        self.elems.hash(state);
    }
}
impl Hash for item::ExternCrate {
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
        self.gist.hash(state);
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
            Lit::Str(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            Lit::ByteStr(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            Lit::Byte(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            Lit::Char(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            Lit::Int(v0) => {
                state.write_u8(4u8);
                v0.hash(state);
            },
            Lit::Float(v0) => {
                state.write_u8(5u8);
                v0.hash(state);
            },
            Lit::Bool(v0) => {
                state.write_u8(6u8);
                v0.hash(state);
            },
            Lit::Verbatim(v0) => {
                state.write_u8(7u8);
                v0.to_string().hash(state);
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
impl Hash for stmt::LocalInit {
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
        TokenStreamHelper(&self.toks).hash(state);
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
            meta::Meta::Path(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            meta::Meta::List(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            meta::Meta::NameValue(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
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
        TokenStreamHelper(&self.toks).hash(state);
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
            pat::Pat::Const(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            pat::Pat::Ident(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            pat::Pat::Lit(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            pat::Pat::Mac(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            pat::Pat::Or(v0) => {
                state.write_u8(4u8);
                v0.hash(state);
            },
            pat::Pat::Paren(v0) => {
                state.write_u8(5u8);
                v0.hash(state);
            },
            pat::Pat::Path(v0) => {
                state.write_u8(6u8);
                v0.hash(state);
            },
            pat::Pat::Range(v0) => {
                state.write_u8(7u8);
                v0.hash(state);
            },
            pat::Pat::Ref(v0) => {
                state.write_u8(8u8);
                v0.hash(state);
            },
            pat::Pat::Rest(v0) => {
                state.write_u8(9u8);
                v0.hash(state);
            },
            pat::Pat::Slice(v0) => {
                state.write_u8(10u8);
                v0.hash(state);
            },
            pat::Pat::Struct(v0) => {
                state.write_u8(11u8);
                v0.hash(state);
            },
            pat::Pat::Tuple(v0) => {
                state.write_u8(12u8);
                v0.hash(state);
            },
            pat::Pat::TupleStruct(v0) => {
                state.write_u8(13u8);
                v0.hash(state);
            },
            pat::Pat::Type(v0) => {
                state.write_u8(14u8);
                v0.hash(state);
            },
            pat::Pat::Verbatim(v0) => {
                state.write_u8(15u8);
                TokenStreamHelper(v0).hash(state);
            },
            pat::Pat::Wild(v0) => {
                state.write_u8(16u8);
                v0.hash(state);
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
            Args::Angled(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            Args::Parenthesized(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
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
impl Hash for RangeLimits {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            RangeLimits::HalfOpen(_) => {
                state.write_u8(0u8);
            },
            RangeLimits::Closed(_) => {
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
        self.reference.hash(state);
        self.mut_.hash(state);
        self.colon.hash(state);
        self.typ.hash(state);
    }
}
impl Hash for ty::Ret {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            ty::Ret::Default => {
                state.write_u8(0u8);
            },
            ty::Ret::Type(_, v1) => {
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
        self.constness.hash(state);
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
            stmt::Stmt::stmt::Local(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            stmt::Stmt::Item(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            stmt::Stmt::Expr(v0, v1) => {
                state.write_u8(2u8);
                v0.hash(state);
                v1.hash(state);
            },
            stmt::Stmt::Mac(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
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
impl Hash for item::Trait::Item {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            item::Trait::Item::Const(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            item::Trait::Item::Fn(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            item::Trait::Item::Type(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            item::Trait::Item::Macro(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            item::Trait::Item::Verbatim(v0) => {
                state.write_u8(4u8);
                TokenStreamHelper(v0).hash(state);
            },
        }
    }
}
impl Hash for item::Trait::Const {
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
impl Hash for item::Trait::Fn {
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
impl Hash for item::Trait::Mac {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mac.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for item::Trait::Type {
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
impl Hash for ty::Type {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            ty::Type::Array(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            ty::Type::Fn(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            ty::Type::Group(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            ty::Type::Impl(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            ty::Type::Infer(v0) => {
                state.write_u8(4u8);
                v0.hash(state);
            },
            ty::Type::Mac(v0) => {
                state.write_u8(5u8);
                v0.hash(state);
            },
            ty::Type::Never(v0) => {
                state.write_u8(6u8);
                v0.hash(state);
            },
            ty::Type::Paren(v0) => {
                state.write_u8(7u8);
                v0.hash(state);
            },
            ty::Type::Path(v0) => {
                state.write_u8(8u8);
                v0.hash(state);
            },
            ty::Type::Ptr(v0) => {
                state.write_u8(9u8);
                v0.hash(state);
            },
            ty::Type::Ref(v0) => {
                state.write_u8(10u8);
                v0.hash(state);
            },
            ty::Type::Slice(v0) => {
                state.write_u8(11u8);
                v0.hash(state);
            },
            ty::Type::TraitObj(v0) => {
                state.write_u8(12u8);
                v0.hash(state);
            },
            ty::Type::Tuple(v0) => {
                state.write_u8(13u8);
                v0.hash(state);
            },
            ty::Type::Verbatim(v0) => {
                state.write_u8(14u8);
                TokenStreamHelper(v0).hash(state);
            },
        }
    }
}
impl Hash for ty::Array {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(state);
        self.len.hash(state);
    }
}
impl Hash for ty::Fn {
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
impl Hash for ty::Group {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(state);
    }
}
impl Hash for ty::Impl {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.bounds.hash(state);
    }
}
impl Hash for ty::Infer {
    fn hash<H>(&self, _state: &mut H)
    where
        H: Hasher,
    {
    }
}
impl Hash for ty::Mac {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.mac.hash(state);
    }
}
impl Hash for ty::Never {
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
            gen::bound::Type::Trait(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            gen::bound::Type::Lifetime(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            gen::bound::Type::Verbatim(v0) => {
                state.write_u8(2u8);
                TokenStreamHelper(v0).hash(state);
            },
        }
    }
}
impl Hash for ty::Paren {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(state);
    }
}
impl Hash for ty::Path {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.qself.hash(state);
        self.path.hash(state);
    }
}
impl Hash for ty::Ptr {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.const_.hash(state);
        self.mut_.hash(state);
        self.elem.hash(state);
    }
}
impl Hash for ty::Ref {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.life.hash(state);
        self.mut_.hash(state);
        self.elem.hash(state);
    }
}
impl Hash for ty::Slice {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(state);
    }
}
impl Hash for ty::TraitObj {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.dyn_.hash(state);
        self.bounds.hash(state);
    }
}
impl Hash for ty::Tuple {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.elems.hash(state);
    }
}
impl Hash for UnOp {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            UnOp::Deref(_) => {
                state.write_u8(0u8);
            },
            UnOp::Not(_) => {
                state.write_u8(1u8);
            },
            UnOp::Neg(_) => {
                state.write_u8(2u8);
            },
        }
    }
}
impl Hash for item::Use::Glob {
    fn hash<H>(&self, _state: &mut H)
    where
        H: Hasher,
    {
    }
}
impl Hash for item::Use::Group {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.elems.hash(state);
    }
}
impl Hash for item::Use::Name {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(state);
    }
}
impl Hash for item::Use::Path {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(state);
        self.tree.hash(state);
    }
}
impl Hash for item::Use::Rename {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(state);
        self.rename.hash(state);
    }
}
impl Hash for item::Use::Tree {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            item::Use::Tree::Path(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            item::Use::Tree::Name(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            item::Use::Tree::Rename(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            item::Use::Tree::Glob(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            item::Use::Tree::Group(v0) => {
                state.write_u8(4u8);
                v0.hash(state);
            },
        }
    }
}
impl Hash for item::Variadic {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.pat.hash(state);
        self.comma.hash(state);
    }
}
impl Hash for data::Variant {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.ident.hash(state);
        self.fields.hash(state);
        self.discriminant.hash(state);
    }
}
impl Hash for VisRestricted {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.in_.hash(state);
        self.path.hash(state);
    }
}
impl Hash for Visibility {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            Visibility::Public(_) => {
                state.write_u8(0u8);
            },
            Visibility::Restricted(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            Visibility::Inherited => {
                state.write_u8(2u8);
            },
        }
    }
}
impl Hash for gen::Where {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.preds.hash(state);
    }
}
impl Hash for gen::Where::Pred {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            gen::Where::Pred::Life(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            gen::Where::Pred::Type(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
        }
    }
}
