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
impl Hash for AttrStyle {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            AttrStyle::Outer => {
                state.write_u8(0u8);
            },
            AttrStyle::Inner(_) => {
                state.write_u8(1u8);
            },
        }
    }
}
impl Hash for Attribute {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.style.hash(state);
        self.meta.hash(state);
    }
}
impl Hash for ty::BareFnArg {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.name.hash(state);
        self.ty.hash(state);
    }
}
impl Hash for ty::BareVari {
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
impl Hash for BoundLifetimes {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.lifes.hash(state);
    }
}
impl Hash for ConstParam {
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
impl Hash for DataEnum {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.variants.hash(state);
    }
}
impl Hash for DataStruct {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.fields.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for DataUnion {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.fields.hash(state);
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
impl Hash for ExprArray {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.elems.hash(state);
    }
}
impl Hash for ExprAssign {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.left.hash(state);
        self.right.hash(state);
    }
}
impl Hash for ExprAsync {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.capture.hash(state);
        self.block.hash(state);
    }
}
impl Hash for ExprAwait {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.base.hash(state);
    }
}
impl Hash for ExprBinary {
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
impl Hash for ExprBlock {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.label.hash(state);
        self.block.hash(state);
    }
}
impl Hash for ExprBreak {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.label.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for ExprCall {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.func.hash(state);
        self.args.hash(state);
    }
}
impl Hash for ExprCast {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
        self.typ.hash(state);
    }
}
impl Hash for ExprClosure {
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
impl Hash for ExprConst {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.block.hash(state);
    }
}
impl Hash for ExprContinue {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.label.hash(state);
    }
}
impl Hash for ExprField {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.base.hash(state);
        self.member.hash(state);
    }
}
impl Hash for ExprForLoop {
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
impl Hash for ExprGroup {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for ExprIf {
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
impl Hash for ExprIndex {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
        self.index.hash(state);
    }
}
impl Hash for ExprInfer {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
    }
}
impl Hash for ExprLet {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.pat.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for ExprLit {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.lit.hash(state);
    }
}
impl Hash for ExprLoop {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.label.hash(state);
        self.body.hash(state);
    }
}
impl Hash for ExprMacro {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mac.hash(state);
    }
}
impl Hash for ExprMatch {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
        self.arms.hash(state);
    }
}
impl Hash for ExprMethodCall {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.receiver.hash(state);
        self.method.hash(state);
        self.turbofish.hash(state);
        self.args.hash(state);
    }
}
impl Hash for ExprParen {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for ExprPath {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.qself.hash(state);
        self.path.hash(state);
    }
}
impl Hash for ExprRange {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.start.hash(state);
        self.limits.hash(state);
        self.end.hash(state);
    }
}
impl Hash for ExprReference {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mut_.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for ExprRepeat {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
        self.len.hash(state);
    }
}
impl Hash for ExprReturn {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for ExprStruct {
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
impl Hash for ExprTry {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for ExprTryBlock {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.block.hash(state);
    }
}
impl Hash for ExprTuple {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.elems.hash(state);
    }
}
impl Hash for ExprUnary {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.op.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for ExprUnsafe {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.block.hash(state);
    }
}
impl Hash for ExprWhile {
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
impl Hash for ExprYield {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.expr.hash(state);
    }
}
impl Hash for Field {
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
impl Hash for FieldMut {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            FieldMut::None => {
                state.write_u8(0u8);
            },
        }
    }
}
impl Hash for patt::Field {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.member.hash(state);
        self.colon.hash(state);
        self.patt.hash(state);
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
impl Hash for Fields {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            Fields::Named(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            Fields::Unnamed(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            Fields::Unit => {
                state.write_u8(2u8);
            },
        }
    }
}
impl Hash for FieldsNamed {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.named.hash(state);
    }
}
impl Hash for FieldsUnnamed {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.unnamed.hash(state);
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
impl Hash for FnArg {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            FnArg::Receiver(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            FnArg::Typed(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
        }
    }
}
impl Hash for ForeignItem {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            ForeignItem::Fn(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            ForeignItem::Static(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            ForeignItem::Type(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            ForeignItem::Macro(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            ForeignItem::Verbatim(v0) => {
                state.write_u8(4u8);
                TokenStreamHelper(v0).hash(state);
            },
        }
    }
}
impl Hash for ForeignItemFn {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.sig.hash(state);
    }
}
impl Hash for ForeignItemMacro {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mac.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for ForeignItemStatic {
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
impl Hash for ForeignItemType {
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
impl Hash for GenericParam {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            GenericParam::Lifetime(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            GenericParam::Type(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            GenericParam::Const(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
        }
    }
}
impl Hash for Generics {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.lt.hash(state);
        self.params.hash(state);
        self.gt.hash(state);
        self.clause.hash(state);
    }
}
impl Hash for ImplItem {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            ImplItem::Const(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            ImplItem::Fn(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            ImplItem::Type(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            ImplItem::Macro(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            ImplItem::Verbatim(v0) => {
                state.write_u8(4u8);
                TokenStreamHelper(v0).hash(state);
            },
        }
    }
}
impl Hash for ImplItemConst {
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
impl Hash for ImplItemFn {
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
impl Hash for ImplItemMacro {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mac.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for ImplItemType {
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
impl Hash for ImplRestriction {
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
            Item::ForeignMod(v0) => {
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
impl Hash for ItemConst {
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
impl Hash for ItemEnum {
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
impl Hash for ItemExternCrate {
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
impl Hash for ItemFn {
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
impl Hash for ItemForeignMod {
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
impl Hash for ItemImpl {
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
impl Hash for ItemMacro {
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
impl Hash for ItemMod {
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
impl Hash for ItemStatic {
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
impl Hash for ItemStruct {
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
impl Hash for ItemTrait {
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
        self.supertraits.hash(state);
        self.items.hash(state);
    }
}
impl Hash for ItemTraitAlias {
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
impl Hash for ItemType {
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
impl Hash for ItemUnion {
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
impl Hash for ItemUse {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.leading_colon.hash(state);
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
impl Hash for LifetimeParam {
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
impl Hash for Local {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.pat.hash(state);
        self.init.hash(state);
    }
}
impl Hash for LocalInit {
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
impl Hash for MacroDelim {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            MacroDelim::Paren(_) => {
                state.write_u8(0u8);
            },
            MacroDelim::Brace(_) => {
                state.write_u8(1u8);
            },
            MacroDelim::Bracket(_) => {
                state.write_u8(2u8);
            },
        }
    }
}
impl Hash for Meta {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            Meta::Path(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            Meta::List(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            Meta::NameValue(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
        }
    }
}
impl Hash for MetaList {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.path.hash(state);
        self.delim.hash(state);
        TokenStreamHelper(&self.toks).hash(state);
    }
}
impl Hash for MetaNameValue {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.path.hash(state);
        self.val.hash(state);
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
impl Hash for patt::Patt {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            patt::Patt::Const(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            patt::Patt::Ident(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            patt::Patt::Lit(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            patt::Patt::Mac(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            patt::Patt::Or(v0) => {
                state.write_u8(4u8);
                v0.hash(state);
            },
            patt::Patt::Paren(v0) => {
                state.write_u8(5u8);
                v0.hash(state);
            },
            patt::Patt::Path(v0) => {
                state.write_u8(6u8);
                v0.hash(state);
            },
            patt::Patt::Range(v0) => {
                state.write_u8(7u8);
                v0.hash(state);
            },
            patt::Patt::Ref(v0) => {
                state.write_u8(8u8);
                v0.hash(state);
            },
            patt::Patt::Rest(v0) => {
                state.write_u8(9u8);
                v0.hash(state);
            },
            patt::Patt::Slice(v0) => {
                state.write_u8(10u8);
                v0.hash(state);
            },
            patt::Patt::Struct(v0) => {
                state.write_u8(11u8);
                v0.hash(state);
            },
            patt::Patt::Tuple(v0) => {
                state.write_u8(12u8);
                v0.hash(state);
            },
            patt::Patt::TupleStruct(v0) => {
                state.write_u8(13u8);
                v0.hash(state);
            },
            patt::Patt::Type(v0) => {
                state.write_u8(14u8);
                v0.hash(state);
            },
            patt::Patt::Verbatim(v0) => {
                state.write_u8(15u8);
                TokenStreamHelper(v0).hash(state);
            },
            patt::Patt::Wild(v0) => {
                state.write_u8(16u8);
                v0.hash(state);
            },
        }
    }
}
impl Hash for patt::Ident {
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
impl Hash for patt::Or {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vert.hash(state);
        self.cases.hash(state);
    }
}
impl Hash for patt::Paren {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.patt.hash(state);
    }
}
impl Hash for patt::Ref {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mut_.hash(state);
        self.patt.hash(state);
    }
}
impl Hash for patt::Rest {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
    }
}
impl Hash for patt::Slice {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.patts.hash(state);
    }
}
impl Hash for patt::Struct {
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
impl Hash for patt::Tuple {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.patts.hash(state);
    }
}
impl Hash for patt::TupleStruct {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.qself.hash(state);
        self.path.hash(state);
        self.patts.hash(state);
    }
}
impl Hash for patt::Type {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.patt.hash(state);
        self.typ.hash(state);
    }
}
impl Hash for patt::Wild {
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
impl Hash for PredLifetime {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.life.hash(state);
        self.bounds.hash(state);
    }
}
impl Hash for PredType {
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
impl Hash for Receiver {
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
impl Hash for Signature {
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
impl Hash for Stmt {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            Stmt::Local(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            Stmt::Item(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            Stmt::Expr(v0, v1) => {
                state.write_u8(2u8);
                v0.hash(state);
                v1.hash(state);
            },
            Stmt::Macro(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
        }
    }
}
impl Hash for StmtMacro {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mac.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for TraitBound {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.paren.hash(state);
        self.modifier.hash(state);
        self.lifes.hash(state);
        self.path.hash(state);
    }
}
impl Hash for TraitBoundModifier {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            TraitBoundModifier::None => {
                state.write_u8(0u8);
            },
            TraitBoundModifier::Maybe(_) => {
                state.write_u8(1u8);
            },
        }
    }
}
impl Hash for TraitItem {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            TraitItem::Const(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            TraitItem::Fn(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            TraitItem::Type(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            TraitItem::Macro(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            TraitItem::Verbatim(v0) => {
                state.write_u8(4u8);
                TokenStreamHelper(v0).hash(state);
            },
        }
    }
}
impl Hash for TraitItemConst {
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
impl Hash for TraitItemFn {
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
impl Hash for TraitItemMacro {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mac.hash(state);
        self.semi.hash(state);
    }
}
impl Hash for TraitItemType {
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
            ty::Type::BareFn(v0) => {
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
impl Hash for ty::BareFn {
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
impl Hash for TypeParam {
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
impl Hash for TypeParamBound {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            TypeParamBound::Trait(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            TypeParamBound::Lifetime(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            TypeParamBound::Verbatim(v0) => {
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
impl Hash for UseGlob {
    fn hash<H>(&self, _state: &mut H)
    where
        H: Hasher,
    {
    }
}
impl Hash for UseGroup {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.items.hash(state);
    }
}
impl Hash for UseName {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(state);
    }
}
impl Hash for UsePath {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(state);
        self.tree.hash(state);
    }
}
impl Hash for UseRename {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(state);
        self.rename.hash(state);
    }
}
impl Hash for UseTree {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            UseTree::Path(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            UseTree::Name(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            UseTree::Rename(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            UseTree::Glob(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            UseTree::Group(v0) => {
                state.write_u8(4u8);
                v0.hash(state);
            },
        }
    }
}
impl Hash for Variadic {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.pat.hash(state);
        self.comma.hash(state);
    }
}
impl Hash for Variant {
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
impl Hash for WhereClause {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.preds.hash(state);
    }
}
impl Hash for WherePred {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            WherePred::Lifetime(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            WherePred::Type(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
        }
    }
}
