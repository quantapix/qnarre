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
impl Hash for AngleBracketedGenericArguments {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.colon2_token.hash(state);
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
        self.generics.hash(state);
        self.value.hash(state);
    }
}
impl Hash for AssocType {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(state);
        self.generics.hash(state);
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
impl Hash for BareFnArg {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.name.hash(state);
        self.ty.hash(state);
    }
}
impl Hash for BareVariadic {
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
        self.lifetimes.hash(state);
    }
}
impl Hash for ConstParam {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.ident.hash(state);
        self.ty.hash(state);
        self.eq_token.hash(state);
        self.default.hash(state);
    }
}
impl Hash for Constraint {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(state);
        self.generics.hash(state);
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
        self.semi_token.hash(state);
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
        self.generics.hash(state);
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
        self.ty.hash(state);
    }
}
impl Hash for ExprClosure {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.lifetimes.hash(state);
        self.constness.hash(state);
        self.movability.hash(state);
        self.asyncness.hash(state);
        self.capture.hash(state);
        self.inputs.hash(state);
        self.output.hash(state);
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
        self.mutability.hash(state);
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
        self.dot2_token.hash(state);
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
        self.colon_token.hash(state);
        self.ty.hash(state);
    }
}
impl Hash for FieldMutability {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            FieldMutability::None => {
                state.write_u8(0u8);
            },
        }
    }
}
impl Hash for FieldPat {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.member.hash(state);
        self.colon_token.hash(state);
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
        self.colon_token.hash(state);
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
        self.semi_token.hash(state);
    }
}
impl Hash for ForeignItemStatic {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.mutability.hash(state);
        self.ident.hash(state);
        self.ty.hash(state);
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
        self.generics.hash(state);
    }
}
impl Hash for GenericArgument {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            GenericArgument::Lifetime(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            GenericArgument::Type(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            GenericArgument::Const(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            GenericArgument::AssocType(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            GenericArgument::AssocConst(v0) => {
                state.write_u8(4u8);
                v0.hash(state);
            },
            GenericArgument::Constraint(v0) => {
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
        self.lt_token.hash(state);
        self.params.hash(state);
        self.gt_token.hash(state);
        self.where_clause.hash(state);
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
        self.defaultness.hash(state);
        self.ident.hash(state);
        self.generics.hash(state);
        self.ty.hash(state);
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
        self.defaultness.hash(state);
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
        self.semi_token.hash(state);
    }
}
impl Hash for ImplItemType {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.defaultness.hash(state);
        self.ident.hash(state);
        self.generics.hash(state);
        self.ty.hash(state);
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
        self.generics.hash(state);
        self.ty.hash(state);
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
        self.generics.hash(state);
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
        self.unsafety.hash(state);
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
        self.defaultness.hash(state);
        self.unsafety.hash(state);
        self.generics.hash(state);
        self.trait_.hash(state);
        self.self_ty.hash(state);
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
        self.semi_token.hash(state);
    }
}
impl Hash for ItemMod {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.unsafety.hash(state);
        self.ident.hash(state);
        self.content.hash(state);
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
        self.mutability.hash(state);
        self.ident.hash(state);
        self.ty.hash(state);
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
        self.generics.hash(state);
        self.fields.hash(state);
        self.semi_token.hash(state);
    }
}
impl Hash for ItemTrait {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.vis.hash(state);
        self.unsafety.hash(state);
        self.auto_token.hash(state);
        self.restriction.hash(state);
        self.ident.hash(state);
        self.generics.hash(state);
        self.colon_token.hash(state);
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
        self.generics.hash(state);
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
        self.generics.hash(state);
        self.ty.hash(state);
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
        self.generics.hash(state);
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
        self.lifetime.hash(state);
        self.colon_token.hash(state);
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
impl Hash for LitBool {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.value.hash(state);
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
        self.delimiter.hash(state);
        TokenStreamHelper(&self.tokens).hash(state);
    }
}
impl Hash for MacroDelimiter {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            MacroDelimiter::Paren(_) => {
                state.write_u8(0u8);
            },
            MacroDelimiter::Brace(_) => {
                state.write_u8(1u8);
            },
            MacroDelimiter::Bracket(_) => {
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
        self.delimiter.hash(state);
        TokenStreamHelper(&self.tokens).hash(state);
    }
}
impl Hash for MetaNameValue {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.path.hash(state);
        self.value.hash(state);
    }
}
impl Hash for ParenthesizedGenericArguments {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.inputs.hash(state);
        self.output.hash(state);
    }
}
impl Hash for Pat {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            Pat::Const(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            Pat::Ident(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            Pat::Lit(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            Pat::Macro(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            Pat::Or(v0) => {
                state.write_u8(4u8);
                v0.hash(state);
            },
            Pat::Paren(v0) => {
                state.write_u8(5u8);
                v0.hash(state);
            },
            Pat::Path(v0) => {
                state.write_u8(6u8);
                v0.hash(state);
            },
            Pat::Range(v0) => {
                state.write_u8(7u8);
                v0.hash(state);
            },
            Pat::Reference(v0) => {
                state.write_u8(8u8);
                v0.hash(state);
            },
            Pat::Rest(v0) => {
                state.write_u8(9u8);
                v0.hash(state);
            },
            Pat::Slice(v0) => {
                state.write_u8(10u8);
                v0.hash(state);
            },
            Pat::Struct(v0) => {
                state.write_u8(11u8);
                v0.hash(state);
            },
            Pat::Tuple(v0) => {
                state.write_u8(12u8);
                v0.hash(state);
            },
            Pat::TupleStruct(v0) => {
                state.write_u8(13u8);
                v0.hash(state);
            },
            Pat::Type(v0) => {
                state.write_u8(14u8);
                v0.hash(state);
            },
            Pat::Verbatim(v0) => {
                state.write_u8(15u8);
                TokenStreamHelper(v0).hash(state);
            },
            Pat::Wild(v0) => {
                state.write_u8(16u8);
                v0.hash(state);
            },
        }
    }
}
impl Hash for PatIdent {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.by_ref.hash(state);
        self.mutability.hash(state);
        self.ident.hash(state);
        self.subpat.hash(state);
    }
}
impl Hash for PatOr {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.leading_vert.hash(state);
        self.cases.hash(state);
    }
}
impl Hash for PatParen {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.pat.hash(state);
    }
}
impl Hash for PatReference {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mutability.hash(state);
        self.pat.hash(state);
    }
}
impl Hash for PatRest {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
    }
}
impl Hash for PatSlice {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.elems.hash(state);
    }
}
impl Hash for PatStruct {
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
impl Hash for PatTuple {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.elems.hash(state);
    }
}
impl Hash for PatTupleStruct {
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
impl Hash for PatType {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.pat.hash(state);
        self.ty.hash(state);
    }
}
impl Hash for PatWild {
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
        self.leading_colon.hash(state);
        self.segments.hash(state);
    }
}
impl Hash for PathArguments {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            PathArguments::None => {
                state.write_u8(0u8);
            },
            PathArguments::AngleBracketed(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            PathArguments::Parenthesized(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
        }
    }
}
impl Hash for PathSegment {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ident.hash(state);
        self.arguments.hash(state);
    }
}
impl Hash for PredicateLifetime {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.lifetime.hash(state);
        self.bounds.hash(state);
    }
}
impl Hash for PredicateType {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.lifetimes.hash(state);
        self.bounded_ty.hash(state);
        self.bounds.hash(state);
    }
}
impl Hash for QSelf {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.ty.hash(state);
        self.position.hash(state);
        self.as_token.hash(state);
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
        self.mutability.hash(state);
        self.colon_token.hash(state);
        self.ty.hash(state);
    }
}
impl Hash for ReturnType {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            ReturnType::Default => {
                state.write_u8(0u8);
            },
            ReturnType::Type(_, v1) => {
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
        self.asyncness.hash(state);
        self.unsafety.hash(state);
        self.abi.hash(state);
        self.ident.hash(state);
        self.generics.hash(state);
        self.inputs.hash(state);
        self.variadic.hash(state);
        self.output.hash(state);
    }
}
impl Hash for StaticMutability {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            StaticMutability::Mut(_) => {
                state.write_u8(0u8);
            },
            StaticMutability::None => {
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
        self.semi_token.hash(state);
    }
}
impl Hash for TraitBound {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.paren_token.hash(state);
        self.modifier.hash(state);
        self.lifetimes.hash(state);
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
        self.generics.hash(state);
        self.ty.hash(state);
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
        self.semi_token.hash(state);
    }
}
impl Hash for TraitItemMacro {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.mac.hash(state);
        self.semi_token.hash(state);
    }
}
impl Hash for TraitItemType {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.attrs.hash(state);
        self.ident.hash(state);
        self.generics.hash(state);
        self.colon_token.hash(state);
        self.bounds.hash(state);
        self.default.hash(state);
    }
}
impl Hash for Type {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            Type::Array(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            Type::BareFn(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
            Type::Group(v0) => {
                state.write_u8(2u8);
                v0.hash(state);
            },
            Type::ImplTrait(v0) => {
                state.write_u8(3u8);
                v0.hash(state);
            },
            Type::Infer(v0) => {
                state.write_u8(4u8);
                v0.hash(state);
            },
            Type::Macro(v0) => {
                state.write_u8(5u8);
                v0.hash(state);
            },
            Type::Never(v0) => {
                state.write_u8(6u8);
                v0.hash(state);
            },
            Type::Paren(v0) => {
                state.write_u8(7u8);
                v0.hash(state);
            },
            Type::Path(v0) => {
                state.write_u8(8u8);
                v0.hash(state);
            },
            Type::Ptr(v0) => {
                state.write_u8(9u8);
                v0.hash(state);
            },
            Type::Reference(v0) => {
                state.write_u8(10u8);
                v0.hash(state);
            },
            Type::Slice(v0) => {
                state.write_u8(11u8);
                v0.hash(state);
            },
            Type::TraitObject(v0) => {
                state.write_u8(12u8);
                v0.hash(state);
            },
            Type::Tuple(v0) => {
                state.write_u8(13u8);
                v0.hash(state);
            },
            Type::Verbatim(v0) => {
                state.write_u8(14u8);
                TokenStreamHelper(v0).hash(state);
            },
        }
    }
}
impl Hash for TypeArray {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(state);
        self.len.hash(state);
    }
}
impl Hash for TypeBareFn {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.lifetimes.hash(state);
        self.unsafety.hash(state);
        self.abi.hash(state);
        self.inputs.hash(state);
        self.variadic.hash(state);
        self.output.hash(state);
    }
}
impl Hash for TypeGroup {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(state);
    }
}
impl Hash for TypeImplTrait {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.bounds.hash(state);
    }
}
impl Hash for TypeInfer {
    fn hash<H>(&self, _state: &mut H)
    where
        H: Hasher,
    {
    }
}
impl Hash for TypeMacro {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.mac.hash(state);
    }
}
impl Hash for TypeNever {
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
        self.colon_token.hash(state);
        self.bounds.hash(state);
        self.eq_token.hash(state);
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
impl Hash for TypeParen {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(state);
    }
}
impl Hash for TypePath {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.qself.hash(state);
        self.path.hash(state);
    }
}
impl Hash for TypePtr {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.const_token.hash(state);
        self.mutability.hash(state);
        self.elem.hash(state);
    }
}
impl Hash for TypeReference {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.lifetime.hash(state);
        self.mutability.hash(state);
        self.elem.hash(state);
    }
}
impl Hash for TypeSlice {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.elem.hash(state);
    }
}
impl Hash for TypeTraitObject {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.dyn_token.hash(state);
        self.bounds.hash(state);
    }
}
impl Hash for TypeTuple {
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
        self.in_token.hash(state);
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
        self.predicates.hash(state);
    }
}
impl Hash for WherePredicate {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            WherePredicate::Lifetime(v0) => {
                state.write_u8(0u8);
                v0.hash(state);
            },
            WherePredicate::Type(v0) => {
                state.write_u8(1u8);
                v0.hash(state);
            },
        }
    }
}
