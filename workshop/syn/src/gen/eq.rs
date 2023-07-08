use crate::TokenStreamHelper;
use crate::*;
impl Eq for Abi {}
impl PartialEq for Abi {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl Eq for AngleBracketedGenericArguments {}
impl PartialEq for AngleBracketedGenericArguments {
    fn eq(&self, other: &Self) -> bool {
        self.colon2_token == other.colon2_token && self.args == other.args
    }
}
impl Eq for Arm {}
impl PartialEq for Arm {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.pat == other.pat
            && self.guard == other.guard
            && self.body == other.body
            && self.comma == other.comma
    }
}
impl Eq for AssocConst {}
impl PartialEq for AssocConst {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident && self.generics == other.generics && self.value == other.value
    }
}
impl Eq for AssocType {}
impl PartialEq for AssocType {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident && self.generics == other.generics && self.ty == other.ty
    }
}
impl Eq for AttrStyle {}
impl PartialEq for AttrStyle {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (AttrStyle::Outer, AttrStyle::Outer) => true,
            (AttrStyle::Inner(_), AttrStyle::Inner(_)) => true,
            _ => false,
        }
    }
}
impl Eq for Attribute {}
impl PartialEq for Attribute {
    fn eq(&self, other: &Self) -> bool {
        self.style == other.style && self.meta == other.meta
    }
}
impl Eq for BareFnArg {}
impl PartialEq for BareFnArg {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.name == other.name && self.ty == other.ty
    }
}
impl Eq for BareVariadic {}
impl PartialEq for BareVariadic {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.name == other.name && self.comma == other.comma
    }
}
impl Eq for BinOp {}
impl PartialEq for BinOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BinOp::Add(_), BinOp::Add(_)) => true,
            (BinOp::Sub(_), BinOp::Sub(_)) => true,
            (BinOp::Mul(_), BinOp::Mul(_)) => true,
            (BinOp::Div(_), BinOp::Div(_)) => true,
            (BinOp::Rem(_), BinOp::Rem(_)) => true,
            (BinOp::And(_), BinOp::And(_)) => true,
            (BinOp::Or(_), BinOp::Or(_)) => true,
            (BinOp::BitXor(_), BinOp::BitXor(_)) => true,
            (BinOp::BitAnd(_), BinOp::BitAnd(_)) => true,
            (BinOp::BitOr(_), BinOp::BitOr(_)) => true,
            (BinOp::Shl(_), BinOp::Shl(_)) => true,
            (BinOp::Shr(_), BinOp::Shr(_)) => true,
            (BinOp::Eq(_), BinOp::Eq(_)) => true,
            (BinOp::Lt(_), BinOp::Lt(_)) => true,
            (BinOp::Le(_), BinOp::Le(_)) => true,
            (BinOp::Ne(_), BinOp::Ne(_)) => true,
            (BinOp::Ge(_), BinOp::Ge(_)) => true,
            (BinOp::Gt(_), BinOp::Gt(_)) => true,
            (BinOp::AddAssign(_), BinOp::AddAssign(_)) => true,
            (BinOp::SubAssign(_), BinOp::SubAssign(_)) => true,
            (BinOp::MulAssign(_), BinOp::MulAssign(_)) => true,
            (BinOp::DivAssign(_), BinOp::DivAssign(_)) => true,
            (BinOp::RemAssign(_), BinOp::RemAssign(_)) => true,
            (BinOp::BitXorAssign(_), BinOp::BitXorAssign(_)) => true,
            (BinOp::BitAndAssign(_), BinOp::BitAndAssign(_)) => true,
            (BinOp::BitOrAssign(_), BinOp::BitOrAssign(_)) => true,
            (BinOp::ShlAssign(_), BinOp::ShlAssign(_)) => true,
            (BinOp::ShrAssign(_), BinOp::ShrAssign(_)) => true,
            _ => false,
        }
    }
}
impl Eq for Block {}
impl PartialEq for Block {
    fn eq(&self, other: &Self) -> bool {
        self.stmts == other.stmts
    }
}
impl Eq for BoundLifetimes {}
impl PartialEq for BoundLifetimes {
    fn eq(&self, other: &Self) -> bool {
        self.lifetimes == other.lifetimes
    }
}
impl Eq for ConstParam {}
impl PartialEq for ConstParam {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ident == other.ident
            && self.ty == other.ty
            && self.eq == other.eq
            && self.default == other.default
    }
}
impl Eq for Constraint {}
impl PartialEq for Constraint {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident && self.generics == other.generics && self.bounds == other.bounds
    }
}
impl Eq for Data {}
impl PartialEq for Data {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Data::Struct(self0), Data::Struct(other0)) => self0 == other0,
            (Data::Enum(self0), Data::Enum(other0)) => self0 == other0,
            (Data::Union(self0), Data::Union(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for DataEnum {}
impl PartialEq for DataEnum {
    fn eq(&self, other: &Self) -> bool {
        self.variants == other.variants
    }
}
impl Eq for DataStruct {}
impl PartialEq for DataStruct {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields && self.semi == other.semi
    }
}
impl Eq for DataUnion {}
impl PartialEq for DataUnion {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}
impl Eq for DeriveInput {}
impl PartialEq for DeriveInput {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.generics == other.generics
            && self.data == other.data
    }
}
impl Eq for Expr {}
impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Expr::Array(self0), Expr::Array(other0)) => self0 == other0,
            (Expr::Assign(self0), Expr::Assign(other0)) => self0 == other0,
            (Expr::Async(self0), Expr::Async(other0)) => self0 == other0,
            (Expr::Await(self0), Expr::Await(other0)) => self0 == other0,
            (Expr::Binary(self0), Expr::Binary(other0)) => self0 == other0,
            (Expr::Block(self0), Expr::Block(other0)) => self0 == other0,
            (Expr::Break(self0), Expr::Break(other0)) => self0 == other0,
            (Expr::Call(self0), Expr::Call(other0)) => self0 == other0,
            (Expr::Cast(self0), Expr::Cast(other0)) => self0 == other0,
            (Expr::Closure(self0), Expr::Closure(other0)) => self0 == other0,
            (Expr::Const(self0), Expr::Const(other0)) => self0 == other0,
            (Expr::Continue(self0), Expr::Continue(other0)) => self0 == other0,
            (Expr::Field(self0), Expr::Field(other0)) => self0 == other0,
            (Expr::ForLoop(self0), Expr::ForLoop(other0)) => self0 == other0,
            (Expr::Group(self0), Expr::Group(other0)) => self0 == other0,
            (Expr::If(self0), Expr::If(other0)) => self0 == other0,
            (Expr::Index(self0), Expr::Index(other0)) => self0 == other0,
            (Expr::Infer(self0), Expr::Infer(other0)) => self0 == other0,
            (Expr::Let(self0), Expr::Let(other0)) => self0 == other0,
            (Expr::Lit(self0), Expr::Lit(other0)) => self0 == other0,
            (Expr::Loop(self0), Expr::Loop(other0)) => self0 == other0,
            (Expr::Macro(self0), Expr::Macro(other0)) => self0 == other0,
            (Expr::Match(self0), Expr::Match(other0)) => self0 == other0,
            (Expr::MethodCall(self0), Expr::MethodCall(other0)) => self0 == other0,
            (Expr::Paren(self0), Expr::Paren(other0)) => self0 == other0,
            (Expr::Path(self0), Expr::Path(other0)) => self0 == other0,
            (Expr::Range(self0), Expr::Range(other0)) => self0 == other0,
            (Expr::Reference(self0), Expr::Reference(other0)) => self0 == other0,
            (Expr::Repeat(self0), Expr::Repeat(other0)) => self0 == other0,
            (Expr::Return(self0), Expr::Return(other0)) => self0 == other0,
            (Expr::Struct(self0), Expr::Struct(other0)) => self0 == other0,
            (Expr::Try(self0), Expr::Try(other0)) => self0 == other0,
            (Expr::TryBlock(self0), Expr::TryBlock(other0)) => self0 == other0,
            (Expr::Tuple(self0), Expr::Tuple(other0)) => self0 == other0,
            (Expr::Unary(self0), Expr::Unary(other0)) => self0 == other0,
            (Expr::Unsafe(self0), Expr::Unsafe(other0)) => self0 == other0,
            (Expr::Verbatim(self0), Expr::Verbatim(other0)) => TokenStreamHelper(self0) == TokenStreamHelper(other0),
            (Expr::While(self0), Expr::While(other0)) => self0 == other0,
            (Expr::Yield(self0), Expr::Yield(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for ExprArray {}
impl PartialEq for ExprArray {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.elems == other.elems
    }
}
impl Eq for ExprAssign {}
impl PartialEq for ExprAssign {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.left == other.left && self.right == other.right
    }
}
impl Eq for ExprAsync {}
impl PartialEq for ExprAsync {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.capture == other.capture && self.block == other.block
    }
}
impl Eq for ExprAwait {}
impl PartialEq for ExprAwait {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.base == other.base
    }
}
impl Eq for ExprBinary {}
impl PartialEq for ExprBinary {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.left == other.left && self.op == other.op && self.right == other.right
    }
}
impl Eq for ExprBlock {}
impl PartialEq for ExprBlock {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.label == other.label && self.block == other.block
    }
}
impl Eq for ExprBreak {}
impl PartialEq for ExprBreak {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.label == other.label && self.expr == other.expr
    }
}
impl Eq for ExprCall {}
impl PartialEq for ExprCall {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.func == other.func && self.args == other.args
    }
}
impl Eq for ExprCast {}
impl PartialEq for ExprCast {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr && self.ty == other.ty
    }
}
impl Eq for ExprClosure {}
impl PartialEq for ExprClosure {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.lifetimes == other.lifetimes
            && self.constness == other.constness
            && self.movability == other.movability
            && self.asyncness == other.asyncness
            && self.capture == other.capture
            && self.inputs == other.inputs
            && self.output == other.output
            && self.body == other.body
    }
}
impl Eq for ExprConst {}
impl PartialEq for ExprConst {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.block == other.block
    }
}
impl Eq for ExprContinue {}
impl PartialEq for ExprContinue {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.label == other.label
    }
}
impl Eq for ExprField {}
impl PartialEq for ExprField {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.base == other.base && self.member == other.member
    }
}
impl Eq for ExprForLoop {}
impl PartialEq for ExprForLoop {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.label == other.label
            && self.pat == other.pat
            && self.expr == other.expr
            && self.body == other.body
    }
}
impl Eq for ExprGroup {}
impl PartialEq for ExprGroup {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr
    }
}
impl Eq for ExprIf {}
impl PartialEq for ExprIf {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.cond == other.cond
            && self.then_branch == other.then_branch
            && self.else_branch == other.else_branch
    }
}
impl Eq for ExprIndex {}
impl PartialEq for ExprIndex {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr && self.index == other.index
    }
}
impl Eq for ExprInfer {}
impl PartialEq for ExprInfer {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
    }
}
impl Eq for ExprLet {}
impl PartialEq for ExprLet {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.pat == other.pat && self.expr == other.expr
    }
}
impl Eq for ExprLit {}
impl PartialEq for ExprLit {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.lit == other.lit
    }
}
impl Eq for ExprLoop {}
impl PartialEq for ExprLoop {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.label == other.label && self.body == other.body
    }
}
impl Eq for ExprMacro {}
impl PartialEq for ExprMacro {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mac == other.mac
    }
}
impl Eq for ExprMatch {}
impl PartialEq for ExprMatch {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr && self.arms == other.arms
    }
}
impl Eq for ExprMethodCall {}
impl PartialEq for ExprMethodCall {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.receiver == other.receiver
            && self.method == other.method
            && self.turbofish == other.turbofish
            && self.args == other.args
    }
}
impl Eq for ExprParen {}
impl PartialEq for ExprParen {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr
    }
}
impl Eq for ExprPath {}
impl PartialEq for ExprPath {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.qself == other.qself && self.path == other.path
    }
}
impl Eq for ExprRange {}
impl PartialEq for ExprRange {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.start == other.start && self.limits == other.limits && self.end == other.end
    }
}
impl Eq for ExprReference {}
impl PartialEq for ExprReference {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mutability == other.mutability && self.expr == other.expr
    }
}
impl Eq for ExprRepeat {}
impl PartialEq for ExprRepeat {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr && self.len == other.len
    }
}
impl Eq for ExprReturn {}
impl PartialEq for ExprReturn {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr
    }
}
impl Eq for ExprStruct {}
impl PartialEq for ExprStruct {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.qself == other.qself
            && self.path == other.path
            && self.fields == other.fields
            && self.dot2_token == other.dot2_token
            && self.rest == other.rest
    }
}
impl Eq for ExprTry {}
impl PartialEq for ExprTry {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr
    }
}
impl Eq for ExprTryBlock {}
impl PartialEq for ExprTryBlock {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.block == other.block
    }
}
impl Eq for ExprTuple {}
impl PartialEq for ExprTuple {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.elems == other.elems
    }
}
impl Eq for ExprUnary {}
impl PartialEq for ExprUnary {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.op == other.op && self.expr == other.expr
    }
}
impl Eq for ExprUnsafe {}
impl PartialEq for ExprUnsafe {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.block == other.block
    }
}
impl Eq for ExprWhile {}
impl PartialEq for ExprWhile {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.label == other.label && self.cond == other.cond && self.body == other.body
    }
}
impl Eq for ExprYield {}
impl PartialEq for ExprYield {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr
    }
}
impl Eq for Field {}
impl PartialEq for Field {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.mutability == other.mutability
            && self.ident == other.ident
            && self.colon_token == other.colon_token
            && self.ty == other.ty
    }
}
impl Eq for FieldMut {}
impl PartialEq for FieldMut {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (FieldMut::None, FieldMut::None) => true,
        }
    }
}
impl Eq for FieldPat {}
impl PartialEq for FieldPat {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.member == other.member && self.colon == other.colon && self.pat == other.pat
    }
}
impl Eq for FieldValue {}
impl PartialEq for FieldValue {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.member == other.member
            && self.colon_token == other.colon_token
            && self.expr == other.expr
    }
}
impl Eq for Fields {}
impl PartialEq for Fields {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Fields::Named(self0), Fields::Named(other0)) => self0 == other0,
            (Fields::Unnamed(self0), Fields::Unnamed(other0)) => self0 == other0,
            (Fields::Unit, Fields::Unit) => true,
            _ => false,
        }
    }
}
impl Eq for FieldsNamed {}
impl PartialEq for FieldsNamed {
    fn eq(&self, other: &Self) -> bool {
        self.named == other.named
    }
}
impl Eq for FieldsUnnamed {}
impl PartialEq for FieldsUnnamed {
    fn eq(&self, other: &Self) -> bool {
        self.unnamed == other.unnamed
    }
}
impl Eq for File {}
impl PartialEq for File {
    fn eq(&self, other: &Self) -> bool {
        self.shebang == other.shebang && self.attrs == other.attrs && self.items == other.items
    }
}
impl Eq for FnArg {}
impl PartialEq for FnArg {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (FnArg::Receiver(self0), FnArg::Receiver(other0)) => self0 == other0,
            (FnArg::Typed(self0), FnArg::Typed(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for ForeignItem {}
impl PartialEq for ForeignItem {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ForeignItem::Fn(self0), ForeignItem::Fn(other0)) => self0 == other0,
            (ForeignItem::Static(self0), ForeignItem::Static(other0)) => self0 == other0,
            (ForeignItem::Type(self0), ForeignItem::Type(other0)) => self0 == other0,
            (ForeignItem::Macro(self0), ForeignItem::Macro(other0)) => self0 == other0,
            (ForeignItem::Verbatim(self0), ForeignItem::Verbatim(other0)) => {
                TokenStreamHelper(self0) == TokenStreamHelper(other0)
            },
            _ => false,
        }
    }
}
impl Eq for ForeignItemFn {}
impl PartialEq for ForeignItemFn {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.vis == other.vis && self.sig == other.sig
    }
}
impl Eq for ForeignItemMacro {}
impl PartialEq for ForeignItemMacro {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mac == other.mac && self.semi_token == other.semi_token
    }
}
impl Eq for ForeignItemStatic {}
impl PartialEq for ForeignItemStatic {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.mutability == other.mutability
            && self.ident == other.ident
            && self.ty == other.ty
    }
}
impl Eq for ForeignItemType {}
impl PartialEq for ForeignItemType {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.generics == other.generics
    }
}
impl Eq for GenericArgument {}
impl PartialEq for GenericArgument {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (GenericArgument::Lifetime(self0), GenericArgument::Lifetime(other0)) => self0 == other0,
            (GenericArgument::Type(self0), GenericArgument::Type(other0)) => self0 == other0,
            (GenericArgument::Const(self0), GenericArgument::Const(other0)) => self0 == other0,
            (GenericArgument::AssocType(self0), GenericArgument::AssocType(other0)) => self0 == other0,
            (GenericArgument::AssocConst(self0), GenericArgument::AssocConst(other0)) => self0 == other0,
            (GenericArgument::Constraint(self0), GenericArgument::Constraint(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for GenericParam {}
impl PartialEq for GenericParam {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (GenericParam::Lifetime(self0), GenericParam::Lifetime(other0)) => self0 == other0,
            (GenericParam::Type(self0), GenericParam::Type(other0)) => self0 == other0,
            (GenericParam::Const(self0), GenericParam::Const(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for Generics {}
impl PartialEq for Generics {
    fn eq(&self, other: &Self) -> bool {
        self.lt == other.lt && self.params == other.params && self.gt == other.gt && self.clause == other.clause
    }
}
impl Eq for ImplItem {}
impl PartialEq for ImplItem {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ImplItem::Const(self0), ImplItem::Const(other0)) => self0 == other0,
            (ImplItem::Fn(self0), ImplItem::Fn(other0)) => self0 == other0,
            (ImplItem::Type(self0), ImplItem::Type(other0)) => self0 == other0,
            (ImplItem::Macro(self0), ImplItem::Macro(other0)) => self0 == other0,
            (ImplItem::Verbatim(self0), ImplItem::Verbatim(other0)) => {
                TokenStreamHelper(self0) == TokenStreamHelper(other0)
            },
            _ => false,
        }
    }
}
impl Eq for ImplItemConst {}
impl PartialEq for ImplItemConst {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.defaultness == other.defaultness
            && self.ident == other.ident
            && self.generics == other.generics
            && self.ty == other.ty
            && self.expr == other.expr
    }
}
impl Eq for ImplItemFn {}
impl PartialEq for ImplItemFn {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.defaultness == other.defaultness
            && self.sig == other.sig
            && self.block == other.block
    }
}
impl Eq for ImplItemMacro {}
impl PartialEq for ImplItemMacro {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mac == other.mac && self.semi_token == other.semi_token
    }
}
impl Eq for ImplItemType {}
impl PartialEq for ImplItemType {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.defaultness == other.defaultness
            && self.ident == other.ident
            && self.generics == other.generics
            && self.ty == other.ty
    }
}
impl Eq for ImplRestriction {}
impl PartialEq for ImplRestriction {
    fn eq(&self, _other: &Self) -> bool {
        match *self {}
    }
}
impl Eq for Item {}
impl PartialEq for Item {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Item::Const(self0), Item::Const(other0)) => self0 == other0,
            (Item::Enum(self0), Item::Enum(other0)) => self0 == other0,
            (Item::ExternCrate(self0), Item::ExternCrate(other0)) => self0 == other0,
            (Item::Fn(self0), Item::Fn(other0)) => self0 == other0,
            (Item::ForeignMod(self0), Item::ForeignMod(other0)) => self0 == other0,
            (Item::Impl(self0), Item::Impl(other0)) => self0 == other0,
            (Item::Macro(self0), Item::Macro(other0)) => self0 == other0,
            (Item::Mod(self0), Item::Mod(other0)) => self0 == other0,
            (Item::Static(self0), Item::Static(other0)) => self0 == other0,
            (Item::Struct(self0), Item::Struct(other0)) => self0 == other0,
            (Item::Trait(self0), Item::Trait(other0)) => self0 == other0,
            (Item::TraitAlias(self0), Item::TraitAlias(other0)) => self0 == other0,
            (Item::Type(self0), Item::Type(other0)) => self0 == other0,
            (Item::Union(self0), Item::Union(other0)) => self0 == other0,
            (Item::Use(self0), Item::Use(other0)) => self0 == other0,
            (Item::Verbatim(self0), Item::Verbatim(other0)) => TokenStreamHelper(self0) == TokenStreamHelper(other0),
            _ => false,
        }
    }
}
impl Eq for ItemConst {}
impl PartialEq for ItemConst {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.generics == other.generics
            && self.ty == other.ty
            && self.expr == other.expr
    }
}
impl Eq for ItemEnum {}
impl PartialEq for ItemEnum {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.generics == other.generics
            && self.variants == other.variants
    }
}
impl Eq for ItemExternCrate {}
impl PartialEq for ItemExternCrate {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.vis == other.vis && self.ident == other.ident && self.rename == other.rename
    }
}
impl Eq for ItemFn {}
impl PartialEq for ItemFn {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.vis == other.vis && self.sig == other.sig && self.block == other.block
    }
}
impl Eq for ItemForeignMod {}
impl PartialEq for ItemForeignMod {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.unsafety == other.unsafety
            && self.abi == other.abi
            && self.items == other.items
    }
}
impl Eq for ItemImpl {}
impl PartialEq for ItemImpl {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.defaultness == other.defaultness
            && self.unsafety == other.unsafety
            && self.generics == other.generics
            && self.trait_ == other.trait_
            && self.self_ty == other.self_ty
            && self.items == other.items
    }
}
impl Eq for ItemMacro {}
impl PartialEq for ItemMacro {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ident == other.ident
            && self.mac == other.mac
            && self.semi_token == other.semi_token
    }
}
impl Eq for ItemMod {}
impl PartialEq for ItemMod {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.unsafety == other.unsafety
            && self.ident == other.ident
            && self.content == other.content
            && self.semi == other.semi
    }
}
impl Eq for ItemStatic {}
impl PartialEq for ItemStatic {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.mutability == other.mutability
            && self.ident == other.ident
            && self.ty == other.ty
            && self.expr == other.expr
    }
}
impl Eq for ItemStruct {}
impl PartialEq for ItemStruct {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.generics == other.generics
            && self.fields == other.fields
            && self.semi_token == other.semi_token
    }
}
impl Eq for ItemTrait {}
impl PartialEq for ItemTrait {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.unsafety == other.unsafety
            && self.auto_token == other.auto_token
            && self.restriction == other.restriction
            && self.ident == other.ident
            && self.generics == other.generics
            && self.colon_token == other.colon_token
            && self.supertraits == other.supertraits
            && self.items == other.items
    }
}
impl Eq for ItemTraitAlias {}
impl PartialEq for ItemTraitAlias {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.generics == other.generics
            && self.bounds == other.bounds
    }
}
impl Eq for ItemType {}
impl PartialEq for ItemType {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.generics == other.generics
            && self.ty == other.ty
    }
}
impl Eq for ItemUnion {}
impl PartialEq for ItemUnion {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.generics == other.generics
            && self.fields == other.fields
    }
}
impl Eq for ItemUse {}
impl PartialEq for ItemUse {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.leading_colon == other.leading_colon
            && self.tree == other.tree
    }
}
impl Eq for Label {}
impl PartialEq for Label {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl Eq for LifetimeParam {}
impl PartialEq for LifetimeParam {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.lifetime == other.lifetime
            && self.colon == other.colon
            && self.bounds == other.bounds
    }
}
impl Eq for Lit {}
impl PartialEq for Lit {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Lit::Str(self0), Lit::Str(other0)) => self0 == other0,
            (Lit::ByteStr(self0), Lit::ByteStr(other0)) => self0 == other0,
            (Lit::Byte(self0), Lit::Byte(other0)) => self0 == other0,
            (Lit::Char(self0), Lit::Char(other0)) => self0 == other0,
            (Lit::Int(self0), Lit::Int(other0)) => self0 == other0,
            (Lit::Float(self0), Lit::Float(other0)) => self0 == other0,
            (Lit::Bool(self0), Lit::Bool(other0)) => self0 == other0,
            (Lit::Verbatim(self0), Lit::Verbatim(other0)) => self0.to_string() == other0.to_string(),
            _ => false,
        }
    }
}
impl Eq for lit::Bool {}
impl PartialEq for lit::Bool {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}
impl Eq for lit::Byte {}
impl Eq for lit::ByteStr {}
impl Eq for lit::Char {}
impl Eq for lit::Float {}
impl Eq for lit::Int {}
impl Eq for lit::Str {}
impl Eq for Local {}
impl PartialEq for Local {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.pat == other.pat && self.init == other.init
    }
}
impl Eq for LocalInit {}
impl PartialEq for LocalInit {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr && self.diverge == other.diverge
    }
}
impl Eq for Macro {}
impl PartialEq for Macro {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.delim == other.delim
            && TokenStreamHelper(&self.toks) == TokenStreamHelper(&other.toks)
    }
}
impl Eq for MacroDelimiter {}
impl PartialEq for MacroDelimiter {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MacroDelimiter::Paren(_), MacroDelimiter::Paren(_)) => true,
            (MacroDelimiter::Brace(_), MacroDelimiter::Brace(_)) => true,
            (MacroDelimiter::Bracket(_), MacroDelimiter::Bracket(_)) => true,
            _ => false,
        }
    }
}
impl Eq for Meta {}
impl PartialEq for Meta {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Meta::Path(self0), Meta::Path(other0)) => self0 == other0,
            (Meta::List(self0), Meta::List(other0)) => self0 == other0,
            (Meta::NameValue(self0), Meta::NameValue(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for MetaList {}
impl PartialEq for MetaList {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.delim == other.delim
            && TokenStreamHelper(&self.toks) == TokenStreamHelper(&other.toks)
    }
}
impl Eq for MetaNameValue {}
impl PartialEq for MetaNameValue {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path && self.val == other.val
    }
}
impl Eq for ParenthesizedGenericArguments {}
impl PartialEq for ParenthesizedGenericArguments {
    fn eq(&self, other: &Self) -> bool {
        self.inputs == other.inputs && self.output == other.output
    }
}
impl Eq for Pat {}
impl PartialEq for Pat {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Pat::Const(self0), Pat::Const(other0)) => self0 == other0,
            (Pat::Ident(self0), Pat::Ident(other0)) => self0 == other0,
            (Pat::Lit(self0), Pat::Lit(other0)) => self0 == other0,
            (Pat::Macro(self0), Pat::Macro(other0)) => self0 == other0,
            (Pat::Or(self0), Pat::Or(other0)) => self0 == other0,
            (Pat::Paren(self0), Pat::Paren(other0)) => self0 == other0,
            (Pat::Path(self0), Pat::Path(other0)) => self0 == other0,
            (Pat::Range(self0), Pat::Range(other0)) => self0 == other0,
            (Pat::Reference(self0), Pat::Reference(other0)) => self0 == other0,
            (Pat::Rest(self0), Pat::Rest(other0)) => self0 == other0,
            (Pat::Slice(self0), Pat::Slice(other0)) => self0 == other0,
            (Pat::Struct(self0), Pat::Struct(other0)) => self0 == other0,
            (Pat::Tuple(self0), Pat::Tuple(other0)) => self0 == other0,
            (Pat::TupleStruct(self0), Pat::TupleStruct(other0)) => self0 == other0,
            (Pat::Type(self0), Pat::Type(other0)) => self0 == other0,
            (Pat::Verbatim(self0), Pat::Verbatim(other0)) => TokenStreamHelper(self0) == TokenStreamHelper(other0),
            (Pat::Wild(self0), Pat::Wild(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for PatIdent {}
impl PartialEq for PatIdent {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ref_ == other.ref_
            && self.mut_ == other.mut_
            && self.ident == other.ident
            && self.subpat == other.subpat
    }
}
impl Eq for PatOr {}
impl PartialEq for PatOr {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.leading_vert == other.leading_vert && self.cases == other.cases
    }
}
impl Eq for PatParen {}
impl PartialEq for PatParen {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.pat == other.pat
    }
}
impl Eq for PatReference {}
impl PartialEq for PatReference {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mutability == other.mutability && self.pat == other.pat
    }
}
impl Eq for PatRest {}
impl PartialEq for PatRest {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
    }
}
impl Eq for PatSlice {}
impl PartialEq for PatSlice {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.elems == other.elems
    }
}
impl Eq for PatStruct {}
impl PartialEq for PatStruct {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.qself == other.qself
            && self.path == other.path
            && self.fields == other.fields
            && self.rest == other.rest
    }
}
impl Eq for PatTuple {}
impl PartialEq for PatTuple {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.elems == other.elems
    }
}
impl Eq for PatTupleStruct {}
impl PartialEq for PatTupleStruct {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.qself == other.qself && self.path == other.path && self.elems == other.elems
    }
}
impl Eq for PatType {}
impl PartialEq for PatType {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.pat == other.pat && self.ty == other.ty
    }
}
impl Eq for PatWild {}
impl PartialEq for PatWild {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
    }
}
impl Eq for Path {}
impl PartialEq for Path {
    fn eq(&self, other: &Self) -> bool {
        self.leading_colon == other.leading_colon && self.segments == other.segments
    }
}
impl Eq for PathArguments {}
impl PartialEq for PathArguments {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PathArguments::None, PathArguments::None) => true,
            (PathArguments::AngleBracketed(self0), PathArguments::AngleBracketed(other0)) => self0 == other0,
            (PathArguments::Parenthesized(self0), PathArguments::Parenthesized(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for PathSegment {}
impl PartialEq for PathSegment {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident && self.arguments == other.arguments
    }
}
impl Eq for PredLifetime {}
impl PartialEq for PredLifetime {
    fn eq(&self, other: &Self) -> bool {
        self.lifetime == other.lifetime && self.bounds == other.bounds
    }
}
impl Eq for PredType {}
impl PartialEq for PredType {
    fn eq(&self, other: &Self) -> bool {
        self.lifetimes == other.lifetimes && self.bounded_ty == other.bounded_ty && self.bounds == other.bounds
    }
}
impl Eq for QSelf {}
impl PartialEq for QSelf {
    fn eq(&self, other: &Self) -> bool {
        self.ty == other.ty && self.position == other.position && self.as_ == other.as_
    }
}
impl Eq for RangeLimits {}
impl PartialEq for RangeLimits {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (RangeLimits::HalfOpen(_), RangeLimits::HalfOpen(_)) => true,
            (RangeLimits::Closed(_), RangeLimits::Closed(_)) => true,
            _ => false,
        }
    }
}
impl Eq for Receiver {}
impl PartialEq for Receiver {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.reference == other.reference
            && self.mutability == other.mutability
            && self.colon_token == other.colon_token
            && self.ty == other.ty
    }
}
impl Eq for ReturnType {}
impl PartialEq for ReturnType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ReturnType::Default, ReturnType::Default) => true,
            (ReturnType::Type(_, self1), ReturnType::Type(_, other1)) => self1 == other1,
            _ => false,
        }
    }
}
impl Eq for Signature {}
impl PartialEq for Signature {
    fn eq(&self, other: &Self) -> bool {
        self.constness == other.constness
            && self.asyncness == other.asyncness
            && self.unsafety == other.unsafety
            && self.abi == other.abi
            && self.ident == other.ident
            && self.generics == other.generics
            && self.inputs == other.inputs
            && self.variadic == other.variadic
            && self.output == other.output
    }
}
impl Eq for StaticMutability {}
impl PartialEq for StaticMutability {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (StaticMutability::Mut(_), StaticMutability::Mut(_)) => true,
            (StaticMutability::None, StaticMutability::None) => true,
            _ => false,
        }
    }
}
impl Eq for Stmt {}
impl PartialEq for Stmt {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Stmt::Local(self0), Stmt::Local(other0)) => self0 == other0,
            (Stmt::Item(self0), Stmt::Item(other0)) => self0 == other0,
            (Stmt::Expr(self0, self1), Stmt::Expr(other0, other1)) => self0 == other0 && self1 == other1,
            (Stmt::Macro(self0), Stmt::Macro(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for StmtMacro {}
impl PartialEq for StmtMacro {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mac == other.mac && self.semi == other.semi
    }
}
impl Eq for TraitBound {}
impl PartialEq for TraitBound {
    fn eq(&self, other: &Self) -> bool {
        self.paren == other.paren
            && self.modifier == other.modifier
            && self.lifetimes == other.lifetimes
            && self.path == other.path
    }
}
impl Eq for TraitBoundModifier {}
impl PartialEq for TraitBoundModifier {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TraitBoundModifier::None, TraitBoundModifier::None) => true,
            (TraitBoundModifier::Maybe(_), TraitBoundModifier::Maybe(_)) => true,
            _ => false,
        }
    }
}
impl Eq for TraitItem {}
impl PartialEq for TraitItem {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TraitItem::Const(self0), TraitItem::Const(other0)) => self0 == other0,
            (TraitItem::Fn(self0), TraitItem::Fn(other0)) => self0 == other0,
            (TraitItem::Type(self0), TraitItem::Type(other0)) => self0 == other0,
            (TraitItem::Macro(self0), TraitItem::Macro(other0)) => self0 == other0,
            (TraitItem::Verbatim(self0), TraitItem::Verbatim(other0)) => {
                TokenStreamHelper(self0) == TokenStreamHelper(other0)
            },
            _ => false,
        }
    }
}
impl Eq for TraitItemConst {}
impl PartialEq for TraitItemConst {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ident == other.ident
            && self.generics == other.generics
            && self.ty == other.ty
            && self.default == other.default
    }
}
impl Eq for TraitItemFn {}
impl PartialEq for TraitItemFn {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.sig == other.sig
            && self.default == other.default
            && self.semi_token == other.semi_token
    }
}
impl Eq for TraitItemMacro {}
impl PartialEq for TraitItemMacro {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mac == other.mac && self.semi_token == other.semi_token
    }
}
impl Eq for TraitItemType {}
impl PartialEq for TraitItemType {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ident == other.ident
            && self.generics == other.generics
            && self.colon_token == other.colon_token
            && self.bounds == other.bounds
            && self.default == other.default
    }
}
impl Eq for Type {}
impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Type::Array(self0), Type::Array(other0)) => self0 == other0,
            (Type::BareFn(self0), Type::BareFn(other0)) => self0 == other0,
            (Type::Group(self0), Type::Group(other0)) => self0 == other0,
            (Type::ImplTrait(self0), Type::ImplTrait(other0)) => self0 == other0,
            (Type::Infer(self0), Type::Infer(other0)) => self0 == other0,
            (Type::Macro(self0), Type::Macro(other0)) => self0 == other0,
            (Type::Never(self0), Type::Never(other0)) => self0 == other0,
            (Type::Paren(self0), Type::Paren(other0)) => self0 == other0,
            (Type::Path(self0), Type::Path(other0)) => self0 == other0,
            (Type::Ptr(self0), Type::Ptr(other0)) => self0 == other0,
            (Type::Reference(self0), Type::Reference(other0)) => self0 == other0,
            (Type::Slice(self0), Type::Slice(other0)) => self0 == other0,
            (Type::TraitObject(self0), Type::TraitObject(other0)) => self0 == other0,
            (Type::Tuple(self0), Type::Tuple(other0)) => self0 == other0,
            (Type::Verbatim(self0), Type::Verbatim(other0)) => TokenStreamHelper(self0) == TokenStreamHelper(other0),
            _ => false,
        }
    }
}
impl Eq for TypeArray {}
impl PartialEq for TypeArray {
    fn eq(&self, other: &Self) -> bool {
        self.elem == other.elem && self.len == other.len
    }
}
impl Eq for TypeBareFn {}
impl PartialEq for TypeBareFn {
    fn eq(&self, other: &Self) -> bool {
        self.lifetimes == other.lifetimes
            && self.unsafe_ == other.unsafe_
            && self.abi == other.abi
            && self.inputs == other.inputs
            && self.variadic == other.variadic
            && self.output == other.output
    }
}
impl Eq for TypeGroup {}
impl PartialEq for TypeGroup {
    fn eq(&self, other: &Self) -> bool {
        self.elem == other.elem
    }
}
impl Eq for TypeImplTrait {}
impl PartialEq for TypeImplTrait {
    fn eq(&self, other: &Self) -> bool {
        self.bounds == other.bounds
    }
}
impl Eq for TypeInfer {}
impl PartialEq for TypeInfer {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for TypeMacro {}
impl PartialEq for TypeMacro {
    fn eq(&self, other: &Self) -> bool {
        self.mac == other.mac
    }
}
impl Eq for TypeNever {}
impl PartialEq for TypeNever {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for TypeParam {}
impl PartialEq for TypeParam {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ident == other.ident
            && self.colon == other.colon
            && self.bounds == other.bounds
            && self.eq == other.eq
            && self.default == other.default
    }
}
impl Eq for TypeParamBound {}
impl PartialEq for TypeParamBound {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TypeParamBound::Trait(self0), TypeParamBound::Trait(other0)) => self0 == other0,
            (TypeParamBound::Lifetime(self0), TypeParamBound::Lifetime(other0)) => self0 == other0,
            (TypeParamBound::Verbatim(self0), TypeParamBound::Verbatim(other0)) => {
                TokenStreamHelper(self0) == TokenStreamHelper(other0)
            },
            _ => false,
        }
    }
}
impl Eq for TypeParen {}
impl PartialEq for TypeParen {
    fn eq(&self, other: &Self) -> bool {
        self.elem == other.elem
    }
}
impl Eq for TypePath {}
impl PartialEq for TypePath {
    fn eq(&self, other: &Self) -> bool {
        self.qself == other.qself && self.path == other.path
    }
}
impl Eq for TypePtr {}
impl PartialEq for TypePtr {
    fn eq(&self, other: &Self) -> bool {
        self.const_ == other.const_ && self.mut_ == other.mut_ && self.elem == other.elem
    }
}
impl Eq for TypeReference {}
impl PartialEq for TypeReference {
    fn eq(&self, other: &Self) -> bool {
        self.lifetime == other.lifetime && self.mut_ == other.mut_ && self.elem == other.elem
    }
}
impl Eq for TypeSlice {}
impl PartialEq for TypeSlice {
    fn eq(&self, other: &Self) -> bool {
        self.elem == other.elem
    }
}
impl Eq for TypeTraitObject {}
impl PartialEq for TypeTraitObject {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_ == other.dyn_ && self.bounds == other.bounds
    }
}
impl Eq for TypeTuple {}
impl PartialEq for TypeTuple {
    fn eq(&self, other: &Self) -> bool {
        self.elems == other.elems
    }
}
impl Eq for UnOp {}
impl PartialEq for UnOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (UnOp::Deref(_), UnOp::Deref(_)) => true,
            (UnOp::Not(_), UnOp::Not(_)) => true,
            (UnOp::Neg(_), UnOp::Neg(_)) => true,
            _ => false,
        }
    }
}
impl Eq for UseGlob {}
impl PartialEq for UseGlob {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for UseGroup {}
impl PartialEq for UseGroup {
    fn eq(&self, other: &Self) -> bool {
        self.items == other.items
    }
}
impl Eq for UseName {}
impl PartialEq for UseName {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident
    }
}
impl Eq for UsePath {}
impl PartialEq for UsePath {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident && self.tree == other.tree
    }
}
impl Eq for UseRename {}
impl PartialEq for UseRename {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident && self.rename == other.rename
    }
}
impl Eq for UseTree {}
impl PartialEq for UseTree {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (UseTree::Path(self0), UseTree::Path(other0)) => self0 == other0,
            (UseTree::Name(self0), UseTree::Name(other0)) => self0 == other0,
            (UseTree::Rename(self0), UseTree::Rename(other0)) => self0 == other0,
            (UseTree::Glob(self0), UseTree::Glob(other0)) => self0 == other0,
            (UseTree::Group(self0), UseTree::Group(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for Variadic {}
impl PartialEq for Variadic {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.pat == other.pat && self.comma == other.comma
    }
}
impl Eq for Variant {}
impl PartialEq for Variant {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ident == other.ident
            && self.fields == other.fields
            && self.discriminant == other.discriminant
    }
}
impl Eq for VisRestricted {}
impl PartialEq for VisRestricted {
    fn eq(&self, other: &Self) -> bool {
        self.in_ == other.in_ && self.path == other.path
    }
}
impl Eq for Visibility {}
impl PartialEq for Visibility {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Visibility::Public(_), Visibility::Public(_)) => true,
            (Visibility::Restricted(self0), Visibility::Restricted(other0)) => self0 == other0,
            (Visibility::Inherited, Visibility::Inherited) => true,
            _ => false,
        }
    }
}
impl Eq for WhereClause {}
impl PartialEq for WhereClause {
    fn eq(&self, other: &Self) -> bool {
        self.preds == other.preds
    }
}
impl Eq for WherePred {}
impl PartialEq for WherePred {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (WherePred::Lifetime(self0), WherePred::Lifetime(other0)) => self0 == other0,
            (WherePred::Type(self0), WherePred::Type(other0)) => self0 == other0,
            _ => false,
        }
    }
}
