use crate::TokenStreamHelper;
use crate::*;
impl Eq for Abi {}
impl PartialEq for Abi {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl Eq for AngledArgs {}
impl PartialEq for AngledArgs {
    fn eq(&self, other: &Self) -> bool {
        self.colon2 == other.colon2 && self.args == other.args
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
        self.ident == other.ident && self.gnrs == other.gnrs && self.val == other.val
    }
}
impl Eq for AssocType {}
impl PartialEq for AssocType {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident && self.gnrs == other.gnrs && self.ty == other.ty
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
impl Eq for ty::BareFnArg {}
impl PartialEq for ty::BareFnArg {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.name == other.name && self.ty == other.ty
    }
}
impl Eq for ty::BareVari {}
impl PartialEq for ty::BareVari {
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
        self.lifes == other.lifes
    }
}
impl Eq for ConstParam {}
impl PartialEq for ConstParam {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ident == other.ident
            && self.typ == other.typ
            && self.eq == other.eq
            && self.default == other.default
    }
}
impl Eq for Constraint {}
impl PartialEq for Constraint {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident && self.gnrs == other.gnrs && self.bounds == other.bounds
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
            && self.gens == other.gens
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
impl Eq for expr::Array {}
impl PartialEq for expr::Array {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.elems == other.elems
    }
}
impl Eq for expr::Assign {}
impl PartialEq for expr::Assign {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.left == other.left && self.right == other.right
    }
}
impl Eq for expr::Async {}
impl PartialEq for expr::Async {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.move_ == other.move_ && self.block == other.block
    }
}
impl Eq for expr::Await {}
impl PartialEq for expr::Await {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr
    }
}
impl Eq for expr::Binary {}
impl PartialEq for expr::Binary {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.left == other.left && self.op == other.op && self.right == other.right
    }
}
impl Eq for expr::Block {}
impl PartialEq for expr::Block {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.label == other.label && self.block == other.block
    }
}
impl Eq for expr::Break {}
impl PartialEq for expr::Break {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.label == other.label && self.expr == other.expr
    }
}
impl Eq for expr::Call {}
impl PartialEq for expr::Call {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.func == other.func && self.args == other.args
    }
}
impl Eq for expr::Cast {}
impl PartialEq for expr::Cast {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr && self.typ == other.typ
    }
}
impl Eq for expr::Closure {}
impl PartialEq for expr::Closure {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.lifes == other.lifes
            && self.const_ == other.const_
            && self.static_ == other.static_
            && self.async_ == other.async_
            && self.move_ == other.move_
            && self.inputs == other.inputs
            && self.ret == other.ret
            && self.body == other.body
    }
}
impl Eq for expr::Const {}
impl PartialEq for expr::Const {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.block == other.block
    }
}
impl Eq for expr::Continue {}
impl PartialEq for expr::Continue {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.label == other.label
    }
}
impl Eq for expr::Field {}
impl PartialEq for expr::Field {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.base == other.base && self.memb == other.memb
    }
}
impl Eq for expr::ForLoop {}
impl PartialEq for expr::ForLoop {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.label == other.label
            && self.pat == other.pat
            && self.expr == other.expr
            && self.body == other.body
    }
}
impl Eq for expr::Groupup {}
impl PartialEq for expr::Groupup {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr
    }
}
impl Eq for expr::If {}
impl PartialEq for expr::If {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.cond == other.cond
            && self.then_branch == other.then_branch
            && self.else_branch == other.else_branch
    }
}
impl Eq for expr::Index {}
impl PartialEq for expr::Index {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr && self.index == other.index
    }
}
impl Eq for expr::Infer {}
impl PartialEq for expr::Infer {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
    }
}
impl Eq for expr::Let {}
impl PartialEq for expr::Let {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.pat == other.pat && self.expr == other.expr
    }
}
impl Eq for expr::Lit {}
impl PartialEq for expr::Lit {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.lit == other.lit
    }
}
impl Eq for expr::Loop {}
impl PartialEq for expr::Loop {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.label == other.label && self.body == other.body
    }
}
impl Eq for expr::Mac {}
impl PartialEq for expr::Mac {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mac == other.mac
    }
}
impl Eq for expr::Match {}
impl PartialEq for expr::Match {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr && self.arms == other.arms
    }
}
impl Eq for expr::MethodCall {}
impl PartialEq for expr::MethodCall {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.expr == other.expr
            && self.method == other.method
            && self.turbofish == other.turbofish
            && self.args == other.args
    }
}
impl Eq for expr::Paren {}
impl PartialEq for expr::Paren {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr
    }
}
impl Eq for expr::Path {}
impl PartialEq for expr::Path {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.qself == other.qself && self.path == other.path
    }
}
impl Eq for expr::Range {}
impl PartialEq for expr::Range {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.beg == other.beg && self.limits == other.limits && self.end == other.end
    }
}
impl Eq for expr::Ref {}
impl PartialEq for expr::Ref {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mut_ == other.mut_ && self.expr == other.expr
    }
}
impl Eq for expr::Repeat {}
impl PartialEq for expr::Repeat {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr && self.len == other.len
    }
}
impl Eq for expr::Returnrn {}
impl PartialEq for expr::Returnrn {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr
    }
}
impl Eq for expr::Struct {}
impl PartialEq for expr::Struct {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.qself == other.qself
            && self.path == other.path
            && self.fields == other.fields
            && self.dot2 == other.dot2
            && self.rest == other.rest
    }
}
impl Eq for expr::Try {}
impl PartialEq for expr::Try {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.expr == other.expr
    }
}
impl Eq for expr::TryBlock {}
impl PartialEq for expr::TryBlock {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.block == other.block
    }
}
impl Eq for expr::Tuple {}
impl PartialEq for expr::Tuple {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.elems == other.elems
    }
}
impl Eq for expr::Unary {}
impl PartialEq for expr::Unary {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.op == other.op && self.expr == other.expr
    }
}
impl Eq for expr::Unsafe {}
impl PartialEq for expr::Unsafe {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.block == other.block
    }
}
impl Eq for expr::While {}
impl PartialEq for expr::While {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.label == other.label && self.cond == other.cond && self.body == other.body
    }
}
impl Eq for expr::Yield {}
impl PartialEq for expr::Yield {
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
            && self.colon == other.colon
            && self.typ == other.typ
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
impl Eq for patt::Field {}
impl PartialEq for patt::Field {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.member == other.member && self.colon == other.colon && self.patt == other.patt
    }
}
impl Eq for FieldValue {}
impl PartialEq for FieldValue {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.member == other.member && self.colon == other.colon && self.expr == other.expr
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
        self.attrs == other.attrs && self.mac == other.mac && self.semi == other.semi
    }
}
impl Eq for ForeignItemStatic {}
impl PartialEq for ForeignItemStatic {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.mut_ == other.mut_
            && self.ident == other.ident
            && self.typ == other.typ
    }
}
impl Eq for ForeignItemType {}
impl PartialEq for ForeignItemType {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.vis == other.vis && self.ident == other.ident && self.gens == other.gens
    }
}
impl Eq for Arg {}
impl PartialEq for Arg {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Arg::Lifetime(self0), Arg::Lifetime(other0)) => self0 == other0,
            (Arg::Type(self0), Arg::Type(other0)) => self0 == other0,
            (Arg::Const(self0), Arg::Const(other0)) => self0 == other0,
            (Arg::AssocType(self0), Arg::AssocType(other0)) => self0 == other0,
            (Arg::AssocConst(self0), Arg::AssocConst(other0)) => self0 == other0,
            (Arg::Constraint(self0), Arg::Constraint(other0)) => self0 == other0,
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
        self.lt == other.lt && self.params == other.params && self.gt == other.gt && self.where_ == other.where_
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
            && self.default_ == other.default_
            && self.ident == other.ident
            && self.gens == other.gens
            && self.typ == other.typ
            && self.expr == other.expr
    }
}
impl Eq for ImplItemFn {}
impl PartialEq for ImplItemFn {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.default_ == other.default_
            && self.sig == other.sig
            && self.block == other.block
    }
}
impl Eq for ImplItemMacro {}
impl PartialEq for ImplItemMacro {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mac == other.mac && self.semi == other.semi
    }
}
impl Eq for ImplItemType {}
impl PartialEq for ImplItemType {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.default_ == other.default_
            && self.ident == other.ident
            && self.gens == other.gens
            && self.typ == other.typ
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
            && self.gens == other.gens
            && self.typ == other.typ
            && self.expr == other.expr
    }
}
impl Eq for ItemEnum {}
impl PartialEq for ItemEnum {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.gens == other.gens
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
        self.attrs == other.attrs && self.unsafe_ == other.unsafe_ && self.abi == other.abi && self.items == other.items
    }
}
impl Eq for ItemImpl {}
impl PartialEq for ItemImpl {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.default_ == other.default_
            && self.unsafe_ == other.unsafe_
            && self.gens == other.gens
            && self.trait_ == other.trait_
            && self.typ == other.typ
            && self.items == other.items
    }
}
impl Eq for ItemMacro {}
impl PartialEq for ItemMacro {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.ident == other.ident && self.mac == other.mac && self.semi == other.semi
    }
}
impl Eq for ItemMod {}
impl PartialEq for ItemMod {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.unsafe_ == other.unsafe_
            && self.ident == other.ident
            && self.gist == other.gist
            && self.semi == other.semi
    }
}
impl Eq for ItemStatic {}
impl PartialEq for ItemStatic {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.mut_ == other.mut_
            && self.ident == other.ident
            && self.typ == other.typ
            && self.expr == other.expr
    }
}
impl Eq for ItemStruct {}
impl PartialEq for ItemStruct {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.gens == other.gens
            && self.fields == other.fields
            && self.semi == other.semi
    }
}
impl Eq for ItemTrait {}
impl PartialEq for ItemTrait {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.unsafe_ == other.unsafe_
            && self.auto_ == other.auto_
            && self.restriction == other.restriction
            && self.ident == other.ident
            && self.gens == other.gens
            && self.colon == other.colon
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
            && self.gens == other.gens
            && self.bounds == other.bounds
    }
}
impl Eq for ItemType {}
impl PartialEq for ItemType {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.gens == other.gens
            && self.typ == other.typ
    }
}
impl Eq for ItemUnion {}
impl PartialEq for ItemUnion {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.gens == other.gens
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
        self.attrs == other.attrs && self.life == other.life && self.colon == other.colon && self.bounds == other.bounds
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
impl Eq for stmt::Local {}
impl PartialEq for stmt::Local {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.pat == other.pat && self.init == other.init
    }
}
impl Eq for stmt::LocalInit {}
impl PartialEq for stmt::LocalInit {
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
impl Eq for MacroDelim {}
impl PartialEq for MacroDelim {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MacroDelim::Paren(_), MacroDelim::Paren(_)) => true,
            (MacroDelim::Brace(_), MacroDelim::Brace(_)) => true,
            (MacroDelim::Bracket(_), MacroDelim::Bracket(_)) => true,
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
        self.path == other.path && self.expr == other.expr
    }
}
impl Eq for ParenthesizedArgs {}
impl PartialEq for ParenthesizedArgs {
    fn eq(&self, other: &Self) -> bool {
        self.ins == other.ins && self.out == other.out
    }
}
impl Eq for patt::Patt {}
impl PartialEq for patt::Patt {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (patt::Patt::Const(self0), patt::Patt::Const(other0)) => self0 == other0,
            (patt::Patt::Ident(self0), patt::Patt::Ident(other0)) => self0 == other0,
            (patt::Patt::Lit(self0), patt::Patt::Lit(other0)) => self0 == other0,
            (patt::Patt::Mac(self0), patt::Patt::Mac(other0)) => self0 == other0,
            (patt::Patt::Or(self0), patt::Patt::Or(other0)) => self0 == other0,
            (patt::Patt::Paren(self0), patt::Patt::Paren(other0)) => self0 == other0,
            (patt::Patt::Path(self0), patt::Patt::Path(other0)) => self0 == other0,
            (patt::Patt::Range(self0), patt::Patt::Range(other0)) => self0 == other0,
            (patt::Patt::Ref(self0), patt::Patt::Ref(other0)) => self0 == other0,
            (patt::Patt::Rest(self0), patt::Patt::Rest(other0)) => self0 == other0,
            (patt::Patt::Slice(self0), patt::Patt::Slice(other0)) => self0 == other0,
            (patt::Patt::Struct(self0), patt::Patt::Struct(other0)) => self0 == other0,
            (patt::Patt::Tuple(self0), patt::Patt::Tuple(other0)) => self0 == other0,
            (patt::Patt::TupleStruct(self0), patt::Patt::TupleStruct(other0)) => self0 == other0,
            (patt::Patt::Type(self0), patt::Patt::Type(other0)) => self0 == other0,
            (patt::Patt::Verbatim(self0), patt::Patt::Verbatim(other0)) => {
                TokenStreamHelper(self0) == TokenStreamHelper(other0)
            },
            (patt::Patt::Wild(self0), patt::Patt::Wild(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for patt::Ident {}
impl PartialEq for patt::Ident {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ref_ == other.ref_
            && self.mut_ == other.mut_
            && self.ident == other.ident
            && self.sub == other.sub
    }
}
impl Eq for patt::Or {}
impl PartialEq for patt::Or {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.vert == other.vert && self.cases == other.cases
    }
}
impl Eq for patt::Paren {}
impl PartialEq for patt::Paren {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.patt == other.patt
    }
}
impl Eq for patt::Ref {}
impl PartialEq for patt::Ref {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mut_ == other.mut_ && self.patt == other.patt
    }
}
impl Eq for patt::Rest {}
impl PartialEq for patt::Rest {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
    }
}
impl Eq for patt::Slice {}
impl PartialEq for patt::Slice {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.elems == other.elems
    }
}
impl Eq for patt::Struct {}
impl PartialEq for patt::Struct {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.qself == other.qself
            && self.path == other.path
            && self.fields == other.fields
            && self.rest == other.rest
    }
}
impl Eq for patt::Tuple {}
impl PartialEq for patt::Tuple {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.elems == other.elems
    }
}
impl Eq for patt::TupleStruct {}
impl PartialEq for patt::TupleStruct {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.qself == other.qself && self.path == other.path && self.elems == other.elems
    }
}
impl Eq for patt::Type {}
impl PartialEq for patt::Type {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.patt == other.patt && self.typ == other.typ
    }
}
impl Eq for patt::Wild {}
impl PartialEq for patt::Wild {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
    }
}
impl Eq for Path {}
impl PartialEq for Path {
    fn eq(&self, other: &Self) -> bool {
        self.colon == other.colon && self.segs == other.segs
    }
}
impl Eq for Args {}
impl PartialEq for Args {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Args::None, Args::None) => true,
            (Args::Angled(self0), Args::Angled(other0)) => self0 == other0,
            (Args::Parenthesized(self0), Args::Parenthesized(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for Segment {}
impl PartialEq for Segment {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident && self.args == other.args
    }
}
impl Eq for PredLifetime {}
impl PartialEq for PredLifetime {
    fn eq(&self, other: &Self) -> bool {
        self.life == other.life && self.bounds == other.bounds
    }
}
impl Eq for PredType {}
impl PartialEq for PredType {
    fn eq(&self, other: &Self) -> bool {
        self.lifes == other.lifes && self.bounded == other.bounded && self.bounds == other.bounds
    }
}
impl Eq for QSelf {}
impl PartialEq for QSelf {
    fn eq(&self, other: &Self) -> bool {
        self.ty == other.ty && self.pos == other.pos && self.as_ == other.as_
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
            && self.mut_ == other.mut_
            && self.colon == other.colon
            && self.typ == other.typ
    }
}
impl Eq for ty::Ret {}
impl PartialEq for ty::Ret {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ty::Ret::Default, ty::Ret::Default) => true,
            (ty::Ret::Type(_, self1), ty::Ret::Type(_, other1)) => self1 == other1,
            _ => false,
        }
    }
}
impl Eq for Signature {}
impl PartialEq for Signature {
    fn eq(&self, other: &Self) -> bool {
        self.constness == other.constness
            && self.async_ == other.async_
            && self.unsafe_ == other.unsafe_
            && self.abi == other.abi
            && self.ident == other.ident
            && self.gens == other.gens
            && self.args == other.args
            && self.vari == other.vari
            && self.ret == other.ret
    }
}
impl Eq for StaticMut {}
impl PartialEq for StaticMut {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (StaticMut::Mut(_), StaticMut::Mut(_)) => true,
            (StaticMut::None, StaticMut::None) => true,
            _ => false,
        }
    }
}
impl Eq for stmt::Stmt {}
impl PartialEq for stmt::Stmt {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (stmt::Stmt::stmt::Local(self0), stmt::Stmt::stmt::Local(other0)) => self0 == other0,
            (stmt::Stmt::Item(self0), stmt::Stmt::Item(other0)) => self0 == other0,
            (stmt::Stmt::Expr(self0, self1), stmt::Stmt::Expr(other0, other1)) => self0 == other0 && self1 == other1,
            (stmt::Stmt::Mac(self0), stmt::Stmt::Mac(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for stmt::Mac {}
impl PartialEq for stmt::Mac {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mac == other.mac && self.semi == other.semi
    }
}
impl Eq for TraitBound {}
impl PartialEq for TraitBound {
    fn eq(&self, other: &Self) -> bool {
        self.paren == other.paren
            && self.modifier == other.modifier
            && self.lifes == other.lifes
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
            && self.gens == other.gens
            && self.typ == other.typ
            && self.default == other.default
    }
}
impl Eq for TraitItemFn {}
impl PartialEq for TraitItemFn {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.sig == other.sig && self.default == other.default && self.semi == other.semi
    }
}
impl Eq for TraitItemMacro {}
impl PartialEq for TraitItemMacro {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mac == other.mac && self.semi == other.semi
    }
}
impl Eq for TraitItemType {}
impl PartialEq for TraitItemType {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ident == other.ident
            && self.gens == other.gens
            && self.colon == other.colon
            && self.bounds == other.bounds
            && self.default == other.default
    }
}
impl Eq for ty::Type {}
impl PartialEq for ty::Type {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ty::Type::Array(self0), ty::Type::Array(other0)) => self0 == other0,
            (ty::Type::BareFn(self0), ty::Type::BareFn(other0)) => self0 == other0,
            (ty::Type::Group(self0), ty::Type::Group(other0)) => self0 == other0,
            (ty::Type::Impl(self0), ty::Type::Impl(other0)) => self0 == other0,
            (ty::Type::Infer(self0), ty::Type::Infer(other0)) => self0 == other0,
            (ty::Type::Mac(self0), ty::Type::Mac(other0)) => self0 == other0,
            (ty::Type::Never(self0), ty::Type::Never(other0)) => self0 == other0,
            (ty::Type::Paren(self0), ty::Type::Paren(other0)) => self0 == other0,
            (ty::Type::Path(self0), ty::Type::Path(other0)) => self0 == other0,
            (ty::Type::Ptr(self0), ty::Type::Ptr(other0)) => self0 == other0,
            (ty::Type::Ref(self0), ty::Type::Ref(other0)) => self0 == other0,
            (ty::Type::Slice(self0), ty::Type::Slice(other0)) => self0 == other0,
            (ty::Type::TraitObj(self0), ty::Type::TraitObj(other0)) => self0 == other0,
            (ty::Type::Tuple(self0), ty::Type::Tuple(other0)) => self0 == other0,
            (ty::Type::Verbatim(self0), ty::Type::Verbatim(other0)) => {
                TokenStreamHelper(self0) == TokenStreamHelper(other0)
            },
            _ => false,
        }
    }
}
impl Eq for ty::Array {}
impl PartialEq for ty::Array {
    fn eq(&self, other: &Self) -> bool {
        self.elem == other.elem && self.len == other.len
    }
}
impl Eq for ty::BareFn {}
impl PartialEq for ty::BareFn {
    fn eq(&self, other: &Self) -> bool {
        self.lifes == other.lifes
            && self.unsafe_ == other.unsafe_
            && self.abi == other.abi
            && self.args == other.args
            && self.vari == other.vari
            && self.ret == other.ret
    }
}
impl Eq for ty::Group {}
impl PartialEq for ty::Group {
    fn eq(&self, other: &Self) -> bool {
        self.elem == other.elem
    }
}
impl Eq for ty::Impl {}
impl PartialEq for ty::Impl {
    fn eq(&self, other: &Self) -> bool {
        self.bounds == other.bounds
    }
}
impl Eq for ty::Infer {}
impl PartialEq for ty::Infer {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for ty::Mac {}
impl PartialEq for ty::Mac {
    fn eq(&self, other: &Self) -> bool {
        self.mac == other.mac
    }
}
impl Eq for ty::Never {}
impl PartialEq for ty::Never {
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
impl Eq for ty::Paren {}
impl PartialEq for ty::Paren {
    fn eq(&self, other: &Self) -> bool {
        self.elem == other.elem
    }
}
impl Eq for ty::Path {}
impl PartialEq for ty::Path {
    fn eq(&self, other: &Self) -> bool {
        self.qself == other.qself && self.path == other.path
    }
}
impl Eq for ty::Ptr {}
impl PartialEq for ty::Ptr {
    fn eq(&self, other: &Self) -> bool {
        self.const_ == other.const_ && self.mut_ == other.mut_ && self.elem == other.elem
    }
}
impl Eq for ty::Ref {}
impl PartialEq for ty::Ref {
    fn eq(&self, other: &Self) -> bool {
        self.life == other.life && self.mut_ == other.mut_ && self.elem == other.elem
    }
}
impl Eq for ty::Slice {}
impl PartialEq for ty::Slice {
    fn eq(&self, other: &Self) -> bool {
        self.elem == other.elem
    }
}
impl Eq for ty::TraitObj {}
impl PartialEq for ty::TraitObj {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_ == other.dyn_ && self.bounds == other.bounds
    }
}
impl Eq for ty::Tuple {}
impl PartialEq for ty::Tuple {
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
