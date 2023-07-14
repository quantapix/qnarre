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
impl Eq for attr::Style {}
impl PartialEq for attr::Style {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (attr::Style::Outer, attr::Style::Outer) => true,
            (attr::Style::Inner(_), attr::Style::Inner(_)) => true,
            _ => false,
        }
    }
}
impl Eq for attr::Attr {}
impl PartialEq for attr::Attr {
    fn eq(&self, other: &Self) -> bool {
        self.style == other.style && self.meta == other.meta
    }
}
impl Eq for typ::FnArg {}
impl PartialEq for typ::FnArg {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.name == other.name && self.typ == other.typ
    }
}
impl Eq for typ::Variadic {}
impl PartialEq for typ::Variadic {
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
impl Eq for Bgen::bound::Lifes {}
impl PartialEq for Bgen::bound::Lifes {
    fn eq(&self, other: &Self) -> bool {
        self.lifes == other.lifes
    }
}
impl Eq for gen::param::Const {}
impl PartialEq for gen::param::Const {
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
impl Eq for data::Enum {}
impl PartialEq for data::Enum {
    fn eq(&self, other: &Self) -> bool {
        self.variants == other.variants
    }
}
impl Eq for data::Struct {}
impl PartialEq for data::Struct {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields && self.semi == other.semi
    }
}
impl Eq for data::Union {}
impl PartialEq for data::Union {
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
impl Eq for data::Field {}
impl PartialEq for data::Field {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.mut_ == other.mut_
            && self.ident == other.ident
            && self.colon == other.colon
            && self.typ == other.typ
    }
}
impl Eq for data::Mut {}
impl PartialEq for data::Mut {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (data::Mut::None, data::Mut::None) => true,
        }
    }
}
impl Eq for pat::Field {}
impl PartialEq for pat::Field {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.member == other.member && self.colon == other.colon && self.pat == other.pat
    }
}
impl Eq for FieldValue {}
impl PartialEq for FieldValue {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.member == other.member && self.colon == other.colon && self.expr == other.expr
    }
}
impl Eq for data::Fields {}
impl PartialEq for data::Fields {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (data::Fields::Named(self0), data::Fields::Named(other0)) => self0 == other0,
            (data::Fields::Unnamed(self0), data::Fields::Unnamed(other0)) => self0 == other0,
            (data::Fields::Unit, data::Fields::Unit) => true,
            _ => false,
        }
    }
}
impl Eq for data::Named {}
impl PartialEq for data::Named {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}
impl Eq for data::Unnamed {}
impl PartialEq for data::Unnamed {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}
impl Eq for item::File {}
impl PartialEq for item::File {
    fn eq(&self, other: &Self) -> bool {
        self.shebang == other.shebang && self.attrs == other.attrs && self.items == other.items
    }
}
impl Eq for item::FnArg {}
impl PartialEq for item::FnArg {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (item::FnArg::Receiver(self0), item::FnArg::Receiver(other0)) => self0 == other0,
            (item::FnArg::Type(self0), item::FnArg::Type(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for item::Foreign::Item {}
impl PartialEq for item::Foreign::Item {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (item::Foreign::Item::Fn(self0), item::Foreign::Item::Fn(other0)) => self0 == other0,
            (item::Foreign::Item::Static(self0), item::Foreign::Item::Static(other0)) => self0 == other0,
            (item::Foreign::Item::Type(self0), item::Foreign::Item::Type(other0)) => self0 == other0,
            (item::Foreign::Item::Macro(self0), item::Foreign::Item::Macro(other0)) => self0 == other0,
            (item::Foreign::Item::Verbatim(self0), item::Foreign::Item::Verbatim(other0)) => {
                TokenStreamHelper(self0) == TokenStreamHelper(other0)
            },
            _ => false,
        }
    }
}
impl Eq for item::Foreign::Fn {}
impl PartialEq for item::Foreign::Fn {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.vis == other.vis && self.sig == other.sig
    }
}
impl Eq for item::Foreign::Mac {}
impl PartialEq for item::Foreign::Mac {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mac == other.mac && self.semi == other.semi
    }
}
impl Eq for item::Foreign::Static {}
impl PartialEq for item::Foreign::Static {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.mut_ == other.mut_
            && self.ident == other.ident
            && self.typ == other.typ
    }
}
impl Eq for item::Foreign::Type {}
impl PartialEq for item::Foreign::Type {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.vis == other.vis && self.ident == other.ident && self.gens == other.gens
    }
}
impl Eq for Arg {}
impl PartialEq for Arg {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Arg::Life(self0), Arg::Life(other0)) => self0 == other0,
            (Arg::Type(self0), Arg::Type(other0)) => self0 == other0,
            (Arg::Const(self0), Arg::Const(other0)) => self0 == other0,
            (Arg::AssocType(self0), Arg::AssocType(other0)) => self0 == other0,
            (Arg::AssocConst(self0), Arg::AssocConst(other0)) => self0 == other0,
            (Arg::Constraint(self0), Arg::Constraint(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for gen::Param {}
impl PartialEq for gen::Param {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (gen::Param::Life(self0), gen::Param::Life(other0)) => self0 == other0,
            (gen::Param::Type(self0), gen::Param::Type(other0)) => self0 == other0,
            (gen::Param::Const(self0), gen::Param::Const(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for gen::Gens {}
impl PartialEq for gen::Gens {
    fn eq(&self, other: &Self) -> bool {
        self.lt == other.lt && self.params == other.params && self.gt == other.gt && self.where_ == other.where_
    }
}
impl Eq for item::Impl::Item {}
impl PartialEq for item::Impl::Item {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (item::Impl::Item::Const(self0), item::Impl::Item::Const(other0)) => self0 == other0,
            (item::Impl::Item::Fn(self0), item::Impl::Item::Fn(other0)) => self0 == other0,
            (item::Impl::Item::Type(self0), item::Impl::Item::Type(other0)) => self0 == other0,
            (item::Impl::Item::Macro(self0), item::Impl::Item::Macro(other0)) => self0 == other0,
            (item::Impl::Item::Verbatim(self0), item::Impl::Item::Verbatim(other0)) => {
                TokenStreamHelper(self0) == TokenStreamHelper(other0)
            },
            _ => false,
        }
    }
}
impl Eq for item::Impl::Const {}
impl PartialEq for item::Impl::Const {
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
impl Eq for item::Impl::Fn {}
impl PartialEq for item::Impl::Fn {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.default_ == other.default_
            && self.sig == other.sig
            && self.block == other.block
    }
}
impl Eq for item::Impl::Mac {}
impl PartialEq for item::Impl::Mac {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mac == other.mac && self.semi == other.semi
    }
}
impl Eq for item::Impl::Type {}
impl PartialEq for item::Impl::Type {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.default_ == other.default_
            && self.ident == other.ident
            && self.gens == other.gens
            && self.typ == other.typ
    }
}
impl Eq for item::Impl::Restriction {}
impl PartialEq for item::Impl::Restriction {
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
            (Item::Extern(self0), Item::Extern(other0)) => self0 == other0,
            (Item::Fn(self0), Item::Fn(other0)) => self0 == other0,
            (Item::Foreign(self0), Item::Foreign(other0)) => self0 == other0,
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
impl Eq for item::Const {}
impl PartialEq for item::Const {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.gens == other.gens
            && self.typ == other.typ
            && self.expr == other.expr
    }
}
impl Eq for item::Enum {}
impl PartialEq for item::Enum {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.gens == other.gens
            && self.variants == other.variants
    }
}
impl Eq for item::Extern {}
impl PartialEq for item::Extern {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.vis == other.vis && self.ident == other.ident && self.rename == other.rename
    }
}
impl Eq for item::Fn {}
impl PartialEq for item::Fn {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.vis == other.vis && self.sig == other.sig && self.block == other.block
    }
}
impl Eq for item::Foreign {}
impl PartialEq for item::Foreign {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.unsafe_ == other.unsafe_ && self.abi == other.abi && self.items == other.items
    }
}
impl Eq for item::Impl {}
impl PartialEq for item::Impl {
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
impl Eq for item::Mac {}
impl PartialEq for item::Mac {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.ident == other.ident && self.mac == other.mac && self.semi == other.semi
    }
}
impl Eq for item::Mod {}
impl PartialEq for item::Mod {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.unsafe_ == other.unsafe_
            && self.ident == other.ident
            && self.items == other.items
            && self.semi == other.semi
    }
}
impl Eq for item::Static {}
impl PartialEq for item::Static {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.mut_ == other.mut_
            && self.ident == other.ident
            && self.typ == other.typ
            && self.expr == other.expr
    }
}
impl Eq for item::Struct {}
impl PartialEq for item::Struct {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.gens == other.gens
            && self.fields == other.fields
            && self.semi == other.semi
    }
}
impl Eq for item::Trait {}
impl PartialEq for item::Trait {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.unsafe_ == other.unsafe_
            && self.auto_ == other.auto_
            && self.restriction == other.restriction
            && self.ident == other.ident
            && self.gens == other.gens
            && self.colon == other.colon
            && self.supers == other.supers
            && self.items == other.items
    }
}
impl Eq for item::TraitAlias {}
impl PartialEq for item::TraitAlias {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.gens == other.gens
            && self.bounds == other.bounds
    }
}
impl Eq for item::Type {}
impl PartialEq for item::Type {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.gens == other.gens
            && self.typ == other.typ
    }
}
impl Eq for item::Union {}
impl PartialEq for item::Union {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.vis == other.vis
            && self.ident == other.ident
            && self.gens == other.gens
            && self.fields == other.fields
    }
}
impl Eq for item::Use {}
impl PartialEq for item::Use {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.vis == other.vis && self.colon == other.colon && self.tree == other.tree
    }
}
impl Eq for Label {}
impl PartialEq for Label {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl Eq for gen::param::Life {}
impl PartialEq for gen::param::Life {
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
impl Eq for stmt::Init {}
impl PartialEq for stmt::Init {
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
impl Eq for tok::Delim {}
impl PartialEq for tok::Delim {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (tok::Delim::Paren(_), tok::Delim::Paren(_)) => true,
            (tok::Delim::Brace(_), tok::Delim::Brace(_)) => true,
            (tok::Delim::Bracket(_), tok::Delim::Bracket(_)) => true,
            _ => false,
        }
    }
}
impl Eq for meta::Meta {}
impl PartialEq for meta::Meta {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (meta::Meta::Path(self0), meta::Meta::Path(other0)) => self0 == other0,
            (meta::Meta::List(self0), meta::Meta::List(other0)) => self0 == other0,
            (meta::Meta::NameValue(self0), meta::Meta::NameValue(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for meta::List {}
impl PartialEq for meta::List {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.delim == other.delim
            && TokenStreamHelper(&self.toks) == TokenStreamHelper(&other.toks)
    }
}
impl Eq for meta::NameValue {}
impl PartialEq for meta::NameValue {
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
impl Eq for pat::Pat {}
impl PartialEq for pat::Pat {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (pat::Pat::Const(self0), pat::Pat::Const(other0)) => self0 == other0,
            (pat::Pat::Ident(self0), pat::Pat::Ident(other0)) => self0 == other0,
            (pat::Pat::Lit(self0), pat::Pat::Lit(other0)) => self0 == other0,
            (pat::Pat::Mac(self0), pat::Pat::Mac(other0)) => self0 == other0,
            (pat::Pat::Or(self0), pat::Pat::Or(other0)) => self0 == other0,
            (pat::Pat::Paren(self0), pat::Pat::Paren(other0)) => self0 == other0,
            (pat::Pat::Path(self0), pat::Pat::Path(other0)) => self0 == other0,
            (pat::Pat::Range(self0), pat::Pat::Range(other0)) => self0 == other0,
            (pat::Pat::Ref(self0), pat::Pat::Ref(other0)) => self0 == other0,
            (pat::Pat::Rest(self0), pat::Pat::Rest(other0)) => self0 == other0,
            (pat::Pat::Slice(self0), pat::Pat::Slice(other0)) => self0 == other0,
            (pat::Pat::Struct(self0), pat::Pat::Struct(other0)) => self0 == other0,
            (pat::Pat::Tuple(self0), pat::Pat::Tuple(other0)) => self0 == other0,
            (pat::Pat::TupleStruct(self0), pat::Pat::TupleStruct(other0)) => self0 == other0,
            (pat::Pat::Type(self0), pat::Pat::Type(other0)) => self0 == other0,
            (pat::Pat::Verbatim(self0), pat::Pat::Verbatim(other0)) => {
                TokenStreamHelper(self0) == TokenStreamHelper(other0)
            },
            (pat::Pat::Wild(self0), pat::Pat::Wild(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for pat::Ident {}
impl PartialEq for pat::Ident {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ref_ == other.ref_
            && self.mut_ == other.mut_
            && self.ident == other.ident
            && self.sub == other.sub
    }
}
impl Eq for pat::Or {}
impl PartialEq for pat::Or {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.vert == other.vert && self.cases == other.cases
    }
}
impl Eq for pat::Paren {}
impl PartialEq for pat::Paren {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.pat == other.pat
    }
}
impl Eq for pat::Ref {}
impl PartialEq for pat::Ref {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mut_ == other.mut_ && self.pat == other.pat
    }
}
impl Eq for pat::Rest {}
impl PartialEq for pat::Rest {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
    }
}
impl Eq for pat::Slice {}
impl PartialEq for pat::Slice {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.elems == other.elems
    }
}
impl Eq for pat::Struct {}
impl PartialEq for pat::Struct {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.qself == other.qself
            && self.path == other.path
            && self.fields == other.fields
            && self.rest == other.rest
    }
}
impl Eq for pat::Tuple {}
impl PartialEq for pat::Tuple {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.elems == other.elems
    }
}
impl Eq for pat::TupleStruct {}
impl PartialEq for pat::TupleStruct {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.qself == other.qself && self.path == other.path && self.elems == other.elems
    }
}
impl Eq for pat::Type {}
impl PartialEq for pat::Type {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.pat == other.pat && self.typ == other.typ
    }
}
impl Eq for pat::Wild {}
impl PartialEq for pat::Wild {
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
impl Eq for gen::Where::Life {}
impl PartialEq for gen::Where::Life {
    fn eq(&self, other: &Self) -> bool {
        self.life == other.life && self.bounds == other.bounds
    }
}
impl Eq for gen::Where::Type {}
impl PartialEq for gen::Where::Type {
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
impl Eq for item::Receiver {}
impl PartialEq for item::Receiver {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ref_ == other.ref_
            && self.mut_ == other.mut_
            && self.colon == other.colon
            && self.typ == other.typ
    }
}
impl Eq for typ::Ret {}
impl PartialEq for typ::Ret {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (typ::Ret::Default, typ::Ret::Default) => true,
            (typ::Ret::Type(_, self1), typ::Ret::Type(_, other1)) => self1 == other1,
            _ => false,
        }
    }
}
impl Eq for item::Sig {}
impl PartialEq for item::Sig {
    fn eq(&self, other: &Self) -> bool {
        self.const_ == other.const_
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
impl Eq for gen::bound::Trait {}
impl PartialEq for gen::bound::Trait {
    fn eq(&self, other: &Self) -> bool {
        self.paren == other.paren && self.modif == other.modif && self.lifes == other.lifes && self.path == other.path
    }
}
impl Eq for gen::bound::Modifier {}
impl PartialEq for gen::bound::Modifier {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (gen::bound::Modifier::None, gen::bound::Modifier::None) => true,
            (gen::bound::Modifier::Maybe(_), gen::bound::Modifier::Maybe(_)) => true,
            _ => false,
        }
    }
}
impl Eq for item::Trait::Item {}
impl PartialEq for item::Trait::Item {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (item::Trait::Item::Const(self0), item::Trait::Item::Const(other0)) => self0 == other0,
            (item::Trait::Item::Fn(self0), item::Trait::Item::Fn(other0)) => self0 == other0,
            (item::Trait::Item::Type(self0), item::Trait::Item::Type(other0)) => self0 == other0,
            (item::Trait::Item::Macro(self0), item::Trait::Item::Macro(other0)) => self0 == other0,
            (item::Trait::Item::Verbatim(self0), item::Trait::Item::Verbatim(other0)) => {
                TokenStreamHelper(self0) == TokenStreamHelper(other0)
            },
            _ => false,
        }
    }
}
impl Eq for item::Trait::Const {}
impl PartialEq for item::Trait::Const {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ident == other.ident
            && self.gens == other.gens
            && self.typ == other.typ
            && self.default == other.default
    }
}
impl Eq for item::Trait::Fn {}
impl PartialEq for item::Trait::Fn {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.sig == other.sig && self.default == other.default && self.semi == other.semi
    }
}
impl Eq for item::Trait::Mac {}
impl PartialEq for item::Trait::Mac {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.mac == other.mac && self.semi == other.semi
    }
}
impl Eq for item::Trait::Type {}
impl PartialEq for item::Trait::Type {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ident == other.ident
            && self.gens == other.gens
            && self.colon == other.colon
            && self.bounds == other.bounds
            && self.default == other.default
    }
}
impl Eq for typ::Type {}
impl PartialEq for typ::Type {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (typ::Type::Array(self0), typ::Type::Array(other0)) => self0 == other0,
            (typ::Type::Fn(self0), typ::Type::Fn(other0)) => self0 == other0,
            (typ::Type::Group(self0), typ::Type::Group(other0)) => self0 == other0,
            (typ::Type::Impl(self0), typ::Type::Impl(other0)) => self0 == other0,
            (typ::Type::Infer(self0), typ::Type::Infer(other0)) => self0 == other0,
            (typ::Type::Mac(self0), typ::Type::Mac(other0)) => self0 == other0,
            (typ::Type::Never(self0), typ::Type::Never(other0)) => self0 == other0,
            (typ::Type::Paren(self0), typ::Type::Paren(other0)) => self0 == other0,
            (typ::Type::Path(self0), typ::Type::Path(other0)) => self0 == other0,
            (typ::Type::Ptr(self0), typ::Type::Ptr(other0)) => self0 == other0,
            (typ::Type::Ref(self0), typ::Type::Ref(other0)) => self0 == other0,
            (typ::Type::Slice(self0), typ::Type::Slice(other0)) => self0 == other0,
            (typ::Type::Trait(self0), typ::Type::Trait(other0)) => self0 == other0,
            (typ::Type::Tuple(self0), typ::Type::Tuple(other0)) => self0 == other0,
            (typ::Type::Verbatim(self0), typ::Type::Verbatim(other0)) => {
                TokenStreamHelper(self0) == TokenStreamHelper(other0)
            },
            _ => false,
        }
    }
}
impl Eq for typ::Array {}
impl PartialEq for typ::Array {
    fn eq(&self, other: &Self) -> bool {
        self.elem == other.elem && self.len == other.len
    }
}
impl Eq for typ::Fn {}
impl PartialEq for typ::Fn {
    fn eq(&self, other: &Self) -> bool {
        self.lifes == other.lifes
            && self.unsafe_ == other.unsafe_
            && self.abi == other.abi
            && self.args == other.args
            && self.vari == other.vari
            && self.ret == other.ret
    }
}
impl Eq for typ::Group {}
impl PartialEq for typ::Group {
    fn eq(&self, other: &Self) -> bool {
        self.elem == other.elem
    }
}
impl Eq for typ::Impl {}
impl PartialEq for typ::Impl {
    fn eq(&self, other: &Self) -> bool {
        self.bounds == other.bounds
    }
}
impl Eq for typ::Infer {}
impl PartialEq for typ::Infer {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for typ::Mac {}
impl PartialEq for typ::Mac {
    fn eq(&self, other: &Self) -> bool {
        self.mac == other.mac
    }
}
impl Eq for typ::Never {}
impl PartialEq for typ::Never {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for gen::param::Type {}
impl PartialEq for gen::param::Type {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ident == other.ident
            && self.colon == other.colon
            && self.bounds == other.bounds
            && self.eq == other.eq
            && self.default == other.default
    }
}
impl Eq for gen::bound::Type {}
impl PartialEq for gen::bound::Type {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (gen::bound::Type::Trait(self0), gen::bound::Type::Trait(other0)) => self0 == other0,
            (gen::bound::Type::Life(self0), gen::bound::Type::Life(other0)) => self0 == other0,
            (gen::bound::Type::Stream(self0), gen::bound::Type::Stream(other0)) => {
                TokenStreamHelper(self0) == TokenStreamHelper(other0)
            },
            _ => false,
        }
    }
}
impl Eq for typ::Paren {}
impl PartialEq for typ::Paren {
    fn eq(&self, other: &Self) -> bool {
        self.elem == other.elem
    }
}
impl Eq for typ::Path {}
impl PartialEq for typ::Path {
    fn eq(&self, other: &Self) -> bool {
        self.qself == other.qself && self.path == other.path
    }
}
impl Eq for typ::Ptr {}
impl PartialEq for typ::Ptr {
    fn eq(&self, other: &Self) -> bool {
        self.const_ == other.const_ && self.mut_ == other.mut_ && self.elem == other.elem
    }
}
impl Eq for typ::Ref {}
impl PartialEq for typ::Ref {
    fn eq(&self, other: &Self) -> bool {
        self.life == other.life && self.mut_ == other.mut_ && self.elem == other.elem
    }
}
impl Eq for typ::Slice {}
impl PartialEq for typ::Slice {
    fn eq(&self, other: &Self) -> bool {
        self.elem == other.elem
    }
}
impl Eq for typ::Trait {}
impl PartialEq for typ::Trait {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_ == other.dyn_ && self.bounds == other.bounds
    }
}
impl Eq for typ::Tuple {}
impl PartialEq for typ::Tuple {
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
impl Eq for item::Use::Glob {}
impl PartialEq for item::Use::Glob {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for item::Use::Group {}
impl PartialEq for item::Use::Group {
    fn eq(&self, other: &Self) -> bool {
        self.elems == other.elems
    }
}
impl Eq for item::Use::Name {}
impl PartialEq for item::Use::Name {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident
    }
}
impl Eq for item::Use::Path {}
impl PartialEq for item::Use::Path {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident && self.tree == other.tree
    }
}
impl Eq for item::Use::Rename {}
impl PartialEq for item::Use::Rename {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident && self.rename == other.rename
    }
}
impl Eq for item::Use::Tree {}
impl PartialEq for item::Use::Tree {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (item::Use::Tree::Path(self0), item::Use::Tree::Path(other0)) => self0 == other0,
            (item::Use::Tree::Name(self0), item::Use::Tree::Name(other0)) => self0 == other0,
            (item::Use::Tree::Rename(self0), item::Use::Tree::Rename(other0)) => self0 == other0,
            (item::Use::Tree::Glob(self0), item::Use::Tree::Glob(other0)) => self0 == other0,
            (item::Use::Tree::Group(self0), item::Use::Tree::Group(other0)) => self0 == other0,
            _ => false,
        }
    }
}
impl Eq for item::Variadic {}
impl PartialEq for item::Variadic {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs && self.pat == other.pat && self.comma == other.comma
    }
}
impl Eq for data::Variant {}
impl PartialEq for data::Variant {
    fn eq(&self, other: &Self) -> bool {
        self.attrs == other.attrs
            && self.ident == other.ident
            && self.fields == other.fields
            && self.discrim == other.discrim
    }
}
impl Eq for data::Restricted {}
impl PartialEq for data::Restricted {
    fn eq(&self, other: &Self) -> bool {
        self.in_ == other.in_ && self.path == other.path
    }
}
impl Eq for data::Visibility {}
impl PartialEq for data::Visibility {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (data::Visibility::Public(_), data::Visibility::Public(_)) => true,
            (data::Visibility::Restricted(self0), data::Visibility::Restricted(other0)) => self0 == other0,
            (data::Visibility::Inherited, data::Visibility::Inherited) => true,
            _ => false,
        }
    }
}
impl Eq for gen::Where {}
impl PartialEq for gen::Where {
    fn eq(&self, other: &Self) -> bool {
        self.preds == other.preds
    }
}
impl Eq for gen::Where::Pred {}
impl PartialEq for gen::Where::Pred {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (gen::Where::Pred::Life(self0), gen::Where::Pred::Life(other0)) => self0 == other0,
            (gen::Where::Pred::Type(self0), gen::Where::Pred::Type(other0)) => self0 == other0,
            _ => false,
        }
    }
}
