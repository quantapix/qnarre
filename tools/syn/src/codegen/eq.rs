use crate::StreamHelper;
use crate::*;

impl Eq for typ::Abi {}
impl PartialEq for typ::Abi {
    fn eq(&self, x: &Self) -> bool {
        self.name == x.name
    }
}
impl Eq for path::Angled {}
impl PartialEq for path::Angled {
    fn eq(&self, x: &Self) -> bool {
        self.colon2 == x.colon2 && self.args == x.args
    }
}
impl Eq for expr::Arm {}
impl PartialEq for expr::Arm {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.pat == x.pat
            && self.guard == x.guard
            && self.body == x.body
            && self.comma == x.comma
    }
}
impl Eq for path::AssocConst {}
impl PartialEq for path::AssocConst {
    fn eq(&self, x: &Self) -> bool {
        self.ident == x.ident && self.args == x.args && self.val == x.val
    }
}
impl Eq for path::AssocType {}
impl PartialEq for path::AssocType {
    fn eq(&self, x: &Self) -> bool {
        self.ident == x.ident && self.args == x.args && self.typ == x.typ
    }
}
impl Eq for attr::Style {}
impl PartialEq for attr::Style {
    fn eq(&self, x: &Self) -> bool {
        match (self, x) {
            (attr::Style::Outer, attr::Style::Outer) => true,
            (attr::Style::Inner(_), attr::Style::Inner(_)) => true,
            _ => false,
        }
    }
}
impl Eq for attr::Attr {}
impl PartialEq for attr::Attr {
    fn eq(&self, x: &Self) -> bool {
        self.style == x.style && self.meta == x.meta
    }
}
impl Eq for typ::FnArg {}
impl PartialEq for typ::FnArg {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.name == x.name && self.typ == x.typ
    }
}
impl Eq for typ::Variadic {}
impl PartialEq for typ::Variadic {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.name == x.name && self.comma == x.comma
    }
}
impl Eq for expr::BinOp {}
impl PartialEq for expr::BinOp {
    fn eq(&self, x: &Self) -> bool {
        use expr::BinOp::*;
        match (self, x) {
            (Add(_), Add(_)) => true,
            (Sub(_), Sub(_)) => true,
            (Mul(_), Mul(_)) => true,
            (Div(_), Div(_)) => true,
            (Rem(_), Rem(_)) => true,
            (And(_), And(_)) => true,
            (Or(_), Or(_)) => true,
            (BitXor(_), BitXor(_)) => true,
            (BitAnd(_), BitAnd(_)) => true,
            (BitOr(_), BitOr(_)) => true,
            (Shl(_), Shl(_)) => true,
            (Shr(_), Shr(_)) => true,
            (Eq(_), Eq(_)) => true,
            (Lt(_), Lt(_)) => true,
            (Le(_), Le(_)) => true,
            (Ne(_), Ne(_)) => true,
            (Ge(_), Ge(_)) => true,
            (Gt(_), Gt(_)) => true,
            (AddAssign(_), AddAssign(_)) => true,
            (SubAssign(_), SubAssign(_)) => true,
            (MulAssign(_), MulAssign(_)) => true,
            (DivAssign(_), DivAssign(_)) => true,
            (RemAssign(_), RemAssign(_)) => true,
            (BitXorAssign(_), BitXorAssign(_)) => true,
            (BitAndAssign(_), BitAndAssign(_)) => true,
            (BitOrAssign(_), BitOrAssign(_)) => true,
            (ShlAssign(_), ShlAssign(_)) => true,
            (ShrAssign(_), ShrAssign(_)) => true,
            _ => false,
        }
    }
}
impl Eq for stmt::Block {}
impl PartialEq for stmt::Block {
    fn eq(&self, x: &Self) -> bool {
        self.stmts == x.stmts
    }
}
impl Eq for gen::bound::Lifes {}
impl PartialEq for gen::bound::Lifes {
    fn eq(&self, x: &Self) -> bool {
        self.lifes == x.lifes
    }
}
impl Eq for gen::param::Const {}
impl PartialEq for gen::param::Const {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.ident == x.ident
            && self.typ == x.typ
            && self.eq == x.eq
            && self.default == x.default
    }
}
impl Eq for path::Constraint {}
impl PartialEq for path::Constraint {
    fn eq(&self, x: &Self) -> bool {
        self.ident == x.ident && self.args == x.args && self.bounds == x.bounds
    }
}
impl Eq for data::Data {}
impl PartialEq for data::Data {
    fn eq(&self, x: &Self) -> bool {
        use data::Data::*;
        match (self, x) {
            (Struct(x), Struct(y)) => x == y,
            (Enum(x), Enum(y)) => x == y,
            (Union(x), Union(y)) => x == y,
            _ => false,
        }
    }
}
impl Eq for data::Enum {}
impl PartialEq for data::Enum {
    fn eq(&self, x: &Self) -> bool {
        self.variants == x.variants
    }
}
impl Eq for data::Struct {}
impl PartialEq for data::Struct {
    fn eq(&self, x: &Self) -> bool {
        self.fields == x.fields && self.semi == x.semi
    }
}
impl Eq for data::Union {}
impl PartialEq for data::Union {
    fn eq(&self, x: &Self) -> bool {
        self.fields == x.fields
    }
}
impl Eq for Input {}
impl PartialEq for Input {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.ident == x.ident
            && self.gens == x.gens
            && self.data == x.data
    }
}
impl Eq for expr::Expr {}
impl PartialEq for expr::Expr {
    fn eq(&self, x: &Self) -> bool {
        use expr::Expr::*;
        match (self, x) {
            (Array(x), Array(y)) => x == y,
            (Assign(x), Assign(y)) => x == y,
            (Async(x), Async(y)) => x == y,
            (Await(x), Await(y)) => x == y,
            (Binary(x), Binary(y)) => x == y,
            (Block(x), Block(y)) => x == y,
            (Break(x), Break(y)) => x == y,
            (Call(x), Call(y)) => x == y,
            (Cast(x), Cast(y)) => x == y,
            (Closure(x), Closure(y)) => x == y,
            (Const(x), Const(y)) => x == y,
            (Continue(x), Continue(y)) => x == y,
            (Field(x), Field(y)) => x == y,
            (ForLoop(x), ForLoop(y)) => x == y,
            (Group(x), Group(y)) => x == y,
            (If(x), If(y)) => x == y,
            (Index(x), Index(y)) => x == y,
            (Infer(x), Infer(y)) => x == y,
            (Let(x), Let(y)) => x == y,
            (Lit(x), Lit(y)) => x == y,
            (Loop(x), Loop(y)) => x == y,
            (Mac(x), Mac(y)) => x == y,
            (Match(x), Match(y)) => x == y,
            (MethodCall(x), MethodCall(y)) => x == y,
            (Parenth(x), Parenth(y)) => x == y,
            (Path(x), Path(y)) => x == y,
            (Range(x), Range(y)) => x == y,
            (Ref(x), Ref(y)) => x == y,
            (Repeat(x), Repeat(y)) => x == y,
            (Return(x), Return(y)) => x == y,
            (Struct(x), Struct(y)) => x == y,
            (Try(x), Try(y)) => x == y,
            (TryBlock(x), TryBlock(y)) => x == y,
            (Tuple(x), Tuple(y)) => x == y,
            (Unary(x), Unary(y)) => x == y,
            (Unsafe(x), Unsafe(y)) => x == y,
            (Verbatim(x), Verbatim(y)) => StreamHelper(x) == StreamHelper(y),
            (While(x), While(y)) => x == y,
            (Yield(x), Yield(y)) => x == y,
            _ => false,
        }
    }
}
impl Eq for expr::Array {}
impl PartialEq for expr::Array {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.elems == x.elems
    }
}
impl Eq for expr::Assign {}
impl PartialEq for expr::Assign {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.left == x.left && self.right == x.right
    }
}
impl Eq for expr::Async {}
impl PartialEq for expr::Async {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.move_ == x.move_ && self.block == x.block
    }
}
impl Eq for expr::Await {}
impl PartialEq for expr::Await {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.expr == x.expr
    }
}
impl Eq for expr::Binary {}
impl PartialEq for expr::Binary {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.left == x.left && self.op == x.op && self.right == x.right
    }
}
impl Eq for expr::Block {}
impl PartialEq for expr::Block {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.label == x.label && self.block == x.block
    }
}
impl Eq for expr::Break {}
impl PartialEq for expr::Break {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.life == x.life && self.val == x.val
    }
}
impl Eq for expr::Call {}
impl PartialEq for expr::Call {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.func == x.func && self.args == x.args
    }
}
impl Eq for expr::Cast {}
impl PartialEq for expr::Cast {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.expr == x.expr && self.typ == x.typ
    }
}
impl Eq for expr::Closure {}
impl PartialEq for expr::Closure {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.lifes == x.lifes
            && self.const_ == x.const_
            && self.static_ == x.static_
            && self.async_ == x.async_
            && self.move_ == x.move_
            && self.ins == x.ins
            && self.ret == x.ret
            && self.body == x.body
    }
}
impl Eq for expr::Const {}
impl PartialEq for expr::Const {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.block == x.block
    }
}
impl Eq for expr::Continue {}
impl PartialEq for expr::Continue {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.life == x.life
    }
}
impl Eq for expr::Field {}
impl PartialEq for expr::Field {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.expr == x.expr && self.memb == x.memb
    }
}
impl Eq for expr::ForLoop {}
impl PartialEq for expr::ForLoop {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.label == x.label
            && self.pat == x.pat
            && self.expr == x.expr
            && self.body == x.body
    }
}
impl Eq for expr::Group {}
impl PartialEq for expr::Group {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.expr == x.expr
    }
}
impl Eq for expr::If {}
impl PartialEq for expr::If {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.cond == x.cond && self.then_ == x.then_ && self.else_ == x.else_
    }
}
impl Eq for expr::Index {}
impl PartialEq for expr::Index {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.expr == x.expr && self.idx == x.idx
    }
}
impl Eq for expr::Infer {}
impl PartialEq for expr::Infer {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
    }
}
impl Eq for expr::Let {}
impl PartialEq for expr::Let {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.pat == x.pat && self.expr == x.expr
    }
}
impl Eq for expr::Lit {}
impl PartialEq for expr::Lit {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.lit == x.lit
    }
}
impl Eq for expr::Loop {}
impl PartialEq for expr::Loop {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.label == x.label && self.body == x.body
    }
}
impl Eq for expr::Mac {}
impl PartialEq for expr::Mac {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.mac == x.mac
    }
}
impl Eq for expr::Match {}
impl PartialEq for expr::Match {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.expr == x.expr && self.arms == x.arms
    }
}
impl Eq for expr::MethodCall {}
impl PartialEq for expr::MethodCall {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.expr == x.expr
            && self.method == x.method
            && self.turbofish == x.turbofish
            && self.args == x.args
    }
}
impl Eq for expr::Parenth {}
impl PartialEq for expr::Parenth {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.expr == x.expr
    }
}
impl Eq for expr::Path {}
impl PartialEq for expr::Path {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.qself == x.qself && self.path == x.path
    }
}
impl Eq for expr::Range {}
impl PartialEq for expr::Range {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.beg == x.beg && self.limits == x.limits && self.end == x.end
    }
}
impl Eq for expr::Ref {}
impl PartialEq for expr::Ref {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.mut_ == x.mut_ && self.expr == x.expr
    }
}
impl Eq for expr::Repeat {}
impl PartialEq for expr::Repeat {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.expr == x.expr && self.len == x.len
    }
}
impl Eq for expr::Return {}
impl PartialEq for expr::Return {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.expr == x.expr
    }
}
impl Eq for expr::Struct {}
impl PartialEq for expr::Struct {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.qself == x.qself
            && self.path == x.path
            && self.fields == x.fields
            && self.dot2 == x.dot2
            && self.rest == x.rest
    }
}
impl Eq for expr::Try {}
impl PartialEq for expr::Try {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.expr == x.expr
    }
}
impl Eq for expr::TryBlock {}
impl PartialEq for expr::TryBlock {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.block == x.block
    }
}
impl Eq for expr::Tuple {}
impl PartialEq for expr::Tuple {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.elems == x.elems
    }
}
impl Eq for expr::Unary {}
impl PartialEq for expr::Unary {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.op == x.op && self.expr == x.expr
    }
}
impl Eq for expr::Unsafe {}
impl PartialEq for expr::Unsafe {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.block == x.block
    }
}
impl Eq for expr::While {}
impl PartialEq for expr::While {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.label == x.label && self.cond == x.cond && self.block == x.block
    }
}
impl Eq for expr::Yield {}
impl PartialEq for expr::Yield {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.expr == x.expr
    }
}
impl Eq for data::Field {}
impl PartialEq for data::Field {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.mut_ == x.mut_
            && self.ident == x.ident
            && self.colon == x.colon
            && self.typ == x.typ
    }
}
impl Eq for data::Mut {}
impl PartialEq for data::Mut {
    fn eq(&self, x: &Self) -> bool {
        match (self, x) {
            (data::Mut::None, data::Mut::None) => true,
        }
    }
}
impl Eq for pat::Field {}
impl PartialEq for pat::Field {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.memb == x.memb && self.colon == x.colon && self.pat == x.pat
    }
}
impl Eq for expr::FieldValue {}
impl PartialEq for expr::FieldValue {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.memb == x.memb && self.colon == x.colon && self.expr == x.expr
    }
}
impl Eq for data::Fields {}
impl PartialEq for data::Fields {
    fn eq(&self, x: &Self) -> bool {
        use data::Fields::*;
        match (self, x) {
            (Named(x), Named(y)) => x == y,
            (Unit, Unit) => true,
            (Unnamed(x), Unnamed(y)) => x == y,
            _ => false,
        }
    }
}
impl Eq for data::Named {}
impl PartialEq for data::Named {
    fn eq(&self, x: &Self) -> bool {
        self.fields == x.fields
    }
}
impl Eq for data::Unnamed {}
impl PartialEq for data::Unnamed {
    fn eq(&self, x: &Self) -> bool {
        self.fields == x.fields
    }
}
impl Eq for item::File {}
impl PartialEq for item::File {
    fn eq(&self, x: &Self) -> bool {
        self.shebang == x.shebang && self.attrs == x.attrs && self.items == x.items
    }
}
impl Eq for item::FnArg {}
impl PartialEq for item::FnArg {
    fn eq(&self, x: &Self) -> bool {
        use item::FnArg::*;
        match (self, x) {
            (Receiver(x), Receiver(y)) => x == y,
            (Type(x), Type(y)) => x == y,
            _ => false,
        }
    }
}
impl Eq for item::foreign::Item {}
impl PartialEq for item::foreign::Item {
    fn eq(&self, x: &Self) -> bool {
        use item::foreign::Item::*;
        match (self, x) {
            (Fn(x), Fn(y)) => x == y,
            (Mac(x), Mac(y)) => x == y,
            (Static(x), Static(y)) => x == y,
            (Type(x), Type(y)) => x == y,
            (Verbatim(x), Verbatim(y)) => StreamHelper(x) == StreamHelper(y),
            _ => false,
        }
    }
}
impl Eq for item::foreign::Fn {}
impl PartialEq for item::foreign::Fn {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.vis == x.vis && self.sig == x.sig
    }
}
impl Eq for item::foreign::Mac {}
impl PartialEq for item::foreign::Mac {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.mac == x.mac && self.semi == x.semi
    }
}
impl Eq for item::foreign::Static {}
impl PartialEq for item::foreign::Static {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.vis == x.vis && self.mut_ == x.mut_ && self.ident == x.ident && self.typ == x.typ
    }
}
impl Eq for item::foreign::Type {}
impl PartialEq for item::foreign::Type {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.vis == x.vis && self.ident == x.ident && self.gens == x.gens
    }
}
impl Eq for path::Arg {}
impl PartialEq for path::Arg {
    fn eq(&self, x: &Self) -> bool {
        use path::Arg::*;
        match (self, x) {
            (AssocConst(x), AssocConst(y)) => x == y,
            (AssocType(x), AssocType(y)) => x == y,
            (Const(x), Const(y)) => x == y,
            (Constraint(x), Constraint(y)) => x == y,
            (Life(x), Life(y)) => x == y,
            (Type(x), Type(y)) => x == y,
            _ => false,
        }
    }
}
impl Eq for gen::Param {}
impl PartialEq for gen::Param {
    fn eq(&self, x: &Self) -> bool {
        use gen::Param::*;
        match (self, x) {
            (Const(x), Const(y)) => x == y,
            (Life(x), Life(y)) => x == y,
            (Type(x), Type(y)) => x == y,
            _ => false,
        }
    }
}
impl Eq for gen::Gens {}
impl PartialEq for gen::Gens {
    fn eq(&self, x: &Self) -> bool {
        self.lt == x.lt && self.params == x.params && self.gt == x.gt && self.where_ == x.where_
    }
}
impl Eq for item::impl_::Item {}
impl PartialEq for item::impl_::Item {
    fn eq(&self, x: &Self) -> bool {
        use item::impl_::Item::*;
        match (self, x) {
            (Const(x), Const(y)) => x == y,
            (Fn(x), Fn(y)) => x == y,
            (Mac(x), Mac(y)) => x == y,
            (Type(x), Type(y)) => x == y,
            (Verbatim(x), Verbatim(y)) => StreamHelper(x) == StreamHelper(y),
            _ => false,
        }
    }
}
impl Eq for item::impl_::Const {}
impl PartialEq for item::impl_::Const {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.default == x.default
            && self.ident == x.ident
            && self.gens == x.gens
            && self.typ == x.typ
            && self.expr == x.expr
    }
}
impl Eq for item::impl_::Fn {}
impl PartialEq for item::impl_::Fn {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.default == x.default
            && self.sig == x.sig
            && self.block == x.block
    }
}
impl Eq for item::impl_::Mac {}
impl PartialEq for item::impl_::Mac {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.mac == x.mac && self.semi == x.semi
    }
}
impl Eq for item::impl_::Type {}
impl PartialEq for item::impl_::Type {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.default == x.default
            && self.ident == x.ident
            && self.gens == x.gens
            && self.typ == x.typ
    }
}
impl Eq for item::impl_::Restriction {}
impl PartialEq for item::impl_::Restriction {
    fn eq(&self, _other: &Self) -> bool {
        match *self {}
    }
}
impl Eq for item::Item {}
impl PartialEq for item::Item {
    fn eq(&self, x: &Self) -> bool {
        use item::Item::*;
        match (self, x) {
            (Const(x), Const(y)) => x == y,
            (Enum(x), Enum(y)) => x == y,
            (Extern(x), Extern(y)) => x == y,
            (Fn(x), Fn(y)) => x == y,
            (Foreign(x), Foreign(y)) => x == y,
            (Impl(x), Impl(y)) => x == y,
            (Mac(x), Mac(y)) => x == y,
            (Mod(x), Mod(y)) => x == y,
            (Static(x), Static(y)) => x == y,
            (Struct(x), Struct(y)) => x == y,
            (Trait(x), Trait(y)) => x == y,
            (TraitAlias(x), TraitAlias(y)) => x == y,
            (Type(x), Type(y)) => x == y,
            (Union(x), Union(y)) => x == y,
            (Use(x), Use(y)) => x == y,
            (Verbatim(x), Verbatim(y)) => StreamHelper(x) == StreamHelper(y),
            _ => false,
        }
    }
}
impl Eq for item::Const {}
impl PartialEq for item::Const {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.ident == x.ident
            && self.gens == x.gens
            && self.typ == x.typ
            && self.expr == x.expr
    }
}
impl Eq for item::Enum {}
impl PartialEq for item::Enum {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.ident == x.ident
            && self.gens == x.gens
            && self.variants == x.variants
    }
}
impl Eq for item::Extern {}
impl PartialEq for item::Extern {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.vis == x.vis && self.ident == x.ident && self.rename == x.rename
    }
}
impl Eq for item::Fn {}
impl PartialEq for item::Fn {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.vis == x.vis && self.sig == x.sig && self.block == x.block
    }
}
impl Eq for item::Foreign {}
impl PartialEq for item::Foreign {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.unsafe_ == x.unsafe_ && self.abi == x.abi && self.items == x.items
    }
}
impl Eq for item::Impl {}
impl PartialEq for item::Impl {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.default == x.default
            && self.unsafe_ == x.unsafe_
            && self.gens == x.gens
            && self.trait_ == x.trait_
            && self.typ == x.typ
            && self.items == x.items
    }
}
impl Eq for item::Mac {}
impl PartialEq for item::Mac {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.ident == x.ident && self.mac == x.mac && self.semi == x.semi
    }
}
impl Eq for item::Mod {}
impl PartialEq for item::Mod {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.unsafe_ == x.unsafe_
            && self.ident == x.ident
            && self.items == x.items
            && self.semi == x.semi
    }
}
impl Eq for item::Static {}
impl PartialEq for item::Static {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.mut_ == x.mut_
            && self.ident == x.ident
            && self.typ == x.typ
            && self.expr == x.expr
    }
}
impl Eq for item::Struct {}
impl PartialEq for item::Struct {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.ident == x.ident
            && self.gens == x.gens
            && self.fields == x.fields
            && self.semi == x.semi
    }
}
impl Eq for item::Trait {}
impl PartialEq for item::Trait {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.unsafe_ == x.unsafe_
            && self.auto == x.auto
            && self.restriction == x.restriction
            && self.ident == x.ident
            && self.gens == x.gens
            && self.colon == x.colon
            && self.supers == x.supers
            && self.items == x.items
    }
}
impl Eq for item::TraitAlias {}
impl PartialEq for item::TraitAlias {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.ident == x.ident
            && self.gens == x.gens
            && self.bounds == x.bounds
    }
}
impl Eq for item::Type {}
impl PartialEq for item::Type {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.vis == x.vis && self.ident == x.ident && self.gens == x.gens && self.typ == x.typ
    }
}
impl Eq for item::Union {}
impl PartialEq for item::Union {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.ident == x.ident
            && self.gens == x.gens
            && self.fields == x.fields
    }
}
impl Eq for item::Use {}
impl PartialEq for item::Use {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.vis == x.vis && self.colon == x.colon && self.tree == x.tree
    }
}
impl Eq for expr::Label {}
impl PartialEq for expr::Label {
    fn eq(&self, x: &Self) -> bool {
        self.name == x.name
    }
}
impl Eq for gen::param::Life {}
impl PartialEq for gen::param::Life {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.life == x.life && self.colon == x.colon && self.bounds == x.bounds
    }
}
impl Eq for lit::Lit {}
impl PartialEq for lit::Lit {
    fn eq(&self, x: &Self) -> bool {
        use lit::Lit::*;
        match (self, x) {
            (Str(x), Str(y)) => x == y,
            (ByteStr(x), ByteStr(y)) => x == y,
            (Byte(x), Byte(y)) => x == y,
            (Char(x), Char(y)) => x == y,
            (Int(x), Int(y)) => x == y,
            (Float(x), Float(y)) => x == y,
            (Bool(x), Bool(y)) => x == y,
            (Verbatim(x), Verbatim(y)) => x.to_string() == y.to_string(),
            _ => false,
        }
    }
}
impl Eq for lit::Bool {}
impl PartialEq for lit::Bool {
    fn eq(&self, x: &Self) -> bool {
        self.val == x.val
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
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.pat == x.pat && self.init == x.init
    }
}
impl Eq for stmt::Init {}
impl PartialEq for stmt::Init {
    fn eq(&self, x: &Self) -> bool {
        self.expr == x.expr && self.diverge == x.diverge
    }
}
impl Eq for mac::Mac {}
impl PartialEq for mac::Mac {
    fn eq(&self, x: &Self) -> bool {
        self.path == x.path && self.delim == x.delim && StreamHelper(&self.toks) == StreamHelper(&x.toks)
    }
}
impl Eq for tok::Delim {}
impl PartialEq for tok::Delim {
    fn eq(&self, x: &Self) -> bool {
        use tok::Delim::*;
        match (self, x) {
            (Brace(_), Brace(_)) => true,
            (Bracket(_), Bracket(_)) => true,
            (Parenth(_), Parenth(_)) => true,
            _ => false,
        }
    }
}
impl Eq for attr::Meta {}
impl PartialEq for attr::Meta {
    fn eq(&self, x: &Self) -> bool {
        use attr::Meta::*;
        match (self, x) {
            (List(x), List(y)) => x == y,
            (NameValue(x), NameValue(y)) => x == y,
            (Path(x), Path(y)) => x == y,
            _ => false,
        }
    }
}
impl Eq for attr::List {}
impl PartialEq for attr::List {
    fn eq(&self, x: &Self) -> bool {
        self.path == x.path && self.delim == x.delim && StreamHelper(&self.toks) == StreamHelper(&x.toks)
    }
}
impl Eq for attr::NameValue {}
impl PartialEq for attr::NameValue {
    fn eq(&self, x: &Self) -> bool {
        self.name == x.name && self.val == x.val
    }
}
impl Eq for path::Parenthed {}
impl PartialEq for path::Parenthed {
    fn eq(&self, x: &Self) -> bool {
        self.args == x.args && self.ret == x.ret
    }
}
impl Eq for pat::Pat {}
impl PartialEq for pat::Pat {
    fn eq(&self, x: &Self) -> bool {
        use pat::Pat::*;
        match (self, x) {
            (Const(x), Const(y)) => x == y,
            (Ident(x), Ident(y)) => x == y,
            (Lit(x), Lit(y)) => x == y,
            (Mac(x), Mac(y)) => x == y,
            (Or(x), Or(y)) => x == y,
            (Parenth(x), Parenth(y)) => x == y,
            (Path(x), Path(y)) => x == y,
            (Range(x), Range(y)) => x == y,
            (Ref(x), Ref(y)) => x == y,
            (Rest(x), Rest(y)) => x == y,
            (Slice(x), Slice(y)) => x == y,
            (Struct(x), Struct(y)) => x == y,
            (Tuple(x), Tuple(y)) => x == y,
            (TupleStruct(x), TupleStruct(y)) => x == y,
            (Type(x), Type(y)) => x == y,
            (Verbatim(x), Verbatim(y)) => StreamHelper(x) == StreamHelper(y),
            (Wild(x), Wild(y)) => x == y,
            _ => false,
        }
    }
}
impl Eq for pat::Ident {}
impl PartialEq for pat::Ident {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.ref_ == x.ref_
            && self.mut_ == x.mut_
            && self.ident == x.ident
            && self.sub == x.sub
    }
}
impl Eq for pat::Or {}
impl PartialEq for pat::Or {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.vert == x.vert && self.cases == x.cases
    }
}
impl Eq for pat::Parenth {}
impl PartialEq for pat::Parenth {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.pat == x.pat
    }
}
impl Eq for pat::Ref {}
impl PartialEq for pat::Ref {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.mut_ == x.mut_ && self.pat == x.pat
    }
}
impl Eq for pat::Rest {}
impl PartialEq for pat::Rest {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
    }
}
impl Eq for pat::Slice {}
impl PartialEq for pat::Slice {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.pats == x.pats
    }
}
impl Eq for pat::Struct {}
impl PartialEq for pat::Struct {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.qself == x.qself
            && self.path == x.path
            && self.fields == x.fields
            && self.rest == x.rest
    }
}
impl Eq for pat::Tuple {}
impl PartialEq for pat::Tuple {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.pats == x.pats
    }
}
impl Eq for pat::TupleStruct {}
impl PartialEq for pat::TupleStruct {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.qself == x.qself && self.path == x.path && self.pats == x.pats
    }
}
impl Eq for pat::Type {}
impl PartialEq for pat::Type {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.pat == x.pat && self.typ == x.typ
    }
}
impl Eq for pat::Wild {}
impl PartialEq for pat::Wild {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
    }
}
impl Eq for Path {}
impl PartialEq for Path {
    fn eq(&self, x: &Self) -> bool {
        self.colon == x.colon && self.segs == x.segs
    }
}
impl Eq for path::Args {}
impl PartialEq for path::Args {
    fn eq(&self, x: &Self) -> bool {
        use path::Args::*;
        match (self, x) {
            (None, None) => true,
            (Angled(x), Angled(y)) => x == y,
            (Parenthed(x), Parenthed(y)) => x == y,
            _ => false,
        }
    }
}
impl Eq for path::Segment {}
impl PartialEq for path::Segment {
    fn eq(&self, x: &Self) -> bool {
        self.ident == x.ident && self.args == x.args
    }
}
impl Eq for gen::where_::Life {}
impl PartialEq for gen::where_::Life {
    fn eq(&self, x: &Self) -> bool {
        self.life == x.life && self.bounds == x.bounds
    }
}
impl Eq for gen::where_::Type {}
impl PartialEq for gen::where_::Type {
    fn eq(&self, x: &Self) -> bool {
        self.lifes == x.lifes && self.typ == x.typ && self.bounds == x.bounds
    }
}
impl Eq for path::QSelf {}
impl PartialEq for path::QSelf {
    fn eq(&self, x: &Self) -> bool {
        self.ty == x.ty && self.pos == x.pos && self.as_ == x.as_
    }
}
impl Eq for expr::Limits {}
impl PartialEq for expr::Limits {
    fn eq(&self, x: &Self) -> bool {
        use expr::Limits::*;
        match (self, x) {
            (Closed(_), Closed(_)) => true,
            (HalfOpen(_), HalfOpen(_)) => true,
            _ => false,
        }
    }
}
impl Eq for item::Receiver {}
impl PartialEq for item::Receiver {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.ref_ == x.ref_
            && self.mut_ == x.mut_
            && self.colon == x.colon
            && self.typ == x.typ
    }
}
impl Eq for typ::Ret {}
impl PartialEq for typ::Ret {
    fn eq(&self, x: &Self) -> bool {
        use typ::Ret::*;
        match (self, x) {
            (Default, Default) => true,
            (Type(_, x), Type(_, y)) => x == y,
            _ => false,
        }
    }
}
impl Eq for item::Sig {}
impl PartialEq for item::Sig {
    fn eq(&self, x: &Self) -> bool {
        self.const_ == x.const_
            && self.async_ == x.async_
            && self.unsafe_ == x.unsafe_
            && self.abi == x.abi
            && self.ident == x.ident
            && self.gens == x.gens
            && self.args == x.args
            && self.vari == x.vari
            && self.ret == x.ret
    }
}
impl Eq for item::StaticMut {}
impl PartialEq for item::StaticMut {
    fn eq(&self, x: &Self) -> bool {
        use item::StaticMut::*;
        match (self, x) {
            (Mut(_), Mut(_)) => true,
            (None, None) => true,
            _ => false,
        }
    }
}
impl Eq for stmt::Stmt {}
impl PartialEq for stmt::Stmt {
    fn eq(&self, x: &Self) -> bool {
        use stmt::Stmt::*;
        match (self, x) {
            (Expr(x, self1), Expr(y, other1)) => x == y && self1 == other1,
            (Item(x), Item(y)) => x == y,
            (Local(x), Local(y)) => x == y,
            (Mac(x), Mac(y)) => x == y,
            _ => false,
        }
    }
}
impl Eq for stmt::Mac {}
impl PartialEq for stmt::Mac {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.mac == x.mac && self.semi == x.semi
    }
}
impl Eq for gen::bound::Trait {}
impl PartialEq for gen::bound::Trait {
    fn eq(&self, x: &Self) -> bool {
        self.parenth == x.parenth && self.modif == x.modif && self.lifes == x.lifes && self.path == x.path
    }
}
impl Eq for gen::bound::Modifier {}
impl PartialEq for gen::bound::Modifier {
    fn eq(&self, x: &Self) -> bool {
        use gen::bound::Modifier::*;
        match (self, x) {
            (Maybe(_), Maybe(_)) => true,
            (None, None) => true,
            _ => false,
        }
    }
}
impl Eq for item::trait_::Item {}
impl PartialEq for item::trait_::Item {
    fn eq(&self, x: &Self) -> bool {
        use item::trait_::Item::*;
        match (self, x) {
            (Const(x), Const(y)) => x == y,
            (Fn(x), Fn(y)) => x == y,
            (Mac(x), Mac(y)) => x == y,
            (Type(x), Type(y)) => x == y,
            (Verbatim(x), Verbatim(y)) => StreamHelper(x) == StreamHelper(y),
            _ => false,
        }
    }
}
impl Eq for item::trait_::Const {}
impl PartialEq for item::trait_::Const {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.ident == x.ident
            && self.gens == x.gens
            && self.typ == x.typ
            && self.default == x.default
    }
}
impl Eq for item::trait_::Fn {}
impl PartialEq for item::trait_::Fn {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.sig == x.sig && self.default == x.default && self.semi == x.semi
    }
}
impl Eq for item::trait_::Mac {}
impl PartialEq for item::trait_::Mac {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.mac == x.mac && self.semi == x.semi
    }
}
impl Eq for item::trait_::Type {}
impl PartialEq for item::trait_::Type {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.ident == x.ident
            && self.gens == x.gens
            && self.colon == x.colon
            && self.bounds == x.bounds
            && self.default == x.default
    }
}
impl Eq for typ::Type {}
impl PartialEq for typ::Type {
    fn eq(&self, x: &Self) -> bool {
        use typ::Type::*;
        match (self, x) {
            (Array(x), Array(y)) => x == y,
            (Fn(x), Fn(y)) => x == y,
            (Group(x), Group(y)) => x == y,
            (Impl(x), Impl(y)) => x == y,
            (Infer(x), Infer(y)) => x == y,
            (Mac(x), Mac(y)) => x == y,
            (Never(x), Never(y)) => x == y,
            (Parenth(x), Parenth(y)) => x == y,
            (Path(x), Path(y)) => x == y,
            (Ptr(x), Ptr(y)) => x == y,
            (Ref(x), Ref(y)) => x == y,
            (Slice(x), Slice(y)) => x == y,
            (Trait(x), Trait(y)) => x == y,
            (Tuple(x), Tuple(y)) => x == y,
            (Verbatim(x), Verbatim(y)) => StreamHelper(x) == StreamHelper(y),
            _ => false,
        }
    }
}
impl Eq for typ::Array {}
impl PartialEq for typ::Array {
    fn eq(&self, x: &Self) -> bool {
        self.elem == x.elem && self.len == x.len
    }
}
impl Eq for typ::Fn {}
impl PartialEq for typ::Fn {
    fn eq(&self, x: &Self) -> bool {
        self.lifes == x.lifes
            && self.unsafe_ == x.unsafe_
            && self.abi == x.abi
            && self.args == x.args
            && self.vari == x.vari
            && self.ret == x.ret
    }
}
impl Eq for typ::Group {}
impl PartialEq for typ::Group {
    fn eq(&self, x: &Self) -> bool {
        self.elem == x.elem
    }
}
impl Eq for typ::Impl {}
impl PartialEq for typ::Impl {
    fn eq(&self, x: &Self) -> bool {
        self.bounds == x.bounds
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
    fn eq(&self, x: &Self) -> bool {
        self.mac == x.mac
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
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.ident == x.ident
            && self.colon == x.colon
            && self.bounds == x.bounds
            && self.eq == x.eq
            && self.default == x.default
    }
}
impl Eq for gen::bound::Type {}
impl PartialEq for gen::bound::Type {
    fn eq(&self, x: &Self) -> bool {
        use gen::bound::Type::*;
        match (self, x) {
            (Life(x), Life(y)) => x == y,
            (Trait(x), Trait(y)) => x == y,
            (Verbatim(x), Verbatim(y)) => StreamHelper(x) == StreamHelper(y),
            _ => false,
        }
    }
}
impl Eq for typ::Parenth {}
impl PartialEq for typ::Parenth {
    fn eq(&self, x: &Self) -> bool {
        self.elem == x.elem
    }
}
impl Eq for typ::Path {}
impl PartialEq for typ::Path {
    fn eq(&self, x: &Self) -> bool {
        self.qself == x.qself && self.path == x.path
    }
}
impl Eq for typ::Ptr {}
impl PartialEq for typ::Ptr {
    fn eq(&self, x: &Self) -> bool {
        self.const_ == x.const_ && self.mut_ == x.mut_ && self.elem == x.elem
    }
}
impl Eq for typ::Ref {}
impl PartialEq for typ::Ref {
    fn eq(&self, x: &Self) -> bool {
        self.life == x.life && self.mut_ == x.mut_ && self.elem == x.elem
    }
}
impl Eq for typ::Slice {}
impl PartialEq for typ::Slice {
    fn eq(&self, x: &Self) -> bool {
        self.elem == x.elem
    }
}
impl Eq for typ::Trait {}
impl PartialEq for typ::Trait {
    fn eq(&self, x: &Self) -> bool {
        self.dyn_ == x.dyn_ && self.bounds == x.bounds
    }
}
impl Eq for typ::Tuple {}
impl PartialEq for typ::Tuple {
    fn eq(&self, x: &Self) -> bool {
        self.elems == x.elems
    }
}
impl Eq for expr::UnOp {}
impl PartialEq for expr::UnOp {
    fn eq(&self, x: &Self) -> bool {
        use expr::UnOp::*;
        match (self, x) {
            (Deref(_), Deref(_)) => true,
            (Neg(_), Neg(_)) => true,
            (Not(_), Not(_)) => true,
            _ => false,
        }
    }
}
impl Eq for item::use_::Glob {}
impl PartialEq for item::use_::Glob {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for item::use_::Group {}
impl PartialEq for item::use_::Group {
    fn eq(&self, x: &Self) -> bool {
        self.trees == x.trees
    }
}
impl Eq for item::use_::Name {}
impl PartialEq for item::use_::Name {
    fn eq(&self, x: &Self) -> bool {
        self.ident == x.ident
    }
}
impl Eq for item::use_::Path {}
impl PartialEq for item::use_::Path {
    fn eq(&self, x: &Self) -> bool {
        self.ident == x.ident && self.tree == x.tree
    }
}
impl Eq for item::use_::Rename {}
impl PartialEq for item::use_::Rename {
    fn eq(&self, x: &Self) -> bool {
        self.ident == x.ident && self.rename == x.rename
    }
}
impl Eq for item::use_::Tree {}
impl PartialEq for item::use_::Tree {
    fn eq(&self, x: &Self) -> bool {
        use item::use_::Tree::*;
        match (self, x) {
            (Glob(x), Glob(y)) => x == y,
            (Group(x), Group(y)) => x == y,
            (Name(x), Name(y)) => x == y,
            (Path(x), Path(y)) => x == y,
            (Rename(x), Rename(y)) => x == y,
            _ => false,
        }
    }
}
impl Eq for item::Variadic {}
impl PartialEq for item::Variadic {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.pat == x.pat && self.comma == x.comma
    }
}
impl Eq for data::Variant {}
impl PartialEq for data::Variant {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.ident == x.ident && self.fields == x.fields && self.discrim == x.discrim
    }
}
impl Eq for data::Restricted {}
impl PartialEq for data::Restricted {
    fn eq(&self, x: &Self) -> bool {
        self.in_ == x.in_ && self.path == x.path
    }
}
impl Eq for data::Visibility {}
impl PartialEq for data::Visibility {
    fn eq(&self, x: &Self) -> bool {
        use data::Visibility::*;
        match (self, x) {
            (Inherited, Inherited) => true,
            (Public(_), Public(_)) => true,
            (Restricted(x), Restricted(y)) => x == y,
            _ => false,
        }
    }
}
impl Eq for gen::Where {}
impl PartialEq for gen::Where {
    fn eq(&self, x: &Self) -> bool {
        self.preds == x.preds
    }
}
impl Eq for gen::where_::Pred {}
impl PartialEq for gen::where_::Pred {
    fn eq(&self, x: &Self) -> bool {
        use gen::where_::Pred::*;
        match (self, x) {
            (Life(x), Life(y)) => x == y,
            (Type(x), Type(y)) => x == y,
            _ => false,
        }
    }
}
