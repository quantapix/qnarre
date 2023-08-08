#![allow(unused_variables)]

use crate::*;

pub fn visit_abi<'a, V>(v: &mut V, self: &'a typ::Abi)
where
    V: Visit<'a> + ?Sized,
{
    if let Some(x) = &self.name {
        x.visit(v);
    }
}
pub fn visit_angle_bracketed_generic_arguments<'a, V>(v: &mut V, self: &'a path::Angled)
where
    V: Visit<'a> + ?Sized,
{
    for el in Puncted::pairs(&self.args) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_arm<'a, V>(v: &mut V, self: &'a expr::Arm)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.pat.visit(v);
    if let Some(x) = &self.guard {
        
        &*(x).1.visit(v);
    }
    &*self.body.visit(v);
}
pub fn visit_assoc_const<'a, V>(v: &mut V, self: &'a path::AssocConst)
where
    V: Visit<'a> + ?Sized,
{
    &self.ident.visit(v);
    if let Some(x) = &self.gnrs {
        x.visit(v);
    }
    &self.val.visit(v);
}
pub fn visit_assoc_type<'a, V>(v: &mut V, self: &'a path::AssocType)
where
    V: Visit<'a> + ?Sized,
{
    &self.ident.visit(v);
    if let Some(x) = &self.gnrs {
        x.visit(v);
    }
    &self.ty.visit(v);
}
pub fn visit_attr_style<'a, V>(v: &mut V, self: &'a attr::Style)
where
    V: Visit<'a> + ?Sized,
{
    match self {
        attr::Style::Outer => {},
        attr::Style::Inner(x) => {
            
        },
    }
}
pub fn visit_attribute<'a, V>(v: &mut V, self: &'a attr::Attr)
where
    V: Visit<'a> + ?Sized,
{
    &self.style.visit(v);
    &self.meta.visit(v);
}
pub fn visit_bare_fn_arg<'a, V>(v: &mut V, self: &'a typ::FnArg)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.name {
        &(x).0.visit(v);
        
    }
    &self.typ.visit(v);
}
pub fn visit_bare_variadic<'a, V>(v: &mut V, self: &'a typ::Variadic)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.name {
        &(x).0.visit(v);
        
    }
}
pub fn visit_bin_op<'a, V>(v: &mut V, self: &'a expr::BinOp)
where
    V: Visit<'a> + ?Sized,
{
    use expr::BinOp::*;
    match self {
        Add(x) => {
            
        },
        Sub(x) => {
            
        },
        Mul(x) => {
            
        },
        Div(x) => {
            
        },
        Rem(x) => {
            
        },
        And(x) => {
            
        },
        Or(x) => {
            
        },
        BitXor(x) => {
            
        },
        BitAnd(x) => {
            
        },
        BitOr(x) => {
            
        },
        Shl(x) => {
            
        },
        Shr(x) => {
            
        },
        Eq(x) => {
            
        },
        Lt(x) => {
            
        },
        Le(x) => {
            
        },
        Ne(x) => {
            
        },
        Ge(x) => {
            
        },
        Gt(x) => {
            
        },
        AddAssign(x) => {
            
        },
        SubAssign(x) => {
            
        },
        MulAssign(x) => {
            
        },
        DivAssign(x) => {
            
        },
        RemAssign(x) => {
            
        },
        BitXorAssign(x) => {
            
        },
        BitAndAssign(x) => {
            
        },
        BitOrAssign(x) => {
            
        },
        ShlAssign(x) => {
            
        },
        ShrAssign(x) => {
            
        },
    }
}
pub fn visit_block<'a, V>(v: &mut V, self: &'a stmt::Block)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.stmts {
        x.visit(v);
    }
}
pub fn visit_bound_lifetimes<'a, V>(v: &mut V, self: &'a gen::bound::Lifes)
where
    V: Visit<'a> + ?Sized,
{
    for el in Puncted::pairs(&self.lifes) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_const_param<'a, V>(v: &mut V, self: &'a gen::param::Const)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.ident.visit(v);
    &self.typ.visit(v);
    if let Some(x) = &self.default {
        x.visit(v);
    }
}
pub fn visit_constraint<'a, V>(v: &mut V, self: &'a path::Constraint)
where
    V: Visit<'a> + ?Sized,
{
    &self.ident.visit(v);
    if let Some(x) = &self.gnrs {
        x.visit(v);
    }
    for el in Puncted::pairs(&self.bounds) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_data<'a, V>(v: &mut V, self: &'a data::Data)
where
    V: Visit<'a> + ?Sized,
{
    match self {
        data::Data::Struct(x) => {
            x.visit(v);
        },
        data::Data::Enum(x) => {
            x.visit(v);
        },
        data::Data::Union(x) => {
            x.visit(v);
        },
    }
}
pub fn visit_data_enum<'a, V>(v: &mut V, self: &'a data::Enum)
where
    V: Visit<'a> + ?Sized,
{
    for el in Puncted::pairs(&self.variants) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_data_struct<'a, V>(v: &mut V, self: &'a data::Struct)
where
    V: Visit<'a> + ?Sized,
{
    &self.fields.visit(v);
}
pub fn visit_data_union<'a, V>(v: &mut V, self: &'a data::Union)
where
    V: Visit<'a> + ?Sized,
{
    &self.fields.visit(v);
}
pub fn visit_derive_input<'a, V>(v: &mut V, self: &'a Input)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    &self.gens.visit(v);
    &self.data.visit(v);
}
pub fn visit_expr<'a, V>(v: &mut V, self: &'a expr::Expr)
where
    V: Visit<'a> + ?Sized,
{
    use expr::Expr::*;
    match self {
        Array(x) => {
            x.visit(v);
        },
        Assign(x) => {
            x.visit(v);
        },
        Async(x) => {
            x.visit(v);
        },
        Await(x) => {
            x.visit(v);
        },
        Binary(x) => {
            x.visit(v);
        },
        Block(x) => {
            x.visit(v);
        },
        Break(x) => {
            x.visit(v);
        },
        Call(x) => {
            x.visit(v);
        },
        Cast(x) => {
            x.visit(v);
        },
        Closure(x) => {
            x.visit(v);
        },
        Const(x) => {
            x.visit(v);
        },
        Continue(x) => {
            x.visit(v);
        },
        Field(x) => {
            x.visit(v);
        },
        ForLoop(x) => {
            x.visit(v);
        },
        Group(x) => {
            x.visit(v);
        },
        If(x) => {
            x.visit(v);
        },
        Index(x) => {
            x.visit(v);
        },
        Infer(x) => {
            x.visit(v);
        },
        Let(x) => {
            x.visit(v);
        },
        Lit(x) => {
            x.visit(v);
        },
        Loop(x) => {
            x.visit(v);
        },
        Mac(x) => {
            x.visit(v);
        },
        Match(x) => {
            x.visit(v);
        },
        MethodCall(x) => {
            x.visit(v);
        },
        Parenth(x) => {
            x.visit(v);
        },
        Path(x) => {
            x.visit(v);
        },
        Range(x) => {
            x.visit(v);
        },
        Ref(x) => {
            x.visit(v);
        },
        Repeat(x) => {
            x.visit(v);
        },
        Return(x) => {
            x.visit(v);
        },
        Struct(x) => {
            x.visit(v);
        },
        Try(x) => {
            x.visit(v);
        },
        TryBlock(x) => {
            x.visit(v);
        },
        Tuple(x) => {
            x.visit(v);
        },
        Unary(x) => {
            x.visit(v);
        },
        Unsafe(x) => {
            x.visit(v);
        },
        Verbatim(x) => {
            
        },
        While(x) => {
            x.visit(v);
        },
        Yield(x) => {
            x.visit(v);
        },
    }
}
pub fn visit_expr_array<'a, V>(v: &mut V, self: &'a expr::Array)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    for el in Puncted::pairs(&self.elems) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_expr_assign<'a, V>(v: &mut V, self: &'a expr::Assign)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.left.visit(v);
    &*self.right.visit(v);
}
pub fn visit_expr_async<'a, V>(v: &mut V, self: &'a expr::Async)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.block.visit(v);
}
pub fn visit_expr_await<'a, V>(v: &mut V, self: &'a expr::Await)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
}
pub fn visit_expr_binary<'a, V>(v: &mut V, self: &'a expr::Binary)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.left.visit(v);
    &self.op.visit(v);
    &*self.right.visit(v);
}
pub fn visit_expr_block<'a, V>(v: &mut V, self: &'a expr::Block)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.label {
        x.visit(v);
    }
    &self.block.visit(v);
}
pub fn visit_expr_break<'a, V>(v: &mut V, self: &'a expr::Break)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.life {
        x.visit(v);
    }
    if let Some(x) = &self.val {
        &**x.visit(v);
    }
}
pub fn visit_expr_call<'a, V>(v: &mut V, self: &'a expr::Call)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.func.visit(v);
    for el in Puncted::pairs(&self.args) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_expr_cast<'a, V>(v: &mut V, self: &'a expr::Cast)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
    &*self.typ.visit(v);
}
pub fn visit_expr_closure<'a, V>(v: &mut V, self: &'a expr::Closure)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.lifes {
        x.visit(v);
    }
    for el in Puncted::pairs(&self.inputs) {
        let x = el.value();
        x.visit(v);
    }
    &self.ret.visit(v);
    &*self.body.visit(v);
}
pub fn visit_expr_const<'a, V>(v: &mut V, self: &'a expr::Const)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.block.visit(v);
}
pub fn visit_expr_continue<'a, V>(v: &mut V, self: &'a expr::Continue)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.life {
        x.visit(v);
    }
}
pub fn visit_expr_field<'a, V>(v: &mut V, self: &'a expr::Field)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
    &self.memb.visit(v);
}
pub fn visit_expr_for_loop<'a, V>(v: &mut V, self: &'a expr::ForLoop)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.label {
        x.visit(v);
    }
    &*self.pat.visit(v);
    &*self.expr.visit(v);
    &self.body.visit(v);
}
pub fn visit_expr_group<'a, V>(v: &mut V, self: &'a expr::Group)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
}
pub fn visit_expr_if<'a, V>(v: &mut V, self: &'a expr::If)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.cond.visit(v);
    &self.then_branch.visit(v);
    if let Some(x) = &self.else_branch {
        
        &*(x).1.visit(v);
    }
}
pub fn visit_expr_index<'a, V>(v: &mut V, self: &'a expr::Index)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
    &*self.index.visit(v);
}
pub fn visit_expr_infer<'a, V>(v: &mut V, self: &'a expr::Infer)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
}
pub fn visit_expr_let<'a, V>(v: &mut V, self: &'a expr::Let)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.pat.visit(v);
    &*self.expr.visit(v);
}
pub fn visit_expr_lit<'a, V>(v: &mut V, self: &'a expr::Lit)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.lit.visit(v);
}
pub fn visit_expr_loop<'a, V>(v: &mut V, self: &'a expr::Loop)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.label {
        x.visit(v);
    }
    &self.body.visit(v);
}
pub fn visit_expr_macro<'a, V>(v: &mut V, self: &'a expr::Mac)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.mac.visit(v);
}
pub fn visit_expr_match<'a, V>(v: &mut V, self: &'a expr::Match)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
    for x in &self.arms {
        x.visit(v);
    }
}
pub fn visit_expr_method_call<'a, V>(v: &mut V, self: &'a expr::MethodCall)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
    &self.method.visit(v);
    if let Some(x) = &self.turbofish {
        x.visit(v);
    }
    for el in Puncted::pairs(&self.args) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_expr_paren<'a, V>(v: &mut V, self: &'a expr::Parenth)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
}
pub fn visit_expr_path<'a, V>(v: &mut V, self: &'a expr::Path)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.qself {
        x.visit(v);
    }
    &self.path.visit(v);
}
pub fn visit_expr_range<'a, V>(v: &mut V, self: &'a expr::Range)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.beg {
        &**x.visit(v);
    }
    &self.limits.visit(v);
    if let Some(x) = &self.end {
        &**x.visit(v);
    }
}
pub fn visit_expr_reference<'a, V>(v: &mut V, self: &'a expr::Ref)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
}
pub fn visit_expr_repeat<'a, V>(v: &mut V, self: &'a expr::Repeat)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
    &*self.len.visit(v);
}
pub fn visit_expr_return<'a, V>(v: &mut V, self: &'a expr::Return)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.expr {
        &**x.visit(v);
    }
}
pub fn visit_expr_struct<'a, V>(v: &mut V, self: &'a expr::Struct)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.qself {
        x.visit(v);
    }
    &self.path.visit(v);
    for el in Puncted::pairs(&self.fields) {
        let x = el.value();
        x.visit(v);
    }
    if let Some(x) = &self.rest {
        &**x.visit(v);
    }
}
pub fn visit_expr_try<'a, V>(v: &mut V, self: &'a expr::Try)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
}
pub fn visit_expr_try_block<'a, V>(v: &mut V, self: &'a expr::TryBlock)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.block.visit(v);
}
pub fn visit_expr_tuple<'a, V>(v: &mut V, self: &'a expr::Tuple)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    for el in Puncted::pairs(&self.elems) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_expr_unary<'a, V>(v: &mut V, self: &'a expr::Unary)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.op.visit(v);
    &*self.expr.visit(v);
}
pub fn visit_expr_unsafe<'a, V>(v: &mut V, self: &'a expr::Unsafe)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.block.visit(v);
}
pub fn visit_expr_while<'a, V>(v: &mut V, self: &'a expr::While)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.label {
        x.visit(v);
    }
    &*self.cond.visit(v);
    &self.body.visit(v);
}
pub fn visit_expr_yield<'a, V>(v: &mut V, self: &'a expr::Yield)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.expr {
        &**x.visit(v);
    }
}
pub fn visit_field<'a, V>(v: &mut V, self: &'a data::Field)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.mut_.visit(v);
    if let Some(x) = &self.ident {
        x.visit(v);
    }
    &self.typ.visit(v);
}
pub fn visit_field_mutability<'a, V>(v: &mut V, self: &'a data::Mut)
where
    V: Visit<'a> + ?Sized,
{
    match self {
        data::Mut::None => {},
    }
}
pub fn visit_field_pat<'a, V>(v: &mut V, self: &'a pat::Field)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.memb.visit(v);
    &*self.pat.visit(v);
}
pub fn visit_field_value<'a, V>(v: &mut V, self: &'a expr::FieldValue)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.member.visit(v);
    &self.expr.visit(v);
}
pub fn visit_fields<'a, V>(v: &mut V, self: &'a data::Fields)
where
    V: Visit<'a> + ?Sized,
{
    match self {
        data::Fields::Named(x) => {
            x.visit(v);
        },
        data::Fields::Unnamed(x) => {
            x.visit(v);
        },
        data::Fields::Unit => {},
    }
}
pub fn visit_fields_named<'a, V>(v: &mut V, self: &'a data::Named)
where
    V: Visit<'a> + ?Sized,
{
    for el in Puncted::pairs(&self.fields) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_fields_unnamed<'a, V>(v: &mut V, self: &'a data::Unnamed)
where
    V: Visit<'a> + ?Sized,
{
    for el in Puncted::pairs(&self.fields) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_file<'a, V>(v: &mut V, self: &'a item::File)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    for x in &self.items {
        x.visit(v);
    }
}
pub fn visit_fn_arg<'a, V>(v: &mut V, self: &'a item::FnArg)
where
    V: Visit<'a> + ?Sized,
{
    use item::FnArg::*;
    match self {
        Receiver(x) => {
            x.visit(v);
        },
        Type(x) => {
            x.visit(v);
        },
    }
}
pub fn visit_foreign_item<'a, V>(v: &mut V, self: &'a item::foreign::Item)
where
    V: Visit<'a> + ?Sized,
{
    use item::foreign::Item::*;
    match self {
        Fn(x) => {
            x.visit(v);
        },
        Static(x) => {
            x.visit(v);
        },
        Type(x) => {
            x.visit(v);
        },
        Mac(x) => {
            x.visit(v);
        },
        Verbatim(x) => {
            
        },
    }
}
pub fn visit_foreign_item_fn<'a, V>(v: &mut V, self: &'a item::foreign::Fn)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.sig.visit(v);
}
pub fn visit_foreign_item_macro<'a, V>(v: &mut V, self: &'a item::foreign::Mac)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.mac.visit(v);
}
pub fn visit_foreign_item_static<'a, V>(v: &mut V, self: &'a item::foreign::Static)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.mut_.visit(v);
    &self.ident.visit(v);
    &*self.typ.visit(v);
}
pub fn visit_foreign_item_type<'a, V>(v: &mut V, self: &'a item::foreign::Type)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    &self.gens.visit(v);
}
pub fn visit_generic_argument<'a, V>(v: &mut V, self: &'a path::Arg)
where
    V: Visit<'a> + ?Sized,
{
    use path::Arg::*;
    match self {
        Life(x) => {
            x.visit(v);
        },
        Type(x) => {
            x.visit(v);
        },
        Const(x) => {
            x.visit(v);
        },
        AssocType(x) => {
            x.visit(v);
        },
        AssocConst(x) => {
            x.visit(v);
        },
        Constraint(x) => {
            x.visit(v);
        },
    }
}
pub fn visit_generic_param<'a, V>(v: &mut V, self: &'a gen::Param)
where
    V: Visit<'a> + ?Sized,
{
    use gen::Param::*;
    match self {
        Life(x) => {
            x.visit(v);
        },
        Type(x) => {
            x.visit(v);
        },
        Const(x) => {
            x.visit(v);
        },
    }
}
pub fn visit_generics<'a, V>(v: &mut V, self: &'a gen::Gens)
where
    V: Visit<'a> + ?Sized,
{
    for el in Puncted::pairs(&self.params) {
        let x = el.value();
        x.visit(v);
    }
    if let Some(x) = &self.where_ {
        x.visit(v);
    }
}
pub fn visit_ident<'a, V>(v: &mut V, self: &'a Ident)
where
    V: Visit<'a> + ?Sized,
{
    &self.span().visit(v);
}
pub fn visit_impl_item<'a, V>(v: &mut V, self: &'a item::impl_::Item)
where
    V: Visit<'a> + ?Sized,
{
    use item::impl_::Item::*;
    match self {
        Const(x) => {
            x.visit(v);
        },
        Fn(x) => {
            x.visit(v);
        },
        Type(x) => {
            x.visit(v);
        },
        Mac(x) => {
            x.visit(v);
        },
        Verbatim(x) => {
            
        },
    }
}
pub fn visit_impl_item_const<'a, V>(v: &mut V, self: &'a item::impl_::Const)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    &self.gens.visit(v);
    &self.typ.visit(v);
    &self.expr.visit(v);
}
pub fn visit_impl_item_fn<'a, V>(v: &mut V, self: &'a item::impl_::Fn)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.sig.visit(v);
    &self.block.visit(v);
}
pub fn visit_impl_item_macro<'a, V>(v: &mut V, self: &'a item::impl_::Mac)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.mac.visit(v);
}
pub fn visit_impl_item_type<'a, V>(v: &mut V, self: &'a item::impl_::Type)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    &self.gens.visit(v);
    &self.typ.visit(v);
}
pub fn visit_impl_restriction<'a, V>(v: &mut V, self: &'a item::impl_::Restriction)
where
    V: Visit<'a> + ?Sized,
{
    match *self {}
}
pub fn visit_index<'a, V>(v: &mut V, self: &'a expr::Index)
where
    V: Visit<'a> + ?Sized,
{
    &self.span.visit(v);
}
pub fn visit_item<'a, V>(v: &mut V, self: &'a item::Item)
where
    V: Visit<'a> + ?Sized,
{
    use item::Item::*;
    match self {
        Const(x) => {
            x.visit(v);
        },
        Enum(x) => {
            x.visit(v);
        },
        Extern(x) => {
            x.visit(v);
        },
        Fn(x) => {
            x.visit(v);
        },
        Foreign(x) => {
            x.visit(v);
        },
        Impl(x) => {
            x.visit(v);
        },
        Mac(x) => {
            x.visit(v);
        },
        Mod(x) => {
            x.visit(v);
        },
        Static(x) => {
            x.visit(v);
        },
        Struct(x) => {
            x.visit(v);
        },
        Trait(x) => {
            x.visit(v);
        },
        TraitAlias(x) => {
            x.visit(v);
        },
        Type(x) => {
            x.visit(v);
        },
        Union(x) => {
            x.visit(v);
        },
        Use(x) => {
            x.visit(v);
        },
        Verbatim(x) => {
            
        },
    }
}
pub fn visit_item_const<'a, V>(v: &mut V, self: &'a item::Const)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    &self.gens.visit(v);
    &*self.typ.visit(v);
    &*self.expr.visit(v);
}
pub fn visit_item_enum<'a, V>(v: &mut V, self: &'a item::Enum)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    &self.gens.visit(v);
    for el in Puncted::pairs(&self.variants) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_item_extern_crate<'a, V>(v: &mut V, self: &'a item::Extern)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    if let Some(x) = &self.rename {
        
        &(x).1.visit(v);
    }
}
pub fn visit_item_fn<'a, V>(v: &mut V, self: &'a item::Fn)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.sig.visit(v);
    &*self.block.visit(v);
}
pub fn visit_item_foreign_mod<'a, V>(v: &mut V, self: &'a item::Foreign)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.abi.visit(v);
    for x in &self.items {
        x.visit(v);
    }
}
pub fn visit_item_impl<'a, V>(v: &mut V, self: &'a item::Impl)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.gens.visit(v);
    if let Some(x) = &self.trait_ {
        
        &(x).1.visit(v);
        
    }
    &*self.typ.visit(v);
    for x in &self.items {
        x.visit(v);
    }
}
pub fn visit_item_macro<'a, V>(v: &mut V, self: &'a item::Mac)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.ident {
        x.visit(v);
    }
    &self.mac.visit(v);
}
pub fn visit_item_mod<'a, V>(v: &mut V, self: &'a item::Mod)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    if let Some(x) = &self.items {
        
        for x in &(x).1 {
            x.visit(v);
        }
    }
}
pub fn visit_item_static<'a, V>(v: &mut V, self: &'a item::Static)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.mut_.visit(v);
    &self.ident.visit(v);
    &*self.typ.visit(v);
    &*self.expr.visit(v);
}
pub fn visit_item_struct<'a, V>(v: &mut V, self: &'a item::Struct)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    &self.gens.visit(v);
    &self.fields.visit(v);
}
pub fn visit_item_trait<'a, V>(v: &mut V, self: &'a item::Trait)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    if let Some(x) = &self.restriction {
        x.visit(v);
    }
    &self.ident.visit(v);
    &self.gens.visit(v);
    for el in Puncted::pairs(&self.supers) {
        let x = el.value();
        x.visit(v);
    }
    for x in &self.items {
        x.visit(v);
    }
}
pub fn visit_item_trait_alias<'a, V>(v: &mut V, self: &'a item::TraitAlias)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    &self.gens.visit(v);
    for el in Puncted::pairs(&self.bounds) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_item_type<'a, V>(v: &mut V, self: &'a item::Type)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    &self.gens.visit(v);
    &*self.typ.visit(v);
}
pub fn visit_item_union<'a, V>(v: &mut V, self: &'a item::Union)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    &self.gens.visit(v);
    &self.fields.visit(v);
}
pub fn visit_item_use<'a, V>(v: &mut V, self: &'a item::Use)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.tree.visit(v);
}
pub fn visit_label<'a, V>(v: &mut V, self: &'a expr::Label)
where
    V: Visit<'a> + ?Sized,
{
    &self.name.visit(v);
}
pub fn visit_lifetime<'a, V>(v: &mut V, self: &'a Life)
where
    V: Visit<'a> + ?Sized,
{
    &self.apos.visit(v);
    &self.ident.visit(v);
}
pub fn visit_lifetime_param<'a, V>(v: &mut V, self: &'a gen::param::Life)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.life.visit(v);
    for el in Puncted::pairs(&self.bounds) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_lit<'a, V>(v: &mut V, self: &'a lit::Lit)
where
    V: Visit<'a> + ?Sized,
{
    use lit::Lit::*;
    match self {
        Str(x) => {
            x.visit(v);
        },
        ByteStr(x) => {
            x.visit(v);
        },
        Byte(x) => {
            x.visit(v);
        },
        Char(x) => {
            x.visit(v);
        },
        Int(x) => {
            x.visit(v);
        },
        Float(x) => {
            x.visit(v);
        },
        Bool(x) => {
            x.visit(v);
        },
        Verbatim(x) => {
            
        },
    }
}
pub fn visit_lit_bool<'a, V>(v: &mut V, self: &'a lit::Bool)
where
    V: Visit<'a> + ?Sized,
{
    &self.span.visit(v);
}
pub fn visit_lit_byte<'a, V>(v: &mut V, self: &'a lit::Byte)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_lit_byte_str<'a, V>(v: &mut V, self: &'a lit::ByteStr)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_lit_char<'a, V>(v: &mut V, self: &'a lit::Char)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_lit_float<'a, V>(v: &mut V, self: &'a lit::Float)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_lit_int<'a, V>(v: &mut V, self: &'a lit::Int)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_lit_str<'a, V>(v: &mut V, self: &'a lit::Str)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_local<'a, V>(v: &mut V, self: &'a stmt::Local)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.pat.visit(v);
    if let Some(x) = &self.init {
        x.visit(v);
    }
}
pub fn visit_local_init<'a, V>(v: &mut V, self: &'a stmt::Init)
where
    V: Visit<'a> + ?Sized,
{
    &*self.expr.visit(v);
    if let Some(x) = &self.diverge {
        
        &*(x).1.visit(v);
    }
}
pub fn visit_macro<'a, V>(v: &mut V, self: &'a mac::Mac)
where
    V: Visit<'a> + ?Sized,
{
    &self.path.visit(v);
    &self.delim.visit(v);
}
pub fn visit_macro_delimiter<'a, V>(v: &mut V, self: &'a tok::Delim)
where
    V: Visit<'a> + ?Sized,
{
    use tok::Delim::*;
    match self {
        Parenth(x) => {
            
        },
        Brace(x) => {
            
        },
        Bracket(x) => {
            
        },
    }
}
pub fn visit_member<'a, V>(v: &mut V, self: &'a expr::Member)
where
    V: Visit<'a> + ?Sized,
{
    use expr::Member::*;
    match self {
        Named(x) => {
            x.visit(v);
        },
        Unnamed(x) => {
            x.visit(v);
        },
    }
}
pub fn visit_meta<'a, V>(v: &mut V, self: &'a attr::Meta)
where
    V: Visit<'a> + ?Sized,
{
    use attr::Meta::*;
    match self {
        Path(x) => {
            x.visit(v);
        },
        List(x) => {
            x.visit(v);
        },
        NameValue(x) => {
            x.visit(v);
        },
    }
}
pub fn visit_meta_list<'a, V>(v: &mut V, self: &'a attr::List)
where
    V: Visit<'a> + ?Sized,
{
    &self.path.visit(v);
    &self.delim.visit(v);
}
pub fn visit_meta_name_value<'a, V>(v: &mut V, self: &'a attr::NameValue)
where
    V: Visit<'a> + ?Sized,
{
    &self.name.visit(v);
    &self.val.visit(v);
}
pub fn visit_parenthesized_generic_arguments<'a, V>(v: &mut V, self: &'a path::Parenthed)
where
    V: Visit<'a> + ?Sized,
{
    for el in Puncted::pairs(&self.ins) {
        let x = el.value();
        x.visit(v);
    }
    &self.out.visit(v);
}
pub fn visit_pat<'a, V>(v: &mut V, self: &'a pat::Pat)
where
    V: Visit<'a> + ?Sized,
{
    use pat::Pat::*;
    match self {
        Const(x) => {
            x.visit(v);
        },
        Ident(x) => {
            x.visit(v);
        },
        Lit(x) => {
            x.visit(v);
        },
        Mac(x) => {
            x.visit(v);
        },
        Or(x) => {
            x.visit(v);
        },
        Parenth(x) => {
            x.visit(v);
        },
        Path(x) => {
            x.visit(v);
        },
        Range(x) => {
            x.visit(v);
        },
        Ref(x) => {
            x.visit(v);
        },
        Rest(x) => {
            x.visit(v);
        },
        Slice(x) => {
            x.visit(v);
        },
        Struct(x) => {
            x.visit(v);
        },
        Tuple(x) => {
            x.visit(v);
        },
        TupleStruct(x) => {
            x.visit(v);
        },
        Type(x) => {
            x.visit(v);
        },
        Verbatim(x) => {
            
        },
        Wild(x) => {
            x.visit(v);
        },
    }
}
pub fn visit_pat_ident<'a, V>(v: &mut V, self: &'a pat::Ident)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.ident.visit(v);
    if let Some(x) = &self.sub {
        
        &*(x).1.visit(v);
    }
}
pub fn visit_pat_or<'a, V>(v: &mut V, self: &'a pat::Or)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    for el in Puncted::pairs(&self.cases) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_pat_paren<'a, V>(v: &mut V, self: &'a pat::Parenth)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.pat.visit(v);
}
pub fn visit_pat_reference<'a, V>(v: &mut V, self: &'a pat::Ref)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.pat.visit(v);
}
pub fn visit_pat_rest<'a, V>(v: &mut V, self: &'a pat::Rest)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
}
pub fn visit_pat_slice<'a, V>(v: &mut V, self: &'a pat::Slice)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    for el in Puncted::pairs(&self.pats) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_pat_struct<'a, V>(v: &mut V, self: &'a pat::Struct)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.qself {
        x.visit(v);
    }
    &self.path.visit(v);
    for el in Puncted::pairs(&self.fields) {
        let x = el.value();
        x.visit(v);
    }
    if let Some(x) = &self.rest {
        x.visit(v);
    }
}
pub fn visit_pat_tuple<'a, V>(v: &mut V, self: &'a pat::Tuple)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    for el in Puncted::pairs(&self.pats) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_pat_tuple_struct<'a, V>(v: &mut V, self: &'a pat::TupleStruct)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.qself {
        x.visit(v);
    }
    &self.path.visit(v);
    for el in Puncted::pairs(&self.pats) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_pat_type<'a, V>(v: &mut V, self: &'a pat::Type)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.pat.visit(v);
    &*self.typ.visit(v);
}
pub fn visit_pat_wild<'a, V>(v: &mut V, self: &'a pat::Wild)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
}
pub fn visit_path<'a, V>(v: &mut V, self: &'a Path)
where
    V: Visit<'a> + ?Sized,
{
    for el in Puncted::pairs(&self.segs) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_path_arguments<'a, V>(v: &mut V, self: &'a path::Args)
where
    V: Visit<'a> + ?Sized,
{
    use path::Args::*;
    match self {
        None => {},
        Angled(x) => {
            x.visit(v);
        },
        Parenthed(x) => {
            x.visit(v);
        },
    }
}
pub fn visit_path_segment<'a, V>(v: &mut V, self: &'a path::Segment)
where
    V: Visit<'a> + ?Sized,
{
    &self.ident.visit(v);
    &self.args.visit(v);
}
pub fn visit_predicate_lifetime<'a, V>(v: &mut V, self: &'a gen::where_::Life)
where
    V: Visit<'a> + ?Sized,
{
    &self.life.visit(v);
    for el in Puncted::pairs(&self.bounds) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_predicate_type<'a, V>(v: &mut V, self: &'a gen::where_::Type)
where
    V: Visit<'a> + ?Sized,
{
    if let Some(x) = &self.lifes {
        x.visit(v);
    }
    &self.bounded.visit(v);
    for el in Puncted::pairs(&self.bounds) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_qself<'a, V>(v: &mut V, self: &'a path::QSelf)
where
    V: Visit<'a> + ?Sized,
{
    &*self.ty.visit(v);
}
pub fn visit_range_limits<'a, V>(v: &mut V, self: &'a expr::Limits)
where
    V: Visit<'a> + ?Sized,
{
    use expr::Limits::*;
    match self {
        HalfOpen(x) => {
            
        },
        Closed(x) => {
            
        },
    }
}
pub fn visit_receiver<'a, V>(v: &mut V, self: &'a item::Receiver)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.ref_ {
        
        if let Some(x) = &(x).1 {
            x.visit(v);
        }
    }
    &*self.typ.visit(v);
}
pub fn visit_return_type<'a, V>(v: &mut V, self: &'a typ::Ret)
where
    V: Visit<'a> + ?Sized,
{
    use typ::Ret::*;
    match self {
        Default => {},
        Type(_binding_0, _binding_1) => {
            
            &**_binding_1.visit(v);
        },
    }
}
pub fn visit_signature<'a, V>(v: &mut V, self: &'a item::Sig)
where
    V: Visit<'a> + ?Sized,
{
    if let Some(x) = &self.abi {
        x.visit(v);
    }
    &self.ident.visit(v);
    &self.gens.visit(v);
    for el in Puncted::pairs(&self.args) {
        let x = el.value();
        x.visit(v);
    }
    if let Some(x) = &self.vari {
        x.visit(v);
    }
    &self.ret.visit(v);
}
pub fn visit_span<'a, V>(v: &mut V, self: &pm2::Span)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_static_mutability<'a, V>(v: &mut V, self: &'a item::StaticMut)
where
    V: Visit<'a> + ?Sized,
{
    use item::StaticMut::*;
    match self {
        Mut(x) => {
            
        },
        None => {},
    }
}
pub fn visit_stmt<'a, V>(v: &mut V, self: &'a stmt::Stmt)
where
    V: Visit<'a> + ?Sized,
{
    use stmt::Stmt::*;
    match self {
        Local(x) => {
            x.visit(v);
        },
        Item(x) => {
            x.visit(v);
        },
        Expr(x, y) => {
            x.visit(v);
            
        },
        Mac(x) => {
            x.visit(v);
        },
    }
}
pub fn visit_stmt_macro<'a, V>(v: &mut V, self: &'a stmt::Mac)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.mac.visit(v);
}
pub fn visit_trait_bound<'a, V>(v: &mut V, self: &'a gen::bound::Trait)
where
    V: Visit<'a> + ?Sized,
{
    &self.modif.visit(v);
    if let Some(x) = &self.lifes {
        x.visit(v);
    }
    &self.path.visit(v);
}
pub fn visit_trait_bound_modifier<'a, V>(v: &mut V, self: &'a gen::bound::Modifier)
where
    V: Visit<'a> + ?Sized,
{
    use gen::bound::Modifier::*;
    match self {
        None => {},
        Maybe(x) => {
            
        },
    }
}
pub fn visit_trait_item<'a, V>(v: &mut V, self: &'a item::trait_::Item)
where
    V: Visit<'a> + ?Sized,
{
    use item::trait_::Item::*;
    match self {
        Const(x) => {
            x.visit(v);
        },
        Fn(x) => {
            x.visit(v);
        },
        Type(x) => {
            x.visit(v);
        },
        Mac(x) => {
            x.visit(v);
        },
        Verbatim(x) => {
            
        },
    }
}
pub fn visit_trait_item_const<'a, V>(v: &mut V, self: &'a item::trait_::Const)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.ident.visit(v);
    &self.gens.visit(v);
    &self.typ.visit(v);
    if let Some(x) = &self.default {
        
        &(x).1.visit(v);
    }
}
pub fn visit_trait_item_fn<'a, V>(v: &mut V, self: &'a item::trait_::Fn)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.sig.visit(v);
    if let Some(x) = &self.default {
        x.visit(v);
    }
}
pub fn visit_trait_item_macro<'a, V>(v: &mut V, self: &'a item::trait_::Mac)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.mac.visit(v);
}
pub fn visit_trait_item_type<'a, V>(v: &mut V, self: &'a item::trait_::Type)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.ident.visit(v);
    &self.gens.visit(v);
    for el in Puncted::pairs(&self.bounds) {
        let x = el.value();
        x.visit(v);
    }
    if let Some(x) = &self.default {
        
        &(x).1.visit(v);
    }
}
pub fn visit_type<'a, V>(v: &mut V, self: &'a typ::Type)
where
    V: Visit<'a> + ?Sized,
{
    use typ::Type::*;
    match self {
        Array(x) => {
            x.visit(v);
        },
        Fn(x) => {
            x.visit(v);
        },
        Group(x) => {
            x.visit(v);
        },
        Impl(x) => {
            x.visit(v);
        },
        Infer(x) => {
            x.visit(v);
        },
        Mac(x) => {
            x.visit(v);
        },
        Never(x) => {
            x.visit(v);
        },
        Parenth(x) => {
            x.visit(v);
        },
        Path(x) => {
            x.visit(v);
        },
        Ptr(x) => {
            x.visit(v);
        },
        Ref(x) => {
            x.visit(v);
        },
        Slice(x) => {
            x.visit(v);
        },
        Trait(x) => {
            x.visit(v);
        },
        Tuple(x) => {
            x.visit(v);
        },
        Verbatim(x) => {
            
        },
    }
}
pub fn visit_type_array<'a, V>(v: &mut V, self: &'a typ::Array)
where
    V: Visit<'a> + ?Sized,
{
    &*self.elem.visit(v);
    &self.len.visit(v);
}
pub fn visit_type_bare_fn<'a, V>(v: &mut V, self: &'a typ::Fn)
where
    V: Visit<'a> + ?Sized,
{
    if let Some(x) = &self.lifes {
        x.visit(v);
    }
    if let Some(x) = &self.abi {
        x.visit(v);
    }
    for el in Puncted::pairs(&self.args) {
        let x = el.value();
        x.visit(v);
    }
    if let Some(x) = &self.vari {
        x.visit(v);
    }
    &self.ret.visit(v);
}
pub fn visit_type_group<'a, V>(v: &mut V, self: &'a typ::Group)
where
    V: Visit<'a> + ?Sized,
{
    &*self.elem.visit(v);
}
pub fn visit_type_impl_trait<'a, V>(v: &mut V, self: &'a typ::Impl)
where
    V: Visit<'a> + ?Sized,
{
    for el in Puncted::pairs(&self.bounds) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_type_infer<'a, V>(v: &mut V, self: &'a typ::Infer)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_type_macro<'a, V>(v: &mut V, self: &'a typ::Mac)
where
    V: Visit<'a> + ?Sized,
{
    &self.mac.visit(v);
}
pub fn visit_type_never<'a, V>(v: &mut V, self: &'a typ::Never)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_type_param<'a, V>(v: &mut V, self: &'a gen::param::Type)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.ident.visit(v);
    for el in Puncted::pairs(&self.bounds) {
        let x = el.value();
        x.visit(v);
    }
    if let Some(x) = &self.default {
        x.visit(v);
    }
}
pub fn visit_type_param_bound<'a, V>(v: &mut V, self: &'a gen::bound::Type)
where
    V: Visit<'a> + ?Sized,
{
    use gen::bound::Type::*;
    match self {
        Trait(x) => {
            x.visit(v);
        },
        Life(x) => {
            x.visit(v);
        },
        Verbatim(x) => {
            
        },
    }
}
pub fn visit_type_paren<'a, V>(v: &mut V, self: &'a typ::Parenth)
where
    V: Visit<'a> + ?Sized,
{
    &*self.elem.visit(v);
}
pub fn visit_type_path<'a, V>(v: &mut V, self: &'a typ::Path)
where
    V: Visit<'a> + ?Sized,
{
    if let Some(x) = &self.qself {
        x.visit(v);
    }
    &self.path.visit(v);
}
pub fn visit_type_ptr<'a, V>(v: &mut V, self: &'a typ::Ptr)
where
    V: Visit<'a> + ?Sized,
{
    &*self.elem.visit(v);
}
pub fn visit_type_reference<'a, V>(v: &mut V, self: &'a typ::Ref)
where
    V: Visit<'a> + ?Sized,
{
    if let Some(x) = &self.life {
        x.visit(v);
    }
    &*self.elem.visit(v);
}
pub fn visit_type_slice<'a, V>(v: &mut V, self: &'a typ::Slice)
where
    V: Visit<'a> + ?Sized,
{
    &*self.elem.visit(v);
}
pub fn visit_type_trait_object<'a, V>(v: &mut V, self: &'a typ::Trait)
where
    V: Visit<'a> + ?Sized,
{
    for el in Puncted::pairs(&self.bounds) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_type_tuple<'a, V>(v: &mut V, self: &'a typ::Tuple)
where
    V: Visit<'a> + ?Sized,
{
    for el in Puncted::pairs(&self.elems) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_un_op<'a, V>(v: &mut V, self: &'a expr::UnOp)
where
    V: Visit<'a> + ?Sized,
{
    use expr::UnOp::*;
    match self {
        Deref(x) => {
            
        },
        Not(x) => {
            
        },
        Neg(x) => {
            
        },
    }
}
pub fn visit_use_glob<'a, V>(v: &mut V, self: &'a item::use_::Glob)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_use_group<'a, V>(v: &mut V, self: &'a item::use_::Group)
where
    V: Visit<'a> + ?Sized,
{
    for el in Puncted::pairs(&self.trees) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_use_name<'a, V>(v: &mut V, self: &'a item::use_::Name)
where
    V: Visit<'a> + ?Sized,
{
    &self.ident.visit(v);
}
pub fn visit_use_path<'a, V>(v: &mut V, self: &'a item::use_::Path)
where
    V: Visit<'a> + ?Sized,
{
    &self.ident.visit(v);
    &*self.tree.visit(v);
}
pub fn visit_use_rename<'a, V>(v: &mut V, self: &'a item::use_::Rename)
where
    V: Visit<'a> + ?Sized,
{
    &self.ident.visit(v);
    &self.rename.visit(v);
}
pub fn visit_use_tree<'a, V>(v: &mut V, self: &'a item::use_::Tree)
where
    V: Visit<'a> + ?Sized,
{
    use item::use_::Tree::*;
    match self {
        Path(x) => {
            x.visit(v);
        },
        Name(x) => {
            x.visit(v);
        },
        Rename(x) => {
            x.visit(v);
        },
        Glob(x) => {
            x.visit(v);
        },
        Group(x) => {
            x.visit(v);
        },
    }
}
pub fn visit_variadic<'a, V>(v: &mut V, self: &'a item::Variadic)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.pat {
        &*(x).0.visit(v);
        
    }
}
pub fn visit_variant<'a, V>(v: &mut V, self: &'a data::Variant)
where
    V: Visit<'a> + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.ident.visit(v);
    &self.fields.visit(v);
    if let Some(x) = &self.discrim {
        
        &(x).1.visit(v);
    }
}
pub fn visit_vis_restricted<'a, V>(v: &mut V, self: &'a data::Restricted)
where
    V: Visit<'a> + ?Sized,
{
    &*self.path.visit(v);
}
pub fn visit_visibility<'a, V>(v: &mut V, self: &'a data::Visibility)
where
    V: Visit<'a> + ?Sized,
{
    use data::Visibility::*;
    match self {
        Public(x) => {
            
        },
        Restricted(x) => {
            x.visit(v);
        },
        Inherited => {},
    }
}
pub fn visit_where_clause<'a, V>(v: &mut V, self: &'a gen::Where)
where
    V: Visit<'a> + ?Sized,
{
    for el in Puncted::pairs(&self.preds) {
        let x = el.value();
        x.visit(v);
    }
}
pub fn visit_where_predicate<'a, V>(v: &mut V, self: &'a gen::where_::Pred)
where
    V: Visit<'a> + ?Sized,
{
    use gen::where_::Pred::*;
    match self {
        Life(x) => {
            x.visit(v);
        },
        Type(x) => {
            x.visit(v);
        },
    }
}
