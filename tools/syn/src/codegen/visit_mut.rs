#![allow(unused_variables)]

use crate::*;

macro_rules! full {
    ($e:expr) => {
        $e
    };
}

trait Visitor {}

trait VisitMut {
    fn visit_mut<V>(&mut self, v: &mut V);
}

pub fn visit_expr_binary_mut<V>(v: &mut V, node: &mut expr::Binary)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.left.visit_mut(v);
    &mut self.op.visit_mut(v);
    &mut *node.right.visit_mut(v);
}
pub fn visit_expr_block_mut<V>(v: &mut V, node: &mut expr::Block)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.label {
        x.visit_mut(v);
    }
    &mut self.block.visit_mut(v);
}
pub fn visit_expr_break_mut<V>(v: &mut V, node: &mut expr::Break)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.life {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.val {
        &mut **it.visit_mut(v);
    }
}
pub fn visit_expr_call_mut<V>(v: &mut V, node: &mut expr::Call)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.func.visit_mut(v);
    for mut el in Puncted::pairs_mut(&mut self.args) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_expr_cast_mut<V>(v: &mut V, node: &mut expr::Cast)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.expr.visit_mut(v);
    &mut *node.typ.visit_mut(v);
}
pub fn visit_expr_closure_mut<V>(v: &mut V, node: &mut expr::Closurere)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.lifes {
        x.visit_mut(v);
    }
    for mut el in Puncted::pairs_mut(&mut self.inputs) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
    &mut self.ret.visit_mut(v);
    &mut *node.body.visit_mut(v);
}
pub fn visit_expr_const_mut<V>(v: &mut V, node: &mut expr::Constst)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.block.visit_mut(v);
}
pub fn visit_expr_continue_mut<V>(v: &mut V, node: &mut expr::Continue)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.life {
        x.visit_mut(v);
    }
}
pub fn visit_expr_field_mut<V>(v: &mut V, node: &mut expr::Field)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.expr.visit_mut(v);
    &mut self.memb.visit_mut(v);
}
pub fn visit_expr_for_loop_mut<V>(v: &mut V, node: &mut expr::ForLoop)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.label {
        x.visit_mut(v);
    }
    &mut *node.pat.visit_mut(v);
    &mut *node.expr.visit_mut(v);
    &mut self.body.visit_mut(v);
}
pub fn visit_expr_group_mut<V>(v: &mut V, node: &mut expr::Group)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.expr.visit_mut(v);
}
pub fn visit_expr_if_mut<V>(v: &mut V, node: &mut expr::If)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.cond.visit_mut(v);
    &mut self.then_branch.visit_mut(v);
    if let Some(x) = &mut self.else_branch {
        &mut *(x).1.visit_mut(v);
    }
}
pub fn visit_expr_index_mut<V>(v: &mut V, node: &mut expr::Index)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.expr.visit_mut(v);
    &mut *node.index.visit_mut(v);
}
pub fn visit_expr_infer_mut<V>(v: &mut V, node: &mut expr::Infer)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
}
pub fn visit_expr_let_mut<V>(v: &mut V, node: &mut expr::Let)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.pat.visit_mut(v);
    &mut *node.expr.visit_mut(v);
}
pub fn visit_expr_lit_mut<V>(v: &mut V, node: &mut expr::Lit)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.lit.visit_mut(v);
}
pub fn visit_expr_loop_mut<V>(v: &mut V, node: &mut expr::Loop)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.label {
        x.visit_mut(v);
    }
    &mut self.body.visit_mut(v);
}
pub fn visit_expr_macro_mut<V>(v: &mut V, node: &mut expr::Mac)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.mac.visit_mut(v);
}
pub fn visit_expr_match_mut<V>(v: &mut V, node: &mut expr::Match)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.expr.visit_mut(v);
    for x in &mut self.arms {
        x.visit_mut(v);
    }
}
pub fn visit_expr_method_call_mut<V>(v: &mut V, node: &mut expr::MethodCall)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.expr.visit_mut(v);
    &mut self.method.visit_mut(v);
    if let Some(x) = &mut self.turbofish {
        x.visit_mut(v);
    }
    for mut el in Puncted::pairs_mut(&mut self.args) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_expr_paren_mut<V>(v: &mut V, node: &mut expr::Parenth)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.expr.visit_mut(v);
}
pub fn visit_expr_path_mut<V>(v: &mut V, node: &mut expr::Path)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.qself {
        x.visit_mut(v);
    }
    &mut self.path.visit_mut(v);
}
pub fn visit_expr_range_mut<V>(v: &mut V, node: &mut expr::Range)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.beg {
        &mut **it.visit_mut(v);
    }
    &mut self.limits.visit_mut(v);
    if let Some(x) = &mut self.end {
        &mut **it.visit_mut(v);
    }
}
pub fn visit_expr_reference_mut<V>(v: &mut V, node: &mut expr::Ref)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.expr.visit_mut(v);
}
pub fn visit_expr_repeat_mut<V>(v: &mut V, node: &mut expr::Repeat)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.expr.visit_mut(v);
    &mut *node.len.visit_mut(v);
}
pub fn visit_expr_return_mut<V>(v: &mut V, node: &mut expr::Return)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.expr {
        &mut **it.visit_mut(v);
    }
}
pub fn visit_expr_struct_mut<V>(v: &mut V, node: &mut expr::Struct)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.qself {
        x.visit_mut(v);
    }
    &mut self.path.visit_mut(v);
    for mut el in Puncted::pairs_mut(&mut self.fields) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.rest {
        &mut **it.visit_mut(v);
    }
}
pub fn visit_expr_try_mut<V>(v: &mut V, node: &mut expr::Tryry)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.expr.visit_mut(v);
}
pub fn visit_expr_try_block_mut<V>(v: &mut V, node: &mut expr::TryBlock)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.block.visit_mut(v);
}
pub fn visit_expr_tuple_mut<V>(v: &mut V, node: &mut expr::Tuple)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    for mut el in Puncted::pairs_mut(&mut self.elems) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_expr_unary_mut<V>(v: &mut V, node: &mut expr::Unary)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.op.visit_mut(v);
    &mut *node.expr.visit_mut(v);
}
pub fn visit_expr_unsafe_mut<V>(v: &mut V, node: &mut expr::Unsafe)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.block.visit_mut(v);
}
pub fn visit_expr_while_mut<V>(v: &mut V, node: &mut expr::While)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.label {
        x.visit_mut(v);
    }
    &mut *node.cond.visit_mut(v);
    &mut self.body.visit_mut(v);
}
pub fn visit_expr_yield_mut<V>(v: &mut V, node: &mut expr::Yield)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.expr {
        &mut **it.visit_mut(v);
    }
}
pub fn visit_field_mut<V>(v: &mut V, node: &mut data::Field)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.mut_.visit_mut(v);
    if let Some(x) = &mut self.ident {
        x.visit_mut(v);
    }
    &mut self.typ.visit_mut(v);
}
pub fn visit_field_mutability_mut<V>(v: &mut V, node: &mut data::Mut)
where
    V: VisitMut + ?Sized,
{
    match node {
        data::Mut::None => {},
    }
}
pub fn visit_field_pat_mut<V>(v: &mut V, node: &mut pat::Field)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.memb.visit_mut(v);
    &mut *node.pat.visit_mut(v);
}
pub fn visit_field_value_mut<V>(v: &mut V, node: &mut FieldValue)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.member.visit_mut(v);
    &mut self.expr.visit_mut(v);
}
pub fn visit_fields_mut<V>(v: &mut V, node: &mut data::Fields)
where
    V: VisitMut + ?Sized,
{
    match node {
        data::Fields::Named(x) => {
            x.visit_mut(v);
        },
        data::Fields::Unnamed(x) => {
            x.visit_mut(v);
        },
        data::Fields::Unit => {},
    }
}
pub fn visit_fields_named_mut<V>(v: &mut V, node: &mut data::Named)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.fields) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_fields_unnamed_mut<V>(v: &mut V, node: &mut data::Unnamed)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.fields) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_file_mut<V>(v: &mut V, node: &mut item::File)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    for x in &mut self.items {
        x.visit_mut(v);
    }
}
pub fn visit_fn_arg_mut<V>(v: &mut V, node: &mut item::FnArg)
where
    V: VisitMut + ?Sized,
{
    match node {
        item::FnArg::Receiver(x) => {
            x.visit_mut(v);
        },
        item::FnArg::Type(x) => {
            x.visit_mut(v);
        },
    }
}
pub fn visit_foreign_item_mut<V>(v: &mut V, node: &mut item::foreign::Item)
where
    V: VisitMut + ?Sized,
{
    match node {
        item::foreign::Item::Fn(x) => {
            x.visit_mut(v);
        },
        item::foreign::Item::Static(x) => {
            x.visit_mut(v);
        },
        item::foreign::Item::Type(x) => {
            x.visit_mut(v);
        },
        item::foreign::Item::Macro(x) => {
            x.visit_mut(v);
        },
        item::foreign::Item::Verbatim(x) => {},
    }
}
pub fn visit_foreign_item_fn_mut<V>(v: &mut V, node: &mut item::foreign::Fn)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.sig.visit_mut(v);
}
pub fn visit_foreign_item_macro_mut<V>(v: &mut V, node: &mut item::foreign::Mac)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.mac.visit_mut(v);
}
pub fn visit_foreign_item_static_mut<V>(v: &mut V, node: &mut item::foreign::Static)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.mut_.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut *node.typ.visit_mut(v);
}
pub fn visit_foreign_item_type_mut<V>(v: &mut V, node: &mut item::foreign::Type)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
}
pub fn visit_generic_argument_mut<V>(v: &mut V, node: &mut Arg)
where
    V: VisitMut + ?Sized,
{
    match node {
        Arg::Life(x) => {
            x.visit_mut(v);
        },
        Arg::Type(x) => {
            x.visit_mut(v);
        },
        Arg::Const(x) => {
            x.visit_mut(v);
        },
        Arg::AssocType(x) => {
            x.visit_mut(v);
        },
        Arg::AssocConst(x) => {
            x.visit_mut(v);
        },
        Arg::Constraint(x) => {
            x.visit_mut(v);
        },
    }
}
pub fn visit_generic_param_mut<V>(v: &mut V, node: &mut gen::Param)
where
    V: VisitMut + ?Sized,
{
    match node {
        gen::Param::Life(x) => {
            x.visit_mut(v);
        },
        gen::Param::Type(x) => {
            x.visit_mut(v);
        },
        gen::Param::Const(x) => {
            x.visit_mut(v);
        },
    }
}
pub fn visit_generics_mut<V>(v: &mut V, node: &mut gen::Gens)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.params) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.where_ {
        x.visit_mut(v);
    }
}
pub fn visit_ident_mut<V>(v: &mut V, node: &mut Ident)
where
    V: VisitMut + ?Sized,
{
    let mut span = node.span();
    &mut span.visit_mut(v);
    node.set_span(span);
}
pub fn visit_impl_item_mut<V>(v: &mut V, node: &mut item::impl_::Item)
where
    V: VisitMut + ?Sized,
{
    match node {
        item::impl_::Item::Const(x) => {
            x.visit_mut(v);
        },
        item::impl_::Item::Fn(x) => {
            x.visit_mut(v);
        },
        item::impl_::Item::Type(x) => {
            x.visit_mut(v);
        },
        item::impl_::Item::Macro(x) => {
            x.visit_mut(v);
        },
        item::impl_::Item::Verbatim(x) => {},
    }
}
pub fn visit_impl_item_const_mut<V>(v: &mut V, node: &mut item::impl_::Const)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    &mut self.typ.visit_mut(v);
    &mut self.expr.visit_mut(v);
}
pub fn visit_impl_item_fn_mut<V>(v: &mut V, node: &mut item::impl_::Fn)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.sig.visit_mut(v);
    &mut self.block.visit_mut(v);
}
pub fn visit_impl_item_macro_mut<V>(v: &mut V, node: &mut item::impl_::Mac)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.mac.visit_mut(v);
}
pub fn visit_impl_item_type_mut<V>(v: &mut V, node: &mut item::impl_::Type)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    &mut self.typ.visit_mut(v);
}
pub fn visit_impl_restriction_mut<V>(v: &mut V, node: &mut item::impl_::Restriction)
where
    V: VisitMut + ?Sized,
{
    match *node {}
}
pub fn visit_index_mut<V>(v: &mut V, node: &mut Index)
where
    V: VisitMut + ?Sized,
{
    &mut self.span.visit_mut(v);
}
pub fn visit_item_mut<V>(v: &mut V, node: &mut Item)
where
    V: VisitMut + ?Sized,
{
    match node {
        Item::Const(x) => {
            x.visit_mut(v);
        },
        Item::Enum(x) => {
            x.visit_mut(v);
        },
        Item::Extern(x) => {
            x.visit_mut(v);
        },
        Item::Fn(x) => {
            x.visit_mut(v);
        },
        Item::Foreign(x) => {
            x.visit_mut(v);
        },
        Item::Impl(x) => {
            x.visit_mut(v);
        },
        Item::Macro(x) => {
            x.visit_mut(v);
        },
        Item::Mod(x) => {
            x.visit_mut(v);
        },
        Item::Static(x) => {
            x.visit_mut(v);
        },
        Item::Struct(x) => {
            x.visit_mut(v);
        },
        Item::Trait(x) => {
            x.visit_mut(v);
        },
        Item::TraitAlias(x) => {
            x.visit_mut(v);
        },
        Item::Type(x) => {
            x.visit_mut(v);
        },
        Item::Union(x) => {
            x.visit_mut(v);
        },
        Item::Use(x) => {
            x.visit_mut(v);
        },
        Item::Stream(x) => {},
    }
}
pub fn visit_item_const_mut<V>(v: &mut V, node: &mut item::Const)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    &mut *node.typ.visit_mut(v);
    &mut *node.expr.visit_mut(v);
}
pub fn visit_item_enum_mut<V>(v: &mut V, node: &mut item::Enum)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    for mut el in Puncted::pairs_mut(&mut self.variants) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_item_extern_crate_mut<V>(v: &mut V, node: &mut item::Extern)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    if let Some(x) = &mut self.rename {
        &mut (x).1.visit_mut(v);
    }
}
pub fn visit_item_fn_mut<V>(v: &mut V, node: &mut item::Fn)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.sig.visit_mut(v);
    &mut *node.block.visit_mut(v);
}
pub fn visit_item_foreign_mod_mut<V>(v: &mut V, node: &mut item::Foreign)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.abi.visit_mut(v);
    for x in &mut self.items {
        x.visit_mut(v);
    }
}
pub fn visit_item_impl_mut<V>(v: &mut V, node: &mut item::Impl)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.gens.visit_mut(v);
    if let Some(x) = &mut self.trait_ {
        &mut (x).1.visit_mut(v);
    }
    &mut *node.typ.visit_mut(v);
    for x in &mut self.items {
        x.visit_mut(v);
    }
}
pub fn visit_item_macro_mut<V>(v: &mut V, node: &mut item::Mac)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.ident {
        x.visit_mut(v);
    }
    &mut self.mac.visit_mut(v);
}
pub fn visit_item_mod_mut<V>(v: &mut V, node: &mut item::Mod)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    if let Some(x) = &mut self.items {
        for x in &mut (x).1 {
            x.visit_mut(v);
        }
    }
}
pub fn visit_item_static_mut<V>(v: &mut V, node: &mut item::Static)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.mut_.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut *node.typ.visit_mut(v);
    &mut *node.expr.visit_mut(v);
}
pub fn visit_item_struct_mut<V>(v: &mut V, node: &mut item::Struct)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    &mut self.fields.visit_mut(v);
}
pub fn visit_item_trait_mut<V>(v: &mut V, node: &mut item::Trait)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    if let Some(x) = &mut self.restriction {
        x.visit_mut(v);
    }
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    for mut el in Puncted::pairs_mut(&mut self.supers) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
    for x in &mut self.items {
        x.visit_mut(v);
    }
}
pub fn visit_item_trait_alias_mut<V>(v: &mut V, node: &mut item::TraitAlias)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_item_type_mut<V>(v: &mut V, node: &mut item::Type)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    &mut *node.typ.visit_mut(v);
}
pub fn visit_item_union_mut<V>(v: &mut V, node: &mut item::Union)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    &mut self.fields.visit_mut(v);
}
pub fn visit_item_use_mut<V>(v: &mut V, node: &mut item::Use)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.tree.visit_mut(v);
}
pub fn visit_label_mut<V>(v: &mut V, node: &mut Label)
where
    V: VisitMut + ?Sized,
{
    &mut self.name.visit_mut(v);
}
pub fn visit_lifetime_mut<V>(v: &mut V, node: &mut Life)
where
    V: VisitMut + ?Sized,
{
    &mut self.apos.visit_mut(v);
    &mut self.ident.visit_mut(v);
}
pub fn visit_lifetime_param_mut<V>(v: &mut V, node: &mut gen::param::Life)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.life.visit_mut(v);
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_lit_mut<V>(v: &mut V, node: &mut Lit)
where
    V: VisitMut + ?Sized,
{
    match node {
        Lit::Str(x) => {
            x.visit_mut(v);
        },
        Lit::ByteStr(x) => {
            x.visit_mut(v);
        },
        Lit::Byte(x) => {
            x.visit_mut(v);
        },
        Lit::Char(x) => {
            x.visit_mut(v);
        },
        Lit::Int(x) => {
            x.visit_mut(v);
        },
        Lit::Float(x) => {
            x.visit_mut(v);
        },
        Lit::Bool(x) => {
            x.visit_mut(v);
        },
        Lit::Stream(x) => {},
    }
}
pub fn visit_lit_bool_mut<V>(v: &mut V, node: &mut lit::Bool)
where
    V: VisitMut + ?Sized,
{
    &mut self.span.visit_mut(v);
}
pub fn visit_lit_byte_mut<V>(v: &mut V, node: &mut lit::Byte)
where
    V: VisitMut + ?Sized,
{
}
pub fn visit_lit_byte_str_mut<V>(v: &mut V, node: &mut lit::ByteStr)
where
    V: VisitMut + ?Sized,
{
}
pub fn visit_lit_char_mut<V>(v: &mut V, node: &mut lit::Char)
where
    V: VisitMut + ?Sized,
{
}
pub fn visit_lit_float_mut<V>(v: &mut V, node: &mut lit::Float)
where
    V: VisitMut + ?Sized,
{
}
pub fn visit_lit_int_mut<V>(v: &mut V, node: &mut lit::Int)
where
    V: VisitMut + ?Sized,
{
}
pub fn visit_lit_str_mut<V>(v: &mut V, node: &mut lit::Str)
where
    V: VisitMut + ?Sized,
{
}
pub fn visit_local_mut<V>(v: &mut V, node: &mut stmt::Local)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.pat.visit_mut(v);
    if let Some(x) = &mut self.init {
        x.visit_mut(v);
    }
}
pub fn visit_local_init_mut<V>(v: &mut V, node: &mut stmt::Init)
where
    V: VisitMut + ?Sized,
{
    &mut *node.expr.visit_mut(v);
    if let Some(x) = &mut self.diverge {
        &mut *(x).1.visit_mut(v);
    }
}
pub fn visit_macro_mut<V>(v: &mut V, node: &mut Macro)
where
    V: VisitMut + ?Sized,
{
    &mut self.path.visit_mut(v);
    &mut self.delim.visit_mut(v);
}
pub fn visit_macro_delimiter_mut<V>(v: &mut V, node: &mut tok::Delim)
where
    V: VisitMut + ?Sized,
{
    match node {
        tok::Delim::Parenth(x) => {},
        tok::Delim::Brace(x) => {},
        tok::Delim::Bracket(x) => {},
    }
}
pub fn visit_member_mut<V>(v: &mut V, node: &mut Member)
where
    V: VisitMut + ?Sized,
{
    match node {
        Member::Named(x) => {
            x.visit_mut(v);
        },
        Member::Unnamed(x) => {
            x.visit_mut(v);
        },
    }
}
pub fn visit_meta_list_mut<V>(v: &mut V, node: &mut attr::List)
where
    V: VisitMut + ?Sized,
{
    &mut self.path.visit_mut(v);
    &mut self.delim.visit_mut(v);
}
pub fn visit_meta_name_value_mut<V>(v: &mut V, node: &mut attr::NameValue)
where
    V: VisitMut + ?Sized,
{
    &mut self.name.visit_mut(v);
    &mut self.val.visit_mut(v);
}
pub fn visit_parenthesized_generic_arguments_mut<V>(v: &mut V, node: &mut path::Parenthed)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.ins) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
    &mut self.out.visit_mut(v);
}
pub fn visit_pat_mut<V>(v: &mut V, node: &mut pat::Pat)
where
    V: VisitMut + ?Sized,
{
    match node {
        pat::Pat::Const(x) => {
            x.visit_mut(v);
        },
        pat::Pat::Ident(x) => {
            x.visit_mut(v);
        },
        pat::Pat::Lit(x) => {
            x.visit_mut(v);
        },
        pat::Pat::Mac(x) => {
            x.visit_mut(v);
        },
        pat::Pat::Or(x) => {
            x.visit_mut(v);
        },
        pat::Pat::Parenth(x) => {
            x.visit_mut(v);
        },
        pat::Pat::Path(x) => {
            x.visit_mut(v);
        },
        pat::Pat::Range(x) => {
            x.visit_mut(v);
        },
        pat::Pat::Ref(x) => {
            x.visit_mut(v);
        },
        pat::Pat::Rest(x) => {
            x.visit_mut(v);
        },
        pat::Pat::Slice(x) => {
            x.visit_mut(v);
        },
        pat::Pat::Struct(x) => {
            x.visit_mut(v);
        },
        pat::Pat::Tuple(x) => {
            x.visit_mut(v);
        },
        pat::Pat::TupleStruct(x) => {
            x.visit_mut(v);
        },
        pat::Pat::Type(x) => {
            x.visit_mut(v);
        },
        pat::Pat::Verbatim(x) => {},
        pat::Pat::Wild(x) => {
            x.visit_mut(v);
        },
    }
}
pub fn visit_pat_ident_mut<V>(v: &mut V, node: &mut pat::Ident)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.ident.visit_mut(v);
    if let Some(x) = &mut self.sub {
        &mut *(x).1.visit_mut(v);
    }
}
pub fn visit_pat_or_mut<V>(v: &mut V, node: &mut pat::Or)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    for mut el in Puncted::pairs_mut(&mut self.cases) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_pat_paren_mut<V>(v: &mut V, node: &mut pat::Parenth)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.pat.visit_mut(v);
}
pub fn visit_pat_reference_mut<V>(v: &mut V, node: &mut pat::Ref)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.pat.visit_mut(v);
}
pub fn visit_pat_rest_mut<V>(v: &mut V, node: &mut pat::Rest)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
}
pub fn visit_pat_slice_mut<V>(v: &mut V, node: &mut pat::Slice)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    for mut el in Puncted::pairs_mut(&mut self.pats) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_pat_struct_mut<V>(v: &mut V, node: &mut pat::Struct)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.qself {
        x.visit_mut(v);
    }
    &mut self.path.visit_mut(v);
    for mut el in Puncted::pairs_mut(&mut self.fields) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.rest {
        x.visit_mut(v);
    }
}
pub fn visit_pat_tuple_mut<V>(v: &mut V, node: &mut pat::Tuple)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    for mut el in Puncted::pairs_mut(&mut self.pats) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_pat_tuple_struct_mut<V>(v: &mut V, node: &mut pat::TupleStruct)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.qself {
        x.visit_mut(v);
    }
    &mut self.path.visit_mut(v);
    for mut el in Puncted::pairs_mut(&mut self.pats) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_pat_type_mut<V>(v: &mut V, node: &mut pat::Type)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *node.pat.visit_mut(v);
    &mut *node.typ.visit_mut(v);
}
pub fn visit_pat_wild_mut<V>(v: &mut V, node: &mut pat::Wild)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
}
pub fn visit_path_mut<V>(v: &mut V, node: &mut Path)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.segs) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_path_arguments_mut<V>(v: &mut V, node: &mut path::Args)
where
    V: VisitMut + ?Sized,
{
    use path::Args::*;
    match node {
        None => {},
        Angled(x) => {
            x.visit_mut(v);
        },
        Parenthed(x) => {
            x.visit_mut(v);
        },
    }
}
pub fn visit_path_segment_mut<V>(v: &mut V, node: &mut Segment)
where
    V: VisitMut + ?Sized,
{
    &mut self.ident.visit_mut(v);
    &mut self.args.visit_mut(v);
}
pub fn visit_predicate_lifetime_mut<V>(v: &mut V, node: &mut gen::Where::Life)
where
    V: VisitMut + ?Sized,
{
    &mut self.life.visit_mut(v);
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_predicate_type_mut<V>(v: &mut V, node: &mut gen::Where::Type)
where
    V: VisitMut + ?Sized,
{
    if let Some(x) = &mut self.lifes {
        x.visit_mut(v);
    }
    &mut self.bounded.visit_mut(v);
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_qself_mut<V>(v: &mut V, node: &mut QSelf)
where
    V: VisitMut + ?Sized,
{
    &mut *node.ty.visit_mut(v);
}
pub fn visit_range_limits_mut<V>(v: &mut V, node: &mut expr::Limits)
where
    V: VisitMut + ?Sized,
{
    match node {
        expr::Limits::HalfOpen(x) => {},
        expr::Limits::Closed(x) => {},
    }
}
pub fn visit_receiver_mut<V>(v: &mut V, node: &mut item::Receiver)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.ref_ {
        if let Some(x) = &mut (x).1 {
            x.visit_mut(v);
        }
    }
    &mut *node.typ.visit_mut(v);
}
pub fn visit_return_type_mut<V>(v: &mut V, node: &mut typ::Ret)
where
    V: VisitMut + ?Sized,
{
    match node {
        typ::Ret::Default => {},
        typ::Ret::Type(_binding_0, _binding_1) => {
            &mut **_binding_1.visit_mut(v);
        },
    }
}
pub fn visit_signature_mut<V>(v: &mut V, node: &mut item::Sig)
where
    V: VisitMut + ?Sized,
{
    if let Some(x) = &mut self.abi {
        x.visit_mut(v);
    }
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    for mut el in Puncted::pairs_mut(&mut self.args) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.vari {
        x.visit_mut(v);
    }
    &mut self.ret.visit_mut(v);
}
pub fn visit_span_mut<V>(v: &mut V, node: &mut pm2::Span)
where
    V: VisitMut + ?Sized,
{
}
pub fn visit_static_mutability_mut<V>(v: &mut V, node: &mut StaticMut)
where
    V: VisitMut + ?Sized,
{
    match node {
        StaticMut::Mut(x) => {},
        StaticMut::None => {},
    }
}
pub fn visit_stmt_mut<V>(v: &mut V, node: &mut stmt::Stmt)
where
    V: VisitMut + ?Sized,
{
    match node {
        stmt::Stmt::stmt::Local(x) => {
            x.visit_mut(v);
        },
        stmt::Stmt::Item(x) => {
            x.visit_mut(v);
        },
        stmt::Stmt::Expr(_binding_0, _binding_1) => {
            x.visit_mut(v);
        },
        stmt::Stmt::Mac(x) => {
            x.visit_mut(v);
        },
    }
}
pub fn visit_stmt_macro_mut<V>(v: &mut V, node: &mut stmt::Mac)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.mac.visit_mut(v);
}
pub fn visit_trait_bound_mut<V>(v: &mut V, node: &mut gen::bound::Trait)
where
    V: VisitMut + ?Sized,
{
    &mut self.modif.visit_mut(v);
    if let Some(x) = &mut self.lifes {
        x.visit_mut(v);
    }
    &mut self.path.visit_mut(v);
}
pub fn visit_trait_bound_modifier_mut<V>(v: &mut V, node: &mut gen::bound::Modifier)
where
    V: VisitMut + ?Sized,
{
    match node {
        gen::bound::Modifier::None => {},
        gen::bound::Modifier::Maybe(x) => {},
    }
}
pub fn visit_trait_item_mut<V>(v: &mut V, node: &mut item::trait_::Item)
where
    V: VisitMut + ?Sized,
{
    match node {
        item::trait_::Item::Const(x) => {
            x.visit_mut(v);
        },
        item::trait_::Item::Fn(x) => {
            x.visit_mut(v);
        },
        item::trait_::Item::Type(x) => {
            x.visit_mut(v);
        },
        item::trait_::Item::Macro(x) => {
            x.visit_mut(v);
        },
        item::trait_::Item::Verbatim(x) => {},
    }
}
pub fn visit_trait_item_const_mut<V>(v: &mut V, node: &mut item::trait_::Const)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    &mut self.typ.visit_mut(v);
    if let Some(x) = &mut self.default {
        &mut (x).1.visit_mut(v);
    }
}
pub fn visit_trait_item_fn_mut<V>(v: &mut V, node: &mut item::trait_::Fn)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.sig.visit_mut(v);
    if let Some(x) = &mut self.default {
        x.visit_mut(v);
    }
}
pub fn visit_trait_item_macro_mut<V>(v: &mut V, node: &mut item::trait_::Mac)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.mac.visit_mut(v);
}
pub fn visit_trait_item_type_mut<V>(v: &mut V, node: &mut item::trait_::Type)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.default {
        &mut (x).1.visit_mut(v);
    }
}
pub fn visit_type_mut<V>(v: &mut V, node: &mut typ::Type)
where
    V: VisitMut + ?Sized,
{
    match node {
        typ::Type::Array(x) => {
            x.visit_mut(v);
        },
        typ::Type::Fn(x) => {
            x.visit_mut(v);
        },
        typ::Type::Group(x) => {
            x.visit_mut(v);
        },
        typ::Type::Impl(x) => {
            x.visit_mut(v);
        },
        typ::Type::Infer(x) => {
            x.visit_mut(v);
        },
        typ::Type::Mac(x) => {
            x.visit_mut(v);
        },
        typ::Type::Never(x) => {
            x.visit_mut(v);
        },
        typ::Type::Parenth(x) => {
            x.visit_mut(v);
        },
        typ::Type::Path(x) => {
            x.visit_mut(v);
        },
        typ::Type::Ptr(x) => {
            x.visit_mut(v);
        },
        typ::Type::Ref(x) => {
            x.visit_mut(v);
        },
        typ::Type::Slice(x) => {
            x.visit_mut(v);
        },
        typ::Type::Trait(x) => {
            x.visit_mut(v);
        },
        typ::Type::Tuple(x) => {
            x.visit_mut(v);
        },
        typ::Type::Stream(x) => {},
    }
}
pub fn visit_type_array_mut<V>(v: &mut V, node: &mut typ::Array)
where
    V: VisitMut + ?Sized,
{
    &mut *node.elem.visit_mut(v);
    &mut self.len.visit_mut(v);
}
pub fn visit_type_bare_fn_mut<V>(v: &mut V, node: &mut typ::Fn)
where
    V: VisitMut + ?Sized,
{
    if let Some(x) = &mut self.lifes {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.abi {
        x.visit_mut(v);
    }
    for mut el in Puncted::pairs_mut(&mut self.args) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.vari {
        x.visit_mut(v);
    }
    &mut self.ret.visit_mut(v);
}
pub fn visit_type_group_mut<V>(v: &mut V, node: &mut typ::Group)
where
    V: VisitMut + ?Sized,
{
    &mut *node.elem.visit_mut(v);
}
pub fn visit_type_impl_trait_mut<V>(v: &mut V, node: &mut typ::Impl)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_type_infer_mut<V>(v: &mut V, node: &mut typ::Infer)
where
    V: VisitMut + ?Sized,
{
}
pub fn visit_type_macro_mut<V>(v: &mut V, node: &mut typ::Mac)
where
    V: VisitMut + ?Sized,
{
    &mut self.mac.visit_mut(v);
}
pub fn visit_type_never_mut<V>(v: &mut V, node: &mut typ::Never)
where
    V: VisitMut + ?Sized,
{
}
pub fn visit_type_param_mut<V>(v: &mut V, node: &mut gen::param::Type)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.ident.visit_mut(v);
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.default {
        x.visit_mut(v);
    }
}
pub fn visit_type_param_bound_mut<V>(v: &mut V, node: &mut gen::bound::Type)
where
    V: VisitMut + ?Sized,
{
    match node {
        gen::bound::Type::Trait(x) => {
            x.visit_mut(v);
        },
        gen::bound::Type::Life(x) => {
            x.visit_mut(v);
        },
        gen::bound::Type::Verbatim(x) => {},
    }
}
pub fn visit_type_paren_mut<V>(v: &mut V, node: &mut typ::Parenth)
where
    V: VisitMut + ?Sized,
{
    &mut *node.elem.visit_mut(v);
}
pub fn visit_type_path_mut<V>(v: &mut V, node: &mut typ::Path)
where
    V: VisitMut + ?Sized,
{
    if let Some(x) = &mut self.qself {
        x.visit_mut(v);
    }
    &mut self.path.visit_mut(v);
}
pub fn visit_type_ptr_mut<V>(v: &mut V, node: &mut typ::Ptr)
where
    V: VisitMut + ?Sized,
{
    &mut *node.elem.visit_mut(v);
}
pub fn visit_type_reference_mut<V>(v: &mut V, node: &mut typ::Ref)
where
    V: VisitMut + ?Sized,
{
    if let Some(x) = &mut self.life {
        x.visit_mut(v);
    }
    &mut *node.elem.visit_mut(v);
}
pub fn visit_type_slice_mut<V>(v: &mut V, node: &mut typ::Slice)
where
    V: VisitMut + ?Sized,
{
    &mut *node.elem.visit_mut(v);
}
pub fn visit_type_trait_object_mut<V>(v: &mut V, node: &mut typ::Trait)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_type_tuple_mut<V>(v: &mut V, node: &mut typ::Tuple)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.elems) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_un_op_mut<V>(v: &mut V, node: &mut UnOp)
where
    V: VisitMut + ?Sized,
{
    match node {
        UnOp::Deref(x) => {},
        UnOp::Not(x) => {},
        UnOp::Neg(x) => {},
    }
}
pub fn visit_use_glob_mut<V>(v: &mut V, node: &mut item::use_::Glob)
where
    V: VisitMut + ?Sized,
{
}
pub fn visit_use_group_mut<V>(v: &mut V, node: &mut item::use_::Group)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.trees) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_use_name_mut<V>(v: &mut V, node: &mut item::use_::Name)
where
    V: VisitMut + ?Sized,
{
    &mut self.ident.visit_mut(v);
}
pub fn visit_use_path_mut<V>(v: &mut V, node: &mut item::use_::Path)
where
    V: VisitMut + ?Sized,
{
    &mut self.ident.visit_mut(v);
    &mut *node.tree.visit_mut(v);
}
pub fn visit_use_rename_mut<V>(v: &mut V, node: &mut item::use_::Rename)
where
    V: VisitMut + ?Sized,
{
    &mut self.ident.visit_mut(v);
    &mut self.rename.visit_mut(v);
}
pub fn visit_use_tree_mut<V>(v: &mut V, node: &mut item::use_::Tree)
where
    V: VisitMut + ?Sized,
{
    match node {
        item::use_::Tree::Path(x) => {
            x.visit_mut(v);
        },
        item::use_::Tree::Name(x) => {
            x.visit_mut(v);
        },
        item::use_::Tree::Rename(x) => {
            x.visit_mut(v);
        },
        item::use_::Tree::Glob(x) => {
            x.visit_mut(v);
        },
        item::use_::Tree::Group(x) => {
            x.visit_mut(v);
        },
    }
}
pub fn visit_variadic_mut<V>(v: &mut V, node: &mut item::Variadic)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.pat {
        &mut *(x).0.visit_mut(v);
    }
}
pub fn visit_variant_mut<V>(v: &mut V, node: &mut data::Variant)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.ident.visit_mut(v);
    &mut self.fields.visit_mut(v);
    if let Some(x) = &mut self.discrim {
        &mut (x).1.visit_mut(v);
    }
}
pub fn visit_vis_restricted_mut<V>(v: &mut V, node: &mut data::Restricted)
where
    V: VisitMut + ?Sized,
{
    &mut *node.path.visit_mut(v);
}
pub fn visit_visibility_mut<V>(v: &mut V, node: &mut data::Visibility)
where
    V: VisitMut + ?Sized,
{
    match node {
        data::Visibility::Public(x) => {},
        data::Visibility::Restricted(x) => {
            x.visit_mut(v);
        },
        data::Visibility::Inherited => {},
    }
}
pub fn visit_where_clause_mut<V>(v: &mut V, node: &mut gen::Where)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.preds) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
pub fn visit_where_predicate_mut<V>(v: &mut V, node: &mut gen::Where::Pred)
where
    V: VisitMut + ?Sized,
{
    match node {
        gen::Where::Pred::Life(x) => {
            x.visit_mut(v);
        },
        gen::Where::Pred::Type(x) => {
            x.visit_mut(v);
        },
    }
}
