#![allow(unused_variables)]

use crate::*;

macro_rules! full {
    ($e:expr) => {
        $e
    };
}
macro_rules! skip {
    ($($tt:tt)*) => {};
}

pub trait Visit<'a> {
    fn visit_abi(&mut self, x: &'a typ::Abi) {
        visit_abi(self, x);
    }
    fn visit_angle_bracketed_generic_arguments(&mut self, x: &'a path::Angled) {
        visit_angle_bracketed_generic_arguments(self, x);
    }
    fn visit_arm(&mut self, x: &'a expr::Arm) {
        visit_arm(self, x);
    }
    fn visit_assoc_const(&mut self, x: &'a path::AssocConst) {
        visit_assoc_const(self, x);
    }
    fn visit_assoc_type(&mut self, x: &'a path::AssocType) {
        visit_assoc_type(self, x);
    }
    fn visit_attr_style(&mut self, x: &'a attr::Style) {
        visit_attr_style(self, x);
    }
    fn visit_attribute(&mut self, x: &'a attr::Attr) {
        visit_attribute(self, x);
    }
    fn visit_bare_fn_arg(&mut self, x: &'a typ::FnArg) {
        visit_bare_fn_arg(self, x);
    }
    fn visit_bare_variadic(&mut self, x: &'a typ::Variadic) {
        visit_bare_variadic(self, x);
    }
    fn visit_bin_op(&mut self, x: &'a expr::BinOp) {
        visit_bin_op(self, x);
    }
    fn visit_block(&mut self, x: &'a stmt::Block) {
        visit_block(self, x);
    }
    fn visit_bound_lifetimes(&mut self, x: &'a gen::bound::Lifes) {
        visit_bound_lifetimes(self, x);
    }
    fn visit_const_param(&mut self, x: &'a gen::param::Const) {
        visit_const_param(self, x);
    }
    fn visit_constraint(&mut self, x: &'a path::Constraint) {
        visit_constraint(self, x);
    }
    fn visit_data(&mut self, x: &'a data::Data) {
        visit_data(self, x);
    }
    fn visit_data_enum(&mut self, x: &'a data::Enum) {
        visit_data_enum(self, x);
    }
    fn visit_data_struct(&mut self, x: &'a data::Struct) {
        visit_data_struct(self, x);
    }
    fn visit_data_union(&mut self, x: &'a data::Union) {
        visit_data_union(self, x);
    }
    fn visit_derive_input(&mut self, x: &'a Input) {
        visit_derive_input(self, x);
    }
    fn visit_expr(&mut self, x: &'a expr::Expr) {
        visit_expr(self, x);
    }
    fn visit_expr_array(&mut self, x: &'a expr::Array) {
        visit_expr_array(self, x);
    }
    fn visit_expr_assign(&mut self, x: &'a expr::Assign) {
        visit_expr_assign(self, x);
    }
    fn visit_expr_async(&mut self, x: &'a expr::Async) {
        visit_expr_async(self, x);
    }
    fn visit_expr_await(&mut self, x: &'a expr::Await) {
        visit_expr_await(self, x);
    }
    fn visit_expr_binary(&mut self, x: &'a expr::Binary) {
        visit_expr_binary(self, x);
    }
    fn visit_expr_block(&mut self, x: &'a expr::Block) {
        visit_expr_block(self, x);
    }
    fn visit_expr_break(&mut self, x: &'a expr::Break) {
        visit_expr_break(self, x);
    }
    fn visit_expr_call(&mut self, x: &'a expr::Call) {
        visit_expr_call(self, x);
    }
    fn visit_expr_cast(&mut self, x: &'a expr::Cast) {
        visit_expr_cast(self, x);
    }
    fn visit_expr_closure(&mut self, x: &'a expr::Closure) {
        visit_expr_closure(self, x);
    }
    fn visit_expr_const(&mut self, x: &'a expr::Const) {
        visit_expr_const(self, x);
    }
    fn visit_expr_continue(&mut self, x: &'a expr::Continue) {
        visit_expr_continue(self, x);
    }
    fn visit_expr_field(&mut self, x: &'a expr::Field) {
        visit_expr_field(self, x);
    }
    fn visit_expr_for_loop(&mut self, x: &'a expr::ForLoop) {
        visit_expr_for_loop(self, x);
    }
    fn visit_expr_group(&mut self, x: &'a expr::Group) {
        visit_expr_group(self, x);
    }
    fn visit_expr_if(&mut self, x: &'a expr::If) {
        visit_expr_if(self, x);
    }
    fn visit_expr_index(&mut self, x: &'a expr::Index) {
        visit_expr_index(self, x);
    }
    fn visit_expr_infer(&mut self, x: &'a expr::Infer) {
        visit_expr_infer(self, x);
    }
    fn visit_expr_let(&mut self, x: &'a expr::Let) {
        visit_expr_let(self, x);
    }
    fn visit_expr_lit(&mut self, x: &'a expr::Lit) {
        visit_expr_lit(self, x);
    }
    fn visit_expr_loop(&mut self, x: &'a expr::Loop) {
        visit_expr_loop(self, x);
    }
    fn visit_expr_macro(&mut self, x: &'a expr::Mac) {
        visit_expr_macro(self, x);
    }
    fn visit_expr_match(&mut self, x: &'a expr::Match) {
        visit_expr_match(self, x);
    }
    fn visit_expr_method_call(&mut self, x: &'a expr::MethodCall) {
        visit_expr_method_call(self, x);
    }
    fn visit_expr_paren(&mut self, x: &'a expr::Parenth) {
        visit_expr_paren(self, x);
    }
    fn visit_expr_path(&mut self, x: &'a expr::Path) {
        visit_expr_path(self, x);
    }
    fn visit_expr_range(&mut self, x: &'a expr::Range) {
        visit_expr_range(self, x);
    }
    fn visit_expr_reference(&mut self, x: &'a expr::Ref) {
        visit_expr_reference(self, x);
    }
    fn visit_expr_repeat(&mut self, x: &'a expr::Repeat) {
        visit_expr_repeat(self, x);
    }
    fn visit_expr_return(&mut self, x: &'a expr::Return) {
        visit_expr_return(self, x);
    }
    fn visit_expr_struct(&mut self, x: &'a expr::Struct) {
        visit_expr_struct(self, x);
    }
    fn visit_expr_try(&mut self, x: &'a expr::Try) {
        visit_expr_try(self, x);
    }
    fn visit_expr_try_block(&mut self, x: &'a expr::TryBlock) {
        visit_expr_try_block(self, x);
    }
    fn visit_expr_tuple(&mut self, x: &'a expr::Tuple) {
        visit_expr_tuple(self, x);
    }
    fn visit_expr_unary(&mut self, x: &'a expr::Unary) {
        visit_expr_unary(self, x);
    }
    fn visit_expr_unsafe(&mut self, x: &'a expr::Unsafe) {
        visit_expr_unsafe(self, x);
    }
    fn visit_expr_while(&mut self, x: &'a expr::While) {
        visit_expr_while(self, x);
    }
    fn visit_expr_yield(&mut self, x: &'a expr::Yield) {
        visit_expr_yield(self, x);
    }
    fn visit_field(&mut self, x: &'a data::Field) {
        visit_field(self, x);
    }
    fn visit_field_mutability(&mut self, x: &'a data::Mut) {
        visit_field_mutability(self, x);
    }
    fn visit_field_pat(&mut self, x: &'a pat::Field) {
        visit_field_pat(self, x);
    }
    fn visit_field_value(&mut self, x: &'a expr::FieldValue) {
        visit_field_value(self, x);
    }
    fn visit_fields(&mut self, x: &'a data::Fields) {
        visit_fields(self, x);
    }
    fn visit_fields_named(&mut self, x: &'a data::Named) {
        visit_fields_named(self, x);
    }
    fn visit_fields_unnamed(&mut self, x: &'a data::Unnamed) {
        visit_fields_unnamed(self, x);
    }
    fn visit_file(&mut self, x: &'a item::File) {
        visit_file(self, x);
    }
    fn visit_fn_arg(&mut self, x: &'a item::FnArg) {
        visit_fn_arg(self, x);
    }
    fn visit_foreign_item(&mut self, x: &'a item::foreign::Item) {
        visit_foreign_item(self, x);
    }
    fn visit_foreign_item_fn(&mut self, x: &'a item::foreign::Fn) {
        visit_foreign_item_fn(self, x);
    }
    fn visit_foreign_item_macro(&mut self, x: &'a item::foreign::Mac) {
        visit_foreign_item_macro(self, x);
    }
    fn visit_foreign_item_static(&mut self, x: &'a item::foreign::Static) {
        visit_foreign_item_static(self, x);
    }
    fn visit_foreign_item_type(&mut self, x: &'a item::foreign::Type) {
        visit_foreign_item_type(self, x);
    }
    fn visit_generic_argument(&mut self, x: &'a path::Arg) {
        visit_generic_argument(self, x);
    }
    fn visit_generic_param(&mut self, x: &'a gen::Param) {
        visit_generic_param(self, x);
    }
    fn visit_generics(&mut self, x: &'a gen::Gens) {
        visit_generics(self, x);
    }
    fn visit_ident(&mut self, x: &'a Ident) {
        visit_ident(self, x);
    }
    fn visit_impl_item(&mut self, x: &'a item::impl_::Item) {
        visit_impl_item(self, x);
    }
    fn visit_impl_item_const(&mut self, x: &'a item::impl_::Const) {
        visit_impl_item_const(self, x);
    }
    fn visit_impl_item_fn(&mut self, x: &'a item::impl_::Fn) {
        visit_impl_item_fn(self, x);
    }
    fn visit_impl_item_macro(&mut self, x: &'a item::impl_::Mac) {
        visit_impl_item_macro(self, x);
    }
    fn visit_impl_item_type(&mut self, x: &'a item::impl_::Type) {
        visit_impl_item_type(self, x);
    }
    fn visit_impl_restriction(&mut self, x: &'a item::impl_::Restriction) {
        visit_impl_restriction(self, x);
    }
    fn visit_index(&mut self, x: &'a expr::Index) {
        visit_index(self, x);
    }
    fn visit_item(&mut self, x: &'a item::Item) {
        visit_item(self, x);
    }
    fn visit_item_const(&mut self, x: &'a item::Const) {
        visit_item_const(self, x);
    }
    fn visit_item_enum(&mut self, x: &'a item::Enum) {
        visit_item_enum(self, x);
    }
    fn visit_item_extern_crate(&mut self, x: &'a item::Extern) {
        visit_item_extern_crate(self, x);
    }
    fn visit_item_fn(&mut self, x: &'a item::Fn) {
        visit_item_fn(self, x);
    }
    fn visit_item_foreign_mod(&mut self, x: &'a item::Foreign) {
        visit_item_foreign_mod(self, x);
    }
    fn visit_item_impl(&mut self, x: &'a item::Impl) {
        visit_item_impl(self, x);
    }
    fn visit_item_macro(&mut self, x: &'a item::Mac) {
        visit_item_macro(self, x);
    }
    fn visit_item_mod(&mut self, x: &'a item::Mod) {
        visit_item_mod(self, x);
    }
    fn visit_item_static(&mut self, x: &'a item::Static) {
        visit_item_static(self, x);
    }
    fn visit_item_struct(&mut self, x: &'a item::Struct) {
        visit_item_struct(self, x);
    }
    fn visit_item_trait(&mut self, x: &'a item::Trait) {
        visit_item_trait(self, x);
    }
    fn visit_item_trait_alias(&mut self, x: &'a item::TraitAlias) {
        visit_item_trait_alias(self, x);
    }
    fn visit_item_type(&mut self, x: &'a item::Type) {
        visit_item_type(self, x);
    }
    fn visit_item_union(&mut self, x: &'a item::Union) {
        visit_item_union(self, x);
    }
    fn visit_item_use(&mut self, x: &'a item::Use) {
        visit_item_use(self, x);
    }
    fn visit_label(&mut self, x: &'a expr::Label) {
        visit_label(self, x);
    }
    fn visit_lifetime(&mut self, x: &'a Life) {
        visit_lifetime(self, x);
    }
    fn visit_lifetime_param(&mut self, x: &'a gen::param::Life) {
        visit_lifetime_param(self, x);
    }
    fn visit_lit(&mut self, x: &'a lit::Lit) {
        visit_lit(self, x);
    }
    fn visit_lit_bool(&mut self, x: &'a lit::Bool) {
        visit_lit_bool(self, x);
    }
    fn visit_lit_byte(&mut self, x: &'a lit::Byte) {
        visit_lit_byte(self, x);
    }
    fn visit_lit_byte_str(&mut self, x: &'a lit::ByteStr) {
        visit_lit_byte_str(self, x);
    }
    fn visit_lit_char(&mut self, x: &'a lit::Char) {
        visit_lit_char(self, x);
    }
    fn visit_lit_float(&mut self, x: &'a lit::Float) {
        visit_lit_float(self, x);
    }
    fn visit_lit_int(&mut self, x: &'a lit::Int) {
        visit_lit_int(self, x);
    }
    fn visit_lit_str(&mut self, x: &'a lit::Str) {
        visit_lit_str(self, x);
    }
    fn visit_local(&mut self, x: &'a stmt::Local) {
        visit_local(self, x);
    }
    fn visit_local_init(&mut self, x: &'a stmt::Init) {
        visit_local_init(self, x);
    }
    fn visit_macro(&mut self, x: &'a mac::Mac) {
        visit_macro(self, x);
    }
    fn visit_macro_delimiter(&mut self, x: &'a tok::Delim) {
        visit_macro_delimiter(self, x);
    }
    fn visit_member(&mut self, x: &'a expr::Member) {
        visit_member(self, x);
    }
    fn visit_meta(&mut self, x: &'a attr::Meta) {
        visit_meta(self, x);
    }
    fn visit_meta_list(&mut self, x: &'a attr::List) {
        visit_meta_list(self, x);
    }
    fn visit_meta_name_value(&mut self, x: &'a attr::NameValue) {
        visit_meta_name_value(self, x);
    }
    fn visit_parenthesized_generic_arguments(&mut self, x: &'a path::Parenthed) {
        visit_parenthesized_generic_arguments(self, x);
    }
    fn visit_pat(&mut self, x: &'a pat::Pat) {
        visit_pat(self, x);
    }
    fn visit_pat_ident(&mut self, x: &'a pat::Ident) {
        visit_pat_ident(self, x);
    }
    fn visit_pat_or(&mut self, x: &'a pat::Or) {
        visit_pat_or(self, x);
    }
    fn visit_pat_paren(&mut self, x: &'a pat::Parenth) {
        visit_pat_paren(self, x);
    }
    fn visit_pat_reference(&mut self, x: &'a pat::Ref) {
        visit_pat_reference(self, x);
    }
    fn visit_pat_rest(&mut self, x: &'a pat::Rest) {
        visit_pat_rest(self, x);
    }
    fn visit_pat_slice(&mut self, x: &'a pat::Slice) {
        visit_pat_slice(self, x);
    }
    fn visit_pat_struct(&mut self, x: &'a pat::Struct) {
        visit_pat_struct(self, x);
    }
    fn visit_pat_tuple(&mut self, x: &'a pat::Tuple) {
        visit_pat_tuple(self, x);
    }
    fn visit_pat_tuple_struct(&mut self, x: &'a pat::TupleStruct) {
        visit_pat_tuple_struct(self, x);
    }
    fn visit_pat_type(&mut self, x: &'a pat::Type) {
        visit_pat_type(self, x);
    }
    fn visit_pat_wild(&mut self, x: &'a pat::Wild) {
        visit_pat_wild(self, x);
    }
    fn visit_path(&mut self, x: &'a path::Path) {
        visit_path(self, x);
    }
    fn visit_path_arguments(&mut self, x: &'a path::Args) {
        visit_path_arguments(self, x);
    }
    fn visit_path_segment(&mut self, x: &'a path::Segment) {
        visit_path_segment(self, x);
    }
    fn visit_predicate_lifetime(&mut self, x: &'a gen::where_::Life) {
        visit_predicate_lifetime(self, x);
    }
    fn visit_predicate_type(&mut self, x: &'a gen::where_::Type) {
        visit_predicate_type(self, x);
    }
    fn visit_qself(&mut self, x: &'a path::QSelf) {
        visit_qself(self, x);
    }
    fn visit_range_limits(&mut self, x: &'a expr::Limits) {
        visit_range_limits(self, x);
    }
    fn visit_receiver(&mut self, x: &'a item::Receiver) {
        visit_receiver(self, x);
    }
    fn visit_return_type(&mut self, x: &'a typ::Ret) {
        visit_return_type(self, x);
    }
    fn visit_signature(&mut self, x: &'a item::Sig) {
        visit_signature(self, x);
    }
    fn visit_span(&mut self, x: &pm2::Span) {
        visit_span(self, x);
    }
    fn visit_static_mutability(&mut self, x: &'a item::StaticMut) {
        visit_static_mutability(self, x);
    }
    fn visit_stmt(&mut self, x: &'a stmt::Stmt) {
        visit_stmt(self, x);
    }
    fn visit_stmt_macro(&mut self, x: &'a stmt::Mac) {
        visit_stmt_macro(self, x);
    }
    fn visit_trait_bound(&mut self, x: &'a gen::bound::Trait) {
        visit_trait_bound(self, x);
    }
    fn visit_trait_bound_modifier(&mut self, x: &'a gen::bound::Modifier) {
        visit_trait_bound_modifier(self, x);
    }
    fn visit_trait_item(&mut self, x: &'a item::trait_::Item) {
        visit_trait_item(self, x);
    }
    fn visit_trait_item_const(&mut self, x: &'a item::trait_::Const) {
        visit_trait_item_const(self, x);
    }
    fn visit_trait_item_fn(&mut self, x: &'a item::trait_::Fn) {
        visit_trait_item_fn(self, x);
    }
    fn visit_trait_item_macro(&mut self, x: &'a item::trait_::Mac) {
        visit_trait_item_macro(self, x);
    }
    fn visit_trait_item_type(&mut self, x: &'a item::trait_::Type) {
        visit_trait_item_type(self, x);
    }
    fn visit_type(&mut self, x: &'a typ::Type) {
        visit_type(self, x);
    }
    fn visit_type_array(&mut self, x: &'a typ::Array) {
        visit_type_array(self, x);
    }
    fn visit_type_bare_fn(&mut self, x: &'a typ::Fn) {
        visit_type_bare_fn(self, x);
    }
    fn visit_type_group(&mut self, x: &'a typ::Group) {
        visit_type_group(self, x);
    }
    fn visit_type_impl_trait(&mut self, x: &'a typ::Impl) {
        visit_type_impl_trait(self, x);
    }
    fn visit_type_infer(&mut self, x: &'a typ::Infer) {
        visit_type_infer(self, x);
    }
    fn visit_type_macro(&mut self, x: &'a typ::Mac) {
        visit_type_macro(self, x);
    }
    fn visit_type_never(&mut self, x: &'a typ::Never) {
        visit_type_never(self, x);
    }
    fn visit_type_param(&mut self, x: &'a gen::param::Type) {
        visit_type_param(self, x);
    }
    fn visit_type_param_bound(&mut self, x: &'a gen::bound::Type) {
        visit_type_param_bound(self, x);
    }
    fn visit_type_paren(&mut self, x: &'a typ::Parenth) {
        visit_type_paren(self, x);
    }
    fn visit_type_path(&mut self, x: &'a typ::Path) {
        visit_type_path(self, x);
    }
    fn visit_type_ptr(&mut self, x: &'a typ::Ptr) {
        visit_type_ptr(self, x);
    }
    fn visit_type_reference(&mut self, x: &'a typ::Ref) {
        visit_type_reference(self, x);
    }
    fn visit_type_slice(&mut self, x: &'a typ::Slice) {
        visit_type_slice(self, x);
    }
    fn visit_type_trait_object(&mut self, x: &'a typ::Trait) {
        visit_type_trait_object(self, x);
    }
    fn visit_type_tuple(&mut self, x: &'a typ::Tuple) {
        visit_type_tuple(self, x);
    }
    fn visit_un_op(&mut self, x: &'a expr::UnOp) {
        visit_un_op(self, x);
    }
    fn visit_use_glob(&mut self, x: &'a item::use_::Glob) {
        visit_use_glob(self, x);
    }
    fn visit_use_group(&mut self, x: &'a item::use_::Group) {
        visit_use_group(self, x);
    }
    fn visit_use_name(&mut self, x: &'a item::use_::Name) {
        visit_use_name(self, x);
    }
    fn visit_use_path(&mut self, x: &'a item::use_::Path) {
        visit_use_path(self, x);
    }
    fn visit_use_rename(&mut self, x: &'a item::use_::Rename) {
        visit_use_rename(self, x);
    }
    fn visit_use_tree(&mut self, x: &'a item::use_::Tree) {
        visit_use_tree(self, x);
    }
    fn visit_variadic(&mut self, x: &'a item::Variadic) {
        visit_variadic(self, x);
    }
    fn visit_variant(&mut self, x: &'a data::Variant) {
        visit_variant(self, x);
    }
    fn visit_vis_restricted(&mut self, x: &'a data::Restricted) {
        visit_vis_restricted(self, x);
    }
    fn visit_visibility(&mut self, x: &'a data::Visibility) {
        visit_visibility(self, x);
    }
    fn visit_where_clause(&mut self, x: &'a gen::Where) {
        visit_where_clause(self, x);
    }
    fn visit_where_predicate(&mut self, x: &'a gen::where_::Pred) {
        visit_where_predicate(self, x);
    }
}
pub fn visit_abi<'a, V>(v: &mut V, node: &'a typ::Abi)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.extern_);
    if let Some(x) = &node.name {
        v.visit_lit_str(x);
    }
}
pub fn visit_angle_bracketed_generic_arguments<'a, V>(v: &mut V, node: &'a path::Angled)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.colon2);
    skip!(node.lt);
    for el in Puncted::pairs(&node.args) {
        let x = el.value();
        v.visit_generic_argument(x);
    }
    skip!(node.gt);
}
pub fn visit_arm<'a, V>(v: &mut V, node: &'a expr::Arm)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_pat(&node.pat);
    if let Some(x) = &node.guard {
        skip!((x).0);
        v.visit_expr(&*(x).1);
    }
    skip!(node.fat_arrow);
    v.visit_expr(&*node.body);
    skip!(node.comma);
}
pub fn visit_assoc_const<'a, V>(v: &mut V, node: &'a path::AssocConst)
where
    V: Visit<'a> + ?Sized,
{
    v.visit_ident(&node.ident);
    if let Some(x) = &node.gnrs {
        v.visit_angle_bracketed_generic_arguments(x);
    }
    skip!(node.eq);
    v.visit_expr(&node.val);
}
pub fn visit_assoc_type<'a, V>(v: &mut V, node: &'a path::AssocType)
where
    V: Visit<'a> + ?Sized,
{
    v.visit_ident(&node.ident);
    if let Some(x) = &node.gnrs {
        v.visit_angle_bracketed_generic_arguments(x);
    }
    skip!(node.eq);
    v.visit_type(&node.ty);
}
pub fn visit_attr_style<'a, V>(v: &mut V, node: &'a attr::Style)
where
    V: Visit<'a> + ?Sized,
{
    match node {
        attr::Style::Outer => {},
        attr::Style::Inner(x) => {
            skip!(x);
        },
    }
}
pub fn visit_attribute<'a, V>(v: &mut V, node: &'a attr::Attr)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.pound);
    v.visit_attr_style(&node.style);
    skip!(node.bracket);
    v.visit_meta(&node.meta);
}
pub fn visit_bare_fn_arg<'a, V>(v: &mut V, node: &'a typ::FnArg)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    if let Some(x) = &node.name {
        v.visit_ident(&(x).0);
        skip!((x).1);
    }
    v.visit_type(&node.typ);
}
pub fn visit_bare_variadic<'a, V>(v: &mut V, node: &'a typ::Variadic)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    if let Some(x) = &node.name {
        v.visit_ident(&(x).0);
        skip!((x).1);
    }
    skip!(node.dots);
    skip!(node.comma);
}
pub fn visit_bin_op<'a, V>(v: &mut V, node: &'a expr::BinOp)
where
    V: Visit<'a> + ?Sized,
{
    use expr::BinOp::*;
    match node {
        Add(x) => {
            skip!(x);
        },
        Sub(x) => {
            skip!(x);
        },
        Mul(x) => {
            skip!(x);
        },
        Div(x) => {
            skip!(x);
        },
        Rem(x) => {
            skip!(x);
        },
        And(x) => {
            skip!(x);
        },
        Or(x) => {
            skip!(x);
        },
        BitXor(x) => {
            skip!(x);
        },
        BitAnd(x) => {
            skip!(x);
        },
        BitOr(x) => {
            skip!(x);
        },
        Shl(x) => {
            skip!(x);
        },
        Shr(x) => {
            skip!(x);
        },
        Eq(x) => {
            skip!(x);
        },
        Lt(x) => {
            skip!(x);
        },
        Le(x) => {
            skip!(x);
        },
        Ne(x) => {
            skip!(x);
        },
        Ge(x) => {
            skip!(x);
        },
        Gt(x) => {
            skip!(x);
        },
        AddAssign(x) => {
            skip!(x);
        },
        SubAssign(x) => {
            skip!(x);
        },
        MulAssign(x) => {
            skip!(x);
        },
        DivAssign(x) => {
            skip!(x);
        },
        RemAssign(x) => {
            skip!(x);
        },
        BitXorAssign(x) => {
            skip!(x);
        },
        BitAndAssign(x) => {
            skip!(x);
        },
        BitOrAssign(x) => {
            skip!(x);
        },
        ShlAssign(x) => {
            skip!(x);
        },
        ShrAssign(x) => {
            skip!(x);
        },
    }
}
pub fn visit_block<'a, V>(v: &mut V, node: &'a stmt::Block)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.brace);
    for x in &node.stmts {
        v.visit_stmt(x);
    }
}
pub fn visit_bound_lifetimes<'a, V>(v: &mut V, node: &'a gen::bound::Lifes)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.for_);
    skip!(node.lt);
    for el in Puncted::pairs(&node.lifes) {
        let x = el.value();
        v.visit_generic_param(x);
    }
    skip!(node.gt);
}
pub fn visit_const_param<'a, V>(v: &mut V, node: &'a gen::param::Const)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.const_);
    v.visit_ident(&node.ident);
    skip!(node.colon);
    v.visit_type(&node.typ);
    skip!(node.eq);
    if let Some(x) = &node.default {
        v.visit_expr(x);
    }
}
pub fn visit_constraint<'a, V>(v: &mut V, node: &'a path::Constraint)
where
    V: Visit<'a> + ?Sized,
{
    v.visit_ident(&node.ident);
    if let Some(x) = &node.gnrs {
        v.visit_angle_bracketed_generic_arguments(x);
    }
    skip!(node.colon);
    for el in Puncted::pairs(&node.bounds) {
        let x = el.value();
        v.visit_type_param_bound(x);
    }
}
pub fn visit_data<'a, V>(v: &mut V, node: &'a data::Data)
where
    V: Visit<'a> + ?Sized,
{
    match node {
        data::Data::Struct(x) => {
            v.visit_data_struct(x);
        },
        data::Data::Enum(x) => {
            v.visit_data_enum(x);
        },
        data::Data::Union(x) => {
            v.visit_data_union(x);
        },
    }
}
pub fn visit_data_enum<'a, V>(v: &mut V, node: &'a data::Enum)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.enum_);
    skip!(node.brace);
    for el in Puncted::pairs(&node.variants) {
        let x = el.value();
        v.visit_variant(x);
    }
}
pub fn visit_data_struct<'a, V>(v: &mut V, node: &'a data::Struct)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.struct_);
    v.visit_fields(&node.fields);
    skip!(node.semi);
}
pub fn visit_data_union<'a, V>(v: &mut V, node: &'a data::Union)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.union_);
    v.visit_fields_named(&node.fields);
}
pub fn visit_derive_input<'a, V>(v: &mut V, node: &'a Input)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    v.visit_data(&node.data);
}
pub fn visit_expr<'a, V>(v: &mut V, node: &'a expr::Expr)
where
    V: Visit<'a> + ?Sized,
{
    use expr::Expr::*;
    match node {
        Array(x) => {
            full!(v.visit_expr_array(x));
        },
        Assign(x) => {
            full!(v.visit_expr_assign(x));
        },
        Async(x) => {
            full!(v.visit_expr_async(x));
        },
        Await(x) => {
            full!(v.visit_expr_await(x));
        },
        Binary(x) => {
            v.visit_expr_binary(x);
        },
        Block(x) => {
            full!(v.visit_expr_block(x));
        },
        Break(x) => {
            full!(v.visit_expr_break(x));
        },
        Call(x) => {
            v.visit_expr_call(x);
        },
        Cast(x) => {
            v.visit_expr_cast(x);
        },
        Closure(x) => {
            full!(v.visit_expr_closure(x));
        },
        Const(x) => {
            full!(v.visit_expr_const(x));
        },
        Continue(x) => {
            full!(v.visit_expr_continue(x));
        },
        Field(x) => {
            v.visit_expr_field(x);
        },
        ForLoop(x) => {
            full!(v.visit_expr_for_loop(x));
        },
        Group(x) => {
            v.visit_expr_group(x);
        },
        If(x) => {
            full!(v.visit_expr_if(x));
        },
        Index(x) => {
            v.visit_expr_index(x);
        },
        Infer(x) => {
            full!(v.visit_expr_infer(x));
        },
        Let(x) => {
            full!(v.visit_expr_let(x));
        },
        Lit(x) => {
            v.visit_expr_lit(x);
        },
        Loop(x) => {
            full!(v.visit_expr_loop(x));
        },
        Mac(x) => {
            v.visit_expr_macro(x);
        },
        Match(x) => {
            full!(v.visit_expr_match(x));
        },
        MethodCall(x) => {
            full!(v.visit_expr_method_call(x));
        },
        Parenth(x) => {
            v.visit_expr_paren(x);
        },
        Path(x) => {
            v.visit_expr_path(x);
        },
        Range(x) => {
            full!(v.visit_expr_range(x));
        },
        Ref(x) => {
            full!(v.visit_expr_reference(x));
        },
        Repeat(x) => {
            full!(v.visit_expr_repeat(x));
        },
        Return(x) => {
            full!(v.visit_expr_return(x));
        },
        Struct(x) => {
            full!(v.visit_expr_struct(x));
        },
        Try(x) => {
            full!(v.visit_expr_try(x));
        },
        TryBlock(x) => {
            full!(v.visit_expr_try_block(x));
        },
        Tuple(x) => {
            full!(v.visit_expr_tuple(x));
        },
        Unary(x) => {
            v.visit_expr_unary(x);
        },
        Unsafe(x) => {
            full!(v.visit_expr_unsafe(x));
        },
        Verbatim(x) => {
            skip!(x);
        },
        While(x) => {
            full!(v.visit_expr_while(x));
        },
        Yield(x) => {
            full!(v.visit_expr_yield(x));
        },
    }
}
pub fn visit_expr_array<'a, V>(v: &mut V, node: &'a expr::Array)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.bracket);
    for el in Puncted::pairs(&node.elems) {
        let x = el.value();
        v.visit_expr(x);
    }
}
pub fn visit_expr_assign<'a, V>(v: &mut V, node: &'a expr::Assign)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_expr(&*node.left);
    skip!(node.eq);
    v.visit_expr(&*node.right);
}
pub fn visit_expr_async<'a, V>(v: &mut V, node: &'a expr::Async)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.async_);
    skip!(node.capture);
    v.visit_block(&node.block);
}
pub fn visit_expr_await<'a, V>(v: &mut V, node: &'a expr::Await)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_expr(&*node.expr);
    skip!(node.dot);
    skip!(node.await_);
}
pub fn visit_expr_binary<'a, V>(v: &mut V, node: &'a expr::Binary)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_expr(&*node.left);
    v.visit_bin_op(&node.op);
    v.visit_expr(&*node.right);
}
pub fn visit_expr_block<'a, V>(v: &mut V, node: &'a expr::Block)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    if let Some(x) = &node.label {
        v.visit_label(x);
    }
    v.visit_block(&node.block);
}
pub fn visit_expr_break<'a, V>(v: &mut V, node: &'a expr::Break)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.break_);
    if let Some(x) = &node.life {
        v.visit_lifetime(x);
    }
    if let Some(x) = &node.val {
        v.visit_expr(&**x);
    }
}
pub fn visit_expr_call<'a, V>(v: &mut V, node: &'a expr::Call)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_expr(&*node.func);
    skip!(node.parenth);
    for el in Puncted::pairs(&node.args) {
        let x = el.value();
        v.visit_expr(x);
    }
}
pub fn visit_expr_cast<'a, V>(v: &mut V, node: &'a expr::Cast)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_expr(&*node.expr);
    skip!(node.as_);
    v.visit_type(&*node.typ);
}
pub fn visit_expr_closure<'a, V>(v: &mut V, node: &'a expr::Closure)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    if let Some(x) = &node.lifes {
        v.visit_bound_lifetimes(x);
    }
    skip!(node.const_);
    skip!(node.movability);
    skip!(node.asyncness);
    skip!(node.capture);
    skip!(node.or1);
    for el in Puncted::pairs(&node.inputs) {
        let x = el.value();
        v.visit_pat(x);
    }
    skip!(node.or2);
    v.visit_return_type(&node.ret);
    v.visit_expr(&*node.body);
}
pub fn visit_expr_const<'a, V>(v: &mut V, node: &'a expr::Const)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.const_);
    v.visit_block(&node.block);
}
pub fn visit_expr_continue<'a, V>(v: &mut V, node: &'a expr::Continue)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.continue_);
    if let Some(x) = &node.life {
        v.visit_lifetime(x);
    }
}
pub fn visit_expr_field<'a, V>(v: &mut V, node: &'a expr::Field)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_expr(&*node.expr);
    skip!(node.dot);
    v.visit_member(&node.memb);
}
pub fn visit_expr_for_loop<'a, V>(v: &mut V, node: &'a expr::ForLoop)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    if let Some(x) = &node.label {
        v.visit_label(x);
    }
    skip!(node.for_);
    v.visit_pat(&*node.pat);
    skip!(node.in_);
    v.visit_expr(&*node.expr);
    v.visit_block(&node.body);
}
pub fn visit_expr_group<'a, V>(v: &mut V, node: &'a expr::Group)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.group);
    v.visit_expr(&*node.expr);
}
pub fn visit_expr_if<'a, V>(v: &mut V, node: &'a expr::If)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.if_);
    v.visit_expr(&*node.cond);
    v.visit_block(&node.then_branch);
    if let Some(x) = &node.else_branch {
        skip!((x).0);
        v.visit_expr(&*(x).1);
    }
}
pub fn visit_expr_index<'a, V>(v: &mut V, node: &'a expr::Index)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_expr(&*node.expr);
    skip!(node.bracket);
    v.visit_expr(&*node.index);
}
pub fn visit_expr_infer<'a, V>(v: &mut V, node: &'a expr::Infer)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.underscore);
}
pub fn visit_expr_let<'a, V>(v: &mut V, node: &'a expr::Let)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.let_);
    v.visit_pat(&*node.pat);
    skip!(node.eq);
    v.visit_expr(&*node.expr);
}
pub fn visit_expr_lit<'a, V>(v: &mut V, node: &'a expr::Lit)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_lit(&node.lit);
}
pub fn visit_expr_loop<'a, V>(v: &mut V, node: &'a expr::Loop)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    if let Some(x) = &node.label {
        v.visit_label(x);
    }
    skip!(node.loop_);
    v.visit_block(&node.body);
}
pub fn visit_expr_macro<'a, V>(v: &mut V, node: &'a expr::Mac)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_macro(&node.mac);
}
pub fn visit_expr_match<'a, V>(v: &mut V, node: &'a expr::Match)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.match_);
    v.visit_expr(&*node.expr);
    skip!(node.brace);
    for x in &node.arms {
        v.visit_arm(x);
    }
}
pub fn visit_expr_method_call<'a, V>(v: &mut V, node: &'a expr::MethodCall)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_expr(&*node.expr);
    skip!(node.dot);
    v.visit_ident(&node.method);
    if let Some(x) = &node.turbofish {
        v.visit_angle_bracketed_generic_arguments(x);
    }
    skip!(node.parenth);
    for el in Puncted::pairs(&node.args) {
        let x = el.value();
        v.visit_expr(x);
    }
}
pub fn visit_expr_paren<'a, V>(v: &mut V, node: &'a expr::Parenth)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.parenth);
    v.visit_expr(&*node.expr);
}
pub fn visit_expr_path<'a, V>(v: &mut V, node: &'a expr::Path)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    if let Some(x) = &node.qself {
        v.visit_qself(x);
    }
    v.visit_path(&node.path);
}
pub fn visit_expr_range<'a, V>(v: &mut V, node: &'a expr::Range)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    if let Some(x) = &node.beg {
        v.visit_expr(&**x);
    }
    v.visit_range_limits(&node.limits);
    if let Some(x) = &node.end {
        v.visit_expr(&**x);
    }
}
pub fn visit_expr_reference<'a, V>(v: &mut V, node: &'a expr::Ref)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.and);
    skip!(node.mut_);
    v.visit_expr(&*node.expr);
}
pub fn visit_expr_repeat<'a, V>(v: &mut V, node: &'a expr::Repeat)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.bracket);
    v.visit_expr(&*node.expr);
    skip!(node.semi);
    v.visit_expr(&*node.len);
}
pub fn visit_expr_return<'a, V>(v: &mut V, node: &'a expr::Return)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.return_);
    if let Some(x) = &node.expr {
        v.visit_expr(&**x);
    }
}
pub fn visit_expr_struct<'a, V>(v: &mut V, node: &'a expr::Struct)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    if let Some(x) = &node.qself {
        v.visit_qself(x);
    }
    v.visit_path(&node.path);
    skip!(node.brace);
    for el in Puncted::pairs(&node.fields) {
        let x = el.value();
        v.visit_field_value(x);
    }
    skip!(node.dot2);
    if let Some(x) = &node.rest {
        v.visit_expr(&**x);
    }
}
pub fn visit_expr_try<'a, V>(v: &mut V, node: &'a expr::Try)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_expr(&*node.expr);
    skip!(node.question);
}
pub fn visit_expr_try_block<'a, V>(v: &mut V, node: &'a expr::TryBlock)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.try_);
    v.visit_block(&node.block);
}
pub fn visit_expr_tuple<'a, V>(v: &mut V, node: &'a expr::Tuple)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.parenth);
    for el in Puncted::pairs(&node.elems) {
        let x = el.value();
        v.visit_expr(x);
    }
}
pub fn visit_expr_unary<'a, V>(v: &mut V, node: &'a expr::Unary)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_un_op(&node.op);
    v.visit_expr(&*node.expr);
}
pub fn visit_expr_unsafe<'a, V>(v: &mut V, node: &'a expr::Unsafe)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.unsafe_);
    v.visit_block(&node.block);
}
pub fn visit_expr_while<'a, V>(v: &mut V, node: &'a expr::While)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    if let Some(x) = &node.label {
        v.visit_label(x);
    }
    skip!(node.while_);
    v.visit_expr(&*node.cond);
    v.visit_block(&node.body);
}
pub fn visit_expr_yield<'a, V>(v: &mut V, node: &'a expr::Yield)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.yield_);
    if let Some(x) = &node.expr {
        v.visit_expr(&**x);
    }
}
pub fn visit_field<'a, V>(v: &mut V, node: &'a data::Field)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    v.visit_field_mutability(&node.mut_);
    if let Some(x) = &node.ident {
        v.visit_ident(x);
    }
    skip!(node.colon);
    v.visit_type(&node.typ);
}
pub fn visit_field_mutability<'a, V>(v: &mut V, node: &'a data::Mut)
where
    V: Visit<'a> + ?Sized,
{
    match node {
        data::Mut::None => {},
    }
}
pub fn visit_field_pat<'a, V>(v: &mut V, node: &'a pat::Field)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_member(&node.memb);
    skip!(node.colon);
    v.visit_pat(&*node.pat);
}
pub fn visit_field_value<'a, V>(v: &mut V, node: &'a expr::FieldValue)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_member(&node.member);
    skip!(node.colon);
    v.visit_expr(&node.expr);
}
pub fn visit_fields<'a, V>(v: &mut V, node: &'a data::Fields)
where
    V: Visit<'a> + ?Sized,
{
    match node {
        data::Fields::Named(x) => {
            v.visit_fields_named(x);
        },
        data::Fields::Unnamed(x) => {
            v.visit_fields_unnamed(x);
        },
        data::Fields::Unit => {},
    }
}
pub fn visit_fields_named<'a, V>(v: &mut V, node: &'a data::Named)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.brace);
    for el in Puncted::pairs(&node.fields) {
        let x = el.value();
        v.visit_field(x);
    }
}
pub fn visit_fields_unnamed<'a, V>(v: &mut V, node: &'a data::Unnamed)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.parenth);
    for el in Puncted::pairs(&node.fields) {
        let x = el.value();
        v.visit_field(x);
    }
}
pub fn visit_file<'a, V>(v: &mut V, node: &'a item::File)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.shebang);
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    for x in &node.items {
        v.visit_item(x);
    }
}
pub fn visit_fn_arg<'a, V>(v: &mut V, node: &'a item::FnArg)
where
    V: Visit<'a> + ?Sized,
{
    use item::FnArg::*;
    match node {
        Receiver(x) => {
            v.visit_receiver(x);
        },
        Type(x) => {
            v.visit_pat_type(x);
        },
    }
}
pub fn visit_foreign_item<'a, V>(v: &mut V, node: &'a item::foreign::Item)
where
    V: Visit<'a> + ?Sized,
{
    use item::foreign::Item::*;
    match node {
        Fn(x) => {
            v.visit_foreign_item_fn(x);
        },
        Static(x) => {
            v.visit_foreign_item_static(x);
        },
        Type(x) => {
            v.visit_foreign_item_type(x);
        },
        Mac(x) => {
            v.visit_foreign_item_macro(x);
        },
        Verbatim(x) => {
            skip!(x);
        },
    }
}
pub fn visit_foreign_item_fn<'a, V>(v: &mut V, node: &'a item::foreign::Fn)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    v.visit_signature(&node.sig);
    skip!(node.semi);
}
pub fn visit_foreign_item_macro<'a, V>(v: &mut V, node: &'a item::foreign::Mac)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_macro(&node.mac);
    skip!(node.semi);
}
pub fn visit_foreign_item_static<'a, V>(v: &mut V, node: &'a item::foreign::Static)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.static_);
    v.visit_static_mutability(&node.mut_);
    v.visit_ident(&node.ident);
    skip!(node.colon);
    v.visit_type(&*node.typ);
    skip!(node.semi);
}
pub fn visit_foreign_item_type<'a, V>(v: &mut V, node: &'a item::foreign::Type)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.type);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.semi);
}
pub fn visit_generic_argument<'a, V>(v: &mut V, node: &'a path::Arg)
where
    V: Visit<'a> + ?Sized,
{
    use path::Arg::*;
    match node {
        Life(x) => {
            v.visit_lifetime(x);
        },
        Type(x) => {
            v.visit_type(x);
        },
        Const(x) => {
            v.visit_expr(x);
        },
        AssocType(x) => {
            v.visit_assoc_type(x);
        },
        AssocConst(x) => {
            v.visit_assoc_const(x);
        },
        Constraint(x) => {
            v.visit_constraint(x);
        },
    }
}
pub fn visit_generic_param<'a, V>(v: &mut V, node: &'a gen::Param)
where
    V: Visit<'a> + ?Sized,
{
    use gen::Param::*;
    match node {
        Life(x) => {
            v.visit_lifetime_param(x);
        },
        Type(x) => {
            v.visit_type_param(x);
        },
        Const(x) => {
            v.visit_const_param(x);
        },
    }
}
pub fn visit_generics<'a, V>(v: &mut V, node: &'a gen::Gens)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.lt);
    for el in Puncted::pairs(&node.params) {
        let x = el.value();
        v.visit_generic_param(x);
    }
    skip!(node.gt);
    if let Some(x) = &node.where_ {
        v.visit_where_clause(x);
    }
}
pub fn visit_ident<'a, V>(v: &mut V, node: &'a Ident)
where
    V: Visit<'a> + ?Sized,
{
    v.visit_span(&node.span());
}
pub fn visit_impl_item<'a, V>(v: &mut V, node: &'a item::impl_::Item)
where
    V: Visit<'a> + ?Sized,
{
    use item::impl_::Item::*;
    match node {
        Const(x) => {
            v.visit_impl_item_const(x);
        },
        Fn(x) => {
            v.visit_impl_item_fn(x);
        },
        Type(x) => {
            v.visit_impl_item_type(x);
        },
        Mac(x) => {
            v.visit_impl_item_macro(x);
        },
        Verbatim(x) => {
            skip!(x);
        },
    }
}
pub fn visit_impl_item_const<'a, V>(v: &mut V, node: &'a item::impl_::Const)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.defaultness);
    skip!(node.const_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.colon);
    v.visit_type(&node.typ);
    skip!(node.eq);
    v.visit_expr(&node.expr);
    skip!(node.semi);
}
pub fn visit_impl_item_fn<'a, V>(v: &mut V, node: &'a item::impl_::Fn)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.defaultness);
    v.visit_signature(&node.sig);
    v.visit_block(&node.block);
}
pub fn visit_impl_item_macro<'a, V>(v: &mut V, node: &'a item::impl_::Mac)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_macro(&node.mac);
    skip!(node.semi);
}
pub fn visit_impl_item_type<'a, V>(v: &mut V, node: &'a item::impl_::Type)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.defaultness);
    skip!(node.type);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.eq);
    v.visit_type(&node.typ);
    skip!(node.semi);
}
pub fn visit_impl_restriction<'a, V>(v: &mut V, node: &'a item::impl_::Restriction)
where
    V: Visit<'a> + ?Sized,
{
    match *node {}
}
pub fn visit_index<'a, V>(v: &mut V, node: &'a expr::Index)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.index);
    v.visit_span(&node.span);
}
pub fn visit_item<'a, V>(v: &mut V, node: &'a item::Item)
where
    V: Visit<'a> + ?Sized,
{
    use item::Item::*;
    match node {
        Const(x) => {
            v.visit_item_const(x);
        },
        Enum(x) => {
            v.visit_item_enum(x);
        },
        Extern(x) => {
            v.visit_item_extern_crate(x);
        },
        Fn(x) => {
            v.visit_item_fn(x);
        },
        Foreign(x) => {
            v.visit_item_foreign_mod(x);
        },
        Impl(x) => {
            v.visit_item_impl(x);
        },
        Mac(x) => {
            v.visit_item_macro(x);
        },
        Mod(x) => {
            v.visit_item_mod(x);
        },
        Static(x) => {
            v.visit_item_static(x);
        },
        Struct(x) => {
            v.visit_item_struct(x);
        },
        Trait(x) => {
            v.visit_item_trait(x);
        },
        TraitAlias(x) => {
            v.visit_item_trait_alias(x);
        },
        Type(x) => {
            v.visit_item_type(x);
        },
        Union(x) => {
            v.visit_item_union(x);
        },
        Use(x) => {
            v.visit_item_use(x);
        },
        Verbatim(x) => {
            skip!(x);
        },
    }
}
pub fn visit_item_const<'a, V>(v: &mut V, node: &'a item::Const)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.const_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.colon);
    v.visit_type(&*node.typ);
    skip!(node.eq);
    v.visit_expr(&*node.expr);
    skip!(node.semi);
}
pub fn visit_item_enum<'a, V>(v: &mut V, node: &'a item::Enum)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.enum_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.brace);
    for el in Puncted::pairs(&node.variants) {
        let x = el.value();
        v.visit_variant(x);
    }
}
pub fn visit_item_extern_crate<'a, V>(v: &mut V, node: &'a item::Extern)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.extern_);
    skip!(node.crate_);
    v.visit_ident(&node.ident);
    if let Some(x) = &node.rename {
        skip!((x).0);
        v.visit_ident(&(x).1);
    }
    skip!(node.semi);
}
pub fn visit_item_fn<'a, V>(v: &mut V, node: &'a item::Fn)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    v.visit_signature(&node.sig);
    v.visit_block(&*node.block);
}
pub fn visit_item_foreign_mod<'a, V>(v: &mut V, node: &'a item::Foreign)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.unsafe_);
    v.visit_abi(&node.abi);
    skip!(node.brace);
    for x in &node.items {
        v.visit_foreign_item(x);
    }
}
pub fn visit_item_impl<'a, V>(v: &mut V, node: &'a item::Impl)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.defaultness);
    skip!(node.unsafe_);
    skip!(node.impl_);
    v.visit_generics(&node.gens);
    if let Some(x) = &node.trait_ {
        skip!((x).0);
        v.visit_path(&(x).1);
        skip!((x).2);
    }
    v.visit_type(&*node.typ);
    skip!(node.brace);
    for x in &node.items {
        v.visit_impl_item(x);
    }
}
pub fn visit_item_macro<'a, V>(v: &mut V, node: &'a item::Mac)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    if let Some(x) = &node.ident {
        v.visit_ident(x);
    }
    v.visit_macro(&node.mac);
    skip!(node.semi);
}
pub fn visit_item_mod<'a, V>(v: &mut V, node: &'a item::Mod)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.unsafe_);
    skip!(node.mod_);
    v.visit_ident(&node.ident);
    if let Some(x) = &node.items {
        skip!((x).0);
        for x in &(x).1 {
            v.visit_item(x);
        }
    }
    skip!(node.semi);
}
pub fn visit_item_static<'a, V>(v: &mut V, node: &'a item::Static)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.static_);
    v.visit_static_mutability(&node.mut_);
    v.visit_ident(&node.ident);
    skip!(node.colon);
    v.visit_type(&*node.typ);
    skip!(node.eq);
    v.visit_expr(&*node.expr);
    skip!(node.semi);
}
pub fn visit_item_struct<'a, V>(v: &mut V, node: &'a item::Struct)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.struct_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    v.visit_fields(&node.fields);
    skip!(node.semi);
}
pub fn visit_item_trait<'a, V>(v: &mut V, node: &'a item::Trait)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.unsafe_);
    skip!(node.auto_);
    if let Some(x) = &node.restriction {
        v.visit_impl_restriction(x);
    }
    skip!(node.trait_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.colon);
    for el in Puncted::pairs(&node.supers) {
        let x = el.value();
        v.visit_type_param_bound(x);
    }
    skip!(node.brace);
    for x in &node.items {
        v.visit_trait_item(x);
    }
}
pub fn visit_item_trait_alias<'a, V>(v: &mut V, node: &'a item::TraitAlias)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.trait_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.eq);
    for el in Puncted::pairs(&node.bounds) {
        let x = el.value();
        v.visit_type_param_bound(x);
    }
    skip!(node.semi);
}
pub fn visit_item_type<'a, V>(v: &mut V, node: &'a item::Type)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.type);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.eq);
    v.visit_type(&*node.typ);
    skip!(node.semi);
}
pub fn visit_item_union<'a, V>(v: &mut V, node: &'a item::Union)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.union_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    v.visit_fields_named(&node.fields);
}
pub fn visit_item_use<'a, V>(v: &mut V, node: &'a item::Use)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_visibility(&node.vis);
    skip!(node.use_);
    skip!(node.colon);
    v.visit_use_tree(&node.tree);
    skip!(node.semi);
}
pub fn visit_label<'a, V>(v: &mut V, node: &'a expr::Label)
where
    V: Visit<'a> + ?Sized,
{
    v.visit_lifetime(&node.name);
    skip!(node.colon);
}
pub fn visit_lifetime<'a, V>(v: &mut V, node: &'a Life)
where
    V: Visit<'a> + ?Sized,
{
    v.visit_span(&node.apos);
    v.visit_ident(&node.ident);
}
pub fn visit_lifetime_param<'a, V>(v: &mut V, node: &'a gen::param::Life)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_lifetime(&node.life);
    skip!(node.colon);
    for el in Puncted::pairs(&node.bounds) {
        let x = el.value();
        v.visit_lifetime(x);
    }
}
pub fn visit_lit<'a, V>(v: &mut V, node: &'a lit::Lit)
where
    V: Visit<'a> + ?Sized,
{
    use lit::Lit::*;
    match node {
        Str(x) => {
            v.visit_lit_str(x);
        },
        ByteStr(x) => {
            v.visit_lit_byte_str(x);
        },
        Byte(x) => {
            v.visit_lit_byte(x);
        },
        Char(x) => {
            v.visit_lit_char(x);
        },
        Int(x) => {
            v.visit_lit_int(x);
        },
        Float(x) => {
            v.visit_lit_float(x);
        },
        Bool(x) => {
            v.visit_lit_bool(x);
        },
        Verbatim(x) => {
            skip!(x);
        },
    }
}
pub fn visit_lit_bool<'a, V>(v: &mut V, node: &'a lit::Bool)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.value);
    v.visit_span(&node.span);
}
pub fn visit_lit_byte<'a, V>(v: &mut V, node: &'a lit::Byte)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_lit_byte_str<'a, V>(v: &mut V, node: &'a lit::ByteStr)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_lit_char<'a, V>(v: &mut V, node: &'a lit::Char)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_lit_float<'a, V>(v: &mut V, node: &'a lit::Float)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_lit_int<'a, V>(v: &mut V, node: &'a lit::Int)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_lit_str<'a, V>(v: &mut V, node: &'a lit::Str)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_local<'a, V>(v: &mut V, node: &'a stmt::Local)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.let_);
    v.visit_pat(&node.pat);
    if let Some(x) = &node.init {
        v.visit_local_init(x);
    }
    skip!(node.semi);
}
pub fn visit_local_init<'a, V>(v: &mut V, node: &'a stmt::Init)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.eq);
    v.visit_expr(&*node.expr);
    if let Some(x) = &node.diverge {
        skip!((x).0);
        v.visit_expr(&*(x).1);
    }
}
pub fn visit_macro<'a, V>(v: &mut V, node: &'a mac::Mac)
where
    V: Visit<'a> + ?Sized,
{
    v.visit_path(&node.path);
    skip!(node.bang);
    v.visit_macro_delimiter(&node.delim);
    skip!(node.tokens);
}
pub fn visit_macro_delimiter<'a, V>(v: &mut V, node: &'a tok::Delim)
where
    V: Visit<'a> + ?Sized,
{
    use tok::Delim::*;
    match node {
        Parenth(x) => {
            skip!(x);
        },
        Brace(x) => {
            skip!(x);
        },
        Bracket(x) => {
            skip!(x);
        },
    }
}
pub fn visit_member<'a, V>(v: &mut V, node: &'a expr::Member)
where
    V: Visit<'a> + ?Sized,
{
    use expr::Member::*;
    match node {
        Named(x) => {
            v.visit_ident(x);
        },
        Unnamed(x) => {
            v.visit_index(x);
        },
    }
}
pub fn visit_meta<'a, V>(v: &mut V, node: &'a attr::Meta)
where
    V: Visit<'a> + ?Sized,
{
    use attr::Meta::*;
    match node {
        Path(x) => {
            v.visit_path(x);
        },
        List(x) => {
            v.visit_meta_list(x);
        },
        NameValue(x) => {
            v.visit_meta_name_value(x);
        },
    }
}
pub fn visit_meta_list<'a, V>(v: &mut V, node: &'a attr::List)
where
    V: Visit<'a> + ?Sized,
{
    v.visit_path(&node.path);
    v.visit_macro_delimiter(&node.delim);
    skip!(node.tokens);
}
pub fn visit_meta_name_value<'a, V>(v: &mut V, node: &'a attr::NameValue)
where
    V: Visit<'a> + ?Sized,
{
    v.visit_path(&node.name);
    skip!(node.eq);
    v.visit_expr(&node.val);
}
pub fn visit_parenthesized_generic_arguments<'a, V>(v: &mut V, node: &'a path::Parenthed)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.parenth);
    for el in Puncted::pairs(&node.ins) {
        let x = el.value();
        v.visit_type(x);
    }
    v.visit_return_type(&node.out);
}
pub fn visit_pat<'a, V>(v: &mut V, node: &'a pat::Pat)
where
    V: Visit<'a> + ?Sized,
{
    use pat::Pat::*;
    match node {
        Const(x) => {
            v.visit_expr_const(x);
        },
        Ident(x) => {
            v.visit_pat_ident(x);
        },
        Lit(x) => {
            v.visit_expr_lit(x);
        },
        Mac(x) => {
            v.visit_expr_macro(x);
        },
        Or(x) => {
            v.visit_pat_or(x);
        },
        Parenth(x) => {
            v.visit_pat_paren(x);
        },
        Path(x) => {
            v.visit_expr_path(x);
        },
        Range(x) => {
            v.visit_expr_range(x);
        },
        Ref(x) => {
            v.visit_pat_reference(x);
        },
        Rest(x) => {
            v.visit_pat_rest(x);
        },
        Slice(x) => {
            v.visit_pat_slice(x);
        },
        Struct(x) => {
            v.visit_pat_struct(x);
        },
        Tuple(x) => {
            v.visit_pat_tuple(x);
        },
        TupleStruct(x) => {
            v.visit_pat_tuple_struct(x);
        },
        Type(x) => {
            v.visit_pat_type(x);
        },
        Verbatim(x) => {
            skip!(x);
        },
        Wild(x) => {
            v.visit_pat_wild(x);
        },
    }
}
pub fn visit_pat_ident<'a, V>(v: &mut V, node: &'a pat::Ident)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.by_ref);
    skip!(node.mut_);
    v.visit_ident(&node.ident);
    if let Some(x) = &node.sub {
        skip!((x).0);
        v.visit_pat(&*(x).1);
    }
}
pub fn visit_pat_or<'a, V>(v: &mut V, node: &'a pat::Or)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.leading_vert);
    for el in Puncted::pairs(&node.cases) {
        let x = el.value();
        v.visit_pat(x);
    }
}
pub fn visit_pat_paren<'a, V>(v: &mut V, node: &'a pat::Parenth)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.parenth);
    v.visit_pat(&*node.pat);
}
pub fn visit_pat_reference<'a, V>(v: &mut V, node: &'a pat::Ref)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.and);
    skip!(node.mut_);
    v.visit_pat(&*node.pat);
}
pub fn visit_pat_rest<'a, V>(v: &mut V, node: &'a pat::Rest)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.dot2);
}
pub fn visit_pat_slice<'a, V>(v: &mut V, node: &'a pat::Slice)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.bracket);
    for el in Puncted::pairs(&node.pats) {
        let x = el.value();
        v.visit_pat(x);
    }
}
pub fn visit_pat_struct<'a, V>(v: &mut V, node: &'a pat::Struct)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    if let Some(x) = &node.qself {
        v.visit_qself(x);
    }
    v.visit_path(&node.path);
    skip!(node.brace);
    for el in Puncted::pairs(&node.fields) {
        let x = el.value();
        v.visit_field_pat(x);
    }
    if let Some(x) = &node.rest {
        v.visit_pat_rest(x);
    }
}
pub fn visit_pat_tuple<'a, V>(v: &mut V, node: &'a pat::Tuple)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.parenth);
    for el in Puncted::pairs(&node.pats) {
        let x = el.value();
        v.visit_pat(x);
    }
}
pub fn visit_pat_tuple_struct<'a, V>(v: &mut V, node: &'a pat::TupleStruct)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    if let Some(x) = &node.qself {
        v.visit_qself(x);
    }
    v.visit_path(&node.path);
    skip!(node.parenth);
    for el in Puncted::pairs(&node.pats) {
        let x = el.value();
        v.visit_pat(x);
    }
}
pub fn visit_pat_type<'a, V>(v: &mut V, node: &'a pat::Type)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_pat(&*node.pat);
    skip!(node.colon);
    v.visit_type(&*node.typ);
}
pub fn visit_pat_wild<'a, V>(v: &mut V, node: &'a pat::Wild)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.underscore);
}
pub fn visit_path<'a, V>(v: &mut V, node: &'a Path)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.colon);
    for el in Puncted::pairs(&node.segs) {
        let x = el.value();
        v.visit_path_segment(x);
    }
}
pub fn visit_path_arguments<'a, V>(v: &mut V, node: &'a path::Args)
where
    V: Visit<'a> + ?Sized,
{
    use path::Args::*;
    match node {
        None => {},
        Angled(x) => {
            v.visit_angle_bracketed_generic_arguments(x);
        },
        Parenthed(x) => {
            v.visit_parenthesized_generic_arguments(x);
        },
    }
}
pub fn visit_path_segment<'a, V>(v: &mut V, node: &'a path::Segment)
where
    V: Visit<'a> + ?Sized,
{
    v.visit_ident(&node.ident);
    v.visit_path_arguments(&node.args);
}
pub fn visit_predicate_lifetime<'a, V>(v: &mut V, node: &'a gen::where_::Life)
where
    V: Visit<'a> + ?Sized,
{
    v.visit_lifetime(&node.life);
    skip!(node.colon);
    for el in Puncted::pairs(&node.bounds) {
        let x = el.value();
        v.visit_lifetime(x);
    }
}
pub fn visit_predicate_type<'a, V>(v: &mut V, node: &'a gen::where_::Type)
where
    V: Visit<'a> + ?Sized,
{
    if let Some(x) = &node.lifes {
        v.visit_bound_lifetimes(x);
    }
    v.visit_type(&node.bounded);
    skip!(node.colon);
    for el in Puncted::pairs(&node.bounds) {
        let x = el.value();
        v.visit_type_param_bound(x);
    }
}
pub fn visit_qself<'a, V>(v: &mut V, node: &'a path::QSelf)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.lt);
    v.visit_type(&*node.ty);
    skip!(node.position);
    skip!(node.as_);
    skip!(node.gt);
}
pub fn visit_range_limits<'a, V>(v: &mut V, node: &'a expr::Limits)
where
    V: Visit<'a> + ?Sized,
{
    use expr::Limits::*;
    match node {
        HalfOpen(x) => {
            skip!(x);
        },
        Closed(x) => {
            skip!(x);
        },
    }
}
pub fn visit_receiver<'a, V>(v: &mut V, node: &'a item::Receiver)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    if let Some(x) = &node.ref_ {
        skip!((x).0);
        if let Some(x) = &(x).1 {
            v.visit_lifetime(x);
        }
    }
    skip!(node.mut_);
    skip!(node.self_);
    skip!(node.colon);
    v.visit_type(&*node.typ);
}
pub fn visit_return_type<'a, V>(v: &mut V, node: &'a typ::Ret)
where
    V: Visit<'a> + ?Sized,
{
    use typ::Ret::*;
    match node {
        Default => {},
        Type(_binding_0, _binding_1) => {
            skip!(x);
            v.visit_type(&**_binding_1);
        },
    }
}
pub fn visit_signature<'a, V>(v: &mut V, node: &'a item::Sig)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.const_);
    skip!(node.asyncness);
    skip!(node.unsafe_);
    if let Some(x) = &node.abi {
        v.visit_abi(x);
    }
    skip!(node.fn_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.parenth);
    for el in Puncted::pairs(&node.args) {
        let x = el.value();
        v.visit_fn_arg(x);
    }
    if let Some(x) = &node.vari {
        v.visit_variadic(x);
    }
    v.visit_return_type(&node.ret);
}
pub fn visit_span<'a, V>(v: &mut V, node: &pm2::Span)
where
    V: Visit<'a> + ?Sized,
{
}
pub fn visit_static_mutability<'a, V>(v: &mut V, node: &'a item::StaticMut)
where
    V: Visit<'a> + ?Sized,
{
    use item::StaticMut::*;
    match node {
        Mut(x) => {
            skip!(x);
        },
        None => {},
    }
}
pub fn visit_stmt<'a, V>(v: &mut V, node: &'a stmt::Stmt)
where
    V: Visit<'a> + ?Sized,
{
    use stmt::Stmt::*;
    match node {
        Local(x) => {
            v.visit_local(x);
        },
        Item(x) => {
            v.visit_item(x);
        },
        Expr(x, y) => {
            v.visit_expr(x);
            skip!(y);
        },
        Mac(x) => {
            v.visit_stmt_macro(x);
        },
    }
}
pub fn visit_stmt_macro<'a, V>(v: &mut V, node: &'a stmt::Mac)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_macro(&node.mac);
    skip!(node.semi);
}
pub fn visit_trait_bound<'a, V>(v: &mut V, node: &'a gen::bound::Trait)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.parenth);
    v.visit_trait_bound_modifier(&node.modif);
    if let Some(x) = &node.lifes {
        v.visit_bound_lifetimes(x);
    }
    v.visit_path(&node.path);
}
pub fn visit_trait_bound_modifier<'a, V>(v: &mut V, node: &'a gen::bound::Modifier)
where
    V: Visit<'a> + ?Sized,
{
    use gen::bound::Modifier::*;
    match node {
        None => {},
        Maybe(x) => {
            skip!(x);
        },
    }
}
pub fn visit_trait_item<'a, V>(v: &mut V, node: &'a item::trait_::Item)
where
    V: Visit<'a> + ?Sized,
{
    use item::trait_::Item::*;
    match node {
        Const(x) => {
            v.visit_trait_item_const(x);
        },
        Fn(x) => {
            v.visit_trait_item_fn(x);
        },
        Type(x) => {
            v.visit_trait_item_type(x);
        },
        Mac(x) => {
            v.visit_trait_item_macro(x);
        },
        Verbatim(x) => {
            skip!(x);
        },
    }
}
pub fn visit_trait_item_const<'a, V>(v: &mut V, node: &'a item::trait_::Const)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.const_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.colon);
    v.visit_type(&node.typ);
    if let Some(x) = &node.default {
        skip!((x).0);
        v.visit_expr(&(x).1);
    }
    skip!(node.semi);
}
pub fn visit_trait_item_fn<'a, V>(v: &mut V, node: &'a item::trait_::Fn)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_signature(&node.sig);
    if let Some(x) = &node.default {
        v.visit_block(x);
    }
    skip!(node.semi);
}
pub fn visit_trait_item_macro<'a, V>(v: &mut V, node: &'a item::trait_::Mac)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_macro(&node.mac);
    skip!(node.semi);
}
pub fn visit_trait_item_type<'a, V>(v: &mut V, node: &'a item::trait_::Type)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    skip!(node.type);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.colon);
    for el in Puncted::pairs(&node.bounds) {
        let x = el.value();
        v.visit_type_param_bound(x);
    }
    if let Some(x) = &node.default {
        skip!((x).0);
        v.visit_type(&(x).1);
    }
    skip!(node.semi);
}
pub fn visit_type<'a, V>(v: &mut V, node: &'a typ::Type)
where
    V: Visit<'a> + ?Sized,
{
    use typ::Type::*;
    match node {
        Array(x) => {
            v.visit_type_array(x);
        },
        Fn(x) => {
            v.visit_type_bare_fn(x);
        },
        Group(x) => {
            v.visit_type_group(x);
        },
        Impl(x) => {
            v.visit_type_impl_trait(x);
        },
        Infer(x) => {
            v.visit_type_infer(x);
        },
        Mac(x) => {
            v.visit_type_macro(x);
        },
        Never(x) => {
            v.visit_type_never(x);
        },
        Parenth(x) => {
            v.visit_type_paren(x);
        },
        Path(x) => {
            v.visit_type_path(x);
        },
        Ptr(x) => {
            v.visit_type_ptr(x);
        },
        Ref(x) => {
            v.visit_type_reference(x);
        },
        Slice(x) => {
            v.visit_type_slice(x);
        },
        Trait(x) => {
            v.visit_type_trait_object(x);
        },
        Tuple(x) => {
            v.visit_type_tuple(x);
        },
        Verbatim(x) => {
            skip!(x);
        },
    }
}
pub fn visit_type_array<'a, V>(v: &mut V, node: &'a typ::Array)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.bracket);
    v.visit_type(&*node.elem);
    skip!(node.semi);
    v.visit_expr(&node.len);
}
pub fn visit_type_bare_fn<'a, V>(v: &mut V, node: &'a typ::Fn)
where
    V: Visit<'a> + ?Sized,
{
    if let Some(x) = &node.lifes {
        v.visit_bound_lifetimes(x);
    }
    skip!(node.unsafe_);
    if let Some(x) = &node.abi {
        v.visit_abi(x);
    }
    skip!(node.fn_);
    skip!(node.parenth);
    for el in Puncted::pairs(&node.args) {
        let x = el.value();
        v.visit_bare_fn_arg(x);
    }
    if let Some(x) = &node.vari {
        v.visit_bare_variadic(x);
    }
    v.visit_return_type(&node.ret);
}
pub fn visit_type_group<'a, V>(v: &mut V, node: &'a typ::Group)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.group);
    v.visit_type(&*node.elem);
}
pub fn visit_type_impl_trait<'a, V>(v: &mut V, node: &'a typ::Impl)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.impl_);
    for el in Puncted::pairs(&node.bounds) {
        let x = el.value();
        v.visit_type_param_bound(x);
    }
}
pub fn visit_type_infer<'a, V>(v: &mut V, node: &'a typ::Infer)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.underscore);
}
pub fn visit_type_macro<'a, V>(v: &mut V, node: &'a typ::Mac)
where
    V: Visit<'a> + ?Sized,
{
    v.visit_macro(&node.mac);
}
pub fn visit_type_never<'a, V>(v: &mut V, node: &'a typ::Never)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.bang);
}
pub fn visit_type_param<'a, V>(v: &mut V, node: &'a gen::param::Type)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_ident(&node.ident);
    skip!(node.colon);
    for el in Puncted::pairs(&node.bounds) {
        let x = el.value();
        v.visit_type_param_bound(x);
    }
    skip!(node.eq);
    if let Some(x) = &node.default {
        v.visit_type(x);
    }
}
pub fn visit_type_param_bound<'a, V>(v: &mut V, node: &'a gen::bound::Type)
where
    V: Visit<'a> + ?Sized,
{
    use gen::bound::Type::*;
    match node {
        Trait(x) => {
            v.visit_trait_bound(x);
        },
        Life(x) => {
            v.visit_lifetime(x);
        },
        Verbatim(x) => {
            skip!(x);
        },
    }
}
pub fn visit_type_paren<'a, V>(v: &mut V, node: &'a typ::Parenth)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.parenth);
    v.visit_type(&*node.elem);
}
pub fn visit_type_path<'a, V>(v: &mut V, node: &'a typ::Path)
where
    V: Visit<'a> + ?Sized,
{
    if let Some(x) = &node.qself {
        v.visit_qself(x);
    }
    v.visit_path(&node.path);
}
pub fn visit_type_ptr<'a, V>(v: &mut V, node: &'a typ::Ptr)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.star);
    skip!(node.const_);
    skip!(node.mut_);
    v.visit_type(&*node.elem);
}
pub fn visit_type_reference<'a, V>(v: &mut V, node: &'a typ::Ref)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.and);
    if let Some(x) = &node.life {
        v.visit_lifetime(x);
    }
    skip!(node.mut_);
    v.visit_type(&*node.elem);
}
pub fn visit_type_slice<'a, V>(v: &mut V, node: &'a typ::Slice)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.bracket);
    v.visit_type(&*node.elem);
}
pub fn visit_type_trait_object<'a, V>(v: &mut V, node: &'a typ::Trait)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.dyn_);
    for el in Puncted::pairs(&node.bounds) {
        let x = el.value();
        v.visit_type_param_bound(x);
    }
}
pub fn visit_type_tuple<'a, V>(v: &mut V, node: &'a typ::Tuple)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.parenth);
    for el in Puncted::pairs(&node.elems) {
        let x = el.value();
        v.visit_type(x);
    }
}
pub fn visit_un_op<'a, V>(v: &mut V, node: &'a expr::UnOp)
where
    V: Visit<'a> + ?Sized,
{
    use expr::UnOp::*;
    match node {
        Deref(x) => {
            skip!(x);
        },
        Not(x) => {
            skip!(x);
        },
        Neg(x) => {
            skip!(x);
        },
    }
}
pub fn visit_use_glob<'a, V>(v: &mut V, node: &'a item::use_::Glob)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.star);
}
pub fn visit_use_group<'a, V>(v: &mut V, node: &'a item::use_::Group)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.brace);
    for el in Puncted::pairs(&node.trees) {
        let x = el.value();
        v.visit_use_tree(x);
    }
}
pub fn visit_use_name<'a, V>(v: &mut V, node: &'a item::use_::Name)
where
    V: Visit<'a> + ?Sized,
{
    v.visit_ident(&node.ident);
}
pub fn visit_use_path<'a, V>(v: &mut V, node: &'a item::use_::Path)
where
    V: Visit<'a> + ?Sized,
{
    v.visit_ident(&node.ident);
    skip!(node.colon2);
    v.visit_use_tree(&*node.tree);
}
pub fn visit_use_rename<'a, V>(v: &mut V, node: &'a item::use_::Rename)
where
    V: Visit<'a> + ?Sized,
{
    v.visit_ident(&node.ident);
    skip!(node.as_);
    v.visit_ident(&node.rename);
}
pub fn visit_use_tree<'a, V>(v: &mut V, node: &'a item::use_::Tree)
where
    V: Visit<'a> + ?Sized,
{
    use item::use_::Tree::*;
    match node {
        Path(x) => {
            v.visit_use_path(x);
        },
        Name(x) => {
            v.visit_use_name(x);
        },
        Rename(x) => {
            v.visit_use_rename(x);
        },
        Glob(x) => {
            v.visit_use_glob(x);
        },
        Group(x) => {
            v.visit_use_group(x);
        },
    }
}
pub fn visit_variadic<'a, V>(v: &mut V, node: &'a item::Variadic)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    if let Some(x) = &node.pat {
        v.visit_pat(&*(x).0);
        skip!((x).1);
    }
    skip!(node.dots);
    skip!(node.comma);
}
pub fn visit_variant<'a, V>(v: &mut V, node: &'a data::Variant)
where
    V: Visit<'a> + ?Sized,
{
    for x in &node.attrs {
        v.visit_attribute(x);
    }
    v.visit_ident(&node.ident);
    v.visit_fields(&node.fields);
    if let Some(x) = &node.discrim {
        skip!((x).0);
        v.visit_expr(&(x).1);
    }
}
pub fn visit_vis_restricted<'a, V>(v: &mut V, node: &'a data::Restricted)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.pub_);
    skip!(node.parenth);
    skip!(node.in_);
    v.visit_path(&*node.path);
}
pub fn visit_visibility<'a, V>(v: &mut V, node: &'a data::Visibility)
where
    V: Visit<'a> + ?Sized,
{
    use data::Visibility::*;
    match node {
        Public(x) => {
            skip!(x);
        },
        Restricted(x) => {
            v.visit_vis_restricted(x);
        },
        Inherited => {},
    }
}
pub fn visit_where_clause<'a, V>(v: &mut V, node: &'a gen::Where)
where
    V: Visit<'a> + ?Sized,
{
    skip!(node.where_);
    for el in Puncted::pairs(&node.preds) {
        let x = el.value();
        v.visit_where_predicate(x);
    }
}
pub fn visit_where_predicate<'a, V>(v: &mut V, node: &'a gen::where_::Pred)
where
    V: Visit<'a> + ?Sized,
{
    use gen::where_::Pred::*;
    match node {
        Life(x) => {
            v.visit_predicate_lifetime(x);
        },
        Type(x) => {
            v.visit_predicate_type(x);
        },
    }
}
