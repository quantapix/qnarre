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

pub trait Visit<'ast> {
    fn visit_abi(&mut self, i: &'ast Abi) {
        visit_abi(self, i);
    }
    fn visit_angle_bracketed_generic_arguments(&mut self, i: &'ast AngledArgs) {
        visit_angle_bracketed_generic_arguments(self, i);
    }
    fn visit_arm(&mut self, i: &'ast Arm) {
        visit_arm(self, i);
    }
    fn visit_assoc_const(&mut self, i: &'ast AssocConst) {
        visit_assoc_const(self, i);
    }
    fn visit_assoc_type(&mut self, i: &'ast AssocType) {
        visit_assoc_type(self, i);
    }
    fn visit_attr_style(&mut self, i: &'ast attr::Style) {
        visit_attr_style(self, i);
    }
    fn visit_attribute(&mut self, i: &'ast attr::Attr) {
        visit_attribute(self, i);
    }
    fn visit_bare_fn_arg(&mut self, i: &'ast typ::FnArg) {
        visit_bare_fn_arg(self, i);
    }
    fn visit_bare_variadic(&mut self, i: &'ast typ::Variadic) {
        visit_bare_variadic(self, i);
    }
    fn visit_bin_op(&mut self, i: &'ast BinOp) {
        visit_bin_op(self, i);
    }
    fn visit_block(&mut self, i: &'ast Block) {
        visit_block(self, i);
    }
    fn visit_bound_lifetimes(&mut self, i: &'ast Bgen::bound::Lifes) {
        visit_bound_lifetimes(self, i);
    }
    fn visit_const_param(&mut self, i: &'ast gen::param::Const) {
        visit_const_param(self, i);
    }
    fn visit_constraint(&mut self, i: &'ast Constraint) {
        visit_constraint(self, i);
    }
    fn visit_data(&mut self, i: &'ast Data) {
        visit_data(self, i);
    }
    fn visit_data_enum(&mut self, i: &'ast data::Enum) {
        visit_data_enum(self, i);
    }
    fn visit_data_struct(&mut self, i: &'ast data::Struct) {
        visit_data_struct(self, i);
    }
    fn visit_data_union(&mut self, i: &'ast data::Union) {
        visit_data_union(self, i);
    }
    fn visit_derive_input(&mut self, i: &'ast Input) {
        visit_derive_input(self, i);
    }
    fn visit_expr(&mut self, i: &'ast Expr) {
        visit_expr(self, i);
    }
    fn visit_expr_array(&mut self, i: &'ast expr::Array) {
        visit_expr_array(self, i);
    }
    fn visit_expr_assign(&mut self, i: &'ast expr::Assign) {
        visit_expr_assign(self, i);
    }
    fn visit_expr_async(&mut self, i: &'ast expr::Async) {
        visit_expr_async(self, i);
    }
    fn visit_expr_await(&mut self, i: &'ast expr::Await) {
        visit_expr_await(self, i);
    }
    fn visit_expr_binary(&mut self, i: &'ast expr::Binary) {
        visit_expr_binary(self, i);
    }
    fn visit_expr_block(&mut self, i: &'ast expr::Block) {
        visit_expr_block(self, i);
    }
    fn visit_expr_break(&mut self, i: &'ast expr::Break) {
        visit_expr_break(self, i);
    }
    fn visit_expr_call(&mut self, i: &'ast expr::Call) {
        visit_expr_call(self, i);
    }
    fn visit_expr_cast(&mut self, i: &'ast expr::Cast) {
        visit_expr_cast(self, i);
    }
    fn visit_expr_closure(&mut self, i: &'ast expr::Closure) {
        visit_expr_closure(self, i);
    }
    fn visit_expr_const(&mut self, i: &'ast expr::Const) {
        visit_expr_const(self, i);
    }
    fn visit_expr_continue(&mut self, i: &'ast expr::Continue) {
        visit_expr_continue(self, i);
    }
    fn visit_expr_field(&mut self, i: &'ast expr::Field) {
        visit_expr_field(self, i);
    }
    fn visit_expr_for_loop(&mut self, i: &'ast expr::ForLoop) {
        visit_expr_for_loop(self, i);
    }
    fn visit_expr_group(&mut self, i: &'ast expr::Group) {
        visit_expr_group(self, i);
    }
    fn visit_expr_if(&mut self, i: &'ast expr::If) {
        visit_expr_if(self, i);
    }
    fn visit_expr_index(&mut self, i: &'ast expr::Index) {
        visit_expr_index(self, i);
    }
    fn visit_expr_infer(&mut self, i: &'ast expr::Infer) {
        visit_expr_infer(self, i);
    }
    fn visit_expr_let(&mut self, i: &'ast expr::Let) {
        visit_expr_let(self, i);
    }
    fn visit_expr_lit(&mut self, i: &'ast expr::Lit) {
        visit_expr_lit(self, i);
    }
    fn visit_expr_loop(&mut self, i: &'ast expr::Loopop) {
        visit_expr_loop(self, i);
    }
    fn visit_expr_macro(&mut self, i: &'ast expr::Mac) {
        visit_expr_macro(self, i);
    }
    fn visit_expr_match(&mut self, i: &'ast expr::Match) {
        visit_expr_match(self, i);
    }
    fn visit_expr_method_call(&mut self, i: &'ast expr::MethodCall) {
        visit_expr_method_call(self, i);
    }
    fn visit_expr_paren(&mut self, i: &'ast expr::Paren) {
        visit_expr_paren(self, i);
    }
    fn visit_expr_path(&mut self, i: &'ast expr::Path) {
        visit_expr_path(self, i);
    }
    fn visit_expr_range(&mut self, i: &'ast expr::Range) {
        visit_expr_range(self, i);
    }
    fn visit_expr_reference(&mut self, i: &'ast expr::Ref) {
        visit_expr_reference(self, i);
    }
    fn visit_expr_repeat(&mut self, i: &'ast expr::Repeat) {
        visit_expr_repeat(self, i);
    }
    fn visit_expr_return(&mut self, i: &'ast expr::Return) {
        visit_expr_return(self, i);
    }
    fn visit_expr_struct(&mut self, i: &'ast expr::ExprStructStruct) {
        visit_expr_struct(self, i);
    }
    fn visit_expr_try(&mut self, i: &'ast expr::Try) {
        visit_expr_try(self, i);
    }
    fn visit_expr_try_block(&mut self, i: &'ast expr::TryBlock) {
        visit_expr_try_block(self, i);
    }
    fn visit_expr_tuple(&mut self, i: &'ast expr::Tuple) {
        visit_expr_tuple(self, i);
    }
    fn visit_expr_unary(&mut self, i: &'ast expr::Unary) {
        visit_expr_unary(self, i);
    }
    fn visit_expr_unsafe(&mut self, i: &'ast expr::Unsafe) {
        visit_expr_unsafe(self, i);
    }
    fn visit_expr_while(&mut self, i: &'ast expr::While) {
        visit_expr_while(self, i);
    }
    fn visit_expr_yield(&mut self, i: &'ast expr::Yield) {
        visit_expr_yield(self, i);
    }
    fn visit_field(&mut self, i: &'ast data::Field) {
        visit_field(self, i);
    }
    fn visit_field_mutability(&mut self, i: &'ast data::Mut) {
        visit_field_mutability(self, i);
    }
    fn visit_field_pat(&mut self, i: &'ast pat::Field) {
        visit_field_pat(self, i);
    }
    fn visit_field_value(&mut self, i: &'ast FieldValue) {
        visit_field_value(self, i);
    }
    fn visit_fields(&mut self, i: &'ast data::Fields) {
        visit_fields(self, i);
    }
    fn visit_fields_named(&mut self, i: &'ast data::Named) {
        visit_fields_named(self, i);
    }
    fn visit_fields_unnamed(&mut self, i: &'ast data::Unnamed) {
        visit_fields_unnamed(self, i);
    }
    fn visit_file(&mut self, i: &'ast item::File) {
        visit_file(self, i);
    }
    fn visit_fn_arg(&mut self, i: &'ast item::FnArg) {
        visit_fn_arg(self, i);
    }
    fn visit_foreign_item(&mut self, i: &'ast item::foreign::Item) {
        visit_foreign_item(self, i);
    }
    fn visit_foreign_item_fn(&mut self, i: &'ast item::foreign::Fn) {
        visit_foreign_item_fn(self, i);
    }
    fn visit_foreign_item_macro(&mut self, i: &'ast item::foreign::Mac) {
        visit_foreign_item_macro(self, i);
    }
    fn visit_foreign_item_static(&mut self, i: &'ast item::foreign::Static) {
        visit_foreign_item_static(self, i);
    }
    fn visit_foreign_item_type(&mut self, i: &'ast item::foreign::Type) {
        visit_foreign_item_type(self, i);
    }
    fn visit_generic_argument(&mut self, i: &'ast Arg) {
        visit_generic_argument(self, i);
    }
    fn visit_generic_param(&mut self, i: &'ast gen::Param) {
        visit_generic_param(self, i);
    }
    fn visit_generics(&mut self, i: &'ast gen::Gens) {
        visit_generics(self, i);
    }
    fn visit_ident(&mut self, i: &'ast Ident) {
        visit_ident(self, i);
    }
    fn visit_impl_item(&mut self, i: &'ast item::impl_::Item) {
        visit_impl_item(self, i);
    }
    fn visit_impl_item_const(&mut self, i: &'ast item::impl_::Const) {
        visit_impl_item_const(self, i);
    }
    fn visit_impl_item_fn(&mut self, i: &'ast item::impl_::Fn) {
        visit_impl_item_fn(self, i);
    }
    fn visit_impl_item_macro(&mut self, i: &'ast item::impl_::Mac) {
        visit_impl_item_macro(self, i);
    }
    fn visit_impl_item_type(&mut self, i: &'ast item::impl_::Type) {
        visit_impl_item_type(self, i);
    }
    fn visit_impl_restriction(&mut self, i: &'ast item::impl_::Restriction) {
        visit_impl_restriction(self, i);
    }
    fn visit_index(&mut self, i: &'ast Index) {
        visit_index(self, i);
    }
    fn visit_item(&mut self, i: &'ast Item) {
        visit_item(self, i);
    }
    fn visit_item_const(&mut self, i: &'ast item::Const) {
        visit_item_const(self, i);
    }
    fn visit_item_enum(&mut self, i: &'ast item::Enum) {
        visit_item_enum(self, i);
    }
    fn visit_item_extern_crate(&mut self, i: &'ast item::Extern) {
        visit_item_extern_crate(self, i);
    }
    fn visit_item_fn(&mut self, i: &'ast item::Fn) {
        visit_item_fn(self, i);
    }
    fn visit_item_foreign_mod(&mut self, i: &'ast item::Foreign) {
        visit_item_foreign_mod(self, i);
    }
    fn visit_item_impl(&mut self, i: &'ast item::Impl) {
        visit_item_impl(self, i);
    }
    fn visit_item_macro(&mut self, i: &'ast item::Mac) {
        visit_item_macro(self, i);
    }
    fn visit_item_mod(&mut self, i: &'ast item::Mod) {
        visit_item_mod(self, i);
    }
    fn visit_item_static(&mut self, i: &'ast item::Static) {
        visit_item_static(self, i);
    }
    fn visit_item_struct(&mut self, i: &'ast item::Struct) {
        visit_item_struct(self, i);
    }
    fn visit_item_trait(&mut self, i: &'ast item::Trait) {
        visit_item_trait(self, i);
    }
    fn visit_item_trait_alias(&mut self, i: &'ast item::TraitAlias) {
        visit_item_trait_alias(self, i);
    }
    fn visit_item_type(&mut self, i: &'ast item::Type) {
        visit_item_type(self, i);
    }
    fn visit_item_union(&mut self, i: &'ast item::Union) {
        visit_item_union(self, i);
    }
    fn visit_item_use(&mut self, i: &'ast item::Use) {
        visit_item_use(self, i);
    }
    fn visit_label(&mut self, i: &'ast Label) {
        visit_label(self, i);
    }
    fn visit_lifetime(&mut self, i: &'ast Life) {
        visit_lifetime(self, i);
    }
    fn visit_lifetime_param(&mut self, i: &'ast gen::param::Life) {
        visit_lifetime_param(self, i);
    }
    fn visit_lit(&mut self, i: &'ast Lit) {
        visit_lit(self, i);
    }
    fn visit_lit_bool(&mut self, i: &'ast lit::Bool) {
        visit_lit_bool(self, i);
    }
    fn visit_lit_byte(&mut self, i: &'ast lit::Byte) {
        visit_lit_byte(self, i);
    }
    fn visit_lit_byte_str(&mut self, i: &'ast lit::ByteStr) {
        visit_lit_byte_str(self, i);
    }
    fn visit_lit_char(&mut self, i: &'ast lit::Char) {
        visit_lit_char(self, i);
    }
    fn visit_lit_float(&mut self, i: &'ast lit::Float) {
        visit_lit_float(self, i);
    }
    fn visit_lit_int(&mut self, i: &'ast lit::Int) {
        visit_lit_int(self, i);
    }
    fn visit_lit_str(&mut self, i: &'ast lit::Str) {
        visit_lit_str(self, i);
    }
    fn visit_local(&mut self, i: &'ast stmt::Local) {
        visit_local(self, i);
    }
    fn visit_local_init(&mut self, i: &'ast stmt::Init) {
        visit_local_init(self, i);
    }
    fn visit_macro(&mut self, i: &'ast Macro) {
        visit_macro(self, i);
    }
    fn visit_macro_delimiter(&mut self, i: &'ast tok::Delim) {
        visit_macro_delimiter(self, i);
    }
    fn visit_member(&mut self, i: &'ast Member) {
        visit_member(self, i);
    }
    fn visit_meta(&mut self, i: &'ast attr::Meta) {
        visit_meta(self, i);
    }
    fn visit_meta_list(&mut self, i: &'ast attr::List) {
        visit_meta_list(self, i);
    }
    fn visit_meta_name_value(&mut self, i: &'ast attr::NameValue) {
        visit_meta_name_value(self, i);
    }
    fn visit_parenthesized_generic_arguments(&mut self, i: &'ast ParenthesizedArgs) {
        visit_parenthesized_generic_arguments(self, i);
    }
    fn visit_pat(&mut self, i: &'ast pat::Pat) {
        visit_pat(self, i);
    }
    fn visit_pat_ident(&mut self, i: &'ast pat::Ident) {
        visit_pat_ident(self, i);
    }
    fn visit_pat_or(&mut self, i: &'ast pat::Or) {
        visit_pat_or(self, i);
    }
    fn visit_pat_paren(&mut self, i: &'ast pat::Paren) {
        visit_pat_paren(self, i);
    }
    fn visit_pat_reference(&mut self, i: &'ast pat::Ref) {
        visit_pat_reference(self, i);
    }
    fn visit_pat_rest(&mut self, i: &'ast pat::Rest) {
        visit_pat_rest(self, i);
    }
    fn visit_pat_slice(&mut self, i: &'ast pat::Slice) {
        visit_pat_slice(self, i);
    }
    fn visit_pat_struct(&mut self, i: &'ast pat::Struct) {
        visit_pat_struct(self, i);
    }
    fn visit_pat_tuple(&mut self, i: &'ast pat::Tuple) {
        visit_pat_tuple(self, i);
    }
    fn visit_pat_tuple_struct(&mut self, i: &'ast pat::TupleStruct) {
        visit_pat_tuple_struct(self, i);
    }
    fn visit_pat_type(&mut self, i: &'ast pat::Type) {
        visit_pat_type(self, i);
    }
    fn visit_pat_wild(&mut self, i: &'ast pat::Wild) {
        visit_pat_wild(self, i);
    }
    fn visit_path(&mut self, i: &'ast Path) {
        visit_path(self, i);
    }
    fn visit_path_arguments(&mut self, i: &'ast Args) {
        visit_path_arguments(self, i);
    }
    fn visit_path_segment(&mut self, i: &'ast Segment) {
        visit_path_segment(self, i);
    }
    fn visit_predicate_lifetime(&mut self, i: &'ast gen::Where::Life) {
        visit_predicate_lifetime(self, i);
    }
    fn visit_predicate_type(&mut self, i: &'ast gen::Where::Type) {
        visit_predicate_type(self, i);
    }
    fn visit_qself(&mut self, i: &'ast QSelf) {
        visit_qself(self, i);
    }
    fn visit_range_limits(&mut self, i: &'ast expr::Limits) {
        visit_range_limits(self, i);
    }
    fn visit_receiver(&mut self, i: &'ast item::Receiver) {
        visit_receiver(self, i);
    }
    fn visit_return_type(&mut self, i: &'ast typ::Ret) {
        visit_return_type(self, i);
    }
    fn visit_signature(&mut self, i: &'ast item::Sig) {
        visit_signature(self, i);
    }
    fn visit_span(&mut self, i: &pm2::Span) {
        visit_span(self, i);
    }
    fn visit_static_mutability(&mut self, i: &'ast StaticMut) {
        visit_static_mutability(self, i);
    }
    fn visit_stmt(&mut self, i: &'ast stmt::Stmt) {
        visit_stmt(self, i);
    }
    fn visit_stmt_macro(&mut self, i: &'ast stmt::Mac) {
        visit_stmt_macro(self, i);
    }
    fn visit_trait_bound(&mut self, i: &'ast gen::bound::Trait) {
        visit_trait_bound(self, i);
    }
    fn visit_trait_bound_modifier(&mut self, i: &'ast gen::bound::Modifier) {
        visit_trait_bound_modifier(self, i);
    }
    fn visit_trait_item(&mut self, i: &'ast item::trait_::Item) {
        visit_trait_item(self, i);
    }
    fn visit_trait_item_const(&mut self, i: &'ast item::trait_::Const) {
        visit_trait_item_const(self, i);
    }
    fn visit_trait_item_fn(&mut self, i: &'ast item::trait_::Fn) {
        visit_trait_item_fn(self, i);
    }
    fn visit_trait_item_macro(&mut self, i: &'ast item::trait_::Mac) {
        visit_trait_item_macro(self, i);
    }
    fn visit_trait_item_type(&mut self, i: &'ast item::trait_::Type) {
        visit_trait_item_type(self, i);
    }
    fn visit_type(&mut self, i: &'ast typ::Type) {
        visit_type(self, i);
    }
    fn visit_type_array(&mut self, i: &'ast typ::Array) {
        visit_type_array(self, i);
    }
    fn visit_type_bare_fn(&mut self, i: &'ast typ::Fn) {
        visit_type_bare_fn(self, i);
    }
    fn visit_type_group(&mut self, i: &'ast typ::Group) {
        visit_type_group(self, i);
    }
    fn visit_type_impl_trait(&mut self, i: &'ast typ::Impl) {
        visit_type_impl_trait(self, i);
    }
    fn visit_type_infer(&mut self, i: &'ast typ::Infer) {
        visit_type_infer(self, i);
    }
    fn visit_type_macro(&mut self, i: &'ast typ::Mac) {
        visit_type_macro(self, i);
    }
    fn visit_type_never(&mut self, i: &'ast typ::Never) {
        visit_type_never(self, i);
    }
    fn visit_type_param(&mut self, i: &'ast gen::param::Type) {
        visit_type_param(self, i);
    }
    fn visit_type_param_bound(&mut self, i: &'ast gen::bound::Type) {
        visit_type_param_bound(self, i);
    }
    fn visit_type_paren(&mut self, i: &'ast typ::Paren) {
        visit_type_paren(self, i);
    }
    fn visit_type_path(&mut self, i: &'ast typ::Path) {
        visit_type_path(self, i);
    }
    fn visit_type_ptr(&mut self, i: &'ast typ::Ptr) {
        visit_type_ptr(self, i);
    }
    fn visit_type_reference(&mut self, i: &'ast typ::Ref) {
        visit_type_reference(self, i);
    }
    fn visit_type_slice(&mut self, i: &'ast typ::Slice) {
        visit_type_slice(self, i);
    }
    fn visit_type_trait_object(&mut self, i: &'ast typ::Trait) {
        visit_type_trait_object(self, i);
    }
    fn visit_type_tuple(&mut self, i: &'ast typ::Tuple) {
        visit_type_tuple(self, i);
    }
    fn visit_un_op(&mut self, i: &'ast UnOp) {
        visit_un_op(self, i);
    }
    fn visit_use_glob(&mut self, i: &'ast item::use_::Glob) {
        visit_use_glob(self, i);
    }
    fn visit_use_group(&mut self, i: &'ast item::use_::Group) {
        visit_use_group(self, i);
    }
    fn visit_use_name(&mut self, i: &'ast item::use_::Name) {
        visit_use_name(self, i);
    }
    fn visit_use_path(&mut self, i: &'ast item::use_::Path) {
        visit_use_path(self, i);
    }
    fn visit_use_rename(&mut self, i: &'ast item::use_::Rename) {
        visit_use_rename(self, i);
    }
    fn visit_use_tree(&mut self, i: &'ast item::use_::Tree) {
        visit_use_tree(self, i);
    }
    fn visit_variadic(&mut self, i: &'ast item::Variadic) {
        visit_variadic(self, i);
    }
    fn visit_variant(&mut self, i: &'ast data::Variant) {
        visit_variant(self, i);
    }
    fn visit_vis_restricted(&mut self, i: &'ast data::Restricted) {
        visit_vis_restricted(self, i);
    }
    fn visit_visibility(&mut self, i: &'ast data::Visibility) {
        visit_visibility(self, i);
    }
    fn visit_where_clause(&mut self, i: &'ast gen::Where) {
        visit_where_clause(self, i);
    }
    fn visit_where_predicate(&mut self, i: &'ast gen::Where::Pred) {
        visit_where_predicate(self, i);
    }
}
pub fn visit_abi<'ast, V>(v: &mut V, node: &'ast Abi)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.extern_);
    if let Some(it) = &node.name {
        v.visit_lit_str(it);
    }
}
pub fn visit_angle_bracketed_generic_arguments<'ast, V>(v: &mut V, node: &'ast AngledArgs)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.colon2);
    skip!(node.lt);
    for el in Puncted::pairs(&node.args) {
        let it = el.value();
        v.visit_generic_argument(it);
    }
    skip!(node.gt);
}
pub fn visit_arm<'ast, V>(v: &mut V, node: &'ast Arm)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_pat(&node.pat);
    if let Some(it) = &node.guard {
        skip!((it).0);
        v.visit_expr(&*(it).1);
    }
    skip!(node.fat_arrow);
    v.visit_expr(&*node.body);
    skip!(node.comma);
}
pub fn visit_assoc_const<'ast, V>(v: &mut V, node: &'ast AssocConst)
where
    V: Visit<'ast> + ?Sized,
{
    v.visit_ident(&node.ident);
    if let Some(it) = &node.gnrs {
        v.visit_angle_bracketed_generic_arguments(it);
    }
    skip!(node.eq);
    v.visit_expr(&node.val);
}
pub fn visit_assoc_type<'ast, V>(v: &mut V, node: &'ast AssocType)
where
    V: Visit<'ast> + ?Sized,
{
    v.visit_ident(&node.ident);
    if let Some(it) = &node.gnrs {
        v.visit_angle_bracketed_generic_arguments(it);
    }
    skip!(node.eq);
    v.visit_type(&node.ty);
}
pub fn visit_attr_style<'ast, V>(v: &mut V, node: &'ast attr::Style)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        attr::Style::Outer => {},
        attr::Style::Inner(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_attribute<'ast, V>(v: &mut V, node: &'ast attr::Attr)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.pound);
    v.visit_attr_style(&node.style);
    skip!(node.bracket);
    v.visit_meta(&node.meta);
}
pub fn visit_bare_fn_arg<'ast, V>(v: &mut V, node: &'ast typ::FnArg)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    if let Some(it) = &node.name {
        v.visit_ident(&(it).0);
        skip!((it).1);
    }
    v.visit_type(&node.typ);
}
pub fn visit_bare_variadic<'ast, V>(v: &mut V, node: &'ast typ::Variadic)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    if let Some(it) = &node.name {
        v.visit_ident(&(it).0);
        skip!((it).1);
    }
    skip!(node.dots);
    skip!(node.comma);
}
pub fn visit_bin_op<'ast, V>(v: &mut V, node: &'ast BinOp)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        BinOp::Add(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::Sub(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::Mul(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::Div(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::Rem(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::And(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::Or(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::BitXor(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::BitAnd(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::BitOr(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::Shl(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::Shr(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::Eq(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::Lt(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::Le(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::Ne(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::Ge(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::Gt(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::AddAssign(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::SubAssign(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::MulAssign(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::DivAssign(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::RemAssign(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::BitXorAssign(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::BitAndAssign(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::BitOrAssign(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::ShlAssign(_binding_0) => {
            skip!(_binding_0);
        },
        BinOp::ShrAssign(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_block<'ast, V>(v: &mut V, node: &'ast Block)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.brace);
    for it in &node.stmts {
        v.visit_stmt(it);
    }
}
pub fn visit_bound_lifetimes<'ast, V>(v: &mut V, node: &'ast Bgen::bound::Lifes)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.for_);
    skip!(node.lt);
    for el in Puncted::pairs(&node.lifes) {
        let it = el.value();
        v.visit_generic_param(it);
    }
    skip!(node.gt);
}
pub fn visit_const_param<'ast, V>(v: &mut V, node: &'ast gen::param::Const)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.const_);
    v.visit_ident(&node.ident);
    skip!(node.colon);
    v.visit_type(&node.typ);
    skip!(node.eq);
    if let Some(it) = &node.default {
        v.visit_expr(it);
    }
}
pub fn visit_constraint<'ast, V>(v: &mut V, node: &'ast Constraint)
where
    V: Visit<'ast> + ?Sized,
{
    v.visit_ident(&node.ident);
    if let Some(it) = &node.gnrs {
        v.visit_angle_bracketed_generic_arguments(it);
    }
    skip!(node.colon);
    for el in Puncted::pairs(&node.bounds) {
        let it = el.value();
        v.visit_type_param_bound(it);
    }
}
pub fn visit_data<'ast, V>(v: &mut V, node: &'ast Data)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        Data::Struct(_binding_0) => {
            v.visit_data_struct(_binding_0);
        },
        Data::Enum(_binding_0) => {
            v.visit_data_enum(_binding_0);
        },
        Data::Union(_binding_0) => {
            v.visit_data_union(_binding_0);
        },
    }
}
pub fn visit_data_enum<'ast, V>(v: &mut V, node: &'ast data::Enum)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.enum_);
    skip!(node.brace);
    for el in Puncted::pairs(&node.variants) {
        let it = el.value();
        v.visit_variant(it);
    }
}
pub fn visit_data_struct<'ast, V>(v: &mut V, node: &'ast data::Struct)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.struct_);
    v.visit_fields(&node.fields);
    skip!(node.semi);
}
pub fn visit_data_union<'ast, V>(v: &mut V, node: &'ast data::Union)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.union_);
    v.visit_fields_named(&node.fields);
}
pub fn visit_derive_input<'ast, V>(v: &mut V, node: &'ast Input)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    v.visit_data(&node.data);
}
pub fn visit_expr<'ast, V>(v: &mut V, node: &'ast Expr)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        Expr::Array(_binding_0) => {
            full!(v.visit_expr_array(_binding_0));
        },
        Expr::Assign(_binding_0) => {
            full!(v.visit_expr_assign(_binding_0));
        },
        Expr::Async(_binding_0) => {
            full!(v.visit_expr_async(_binding_0));
        },
        Expr::Await(_binding_0) => {
            full!(v.visit_expr_await(_binding_0));
        },
        Expr::Binary(_binding_0) => {
            v.visit_expr_binary(_binding_0);
        },
        Expr::Block(_binding_0) => {
            full!(v.visit_expr_block(_binding_0));
        },
        Expr::Break(_binding_0) => {
            full!(v.visit_expr_break(_binding_0));
        },
        Expr::Call(_binding_0) => {
            v.visit_expr_call(_binding_0);
        },
        Expr::Cast(_binding_0) => {
            v.visit_expr_cast(_binding_0);
        },
        Expr::Closure(_binding_0) => {
            full!(v.visit_expr_closure(_binding_0));
        },
        Expr::Const(_binding_0) => {
            full!(v.visit_expr_const(_binding_0));
        },
        Expr::Continue(_binding_0) => {
            full!(v.visit_expr_continue(_binding_0));
        },
        Expr::Field(_binding_0) => {
            v.visit_expr_field(_binding_0);
        },
        Expr::ForLoop(_binding_0) => {
            full!(v.visit_expr_for_loop(_binding_0));
        },
        Expr::Group(_binding_0) => {
            v.visit_expr_group(_binding_0);
        },
        Expr::If(_binding_0) => {
            full!(v.visit_expr_if(_binding_0));
        },
        Expr::Index(_binding_0) => {
            v.visit_expr_index(_binding_0);
        },
        Expr::Infer(_binding_0) => {
            full!(v.visit_expr_infer(_binding_0));
        },
        Expr::Let(_binding_0) => {
            full!(v.visit_expr_let(_binding_0));
        },
        Expr::Lit(_binding_0) => {
            v.visit_expr_lit(_binding_0);
        },
        Expr::Loop(_binding_0) => {
            full!(v.visit_expr_loop(_binding_0));
        },
        Expr::Macro(_binding_0) => {
            v.visit_expr_macro(_binding_0);
        },
        Expr::Match(_binding_0) => {
            full!(v.visit_expr_match(_binding_0));
        },
        Expr::MethodCall(_binding_0) => {
            full!(v.visit_expr_method_call(_binding_0));
        },
        Expr::Paren(_binding_0) => {
            v.visit_expr_paren(_binding_0);
        },
        Expr::Path(_binding_0) => {
            v.visit_expr_path(_binding_0);
        },
        Expr::Range(_binding_0) => {
            full!(v.visit_expr_range(_binding_0));
        },
        Expr::Reference(_binding_0) => {
            full!(v.visit_expr_reference(_binding_0));
        },
        Expr::Repeat(_binding_0) => {
            full!(v.visit_expr_repeat(_binding_0));
        },
        Expr::Return(_binding_0) => {
            full!(v.visit_expr_return(_binding_0));
        },
        Expr::Struct(_binding_0) => {
            full!(v.visit_expr_struct(_binding_0));
        },
        Expr::Try(_binding_0) => {
            full!(v.visit_expr_try(_binding_0));
        },
        Expr::TryBlock(_binding_0) => {
            full!(v.visit_expr_try_block(_binding_0));
        },
        Expr::Tuple(_binding_0) => {
            full!(v.visit_expr_tuple(_binding_0));
        },
        Expr::Unary(_binding_0) => {
            v.visit_expr_unary(_binding_0);
        },
        Expr::Unsafe(_binding_0) => {
            full!(v.visit_expr_unsafe(_binding_0));
        },
        Expr::Stream(_binding_0) => {
            skip!(_binding_0);
        },
        Expr::While(_binding_0) => {
            full!(v.visit_expr_while(_binding_0));
        },
        Expr::Yield(_binding_0) => {
            full!(v.visit_expr_yield(_binding_0));
        },
    }
}
pub fn visit_expr_array<'ast, V>(v: &mut V, node: &'ast expr::Array)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.bracket);
    for el in Puncted::pairs(&node.elems) {
        let it = el.value();
        v.visit_expr(it);
    }
}
pub fn visit_expr_assign<'ast, V>(v: &mut V, node: &'ast expr::Assign)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_expr(&*node.left);
    skip!(node.eq);
    v.visit_expr(&*node.right);
}
pub fn visit_expr_async<'ast, V>(v: &mut V, node: &'ast expr::Async)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.async_);
    skip!(node.capture);
    v.visit_block(&node.block);
}
pub fn visit_expr_await<'ast, V>(v: &mut V, node: &'ast expr::Await)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_expr(&*node.expr);
    skip!(node.dot);
    skip!(node.await_);
}
pub fn visit_expr_binary<'ast, V>(v: &mut V, node: &'ast expr::Binary)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_expr(&*node.left);
    v.visit_bin_op(&node.op);
    v.visit_expr(&*node.right);
}
pub fn visit_expr_block<'ast, V>(v: &mut V, node: &'ast expr::Block)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    if let Some(it) = &node.label {
        v.visit_label(it);
    }
    v.visit_block(&node.block);
}
pub fn visit_expr_break<'ast, V>(v: &mut V, node: &'ast expr::Break)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.break_);
    if let Some(it) = &node.life {
        v.visit_lifetime(it);
    }
    if let Some(it) = &node.val {
        v.visit_expr(&**it);
    }
}
pub fn visit_expr_call<'ast, V>(v: &mut V, node: &'ast expr::Call)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_expr(&*node.func);
    skip!(node.paren);
    for el in Puncted::pairs(&node.args) {
        let it = el.value();
        v.visit_expr(it);
    }
}
pub fn visit_expr_cast<'ast, V>(v: &mut V, node: &'ast expr::Cast)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_expr(&*node.expr);
    skip!(node.as_);
    v.visit_type(&*node.typ);
}
pub fn visit_expr_closure<'ast, V>(v: &mut V, node: &'ast expr::Closure)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    if let Some(it) = &node.lifes {
        v.visit_bound_lifetimes(it);
    }
    skip!(node.const_);
    skip!(node.movability);
    skip!(node.asyncness);
    skip!(node.capture);
    skip!(node.or1);
    for el in Puncted::pairs(&node.inputs) {
        let it = el.value();
        v.visit_pat(it);
    }
    skip!(node.or2);
    v.visit_return_type(&node.ret);
    v.visit_expr(&*node.body);
}
pub fn visit_expr_const<'ast, V>(v: &mut V, node: &'ast expr::Const)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.const_);
    v.visit_block(&node.block);
}
pub fn visit_expr_continue<'ast, V>(v: &mut V, node: &'ast expr::Continue)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.continue_);
    if let Some(it) = &node.life {
        v.visit_lifetime(it);
    }
}
pub fn visit_expr_field<'ast, V>(v: &mut V, node: &'ast expr::Field)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_expr(&*node.expr);
    skip!(node.dot);
    v.visit_member(&node.memb);
}
pub fn visit_expr_for_loop<'ast, V>(v: &mut V, node: &'ast expr::ForLoop)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    if let Some(it) = &node.label {
        v.visit_label(it);
    }
    skip!(node.for_);
    v.visit_pat(&*node.pat);
    skip!(node.in_);
    v.visit_expr(&*node.expr);
    v.visit_block(&node.body);
}
pub fn visit_expr_group<'ast, V>(v: &mut V, node: &'ast expr::Group)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.group);
    v.visit_expr(&*node.expr);
}
pub fn visit_expr_if<'ast, V>(v: &mut V, node: &'ast expr::If)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.if_);
    v.visit_expr(&*node.cond);
    v.visit_block(&node.then_branch);
    if let Some(it) = &node.else_branch {
        skip!((it).0);
        v.visit_expr(&*(it).1);
    }
}
pub fn visit_expr_index<'ast, V>(v: &mut V, node: &'ast expr::Index)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_expr(&*node.expr);
    skip!(node.bracket);
    v.visit_expr(&*node.index);
}
pub fn visit_expr_infer<'ast, V>(v: &mut V, node: &'ast expr::Infer)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.underscore);
}
pub fn visit_expr_let<'ast, V>(v: &mut V, node: &'ast expr::Let)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.let_);
    v.visit_pat(&*node.pat);
    skip!(node.eq);
    v.visit_expr(&*node.expr);
}
pub fn visit_expr_lit<'ast, V>(v: &mut V, node: &'ast expr::Lit)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_lit(&node.lit);
}
pub fn visit_expr_loop<'ast, V>(v: &mut V, node: &'ast expr::Loopop)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    if let Some(it) = &node.label {
        v.visit_label(it);
    }
    skip!(node.loop_);
    v.visit_block(&node.body);
}
pub fn visit_expr_macro<'ast, V>(v: &mut V, node: &'ast expr::Mac)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_macro(&node.mac);
}
pub fn visit_expr_match<'ast, V>(v: &mut V, node: &'ast expr::Match)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.match_);
    v.visit_expr(&*node.expr);
    skip!(node.brace);
    for it in &node.arms {
        v.visit_arm(it);
    }
}
pub fn visit_expr_method_call<'ast, V>(v: &mut V, node: &'ast expr::MethodCall)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_expr(&*node.expr);
    skip!(node.dot);
    v.visit_ident(&node.method);
    if let Some(it) = &node.turbofish {
        v.visit_angle_bracketed_generic_arguments(it);
    }
    skip!(node.paren);
    for el in Puncted::pairs(&node.args) {
        let it = el.value();
        v.visit_expr(it);
    }
}
pub fn visit_expr_paren<'ast, V>(v: &mut V, node: &'ast expr::Paren)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.paren);
    v.visit_expr(&*node.expr);
}
pub fn visit_expr_path<'ast, V>(v: &mut V, node: &'ast expr::Path)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    if let Some(it) = &node.qself {
        v.visit_qself(it);
    }
    v.visit_path(&node.path);
}
pub fn visit_expr_range<'ast, V>(v: &mut V, node: &'ast expr::Range)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    if let Some(it) = &node.beg {
        v.visit_expr(&**it);
    }
    v.visit_range_limits(&node.limits);
    if let Some(it) = &node.end {
        v.visit_expr(&**it);
    }
}
pub fn visit_expr_reference<'ast, V>(v: &mut V, node: &'ast expr::Ref)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.and);
    skip!(node.mut_);
    v.visit_expr(&*node.expr);
}
pub fn visit_expr_repeat<'ast, V>(v: &mut V, node: &'ast expr::Repeat)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.bracket);
    v.visit_expr(&*node.expr);
    skip!(node.semi);
    v.visit_expr(&*node.len);
}
pub fn visit_expr_return<'ast, V>(v: &mut V, node: &'ast expr::Return)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.return_);
    if let Some(it) = &node.expr {
        v.visit_expr(&**it);
    }
}
pub fn visit_expr_struct<'ast, V>(v: &mut V, node: &'ast expr::ExprStructStruct)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    if let Some(it) = &node.qself {
        v.visit_qself(it);
    }
    v.visit_path(&node.path);
    skip!(node.brace);
    for el in Puncted::pairs(&node.fields) {
        let it = el.value();
        v.visit_field_value(it);
    }
    skip!(node.dot2);
    if let Some(it) = &node.rest {
        v.visit_expr(&**it);
    }
}
pub fn visit_expr_try<'ast, V>(v: &mut V, node: &'ast expr::Try)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_expr(&*node.expr);
    skip!(node.question);
}
pub fn visit_expr_try_block<'ast, V>(v: &mut V, node: &'ast expr::TryBlock)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.try_);
    v.visit_block(&node.block);
}
pub fn visit_expr_tuple<'ast, V>(v: &mut V, node: &'ast expr::Tuple)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.paren);
    for el in Puncted::pairs(&node.elems) {
        let it = el.value();
        v.visit_expr(it);
    }
}
pub fn visit_expr_unary<'ast, V>(v: &mut V, node: &'ast expr::Unary)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_un_op(&node.op);
    v.visit_expr(&*node.expr);
}
pub fn visit_expr_unsafe<'ast, V>(v: &mut V, node: &'ast expr::Unsafe)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.unsafe_);
    v.visit_block(&node.block);
}
pub fn visit_expr_while<'ast, V>(v: &mut V, node: &'ast expr::While)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    if let Some(it) = &node.label {
        v.visit_label(it);
    }
    skip!(node.while_);
    v.visit_expr(&*node.cond);
    v.visit_block(&node.body);
}
pub fn visit_expr_yield<'ast, V>(v: &mut V, node: &'ast expr::Yield)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.yield_);
    if let Some(it) = &node.expr {
        v.visit_expr(&**it);
    }
}
pub fn visit_field<'ast, V>(v: &mut V, node: &'ast data::Field)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    v.visit_field_mutability(&node.mut_);
    if let Some(it) = &node.ident {
        v.visit_ident(it);
    }
    skip!(node.colon);
    v.visit_type(&node.typ);
}
pub fn visit_field_mutability<'ast, V>(v: &mut V, node: &'ast data::Mut)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        data::Mut::None => {},
    }
}
pub fn visit_field_pat<'ast, V>(v: &mut V, node: &'ast pat::Field)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_member(&node.memb);
    skip!(node.colon);
    v.visit_pat(&*node.pat);
}
pub fn visit_field_value<'ast, V>(v: &mut V, node: &'ast FieldValue)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_member(&node.member);
    skip!(node.colon);
    v.visit_expr(&node.expr);
}
pub fn visit_fields<'ast, V>(v: &mut V, node: &'ast data::Fields)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        data::Fields::Named(_binding_0) => {
            v.visit_fields_named(_binding_0);
        },
        data::Fields::Unnamed(_binding_0) => {
            v.visit_fields_unnamed(_binding_0);
        },
        data::Fields::Unit => {},
    }
}
pub fn visit_fields_named<'ast, V>(v: &mut V, node: &'ast data::Named)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.brace);
    for el in Puncted::pairs(&node.fields) {
        let it = el.value();
        v.visit_field(it);
    }
}
pub fn visit_fields_unnamed<'ast, V>(v: &mut V, node: &'ast data::Unnamed)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.paren);
    for el in Puncted::pairs(&node.fields) {
        let it = el.value();
        v.visit_field(it);
    }
}
pub fn visit_file<'ast, V>(v: &mut V, node: &'ast item::File)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.shebang);
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    for it in &node.items {
        v.visit_item(it);
    }
}
pub fn visit_fn_arg<'ast, V>(v: &mut V, node: &'ast item::FnArg)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        item::FnArg::Receiver(_binding_0) => {
            v.visit_receiver(_binding_0);
        },
        item::FnArg::Type(_binding_0) => {
            v.visit_pat_type(_binding_0);
        },
    }
}
pub fn visit_foreign_item<'ast, V>(v: &mut V, node: &'ast item::foreign::Item)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        item::foreign::Item::Fn(_binding_0) => {
            v.visit_foreign_item_fn(_binding_0);
        },
        item::foreign::Item::Static(_binding_0) => {
            v.visit_foreign_item_static(_binding_0);
        },
        item::foreign::Item::Type(_binding_0) => {
            v.visit_foreign_item_type(_binding_0);
        },
        item::foreign::Item::Macro(_binding_0) => {
            v.visit_foreign_item_macro(_binding_0);
        },
        item::foreign::Item::Stream(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_foreign_item_fn<'ast, V>(v: &mut V, node: &'ast item::foreign::Fn)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    v.visit_signature(&node.sig);
    skip!(node.semi);
}
pub fn visit_foreign_item_macro<'ast, V>(v: &mut V, node: &'ast item::foreign::Mac)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_macro(&node.mac);
    skip!(node.semi);
}
pub fn visit_foreign_item_static<'ast, V>(v: &mut V, node: &'ast item::foreign::Static)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    skip!(node.static_);
    v.visit_static_mutability(&node.mut_);
    v.visit_ident(&node.ident);
    skip!(node.colon);
    v.visit_type(&*node.typ);
    skip!(node.semi);
}
pub fn visit_foreign_item_type<'ast, V>(v: &mut V, node: &'ast item::foreign::Type)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    skip!(node.type);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.semi);
}
pub fn visit_generic_argument<'ast, V>(v: &mut V, node: &'ast Arg)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        Arg::Life(_binding_0) => {
            v.visit_lifetime(_binding_0);
        },
        Arg::Type(_binding_0) => {
            v.visit_type(_binding_0);
        },
        Arg::Const(_binding_0) => {
            v.visit_expr(_binding_0);
        },
        Arg::AssocType(_binding_0) => {
            v.visit_assoc_type(_binding_0);
        },
        Arg::AssocConst(_binding_0) => {
            v.visit_assoc_const(_binding_0);
        },
        Arg::Constraint(_binding_0) => {
            v.visit_constraint(_binding_0);
        },
    }
}
pub fn visit_generic_param<'ast, V>(v: &mut V, node: &'ast gen::Param)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        gen::Param::Life(_binding_0) => {
            v.visit_lifetime_param(_binding_0);
        },
        gen::Param::Type(_binding_0) => {
            v.visit_type_param(_binding_0);
        },
        gen::Param::Const(_binding_0) => {
            v.visit_const_param(_binding_0);
        },
    }
}
pub fn visit_generics<'ast, V>(v: &mut V, node: &'ast gen::Gens)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.lt);
    for el in Puncted::pairs(&node.params) {
        let it = el.value();
        v.visit_generic_param(it);
    }
    skip!(node.gt);
    if let Some(it) = &node.where_ {
        v.visit_where_clause(it);
    }
}
pub fn visit_ident<'ast, V>(v: &mut V, node: &'ast Ident)
where
    V: Visit<'ast> + ?Sized,
{
    v.visit_span(&node.span());
}
pub fn visit_impl_item<'ast, V>(v: &mut V, node: &'ast item::impl_::Item)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        item::impl_::Item::Const(_binding_0) => {
            v.visit_impl_item_const(_binding_0);
        },
        item::impl_::Item::Fn(_binding_0) => {
            v.visit_impl_item_fn(_binding_0);
        },
        item::impl_::Item::Type(_binding_0) => {
            v.visit_impl_item_type(_binding_0);
        },
        item::impl_::Item::Macro(_binding_0) => {
            v.visit_impl_item_macro(_binding_0);
        },
        item::impl_::Item::Stream(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_impl_item_const<'ast, V>(v: &mut V, node: &'ast item::impl_::Const)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
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
pub fn visit_impl_item_fn<'ast, V>(v: &mut V, node: &'ast item::impl_::Fn)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    skip!(node.defaultness);
    v.visit_signature(&node.sig);
    v.visit_block(&node.block);
}
pub fn visit_impl_item_macro<'ast, V>(v: &mut V, node: &'ast item::impl_::Mac)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_macro(&node.mac);
    skip!(node.semi);
}
pub fn visit_impl_item_type<'ast, V>(v: &mut V, node: &'ast item::impl_::Type)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
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
pub fn visit_impl_restriction<'ast, V>(v: &mut V, node: &'ast item::impl_::Restriction)
where
    V: Visit<'ast> + ?Sized,
{
    match *node {}
}
pub fn visit_index<'ast, V>(v: &mut V, node: &'ast Index)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.index);
    v.visit_span(&node.span);
}
pub fn visit_item<'ast, V>(v: &mut V, node: &'ast Item)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        Item::Const(_binding_0) => {
            v.visit_item_const(_binding_0);
        },
        Item::Enum(_binding_0) => {
            v.visit_item_enum(_binding_0);
        },
        Item::Extern(_binding_0) => {
            v.visit_item_extern_crate(_binding_0);
        },
        Item::Fn(_binding_0) => {
            v.visit_item_fn(_binding_0);
        },
        Item::Foreign(_binding_0) => {
            v.visit_item_foreign_mod(_binding_0);
        },
        Item::Impl(_binding_0) => {
            v.visit_item_impl(_binding_0);
        },
        Item::Macro(_binding_0) => {
            v.visit_item_macro(_binding_0);
        },
        Item::Mod(_binding_0) => {
            v.visit_item_mod(_binding_0);
        },
        Item::Static(_binding_0) => {
            v.visit_item_static(_binding_0);
        },
        Item::Struct(_binding_0) => {
            v.visit_item_struct(_binding_0);
        },
        Item::Trait(_binding_0) => {
            v.visit_item_trait(_binding_0);
        },
        Item::TraitAlias(_binding_0) => {
            v.visit_item_trait_alias(_binding_0);
        },
        Item::Type(_binding_0) => {
            v.visit_item_type(_binding_0);
        },
        Item::Union(_binding_0) => {
            v.visit_item_union(_binding_0);
        },
        Item::Use(_binding_0) => {
            v.visit_item_use(_binding_0);
        },
        Item::Stream(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_item_const<'ast, V>(v: &mut V, node: &'ast item::Const)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
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
pub fn visit_item_enum<'ast, V>(v: &mut V, node: &'ast item::Enum)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    skip!(node.enum_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.brace);
    for el in Puncted::pairs(&node.variants) {
        let it = el.value();
        v.visit_variant(it);
    }
}
pub fn visit_item_extern_crate<'ast, V>(v: &mut V, node: &'ast item::Extern)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    skip!(node.extern_);
    skip!(node.crate_);
    v.visit_ident(&node.ident);
    if let Some(it) = &node.rename {
        skip!((it).0);
        v.visit_ident(&(it).1);
    }
    skip!(node.semi);
}
pub fn visit_item_fn<'ast, V>(v: &mut V, node: &'ast item::Fn)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    v.visit_signature(&node.sig);
    v.visit_block(&*node.block);
}
pub fn visit_item_foreign_mod<'ast, V>(v: &mut V, node: &'ast item::Foreign)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.unsafe_);
    v.visit_abi(&node.abi);
    skip!(node.brace);
    for it in &node.items {
        v.visit_foreign_item(it);
    }
}
pub fn visit_item_impl<'ast, V>(v: &mut V, node: &'ast item::Impl)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.defaultness);
    skip!(node.unsafe_);
    skip!(node.impl_);
    v.visit_generics(&node.gens);
    if let Some(it) = &node.trait_ {
        skip!((it).0);
        v.visit_path(&(it).1);
        skip!((it).2);
    }
    v.visit_type(&*node.typ);
    skip!(node.brace);
    for it in &node.items {
        v.visit_impl_item(it);
    }
}
pub fn visit_item_macro<'ast, V>(v: &mut V, node: &'ast item::Mac)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    if let Some(it) = &node.ident {
        v.visit_ident(it);
    }
    v.visit_macro(&node.mac);
    skip!(node.semi);
}
pub fn visit_item_mod<'ast, V>(v: &mut V, node: &'ast item::Mod)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    skip!(node.unsafe_);
    skip!(node.mod_);
    v.visit_ident(&node.ident);
    if let Some(it) = &node.items {
        skip!((it).0);
        for it in &(it).1 {
            v.visit_item(it);
        }
    }
    skip!(node.semi);
}
pub fn visit_item_static<'ast, V>(v: &mut V, node: &'ast item::Static)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
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
pub fn visit_item_struct<'ast, V>(v: &mut V, node: &'ast item::Struct)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    skip!(node.struct_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    v.visit_fields(&node.fields);
    skip!(node.semi);
}
pub fn visit_item_trait<'ast, V>(v: &mut V, node: &'ast item::Trait)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    skip!(node.unsafe_);
    skip!(node.auto_);
    if let Some(it) = &node.restriction {
        v.visit_impl_restriction(it);
    }
    skip!(node.trait_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.colon);
    for el in Puncted::pairs(&node.supers) {
        let it = el.value();
        v.visit_type_param_bound(it);
    }
    skip!(node.brace);
    for it in &node.items {
        v.visit_trait_item(it);
    }
}
pub fn visit_item_trait_alias<'ast, V>(v: &mut V, node: &'ast item::TraitAlias)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    skip!(node.trait_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.eq);
    for el in Puncted::pairs(&node.bounds) {
        let it = el.value();
        v.visit_type_param_bound(it);
    }
    skip!(node.semi);
}
pub fn visit_item_type<'ast, V>(v: &mut V, node: &'ast item::Type)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    skip!(node.type);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.eq);
    v.visit_type(&*node.typ);
    skip!(node.semi);
}
pub fn visit_item_union<'ast, V>(v: &mut V, node: &'ast item::Union)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    skip!(node.union_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    v.visit_fields_named(&node.fields);
}
pub fn visit_item_use<'ast, V>(v: &mut V, node: &'ast item::Use)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_visibility(&node.vis);
    skip!(node.use_);
    skip!(node.leading_colon);
    v.visit_use_tree(&node.tree);
    skip!(node.semi);
}
pub fn visit_label<'ast, V>(v: &mut V, node: &'ast Label)
where
    V: Visit<'ast> + ?Sized,
{
    v.visit_lifetime(&node.name);
    skip!(node.colon);
}
pub fn visit_lifetime<'ast, V>(v: &mut V, node: &'ast Life)
where
    V: Visit<'ast> + ?Sized,
{
    v.visit_span(&node.apos);
    v.visit_ident(&node.ident);
}
pub fn visit_lifetime_param<'ast, V>(v: &mut V, node: &'ast gen::param::Life)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_lifetime(&node.life);
    skip!(node.colon);
    for el in Puncted::pairs(&node.bounds) {
        let it = el.value();
        v.visit_lifetime(it);
    }
}
pub fn visit_lit<'ast, V>(v: &mut V, node: &'ast Lit)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        Lit::Str(_binding_0) => {
            v.visit_lit_str(_binding_0);
        },
        Lit::ByteStr(_binding_0) => {
            v.visit_lit_byte_str(_binding_0);
        },
        Lit::Byte(_binding_0) => {
            v.visit_lit_byte(_binding_0);
        },
        Lit::Char(_binding_0) => {
            v.visit_lit_char(_binding_0);
        },
        Lit::Int(_binding_0) => {
            v.visit_lit_int(_binding_0);
        },
        Lit::Float(_binding_0) => {
            v.visit_lit_float(_binding_0);
        },
        Lit::Bool(_binding_0) => {
            v.visit_lit_bool(_binding_0);
        },
        Lit::Stream(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_lit_bool<'ast, V>(v: &mut V, node: &'ast lit::Bool)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.value);
    v.visit_span(&node.span);
}
pub fn visit_lit_byte<'ast, V>(v: &mut V, node: &'ast lit::Byte)
where
    V: Visit<'ast> + ?Sized,
{
}
pub fn visit_lit_byte_str<'ast, V>(v: &mut V, node: &'ast lit::ByteStr)
where
    V: Visit<'ast> + ?Sized,
{
}
pub fn visit_lit_char<'ast, V>(v: &mut V, node: &'ast lit::Char)
where
    V: Visit<'ast> + ?Sized,
{
}
pub fn visit_lit_float<'ast, V>(v: &mut V, node: &'ast lit::Float)
where
    V: Visit<'ast> + ?Sized,
{
}
pub fn visit_lit_int<'ast, V>(v: &mut V, node: &'ast lit::Int)
where
    V: Visit<'ast> + ?Sized,
{
}
pub fn visit_lit_str<'ast, V>(v: &mut V, node: &'ast lit::Str)
where
    V: Visit<'ast> + ?Sized,
{
}
pub fn visit_local<'ast, V>(v: &mut V, node: &'ast stmt::Local)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.let_);
    v.visit_pat(&node.pat);
    if let Some(it) = &node.init {
        v.visit_local_init(it);
    }
    skip!(node.semi);
}
pub fn visit_local_init<'ast, V>(v: &mut V, node: &'ast stmt::Init)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.eq);
    v.visit_expr(&*node.expr);
    if let Some(it) = &node.diverge {
        skip!((it).0);
        v.visit_expr(&*(it).1);
    }
}
pub fn visit_macro<'ast, V>(v: &mut V, node: &'ast Macro)
where
    V: Visit<'ast> + ?Sized,
{
    v.visit_path(&node.path);
    skip!(node.bang);
    v.visit_macro_delimiter(&node.delim);
    skip!(node.tokens);
}
pub fn visit_macro_delimiter<'ast, V>(v: &mut V, node: &'ast tok::Delim)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        tok::Delim::Paren(_binding_0) => {
            skip!(_binding_0);
        },
        tok::Delim::Brace(_binding_0) => {
            skip!(_binding_0);
        },
        tok::Delim::Bracket(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_member<'ast, V>(v: &mut V, node: &'ast Member)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        Member::Named(_binding_0) => {
            v.visit_ident(_binding_0);
        },
        Member::Unnamed(_binding_0) => {
            v.visit_index(_binding_0);
        },
    }
}
pub fn visit_meta<'ast, V>(v: &mut V, node: &'ast attr::Meta)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        attr::Meta::Path(_binding_0) => {
            v.visit_path(_binding_0);
        },
        attr::Meta::List(_binding_0) => {
            v.visit_meta_list(_binding_0);
        },
        attr::Meta::NameValue(_binding_0) => {
            v.visit_meta_name_value(_binding_0);
        },
    }
}
pub fn visit_meta_list<'ast, V>(v: &mut V, node: &'ast attr::List)
where
    V: Visit<'ast> + ?Sized,
{
    v.visit_path(&node.path);
    v.visit_macro_delimiter(&node.delim);
    skip!(node.tokens);
}
pub fn visit_meta_name_value<'ast, V>(v: &mut V, node: &'ast attr::NameValue)
where
    V: Visit<'ast> + ?Sized,
{
    v.visit_path(&node.name);
    skip!(node.eq);
    v.visit_expr(&node.val);
}
pub fn visit_parenthesized_generic_arguments<'ast, V>(v: &mut V, node: &'ast ParenthesizedArgs)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.paren);
    for el in Puncted::pairs(&node.ins) {
        let it = el.value();
        v.visit_type(it);
    }
    v.visit_return_type(&node.out);
}
pub fn visit_pat<'ast, V>(v: &mut V, node: &'ast pat::Pat)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        pat::Pat::Const(_binding_0) => {
            v.visit_expr_const(_binding_0);
        },
        pat::Pat::Ident(_binding_0) => {
            v.visit_pat_ident(_binding_0);
        },
        pat::Pat::Lit(_binding_0) => {
            v.visit_expr_lit(_binding_0);
        },
        pat::Pat::Mac(_binding_0) => {
            v.visit_expr_macro(_binding_0);
        },
        pat::Pat::Or(_binding_0) => {
            v.visit_pat_or(_binding_0);
        },
        pat::Pat::Paren(_binding_0) => {
            v.visit_pat_paren(_binding_0);
        },
        pat::Pat::Path(_binding_0) => {
            v.visit_expr_path(_binding_0);
        },
        pat::Pat::Range(_binding_0) => {
            v.visit_expr_range(_binding_0);
        },
        pat::Pat::Ref(_binding_0) => {
            v.visit_pat_reference(_binding_0);
        },
        pat::Pat::Rest(_binding_0) => {
            v.visit_pat_rest(_binding_0);
        },
        pat::Pat::Slice(_binding_0) => {
            v.visit_pat_slice(_binding_0);
        },
        pat::Pat::Struct(_binding_0) => {
            v.visit_pat_struct(_binding_0);
        },
        pat::Pat::Tuple(_binding_0) => {
            v.visit_pat_tuple(_binding_0);
        },
        pat::Pat::TupleStruct(_binding_0) => {
            v.visit_pat_tuple_struct(_binding_0);
        },
        pat::Pat::Type(_binding_0) => {
            v.visit_pat_type(_binding_0);
        },
        pat::Pat::Stream(_binding_0) => {
            skip!(_binding_0);
        },
        pat::Pat::Wild(_binding_0) => {
            v.visit_pat_wild(_binding_0);
        },
    }
}
pub fn visit_pat_ident<'ast, V>(v: &mut V, node: &'ast pat::Ident)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.by_ref);
    skip!(node.mut_);
    v.visit_ident(&node.ident);
    if let Some(it) = &node.sub {
        skip!((it).0);
        v.visit_pat(&*(it).1);
    }
}
pub fn visit_pat_or<'ast, V>(v: &mut V, node: &'ast pat::Or)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.leading_vert);
    for el in Puncted::pairs(&node.cases) {
        let it = el.value();
        v.visit_pat(it);
    }
}
pub fn visit_pat_paren<'ast, V>(v: &mut V, node: &'ast pat::Paren)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.paren);
    v.visit_pat(&*node.pat);
}
pub fn visit_pat_reference<'ast, V>(v: &mut V, node: &'ast pat::Ref)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.and);
    skip!(node.mut_);
    v.visit_pat(&*node.pat);
}
pub fn visit_pat_rest<'ast, V>(v: &mut V, node: &'ast pat::Rest)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.dot2);
}
pub fn visit_pat_slice<'ast, V>(v: &mut V, node: &'ast pat::Slice)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.bracket);
    for el in Puncted::pairs(&node.elems) {
        let it = el.value();
        v.visit_pat(it);
    }
}
pub fn visit_pat_struct<'ast, V>(v: &mut V, node: &'ast pat::Struct)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    if let Some(it) = &node.qself {
        v.visit_qself(it);
    }
    v.visit_path(&node.path);
    skip!(node.brace);
    for el in Puncted::pairs(&node.fields) {
        let it = el.value();
        v.visit_field_pat(it);
    }
    if let Some(it) = &node.rest {
        v.visit_pat_rest(it);
    }
}
pub fn visit_pat_tuple<'ast, V>(v: &mut V, node: &'ast pat::Tuple)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.paren);
    for el in Puncted::pairs(&node.elems) {
        let it = el.value();
        v.visit_pat(it);
    }
}
pub fn visit_pat_tuple_struct<'ast, V>(v: &mut V, node: &'ast pat::TupleStruct)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    if let Some(it) = &node.qself {
        v.visit_qself(it);
    }
    v.visit_path(&node.path);
    skip!(node.paren);
    for el in Puncted::pairs(&node.elems) {
        let it = el.value();
        v.visit_pat(it);
    }
}
pub fn visit_pat_type<'ast, V>(v: &mut V, node: &'ast pat::Type)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_pat(&*node.pat);
    skip!(node.colon);
    v.visit_type(&*node.typ);
}
pub fn visit_pat_wild<'ast, V>(v: &mut V, node: &'ast pat::Wild)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.underscore);
}
pub fn visit_path<'ast, V>(v: &mut V, node: &'ast Path)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.leading_colon);
    for el in Puncted::pairs(&node.segs) {
        let it = el.value();
        v.visit_path_segment(it);
    }
}
pub fn visit_path_arguments<'ast, V>(v: &mut V, node: &'ast Args)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        Args::None => {},
        Args::Angled(_binding_0) => {
            v.visit_angle_bracketed_generic_arguments(_binding_0);
        },
        Args::Parenthesized(_binding_0) => {
            v.visit_parenthesized_generic_arguments(_binding_0);
        },
    }
}
pub fn visit_path_segment<'ast, V>(v: &mut V, node: &'ast Segment)
where
    V: Visit<'ast> + ?Sized,
{
    v.visit_ident(&node.ident);
    v.visit_path_arguments(&node.args);
}
pub fn visit_predicate_lifetime<'ast, V>(v: &mut V, node: &'ast gen::Where::Life)
where
    V: Visit<'ast> + ?Sized,
{
    v.visit_lifetime(&node.life);
    skip!(node.colon);
    for el in Puncted::pairs(&node.bounds) {
        let it = el.value();
        v.visit_lifetime(it);
    }
}
pub fn visit_predicate_type<'ast, V>(v: &mut V, node: &'ast gen::Where::Type)
where
    V: Visit<'ast> + ?Sized,
{
    if let Some(it) = &node.lifes {
        v.visit_bound_lifetimes(it);
    }
    v.visit_type(&node.bounded);
    skip!(node.colon);
    for el in Puncted::pairs(&node.bounds) {
        let it = el.value();
        v.visit_type_param_bound(it);
    }
}
pub fn visit_qself<'ast, V>(v: &mut V, node: &'ast QSelf)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.lt);
    v.visit_type(&*node.ty);
    skip!(node.position);
    skip!(node.as_);
    skip!(node.gt);
}
pub fn visit_range_limits<'ast, V>(v: &mut V, node: &'ast expr::Limits)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        expr::Limits::HalfOpen(_binding_0) => {
            skip!(_binding_0);
        },
        expr::Limits::Closed(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_receiver<'ast, V>(v: &mut V, node: &'ast item::Receiver)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    if let Some(it) = &node.ref_ {
        skip!((it).0);
        if let Some(it) = &(it).1 {
            v.visit_lifetime(it);
        }
    }
    skip!(node.mut_);
    skip!(node.self_);
    skip!(node.colon);
    v.visit_type(&*node.typ);
}
pub fn visit_return_type<'ast, V>(v: &mut V, node: &'ast typ::Ret)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        typ::Ret::Default => {},
        typ::Ret::Type(_binding_0, _binding_1) => {
            skip!(_binding_0);
            v.visit_type(&**_binding_1);
        },
    }
}
pub fn visit_signature<'ast, V>(v: &mut V, node: &'ast item::Sig)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.const_);
    skip!(node.asyncness);
    skip!(node.unsafe_);
    if let Some(it) = &node.abi {
        v.visit_abi(it);
    }
    skip!(node.fn_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.paren);
    for el in Puncted::pairs(&node.args) {
        let it = el.value();
        v.visit_fn_arg(it);
    }
    if let Some(it) = &node.vari {
        v.visit_variadic(it);
    }
    v.visit_return_type(&node.ret);
}
pub fn visit_span<'ast, V>(v: &mut V, node: &pm2::Span)
where
    V: Visit<'ast> + ?Sized,
{
}
pub fn visit_static_mutability<'ast, V>(v: &mut V, node: &'ast StaticMut)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        StaticMut::Mut(_binding_0) => {
            skip!(_binding_0);
        },
        StaticMut::None => {},
    }
}
pub fn visit_stmt<'ast, V>(v: &mut V, node: &'ast stmt::Stmt)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        stmt::Stmt::stmt::Local(_binding_0) => {
            v.visit_local(_binding_0);
        },
        stmt::Stmt::Item(_binding_0) => {
            v.visit_item(_binding_0);
        },
        stmt::Stmt::Expr(_binding_0, _binding_1) => {
            v.visit_expr(_binding_0);
            skip!(_binding_1);
        },
        stmt::Stmt::Mac(_binding_0) => {
            v.visit_stmt_macro(_binding_0);
        },
    }
}
pub fn visit_stmt_macro<'ast, V>(v: &mut V, node: &'ast stmt::Mac)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_macro(&node.mac);
    skip!(node.semi);
}
pub fn visit_trait_bound<'ast, V>(v: &mut V, node: &'ast gen::bound::Trait)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.paren);
    v.visit_trait_bound_modifier(&node.modif);
    if let Some(it) = &node.lifes {
        v.visit_bound_lifetimes(it);
    }
    v.visit_path(&node.path);
}
pub fn visit_trait_bound_modifier<'ast, V>(v: &mut V, node: &'ast gen::bound::Modifier)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        gen::bound::Modifier::None => {},
        gen::bound::Modifier::Maybe(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_trait_item<'ast, V>(v: &mut V, node: &'ast item::trait_::Item)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        item::trait_::Item::Const(_binding_0) => {
            v.visit_trait_item_const(_binding_0);
        },
        item::trait_::Item::Fn(_binding_0) => {
            v.visit_trait_item_fn(_binding_0);
        },
        item::trait_::Item::Type(_binding_0) => {
            v.visit_trait_item_type(_binding_0);
        },
        item::trait_::Item::Macro(_binding_0) => {
            v.visit_trait_item_macro(_binding_0);
        },
        item::trait_::Item::Stream(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_trait_item_const<'ast, V>(v: &mut V, node: &'ast item::trait_::Const)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.const_);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.colon);
    v.visit_type(&node.typ);
    if let Some(it) = &node.default {
        skip!((it).0);
        v.visit_expr(&(it).1);
    }
    skip!(node.semi);
}
pub fn visit_trait_item_fn<'ast, V>(v: &mut V, node: &'ast item::trait_::Fn)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_signature(&node.sig);
    if let Some(it) = &node.default {
        v.visit_block(it);
    }
    skip!(node.semi);
}
pub fn visit_trait_item_macro<'ast, V>(v: &mut V, node: &'ast item::trait_::Mac)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_macro(&node.mac);
    skip!(node.semi);
}
pub fn visit_trait_item_type<'ast, V>(v: &mut V, node: &'ast item::trait_::Type)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    skip!(node.type);
    v.visit_ident(&node.ident);
    v.visit_generics(&node.gens);
    skip!(node.colon);
    for el in Puncted::pairs(&node.bounds) {
        let it = el.value();
        v.visit_type_param_bound(it);
    }
    if let Some(it) = &node.default {
        skip!((it).0);
        v.visit_type(&(it).1);
    }
    skip!(node.semi);
}
pub fn visit_type<'ast, V>(v: &mut V, node: &'ast typ::Type)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        typ::Type::Array(_binding_0) => {
            v.visit_type_array(_binding_0);
        },
        typ::Type::Fn(_binding_0) => {
            v.visit_type_bare_fn(_binding_0);
        },
        typ::Type::Group(_binding_0) => {
            v.visit_type_group(_binding_0);
        },
        typ::Type::Impl(_binding_0) => {
            v.visit_type_impl_trait(_binding_0);
        },
        typ::Type::Infer(_binding_0) => {
            v.visit_type_infer(_binding_0);
        },
        typ::Type::Mac(_binding_0) => {
            v.visit_type_macro(_binding_0);
        },
        typ::Type::Never(_binding_0) => {
            v.visit_type_never(_binding_0);
        },
        typ::Type::Paren(_binding_0) => {
            v.visit_type_paren(_binding_0);
        },
        typ::Type::Path(_binding_0) => {
            v.visit_type_path(_binding_0);
        },
        typ::Type::Ptr(_binding_0) => {
            v.visit_type_ptr(_binding_0);
        },
        typ::Type::Ref(_binding_0) => {
            v.visit_type_reference(_binding_0);
        },
        typ::Type::Slice(_binding_0) => {
            v.visit_type_slice(_binding_0);
        },
        typ::Type::Trait(_binding_0) => {
            v.visit_type_trait_object(_binding_0);
        },
        typ::Type::Tuple(_binding_0) => {
            v.visit_type_tuple(_binding_0);
        },
        typ::Type::Stream(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_type_array<'ast, V>(v: &mut V, node: &'ast typ::Array)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.bracket);
    v.visit_type(&*node.elem);
    skip!(node.semi);
    v.visit_expr(&node.len);
}
pub fn visit_type_bare_fn<'ast, V>(v: &mut V, node: &'ast typ::Fn)
where
    V: Visit<'ast> + ?Sized,
{
    if let Some(it) = &node.lifes {
        v.visit_bound_lifetimes(it);
    }
    skip!(node.unsafe_);
    if let Some(it) = &node.abi {
        v.visit_abi(it);
    }
    skip!(node.fn_);
    skip!(node.paren);
    for el in Puncted::pairs(&node.args) {
        let it = el.value();
        v.visit_bare_fn_arg(it);
    }
    if let Some(it) = &node.vari {
        v.visit_bare_variadic(it);
    }
    v.visit_return_type(&node.ret);
}
pub fn visit_type_group<'ast, V>(v: &mut V, node: &'ast typ::Group)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.group);
    v.visit_type(&*node.elem);
}
pub fn visit_type_impl_trait<'ast, V>(v: &mut V, node: &'ast typ::Impl)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.impl_);
    for el in Puncted::pairs(&node.bounds) {
        let it = el.value();
        v.visit_type_param_bound(it);
    }
}
pub fn visit_type_infer<'ast, V>(v: &mut V, node: &'ast typ::Infer)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.underscore);
}
pub fn visit_type_macro<'ast, V>(v: &mut V, node: &'ast typ::Mac)
where
    V: Visit<'ast> + ?Sized,
{
    v.visit_macro(&node.mac);
}
pub fn visit_type_never<'ast, V>(v: &mut V, node: &'ast typ::Never)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.bang);
}
pub fn visit_type_param<'ast, V>(v: &mut V, node: &'ast gen::param::Type)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_ident(&node.ident);
    skip!(node.colon);
    for el in Puncted::pairs(&node.bounds) {
        let it = el.value();
        v.visit_type_param_bound(it);
    }
    skip!(node.eq);
    if let Some(it) = &node.default {
        v.visit_type(it);
    }
}
pub fn visit_type_param_bound<'ast, V>(v: &mut V, node: &'ast gen::bound::Type)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        gen::bound::Type::Trait(_binding_0) => {
            v.visit_trait_bound(_binding_0);
        },
        gen::bound::Type::Life(_binding_0) => {
            v.visit_lifetime(_binding_0);
        },
        gen::bound::Type::Verbatim(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_type_paren<'ast, V>(v: &mut V, node: &'ast typ::Paren)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.paren);
    v.visit_type(&*node.elem);
}
pub fn visit_type_path<'ast, V>(v: &mut V, node: &'ast typ::Path)
where
    V: Visit<'ast> + ?Sized,
{
    if let Some(it) = &node.qself {
        v.visit_qself(it);
    }
    v.visit_path(&node.path);
}
pub fn visit_type_ptr<'ast, V>(v: &mut V, node: &'ast typ::Ptr)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.star);
    skip!(node.const_);
    skip!(node.mut_);
    v.visit_type(&*node.elem);
}
pub fn visit_type_reference<'ast, V>(v: &mut V, node: &'ast typ::Ref)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.and);
    if let Some(it) = &node.life {
        v.visit_lifetime(it);
    }
    skip!(node.mut_);
    v.visit_type(&*node.elem);
}
pub fn visit_type_slice<'ast, V>(v: &mut V, node: &'ast typ::Slice)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.bracket);
    v.visit_type(&*node.elem);
}
pub fn visit_type_trait_object<'ast, V>(v: &mut V, node: &'ast typ::Trait)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.dyn_);
    for el in Puncted::pairs(&node.bounds) {
        let it = el.value();
        v.visit_type_param_bound(it);
    }
}
pub fn visit_type_tuple<'ast, V>(v: &mut V, node: &'ast typ::Tuple)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.paren);
    for el in Puncted::pairs(&node.elems) {
        let it = el.value();
        v.visit_type(it);
    }
}
pub fn visit_un_op<'ast, V>(v: &mut V, node: &'ast UnOp)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        UnOp::Deref(_binding_0) => {
            skip!(_binding_0);
        },
        UnOp::Not(_binding_0) => {
            skip!(_binding_0);
        },
        UnOp::Neg(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_use_glob<'ast, V>(v: &mut V, node: &'ast item::use_::Glob)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.star);
}
pub fn visit_use_group<'ast, V>(v: &mut V, node: &'ast item::use_::Group)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.brace);
    for el in Puncted::pairs(&node.trees) {
        let it = el.value();
        v.visit_use_tree(it);
    }
}
pub fn visit_use_name<'ast, V>(v: &mut V, node: &'ast item::use_::Name)
where
    V: Visit<'ast> + ?Sized,
{
    v.visit_ident(&node.ident);
}
pub fn visit_use_path<'ast, V>(v: &mut V, node: &'ast item::use_::Path)
where
    V: Visit<'ast> + ?Sized,
{
    v.visit_ident(&node.ident);
    skip!(node.colon2);
    v.visit_use_tree(&*node.tree);
}
pub fn visit_use_rename<'ast, V>(v: &mut V, node: &'ast item::use_::Rename)
where
    V: Visit<'ast> + ?Sized,
{
    v.visit_ident(&node.ident);
    skip!(node.as_);
    v.visit_ident(&node.rename);
}
pub fn visit_use_tree<'ast, V>(v: &mut V, node: &'ast item::use_::Tree)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        item::use_::Tree::Path(_binding_0) => {
            v.visit_use_path(_binding_0);
        },
        item::use_::Tree::Name(_binding_0) => {
            v.visit_use_name(_binding_0);
        },
        item::use_::Tree::Rename(_binding_0) => {
            v.visit_use_rename(_binding_0);
        },
        item::use_::Tree::Glob(_binding_0) => {
            v.visit_use_glob(_binding_0);
        },
        item::use_::Tree::Group(_binding_0) => {
            v.visit_use_group(_binding_0);
        },
    }
}
pub fn visit_variadic<'ast, V>(v: &mut V, node: &'ast item::Variadic)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    if let Some(it) = &node.pat {
        v.visit_pat(&*(it).0);
        skip!((it).1);
    }
    skip!(node.dots);
    skip!(node.comma);
}
pub fn visit_variant<'ast, V>(v: &mut V, node: &'ast data::Variant)
where
    V: Visit<'ast> + ?Sized,
{
    for it in &node.attrs {
        v.visit_attribute(it);
    }
    v.visit_ident(&node.ident);
    v.visit_fields(&node.fields);
    if let Some(it) = &node.discrim {
        skip!((it).0);
        v.visit_expr(&(it).1);
    }
}
pub fn visit_vis_restricted<'ast, V>(v: &mut V, node: &'ast data::Restricted)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.pub_);
    skip!(node.paren);
    skip!(node.in_);
    v.visit_path(&*node.path);
}
pub fn visit_visibility<'ast, V>(v: &mut V, node: &'ast data::Visibility)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        data::Visibility::Public(_binding_0) => {
            skip!(_binding_0);
        },
        data::Visibility::Restricted(_binding_0) => {
            v.visit_vis_restricted(_binding_0);
        },
        data::Visibility::Inherited => {},
    }
}
pub fn visit_where_clause<'ast, V>(v: &mut V, node: &'ast gen::Where)
where
    V: Visit<'ast> + ?Sized,
{
    skip!(node.where_);
    for el in Puncted::pairs(&node.preds) {
        let it = el.value();
        v.visit_where_predicate(it);
    }
}
pub fn visit_where_predicate<'ast, V>(v: &mut V, node: &'ast gen::Where::Pred)
where
    V: Visit<'ast> + ?Sized,
{
    match node {
        gen::Where::Pred::Life(_binding_0) => {
            v.visit_predicate_lifetime(_binding_0);
        },
        gen::Where::Pred::Type(_binding_0) => {
            v.visit_predicate_type(_binding_0);
        },
    }
}
