#![allow(unused_variables)]

use crate::punct::Punctuated;
use crate::*;
use proc_macro2::Span;
macro_rules! full {
    ($e:expr) => {
        $e
    };
}
#[cfg(all(feature = "derive", not(feature = "full")))]
macro_rules! full {
    ($e:expr) => {
        unreachable!()
    };
}
macro_rules! skip {
    ($($tt:tt)*) => {};
}
pub trait VisitMut {
    fn visit_abi_mut(&mut self, i: &mut Abi) {
        visit_abi_mut(self, i);
    }
    fn visit_angle_bracketed_generic_arguments_mut(&mut self, i: &mut AngledArgs) {
        visit_angle_bracketed_generic_arguments_mut(self, i);
    }
    fn visit_arm_mut(&mut self, i: &mut Arm) {
        visit_arm_mut(self, i);
    }
    fn visit_assoc_const_mut(&mut self, i: &mut AssocConst) {
        visit_assoc_const_mut(self, i);
    }
    fn visit_assoc_type_mut(&mut self, i: &mut AssocType) {
        visit_assoc_type_mut(self, i);
    }
    fn visit_attr_style_mut(&mut self, i: &mut AttrStyle) {
        visit_attr_style_mut(self, i);
    }
    fn visit_attribute_mut(&mut self, i: &mut Attribute) {
        visit_attribute_mut(self, i);
    }
    fn visit_bare_fn_arg_mut(&mut self, i: &mut ty::BareFnArg) {
        visit_bare_fn_arg_mut(self, i);
    }
    fn visit_bare_variadic_mut(&mut self, i: &mut ty::BareVari) {
        visit_bare_variadic_mut(self, i);
    }
    fn visit_bin_op_mut(&mut self, i: &mut BinOp) {
        visit_bin_op_mut(self, i);
    }
    fn visit_block_mut(&mut self, i: &mut Block) {
        visit_block_mut(self, i);
    }
    fn visit_bound_lifetimes_mut(&mut self, i: &mut BoundLifetimes) {
        visit_bound_lifetimes_mut(self, i);
    }
    fn visit_const_param_mut(&mut self, i: &mut ConstParam) {
        visit_const_param_mut(self, i);
    }
    fn visit_constraint_mut(&mut self, i: &mut Constraint) {
        visit_constraint_mut(self, i);
    }
    fn visit_data_mut(&mut self, i: &mut Data) {
        visit_data_mut(self, i);
    }
    fn visit_data_enum_mut(&mut self, i: &mut DataEnum) {
        visit_data_enum_mut(self, i);
    }
    fn visit_data_struct_mut(&mut self, i: &mut DataStruct) {
        visit_data_struct_mut(self, i);
    }
    fn visit_data_union_mut(&mut self, i: &mut DataUnion) {
        visit_data_union_mut(self, i);
    }
    fn visit_derive_input_mut(&mut self, i: &mut DeriveInput) {
        visit_derive_input_mut(self, i);
    }
    fn visit_expr_mut(&mut self, i: &mut Expr) {
        visit_expr_mut(self, i);
    }
    fn visit_expr_array_mut(&mut self, i: &mut ExprArray) {
        visit_expr_array_mut(self, i);
    }
    fn visit_expr_assign_mut(&mut self, i: &mut ExprAssign) {
        visit_expr_assign_mut(self, i);
    }
    fn visit_expr_async_mut(&mut self, i: &mut ExprAsync) {
        visit_expr_async_mut(self, i);
    }
    fn visit_expr_await_mut(&mut self, i: &mut ExprAwait) {
        visit_expr_await_mut(self, i);
    }
    fn visit_expr_binary_mut(&mut self, i: &mut ExprBinary) {
        visit_expr_binary_mut(self, i);
    }
    fn visit_expr_block_mut(&mut self, i: &mut ExprBlock) {
        visit_expr_block_mut(self, i);
    }
    fn visit_expr_break_mut(&mut self, i: &mut ExprBreak) {
        visit_expr_break_mut(self, i);
    }
    fn visit_expr_call_mut(&mut self, i: &mut ExprCall) {
        visit_expr_call_mut(self, i);
    }
    fn visit_expr_cast_mut(&mut self, i: &mut ExprCast) {
        visit_expr_cast_mut(self, i);
    }
    fn visit_expr_closure_mut(&mut self, i: &mut ExprClosure) {
        visit_expr_closure_mut(self, i);
    }
    fn visit_expr_const_mut(&mut self, i: &mut ExprConst) {
        visit_expr_const_mut(self, i);
    }
    fn visit_expr_continue_mut(&mut self, i: &mut ExprContinue) {
        visit_expr_continue_mut(self, i);
    }
    fn visit_expr_field_mut(&mut self, i: &mut ExprField) {
        visit_expr_field_mut(self, i);
    }
    fn visit_expr_for_loop_mut(&mut self, i: &mut ExprForLoop) {
        visit_expr_for_loop_mut(self, i);
    }
    fn visit_expr_group_mut(&mut self, i: &mut ExprGroup) {
        visit_expr_group_mut(self, i);
    }
    fn visit_expr_if_mut(&mut self, i: &mut ExprIf) {
        visit_expr_if_mut(self, i);
    }
    fn visit_expr_index_mut(&mut self, i: &mut ExprIndex) {
        visit_expr_index_mut(self, i);
    }
    fn visit_expr_infer_mut(&mut self, i: &mut ExprInfer) {
        visit_expr_infer_mut(self, i);
    }
    fn visit_expr_let_mut(&mut self, i: &mut ExprLet) {
        visit_expr_let_mut(self, i);
    }
    fn visit_expr_lit_mut(&mut self, i: &mut ExprLit) {
        visit_expr_lit_mut(self, i);
    }
    fn visit_expr_loop_mut(&mut self, i: &mut ExprLoop) {
        visit_expr_loop_mut(self, i);
    }
    fn visit_expr_macro_mut(&mut self, i: &mut ExprMacro) {
        visit_expr_macro_mut(self, i);
    }
    fn visit_expr_match_mut(&mut self, i: &mut ExprMatch) {
        visit_expr_match_mut(self, i);
    }
    fn visit_expr_method_call_mut(&mut self, i: &mut ExprMethodCall) {
        visit_expr_method_call_mut(self, i);
    }
    fn visit_expr_paren_mut(&mut self, i: &mut ExprParen) {
        visit_expr_paren_mut(self, i);
    }
    fn visit_expr_path_mut(&mut self, i: &mut ExprPath) {
        visit_expr_path_mut(self, i);
    }
    fn visit_expr_range_mut(&mut self, i: &mut ExprRange) {
        visit_expr_range_mut(self, i);
    }
    fn visit_expr_reference_mut(&mut self, i: &mut ExprReference) {
        visit_expr_reference_mut(self, i);
    }
    fn visit_expr_repeat_mut(&mut self, i: &mut ExprRepeat) {
        visit_expr_repeat_mut(self, i);
    }
    fn visit_expr_return_mut(&mut self, i: &mut ExprReturn) {
        visit_expr_return_mut(self, i);
    }
    fn visit_expr_struct_mut(&mut self, i: &mut ExprStruct) {
        visit_expr_struct_mut(self, i);
    }
    fn visit_expr_try_mut(&mut self, i: &mut ExprTry) {
        visit_expr_try_mut(self, i);
    }
    fn visit_expr_try_block_mut(&mut self, i: &mut ExprTryBlock) {
        visit_expr_try_block_mut(self, i);
    }
    fn visit_expr_tuple_mut(&mut self, i: &mut ExprTuple) {
        visit_expr_tuple_mut(self, i);
    }
    fn visit_expr_unary_mut(&mut self, i: &mut ExprUnary) {
        visit_expr_unary_mut(self, i);
    }
    fn visit_expr_unsafe_mut(&mut self, i: &mut ExprUnsafe) {
        visit_expr_unsafe_mut(self, i);
    }
    fn visit_expr_while_mut(&mut self, i: &mut ExprWhile) {
        visit_expr_while_mut(self, i);
    }
    fn visit_expr_yield_mut(&mut self, i: &mut ExprYield) {
        visit_expr_yield_mut(self, i);
    }
    fn visit_field_mut(&mut self, i: &mut Field) {
        visit_field_mut(self, i);
    }
    fn visit_field_mutability_mut(&mut self, i: &mut FieldMut) {
        visit_field_mutability_mut(self, i);
    }
    fn visit_field_pat_mut(&mut self, i: &mut patt::Field) {
        visit_field_pat_mut(self, i);
    }
    fn visit_field_value_mut(&mut self, i: &mut FieldValue) {
        visit_field_value_mut(self, i);
    }
    fn visit_fields_mut(&mut self, i: &mut Fields) {
        visit_fields_mut(self, i);
    }
    fn visit_fields_named_mut(&mut self, i: &mut FieldsNamed) {
        visit_fields_named_mut(self, i);
    }
    fn visit_fields_unnamed_mut(&mut self, i: &mut FieldsUnnamed) {
        visit_fields_unnamed_mut(self, i);
    }
    fn visit_file_mut(&mut self, i: &mut File) {
        visit_file_mut(self, i);
    }
    fn visit_fn_arg_mut(&mut self, i: &mut FnArg) {
        visit_fn_arg_mut(self, i);
    }
    fn visit_foreign_item_mut(&mut self, i: &mut ForeignItem) {
        visit_foreign_item_mut(self, i);
    }
    fn visit_foreign_item_fn_mut(&mut self, i: &mut ForeignItemFn) {
        visit_foreign_item_fn_mut(self, i);
    }
    fn visit_foreign_item_macro_mut(&mut self, i: &mut ForeignItemMacro) {
        visit_foreign_item_macro_mut(self, i);
    }
    fn visit_foreign_item_static_mut(&mut self, i: &mut ForeignItemStatic) {
        visit_foreign_item_static_mut(self, i);
    }
    fn visit_foreign_item_type_mut(&mut self, i: &mut ForeignItemType) {
        visit_foreign_item_type_mut(self, i);
    }
    fn visit_generic_argument_mut(&mut self, i: &mut Arg) {
        visit_generic_argument_mut(self, i);
    }
    fn visit_generic_param_mut(&mut self, i: &mut GenericParam) {
        visit_generic_param_mut(self, i);
    }
    fn visit_generics_mut(&mut self, i: &mut Generics) {
        visit_generics_mut(self, i);
    }
    fn visit_ident_mut(&mut self, i: &mut Ident) {
        visit_ident_mut(self, i);
    }
    fn visit_impl_item_mut(&mut self, i: &mut ImplItem) {
        visit_impl_item_mut(self, i);
    }
    fn visit_impl_item_const_mut(&mut self, i: &mut ImplItemConst) {
        visit_impl_item_const_mut(self, i);
    }
    fn visit_impl_item_fn_mut(&mut self, i: &mut ImplItemFn) {
        visit_impl_item_fn_mut(self, i);
    }
    fn visit_impl_item_macro_mut(&mut self, i: &mut ImplItemMacro) {
        visit_impl_item_macro_mut(self, i);
    }
    fn visit_impl_item_type_mut(&mut self, i: &mut ImplItemType) {
        visit_impl_item_type_mut(self, i);
    }
    fn visit_impl_restriction_mut(&mut self, i: &mut ImplRestriction) {
        visit_impl_restriction_mut(self, i);
    }
    fn visit_index_mut(&mut self, i: &mut Index) {
        visit_index_mut(self, i);
    }
    fn visit_item_mut(&mut self, i: &mut Item) {
        visit_item_mut(self, i);
    }
    fn visit_item_const_mut(&mut self, i: &mut ItemConst) {
        visit_item_const_mut(self, i);
    }
    fn visit_item_enum_mut(&mut self, i: &mut ItemEnum) {
        visit_item_enum_mut(self, i);
    }
    fn visit_item_extern_crate_mut(&mut self, i: &mut ItemExternCrate) {
        visit_item_extern_crate_mut(self, i);
    }
    fn visit_item_fn_mut(&mut self, i: &mut ItemFn) {
        visit_item_fn_mut(self, i);
    }
    fn visit_item_foreign_mod_mut(&mut self, i: &mut ItemForeignMod) {
        visit_item_foreign_mod_mut(self, i);
    }
    fn visit_item_impl_mut(&mut self, i: &mut ItemImpl) {
        visit_item_impl_mut(self, i);
    }
    fn visit_item_macro_mut(&mut self, i: &mut ItemMacro) {
        visit_item_macro_mut(self, i);
    }
    fn visit_item_mod_mut(&mut self, i: &mut ItemMod) {
        visit_item_mod_mut(self, i);
    }
    fn visit_item_static_mut(&mut self, i: &mut ItemStatic) {
        visit_item_static_mut(self, i);
    }
    fn visit_item_struct_mut(&mut self, i: &mut ItemStruct) {
        visit_item_struct_mut(self, i);
    }
    fn visit_item_trait_mut(&mut self, i: &mut ItemTrait) {
        visit_item_trait_mut(self, i);
    }
    fn visit_item_trait_alias_mut(&mut self, i: &mut ItemTraitAlias) {
        visit_item_trait_alias_mut(self, i);
    }
    fn visit_item_type_mut(&mut self, i: &mut ItemType) {
        visit_item_type_mut(self, i);
    }
    fn visit_item_union_mut(&mut self, i: &mut ItemUnion) {
        visit_item_union_mut(self, i);
    }
    fn visit_item_use_mut(&mut self, i: &mut ItemUse) {
        visit_item_use_mut(self, i);
    }
    fn visit_label_mut(&mut self, i: &mut Label) {
        visit_label_mut(self, i);
    }
    fn visit_lifetime_mut(&mut self, i: &mut Lifetime) {
        visit_lifetime_mut(self, i);
    }
    fn visit_lifetime_param_mut(&mut self, i: &mut LifetimeParam) {
        visit_lifetime_param_mut(self, i);
    }
    fn visit_lit_mut(&mut self, i: &mut Lit) {
        visit_lit_mut(self, i);
    }
    fn visit_lit_bool_mut(&mut self, i: &mut lit::Bool) {
        visit_lit_bool_mut(self, i);
    }
    fn visit_lit_byte_mut(&mut self, i: &mut lit::Byte) {
        visit_lit_byte_mut(self, i);
    }
    fn visit_lit_byte_str_mut(&mut self, i: &mut lit::ByteStr) {
        visit_lit_byte_str_mut(self, i);
    }
    fn visit_lit_char_mut(&mut self, i: &mut lit::Char) {
        visit_lit_char_mut(self, i);
    }
    fn visit_lit_float_mut(&mut self, i: &mut lit::Float) {
        visit_lit_float_mut(self, i);
    }
    fn visit_lit_int_mut(&mut self, i: &mut lit::Int) {
        visit_lit_int_mut(self, i);
    }
    fn visit_lit_str_mut(&mut self, i: &mut lit::Str) {
        visit_lit_str_mut(self, i);
    }
    fn visit_local_mut(&mut self, i: &mut Local) {
        visit_local_mut(self, i);
    }
    fn visit_local_init_mut(&mut self, i: &mut LocalInit) {
        visit_local_init_mut(self, i);
    }
    fn visit_macro_mut(&mut self, i: &mut Macro) {
        visit_macro_mut(self, i);
    }
    fn visit_macro_delimiter_mut(&mut self, i: &mut MacroDelim) {
        visit_macro_delimiter_mut(self, i);
    }
    fn visit_member_mut(&mut self, i: &mut Member) {
        visit_member_mut(self, i);
    }
    fn visit_meta_mut(&mut self, i: &mut Meta) {
        visit_meta_mut(self, i);
    }
    fn visit_meta_list_mut(&mut self, i: &mut MetaList) {
        visit_meta_list_mut(self, i);
    }
    fn visit_meta_name_value_mut(&mut self, i: &mut MetaNameValue) {
        visit_meta_name_value_mut(self, i);
    }
    fn visit_parenthesized_generic_arguments_mut(&mut self, i: &mut ParenthesizedArgs) {
        visit_parenthesized_generic_arguments_mut(self, i);
    }
    fn visit_pat_mut(&mut self, i: &mut patt::Patt) {
        visit_pat_mut(self, i);
    }
    fn visit_pat_ident_mut(&mut self, i: &mut patt::Ident) {
        visit_pat_ident_mut(self, i);
    }
    fn visit_pat_or_mut(&mut self, i: &mut patt::Or) {
        visit_pat_or_mut(self, i);
    }
    fn visit_pat_paren_mut(&mut self, i: &mut patt::Paren) {
        visit_pat_paren_mut(self, i);
    }
    fn visit_pat_reference_mut(&mut self, i: &mut patt::Ref) {
        visit_pat_reference_mut(self, i);
    }
    fn visit_pat_rest_mut(&mut self, i: &mut patt::Rest) {
        visit_pat_rest_mut(self, i);
    }
    fn visit_pat_slice_mut(&mut self, i: &mut patt::Slice) {
        visit_pat_slice_mut(self, i);
    }
    fn visit_pat_struct_mut(&mut self, i: &mut patt::Struct) {
        visit_pat_struct_mut(self, i);
    }
    fn visit_pat_tuple_mut(&mut self, i: &mut patt::Tuple) {
        visit_pat_tuple_mut(self, i);
    }
    fn visit_pat_tuple_struct_mut(&mut self, i: &mut patt::TupleStruct) {
        visit_pat_tuple_struct_mut(self, i);
    }
    fn visit_pat_type_mut(&mut self, i: &mut patt::Type) {
        visit_pat_type_mut(self, i);
    }
    fn visit_pat_wild_mut(&mut self, i: &mut patt::Wild) {
        visit_pat_wild_mut(self, i);
    }
    fn visit_path_mut(&mut self, i: &mut Path) {
        visit_path_mut(self, i);
    }
    fn visit_path_arguments_mut(&mut self, i: &mut Args) {
        visit_path_arguments_mut(self, i);
    }
    fn visit_path_segment_mut(&mut self, i: &mut Segment) {
        visit_path_segment_mut(self, i);
    }
    fn visit_predicate_lifetime_mut(&mut self, i: &mut PredLifetime) {
        visit_predicate_lifetime_mut(self, i);
    }
    fn visit_predicate_type_mut(&mut self, i: &mut PredType) {
        visit_predicate_type_mut(self, i);
    }
    fn visit_qself_mut(&mut self, i: &mut QSelf) {
        visit_qself_mut(self, i);
    }
    fn visit_range_limits_mut(&mut self, i: &mut RangeLimits) {
        visit_range_limits_mut(self, i);
    }
    fn visit_receiver_mut(&mut self, i: &mut Receiver) {
        visit_receiver_mut(self, i);
    }
    fn visit_return_type_mut(&mut self, i: &mut ty::Ret) {
        visit_return_type_mut(self, i);
    }
    fn visit_signature_mut(&mut self, i: &mut Signature) {
        visit_signature_mut(self, i);
    }
    fn visit_span_mut(&mut self, i: &mut Span) {
        visit_span_mut(self, i);
    }
    fn visit_static_mutability_mut(&mut self, i: &mut StaticMut) {
        visit_static_mutability_mut(self, i);
    }
    fn visit_stmt_mut(&mut self, i: &mut Stmt) {
        visit_stmt_mut(self, i);
    }
    fn visit_stmt_macro_mut(&mut self, i: &mut StmtMacro) {
        visit_stmt_macro_mut(self, i);
    }
    fn visit_trait_bound_mut(&mut self, i: &mut TraitBound) {
        visit_trait_bound_mut(self, i);
    }
    fn visit_trait_bound_modifier_mut(&mut self, i: &mut TraitBoundModifier) {
        visit_trait_bound_modifier_mut(self, i);
    }
    fn visit_trait_item_mut(&mut self, i: &mut TraitItem) {
        visit_trait_item_mut(self, i);
    }
    fn visit_trait_item_const_mut(&mut self, i: &mut TraitItemConst) {
        visit_trait_item_const_mut(self, i);
    }
    fn visit_trait_item_fn_mut(&mut self, i: &mut TraitItemFn) {
        visit_trait_item_fn_mut(self, i);
    }
    fn visit_trait_item_macro_mut(&mut self, i: &mut TraitItemMacro) {
        visit_trait_item_macro_mut(self, i);
    }
    fn visit_trait_item_type_mut(&mut self, i: &mut TraitItemType) {
        visit_trait_item_type_mut(self, i);
    }
    fn visit_type_mut(&mut self, i: &mut ty::Type) {
        visit_type_mut(self, i);
    }
    fn visit_type_array_mut(&mut self, i: &mut ty::Array) {
        visit_type_array_mut(self, i);
    }
    fn visit_type_bare_fn_mut(&mut self, i: &mut ty::BareFn) {
        visit_type_bare_fn_mut(self, i);
    }
    fn visit_type_group_mut(&mut self, i: &mut ty::Group) {
        visit_type_group_mut(self, i);
    }
    fn visit_type_impl_trait_mut(&mut self, i: &mut ty::Impl) {
        visit_type_impl_trait_mut(self, i);
    }
    fn visit_type_infer_mut(&mut self, i: &mut ty::Infer) {
        visit_type_infer_mut(self, i);
    }
    fn visit_type_macro_mut(&mut self, i: &mut ty::Mac) {
        visit_type_macro_mut(self, i);
    }
    fn visit_type_never_mut(&mut self, i: &mut ty::Never) {
        visit_type_never_mut(self, i);
    }
    fn visit_type_param_mut(&mut self, i: &mut TypeParam) {
        visit_type_param_mut(self, i);
    }
    fn visit_type_param_bound_mut(&mut self, i: &mut TypeParamBound) {
        visit_type_param_bound_mut(self, i);
    }
    fn visit_type_paren_mut(&mut self, i: &mut ty::Paren) {
        visit_type_paren_mut(self, i);
    }
    fn visit_type_path_mut(&mut self, i: &mut ty::Path) {
        visit_type_path_mut(self, i);
    }
    fn visit_type_ptr_mut(&mut self, i: &mut ty::Ptr) {
        visit_type_ptr_mut(self, i);
    }
    fn visit_type_reference_mut(&mut self, i: &mut ty::Ref) {
        visit_type_reference_mut(self, i);
    }
    fn visit_type_slice_mut(&mut self, i: &mut ty::Slice) {
        visit_type_slice_mut(self, i);
    }
    fn visit_type_trait_object_mut(&mut self, i: &mut ty::TraitObj) {
        visit_type_trait_object_mut(self, i);
    }
    fn visit_type_tuple_mut(&mut self, i: &mut ty::Tuple) {
        visit_type_tuple_mut(self, i);
    }
    fn visit_un_op_mut(&mut self, i: &mut UnOp) {
        visit_un_op_mut(self, i);
    }
    fn visit_use_glob_mut(&mut self, i: &mut UseGlob) {
        visit_use_glob_mut(self, i);
    }
    fn visit_use_group_mut(&mut self, i: &mut UseGroup) {
        visit_use_group_mut(self, i);
    }
    fn visit_use_name_mut(&mut self, i: &mut UseName) {
        visit_use_name_mut(self, i);
    }
    fn visit_use_path_mut(&mut self, i: &mut UsePath) {
        visit_use_path_mut(self, i);
    }
    fn visit_use_rename_mut(&mut self, i: &mut UseRename) {
        visit_use_rename_mut(self, i);
    }
    fn visit_use_tree_mut(&mut self, i: &mut UseTree) {
        visit_use_tree_mut(self, i);
    }
    fn visit_variadic_mut(&mut self, i: &mut Variadic) {
        visit_variadic_mut(self, i);
    }
    fn visit_variant_mut(&mut self, i: &mut Variant) {
        visit_variant_mut(self, i);
    }
    fn visit_vis_restricted_mut(&mut self, i: &mut VisRestricted) {
        visit_vis_restricted_mut(self, i);
    }
    fn visit_visibility_mut(&mut self, i: &mut Visibility) {
        visit_visibility_mut(self, i);
    }
    fn visit_where_clause_mut(&mut self, i: &mut WhereClause) {
        visit_where_clause_mut(self, i);
    }
    fn visit_where_predicate_mut(&mut self, i: &mut WherePred) {
        visit_where_predicate_mut(self, i);
    }
}
pub fn visit_abi_mut<V>(v: &mut V, node: &mut Abi)
where
    V: VisitMut + ?Sized,
{
    skip!(node.extern_);
    if let Some(it) = &mut node.name {
        v.visit_lit_str_mut(it);
    }
}
pub fn visit_angle_bracketed_generic_arguments_mut<V>(v: &mut V, node: &mut AngledArgs)
where
    V: VisitMut + ?Sized,
{
    skip!(node.colon2);
    skip!(node.lt);
    for mut el in Punctuated::pairs_mut(&mut node.args) {
        let it = el.value_mut();
        v.visit_generic_argument_mut(it);
    }
    skip!(node.gt);
}
pub fn visit_arm_mut<V>(v: &mut V, node: &mut Arm)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_pat_mut(&mut node.pat);
    if let Some(it) = &mut node.guard {
        skip!((it).0);
        v.visit_expr_mut(&mut *(it).1);
    }
    skip!(node.fat_arrow);
    v.visit_expr_mut(&mut *node.body);
    skip!(node.comma);
}
pub fn visit_assoc_const_mut<V>(v: &mut V, node: &mut AssocConst)
where
    V: VisitMut + ?Sized,
{
    v.visit_ident_mut(&mut node.ident);
    if let Some(it) = &mut node.gnrs {
        v.visit_angle_bracketed_generic_arguments_mut(it);
    }
    skip!(node.eq);
    v.visit_expr_mut(&mut node.val);
}
pub fn visit_assoc_type_mut<V>(v: &mut V, node: &mut AssocType)
where
    V: VisitMut + ?Sized,
{
    v.visit_ident_mut(&mut node.ident);
    if let Some(it) = &mut node.gnrs {
        v.visit_angle_bracketed_generic_arguments_mut(it);
    }
    skip!(node.eq);
    v.visit_type_mut(&mut node.ty);
}
pub fn visit_attr_style_mut<V>(v: &mut V, node: &mut AttrStyle)
where
    V: VisitMut + ?Sized,
{
    match node {
        AttrStyle::Outer => {},
        AttrStyle::Inner(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_attribute_mut<V>(v: &mut V, node: &mut Attribute)
where
    V: VisitMut + ?Sized,
{
    skip!(node.pound);
    v.visit_attr_style_mut(&mut node.style);
    skip!(node.bracket);
    v.visit_meta_mut(&mut node.meta);
}
pub fn visit_bare_fn_arg_mut<V>(v: &mut V, node: &mut ty::BareFnArg)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    if let Some(it) = &mut node.name {
        v.visit_ident_mut(&mut (it).0);
        skip!((it).1);
    }
    v.visit_type_mut(&mut node.ty);
}
pub fn visit_bare_variadic_mut<V>(v: &mut V, node: &mut ty::BareVari)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    if let Some(it) = &mut node.name {
        v.visit_ident_mut(&mut (it).0);
        skip!((it).1);
    }
    skip!(node.dots);
    skip!(node.comma);
}
pub fn visit_bin_op_mut<V>(v: &mut V, node: &mut BinOp)
where
    V: VisitMut + ?Sized,
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
pub fn visit_block_mut<V>(v: &mut V, node: &mut Block)
where
    V: VisitMut + ?Sized,
{
    skip!(node.brace);
    for it in &mut node.stmts {
        v.visit_stmt_mut(it);
    }
}
pub fn visit_bound_lifetimes_mut<V>(v: &mut V, node: &mut BoundLifetimes)
where
    V: VisitMut + ?Sized,
{
    skip!(node.for_);
    skip!(node.lt);
    for mut el in Punctuated::pairs_mut(&mut node.lifes) {
        let it = el.value_mut();
        v.visit_generic_param_mut(it);
    }
    skip!(node.gt);
}
pub fn visit_const_param_mut<V>(v: &mut V, node: &mut ConstParam)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.const_);
    v.visit_ident_mut(&mut node.ident);
    skip!(node.colon);
    v.visit_type_mut(&mut node.typ);
    skip!(node.eq);
    if let Some(it) = &mut node.default {
        v.visit_expr_mut(it);
    }
}
pub fn visit_constraint_mut<V>(v: &mut V, node: &mut Constraint)
where
    V: VisitMut + ?Sized,
{
    v.visit_ident_mut(&mut node.ident);
    if let Some(it) = &mut node.gnrs {
        v.visit_angle_bracketed_generic_arguments_mut(it);
    }
    skip!(node.colon);
    for mut el in Punctuated::pairs_mut(&mut node.bounds) {
        let it = el.value_mut();
        v.visit_type_param_bound_mut(it);
    }
}
pub fn visit_data_mut<V>(v: &mut V, node: &mut Data)
where
    V: VisitMut + ?Sized,
{
    match node {
        Data::Struct(_binding_0) => {
            v.visit_data_struct_mut(_binding_0);
        },
        Data::Enum(_binding_0) => {
            v.visit_data_enum_mut(_binding_0);
        },
        Data::Union(_binding_0) => {
            v.visit_data_union_mut(_binding_0);
        },
    }
}
pub fn visit_data_enum_mut<V>(v: &mut V, node: &mut DataEnum)
where
    V: VisitMut + ?Sized,
{
    skip!(node.enum_);
    skip!(node.brace);
    for mut el in Punctuated::pairs_mut(&mut node.variants) {
        let it = el.value_mut();
        v.visit_variant_mut(it);
    }
}
pub fn visit_data_struct_mut<V>(v: &mut V, node: &mut DataStruct)
where
    V: VisitMut + ?Sized,
{
    skip!(node.struct_);
    v.visit_fields_mut(&mut node.fields);
    skip!(node.semi);
}
pub fn visit_data_union_mut<V>(v: &mut V, node: &mut DataUnion)
where
    V: VisitMut + ?Sized,
{
    skip!(node.union_);
    v.visit_fields_named_mut(&mut node.fields);
}
pub fn visit_derive_input_mut<V>(v: &mut V, node: &mut DeriveInput)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    v.visit_ident_mut(&mut node.ident);
    v.visit_generics_mut(&mut node.gens);
    v.visit_data_mut(&mut node.data);
}
pub fn visit_expr_mut<V>(v: &mut V, node: &mut Expr)
where
    V: VisitMut + ?Sized,
{
    match node {
        Expr::Array(_binding_0) => {
            full!(v.visit_expr_array_mut(_binding_0));
        },
        Expr::Assign(_binding_0) => {
            full!(v.visit_expr_assign_mut(_binding_0));
        },
        Expr::Async(_binding_0) => {
            full!(v.visit_expr_async_mut(_binding_0));
        },
        Expr::Await(_binding_0) => {
            full!(v.visit_expr_await_mut(_binding_0));
        },
        Expr::Binary(_binding_0) => {
            v.visit_expr_binary_mut(_binding_0);
        },
        Expr::Block(_binding_0) => {
            full!(v.visit_expr_block_mut(_binding_0));
        },
        Expr::Break(_binding_0) => {
            full!(v.visit_expr_break_mut(_binding_0));
        },
        Expr::Call(_binding_0) => {
            v.visit_expr_call_mut(_binding_0);
        },
        Expr::Cast(_binding_0) => {
            v.visit_expr_cast_mut(_binding_0);
        },
        Expr::Closure(_binding_0) => {
            full!(v.visit_expr_closure_mut(_binding_0));
        },
        Expr::Const(_binding_0) => {
            full!(v.visit_expr_const_mut(_binding_0));
        },
        Expr::Continue(_binding_0) => {
            full!(v.visit_expr_continue_mut(_binding_0));
        },
        Expr::Field(_binding_0) => {
            v.visit_expr_field_mut(_binding_0);
        },
        Expr::ForLoop(_binding_0) => {
            full!(v.visit_expr_for_loop_mut(_binding_0));
        },
        Expr::Group(_binding_0) => {
            v.visit_expr_group_mut(_binding_0);
        },
        Expr::If(_binding_0) => {
            full!(v.visit_expr_if_mut(_binding_0));
        },
        Expr::Index(_binding_0) => {
            v.visit_expr_index_mut(_binding_0);
        },
        Expr::Infer(_binding_0) => {
            full!(v.visit_expr_infer_mut(_binding_0));
        },
        Expr::Let(_binding_0) => {
            full!(v.visit_expr_let_mut(_binding_0));
        },
        Expr::Lit(_binding_0) => {
            v.visit_expr_lit_mut(_binding_0);
        },
        Expr::Loop(_binding_0) => {
            full!(v.visit_expr_loop_mut(_binding_0));
        },
        Expr::Macro(_binding_0) => {
            v.visit_expr_macro_mut(_binding_0);
        },
        Expr::Match(_binding_0) => {
            full!(v.visit_expr_match_mut(_binding_0));
        },
        Expr::MethodCall(_binding_0) => {
            full!(v.visit_expr_method_call_mut(_binding_0));
        },
        Expr::Paren(_binding_0) => {
            v.visit_expr_paren_mut(_binding_0);
        },
        Expr::Path(_binding_0) => {
            v.visit_expr_path_mut(_binding_0);
        },
        Expr::Range(_binding_0) => {
            full!(v.visit_expr_range_mut(_binding_0));
        },
        Expr::Reference(_binding_0) => {
            full!(v.visit_expr_reference_mut(_binding_0));
        },
        Expr::Repeat(_binding_0) => {
            full!(v.visit_expr_repeat_mut(_binding_0));
        },
        Expr::Return(_binding_0) => {
            full!(v.visit_expr_return_mut(_binding_0));
        },
        Expr::Struct(_binding_0) => {
            full!(v.visit_expr_struct_mut(_binding_0));
        },
        Expr::Try(_binding_0) => {
            full!(v.visit_expr_try_mut(_binding_0));
        },
        Expr::TryBlock(_binding_0) => {
            full!(v.visit_expr_try_block_mut(_binding_0));
        },
        Expr::Tuple(_binding_0) => {
            full!(v.visit_expr_tuple_mut(_binding_0));
        },
        Expr::Unary(_binding_0) => {
            v.visit_expr_unary_mut(_binding_0);
        },
        Expr::Unsafe(_binding_0) => {
            full!(v.visit_expr_unsafe_mut(_binding_0));
        },
        Expr::Verbatim(_binding_0) => {
            skip!(_binding_0);
        },
        Expr::While(_binding_0) => {
            full!(v.visit_expr_while_mut(_binding_0));
        },
        Expr::Yield(_binding_0) => {
            full!(v.visit_expr_yield_mut(_binding_0));
        },
    }
}
pub fn visit_expr_array_mut<V>(v: &mut V, node: &mut ExprArray)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.bracket);
    for mut el in Punctuated::pairs_mut(&mut node.elems) {
        let it = el.value_mut();
        v.visit_expr_mut(it);
    }
}
pub fn visit_expr_assign_mut<V>(v: &mut V, node: &mut ExprAssign)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_expr_mut(&mut *node.left);
    skip!(node.eq);
    v.visit_expr_mut(&mut *node.right);
}
pub fn visit_expr_async_mut<V>(v: &mut V, node: &mut ExprAsync)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.async_);
    skip!(node.capture);
    v.visit_block_mut(&mut node.block);
}
pub fn visit_expr_await_mut<V>(v: &mut V, node: &mut ExprAwait)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_expr_mut(&mut *node.base);
    skip!(node.dot);
    skip!(node.await_);
}
pub fn visit_expr_binary_mut<V>(v: &mut V, node: &mut ExprBinary)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_expr_mut(&mut *node.left);
    v.visit_bin_op_mut(&mut node.op);
    v.visit_expr_mut(&mut *node.right);
}
pub fn visit_expr_block_mut<V>(v: &mut V, node: &mut ExprBlock)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    if let Some(it) = &mut node.label {
        v.visit_label_mut(it);
    }
    v.visit_block_mut(&mut node.block);
}
pub fn visit_expr_break_mut<V>(v: &mut V, node: &mut ExprBreak)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.break_);
    if let Some(it) = &mut node.label {
        v.visit_lifetime_mut(it);
    }
    if let Some(it) = &mut node.expr {
        v.visit_expr_mut(&mut **it);
    }
}
pub fn visit_expr_call_mut<V>(v: &mut V, node: &mut ExprCall)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_expr_mut(&mut *node.func);
    skip!(node.paren);
    for mut el in Punctuated::pairs_mut(&mut node.args) {
        let it = el.value_mut();
        v.visit_expr_mut(it);
    }
}
pub fn visit_expr_cast_mut<V>(v: &mut V, node: &mut ExprCast)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_expr_mut(&mut *node.expr);
    skip!(node.as_);
    v.visit_type_mut(&mut *node.typ);
}
pub fn visit_expr_closure_mut<V>(v: &mut V, node: &mut ExprClosure)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    if let Some(it) = &mut node.lifes {
        v.visit_bound_lifetimes_mut(it);
    }
    skip!(node.constness);
    skip!(node.movability);
    skip!(node.asyncness);
    skip!(node.capture);
    skip!(node.or1);
    for mut el in Punctuated::pairs_mut(&mut node.inputs) {
        let it = el.value_mut();
        v.visit_pat_mut(it);
    }
    skip!(node.or2);
    v.visit_return_type_mut(&mut node.ret);
    v.visit_expr_mut(&mut *node.body);
}
pub fn visit_expr_const_mut<V>(v: &mut V, node: &mut ExprConst)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.const_);
    v.visit_block_mut(&mut node.block);
}
pub fn visit_expr_continue_mut<V>(v: &mut V, node: &mut ExprContinue)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.continue_);
    if let Some(it) = &mut node.label {
        v.visit_lifetime_mut(it);
    }
}
pub fn visit_expr_field_mut<V>(v: &mut V, node: &mut ExprField)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_expr_mut(&mut *node.base);
    skip!(node.dot);
    v.visit_member_mut(&mut node.member);
}
pub fn visit_expr_for_loop_mut<V>(v: &mut V, node: &mut ExprForLoop)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    if let Some(it) = &mut node.label {
        v.visit_label_mut(it);
    }
    skip!(node.for_);
    v.visit_pat_mut(&mut *node.pat);
    skip!(node.in_);
    v.visit_expr_mut(&mut *node.expr);
    v.visit_block_mut(&mut node.body);
}
pub fn visit_expr_group_mut<V>(v: &mut V, node: &mut ExprGroup)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.group);
    v.visit_expr_mut(&mut *node.expr);
}
pub fn visit_expr_if_mut<V>(v: &mut V, node: &mut ExprIf)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.if_);
    v.visit_expr_mut(&mut *node.cond);
    v.visit_block_mut(&mut node.then_branch);
    if let Some(it) = &mut node.else_branch {
        skip!((it).0);
        v.visit_expr_mut(&mut *(it).1);
    }
}
pub fn visit_expr_index_mut<V>(v: &mut V, node: &mut ExprIndex)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_expr_mut(&mut *node.expr);
    skip!(node.bracket);
    v.visit_expr_mut(&mut *node.index);
}
pub fn visit_expr_infer_mut<V>(v: &mut V, node: &mut ExprInfer)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.underscore);
}
pub fn visit_expr_let_mut<V>(v: &mut V, node: &mut ExprLet)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.let_);
    v.visit_pat_mut(&mut *node.pat);
    skip!(node.eq);
    v.visit_expr_mut(&mut *node.expr);
}
pub fn visit_expr_lit_mut<V>(v: &mut V, node: &mut ExprLit)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_lit_mut(&mut node.lit);
}
pub fn visit_expr_loop_mut<V>(v: &mut V, node: &mut ExprLoop)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    if let Some(it) = &mut node.label {
        v.visit_label_mut(it);
    }
    skip!(node.loop_);
    v.visit_block_mut(&mut node.body);
}
pub fn visit_expr_macro_mut<V>(v: &mut V, node: &mut ExprMacro)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_macro_mut(&mut node.mac);
}
pub fn visit_expr_match_mut<V>(v: &mut V, node: &mut ExprMatch)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.match_);
    v.visit_expr_mut(&mut *node.expr);
    skip!(node.brace);
    for it in &mut node.arms {
        v.visit_arm_mut(it);
    }
}
pub fn visit_expr_method_call_mut<V>(v: &mut V, node: &mut ExprMethodCall)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_expr_mut(&mut *node.receiver);
    skip!(node.dot);
    v.visit_ident_mut(&mut node.method);
    if let Some(it) = &mut node.turbofish {
        v.visit_angle_bracketed_generic_arguments_mut(it);
    }
    skip!(node.paren);
    for mut el in Punctuated::pairs_mut(&mut node.args) {
        let it = el.value_mut();
        v.visit_expr_mut(it);
    }
}
pub fn visit_expr_paren_mut<V>(v: &mut V, node: &mut ExprParen)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.paren);
    v.visit_expr_mut(&mut *node.expr);
}
pub fn visit_expr_path_mut<V>(v: &mut V, node: &mut ExprPath)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    if let Some(it) = &mut node.qself {
        v.visit_qself_mut(it);
    }
    v.visit_path_mut(&mut node.path);
}
pub fn visit_expr_range_mut<V>(v: &mut V, node: &mut ExprRange)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    if let Some(it) = &mut node.start {
        v.visit_expr_mut(&mut **it);
    }
    v.visit_range_limits_mut(&mut node.limits);
    if let Some(it) = &mut node.end {
        v.visit_expr_mut(&mut **it);
    }
}
pub fn visit_expr_reference_mut<V>(v: &mut V, node: &mut ExprReference)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.and);
    skip!(node.mutability);
    v.visit_expr_mut(&mut *node.expr);
}
pub fn visit_expr_repeat_mut<V>(v: &mut V, node: &mut ExprRepeat)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.bracket);
    v.visit_expr_mut(&mut *node.expr);
    skip!(node.semi);
    v.visit_expr_mut(&mut *node.len);
}
pub fn visit_expr_return_mut<V>(v: &mut V, node: &mut ExprReturn)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.return_);
    if let Some(it) = &mut node.expr {
        v.visit_expr_mut(&mut **it);
    }
}
pub fn visit_expr_struct_mut<V>(v: &mut V, node: &mut ExprStruct)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    if let Some(it) = &mut node.qself {
        v.visit_qself_mut(it);
    }
    v.visit_path_mut(&mut node.path);
    skip!(node.brace);
    for mut el in Punctuated::pairs_mut(&mut node.fields) {
        let it = el.value_mut();
        v.visit_field_value_mut(it);
    }
    skip!(node.dot2);
    if let Some(it) = &mut node.rest {
        v.visit_expr_mut(&mut **it);
    }
}
pub fn visit_expr_try_mut<V>(v: &mut V, node: &mut ExprTry)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_expr_mut(&mut *node.expr);
    skip!(node.question);
}
pub fn visit_expr_try_block_mut<V>(v: &mut V, node: &mut ExprTryBlock)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.try_);
    v.visit_block_mut(&mut node.block);
}
pub fn visit_expr_tuple_mut<V>(v: &mut V, node: &mut ExprTuple)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.paren);
    for mut el in Punctuated::pairs_mut(&mut node.elems) {
        let it = el.value_mut();
        v.visit_expr_mut(it);
    }
}
pub fn visit_expr_unary_mut<V>(v: &mut V, node: &mut ExprUnary)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_un_op_mut(&mut node.op);
    v.visit_expr_mut(&mut *node.expr);
}
pub fn visit_expr_unsafe_mut<V>(v: &mut V, node: &mut ExprUnsafe)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.unsafe_);
    v.visit_block_mut(&mut node.block);
}
pub fn visit_expr_while_mut<V>(v: &mut V, node: &mut ExprWhile)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    if let Some(it) = &mut node.label {
        v.visit_label_mut(it);
    }
    skip!(node.while_);
    v.visit_expr_mut(&mut *node.cond);
    v.visit_block_mut(&mut node.body);
}
pub fn visit_expr_yield_mut<V>(v: &mut V, node: &mut ExprYield)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.yield_);
    if let Some(it) = &mut node.expr {
        v.visit_expr_mut(&mut **it);
    }
}
pub fn visit_field_mut<V>(v: &mut V, node: &mut Field)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    v.visit_field_mutability_mut(&mut node.mutability);
    if let Some(it) = &mut node.ident {
        v.visit_ident_mut(it);
    }
    skip!(node.colon);
    v.visit_type_mut(&mut node.typ);
}
pub fn visit_field_mutability_mut<V>(v: &mut V, node: &mut FieldMut)
where
    V: VisitMut + ?Sized,
{
    match node {
        FieldMut::None => {},
    }
}
pub fn visit_field_pat_mut<V>(v: &mut V, node: &mut patt::Field)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_member_mut(&mut node.member);
    skip!(node.colon);
    v.visit_pat_mut(&mut *node.patt);
}
pub fn visit_field_value_mut<V>(v: &mut V, node: &mut FieldValue)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_member_mut(&mut node.member);
    skip!(node.colon);
    v.visit_expr_mut(&mut node.expr);
}
pub fn visit_fields_mut<V>(v: &mut V, node: &mut Fields)
where
    V: VisitMut + ?Sized,
{
    match node {
        Fields::Named(_binding_0) => {
            v.visit_fields_named_mut(_binding_0);
        },
        Fields::Unnamed(_binding_0) => {
            v.visit_fields_unnamed_mut(_binding_0);
        },
        Fields::Unit => {},
    }
}
pub fn visit_fields_named_mut<V>(v: &mut V, node: &mut FieldsNamed)
where
    V: VisitMut + ?Sized,
{
    skip!(node.brace);
    for mut el in Punctuated::pairs_mut(&mut node.named) {
        let it = el.value_mut();
        v.visit_field_mut(it);
    }
}
pub fn visit_fields_unnamed_mut<V>(v: &mut V, node: &mut FieldsUnnamed)
where
    V: VisitMut + ?Sized,
{
    skip!(node.paren);
    for mut el in Punctuated::pairs_mut(&mut node.unnamed) {
        let it = el.value_mut();
        v.visit_field_mut(it);
    }
}
pub fn visit_file_mut<V>(v: &mut V, node: &mut File)
where
    V: VisitMut + ?Sized,
{
    skip!(node.shebang);
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    for it in &mut node.items {
        v.visit_item_mut(it);
    }
}
pub fn visit_fn_arg_mut<V>(v: &mut V, node: &mut FnArg)
where
    V: VisitMut + ?Sized,
{
    match node {
        FnArg::Receiver(_binding_0) => {
            v.visit_receiver_mut(_binding_0);
        },
        FnArg::Typed(_binding_0) => {
            v.visit_pat_type_mut(_binding_0);
        },
    }
}
pub fn visit_foreign_item_mut<V>(v: &mut V, node: &mut ForeignItem)
where
    V: VisitMut + ?Sized,
{
    match node {
        ForeignItem::Fn(_binding_0) => {
            v.visit_foreign_item_fn_mut(_binding_0);
        },
        ForeignItem::Static(_binding_0) => {
            v.visit_foreign_item_static_mut(_binding_0);
        },
        ForeignItem::Type(_binding_0) => {
            v.visit_foreign_item_type_mut(_binding_0);
        },
        ForeignItem::Macro(_binding_0) => {
            v.visit_foreign_item_macro_mut(_binding_0);
        },
        ForeignItem::Verbatim(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_foreign_item_fn_mut<V>(v: &mut V, node: &mut ForeignItemFn)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    v.visit_signature_mut(&mut node.sig);
    skip!(node.semi);
}
pub fn visit_foreign_item_macro_mut<V>(v: &mut V, node: &mut ForeignItemMacro)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_macro_mut(&mut node.mac);
    skip!(node.semi);
}
pub fn visit_foreign_item_static_mut<V>(v: &mut V, node: &mut ForeignItemStatic)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.static_);
    v.visit_static_mutability_mut(&mut node.mut_);
    v.visit_ident_mut(&mut node.ident);
    skip!(node.colon);
    v.visit_type_mut(&mut *node.typ);
    skip!(node.semi);
}
pub fn visit_foreign_item_type_mut<V>(v: &mut V, node: &mut ForeignItemType)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.type);
    v.visit_ident_mut(&mut node.ident);
    v.visit_generics_mut(&mut node.gens);
    skip!(node.semi);
}
pub fn visit_generic_argument_mut<V>(v: &mut V, node: &mut Arg)
where
    V: VisitMut + ?Sized,
{
    match node {
        Arg::Lifetime(_binding_0) => {
            v.visit_lifetime_mut(_binding_0);
        },
        Arg::Type(_binding_0) => {
            v.visit_type_mut(_binding_0);
        },
        Arg::Const(_binding_0) => {
            v.visit_expr_mut(_binding_0);
        },
        Arg::AssocType(_binding_0) => {
            v.visit_assoc_type_mut(_binding_0);
        },
        Arg::AssocConst(_binding_0) => {
            v.visit_assoc_const_mut(_binding_0);
        },
        Arg::Constraint(_binding_0) => {
            v.visit_constraint_mut(_binding_0);
        },
    }
}
pub fn visit_generic_param_mut<V>(v: &mut V, node: &mut GenericParam)
where
    V: VisitMut + ?Sized,
{
    match node {
        GenericParam::Lifetime(_binding_0) => {
            v.visit_lifetime_param_mut(_binding_0);
        },
        GenericParam::Type(_binding_0) => {
            v.visit_type_param_mut(_binding_0);
        },
        GenericParam::Const(_binding_0) => {
            v.visit_const_param_mut(_binding_0);
        },
    }
}
pub fn visit_generics_mut<V>(v: &mut V, node: &mut Generics)
where
    V: VisitMut + ?Sized,
{
    skip!(node.lt);
    for mut el in Punctuated::pairs_mut(&mut node.params) {
        let it = el.value_mut();
        v.visit_generic_param_mut(it);
    }
    skip!(node.gt);
    if let Some(it) = &mut node.clause {
        v.visit_where_clause_mut(it);
    }
}
pub fn visit_ident_mut<V>(v: &mut V, node: &mut Ident)
where
    V: VisitMut + ?Sized,
{
    let mut span = node.span();
    v.visit_span_mut(&mut span);
    node.set_span(span);
}
pub fn visit_impl_item_mut<V>(v: &mut V, node: &mut ImplItem)
where
    V: VisitMut + ?Sized,
{
    match node {
        ImplItem::Const(_binding_0) => {
            v.visit_impl_item_const_mut(_binding_0);
        },
        ImplItem::Fn(_binding_0) => {
            v.visit_impl_item_fn_mut(_binding_0);
        },
        ImplItem::Type(_binding_0) => {
            v.visit_impl_item_type_mut(_binding_0);
        },
        ImplItem::Macro(_binding_0) => {
            v.visit_impl_item_macro_mut(_binding_0);
        },
        ImplItem::Verbatim(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_impl_item_const_mut<V>(v: &mut V, node: &mut ImplItemConst)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.defaultness);
    skip!(node.const_);
    v.visit_ident_mut(&mut node.ident);
    v.visit_generics_mut(&mut node.gens);
    skip!(node.colon);
    v.visit_type_mut(&mut node.typ);
    skip!(node.eq);
    v.visit_expr_mut(&mut node.expr);
    skip!(node.semi);
}
pub fn visit_impl_item_fn_mut<V>(v: &mut V, node: &mut ImplItemFn)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.defaultness);
    v.visit_signature_mut(&mut node.sig);
    v.visit_block_mut(&mut node.block);
}
pub fn visit_impl_item_macro_mut<V>(v: &mut V, node: &mut ImplItemMacro)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_macro_mut(&mut node.mac);
    skip!(node.semi);
}
pub fn visit_impl_item_type_mut<V>(v: &mut V, node: &mut ImplItemType)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.defaultness);
    skip!(node.type);
    v.visit_ident_mut(&mut node.ident);
    v.visit_generics_mut(&mut node.gens);
    skip!(node.eq);
    v.visit_type_mut(&mut node.typ);
    skip!(node.semi);
}
pub fn visit_impl_restriction_mut<V>(v: &mut V, node: &mut ImplRestriction)
where
    V: VisitMut + ?Sized,
{
    match *node {}
}
pub fn visit_index_mut<V>(v: &mut V, node: &mut Index)
where
    V: VisitMut + ?Sized,
{
    skip!(node.index);
    v.visit_span_mut(&mut node.span);
}
pub fn visit_item_mut<V>(v: &mut V, node: &mut Item)
where
    V: VisitMut + ?Sized,
{
    match node {
        Item::Const(_binding_0) => {
            v.visit_item_const_mut(_binding_0);
        },
        Item::Enum(_binding_0) => {
            v.visit_item_enum_mut(_binding_0);
        },
        Item::ExternCrate(_binding_0) => {
            v.visit_item_extern_crate_mut(_binding_0);
        },
        Item::Fn(_binding_0) => {
            v.visit_item_fn_mut(_binding_0);
        },
        Item::ForeignMod(_binding_0) => {
            v.visit_item_foreign_mod_mut(_binding_0);
        },
        Item::Impl(_binding_0) => {
            v.visit_item_impl_mut(_binding_0);
        },
        Item::Macro(_binding_0) => {
            v.visit_item_macro_mut(_binding_0);
        },
        Item::Mod(_binding_0) => {
            v.visit_item_mod_mut(_binding_0);
        },
        Item::Static(_binding_0) => {
            v.visit_item_static_mut(_binding_0);
        },
        Item::Struct(_binding_0) => {
            v.visit_item_struct_mut(_binding_0);
        },
        Item::Trait(_binding_0) => {
            v.visit_item_trait_mut(_binding_0);
        },
        Item::TraitAlias(_binding_0) => {
            v.visit_item_trait_alias_mut(_binding_0);
        },
        Item::Type(_binding_0) => {
            v.visit_item_type_mut(_binding_0);
        },
        Item::Union(_binding_0) => {
            v.visit_item_union_mut(_binding_0);
        },
        Item::Use(_binding_0) => {
            v.visit_item_use_mut(_binding_0);
        },
        Item::Verbatim(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_item_const_mut<V>(v: &mut V, node: &mut ItemConst)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.const_);
    v.visit_ident_mut(&mut node.ident);
    v.visit_generics_mut(&mut node.gens);
    skip!(node.colon);
    v.visit_type_mut(&mut *node.typ);
    skip!(node.eq);
    v.visit_expr_mut(&mut *node.expr);
    skip!(node.semi);
}
pub fn visit_item_enum_mut<V>(v: &mut V, node: &mut ItemEnum)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.enum_);
    v.visit_ident_mut(&mut node.ident);
    v.visit_generics_mut(&mut node.gens);
    skip!(node.brace);
    for mut el in Punctuated::pairs_mut(&mut node.variants) {
        let it = el.value_mut();
        v.visit_variant_mut(it);
    }
}
pub fn visit_item_extern_crate_mut<V>(v: &mut V, node: &mut ItemExternCrate)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.extern_);
    skip!(node.crate_);
    v.visit_ident_mut(&mut node.ident);
    if let Some(it) = &mut node.rename {
        skip!((it).0);
        v.visit_ident_mut(&mut (it).1);
    }
    skip!(node.semi);
}
pub fn visit_item_fn_mut<V>(v: &mut V, node: &mut ItemFn)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    v.visit_signature_mut(&mut node.sig);
    v.visit_block_mut(&mut *node.block);
}
pub fn visit_item_foreign_mod_mut<V>(v: &mut V, node: &mut ItemForeignMod)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.unsafety);
    v.visit_abi_mut(&mut node.abi);
    skip!(node.brace);
    for it in &mut node.items {
        v.visit_foreign_item_mut(it);
    }
}
pub fn visit_item_impl_mut<V>(v: &mut V, node: &mut ItemImpl)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.defaultness);
    skip!(node.unsafety);
    skip!(node.impl_);
    v.visit_generics_mut(&mut node.gens);
    if let Some(it) = &mut node.trait_ {
        skip!((it).0);
        v.visit_path_mut(&mut (it).1);
        skip!((it).2);
    }
    v.visit_type_mut(&mut *node.typ);
    skip!(node.brace);
    for it in &mut node.items {
        v.visit_impl_item_mut(it);
    }
}
pub fn visit_item_macro_mut<V>(v: &mut V, node: &mut ItemMacro)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    if let Some(it) = &mut node.ident {
        v.visit_ident_mut(it);
    }
    v.visit_macro_mut(&mut node.mac);
    skip!(node.semi);
}
pub fn visit_item_mod_mut<V>(v: &mut V, node: &mut ItemMod)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.unsafety);
    skip!(node.mod_);
    v.visit_ident_mut(&mut node.ident);
    if let Some(it) = &mut node.gist {
        skip!((it).0);
        for it in &mut (it).1 {
            v.visit_item_mut(it);
        }
    }
    skip!(node.semi);
}
pub fn visit_item_static_mut<V>(v: &mut V, node: &mut ItemStatic)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.static_);
    v.visit_static_mutability_mut(&mut node.mut_);
    v.visit_ident_mut(&mut node.ident);
    skip!(node.colon);
    v.visit_type_mut(&mut *node.typ);
    skip!(node.eq);
    v.visit_expr_mut(&mut *node.expr);
    skip!(node.semi);
}
pub fn visit_item_struct_mut<V>(v: &mut V, node: &mut ItemStruct)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.struct_);
    v.visit_ident_mut(&mut node.ident);
    v.visit_generics_mut(&mut node.gens);
    v.visit_fields_mut(&mut node.fields);
    skip!(node.semi);
}
pub fn visit_item_trait_mut<V>(v: &mut V, node: &mut ItemTrait)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.unsafety);
    skip!(node.auto_);
    if let Some(it) = &mut node.restriction {
        v.visit_impl_restriction_mut(it);
    }
    skip!(node.trait_);
    v.visit_ident_mut(&mut node.ident);
    v.visit_generics_mut(&mut node.gens);
    skip!(node.colon);
    for mut el in Punctuated::pairs_mut(&mut node.supertraits) {
        let it = el.value_mut();
        v.visit_type_param_bound_mut(it);
    }
    skip!(node.brace);
    for it in &mut node.items {
        v.visit_trait_item_mut(it);
    }
}
pub fn visit_item_trait_alias_mut<V>(v: &mut V, node: &mut ItemTraitAlias)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.trait_);
    v.visit_ident_mut(&mut node.ident);
    v.visit_generics_mut(&mut node.gens);
    skip!(node.eq);
    for mut el in Punctuated::pairs_mut(&mut node.bounds) {
        let it = el.value_mut();
        v.visit_type_param_bound_mut(it);
    }
    skip!(node.semi);
}
pub fn visit_item_type_mut<V>(v: &mut V, node: &mut ItemType)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.type);
    v.visit_ident_mut(&mut node.ident);
    v.visit_generics_mut(&mut node.gens);
    skip!(node.eq);
    v.visit_type_mut(&mut *node.typ);
    skip!(node.semi);
}
pub fn visit_item_union_mut<V>(v: &mut V, node: &mut ItemUnion)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.union_);
    v.visit_ident_mut(&mut node.ident);
    v.visit_generics_mut(&mut node.gens);
    v.visit_fields_named_mut(&mut node.fields);
}
pub fn visit_item_use_mut<V>(v: &mut V, node: &mut ItemUse)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_visibility_mut(&mut node.vis);
    skip!(node.use_);
    skip!(node.leading_colon);
    v.visit_use_tree_mut(&mut node.tree);
    skip!(node.semi);
}
pub fn visit_label_mut<V>(v: &mut V, node: &mut Label)
where
    V: VisitMut + ?Sized,
{
    v.visit_lifetime_mut(&mut node.name);
    skip!(node.colon);
}
pub fn visit_lifetime_mut<V>(v: &mut V, node: &mut Lifetime)
where
    V: VisitMut + ?Sized,
{
    v.visit_span_mut(&mut node.apostrophe);
    v.visit_ident_mut(&mut node.ident);
}
pub fn visit_lifetime_param_mut<V>(v: &mut V, node: &mut LifetimeParam)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_lifetime_mut(&mut node.life);
    skip!(node.colon);
    for mut el in Punctuated::pairs_mut(&mut node.bounds) {
        let it = el.value_mut();
        v.visit_lifetime_mut(it);
    }
}
pub fn visit_lit_mut<V>(v: &mut V, node: &mut Lit)
where
    V: VisitMut + ?Sized,
{
    match node {
        Lit::Str(_binding_0) => {
            v.visit_lit_str_mut(_binding_0);
        },
        Lit::ByteStr(_binding_0) => {
            v.visit_lit_byte_str_mut(_binding_0);
        },
        Lit::Byte(_binding_0) => {
            v.visit_lit_byte_mut(_binding_0);
        },
        Lit::Char(_binding_0) => {
            v.visit_lit_char_mut(_binding_0);
        },
        Lit::Int(_binding_0) => {
            v.visit_lit_int_mut(_binding_0);
        },
        Lit::Float(_binding_0) => {
            v.visit_lit_float_mut(_binding_0);
        },
        Lit::Bool(_binding_0) => {
            v.visit_lit_bool_mut(_binding_0);
        },
        Lit::Verbatim(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_lit_bool_mut<V>(v: &mut V, node: &mut lit::Bool)
where
    V: VisitMut + ?Sized,
{
    skip!(node.value);
    v.visit_span_mut(&mut node.span);
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
pub fn visit_local_mut<V>(v: &mut V, node: &mut Local)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.let_);
    v.visit_pat_mut(&mut node.pat);
    if let Some(it) = &mut node.init {
        v.visit_local_init_mut(it);
    }
    skip!(node.semi);
}
pub fn visit_local_init_mut<V>(v: &mut V, node: &mut LocalInit)
where
    V: VisitMut + ?Sized,
{
    skip!(node.eq);
    v.visit_expr_mut(&mut *node.expr);
    if let Some(it) = &mut node.diverge {
        skip!((it).0);
        v.visit_expr_mut(&mut *(it).1);
    }
}
pub fn visit_macro_mut<V>(v: &mut V, node: &mut Macro)
where
    V: VisitMut + ?Sized,
{
    v.visit_path_mut(&mut node.path);
    skip!(node.bang);
    v.visit_macro_delimiter_mut(&mut node.delim);
    skip!(node.tokens);
}
pub fn visit_macro_delimiter_mut<V>(v: &mut V, node: &mut MacroDelim)
where
    V: VisitMut + ?Sized,
{
    match node {
        MacroDelim::Paren(_binding_0) => {
            skip!(_binding_0);
        },
        MacroDelim::Brace(_binding_0) => {
            skip!(_binding_0);
        },
        MacroDelim::Bracket(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_member_mut<V>(v: &mut V, node: &mut Member)
where
    V: VisitMut + ?Sized,
{
    match node {
        Member::Named(_binding_0) => {
            v.visit_ident_mut(_binding_0);
        },
        Member::Unnamed(_binding_0) => {
            v.visit_index_mut(_binding_0);
        },
    }
}
pub fn visit_meta_mut<V>(v: &mut V, node: &mut Meta)
where
    V: VisitMut + ?Sized,
{
    match node {
        Meta::Path(_binding_0) => {
            v.visit_path_mut(_binding_0);
        },
        Meta::List(_binding_0) => {
            v.visit_meta_list_mut(_binding_0);
        },
        Meta::NameValue(_binding_0) => {
            v.visit_meta_name_value_mut(_binding_0);
        },
    }
}
pub fn visit_meta_list_mut<V>(v: &mut V, node: &mut MetaList)
where
    V: VisitMut + ?Sized,
{
    v.visit_path_mut(&mut node.path);
    v.visit_macro_delimiter_mut(&mut node.delim);
    skip!(node.tokens);
}
pub fn visit_meta_name_value_mut<V>(v: &mut V, node: &mut MetaNameValue)
where
    V: VisitMut + ?Sized,
{
    v.visit_path_mut(&mut node.path);
    skip!(node.eq);
    v.visit_expr_mut(&mut node.val);
}
pub fn visit_parenthesized_generic_arguments_mut<V>(v: &mut V, node: &mut ParenthesizedArgs)
where
    V: VisitMut + ?Sized,
{
    skip!(node.paren);
    for mut el in Punctuated::pairs_mut(&mut node.ins) {
        let it = el.value_mut();
        v.visit_type_mut(it);
    }
    v.visit_return_type_mut(&mut node.out);
}
pub fn visit_pat_mut<V>(v: &mut V, node: &mut patt::Patt)
where
    V: VisitMut + ?Sized,
{
    match node {
        patt::Patt::Const(_binding_0) => {
            v.visit_expr_const_mut(_binding_0);
        },
        patt::Patt::Ident(_binding_0) => {
            v.visit_pat_ident_mut(_binding_0);
        },
        patt::Patt::Lit(_binding_0) => {
            v.visit_expr_lit_mut(_binding_0);
        },
        patt::Patt::Mac(_binding_0) => {
            v.visit_expr_macro_mut(_binding_0);
        },
        patt::Patt::Or(_binding_0) => {
            v.visit_pat_or_mut(_binding_0);
        },
        patt::Patt::Paren(_binding_0) => {
            v.visit_pat_paren_mut(_binding_0);
        },
        patt::Patt::Path(_binding_0) => {
            v.visit_expr_path_mut(_binding_0);
        },
        patt::Patt::Range(_binding_0) => {
            v.visit_expr_range_mut(_binding_0);
        },
        patt::Patt::Ref(_binding_0) => {
            v.visit_pat_reference_mut(_binding_0);
        },
        patt::Patt::Rest(_binding_0) => {
            v.visit_pat_rest_mut(_binding_0);
        },
        patt::Patt::Slice(_binding_0) => {
            v.visit_pat_slice_mut(_binding_0);
        },
        patt::Patt::Struct(_binding_0) => {
            v.visit_pat_struct_mut(_binding_0);
        },
        patt::Patt::Tuple(_binding_0) => {
            v.visit_pat_tuple_mut(_binding_0);
        },
        patt::Patt::TupleStruct(_binding_0) => {
            v.visit_pat_tuple_struct_mut(_binding_0);
        },
        patt::Patt::Type(_binding_0) => {
            v.visit_pat_type_mut(_binding_0);
        },
        patt::Patt::Verbatim(_binding_0) => {
            skip!(_binding_0);
        },
        patt::Patt::Wild(_binding_0) => {
            v.visit_pat_wild_mut(_binding_0);
        },
    }
}
pub fn visit_pat_ident_mut<V>(v: &mut V, node: &mut patt::Ident)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.by_ref);
    skip!(node.mutability);
    v.visit_ident_mut(&mut node.ident);
    if let Some(it) = &mut node.sub {
        skip!((it).0);
        v.visit_pat_mut(&mut *(it).1);
    }
}
pub fn visit_pat_or_mut<V>(v: &mut V, node: &mut patt::Or)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.leading_vert);
    for mut el in Punctuated::pairs_mut(&mut node.cases) {
        let it = el.value_mut();
        v.visit_pat_mut(it);
    }
}
pub fn visit_pat_paren_mut<V>(v: &mut V, node: &mut patt::Paren)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.paren);
    v.visit_pat_mut(&mut *node.patt);
}
pub fn visit_pat_reference_mut<V>(v: &mut V, node: &mut patt::Ref)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.and);
    skip!(node.mutability);
    v.visit_pat_mut(&mut *node.patt);
}
pub fn visit_pat_rest_mut<V>(v: &mut V, node: &mut patt::Rest)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.dot2);
}
pub fn visit_pat_slice_mut<V>(v: &mut V, node: &mut patt::Slice)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.bracket);
    for mut el in Punctuated::pairs_mut(&mut node.patts) {
        let it = el.value_mut();
        v.visit_pat_mut(it);
    }
}
pub fn visit_pat_struct_mut<V>(v: &mut V, node: &mut patt::Struct)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    if let Some(it) = &mut node.qself {
        v.visit_qself_mut(it);
    }
    v.visit_path_mut(&mut node.path);
    skip!(node.brace);
    for mut el in Punctuated::pairs_mut(&mut node.fields) {
        let it = el.value_mut();
        v.visit_field_pat_mut(it);
    }
    if let Some(it) = &mut node.rest {
        v.visit_pat_rest_mut(it);
    }
}
pub fn visit_pat_tuple_mut<V>(v: &mut V, node: &mut patt::Tuple)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.paren);
    for mut el in Punctuated::pairs_mut(&mut node.patts) {
        let it = el.value_mut();
        v.visit_pat_mut(it);
    }
}
pub fn visit_pat_tuple_struct_mut<V>(v: &mut V, node: &mut patt::TupleStruct)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    if let Some(it) = &mut node.qself {
        v.visit_qself_mut(it);
    }
    v.visit_path_mut(&mut node.path);
    skip!(node.paren);
    for mut el in Punctuated::pairs_mut(&mut node.patts) {
        let it = el.value_mut();
        v.visit_pat_mut(it);
    }
}
pub fn visit_pat_type_mut<V>(v: &mut V, node: &mut patt::Type)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_pat_mut(&mut *node.patt);
    skip!(node.colon);
    v.visit_type_mut(&mut *node.typ);
}
pub fn visit_pat_wild_mut<V>(v: &mut V, node: &mut patt::Wild)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.underscore);
}
pub fn visit_path_mut<V>(v: &mut V, node: &mut Path)
where
    V: VisitMut + ?Sized,
{
    skip!(node.leading_colon);
    for mut el in Punctuated::pairs_mut(&mut node.segs) {
        let it = el.value_mut();
        v.visit_path_segment_mut(it);
    }
}
pub fn visit_path_arguments_mut<V>(v: &mut V, node: &mut Args)
where
    V: VisitMut + ?Sized,
{
    match node {
        Args::None => {},
        Args::Angled(_binding_0) => {
            v.visit_angle_bracketed_generic_arguments_mut(_binding_0);
        },
        Args::Parenthesized(_binding_0) => {
            v.visit_parenthesized_generic_arguments_mut(_binding_0);
        },
    }
}
pub fn visit_path_segment_mut<V>(v: &mut V, node: &mut Segment)
where
    V: VisitMut + ?Sized,
{
    v.visit_ident_mut(&mut node.ident);
    v.visit_path_arguments_mut(&mut node.args);
}
pub fn visit_predicate_lifetime_mut<V>(v: &mut V, node: &mut PredLifetime)
where
    V: VisitMut + ?Sized,
{
    v.visit_lifetime_mut(&mut node.life);
    skip!(node.colon);
    for mut el in Punctuated::pairs_mut(&mut node.bounds) {
        let it = el.value_mut();
        v.visit_lifetime_mut(it);
    }
}
pub fn visit_predicate_type_mut<V>(v: &mut V, node: &mut PredType)
where
    V: VisitMut + ?Sized,
{
    if let Some(it) = &mut node.lifes {
        v.visit_bound_lifetimes_mut(it);
    }
    v.visit_type_mut(&mut node.bounded);
    skip!(node.colon);
    for mut el in Punctuated::pairs_mut(&mut node.bounds) {
        let it = el.value_mut();
        v.visit_type_param_bound_mut(it);
    }
}
pub fn visit_qself_mut<V>(v: &mut V, node: &mut QSelf)
where
    V: VisitMut + ?Sized,
{
    skip!(node.lt);
    v.visit_type_mut(&mut *node.ty);
    skip!(node.position);
    skip!(node.as_);
    skip!(node.gt);
}
pub fn visit_range_limits_mut<V>(v: &mut V, node: &mut RangeLimits)
where
    V: VisitMut + ?Sized,
{
    match node {
        RangeLimits::HalfOpen(_binding_0) => {
            skip!(_binding_0);
        },
        RangeLimits::Closed(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_receiver_mut<V>(v: &mut V, node: &mut Receiver)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    if let Some(it) = &mut node.reference {
        skip!((it).0);
        if let Some(it) = &mut (it).1 {
            v.visit_lifetime_mut(it);
        }
    }
    skip!(node.mutability);
    skip!(node.self_);
    skip!(node.colon);
    v.visit_type_mut(&mut *node.typ);
}
pub fn visit_return_type_mut<V>(v: &mut V, node: &mut ty::Ret)
where
    V: VisitMut + ?Sized,
{
    match node {
        ty::Ret::Default => {},
        ty::Ret::Type(_binding_0, _binding_1) => {
            skip!(_binding_0);
            v.visit_type_mut(&mut **_binding_1);
        },
    }
}
pub fn visit_signature_mut<V>(v: &mut V, node: &mut Signature)
where
    V: VisitMut + ?Sized,
{
    skip!(node.constness);
    skip!(node.asyncness);
    skip!(node.unsafety);
    if let Some(it) = &mut node.abi {
        v.visit_abi_mut(it);
    }
    skip!(node.fn_);
    v.visit_ident_mut(&mut node.ident);
    v.visit_generics_mut(&mut node.gens);
    skip!(node.paren);
    for mut el in Punctuated::pairs_mut(&mut node.args) {
        let it = el.value_mut();
        v.visit_fn_arg_mut(it);
    }
    if let Some(it) = &mut node.vari {
        v.visit_variadic_mut(it);
    }
    v.visit_return_type_mut(&mut node.ret);
}
pub fn visit_span_mut<V>(v: &mut V, node: &mut Span)
where
    V: VisitMut + ?Sized,
{
}
pub fn visit_static_mutability_mut<V>(v: &mut V, node: &mut StaticMut)
where
    V: VisitMut + ?Sized,
{
    match node {
        StaticMut::Mut(_binding_0) => {
            skip!(_binding_0);
        },
        StaticMut::None => {},
    }
}
pub fn visit_stmt_mut<V>(v: &mut V, node: &mut Stmt)
where
    V: VisitMut + ?Sized,
{
    match node {
        Stmt::Local(_binding_0) => {
            v.visit_local_mut(_binding_0);
        },
        Stmt::Item(_binding_0) => {
            v.visit_item_mut(_binding_0);
        },
        Stmt::Expr(_binding_0, _binding_1) => {
            v.visit_expr_mut(_binding_0);
            skip!(_binding_1);
        },
        Stmt::Macro(_binding_0) => {
            v.visit_stmt_macro_mut(_binding_0);
        },
    }
}
pub fn visit_stmt_macro_mut<V>(v: &mut V, node: &mut StmtMacro)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_macro_mut(&mut node.mac);
    skip!(node.semi);
}
pub fn visit_trait_bound_mut<V>(v: &mut V, node: &mut TraitBound)
where
    V: VisitMut + ?Sized,
{
    skip!(node.paren);
    v.visit_trait_bound_modifier_mut(&mut node.modifier);
    if let Some(it) = &mut node.lifes {
        v.visit_bound_lifetimes_mut(it);
    }
    v.visit_path_mut(&mut node.path);
}
pub fn visit_trait_bound_modifier_mut<V>(v: &mut V, node: &mut TraitBoundModifier)
where
    V: VisitMut + ?Sized,
{
    match node {
        TraitBoundModifier::None => {},
        TraitBoundModifier::Maybe(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_trait_item_mut<V>(v: &mut V, node: &mut TraitItem)
where
    V: VisitMut + ?Sized,
{
    match node {
        TraitItem::Const(_binding_0) => {
            v.visit_trait_item_const_mut(_binding_0);
        },
        TraitItem::Fn(_binding_0) => {
            v.visit_trait_item_fn_mut(_binding_0);
        },
        TraitItem::Type(_binding_0) => {
            v.visit_trait_item_type_mut(_binding_0);
        },
        TraitItem::Macro(_binding_0) => {
            v.visit_trait_item_macro_mut(_binding_0);
        },
        TraitItem::Verbatim(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_trait_item_const_mut<V>(v: &mut V, node: &mut TraitItemConst)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.const_);
    v.visit_ident_mut(&mut node.ident);
    v.visit_generics_mut(&mut node.gens);
    skip!(node.colon);
    v.visit_type_mut(&mut node.typ);
    if let Some(it) = &mut node.default {
        skip!((it).0);
        v.visit_expr_mut(&mut (it).1);
    }
    skip!(node.semi);
}
pub fn visit_trait_item_fn_mut<V>(v: &mut V, node: &mut TraitItemFn)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_signature_mut(&mut node.sig);
    if let Some(it) = &mut node.default {
        v.visit_block_mut(it);
    }
    skip!(node.semi);
}
pub fn visit_trait_item_macro_mut<V>(v: &mut V, node: &mut TraitItemMacro)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_macro_mut(&mut node.mac);
    skip!(node.semi);
}
pub fn visit_trait_item_type_mut<V>(v: &mut V, node: &mut TraitItemType)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    skip!(node.type);
    v.visit_ident_mut(&mut node.ident);
    v.visit_generics_mut(&mut node.gens);
    skip!(node.colon);
    for mut el in Punctuated::pairs_mut(&mut node.bounds) {
        let it = el.value_mut();
        v.visit_type_param_bound_mut(it);
    }
    if let Some(it) = &mut node.default {
        skip!((it).0);
        v.visit_type_mut(&mut (it).1);
    }
    skip!(node.semi);
}
pub fn visit_type_mut<V>(v: &mut V, node: &mut ty::Type)
where
    V: VisitMut + ?Sized,
{
    match node {
        ty::Type::Array(_binding_0) => {
            v.visit_type_array_mut(_binding_0);
        },
        ty::Type::BareFn(_binding_0) => {
            v.visit_type_bare_fn_mut(_binding_0);
        },
        ty::Type::Group(_binding_0) => {
            v.visit_type_group_mut(_binding_0);
        },
        ty::Type::Impl(_binding_0) => {
            v.visit_type_impl_trait_mut(_binding_0);
        },
        ty::Type::Infer(_binding_0) => {
            v.visit_type_infer_mut(_binding_0);
        },
        ty::Type::Mac(_binding_0) => {
            v.visit_type_macro_mut(_binding_0);
        },
        ty::Type::Never(_binding_0) => {
            v.visit_type_never_mut(_binding_0);
        },
        ty::Type::Paren(_binding_0) => {
            v.visit_type_paren_mut(_binding_0);
        },
        ty::Type::Path(_binding_0) => {
            v.visit_type_path_mut(_binding_0);
        },
        ty::Type::Ptr(_binding_0) => {
            v.visit_type_ptr_mut(_binding_0);
        },
        ty::Type::Ref(_binding_0) => {
            v.visit_type_reference_mut(_binding_0);
        },
        ty::Type::Slice(_binding_0) => {
            v.visit_type_slice_mut(_binding_0);
        },
        ty::Type::TraitObj(_binding_0) => {
            v.visit_type_trait_object_mut(_binding_0);
        },
        ty::Type::Tuple(_binding_0) => {
            v.visit_type_tuple_mut(_binding_0);
        },
        ty::Type::Verbatim(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_type_array_mut<V>(v: &mut V, node: &mut ty::Array)
where
    V: VisitMut + ?Sized,
{
    skip!(node.bracket);
    v.visit_type_mut(&mut *node.elem);
    skip!(node.semi);
    v.visit_expr_mut(&mut node.len);
}
pub fn visit_type_bare_fn_mut<V>(v: &mut V, node: &mut ty::BareFn)
where
    V: VisitMut + ?Sized,
{
    if let Some(it) = &mut node.lifes {
        v.visit_bound_lifetimes_mut(it);
    }
    skip!(node.unsafety);
    if let Some(it) = &mut node.abi {
        v.visit_abi_mut(it);
    }
    skip!(node.fn_);
    skip!(node.paren);
    for mut el in Punctuated::pairs_mut(&mut node.args) {
        let it = el.value_mut();
        v.visit_bare_fn_arg_mut(it);
    }
    if let Some(it) = &mut node.vari {
        v.visit_bare_variadic_mut(it);
    }
    v.visit_return_type_mut(&mut node.ret);
}
pub fn visit_type_group_mut<V>(v: &mut V, node: &mut ty::Group)
where
    V: VisitMut + ?Sized,
{
    skip!(node.group);
    v.visit_type_mut(&mut *node.elem);
}
pub fn visit_type_impl_trait_mut<V>(v: &mut V, node: &mut ty::Impl)
where
    V: VisitMut + ?Sized,
{
    skip!(node.impl_);
    for mut el in Punctuated::pairs_mut(&mut node.bounds) {
        let it = el.value_mut();
        v.visit_type_param_bound_mut(it);
    }
}
pub fn visit_type_infer_mut<V>(v: &mut V, node: &mut ty::Infer)
where
    V: VisitMut + ?Sized,
{
    skip!(node.underscore);
}
pub fn visit_type_macro_mut<V>(v: &mut V, node: &mut ty::Mac)
where
    V: VisitMut + ?Sized,
{
    v.visit_macro_mut(&mut node.mac);
}
pub fn visit_type_never_mut<V>(v: &mut V, node: &mut ty::Never)
where
    V: VisitMut + ?Sized,
{
    skip!(node.bang);
}
pub fn visit_type_param_mut<V>(v: &mut V, node: &mut TypeParam)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_ident_mut(&mut node.ident);
    skip!(node.colon);
    for mut el in Punctuated::pairs_mut(&mut node.bounds) {
        let it = el.value_mut();
        v.visit_type_param_bound_mut(it);
    }
    skip!(node.eq);
    if let Some(it) = &mut node.default {
        v.visit_type_mut(it);
    }
}
pub fn visit_type_param_bound_mut<V>(v: &mut V, node: &mut TypeParamBound)
where
    V: VisitMut + ?Sized,
{
    match node {
        TypeParamBound::Trait(_binding_0) => {
            v.visit_trait_bound_mut(_binding_0);
        },
        TypeParamBound::Lifetime(_binding_0) => {
            v.visit_lifetime_mut(_binding_0);
        },
        TypeParamBound::Verbatim(_binding_0) => {
            skip!(_binding_0);
        },
    }
}
pub fn visit_type_paren_mut<V>(v: &mut V, node: &mut ty::Paren)
where
    V: VisitMut + ?Sized,
{
    skip!(node.paren);
    v.visit_type_mut(&mut *node.elem);
}
pub fn visit_type_path_mut<V>(v: &mut V, node: &mut ty::Path)
where
    V: VisitMut + ?Sized,
{
    if let Some(it) = &mut node.qself {
        v.visit_qself_mut(it);
    }
    v.visit_path_mut(&mut node.path);
}
pub fn visit_type_ptr_mut<V>(v: &mut V, node: &mut ty::Ptr)
where
    V: VisitMut + ?Sized,
{
    skip!(node.star);
    skip!(node.const_);
    skip!(node.mutability);
    v.visit_type_mut(&mut *node.elem);
}
pub fn visit_type_reference_mut<V>(v: &mut V, node: &mut ty::Ref)
where
    V: VisitMut + ?Sized,
{
    skip!(node.and);
    if let Some(it) = &mut node.life {
        v.visit_lifetime_mut(it);
    }
    skip!(node.mutability);
    v.visit_type_mut(&mut *node.elem);
}
pub fn visit_type_slice_mut<V>(v: &mut V, node: &mut ty::Slice)
where
    V: VisitMut + ?Sized,
{
    skip!(node.bracket);
    v.visit_type_mut(&mut *node.elem);
}
pub fn visit_type_trait_object_mut<V>(v: &mut V, node: &mut ty::TraitObj)
where
    V: VisitMut + ?Sized,
{
    skip!(node.dyn_);
    for mut el in Punctuated::pairs_mut(&mut node.bounds) {
        let it = el.value_mut();
        v.visit_type_param_bound_mut(it);
    }
}
pub fn visit_type_tuple_mut<V>(v: &mut V, node: &mut ty::Tuple)
where
    V: VisitMut + ?Sized,
{
    skip!(node.paren);
    for mut el in Punctuated::pairs_mut(&mut node.elems) {
        let it = el.value_mut();
        v.visit_type_mut(it);
    }
}
pub fn visit_un_op_mut<V>(v: &mut V, node: &mut UnOp)
where
    V: VisitMut + ?Sized,
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
pub fn visit_use_glob_mut<V>(v: &mut V, node: &mut UseGlob)
where
    V: VisitMut + ?Sized,
{
    skip!(node.star);
}
pub fn visit_use_group_mut<V>(v: &mut V, node: &mut UseGroup)
where
    V: VisitMut + ?Sized,
{
    skip!(node.brace);
    for mut el in Punctuated::pairs_mut(&mut node.items) {
        let it = el.value_mut();
        v.visit_use_tree_mut(it);
    }
}
pub fn visit_use_name_mut<V>(v: &mut V, node: &mut UseName)
where
    V: VisitMut + ?Sized,
{
    v.visit_ident_mut(&mut node.ident);
}
pub fn visit_use_path_mut<V>(v: &mut V, node: &mut UsePath)
where
    V: VisitMut + ?Sized,
{
    v.visit_ident_mut(&mut node.ident);
    skip!(node.colon2);
    v.visit_use_tree_mut(&mut *node.tree);
}
pub fn visit_use_rename_mut<V>(v: &mut V, node: &mut UseRename)
where
    V: VisitMut + ?Sized,
{
    v.visit_ident_mut(&mut node.ident);
    skip!(node.as_);
    v.visit_ident_mut(&mut node.rename);
}
pub fn visit_use_tree_mut<V>(v: &mut V, node: &mut UseTree)
where
    V: VisitMut + ?Sized,
{
    match node {
        UseTree::Path(_binding_0) => {
            v.visit_use_path_mut(_binding_0);
        },
        UseTree::Name(_binding_0) => {
            v.visit_use_name_mut(_binding_0);
        },
        UseTree::Rename(_binding_0) => {
            v.visit_use_rename_mut(_binding_0);
        },
        UseTree::Glob(_binding_0) => {
            v.visit_use_glob_mut(_binding_0);
        },
        UseTree::Group(_binding_0) => {
            v.visit_use_group_mut(_binding_0);
        },
    }
}
pub fn visit_variadic_mut<V>(v: &mut V, node: &mut Variadic)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    if let Some(it) = &mut node.pat {
        v.visit_pat_mut(&mut *(it).0);
        skip!((it).1);
    }
    skip!(node.dots);
    skip!(node.comma);
}
pub fn visit_variant_mut<V>(v: &mut V, node: &mut Variant)
where
    V: VisitMut + ?Sized,
{
    for it in &mut node.attrs {
        v.visit_attribute_mut(it);
    }
    v.visit_ident_mut(&mut node.ident);
    v.visit_fields_mut(&mut node.fields);
    if let Some(it) = &mut node.discriminant {
        skip!((it).0);
        v.visit_expr_mut(&mut (it).1);
    }
}
pub fn visit_vis_restricted_mut<V>(v: &mut V, node: &mut VisRestricted)
where
    V: VisitMut + ?Sized,
{
    skip!(node.pub_);
    skip!(node.paren);
    skip!(node.in_);
    v.visit_path_mut(&mut *node.path);
}
pub fn visit_visibility_mut<V>(v: &mut V, node: &mut Visibility)
where
    V: VisitMut + ?Sized,
{
    match node {
        Visibility::Public(_binding_0) => {
            skip!(_binding_0);
        },
        Visibility::Restricted(_binding_0) => {
            v.visit_vis_restricted_mut(_binding_0);
        },
        Visibility::Inherited => {},
    }
}
pub fn visit_where_clause_mut<V>(v: &mut V, node: &mut WhereClause)
where
    V: VisitMut + ?Sized,
{
    skip!(node.where_);
    for mut el in Punctuated::pairs_mut(&mut node.preds) {
        let it = el.value_mut();
        v.visit_where_predicate_mut(it);
    }
}
pub fn visit_where_predicate_mut<V>(v: &mut V, node: &mut WherePred)
where
    V: VisitMut + ?Sized,
{
    match node {
        WherePred::Lifetime(_binding_0) => {
            v.visit_predicate_lifetime_mut(_binding_0);
        },
        WherePred::Type(_binding_0) => {
            v.visit_predicate_type_mut(_binding_0);
        },
    }
}
