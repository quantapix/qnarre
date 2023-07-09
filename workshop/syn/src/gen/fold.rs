#![allow(unreachable_code, unused_variables)]
#![allow(clippy::match_wildcard_for_single_variants, clippy::needless_match)]

use crate::gen::helper::fold::*;
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
pub trait Fold {
    fn fold_abi(&mut self, i: Abi) -> Abi {
        fold_abi(self, i)
    }
    fn fold_angle_bracketed_generic_arguments(&mut self, i: AngledArgs) -> AngledArgs {
        fold_angle_bracketed_generic_arguments(self, i)
    }
    fn fold_arm(&mut self, i: Arm) -> Arm {
        fold_arm(self, i)
    }
    fn fold_assoc_const(&mut self, i: AssocConst) -> AssocConst {
        fold_assoc_const(self, i)
    }
    fn fold_assoc_type(&mut self, i: AssocType) -> AssocType {
        fold_assoc_type(self, i)
    }
    fn fold_attr_style(&mut self, i: AttrStyle) -> AttrStyle {
        fold_attr_style(self, i)
    }
    fn fold_attribute(&mut self, i: Attribute) -> Attribute {
        fold_attribute(self, i)
    }
    fn fold_bare_fn_arg(&mut self, i: ty::BareFnArg) -> ty::BareFnArg {
        fold_bare_fn_arg(self, i)
    }
    fn fold_bare_variadic(&mut self, i: ty::BareVari) -> ty::BareVari {
        fold_bare_variadic(self, i)
    }
    fn fold_bin_op(&mut self, i: BinOp) -> BinOp {
        fold_bin_op(self, i)
    }
    fn fold_block(&mut self, i: Block) -> Block {
        fold_block(self, i)
    }
    fn fold_bound_lifetimes(&mut self, i: BoundLifetimes) -> BoundLifetimes {
        fold_bound_lifetimes(self, i)
    }
    fn fold_const_param(&mut self, i: ConstParam) -> ConstParam {
        fold_const_param(self, i)
    }
    fn fold_constraint(&mut self, i: Constraint) -> Constraint {
        fold_constraint(self, i)
    }
    fn fold_data(&mut self, i: Data) -> Data {
        fold_data(self, i)
    }
    fn fold_data_enum(&mut self, i: DataEnum) -> DataEnum {
        fold_data_enum(self, i)
    }
    fn fold_data_struct(&mut self, i: DataStruct) -> DataStruct {
        fold_data_struct(self, i)
    }
    fn fold_data_union(&mut self, i: DataUnion) -> DataUnion {
        fold_data_union(self, i)
    }
    fn fold_derive_input(&mut self, i: DeriveInput) -> DeriveInput {
        fold_derive_input(self, i)
    }
    fn fold_expr(&mut self, i: Expr) -> Expr {
        fold_expr(self, i)
    }
    fn fold_expr_array(&mut self, i: expr::Array) -> expr::Array {
        fold_expr_array(self, i)
    }
    fn fold_expr_assign(&mut self, i: expr::Assign) -> expr::Assign {
        fold_expr_assign(self, i)
    }
    fn fold_expr_async(&mut self, i: expr::Async) -> expr::Async {
        fold_expr_async(self, i)
    }
    fn fold_expr_await(&mut self, i: expr::Await) -> expr::Await {
        fold_expr_await(self, i)
    }
    fn fold_expr_binary(&mut self, i: expr::Binary) -> expr::Binary {
        fold_expr_binary(self, i)
    }
    fn fold_expr_block(&mut self, i: expr::Block) -> expr::Block {
        fold_expr_block(self, i)
    }
    fn fold_expr_break(&mut self, i: expr::Break) -> expr::Break {
        fold_expr_break(self, i)
    }
    fn fold_expr_call(&mut self, i: expr::Call) -> expr::Call {
        fold_expr_call(self, i)
    }
    fn fold_expr_cast(&mut self, i: expr::Cast) -> expr::Cast {
        fold_expr_cast(self, i)
    }
    fn fold_expr_closure(&mut self, i: expr::Closure) -> expr::Closure {
        fold_expr_closure(self, i)
    }
    fn fold_expr_const(&mut self, i: expr::Const) -> expr::Const {
        fold_expr_const(self, i)
    }
    fn fold_expr_continue(&mut self, i: expr::Continue) -> expr::Continue {
        fold_expr_continue(self, i)
    }
    fn fold_expr_field(&mut self, i: expr::Field) -> expr::Field {
        fold_expr_field(self, i)
    }
    fn fold_expr_for_loop(&mut self, i: expr::ForLoop) -> expr::ForLoop {
        fold_expr_for_loop(self, i)
    }
    fn fold_expr_group(&mut self, i: expr::Group) -> expr::Group {
        fold_expr_group(self, i)
    }
    fn fold_expr_if(&mut self, i: expr::If) -> expr::If {
        fold_expr_if(self, i)
    }
    fn fold_expr_index(&mut self, i: expr::Index) -> expr::Index {
        fold_expr_index(self, i)
    }
    fn fold_expr_infer(&mut self, i: expr::Infer) -> expr::Infer {
        fold_expr_infer(self, i)
    }
    fn fold_expr_let(&mut self, i: expr::Let) -> expr::Let {
        fold_expr_let(self, i)
    }
    fn fold_expr_lit(&mut self, i: expr::Lit) -> expr::Lit {
        fold_expr_lit(self, i)
    }
    fn fold_expr_loop(&mut self, i: expr::Loop) -> expr::Loop {
        fold_expr_loop(self, i)
    }
    fn fold_expr_macro(&mut self, i: expr::Mac) -> expr::Mac {
        fold_expr_macro(self, i)
    }
    fn fold_expr_match(&mut self, i: expr::Match) -> expr::Match {
        fold_expr_match(self, i)
    }
    fn fold_expr_method_call(&mut self, i: expr::MethodCall) -> expr::MethodCall {
        fold_expr_method_call(self, i)
    }
    fn fold_expr_paren(&mut self, i: expr::Paren) -> expr::Paren {
        fold_expr_paren(self, i)
    }
    fn fold_expr_path(&mut self, i: expr::Path) -> expr::Path {
        fold_expr_path(self, i)
    }
    fn fold_expr_range(&mut self, i: expr::Range) -> expr::Range {
        fold_expr_range(self, i)
    }
    fn fold_expr_reference(&mut self, i: expr::Ref) -> expr::Ref {
        fold_expr_reference(self, i)
    }
    fn fold_expr_repeat(&mut self, i: expr::Repeat) -> expr::Repeat {
        fold_expr_repeat(self, i)
    }
    fn fold_expr_return(&mut self, i: expr::Return) -> expr::Return {
        fold_expr_return(self, i)
    }
    fn fold_expr_struct(&mut self, i: expr::Struct) -> expr::Struct {
        fold_expr_struct(self, i)
    }
    fn fold_expr_try(&mut self, i: expr::Try) -> expr::Try {
        fold_expr_try(self, i)
    }
    fn fold_expr_try_block(&mut self, i: expr::TryBlock) -> expr::TryBlock {
        fold_expr_try_block(self, i)
    }
    fn fold_expr_tuple(&mut self, i: expr::Tuple) -> expr::Tuple {
        fold_expr_tuple(self, i)
    }
    fn fold_expr_unary(&mut self, i: expr::Unary) -> expr::Unary {
        fold_expr_unary(self, i)
    }
    fn fold_expr_unsafe(&mut self, i: expr::Unsafe) -> expr::Unsafe {
        fold_expr_unsafe(self, i)
    }
    fn fold_expr_while(&mut self, i: expr::While) -> expr::While {
        fold_expr_while(self, i)
    }
    fn fold_expr_yield(&mut self, i: expr::Yield) -> expr::Yield {
        fold_expr_yield(self, i)
    }
    fn fold_field(&mut self, i: Field) -> Field {
        fold_field(self, i)
    }
    fn fold_field_mutability(&mut self, i: FieldMut) -> FieldMut {
        fold_field_mutability(self, i)
    }
    fn fold_field_pat(&mut self, i: patt::Fieldeld) patt::Field:Field {
        fold_field_pat(self, i)
    }
    fn fold_field_value(&mut self, i: FieldValue) -> FieldValue {
        fold_field_value(self, i)
    }
    fn fold_fields(&mut self, i: Fields) -> Fields {
        fold_fields(self, i)
    }
    fn fold_fields_named(&mut self, i: FieldsNamed) -> FieldsNamed {
        fold_fields_named(self, i)
    }
    fn fold_fields_unnamed(&mut self, i: FieldsUnnamed) -> FieldsUnnamed {
        fold_fields_unnamed(self, i)
    }
    fn fold_file(&mut self, i: File) -> File {
        fold_file(self, i)
    }
    fn fold_fn_arg(&mut self, i: FnArg) -> FnArg {
        fold_fn_arg(self, i)
    }
    fn fold_foreign_item(&mut self, i: ForeignItem) -> ForeignItem {
        fold_foreign_item(self, i)
    }
    fn fold_foreign_item_fn(&mut self, i: ForeignItemFn) -> ForeignItemFn {
        fold_foreign_item_fn(self, i)
    }
    fn fold_foreign_item_macro(&mut self, i: ForeignItemMacro) -> ForeignItemMacro {
        fold_foreign_item_macro(self, i)
    }
    fn fold_foreign_item_static(&mut self, i: ForeignItemStatic) -> ForeignItemStatic {
        fold_foreign_item_static(self, i)
    }
    fn fold_foreign_item_type(&mut self, i: ForeignItemType) -> ForeignItemType {
        fold_foreign_item_type(self, i)
    }
    fn fold_generic_argument(&mut self, i: Arg) -> Arg {
        fold_generic_argument(self, i)
    }
    fn fold_generic_param(&mut self, i: GenericParam) -> GenericParam {
        fold_generic_param(self, i)
    }
    fn fold_generics(&mut self, i: Generics) -> Generics {
        fold_generics(self, i)
    }
    fn fold_ident(&mut self, i: Ident) -> Ident {
        fold_ident(self, i)
    }
    fn fold_impl_item(&mut self, i: ImplItem) -> ImplItem {
        fold_impl_item(self, i)
    }
    fn fold_impl_item_const(&mut self, i: ImplItemConst) -> ImplItemConst {
        fold_impl_item_const(self, i)
    }
    fn fold_impl_item_fn(&mut self, i: ImplItemFn) -> ImplItemFn {
        fold_impl_item_fn(self, i)
    }
    fn fold_impl_item_macro(&mut self, i: ImplItemMacro) -> ImplItemMacro {
        fold_impl_item_macro(self, i)
    }
    fn fold_impl_item_type(&mut self, i: ImplItemType) -> ImplItemType {
        fold_impl_item_type(self, i)
    }
    fn fold_impl_restriction(&mut self, i: ImplRestriction) -> ImplRestriction {
        fold_impl_restriction(self, i)
    }
    fn fold_index(&mut self, i: Index) -> Index {
        fold_index(self, i)
    }
    fn fold_item(&mut self, i: Item) -> Item {
        fold_item(self, i)
    }
    fn fold_item_const(&mut self, i: ItemConst) -> ItemConst {
        fold_item_const(self, i)
    }
    fn fold_item_enum(&mut self, i: ItemEnum) -> ItemEnum {
        fold_item_enum(self, i)
    }
    fn fold_item_extern_crate(&mut self, i: ItemExternCrate) -> ItemExternCrate {
        fold_item_extern_crate(self, i)
    }
    fn fold_item_fn(&mut self, i: ItemFn) -> ItemFn {
        fold_item_fn(self, i)
    }
    fn fold_item_foreign_mod(&mut self, i: ItemForeignMod) -> ItemForeignMod {
        fold_item_foreign_mod(self, i)
    }
    fn fold_item_impl(&mut self, i: ItemImpl) -> ItemImpl {
        fold_item_impl(self, i)
    }
    fn fold_item_macro(&mut self, i: ItemMacro) -> ItemMacro {
        fold_item_macro(self, i)
    }
    fn fold_item_mod(&mut self, i: ItemMod) -> ItemMod {
        fold_item_mod(self, i)
    }
    fn fold_item_static(&mut self, i: ItemStatic) -> ItemStatic {
        fold_item_static(self, i)
    }
    fn fold_item_struct(&mut self, i: ItemStruct) -> ItemStruct {
        fold_item_struct(self, i)
    }
    fn fold_item_trait(&mut self, i: ItemTrait) -> ItemTrait {
        fold_item_trait(self, i)
    }
    fn fold_item_trait_alias(&mut self, i: ItemTraitAlias) -> ItemTraitAlias {
        fold_item_trait_alias(self, i)
    }
    fn fold_item_type(&mut self, i: ItemType) -> ItemType {
        fold_item_type(self, i)
    }
    fn fold_item_union(&mut self, i: ItemUnion) -> ItemUnion {
        fold_item_union(self, i)
    }
    fn fold_item_use(&mut self, i: ItemUse) -> ItemUse {
        fold_item_use(self, i)
    }
    fn fold_label(&mut self, i: Label) -> Label {
        fold_label(self, i)
    }
    fn fold_lifetime(&mut self, i: Lifetime) -> Lifetime {
        fold_lifetime(self, i)
    }
    fn fold_lifetime_param(&mut self, i: LifetimeParam) -> LifetimeParam {
        fold_lifetime_param(self, i)
    }
    fn fold_lit(&mut self, i: Lit) -> Lit {
        fold_lit(self, i)
    }
    fn fold_lit_bool(&mut self, i: lit::Bool) -> lit::Bool {
        fold_lit_bool(self, i)
    }
    fn fold_lit_byte(&mut self, i: lit::Byte) -> lit::Byte {
        fold_lit_byte(self, i)
    }
    fn fold_lit_byte_str(&mut self, i: lit::ByteStr) -> lit::ByteStr {
        fold_lit_byte_str(self, i)
    }
    fn fold_lit_char(&mut self, i: lit::Char) -> lit::Char {
        fold_lit_char(self, i)
    }
    fn fold_lit_float(&mut self, i: lit::Float) -> lit::Float {
        fold_lit_float(self, i)
    }
    fn fold_lit_int(&mut self, i: lit::Int) -> lit::Int {
        fold_lit_int(self, i)
    }
    fn fold_lit_str(&mut self, i: lit::Str) -> lit::Str {
        fold_lit_str(self, i)
    }
    fn fold_local(&mut self, i: stmt::Local) -> stmt::Local {
        fold_local(self, i)
    }
    fn fold_local_init(&mut self, i: stmt::LocalInit) -> stmt::LocalInit {
        fold_local_init(self, i)
    }
    fn fold_macro(&mut self, i: Macro) -> Macro {
        fold_macro(self, i)
    }
    fn fold_macro_delimiter(&mut self, i: MacroDelim) -> MacroDelim {
        fold_macro_delimiter(self, i)
    }
    fn fold_member(&mut self, i: Member) -> Member {
        fold_member(self, i)
    }
    fn fold_meta(&mut self, i: Meta) -> Meta {
        fold_meta(self, i)
    }
    fn fold_meta_list(&mut self, i: MetaList) -> MetaList {
        fold_meta_list(self, i)
    }
    fn fold_meta_name_value(&mut self, i: MetaNameValue) -> MetaNameValue {
        fold_meta_name_value(self, i)
    }
    fn fold_parenthesized_generic_arguments(&mut self, i: ParenthesizedArgs) -> ParenthesizedArgs {
        fold_parenthesized_generic_arguments(self, i)
    }
    fn fold_pat(&mut self, i: patt::Patt) -> patt::Patt {
        fold_pat(self, i)
    }
    fn fold_pat_ident(&mut self, i: patt::Ident) -> patt::Ident {
        fold_pat_ident(self, i)
    }
    fn fold_pat_or(&mut self, i: patt::Or) -> patt::Or {
        fold_pat_or(self, i)
    }
    fn fold_pat_paren(&mut self, i: patt::Paren) -> patt::Paren {
        fold_pat_paren(self, i)
    }
    fn fold_pat_reference(&mut self, i: patt::Ref) -> patt::Ref {
        fold_pat_reference(self, i)
    }
    fn fold_pat_rest(&mut self, i: patt::Restest) patt::Rest::Rest {
        fold_pat_rest(self, i)
    }
    fn fold_pat_slice(&mut self, i: patt::Slice) -> patt::Slice {
        fold_pat_slice(self, i)
    }
    fn fold_pat_struct(&mut self, i: patt::Struct) -> patt::Struct {
        fold_pat_struct(self, i)
    }
    fn fold_pat_tuple(&mut self, i: patt::Tuple) -> patt::Tuple {
        fold_pat_tuple(self, i)
    }
    fn fold_pat_tuple_struct(&mut self, i: patt::TupleStructuct) patt::TupleStructStruct {
        fold_pat_tuple_struct(self, i)
    }
    fn fold_pat_type(&mut self, i: patt::Type) -> patt::Type {
        fold_pat_type(self, i)
    }
    fn fold_pat_wild(&mut self, i: patt::Wild) -> patt::Wild {
        fold_pat_wild(self, i)
    }
    fn fold_path(&mut self, i: Path) -> Path {
        fold_path(self, i)
    }
    fn fold_path_arguments(&mut self, i: Args) -> Args {
        fold_path_arguments(self, i)
    }
    fn fold_path_segment(&mut self, i: Segment) -> Segment {
        fold_path_segment(self, i)
    }
    fn fold_predicate_lifetime(&mut self, i: PredLifetime) -> PredLifetime {
        fold_predicate_lifetime(self, i)
    }
    fn fold_predicate_type(&mut self, i: PredType) -> PredType {
        fold_predicate_type(self, i)
    }
    fn fold_qself(&mut self, i: QSelf) -> QSelf {
        fold_qself(self, i)
    }
    fn fold_range_limits(&mut self, i: RangeLimits) -> RangeLimits {
        fold_range_limits(self, i)
    }
    fn fold_receiver(&mut self, i: Receiver) -> Receiver {
        fold_receiver(self, i)
    }
    fn fold_return_type(&mut self, i: ty::Ret) -> ty::Ret {
        fold_return_type(self, i)
    }
    fn fold_signature(&mut self, i: Signature) -> Signature {
        fold_signature(self, i)
    }
    fn fold_span(&mut self, i: Span) -> Span {
        fold_span(self, i)
    }
    fn fold_static_mutability(&mut self, i: StaticMut) -> StaticMut {
        fold_static_mutability(self, i)
    }
    fn fold_stmt(&mut self, i: stmt::Stmt) -> stmt::Stmt {
        fold_stmt(self, i)
    }
    fn fold_stmt_macro(&mut self, i: stmt::Mac) -> stmt::Mac {
        fold_stmt_macro(self, i)
    }
    fn fold_trait_bound(&mut self, i: TraitBound) -> TraitBound {
        fold_trait_bound(self, i)
    }
    fn fold_trait_bound_modifier(&mut self, i: TraitBoundModifier) -> TraitBoundModifier {
        fold_trait_bound_modifier(self, i)
    }
    fn fold_trait_item(&mut self, i: TraitItem) -> TraitItem {
        fold_trait_item(self, i)
    }
    fn fold_trait_item_const(&mut self, i: TraitItemConst) -> TraitItemConst {
        fold_trait_item_const(self, i)
    }
    fn fold_trait_item_fn(&mut self, i: TraitItemFn) -> TraitItemFn {
        fold_trait_item_fn(self, i)
    }
    fn fold_trait_item_macro(&mut self, i: TraitItemMacro) -> TraitItemMacro {
        fold_trait_item_macro(self, i)
    }
    fn fold_trait_item_type(&mut self, i: TraitItemType) -> TraitItemType {
        fold_trait_item_type(self, i)
    }
    fn fold_type(&mut self, i: ty::Type) -> ty::Type {
        fold_type(self, i)
    }
    fn fold_type_array(&mut self, i: ty::Array) -> ty::Array {
        fold_type_array(self, i)
    }
    fn fold_type_bare_fn(&mut self, i: ty::BareFn) -> ty::BareFn {
        fold_type_bare_fn(self, i)
    }
    fn fold_type_group(&mut self, i: ty::Group) -> ty::Group {
        fold_type_group(self, i)
    }
    fn fold_type_impl_trait(&mut self, i: ty::Impl) -> ty::Impl {
        fold_type_impl_trait(self, i)
    }
    fn fold_type_infer(&mut self, i: ty::Infer) -> ty::Infer {
        fold_type_infer(self, i)
    }
    fn fold_type_macro(&mut self, i: ty::Mac) -> ty::Mac {
        fold_type_macro(self, i)
    }
    fn fold_type_never(&mut self, i: ty::Never) -> ty::Never {
        fold_type_never(self, i)
    }
    fn fold_type_param(&mut self, i: TypeParam) -> TypeParam {
        fold_type_param(self, i)
    }
    fn fold_type_param_bound(&mut self, i: TypeParamBound) -> TypeParamBound {
        fold_type_param_bound(self, i)
    }
    fn fold_type_paren(&mut self, i: ty::Paren) -> ty::Paren {
        fold_type_paren(self, i)
    }
    fn fold_type_path(&mut self, i: ty::Path) -> ty::Path {
        fold_type_path(self, i)
    }
    fn fold_type_ptr(&mut self, i: ty::Ptr) -> ty::Ptr {
        fold_type_ptr(self, i)
    }
    fn fold_type_reference(&mut self, i: ty::Ref) -> ty::Ref {
        fold_type_reference(self, i)
    }
    fn fold_type_slice(&mut self, i: ty::Slice) -> ty::Slice {
        fold_type_slice(self, i)
    }
    fn fold_type_trait_object(&mut self, i: ty::TraitObj) -> ty::TraitObj {
        fold_type_trait_object(self, i)
    }
    fn fold_type_tuple(&mut self, i: ty::Tuple) -> ty::Tuple {
        fold_type_tuple(self, i)
    }
    fn fold_un_op(&mut self, i: UnOp) -> UnOp {
        fold_un_op(self, i)
    }
    fn fold_use_glob(&mut self, i: UseGlob) -> UseGlob {
        fold_use_glob(self, i)
    }
    fn fold_use_group(&mut self, i: UseGroup) -> UseGroup {
        fold_use_group(self, i)
    }
    fn fold_use_name(&mut self, i: UseName) -> UseName {
        fold_use_name(self, i)
    }
    fn fold_use_path(&mut self, i: UsePath) -> UsePath {
        fold_use_path(self, i)
    }
    fn fold_use_rename(&mut self, i: UseRename) -> UseRename {
        fold_use_rename(self, i)
    }
    fn fold_use_tree(&mut self, i: UseTree) -> UseTree {
        fold_use_tree(self, i)
    }
    fn fold_variadic(&mut self, i: Variadic) -> Variadic {
        fold_variadic(self, i)
    }
    fn fold_variant(&mut self, i: Variant) -> Variant {
        fold_variant(self, i)
    }
    fn fold_vis_restricted(&mut self, i: VisRestricted) -> VisRestricted {
        fold_vis_restricted(self, i)
    }
    fn fold_visibility(&mut self, i: Visibility) -> Visibility {
        fold_visibility(self, i)
    }
    fn fold_where_clause(&mut self, i: WhereClause) -> WhereClause {
        fold_where_clause(self, i)
    }
    fn fold_where_predicate(&mut self, i: WherePred) -> WherePred {
        fold_where_predicate(self, i)
    }
}
pub fn fold_abi<F>(f: &mut F, node: Abi) -> Abi
where
    F: Fold + ?Sized,
{
    Abi {
        extern_: node.extern_,
        name: (node.name).map(|it| f.fold_lit_str(it)),
    }
}
pub fn fold_angle_bracketed_generic_arguments<F>(f: &mut F, node: AngledArgs) -> AngledArgs
where
    F: Fold + ?Sized,
{
    AngledArgs {
        colon2: node.colon2,
        lt: node.lt,
        args: FoldHelper::lift(node.args, |it| f.fold_generic_argument(it)),
        gt: node.gt,
    }
}
pub fn fold_arm<F>(f: &mut F, node: Arm) -> Arm
where
    F: Fold + ?Sized,
{
    Arm {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        pat: f.fold_pat(node.pat),
        guard: (node.guard).map(|it| ((it).0, Box::new(f.fold_expr(*(it).1)))),
        fat_arrow: node.fat_arrow,
        body: Box::new(f.fold_expr(*node.body)),
        comma: node.comma,
    }
}
pub fn fold_assoc_const<F>(f: &mut F, node: AssocConst) -> AssocConst
where
    F: Fold + ?Sized,
{
    AssocConst {
        ident: f.fold_ident(node.ident),
        gnrs: (node.gnrs).map(|it| f.fold_angle_bracketed_generic_arguments(it)),
        eq: node.eq,
        val: f.fold_expr(node.val),
    }
}
pub fn fold_assoc_type<F>(f: &mut F, node: AssocType) -> AssocType
where
    F: Fold + ?Sized,
{
    AssocType {
        ident: f.fold_ident(node.ident),
        gnrs: (node.gnrs).map(|it| f.fold_angle_bracketed_generic_arguments(it)),
        eq: node.eq,
        ty: f.fold_type(node.ty),
    }
}
pub fn fold_attr_style<F>(f: &mut F, node: AttrStyle) -> AttrStyle
where
    F: Fold + ?Sized,
{
    match node {
        AttrStyle::Outer => AttrStyle::Outer,
        AttrStyle::Inner(_binding_0) => AttrStyle::Inner(_binding_0),
    }
}
pub fn fold_attribute<F>(f: &mut F, node: Attribute) -> Attribute
where
    F: Fold + ?Sized,
{
    Attribute {
        pound: node.pound,
        style: f.fold_attr_style(node.style),
        bracket: node.bracket,
        meta: f.fold_meta(node.meta),
    }
}
pub fn fold_bare_fn_arg<F>(f: &mut F, node: ty::BareFnArg) -> ty::BareFnArg
where
    F: Fold + ?Sized,
{
    ty::BareFnArg {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        name: (node.name).map(|it| (f.fold_ident((it).0), (it).1)),
        ty: f.fold_type(node.ty),
    }
}
pub fn fold_bare_variadic<F>(f: &mut F, node: ty::BareVari) -> ty::BareVari
where
    F: Fold + ?Sized,
{
    ty::BareVari {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        name: (node.name).map(|it| (f.fold_ident((it).0), (it).1)),
        dots: node.dots,
        comma: node.comma,
    }
}
pub fn fold_bin_op<F>(f: &mut F, node: BinOp) -> BinOp
where
    F: Fold + ?Sized,
{
    match node {
        BinOp::Add(_binding_0) => BinOp::Add(_binding_0),
        BinOp::Sub(_binding_0) => BinOp::Sub(_binding_0),
        BinOp::Mul(_binding_0) => BinOp::Mul(_binding_0),
        BinOp::Div(_binding_0) => BinOp::Div(_binding_0),
        BinOp::Rem(_binding_0) => BinOp::Rem(_binding_0),
        BinOp::And(_binding_0) => BinOp::And(_binding_0),
        BinOp::Or(_binding_0) => BinOp::Or(_binding_0),
        BinOp::BitXor(_binding_0) => BinOp::BitXor(_binding_0),
        BinOp::BitAnd(_binding_0) => BinOp::BitAnd(_binding_0),
        BinOp::BitOr(_binding_0) => BinOp::BitOr(_binding_0),
        BinOp::Shl(_binding_0) => BinOp::Shl(_binding_0),
        BinOp::Shr(_binding_0) => BinOp::Shr(_binding_0),
        BinOp::Eq(_binding_0) => BinOp::Eq(_binding_0),
        BinOp::Lt(_binding_0) => BinOp::Lt(_binding_0),
        BinOp::Le(_binding_0) => BinOp::Le(_binding_0),
        BinOp::Ne(_binding_0) => BinOp::Ne(_binding_0),
        BinOp::Ge(_binding_0) => BinOp::Ge(_binding_0),
        BinOp::Gt(_binding_0) => BinOp::Gt(_binding_0),
        BinOp::AddAssign(_binding_0) => BinOp::AddAssign(_binding_0),
        BinOp::SubAssign(_binding_0) => BinOp::SubAssign(_binding_0),
        BinOp::MulAssign(_binding_0) => BinOp::MulAssign(_binding_0),
        BinOp::DivAssign(_binding_0) => BinOp::DivAssign(_binding_0),
        BinOp::RemAssign(_binding_0) => BinOp::RemAssign(_binding_0),
        BinOp::BitXorAssign(_binding_0) => BinOp::BitXorAssign(_binding_0),
        BinOp::BitAndAssign(_binding_0) => BinOp::BitAndAssign(_binding_0),
        BinOp::BitOrAssign(_binding_0) => BinOp::BitOrAssign(_binding_0),
        BinOp::ShlAssign(_binding_0) => BinOp::ShlAssign(_binding_0),
        BinOp::ShrAssign(_binding_0) => BinOp::ShrAssign(_binding_0),
    }
}
pub fn fold_block<F>(f: &mut F, node: Block) -> Block
where
    F: Fold + ?Sized,
{
    Block {
        brace: node.brace,
        stmts: FoldHelper::lift(node.stmts, |it| f.fold_stmt(it)),
    }
}
pub fn fold_bound_lifetimes<F>(f: &mut F, node: BoundLifetimes) -> BoundLifetimes
where
    F: Fold + ?Sized,
{
    BoundLifetimes {
        for_: node.for_,
        lt: node.lt,
        lifes: FoldHelper::lift(node.lifes, |it| f.fold_generic_param(it)),
        gt: node.gt,
    }
}
pub fn fold_const_param<F>(f: &mut F, node: ConstParam) -> ConstParam
where
    F: Fold + ?Sized,
{
    ConstParam {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        const_: node.const_,
        ident: f.fold_ident(node.ident),
        colon: node.colon,
        typ: f.fold_type(node.typ),
        eq: node.eq,
        default: (node.default).map(|it| f.fold_expr(it)),
    }
}
pub fn fold_constraint<F>(f: &mut F, node: Constraint) -> Constraint
where
    F: Fold + ?Sized,
{
    Constraint {
        ident: f.fold_ident(node.ident),
        gnrs: (node.gnrs).map(|it| f.fold_angle_bracketed_generic_arguments(it)),
        colon: node.colon,
        bounds: FoldHelper::lift(node.bounds, |it| f.fold_type_param_bound(it)),
    }
}
pub fn fold_data<F>(f: &mut F, node: Data) -> Data
where
    F: Fold + ?Sized,
{
    match node {
        Data::Struct(_binding_0) => Data::Struct(f.fold_data_struct(_binding_0)),
        Data::Enum(_binding_0) => Data::Enum(f.fold_data_enum(_binding_0)),
        Data::Union(_binding_0) => Data::Union(f.fold_data_union(_binding_0)),
    }
}
pub fn fold_data_enum<F>(f: &mut F, node: DataEnum) -> DataEnum
where
    F: Fold + ?Sized,
{
    DataEnum {
        enum_: node.enum_,
        brace: node.brace,
        variants: FoldHelper::lift(node.variants, |it| f.fold_variant(it)),
    }
}
pub fn fold_data_struct<F>(f: &mut F, node: DataStruct) -> DataStruct
where
    F: Fold + ?Sized,
{
    DataStruct {
        struct_: node.struct_,
        fields: f.fold_fields(node.fields),
        semi: node.semi,
    }
}
pub fn fold_data_union<F>(f: &mut F, node: DataUnion) -> DataUnion
where
    F: Fold + ?Sized,
{
    DataUnion {
        union_: node.union_,
        fields: f.fold_fields_named(node.fields),
    }
}
pub fn fold_derive_input<F>(f: &mut F, node: DeriveInput) -> DeriveInput
where
    F: Fold + ?Sized,
{
    DeriveInput {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        data: f.fold_data(node.data),
    }
}
pub fn fold_expr<F>(f: &mut F, node: Expr) -> Expr
where
    F: Fold + ?Sized,
{
    match node {
        Expr::Array(_binding_0) => Expr::Array(full!(f.fold_expr_array(_binding_0))),
        Expr::Assign(_binding_0) => Expr::Assign(full!(f.fold_expr_assign(_binding_0))),
        Expr::Async(_binding_0) => Expr::Async(full!(f.fold_expr_async(_binding_0))),
        Expr::Await(_binding_0) => Expr::Await(full!(f.fold_expr_await(_binding_0))),
        Expr::Binary(_binding_0) => Expr::Binary(f.fold_expr_binary(_binding_0)),
        Expr::Block(_binding_0) => Expr::Block(full!(f.fold_expr_block(_binding_0))),
        Expr::Break(_binding_0) => Expr::Break(full!(f.fold_expr_break(_binding_0))),
        Expr::Call(_binding_0) => Expr::Call(f.fold_expr_call(_binding_0)),
        Expr::Cast(_binding_0) => Expr::Cast(f.fold_expr_cast(_binding_0)),
        Expr::Closure(_binding_0) => Expr::Closure(full!(f.fold_expr_closure(_binding_0))),
        Expr::Const(_binding_0) => Expr::Const(full!(f.fold_expr_const(_binding_0))),
        Expr::Continue(_binding_0) => Expr::Continue(full!(f.fold_expr_continue(_binding_0))),
        Expr::Field(_binding_0) => Expr::Field(f.fold_expr_field(_binding_0)),
        Expr::ForLoop(_binding_0) => Expr::ForLoop(full!(f.fold_expr_for_loop(_binding_0))),
        Expr::Group(_binding_0) => Expr::Group(f.fold_expr_group(_binding_0)),
        Expr::If(_binding_0) => Expr::If(full!(f.fold_expr_if(_binding_0))),
        Expr::Index(_binding_0) => Expr::Index(f.fold_expr_index(_binding_0)),
        Expr::Infer(_binding_0) => Expr::Infer(full!(f.fold_expr_infer(_binding_0))),
        Expr::Let(_binding_0) => Expr::Let(full!(f.fold_expr_let(_binding_0))),
        Expr::Lit(_binding_0) => Expr::Lit(f.fold_expr_lit(_binding_0)),
        Expr::Loop(_binding_0) => Expr::Loop(full!(f.fold_expr_loop(_binding_0))),
        Expr::Macro(_binding_0) => Expr::Macro(f.fold_expr_macro(_binding_0)),
        Expr::Match(_binding_0) => Expr::Match(full!(f.fold_expr_match(_binding_0))),
        Expr::MethodCall(_binding_0) => Expr::MethodCall(full!(f.fold_expr_method_call(_binding_0))),
        Expr::Paren(_binding_0) => Expr::Paren(f.fold_expr_paren(_binding_0)),
        Expr::Path(_binding_0) => Expr::Path(f.fold_expr_path(_binding_0)),
        Expr::Range(_binding_0) => Expr::Range(full!(f.fold_expr_range(_binding_0))),
        Expr::Reference(_binding_0) => Expr::Reference(full!(f.fold_expr_reference(_binding_0))),
        Expr::Repeat(_binding_0) => Expr::Repeat(full!(f.fold_expr_repeat(_binding_0))),
        Expr::Return(_binding_0) => Expr::Return(full!(f.fold_expr_return(_binding_0))),
        Expr::Struct(_binding_0) => Expr::Struct(full!(f.fold_expr_struct(_binding_0))),
        Expr::Try(_binding_0) => Expr::Try(full!(f.fold_expr_try(_binding_0))),
        Expr::TryBlock(_binding_0) => Expr::TryBlock(full!(f.fold_expr_try_block(_binding_0))),
        Expr::Tuple(_binding_0) => Expr::Tuple(full!(f.fold_expr_tuple(_binding_0))),
        Expr::Unary(_binding_0) => Expr::Unary(f.fold_expr_unary(_binding_0)),
        Expr::Unsafe(_binding_0) => Expr::Unsafe(full!(f.fold_expr_unsafe(_binding_0))),
        Expr::Verbatim(_binding_0) => Expr::Verbatim(_binding_0),
        Expr::While(_binding_0) => Expr::While(full!(f.fold_expr_while(_binding_0))),
        Expr::Yield(_binding_0) => Expr::Yield(full!(f.fold_expr_yield(_binding_0))),
    }
}
pub fn fold_expr_array<F>(f: &mut F, node: expr::Array) -> expr::Array
where
    F: Fold + ?Sized,
{
    expr::Array {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        bracket: node.bracket,
        elems: FoldHelper::lift(node.elems, |it| f.fold_expr(it)),
    }
}
pub fn fold_expr_assign<F>(f: &mut F, node: expr::Assign) -> expr::Assign
where
    F: Fold + ?Sized,
{
    expr::Assign {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        left: Box::new(f.fold_expr(*node.left)),
        eq: node.eq,
        right: Box::new(f.fold_expr(*node.right)),
    }
}
pub fn fold_expr_async<F>(f: &mut F, node: expr::Async) -> expr::Async
where
    F: Fold + ?Sized,
{
    expr::Async {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        async_: node.async_,
        move_: node.move_,
        block: f.fold_block(node.block),
    }
}
pub fn fold_expr_await<F>(f: &mut F, node: expr::Await) -> expr::Await
where
    F: Fold + ?Sized,
{
    expr::Await {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        expr: Box::new(f.fold_expr(*node.expr)),
        dot: node.dot,
        await_: node.await_,
    }
}
pub fn fold_expr_binary<F>(f: &mut F, node: expr::Binary) -> expr::Binary
where
    F: Fold + ?Sized,
{
    expr::Binary {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        left: Box::new(f.fold_expr(*node.left)),
        op: f.fold_bin_op(node.op),
        right: Box::new(f.fold_expr(*node.right)),
    }
}
pub fn fold_expr_block<F>(f: &mut F, node: expr::Block) -> expr::Block
where
    F: Fold + ?Sized,
{
    expr::Block {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        label: (node.label).map(|it| f.fold_label(it)),
        block: f.fold_block(node.block),
    }
}
pub fn fold_expr_break<F>(f: &mut F, node: expr::Break) -> expr::Break
where
    F: Fold + ?Sized,
{
    expr::Break {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        break_: node.break_,
        label: (node.label).map(|it| f.fold_lifetime(it)),
        expr: (node.expr).map(|it| Box::new(f.fold_expr(*it))),
    }
}
pub fn fold_expr_call<F>(f: &mut F, node: expr::Call) -> expr::Call
where
    F: Fold + ?Sized,
{
    expr::Call {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        func: Box::new(f.fold_expr(*node.func)),
        paren: node.paren,
        args: FoldHelper::lift(node.args, |it| f.fold_expr(it)),
    }
}
pub fn fold_expr_cast<F>(f: &mut F, node: expr::Cast) -> expr::Cast
where
    F: Fold + ?Sized,
{
    expr::Cast {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        expr: Box::new(f.fold_expr(*node.expr)),
        as_: node.as_,
        typ: Box::new(f.fold_type(*node.typ)),
    }
}
pub fn fold_expr_closure<F>(f: &mut F, node: expr::Closure) -> expr::Closure
where
    F: Fold + ?Sized,
{
    expr::Closure {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        lifes: (node.lifes).map(|it| f.fold_bound_lifetimes(it)),
        const_: node.const_,
        static_: node.static_,
        async_: node.async_,
        move_: node.move_,
        or1: node.or1,
        inputs: FoldHelper::lift(node.inputs, |it| f.fold_pat(it)),
        or2: node.or2,
        ret: f.fold_return_type(node.ret),
        body: Box::new(f.fold_expr(*node.body)),
    }
}
pub fn fold_expr_const<F>(f: &mut F, node: expr::Const) -> expr::Const
where
    F: Fold + ?Sized,
{
    expr::Const {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        const_: node.const_,
        block: f.fold_block(node.block),
    }
}
pub fn fold_expr_continue<F>(f: &mut F, node: expr::Continue) -> expr::Continue
where
    F: Fold + ?Sized,
{
    expr::Continue {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        continue_: node.continue_,
        label: (node.label).map(|it| f.fold_lifetime(it)),
    }
}
pub fn fold_expr_field<F>(f: &mut F, node: expr::Field) -> expr::Field
where
    F: Fold + ?Sized,
{
    expr::Field {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        base: Box::new(f.fold_expr(*node.base)),
        dot: node.dot,
        memb: f.fold_member(node.memb),
    }
}
pub fn fold_expr_for_loop<F>(f: &mut F, node: expr::ForLoop) -> expr::ForLoop
where
    F: Fold + ?Sized,
{
    expr::ForLoop {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        label: (node.label).map(|it| f.fold_label(it)),
        for_: node.for_,
        pat: Box::new(f.fold_pat(*node.pat)),
        in_: node.in_,
        expr: Box::new(f.fold_expr(*node.expr)),
        body: f.fold_block(node.body),
    }
}
pub fn fold_expr_group<F>(f: &mut F, node: expr::Group) -> expr::Group
where
    F: Fold + ?Sized,
{
    expr::Group {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        group: node.group,
        expr: Box::new(f.fold_expr(*node.expr)),
    }
}
pub fn fold_expr_if<F>(f: &mut F, node: expr::If) -> expr::If
where
    F: Fold + ?Sized,
{
    expr::If {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        if_: node.if_,
        cond: Box::new(f.fold_expr(*node.cond)),
        then_branch: f.fold_block(node.then_branch),
        else_branch: (node.else_branch).map(|it| ((it).0, Box::new(f.fold_expr(*(it).1)))),
    }
}
pub fn fold_expr_index<F>(f: &mut F, node: expr::Index) -> expr::Index
where
    F: Fold + ?Sized,
{
    expr::Index {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        expr: Box::new(f.fold_expr(*node.expr)),
        bracket: node.bracket,
        index: Box::new(f.fold_expr(*node.index)),
    }
}
pub fn fold_expr_infer<F>(f: &mut F, node: expr::Infer) -> expr::Infer
where
    F: Fold + ?Sized,
{
    expr::Infer {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        underscore: node.underscore,
    }
}
pub fn fold_expr_let<F>(f: &mut F, node: expr::Let) -> expr::Let
where
    F: Fold + ?Sized,
{
    expr::Let {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        let_: node.let_,
        pat: Box::new(f.fold_pat(*node.pat)),
        eq: node.eq,
        expr: Box::new(f.fold_expr(*node.expr)),
    }
}
pub fn fold_expr_lit<F>(f: &mut F, node: expr::Lit) -> expr::Lit
where
    F: Fold + ?Sized,
{
    expr::Lit {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        lit: f.fold_lit(node.lit),
    }
}
pub fn fold_expr_loop<F>(f: &mut F, node: expr::Loop) -> expr::Loop
where
    F: Fold + ?Sized,
{
    expr::Loop {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        label: (node.label).map(|it| f.fold_label(it)),
        loop_: node.loop_,
        body: f.fold_block(node.body),
    }
}
pub fn fold_expr_macro<F>(f: &mut F, node: expr::Mac) -> expr::Mac
where
    F: Fold + ?Sized,
{
    expr::Mac {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        mac: f.fold_macro(node.mac),
    }
}
pub fn fold_expr_match<F>(f: &mut F, node: expr::Match) -> expr::Match
where
    F: Fold + ?Sized,
{
    expr::Match {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        match_: node.match_,
        expr: Box::new(f.fold_expr(*node.expr)),
        brace: node.brace,
        arms: FoldHelper::lift(node.arms, |it| f.fold_arm(it)),
    }
}
pub fn fold_expr_method_call<F>(f: &mut F, node: expr::MethodCall) -> expr::MethodCall
where
    F: Fold + ?Sized,
{
    expr::MethodCall {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        expr: Box::new(f.fold_expr(*node.expr)),
        dot: node.dot,
        method: f.fold_ident(node.method),
        turbofish: (node.turbofish).map(|it| f.fold_angle_bracketed_generic_arguments(it)),
        paren: node.paren,
        args: FoldHelper::lift(node.args, |it| f.fold_expr(it)),
    }
}
pub fn fold_expr_paren<F>(f: &mut F, node: expr::Paren) -> expr::Paren
where
    F: Fold + ?Sized,
{
    expr::Paren {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        paren: node.paren,
        expr: Box::new(f.fold_expr(*node.expr)),
    }
}
pub fn fold_expr_path<F>(f: &mut F, node: expr::Path) -> expr::Path
where
    F: Fold + ?Sized,
{
    expr::Path {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        qself: (node.qself).map(|it| f.fold_qself(it)),
        path: f.fold_path(node.path),
    }
}
pub fn fold_expr_range<F>(f: &mut F, node: expr::Range) -> expr::Range
where
    F: Fold + ?Sized,
{
    expr::Range {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        beg: (node.beg).map(|it| Box::new(f.fold_expr(*it))),
        limits: f.fold_range_limits(node.limits),
        end: (node.end).map(|it| Box::new(f.fold_expr(*it))),
    }
}
pub fn fold_expr_reference<F>(f: &mut F, node: expr::Ref) -> expr::Ref
where
    F: Fold + ?Sized,
{
    expr::Ref {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        and: node.and,
        mut_: node.mut_,
        expr: Box::new(f.fold_expr(*node.expr)),
    }
}
pub fn fold_expr_repeat<F>(f: &mut F, node: expr::Repeat) -> expr::Repeat
where
    F: Fold + ?Sized,
{
    expr::Repeat {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        bracket: node.bracket,
        expr: Box::new(f.fold_expr(*node.expr)),
        semi: node.semi,
        len: Box::new(f.fold_expr(*node.len)),
    }
}
pub fn fold_expr_return<F>(f: &mut F, node: expr::Return) -> expr::Return
where
    F: Fold + ?Sized,
{
    expr::Return {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        return_: node.return_,
        expr: (node.expr).map(|it| Box::new(f.fold_expr(*it))),
    }
}
pub fn fold_expr_struct<F>(f: &mut F, node: expr::Struct) -> expr::Struct
where
    F: Fold + ?Sized,
{
    expr::Struct {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        qself: (node.qself).map(|it| f.fold_qself(it)),
        path: f.fold_path(node.path),
        brace: node.brace,
        fields: FoldHelper::lift(node.fields, |it| f.fold_field_value(it)),
        dot2: node.dot2,
        rest: (node.rest).map(|it| Box::new(f.fold_expr(*it))),
    }
}
pub fn fold_expr_try<F>(f: &mut F, node: expr::Try) -> expr::Try
where
    F: Fold + ?Sized,
{
    expr::Try {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        expr: Box::new(f.fold_expr(*node.expr)),
        question: node.question,
    }
}
pub fn fold_expr_try_block<F>(f: &mut F, node: expr::TryBlock) -> expr::TryBlock
where
    F: Fold + ?Sized,
{
    expr::TryBlock {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        try_: node.try_,
        block: f.fold_block(node.block),
    }
}
pub fn fold_expr_tuple<F>(f: &mut F, node: expr::Tuple) -> expr::Tuple
where
    F: Fold + ?Sized,
{
    expr::Tuple {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        paren: node.paren,
        elems: FoldHelper::lift(node.elems, |it| f.fold_expr(it)),
    }
}
pub fn fold_expr_unary<F>(f: &mut F, node: expr::Unary) -> expr::Unary
where
    F: Fold + ?Sized,
{
    expr::Unary {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        op: f.fold_un_op(node.op),
        expr: Box::new(f.fold_expr(*node.expr)),
    }
}
pub fn fold_expr_unsafe<F>(f: &mut F, node: expr::Unsafe) -> expr::Unsafe
where
    F: Fold + ?Sized,
{
    expr::Unsafe {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        unsafe_: node.unsafe_,
        block: f.fold_block(node.block),
    }
}
pub fn fold_expr_while<F>(f: &mut F, node: expr::While) -> expr::While
where
    F: Fold + ?Sized,
{
    expr::While {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        label: (node.label).map(|it| f.fold_label(it)),
        while_: node.while_,
        cond: Box::new(f.fold_expr(*node.cond)),
        body: f.fold_block(node.body),
    }
}
pub fn fold_expr_yield<F>(f: &mut F, node: expr::Yield) -> expr::Yield
where
    F: Fold + ?Sized,
{
    expr::Yield {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        yield_: node.yield_,
        expr: (node.expr).map(|it| Box::new(f.fold_expr(*it))),
    }
}
pub fn fold_field<F>(f: &mut F, node: Field) -> Field
where
    F: Fold + ?Sized,
{
    Field {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        mutability: f.fold_field_mutability(node.mutability),
        ident: (node.ident).map(|it| f.fold_ident(it)),
        colon: node.colon,
        typ: f.fold_type(node.typ),
    }
}
pub fn fold_field_mutability<F>(f: &mut F, node: FieldMut) -> FieldMut
where
    F: Fold + ?Sized,
{
    match node {
        FieldMut::None => FieldMut::None,
    }
}
pub fn fold_field_pat<F>(f: &mut F, node: patt::Fieldeld) patt::Field:Field
where
    F: Fold + ?Sized,
{
    patt::Fieldeld {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        member: f.fold_member(node.member),
        colon: node.colon,
        pat: Box::new(f.fold_pat(*node.pat)),
    }
}
pub fn fold_field_value<F>(f: &mut F, node: FieldValue) -> FieldValue
where
    F: Fold + ?Sized,
{
    FieldValue {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        member: f.fold_member(node.member),
        colon: node.colon,
        expr: f.fold_expr(node.expr),
    }
}
pub fn fold_fields<F>(f: &mut F, node: Fields) -> Fields
where
    F: Fold + ?Sized,
{
    match node {
        Fields::Named(_binding_0) => Fields::Named(f.fold_fields_named(_binding_0)),
        Fields::Unnamed(_binding_0) => Fields::Unnamed(f.fold_fields_unnamed(_binding_0)),
        Fields::Unit => Fields::Unit,
    }
}
pub fn fold_fields_named<F>(f: &mut F, node: FieldsNamed) -> FieldsNamed
where
    F: Fold + ?Sized,
{
    FieldsNamed {
        brace: node.brace,
        named: FoldHelper::lift(node.named, |it| f.fold_field(it)),
    }
}
pub fn fold_fields_unnamed<F>(f: &mut F, node: FieldsUnnamed) -> FieldsUnnamed
where
    F: Fold + ?Sized,
{
    FieldsUnnamed {
        paren: node.paren,
        unnamed: FoldHelper::lift(node.unnamed, |it| f.fold_field(it)),
    }
}
pub fn fold_file<F>(f: &mut F, node: File) -> File
where
    F: Fold + ?Sized,
{
    File {
        shebang: node.shebang,
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        items: FoldHelper::lift(node.items, |it| f.fold_item(it)),
    }
}
pub fn fold_fn_arg<F>(f: &mut F, node: FnArg) -> FnArg
where
    F: Fold + ?Sized,
{
    match node {
        FnArg::Receiver(_binding_0) => FnArg::Receiver(f.fold_receiver(_binding_0)),
        FnArg::Typed(_binding_0) => FnArg::Typed(f.fold_pat_type(_binding_0)),
    }
}
pub fn fold_foreign_item<F>(f: &mut F, node: ForeignItem) -> ForeignItem
where
    F: Fold + ?Sized,
{
    match node {
        ForeignItem::Fn(_binding_0) => ForeignItem::Fn(f.fold_foreign_item_fn(_binding_0)),
        ForeignItem::Static(_binding_0) => ForeignItem::Static(f.fold_foreign_item_static(_binding_0)),
        ForeignItem::Type(_binding_0) => ForeignItem::Type(f.fold_foreign_item_type(_binding_0)),
        ForeignItem::Macro(_binding_0) => ForeignItem::Macro(f.fold_foreign_item_macro(_binding_0)),
        ForeignItem::Verbatim(_binding_0) => ForeignItem::Verbatim(_binding_0),
    }
}
pub fn fold_foreign_item_fn<F>(f: &mut F, node: ForeignItemFn) -> ForeignItemFn
where
    F: Fold + ?Sized,
{
    ForeignItemFn {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        sig: f.fold_signature(node.sig),
        semi: node.semi,
    }
}
pub fn fold_foreign_item_macro<F>(f: &mut F, node: ForeignItemMacro) -> ForeignItemMacro
where
    F: Fold + ?Sized,
{
    ForeignItemMacro {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        mac: f.fold_macro(node.mac),
        semi: node.semi,
    }
}
pub fn fold_foreign_item_static<F>(f: &mut F, node: ForeignItemStatic) -> ForeignItemStatic
where
    F: Fold + ?Sized,
{
    ForeignItemStatic {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        static_: node.static_,
        mut_: f.fold_static_mutability(node.mut_),
        ident: f.fold_ident(node.ident),
        colon: node.colon,
        typ: Box::new(f.fold_type(*node.typ)),
        semi: node.semi,
    }
}
pub fn fold_foreign_item_type<F>(f: &mut F, node: ForeignItemType) -> ForeignItemType
where
    F: Fold + ?Sized,
{
    ForeignItemType {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        type: node.type,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        semi: node.semi,
    }
}
pub fn fold_generic_argument<F>(f: &mut F, node: Arg) -> Arg
where
    F: Fold + ?Sized,
{
    match node {
        Arg::Lifetime(_binding_0) => Arg::Lifetime(f.fold_lifetime(_binding_0)),
        Arg::Type(_binding_0) => Arg::Type(f.fold_type(_binding_0)),
        Arg::Const(_binding_0) => Arg::Const(f.fold_expr(_binding_0)),
        Arg::AssocType(_binding_0) => Arg::AssocType(f.fold_assoc_type(_binding_0)),
        Arg::AssocConst(_binding_0) => Arg::AssocConst(f.fold_assoc_const(_binding_0)),
        Arg::Constraint(_binding_0) => Arg::Constraint(f.fold_constraint(_binding_0)),
    }
}
pub fn fold_generic_param<F>(f: &mut F, node: GenericParam) -> GenericParam
where
    F: Fold + ?Sized,
{
    match node {
        GenericParam::Lifetime(_binding_0) => GenericParam::Lifetime(f.fold_lifetime_param(_binding_0)),
        GenericParam::Type(_binding_0) => GenericParam::Type(f.fold_type_param(_binding_0)),
        GenericParam::Const(_binding_0) => GenericParam::Const(f.fold_const_param(_binding_0)),
    }
}
pub fn fold_generics<F>(f: &mut F, node: Generics) -> Generics
where
    F: Fold + ?Sized,
{
    Generics {
        lt: node.lt,
        params: FoldHelper::lift(node.params, |it| f.fold_generic_param(it)),
        gt: node.gt,
        clause: (node.clause).map(|it| f.fold_where_clause(it)),
    }
}
pub fn fold_ident<F>(f: &mut F, node: Ident) -> Ident
where
    F: Fold + ?Sized,
{
    let mut node = node;
    let span = f.fold_span(node.span());
    node.set_span(span);
    node
}
pub fn fold_impl_item<F>(f: &mut F, node: ImplItem) -> ImplItem
where
    F: Fold + ?Sized,
{
    match node {
        ImplItem::Const(_binding_0) => ImplItem::Const(f.fold_impl_item_const(_binding_0)),
        ImplItem::Fn(_binding_0) => ImplItem::Fn(f.fold_impl_item_fn(_binding_0)),
        ImplItem::Type(_binding_0) => ImplItem::Type(f.fold_impl_item_type(_binding_0)),
        ImplItem::Macro(_binding_0) => ImplItem::Macro(f.fold_impl_item_macro(_binding_0)),
        ImplItem::Verbatim(_binding_0) => ImplItem::Verbatim(_binding_0),
    }
}
pub fn fold_impl_item_const<F>(f: &mut F, node: ImplItemConst) -> ImplItemConst
where
    F: Fold + ?Sized,
{
    ImplItemConst {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        default_: node.default_,
        const_: node.const_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        colon: node.colon,
        typ: f.fold_type(node.typ),
        eq: node.eq,
        expr: f.fold_expr(node.expr),
        semi: node.semi,
    }
}
pub fn fold_impl_item_fn<F>(f: &mut F, node: ImplItemFn) -> ImplItemFn
where
    F: Fold + ?Sized,
{
    ImplItemFn {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        default_: node.default_,
        sig: f.fold_signature(node.sig),
        block: f.fold_block(node.block),
    }
}
pub fn fold_impl_item_macro<F>(f: &mut F, node: ImplItemMacro) -> ImplItemMacro
where
    F: Fold + ?Sized,
{
    ImplItemMacro {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        mac: f.fold_macro(node.mac),
        semi: node.semi,
    }
}
pub fn fold_impl_item_type<F>(f: &mut F, node: ImplItemType) -> ImplItemType
where
    F: Fold + ?Sized,
{
    ImplItemType {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        default_: node.default_,
        type: node.type,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        eq: node.eq,
        typ: f.fold_type(node.typ),
        semi: node.semi,
    }
}
pub fn fold_impl_restriction<F>(f: &mut F, node: ImplRestriction) -> ImplRestriction
where
    F: Fold + ?Sized,
{
    match node {}
}
pub fn fold_index<F>(f: &mut F, node: Index) -> Index
where
    F: Fold + ?Sized,
{
    Index {
        index: node.index,
        span: f.fold_span(node.span),
    }
}
pub fn fold_item<F>(f: &mut F, node: Item) -> Item
where
    F: Fold + ?Sized,
{
    match node {
        Item::Const(_binding_0) => Item::Const(f.fold_item_const(_binding_0)),
        Item::Enum(_binding_0) => Item::Enum(f.fold_item_enum(_binding_0)),
        Item::ExternCrate(_binding_0) => Item::ExternCrate(f.fold_item_extern_crate(_binding_0)),
        Item::Fn(_binding_0) => Item::Fn(f.fold_item_fn(_binding_0)),
        Item::ForeignMod(_binding_0) => Item::ForeignMod(f.fold_item_foreign_mod(_binding_0)),
        Item::Impl(_binding_0) => Item::Impl(f.fold_item_impl(_binding_0)),
        Item::Macro(_binding_0) => Item::Macro(f.fold_item_macro(_binding_0)),
        Item::Mod(_binding_0) => Item::Mod(f.fold_item_mod(_binding_0)),
        Item::Static(_binding_0) => Item::Static(f.fold_item_static(_binding_0)),
        Item::Struct(_binding_0) => Item::Struct(f.fold_item_struct(_binding_0)),
        Item::Trait(_binding_0) => Item::Trait(f.fold_item_trait(_binding_0)),
        Item::TraitAlias(_binding_0) => Item::TraitAlias(f.fold_item_trait_alias(_binding_0)),
        Item::Type(_binding_0) => Item::Type(f.fold_item_type(_binding_0)),
        Item::Union(_binding_0) => Item::Union(f.fold_item_union(_binding_0)),
        Item::Use(_binding_0) => Item::Use(f.fold_item_use(_binding_0)),
        Item::Verbatim(_binding_0) => Item::Verbatim(_binding_0),
    }
}
pub fn fold_item_const<F>(f: &mut F, node: ItemConst) -> ItemConst
where
    F: Fold + ?Sized,
{
    ItemConst {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        const_: node.const_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        colon: node.colon,
        typ: Box::new(f.fold_type(*node.typ)),
        eq: node.eq,
        expr: Box::new(f.fold_expr(*node.expr)),
        semi: node.semi,
    }
}
pub fn fold_item_enum<F>(f: &mut F, node: ItemEnum) -> ItemEnum
where
    F: Fold + ?Sized,
{
    ItemEnum {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        enum_: node.enum_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        brace: node.brace,
        variants: FoldHelper::lift(node.variants, |it| f.fold_variant(it)),
    }
}
pub fn fold_item_extern_crate<F>(f: &mut F, node: ItemExternCrate) -> ItemExternCrate
where
    F: Fold + ?Sized,
{
    ItemExternCrate {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        extern_: node.extern_,
        crate_: node.crate_,
        ident: f.fold_ident(node.ident),
        rename: (node.rename).map(|it| ((it).0, f.fold_ident((it).1))),
        semi: node.semi,
    }
}
pub fn fold_item_fn<F>(f: &mut F, node: ItemFn) -> ItemFn
where
    F: Fold + ?Sized,
{
    ItemFn {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        sig: f.fold_signature(node.sig),
        block: Box::new(f.fold_block(*node.block)),
    }
}
pub fn fold_item_foreign_mod<F>(f: &mut F, node: ItemForeignMod) -> ItemForeignMod
where
    F: Fold + ?Sized,
{
    ItemForeignMod {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        unsafe_: node.unsafe_,
        abi: f.fold_abi(node.abi),
        brace: node.brace,
        items: FoldHelper::lift(node.items, |it| f.fold_foreign_item(it)),
    }
}
pub fn fold_item_impl<F>(f: &mut F, node: ItemImpl) -> ItemImpl
where
    F: Fold + ?Sized,
{
    ItemImpl {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        default_: node.default_,
        unsafe_: node.unsafe_,
        impl_: node.impl_,
        gens: f.fold_generics(node.gens),
        trait_: (node.trait_).map(|it| ((it).0, f.fold_path((it).1), (it).2)),
        typ: Box::new(f.fold_type(*node.typ)),
        brace: node.brace,
        items: FoldHelper::lift(node.items, |it| f.fold_impl_item(it)),
    }
}
pub fn fold_item_macro<F>(f: &mut F, node: ItemMacro) -> ItemMacro
where
    F: Fold + ?Sized,
{
    ItemMacro {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        ident: (node.ident).map(|it| f.fold_ident(it)),
        mac: f.fold_macro(node.mac),
        semi: node.semi,
    }
}
pub fn fold_item_mod<F>(f: &mut F, node: ItemMod) -> ItemMod
where
    F: Fold + ?Sized,
{
    ItemMod {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        unsafe_: node.unsafe_,
        mod_: node.mod_,
        ident: f.fold_ident(node.ident),
        gist: (node.gist).map(|it| ((it).0, FoldHelper::lift((it).1, |it| f.fold_item(it)))),
        semi: node.semi,
    }
}
pub fn fold_item_static<F>(f: &mut F, node: ItemStatic) -> ItemStatic
where
    F: Fold + ?Sized,
{
    ItemStatic {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        static_: node.static_,
        mut_: f.fold_static_mutability(node.mut_),
        ident: f.fold_ident(node.ident),
        colon: node.colon,
        typ: Box::new(f.fold_type(*node.typ)),
        eq: node.eq,
        expr: Box::new(f.fold_expr(*node.expr)),
        semi: node.semi,
    }
}
pub fn fold_item_struct<F>(f: &mut F, node: ItemStruct) -> ItemStruct
where
    F: Fold + ?Sized,
{
    ItemStruct {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        struct_: node.struct_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        fields: f.fold_fields(node.fields),
        semi: node.semi,
    }
}
pub fn fold_item_trait<F>(f: &mut F, node: ItemTrait) -> ItemTrait
where
    F: Fold + ?Sized,
{
    ItemTrait {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        unsafe_: node.unsafe_,
        auto_: node.auto_,
        restriction: (node.restriction).map(|it| f.fold_impl_restriction(it)),
        trait_: node.trait_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        colon: node.colon,
        supertraits: FoldHelper::lift(node.supertraits, |it| f.fold_type_param_bound(it)),
        brace: node.brace,
        items: FoldHelper::lift(node.items, |it| f.fold_trait_item(it)),
    }
}
pub fn fold_item_trait_alias<F>(f: &mut F, node: ItemTraitAlias) -> ItemTraitAlias
where
    F: Fold + ?Sized,
{
    ItemTraitAlias {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        trait_: node.trait_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        eq: node.eq,
        bounds: FoldHelper::lift(node.bounds, |it| f.fold_type_param_bound(it)),
        semi: node.semi,
    }
}
pub fn fold_item_type<F>(f: &mut F, node: ItemType) -> ItemType
where
    F: Fold + ?Sized,
{
    ItemType {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        type: node.type,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        eq: node.eq,
        typ: Box::new(f.fold_type(*node.typ)),
        semi: node.semi,
    }
}
pub fn fold_item_union<F>(f: &mut F, node: ItemUnion) -> ItemUnion
where
    F: Fold + ?Sized,
{
    ItemUnion {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        union_: node.union_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        fields: f.fold_fields_named(node.fields),
    }
}
pub fn fold_item_use<F>(f: &mut F, node: ItemUse) -> ItemUse
where
    F: Fold + ?Sized,
{
    ItemUse {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        use_: node.use_,
        leading_colon: node.leading_colon,
        tree: f.fold_use_tree(node.tree),
        semi: node.semi,
    }
}
pub fn fold_label<F>(f: &mut F, node: Label) -> Label
where
    F: Fold + ?Sized,
{
    Label {
        name: f.fold_lifetime(node.name),
        colon: node.colon,
    }
}
pub fn fold_lifetime<F>(f: &mut F, node: Lifetime) -> Lifetime
where
    F: Fold + ?Sized,
{
    Lifetime {
        apostrophe: f.fold_span(node.apostrophe),
        ident: f.fold_ident(node.ident),
    }
}
pub fn fold_lifetime_param<F>(f: &mut F, node: LifetimeParam) -> LifetimeParam
where
    F: Fold + ?Sized,
{
    LifetimeParam {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        life: f.fold_lifetime(node.life),
        colon: node.colon,
        bounds: FoldHelper::lift(node.bounds, |it| f.fold_lifetime(it)),
    }
}
pub fn fold_lit<F>(f: &mut F, node: Lit) -> Lit
where
    F: Fold + ?Sized,
{
    match node {
        Lit::Str(_binding_0) => Lit::Str(f.fold_lit_str(_binding_0)),
        Lit::ByteStr(_binding_0) => Lit::ByteStr(f.fold_lit_byte_str(_binding_0)),
        Lit::Byte(_binding_0) => Lit::Byte(f.fold_lit_byte(_binding_0)),
        Lit::Char(_binding_0) => Lit::Char(f.fold_lit_char(_binding_0)),
        Lit::Int(_binding_0) => Lit::Int(f.fold_lit_int(_binding_0)),
        Lit::Float(_binding_0) => Lit::Float(f.fold_lit_float(_binding_0)),
        Lit::Bool(_binding_0) => Lit::Bool(f.fold_lit_bool(_binding_0)),
        Lit::Verbatim(_binding_0) => Lit::Verbatim(_binding_0),
    }
}
pub fn fold_lit_bool<F>(f: &mut F, node: lit::Bool) -> lit::Bool
where
    F: Fold + ?Sized,
{
    lit::Bool {
        val: node.val,
        span: f.fold_span(node.span),
    }
}
pub fn fold_lit_byte<F>(f: &mut F, node: lit::Byte) -> lit::Byte
where
    F: Fold + ?Sized,
{
    let span = f.fold_span(node.span());
    let mut node = node;
    node.set_span(span);
    node
}
pub fn fold_lit_byte_str<F>(f: &mut F, node: lit::ByteStr) -> lit::ByteStr
where
    F: Fold + ?Sized,
{
    let span = f.fold_span(node.span());
    let mut node = node;
    node.set_span(span);
    node
}
pub fn fold_lit_char<F>(f: &mut F, node: lit::Char) -> lit::Char
where
    F: Fold + ?Sized,
{
    let span = f.fold_span(node.span());
    let mut node = node;
    node.set_span(span);
    node
}
pub fn fold_lit_float<F>(f: &mut F, node: lit::Float) -> lit::Float
where
    F: Fold + ?Sized,
{
    let span = f.fold_span(node.span());
    let mut node = node;
    node.set_span(span);
    node
}
pub fn fold_lit_int<F>(f: &mut F, node: lit::Int) -> lit::Int
where
    F: Fold + ?Sized,
{
    let span = f.fold_span(node.span());
    let mut node = node;
    node.set_span(span);
    node
}
pub fn fold_lit_str<F>(f: &mut F, node: lit::Str) -> lit::Str
where
    F: Fold + ?Sized,
{
    let span = f.fold_span(node.span());
    let mut node = node;
    node.set_span(span);
    node
}
pub fn fold_local<F>(f: &mut F, node: stmt::Local) -> stmt::Local
where
    F: Fold + ?Sized,
{
    stmt::Local {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        let_: node.let_,
        pat: f.fold_pat(node.pat),
        init: (node.init).map(|it| f.fold_local_init(it)),
        semi: node.semi,
    }
}
pub fn fold_local_init<F>(f: &mut F, node: stmt::LocalInit) -> stmt::LocalInit
where
    F: Fold + ?Sized,
{
    stmt::LocalInit {
        eq: node.eq,
        expr: Box::new(f.fold_expr(*node.expr)),
        diverge: (node.diverge).map(|it| ((it).0, Box::new(f.fold_expr(*(it).1)))),
    }
}
pub fn fold_macro<F>(f: &mut F, node: Macro) -> Macro
where
    F: Fold + ?Sized,
{
    Macro {
        path: f.fold_path(node.path),
        bang: node.bang,
        delim: f.fold_macro_delimiter(node.delim),
        toks: node.toks,
    }
}
pub fn fold_macro_delimiter<F>(f: &mut F, node: MacroDelim) -> MacroDelim
where
    F: Fold + ?Sized,
{
    match node {
        MacroDelim::Paren(_binding_0) => MacroDelim::Paren(_binding_0),
        MacroDelim::Brace(_binding_0) => MacroDelim::Brace(_binding_0),
        MacroDelim::Bracket(_binding_0) => MacroDelim::Bracket(_binding_0),
    }
}
pub fn fold_member<F>(f: &mut F, node: Member) -> Member
where
    F: Fold + ?Sized,
{
    match node {
        Member::Named(_binding_0) => Member::Named(f.fold_ident(_binding_0)),
        Member::Unnamed(_binding_0) => Member::Unnamed(f.fold_index(_binding_0)),
    }
}
pub fn fold_meta<F>(f: &mut F, node: Meta) -> Meta
where
    F: Fold + ?Sized,
{
    match node {
        Meta::Path(_binding_0) => Meta::Path(f.fold_path(_binding_0)),
        Meta::List(_binding_0) => Meta::List(f.fold_meta_list(_binding_0)),
        Meta::NameValue(_binding_0) => Meta::NameValue(f.fold_meta_name_value(_binding_0)),
    }
}
pub fn fold_meta_list<F>(f: &mut F, node: MetaList) -> MetaList
where
    F: Fold + ?Sized,
{
    MetaList {
        path: f.fold_path(node.path),
        delim: f.fold_macro_delimiter(node.delim),
        toks: node.toks,
    }
}
pub fn fold_meta_name_value<F>(f: &mut F, node: MetaNameValue) -> MetaNameValue
where
    F: Fold + ?Sized,
{
    MetaNameValue {
        path: f.fold_path(node.path),
        eq: node.eq,
        val: f.fold_expr(node.val),
    }
}
pub fn fold_parenthesized_generic_arguments<F>(f: &mut F, node: ParenthesizedArgs) -> ParenthesizedArgs
where
    F: Fold + ?Sized,
{
    ParenthesizedArgs {
        paren: node.paren,
        ins: FoldHelper::lift(node.ins, |it| f.fold_type(it)),
        out: f.fold_return_type(node.out),
    }
}
pub fn fold_pat<F>(f: &mut F, node: patt::Patt) -> patt::Patt
where
    F: Fold + ?Sized,
{
    match node {
        patt::Patt::Const(_binding_0) => patt::Patt::Const(f.fold_expr_const(_binding_0)),
        patt::Patt::Ident(_binding_0) => patt::Patt::Ident(f.fold_pat_ident(_binding_0)),
        patt::Patt::Lit(_binding_0) => patt::Patt::Lit(f.fold_expr_lit(_binding_0)),
        patt::Patt::Mac(_binding_0) => patt::Patt::Mac(f.fold_expr_macro(_binding_0)),
        patt::Patt::Or(_binding_0) => patt::Patt::Or(f.fold_pat_or(_binding_0)),
        patt::Patt::Paren(_binding_0) => patt::Patt::Paren(f.fold_pat_paren(_binding_0)),
        patt::Patt::Path(_binding_0) => patt::Patt::Path(f.fold_expr_path(_binding_0)),
        patt::Patt::Range(_binding_0) => patt::Patt::Range(f.fold_expr_range(_binding_0)),
        patt::Patt::Ref(_binding_0) => patt::Patt::Ref(f.fold_pat_reference(_binding_0)),
        patt::Patt::Rest(_binding_0) => patt::Patt::Rest(f.fold_pat_rest(_binding_0)),
        patt::Patt::Slice(_binding_0) => patt::Patt::Slice(f.fold_pat_slice(_binding_0)),
        patt::Patt::Struct(_binding_0) => patt::Patt::Struct(f.fold_pat_struct(_binding_0)),
        patt::Patt::Tuple(_binding_0) => patt::Patt::Tuple(f.fold_pat_tuple(_binding_0)),
        patt::Patt::TupleStruct(_binding_0) => patt::Patt::TupleStruct(f.fold_pat_tuple_struct(_binding_0)),
        patt::Patt::Type(_binding_0) => patt::Patt::Type(f.fold_pat_type(_binding_0)),
        patt::Patt::Verbatim(_binding_0) => patt::Patt::Verbatim(_binding_0),
        patt::Patt::Wild(_binding_0) => patt::Patt::Wild(f.fold_pat_wild(_binding_0)),
    }
}
pub fn fold_pat_ident<F>(f: &mut F, node: patt::Ident) -> patt::Ident
where
    F: Fold + ?Sized,
{
    patt::Ident {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        ref_: node.ref_,
        mut_: node.mut_,
        ident: f.fold_ident(node.ident),
        sub: (node.sub).map(|it| ((it).0, Box::new(f.fold_pat(*(it).1)))),
    }
}
pub fn fold_pat_or<F>(f: &mut F, node: patt::Or) -> patt::Or
where
    F: Fold + ?Sized,
{
    patt::Or {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vert: node.vert,
        cases: FoldHelper::lift(node.cases, |it| f.fold_pat(it)),
    }
}
pub fn fold_pat_paren<F>(f: &mut F, node: patt::Paren) -> patt::Paren
where
    F: Fold + ?Sized,
{
    patt::Paren {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        paren: node.paren,
        patt: Box::new(f.fold_pat(*node.patt)),
    }
}
pub fn fold_pat_reference<F>(f: &mut F, node: patt::Ref) -> patt::Ref
where
    F: Fold + ?Sized,
{
    patt::Ref {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        and: node.and,
        mut_: node.mut_,
        patt: Box::new(f.fold_pat(*node.patt)),
    }
}
pub fn fold_pat_rest<F>(f: &mut F, node: patt::Restest) patt::Rest::Rest
where
    F: Fold + ?Sized,
{
    patt::Restest {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        dot2: node.dot2,
    }
}
pub fn fold_pat_slice<F>(f: &mut F, node: patt::Slice) -> patt::Slice
where
    F: Fold + ?Sized,
{
    patt::Slice {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        bracket: node.bracket,
        elems: FoldHelper::lift(node.elems, |it| f.fold_pat(it)),
    }
}
pub fn fold_pat_struct<F>(f: &mut F, node: patt::Struct) -> patt::Struct
where
    F: Fold + ?Sized,
{
    patt::Struct {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        qself: (node.qself).map(|it| f.fold_qself(it)),
        path: f.fold_path(node.path),
        brace: node.brace,
        fields: FoldHelper::lift(node.fields, |it| f.fold_field_pat(it)),
        rest: (node.rest).map(|it| f.fold_pat_rest(it)),
    }
}
pub fn fold_pat_tuple<F>(f: &mut F, node: patt::Tuple) -> patt::Tuple
where
    F: Fold + ?Sized,
{
    patt::Tuple {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        paren: node.paren,
        elems: FoldHelper::lift(node.elems, |it| f.fold_pat(it)),
    }
}
pub fn fold_pat_tuple_struct<F>(f: &mut F, node: patt::TupleStructuct) patt::TupleStructStruct
where
    F: Fold + ?Sized,
{
    patt::TupleStructuct {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        qself: (node.qself).map(|it| f.fold_qself(it)),
        path: f.fold_path(node.path),
        paren: node.paren,
        elems: FoldHelper::lift(node.elems, |it| f.fold_pat(it)),
    }
}
pub fn fold_pat_type<F>(f: &mut F, node: patt::Type) -> patt::Type
where
    F: Fold + ?Sized,
{
    patt::Type {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        patt: Box::new(f.fold_pat(*node.patt)),
        colon: node.colon,
        typ: Box::new(f.fold_type(*node.typ)),
    }
}
pub fn fold_pat_wild<F>(f: &mut F, node: patt::Wild) -> patt::Wild
where
    F: Fold + ?Sized,
{
    patt::Wild {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        underscore: node.underscore,
    }
}
pub fn fold_path<F>(f: &mut F, node: Path) -> Path
where
    F: Fold + ?Sized,
{
    Path {
        colon: node.colon,
        segs: FoldHelper::lift(node.segs, |it| f.fold_path_segment(it)),
    }
}
pub fn fold_path_arguments<F>(f: &mut F, node: Args) -> Args
where
    F: Fold + ?Sized,
{
    match node {
        Args::None => Args::None,
        Args::Angled(_binding_0) => Args::Angled(f.fold_angle_bracketed_generic_arguments(_binding_0)),
        Args::Parenthesized(_binding_0) => Args::Parenthesized(f.fold_parenthesized_generic_arguments(_binding_0)),
    }
}
pub fn fold_path_segment<F>(f: &mut F, node: Segment) -> Segment
where
    F: Fold + ?Sized,
{
    Segment {
        ident: f.fold_ident(node.ident),
        args: f.fold_path_arguments(node.args),
    }
}
pub fn fold_predicate_lifetime<F>(f: &mut F, node: PredLifetime) -> PredLifetime
where
    F: Fold + ?Sized,
{
    PredLifetime {
        life: f.fold_lifetime(node.life),
        colon: node.colon,
        bounds: FoldHelper::lift(node.bounds, |it| f.fold_lifetime(it)),
    }
}
pub fn fold_predicate_type<F>(f: &mut F, node: PredType) -> PredType
where
    F: Fold + ?Sized,
{
    PredType {
        lifes: (node.lifes).map(|it| f.fold_bound_lifetimes(it)),
        bounded: f.fold_type(node.bounded),
        colon: node.colon,
        bounds: FoldHelper::lift(node.bounds, |it| f.fold_type_param_bound(it)),
    }
}
pub fn fold_qself<F>(f: &mut F, node: QSelf) -> QSelf
where
    F: Fold + ?Sized,
{
    QSelf {
        lt: node.lt,
        ty: Box::new(f.fold_type(*node.ty)),
        pos: node.pos,
        as_: node.as_,
        gt_: node.gt_,
    }
}
pub fn fold_range_limits<F>(f: &mut F, node: RangeLimits) -> RangeLimits
where
    F: Fold + ?Sized,
{
    match node {
        RangeLimits::HalfOpen(_binding_0) => RangeLimits::HalfOpen(_binding_0),
        RangeLimits::Closed(_binding_0) => RangeLimits::Closed(_binding_0),
    }
}
pub fn fold_receiver<F>(f: &mut F, node: Receiver) -> Receiver
where
    F: Fold + ?Sized,
{
    Receiver {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        reference: (node.reference).map(|it| ((it).0, ((it).1).map(|it| f.fold_lifetime(it)))),
        mut_: node.mut_,
        self_: node.self_,
        colon: node.colon,
        typ: Box::new(f.fold_type(*node.typ)),
    }
}
pub fn fold_return_type<F>(f: &mut F, node: ty::Ret) -> ty::Ret
where
    F: Fold + ?Sized,
{
    match node {
        ty::Ret::Default => ty::Ret::Default,
        ty::Ret::Type(_binding_0, _binding_1) => ty::Ret::Type(_binding_0, Box::new(f.fold_type(*_binding_1))),
    }
}
pub fn fold_signature<F>(f: &mut F, node: Signature) -> Signature
where
    F: Fold + ?Sized,
{
    Signature {
        constness: node.constness,
        async_: node.async_,
        unsafe_: node.unsafe_,
        abi: (node.abi).map(|it| f.fold_abi(it)),
        fn_: node.fn_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        paren: node.paren,
        args: FoldHelper::lift(node.args, |it| f.fold_fn_arg(it)),
        vari: (node.vari).map(|it| f.fold_variadic(it)),
        ret: f.fold_return_type(node.ret),
    }
}
pub fn fold_span<F>(f: &mut F, node: Span) -> Span
where
    F: Fold + ?Sized,
{
    node
}
pub fn fold_static_mutability<F>(f: &mut F, node: StaticMut) -> StaticMut
where
    F: Fold + ?Sized,
{
    match node {
        StaticMut::Mut(_binding_0) => StaticMut::Mut(_binding_0),
        StaticMut::None => StaticMut::None,
    }
}
pub fn fold_stmt<F>(f: &mut F, node: stmt::Stmt) -> stmt::Stmt
where
    F: Fold + ?Sized,
{
    match node {
        stmt::Stmt::stmt::Local(_binding_0) => stmt::Stmt::stmt::Local(f.fold_local(_binding_0)),
        stmt::Stmt::Item(_binding_0) => stmt::Stmt::Item(f.fold_item(_binding_0)),
        stmt::Stmt::Expr(_binding_0, _binding_1) => stmt::Stmt::Expr(f.fold_expr(_binding_0), _binding_1),
        stmt::Stmt::Mac(_binding_0) => stmt::Stmt::Mac(f.fold_stmt_macro(_binding_0)),
    }
}
pub fn fold_stmt_macro<F>(f: &mut F, node: stmt::Mac) -> stmt::Mac
where
    F: Fold + ?Sized,
{
    stmt::Mac {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        mac: f.fold_macro(node.mac),
        semi: node.semi,
    }
}
pub fn fold_trait_bound<F>(f: &mut F, node: TraitBound) -> TraitBound
where
    F: Fold + ?Sized,
{
    TraitBound {
        paren: node.paren,
        modifier: f.fold_trait_bound_modifier(node.modifier),
        lifes: (node.lifes).map(|it| f.fold_bound_lifetimes(it)),
        path: f.fold_path(node.path),
    }
}
pub fn fold_trait_bound_modifier<F>(f: &mut F, node: TraitBoundModifier) -> TraitBoundModifier
where
    F: Fold + ?Sized,
{
    match node {
        TraitBoundModifier::None => TraitBoundModifier::None,
        TraitBoundModifier::Maybe(_binding_0) => TraitBoundModifier::Maybe(_binding_0),
    }
}
pub fn fold_trait_item<F>(f: &mut F, node: TraitItem) -> TraitItem
where
    F: Fold + ?Sized,
{
    match node {
        TraitItem::Const(_binding_0) => TraitItem::Const(f.fold_trait_item_const(_binding_0)),
        TraitItem::Fn(_binding_0) => TraitItem::Fn(f.fold_trait_item_fn(_binding_0)),
        TraitItem::Type(_binding_0) => TraitItem::Type(f.fold_trait_item_type(_binding_0)),
        TraitItem::Macro(_binding_0) => TraitItem::Macro(f.fold_trait_item_macro(_binding_0)),
        TraitItem::Verbatim(_binding_0) => TraitItem::Verbatim(_binding_0),
    }
}
pub fn fold_trait_item_const<F>(f: &mut F, node: TraitItemConst) -> TraitItemConst
where
    F: Fold + ?Sized,
{
    TraitItemConst {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        const_: node.const_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        colon: node.colon,
        typ: f.fold_type(node.typ),
        default: (node.default).map(|it| ((it).0, f.fold_expr((it).1))),
        semi: node.semi,
    }
}
pub fn fold_trait_item_fn<F>(f: &mut F, node: TraitItemFn) -> TraitItemFn
where
    F: Fold + ?Sized,
{
    TraitItemFn {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        sig: f.fold_signature(node.sig),
        default: (node.default).map(|it| f.fold_block(it)),
        semi: node.semi,
    }
}
pub fn fold_trait_item_macro<F>(f: &mut F, node: TraitItemMacro) -> TraitItemMacro
where
    F: Fold + ?Sized,
{
    TraitItemMacro {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        mac: f.fold_macro(node.mac),
        semi: node.semi,
    }
}
pub fn fold_trait_item_type<F>(f: &mut F, node: TraitItemType) -> TraitItemType
where
    F: Fold + ?Sized,
{
    TraitItemType {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        type: node.type,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        colon: node.colon,
        bounds: FoldHelper::lift(node.bounds, |it| f.fold_type_param_bound(it)),
        default: (node.default).map(|it| ((it).0, f.fold_type((it).1))),
        semi: node.semi,
    }
}
pub fn fold_type<F>(f: &mut F, node: ty::Type) -> ty::Type
where
    F: Fold + ?Sized,
{
    match node {
        ty::Type::Array(_binding_0) => ty::Type::Array(f.fold_type_array(_binding_0)),
        ty::Type::BareFn(_binding_0) => ty::Type::BareFn(f.fold_type_bare_fn(_binding_0)),
        ty::Type::Group(_binding_0) => ty::Type::Group(f.fold_type_group(_binding_0)),
        ty::Type::Impl(_binding_0) => ty::Type::Impl(f.fold_type_impl_trait(_binding_0)),
        ty::Type::Infer(_binding_0) => ty::Type::Infer(f.fold_type_infer(_binding_0)),
        ty::Type::Mac(_binding_0) => ty::Type::Mac(f.fold_type_macro(_binding_0)),
        ty::Type::Never(_binding_0) => ty::Type::Never(f.fold_type_never(_binding_0)),
        ty::Type::Paren(_binding_0) => ty::Type::Paren(f.fold_type_paren(_binding_0)),
        ty::Type::Path(_binding_0) => ty::Type::Path(f.fold_type_path(_binding_0)),
        ty::Type::Ptr(_binding_0) => ty::Type::Ptr(f.fold_type_ptr(_binding_0)),
        ty::Type::Ref(_binding_0) => ty::Type::Ref(f.fold_type_reference(_binding_0)),
        ty::Type::Slice(_binding_0) => ty::Type::Slice(f.fold_type_slice(_binding_0)),
        ty::Type::TraitObj(_binding_0) => ty::Type::TraitObj(f.fold_type_trait_object(_binding_0)),
        ty::Type::Tuple(_binding_0) => ty::Type::Tuple(f.fold_type_tuple(_binding_0)),
        ty::Type::Verbatim(_binding_0) => ty::Type::Verbatim(_binding_0),
    }
}
pub fn fold_type_array<F>(f: &mut F, node: ty::Array) -> ty::Array
where
    F: Fold + ?Sized,
{
    ty::Array {
        bracket: node.bracket,
        elem: Box::new(f.fold_type(*node.elem)),
        semi: node.semi,
        len: f.fold_expr(node.len),
    }
}
pub fn fold_type_bare_fn<F>(f: &mut F, node: ty::BareFn) -> ty::BareFn
where
    F: Fold + ?Sized,
{
    ty::BareFn {
        lifes: (node.lifes).map(|it| f.fold_bound_lifetimes(it)),
        unsafe_: node.unsafe_,
        abi: (node.abi).map(|it| f.fold_abi(it)),
        fn_: node.fn_,
        paren: node.paren,
        args: FoldHelper::lift(node.args, |it| f.fold_bare_fn_arg(it)),
        vari: (node.vari).map(|it| f.fold_bare_variadic(it)),
        ret: f.fold_return_type(node.ret),
    }
}
pub fn fold_type_group<F>(f: &mut F, node: ty::Group) -> ty::Group
where
    F: Fold + ?Sized,
{
    ty::Group {
        group: node.group,
        elem: Box::new(f.fold_type(*node.elem)),
    }
}
pub fn fold_type_impl_trait<F>(f: &mut F, node: ty::Impl) -> ty::Impl
where
    F: Fold + ?Sized,
{
    ty::Impl {
        impl_: node.impl_,
        bounds: FoldHelper::lift(node.bounds, |it| f.fold_type_param_bound(it)),
    }
}
pub fn fold_type_infer<F>(f: &mut F, node: ty::Infer) -> ty::Infer
where
    F: Fold + ?Sized,
{
    ty::Infer {
        underscore: node.underscore,
    }
}
pub fn fold_type_macro<F>(f: &mut F, node: ty::Mac) -> ty::Mac
where
    F: Fold + ?Sized,
{
    ty::Mac {
        mac: f.fold_macro(node.mac),
    }
}
pub fn fold_type_never<F>(f: &mut F, node: ty::Never) -> ty::Never
where
    F: Fold + ?Sized,
{
    ty::Never { bang: node.bang }
}
pub fn fold_type_param<F>(f: &mut F, node: TypeParam) -> TypeParam
where
    F: Fold + ?Sized,
{
    TypeParam {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        ident: f.fold_ident(node.ident),
        colon: node.colon,
        bounds: FoldHelper::lift(node.bounds, |it| f.fold_type_param_bound(it)),
        eq: node.eq,
        default: (node.default).map(|it| f.fold_type(it)),
    }
}
pub fn fold_type_param_bound<F>(f: &mut F, node: TypeParamBound) -> TypeParamBound
where
    F: Fold + ?Sized,
{
    match node {
        TypeParamBound::Trait(_binding_0) => TypeParamBound::Trait(f.fold_trait_bound(_binding_0)),
        TypeParamBound::Lifetime(_binding_0) => TypeParamBound::Lifetime(f.fold_lifetime(_binding_0)),
        TypeParamBound::Verbatim(_binding_0) => TypeParamBound::Verbatim(_binding_0),
    }
}
pub fn fold_type_paren<F>(f: &mut F, node: ty::Paren) -> ty::Paren
where
    F: Fold + ?Sized,
{
    ty::Paren {
        paren: node.paren,
        elem: Box::new(f.fold_type(*node.elem)),
    }
}
pub fn fold_type_path<F>(f: &mut F, node: ty::Path) -> ty::Path
where
    F: Fold + ?Sized,
{
    ty::Path {
        qself: (node.qself).map(|it| f.fold_qself(it)),
        path: f.fold_path(node.path),
    }
}
pub fn fold_type_ptr<F>(f: &mut F, node: ty::Ptr) -> ty::Ptr
where
    F: Fold + ?Sized,
{
    ty::Ptr {
        star: node.star,
        const_: node.const_,
        mut_: node.mut_,
        elem: Box::new(f.fold_type(*node.elem)),
    }
}
pub fn fold_type_reference<F>(f: &mut F, node: ty::Ref) -> ty::Ref
where
    F: Fold + ?Sized,
{
    ty::Ref {
        and: node.and,
        life: (node.life).map(|it| f.fold_lifetime(it)),
        mut_: node.mut_,
        elem: Box::new(f.fold_type(*node.elem)),
    }
}
pub fn fold_type_slice<F>(f: &mut F, node: ty::Slice) -> ty::Slice
where
    F: Fold + ?Sized,
{
    ty::Slice {
        bracket: node.bracket,
        elem: Box::new(f.fold_type(*node.elem)),
    }
}
pub fn fold_type_trait_object<F>(f: &mut F, node: ty::TraitObj) -> ty::TraitObj
where
    F: Fold + ?Sized,
{
    ty::TraitObj {
        dyn_: node.dyn_,
        bounds: FoldHelper::lift(node.bounds, |it| f.fold_type_param_bound(it)),
    }
}
pub fn fold_type_tuple<F>(f: &mut F, node: ty::Tuple) -> ty::Tuple
where
    F: Fold + ?Sized,
{
    ty::Tuple {
        paren: node.paren,
        elems: FoldHelper::lift(node.elems, |it| f.fold_type(it)),
    }
}
pub fn fold_un_op<F>(f: &mut F, node: UnOp) -> UnOp
where
    F: Fold + ?Sized,
{
    match node {
        UnOp::Deref(_binding_0) => UnOp::Deref(_binding_0),
        UnOp::Not(_binding_0) => UnOp::Not(_binding_0),
        UnOp::Neg(_binding_0) => UnOp::Neg(_binding_0),
    }
}
pub fn fold_use_glob<F>(f: &mut F, node: UseGlob) -> UseGlob
where
    F: Fold + ?Sized,
{
    UseGlob {
        star: node.star,
    }
}
pub fn fold_use_group<F>(f: &mut F, node: UseGroup) -> UseGroup
where
    F: Fold + ?Sized,
{
    UseGroup {
        brace: node.brace,
        items: FoldHelper::lift(node.items, |it| f.fold_use_tree(it)),
    }
}
pub fn fold_use_name<F>(f: &mut F, node: UseName) -> UseName
where
    F: Fold + ?Sized,
{
    UseName {
        ident: f.fold_ident(node.ident),
    }
}
pub fn fold_use_path<F>(f: &mut F, node: UsePath) -> UsePath
where
    F: Fold + ?Sized,
{
    UsePath {
        ident: f.fold_ident(node.ident),
        colon2: node.colon2,
        tree: Box::new(f.fold_use_tree(*node.tree)),
    }
}
pub fn fold_use_rename<F>(f: &mut F, node: UseRename) -> UseRename
where
    F: Fold + ?Sized,
{
    UseRename {
        ident: f.fold_ident(node.ident),
        as_: node.as_,
        rename: f.fold_ident(node.rename),
    }
}
pub fn fold_use_tree<F>(f: &mut F, node: UseTree) -> UseTree
where
    F: Fold + ?Sized,
{
    match node {
        UseTree::Path(_binding_0) => UseTree::Path(f.fold_use_path(_binding_0)),
        UseTree::Name(_binding_0) => UseTree::Name(f.fold_use_name(_binding_0)),
        UseTree::Rename(_binding_0) => UseTree::Rename(f.fold_use_rename(_binding_0)),
        UseTree::Glob(_binding_0) => UseTree::Glob(f.fold_use_glob(_binding_0)),
        UseTree::Group(_binding_0) => UseTree::Group(f.fold_use_group(_binding_0)),
    }
}
pub fn fold_variadic<F>(f: &mut F, node: Variadic) -> Variadic
where
    F: Fold + ?Sized,
{
    Variadic {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        pat: (node.pat).map(|it| (Box::new(f.fold_pat(*(it).0)), (it).1)),
        dots: node.dots,
        comma: node.comma,
    }
}
pub fn fold_variant<F>(f: &mut F, node: Variant) -> Variant
where
    F: Fold + ?Sized,
{
    Variant {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        ident: f.fold_ident(node.ident),
        fields: f.fold_fields(node.fields),
        discriminant: (node.discriminant).map(|it| ((it).0, f.fold_expr((it).1))),
    }
}
pub fn fold_vis_restricted<F>(f: &mut F, node: VisRestricted) -> VisRestricted
where
    F: Fold + ?Sized,
{
    VisRestricted {
        pub_: node.pub_,
        paren: node.paren,
        in_: node.in_,
        path: Box::new(f.fold_path(*node.path)),
    }
}
pub fn fold_visibility<F>(f: &mut F, node: Visibility) -> Visibility
where
    F: Fold + ?Sized,
{
    match node {
        Visibility::Public(_binding_0) => Visibility::Public(_binding_0),
        Visibility::Restricted(_binding_0) => Visibility::Restricted(f.fold_vis_restricted(_binding_0)),
        Visibility::Inherited => Visibility::Inherited,
    }
}
pub fn fold_where_clause<F>(f: &mut F, node: WhereClause) -> WhereClause
where
    F: Fold + ?Sized,
{
    WhereClause {
        where_: node.where_,
        preds: FoldHelper::lift(node.preds, |it| f.fold_where_predicate(it)),
    }
}
pub fn fold_where_predicate<F>(f: &mut F, node: WherePred) -> WherePred
where
    F: Fold + ?Sized,
{
    match node {
        WherePred::Lifetime(_binding_0) => WherePred::Lifetime(f.fold_predicate_lifetime(_binding_0)),
        WherePred::Type(_binding_0) => WherePred::Type(f.fold_predicate_type(_binding_0)),
    }
}
