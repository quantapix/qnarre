#![allow(unreachable_code, unused_variables)]
#![allow(clippy::match_wildcard_for_single_variants, clippy::needless_match)]

use crate::*;

macro_rules! full {
    ($e:expr) => {
        $e
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
    fn fold_attr_style(&mut self, i: attr::Style) -> attr::Style {
        fold_attr_style(self, i)
    }
    fn fold_attribute(&mut self, i: attr::Attr) -> attr::Attr {
        fold_attribute(self, i)
    }
    fn fold_bare_fn_arg(&mut self, i: typ::FnArg) -> typ::FnArg {
        fold_bare_fn_arg(self, i)
    }
    fn fold_bare_variadic(&mut self, i: typ::Variadic) -> typ::Variadic {
        fold_bare_variadic(self, i)
    }
    fn fold_bin_op(&mut self, i: BinOp) -> BinOp {
        fold_bin_op(self, i)
    }
    fn fold_block(&mut self, i: Block) -> Block {
        fold_block(self, i)
    }
    fn fold_bound_lifetimes(&mut self, i: Bgen::bound::Lifes) -> Bgen::bound::Lifes {
        fold_bound_lifetimes(self, i)
    }
    fn fold_const_param(&mut self, i: gen::param::Const) -> gen::param::Const {
        fold_const_param(self, i)
    }
    fn fold_constraint(&mut self, i: Constraint) -> Constraint {
        fold_constraint(self, i)
    }
    fn fold_data(&mut self, i: Data) -> Data {
        fold_data(self, i)
    }
    fn fold_data_enum(&mut self, i: data::Enum) -> data::Enum {
        fold_data_enum(self, i)
    }
    fn fold_data_struct(&mut self, i: data::Struct) -> data::Struct {
        fold_data_struct(self, i)
    }
    fn fold_data_union(&mut self, i: data::Union) -> data::Union {
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
    fn fold_field_mutability(&mut self, i: data::Mut) -> data::Mut {
        fold_field_mutability(self, i)
    }
    fn fold_field_pat(&mut self, i: pat::Fieldeld) -> pat::Field {
        fold_field_pat(self, i)
    }
    fn fold_field_value(&mut self, i: FieldValue) -> FieldValue {
        fold_field_value(self, i)
    }
    fn fold_fields(&mut self, i: data::Fields) -> data::Fields {
        fold_fields(self, i)
    }
    fn fold_fields_named(&mut self, i: data::Named) -> data::Named {
        fold_fields_named(self, i)
    }
    fn fold_fields_unnamed(&mut self, i: data::Unnamed) -> data::Unnamed {
        fold_fields_unnamed(self, i)
    }
    fn fold_file(&mut self, i: item::File) -> item::File {
        fold_file(self, i)
    }
    fn fold_fn_arg(&mut self, i: item::FnArg) -> item::FnArg {
        fold_fn_arg(self, i)
    }
    fn fold_foreign_item(&mut self, i: item::foreign::Item) -> item::foreign::Item {
        fold_foreign_item(self, i)
    }
    fn fold_foreign_item_fn(&mut self, i: item::foreign::Fn) -> item::foreign::Fn {
        fold_foreign_item_fn(self, i)
    }
    fn fold_foreign_item_macro(&mut self, i: item::foreign::Mac) -> item::foreign::Mac {
        fold_foreign_item_macro(self, i)
    }
    fn fold_foreign_item_static(&mut self, i: item::foreign::Static) -> item::foreign::Static {
        fold_foreign_item_static(self, i)
    }
    fn fold_foreign_item_type(&mut self, i: item::foreign::Type) -> item::foreign::Type {
        fold_foreign_item_type(self, i)
    }
    fn fold_generic_argument(&mut self, i: Arg) -> Arg {
        fold_generic_argument(self, i)
    }
    fn fold_generic_param(&mut self, i: gen::Param) -> gen::Param {
        fold_generic_param(self, i)
    }
    fn fold_generics(&mut self, i: gen::Gens) -> gen::Gens {
        fold_generics(self, i)
    }
    fn fold_ident(&mut self, i: Ident) -> Ident {
        fold_ident(self, i)
    }
    fn fold_impl_item(&mut self, i: item::impl_::Item) -> item::impl_::Item {
        fold_impl_item(self, i)
    }
    fn fold_impl_item_const(&mut self, i: item::impl_::Const) -> item::impl_::Const {
        fold_impl_item_const(self, i)
    }
    fn fold_impl_item_fn(&mut self, i: item::impl_::Fn) -> item::impl_::Fn {
        fold_impl_item_fn(self, i)
    }
    fn fold_impl_item_macro(&mut self, i: item::impl_::Mac) -> item::impl_::Mac {
        fold_impl_item_macro(self, i)
    }
    fn fold_impl_item_type(&mut self, i: item::impl_::Type) -> item::impl_::Type {
        fold_impl_item_type(self, i)
    }
    fn fold_impl_restriction(&mut self, i: item::impl_::Restriction) -> item::impl_::Restriction {
        fold_impl_restriction(self, i)
    }
    fn fold_index(&mut self, i: Index) -> Index {
        fold_index(self, i)
    }
    fn fold_item(&mut self, i: Item) -> Item {
        fold_item(self, i)
    }
    fn fold_item_const(&mut self, i: item::Const) -> item::Const {
        fold_item_const(self, i)
    }
    fn fold_item_enum(&mut self, i: item::Enum) -> item::Enum {
        fold_item_enum(self, i)
    }
    fn fold_item_extern_crate(&mut self, i: item::Extern) -> item::Extern {
        fold_item_extern_crate(self, i)
    }
    fn fold_item_fn(&mut self, i: item::Fn) -> item::Fn {
        fold_item_fn(self, i)
    }
    fn fold_item_foreign_mod(&mut self, i: item::Foreign) -> item::Foreign {
        fold_item_foreign_mod(self, i)
    }
    fn fold_item_impl(&mut self, i: item::Impl) -> item::Impl {
        fold_item_impl(self, i)
    }
    fn fold_item_macro(&mut self, i: item::Mac) -> item::Mac {
        fold_item_macro(self, i)
    }
    fn fold_item_mod(&mut self, i: item::Mod) -> item::Mod {
        fold_item_mod(self, i)
    }
    fn fold_item_static(&mut self, i: item::Static) -> item::Static {
        fold_item_static(self, i)
    }
    fn fold_item_struct(&mut self, i: item::Struct) -> item::Struct {
        fold_item_struct(self, i)
    }
    fn fold_item_trait(&mut self, i: item::Trait) -> item::Trait {
        fold_item_trait(self, i)
    }
    fn fold_item_trait_alias(&mut self, i: item::TraitAlias) -> item::TraitAlias {
        fold_item_trait_alias(self, i)
    }
    fn fold_item_type(&mut self, i: item::Type) -> item::Type {
        fold_item_type(self, i)
    }
    fn fold_item_union(&mut self, i: item::Union) -> item::Union {
        fold_item_union(self, i)
    }
    fn fold_item_use(&mut self, i: item::Use) -> item::Use {
        fold_item_use(self, i)
    }
    fn fold_label(&mut self, i: Label) -> Label {
        fold_label(self, i)
    }
    fn fold_lifetime(&mut self, i: Life) -> Life {
        fold_lifetime(self, i)
    }
    fn fold_lifetime_param(&mut self, i: gen::param::Life) -> gen::param::Life {
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
    fn fold_local_init(&mut self, i: stmt::Init) -> stmt::Init {
        fold_local_init(self, i)
    }
    fn fold_macro(&mut self, i: Macro) -> Macro {
        fold_macro(self, i)
    }
    fn fold_macro_delimiter(&mut self, i: tok::Delim) -> tok::Delim {
        fold_macro_delimiter(self, i)
    }
    fn fold_member(&mut self, i: Member) -> Member {
        fold_member(self, i)
    }
    fn fold_meta(&mut self, i: attr::Meta) -> attr::Meta {
        fold_meta(self, i)
    }
    fn fold_meta_list(&mut self, i: attr::List) -> attr::List {
        fold_meta_list(self, i)
    }
    fn fold_meta_name_value(&mut self, i: attr::NameValue) -> attr::NameValue {
        fold_meta_name_value(self, i)
    }
    fn fold_parenthesized_generic_arguments(&mut self, i: ParenthesizedArgs) -> ParenthesizedArgs {
        fold_parenthesized_generic_arguments(self, i)
    }
    fn fold_pat(&mut self, i: pat::Pat) -> pat::Pat {
        fold_pat(self, i)
    }
    fn fold_pat_ident(&mut self, i: pat::Ident) -> pat::Ident {
        fold_pat_ident(self, i)
    }
    fn fold_pat_or(&mut self, i: pat::Or) -> pat::Or {
        fold_pat_or(self, i)
    }
    fn fold_pat_paren(&mut self, i: pat::Paren) -> pat::Paren {
        fold_pat_paren(self, i)
    }
    fn fold_pat_reference(&mut self, i: pat::Ref) -> pat::Ref {
        fold_pat_reference(self, i)
    }
    fn fold_pat_rest(&mut self, i: pat::Restest) -> pat::Rest {
        fold_pat_rest(self, i)
    }
    fn fold_pat_slice(&mut self, i: pat::Slice) -> pat::Slice {
        fold_pat_slice(self, i)
    }
    fn fold_pat_struct(&mut self, i: pat::Struct) -> pat::Struct {
        fold_pat_struct(self, i)
    }
    fn fold_pat_tuple(&mut self, i: pat::Tuple) -> pat::Tuple {
        fold_pat_tuple(self, i)
    }
    fn fold_pat_tuple_struct(&mut self, i: pat::TupleStructuct) -> pat::TupleStruct {
        fold_pat_tuple_struct(self, i)
    }
    fn fold_pat_type(&mut self, i: pat::Type) -> pat::Type {
        fold_pat_type(self, i)
    }
    fn fold_pat_wild(&mut self, i: pat::Wild) -> pat::Wild {
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
    fn fold_predicate_lifetime(&mut self, i: gen::Where::Life) -> gen::Where::Life {
        fold_predicate_lifetime(self, i)
    }
    fn fold_predicate_type(&mut self, i: gen::Where::Type) -> gen::Where::Type {
        fold_predicate_type(self, i)
    }
    fn fold_qself(&mut self, i: QSelf) -> QSelf {
        fold_qself(self, i)
    }
    fn fold_range_limits(&mut self, i: expr::Limits) -> expr::Limits {
        fold_range_limits(self, i)
    }
    fn fold_receiver(&mut self, i: item::Receiver) -> item::Receiver {
        fold_receiver(self, i)
    }
    fn fold_return_type(&mut self, i: typ::Ret) -> typ::Ret {
        fold_return_type(self, i)
    }
    fn fold_signature(&mut self, i: item::Sig) -> item::Sig {
        fold_signature(self, i)
    }
    fn fold_span(&mut self, i: pm2::Span) -> pm2::Span {
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
    fn fold_trait_bound(&mut self, i: gen::bound::Trait) -> gen::bound::Trait {
        fold_trait_bound(self, i)
    }
    fn fold_trait_bound_modifier(&mut self, i: gen::bound::Modifier) -> gen::bound::Modifier {
        fold_trait_bound_modifier(self, i)
    }
    fn fold_trait_item(&mut self, i: item::trait_::Item) -> item::trait_::Item {
        fold_trait_item(self, i)
    }
    fn fold_trait_item_const(&mut self, i: item::trait_::Const) -> item::trait_::Const {
        fold_trait_item_const(self, i)
    }
    fn fold_trait_item_fn(&mut self, i: item::trait_::Fn) -> item::trait_::Fn {
        fold_trait_item_fn(self, i)
    }
    fn fold_trait_item_macro(&mut self, i: item::trait_::Mac) -> item::trait_::Mac {
        fold_trait_item_macro(self, i)
    }
    fn fold_trait_item_type(&mut self, i: item::trait_::Type) -> item::trait_::Type {
        fold_trait_item_type(self, i)
    }
    fn fold_type(&mut self, i: typ::Type) -> typ::Type {
        fold_type(self, i)
    }
    fn fold_type_array(&mut self, i: typ::Array) -> typ::Array {
        fold_type_array(self, i)
    }
    fn fold_type_bare_fn(&mut self, i: typ::Fn) -> typ::Fn {
        fold_type_bare_fn(self, i)
    }
    fn fold_type_group(&mut self, i: typ::Group) -> typ::Group {
        fold_type_group(self, i)
    }
    fn fold_type_impl_trait(&mut self, i: typ::Impl) -> typ::Impl {
        fold_type_impl_trait(self, i)
    }
    fn fold_type_infer(&mut self, i: typ::Infer) -> typ::Infer {
        fold_type_infer(self, i)
    }
    fn fold_type_macro(&mut self, i: typ::Mac) -> typ::Mac {
        fold_type_macro(self, i)
    }
    fn fold_type_never(&mut self, i: typ::Never) -> typ::Never {
        fold_type_never(self, i)
    }
    fn fold_type_param(&mut self, i: gen::param::Type) -> gen::param::Type {
        fold_type_param(self, i)
    }
    fn fold_type_param_bound(&mut self, i: gen::bound::Type) -> gen::bound::Type {
        fold_type_param_bound(self, i)
    }
    fn fold_type_paren(&mut self, i: typ::Paren) -> typ::Paren {
        fold_type_paren(self, i)
    }
    fn fold_type_path(&mut self, i: typ::Path) -> typ::Path {
        fold_type_path(self, i)
    }
    fn fold_type_ptr(&mut self, i: typ::Ptr) -> typ::Ptr {
        fold_type_ptr(self, i)
    }
    fn fold_type_reference(&mut self, i: typ::Ref) -> typ::Ref {
        fold_type_reference(self, i)
    }
    fn fold_type_slice(&mut self, i: typ::Slice) -> typ::Slice {
        fold_type_slice(self, i)
    }
    fn fold_type_trait_object(&mut self, i: typ::Trait) -> typ::Trait {
        fold_type_trait_object(self, i)
    }
    fn fold_type_tuple(&mut self, i: typ::Tuple) -> typ::Tuple {
        fold_type_tuple(self, i)
    }
    fn fold_un_op(&mut self, i: UnOp) -> UnOp {
        fold_un_op(self, i)
    }
    fn fold_use_glob(&mut self, i: item::use_::Glob) -> item::use_::Glob {
        fold_use_glob(self, i)
    }
    fn fold_use_group(&mut self, i: item::use_::Group) -> item::use_::Group {
        fold_use_group(self, i)
    }
    fn fold_use_name(&mut self, i: item::use_::Name) -> item::use_::Name {
        fold_use_name(self, i)
    }
    fn fold_use_path(&mut self, i: item::use_::Path) -> item::use_::Path {
        fold_use_path(self, i)
    }
    fn fold_use_rename(&mut self, i: item::use_::Rename) -> item::use_::Rename {
        fold_use_rename(self, i)
    }
    fn fold_use_tree(&mut self, i: item::use_::Tree) -> item::use_::Tree {
        fold_use_tree(self, i)
    }
    fn fold_variadic(&mut self, i: item::Variadic) -> item::Variadic {
        fold_variadic(self, i)
    }
    fn fold_variant(&mut self, i: data::Variant) -> data::Variant {
        fold_variant(self, i)
    }
    fn fold_vis_restricted(&mut self, i: data::Restricted) -> data::Restricted {
        fold_vis_restricted(self, i)
    }
    fn fold_visibility(&mut self, i: data::Visibility) -> data::Visibility {
        fold_visibility(self, i)
    }
    fn fold_where_clause(&mut self, i: gen::Where) -> gen::Where {
        fold_where_clause(self, i)
    }
    fn fold_where_predicate(&mut self, i: gen::Where::Pred) -> gen::Where::Pred {
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
pub fn fold_assoc_const<F>(f: &mut F, node: path::AssocConst) -> path::AssocConst
where
    F: Fold + ?Sized,
{
    path::AssocConst {
        ident: f.fold_ident(node.ident),
        args: (node.args).map(|it| f.fold_angle_bracketed_generic_arguments(it)),
        eq: node.eq,
        val: f.fold_expr(node.val),
    }
}
pub fn fold_assoc_type<F>(f: &mut F, node: path::AssocType) -> path::AssocType
where
    F: Fold + ?Sized,
{
    path::AssocType {
        ident: f.fold_ident(node.ident),
        args: (node.args).map(|it| f.fold_angle_bracketed_generic_arguments(it)),
        eq: node.eq,
        typ: f.fold_type(node.typ),
    }
}
pub fn fold_attr_style<F>(f: &mut F, node: attr::Style) -> attr::Style
where
    F: Fold + ?Sized,
{
    match node {
        attr::Style::Outer => attr::Style::Outer,
        attr::Style::Inner(_binding_0) => attr::Style::Inner(_binding_0),
    }
}
pub fn fold_attribute<F>(f: &mut F, node: attr::Attr) -> attr::Attr
where
    F: Fold + ?Sized,
{
    attr::Attr {
        pound: node.pound,
        style: f.fold_attr_style(node.style),
        bracket: node.bracket,
        meta: f.fold_meta(node.meta),
    }
}
pub fn fold_bare_fn_arg<F>(f: &mut F, node: typ::FnArg) -> typ::FnArg
where
    F: Fold + ?Sized,
{
    typ::FnArg {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        name: (node.name).map(|it| (f.fold_ident((it).0), (it).1)),
        typ: f.fold_type(node.typ),
    }
}
pub fn fold_bare_variadic<F>(f: &mut F, node: typ::Variadic) -> typ::Variadic
where
    F: Fold + ?Sized,
{
    typ::Variadic {
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
pub fn fold_bound_lifetimes<F>(f: &mut F, node: Bgen::bound::Lifes) -> Bgen::bound::Lifes
where
    F: Fold + ?Sized,
{
    Bgen::bound::Lifes {
        for_: node.for_,
        lt: node.lt,
        lifes: FoldHelper::lift(node.lifes, |it| f.fold_generic_param(it)),
        gt: node.gt,
    }
}
pub fn fold_const_param<F>(f: &mut F, node: gen::param::Const) -> gen::param::Const
where
    F: Fold + ?Sized,
{
    gen::param::Const {
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
pub fn fold_data_enum<F>(f: &mut F, node: data::Enum) -> data::Enum
where
    F: Fold + ?Sized,
{
    data::Enum {
        enum_: node.enum_,
        brace: node.brace,
        variants: FoldHelper::lift(node.variants, |it| f.fold_variant(it)),
    }
}
pub fn fold_data_struct<F>(f: &mut F, node: data::Struct) -> data::Struct
where
    F: Fold + ?Sized,
{
    data::Struct {
        struct_: node.struct_,
        fields: f.fold_fields(node.fields),
        semi: node.semi,
    }
}
pub fn fold_data_union<F>(f: &mut F, node: data::Union) -> data::Union
where
    F: Fold + ?Sized,
{
    data::Union {
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
        Expr::Stream(_binding_0) => Expr::Stream(_binding_0),
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
        life: (node.life).map(|it| f.fold_lifetime(it)),
        val: (node.val).map(|it| Box::new(f.fold_expr(*it))),
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
        ins: FoldHelper::lift(node.inputs, |it| f.fold_pat(it)),
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
        life: (node.life).map(|it| f.fold_lifetime(it)),
    }
}
pub fn fold_expr_field<F>(f: &mut F, node: expr::Field) -> expr::Field
where
    F: Fold + ?Sized,
{
    expr::Field {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        expr: Box::new(f.fold_expr(*node.expr)),
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
        then_: f.fold_block(node.then_),
        else_: (node.else_).map(|it| ((it).0, Box::new(f.fold_expr(*(it).1)))),
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
        idx: Box::new(f.fold_expr(*node.idx)),
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
        block: f.fold_block(node.block),
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
pub fn fold_field<F>(f: &mut F, node: data::Field) -> data::Field
where
    F: Fold + ?Sized,
{
    data::Field {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        mut_: f.fold_field_mutability(node.mut_),
        ident: (node.ident).map(|it| f.fold_ident(it)),
        colon: node.colon,
        typ: f.fold_type(node.typ),
    }
}
pub fn fold_field_mutability<F>(f: &mut F, node: data::Mut) -> data::Mut
where
    F: Fold + ?Sized,
{
    match node {
        data::Mut::None => data::Mut::None,
    }
}
pub fn fold_field_pat<F>(f: &mut F, node: pat::Field) -> pat::Field
where
    F: Fold + ?Sized,
{
    pat::Fieldeld {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        member: f.fold_member(node.memb),
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
pub fn fold_fields<F>(f: &mut F, node: data::Fields) -> data::Fields
where
    F: Fold + ?Sized,
{
    match node {
        data::Fields::Named(_binding_0) => data::Fields::Named(f.fold_fields_named(_binding_0)),
        data::Fields::Unnamed(_binding_0) => data::Fields::Unnamed(f.fold_fields_unnamed(_binding_0)),
        data::Fields::Unit => data::Fields::Unit,
    }
}
pub fn fold_fields_named<F>(f: &mut F, node: data::Named) -> data::Named
where
    F: Fold + ?Sized,
{
    data::Named {
        brace: node.brace,
        fields: FoldHelper::lift(node.fields, |it| f.fold_field(it)),
    }
}
pub fn fold_fields_unnamed<F>(f: &mut F, node: data::Unnamed) -> data::Unnamed
where
    F: Fold + ?Sized,
{
    data::Unnamed {
        paren: node.paren,
        fields: FoldHelper::lift(node.fields, |it| f.fold_field(it)),
    }
}
pub fn fold_file<F>(f: &mut F, node: item::File) -> item::File
where
    F: Fold + ?Sized,
{
    item::File {
        shebang: node.shebang,
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        items: FoldHelper::lift(node.items, |it| f.fold_item(it)),
    }
}
pub fn fold_fn_arg<F>(f: &mut F, node: item::FnArg) -> item::FnArg
where
    F: Fold + ?Sized,
{
    match node {
        item::FnArg::Receiver(_binding_0) => item::FnArg::Receiver(f.fold_receiver(_binding_0)),
        item::FnArg::Type(_binding_0) => item::FnArg::Type(f.fold_pat_type(_binding_0)),
    }
}
pub fn fold_foreign_item<F>(f: &mut F, node: item::foreign::Item) -> item::foreign::Item
where
    F: Fold + ?Sized,
{
    match node {
        item::foreign::Item::Fn(_binding_0) => item::foreign::Item::Fn(f.fold_foreign_item_fn(_binding_0)),
        item::foreign::Item::Static(_binding_0) => item::foreign::Item::Static(f.fold_foreign_item_static(_binding_0)),
        item::foreign::Item::Type(_binding_0) => item::foreign::Item::Type(f.fold_foreign_item_type(_binding_0)),
        item::foreign::Item::Macro(_binding_0) => item::foreign::Item::Macro(f.fold_foreign_item_macro(_binding_0)),
        item::foreign::Item::Stream(_binding_0) => item::foreign::Item::Stream(_binding_0),
    }
}
pub fn fold_foreign_item_fn<F>(f: &mut F, node: item::foreign::Fn) -> item::foreign::Fn
where
    F: Fold + ?Sized,
{
    item::foreign::Fn {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        sig: f.fold_signature(node.sig),
        semi: node.semi,
    }
}
pub fn fold_foreign_item_macro<F>(f: &mut F, node: item::foreign::Mac) -> item::foreign::Mac
where
    F: Fold + ?Sized,
{
    item::foreign::Mac {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        mac: f.fold_macro(node.mac),
        semi: node.semi,
    }
}
pub fn fold_foreign_item_static<F>(f: &mut F, node: item::foreign::Static) -> item::foreign::Static
where
    F: Fold + ?Sized,
{
    item::foreign::Static {
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
pub fn fold_foreign_item_type<F>(f: &mut F, node: item::foreign::Type) -> item::foreign::Type
where
    F: Fold + ?Sized,
{
    item::foreign::Type {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        type_: node.type_,
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
        Arg::Life(_binding_0) => Arg::Life(f.fold_lifetime(_binding_0)),
        Arg::Type(_binding_0) => Arg::Type(f.fold_type(_binding_0)),
        Arg::Const(_binding_0) => Arg::Const(f.fold_expr(_binding_0)),
        Arg::AssocType(_binding_0) => Arg::AssocType(f.fold_assoc_type(_binding_0)),
        Arg::AssocConst(_binding_0) => Arg::AssocConst(f.fold_assoc_const(_binding_0)),
        Arg::Constraint(_binding_0) => Arg::Constraint(f.fold_constraint(_binding_0)),
    }
}
pub fn fold_generic_param<F>(f: &mut F, node: gen::Param) -> gen::Param
where
    F: Fold + ?Sized,
{
    match node {
        gen::Param::Life(_binding_0) => gen::Param::Life(f.fold_lifetime_param(_binding_0)),
        gen::Param::Type(_binding_0) => gen::Param::Type(f.fold_type_param(_binding_0)),
        gen::Param::Const(_binding_0) => gen::Param::Const(f.fold_const_param(_binding_0)),
    }
}
pub fn fold_generics<F>(f: &mut F, node: gen::Gens) -> gen::Gens
where
    F: Fold + ?Sized,
{
    gen::Gens {
        lt: node.lt,
        ps: FoldHelper::lift(node.ps, |it| f.fold_generic_param(it)),
        gt: node.gt,
        where_: (node.where_).map(|it| f.fold_where_clause(it)),
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
pub fn fold_impl_item<F>(f: &mut F, node: item::impl_::Item) -> item::impl_::Item
where
    F: Fold + ?Sized,
{
    match node {
        item::impl_::Item::Const(_binding_0) => item::impl_::Item::Const(f.fold_impl_item_const(_binding_0)),
        item::impl_::Item::Fn(_binding_0) => item::impl_::Item::Fn(f.fold_impl_item_fn(_binding_0)),
        item::impl_::Item::Type(_binding_0) => item::impl_::Item::Type(f.fold_impl_item_type(_binding_0)),
        item::impl_::Item::Macro(_binding_0) => item::impl_::Item::Macro(f.fold_impl_item_macro(_binding_0)),
        item::impl_::Item::Stream(_binding_0) => item::impl_::Item::Stream(_binding_0),
    }
}
pub fn fold_impl_item_const<F>(f: &mut F, node: item::impl_::Const) -> item::impl_::Const
where
    F: Fold + ?Sized,
{
    item::impl_::Const {
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
pub fn fold_impl_item_fn<F>(f: &mut F, node: item::impl_::Fn) -> item::impl_::Fn
where
    F: Fold + ?Sized,
{
    item::impl_::Fn {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        default_: node.default_,
        sig: f.fold_signature(node.sig),
        block: f.fold_block(node.block),
    }
}
pub fn fold_impl_item_macro<F>(f: &mut F, node: item::impl_::Mac) -> item::impl_::Mac
where
    F: Fold + ?Sized,
{
    item::impl_::Mac {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        mac: f.fold_macro(node.mac),
        semi: node.semi,
    }
}
pub fn fold_impl_item_type<F>(f: &mut F, node: item::impl_::Type) -> item::impl_::Type
where
    F: Fold + ?Sized,
{
    item::impl_::Type {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        default_: node.default_,
        type_: node.type_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        eq: node.eq,
        typ: f.fold_type(node.typ),
        semi: node.semi,
    }
}
pub fn fold_impl_restriction<F>(f: &mut F, node: item::impl_::Restriction) -> item::impl_::Restriction
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
        Item::Extern(_binding_0) => Item::Extern(f.fold_item_extern_crate(_binding_0)),
        Item::Fn(_binding_0) => Item::Fn(f.fold_item_fn(_binding_0)),
        Item::Foreign(_binding_0) => Item::Foreign(f.fold_item_foreign_mod(_binding_0)),
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
        Item::Stream(_binding_0) => Item::Stream(_binding_0),
    }
}
pub fn fold_item_const<F>(f: &mut F, node: item::Const) -> item::Const
where
    F: Fold + ?Sized,
{
    item::Const {
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
pub fn fold_item_enum<F>(f: &mut F, node: item::Enum) -> item::Enum
where
    F: Fold + ?Sized,
{
    item::Enum {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        enum_: node.enum_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        brace: node.brace,
        variants: FoldHelper::lift(node.variants, |it| f.fold_variant(it)),
    }
}
pub fn fold_item_extern_crate<F>(f: &mut F, node: item::Extern) -> item::Extern
where
    F: Fold + ?Sized,
{
    item::Extern {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        extern_: node.extern_,
        crate_: node.crate_,
        ident: f.fold_ident(node.ident),
        rename: (node.rename).map(|it| ((it).0, f.fold_ident((it).1))),
        semi: node.semi,
    }
}
pub fn fold_item_fn<F>(f: &mut F, node: item::Fn) -> item::Fn
where
    F: Fold + ?Sized,
{
    item::Fn {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        sig: f.fold_signature(node.sig),
        block: Box::new(f.fold_block(*node.block)),
    }
}
pub fn fold_item_foreign_mod<F>(f: &mut F, node: item::Foreign) -> item::Foreign
where
    F: Fold + ?Sized,
{
    item::Foreign {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        unsafe_: node.unsafe_,
        abi: f.fold_abi(node.abi),
        brace: node.brace,
        items: FoldHelper::lift(node.items, |it| f.fold_foreign_item(it)),
    }
}
pub fn fold_item_impl<F>(f: &mut F, node: item::Impl) -> item::Impl
where
    F: Fold + ?Sized,
{
    item::Impl {
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
pub fn fold_item_macro<F>(f: &mut F, node: item::Mac) -> item::Mac
where
    F: Fold + ?Sized,
{
    item::Mac {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        ident: (node.ident).map(|it| f.fold_ident(it)),
        mac: f.fold_macro(node.mac),
        semi: node.semi,
    }
}
pub fn fold_item_mod<F>(f: &mut F, node: item::Mod) -> item::Mod
where
    F: Fold + ?Sized,
{
    item::Mod {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        unsafe_: node.unsafe_,
        mod_: node.mod_,
        ident: f.fold_ident(node.ident),
        items: (node.items).map(|it| ((it).0, FoldHelper::lift((it).1, |it| f.fold_item(it)))),
        semi: node.semi,
    }
}
pub fn fold_item_static<F>(f: &mut F, node: item::Static) -> item::Static
where
    F: Fold + ?Sized,
{
    item::Static {
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
pub fn fold_item_struct<F>(f: &mut F, node: item::Struct) -> item::Struct
where
    F: Fold + ?Sized,
{
    item::Struct {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        struct_: node.struct_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        fields: f.fold_fields(node.fields),
        semi: node.semi,
    }
}
pub fn fold_item_trait<F>(f: &mut F, node: item::Trait) -> item::Trait
where
    F: Fold + ?Sized,
{
    item::Trait {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        unsafe_: node.unsafe_,
        auto_: node.auto_,
        restriction: (node.restriction).map(|it| f.fold_impl_restriction(it)),
        trait_: node.trait_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        colon: node.colon,
        supers: FoldHelper::lift(node.supers, |it| f.fold_type_param_bound(it)),
        brace: node.brace,
        items: FoldHelper::lift(node.items, |it| f.fold_trait_item(it)),
    }
}
pub fn fold_item_trait_alias<F>(f: &mut F, node: item::TraitAlias) -> item::TraitAlias
where
    F: Fold + ?Sized,
{
    item::TraitAlias {
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
pub fn fold_item_type<F>(f: &mut F, node: item::Type) -> item::Type
where
    F: Fold + ?Sized,
{
    item::Type {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        type_: node.type_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        eq: node.eq,
        typ: Box::new(f.fold_type(*node.typ)),
        semi: node.semi,
    }
}
pub fn fold_item_union<F>(f: &mut F, node: item::Union) -> item::Union
where
    F: Fold + ?Sized,
{
    item::Union {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        union_: node.union_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        fields: f.fold_fields_named(node.fields),
    }
}
pub fn fold_item_use<F>(f: &mut F, node: item::Use) -> item::Use
where
    F: Fold + ?Sized,
{
    item::Use {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vis: f.fold_visibility(node.vis),
        use_: node.use_,
        colon: node.colon,
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
pub fn fold_lifetime<F>(f: &mut F, node: Life) -> Life
where
    F: Fold + ?Sized,
{
    Life {
        apos: f.fold_span(node.apos),
        ident: f.fold_ident(node.ident),
    }
}
pub fn fold_lifetime_param<F>(f: &mut F, node: gen::param::Life) -> gen::param::Life
where
    F: Fold + ?Sized,
{
    gen::param::Life {
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
        Lit::Stream(_binding_0) => Lit::Stream(_binding_0),
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
pub fn fold_local_init<F>(f: &mut F, node: stmt::Init) -> stmt::Init
where
    F: Fold + ?Sized,
{
    stmt::Init {
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
pub fn fold_macro_delimiter<F>(f: &mut F, node: tok::Delim) -> tok::Delim
where
    F: Fold + ?Sized,
{
    match node {
        tok::Delim::Paren(_binding_0) => tok::Delim::Paren(_binding_0),
        tok::Delim::Brace(_binding_0) => tok::Delim::Brace(_binding_0),
        tok::Delim::Bracket(_binding_0) => tok::Delim::Bracket(_binding_0),
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
pub fn fold_meta<F>(f: &mut F, node: attr::Meta) -> attr::Meta
where
    F: Fold + ?Sized,
{
    match node {
        attr::Meta::Path(_binding_0) => attr::Meta::Path(f.fold_path(_binding_0)),
        attr::Meta::List(_binding_0) => attr::Meta::List(f.fold_meta_list(_binding_0)),
        attr::Meta::NameValue(_binding_0) => attr::Meta::NameValue(f.fold_meta_name_value(_binding_0)),
    }
}
pub fn fold_meta_list<F>(f: &mut F, node: attr::List) -> attr::List
where
    F: Fold + ?Sized,
{
    attr::List {
        path: f.fold_path(node.path),
        delim: f.fold_macro_delimiter(node.delim),
        toks: node.toks,
    }
}
pub fn fold_meta_name_value<F>(f: &mut F, node: attr::NameValue) -> attr::NameValue
where
    F: Fold + ?Sized,
{
    attr::NameValue {
        name: f.fold_path(node.name),
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
pub fn fold_pat<F>(f: &mut F, node: pat::Pat) -> pat::Pat
where
    F: Fold + ?Sized,
{
    match node {
        pat::Pat::Const(_binding_0) => pat::Pat::Const(f.fold_expr_const(_binding_0)),
        pat::Pat::Ident(_binding_0) => pat::Pat::Ident(f.fold_pat_ident(_binding_0)),
        pat::Pat::Lit(_binding_0) => pat::Pat::Lit(f.fold_expr_lit(_binding_0)),
        pat::Pat::Mac(_binding_0) => pat::Pat::Mac(f.fold_expr_macro(_binding_0)),
        pat::Pat::Or(_binding_0) => pat::Pat::Or(f.fold_pat_or(_binding_0)),
        pat::Pat::Paren(_binding_0) => pat::Pat::Paren(f.fold_pat_paren(_binding_0)),
        pat::Pat::Path(_binding_0) => pat::Pat::Path(f.fold_expr_path(_binding_0)),
        pat::Pat::Range(_binding_0) => pat::Pat::Range(f.fold_expr_range(_binding_0)),
        pat::Pat::Ref(_binding_0) => pat::Pat::Ref(f.fold_pat_reference(_binding_0)),
        pat::Pat::Rest(_binding_0) => pat::Pat::Rest(f.fold_pat_rest(_binding_0)),
        pat::Pat::Slice(_binding_0) => pat::Pat::Slice(f.fold_pat_slice(_binding_0)),
        pat::Pat::Struct(_binding_0) => pat::Pat::Struct(f.fold_pat_struct(_binding_0)),
        pat::Pat::Tuple(_binding_0) => pat::Pat::Tuple(f.fold_pat_tuple(_binding_0)),
        pat::Pat::TupleStruct(_binding_0) => pat::Pat::TupleStruct(f.fold_pat_tuple_struct(_binding_0)),
        pat::Pat::Type(_binding_0) => pat::Pat::Type(f.fold_pat_type(_binding_0)),
        pat::Pat::Stream(_binding_0) => pat::Pat::Stream(_binding_0),
        pat::Pat::Wild(_binding_0) => pat::Pat::Wild(f.fold_pat_wild(_binding_0)),
    }
}
pub fn fold_pat_ident<F>(f: &mut F, node: pat::Ident) -> pat::Ident
where
    F: Fold + ?Sized,
{
    pat::Ident {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        ref_: node.ref_,
        mut_: node.mut_,
        ident: f.fold_ident(node.ident),
        sub: (node.sub).map(|it| ((it).0, Box::new(f.fold_pat(*(it).1)))),
    }
}
pub fn fold_pat_or<F>(f: &mut F, node: pat::Or) -> pat::Or
where
    F: Fold + ?Sized,
{
    pat::Or {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        vert: node.vert,
        cases: FoldHelper::lift(node.cases, |it| f.fold_pat(it)),
    }
}
pub fn fold_pat_paren<F>(f: &mut F, node: pat::Paren) -> pat::Paren
where
    F: Fold + ?Sized,
{
    pat::Paren {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        paren: node.paren,
        pat: Box::new(f.fold_pat(*node.pat)),
    }
}
pub fn fold_pat_reference<F>(f: &mut F, node: pat::Ref) -> pat::Ref
where
    F: Fold + ?Sized,
{
    pat::Ref {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        and: node.and,
        mut_: node.mut_,
        pat: Box::new(f.fold_pat(*node.pat)),
    }
}
pub fn fold_pat_rest<F>(f: &mut F, node: pat::Rest) -> pat::Rest
where
    F: Fold + ?Sized,
{
    pat::Restest {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        dot2: node.dot2,
    }
}
pub fn fold_pat_slice<F>(f: &mut F, node: pat::Slice) -> pat::Slice
where
    F: Fold + ?Sized,
{
    pat::Slice {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        bracket: node.bracket,
        elems: FoldHelper::lift(node.elems, |it| f.fold_pat(it)),
    }
}
pub fn fold_pat_struct<F>(f: &mut F, node: pat::Struct) -> pat::Struct
where
    F: Fold + ?Sized,
{
    pat::Struct {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        qself: (node.qself).map(|it| f.fold_qself(it)),
        path: f.fold_path(node.path),
        brace: node.brace,
        fields: FoldHelper::lift(node.fields, |it| f.fold_field_pat(it)),
        rest: (node.rest).map(|it| f.fold_pat_rest(it)),
    }
}
pub fn fold_pat_tuple<F>(f: &mut F, node: pat::Tuple) -> pat::Tuple
where
    F: Fold + ?Sized,
{
    pat::Tuple {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        paren: node.paren,
        elems: FoldHelper::lift(node.elems, |it| f.fold_pat(it)),
    }
}
pub fn fold_pat_tuple_struct<F>(f: &mut F, node: pat::TupleStruct) -> pat::TupleStruct
where
    F: Fold + ?Sized,
{
    pat::TupleStructuct {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        qself: (node.qself).map(|it| f.fold_qself(it)),
        path: f.fold_path(node.path),
        paren: node.paren,
        elems: FoldHelper::lift(node.elems, |it| f.fold_pat(it)),
    }
}
pub fn fold_pat_type<F>(f: &mut F, node: pat::Type) -> pat::Type
where
    F: Fold + ?Sized,
{
    pat::Type {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        pat: Box::new(f.fold_pat(*node.pat)),
        colon: node.colon,
        typ: Box::new(f.fold_type(*node.typ)),
    }
}
pub fn fold_pat_wild<F>(f: &mut F, node: pat::Wild) -> pat::Wild
where
    F: Fold + ?Sized,
{
    pat::Wild {
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
pub fn fold_predicate_lifetime<F>(f: &mut F, node: gen::Where::Life) -> gen::Where::Life
where
    F: Fold + ?Sized,
{
    gen::Where::Life {
        life: f.fold_lifetime(node.life),
        colon: node.colon,
        bounds: FoldHelper::lift(node.bounds, |it| f.fold_lifetime(it)),
    }
}
pub fn fold_predicate_type<F>(f: &mut F, node: gen::Where::Type) -> gen::Where::Type
where
    F: Fold + ?Sized,
{
    gen::Where::Type {
        lifes: (node.lifes).map(|it| f.fold_bound_lifetimes(it)),
        bounded: f.fold_type(node.bounded),
        colon: node.colon,
        bounds: FoldHelper::lift(node.bounds, |it| f.fold_type_param_bound(it)),
    }
}
pub fn fold_qself<F>(f: &mut F, node: path::QSelf) -> path::QSelf
where
    F: Fold + ?Sized,
{
    path::QSelf {
        lt: node.lt,
        typ: Box::new(f.fold_type(*node.typ)),
        pos: node.pos,
        as_: node.as_,
        gt: node.gt,
    }
}
pub fn fold_range_limits<F>(f: &mut F, node: expr::Limits) -> expr::Limits
where
    F: Fold + ?Sized,
{
    match node {
        expr::Limits::HalfOpen(_binding_0) => expr::Limits::HalfOpen(_binding_0),
        expr::Limits::Closed(_binding_0) => expr::Limits::Closed(_binding_0),
    }
}
pub fn fold_receiver<F>(f: &mut F, node: item::Receiver) -> item::Receiver
where
    F: Fold + ?Sized,
{
    item::Receiver {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        ref_: (node.ref_).map(|it| ((it).0, ((it).1).map(|it| f.fold_lifetime(it)))),
        mut_: node.mut_,
        self_: node.self_,
        colon: node.colon,
        typ: Box::new(f.fold_type(*node.typ)),
    }
}
pub fn fold_return_type<F>(f: &mut F, node: typ::Ret) -> typ::Ret
where
    F: Fold + ?Sized,
{
    match node {
        typ::Ret::Default => typ::Ret::Default,
        typ::Ret::Type(_binding_0, _binding_1) => typ::Ret::Type(_binding_0, Box::new(f.fold_type(*_binding_1))),
    }
}
pub fn fold_signature<F>(f: &mut F, node: item::Sig) -> item::Sig
where
    F: Fold + ?Sized,
{
    item::Sig {
        const_: node.const_,
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
pub fn fold_span<F>(f: &mut F, node: pm2::Span) -> pm2::Span
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
pub fn fold_trait_bound<F>(f: &mut F, node: gen::bound::Trait) -> gen::bound::Trait
where
    F: Fold + ?Sized,
{
    gen::bound::Trait {
        paren: node.paren,
        modif: f.fold_trait_bound_modifier(node.modif),
        lifes: (node.lifes).map(|it| f.fold_bound_lifetimes(it)),
        path: f.fold_path(node.path),
    }
}
pub fn fold_trait_bound_modifier<F>(f: &mut F, node: gen::bound::Modifier) -> gen::bound::Modifier
where
    F: Fold + ?Sized,
{
    match node {
        gen::bound::Modifier::None => gen::bound::Modifier::None,
        gen::bound::Modifier::Maybe(_binding_0) => gen::bound::Modifier::Maybe(_binding_0),
    }
}
pub fn fold_trait_item<F>(f: &mut F, node: item::trait_::Item) -> item::trait_::Item
where
    F: Fold + ?Sized,
{
    match node {
        item::trait_::Item::Const(_binding_0) => item::trait_::Item::Const(f.fold_trait_item_const(_binding_0)),
        item::trait_::Item::Fn(_binding_0) => item::trait_::Item::Fn(f.fold_trait_item_fn(_binding_0)),
        item::trait_::Item::Type(_binding_0) => item::trait_::Item::Type(f.fold_trait_item_type(_binding_0)),
        item::trait_::Item::Macro(_binding_0) => item::trait_::Item::Macro(f.fold_trait_item_macro(_binding_0)),
        item::trait_::Item::Stream(_binding_0) => item::trait_::Item::Stream(_binding_0),
    }
}
pub fn fold_trait_item_const<F>(f: &mut F, node: item::trait_::Const) -> item::trait_::Const
where
    F: Fold + ?Sized,
{
    item::trait_::Const {
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
pub fn fold_trait_item_fn<F>(f: &mut F, node: item::trait_::Fn) -> item::trait_::Fn
where
    F: Fold + ?Sized,
{
    item::trait_::Fn {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        sig: f.fold_signature(node.sig),
        default: (node.default).map(|it| f.fold_block(it)),
        semi: node.semi,
    }
}
pub fn fold_trait_item_macro<F>(f: &mut F, node: item::trait_::Mac) -> item::trait_::Mac
where
    F: Fold + ?Sized,
{
    item::trait_::Mac {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        mac: f.fold_macro(node.mac),
        semi: node.semi,
    }
}
pub fn fold_trait_item_type<F>(f: &mut F, node: item::trait_::Type) -> item::trait_::Type
where
    F: Fold + ?Sized,
{
    item::trait_::Type {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        type_: node.type_,
        ident: f.fold_ident(node.ident),
        gens: f.fold_generics(node.gens),
        colon: node.colon,
        bounds: FoldHelper::lift(node.bounds, |it| f.fold_type_param_bound(it)),
        default: (node.default).map(|it| ((it).0, f.fold_type((it).1))),
        semi: node.semi,
    }
}
pub fn fold_type<F>(f: &mut F, node: typ::Type) -> typ::Type
where
    F: Fold + ?Sized,
{
    match node {
        typ::Type::Array(_binding_0) => typ::Type::Array(f.fold_type_array(_binding_0)),
        typ::Type::Fn(_binding_0) => typ::Type::Fn(f.fold_type_bare_fn(_binding_0)),
        typ::Type::Group(_binding_0) => typ::Type::Group(f.fold_type_group(_binding_0)),
        typ::Type::Impl(_binding_0) => typ::Type::Impl(f.fold_type_impl_trait(_binding_0)),
        typ::Type::Infer(_binding_0) => typ::Type::Infer(f.fold_type_infer(_binding_0)),
        typ::Type::Mac(_binding_0) => typ::Type::Mac(f.fold_type_macro(_binding_0)),
        typ::Type::Never(_binding_0) => typ::Type::Never(f.fold_type_never(_binding_0)),
        typ::Type::Paren(_binding_0) => typ::Type::Paren(f.fold_type_paren(_binding_0)),
        typ::Type::Path(_binding_0) => typ::Type::Path(f.fold_type_path(_binding_0)),
        typ::Type::Ptr(_binding_0) => typ::Type::Ptr(f.fold_type_ptr(_binding_0)),
        typ::Type::Ref(_binding_0) => typ::Type::Ref(f.fold_type_reference(_binding_0)),
        typ::Type::Slice(_binding_0) => typ::Type::Slice(f.fold_type_slice(_binding_0)),
        typ::Type::Trait(_binding_0) => typ::Type::Trait(f.fold_type_trait_object(_binding_0)),
        typ::Type::Tuple(_binding_0) => typ::Type::Tuple(f.fold_type_tuple(_binding_0)),
        typ::Type::Stream(_binding_0) => typ::Type::Stream(_binding_0),
    }
}
pub fn fold_type_array<F>(f: &mut F, node: typ::Array) -> typ::Array
where
    F: Fold + ?Sized,
{
    typ::Array {
        bracket: node.bracket,
        elem: Box::new(f.fold_type(*node.elem)),
        semi: node.semi,
        len: f.fold_expr(node.len),
    }
}
pub fn fold_type_bare_fn<F>(f: &mut F, node: typ::Fn) -> typ::Fn
where
    F: Fold + ?Sized,
{
    typ::Fn {
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
pub fn fold_type_group<F>(f: &mut F, node: typ::Group) -> typ::Group
where
    F: Fold + ?Sized,
{
    typ::Group {
        group: node.group,
        elem: Box::new(f.fold_type(*node.elem)),
    }
}
pub fn fold_type_impl_trait<F>(f: &mut F, node: typ::Impl) -> typ::Impl
where
    F: Fold + ?Sized,
{
    typ::Impl {
        impl_: node.impl_,
        bounds: FoldHelper::lift(node.bounds, |it| f.fold_type_param_bound(it)),
    }
}
pub fn fold_type_infer<F>(f: &mut F, node: typ::Infer) -> typ::Infer
where
    F: Fold + ?Sized,
{
    typ::Infer {
        underscore: node.underscore,
    }
}
pub fn fold_type_macro<F>(f: &mut F, node: typ::Mac) -> typ::Mac
where
    F: Fold + ?Sized,
{
    typ::Mac {
        mac: f.fold_macro(node.mac),
    }
}
pub fn fold_type_never<F>(f: &mut F, node: typ::Never) -> typ::Never
where
    F: Fold + ?Sized,
{
    typ::Never { bang: node.bang }
}
pub fn fold_type_param<F>(f: &mut F, node: gen::param::Type) -> gen::param::Type
where
    F: Fold + ?Sized,
{
    gen::param::Type {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        ident: f.fold_ident(node.ident),
        colon: node.colon,
        bounds: FoldHelper::lift(node.bounds, |it| f.fold_type_param_bound(it)),
        eq: node.eq,
        default: (node.default).map(|it| f.fold_type(it)),
    }
}
pub fn fold_type_param_bound<F>(f: &mut F, node: gen::bound::Type) -> gen::bound::Type
where
    F: Fold + ?Sized,
{
    match node {
        gen::bound::Type::Trait(_binding_0) => gen::bound::Type::Trait(f.fold_trait_bound(_binding_0)),
        gen::bound::Type::Life(_binding_0) => gen::bound::Type::Life(f.fold_lifetime(_binding_0)),
        gen::bound::Type::Stream(_binding_0) => gen::bound::Type::Stream(_binding_0),
    }
}
pub fn fold_type_paren<F>(f: &mut F, node: typ::Paren) -> typ::Paren
where
    F: Fold + ?Sized,
{
    typ::Paren {
        paren: node.paren,
        elem: Box::new(f.fold_type(*node.elem)),
    }
}
pub fn fold_type_path<F>(f: &mut F, node: typ::Path) -> typ::Path
where
    F: Fold + ?Sized,
{
    typ::Path {
        qself: (node.qself).map(|it| f.fold_qself(it)),
        path: f.fold_path(node.path),
    }
}
pub fn fold_type_ptr<F>(f: &mut F, node: typ::Ptr) -> typ::Ptr
where
    F: Fold + ?Sized,
{
    typ::Ptr {
        star: node.star,
        const_: node.const_,
        mut_: node.mut_,
        elem: Box::new(f.fold_type(*node.elem)),
    }
}
pub fn fold_type_reference<F>(f: &mut F, node: typ::Ref) -> typ::Ref
where
    F: Fold + ?Sized,
{
    typ::Ref {
        and: node.and,
        life: (node.life).map(|it| f.fold_lifetime(it)),
        mut_: node.mut_,
        elem: Box::new(f.fold_type(*node.elem)),
    }
}
pub fn fold_type_slice<F>(f: &mut F, node: typ::Slice) -> typ::Slice
where
    F: Fold + ?Sized,
{
    typ::Slice {
        bracket: node.bracket,
        elem: Box::new(f.fold_type(*node.elem)),
    }
}
pub fn fold_type_trait_object<F>(f: &mut F, node: typ::Trait) -> typ::Trait
where
    F: Fold + ?Sized,
{
    typ::Trait {
        dyn_: node.dyn_,
        bounds: FoldHelper::lift(node.bounds, |it| f.fold_type_param_bound(it)),
    }
}
pub fn fold_type_tuple<F>(f: &mut F, node: typ::Tuple) -> typ::Tuple
where
    F: Fold + ?Sized,
{
    typ::Tuple {
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
pub fn fold_use_glob<F>(f: &mut F, node: item::use_::Glob) -> item::use_::Glob
where
    F: Fold + ?Sized,
{
    item::use_::Glob { star: node.star }
}
pub fn fold_use_group<F>(f: &mut F, node: item::use_::Group) -> item::use_::Group
where
    F: Fold + ?Sized,
{
    item::use_::Group {
        brace: node.brace,
        elems: FoldHelper::lift(node.elems, |it| f.fold_use_tree(it)),
    }
}
pub fn fold_use_name<F>(f: &mut F, node: item::use_::Name) -> item::use_::Name
where
    F: Fold + ?Sized,
{
    item::use_::Name {
        ident: f.fold_ident(node.ident),
    }
}
pub fn fold_use_path<F>(f: &mut F, node: item::use_::Path) -> item::use_::Path
where
    F: Fold + ?Sized,
{
    item::use_::Path {
        ident: f.fold_ident(node.ident),
        colon2: node.colon2,
        tree: Box::new(f.fold_use_tree(*node.tree)),
    }
}
pub fn fold_use_rename<F>(f: &mut F, node: item::use_::Rename) -> item::use_::Rename
where
    F: Fold + ?Sized,
{
    item::use_::Rename {
        ident: f.fold_ident(node.ident),
        as_: node.as_,
        rename: f.fold_ident(node.rename),
    }
}
pub fn fold_use_tree<F>(f: &mut F, node: item::use_::Tree) -> item::use_::Tree
where
    F: Fold + ?Sized,
{
    match node {
        item::use_::Tree::Path(_binding_0) => item::use_::Tree::Path(f.fold_use_path(_binding_0)),
        item::use_::Tree::Name(_binding_0) => item::use_::Tree::Name(f.fold_use_name(_binding_0)),
        item::use_::Tree::Rename(_binding_0) => item::use_::Tree::Rename(f.fold_use_rename(_binding_0)),
        item::use_::Tree::Glob(_binding_0) => item::use_::Tree::Glob(f.fold_use_glob(_binding_0)),
        item::use_::Tree::Group(_binding_0) => item::use_::Tree::Group(f.fold_use_group(_binding_0)),
    }
}
pub fn fold_variadic<F>(f: &mut F, node: item::Variadic) -> item::Variadic
where
    F: Fold + ?Sized,
{
    item::Variadic {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        pat: (node.pat).map(|it| (Box::new(f.fold_pat(*(it).0)), (it).1)),
        dots: node.dots,
        comma: node.comma,
    }
}
pub fn fold_variant<F>(f: &mut F, node: data::Variant) -> data::Variant
where
    F: Fold + ?Sized,
{
    data::Variant {
        attrs: FoldHelper::lift(node.attrs, |it| f.fold_attribute(it)),
        ident: f.fold_ident(node.ident),
        fields: f.fold_fields(node.fields),
        discrim: (node.discrim).map(|it| ((it).0, f.fold_expr((it).1))),
    }
}
pub fn fold_vis_restricted<F>(f: &mut F, node: data::Restricted) -> data::Restricted
where
    F: Fold + ?Sized,
{
    data::Restricted {
        pub_: node.pub_,
        paren: node.paren,
        in_: node.in_,
        path: Box::new(f.fold_path(*node.path)),
    }
}
pub fn fold_visibility<F>(f: &mut F, node: data::Visibility) -> data::Visibility
where
    F: Fold + ?Sized,
{
    match node {
        data::Visibility::Public(_binding_0) => data::Visibility::Public(_binding_0),
        data::Visibility::Restricted(_binding_0) => data::Visibility::Restricted(f.fold_vis_restricted(_binding_0)),
        data::Visibility::Inherited => data::Visibility::Inherited,
    }
}
pub fn fold_where_clause<F>(f: &mut F, node: gen::Where) -> gen::Where
where
    F: Fold + ?Sized,
{
    gen::Where {
        where_: node.where_,
        preds: FoldHelper::lift(node.preds, |it| f.fold_where_predicate(it)),
    }
}
pub fn fold_where_predicate<F>(f: &mut F, node: gen::Where::Pred) -> gen::Where::Pred
where
    F: Fold + ?Sized,
{
    match node {
        gen::Where::Pred::Life(_binding_0) => gen::Where::Pred::Life(f.fold_predicate_lifetime(_binding_0)),
        gen::Where::Pred::Type(_binding_0) => gen::Where::Pred::Type(f.fold_predicate_type(_binding_0)),
    }
}
