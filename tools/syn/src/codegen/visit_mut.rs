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

impl VisitMut for attr::Style {
    fn visit_mut<V>(&mut self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        use attr::Style::*;
        match self {
            Inner(_) => {},
            Outer => {},
        }
    }
}
impl VisitMut for attr::Attr {
    fn visit_mut<V>(&mut self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        &mut self.style.visit_mut(v);
        &mut self.meta.visit_mut(v);
    }
}

impl VisitMut for attr::Meta {
    fn visit_mut<V>(&mut self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        use attr::Meta::*;
        match self {
            Path(x) => {
                x.visit_mut(v);
            },
            List(x) => {
                x.visit_mut(v);
            },
            NameValue(x) => {
                x.visit_mut(v);
            },
        }
    }
}

impl VisitMut for typ::Abi {
    fn visit_mut<V>(&mut self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        if let Some(x) = &mut self.name {
            x.visit_mut(v);
        }
    }
}
impl VisitMut for path::Angled {
    fn visit_mut<V>(&mut self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        for mut y in Puncted::pairs_mut(&mut self.args) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
    }
}
impl VisitMut for expr::Arm {
    fn visit_mut<V>(&mut self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut self.pat.visit_pat_mut(v);
        if let Some(x) = &mut self.guard {
            &mut *(x).1.visit_expr_mut(v);
        }
        &mut *self.body.visit_expr_mut(v);
    }
}
impl VisitMut for path::AssocConst {
    fn visit_mut<V>(&mut self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        &mut self.ident.visit_ident_mut(v);
        if let Some(x) = &mut self.args {
            x.visit_mut(v);
        }
        &mut self.val.visit_mut(v);
    }
}
impl VisitMut for path::AssocType {
    fn visit_mut<V>(&mut self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        &mut self.ident.visit_ident_mut(v);
        if let Some(x) = &mut self.args {
            x.visit_mut(v);
        }
        &mut self.typ.visit_mut();
    }
}
impl VisitMut for typ::FnArg {
    fn visit_mut<V>(&mut self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        if let Some(x) = &mut self.name {
            &mut (x).0.visit_mut(v);
        }
        &mut self.typ.visit_mut(v);
    }
}
impl VisitMut for typ::Variadic {
    fn visit_mut<V>(&mut self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        if let Some(x) = &mut self.name {
            &mut (x).0.visit_ident_mut(v);
        }
    }
}
impl VisitMut for expr::BinOp {
    fn visit_mut<V>(&mut self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        use expr::BinOp::*;
        match self {
            Add(x) => {},
            Sub(x) => {},
            Mul(x) => {},
            Div(x) => {},
            Rem(x) => {},
            And(x) => {},
            Or(x) => {},
            BitXor(x) => {},
            BitAnd(x) => {},
            BitOr(x) => {},
            Shl(x) => {},
            Shr(x) => {},
            Eq(x) => {},
            Lt(x) => {},
            Le(x) => {},
            Ne(x) => {},
            Ge(x) => {},
            Gt(x) => {},
            AddAssign(x) => {},
            SubAssign(x) => {},
            MulAssign(x) => {},
            DivAssign(x) => {},
            RemAssign(x) => {},
            BitXorAssign(x) => {},
            BitAndAssign(x) => {},
            BitOrAssign(x) => {},
            ShlAssign(x) => {},
            ShrAssign(x) => {},
        }
    }
}
impl VisitMut for stmt::Block {
    fn visit_mut<V>(&mut self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        for x in &mut self.stmts {
            x.visit_mut(v);
        }
    }
}
impl VisitMut for gen::bound::Lifes {
    fn visit_mut<V>(&mut self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        for mut y in Puncted::pairs_mut(&mut self.lifes) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
    }
}
impl VisitMut for gen::param::Const {
    fn visit_mut<V>(&mut self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut self.ident.visit_mut(v);
        &mut self.typ.visit_mut(v);
        if let Some(x) = &mut self.default {
            x.visit_expr_mut(v);
        }
    }
}
pub fn visit_constraint_mut<V>(v: &mut V, node: &mut Constraint)
where
    V: VisitMut + ?Sized,
{
    v.visit_ident_mut(&mut self.ident);
    if let Some(x) = &mut self.gnrs {
        v.visit_angle_bracketed_generic_arguments_mut(x);
    }
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        v.visit_type_param_bound_mut(x);
    }
}
pub fn visit_data_mut<V>(v: &mut V, node: &mut Data)
where
    V: VisitMut + ?Sized,
{
    match node {
        Data::Struct(x) => {
            v.visit_data_struct_mut(x);
        },
        Data::Enum(x) => {
            v.visit_data_enum_mut(x);
        },
        Data::Union(x) => {
            v.visit_data_union_mut(x);
        },
    }
}
pub fn visit_data_enum_mut<V>(v: &mut V, node: &mut data::Enum)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.variants) {
        let x = el.value_mut();
        v.visit_variant_mut(x);
    }
}
pub fn visit_data_struct_mut<V>(v: &mut V, node: &mut data::Struct)
where
    V: VisitMut + ?Sized,
{
    v.visit_fields_mut(&mut self.fields);
}
pub fn visit_data_union_mut<V>(v: &mut V, node: &mut data::Union)
where
    V: VisitMut + ?Sized,
{
    v.visit_fields_named_mut(&mut self.fields);
}
pub fn visit_derive_input_mut<V>(v: &mut V, node: &mut Input)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    v.visit_ident_mut(&mut self.ident);
    v.visit_generics_mut(&mut self.gens);
    v.visit_data_mut(&mut self.data);
}
pub fn visit_expr_mut<V>(v: &mut V, node: &mut Expr)
where
    V: VisitMut + ?Sized,
{
    match node {
        Expr::Array(x) => {
            full!(v.visit_expr_array_mut(x));
        },
        Expr::Assign(x) => {
            full!(v.visit_expr_assign_mut(x));
        },
        Expr::Async(x) => {
            full!(v.visit_expr_async_mut(x));
        },
        Expr::Await(x) => {
            full!(v.visit_expr_await_mut(x));
        },
        Expr::Binary(x) => {
            v.visit_expr_binary_mut(x);
        },
        Expr::Block(x) => {
            full!(v.visit_expr_block_mut(x));
        },
        Expr::Break(x) => {
            full!(v.visit_expr_break_mut(x));
        },
        Expr::Call(x) => {
            v.visit_expr_call_mut(x);
        },
        Expr::Cast(x) => {
            v.visit_expr_cast_mut(x);
        },
        Expr::Closure(x) => {
            full!(v.visit_expr_closure_mut(x));
        },
        Expr::Const(x) => {
            full!(v.visit_expr_const_mut(x));
        },
        Expr::Continue(x) => {
            full!(v.visit_expr_continue_mut(x));
        },
        Expr::Field(x) => {
            v.visit_expr_field_mut(x);
        },
        Expr::ForLoop(x) => {
            full!(v.visit_expr_for_loop_mut(x));
        },
        Expr::Group(x) => {
            v.visit_expr_group_mut(x);
        },
        Expr::If(x) => {
            full!(v.visit_expr_if_mut(x));
        },
        Expr::Index(x) => {
            v.visit_expr_index_mut(x);
        },
        Expr::Infer(x) => {
            full!(v.visit_expr_infer_mut(x));
        },
        Expr::Let(x) => {
            full!(v.visit_expr_let_mut(x));
        },
        Expr::Lit(x) => {
            v.visit_expr_lit_mut(x);
        },
        Expr::Loop(x) => {
            full!(v.visit_expr_loop_mut(x));
        },
        Expr::Macro(x) => {
            v.visit_expr_macro_mut(x);
        },
        Expr::Match(x) => {
            full!(v.visit_expr_match_mut(x));
        },
        Expr::MethodCall(x) => {
            full!(v.visit_expr_method_call_mut(x));
        },
        Expr::Parenth(x) => {
            v.visit_expr_paren_mut(x);
        },
        Expr::Path(x) => {
            v.visit_expr_path_mut(x);
        },
        Expr::Range(x) => {
            full!(v.visit_expr_range_mut(x));
        },
        Expr::Reference(x) => {
            full!(v.visit_expr_reference_mut(x));
        },
        Expr::Repeat(x) => {
            full!(v.visit_expr_repeat_mut(x));
        },
        Expr::Return(x) => {
            full!(v.visit_expr_return_mut(x));
        },
        Expr::Struct(x) => {
            full!(v.visit_expr_struct_mut(x));
        },
        Expr::Try(x) => {
            full!(v.visit_expr_try_mut(x));
        },
        Expr::TryBlock(x) => {
            full!(v.visit_expr_try_block_mut(x));
        },
        Expr::Tuple(x) => {
            full!(v.visit_expr_tuple_mut(x));
        },
        Expr::Unary(x) => {
            v.visit_expr_unary_mut(x);
        },
        Expr::Unsafe(x) => {
            full!(v.visit_expr_unsafe_mut(x));
        },
        Expr::Stream(x) => {},
        Expr::While(x) => {
            full!(v.visit_expr_while_mut(x));
        },
        Expr::Yield(x) => {
            full!(v.visit_expr_yield_mut(x));
        },
    }
}
pub fn visit_expr_array_mut<V>(v: &mut V, node: &mut expr::Array)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    for mut el in Puncted::pairs_mut(&mut self.elems) {
        let x = el.value_mut();
        v.visit_expr_mut(x);
    }
}
pub fn visit_expr_assign_mut<V>(v: &mut V, node: &mut expr::Assign)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_expr_mut(&mut *node.left);
    v.visit_expr_mut(&mut *node.right);
}
pub fn visit_expr_async_mut<V>(v: &mut V, node: &mut expr::Async)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_block_mut(&mut self.block);
}
pub fn visit_expr_await_mut<V>(v: &mut V, node: &mut expr::Await)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_expr_mut(&mut *node.expr);
}
pub fn visit_expr_binary_mut<V>(v: &mut V, node: &mut expr::Binary)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_expr_mut(&mut *node.left);
    v.visit_bin_op_mut(&mut self.op);
    v.visit_expr_mut(&mut *node.right);
}
pub fn visit_expr_block_mut<V>(v: &mut V, node: &mut expr::Block)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.label {
        v.visit_label_mut(x);
    }
    v.visit_block_mut(&mut self.block);
}
pub fn visit_expr_break_mut<V>(v: &mut V, node: &mut expr::Break)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.life {
        v.visit_lifetime_mut(x);
    }
    if let Some(x) = &mut self.val {
        v.visit_expr_mut(&mut **it);
    }
}
pub fn visit_expr_call_mut<V>(v: &mut V, node: &mut expr::Call)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_expr_mut(&mut *node.func);
    for mut el in Puncted::pairs_mut(&mut self.args) {
        let x = el.value_mut();
        v.visit_expr_mut(x);
    }
}
pub fn visit_expr_cast_mut<V>(v: &mut V, node: &mut expr::Cast)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_expr_mut(&mut *node.expr);
    v.visit_type_mut(&mut *node.typ);
}
pub fn visit_expr_closure_mut<V>(v: &mut V, node: &mut expr::Closurere)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.lifes {
        v.visit_bound_lifetimes_mut(x);
    }
    for mut el in Puncted::pairs_mut(&mut self.inputs) {
        let x = el.value_mut();
        v.visit_pat_mut(x);
    }
    v.visit_return_type_mut(&mut self.ret);
    v.visit_expr_mut(&mut *node.body);
}
pub fn visit_expr_const_mut<V>(v: &mut V, node: &mut expr::Constst)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_block_mut(&mut self.block);
}
pub fn visit_expr_continue_mut<V>(v: &mut V, node: &mut expr::Continue)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.life {
        v.visit_lifetime_mut(x);
    }
}
pub fn visit_expr_field_mut<V>(v: &mut V, node: &mut expr::Field)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_expr_mut(&mut *node.expr);
    v.visit_member_mut(&mut self.memb);
}
pub fn visit_expr_for_loop_mut<V>(v: &mut V, node: &mut expr::ForLoop)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.label {
        v.visit_label_mut(x);
    }
    v.visit_pat_mut(&mut *node.pat);
    v.visit_expr_mut(&mut *node.expr);
    v.visit_block_mut(&mut self.body);
}
pub fn visit_expr_group_mut<V>(v: &mut V, node: &mut expr::Group)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_expr_mut(&mut *node.expr);
}
pub fn visit_expr_if_mut<V>(v: &mut V, node: &mut expr::If)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_expr_mut(&mut *node.cond);
    v.visit_block_mut(&mut self.then_branch);
    if let Some(x) = &mut self.else_branch {
        v.visit_expr_mut(&mut *(x).1);
    }
}
pub fn visit_expr_index_mut<V>(v: &mut V, node: &mut expr::Index)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_expr_mut(&mut *node.expr);
    v.visit_expr_mut(&mut *node.index);
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
    v.visit_pat_mut(&mut *node.pat);
    v.visit_expr_mut(&mut *node.expr);
}
pub fn visit_expr_lit_mut<V>(v: &mut V, node: &mut expr::Lit)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_lit_mut(&mut self.lit);
}
pub fn visit_expr_loop_mut<V>(v: &mut V, node: &mut expr::Loop)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.label {
        v.visit_label_mut(x);
    }
    v.visit_block_mut(&mut self.body);
}
pub fn visit_expr_macro_mut<V>(v: &mut V, node: &mut expr::Mac)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_macro_mut(&mut self.mac);
}
pub fn visit_expr_match_mut<V>(v: &mut V, node: &mut expr::Match)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_expr_mut(&mut *node.expr);
    for x in &mut self.arms {
        v.visit_arm_mut(x);
    }
}
pub fn visit_expr_method_call_mut<V>(v: &mut V, node: &mut expr::MethodCall)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_expr_mut(&mut *node.expr);
    v.visit_ident_mut(&mut self.method);
    if let Some(x) = &mut self.turbofish {
        v.visit_angle_bracketed_generic_arguments_mut(x);
    }
    for mut el in Puncted::pairs_mut(&mut self.args) {
        let x = el.value_mut();
        v.visit_expr_mut(x);
    }
}
pub fn visit_expr_paren_mut<V>(v: &mut V, node: &mut expr::Parenth)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_expr_mut(&mut *node.expr);
}
pub fn visit_expr_path_mut<V>(v: &mut V, node: &mut expr::Path)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.qself {
        v.visit_qself_mut(x);
    }
    v.visit_path_mut(&mut self.path);
}
pub fn visit_expr_range_mut<V>(v: &mut V, node: &mut expr::Range)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.beg {
        v.visit_expr_mut(&mut **it);
    }
    v.visit_range_limits_mut(&mut self.limits);
    if let Some(x) = &mut self.end {
        v.visit_expr_mut(&mut **it);
    }
}
pub fn visit_expr_reference_mut<V>(v: &mut V, node: &mut expr::Ref)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_expr_mut(&mut *node.expr);
}
pub fn visit_expr_repeat_mut<V>(v: &mut V, node: &mut expr::Repeat)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_expr_mut(&mut *node.expr);
    v.visit_expr_mut(&mut *node.len);
}
pub fn visit_expr_return_mut<V>(v: &mut V, node: &mut expr::Return)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.expr {
        v.visit_expr_mut(&mut **it);
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
        v.visit_qself_mut(x);
    }
    v.visit_path_mut(&mut self.path);
    for mut el in Puncted::pairs_mut(&mut self.fields) {
        let x = el.value_mut();
        v.visit_field_value_mut(x);
    }
    if let Some(x) = &mut self.rest {
        v.visit_expr_mut(&mut **it);
    }
}
pub fn visit_expr_try_mut<V>(v: &mut V, node: &mut expr::Tryry)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_expr_mut(&mut *node.expr);
}
pub fn visit_expr_try_block_mut<V>(v: &mut V, node: &mut expr::TryBlock)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_block_mut(&mut self.block);
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
        v.visit_expr_mut(x);
    }
}
pub fn visit_expr_unary_mut<V>(v: &mut V, node: &mut expr::Unary)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_un_op_mut(&mut self.op);
    v.visit_expr_mut(&mut *node.expr);
}
pub fn visit_expr_unsafe_mut<V>(v: &mut V, node: &mut expr::Unsafe)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_block_mut(&mut self.block);
}
pub fn visit_expr_while_mut<V>(v: &mut V, node: &mut expr::While)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.label {
        v.visit_label_mut(x);
    }
    v.visit_expr_mut(&mut *node.cond);
    v.visit_block_mut(&mut self.body);
}
pub fn visit_expr_yield_mut<V>(v: &mut V, node: &mut expr::Yield)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.expr {
        v.visit_expr_mut(&mut **it);
    }
}
pub fn visit_field_mut<V>(v: &mut V, node: &mut data::Field)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    v.visit_field_mutability_mut(&mut self.mut_);
    if let Some(x) = &mut self.ident {
        v.visit_ident_mut(x);
    }
    v.visit_type_mut(&mut self.typ);
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
    v.visit_member_mut(&mut self.memb);
    v.visit_pat_mut(&mut *node.pat);
}
pub fn visit_field_value_mut<V>(v: &mut V, node: &mut FieldValue)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_member_mut(&mut self.member);
    v.visit_expr_mut(&mut self.expr);
}
pub fn visit_fields_mut<V>(v: &mut V, node: &mut data::Fields)
where
    V: VisitMut + ?Sized,
{
    match node {
        data::Fields::Named(x) => {
            v.visit_fields_named_mut(x);
        },
        data::Fields::Unnamed(x) => {
            v.visit_fields_unnamed_mut(x);
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
        v.visit_field_mut(x);
    }
}
pub fn visit_fields_unnamed_mut<V>(v: &mut V, node: &mut data::Unnamed)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.fields) {
        let x = el.value_mut();
        v.visit_field_mut(x);
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
        v.visit_item_mut(x);
    }
}
pub fn visit_fn_arg_mut<V>(v: &mut V, node: &mut item::FnArg)
where
    V: VisitMut + ?Sized,
{
    match node {
        item::FnArg::Receiver(x) => {
            v.visit_receiver_mut(x);
        },
        item::FnArg::Type(x) => {
            v.visit_pat_type_mut(x);
        },
    }
}
pub fn visit_foreign_item_mut<V>(v: &mut V, node: &mut item::foreign::Item)
where
    V: VisitMut + ?Sized,
{
    match node {
        item::foreign::Item::Fn(x) => {
            v.visit_foreign_item_fn_mut(x);
        },
        item::foreign::Item::Static(x) => {
            v.visit_foreign_item_static_mut(x);
        },
        item::foreign::Item::Type(x) => {
            v.visit_foreign_item_type_mut(x);
        },
        item::foreign::Item::Macro(x) => {
            v.visit_foreign_item_macro_mut(x);
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
    v.visit_visibility_mut(&mut self.vis);
    v.visit_signature_mut(&mut self.sig);
}
pub fn visit_foreign_item_macro_mut<V>(v: &mut V, node: &mut item::foreign::Mac)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_macro_mut(&mut self.mac);
}
pub fn visit_foreign_item_static_mut<V>(v: &mut V, node: &mut item::foreign::Static)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    v.visit_static_mutability_mut(&mut self.mut_);
    v.visit_ident_mut(&mut self.ident);
    v.visit_type_mut(&mut *node.typ);
}
pub fn visit_foreign_item_type_mut<V>(v: &mut V, node: &mut item::foreign::Type)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    v.visit_ident_mut(&mut self.ident);
    v.visit_generics_mut(&mut self.gens);
}
pub fn visit_generic_argument_mut<V>(v: &mut V, node: &mut Arg)
where
    V: VisitMut + ?Sized,
{
    match node {
        Arg::Life(x) => {
            v.visit_lifetime_mut(x);
        },
        Arg::Type(x) => {
            v.visit_type_mut(x);
        },
        Arg::Const(x) => {
            v.visit_expr_mut(x);
        },
        Arg::AssocType(x) => {
            v.visit_assoc_type_mut(x);
        },
        Arg::AssocConst(x) => {
            v.visit_assoc_const_mut(x);
        },
        Arg::Constraint(x) => {
            v.visit_constraint_mut(x);
        },
    }
}
pub fn visit_generic_param_mut<V>(v: &mut V, node: &mut gen::Param)
where
    V: VisitMut + ?Sized,
{
    match node {
        gen::Param::Life(x) => {
            v.visit_lifetime_param_mut(x);
        },
        gen::Param::Type(x) => {
            v.visit_type_param_mut(x);
        },
        gen::Param::Const(x) => {
            v.visit_const_param_mut(x);
        },
    }
}
pub fn visit_generics_mut<V>(v: &mut V, node: &mut gen::Gens)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.params) {
        let x = el.value_mut();
        v.visit_generic_param_mut(x);
    }
    if let Some(x) = &mut self.where_ {
        v.visit_where_clause_mut(x);
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
pub fn visit_impl_item_mut<V>(v: &mut V, node: &mut item::impl_::Item)
where
    V: VisitMut + ?Sized,
{
    match node {
        item::impl_::Item::Const(x) => {
            v.visit_impl_item_const_mut(x);
        },
        item::impl_::Item::Fn(x) => {
            v.visit_impl_item_fn_mut(x);
        },
        item::impl_::Item::Type(x) => {
            v.visit_impl_item_type_mut(x);
        },
        item::impl_::Item::Macro(x) => {
            v.visit_impl_item_macro_mut(x);
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
    v.visit_visibility_mut(&mut self.vis);
    v.visit_ident_mut(&mut self.ident);
    v.visit_generics_mut(&mut self.gens);
    v.visit_type_mut(&mut self.typ);
    v.visit_expr_mut(&mut self.expr);
}
pub fn visit_impl_item_fn_mut<V>(v: &mut V, node: &mut item::impl_::Fn)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    v.visit_signature_mut(&mut self.sig);
    v.visit_block_mut(&mut self.block);
}
pub fn visit_impl_item_macro_mut<V>(v: &mut V, node: &mut item::impl_::Mac)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_macro_mut(&mut self.mac);
}
pub fn visit_impl_item_type_mut<V>(v: &mut V, node: &mut item::impl_::Type)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    v.visit_ident_mut(&mut self.ident);
    v.visit_generics_mut(&mut self.gens);
    v.visit_type_mut(&mut self.typ);
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
    v.visit_span_mut(&mut self.span);
}
pub fn visit_item_mut<V>(v: &mut V, node: &mut Item)
where
    V: VisitMut + ?Sized,
{
    match node {
        Item::Const(x) => {
            v.visit_item_const_mut(x);
        },
        Item::Enum(x) => {
            v.visit_item_enum_mut(x);
        },
        Item::Extern(x) => {
            v.visit_item_extern_crate_mut(x);
        },
        Item::Fn(x) => {
            v.visit_item_fn_mut(x);
        },
        Item::Foreign(x) => {
            v.visit_item_foreign_mod_mut(x);
        },
        Item::Impl(x) => {
            v.visit_item_impl_mut(x);
        },
        Item::Macro(x) => {
            v.visit_item_macro_mut(x);
        },
        Item::Mod(x) => {
            v.visit_item_mod_mut(x);
        },
        Item::Static(x) => {
            v.visit_item_static_mut(x);
        },
        Item::Struct(x) => {
            v.visit_item_struct_mut(x);
        },
        Item::Trait(x) => {
            v.visit_item_trait_mut(x);
        },
        Item::TraitAlias(x) => {
            v.visit_item_trait_alias_mut(x);
        },
        Item::Type(x) => {
            v.visit_item_type_mut(x);
        },
        Item::Union(x) => {
            v.visit_item_union_mut(x);
        },
        Item::Use(x) => {
            v.visit_item_use_mut(x);
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
    v.visit_visibility_mut(&mut self.vis);
    v.visit_ident_mut(&mut self.ident);
    v.visit_generics_mut(&mut self.gens);
    v.visit_type_mut(&mut *node.typ);
    v.visit_expr_mut(&mut *node.expr);
}
pub fn visit_item_enum_mut<V>(v: &mut V, node: &mut item::Enum)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    v.visit_ident_mut(&mut self.ident);
    v.visit_generics_mut(&mut self.gens);
    for mut el in Puncted::pairs_mut(&mut self.variants) {
        let x = el.value_mut();
        v.visit_variant_mut(x);
    }
}
pub fn visit_item_extern_crate_mut<V>(v: &mut V, node: &mut item::Extern)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    v.visit_ident_mut(&mut self.ident);
    if let Some(x) = &mut self.rename {
        v.visit_ident_mut(&mut (x).1);
    }
}
pub fn visit_item_fn_mut<V>(v: &mut V, node: &mut item::Fn)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    v.visit_signature_mut(&mut self.sig);
    v.visit_block_mut(&mut *node.block);
}
pub fn visit_item_foreign_mod_mut<V>(v: &mut V, node: &mut item::Foreign)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_abi_mut(&mut self.abi);
    for x in &mut self.items {
        v.visit_foreign_item_mut(x);
    }
}
pub fn visit_item_impl_mut<V>(v: &mut V, node: &mut item::Impl)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_generics_mut(&mut self.gens);
    if let Some(x) = &mut self.trait_ {
        v.visit_path_mut(&mut (x).1);
    }
    v.visit_type_mut(&mut *node.typ);
    for x in &mut self.items {
        v.visit_impl_item_mut(x);
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
        v.visit_ident_mut(x);
    }
    v.visit_macro_mut(&mut self.mac);
}
pub fn visit_item_mod_mut<V>(v: &mut V, node: &mut item::Mod)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    v.visit_ident_mut(&mut self.ident);
    if let Some(x) = &mut self.items {
        for x in &mut (x).1 {
            v.visit_item_mut(x);
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
    v.visit_visibility_mut(&mut self.vis);
    v.visit_static_mutability_mut(&mut self.mut_);
    v.visit_ident_mut(&mut self.ident);
    v.visit_type_mut(&mut *node.typ);
    v.visit_expr_mut(&mut *node.expr);
}
pub fn visit_item_struct_mut<V>(v: &mut V, node: &mut item::Struct)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    v.visit_ident_mut(&mut self.ident);
    v.visit_generics_mut(&mut self.gens);
    v.visit_fields_mut(&mut self.fields);
}
pub fn visit_item_trait_mut<V>(v: &mut V, node: &mut item::Trait)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    if let Some(x) = &mut self.restriction {
        v.visit_impl_restriction_mut(x);
    }
    v.visit_ident_mut(&mut self.ident);
    v.visit_generics_mut(&mut self.gens);
    for mut el in Puncted::pairs_mut(&mut self.supers) {
        let x = el.value_mut();
        v.visit_type_param_bound_mut(x);
    }
    for x in &mut self.items {
        v.visit_trait_item_mut(x);
    }
}
pub fn visit_item_trait_alias_mut<V>(v: &mut V, node: &mut item::TraitAlias)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    v.visit_ident_mut(&mut self.ident);
    v.visit_generics_mut(&mut self.gens);
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        v.visit_type_param_bound_mut(x);
    }
}
pub fn visit_item_type_mut<V>(v: &mut V, node: &mut item::Type)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    v.visit_ident_mut(&mut self.ident);
    v.visit_generics_mut(&mut self.gens);
    v.visit_type_mut(&mut *node.typ);
}
pub fn visit_item_union_mut<V>(v: &mut V, node: &mut item::Union)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    v.visit_ident_mut(&mut self.ident);
    v.visit_generics_mut(&mut self.gens);
    v.visit_fields_named_mut(&mut self.fields);
}
pub fn visit_item_use_mut<V>(v: &mut V, node: &mut item::Use)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_visibility_mut(&mut self.vis);
    v.visit_use_tree_mut(&mut self.tree);
}
pub fn visit_label_mut<V>(v: &mut V, node: &mut Label)
where
    V: VisitMut + ?Sized,
{
    v.visit_lifetime_mut(&mut self.name);
}
pub fn visit_lifetime_mut<V>(v: &mut V, node: &mut Life)
where
    V: VisitMut + ?Sized,
{
    v.visit_span_mut(&mut self.apos);
    v.visit_ident_mut(&mut self.ident);
}
pub fn visit_lifetime_param_mut<V>(v: &mut V, node: &mut gen::param::Life)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_lifetime_mut(&mut self.life);
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        v.visit_lifetime_mut(x);
    }
}
pub fn visit_lit_mut<V>(v: &mut V, node: &mut Lit)
where
    V: VisitMut + ?Sized,
{
    match node {
        Lit::Str(x) => {
            v.visit_lit_str_mut(x);
        },
        Lit::ByteStr(x) => {
            v.visit_lit_byte_str_mut(x);
        },
        Lit::Byte(x) => {
            v.visit_lit_byte_mut(x);
        },
        Lit::Char(x) => {
            v.visit_lit_char_mut(x);
        },
        Lit::Int(x) => {
            v.visit_lit_int_mut(x);
        },
        Lit::Float(x) => {
            v.visit_lit_float_mut(x);
        },
        Lit::Bool(x) => {
            v.visit_lit_bool_mut(x);
        },
        Lit::Stream(x) => {},
    }
}
pub fn visit_lit_bool_mut<V>(v: &mut V, node: &mut lit::Bool)
where
    V: VisitMut + ?Sized,
{
    v.visit_span_mut(&mut self.span);
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
    v.visit_pat_mut(&mut self.pat);
    if let Some(x) = &mut self.init {
        v.visit_local_init_mut(x);
    }
}
pub fn visit_local_init_mut<V>(v: &mut V, node: &mut stmt::Init)
where
    V: VisitMut + ?Sized,
{
    v.visit_expr_mut(&mut *node.expr);
    if let Some(x) = &mut self.diverge {
        v.visit_expr_mut(&mut *(x).1);
    }
}
pub fn visit_macro_mut<V>(v: &mut V, node: &mut Macro)
where
    V: VisitMut + ?Sized,
{
    v.visit_path_mut(&mut self.path);
    v.visit_macro_delimiter_mut(&mut self.delim);
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
            v.visit_ident_mut(x);
        },
        Member::Unnamed(x) => {
            v.visit_index_mut(x);
        },
    }
}
pub fn visit_meta_list_mut<V>(v: &mut V, node: &mut attr::List)
where
    V: VisitMut + ?Sized,
{
    v.visit_path_mut(&mut self.path);
    v.visit_macro_delimiter_mut(&mut self.delim);
}
pub fn visit_meta_name_value_mut<V>(v: &mut V, node: &mut attr::NameValue)
where
    V: VisitMut + ?Sized,
{
    v.visit_path_mut(&mut self.name);
    v.visit_expr_mut(&mut self.val);
}
pub fn visit_parenthesized_generic_arguments_mut<V>(v: &mut V, node: &mut path::Parenthed)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.ins) {
        let x = el.value_mut();
        v.visit_type_mut(x);
    }
    v.visit_return_type_mut(&mut self.out);
}
pub fn visit_pat_mut<V>(v: &mut V, node: &mut pat::Pat)
where
    V: VisitMut + ?Sized,
{
    match node {
        pat::Pat::Const(x) => {
            v.visit_expr_const_mut(x);
        },
        pat::Pat::Ident(x) => {
            v.visit_pat_ident_mut(x);
        },
        pat::Pat::Lit(x) => {
            v.visit_expr_lit_mut(x);
        },
        pat::Pat::Mac(x) => {
            v.visit_expr_macro_mut(x);
        },
        pat::Pat::Or(x) => {
            v.visit_pat_or_mut(x);
        },
        pat::Pat::Parenth(x) => {
            v.visit_pat_paren_mut(x);
        },
        pat::Pat::Path(x) => {
            v.visit_expr_path_mut(x);
        },
        pat::Pat::Range(x) => {
            v.visit_expr_range_mut(x);
        },
        pat::Pat::Ref(x) => {
            v.visit_pat_reference_mut(x);
        },
        pat::Pat::Rest(x) => {
            v.visit_pat_rest_mut(x);
        },
        pat::Pat::Slice(x) => {
            v.visit_pat_slice_mut(x);
        },
        pat::Pat::Struct(x) => {
            v.visit_pat_struct_mut(x);
        },
        pat::Pat::Tuple(x) => {
            v.visit_pat_tuple_mut(x);
        },
        pat::Pat::TupleStruct(x) => {
            v.visit_pat_tuple_struct_mut(x);
        },
        pat::Pat::Type(x) => {
            v.visit_pat_type_mut(x);
        },
        pat::Pat::Verbatim(x) => {},
        pat::Pat::Wild(x) => {
            v.visit_pat_wild_mut(x);
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
    v.visit_ident_mut(&mut self.ident);
    if let Some(x) = &mut self.sub {
        v.visit_pat_mut(&mut *(x).1);
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
        v.visit_pat_mut(x);
    }
}
pub fn visit_pat_paren_mut<V>(v: &mut V, node: &mut pat::Parenth)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_pat_mut(&mut *node.pat);
}
pub fn visit_pat_reference_mut<V>(v: &mut V, node: &mut pat::Ref)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_pat_mut(&mut *node.pat);
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
        v.visit_pat_mut(x);
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
        v.visit_qself_mut(x);
    }
    v.visit_path_mut(&mut self.path);
    for mut el in Puncted::pairs_mut(&mut self.fields) {
        let x = el.value_mut();
        v.visit_field_pat_mut(x);
    }
    if let Some(x) = &mut self.rest {
        v.visit_pat_rest_mut(x);
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
        v.visit_pat_mut(x);
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
        v.visit_qself_mut(x);
    }
    v.visit_path_mut(&mut self.path);
    for mut el in Puncted::pairs_mut(&mut self.pats) {
        let x = el.value_mut();
        v.visit_pat_mut(x);
    }
}
pub fn visit_pat_type_mut<V>(v: &mut V, node: &mut pat::Type)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_pat_mut(&mut *node.pat);
    v.visit_type_mut(&mut *node.typ);
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
        v.visit_path_segment_mut(x);
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
            v.visit_angle_bracketed_generic_arguments_mut(x);
        },
        Parenthed(x) => {
            v.visit_parenthesized_generic_arguments_mut(x);
        },
    }
}
pub fn visit_path_segment_mut<V>(v: &mut V, node: &mut Segment)
where
    V: VisitMut + ?Sized,
{
    v.visit_ident_mut(&mut self.ident);
    v.visit_path_arguments_mut(&mut self.args);
}
pub fn visit_predicate_lifetime_mut<V>(v: &mut V, node: &mut gen::Where::Life)
where
    V: VisitMut + ?Sized,
{
    v.visit_lifetime_mut(&mut self.life);
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        v.visit_lifetime_mut(x);
    }
}
pub fn visit_predicate_type_mut<V>(v: &mut V, node: &mut gen::Where::Type)
where
    V: VisitMut + ?Sized,
{
    if let Some(x) = &mut self.lifes {
        v.visit_bound_lifetimes_mut(x);
    }
    v.visit_type_mut(&mut self.bounded);
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        v.visit_type_param_bound_mut(x);
    }
}
pub fn visit_qself_mut<V>(v: &mut V, node: &mut QSelf)
where
    V: VisitMut + ?Sized,
{
    v.visit_type_mut(&mut *node.ty);
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
            v.visit_lifetime_mut(x);
        }
    }
    v.visit_type_mut(&mut *node.typ);
}
pub fn visit_return_type_mut<V>(v: &mut V, node: &mut typ::Ret)
where
    V: VisitMut + ?Sized,
{
    match node {
        typ::Ret::Default => {},
        typ::Ret::Type(_binding_0, _binding_1) => {
            v.visit_type_mut(&mut **_binding_1);
        },
    }
}
pub fn visit_signature_mut<V>(v: &mut V, node: &mut item::Sig)
where
    V: VisitMut + ?Sized,
{
    if let Some(x) = &mut self.abi {
        v.visit_abi_mut(x);
    }
    v.visit_ident_mut(&mut self.ident);
    v.visit_generics_mut(&mut self.gens);
    for mut el in Puncted::pairs_mut(&mut self.args) {
        let x = el.value_mut();
        v.visit_fn_arg_mut(x);
    }
    if let Some(x) = &mut self.vari {
        v.visit_variadic_mut(x);
    }
    v.visit_return_type_mut(&mut self.ret);
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
            v.visit_local_mut(x);
        },
        stmt::Stmt::Item(x) => {
            v.visit_item_mut(x);
        },
        stmt::Stmt::Expr(_binding_0, _binding_1) => {
            v.visit_expr_mut(x);
        },
        stmt::Stmt::Mac(x) => {
            v.visit_stmt_macro_mut(x);
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
    v.visit_macro_mut(&mut self.mac);
}
pub fn visit_trait_bound_mut<V>(v: &mut V, node: &mut gen::bound::Trait)
where
    V: VisitMut + ?Sized,
{
    v.visit_trait_bound_modifier_mut(&mut self.modif);
    if let Some(x) = &mut self.lifes {
        v.visit_bound_lifetimes_mut(x);
    }
    v.visit_path_mut(&mut self.path);
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
            v.visit_trait_item_const_mut(x);
        },
        item::trait_::Item::Fn(x) => {
            v.visit_trait_item_fn_mut(x);
        },
        item::trait_::Item::Type(x) => {
            v.visit_trait_item_type_mut(x);
        },
        item::trait_::Item::Macro(x) => {
            v.visit_trait_item_macro_mut(x);
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
    v.visit_ident_mut(&mut self.ident);
    v.visit_generics_mut(&mut self.gens);
    v.visit_type_mut(&mut self.typ);
    if let Some(x) = &mut self.default {
        v.visit_expr_mut(&mut (x).1);
    }
}
pub fn visit_trait_item_fn_mut<V>(v: &mut V, node: &mut item::trait_::Fn)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_signature_mut(&mut self.sig);
    if let Some(x) = &mut self.default {
        v.visit_block_mut(x);
    }
}
pub fn visit_trait_item_macro_mut<V>(v: &mut V, node: &mut item::trait_::Mac)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_macro_mut(&mut self.mac);
}
pub fn visit_trait_item_type_mut<V>(v: &mut V, node: &mut item::trait_::Type)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_ident_mut(&mut self.ident);
    v.visit_generics_mut(&mut self.gens);
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        v.visit_type_param_bound_mut(x);
    }
    if let Some(x) = &mut self.default {
        v.visit_type_mut(&mut (x).1);
    }
}
pub fn visit_type_mut<V>(v: &mut V, node: &mut typ::Type)
where
    V: VisitMut + ?Sized,
{
    match node {
        typ::Type::Array(x) => {
            v.visit_type_array_mut(x);
        },
        typ::Type::Fn(x) => {
            v.visit_type_bare_fn_mut(x);
        },
        typ::Type::Group(x) => {
            v.visit_type_group_mut(x);
        },
        typ::Type::Impl(x) => {
            v.visit_type_impl_trait_mut(x);
        },
        typ::Type::Infer(x) => {
            v.visit_type_infer_mut(x);
        },
        typ::Type::Mac(x) => {
            v.visit_type_macro_mut(x);
        },
        typ::Type::Never(x) => {
            v.visit_type_never_mut(x);
        },
        typ::Type::Parenth(x) => {
            v.visit_type_paren_mut(x);
        },
        typ::Type::Path(x) => {
            v.visit_type_path_mut(x);
        },
        typ::Type::Ptr(x) => {
            v.visit_type_ptr_mut(x);
        },
        typ::Type::Ref(x) => {
            v.visit_type_reference_mut(x);
        },
        typ::Type::Slice(x) => {
            v.visit_type_slice_mut(x);
        },
        typ::Type::Trait(x) => {
            v.visit_type_trait_object_mut(x);
        },
        typ::Type::Tuple(x) => {
            v.visit_type_tuple_mut(x);
        },
        typ::Type::Stream(x) => {},
    }
}
pub fn visit_type_array_mut<V>(v: &mut V, node: &mut typ::Array)
where
    V: VisitMut + ?Sized,
{
    v.visit_type_mut(&mut *node.elem);
    v.visit_expr_mut(&mut self.len);
}
pub fn visit_type_bare_fn_mut<V>(v: &mut V, node: &mut typ::Fn)
where
    V: VisitMut + ?Sized,
{
    if let Some(x) = &mut self.lifes {
        v.visit_bound_lifetimes_mut(x);
    }
    if let Some(x) = &mut self.abi {
        v.visit_abi_mut(x);
    }
    for mut el in Puncted::pairs_mut(&mut self.args) {
        let x = el.value_mut();
        v.visit_bare_fn_arg_mut(x);
    }
    if let Some(x) = &mut self.vari {
        v.visit_bare_variadic_mut(x);
    }
    v.visit_return_type_mut(&mut self.ret);
}
pub fn visit_type_group_mut<V>(v: &mut V, node: &mut typ::Group)
where
    V: VisitMut + ?Sized,
{
    v.visit_type_mut(&mut *node.elem);
}
pub fn visit_type_impl_trait_mut<V>(v: &mut V, node: &mut typ::Impl)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        v.visit_type_param_bound_mut(x);
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
    v.visit_macro_mut(&mut self.mac);
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
    v.visit_ident_mut(&mut self.ident);
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        v.visit_type_param_bound_mut(x);
    }
    if let Some(x) = &mut self.default {
        v.visit_type_mut(x);
    }
}
pub fn visit_type_param_bound_mut<V>(v: &mut V, node: &mut gen::bound::Type)
where
    V: VisitMut + ?Sized,
{
    match node {
        gen::bound::Type::Trait(x) => {
            v.visit_trait_bound_mut(x);
        },
        gen::bound::Type::Life(x) => {
            v.visit_lifetime_mut(x);
        },
        gen::bound::Type::Verbatim(x) => {},
    }
}
pub fn visit_type_paren_mut<V>(v: &mut V, node: &mut typ::Parenth)
where
    V: VisitMut + ?Sized,
{
    v.visit_type_mut(&mut *node.elem);
}
pub fn visit_type_path_mut<V>(v: &mut V, node: &mut typ::Path)
where
    V: VisitMut + ?Sized,
{
    if let Some(x) = &mut self.qself {
        v.visit_qself_mut(x);
    }
    v.visit_path_mut(&mut self.path);
}
pub fn visit_type_ptr_mut<V>(v: &mut V, node: &mut typ::Ptr)
where
    V: VisitMut + ?Sized,
{
    v.visit_type_mut(&mut *node.elem);
}
pub fn visit_type_reference_mut<V>(v: &mut V, node: &mut typ::Ref)
where
    V: VisitMut + ?Sized,
{
    if let Some(x) = &mut self.life {
        v.visit_lifetime_mut(x);
    }
    v.visit_type_mut(&mut *node.elem);
}
pub fn visit_type_slice_mut<V>(v: &mut V, node: &mut typ::Slice)
where
    V: VisitMut + ?Sized,
{
    v.visit_type_mut(&mut *node.elem);
}
pub fn visit_type_trait_object_mut<V>(v: &mut V, node: &mut typ::Trait)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        v.visit_type_param_bound_mut(x);
    }
}
pub fn visit_type_tuple_mut<V>(v: &mut V, node: &mut typ::Tuple)
where
    V: VisitMut + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.elems) {
        let x = el.value_mut();
        v.visit_type_mut(x);
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
        v.visit_use_tree_mut(x);
    }
}
pub fn visit_use_name_mut<V>(v: &mut V, node: &mut item::use_::Name)
where
    V: VisitMut + ?Sized,
{
    v.visit_ident_mut(&mut self.ident);
}
pub fn visit_use_path_mut<V>(v: &mut V, node: &mut item::use_::Path)
where
    V: VisitMut + ?Sized,
{
    v.visit_ident_mut(&mut self.ident);
    v.visit_use_tree_mut(&mut *node.tree);
}
pub fn visit_use_rename_mut<V>(v: &mut V, node: &mut item::use_::Rename)
where
    V: VisitMut + ?Sized,
{
    v.visit_ident_mut(&mut self.ident);
    v.visit_ident_mut(&mut self.rename);
}
pub fn visit_use_tree_mut<V>(v: &mut V, node: &mut item::use_::Tree)
where
    V: VisitMut + ?Sized,
{
    match node {
        item::use_::Tree::Path(x) => {
            v.visit_use_path_mut(x);
        },
        item::use_::Tree::Name(x) => {
            v.visit_use_name_mut(x);
        },
        item::use_::Tree::Rename(x) => {
            v.visit_use_rename_mut(x);
        },
        item::use_::Tree::Glob(x) => {
            v.visit_use_glob_mut(x);
        },
        item::use_::Tree::Group(x) => {
            v.visit_use_group_mut(x);
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
        v.visit_pat_mut(&mut *(x).0);
    }
}
pub fn visit_variant_mut<V>(v: &mut V, node: &mut data::Variant)
where
    V: VisitMut + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    v.visit_ident_mut(&mut self.ident);
    v.visit_fields_mut(&mut self.fields);
    if let Some(x) = &mut self.discrim {
        v.visit_expr_mut(&mut (x).1);
    }
}
pub fn visit_vis_restricted_mut<V>(v: &mut V, node: &mut data::Restricted)
where
    V: VisitMut + ?Sized,
{
    v.visit_path_mut(&mut *node.path);
}
pub fn visit_visibility_mut<V>(v: &mut V, node: &mut data::Visibility)
where
    V: VisitMut + ?Sized,
{
    match node {
        data::Visibility::Public(x) => {},
        data::Visibility::Restricted(x) => {
            v.visit_vis_restricted_mut(x);
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
        v.visit_where_predicate_mut(x);
    }
}
pub fn visit_where_predicate_mut<V>(v: &mut V, node: &mut gen::Where::Pred)
where
    V: VisitMut + ?Sized,
{
    match node {
        gen::Where::Pred::Life(x) => {
            v.visit_predicate_lifetime_mut(x);
        },
        gen::Where::Pred::Type(x) => {
            v.visit_predicate_type_mut(x);
        },
    }
}
