#![allow(unused_variables)]

use crate::*;


impl Visit for gen::bound::Lifes {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for el in Puncted::pairs(&self.lifes) {
        let x = el.value();
        x.visit(v);
    }
}
}
impl Visit for gen::param::Const {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for expr::Expr {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for expr::Array {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    for el in Puncted::pairs(&self.elems) {
        let x = el.value();
        x.visit(v);
    }
}
}
impl Visit for expr::Assign {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.left.visit(v);
    &*self.right.visit(v);
}
}
impl Visit for expr::Async {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.block.visit(v);
}
}
impl Visit for expr::Await {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
}
}
impl Visit for expr::Binary {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.left.visit(v);
    &self.op.visit(v);
    &*self.right.visit(v);
}
}
impl Visit for expr::Block {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.label {
        x.visit(v);
    }
    &self.block.visit(v);
}
}
impl Visit for expr::Break {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for expr::Call {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for expr::Cast {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
    &*self.typ.visit(v);
}
}
impl Visit for expr::Closure {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for expr::Const {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.block.visit(v);
}
}
impl Visit for expr::Continue {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.life {
        x.visit(v);
    }
}
}
impl Visit for expr::Field {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
    &self.memb.visit(v);
}
}
impl Visit for expr::ForLoop {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for expr::Group {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
}
}
impl Visit for expr::If {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for expr::Index {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
    &*self.index.visit(v);
}
}
impl Visit for expr::Infer {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
}
}
impl Visit for expr::Let {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.pat.visit(v);
    &*self.expr.visit(v);
}
}
impl Visit for expr::Lit {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.lit.visit(v);
}
}
impl Visit for expr::Loop {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.label {
        x.visit(v);
    }
    &self.body.visit(v);
}
}
impl Visit for expr::Mac {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.mac.visit(v);
}
}
impl Visit for expr::Match {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
    for x in &self.arms {
        x.visit(v);
    }
}
}
impl Visit for expr::MethodCall {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for expr::Parenth {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
}
}
impl Visit for expr::Path {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.qself {
        x.visit(v);
    }
    &self.path.visit(v);
}
}
impl Visit for expr::Range {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for expr::Ref {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
}
}
impl Visit for expr::Repeat {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
    &*self.len.visit(v);
}
}
impl Visit for expr::Return {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.expr {
        &**x.visit(v);
    }
}
}
impl Visit for expr::Struct {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for expr::Try {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.expr.visit(v);
}
}
impl Visit for expr::TryBlock {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.block.visit(v);
}
}
impl Visit for expr::Tuple {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    for el in Puncted::pairs(&self.elems) {
        let x = el.value();
        x.visit(v);
    }
}
}
impl Visit for expr::Unary {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.op.visit(v);
    &*self.expr.visit(v);
}
}
impl Visit for expr::Unsafe {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.block.visit(v);
}
}
impl Visit for expr::While {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for expr::Yield {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.expr {
        &**x.visit(v);
    }
}
}
impl Visit for data::Field {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for data::Mut {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
        data::Mut::None => {},
    }
}
}
impl Visit for pat::Field {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.memb.visit(v);
    &*self.pat.visit(v);
}
}
impl Visit for expr::FieldValue {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.member.visit(v);
    &self.expr.visit(v);
}
}
impl Visit for data::Fields {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for data::Named {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for el in Puncted::pairs(&self.fields) {
        let x = el.value();
        x.visit(v);
    }
}
}
impl Visit for data::Unnamed {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for el in Puncted::pairs(&self.fields) {
        let x = el.value();
        x.visit(v);
    }
}
}
impl Visit for item::File {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    for x in &self.items {
        x.visit(v);
    }
}
}
impl Visit for item::FnArg {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::foreign::Item {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::foreign::Fn {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.sig.visit(v);
}
}
impl Visit for item::foreign::Mac {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.mac.visit(v);
}
}
impl Visit for item::foreign::Static {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.mut_.visit(v);
    &self.ident.visit(v);
    &*self.typ.visit(v);
}
}
impl Visit for item::foreign::Type {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    &self.gens.visit(v);
}
}
impl Visit for path::Arg {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for gen::Param {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for gen::Gens {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for el in Puncted::pairs(&self.params) {
        let x = el.value();
        x.visit(v);
    }
    if let Some(x) = &self.where_ {
        x.visit(v);
    }
}
}
impl Visit for Ident {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.span().visit(v);
}
}
impl Visit for item::impl_::Item {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::impl_::Const {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::impl_::Fn {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.sig.visit(v);
    &self.block.visit(v);
}
}
impl Visit for item::impl_::Mac {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.mac.visit(v);
}
}
impl Visit for item::impl_::Type {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    &self.gens.visit(v);
    &self.typ.visit(v);
}
}
impl Visit for item::impl_::Restriction {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match *self {}
}
}
impl Visit for expr::Index {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.span.visit(v);
}
}
impl Visit for item::Item {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::Const {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::Enum {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::Extern {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::Fn {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.sig.visit(v);
    &*self.block.visit(v);
}
}
impl Visit for item::Foreign {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.abi.visit(v);
    for x in &self.items {
        x.visit(v);
    }
}
}
impl Visit for item::Impl {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::Mac {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.ident {
        x.visit(v);
    }
    &self.mac.visit(v);
}
}
impl Visit for item::Mod {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::Static {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::Struct {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    &self.gens.visit(v);
    &self.fields.visit(v);
}
}
impl Visit for item::Trait {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::TraitAlias {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::Type {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    &self.gens.visit(v);
    &*self.typ.visit(v);
}
}
impl Visit for item::Union {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.ident.visit(v);
    &self.gens.visit(v);
    &self.fields.visit(v);
}
}
impl Visit for item::Use {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.vis.visit(v);
    &self.tree.visit(v);
}
}
impl Visit for expr::Label {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.name.visit(v);
}
}
impl Visit for Life {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.apos.visit(v);
    &self.ident.visit(v);
}
}
impl Visit for gen::param::Life {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for lit::Lit {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for lit::Bool {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.span.visit(v);
}
}
impl Visit for lit::Byte {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl Visit for lit::ByteStr {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl Visit for lit::Char {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl Visit for lit::Float {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl Visit for lit::Int {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl Visit for lit::Str {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl Visit for stmt::Local {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.pat.visit(v);
    if let Some(x) = &self.init {
        x.visit(v);
    }
}
}
impl Visit for stmt::Init {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &*self.expr.visit(v);
    if let Some(x) = &self.diverge {
        
        &*(x).1.visit(v);
    }
}
}
impl Visit for mac::Mac {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.path.visit(v);
    &self.delim.visit(v);
}
}
impl Visit for tok::Delim {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for expr::Member {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for attr::Meta {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for attr::List {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.path.visit(v);
    &self.delim.visit(v);
}
}
impl Visit for attr::NameValue {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.name.visit(v);
    &self.val.visit(v);
}
}
impl Visit for path::Parenthed {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for el in Puncted::pairs(&self.ins) {
        let x = el.value();
        x.visit(v);
    }
    &self.out.visit(v);
}
}
impl Visit for pat::Pat {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for pat::Ident {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.ident.visit(v);
    if let Some(x) = &self.sub {
        
        &*(x).1.visit(v);
    }
}
}
impl Visit for pat::Or {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    for el in Puncted::pairs(&self.cases) {
        let x = el.value();
        x.visit(v);
    }
}
}
impl Visit for pat::Parenth {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.pat.visit(v);
}
}
impl Visit for pat::Ref {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.pat.visit(v);
}
}
impl Visit for pat::Rest {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
}
}
impl Visit for pat::Slice {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    for el in Puncted::pairs(&self.pats) {
        let x = el.value();
        x.visit(v);
    }
}
}
impl Visit for pat::Struct {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for pat::Tuple {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    for el in Puncted::pairs(&self.pats) {
        let x = el.value();
        x.visit(v);
    }
}
}
impl Visit for pat::TupleStruct {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for pat::Type {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &*self.pat.visit(v);
    &*self.typ.visit(v);
}
}
impl Visit for pat::Wild {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
}
}
impl Visit for Path {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for el in Puncted::pairs(&self.segs) {
        let x = el.value();
        x.visit(v);
    }
}
}
impl Visit for path::Args {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for path::Segment {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.ident.visit(v);
    &self.args.visit(v);
}
}
impl Visit for gen::where_::Life {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.life.visit(v);
    for el in Puncted::pairs(&self.bounds) {
        let x = el.value();
        x.visit(v);
    }
}
}
impl Visit for gen::where_::Type {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for path::QSelf {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &*self.ty.visit(v);
}
}
impl Visit for expr::Limits {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    use expr::Limits::*;
    match self {
        HalfOpen(x) => {
            
        },
        Closed(x) => {
            
        },
    }
}
}
impl Visit for item::Receiver {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for typ::Ret {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    use typ::Ret::*;
    match self {
        Default => {},
        Type(_binding_0, _binding_1) => {
            
            &**_binding_1.visit(v);
        },
    }
}
}
impl Visit for item::Sig {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
    V: Visitor + ?Sized,
{
}
}
impl Visit for item::StaticMut {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    use item::StaticMut::*;
    match self {
        Mut(x) => {
            
        },
        None => {},
    }
}
}
impl Visit for stmt::Stmt {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for stmt::Mac {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.mac.visit(v);
}
}
impl Visit for gen::bound::Trait {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.modif.visit(v);
    if let Some(x) = &self.lifes {
        x.visit(v);
    }
    &self.path.visit(v);
}
}
impl Visit for gen::bound::Modifier {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    use gen::bound::Modifier::*;
    match self {
        None => {},
        Maybe(x) => {
            
        },
    }
}
}
impl Visit for item::trait_::Item {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::trait_::Const {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::trait_::Fn {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.sig.visit(v);
    if let Some(x) = &self.default {
        x.visit(v);
    }
}
}
impl Visit for item::trait_::Mac {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    &self.mac.visit(v);
}
}
impl Visit for item::trait_::Type {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for typ::Type {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for typ::Array {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &*self.elem.visit(v);
    &self.len.visit(v);
}
}
impl Visit for typ::Fn {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for typ::Group {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &*self.elem.visit(v);
}
}
impl Visit for typ::Impl {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for el in Puncted::pairs(&self.bounds) {
        let x = el.value();
        x.visit(v);
    }
}
}
impl Visit for typ::Infer {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl Visit for typ::Mac {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.mac.visit(v);
}
}
impl Visit for typ::Never {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl Visit for gen::param::Type {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for gen::bound::Type {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for typ::Parenth {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &*self.elem.visit(v);
}
}
impl Visit for typ::Path {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    if let Some(x) = &self.qself {
        x.visit(v);
    }
    &self.path.visit(v);
}
}
impl Visit for typ::Ptr {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &*self.elem.visit(v);
}
}
impl Visit for typ::Ref {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    if let Some(x) = &self.life {
        x.visit(v);
    }
    &*self.elem.visit(v);
}
}
impl Visit for typ::Slice {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &*self.elem.visit(v);
}
}
impl Visit for typ::Trait {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for el in Puncted::pairs(&self.bounds) {
        let x = el.value();
        x.visit(v);
    }
}
}
impl Visit for typ::Tuple {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for el in Puncted::pairs(&self.elems) {
        let x = el.value();
        x.visit(v);
    }
}
}
impl Visit for expr::UnOp {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::use_::Glob {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl Visit for item::use_::Group {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for el in Puncted::pairs(&self.trees) {
        let x = el.value();
        x.visit(v);
    }
}
}
impl Visit for item::use_::Name {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.ident.visit(v);
}
}
impl Visit for item::use_::Path {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.ident.visit(v);
    &*self.tree.visit(v);
}
}
impl Visit for item::use_::Rename {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.ident.visit(v);
    &self.rename.visit(v);
}
}
impl Visit for item::use_::Tree {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for item::Variadic {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &self.attrs {
        x.visit(v);
    }
    if let Some(x) = &self.pat {
        &*(x).0.visit(v);
        
    }
}
}
impl Visit for data::Variant {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for data::Restricted {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &*self.path.visit(v);
}
}
impl Visit for data::Visibility {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl Visit for gen::Where {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for el in Puncted::pairs(&self.preds) {
        let x = el.value();
        x.visit(v);
    }
}
}
impl Visit for gen::where_::Pred {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
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
