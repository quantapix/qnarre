#![allow(unused_variables)]

use crate::*;


impl Visit for gen::bound::Lifes {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for y in Puncted::pairs(&self.lifes) {
        let x = y.value();
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
    for y in Puncted::pairs(&self.params) {
        let x = y.value();
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
    for y in Puncted::pairs(&self.variants) {
        let x = y.value();
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
    for y in Puncted::pairs(&self.supers) {
        let x = y.value();
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
    for y in Puncted::pairs(&self.bounds) {
        let x = y.value();
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
    for y in Puncted::pairs(&self.bounds) {
        let x = y.value();
        x.visit(v);
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
impl Visit for gen::where_::Life {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &self.life.visit(v);
    for y in Puncted::pairs(&self.bounds) {
        let x = y.value();
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
    for y in Puncted::pairs(&self.bounds) {
        let x = y.value();
        x.visit(v);
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
    for y in Puncted::pairs(&self.args) {
        let x = y.value();
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
    for y in Puncted::pairs(&self.bounds) {
        let x = y.value();
        x.visit(v);
    }
    if let Some(x) = &self.default {
        
        &(x).1.visit(v);
    }
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
    for y in Puncted::pairs(&self.bounds) {
        let x = y.value();
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
    for y in Puncted::pairs(&self.trees) {
        let x = y.value();
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
impl Visit for gen::Where {
fn visit<V>(&self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for y in Puncted::pairs(&self.preds) {
        let x = y.value();
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
