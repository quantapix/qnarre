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

impl VisitMut for pat::Field {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.memb.visit_mut(v);
    &mut *self.pat.visit_mut(v);
}
}
impl VisitMut for item::File {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    for x in &mut self.items {
        x.visit_mut(v);
    }
}
}
impl VisitMut for item::FnArg {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
        item::FnArg::Receiver(x) => {
            x.visit_mut(v);
        },
        item::FnArg::Type(x) => {
            x.visit_mut(v);
        },
    }
}
}
impl VisitMut for item::foreign::Item {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
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
}
impl VisitMut for item::foreign::Fn {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.sig.visit_mut(v);
}
}
impl VisitMut for item::foreign::Mac {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.mac.visit_mut(v);
}
}
impl VisitMut for item::foreign::Static {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.mut_.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut *self.typ.visit_mut(v);
}
}
impl VisitMut for item::foreign::Type {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
}
}
impl VisitMut for Arg {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
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
}
impl VisitMut for gen::Param {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
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
}
impl VisitMut for gen::Gens {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.params) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.where_ {
        x.visit_mut(v);
    }
}
}
impl VisitMut for Ident {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    let mut span = self.span();
    &mut span.visit_mut(v);
    self.set_span(span);
}
}
impl VisitMut for item::impl_::Item {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
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
}
impl VisitMut for item::impl_::Const {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl VisitMut for item::impl_::Fn {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.sig.visit_mut(v);
    &mut self.block.visit_mut(v);
}
}
impl VisitMut for item::impl_::Mac {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.mac.visit_mut(v);
}
}
impl VisitMut for item::impl_::Type {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    &mut self.typ.visit_mut(v);
}
}
impl VisitMut for item::impl_::Restriction {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match *self {}
}
}
impl VisitMut for Index {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut self.span.visit_mut(v);
}
}
impl VisitMut for Item {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
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
}
impl VisitMut for item::Const {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    &mut *self.typ.visit_mut(v);
    &mut *self.expr.visit_mut(v);
}
}
impl VisitMut for item::Enum {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl VisitMut for item::Extern {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl VisitMut for item::Fn {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.sig.visit_mut(v);
    &mut *self.block.visit_mut(v);
}
}
impl VisitMut for item::Foreign {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.abi.visit_mut(v);
    for x in &mut self.items {
        x.visit_mut(v);
    }
}
}
impl VisitMut for item::Impl {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.gens.visit_mut(v);
    if let Some(x) = &mut self.trait_ {
        &mut (x).1.visit_mut(v);
    }
    &mut *self.typ.visit_mut(v);
    for x in &mut self.items {
        x.visit_mut(v);
    }
}
}
impl VisitMut for item::Mac {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.ident {
        x.visit_mut(v);
    }
    &mut self.mac.visit_mut(v);
}
}
impl VisitMut for item::Mod {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl VisitMut for item::Static {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.mut_.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut *self.typ.visit_mut(v);
    &mut *self.expr.visit_mut(v);
}
}
impl VisitMut for item::Struct {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    &mut self.fields.visit_mut(v);
}
}
impl VisitMut for item::Trait {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl VisitMut for item::TraitAlias {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl VisitMut for item::Type {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    &mut *self.typ.visit_mut(v);
}
}
impl VisitMut for item::Union {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.ident.visit_mut(v);
    &mut self.gens.visit_mut(v);
    &mut self.fields.visit_mut(v);
}
}
impl VisitMut for item::Use {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.vis.visit_mut(v);
    &mut self.tree.visit_mut(v);
}
}
impl VisitMut for Label {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut self.name.visit_mut(v);
}
}
impl VisitMut for Life {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut self.apos.visit_mut(v);
    &mut self.ident.visit_mut(v);
}
}
impl VisitMut for gen::param::Life {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl VisitMut for Lit {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
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
}
impl VisitMut for lit::Bool {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut self.span.visit_mut(v);
}
}
impl VisitMut for lit::Byte {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl VisitMut for lit::ByteStr {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl VisitMut for lit::Char {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl VisitMut for lit::Float {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl VisitMut for lit::Int {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl VisitMut for lit::Str {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl VisitMut for stmt::Local {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.pat.visit_mut(v);
    if let Some(x) = &mut self.init {
        x.visit_mut(v);
    }
}
}
impl VisitMut for stmt::Init {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut *self.expr.visit_mut(v);
    if let Some(x) = &mut self.diverge {
        &mut *(x).1.visit_mut(v);
    }
}
}
impl VisitMut for Macro {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut self.path.visit_mut(v);
    &mut self.delim.visit_mut(v);
}
}
impl VisitMut for tok::Delim {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
        tok::Delim::Parenth(x) => {},
        tok::Delim::Brace(x) => {},
        tok::Delim::Bracket(x) => {},
    }
}
}
impl VisitMut for Member {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
        Member::Named(x) => {
            x.visit_mut(v);
        },
        Member::Unnamed(x) => {
            x.visit_mut(v);
        },
    }
}
}
impl VisitMut for attr::List {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut self.path.visit_mut(v);
    &mut self.delim.visit_mut(v);
}
}
impl VisitMut for attr::NameValue {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut self.name.visit_mut(v);
    &mut self.val.visit_mut(v);
}
}
impl VisitMut for path::Parenthed {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.ins) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
    &mut self.out.visit_mut(v);
}
}
impl VisitMut for pat::Pat {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
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
}
impl VisitMut for pat::Ident {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.ident.visit_mut(v);
    if let Some(x) = &mut self.sub {
        &mut *(x).1.visit_mut(v);
    }
}
}
impl VisitMut for pat::Or {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    for mut el in Puncted::pairs_mut(&mut self.cases) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
}
impl VisitMut for pat::Parenth {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *self.pat.visit_mut(v);
}
}
impl VisitMut for pat::Ref {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *self.pat.visit_mut(v);
}
}
impl VisitMut for pat::Rest {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
}
}
impl VisitMut for pat::Slice {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    for mut el in Puncted::pairs_mut(&mut self.pats) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
}
impl VisitMut for pat::Struct {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl VisitMut for pat::Tuple {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    for mut el in Puncted::pairs_mut(&mut self.pats) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
}
impl VisitMut for pat::TupleStruct {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl VisitMut for pat::Type {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut *self.pat.visit_mut(v);
    &mut *self.typ.visit_mut(v);
}
}
impl VisitMut for pat::Wild {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
}
}
impl VisitMut for Path {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.segs) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
}
impl VisitMut for path::Args {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    use path::Args::*;
    match self {
        None => {},
        Angled(x) => {
            x.visit_mut(v);
        },
        Parenthed(x) => {
            x.visit_mut(v);
        },
    }
}
}
impl VisitMut for Segment {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut self.ident.visit_mut(v);
    &mut self.args.visit_mut(v);
}
}
impl VisitMut for gen::Where::Life {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut self.life.visit_mut(v);
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
}
impl VisitMut for gen::Where::Type {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl VisitMut for QSelf {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut *self.typ.visit_mut(v);
}
}
impl VisitMut for item::Receiver {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.ref_ {
        if let Some(x) = &mut (x).1 {
            x.visit_mut(v);
        }
    }
    &mut *self.typ.visit_mut(v);
}
}
impl VisitMut for typ::Ret {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
        typ::Ret::Default => {},
        typ::Ret::Type(_binding_0, _binding_1) => {
            &mut **_binding_1.visit_mut(v);
        },
    }
}
}
impl VisitMut for item::Sig {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl VisitMut for pm2::Span {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl VisitMut for StaticMut {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
        StaticMut::Mut(x) => {},
        StaticMut::None => {},
    }
}
}
impl VisitMut for stmt::Stmt {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
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
}
impl VisitMut for stmt::Mac {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.mac.visit_mut(v);
}
}
impl VisitMut for gen::bound::Trait {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut self.modif.visit_mut(v);
    if let Some(x) = &mut self.lifes {
        x.visit_mut(v);
    }
    &mut self.path.visit_mut(v);
}
}
impl VisitMut for gen::bound::Modifier {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
        gen::bound::Modifier::None => {},
        gen::bound::Modifier::Maybe(x) => {},
    }
}
}
impl VisitMut for item::trait_::Item {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
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
}
impl VisitMut for item::trait_::Const {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl VisitMut for item::trait_::Fn {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.sig.visit_mut(v);
    if let Some(x) = &mut self.default {
        x.visit_mut(v);
    }
}
}
impl VisitMut for item::trait_::Mac {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    &mut self.mac.visit_mut(v);
}
}
impl VisitMut for item::trait_::Type {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl VisitMut for typ::Type {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
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
}
impl VisitMut for typ::Array {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut *self.elem.visit_mut(v);
    &mut self.len.visit_mut(v);
}
}
impl VisitMut for typ::Fn {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl VisitMut for typ::Group {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut *self.elem.visit_mut(v);
}
}
impl VisitMut for typ::Impl {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
}
impl VisitMut for typ::Infer {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl VisitMut for typ::Mac {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut self.mac.visit_mut(v);
}
}
impl VisitMut for typ::Never {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl VisitMut for gen::param::Type {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
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
}
impl VisitMut for gen::bound::Type {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
        gen::bound::Type::Trait(x) => {
            x.visit_mut(v);
        },
        gen::bound::Type::Life(x) => {
            x.visit_mut(v);
        },
        gen::bound::Type::Verbatim(x) => {},
    }
}
}
impl VisitMut for typ::Parenth {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut *self.elem.visit_mut(v);
}
}
impl VisitMut for typ::Path {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    if let Some(x) = &mut self.qself {
        x.visit_mut(v);
    }
    &mut self.path.visit_mut(v);
}
}
impl VisitMut for typ::Ptr {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut *self.elem.visit_mut(v);
}
}
impl VisitMut for typ::Ref {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    if let Some(x) = &mut self.life {
        x.visit_mut(v);
    }
    &mut *self.elem.visit_mut(v);
}
}
impl VisitMut for typ::Slice {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut *self.elem.visit_mut(v);
}
}
impl VisitMut for typ::Trait {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.bounds) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
}
impl VisitMut for typ::Tuple {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.elems) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
}
impl VisitMut for UnOp {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
        UnOp::Deref(x) => {},
        UnOp::Not(x) => {},
        UnOp::Neg(x) => {},
    }
}
}
impl VisitMut for item::use_::Glob {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
}
}
impl VisitMut for item::use_::Group {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.trees) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
}
impl VisitMut for item::use_::Name {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut self.ident.visit_mut(v);
}
}
impl VisitMut for item::use_::Path {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut self.ident.visit_mut(v);
    &mut *self.tree.visit_mut(v);
}
}
impl VisitMut for item::use_::Rename {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    &mut self.ident.visit_mut(v);
    &mut self.rename.visit_mut(v);
}
}
impl VisitMut for item::use_::Tree {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
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
}
impl VisitMut for item::Variadic {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for x in &mut self.attrs {
        x.visit_mut(v);
    }
    if let Some(x) = &mut self.pat {
        &mut *(x).0.visit_mut(v);
    }
}
}
impl VisitMut for gen::Where {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    for mut el in Puncted::pairs_mut(&mut self.preds) {
        let x = el.value_mut();
        x.visit_mut(v);
    }
}
}
impl VisitMut for gen::Where::Pred {
    fn visit_mut<V>(&mut self, v: &mut V)
where
    V: Visitor + ?Sized,
{
    match self {
        gen::Where::Pred::Life(x) => {
            x.visit_mut(v);
        },
        gen::Where::Pred::Type(x) => {
            x.visit_mut(v);
        },
    }
}
