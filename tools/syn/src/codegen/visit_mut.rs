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
            item::foreign::Item::Verbatim(_) => {},
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
            item::impl_::Item::Verbatim(_) => {},
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
            Item::Verbatim(_) => {},
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
        for mut y in Puncted::pairs_mut(&mut self.variants) {
            let x = y.value_mut();
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
        for mut y in Puncted::pairs_mut(&mut self.supers) {
            let x = y.value_mut();
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
        for mut y in Puncted::pairs_mut(&mut self.bounds) {
            let x = y.value_mut();
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
impl VisitMut for Life {
    fn visit_mut<V>(&mut self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        &mut self.apos.visit_mut(v);
        &mut self.ident.visit_mut(v);
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
            tok::Delim::Parenth(_) => {},
            tok::Delim::Brace(_) => {},
            tok::Delim::Bracket(_) => {},
        }
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
        for mut y in Puncted::pairs_mut(&mut self.args) {
            let x = y.value_mut();
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
            StaticMut::Mut(_) => {},
            StaticMut::None => {},
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
            item::trait_::Item::Verbatim(_) => {},
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
        for mut y in Puncted::pairs_mut(&mut self.bounds) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
        if let Some(x) = &mut self.default {
            &mut (x).1.visit_mut(v);
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
        for mut y in Puncted::pairs_mut(&mut self.trees) {
            let x = y.value_mut();
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
