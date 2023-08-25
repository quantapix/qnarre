#![allow(unreachable_code, unused_variables)]
#![allow(clippy::match_wildcard_for_single_variants, clippy::needless_match)]

use crate::*;

macro_rules! full {
    ($e:expr) => {
        $e
    };
}

impl<F> Fold for gen::bound::Lifes
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        Bgen::bound::Lifes {
            for_: self.for_,
            lt: self.lt,
            lifes: FoldHelper::lift(self.lifes, |x| x.fold(f)),
            gt: self.gt,
        }
    }
}
impl<F> Fold for gen::param::Const
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::param::Const {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            const_: self.const_,
            ident: self.ident.fold(f),
            colon: self.colon,
            typ: self.typ.fold(f),
            eq: self.eq,
            default: (self.default).map(|x| x.fold(f)),
        }
    }
}
impl<F> Fold for gen::Param
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            gen::Param::Life(x) => gen::Param::Life(x.fold(f)),
            gen::Param::Type(x) => gen::Param::Type(x.fold(f)),
            gen::Param::Const(x) => gen::Param::Const(x.fold(f)),
        }
    }
}
impl<F> Fold for gen::Gens
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::Gens {
            lt: self.lt,
            params: FoldHelper::lift(self.params, |x| x.fold(f)),
            gt: self.gt,
            where_: (self.where_).map(|x| x.fold(f)),
        }
    }
}
impl<F> Fold for gen::param::Life
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::param::Life {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            life: self.life.fold(f),
            colon: self.colon,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for gen::Where::Life
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::Where::Life {
            life: self.life.fold(f),
            colon: self.colon,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for gen::Where::Type
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::Where::Type {
            lifes: (self.lifes).map(|x| x.fold(f)),
            bounded: self.bounded.fold(f),
            colon: self.colon,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for pm2::Span
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        node
    }
}
impl<F> Fold for gen::bound::Trait
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::bound::Trait {
            parenth: self.parenth,
            modif: self.modif.fold(f),
            lifes: (self.lifes).map(|x| x.fold(f)),
            path: self.path.fold(f),
        }
    }
}
impl<F> Fold for gen::bound::Modifier
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            gen::bound::Modifier::None => gen::bound::Modifier::None,
            gen::bound::Modifier::Maybe(x) => gen::bound::Modifier::Maybe(x),
        }
    }
}
impl<F> Fold for gen::param::Type
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::param::Type {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            ident: self.ident.fold(f),
            colon: self.colon,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
            eq: self.eq,
            default: (self.default).map(|x| x.fold(f)),
        }
    }
}
impl<F> Fold for gen::bound::Type
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            gen::bound::Type::Trait(x) => gen::bound::Type::Trait(x.fold(f)),
            gen::bound::Type::Life(x) => gen::bound::Type::Life(x.fold(f)),
            gen::bound::Type::Verbatim(x) => gen::bound::Type::Verbatim(x),
        }
    }
}
impl<F> Fold for gen::Where
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        gen::Where {
            where_: self.where_,
            preds: FoldHelper::lift(self.preds, |x| x.fold(f)),
        }
    }
}
impl<F> Fold for gen::Where::Pred
where
    F: Folder + ?Sized,
{
    fn fold(&self, f: &mut F) {
        match self {
            gen::Where::Pred::Life(x) => gen::Where::Pred::Life(x.fold(f)),
            gen::Where::Pred::Type(x) => gen::Where::Pred::Type(x.fold(f)),
        }
    }
}
