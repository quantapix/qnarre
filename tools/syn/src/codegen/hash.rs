use crate::StreamHelper;
use crate::*;
use std::hash::{Hash, Hasher};
impl<H> Hash for gen::bound::Lifes
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.lifes.hash(h);
    }
}
impl<H> Hash for gen::param::Const
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.typ.hash(h);
        self.eq.hash(h);
        self.default.hash(h);
    }
}
impl<H> Hash for gen::Param
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use gen::Param::*;
        match self {
            Life(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Const(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
        }
    }
}
impl<H> Hash for gen::Gens
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.lt.hash(h);
        self.params.hash(h);
        self.gt.hash(h);
        self.where_.hash(h);
    }
}
impl<H> Hash for gen::param::Life
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.life.hash(h);
        self.colon.hash(h);
        self.bounds.hash(h);
    }
}
impl<H> Hash for mac::Mac
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.path.hash(h);
        self.delim.hash(h);
        StreamHelper(&self.toks).hash(h);
    }
}
impl<H> Hash for gen::where_::Life
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.life.hash(h);
        self.bounds.hash(h);
    }
}
impl<H> Hash for gen::where_::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.lifes.hash(h);
        self.typ.hash(h);
        self.bounds.hash(h);
    }
}
impl<H> Hash for gen::bound::Trait
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.parenth.hash(h);
        self.modif.hash(h);
        self.lifes.hash(h);
        self.path.hash(h);
    }
}
impl<H> Hash for gen::bound::Modifier
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use gen::bound::Modifier::*;
        match self {
            None => {
                h.write_u8(0u8);
            },
            Maybe(_) => {
                h.write_u8(1u8);
            },
        }
    }
}
impl<H> Hash for gen::param::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.colon.hash(h);
        self.bounds.hash(h);
        self.eq.hash(h);
        self.default.hash(h);
    }
}
impl<H> Hash for gen::bound::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use gen::bound::Type::*;
        match self {
            Trait(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Life(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(2u8);
                StreamHelper(x).hash(h);
            },
        }
    }
}
impl<H> Hash for gen::Where
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.preds.hash(h);
    }
}
impl<H> Hash for gen::where_::Pred
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use gen::where_::Pred::*;
        match self {
            Life(v0) => {
                h.write_u8(0u8);
                v0.hash(h);
            },
            Type(v0) => {
                h.write_u8(1u8);
                v0.hash(h);
            },
        }
    }
}
