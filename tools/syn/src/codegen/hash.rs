use crate::StreamHelper;
use crate::*;
use std::hash::{Hash, Hasher};
impl<H> Hash for stmt::Block
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.stmts.hash(h);
    }
}
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
impl<H> Hash for pat::Field
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.memb.hash(h);
        self.colon.hash(h);
        self.pat.hash(h);
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
impl<H> Hash for lit::Lit
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use lit::Lit::*;
        match self {
            Str(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            ByteStr(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Byte(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Char(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Int(x) => {
                h.write_u8(4u8);
                x.hash(h);
            },
            Float(x) => {
                h.write_u8(5u8);
                x.hash(h);
            },
            Bool(x) => {
                h.write_u8(6u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(7u8);
                x.to_string().hash(h);
            },
        }
    }
}
impl<H> Hash for lit::Bool
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.val.hash(h);
    }
}
impl<H> Hash for stmt::Local
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.init.hash(h);
    }
}
impl<H> Hash for stmt::Init
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.expr.hash(h);
        self.diverge.hash(h);
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
impl<H> Hash for tok::Delim
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use tok::Delim::*;
        match self {
            Parenth(_) => {
                h.write_u8(0u8);
            },
            Brace(_) => {
                h.write_u8(1u8);
            },
            Bracket(_) => {
                h.write_u8(2u8);
            },
        }
    }
}
impl<H> Hash for pat::Pat
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use pat::Pat::*;
        match self {
            Const(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Ident(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Lit(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Mac(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Or(x) => {
                h.write_u8(4u8);
                x.hash(h);
            },
            Parenth(x) => {
                h.write_u8(5u8);
                x.hash(h);
            },
            Path(x) => {
                h.write_u8(6u8);
                x.hash(h);
            },
            Range(x) => {
                h.write_u8(7u8);
                x.hash(h);
            },
            Ref(x) => {
                h.write_u8(8u8);
                x.hash(h);
            },
            Rest(x) => {
                h.write_u8(9u8);
                x.hash(h);
            },
            Slice(x) => {
                h.write_u8(10u8);
                x.hash(h);
            },
            Struct(x) => {
                h.write_u8(11u8);
                x.hash(h);
            },
            Tuple(x) => {
                h.write_u8(12u8);
                x.hash(h);
            },
            TupleStruct(x) => {
                h.write_u8(13u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(14u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(15u8);
                StreamHelper(x).hash(h);
            },
            Wild(x) => {
                h.write_u8(16u8);
                x.hash(h);
            },
        }
    }
}
impl<H> Hash for pat::Ident
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ref_.hash(h);
        self.mut_.hash(h);
        self.ident.hash(h);
        self.sub.hash(h);
    }
}
impl<H> Hash for pat::Or
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vert.hash(h);
        self.cases.hash(h);
    }
}
impl<H> Hash for pat::Parenth
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pat.hash(h);
    }
}
impl<H> Hash for pat::Ref
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.mut_.hash(h);
        self.pat.hash(h);
    }
}
impl<H> Hash for pat::Rest
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
    }
}
impl<H> Hash for pat::Slice
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pats.hash(h);
    }
}
impl<H> Hash for pat::Struct
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.qself.hash(h);
        self.path.hash(h);
        self.fields.hash(h);
        self.rest.hash(h);
    }
}
impl<H> Hash for pat::Tuple
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pats.hash(h);
    }
}
impl<H> Hash for pat::TupleStruct
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.qself.hash(h);
        self.path.hash(h);
        self.pats.hash(h);
    }
}
impl<H> Hash for pat::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.typ.hash(h);
    }
}
impl<H> Hash for pat::Wild
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
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
impl<H> Hash for stmt::Stmt
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use stmt::Stmt::*;
        match self {
            Local(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Item(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Expr(x, v1) => {
                h.write_u8(2u8);
                x.hash(h);
                v1.hash(h);
            },
            Mac(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
        }
    }
}
impl<H> Hash for stmt::Mac
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
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
