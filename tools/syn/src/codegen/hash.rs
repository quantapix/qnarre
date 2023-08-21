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
impl<H> Hash for item::File
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.shebang.hash(h);
        self.attrs.hash(h);
        self.items.hash(h);
    }
}
impl<H> Hash for item::FnArg
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use item::FnArg::*;
        match self {
            Receiver(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
        }
    }
}
impl<H> Hash for item::foreign::Item
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use item::foreign::Item::*;
        match self {
            Fn(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Static(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Mac(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(4u8);
                StreamHelper(x).hash(h);
            },
        }
    }
}
impl<H> Hash for item::foreign::Fn
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.sig.hash(h);
    }
}
impl<H> Hash for item::foreign::Mac
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for item::foreign::Static
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.mut_.hash(h);
        self.ident.hash(h);
        self.typ.hash(h);
    }
}
impl<H> Hash for item::foreign::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
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
impl<H> Hash for item::impl_::Item
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use item::impl_::Item::*;
        match self {
            Const(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Fn(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Mac(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(4u8);
                StreamHelper(x).hash(h);
            },
        }
    }
}
impl<H> Hash for item::impl_::Const
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.default.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.typ.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for item::impl_::Fn
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.default.hash(h);
        self.sig.hash(h);
        self.block.hash(h);
    }
}
impl<H> Hash for item::impl_::Mac
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for item::impl_::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.default.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.typ.hash(h);
    }
}
impl<H> Hash for item::impl_::Restriction
where
    H: Hasher,
{
    fn hash(&self, _h: &mut H) {
        match *self {}
    }
}
impl<H> Hash for item::Item
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use item::Item::*;
        match self {
            Const(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Enum(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Extern(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Fn(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Foreign(x) => {
                h.write_u8(4u8);
                x.hash(h);
            },
            Impl(x) => {
                h.write_u8(5u8);
                x.hash(h);
            },
            Mac(x) => {
                h.write_u8(6u8);
                x.hash(h);
            },
            Mod(x) => {
                h.write_u8(7u8);
                x.hash(h);
            },
            Static(x) => {
                h.write_u8(8u8);
                x.hash(h);
            },
            Struct(x) => {
                h.write_u8(9u8);
                x.hash(h);
            },
            Trait(x) => {
                h.write_u8(10u8);
                x.hash(h);
            },
            TraitAlias(x) => {
                h.write_u8(11u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(12u8);
                x.hash(h);
            },
            Union(x) => {
                h.write_u8(13u8);
                x.hash(h);
            },
            Use(x) => {
                h.write_u8(14u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(15u8);
                StreamHelper(x).hash(h);
            },
        }
    }
}
impl<H> Hash for item::Const
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.typ.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for item::Enum
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.variants.hash(h);
    }
}
impl<H> Hash for item::Extern
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.rename.hash(h);
    }
}
impl<H> Hash for item::Fn
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.sig.hash(h);
        self.block.hash(h);
    }
}
impl<H> Hash for item::Foreign
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.unsafe_.hash(h);
        self.abi.hash(h);
        self.items.hash(h);
    }
}
impl<H> Hash for item::Impl
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.default.hash(h);
        self.unsafe_.hash(h);
        self.gens.hash(h);
        self.trait_.hash(h);
        self.typ.hash(h);
        self.items.hash(h);
    }
}
impl<H> Hash for item::Mac
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for item::Mod
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.unsafe_.hash(h);
        self.ident.hash(h);
        self.items.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for item::Static
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.mut_.hash(h);
        self.ident.hash(h);
        self.typ.hash(h);
        self.expr.hash(h);
    }
}
impl<H> Hash for item::Struct
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.fields.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for item::Trait
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.unsafe_.hash(h);
        self.auto.hash(h);
        self.restriction.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.colon.hash(h);
        self.supers.hash(h);
        self.items.hash(h);
    }
}
impl<H> Hash for item::TraitAlias
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.bounds.hash(h);
    }
}
impl<H> Hash for item::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.typ.hash(h);
    }
}
impl<H> Hash for item::Union
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.fields.hash(h);
    }
}
impl<H> Hash for item::Use
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.colon.hash(h);
        self.tree.hash(h);
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
impl<H> Hash for item::Receiver
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ref_.hash(h);
        self.mut_.hash(h);
        self.colon.hash(h);
        self.typ.hash(h);
    }
}
impl<H> Hash for item::Sig
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.const_.hash(h);
        self.async_.hash(h);
        self.unsafe_.hash(h);
        self.abi.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.args.hash(h);
        self.vari.hash(h);
        self.ret.hash(h);
    }
}
impl<H> Hash for item::StaticMut
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use item::StaticMut::*;
        match self {
            Mut(_) => {
                h.write_u8(0u8);
            },
            None => {
                h.write_u8(1u8);
            },
        }
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
impl<H> Hash for item::trait_::Item
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use item::trait_::Item::*;
        match self {
            Const(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Fn(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Mac(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(4u8);
                StreamHelper(x).hash(h);
            },
        }
    }
}
impl<H> Hash for item::trait_::Const
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.typ.hash(h);
        self.default.hash(h);
    }
}
impl<H> Hash for item::trait_::Fn
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.sig.hash(h);
        self.default.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for item::trait_::Mac
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
    }
}
impl<H> Hash for item::trait_::Type
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.colon.hash(h);
        self.bounds.hash(h);
        self.default.hash(h);
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
impl<H> Hash for item::use_::Glob
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {}
}
impl<H> Hash for item::use_::Group
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.trees.hash(h);
    }
}
impl<H> Hash for item::use_::Name
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.ident.hash(h);
    }
}
impl<H> Hash for item::use_::Path
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.ident.hash(h);
        self.tree.hash(h);
    }
}
impl<H> Hash for item::use_::Rename
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.ident.hash(h);
        self.rename.hash(h);
    }
}
impl<H> Hash for item::use_::Tree
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use item::use_::Tree::*;
        match self {
            Path(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Name(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Rename(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Glob(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Group(x) => {
                h.write_u8(4u8);
                x.hash(h);
            },
        }
    }
}
impl<H> Hash for item::Variadic
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.comma.hash(h);
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
