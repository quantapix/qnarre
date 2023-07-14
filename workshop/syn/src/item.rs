use crate::attr::Filter;

use super::*;
use std::mem;

pub struct File {
    pub shebang: Option<String>,
    pub attrs: Vec<attr::Attr>,
    pub items: Vec<Item>,
}
impl Parse for File {
    fn parse(s: Stream) -> Res<Self> {
        Ok(File {
            shebang: None,
            attrs: s.call(attr::Attr::parse_inners)?,
            items: {
                let mut ys = Vec::new();
                while !s.is_empty() {
                    ys.push(s.parse()?);
                }
                ys
            },
        })
    }
}
impl ToTokens for File {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.inners());
        ys.append_all(&self.items);
    }
}

ast_enum_of_structs! {
    pub enum Item {
        Const(Const),
        Enum(Enum),
        Extern(Extern),
        Fn(Fn),
        Foreign(Foreign),
        Impl(Impl),
        Mac(Mac),
        Mod(Mod),
        Static(Static),
        Stream(Stream),
        Struct(Struct),
        Trait(Trait),
        TraitAlias(TraitAlias),
        Type(Type),
        Union(Union),
        Use(Use),
    }
}
impl Item {
    pub fn replace_attrs(&mut self, ys: Vec<attr::Attr>) -> Vec<attr::Attr> {
        match self {
            Const(Const { attrs, .. })
            | Enum(Enum { attrs, .. })
            | Extern(Extern { attrs, .. })
            | Fn(Fn { attrs, .. })
            | Foreign(Foreign { attrs, .. })
            | Impl(Impl { attrs, .. })
            | Mac(Mac { attrs, .. })
            | Mod(Mod { attrs, .. })
            | Static(Static { attrs, .. })
            | Struct(Struct { attrs, .. })
            | Trait(Trait { attrs, .. })
            | TraitAlias(TraitAlias { attrs, .. })
            | Type(Type { attrs, .. })
            | Union(Union { attrs, .. })
            | Use(Use { attrs, .. }) => mem::replace(attrs, ys),
            Stream(_) => Vec::new(),
        }
    }
}
impl From<DeriveInput> for Item {
    fn from(x: DeriveInput) -> Item {
        match x.data {
            data::Data::Struct(y) => Item::Struct(Struct {
                attrs: x.attrs,
                vis: x.vis,
                struct_: y.struct_,
                ident: x.ident,
                gens: x.gens,
                fields: y.fields,
                semi: y.semi,
            }),
            data::Data::Enum(y) => Item::Enum(Enum {
                attrs: x.attrs,
                vis: x.vis,
                enum_: y.enum_,
                ident: x.ident,
                gens: x.gens,
                brace: y.brace,
                variants: y.variants,
            }),
            data::Data::Union(y) => Item::Union(Union {
                attrs: x.attrs,
                vis: x.vis,
                union_: y.union_,
                ident: x.ident,
                gens: x.gens,
                fields: y.fields,
            }),
        }
    }
}
impl Parse for Item {
    fn parse(s: Stream) -> Res<Self> {
        let beg = s.fork();
        let attrs = s.call(attr::Attr::parse_outers)?;
        parse_rest_of_item(beg, attrs, s)
    }
}

pub struct Const {
    pub attrs: Vec<attr::Attr>,
    pub vis: data::Visibility,
    pub const_: Token![const],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub colon: Token![:],
    pub typ: Box<typ::Type>,
    pub eq: Token![=],
    pub expr: Box<expr::Expr>,
    pub semi: Token![;],
}
impl Parse for Const {
    fn parse(s: Stream) -> Res<Self> {
        Ok(Const {
            attrs: s.call(attr::Attr::parse_outers)?,
            vis: s.parse()?,
            const_: s.parse()?,
            ident: {
                let y = s.look1();
                if y.peek(Ident) || y.peek(Token![_]) {
                    s.call(Ident::parse_any)?
                } else {
                    return Err(y.error());
                }
            },
            gens: gen::Gens::default(),
            colon: s.parse()?,
            typ: s.parse()?,
            eq: s.parse()?,
            expr: s.parse()?,
            semi: s.parse()?,
        })
    }
}
impl ToTokens for Const {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.vis.to_tokens(ys);
        self.const_.to_tokens(ys);
        self.ident.to_tokens(ys);
        self.colon.to_tokens(ys);
        self.typ.to_tokens(ys);
        self.eq.to_tokens(ys);
        self.expr.to_tokens(ys);
        self.semi.to_tokens(ys);
    }
}

pub struct Enum {
    pub attrs: Vec<attr::Attr>,
    pub vis: data::Visibility,
    pub enum_: Token![enum],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub brace: tok::Brace,
    pub variants: Puncted<data::Variant, Token![,]>,
}
impl From<Enum> for DeriveInput {
    fn from(x: Enum) -> DeriveInput {
        DeriveInput {
            attrs: x.attrs,
            vis: x.vis,
            ident: x.ident,
            gens: x.gens,
            data: data::Data::Enum(data::Enum {
                enum_: x.enum_,
                brace: x.brace,
                variants: x.variants,
            }),
        }
    }
}
impl Parse for Enum {
    fn parse(s: Stream) -> Res<Self> {
        let attrs = s.call(attr::Attr::parse_outers)?;
        let vis = s.parse::<data::Visibility>()?;
        let enum_ = s.parse::<Token![enum]>()?;
        let ident = s.parse::<Ident>()?;
        let gens = s.parse::<gen::Gens>()?;
        let (where_, brace, variants) = data::parse_enum(s)?;
        Ok(Enum {
            attrs,
            vis,
            enum_,
            ident,
            gens: gen::Gens { where_, ..gens },
            brace,
            variants,
        })
    }
}
impl ToTokens for Enum {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.vis.to_tokens(ys);
        self.enum_.to_tokens(ys);
        self.ident.to_tokens(ys);
        self.gens.to_tokens(ys);
        self.gens.where_.to_tokens(ys);
        self.brace.surround(ys, |ys| {
            self.variants.to_tokens(ys);
        });
    }
}

pub struct Extern {
    pub attrs: Vec<attr::Attr>,
    pub vis: data::Visibility,
    pub extern_: Token![extern],
    pub crate_: Token![crate],
    pub ident: Ident,
    pub rename: Option<(Token![as], Ident)>,
    pub semi: Token![;],
}
impl Parse for Extern {
    fn parse(s: Stream) -> Res<Self> {
        Ok(Extern {
            attrs: s.call(attr::Attr::parse_outers)?,
            vis: s.parse()?,
            extern_: s.parse()?,
            crate_: s.parse()?,
            ident: {
                if s.peek(Token![self]) {
                    s.call(Ident::parse_any)?
                } else {
                    s.parse()?
                }
            },
            rename: {
                if s.peek(Token![as]) {
                    let as_: Token![as] = s.parse()?;
                    let y: Ident = if s.peek(Token![_]) {
                        Ident::from(s.parse::<Token![_]>()?)
                    } else {
                        s.parse()?
                    };
                    Some((as_, y))
                } else {
                    None
                }
            },
            semi: s.parse()?,
        })
    }
}
impl ToTokens for Extern {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.vis.to_tokens(ys);
        self.extern_.to_tokens(ys);
        self.crate_.to_tokens(ys);
        self.ident.to_tokens(ys);
        if let Some((as_, y)) = &self.rename {
            as_.to_tokens(ys);
            y.to_tokens(ys);
        }
        self.semi.to_tokens(ys);
    }
}

pub struct Fn {
    pub attrs: Vec<attr::Attr>,
    pub vis: data::Visibility,
    pub sig: Sig,
    pub block: Box<stmt::Block>,
}
impl Parse for Fn {
    fn parse(s: Stream) -> Res<Self> {
        let attrs = s.call(attr::Attr::parse_outers)?;
        let vis: data::Visibility = s.parse()?;
        let sig: Sig = s.parse()?;
        parse_rest_of_fn(s, attrs, vis, sig)
    }
}
impl ToTokens for Fn {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.vis.to_tokens(ys);
        self.sig.to_tokens(ys);
        self.block.brace.surround(ys, |ys| {
            ys.append_all(self.attrs.inners());
            ys.append_all(&self.block.stmts);
        });
    }
}

pub struct Foreign {
    pub attrs: Vec<attr::Attr>,
    pub unsafe_: Option<Token![unsafe]>,
    pub abi: Abi,
    pub brace: tok::Brace,
    pub items: Vec<Foreign::Item>,
}
impl Parse for Foreign {
    fn parse(s: Stream) -> Res<Self> {
        let mut attrs = s.call(attr::Attr::parse_outers)?;
        let unsafe_: Option<Token![unsafe]> = s.parse()?;
        let abi: Abi = s.parse()?;
        let y;
        let brace = braced!(y in s);
        attr::parse_inners(&y, &mut attrs)?;
        let mut items = Vec::new();
        while !y.is_empty() {
            items.push(y.parse()?);
        }
        Ok(Foreign {
            attrs,
            unsafe_,
            abi,
            brace,
            items,
        })
    }
}
impl ToTokens for Foreign {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.unsafe_.to_tokens(ys);
        self.abi.to_tokens(ys);
        self.brace.surround(ys, |ys| {
            ys.append_all(self.attrs.inners());
            ys.append_all(&self.items);
        });
    }
}

pub struct Impl {
    pub attrs: Vec<attr::Attr>,
    pub default_: Option<Token![default]>,
    pub unsafe_: Option<Token![unsafe]>,
    pub impl_: Token![impl],
    pub gens: gen::Gens,
    pub trait_: Option<(Option<Token![!]>, Path, Token![for])>,
    pub typ: Box<typ::Type>,
    pub brace: tok::Brace,
    pub items: Vec<Impl::Item>,
}
impl Parse for Impl {
    fn parse(s: Stream) -> Res<Self> {
        let verbatim = false;
        parse_impl(s, verbatim).map(Option::unwrap)
    }
}
impl ToTokens for Impl {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.default_.to_tokens(ys);
        self.unsafe_.to_tokens(ys);
        self.impl_.to_tokens(ys);
        self.gens.to_tokens(ys);
        if let Some((polarity, path, for_)) = &self.trait_ {
            polarity.to_tokens(ys);
            path.to_tokens(ys);
            for_.to_tokens(ys);
        }
        self.typ.to_tokens(ys);
        self.gens.where_.to_tokens(ys);
        self.brace.surround(ys, |ys| {
            ys.append_all(self.attrs.inners());
            ys.append_all(&self.items);
        });
    }
}

pub struct Mac {
    pub attrs: Vec<attr::Attr>,
    pub ident: Option<Ident>,
    pub mac: mac::Mac,
    pub semi: Option<Token![;]>,
}
impl Parse for Mac {
    fn parse(s: Stream) -> Res<Self> {
        let attrs = s.call(attr::Attr::parse_outers)?;
        let path = s.call(Path::parse_mod_style)?;
        let bang: Token![!] = s.parse()?;
        let ident: Option<Ident> = if s.peek(Token![try]) {
            s.call(Ident::parse_any).map(Some)
        } else {
            s.parse()
        }?;
        let (delim, toks) = s.call(mac::parse_delim)?;
        let semi: Option<Token![;]> = if !delim.is_brace() { Some(s.parse()?) } else { None };
        Ok(Mac {
            attrs,
            ident,
            mac: mac::Mac {
                path,
                bang,
                delim,
                toks,
            },
            semi,
        })
    }
}
impl ToTokens for Mac {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.mac.path.to_tokens(ys);
        self.mac.bang.to_tokens(ys);
        self.ident.to_tokens(ys);
        match &self.mac.delim {
            tok::Delim::Paren(x) => {
                x.surround(ys, |ys| self.mac.toks.to_tokens(ys));
            },
            tok::Delim::Brace(x) => {
                x.surround(ys, |ys| self.mac.toks.to_tokens(ys));
            },
            tok::Delim::Bracket(x) => {
                x.surround(ys, |ys| self.mac.toks.to_tokens(ys));
            },
        }
        self.semi.to_tokens(ys);
    }
}

pub struct Mod {
    pub attrs: Vec<attr::Attr>,
    pub vis: data::Visibility,
    pub unsafe_: Option<Token![unsafe]>,
    pub mod_: Token![mod],
    pub ident: Ident,
    pub items: Option<(tok::Brace, Vec<Item>)>,
    pub semi: Option<Token![;]>,
}
impl Parse for Mod {
    fn parse(s: Stream) -> Res<Self> {
        let mut attrs = s.call(attr::Attr::parse_outers)?;
        let vis: data::Visibility = s.parse()?;
        let unsafe_: Option<Token![unsafe]> = s.parse()?;
        let mod_: Token![mod] = s.parse()?;
        let ident: Ident = if s.peek(Token![try]) {
            s.call(Ident::parse_any)
        } else {
            s.parse()
        }?;
        let look = s.look1();
        if look.peek(Token![;]) {
            Ok(Mod {
                attrs,
                vis,
                unsafe_,
                mod_,
                ident,
                items: None,
                semi: Some(s.parse()?),
            })
        } else if look.peek(tok::Brace) {
            let y;
            let brace = braced!(y in s);
            attr::parse_inners(&y, &mut attrs)?;
            let mut items = Vec::new();
            while !y.is_empty() {
                items.push(y.parse()?);
            }
            Ok(Mod {
                attrs,
                vis,
                unsafe_,
                mod_,
                ident,
                items: Some((brace, items)),
                semi: None,
            })
        } else {
            Err(look.error())
        }
    }
}
impl ToTokens for Mod {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.vis.to_tokens(ys);
        self.unsafe_.to_tokens(ys);
        self.mod_.to_tokens(ys);
        self.ident.to_tokens(ys);
        if let Some((brace, xs)) = &self.items {
            brace.surround(ys, |ys| {
                ys.append_all(self.attrs.inners());
                ys.append_all(xs);
            });
        } else {
            TokensOrDefault(&self.semi).to_tokens(ys);
        }
    }
}

pub struct Static {
    pub attrs: Vec<attr::Attr>,
    pub vis: data::Visibility,
    pub static_: Token![static],
    pub mut_: StaticMut,
    pub ident: Ident,
    pub colon: Token![:],
    pub typ: Box<typ::Type>,
    pub eq: Token![=],
    pub expr: Box<expr::Expr>,
    pub semi: Token![;],
}
impl Parse for Static {
    fn parse(s: Stream) -> Res<Self> {
        Ok(Static {
            attrs: s.call(attr::Attr::parse_outers)?,
            vis: s.parse()?,
            static_: s.parse()?,
            mut_: s.parse()?,
            ident: s.parse()?,
            colon: s.parse()?,
            typ: s.parse()?,
            eq: s.parse()?,
            expr: s.parse()?,
            semi: s.parse()?,
        })
    }
}
impl ToTokens for Static {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.vis.to_tokens(ys);
        self.static_.to_tokens(ys);
        self.mut_.to_tokens(ys);
        self.ident.to_tokens(ys);
        self.colon.to_tokens(ys);
        self.typ.to_tokens(ys);
        self.eq.to_tokens(ys);
        self.expr.to_tokens(ys);
        self.semi.to_tokens(ys);
    }
}

pub struct Struct {
    pub attrs: Vec<attr::Attr>,
    pub vis: data::Visibility,
    pub struct_: Token![struct],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub fields: data::Fields,
    pub semi: Option<Token![;]>,
}
impl From<Struct> for DeriveInput {
    fn from(x: Struct) -> DeriveInput {
        DeriveInput {
            attrs: x.attrs,
            vis: x.vis,
            ident: x.ident,
            gens: x.gens,
            data: data::Data::Struct(data::Struct {
                struct_: x.struct_,
                fields: x.fields,
                semi: x.semi,
            }),
        }
    }
}
impl Parse for Struct {
    fn parse(s: Stream) -> Res<Self> {
        let attrs = s.call(attr::Attr::parse_outers)?;
        let vis = s.parse::<data::Visibility>()?;
        let struct_ = s.parse::<Token![struct]>()?;
        let ident = s.parse::<Ident>()?;
        let gens = s.parse::<gen::Gens>()?;
        let (where_, fields, semi) = data::parse_struct(s)?;
        Ok(Struct {
            attrs,
            vis,
            struct_,
            ident,
            gens: gen::Gens { where_, ..gens },
            fields,
            semi,
        })
    }
}
impl ToTokens for Struct {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.vis.to_tokens(ys);
        self.struct_.to_tokens(ys);
        self.ident.to_tokens(ys);
        self.gens.to_tokens(ys);
        match &self.fields {
            data::Fields::Named(x) => {
                self.gens.where_.to_tokens(ys);
                x.to_tokens(ys);
            },
            data::Fields::Unnamed(x) => {
                x.to_tokens(ys);
                self.gens.where_.to_tokens(ys);
                TokensOrDefault(&self.semi).to_tokens(ys);
            },
            data::Fields::Unit => {
                self.gens.where_.to_tokens(ys);
                TokensOrDefault(&self.semi).to_tokens(ys);
            },
        }
    }
}

pub struct Trait {
    pub attrs: Vec<attr::Attr>,
    pub vis: data::Visibility,
    pub unsafe_: Option<Token![unsafe]>,
    pub auto_: Option<Token![auto]>,
    pub restriction: Option<Impl::Restriction>,
    pub trait_: Token![trait],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub colon: Option<Token![:]>,
    pub supers: Puncted<gen::bound::Type, Token![+]>,
    pub brace: tok::Brace,
    pub items: Vec<Trait::Item>,
}
impl Parse for Trait {
    fn parse(s: Stream) -> Res<Self> {
        let attrs = s.call(attr::Attr::parse_outers)?;
        let vis: data::Visibility = s.parse()?;
        let unsafe_: Option<Token![unsafe]> = s.parse()?;
        let auto_: Option<Token![auto]> = s.parse()?;
        let trait_: Token![trait] = s.parse()?;
        let ident: Ident = s.parse()?;
        let gens: gen::Gens = s.parse()?;
        parse_rest_of_trait(s, attrs, vis, unsafe_, auto_, trait_, ident, gens)
    }
}
impl ToTokens for Trait {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.vis.to_tokens(ys);
        self.unsafe_.to_tokens(ys);
        self.auto_.to_tokens(ys);
        self.trait_.to_tokens(ys);
        self.ident.to_tokens(ys);
        self.gens.to_tokens(ys);
        if !self.supers.is_empty() {
            TokensOrDefault(&self.colon).to_tokens(ys);
            self.supers.to_tokens(ys);
        }
        self.gens.where_.to_tokens(ys);
        self.brace.surround(ys, |ys| {
            ys.append_all(self.attrs.inners());
            ys.append_all(&self.items);
        });
    }
}

pub struct TraitAlias {
    pub attrs: Vec<attr::Attr>,
    pub vis: data::Visibility,
    pub trait_: Token![trait],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub eq: Token![=],
    pub bounds: Puncted<gen::bound::Type, Token![+]>,
    pub semi: Token![;],
}
impl Parse for TraitAlias {
    fn parse(s: Stream) -> Res<Self> {
        let (attrs, vis, trait_, ident, gens) = parse_start_of_trait_alias(s)?;
        parse_rest_of_trait_alias(s, attrs, vis, trait_, ident, gens)
    }
}
impl ToTokens for TraitAlias {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.vis.to_tokens(ys);
        self.trait_.to_tokens(ys);
        self.ident.to_tokens(ys);
        self.gens.to_tokens(ys);
        self.eq.to_tokens(ys);
        self.bounds.to_tokens(ys);
        self.gens.where_.to_tokens(ys);
        self.semi.to_tokens(ys);
    }
}

pub struct Type {
    pub attrs: Vec<attr::Attr>,
    pub vis: data::Visibility,
    pub type_: Token![type],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub eq: Token![=],
    pub typ: Box<typ::Type>,
    pub semi: Token![;],
}
impl Parse for Type {
    fn parse(s: Stream) -> Res<Self> {
        Ok(Type {
            attrs: s.call(attr::Attr::parse_outers)?,
            vis: s.parse()?,
            type_: s.parse()?,
            ident: s.parse()?,
            gens: {
                let mut y: gen::Gens = s.parse()?;
                y.where_ = s.parse()?;
                y
            },
            eq: s.parse()?,
            typ: s.parse()?,
            semi: s.parse()?,
        })
    }
}
impl ToTokens for Type {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.vis.to_tokens(ys);
        self.type_.to_tokens(ys);
        self.ident.to_tokens(ys);
        self.gens.to_tokens(ys);
        self.gens.where_.to_tokens(ys);
        self.eq.to_tokens(ys);
        self.typ.to_tokens(ys);
        self.semi.to_tokens(ys);
    }
}

pub struct Union {
    pub attrs: Vec<attr::Attr>,
    pub vis: data::Visibility,
    pub union_: Token![union],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub fields: data::Named,
}
impl From<Union> for DeriveInput {
    fn from(x: Union) -> DeriveInput {
        DeriveInput {
            attrs: x.attrs,
            vis: x.vis,
            ident: x.ident,
            gens: x.gens,
            data: data::Data::Union(data::Union {
                union_: x.union_,
                fields: x.fields,
            }),
        }
    }
}
impl Parse for Union {
    fn parse(s: Stream) -> Res<Self> {
        let attrs = s.call(attr::Attr::parse_outers)?;
        let vis = s.parse::<data::Visibility>()?;
        let union_ = s.parse::<Token![union]>()?;
        let ident = s.parse::<Ident>()?;
        let gens = s.parse::<gen::Gens>()?;
        let (where_, fields) = data::parse_union(s)?;
        Ok(Union {
            attrs,
            vis,
            union_,
            ident,
            gens: gen::Gens { where_, ..gens },
            fields,
        })
    }
}
impl ToTokens for Union {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.vis.to_tokens(ys);
        self.union_.to_tokens(ys);
        self.ident.to_tokens(ys);
        self.gens.to_tokens(ys);
        self.gens.where_.to_tokens(ys);
        self.fields.to_tokens(ys);
    }
}

pub struct Use {
    pub attrs: Vec<attr::Attr>,
    pub vis: data::Visibility,
    pub use_: Token![use],
    pub colon: Option<Token![::]>,
    pub tree: Use::Tree,
    pub semi: Token![;],
}
impl Parse for Use {
    fn parse(s: Stream) -> Res<Self> {
        let root = false;
        parse_item_use(s, root).map(Option::unwrap)
    }
}
impl ToTokens for Use {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.vis.to_tokens(ys);
        self.use_.to_tokens(ys);
        self.colon.to_tokens(ys);
        self.tree.to_tokens(ys);
        self.semi.to_tokens(ys);
    }
}
fn parse_item_use(s: Stream, root: bool) -> Res<Option<Use>> {
    let attrs = s.call(attr::Attr::parse_outers)?;
    let vis: data::Visibility = s.parse()?;
    let use_: Token![use] = s.parse()?;
    let colon: Option<Token![::]> = s.parse()?;
    let tree = Use::parse_tree(s, root && colon.is_none())?;
    let semi: Token![;] = s.parse()?;
    let tree = match tree {
        Some(x) => x,
        None => return Ok(None),
    };
    Ok(Some(Use {
        attrs,
        vis,
        use_,
        colon,
        tree,
        semi,
    }))
}

pub struct Receiver {
    pub attrs: Vec<attr::Attr>,
    pub ref_: Option<(Token![&], Option<Life>)>,
    pub mut_: Option<Token![mut]>,
    pub self_: Token![self],
    pub colon: Option<Token![:]>,
    pub typ: Box<typ::Type>,
}
impl Receiver {
    pub fn life(&self) -> Option<&Life> {
        self.ref_.as_ref()?.1.as_ref()
    }
}
impl Parse for Receiver {
    fn parse(s: Stream) -> Res<Self> {
        let ref_ = if s.peek(Token![&]) {
            let amper: Token![&] = s.parse()?;
            let life: Option<Life> = s.parse()?;
            Some((amper, life))
        } else {
            None
        };
        let mut_: Option<Token![mut]> = s.parse()?;
        let self_: Token![self] = s.parse()?;
        let colon: Option<Token![:]> = if ref_.is_some() { None } else { s.parse()? };
        let typ: typ::Type = if colon.is_some() {
            s.parse()?
        } else {
            let mut y = typ::Type::Path(typ::Path {
                qself: None,
                path: Path::from(Ident::new("Self", self_.span)),
            });
            if let Some((amper, life)) = ref_.as_ref() {
                y = typ::Type::Ref(typ::Ref {
                    and: Token![&](amper.span),
                    life: life.clone(),
                    mut_: mut_.as_ref().map(|x| Token![mut](x.span)),
                    elem: Box::new(y),
                });
            }
            y
        };
        Ok(Receiver {
            attrs: Vec::new(),
            ref_,
            mut_,
            self_,
            colon,
            typ: Box::new(typ),
        })
    }
}
impl ToTokens for Receiver {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        if let Some((amper, life)) = &self.ref_ {
            amper.to_tokens(ys);
            life.to_tokens(ys);
        }
        self.mut_.to_tokens(ys);
        self.self_.to_tokens(ys);
        if let Some(x) = &self.colon {
            x.to_tokens(ys);
            self.typ.to_tokens(ys);
        } else {
            let consistent = match (&self.ref_, &self.mut_, &*self.typ) {
                (Some(_), mut_, typ::Type::Ref(x)) => {
                    mut_.is_some() == x.mut_.is_some()
                        && match &*x.elem {
                            typ::Type::Path(x) => x.qself.is_none() && x.path.is_ident("Self"),
                            _ => false,
                        }
                },
                (None, _, typ::Type::Path(x)) => x.qself.is_none() && x.path.is_ident("Self"),
                _ => false,
            };
            if !consistent {
                <Token![:]>::default().to_tokens(ys);
                self.typ.to_tokens(ys);
            }
        }
    }
}

ast_enum_of_structs! {
    pub enum FnArg {
        Receiver(Receiver),
        Type(pat::Type),
    }
}
impl Parse for FnArg {
    fn parse(s: Stream) -> Res<Self> {
        let variadic = false;
        let attrs = s.call(attr::Attr::parse_outers)?;
        use FnArgOrVariadic::*;
        match parse_fn_arg_or_variadic(s, attrs, variadic)? {
            FnArg(x) => Ok(x),
            Variadic(_) => unreachable!(),
        }
    }
}

enum FnArgOrVariadic {
    FnArg(FnArg),
    Variadic(Variadic),
}

pub struct Sig {
    pub const_: Option<Token![const]>,
    pub async_: Option<Token![async]>,
    pub unsafe_: Option<Token![unsafe]>,
    pub abi: Option<Abi>,
    pub fn_: Token![fn],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub paren: tok::Paren,
    pub args: Puncted<FnArg, Token![,]>,
    pub vari: Option<Variadic>,
    pub ret: typ::Ret,
}
impl Sig {
    pub fn receiver(&self) -> Option<&Receiver> {
        let y = self.args.first()?;
        match y {
            FnArg::Receiver(&x) => Some(x),
            FnArg::Type(_) => None,
        }
    }
}
impl Parse for Sig {
    fn parse(s: Stream) -> Res<Self> {
        let const_: Option<Token![const]> = s.parse()?;
        let async_: Option<Token![async]> = s.parse()?;
        let unsafe_: Option<Token![unsafe]> = s.parse()?;
        let abi: Option<Abi> = s.parse()?;
        let fn_: Token![fn] = s.parse()?;
        let ident: Ident = s.parse()?;
        let mut gens: gen::Gens = s.parse()?;
        let y;
        let paren = parenthesized!(y in s);
        let (args, vari) = parse_fn_args(&y)?;
        let ret: typ::Ret = s.parse()?;
        gens.where_ = s.parse()?;
        Ok(Sig {
            const_,
            async_,
            unsafe_,
            abi,
            fn_,
            ident,
            gens,
            paren,
            args,
            vari,
            ret,
        })
    }
}
impl ToTokens for Sig {
    fn to_tokens(&self, ys: &mut Stream) {
        self.const_.to_tokens(ys);
        self.async_.to_tokens(ys);
        self.unsafe_.to_tokens(ys);
        self.abi.to_tokens(ys);
        self.fn_.to_tokens(ys);
        self.ident.to_tokens(ys);
        self.gens.to_tokens(ys);
        self.paren.surround(ys, |ys| {
            self.args.to_tokens(ys);
            if let Some(x) = &self.vari {
                if !self.args.empty_or_trailing() {
                    <Token![,]>::default().to_tokens(ys);
                }
                x.to_tokens(ys);
            }
        });
        self.ret.to_tokens(ys);
        self.gens.where_.to_tokens(ys);
    }
}

pub struct Variadic {
    pub attrs: Vec<attr::Attr>,
    pub pat: Option<(Box<pat::Pat>, Token![:])>,
    pub dots: Token![...],
    pub comma: Option<Token![,]>,
}
impl ToTokens for Variadic {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        if let Some((pat, colon)) = &self.pat {
            pat.to_tokens(ys);
            colon.to_tokens(ys);
        }
        self.dots.to_tokens(ys);
        self.comma.to_tokens(ys);
    }
}

pub enum StaticMut {
    Mut(Token![mut]),
    None,
}
impl Parse for StaticMut {
    fn parse(s: Stream) -> Res<Self> {
        let mut_: Option<Token![mut]> = s.parse()?;
        Ok(mut_.map_or(StaticMut::None, StaticMut::Mut))
    }
}
impl ToTokens for StaticMut {
    fn to_tokens(&self, ys: &mut Stream) {
        match self {
            StaticMut::None => {},
            StaticMut::Mut(x) => x.to_tokens(ys),
        }
    }
}

pub mod Foreign {
    use super::*;
    ast_enum_of_structs! {
        pub enum Item {
            Fn(Fn),
            Mac(Mac),
            Static(Static),
            Stream(Stream),
            Type(Type),
        }
    }
    impl Parse for Item {
        fn parse(s: Stream) -> Res<Self> {
            let beg = s.fork();
            let mut attrs = s.call(attr::Attr::parse_outers)?;
            let ahead = s.fork();
            let vis: data::Visibility = ahead.parse()?;
            let look = ahead.look1();
            let mut y = if look.peek(Token![fn]) || peek_signature(&ahead) {
                let vis: data::Visibility = s.parse()?;
                let sig: Sig = s.parse()?;
                if s.peek(tok::Brace) {
                    let y;
                    braced!(y in s);
                    y.call(attr::Attr::parse_inners)?;
                    y.call(stmt::Block::parse_within)?;
                    Ok(Item::Stream(parse::parse_verbatim(&beg, s)))
                } else {
                    Ok(Item::Fn(Fn {
                        attrs: Vec::new(),
                        vis,
                        sig,
                        semi: s.parse()?,
                    }))
                }
            } else if look.peek(Token![static]) {
                let vis = s.parse()?;
                let static_ = s.parse()?;
                let mut_ = s.parse()?;
                let ident = s.parse()?;
                let colon = s.parse()?;
                let typ = s.parse()?;
                if s.peek(Token![=]) {
                    s.parse::<Token![=]>()?;
                    s.parse::<expr::Expr>()?;
                    s.parse::<Token![;]>()?;
                    Ok(Item::Stream(parse::parse_verbatim(&beg, s)))
                } else {
                    Ok(Item::Static(Static {
                        attrs: Vec::new(),
                        vis,
                        static_,
                        mut_,
                        ident,
                        colon,
                        typ,
                        semi: s.parse()?,
                    }))
                }
            } else if look.peek(Token![type]) {
                parse_foreign_item_type(beg, s)
            } else if vis.is_inherited()
                && (look.peek(Ident)
                    || look.peek(Token![self])
                    || look.peek(Token![super])
                    || look.peek(Token![crate])
                    || look.peek(Token![::]))
            {
                s.parse().map(Item::Mac)
            } else {
                Err(look.error())
            }?;
            let ys = match &mut y {
                Item::Fn(x) => &mut x.attrs,
                Item::Static(x) => &mut x.attrs,
                Item::Type(x) => &mut x.attrs,
                Item::Mac(x) => &mut x.attrs,
                Item::Stream(_) => return Ok(y),
            };
            attrs.append(ys);
            *ys = attrs;
            Ok(y)
        }
    }

    pub struct Fn {
        pub attrs: Vec<attr::Attr>,
        pub vis: data::Visibility,
        pub sig: Sig,
        pub semi: Token![;],
    }
    impl Parse for Fn {
        fn parse(s: Stream) -> Res<Self> {
            let attrs = s.call(attr::Attr::parse_outers)?;
            let vis: data::Visibility = s.parse()?;
            let sig: Sig = s.parse()?;
            let semi: Token![;] = s.parse()?;
            Ok(Fn { attrs, vis, sig, semi })
        }
    }
    impl ToTokens for Fn {
        fn to_tokens(&self, ys: &mut Stream) {
            ys.append_all(self.attrs.outers());
            self.vis.to_tokens(ys);
            self.sig.to_tokens(ys);
            self.semi.to_tokens(ys);
        }
    }

    pub struct Static {
        pub attrs: Vec<attr::Attr>,
        pub vis: data::Visibility,
        pub static_: Token![static],
        pub mut_: StaticMut,
        pub ident: Ident,
        pub colon: Token![:],
        pub typ: Box<typ::Type>,
        pub semi: Token![;],
    }
    impl Parse for Static {
        fn parse(s: Stream) -> Res<Self> {
            Ok(Static {
                attrs: s.call(attr::Attr::parse_outers)?,
                vis: s.parse()?,
                static_: s.parse()?,
                mut_: s.parse()?,
                ident: s.parse()?,
                colon: s.parse()?,
                typ: s.parse()?,
                semi: s.parse()?,
            })
        }
    }
    impl ToTokens for Static {
        fn to_tokens(&self, ys: &mut Stream) {
            ys.append_all(self.attrs.outers());
            self.vis.to_tokens(ys);
            self.static_.to_tokens(ys);
            self.mut_.to_tokens(ys);
            self.ident.to_tokens(ys);
            self.colon.to_tokens(ys);
            self.typ.to_tokens(ys);
            self.semi.to_tokens(ys);
        }
    }

    pub struct Type {
        pub attrs: Vec<attr::Attr>,
        pub vis: data::Visibility,
        pub type_: Token![type],
        pub ident: Ident,
        pub gens: gen::Gens,
        pub semi: Token![;],
    }
    impl Parse for Type {
        fn parse(s: Stream) -> Res<Self> {
            Ok(Type {
                attrs: s.call(attr::Attr::parse_outers)?,
                vis: s.parse()?,
                type_: s.parse()?,
                ident: s.parse()?,
                gens: {
                    let mut y: gen::Gens = s.parse()?;
                    y.where_ = s.parse()?;
                    y
                },
                semi: s.parse()?,
            })
        }
    }
    impl ToTokens for Type {
        fn to_tokens(&self, ys: &mut Stream) {
            ys.append_all(self.attrs.outers());
            self.vis.to_tokens(ys);
            self.type_.to_tokens(ys);
            self.ident.to_tokens(ys);
            self.gens.to_tokens(ys);
            self.gens.where_.to_tokens(ys);
            self.semi.to_tokens(ys);
        }
    }

    pub struct Mac {
        pub attrs: Vec<attr::Attr>,
        pub mac: mac::Mac,
        pub semi: Option<Token![;]>,
    }
    impl Parse for Mac {
        fn parse(s: Stream) -> Res<Self> {
            let attrs = s.call(attr::Attr::parse_outers)?;
            let mac: mac::Mac = s.parse()?;
            let semi: Option<Token![;]> = if mac.delim.is_brace() { None } else { Some(s.parse()?) };
            Ok(Mac { attrs, mac, semi })
        }
    }
    impl ToTokens for Mac {
        fn to_tokens(&self, ys: &mut Stream) {
            ys.append_all(self.attrs.outers());
            self.mac.to_tokens(ys);
            self.semi.to_tokens(ys);
        }
    }
}
pub mod Impl {
    use super::*;
    ast_enum_of_structs! {
        pub enum Item {
            Const(Const),
            Fn(Fn),
            Mac(Mac),
            Stream(Stream),
            Type(Type),
        }
    }
    impl Parse for Item {
        fn parse(s: Stream) -> Res<Self> {
            let beg = s.fork();
            let mut attrs = s.call(attr::Attr::parse_outers)?;
            let ahead = s.fork();
            let vis: data::Visibility = ahead.parse()?;
            let mut look = ahead.look1();
            let default_ = if look.peek(Token![default]) && !ahead.peek2(Token![!]) {
                let y: Token![default] = ahead.parse()?;
                look = ahead.look1();
                Some(y)
            } else {
                None
            };
            let mut y = if look.peek(Token![fn]) || peek_signature(&ahead) {
                let omitted = true;
                if let Some(x) = parse_impl_item_fn(s, omitted)? {
                    Ok(Item::Fn(x))
                } else {
                    Ok(Item::Stream(parse::parse_verbatim(&beg, s)))
                }
            } else if look.peek(Token![const]) {
                s.advance_to(&ahead);
                let const_: Token![const] = s.parse()?;
                let look = s.look1();
                let ident = if look.peek(Ident) || look.peek(Token![_]) {
                    s.call(Ident::parse_any)?
                } else {
                    return Err(look.error());
                };
                let colon: Token![:] = s.parse()?;
                let typ: typ::Type = s.parse()?;
                if let Some(eq) = s.parse()? {
                    return Ok(Item::Const(Const {
                        attrs,
                        vis,
                        default_,
                        const_,
                        ident,
                        gens: gen::Gens::default(),
                        colon,
                        typ,
                        eq,
                        expr: s.parse()?,
                        semi: s.parse()?,
                    }));
                } else {
                    s.parse::<Token![;]>()?;
                    return Ok(Item::Stream(parse::parse_verbatim(&beg, s)));
                }
            } else if look.peek(Token![type]) {
                parse_impl_item_type(beg, s)
            } else if vis.is_inherited()
                && default_.is_none()
                && (look.peek(Ident)
                    || look.peek(Token![self])
                    || look.peek(Token![super])
                    || look.peek(Token![crate])
                    || look.peek(Token![::]))
            {
                s.parse().map(Item::Mac)
            } else {
                Err(look.error())
            }?;
            let ys = match &mut y {
                Item::Const(x) => &mut x.attrs,
                Item::Fn(x) => &mut x.attrs,
                Item::Type(x) => &mut x.attrs,
                Item::Mac(x) => &mut x.attrs,
                Item::Stream(_) => return Ok(y),
            };
            attrs.append(ys);
            *ys = attrs;
            Ok(y)
        }
    }

    pub struct Const {
        pub attrs: Vec<attr::Attr>,
        pub vis: data::Visibility,
        pub default_: Option<Token![default]>,
        pub const_: Token![const],
        pub ident: Ident,
        pub gens: gen::Gens,
        pub colon: Token![:],
        pub typ: typ::Type,
        pub eq: Token![=],
        pub expr: expr::Expr,
        pub semi: Token![;],
    }
    impl Parse for Const {
        fn parse(s: Stream) -> Res<Self> {
            Ok(Const {
                attrs: s.call(attr::Attr::parse_outers)?,
                vis: s.parse()?,
                default_: s.parse()?,
                const_: s.parse()?,
                ident: {
                    let look = s.look1();
                    if look.peek(Ident) || look.peek(Token![_]) {
                        s.call(Ident::parse_any)?
                    } else {
                        return Err(look.error());
                    }
                },
                gens: gen::Gens::default(),
                colon: s.parse()?,
                typ: s.parse()?,
                eq: s.parse()?,
                expr: s.parse()?,
                semi: s.parse()?,
            })
        }
    }
    impl ToTokens for Const {
        fn to_tokens(&self, ys: &mut Stream) {
            ys.append_all(self.attrs.outers());
            self.vis.to_tokens(ys);
            self.default_.to_tokens(ys);
            self.const_.to_tokens(ys);
            self.ident.to_tokens(ys);
            self.colon.to_tokens(ys);
            self.typ.to_tokens(ys);
            self.eq.to_tokens(ys);
            self.expr.to_tokens(ys);
            self.semi.to_tokens(ys);
        }
    }

    pub struct Fn {
        pub attrs: Vec<attr::Attr>,
        pub vis: data::Visibility,
        pub default_: Option<Token![default]>,
        pub sig: Sig,
        pub block: stmt::Block,
    }
    impl Parse for Fn {
        fn parse(s: Stream) -> Res<Self> {
            let omitted = false;
            parse_impl_item_fn(s, omitted).map(Option::unwrap)
        }
    }
    impl ToTokens for Fn {
        fn to_tokens(&self, ys: &mut Stream) {
            ys.append_all(self.attrs.outers());
            self.vis.to_tokens(ys);
            self.default_.to_tokens(ys);
            self.sig.to_tokens(ys);
            self.block.brace.surround(ys, |ys| {
                ys.append_all(self.attrs.inners());
                ys.append_all(&self.block.stmts);
            });
        }
    }

    pub struct Type {
        pub attrs: Vec<attr::Attr>,
        pub vis: data::Visibility,
        pub default_: Option<Token![default]>,
        pub type_: Token![type],
        pub ident: Ident,
        pub gens: gen::Gens,
        pub eq: Token![=],
        pub typ: typ::Type,
        pub semi: Token![;],
    }
    impl Parse for Type {
        fn parse(s: Stream) -> Res<Self> {
            let attrs = s.call(attr::Attr::parse_outers)?;
            let vis: data::Visibility = s.parse()?;
            let default_: Option<Token![default]> = s.parse()?;
            let type_: Token![type] = s.parse()?;
            let ident: Ident = s.parse()?;
            let mut gens: gen::Gens = s.parse()?;
            let eq: Token![=] = s.parse()?;
            let typ: typ::Type = s.parse()?;
            gens.where_ = s.parse()?;
            let semi: Token![;] = s.parse()?;
            Ok(Type {
                attrs,
                vis,
                default_,
                type_,
                ident,
                gens,
                eq,
                typ,
                semi,
            })
        }
    }
    impl ToTokens for Type {
        fn to_tokens(&self, ys: &mut Stream) {
            ys.append_all(self.attrs.outers());
            self.vis.to_tokens(ys);
            self.default_.to_tokens(ys);
            self.type_.to_tokens(ys);
            self.ident.to_tokens(ys);
            self.gens.to_tokens(ys);
            self.eq.to_tokens(ys);
            self.typ.to_tokens(ys);
            self.gens.where_.to_tokens(ys);
            self.semi.to_tokens(ys);
        }
    }

    pub struct Mac {
        pub attrs: Vec<attr::Attr>,
        pub mac: mac::Mac,
        pub semi: Option<Token![;]>,
    }
    impl Parse for Mac {
        fn parse(s: Stream) -> Res<Self> {
            let attrs = s.call(attr::Attr::parse_outers)?;
            let mac: mac::Mac = s.parse()?;
            let semi: Option<Token![;]> = if mac.delim.is_brace() { None } else { Some(s.parse()?) };
            Ok(Mac { attrs, mac, semi })
        }
    }
    impl ToTokens for Mac {
        fn to_tokens(&self, ys: &mut Stream) {
            ys.append_all(self.attrs.outers());
            self.mac.to_tokens(ys);
            self.semi.to_tokens(ys);
        }
    }

    pub enum Restriction {}
}
pub mod Trait {
    use super::*;
    ast_enum_of_structs! {
        pub enum Item {
            Const(Const),
            Fn(Fn),
            Type(Type),
            Mac(Mac),
            Stream(Stream),
        }
    }
    impl Parse for Item {
        fn parse(s: Stream) -> Res<Self> {
            let beg = s.fork();
            let mut attrs = s.call(attr::Attr::parse_outers)?;
            let vis: data::Visibility = s.parse()?;
            let default_: Option<Token![default]> = s.parse()?;
            let ahead = s.fork();
            let look = ahead.look1();
            let mut y = if look.peek(Token![fn]) || peek_signature(&ahead) {
                s.parse().map(Item::Fn)
            } else if look.peek(Token![const]) {
                ahead.parse::<Token![const]>()?;
                let look = ahead.look1();
                if look.peek(Ident) || look.peek(Token![_]) {
                    s.parse().map(Item::Const)
                } else if look.peek(Token![async])
                    || look.peek(Token![unsafe])
                    || look.peek(Token![extern])
                    || look.peek(Token![fn])
                {
                    s.parse().map(Item::Fn)
                } else {
                    Err(look.error())
                }
            } else if look.peek(Token![type]) {
                parse_trait_item_type(beg.fork(), s)
            } else if vis.is_inherited()
                && default_.is_none()
                && (look.peek(Ident)
                    || look.peek(Token![self])
                    || look.peek(Token![super])
                    || look.peek(Token![crate])
                    || look.peek(Token![::]))
            {
                s.parse().map(Item::Mac)
            } else {
                Err(look.error())
            }?;
            match (vis, default_) {
                (data::Visibility::Inherited, None) => {},
                _ => return Ok(Item::Stream(parse::parse_verbatim(&beg, s))),
            }
            let ys = match &mut y {
                Item::Const(item) => &mut item.attrs,
                Item::Fn(item) => &mut item.attrs,
                Item::Type(item) => &mut item.attrs,
                Item::Mac(item) => &mut item.attrs,
                Item::Stream(_) => unreachable!(),
            };
            attrs.append(ys);
            *ys = attrs;
            Ok(y)
        }
    }

    pub struct Const {
        pub attrs: Vec<attr::Attr>,
        pub const_: Token![const],
        pub ident: Ident,
        pub gens: gen::Gens,
        pub colon: Token![:],
        pub typ: typ::Type,
        pub default: Option<(Token![=], expr::Expr)>,
        pub semi: Token![;],
    }
    impl Parse for Const {
        fn parse(s: Stream) -> Res<Self> {
            Ok(Const {
                attrs: s.call(attr::Attr::parse_outers)?,
                const_: s.parse()?,
                ident: {
                    let look = s.look1();
                    if look.peek(Ident) || look.peek(Token![_]) {
                        s.call(Ident::parse_any)?
                    } else {
                        return Err(look.error());
                    }
                },
                gens: gen::Gens::default(),
                colon: s.parse()?,
                typ: s.parse()?,
                default: {
                    if s.peek(Token![=]) {
                        let eq: Token![=] = s.parse()?;
                        let default: expr::Expr = s.parse()?;
                        Some((eq, default))
                    } else {
                        None
                    }
                },
                semi: s.parse()?,
            })
        }
    }
    impl ToTokens for Const {
        fn to_tokens(&self, ys: &mut Stream) {
            ys.append_all(self.attrs.outers());
            self.const_.to_tokens(ys);
            self.ident.to_tokens(ys);
            self.colon.to_tokens(ys);
            self.typ.to_tokens(ys);
            if let Some((eq, x)) = &self.default {
                eq.to_tokens(ys);
                x.to_tokens(ys);
            }
            self.semi.to_tokens(ys);
        }
    }

    pub struct Fn {
        pub attrs: Vec<attr::Attr>,
        pub sig: Sig,
        pub default: Option<stmt::Block>,
        pub semi: Option<Token![;]>,
    }
    impl Parse for Fn {
        fn parse(s: Stream) -> Res<Self> {
            let mut attrs = s.call(attr::Attr::parse_outers)?;
            let sig: Sig = s.parse()?;
            let look = s.look1();
            let (brace, stmts, semi) = if look.peek(tok::Brace) {
                let y;
                let brace = braced!(y in s);
                attr::parse_inners(&y, &mut attrs)?;
                let stmts = y.call(stmt::Block::parse_within)?;
                (Some(brace), stmts, None)
            } else if look.peek(Token![;]) {
                let semi: Token![;] = s.parse()?;
                (None, Vec::new(), Some(semi))
            } else {
                return Err(look.error());
            };
            Ok(Fn {
                attrs,
                sig,
                default: brace.map(|brace| Block { brace, stmts }),
                semi,
            })
        }
    }
    impl ToTokens for Fn {
        fn to_tokens(&self, ys: &mut Stream) {
            ys.append_all(self.attrs.outers());
            self.sig.to_tokens(ys);
            match &self.default {
                Some(block) => {
                    block.brace.surround(ys, |ys| {
                        ys.append_all(self.attrs.inners());
                        ys.append_all(&block.stmts);
                    });
                },
                None => {
                    TokensOrDefault(&self.semi).to_tokens(ys);
                },
            }
        }
    }

    pub struct Type {
        pub attrs: Vec<attr::Attr>,
        pub type_: Token![type],
        pub ident: Ident,
        pub gens: gen::Gens,
        pub colon: Option<Token![:]>,
        pub bounds: Puncted<gen::bound::Type, Token![+]>,
        pub default: Option<(Token![=], typ::Type)>,
        pub semi: Token![;],
    }
    impl Parse for Type {
        fn parse(s: Stream) -> Res<Self> {
            let attrs = s.call(attr::Attr::parse_outers)?;
            let type_: Token![type] = s.parse()?;
            let ident: Ident = s.parse()?;
            let mut gens: gen::Gens = s.parse()?;
            let (colon, bounds) = Flexible::parse_optional_bounds(s)?;
            let default = Flexible::parse_optional_definition(s)?;
            gens.where_ = s.parse()?;
            let semi: Token![;] = s.parse()?;
            Ok(Type {
                attrs,
                type_,
                ident,
                gens,
                colon,
                bounds,
                default,
                semi,
            })
        }
    }
    impl ToTokens for Type {
        fn to_tokens(&self, ys: &mut Stream) {
            ys.append_all(self.attrs.outers());
            self.type_.to_tokens(ys);
            self.ident.to_tokens(ys);
            self.gens.to_tokens(ys);
            if !self.bounds.is_empty() {
                TokensOrDefault(&self.colon).to_tokens(ys);
                self.bounds.to_tokens(ys);
            }
            if let Some((eq, x)) = &self.default {
                eq.to_tokens(ys);
                x.to_tokens(ys);
            }
            self.gens.where_.to_tokens(ys);
            self.semi.to_tokens(ys);
        }
    }

    pub struct Mac {
        pub attrs: Vec<attr::Attr>,
        pub mac: mac::Mac,
        pub semi: Option<Token![;]>,
    }
    impl Parse for Mac {
        fn parse(s: Stream) -> Res<Self> {
            let attrs = s.call(attr::Attr::parse_outers)?;
            let mac: mac::Mac = s.parse()?;
            let semi: Option<Token![;]> = if mac.delim.is_brace() { None } else { Some(s.parse()?) };
            Ok(Mac { attrs, mac, semi })
        }
    }
    impl ToTokens for Mac {
        fn to_tokens(&self, ys: &mut Stream) {
            ys.append_all(self.attrs.outers());
            self.mac.to_tokens(ys);
            self.semi.to_tokens(ys);
        }
    }
}
pub mod Use {
    use super::*;
    ast_enum_of_structs! {
        pub enum Tree {
            Glob(Glob),
            Group(Group),
            Name(Name),
            Path(Path),
            Rename(Rename),
        }
    }
    impl Parse for Tree {
        fn parse(s: Stream) -> Res<Tree> {
            let root = false;
            parse_tree(s, root).map(Option::unwrap)
        }
    }
    fn parse_tree(s: Stream, root: bool) -> Res<Option<Tree>> {
        let look = s.look1();
        if look.peek(Ident)
            || look.peek(Token![self])
            || look.peek(Token![super])
            || look.peek(Token![crate])
            || look.peek(Token![try])
        {
            let ident = s.call(Ident::parse_any)?;
            if s.peek(Token![::]) {
                Ok(Some(Tree::Path(Path {
                    ident,
                    colon2: s.parse()?,
                    tree: Box::new(s.parse()?),
                })))
            } else if s.peek(Token![as]) {
                Ok(Some(Tree::Rename(Rename {
                    ident,
                    as_: s.parse()?,
                    rename: {
                        if s.peek(Ident) {
                            s.parse()?
                        } else if s.peek(Token![_]) {
                            Ident::from(s.parse::<Token![_]>()?)
                        } else {
                            return Err(s.error("expected identifier or underscore"));
                        }
                    },
                })))
            } else {
                Ok(Some(Tree::Name(Name { ident })))
            }
        } else if look.peek(Token![*]) {
            Ok(Some(Tree::Glob(Glob { star: s.parse()? })))
        } else if look.peek(tok::Brace) {
            let y;
            let brace = braced!(y in s);
            let mut elems = Puncted::new();
            let mut has_root = false;
            loop {
                if y.is_empty() {
                    break;
                }
                let starts_with_root = root && y.parse::<Option<Token![::]>>()?.is_some();
                has_root |= starts_with_root;
                match parse_tree(&y, root && !starts_with_root)? {
                    Some(x) => elems.push_value(x),
                    None => has_root = true,
                }
                if y.is_empty() {
                    break;
                }
                let comma: Token![,] = y.parse()?;
                elems.push_punct(comma);
            }
            if has_root {
                Ok(None)
            } else {
                Ok(Some(Tree::Group(Group { brace, elems })))
            }
        } else {
            Err(look.error())
        }
    }

    pub struct Path {
        pub ident: Ident,
        pub colon2: Token![::],
        pub tree: Box<Tree>,
    }
    impl ToTokens for Path {
        fn to_tokens(&self, ys: &mut Stream) {
            self.ident.to_tokens(ys);
            self.colon2.to_tokens(ys);
            self.tree.to_tokens(ys);
        }
    }

    pub struct Name {
        pub ident: Ident,
    }
    impl ToTokens for Name {
        fn to_tokens(&self, ys: &mut Stream) {
            self.ident.to_tokens(ys);
        }
    }

    pub struct Rename {
        pub ident: Ident,
        pub as_: Token![as],
        pub rename: Ident,
    }
    impl ToTokens for Rename {
        fn to_tokens(&self, ys: &mut Stream) {
            self.ident.to_tokens(ys);
            self.as_.to_tokens(ys);
            self.rename.to_tokens(ys);
        }
    }

    pub struct Glob {
        pub star: Token![*],
    }
    impl ToTokens for Glob {
        fn to_tokens(&self, ys: &mut Stream) {
            self.star.to_tokens(ys);
        }
    }

    pub struct Group {
        pub brace: tok::Brace,
        pub elems: Puncted<Tree, Token![,]>,
    }
    impl ToTokens for Group {
        fn to_tokens(&self, ys: &mut Stream) {
            self.brace.surround(ys, |ys| {
                self.elems.to_tokens(ys);
            });
        }
    }
}

enum TypeDefault {
    Optional,
    Disallowed,
}
enum WhereLoc {
    BeforeEq,
    AfterEq,
    Both,
}

struct Flexible {
    vis: data::Visibility,
    default_: Option<Token![default]>,
    type_: Token![type],
    ident: Ident,
    gens: gen::Gens,
    colon: Option<Token![:]>,
    bounds: Puncted<gen::bound::Type, Token![+]>,
    typ: Option<(Token![=], typ::Type)>,
    semi: Token![;],
}
impl Flexible {
    fn parse(s: Stream, allow_default: TypeDefault, where_loc: WhereLoc) -> Res<Self> {
        let vis: data::Visibility = s.parse()?;
        let default_: Option<Token![default]> = match allow_default {
            TypeDefault::Optional => s.parse()?,
            TypeDefault::Disallowed => None,
        };
        let type_: Token![type] = s.parse()?;
        let ident: Ident = s.parse()?;
        let mut gens: gen::Gens = s.parse()?;
        let (colon, bounds) = Self::parse_optional_bounds(s)?;
        match where_loc {
            WhereLoc::BeforeEq | WhereLoc::Both => {
                gens.where_ = s.parse()?;
            },
            WhereLoc::AfterEq => {},
        }
        let ty = Self::parse_optional_definition(s)?;
        match where_loc {
            WhereLoc::AfterEq | WhereLoc::Both if gens.where_.is_none() => {
                gens.where_ = s.parse()?;
            },
            _ => {},
        }
        let semi: Token![;] = s.parse()?;
        Ok(Flexible {
            vis,
            default_,
            type_,
            ident,
            gens,
            colon,
            bounds,
            typ: ty,
            semi,
        })
    }
    fn parse_optional_bounds(s: Stream) -> Res<(Option<Token![:]>, Puncted<gen::bound::Type, Token![+]>)> {
        let colon: Option<Token![:]> = s.parse()?;
        let mut ys = Puncted::new();
        if colon.is_some() {
            loop {
                if s.peek(Token![where]) || s.peek(Token![=]) || s.peek(Token![;]) {
                    break;
                }
                ys.push_value(s.parse::<gen::bound::Type>()?);
                if s.peek(Token![where]) || s.peek(Token![=]) || s.peek(Token![;]) {
                    break;
                }
                ys.push_punct(s.parse::<Token![+]>()?);
            }
        }
        Ok((colon, ys))
    }
    fn parse_optional_definition(s: Stream) -> Res<Option<(Token![=], typ::Type)>> {
        let eq: Option<Token![=]> = s.parse()?;
        if let Some(eq) = eq {
            let y: typ::Type = s.parse()?;
            Ok(Some((eq, y)))
        } else {
            Ok(None)
        }
    }
}

pub fn parse_rest_of_item(beg: Buffer, mut attrs: Vec<attr::Attr>, s: Stream) -> Res<Item> {
    let ahead = s.fork();
    let vis: data::Visibility = ahead.parse()?;
    let look = ahead.look1();
    let mut item = if look.peek(Token![fn]) || peek_signature(&ahead) {
        let vis: data::Visibility = s.parse()?;
        let sig: Sig = s.parse()?;
        if s.peek(Token![;]) {
            s.parse::<Token![;]>()?;
            Ok(Item::Stream(parse::parse_verbatim(&beg, s)))
        } else {
            parse_rest_of_fn(s, Vec::new(), vis, sig).map(Item::Fn)
        }
    } else if look.peek(Token![extern]) {
        ahead.parse::<Token![extern]>()?;
        let look = ahead.look1();
        if look.peek(Token![crate]) {
            s.parse().map(Item::Extern)
        } else if look.peek(tok::Brace) {
            s.parse().map(Item::Foreign)
        } else if look.peek(lit::Str) {
            ahead.parse::<lit::Str>()?;
            let look = ahead.look1();
            if look.peek(tok::Brace) {
                s.parse().map(Item::Foreign)
            } else {
                Err(look.error())
            }
        } else {
            Err(look.error())
        }
    } else if look.peek(Token![use]) {
        let allow_crate_root_in_path = true;
        match parse_item_use(s, allow_crate_root_in_path)? {
            Some(item_use) => Ok(Item::Use(item_use)),
            None => Ok(Item::Stream(parse::parse_verbatim(&beg, s))),
        }
    } else if look.peek(Token![static]) {
        let vis = s.parse()?;
        let static_ = s.parse()?;
        let mut_ = s.parse()?;
        let ident = s.parse()?;
        if s.peek(Token![=]) {
            s.parse::<Token![=]>()?;
            s.parse::<expr::Expr>()?;
            s.parse::<Token![;]>()?;
            Ok(Item::Stream(parse::parse_verbatim(&beg, s)))
        } else {
            let colon = s.parse()?;
            let ty = s.parse()?;
            if s.peek(Token![;]) {
                s.parse::<Token![;]>()?;
                Ok(Item::Stream(parse::parse_verbatim(&beg, s)))
            } else {
                Ok(Item::Static(Static {
                    attrs: Vec::new(),
                    vis,
                    static_,
                    mut_,
                    ident,
                    colon,
                    typ: ty,
                    eq: s.parse()?,
                    expr: s.parse()?,
                    semi: s.parse()?,
                }))
            }
        }
    } else if look.peek(Token![const]) {
        let vis = s.parse()?;
        let const_: Token![const] = s.parse()?;
        let look = s.look1();
        let ident = if look.peek(Ident) || look.peek(Token![_]) {
            s.call(Ident::parse_any)?
        } else {
            return Err(look.error());
        };
        let colon = s.parse()?;
        let ty = s.parse()?;
        if s.peek(Token![;]) {
            s.parse::<Token![;]>()?;
            Ok(Item::Stream(parse::parse_verbatim(&beg, s)))
        } else {
            Ok(Item::Const(Const {
                attrs: Vec::new(),
                vis,
                const_,
                ident,
                gens: gen::Gens::default(),
                colon,
                typ: ty,
                eq: s.parse()?,
                expr: s.parse()?,
                semi: s.parse()?,
            }))
        }
    } else if look.peek(Token![unsafe]) {
        ahead.parse::<Token![unsafe]>()?;
        let look = ahead.look1();
        if look.peek(Token![trait]) || look.peek(Token![auto]) && ahead.peek2(Token![trait]) {
            s.parse().map(Item::Trait)
        } else if look.peek(Token![impl]) {
            let allow_verbatim_impl = true;
            if let Some(item) = parse_impl(s, allow_verbatim_impl)? {
                Ok(Item::Impl(item))
            } else {
                Ok(Item::Stream(parse::parse_verbatim(&beg, s)))
            }
        } else if look.peek(Token![extern]) {
            s.parse().map(Item::Foreign)
        } else if look.peek(Token![mod]) {
            s.parse().map(Item::Mod)
        } else {
            Err(look.error())
        }
    } else if look.peek(Token![mod]) {
        s.parse().map(Item::Mod)
    } else if look.peek(Token![type]) {
        parse_item_type(beg, s)
    } else if look.peek(Token![struct]) {
        s.parse().map(Item::Struct)
    } else if look.peek(Token![enum]) {
        s.parse().map(Item::Enum)
    } else if look.peek(Token![union]) && ahead.peek2(Ident) {
        s.parse().map(Item::Union)
    } else if look.peek(Token![trait]) {
        s.call(parse_trait_or_trait_alias)
    } else if look.peek(Token![auto]) && ahead.peek2(Token![trait]) {
        s.parse().map(Item::Trait)
    } else if look.peek(Token![impl]) || look.peek(Token![default]) && !ahead.peek2(Token![!]) {
        let allow_verbatim_impl = true;
        if let Some(item) = parse_impl(s, allow_verbatim_impl)? {
            Ok(Item::Impl(item))
        } else {
            Ok(Item::Stream(parse::parse_verbatim(&beg, s)))
        }
    } else if look.peek(Token![macro]) {
        s.advance_to(&ahead);
        parse_macro2(beg, vis, s)
    } else if vis.is_inherited()
        && (look.peek(Ident)
            || look.peek(Token![self])
            || look.peek(Token![super])
            || look.peek(Token![crate])
            || look.peek(Token![::]))
    {
        s.parse().map(Item::Mac)
    } else {
        Err(look.error())
    }?;
    attrs.extend(item.replace_attrs(Vec::new()));
    item.replace_attrs(attrs);
    Ok(item)
}
fn parse_macro2(beg: Buffer, _: data::Visibility, s: Stream) -> Res<Item> {
    s.parse::<Token![macro]>()?;
    s.parse::<Ident>()?;
    let mut look = s.look1();
    if look.peek(tok::Paren) {
        let y;
        parenthesized!(y in s);
        y.parse::<Stream>()?;
        look = s.look1();
    }
    if look.peek(tok::Brace) {
        let y;
        braced!(y in s);
        y.parse::<Stream>()?;
    } else {
        return Err(look.error());
    }
    Ok(Item::Stream(parse::parse_verbatim(&beg, s)))
}
fn peek_signature(s: Stream) -> bool {
    let y = s.fork();
    y.parse::<Option<Token![const]>>().is_ok()
        && y.parse::<Option<Token![async]>>().is_ok()
        && y.parse::<Option<Token![unsafe]>>().is_ok()
        && y.parse::<Option<Abi>>().is_ok()
        && y.peek(Token![fn])
}
fn parse_rest_of_fn(s: Stream, mut attrs: Vec<attr::Attr>, vis: data::Visibility, sig: Sig) -> Res<Fn> {
    let y;
    let brace = braced!(y in s);
    attr::parse_inners(&y, &mut attrs)?;
    let stmts = y.call(stmt::Block::parse_within)?;
    Ok(Fn {
        attrs,
        vis,
        sig,
        block: Box::new(Block { brace, stmts }),
    })
}
fn parse_fn_arg_or_variadic(s: Stream, attrs: Vec<attr::Attr>, variadic: bool) -> Res<FnArgOrVariadic> {
    let ahead = s.fork();
    if let Ok(mut receiver) = ahead.parse::<Receiver>() {
        s.advance_to(&ahead);
        receiver.attrs = attrs;
        return Ok(FnArgOrVariadic::FnArg(FnArg::Receiver(receiver)));
    }
    if s.peek(Ident) && s.peek2(Token![<]) {
        let span = s.fork().parse::<Ident>()?.span();
        return Ok(FnArgOrVariadic::FnArg(FnArg::Typed(pat::Type {
            attrs,
            pat: Box::new(pat::Pat::Wild(pat::Wild {
                attrs: Vec::new(),
                underscore: Token![_](span),
            })),
            colon: Token![:](span),
            typ: s.parse()?,
        })));
    }
    let pat = Box::new(pat::Pat::parse_single(s)?);
    let colon: Token![:] = s.parse()?;
    if variadic {
        if let Some(dots) = s.parse::<Option<Token![...]>>()? {
            return Ok(FnArgOrVariadic::Variadic(Variadic {
                attrs,
                pat: Some((pat, colon)),
                dots,
                comma: None,
            }));
        }
    }
    Ok(FnArgOrVariadic::FnArg(FnArg::Typed(pat::Type {
        attrs,
        pat,
        colon,
        typ: s.parse()?,
    })))
}
fn parse_fn_args(s: Stream) -> Res<(Puncted<FnArg, Token![,]>, Option<Variadic>)> {
    let mut ys = Puncted::new();
    let mut vari = None;
    let mut has_receiver = false;
    while !s.is_empty() {
        let attrs = s.call(attr::Attr::parse_outers)?;
        if let Some(dots) = s.parse::<Option<Token![...]>>()? {
            vari = Some(Variadic {
                attrs,
                pat: None,
                dots,
                comma: if s.is_empty() { None } else { Some(s.parse()?) },
            });
            break;
        }
        let variadic = true;
        let y = match parse_fn_arg_or_variadic(s, attrs, variadic)? {
            FnArgOrVariadic::FnArg(x) => x,
            FnArgOrVariadic::Variadic(x) => {
                vari = Some(Variadic {
                    comma: if x.is_empty() { None } else { Some(x.parse()?) },
                    ..x
                });
                break;
            },
        };
        match &y {
            FnArg::Receiver(x) if has_receiver => {
                return Err(Err::new(x.self_.span, "unexpected second method receiver"));
            },
            FnArg::Receiver(x) if !ys.is_empty() => {
                return Err(Err::new(x.self_.span, "unexpected method receiver"));
            },
            FnArg::Receiver(_) => has_receiver = true,
            FnArg::Typed(_) => {},
        }
        ys.push_value(y);
        if s.is_empty() {
            break;
        }
        let y: Token![,] = s.parse()?;
        ys.push_punct(y);
    }
    Ok((ys, vari))
}
fn parse_foreign_item_type(beg: Buffer, s: Stream) -> Res<Foreign::Item> {
    let Flexible {
        vis,
        default_: _,
        type_,
        ident,
        gens,
        colon,
        bounds: _,
        typ,
        semi,
    } = Flexible::parse(s, TypeDefault::Disallowed, WhereLoc::Both)?;
    if colon.is_some() || typ.is_some() {
        Ok(Foreign::Item::Stream(parse::parse_verbatim(&beg, s)))
    } else {
        Ok(Foreign::Item::Type(Foreign::Type {
            attrs: Vec::new(),
            vis,
            type_,
            ident,
            gens,
            semi,
        }))
    }
}
fn parse_item_type(beg: Buffer, s: Stream) -> Res<Item> {
    let Flexible {
        vis,
        default_: _,
        type_,
        ident,
        gens,
        colon,
        bounds: _,
        typ,
        semi,
    } = Flexible::parse(s, TypeDefault::Disallowed, WhereLoc::BeforeEq)?;
    let (eq, typ) = match typ {
        Some(x) if colon.is_none() => x,
        _ => return Ok(Item::Stream(parse::parse_verbatim(&beg, s))),
    };
    Ok(Item::Type(Type {
        attrs: Vec::new(),
        vis,
        type_,
        ident,
        gens,
        eq,
        typ: Box::new(typ),
        semi,
    }))
}
fn parse_trait_or_trait_alias(s: Stream) -> Res<Item> {
    let (attrs, vis, trait_, ident, gens) = parse_start_of_trait_alias(s)?;
    let look = s.look1();
    if look.peek(tok::Brace) || look.peek(Token![:]) || look.peek(Token![where]) {
        let unsafe_ = None;
        let auto_ = None;
        parse_rest_of_trait(s, attrs, vis, unsafe_, auto_, trait_, ident, gens).map(Item::Trait)
    } else if look.peek(Token![=]) {
        parse_rest_of_trait_alias(s, attrs, vis, trait_, ident, gens).map(Item::TraitAlias)
    } else {
        Err(look.error())
    }
}
fn parse_rest_of_trait(
    s: Stream,
    mut attrs: Vec<attr::Attr>,
    vis: data::Visibility,
    unsafe_: Option<Token![unsafe]>,
    auto_: Option<Token![auto]>,
    trait_: Token![trait],
    ident: Ident,
    mut gens: gen::Gens,
) -> Res<Trait> {
    let colon: Option<Token![:]> = s.parse()?;
    let mut supers = Puncted::new();
    if colon.is_some() {
        loop {
            if s.peek(Token![where]) || s.peek(tok::Brace) {
                break;
            }
            supers.push_value(s.parse()?);
            if s.peek(Token![where]) || s.peek(tok::Brace) {
                break;
            }
            supers.push_punct(s.parse()?);
        }
    }
    gens.where_ = s.parse()?;
    let y;
    let brace = braced!(y in s);
    attr::parse_inners(&y, &mut attrs)?;
    let mut items = Vec::new();
    while !y.is_empty() {
        items.push(y.parse()?);
    }
    Ok(Trait {
        attrs,
        vis,
        unsafe_,
        auto_,
        restriction: None,
        trait_,
        ident,
        gens,
        colon,
        supers,
        brace,
        items,
    })
}
fn parse_start_of_trait_alias(s: Stream) -> Res<(Vec<attr::Attr>, data::Visibility, Token![trait], Ident, gen::Gens)> {
    let attrs = s.call(attr::Attr::parse_outers)?;
    let vis: data::Visibility = s.parse()?;
    let trait_: Token![trait] = s.parse()?;
    let ident: Ident = s.parse()?;
    let gens: gen::Gens = s.parse()?;
    Ok((attrs, vis, trait_, ident, gens))
}
fn parse_rest_of_trait_alias(
    s: Stream,
    attrs: Vec<attr::Attr>,
    vis: data::Visibility,
    trait_: Token![trait],
    ident: Ident,
    mut gens: gen::Gens,
) -> Res<TraitAlias> {
    let eq: Token![=] = s.parse()?;
    let mut bounds = Puncted::new();
    loop {
        if s.peek(Token![where]) || s.peek(Token![;]) {
            break;
        }
        bounds.push_value(s.parse()?);
        if s.peek(Token![where]) || s.peek(Token![;]) {
            break;
        }
        bounds.push_punct(s.parse()?);
    }
    gens.where_ = s.parse()?;
    let semi: Token![;] = s.parse()?;
    Ok(TraitAlias {
        attrs,
        vis,
        trait_,
        ident,
        gens,
        eq,
        bounds,
        semi,
    })
}
fn parse_trait_item_type(beg: Buffer, s: Stream) -> Res<Trait::Item> {
    let Flexible {
        vis,
        default_: _,
        type_,
        ident,
        gens,
        colon,
        bounds,
        typ,
        semi,
    } = Flexible::parse(s, TypeDefault::Disallowed, WhereLoc::AfterEq)?;
    if vis.is_some() {
        Ok(Trait::Item::Stream(parse::parse_verbatim(&beg, s)))
    } else {
        Ok(Trait::Item::Type(Trait::Type {
            attrs: Vec::new(),
            type_,
            ident,
            gens,
            colon,
            bounds,
            default: typ,
            semi,
        }))
    }
}
fn parse_impl(s: Stream, verbatim: bool) -> Res<Option<Impl>> {
    let mut attrs = s.call(attr::Attr::parse_outers)?;
    let has_visibility = verbatim && s.parse::<data::Visibility>()?.is_some();
    let default_: Option<Token![default]> = s.parse()?;
    let unsafe_: Option<Token![unsafe]> = s.parse()?;
    let impl_: Token![impl] = s.parse()?;
    let has_gens = s.peek(Token![<])
        && (s.peek2(Token![>])
            || s.peek2(Token![#])
            || (s.peek2(Ident) || s.peek2(Life))
                && (s.peek3(Token![:]) || s.peek3(Token![,]) || s.peek3(Token![>]) || s.peek3(Token![=]))
            || s.peek2(Token![const]));
    let mut gens: gen::Gens = if has_gens { s.parse()? } else { gen::Gens::default() };
    let is_const = verbatim && (s.peek(Token![const]) || s.peek(Token![?]) && s.peek2(Token![const]));
    if is_const {
        s.parse::<Option<Token![?]>>()?;
        s.parse::<Token![const]>()?;
    }
    let begin = s.fork();
    let polarity = if s.peek(Token![!]) && !s.peek2(tok::Brace) {
        Some(s.parse::<Token![!]>()?)
    } else {
        None
    };
    let mut first: typ::Type = s.parse()?;
    let self_ty: typ::Type;
    let trait_;
    let is_impl_for = s.peek(Token![for]);
    if is_impl_for {
        let for_: Token![for] = s.parse()?;
        let mut first_ty_ref = &first;
        while let typ::Type::Group(ty) = first_ty_ref {
            first_ty_ref = &ty.elem;
        }
        if let typ::Type::Path(typ::Path { qself: None, .. }) = first_ty_ref {
            while let typ::Type::Group(x) = first {
                first = *x.elem;
            }
            if let typ::Type::Path(typ::Path { qself: None, path }) = first {
                trait_ = Some((polarity, path, for_));
            } else {
                unreachable!();
            }
        } else if !verbatim {
            return Err(Err::new_spanned(first_ty_ref, "expected trait path"));
        } else {
            trait_ = None;
        }
        self_ty = s.parse()?;
    } else {
        trait_ = None;
        self_ty = if polarity.is_none() {
            first
        } else {
            typ::Type::Stream(parse::parse_verbatim(&begin, s))
        };
    }
    gens.where_ = s.parse()?;
    let y;
    let brace = braced!(y in s);
    attr::parse_inners(&y, &mut attrs)?;
    let mut items = Vec::new();
    while !y.is_empty() {
        items.push(y.parse()?);
    }
    if has_visibility || is_const || is_impl_for && trait_.is_none() {
        Ok(None)
    } else {
        Ok(Some(Impl {
            attrs,
            default_,
            unsafe_,
            impl_,
            gens,
            trait_,
            typ: Box::new(self_ty),
            brace,
            items,
        }))
    }
}
fn parse_impl_item_fn(s: Stream, omitted: bool) -> Res<Option<Impl::Fn>> {
    let mut attrs = s.call(attr::Attr::parse_outers)?;
    let vis: data::Visibility = s.parse()?;
    let default_: Option<Token![default]> = s.parse()?;
    let sig: Sig = s.parse()?;
    if omitted && s.parse::<Option<Token![;]>>()?.is_some() {
        return Ok(None);
    }
    let y;
    let brace = braced!(y in s);
    attrs.extend(y.call(attr::Attr::parse_inners)?);
    let block = Block {
        brace,
        stmts: y.call(stmt::Block::parse_within)?,
    };
    Ok(Some(Impl::Fn {
        attrs,
        vis,
        default_,
        sig,
        block,
    }))
}
fn parse_impl_item_type(beg: Buffer, s: Stream) -> Res<Impl::Item> {
    let Flexible {
        vis,
        default_,
        type_,
        ident,
        gens,
        colon,
        bounds: _,
        typ,
        semi,
    } = Flexible::parse(s, TypeDefault::Optional, WhereLoc::AfterEq)?;
    let (eq, typ) = match typ {
        Some(x) if colon.is_none() => x,
        _ => return Ok(Impl::Item::Stream(parse::parse_verbatim(&beg, s))),
    };
    Ok(Impl::Item::Type(Impl::Type {
        attrs: Vec::new(),
        vis,
        default_,
        type_,
        ident,
        gens,
        eq,
        typ,
        semi,
    }))
}
