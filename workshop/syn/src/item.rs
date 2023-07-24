use super::*;
use crate::attr::Filter;
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
impl Lower for File {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.inners());
        s.append_all(&self.items);
    }
}
impl Pretty for File {
    fn pretty(&self, p: &mut Print) {
        p.cbox(0);
        if let Some(x) = &self.shebang {
            p.word(x.clone());
            p.hardbreak();
        }
        p.inner_attrs(&self.attrs);
        for x in &self.items {
            x.pretty(p);
        }
        p.end();
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
        Struct(Struct),
        Trait(Trait),
        TraitAlias(TraitAlias),
        Type(Type),
        Union(Union),
        Use(Use),
        Verbatim(Verbatim),
    }
}
impl Item {
    pub fn replace_attrs(&mut self, ys: Vec<attr::Attr>) -> Vec<attr::Attr> {
        match self {
            Item::Const(Const { attrs, .. })
            | Item::Enum(Enum { attrs, .. })
            | Item::Extern(Extern { attrs, .. })
            | Item::Fn(Fn { attrs, .. })
            | Item::Foreign(Foreign { attrs, .. })
            | Item::Impl(Impl { attrs, .. })
            | Item::Mac(Mac { attrs, .. })
            | Item::Mod(Mod { attrs, .. })
            | Item::Static(Static { attrs, .. })
            | Item::Struct(Struct { attrs, .. })
            | Item::Trait(Trait { attrs, .. })
            | Item::TraitAlias(TraitAlias { attrs, .. })
            | Item::Type(Type { attrs, .. })
            | Item::Union(Union { attrs, .. })
            | Item::Use(Use { attrs, .. }) => mem::replace(attrs, ys),
            Item::Verbatim(_) => Vec::new(),
        }
    }
}
impl From<Input> for Item {
    fn from(x: Input) -> Item {
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
impl Pretty for Item {
    fn pretty(&self, p: &mut Print) {
        use Item::*;
        match self {
            Const(x) => x.pretty(p),
            Enum(x) => x.pretty(p),
            Extern(x) => x.pretty(p),
            Fn(x) => x.pretty(p),
            Foreign(x) => x.pretty(p),
            Impl(x) => x.pretty(p),
            Mac(x) => x.pretty(p),
            Mod(x) => x.pretty(p),
            Static(x) => x.pretty(p),
            Struct(x) => x.pretty(p),
            Trait(x) => x.pretty(p),
            TraitAlias(x) => x.pretty(p),
            Type(x) => x.pretty(p),
            Union(x) => x.pretty(p),
            Use(x) => x.pretty(p),
            Verbatim(x) => x.pretty(p),
        }
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
                if y.peek(ident::Ident) || y.peek(Token![_]) {
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
impl Lower for Const {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.vis.lower(s);
        self.const_.lower(s);
        self.ident.lower(s);
        self.colon.lower(s);
        self.typ.lower(s);
        self.eq.lower(s);
        self.expr.lower(s);
        self.semi.lower(s);
    }
}
impl Pretty for Const {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(0);
        &self.vis.pretty(p);
        p.word("const ");
        &self.ident.pretty(p);
        &self.gens.pretty(p);
        p.word(": ");
        &self.typ.pretty(p);
        p.word(" = ");
        p.neverbreak();
        &self.expr.pretty(p);
        p.word(";");
        p.end();
        p.hardbreak();
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
impl From<Enum> for Input {
    fn from(x: Enum) -> Input {
        Input {
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
impl Lower for Enum {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.vis.lower(s);
        self.enum_.lower(s);
        self.ident.lower(s);
        self.gens.lower(s);
        self.gens.where_.lower(s);
        self.brace.surround(s, |s| {
            self.variants.lower(s);
        });
    }
}
impl Pretty for Enum {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        &self.vis.pretty(p);
        p.word("enum ");
        &self.ident.pretty(p);
        &self.gens.pretty(p);
        p.where_for_body(&self.gens.where_);
        p.word("{");
        p.hardbreak_if_nonempty();
        for x in &self.variants {
            x.pretty(p);
            p.word(",");
            p.hardbreak();
        }
        p.offset(-INDENT);
        p.end();
        p.word("}");
        p.hardbreak();
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
impl Lower for Extern {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.vis.lower(s);
        self.extern_.lower(s);
        self.crate_.lower(s);
        self.ident.lower(s);
        if let Some((as_, x)) = &self.rename {
            as_.lower(s);
            x.lower(s);
        }
        self.semi.lower(s);
    }
}
impl Pretty for Extern {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        &self.vis.pretty(p);
        p.word("extern crate ");
        &self.ident.pretty(p);
        if let Some((_, x)) = &self.rename {
            p.word(" as ");
            x.pretty(p);
        }
        p.word(";");
        p.hardbreak();
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
impl Lower for Fn {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.vis.lower(s);
        self.sig.lower(s);
        self.block.brace.surround(s, |s| {
            s.append_all(self.attrs.inners());
            s.append_all(&self.block.stmts);
        });
    }
}
impl Pretty for Fn {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        &self.vis.pretty(p);
        &self.sig.pretty(p);
        p.where_for_body(&self.sig.gens.where_);
        p.word("{");
        p.hardbreak_if_nonempty();
        p.inner_attrs(&self.attrs);
        for x in &self.block.stmts {
            x.pretty(p);
        }
        p.offset(-INDENT);
        p.end();
        p.word("}");
        p.hardbreak();
    }
}

pub struct Foreign {
    pub attrs: Vec<attr::Attr>,
    pub unsafe_: Option<Token![unsafe]>,
    pub abi: typ::Abi,
    pub brace: tok::Brace,
    pub items: Vec<foreign::Item>,
}
impl Parse for Foreign {
    fn parse(s: Stream) -> Res<Self> {
        let mut attrs = s.call(attr::Attr::parse_outers)?;
        let unsafe_: Option<Token![unsafe]> = s.parse()?;
        let abi: typ::Abi = s.parse()?;
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
impl Lower for Foreign {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.unsafe_.lower(s);
        self.abi.lower(s);
        self.brace.surround(s, |s| {
            s.append_all(self.attrs.inners());
            s.append_all(&self.items);
        });
    }
}
impl Pretty for Foreign {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        if self.unsafe_.is_some() {
            p.word("unsafe ");
        }
        &self.abi.pretty(p);
        p.word("{");
        p.hardbreak_if_nonempty();
        p.inner_attrs(&self.attrs);
        for x in &self.items {
            x.pretty(p);
        }
        p.offset(-INDENT);
        p.end();
        p.word("}");
        p.hardbreak();
    }
}

pub struct Impl {
    pub attrs: Vec<attr::Attr>,
    pub default: Option<Token![default]>,
    pub unsafe_: Option<Token![unsafe]>,
    pub impl_: Token![impl],
    pub gens: gen::Gens,
    pub trait_: Option<(Option<Token![!]>, Path, Token![for])>,
    pub typ: Box<typ::Type>,
    pub brace: tok::Brace,
    pub items: Vec<impl_::Item>,
}
impl Parse for Impl {
    fn parse(s: Stream) -> Res<Self> {
        let verbatim = false;
        parse_impl(s, verbatim).map(Option::unwrap)
    }
}
impl Lower for Impl {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.default.lower(s);
        self.unsafe_.lower(s);
        self.impl_.lower(s);
        self.gens.lower(s);
        if let Some((polarity, path, for_)) = &self.trait_ {
            polarity.lower(s);
            path.lower(s);
            for_.lower(s);
        }
        self.typ.lower(s);
        self.gens.where_.lower(s);
        self.brace.surround(s, |s| {
            s.append_all(self.attrs.inners());
            s.append_all(&self.items);
        });
    }
}
impl Pretty for Impl {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        p.ibox(-INDENT);
        p.cbox(INDENT);
        if self.default.is_some() {
            p.word("default ");
        }
        if self.unsafe_.is_some() {
            p.word("unsafe ");
        }
        p.word("impl");
        &self.gens.pretty(p);
        p.end();
        p.nbsp();
        if let Some((neg, x, _)) = &self.trait_ {
            if neg.is_some() {
                p.word("!");
            }
            x.pretty(p, path::Kind::Type);
            p.space();
            p.word("for ");
        }
        &self.typ.pretty(p);
        p.end();
        p.where_for_body(&self.gens.where_);
        p.word("{");
        p.hardbreak_if_nonempty();
        p.inner_attrs(&self.attrs);
        for x in &self.items {
            x.pretty(p);
        }
        p.offset(-INDENT);
        p.end();
        p.word("}");
        p.hardbreak();
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
impl Lower for Mac {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.mac.path.lower(s);
        self.mac.bang.lower(s);
        self.ident.lower(s);
        match &self.mac.delim {
            tok::Delim::Paren(x) => {
                x.surround(s, |s| self.mac.toks.lower(s));
            },
            tok::Delim::Brace(x) => {
                x.surround(s, |s| self.mac.toks.lower(s));
            },
            tok::Delim::Bracket(x) => {
                x.surround(s, |s| self.mac.toks.lower(s));
            },
        }
        self.semi.lower(s);
    }
}
impl Pretty for Mac {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        let semi = true;
        &self.mac.pretty(p, self.ident.as_ref(), semi);
        p.hardbreak();
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
impl Lower for Mod {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.vis.lower(s);
        self.unsafe_.lower(s);
        self.mod_.lower(s);
        self.ident.lower(s);
        if let Some((brace, xs)) = &self.items {
            brace.surround(s, |s| {
                s.append_all(self.attrs.inners());
                s.append_all(xs);
            });
        } else {
            ToksOrDefault(&self.semi).lower(s);
        }
    }
}
impl Pretty for Mod {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        &self.vis.pretty(p);
        if self.unsafe_.is_some() {
            p.word("unsafe ");
        }
        p.word("mod ");
        &self.ident.pretty(p);
        if let Some((_, xs)) = &self.items {
            p.word(" {");
            p.hardbreak_if_nonempty();
            p.inner_attrs(&self.attrs);
            for x in xs {
                x.pretty(p);
            }
            p.offset(-INDENT);
            p.end();
            p.word("}");
        } else {
            p.word(";");
            p.end();
        }
        p.hardbreak();
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
impl Lower for Static {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.vis.lower(s);
        self.static_.lower(s);
        self.mut_.lower(s);
        self.ident.lower(s);
        self.colon.lower(s);
        self.typ.lower(s);
        self.eq.lower(s);
        self.expr.lower(s);
        self.semi.lower(s);
    }
}
impl Pretty for Static {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(0);
        &self.vis.pretty(p);
        p.word("static ");
        &self.mut_.pretty(p);
        &self.ident.pretty(p);
        p.word(": ");
        &self.typ.pretty(p);
        p.word(" = ");
        p.neverbreak();
        &self.expr.pretty(p);
        p.word(";");
        p.end();
        p.hardbreak();
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
impl From<Struct> for Input {
    fn from(x: Struct) -> Input {
        Input {
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
impl Lower for Struct {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.vis.lower(s);
        self.struct_.lower(s);
        self.ident.lower(s);
        self.gens.lower(s);
        match &self.fields {
            data::Fields::Named(x) => {
                self.gens.where_.lower(s);
                x.lower(s);
            },
            data::Fields::Unnamed(x) => {
                x.lower(s);
                self.gens.where_.lower(s);
                ToksOrDefault(&self.semi).lower(s);
            },
            data::Fields::Unit => {
                self.gens.where_.lower(s);
                ToksOrDefault(&self.semi).lower(s);
            },
        }
    }
}
impl Pretty for Struct {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        &self.vis.pretty(p);
        p.word("struct ");
        &self.ident.pretty(p);
        &self.gens.pretty(p);
        use data::Fields::*;
        match &self.fields {
            Named(xs) => {
                p.where_for_body(&self.gens.where_);
                p.word("{");
                p.hardbreak_if_nonempty();
                for x in &xs.fields {
                    x.pretty(p);
                    p.word(",");
                    p.hardbreak();
                }
                p.offset(-INDENT);
                p.end();
                p.word("}");
            },
            Unnamed(x) => {
                x.pretty(p);
                p.where_with_semi(&self.gens.where_);
                p.end();
            },
            Unit => {
                p.where_with_semi(&self.gens.where_);
                p.end();
            },
        }
        p.hardbreak();
    }
}

pub struct Trait {
    pub attrs: Vec<attr::Attr>,
    pub vis: data::Visibility,
    pub unsafe_: Option<Token![unsafe]>,
    pub auto: Option<Token![auto]>,
    pub restriction: Option<impl_::Restriction>,
    pub trait_: Token![trait],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub colon: Option<Token![:]>,
    pub supers: Puncted<gen::bound::Type, Token![+]>,
    pub brace: tok::Brace,
    pub items: Vec<trait_::Item>,
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
impl Lower for Trait {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.vis.lower(s);
        self.unsafe_.lower(s);
        self.auto.lower(s);
        self.trait_.lower(s);
        self.ident.lower(s);
        self.gens.lower(s);
        if !self.supers.is_empty() {
            ToksOrDefault(&self.colon).lower(s);
            self.supers.lower(s);
        }
        self.gens.where_.lower(s);
        self.brace.surround(s, |s| {
            s.append_all(self.attrs.inners());
            s.append_all(&self.items);
        });
    }
}
impl Pretty for Trait {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        &self.vis.pretty(p);
        if self.unsafe_.is_some() {
            p.word("unsafe ");
        }
        if self.auto.is_some() {
            p.word("auto ");
        }
        p.word("trait ");
        &self.ident.pretty(p);
        &self.gens.pretty(p);
        for x in self.supers.iter().delimited() {
            if x.is_first {
                p.word(": ");
            } else {
                p.word(" + ");
            }
            &x.pretty(p);
        }
        p.where_for_body(&self.gens.where_);
        p.word("{");
        p.hardbreak_if_nonempty();
        p.inner_attrs(&self.attrs);
        for x in &self.items {
            x.pretty(p);
        }
        p.offset(-INDENT);
        p.end();
        p.word("}");
        p.hardbreak();
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
impl Lower for TraitAlias {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.vis.lower(s);
        self.trait_.lower(s);
        self.ident.lower(s);
        self.gens.lower(s);
        self.eq.lower(s);
        self.bounds.lower(s);
        self.gens.where_.lower(s);
        self.semi.lower(s);
    }
}
impl Pretty for TraitAlias {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        &self.vis.pretty(p);
        p.word("trait ");
        &self.ident.pretty(p);
        &self.gens.pretty(p);
        p.word(" = ");
        p.neverbreak();
        for x in self.bounds.iter().delimited() {
            if !x.is_first {
                p.space();
                p.word("+ ");
            }
            &x.pretty(p);
        }
        p.where_with_semi(&self.gens.where_);
        p.end();
        p.hardbreak();
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
impl Lower for Type {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.vis.lower(s);
        self.type_.lower(s);
        self.ident.lower(s);
        self.gens.lower(s);
        self.gens.where_.lower(s);
        self.eq.lower(s);
        self.typ.lower(s);
        self.semi.lower(s);
    }
}
impl Pretty for Type {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        &self.vis.pretty(p);
        p.word("type ");
        &self.ident.pretty(p);
        &self.gens.pretty(p);
        p.where_oneline(&self.gens.where_);
        p.word("= ");
        p.neverbreak();
        p.ibox(-INDENT);
        &self.typ.pretty(p);
        p.end();
        p.word(";");
        p.end();
        p.hardbreak();
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
impl From<Union> for Input {
    fn from(x: Union) -> Input {
        Input {
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
impl Lower for Union {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.vis.lower(s);
        self.union_.lower(s);
        self.ident.lower(s);
        self.gens.lower(s);
        self.gens.where_.lower(s);
        self.fields.lower(s);
    }
}
impl Pretty for Union {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        &self.vis.pretty(p);
        p.word("union ");
        &self.ident.pretty(p);
        &self.gens.pretty(p);
        p.where_for_body(&self.gens.where_);
        p.word("{");
        p.hardbreak_if_nonempty();
        for x in &self.fields.fields {
            x.pretty(p);
            p.word(",");
            p.hardbreak();
        }
        p.offset(-INDENT);
        p.end();
        p.word("}");
        p.hardbreak();
    }
}

pub struct Use {
    pub attrs: Vec<attr::Attr>,
    pub vis: data::Visibility,
    pub use_: Token![use],
    pub colon: Option<Token![::]>,
    pub tree: use_::Tree,
    pub semi: Token![;],
}
impl Parse for Use {
    fn parse(s: Stream) -> Res<Self> {
        let root = false;
        parse_item_use(s, root).map(Option::unwrap)
    }
}
impl Lower for Use {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.vis.lower(s);
        self.use_.lower(s);
        self.colon.lower(s);
        self.tree.lower(s);
        self.semi.lower(s);
    }
}
impl Pretty for Use {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        &self.vis.pretty(p);
        p.word("use ");
        if self.colon.is_some() {
            p.word("::");
        }
        &self.tree.pretty(p);
        p.word(";");
        p.hardbreak();
    }
}
fn parse_item_use(s: Stream, root: bool) -> Res<Option<Use>> {
    let attrs = s.call(attr::Attr::parse_outers)?;
    let vis: data::Visibility = s.parse()?;
    let use_: Token![use] = s.parse()?;
    let colon: Option<Token![::]> = s.parse()?;
    let tree = use_::parse_tree(s, root && colon.is_none())?;
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

pub struct Verbatim(pub pm2::Stream);
impl Pretty for Verbatim {
    fn pretty(&self, p: &mut Print) {
        use verbatim::{FlexibleItemConst, FlexibleItemFn, FlexibleItemStatic, FlexibleItemType, WhereClauseLocation};
        enum ItemVerbatim {
            Ellipsis,
            Empty,
            FlexConst(FlexibleItemConst),
            FlexFn(FlexibleItemFn),
            FlexImpl(FlexImpl),
            FlexStatic(FlexibleItemStatic),
            FlexType(FlexibleItemType),
            Mac(Mac),
            UseBrace(UseBrace),
        }
        struct FlexImpl {
            attrs: Vec<attr::Attr>,
            vis: data::Visibility,
            defaultness: bool,
            unsafe_: bool,
            gens: gen::Gens,
            const_: ImplConstness,
            neg: bool,
            trait_: Option<Type>,
            typ: Type,
            impls: Vec<Impl>,
        }
        enum ImplConstness {
            None,
            MaybeConst,
            Const,
        }
        struct Mac {
            attrs: Vec<attr::Attr>,
            vis: data::Visibility,
            ident: Ident,
            args: Option<Stream>,
            body: Stream,
        }
        struct UseBrace {
            attrs: Vec<attr::Attr>,
            vis: data::Visibility,
            trees: punct::Puncted<RootUseTree, Token![,]>,
        }
        struct RootUseTree {
            colon: Option<Token![::]>,
            inner: use_::Tree,
        }
        impl parse::Parse for ImplConstness {
            fn parse(s: parse::Stream) -> Res<Self> {
                use ImplConstness::*;
                if s.parse::<Option<Token![?]>>()?.is_some() {
                    s.parse::<Token![const]>()?;
                    Ok(MaybeConst)
                } else if s.parse::<Option<Token![const]>>()?.is_some() {
                    Ok(Const)
                } else {
                    Ok(None)
                }
            }
        }
        impl parse::Parse for RootUseTree {
            fn parse(s: parse::Stream) -> Res<Self> {
                Ok(RootUseTree {
                    colon: s.parse()?,
                    inner: s.parse()?,
                })
            }
        }
        impl parse::Parse for ItemVerbatim {
            fn parse(s: parse::Stream) -> Res<Self> {
                if s.is_empty() {
                    return Ok(ItemVerbatim::Empty);
                } else if s.peek(Token![...]) {
                    s.parse::<Token![...]>()?;
                    return Ok(ItemVerbatim::Ellipsis);
                }
                let mut attrs = s.call(attr::Attr::parse_outer)?;
                let vis: data::Visibility = s.parse()?;
                let look = s.lookahead1();
                if look.peek(Token![const]) && (s.peek2(Ident) || s.peek2(Token![_])) {
                    let default = false;
                    let y = FlexibleItemConst::parse(attrs, vis, default, s)?;
                    Ok(ItemVerbatim::FlexConst(y))
                } else if s.peek(Token![const])
                    || look.peek(Token![async])
                    || look.peek(Token![unsafe]) && !s.peek2(Token![impl])
                    || look.peek(Token![extern])
                    || look.peek(Token![fn])
                {
                    let default = false;
                    let y = FlexibleItemFn::parse(attrs, vis, default, s)?;
                    Ok(ItemVerbatim::FlexFn(y))
                } else if look.peek(Token![default]) || s.peek(Token![unsafe]) || look.peek(Token![impl]) {
                    let default = s.parse::<Option<Token![default]>>()?.is_some();
                    let unsafe_ = s.parse::<Option<Token![unsafe]>>()?.is_some();
                    s.parse::<Token![impl]>()?;
                    let has_generics = s.peek(Token![<])
                        && (s.peek2(Token![>])
                            || s.peek2(Token![#])
                            || (s.peek2(Ident) || s.peek2(Lifetime))
                                && (s.peek3(Token![:])
                                    || s.peek3(Token![,])
                                    || s.peek3(Token![>])
                                    || s.peek3(Token![=]))
                            || s.peek2(Token![const]));
                    let mut gens: gen::Gens = if has_generics { s.parse()? } else { gen::Gens::default() };
                    let const_: ImplConstness = s.parse()?;
                    let neg = !s.peek2(tok::Brace) && s.parse::<Option<Token![!]>>()?.is_some();
                    let first: Type = s.parse()?;
                    let (trait_, typ) = if s.parse::<Option<Token![for]>>()?.is_some() {
                        (Some(first), s.parse()?)
                    } else {
                        (None, first)
                    };
                    gens.where_ = s.parse()?;
                    let y;
                    braced!(y in s);
                    let inner_attrs = y.call(attr::Attr::parse_inner)?;
                    attrs.extend(inner_attrs);
                    let mut impls = Vec::new();
                    while !y.is_empty() {
                        impls.push(y.parse()?);
                    }
                    Ok(ItemVerbatim::FlexImpl(FlexImpl {
                        attrs,
                        vis,
                        default,
                        unsafe_,
                        gens,
                        const_,
                        neg,
                        trait_,
                        typ,
                        impls,
                    }))
                } else if look.peek(Token![macro]) {
                    s.parse::<Token![macro]>()?;
                    let ident: Ident = s.parse()?;
                    let args = if s.peek(tok::Paren) {
                        let y;
                        parenthesized!(y in s);
                        Some(y.parse::<Stream>()?)
                    } else {
                        None
                    };
                    let y;
                    braced!(y in s);
                    let body: Stream = y.parse()?;
                    Ok(ItemVerbatim::Mac(Mac {
                        attrs,
                        vis,
                        ident,
                        args,
                        body,
                    }))
                } else if look.peek(Token![static]) {
                    let y = FlexibleItemStatic::parse(attrs, vis, s)?;
                    Ok(ItemVerbatim::FlexStatic(y))
                } else if look.peek(Token![type]) {
                    let default = false;
                    let y = FlexibleItemType::parse(attrs, vis, default, s, WhereClauseLocation::BeforeEq)?;
                    Ok(ItemVerbatim::FlexType(y))
                } else if look.peek(Token![use]) {
                    s.parse::<Token![use]>()?;
                    let y;
                    braced!(y in s);
                    let trees = y.parse_terminated(RootUseTree::parse, Token![,])?;
                    s.parse::<Token![;]>()?;
                    Ok(ItemVerbatim::UseBrace(UseBrace { attrs, vis, trees }))
                } else {
                    Err(look.error())
                }
            }
        }
        let item: ItemVerbatim = match parse2(self.clone()) {
            Ok(x) => x,
            Err(_) => unimplemented!("Item::Verbatim `{}`", self),
        };
        use ItemVerbatim::*;
        match item {
            Empty => {
                p.hardbreak();
            },
            Ellipsis => {
                p.word("...");
                p.hardbreak();
            },
            FlexConst(x) => {
                &x.pretty(p);
            },
            FlexFn(x) => {
                &x.pretty(p);
            },
            FlexImpl(x) => {
                p.outer_attrs(&x.attrs);
                p.cbox(INDENT);
                p.ibox(-INDENT);
                p.cbox(INDENT);
                &x.vis.pretty(p);
                if x.defaultness {
                    p.word("default ");
                }
                if x.unsafe_ {
                    p.word("unsafe ");
                }
                p.word("impl");
                &x.gens.pretty(p);
                p.end();
                p.nbsp();
                match x.const_ {
                    ImplConstness::None => {},
                    ImplConstness::MaybeConst => p.word("?const "),
                    ImplConstness::Const => p.word("const "),
                }
                if x.neg {
                    p.word("!");
                }
                if let Some(x) = &x.trait_ {
                    x.pretty(p);
                    p.space();
                    p.word("for ");
                }
                &x.typ.pretty(p);
                p.end();
                p.where_for_body(&x.gens.where_);
                p.word("{");
                p.hardbreak_if_nonempty();
                p.inner_attrs(&x.attrs);
                for x in &x.impls {
                    x.pretty(p);
                }
                p.offset(-INDENT);
                p.end();
                p.word("}");
                p.hardbreak();
            },
            Mac(x) => {
                p.outer_attrs(&x.attrs);
                &x.vis.pretty(p);
                p.word("macro ");
                &x.ident.pretty(p);
                if let Some(x) = &x.args {
                    p.word("(");
                    p.cbox(INDENT);
                    p.zerobreak();
                    p.ibox(0);
                    p.macro_rules_tokens(x.clone(), true);
                    p.end();
                    p.zerobreak();
                    p.offset(-INDENT);
                    p.end();
                    p.word(")");
                }
                p.word(" {");
                if !x.body.is_empty() {
                    p.neverbreak();
                    p.cbox(INDENT);
                    p.hardbreak();
                    p.ibox(0);
                    p.macro_rules_tokens(x.body.clone(), false);
                    p.end();
                    p.hardbreak();
                    p.offset(-INDENT);
                    p.end();
                }
                p.word("}");
                p.hardbreak();
            },
            FlexStatic(x) => {
                p.flexible_item_static(&x);
            },
            FlexType(x) => {
                p.flexible_item_type(&x);
            },
            UseBrace(x) => {
                p.outer_attrs(&x.attrs);
                &x.vis.pretty(p);
                p.word("use ");
                if x.trees.len() == 1 {
                    p.word("::");
                    p.use_tree(&x.trees[0].inner);
                } else {
                    p.cbox(INDENT);
                    p.word("{");
                    p.zerobreak();
                    p.ibox(0);
                    for x in x.trees.iter().delimited() {
                        if x.colon.is_some() {
                            p.word("::");
                        }
                        &x.inner.pretty(p);
                        if !x.is_last {
                            p.word(",");
                            let mut y = &x.inner;
                            while let use_::Tree::Path(x) = y {
                                y = &x.tree;
                            }
                            if let use_::Tree::Group(_) = y {
                                p.hardbreak();
                            } else {
                                p.space();
                            }
                        }
                    }
                    p.end();
                    p.trailing_comma(true);
                    p.offset(-INDENT);
                    p.word("}");
                    p.end();
                }
                p.word(";");
                p.hardbreak();
            },
        }
    }
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
impl Lower for Receiver {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        if let Some((amper, life)) = &self.ref_ {
            amper.lower(s);
            life.lower(s);
        }
        self.mut_.lower(s);
        self.self_.lower(s);
        if let Some(x) = &self.colon {
            x.lower(s);
            self.typ.lower(s);
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
                <Token![:]>::default().lower(s);
                self.typ.lower(s);
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
    pub abi: Option<typ::Abi>,
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
        let abi: Option<typ::Abi> = s.parse()?;
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
impl Lower for Sig {
    fn lower(&self, s: &mut Stream) {
        self.const_.lower(s);
        self.async_.lower(s);
        self.unsafe_.lower(s);
        self.abi.lower(s);
        self.fn_.lower(s);
        self.ident.lower(s);
        self.gens.lower(s);
        self.paren.surround(s, |s| {
            self.args.lower(s);
            if let Some(x) = &self.vari {
                if !self.args.empty_or_trailing() {
                    <Token![,]>::default().lower(s);
                }
                x.lower(s);
            }
        });
        self.ret.lower(s);
        self.gens.where_.lower(s);
    }
}

pub struct Variadic {
    pub attrs: Vec<attr::Attr>,
    pub pat: Option<(Box<pat::Pat>, Token![:])>,
    pub dots: Token![...],
    pub comma: Option<Token![,]>,
}
impl Lower for Variadic {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        if let Some((pat, colon)) = &self.pat {
            pat.lower(s);
            colon.lower(s);
        }
        self.dots.lower(s);
        self.comma.lower(s);
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
impl Lower for StaticMut {
    fn lower(&self, s: &mut Stream) {
        match self {
            StaticMut::None => {},
            StaticMut::Mut(x) => x.lower(s),
        }
    }
}

pub mod foreign {
    use super::*;
    ast_enum_of_structs! {
        pub enum Item {
            Fn(Fn),
            Mac(Mac),
            Static(Static),
            Type(Type),
            Stream(pm2::Stream),
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
                && (look.peek(ident::Ident)
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
    impl Lower for Fn {
        fn lower(&self, s: &mut Stream) {
            s.append_all(self.attrs.outers());
            self.vis.lower(s);
            self.sig.lower(s);
            self.semi.lower(s);
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
    impl Lower for Static {
        fn lower(&self, s: &mut Stream) {
            s.append_all(self.attrs.outers());
            self.vis.lower(s);
            self.static_.lower(s);
            self.mut_.lower(s);
            self.ident.lower(s);
            self.colon.lower(s);
            self.typ.lower(s);
            self.semi.lower(s);
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
    impl Lower for Type {
        fn lower(&self, s: &mut Stream) {
            s.append_all(self.attrs.outers());
            self.vis.lower(s);
            self.type_.lower(s);
            self.ident.lower(s);
            self.gens.lower(s);
            self.gens.where_.lower(s);
            self.semi.lower(s);
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
    impl Lower for Mac {
        fn lower(&self, s: &mut Stream) {
            s.append_all(self.attrs.outers());
            self.mac.lower(s);
            self.semi.lower(s);
        }
    }
}
pub mod impl_ {
    use super::*;
    ast_enum_of_structs! {
        pub enum Item {
            Const(Const),
            Fn(Fn),
            Mac(Mac),
            Type(Type),
            Stream(pm2::Stream),
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
                let ident = if look.peek(ident::Ident) || look.peek(Token![_]) {
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
                && (look.peek(ident::Ident)
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
                    if look.peek(ident::Ident) || look.peek(Token![_]) {
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
    impl Lower for Const {
        fn lower(&self, s: &mut Stream) {
            s.append_all(self.attrs.outers());
            self.vis.lower(s);
            self.default_.lower(s);
            self.const_.lower(s);
            self.ident.lower(s);
            self.colon.lower(s);
            self.typ.lower(s);
            self.eq.lower(s);
            self.expr.lower(s);
            self.semi.lower(s);
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
    impl Lower for Fn {
        fn lower(&self, s: &mut Stream) {
            s.append_all(self.attrs.outers());
            self.vis.lower(s);
            self.default_.lower(s);
            self.sig.lower(s);
            self.block.brace.surround(s, |s| {
                s.append_all(self.attrs.inners());
                s.append_all(&self.block.stmts);
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
    impl Lower for Type {
        fn lower(&self, s: &mut Stream) {
            s.append_all(self.attrs.outers());
            self.vis.lower(s);
            self.default_.lower(s);
            self.type_.lower(s);
            self.ident.lower(s);
            self.gens.lower(s);
            self.eq.lower(s);
            self.typ.lower(s);
            self.gens.where_.lower(s);
            self.semi.lower(s);
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
    impl Lower for Mac {
        fn lower(&self, s: &mut Stream) {
            s.append_all(self.attrs.outers());
            self.mac.lower(s);
            self.semi.lower(s);
        }
    }

    pub enum Restriction {}
}
pub mod trait_ {
    use super::*;
    ast_enum_of_structs! {
        pub enum Item {
            Const(Const),
            Fn(Fn),
            Type(Type),
            Mac(Mac),
            Stream(pm2::Stream),
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
                if look.peek(ident::Ident) || look.peek(Token![_]) {
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
                && (look.peek(ident::Ident)
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
                    if look.peek(ident::Ident) || look.peek(Token![_]) {
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
    impl Lower for Const {
        fn lower(&self, s: &mut Stream) {
            s.append_all(self.attrs.outers());
            self.const_.lower(s);
            self.ident.lower(s);
            self.colon.lower(s);
            self.typ.lower(s);
            if let Some((eq, x)) = &self.default {
                eq.lower(s);
                x.lower(s);
            }
            self.semi.lower(s);
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
                default: brace.map(|brace| stmt::Block { brace, stmts }),
                semi,
            })
        }
    }
    impl Lower for Fn {
        fn lower(&self, s: &mut Stream) {
            s.append_all(self.attrs.outers());
            self.sig.lower(s);
            match &self.default {
                Some(block) => {
                    block.brace.surround(s, |s| {
                        s.append_all(self.attrs.inners());
                        s.append_all(&block.stmts);
                    });
                },
                None => {
                    ToksOrDefault(&self.semi).lower(s);
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
    impl Lower for Type {
        fn lower(&self, s: &mut Stream) {
            s.append_all(self.attrs.outers());
            self.type_.lower(s);
            self.ident.lower(s);
            self.gens.lower(s);
            if !self.bounds.is_empty() {
                ToksOrDefault(&self.colon).lower(s);
                self.bounds.lower(s);
            }
            if let Some((eq, x)) = &self.default {
                eq.lower(s);
                x.lower(s);
            }
            self.gens.where_.lower(s);
            self.semi.lower(s);
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
    impl Lower for Mac {
        fn lower(&self, s: &mut Stream) {
            s.append_all(self.attrs.outers());
            self.mac.lower(s);
            self.semi.lower(s);
        }
    }
}
pub mod use_ {
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
    impl Pretty for Tree {
        fn pretty(&self, p: &mut Print) {
            use Tree::*;
            match self {
                Glob(x) => x.pretty(p),
                Group(x) => x.pretty(p),
                Name(x) => x.pretty(p),
                Path(x) => x.pretty(p),
                Rename(x) => x.pretty(p),
            }
        }
    }
    pub fn parse_tree(s: Stream, root: bool) -> Res<Option<Tree>> {
        let look = s.look1();
        if look.peek(ident::Ident)
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
                        if s.peek(ident::Ident) {
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
                Ok(Some(Tree::Group(Group { brace, trees: elems })))
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
    impl Lower for Path {
        fn lower(&self, s: &mut Stream) {
            self.ident.lower(s);
            self.colon2.lower(s);
            self.tree.lower(s);
        }
    }
    impl Pretty for Path {
        fn pretty(&self, p: &mut Print) {
            &self.ident.pretty(p);
            p.word("::");
            &self.tree.pretty(p);
        }
    }

    pub struct Name {
        pub ident: Ident,
    }
    impl Lower for Name {
        fn lower(&self, s: &mut Stream) {
            self.ident.lower(s);
        }
    }
    impl Pretty for Name {
        fn pretty(&self, p: &mut Print) {
            &self.ident.pretty(p);
        }
    }

    pub struct Rename {
        pub ident: Ident,
        pub as_: Token![as],
        pub rename: Ident,
    }
    impl Lower for Rename {
        fn lower(&self, s: &mut Stream) {
            self.ident.lower(s);
            self.as_.lower(s);
            self.rename.lower(s);
        }
    }
    impl Pretty for Rename {
        fn pretty(&self, p: &mut Print) {
            &self.ident.pretty(p);
            p.word(" as ");
            &self.rename.pretty(p);
        }
    }

    pub struct Glob {
        pub star: Token![*],
    }
    impl Lower for Glob {
        fn lower(&self, s: &mut Stream) {
            self.star.lower(s);
        }
    }
    impl Pretty for Glob {
        fn pretty(&self, p: &mut Print) {
            let _ = self;
            p.word("*");
        }
    }
    pub struct Group {
        pub brace: tok::Brace,
        pub trees: Puncted<Tree, Token![,]>,
    }
    impl Lower for Group {
        fn lower(&self, s: &mut Stream) {
            self.brace.surround(s, |s| {
                self.trees.lower(s);
            });
        }
    }
    impl Pretty for Group {
        fn pretty(&self, p: &mut Print) {
            if self.trees.is_empty() {
                p.word("{}");
            } else if self.trees.len() == 1 {
                &self.trees[0].pretty(p);
            } else {
                p.cbox(INDENT);
                p.word("{");
                p.zerobreak();
                p.ibox(0);
                for x in self.trees.iter().delimited() {
                    &x.pretty(p);
                    if !x.is_last {
                        p.word(",");
                        let mut y = *x;
                        while let Tree::Path(x) = y {
                            y = &x.tree;
                        }
                        if let Tree::Group(_) = y {
                            p.hardbreak();
                        } else {
                            p.space();
                        }
                    }
                }
                p.end();
                p.trailing_comma(true);
                p.offset(-INDENT);
                p.word("}");
                p.end();
            }
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

pub fn parse_rest_of_item(beg: parse::Buffer, mut attrs: Vec<attr::Attr>, s: Stream) -> Res<Item> {
    let ahead = s.fork();
    let vis: data::Visibility = ahead.parse()?;
    let look = ahead.look1();
    let mut item = if look.peek(Token![fn]) || peek_signature(&ahead) {
        let vis: data::Visibility = s.parse()?;
        let sig: Sig = s.parse()?;
        if s.peek(Token![;]) {
            s.parse::<Token![;]>()?;
            Ok(Item::Verbatim(parse::parse_verbatim(&beg, s)))
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
            None => Ok(Item::Verbatim(parse::parse_verbatim(&beg, s))),
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
            Ok(Item::Verbatim(parse::parse_verbatim(&beg, s)))
        } else {
            let colon = s.parse()?;
            let ty = s.parse()?;
            if s.peek(Token![;]) {
                s.parse::<Token![;]>()?;
                Ok(Item::Verbatim(parse::parse_verbatim(&beg, s)))
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
        let ident = if look.peek(ident::Ident) || look.peek(Token![_]) {
            s.call(Ident::parse_any)?
        } else {
            return Err(look.error());
        };
        let colon = s.parse()?;
        let ty = s.parse()?;
        if s.peek(Token![;]) {
            s.parse::<Token![;]>()?;
            Ok(Item::Verbatim(parse::parse_verbatim(&beg, s)))
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
                Ok(Item::Verbatim(parse::parse_verbatim(&beg, s)))
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
    } else if look.peek(Token![union]) && ahead.peek2(ident::Ident) {
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
            Ok(Item::Verbatim(parse::parse_verbatim(&beg, s)))
        }
    } else if look.peek(Token![macro]) {
        s.advance_to(&ahead);
        parse_macro2(beg, vis, s)
    } else if vis.is_inherited()
        && (look.peek(ident::Ident)
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
fn parse_macro2(beg: parse::Buffer, _: data::Visibility, s: Stream) -> Res<Item> {
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
    Ok(Item::Verbatim(parse::parse_verbatim(&beg, s)))
}
fn peek_signature(s: Stream) -> bool {
    let y = s.fork();
    y.parse::<Option<Token![const]>>().is_ok()
        && y.parse::<Option<Token![async]>>().is_ok()
        && y.parse::<Option<Token![unsafe]>>().is_ok()
        && y.parse::<Option<typ::Abi>>().is_ok()
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
        block: Box::new(stmt::Block { brace, stmts }),
    })
}
fn parse_fn_arg_or_variadic(s: Stream, attrs: Vec<attr::Attr>, variadic: bool) -> Res<FnArgOrVariadic> {
    let ahead = s.fork();
    if let Ok(mut receiver) = ahead.parse::<Receiver>() {
        s.advance_to(&ahead);
        receiver.attrs = attrs;
        return Ok(FnArgOrVariadic::FnArg(FnArg::Receiver(receiver)));
    }
    if s.peek(ident::Ident) && s.peek2(Token![<]) {
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
    let pat = Box::new(pat::Pat::parse_one(s)?);
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
fn parse_foreign_item_type(beg: parse::Buffer, s: Stream) -> Res<foreign::Item> {
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
fn parse_item_type(beg: parse::Buffer, s: Stream) -> Res<Item> {
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
        _ => return Ok(Item::Verbatim(parse::parse_verbatim(&beg, s))),
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
        auto: auto_,
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
fn parse_trait_item_type(beg: parse::Buffer, s: Stream) -> Res<trait_::Item> {
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
        Ok(trait_::Item::Stream(parse::parse_verbatim(&beg, s)))
    } else {
        Ok(trait_::Item::Type(trait_::Type {
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
            || (s.peek2(ident::Ident) || s.peek2(Life))
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
            default: default_,
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
fn parse_impl_item_fn(s: Stream, omitted: bool) -> Res<Option<impl_::Fn>> {
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
    let block = stmt::Block {
        brace,
        stmts: y.call(stmt::Block::parse_within)?,
    };
    Ok(Some(impl_::Fn {
        attrs,
        vis,
        default_,
        sig,
        block,
    }))
}
fn parse_impl_item_type(beg: parse::Buffer, s: Stream) -> Res<impl_::Item> {
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
        _ => return Ok(impl_::Item::Stream(parse::parse_verbatim(&beg, s))),
    };
    Ok(impl_::Item::Type(impl_::Type {
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
