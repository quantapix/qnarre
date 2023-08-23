use super::*;
use crate::attr::Filter;

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
impl Clone for File {
    fn clone(&self) -> Self {
        File {
            shebang: self.shebang.clone(),
            attrs: self.attrs.clone(),
            items: self.items.clone(),
        }
    }
}
impl Eq for File {}
impl PartialEq for File {
    fn eq(&self, x: &Self) -> bool {
        self.shebang == x.shebang && self.attrs == x.attrs && self.items == x.items
    }
}
impl<H> Hash for File
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.shebang.hash(h);
        self.attrs.hash(h);
        self.items.hash(h);
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
impl<V> Visit for File
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        for x in &self.items {
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        for x in &mut self.items {
            x.visit_mut(v);
        }
    }
}

enum_of_structs! {
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
        Alias(Alias),
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
            | Item::Alias(Alias { attrs, .. })
            | Item::Type(Type { attrs, .. })
            | Item::Union(Union { attrs, .. })
            | Item::Use(Use { attrs, .. }) => std::mem::replace(attrs, ys),
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
impl Clone for Item {
    fn clone(&self) -> Self {
        use Item::*;
        match self {
            Const(x) => Const(x.clone()),
            Enum(x) => Enum(x.clone()),
            Extern(x) => Extern(x.clone()),
            Fn(x) => Fn(x.clone()),
            Foreign(x) => Foreign(x.clone()),
            Impl(x) => Impl(x.clone()),
            Mac(x) => Mac(x.clone()),
            Mod(x) => Mod(x.clone()),
            Static(x) => Static(x.clone()),
            Struct(x) => Struct(x.clone()),
            Trait(x) => Trait(x.clone()),
            Alias(x) => Alias(x.clone()),
            Type(x) => Type(x.clone()),
            Union(x) => Union(x.clone()),
            Use(x) => Use(x.clone()),
            Verbatim(x) => Verbatim(x.clone()),
        }
    }
}
impl Eq for Item {}
impl PartialEq for Item {
    fn eq(&self, x: &Self) -> bool {
        use Item::*;
        match (self, x) {
            (Const(x), Const(y)) => x == y,
            (Enum(x), Enum(y)) => x == y,
            (Extern(x), Extern(y)) => x == y,
            (Fn(x), Fn(y)) => x == y,
            (Foreign(x), Foreign(y)) => x == y,
            (Impl(x), Impl(y)) => x == y,
            (Mac(x), Mac(y)) => x == y,
            (Mod(x), Mod(y)) => x == y,
            (Static(x), Static(y)) => x == y,
            (Struct(x), Struct(y)) => x == y,
            (Trait(x), Trait(y)) => x == y,
            (Alias(x), Alias(y)) => x == y,
            (Type(x), Type(y)) => x == y,
            (Union(x), Union(y)) => x == y,
            (Use(x), Use(y)) => x == y,
            (Verbatim(x), Verbatim(y)) => StreamHelper(x) == StreamHelper(y),
            _ => false,
        }
    }
}
impl<H> Hash for Item
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use Item::*;
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
            Alias(x) => {
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
            Alias(x) => x.pretty(p),
            Type(x) => x.pretty(p),
            Union(x) => x.pretty(p),
            Use(x) => x.pretty(p),
            Verbatim(x) => x.pretty(p),
        }
    }
}
impl<V> Visit for Item
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        use Item::*;
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
            Alias(x) => {
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
            Verbatim(_) => {},
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        use Item::*;
        match self {
            Const(x) => {
                x.visit_mut(v);
            },
            Enum(x) => {
                x.visit_mut(v);
            },
            Extern(x) => {
                x.visit_mut(v);
            },
            Fn(x) => {
                x.visit_mut(v);
            },
            Foreign(x) => {
                x.visit_mut(v);
            },
            Impl(x) => {
                x.visit_mut(v);
            },
            Mac(x) => {
                x.visit_mut(v);
            },
            Mod(x) => {
                x.visit_mut(v);
            },
            Static(x) => {
                x.visit_mut(v);
            },
            Struct(x) => {
                x.visit_mut(v);
            },
            Trait(x) => {
                x.visit_mut(v);
            },
            Alias(x) => {
                x.visit_mut(v);
            },
            Type(x) => {
                x.visit_mut(v);
            },
            Union(x) => {
                x.visit_mut(v);
            },
            Use(x) => {
                x.visit_mut(v);
            },
            Verbatim(_) => {},
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
impl Clone for Const {
    fn clone(&self) -> Self {
        Const {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            const_: self.const_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
            eq: self.eq.clone(),
            expr: self.expr.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Eq for Const {}
impl PartialEq for Const {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.ident == x.ident
            && self.gens == x.gens
            && self.typ == x.typ
            && self.expr == x.expr
    }
}
impl<H> Hash for Const
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
impl<V> Visit for Const
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.vis.visit(v);
        &self.ident.visit(v);
        &self.gens.visit(v);
        &*self.typ.visit(v);
        &*self.expr.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
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
impl Clone for Enum {
    fn clone(&self) -> Self {
        Enum {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            enum_: self.enum_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            brace: self.brace.clone(),
            variants: self.variants.clone(),
        }
    }
}
impl Eq for Enum {}
impl PartialEq for Enum {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.ident == x.ident
            && self.gens == x.gens
            && self.variants == x.variants
    }
}
impl<H> Hash for Enum
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
impl<V> Visit for Enum
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
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
    fn visit_mut(&mut self, v: &mut V) {
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
impl Clone for Extern {
    fn clone(&self) -> Self {
        Extern {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            extern_: self.extern_.clone(),
            crate_: self.crate_.clone(),
            ident: self.ident.clone(),
            rename: self.rename.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Eq for Extern {}
impl PartialEq for Extern {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.vis == x.vis && self.ident == x.ident && self.rename == x.rename
    }
}
impl<H> Hash for Extern
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
impl<V> Visit for Extern
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.vis.visit(v);
        &self.ident.visit(v);
        if let Some(x) = &self.rename {
            &(x).1.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
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
impl Clone for Fn {
    fn clone(&self) -> Self {
        Fn {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            sig: self.sig.clone(),
            block: self.block.clone(),
        }
    }
}
impl Eq for Fn {}
impl PartialEq for Fn {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.vis == x.vis && self.sig == x.sig && self.block == x.block
    }
}
impl<H> Hash for Fn
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
impl<V> Visit for Fn
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.vis.visit(v);
        &self.sig.visit(v);
        &*self.block.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut self.vis.visit_mut(v);
        &mut self.sig.visit_mut(v);
        &mut *self.block.visit_mut(v);
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
impl Clone for Foreign {
    fn clone(&self) -> Self {
        Foreign {
            attrs: self.attrs.clone(),
            unsafe_: self.unsafe_.clone(),
            abi: self.abi.clone(),
            brace: self.brace.clone(),
            items: self.items.clone(),
        }
    }
}
impl Eq for Foreign {}
impl PartialEq for Foreign {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.unsafe_ == x.unsafe_ && self.abi == x.abi && self.items == x.items
    }
}
impl<H> Hash for Foreign
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
impl<V> Visit for Foreign
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.abi.visit(v);
        for x in &self.items {
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut self.abi.visit_mut(v);
        for x in &mut self.items {
            x.visit_mut(v);
        }
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
impl Clone for Impl {
    fn clone(&self) -> Self {
        Impl {
            attrs: self.attrs.clone(),
            default: self.default.clone(),
            unsafe_: self.unsafe_.clone(),
            impl_: self.impl_.clone(),
            gens: self.gens.clone(),
            trait_: self.trait_.clone(),
            typ: self.typ.clone(),
            brace: self.brace.clone(),
            items: self.items.clone(),
        }
    }
}
impl Eq for Impl {}
impl PartialEq for Impl {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.default == x.default
            && self.unsafe_ == x.unsafe_
            && self.gens == x.gens
            && self.trait_ == x.trait_
            && self.typ == x.typ
            && self.items == x.items
    }
}
impl<H> Hash for Impl
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
            x.pretty_with_args(p, path::Kind::Type);
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
impl<V> Visit for Impl
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
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
    fn visit_mut(&mut self, v: &mut V) {
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
            tok::Delim::Parenth(x) => {
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
impl Clone for Mac {
    fn clone(&self) -> Self {
        Mac {
            attrs: self.attrs.clone(),
            ident: self.ident.clone(),
            mac: self.mac.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Eq for Mac {}
impl PartialEq for Mac {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.ident == x.ident && self.mac == x.mac && self.semi == x.semi
    }
}
impl<H> Hash for Mac
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
impl Pretty for Mac {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        let semi = true;
        &self.mac.pretty_with_args(p, (self.ident.as_ref(), semi));
        p.hardbreak();
    }
}
impl<V> Visit for Mac
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        if let Some(x) = &self.ident {
            x.visit(v);
        }
        &self.mac.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        if let Some(x) = &mut self.ident {
            x.visit_mut(v);
        }
        &mut self.mac.visit_mut(v);
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
impl Clone for Mod {
    fn clone(&self) -> Self {
        Mod {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            unsafe_: self.unsafe_.clone(),
            mod_: self.mod_.clone(),
            ident: self.ident.clone(),
            items: self.items.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Eq for Mod {}
impl PartialEq for Mod {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.unsafe_ == x.unsafe_
            && self.ident == x.ident
            && self.items == x.items
            && self.semi == x.semi
    }
}
impl<H> Hash for Mod
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
impl<V> Visit for Mod
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
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
    fn visit_mut(&mut self, v: &mut V) {
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
impl Clone for Static {
    fn clone(&self) -> Self {
        Static {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            static_: self.static_.clone(),
            mut_: self.mut_.clone(),
            ident: self.ident.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
            eq: self.eq.clone(),
            expr: self.expr.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Eq for Static {}
impl PartialEq for Static {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.mut_ == x.mut_
            && self.ident == x.ident
            && self.typ == x.typ
            && self.expr == x.expr
    }
}
impl<H> Hash for Static
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
impl<V> Visit for Static
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.vis.visit(v);
        &self.mut_.visit(v);
        &self.ident.visit(v);
        &*self.typ.visit(v);
        &*self.expr.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
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
impl Clone for Struct {
    fn clone(&self) -> Self {
        Struct {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            struct_: self.struct_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            fields: self.fields.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Eq for Struct {}
impl PartialEq for Struct {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.ident == x.ident
            && self.gens == x.gens
            && self.fields == x.fields
            && self.semi == x.semi
    }
}
impl<H> Hash for Struct
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
impl<V> Visit for Struct
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.vis.visit(v);
        &self.ident.visit(v);
        &self.gens.visit(v);
        &self.fields.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut self.vis.visit_mut(v);
        &mut self.ident.visit_mut(v);
        &mut self.gens.visit_mut(v);
        &mut self.fields.visit_mut(v);
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
impl Clone for Trait {
    fn clone(&self) -> Self {
        Trait {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            unsafe_: self.unsafe_.clone(),
            auto: self.auto.clone(),
            restriction: self.restriction.clone(),
            trait_: self.trait_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            colon: self.colon.clone(),
            supers: self.supers.clone(),
            brace: self.brace.clone(),
            items: self.items.clone(),
        }
    }
}
impl Eq for Trait {}
impl PartialEq for Trait {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.unsafe_ == x.unsafe_
            && self.auto == x.auto
            && self.restriction == x.restriction
            && self.ident == x.ident
            && self.gens == x.gens
            && self.colon == x.colon
            && self.supers == x.supers
            && self.items == x.items
    }
}
impl<H> Hash for Trait
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
impl<V> Visit for Trait
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
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
    fn visit_mut(&mut self, v: &mut V) {
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

pub struct Alias {
    pub attrs: Vec<attr::Attr>,
    pub vis: data::Visibility,
    pub trait_: Token![trait],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub eq: Token![=],
    pub bounds: Puncted<gen::bound::Type, Token![+]>,
    pub semi: Token![;],
}
impl Parse for Alias {
    fn parse(s: Stream) -> Res<Self> {
        let (attrs, vis, trait_, ident, gens) = parse_start_of_trait_alias(s)?;
        parse_rest_of_trait_alias(s, attrs, vis, trait_, ident, gens)
    }
}
impl Lower for Alias {
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
impl Clone for Alias {
    fn clone(&self) -> Self {
        Alias {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            trait_: self.trait_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            eq: self.eq.clone(),
            bounds: self.bounds.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Eq for Alias {}
impl PartialEq for Alias {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.ident == x.ident
            && self.gens == x.gens
            && self.bounds == x.bounds
    }
}
impl<H> Hash for Alias
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
impl Pretty for Alias {
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
impl<V> Visit for Alias
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
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
    fn visit_mut(&mut self, v: &mut V) {
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
impl Clone for Type {
    fn clone(&self) -> Self {
        Type {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            type_: self.type_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            eq: self.eq.clone(),
            typ: self.typ.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Eq for Type {}
impl PartialEq for Type {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.vis == x.vis && self.ident == x.ident && self.gens == x.gens && self.typ == x.typ
    }
}
impl<H> Hash for Type
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
impl<V> Visit for Type
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.vis.visit(v);
        &self.ident.visit(v);
        &self.gens.visit(v);
        &*self.typ.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut self.vis.visit_mut(v);
        &mut self.ident.visit_mut(v);
        &mut self.gens.visit_mut(v);
        &mut *self.typ.visit_mut(v);
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
impl Clone for Union {
    fn clone(&self) -> Self {
        Union {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            union_: self.union_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            fields: self.fields.clone(),
        }
    }
}
impl Eq for Union {}
impl PartialEq for Union {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.vis == x.vis
            && self.ident == x.ident
            && self.gens == x.gens
            && self.fields == x.fields
    }
}
impl<H> Hash for Union
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
impl<V> Visit for Union
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.vis.visit(v);
        &self.ident.visit(v);
        &self.gens.visit(v);
        &self.fields.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut self.vis.visit_mut(v);
        &mut self.ident.visit_mut(v);
        &mut self.gens.visit_mut(v);
        &mut self.fields.visit_mut(v);
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
impl Clone for Use {
    fn clone(&self) -> Self {
        Use {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            use_: self.use_.clone(),
            colon: self.colon.clone(),
            tree: self.tree.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Eq for Use {}
impl PartialEq for Use {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.vis == x.vis && self.colon == x.colon && self.tree == x.tree
    }
}
impl<H> Hash for Use
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
impl<V> Visit for Use
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.vis.visit(v);
        &self.tree.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut self.vis.visit_mut(v);
        &mut self.tree.visit_mut(v);
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
        enum Type {
            Ellipsis,
            Empty,
            FlexConst(flex::Const),
            FlexFn(flex::Fn),
            FlexImpl(FlexImpl),
            FlexStatic(flex::Static),
            FlexType(flex::Type),
            Mac(Mac),
            UseBrace(UseBrace),
        }
        struct FlexImpl {
            attrs: Vec<attr::Attr>,
            vis: data::Visibility,
            default: bool,
            unsafe_: bool,
            gens: gen::Gens,
            const_: Const,
            neg: bool,
            trait_: Option<Type>,
            typ: Type,
            impls: Vec<Impl>,
        }
        enum Const {
            None,
            MaybeConst,
            Const,
        }
        struct Mac {
            attrs: Vec<attr::Attr>,
            vis: data::Visibility,
            ident: Ident,
            args: Option<pm2::Stream>,
            body: pm2::Stream,
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
        impl parse::Parse for Const {
            fn parse(s: parse::Stream) -> Res<Self> {
                if s.parse::<Option<Token![?]>>()?.is_some() {
                    s.parse::<Token![const]>()?;
                    Ok(Const::MaybeConst)
                } else if s.parse::<Option<Token![const]>>()?.is_some() {
                    Ok(Const::Const)
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
        impl parse::Parse for Type {
            fn parse(s: parse::Stream) -> Res<Self> {
                if s.is_empty() {
                    return Ok(Type::Empty);
                } else if s.peek(Token![...]) {
                    s.parse::<Token![...]>()?;
                    return Ok(Type::Ellipsis);
                }
                let mut attrs = s.call(attr::Attr::parse_outer)?;
                let vis: data::Visibility = s.parse()?;
                let look = s.lookahead1();
                if look.peek(Token![const]) && (s.peek2(ident::Ident) || s.peek2(Token![_])) {
                    let default = false;
                    let y = flex::Const::parse(attrs, vis, default, s)?;
                    Ok(Type::FlexConst(y))
                } else if s.peek(Token![const])
                    || look.peek(Token![async])
                    || look.peek(Token![unsafe]) && !s.peek2(Token![impl])
                    || look.peek(Token![extern])
                    || look.peek(Token![fn])
                {
                    let default = false;
                    let y = flex::Fn::parse(attrs, vis, default, s)?;
                    Ok(Type::FlexFn(y))
                } else if look.peek(Token![default]) || s.peek(Token![unsafe]) || look.peek(Token![impl]) {
                    let default = s.parse::<Option<Token![default]>>()?.is_some();
                    let unsafe_ = s.parse::<Option<Token![unsafe]>>()?.is_some();
                    s.parse::<Token![impl]>()?;
                    let has_generics = s.peek(Token![<])
                        && (s.peek2(Token![>])
                            || s.peek2(Token![#])
                            || (s.peek2(ident::Ident) || s.peek2(Life))
                                && (s.peek3(Token![:])
                                    || s.peek3(Token![,])
                                    || s.peek3(Token![>])
                                    || s.peek3(Token![=]))
                            || s.peek2(Token![const]));
                    let mut gens: gen::Gens = if has_generics { s.parse()? } else { gen::Gens::default() };
                    let const_: Const = s.parse()?;
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
                    let ys = y.call(attr::Attr::parse_inner)?;
                    attrs.extend(ys);
                    let mut impls = Vec::new();
                    while !y.is_empty() {
                        impls.push(y.parse()?);
                    }
                    Ok(Type::FlexImpl(FlexImpl {
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
                    let args = if s.peek(tok::Parenth) {
                        let y;
                        parenthed!(y in s);
                        Some(y.parse::<Stream>()?)
                    } else {
                        None
                    };
                    let y;
                    braced!(y in s);
                    let body: Stream = y.parse()?;
                    Ok(Type::Mac(Mac {
                        attrs,
                        vis,
                        ident,
                        args,
                        body,
                    }))
                } else if look.peek(Token![static]) {
                    let y = flex::Static::parse(attrs, vis, s)?;
                    Ok(Type::FlexStatic(y))
                } else if look.peek(Token![type]) {
                    let default = false;
                    let y = flex::Type::parse(attrs, vis, default, s, WhereLoc::BeforeEq)?;
                    Ok(Type::FlexType(y))
                } else if look.peek(Token![use]) {
                    s.parse::<Token![use]>()?;
                    let y;
                    braced!(y in s);
                    let trees = y.parse_terminated(RootUseTree::parse, Token![,])?;
                    s.parse::<Token![;]>()?;
                    Ok(Type::UseBrace(UseBrace { attrs, vis, trees }))
                } else {
                    Err(look.error())
                }
            }
        }
        let y: Type = match parse2(self.clone()) {
            Ok(x) => x,
            Err(_) => unimplemented!("Item::Verbatim `{}`", self),
        };
        use Type::*;
        match y {
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
                if x.default {
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
                    Const::None => {},
                    Const::MaybeConst => p.word("?const "),
                    Const::Const => p.word("const "),
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
impl Clone for Receiver {
    fn clone(&self) -> Self {
        Receiver {
            attrs: self.attrs.clone(),
            ref_: self.ref_.clone(),
            mut_: self.mut_.clone(),
            self_: self.self_.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
        }
    }
}
impl Eq for Receiver {}
impl PartialEq for Receiver {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.ref_ == x.ref_
            && self.mut_ == x.mut_
            && self.colon == x.colon
            && self.typ == x.typ
    }
}
impl<H> Hash for Receiver
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
impl Pretty for Receiver {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        if let Some((_, x)) = &self.ref_ {
            p.word("&");
            if let Some(x) = x {
                x.pretty(p);
                p.nbsp();
            }
        }
        if self.mut_.is_some() {
            p.word("mut ");
        }
        p.word("self");
        if self.colon.is_some() {
            p.word(": ");
            &self.typ.pretty(p);
        } else {
            let consistent = match (&self.ref_, &self.mut_, &*self.typ) {
                (Some(_), mut_, typ::Ref(x)) => {
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
                p.word(": ");
                &self.typ.pretty(p);
            }
        }
    }
}
impl<V> Visit for Receiver
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
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
    fn visit_mut(&mut self, v: &mut V) {
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

enum_of_structs! {
    pub enum FnArg {
        Receiver(Receiver),
        Type(pat::Type),
    }
}
impl Parse for FnArg {
    fn parse(s: Stream) -> Res<Self> {
        let variadic = false;
        let attrs = s.call(attr::Attr::parse_outers)?;
        use FnArgOrVari::*;
        match parse_fn_arg_or_variadic(s, attrs, variadic)? {
            FnArg(x) => Ok(x),
            Variadic(_) => unreachable!(),
        }
    }
}
impl Clone for FnArg {
    fn clone(&self) -> Self {
        use FnArg::*;
        match self {
            Receiver(x) => Receiver(x.clone()),
            Type(x) => Type(x.clone()),
        }
    }
}
impl Eq for FnArg {}
impl PartialEq for FnArg {
    fn eq(&self, x: &Self) -> bool {
        use FnArg::*;
        match (self, x) {
            (Receiver(x), Receiver(y)) => x == y,
            (Type(x), Type(y)) => x == y,
            _ => false,
        }
    }
}
impl<H> Hash for FnArg
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use FnArg::*;
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
impl Pretty for FnArg {
    fn pretty(&self, p: &mut Print) {
        use FnArg::*;
        match self {
            Receiver(x) => x.pretty(p),
            Type(x) => x.pretty(p),
        }
    }
}
impl<V> Visit for FnArg
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        use FnArg::*;
        match self {
            Receiver(x) => {
                x.visit(v);
            },
            Type(x) => {
                x.visit(v);
            },
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        use FnArg::*;
        match self {
            Receiver(x) => {
                x.visit_mut(v);
            },
            Type(x) => {
                x.visit_mut(v);
            },
        }
    }
}

enum FnArgOrVari {
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
    pub parenth: tok::Parenth,
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
        let parenth = parenthed!(y in s);
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
            parenth,
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
        self.parenth.surround(s, |s| {
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
impl Clone for Sig {
    fn clone(&self) -> Self {
        Sig {
            const_: self.const_.clone(),
            async_: self.async_.clone(),
            unsafe_: self.unsafe_.clone(),
            abi: self.abi.clone(),
            fn_: self.fn_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            parenth: self.parenth.clone(),
            args: self.args.clone(),
            vari: self.vari.clone(),
            ret: self.ret.clone(),
        }
    }
}
impl Eq for Sig {}
impl PartialEq for Sig {
    fn eq(&self, x: &Self) -> bool {
        self.const_ == x.const_
            && self.async_ == x.async_
            && self.unsafe_ == x.unsafe_
            && self.abi == x.abi
            && self.ident == x.ident
            && self.gens == x.gens
            && self.args == x.args
            && self.vari == x.vari
            && self.ret == x.ret
    }
}
impl<H> Hash for Sig
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
impl Pretty for Sig {
    fn pretty(&self, p: &mut Print) {
        if self.const_.is_some() {
            p.word("const ");
        }
        if self.async_.is_some() {
            p.word("async ");
        }
        if self.unsafe_.is_some() {
            p.word("unsafe ");
        }
        if let Some(x) = &self.abi {
            p.abi(x);
        }
        p.word("fn ");
        &self.ident.pretty(p);
        &self.gens.pretty(p);
        p.word("(");
        p.neverbreak();
        p.cbox(0);
        p.zerobreak();
        for x in self.args.iter().delimited() {
            &x.pretty(p);
            let last = x.is_last && self.vari.is_none();
            p.trailing_comma(last);
        }
        if let Some(x) = &self.vari {
            x.pretty(p);
            p.zerobreak();
        }
        p.offset(-INDENT);
        p.end();
        p.word(")");
        p.cbox(-INDENT);
        &self.ret.pretty(p);
        p.end();
    }
}
impl<V> Visit for Sig
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
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
    fn visit_mut(&mut self, v: &mut V) {
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
impl Clone for StaticMut {
    fn clone(&self) -> Self {
        use StaticMut::*;
        match self {
            Mut(x) => Mut(x.clone()),
            None => None,
        }
    }
}
impl Eq for StaticMut {}
impl PartialEq for StaticMut {
    fn eq(&self, x: &Self) -> bool {
        use StaticMut::*;
        match (self, x) {
            (Mut(_), Mut(_)) => true,
            (None, None) => true,
            _ => false,
        }
    }
}
impl<H> Hash for StaticMut
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        use StaticMut::*;
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
impl Pretty for StaticMut {
    fn pretty(&self, p: &mut Print) {
        use StaticMut::*;
        match self {
            Mut(_) => p.word("mut "),
            None => {},
        }
    }
}
impl<V> Visit for StaticMut
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        use StaticMut::*;
        match self {
            Mut(_) => {},
            None => {},
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        use StaticMut::*;
        match self {
            Mut(_) => {},
            None => {},
        }
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
impl Clone for Variadic {
    fn clone(&self) -> Self {
        Variadic {
            attrs: self.attrs.clone(),
            pat: self.pat.clone(),
            dots: self.dots.clone(),
            comma: self.comma.clone(),
        }
    }
}
impl Eq for Variadic {}
impl PartialEq for Variadic {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.pat == x.pat && self.comma == x.comma
    }
}
impl<H> Hash for Variadic
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.comma.hash(h);
    }
}
impl Pretty for Variadic {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        if let Some((x, _)) = &self.pat {
            x.pretty(p);
            p.word(": ");
        }
        p.word("...");
    }
}
impl<V> Visit for Variadic
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        if let Some(x) = &self.pat {
            &*(x).0.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        if let Some(x) = &mut self.pat {
            &mut *(x).0.visit_mut(v);
        }
    }
}

pub mod foreign {
    use super::*;
    enum_of_structs! {
        pub enum Item {
            Fn(Fn),
            Mac(Mac),
            Static(Static),
            Type(Type),
            Verbatim(Verbatim),
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
                    Ok(Item::Verbatim(parse::parse_verbatim(&beg, s)))
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
                    Ok(Item::Verbatim(parse::parse_verbatim(&beg, s)))
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
                Item::Verbatim(_) => return Ok(y),
            };
            attrs.append(ys);
            *ys = attrs;
            Ok(y)
        }
    }
    impl Clone for Item {
        fn clone(&self) -> Self {
            use self::Item::*;
            match self {
                Fn(x) => Fn(x.clone()),
                Static(x) => Static(x.clone()),
                Type(x) => Type(x.clone()),
                Mac(x) => Mac(x.clone()),
                Verbatim(x) => Verbatim(x.clone()),
            }
        }
    }
    impl Eq for Item {}
    impl PartialEq for Item {
        fn eq(&self, x: &Self) -> bool {
            use self::Item::*;
            match (self, x) {
                (Fn(x), Fn(y)) => x == y,
                (Mac(x), Mac(y)) => x == y,
                (Static(x), Static(y)) => x == y,
                (Type(x), Type(y)) => x == y,
                (Verbatim(x), Verbatim(y)) => StreamHelper(x) == StreamHelper(y),
                _ => false,
            }
        }
    }
    impl<H> Hash for Item
    where
        H: Hasher,
    {
        fn hash(&self, h: &mut H) {
            use self::Item::*;
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
    impl Pretty for Item {
        fn pretty(&self, p: &mut Print) {
            use self::Item::*;
            match self {
                Fn(x) => x.pretty(p),
                Mac(x) => x.pretty(p),
                Static(x) => x.pretty(p),
                Type(x) => x.pretty(p),
                Verbatim(x) => x.pretty(p),
            }
        }
    }
    impl<V> Visit for Item
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            use self::Item::*;
            match self {
                Fn(x) => {
                    x.visit(v);
                },
                Mac(x) => {
                    x.visit(v);
                },
                Static(x) => {
                    x.visit(v);
                },
                Type(x) => {
                    x.visit(v);
                },
                Verbatim(_) => {},
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            use self::Item::*;
            match self {
                Fn(x) => {
                    x.visit_mut(v);
                },
                Mac(x) => {
                    x.visit_mut(v);
                },
                Static(x) => {
                    x.visit_mut(v);
                },
                Type(x) => {
                    x.visit_mut(v);
                },
                Verbatim(_) => {},
            }
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
    impl Clone for Fn {
        fn clone(&self) -> Self {
            Fn {
                attrs: self.attrs.clone(),
                vis: self.vis.clone(),
                sig: self.sig.clone(),
                semi: self.semi.clone(),
            }
        }
    }
    impl Eq for Fn {}
    impl PartialEq for Fn {
        fn eq(&self, x: &Self) -> bool {
            self.attrs == x.attrs && self.vis == x.vis && self.sig == x.sig
        }
    }
    impl<H> Hash for Fn
    where
        H: Hasher,
    {
        fn hash(&self, h: &mut H) {
            self.attrs.hash(h);
            self.vis.hash(h);
            self.sig.hash(h);
        }
    }
    impl Pretty for Fn {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            p.cbox(INDENT);
            &self.vis.pretty(p);
            &self.sig.pretty(p);
            p.where_with_semi(&self.sig.gens.where_);
            p.end();
            p.hardbreak();
        }
    }
    impl<V> Visit for Fn
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            for x in &self.attrs {
                x.visit(v);
            }
            &self.vis.visit(v);
            &self.sig.visit(v);
        }
        fn visit_mut(&mut self, v: &mut V) {
            for x in &mut self.attrs {
                x.visit_mut(v);
            }
            &mut self.vis.visit_mut(v);
            &mut self.sig.visit_mut(v);
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
    impl Clone for Mac {
        fn clone(&self) -> Self {
            Mac {
                attrs: self.attrs.clone(),
                mac: self.mac.clone(),
                semi: self.semi.clone(),
            }
        }
    }
    impl Eq for Mac {}
    impl PartialEq for Mac {
        fn eq(&self, x: &Self) -> bool {
            self.attrs == x.attrs && self.mac == x.mac && self.semi == x.semi
        }
    }
    impl<H> Hash for Mac
    where
        H: Hasher,
    {
        fn hash(&self, h: &mut H) {
            self.attrs.hash(h);
            self.mac.hash(h);
            self.semi.hash(h);
        }
    }
    impl Pretty for Mac {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            let semi = true;
            &self.mac.pretty_with_args(p, (None, semi));
            p.hardbreak();
        }
    }
    impl<V> Visit for Mac
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            for x in &self.attrs {
                x.visit(v);
            }
            &self.mac.visit(v);
        }
        fn visit_mut(&mut self, v: &mut V) {
            for x in &mut self.attrs {
                x.visit_mut(v);
            }
            &mut self.mac.visit_mut(v);
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
    impl Clone for Static {
        fn clone(&self) -> Self {
            Static {
                attrs: self.attrs.clone(),
                vis: self.vis.clone(),
                static_: self.static_.clone(),
                mut_: self.mut_.clone(),
                ident: self.ident.clone(),
                colon: self.colon.clone(),
                typ: self.typ.clone(),
                semi: self.semi.clone(),
            }
        }
    }
    impl Eq for Static {}
    impl PartialEq for Static {
        fn eq(&self, x: &Self) -> bool {
            self.attrs == x.attrs
                && self.vis == x.vis
                && self.mut_ == x.mut_
                && self.ident == x.ident
                && self.typ == x.typ
        }
    }
    impl<H> Hash for Static
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
            p.word(";");
            p.end();
            p.hardbreak();
        }
    }
    impl<V> Visit for Static
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            for x in &self.attrs {
                x.visit(v);
            }
            &self.vis.visit(v);
            &self.mut_.visit(v);
            &self.ident.visit(v);
            &*self.typ.visit(v);
        }
        fn visit_mut(&mut self, v: &mut V) {
            for x in &mut self.attrs {
                x.visit_mut(v);
            }
            &mut self.vis.visit_mut(v);
            &mut self.mut_.visit_mut(v);
            &mut self.ident.visit_mut(v);
            &mut *self.typ.visit_mut(v);
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
    impl Clone for Type {
        fn clone(&self) -> Self {
            Type {
                attrs: self.attrs.clone(),
                vis: self.vis.clone(),
                type_: self.type_.clone(),
                ident: self.ident.clone(),
                gens: self.gens.clone(),
                semi: self.semi.clone(),
            }
        }
    }
    impl Eq for Type {}
    impl PartialEq for Type {
        fn eq(&self, x: &Self) -> bool {
            self.attrs == x.attrs && self.vis == x.vis && self.ident == x.ident && self.gens == x.gens
        }
    }
    impl<H> Hash for Type
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
    impl Pretty for Type {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            p.cbox(0);
            &self.vis.pretty(p);
            p.word("type ");
            &self.ident.pretty(p);
            &self.gens.pretty(p);
            p.word(";");
            p.end();
            p.hardbreak();
        }
    }
    impl<V> Visit for Type
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            for x in &self.attrs {
                x.visit(v);
            }
            &self.vis.visit(v);
            &self.ident.visit(v);
            &self.gens.visit(v);
        }
        fn visit_mut(&mut self, v: &mut V) {
            for x in &mut self.attrs {
                x.visit_mut(v);
            }
            &mut self.vis.visit_mut(v);
            &mut self.ident.visit_mut(v);
            &mut self.gens.visit_mut(v);
        }
    }

    pub struct Verbatim(pub pm2::Stream);
    impl Pretty for Verbatim {
        fn pretty(&self, p: &mut Print) {
            enum Type {
                Ellipsis,
                Empty,
                FlexFn(flex::Fn),
                FlexStatic(flex::Static),
                FlexType(flex::Type),
            }
            impl parse::Parse for Type {
                fn parse(s: parse::Stream) -> Res<Self> {
                    if s.is_empty() {
                        return Ok(Type::Empty);
                    } else if s.peek(Token![...]) {
                        s.parse::<Token![...]>()?;
                        return Ok(Type::Ellipsis);
                    }
                    let attrs = s.call(attr::Attr::parse_outer)?;
                    let vis: data::Visibility = s.parse()?;
                    let default = false;
                    let look = s.lookahead1();
                    if look.peek(Token![const])
                        || look.peek(Token![async])
                        || look.peek(Token![unsafe])
                        || look.peek(Token![extern])
                        || look.peek(Token![fn])
                    {
                        let y = flex::Fn::parse(attrs, vis, default, s)?;
                        Ok(Type::FlexFn(y))
                    } else if look.peek(Token![static]) {
                        let y = flex::Static::parse(attrs, vis, s)?;
                        Ok(Type::FlexStatic(y))
                    } else if look.peek(Token![type]) {
                        let y = flex::Type::parse(attrs, vis, default, s, WhereLoc::Both)?;
                        Ok(Type::FlexType(y))
                    } else {
                        Err(look.error())
                    }
                }
            }
            let y: Type = match parse2(self.clone()) {
                Ok(x) => x,
                Err(_) => unimplemented!("foreign::Item::Verbatim `{}`", self),
            };
            match y {
                Type::Empty => {
                    p.hardbreak();
                },
                Type::Ellipsis => {
                    p.word("...");
                    p.hardbreak();
                },
                Type::FlexFn(x) => {
                    p.flexible_item_fn(&x);
                },
                Type::FlexStatic(x) => {
                    p.flexible_item_static(&x);
                },
                Type::FlexType(x) => {
                    p.flexible_item_type(&x);
                },
            }
        }
    }
}
pub mod impl_ {
    use super::*;
    enum_of_structs! {
        pub enum Item {
            Const(Const),
            Fn(Fn),
            Mac(Mac),
            Type(Type),
            Verbatim(Verbatim),
        }
    }
    impl Parse for Item {
        fn parse(s: Stream) -> Res<Self> {
            let beg = s.fork();
            let mut attrs = s.call(attr::Attr::parse_outers)?;
            let ahead = s.fork();
            let vis: data::Visibility = ahead.parse()?;
            let mut look = ahead.look1();
            let default = if look.peek(Token![default]) && !ahead.peek2(Token![!]) {
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
                    Ok(Item::Verbatim(parse::parse_verbatim(&beg, s)))
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
                        default,
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
                    return Ok(Item::Verbatim(parse::parse_verbatim(&beg, s)));
                }
            } else if look.peek(Token![type]) {
                parse_impl_item_type(beg, s)
            } else if vis.is_inherited()
                && default.is_none()
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
                Item::Verbatim(_) => return Ok(y),
            };
            attrs.append(ys);
            *ys = attrs;
            Ok(y)
        }
    }
    impl Clone for Item {
        fn clone(&self) -> Self {
            use self::Item::*;
            match self {
                Const(x) => Const(x.clone()),
                Fn(x) => Fn(x.clone()),
                Mac(x) => Mac(x.clone()),
                Type(x) => Type(x.clone()),
                Verbatim(x) => Verbatim(x.clone()),
            }
        }
    }
    impl Eq for Item {}
    impl PartialEq for Item {
        fn eq(&self, x: &Self) -> bool {
            use self::Item::*;
            match (self, x) {
                (Const(x), Const(y)) => x == y,
                (Fn(x), Fn(y)) => x == y,
                (Mac(x), Mac(y)) => x == y,
                (Type(x), Type(y)) => x == y,
                (Verbatim(x), Verbatim(y)) => StreamHelper(x) == StreamHelper(y),
                _ => false,
            }
        }
    }
    impl<H> Hash for Item
    where
        H: Hasher,
    {
        fn hash(&self, h: &mut H) {
            use self::Item::*;
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
    impl Pretty for Item {
        fn pretty(&self, p: &mut Print) {
            use self::Item::*;
            match self {
                Const(x) => x.pretty(p),
                Fn(x) => x.pretty(p),
                Mac(x) => x.pretty(p),
                Type(x) => x.pretty(p),
                Verbatim(x) => x.pretty(p),
            }
        }
    }
    impl<V> Visit for Item
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            use self::Item::*;
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
                Verbatim(_) => {},
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            use self::Item::*;
            match self {
                Const(x) => {
                    x.visit_mut(v);
                },
                Fn(x) => {
                    x.visit_mut(v);
                },
                Type(x) => {
                    x.visit_mut(v);
                },
                Mac(x) => {
                    x.visit_mut(v);
                },
                Verbatim(_) => {},
            }
        }
    }

    pub struct Const {
        pub attrs: Vec<attr::Attr>,
        pub vis: data::Visibility,
        pub default: Option<Token![default]>,
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
                default: s.parse()?,
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
            self.default.lower(s);
            self.const_.lower(s);
            self.ident.lower(s);
            self.colon.lower(s);
            self.typ.lower(s);
            self.eq.lower(s);
            self.expr.lower(s);
            self.semi.lower(s);
        }
    }
    impl Clone for Const {
        fn clone(&self) -> Self {
            Const {
                attrs: self.attrs.clone(),
                vis: self.vis.clone(),
                default: self.default.clone(),
                const_: self.const_.clone(),
                ident: self.ident.clone(),
                gens: self.gens.clone(),
                colon: self.colon.clone(),
                typ: self.typ.clone(),
                eq: self.eq.clone(),
                expr: self.expr.clone(),
                semi: self.semi.clone(),
            }
        }
    }
    impl Eq for Const {}
    impl PartialEq for Const {
        fn eq(&self, x: &Self) -> bool {
            self.attrs == x.attrs
                && self.vis == x.vis
                && self.default == x.default
                && self.ident == x.ident
                && self.gens == x.gens
                && self.typ == x.typ
                && self.expr == x.expr
        }
    }
    impl<H> Hash for Const
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
    impl Pretty for Const {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            p.cbox(0);
            &self.vis.pretty(p);
            if self.default.is_some() {
                p.word("default ");
            }
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
    impl<V> Visit for Const
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            for x in &self.attrs {
                x.visit(v);
            }
            &self.vis.visit(v);
            &self.ident.visit(v);
            &self.gens.visit(v);
            &self.typ.visit(v);
            &self.expr.visit(v);
        }
        fn visit_mut(&mut self, v: &mut V) {
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

    pub struct Fn {
        pub attrs: Vec<attr::Attr>,
        pub vis: data::Visibility,
        pub default: Option<Token![default]>,
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
            self.default.lower(s);
            self.sig.lower(s);
            self.block.brace.surround(s, |s| {
                s.append_all(self.attrs.inners());
                s.append_all(&self.block.stmts);
            });
        }
    }
    impl Clone for Fn {
        fn clone(&self) -> Self {
            Fn {
                attrs: self.attrs.clone(),
                vis: self.vis.clone(),
                default: self.default.clone(),
                sig: self.sig.clone(),
                block: self.block.clone(),
            }
        }
    }
    impl Eq for Fn {}
    impl PartialEq for Fn {
        fn eq(&self, x: &Self) -> bool {
            self.attrs == x.attrs
                && self.vis == x.vis
                && self.default == x.default
                && self.sig == x.sig
                && self.block == x.block
        }
    }
    impl<H> Hash for Fn
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
    impl Pretty for Fn {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            p.cbox(INDENT);
            &self.vis.pretty(p);
            if self.default.is_some() {
                p.word("default ");
            }
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
    impl<V> Visit for Fn
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            for x in &self.attrs {
                x.visit(v);
            }
            &self.vis.visit(v);
            &self.sig.visit(v);
            &self.block.visit(v);
        }
        fn visit_mut(&mut self, v: &mut V) {
            for x in &mut self.attrs {
                x.visit_mut(v);
            }
            &mut self.vis.visit_mut(v);
            &mut self.sig.visit_mut(v);
            &mut self.block.visit_mut(v);
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
    impl Clone for Mac {
        fn clone(&self) -> Self {
            Mac {
                attrs: self.attrs.clone(),
                mac: self.mac.clone(),
                semi: self.semi.clone(),
            }
        }
    }
    impl Eq for Mac {}
    impl PartialEq for Mac {
        fn eq(&self, x: &Self) -> bool {
            self.attrs == x.attrs && self.mac == x.mac && self.semi == x.semi
        }
    }
    impl<H> Hash for Mac
    where
        H: Hasher,
    {
        fn hash(&self, h: &mut H) {
            self.attrs.hash(h);
            self.mac.hash(h);
            self.semi.hash(h);
        }
    }
    impl Pretty for Mac {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            let semi = true;
            &self.mac.pretty_with_args(p, (None, semi));
            p.hardbreak();
        }
    }
    impl<V> Visit for Mac
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            for x in &self.attrs {
                x.visit(v);
            }
            &self.mac.visit(v);
        }
        fn visit_mut(&mut self, v: &mut V) {
            for x in &mut self.attrs {
                x.visit_mut(v);
            }
            &mut self.mac.visit_mut(v);
        }
    }

    pub struct Type {
        pub attrs: Vec<attr::Attr>,
        pub vis: data::Visibility,
        pub default: Option<Token![default]>,
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
            let default: Option<Token![default]> = s.parse()?;
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
                default,
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
            self.default.lower(s);
            self.type_.lower(s);
            self.ident.lower(s);
            self.gens.lower(s);
            self.eq.lower(s);
            self.typ.lower(s);
            self.gens.where_.lower(s);
            self.semi.lower(s);
        }
    }
    impl Clone for Type {
        fn clone(&self) -> Self {
            Type {
                attrs: self.attrs.clone(),
                vis: self.vis.clone(),
                default: self.default.clone(),
                type_: self.type_.clone(),
                ident: self.ident.clone(),
                gens: self.gens.clone(),
                eq: self.eq.clone(),
                typ: self.typ.clone(),
                semi: self.semi.clone(),
            }
        }
    }
    impl Eq for Type {}
    impl PartialEq for Type {
        fn eq(&self, x: &Self) -> bool {
            self.attrs == x.attrs
                && self.vis == x.vis
                && self.default == x.default
                && self.ident == x.ident
                && self.gens == x.gens
                && self.typ == x.typ
        }
    }
    impl<H> Hash for Type
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
    impl Pretty for Type {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            p.cbox(INDENT);
            &self.vis.pretty(p);
            if self.default.is_some() {
                p.word("default ");
            }
            p.word("type ");
            &self.ident.pretty(p);
            &self.gens.pretty(p);
            p.word(" = ");
            p.neverbreak();
            p.ibox(-INDENT);
            &self.typ.pretty(p);
            p.end();
            p.where_oneline_with_semi(&self.gens.where_);
            p.end();
            p.hardbreak();
        }
    }
    impl<V> Visit for Type
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            for x in &self.attrs {
                x.visit(v);
            }
            &self.vis.visit(v);
            &self.ident.visit(v);
            &self.gens.visit(v);
            &self.typ.visit(v);
        }
        fn visit_mut(&mut self, v: &mut V) {
            for x in &mut self.attrs {
                x.visit_mut(v);
            }
            &mut self.vis.visit_mut(v);
            &mut self.ident.visit_mut(v);
            &mut self.gens.visit_mut(v);
            &mut self.typ.visit_mut(v);
        }
    }

    pub struct Verbatim(pub pm2::Stream);
    impl Pretty for Verbatim {
        fn pretty(&self, p: &mut Print) {
            enum Type {
                Ellipsis,
                Empty,
                FlexConst(flex::Const),
                FlexFn(flex::Fn),
                FlexType(flex::Type),
            }
            use Type::*;
            impl parse::Parse for Type {
                fn parse(s: parse::Stream) -> Res<Self> {
                    if s.is_empty() {
                        return Ok(Empty);
                    } else if s.peek(Token![...]) {
                        s.parse::<Token![...]>()?;
                        return Ok(Ellipsis);
                    }
                    let attrs = s.call(attr::Attr::parse_outer)?;
                    let vis: data::Visibility = s.parse()?;
                    let default = s.parse::<Option<Token![default]>>()?.is_some();
                    let look = s.lookahead1();
                    if look.peek(Token![const]) && (s.peek2(ident::Ident) || s.peek2(Token![_])) {
                        let y = flex::Const::parse(attrs, vis, default, s)?;
                        Ok(FlexConst(y))
                    } else if s.peek(Token![const])
                        || look.peek(Token![async])
                        || look.peek(Token![unsafe])
                        || look.peek(Token![extern])
                        || look.peek(Token![fn])
                    {
                        let y = flex::Fn::parse(attrs, vis, default, s)?;
                        Ok(FlexFn(y))
                    } else if look.peek(Token![type]) {
                        let y = flex::Type::parse(attrs, vis, default, s, WhereLoc::AfterEq)?;
                        Ok(FlexType(y))
                    } else {
                        Err(look.error())
                    }
                }
            }
            let y: Type = match parse2(self.clone()) {
                Ok(x) => x,
                Err(_) => unimplemented!("imp_::Item::Verbatim `{}`", self),
            };
            match y {
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
                FlexType(x) => {
                    &x.pretty(p);
                },
            }
        }
    }

    pub enum Restriction {}
    impl Clone for Restriction {
        fn clone(&self) -> Self {
            match *self {}
        }
    }
    impl Eq for Restriction {}
    impl PartialEq for Restriction {
        fn eq(&self, _other: &Self) -> bool {
            match *self {}
        }
    }
    impl<H> Hash for Restriction
    where
        H: Hasher,
    {
        fn hash(&self, _h: &mut H) {
            match *self {}
        }
    }
    impl<V> Visit for Restriction
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            match *self {}
        }
        fn visit_mut(&mut self, v: &mut V) {
            match *self {}
        }
    }
}
pub mod trait_ {
    use super::*;
    enum_of_structs! {
        pub enum Item {
            Const(Const),
            Fn(Fn),
            Mac(Mac),
            Type(Type),
            Verbatim(Verbatim),
        }
    }
    impl Parse for Item {
        fn parse(s: Stream) -> Res<Self> {
            let beg = s.fork();
            let mut attrs = s.call(attr::Attr::parse_outers)?;
            let vis: data::Visibility = s.parse()?;
            let default: Option<Token![default]> = s.parse()?;
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
                && default.is_none()
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
            match (vis, default) {
                (data::Visibility::Inherited, None) => {},
                _ => return Ok(Item::Verbatim(parse::parse_verbatim(&beg, s))),
            }
            let ys = match &mut y {
                Item::Const(item) => &mut item.attrs,
                Item::Fn(item) => &mut item.attrs,
                Item::Type(item) => &mut item.attrs,
                Item::Mac(item) => &mut item.attrs,
                Item::Verbatim(_) => unreachable!(),
            };
            attrs.append(ys);
            *ys = attrs;
            Ok(y)
        }
    }
    impl Clone for Item {
        fn clone(&self) -> Self {
            use self::Item::*;
            match self {
                Const(x) => Const(x.clone()),
                Fn(x) => Fn(x.clone()),
                Mac(x) => Mac(x.clone()),
                Type(x) => Type(x.clone()),
                Verbatim(x) => Verbatim(x.clone()),
            }
        }
    }
    impl Eq for Item {}
    impl PartialEq for Item {
        fn eq(&self, x: &Self) -> bool {
            use self::Item::*;
            match (self, x) {
                (Const(x), Const(y)) => x == y,
                (Fn(x), Fn(y)) => x == y,
                (Mac(x), Mac(y)) => x == y,
                (Type(x), Type(y)) => x == y,
                (Verbatim(x), Verbatim(y)) => StreamHelper(x) == StreamHelper(y),
                _ => false,
            }
        }
    }
    impl<H> Hash for Item
    where
        H: Hasher,
    {
        fn hash(&self, h: &mut H) {
            use self::Item::*;
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
    impl Pretty for Item {
        fn pretty(&self, p: &mut Print) {
            use self::Item::*;
            match self {
                Const(x) => x.pretty(p),
                Fn(x) => x.pretty(p),
                Mac(x) => x.pretty(p),
                Type(x) => x.pretty(p),
                Verbatim(x) => x.pretty(p),
            }
        }
    }
    impl<V> Visit for Item
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            use self::Item::*;
            match self {
                Const(x) => {
                    x.visit(v);
                },
                Fn(x) => {
                    x.visit(v);
                },
                Mac(x) => {
                    x.visit(v);
                },
                Type(x) => {
                    x.visit(v);
                },
                Verbatim(_) => {},
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            use self::Item::*;
            match self {
                Const(x) => {
                    x.visit_mut(v);
                },
                Fn(x) => {
                    x.visit_mut(v);
                },
                Mac(x) => {
                    x.visit_mut(v);
                },
                Type(x) => {
                    x.visit_mut(v);
                },
                Verbatim(_) => {},
            }
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
    impl Clone for Const {
        fn clone(&self) -> Self {
            Const {
                attrs: self.attrs.clone(),
                const_: self.const_.clone(),
                ident: self.ident.clone(),
                gens: self.gens.clone(),
                colon: self.colon.clone(),
                typ: self.typ.clone(),
                default: self.default.clone(),
                semi: self.semi.clone(),
            }
        }
    }
    impl Eq for Const {}
    impl PartialEq for Const {
        fn eq(&self, x: &Self) -> bool {
            self.attrs == x.attrs
                && self.ident == x.ident
                && self.gens == x.gens
                && self.typ == x.typ
                && self.default == x.default
        }
    }
    impl<H> Hash for Const
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
    impl Pretty for Const {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            p.cbox(0);
            p.word("const ");
            &self.ident.pretty(p);
            &self.gens.pretty(p);
            p.word(": ");
            &self.typ.pretty(p);
            if let Some((_, x)) = &self.default {
                p.word(" = ");
                p.neverbreak();
                x.pretty(p);
            }
            p.word(";");
            p.end();
            p.hardbreak();
        }
    }
    impl<V> Visit for Const
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
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
        fn visit_mut(&mut self, v: &mut V) {
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
    impl Clone for Fn {
        fn clone(&self) -> Self {
            Fn {
                attrs: self.attrs.clone(),
                sig: self.sig.clone(),
                default: self.default.clone(),
                semi: self.semi.clone(),
            }
        }
    }
    impl Eq for Fn {}
    impl PartialEq for Fn {
        fn eq(&self, x: &Self) -> bool {
            self.attrs == x.attrs && self.sig == x.sig && self.default == x.default && self.semi == x.semi
        }
    }
    impl<H> Hash for Fn
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
    impl Pretty for Fn {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            p.cbox(INDENT);
            &self.sig.pretty(p);
            if let Some(x) = &self.default {
                p.where_for_body(&self.sig.gens.where_);
                p.word("{");
                p.hardbreak_if_nonempty();
                p.inner_attrs(&self.attrs);
                for x in &x.stmts {
                    x.pretty(p);
                }
                p.offset(-INDENT);
                p.end();
                p.word("}");
            } else {
                p.where_with_semi(&self.sig.gens.where_);
                p.end();
            }
            p.hardbreak();
        }
    }
    impl<V> Visit for Fn
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            for x in &self.attrs {
                x.visit(v);
            }
            &self.sig.visit(v);
            if let Some(x) = &self.default {
                x.visit(v);
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            for x in &mut self.attrs {
                x.visit_mut(v);
            }
            &mut self.sig.visit_mut(v);
            if let Some(x) = &mut self.default {
                x.visit_mut(v);
            }
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
    impl Clone for Mac {
        fn clone(&self) -> Self {
            Mac {
                attrs: self.attrs.clone(),
                mac: self.mac.clone(),
                semi: self.semi.clone(),
            }
        }
    }
    impl Eq for Mac {}
    impl PartialEq for Mac {
        fn eq(&self, x: &Self) -> bool {
            self.attrs == x.attrs && self.mac == x.mac && self.semi == x.semi
        }
    }
    impl<H> Hash for Mac
    where
        H: Hasher,
    {
        fn hash(&self, h: &mut H) {
            self.attrs.hash(h);
            self.mac.hash(h);
            self.semi.hash(h);
        }
    }
    impl Pretty for Mac {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            let semi = true;
            &self.mac.pretty_with_args(p, (None, semi));
            p.hardbreak();
        }
    }
    impl<V> Visit for Mac
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            for x in &self.attrs {
                x.visit(v);
            }
            &self.mac.visit(v);
        }
        fn visit_mut(&mut self, v: &mut V) {
            for x in &mut self.attrs {
                x.visit_mut(v);
            }
            &mut self.mac.visit_mut(v);
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
    impl Clone for Type {
        fn clone(&self) -> Self {
            Type {
                attrs: self.attrs.clone(),
                type_: self.type_.clone(),
                ident: self.ident.clone(),
                gens: self.gens.clone(),
                colon: self.colon.clone(),
                bounds: self.bounds.clone(),
                default: self.default.clone(),
                semi: self.semi.clone(),
            }
        }
    }
    impl Eq for Type {}
    impl PartialEq for Type {
        fn eq(&self, x: &Self) -> bool {
            self.attrs == x.attrs
                && self.ident == x.ident
                && self.gens == x.gens
                && self.colon == x.colon
                && self.bounds == x.bounds
                && self.default == x.default
        }
    }
    impl<H> Hash for Type
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
    impl Pretty for Type {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            p.cbox(INDENT);
            p.word("type ");
            &self.ident.pretty(p);
            &self.gens.pretty(p);
            for x in self.bounds.iter().delimited() {
                if x.is_first {
                    p.word(": ");
                } else {
                    p.space();
                    p.word("+ ");
                }
                &x.pretty(p);
            }
            if let Some((_, x)) = &self.default {
                p.word(" = ");
                p.neverbreak();
                p.ibox(-INDENT);
                x.pretty(p);
                p.end();
            }
            p.where_oneline_with_semi(&self.gens.where_);
            p.end();
            p.hardbreak();
        }
    }
    impl<V> Visit for Type
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
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
        fn visit_mut(&mut self, v: &mut V) {
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

    pub struct Verbatim(pub pm2::Stream);
    impl Pretty for Verbatim {
        fn pretty(&self, p: &mut Print) {
            enum Type {
                Empty,
                Ellipsis,
                FlexType(flex::Type),
                PubOrDefault(PubOrDefault),
            }
            use Type::*;
            struct PubOrDefault {
                attrs: Vec<attr::Attr>,
                vis: data::Visibility,
                default: bool,
                trait_: Trait,
            }
            impl parse::Parse for Type {
                fn parse(s: parse::Stream) -> Res<Self> {
                    if s.is_empty() {
                        return Ok(Empty);
                    } else if s.peek(Token![...]) {
                        s.parse::<Token![...]>()?;
                        return Ok(Ellipsis);
                    }
                    let attrs = s.call(attr::Attr::parse_outer)?;
                    let vis: data::Visibility = s.parse()?;
                    let default = s.parse::<Option<Token![default]>>()?.is_some();
                    let look = s.lookahead1();
                    if look.peek(Token![type]) {
                        let y = flex::Type::parse(attrs, vis, default, s, WhereLoc::AfterEq)?;
                        Ok(FlexType(y))
                    } else if (look.peek(Token![const])
                        || look.peek(Token![async])
                        || look.peek(Token![unsafe])
                        || look.peek(Token![extern])
                        || look.peek(Token![fn]))
                        && (!matches!(vis, data::Visibility::Inherited) || default)
                    {
                        Ok(PubOrDefault(PubOrDefault {
                            attrs,
                            vis,
                            default,
                            trait_: s.parse()?,
                        }))
                    } else {
                        Err(look.error())
                    }
                }
            }
            let y: Type = match parse2(self.clone()) {
                Ok(x) => x,
                Err(_) => unimplemented!("trait_::Item::Verbatim `{}`", self),
            };
            match y {
                Empty => {
                    p.hardbreak();
                },
                Ellipsis => {
                    p.word("...");
                    p.hardbreak();
                },
                FlexType(x) => {
                    &x.pretty(p);
                },
                PubOrDefault(x) => {
                    p.outer_attrs(&x.attrs);
                    &x.vis.pretty(p);
                    if x.default {
                        p.word("default ");
                    }
                    &x.trait_.pretty(p);
                },
            }
        }
    }
}
pub mod use_ {
    use super::*;
    enum_of_structs! {
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
    impl Clone for Tree {
        fn clone(&self) -> Self {
            use Tree::*;
            match self {
                Glob(x) => Glob(x.clone()),
                Group(x) => Group(x.clone()),
                Name(x) => Name(x.clone()),
                Path(x) => Path(x.clone()),
                Rename(x) => Rename(x.clone()),
            }
        }
    }
    impl Eq for Tree {}
    impl PartialEq for Tree {
        fn eq(&self, x: &Self) -> bool {
            use Tree::*;
            match (self, x) {
                (Glob(x), Glob(y)) => x == y,
                (Group(x), Group(y)) => x == y,
                (Name(x), Name(y)) => x == y,
                (Path(x), Path(y)) => x == y,
                (Rename(x), Rename(y)) => x == y,
                _ => false,
            }
        }
    }
    impl<H> Hash for Tree
    where
        H: Hasher,
    {
        fn hash(&self, h: &mut H) {
            use Tree::*;
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
    impl<V> Visit for Tree
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            use Tree::*;
            match self {
                Glob(x) => {
                    x.visit(v);
                },
                Group(x) => {
                    x.visit(v);
                },
                Name(x) => {
                    x.visit(v);
                },
                Path(x) => {
                    x.visit(v);
                },
                Rename(x) => {
                    x.visit(v);
                },
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            use Tree::*;
            match self {
                Glob(x) => {
                    x.visit_mut(v);
                },
                Group(x) => {
                    x.visit_mut(v);
                },
                Name(x) => {
                    x.visit_mut(v);
                },
                Path(x) => {
                    x.visit_mut(v);
                },
                Rename(x) => {
                    x.visit_mut(v);
                },
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

    pub struct Glob {
        pub star: Token![*],
    }
    impl Lower for Glob {
        fn lower(&self, s: &mut Stream) {
            self.star.lower(s);
        }
    }
    impl Clone for Glob {
        fn clone(&self) -> Self {
            Glob {
                star: self.star.clone(),
            }
        }
    }
    impl Eq for Glob {}
    impl PartialEq for Glob {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl<H> Hash for Glob
    where
        H: Hasher,
    {
        fn hash(&self, h: &mut H) {}
    }
    impl Pretty for Glob {
        fn pretty(&self, p: &mut Print) {
            let _ = self;
            p.word("*");
        }
    }
    impl<V> Visit for Glob
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {}
        fn visit_mut(&mut self, v: &mut V) {}
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
    impl Clone for Group {
        fn clone(&self) -> Self {
            Group {
                brace: self.brace.clone(),
                trees: self.trees.clone(),
            }
        }
    }
    impl Eq for Group {}
    impl PartialEq for Group {
        fn eq(&self, x: &Self) -> bool {
            self.trees == x.trees
        }
    }
    impl<H> Hash for Group
    where
        H: Hasher,
    {
        fn hash(&self, h: &mut H) {
            self.trees.hash(h);
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
    impl<V> Visit for Group
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            for y in Puncted::pairs(&self.trees) {
                let x = y.value();
                x.visit(v);
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            for mut y in Puncted::pairs_mut(&mut self.trees) {
                let x = y.value_mut();
                x.visit_mut(v);
            }
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
    impl Clone for Name {
        fn clone(&self) -> Self {
            Name {
                ident: self.ident.clone(),
            }
        }
    }
    impl Eq for Name {}
    impl PartialEq for Name {
        fn eq(&self, x: &Self) -> bool {
            self.ident == x.ident
        }
    }
    impl<H> Hash for Name
    where
        H: Hasher,
    {
        fn hash(&self, h: &mut H) {
            self.ident.hash(h);
        }
    }
    impl Pretty for Name {
        fn pretty(&self, p: &mut Print) {
            &self.ident.pretty(p);
        }
    }
    impl<V> Visit for Name
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            &self.ident.visit(v);
        }
        fn visit_mut(&mut self, v: &mut V) {
            &mut self.ident.visit_mut(v);
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
    impl Clone for Path {
        fn clone(&self) -> Self {
            Path {
                ident: self.ident.clone(),
                colon2: self.colon2.clone(),
                tree: self.tree.clone(),
            }
        }
    }
    impl Eq for Path {}
    impl PartialEq for Path {
        fn eq(&self, x: &Self) -> bool {
            self.ident == x.ident && self.tree == x.tree
        }
    }
    impl<H> Hash for Path
    where
        H: Hasher,
    {
        fn hash(&self, h: &mut H) {
            self.ident.hash(h);
            self.tree.hash(h);
        }
    }
    impl Pretty for Path {
        fn pretty(&self, p: &mut Print) {
            &self.ident.pretty(p);
            p.word("::");
            &self.tree.pretty(p);
        }
    }
    impl<V> Visit for Path
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            &self.ident.visit(v);
            &*self.tree.visit(v);
        }
        fn visit_mut(&mut self, v: &mut V) {
            &mut self.ident.visit_mut(v);
            &mut *self.tree.visit_mut(v);
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
    impl Clone for Rename {
        fn clone(&self) -> Self {
            Rename {
                ident: self.ident.clone(),
                as_: self.as_.clone(),
                rename: self.rename.clone(),
            }
        }
    }
    impl Eq for Rename {}
    impl PartialEq for Rename {
        fn eq(&self, x: &Self) -> bool {
            self.ident == x.ident && self.rename == x.rename
        }
    }
    impl<H> Hash for Rename
    where
        H: Hasher,
    {
        fn hash(&self, h: &mut H) {
            self.ident.hash(h);
            self.rename.hash(h);
        }
    }
    impl Pretty for Rename {
        fn pretty(&self, p: &mut Print) {
            &self.ident.pretty(p);
            p.word(" as ");
            &self.rename.pretty(p);
        }
    }
    impl<V> Visit for Rename
    where
        V: Visitor + ?Sized,
    {
        fn visit(&self, v: &mut V) {
            &self.ident.visit(v);
            &self.rename.visit(v);
        }
        fn visit_mut(&mut self, v: &mut V) {
            &mut self.ident.visit_mut(v);
            &mut self.rename.visit_mut(v);
        }
    }
}

mod flex {
    use super::*;

    pub struct Const {
        pub attrs: Vec<attr::Attr>,
        pub vis: data::Visibility,
        pub default: bool,
        pub ident: Ident,
        pub typ: typ::Type,
    }
    impl Const {
        pub fn parse(attrs: Vec<attr::Attr>, vis: data::Visibility, default: bool, s: parse::Stream) -> Res<Self> {
            s.parse::<Token![const]>()?;
            let ident = s.call(Ident::parse_any)?;
            s.parse::<Token![:]>()?;
            let ty: Type = s.parse()?;
            s.parse::<Token![;]>()?;
            Ok(Const {
                attrs,
                vis,
                default,
                ident,
                typ: ty,
            })
        }
    }
    impl Pretty for Const {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            p.cbox(0);
            &self.vis.pretty(p);
            if self.default {
                p.word("default ");
            }
            p.word("const ");
            &self.ident.pretty(p);
            p.word(": ");
            &self.typ.pretty(p);
            p.word(";");
            p.end();
            p.hardbreak();
        }
    }

    pub struct Fn {
        pub attrs: Vec<attr::Attr>,
        pub vis: data::Visibility,
        pub default: bool,
        pub sig: Sig,
        pub stmts: Option<Vec<stmt::Stmt>>,
    }
    impl Fn {
        pub fn parse(mut attrs: Vec<attr::Attr>, vis: data::Visibility, default: bool, s: parse::Stream) -> Res<Self> {
            let sig: Sig = s.parse()?;
            let look = s.lookahead1();
            let body = if look.peek(Token![;]) {
                s.parse::<Token![;]>()?;
                None
            } else if look.peek(tok::Brace) {
                let y;
                braced!(y in s);
                attrs.extend(y.call(attr::Attr::parse_inner)?);
                Some(y.call(stmt::Block::parse_within)?)
            } else {
                return Err(look.error());
            };
            Ok(Fn {
                attrs,
                vis,
                default,
                sig,
                stmts: body,
            })
        }
    }
    impl Pretty for Fn {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            p.cbox(INDENT);
            &self.vis.pretty(p);
            if self.default {
                p.word("default ");
            }
            &self.sig.pretty(p);
            if let Some(xs) = &self.stmts {
                p.where_clause_for_body(&self.sig.gens.where_);
                p.word("{");
                p.hardbreak_if_nonempty();
                p.inner_attrs(&self.attrs);
                for x in xs {
                    x.pretty(p);
                }
                p.offset(-INDENT);
                p.end();
                p.word("}");
            } else {
                p.where_clause_semi(&self.sig.gens.where_);
                p.end();
            }
            p.hardbreak();
        }
    }

    pub struct Static {
        pub attrs: Vec<attr::Attr>,
        pub vis: data::Visibility,
        pub mut_: StaticMut,
        pub ident: Ident,
        pub typ: Option<typ::Type>,
        pub expr: Option<expr::Expr>,
    }
    impl Static {
        pub fn parse(attrs: Vec<attr::Attr>, vis: data::Visibility, s: parse::Stream) -> Res<Self> {
            s.parse::<Token![static]>()?;
            let mut_: StaticMut = s.parse()?;
            let ident = s.parse()?;
            let look = s.lookahead1();
            let has_type = look.peek(Token![:]);
            let has_expr = look.peek(Token![=]);
            if !has_type && !has_expr {
                return Err(look.error());
            }
            let typ: Option<Type> = if has_type {
                s.parse::<Token![:]>()?;
                s.parse().map(Some)?
            } else {
                None
            };
            let expr: Option<expr::Expr> = if s.parse::<Option<Token![=]>>()?.is_some() {
                s.parse().map(Some)?
            } else {
                None
            };
            s.parse::<Token![;]>()?;
            Ok(Static {
                attrs,
                vis,
                mut_,
                ident,
                typ,
                expr,
            })
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
            if let Some(x) = &self.typ {
                p.word(": ");
                x.pretty(p);
            }
            if let Some(x) = &self.expr {
                p.word(" = ");
                p.neverbreak();
                x.pretty(p);
            }
            p.word(";");
            p.end();
            p.hardbreak();
        }
    }

    pub struct Type {
        pub attrs: Vec<attr::Attr>,
        pub vis: data::Visibility,
        pub default: bool,
        pub ident: Ident,
        pub gens: gen::Gens,
        pub bounds: Vec<gen::bound::Type>,
        pub typ: Option<typ::Type>,
        pub where_: Option<gen::Where>,
    }
    impl Type {
        pub fn parse(
            attrs: Vec<attr::Attr>,
            vis: data::Visibility,
            default: bool,
            s: parse::Stream,
            loc: WhereLoc,
        ) -> Res<Self> {
            s.parse::<Token![type]>()?;
            let ident: Ident = s.parse()?;
            let mut gens: gen::Gens = s.parse()?;
            let mut bounds = Vec::new();
            if s.parse::<Option<Token![:]>>()?.is_some() {
                loop {
                    if s.peek(Token![where]) || s.peek(Token![=]) || s.peek(Token![;]) {
                        break;
                    }
                    bounds.push(s.parse::<gen::bound::Type>()?);
                    if s.peek(Token![where]) || s.peek(Token![=]) || s.peek(Token![;]) {
                        break;
                    }
                    s.parse::<Token![+]>()?;
                }
            }
            use WhereLoc::*;
            match loc {
                BeforeEq | Both => {
                    gens.where_ = s.parse()?;
                },
                AfterEq => {},
            }
            let typ = if s.parse::<Option<Token![=]>>()?.is_some() {
                Some(s.parse()?)
            } else {
                None
            };
            let where_ = match loc {
                AfterEq | Both if gens.where_.is_none() => s.parse()?,
                _ => None,
            };
            s.parse::<Token![;]>()?;
            Ok(Type {
                attrs,
                vis,
                default,
                ident,
                gens,
                bounds,
                typ,
                where_,
            })
        }
    }
    impl Pretty for Type {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            p.cbox(INDENT);
            &self.vis.pretty(p);
            if self.default {
                p.word("default ");
            }
            p.word("type ");
            &self.ident.pretty(p);
            &self.gens.pretty(p);
            for x in self.bounds.iter().delimited() {
                if x.is_first {
                    p.word(": ");
                } else {
                    p.space();
                    p.word("+ ");
                }
                &x.pretty(p);
            }
            if let Some(x) = &self.typ {
                p.where_clause_oneline(&self.gens.where_);
                p.word("= ");
                p.neverbreak();
                p.ibox(-INDENT);
                x.pretty(p);
                p.end();
                p.where_clause_oneline_semi(&self.where_);
            } else {
                p.where_clause_oneline_semi(&self.gens.where_);
            }
            p.end();
            p.hardbreak();
        }
    }
}

enum TypeDefault {
    Optional,
    Disallowed,
}
enum WhereLoc {
    AfterEq,
    BeforeEq,
    Both,
}

struct Flexible {
    vis: data::Visibility,
    default: Option<Token![default]>,
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
        let default: Option<Token![default]> = match allow_default {
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
            default,
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
    if look.peek(tok::Parenth) {
        let y;
        parenthed!(y in s);
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
fn parse_fn_arg_or_variadic(s: Stream, attrs: Vec<attr::Attr>, variadic: bool) -> Res<FnArgOrVari> {
    let ahead = s.fork();
    if let Ok(mut receiver) = ahead.parse::<Receiver>() {
        s.advance_to(&ahead);
        receiver.attrs = attrs;
        return Ok(FnArgOrVari::FnArg(FnArg::Receiver(receiver)));
    }
    if s.peek(ident::Ident) && s.peek2(Token![<]) {
        let span = s.fork().parse::<Ident>()?.span();
        return Ok(FnArgOrVari::FnArg(FnArg::Typed(pat::Type {
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
            return Ok(FnArgOrVari::Variadic(Variadic {
                attrs,
                pat: Some((pat, colon)),
                dots,
                comma: None,
            }));
        }
    }
    Ok(FnArgOrVari::FnArg(FnArg::Typed(pat::Type {
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
            FnArgOrVari::FnArg(x) => x,
            FnArgOrVari::Variadic(x) => {
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
        default: _,
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
        default: _,
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
        parse_rest_of_trait_alias(s, attrs, vis, trait_, ident, gens).map(Item::Alias)
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
) -> Res<Alias> {
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
    Ok(Alias {
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
        default: _,
        type_,
        ident,
        gens,
        colon,
        bounds,
        typ,
        semi,
    } = Flexible::parse(s, TypeDefault::Disallowed, WhereLoc::AfterEq)?;
    if vis.is_some() {
        Ok(trait_::Item::Verbatim(parse::parse_verbatim(&beg, s)))
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
    let default: Option<Token![default]> = s.parse()?;
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
            default,
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
    let default: Option<Token![default]> = s.parse()?;
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
        default,
        sig,
        block,
    }))
}
fn parse_impl_item_type(beg: parse::Buffer, s: Stream) -> Res<impl_::Item> {
    let Flexible {
        vis,
        default,
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
        _ => return Ok(impl_::Item::Verbatim(parse::parse_verbatim(&beg, s))),
    };
    Ok(impl_::Item::Type(impl_::Type {
        attrs: Vec::new(),
        vis,
        default,
        type_,
        ident,
        gens,
        eq,
        typ,
        semi,
    }))
}
