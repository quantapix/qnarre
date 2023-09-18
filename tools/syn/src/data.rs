use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Input {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub ident: Ident,
    pub gens: gen::Gens,
    pub data: Data,
}
impl Parse for Input {
    fn parse(s: Stream) -> Res<Self> {
        let attrs = s.call(attr::Attr::parse_outers)?;
        let vis = s.parse::<Visibility>()?;
        let look = s.look1();
        if look.peek(Token![struct]) {
            let struct_ = s.parse::<Token![struct]>()?;
            let ident = s.parse::<Ident>()?;
            let gens = s.parse::<gen::Gens>()?;
            let (where_, fields, semi) = parse_struct(s)?;
            Ok(Input {
                attrs,
                vis,
                ident,
                gens: gen::Gens { where_, ..gens },
                data: Data::Struct(Struct { struct_, fields, semi }),
            })
        } else if look.peek(Token![enum]) {
            let enum_ = s.parse::<Token![enum]>()?;
            let ident = s.parse::<Ident>()?;
            let gens = s.parse::<gen::Gens>()?;
            let (where_, brace, variants) = parse_enum(s)?;
            Ok(Input {
                attrs,
                vis,
                ident,
                gens: gen::Gens { where_, ..gens },
                data: Data::Enum(Enum { enum_, brace, variants }),
            })
        } else if look.peek(Token![union]) {
            let union_ = s.parse::<Token![union]>()?;
            let ident = s.parse::<Ident>()?;
            let gens = s.parse::<gen::Gens>()?;
            let (where_, fields) = parse_union(s)?;
            Ok(Input {
                attrs,
                vis,
                ident,
                gens: gen::Gens { where_, ..gens },
                data: Data::Union(Union { union_, fields }),
            })
        } else {
            Err(look.error())
        }
    }
}
impl Lower for Input {
    fn lower(&self, s: &mut Stream) {
        for x in self.attrs.outers() {
            x.lower(s);
        }
        self.vis.lower(s);
        match &self.data {
            Data::Struct(x) => x.struct_.lower(s),
            Data::Enum(x) => x.enum_.lower(s),
            Data::Union(x) => x.union_.lower(s),
        }
        self.ident.lower(s);
        self.gens.lower(s);
        match &self.data {
            Data::Struct(data) => match &data.fields {
                Fields::Named(x) => {
                    self.gens.where_.lower(s);
                    x.lower(s);
                },
                Fields::Unnamed(x) => {
                    x.lower(s);
                    self.gens.where_.lower(s);
                    ToksOrDefault(&data.semi).lower(s);
                },
                Fields::Unit => {
                    self.gens.where_.lower(s);
                    ToksOrDefault(&data.semi).lower(s);
                },
            },
            Data::Enum(x) => {
                self.gens.where_.lower(s);
                x.brace.surround(s, |s| {
                    x.variants.lower(s);
                });
            },
            Data::Union(x) => {
                self.gens.where_.lower(s);
                x.fields.lower(s);
            },
        }
    }
}
impl<F: Folder + ?Sized> Fold for Input {
    fn fold(&self, f: &mut F) {
        Input {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            ident: self.ident.fold(f),
            gens: self.gens.fold(f),
            data: self.data.fold(f),
        }
    }
}
impl<H: Hasher> Hash for Input {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.ident.hash(h);
        self.gens.hash(h);
        self.data.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Input {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.vis.visit(v);
        &self.ident.visit(v);
        &self.gens.visit(v);
        &self.data.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut self.vis.visit_mut(v);
        &mut self.ident.visit_mut(v);
        &mut self.gens.visit_mut(v);
        &mut self.data.visit_mut(v);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Visibility {
    Public(Token![pub]),
    Restricted(Restricted),
    Inherited,
}
impl Visibility {
    pub fn is_some(&self) -> bool {
        match self {
            Visibility::Inherited => false,
            _ => true,
        }
    }
    pub fn is_inherited(&self) -> bool {
        match self {
            Visibility::Inherited => true,
            _ => false,
        }
    }
    fn parse_pub(s: Stream) -> Res<Self> {
        let pub_ = s.parse::<Token![pub]>()?;
        if s.peek(tok::Parenth) {
            let ahead = s.fork();
            let y;
            let parenth = parenthed!(y in ahead);
            if y.peek(Token![crate]) || y.peek(Token![self]) || y.peek(Token![super]) {
                let path = y.call(Ident::parse_any)?;
                if y.is_empty() {
                    s.advance_to(&ahead);
                    return Ok(Visibility::Restricted(Restricted {
                        pub_,
                        parenth,
                        in_: None,
                        path: Box::new(Path::from(path)),
                    }));
                }
            } else if y.peek(Token![in]) {
                let in_: Token![in] = y.parse()?;
                let path = y.call(Path::parse_mod_style)?;
                s.advance_to(&ahead);
                return Ok(Visibility::Restricted(Restricted {
                    pub_,
                    parenth,
                    in_: Some(in_),
                    path: Box::new(path),
                }));
            }
        }
        Ok(Visibility::Public(pub_))
    }
}
impl Parse for Visibility {
    fn parse(s: Stream) -> Res<Self> {
        if s.peek(tok::Group) {
            let ahead = s.fork();
            let y = parse::parse_group(&ahead)?;
            if y.buf.is_empty() {
                s.advance_to(&ahead);
                return Ok(Visibility::Inherited);
            }
        }
        if s.peek(Token![pub]) {
            Self::parse_pub(s)
        } else {
            Ok(Visibility::Inherited)
        }
    }
}
impl Lower for Visibility {
    fn lower(&self, s: &mut Stream) {
        match self {
            Visibility::Public(x) => x.lower(s),
            Visibility::Restricted(x) => x.lower(s),
            Visibility::Inherited => {},
        }
    }
}
impl Pretty for Visibility {
    fn pretty(&self, p: &mut Print) {
        use Visibility::*;
        match self {
            Public(_) => p.word("pub "),
            Restricted(x) => x.pretty(p),
            Inherited => {},
        }
    }
}
impl<F: Folder + ?Sized> Fold for Visibility {
    fn fold(&self, f: &mut F) {
        use Visibility::*;
        match self {
            Public(x) => Public(x),
            Restricted(x) => Restricted(x.fold(f)),
            Inherited => Inherited,
        }
    }
}
impl<H: Hasher> Hash for Visibility {
    fn hash(&self, h: &mut H) {
        use Visibility::*;
        match self {
            Public(_) => {
                h.write_u8(0u8);
            },
            Restricted(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Inherited => {
                h.write_u8(2u8);
            },
        }
    }
}
impl<V: Visitor + ?Sized> Visit for Visibility {
    fn visit(&self, v: &mut V) {
        use Visibility::*;
        match self {
            Inherited => {},
            Public(_) => {},
            Restricted(x) => {
                x.visit(v);
            },
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        use Visibility::*;
        match self {
            Inherited => {},
            Public(_) => {},
            Restricted(x) => {
                x.visit_mut(v);
            },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Restricted {
    pub pub_: Token![pub],
    pub parenth: tok::Parenth,
    pub in_: Option<Token![in]>,
    pub path: Box<Path>,
}
impl Lower for Restricted {
    fn lower(&self, s: &mut Stream) {
        self.pub_.lower(s);
        self.parenth.surround(s, |s| {
            self.in_.lower(s);
            self.path.lower(s);
        });
    }
}
impl Pretty for Restricted {
    fn pretty(&self, p: &mut Print) {
        p.word("pub(");
        let omit_in = self
            .path
            .get_ident()
            .map_or(false, |x| matches!(x.to_string().as_str(), "self" | "super" | "crate"));
        if !omit_in {
            p.word("in ");
        }
        p.path(&self.path, path::Kind::Simple);
        p.word(") ");
    }
}
impl<F: Folder + ?Sized> Fold for Restricted {
    fn fold(&self, f: &mut F) {
        Restricted {
            pub_: self.pub_,
            parenth: self.parenth,
            in_: self.in_,
            path: Box::new(*self.path.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Restricted {
    fn hash(&self, h: &mut H) {
        self.in_.hash(h);
        self.path.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Restricted {
    fn visit(&self, v: &mut V) {
        &*self.path.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut *self.path.visit_mut(v);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Data {
    Enum(Enum),
    Struct(Struct),
    Union(Union),
}
impl<F: Folder + ?Sized> Fold for Data {
    fn fold(&self, f: &mut F) {
        use Data::*;
        match self {
            Enum(x) => Enum(x.fold(f)),
            Struct(x) => Struct(x.fold(f)),
            Union(x) => Union(x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Data {
    fn hash(&self, h: &mut H) {
        use Data::*;
        match self {
            Enum(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Struct(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Union(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
        }
    }
}
impl<V: Visitor + ?Sized> Visit for Data {
    fn visit(&self, v: &mut V) {
        use Data::*;
        match self {
            Enum(x) => {
                x.visit(v);
            },
            Struct(x) => {
                x.visit(v);
            },
            Union(x) => {
                x.visit(v);
            },
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        use Data::*;
        match self {
            Enum(x) => {
                x.visit_mut(v);
            },
            Struct(x) => {
                x.visit_mut(v);
            },
            Union(x) => {
                x.visit_mut(v);
            },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Enum {
    pub enum_: Token![enum],
    pub brace: tok::Brace,
    pub variants: Puncted<Variant, Token![,]>,
}
impl<F: Folder + ?Sized> Fold for Enum {
    fn fold(&self, f: &mut F) {
        Enum {
            enum_: self.enum_,
            brace: self.brace,
            variants: FoldHelper::lift(self.variants, |x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Enum {
    fn hash(&self, h: &mut H) {
        self.variants.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Enum {
    fn visit(&self, v: &mut V) {
        for y in Puncted::pairs(&self.variants) {
            let x = y.value();
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for mut y in Puncted::pairs_mut(&mut self.variants) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Struct {
    pub struct_: Token![struct],
    pub fields: Fields,
    pub semi: Option<Token![;]>,
}
impl<F: Folder + ?Sized> Fold for Struct {
    fn fold(&self, f: &mut F) {
        Struct {
            struct_: self.struct_,
            fields: self.fields.fold(f),
            semi: self.semi,
        }
    }
}
impl<H: Hasher> Hash for Struct {
    fn hash(&self, h: &mut H) {
        self.fields.hash(h);
        self.semi.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Struct {
    fn visit(&self, v: &mut V) {
        &self.fields.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut self.fields.visit_mut(v);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Union {
    pub union_: Token![union],
    pub fields: Named,
}
impl<F: Folder + ?Sized> Fold for Union {
    fn fold(&self, f: &mut F) {
        Union {
            union_: self.union_,
            fields: self.fields.fold(f),
        }
    }
}
impl<H: Hasher> Hash for Union {
    fn hash(&self, h: &mut H) {
        self.fields.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Union {
    fn visit(&self, v: &mut V) {
        &self.fields.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut self.fields.visit_mut(v);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Variant {
    pub attrs: Vec<attr::Attr>,
    pub ident: Ident,
    pub fields: Fields,
    pub discrim: Option<(Token![=], expr::Expr)>,
}
impl Parse for Variant {
    fn parse(s: Stream) -> Res<Self> {
        let attrs = s.call(attr::Attr::parse_outers)?;
        let _: Visibility = s.parse()?;
        let ident: Ident = s.parse()?;
        let fields = if s.peek(tok::Brace) {
            Fields::Named(s.parse()?)
        } else if s.peek(tok::Parenth) {
            Fields::Unnamed(s.parse()?)
        } else {
            Fields::Unit
        };
        let discrim = if s.peek(Token![=]) {
            let eq: Token![=] = s.parse()?;
            let y: expr::Expr = s.parse()?;
            Some((eq, y))
        } else {
            None
        };
        Ok(Variant {
            attrs,
            ident,
            fields,
            discrim,
        })
    }
}
impl Lower for Variant {
    fn lower(&self, s: &mut Stream) {
        s.append_all(&self.attrs);
        self.ident.lower(s);
        self.fields.lower(s);
        if let Some((eq, disc)) = &self.discrim {
            eq.lower(s);
            disc.lower(s);
        }
    }
}
impl Pretty for Variant {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.ident(&self.ident);
        use Fields::*;
        match &self.fields {
            Named(xs) => {
                p.nbsp();
                p.word("{");
                p.cbox(INDENT);
                p.space();
                for x in xs.fields.iter().delimited() {
                    p.field(&x);
                    p.trailing_comma_or_space(x.is_last);
                }
                p.offset(-INDENT);
                p.end();
                p.word("}");
            },
            Unnamed(xs) => {
                p.cbox(INDENT);
                p.fields_unnamed(xs);
                p.end();
            },
            Unit => {},
        }
        if let Some((_, x)) = &self.discrim {
            p.word(" = ");
            p.expr(x);
        }
    }
}
impl<F: Folder + ?Sized> Fold for Variant {
    fn fold(&self, f: &mut F) {
        Variant {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            ident: self.ident.fold(f),
            fields: self.fields.fold(f),
            discrim: (self.discrim).map(|x| ((x).0, (x).1.fold(f))),
        }
    }
}
impl<H: Hasher> Hash for Variant {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ident.hash(h);
        self.fields.hash(h);
        self.discrim.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Variant {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.ident.visit(v);
        &self.fields.visit(v);
        if let Some(x) = &self.discrim {
            &(x).1.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut self.ident.visit_mut(v);
        &mut self.fields.visit_mut(v);
        if let Some(x) = &mut self.discrim {
            &mut (x).1.visit_mut(v);
        }
    }
}

enum_of_structs! {
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub enum Fields {
        Named(Named),
        Unnamed(Unnamed),
        Unit,
    }
}
impl Fields {
    pub fn iter(&self) -> punct::Iter<Field> {
        use Fields::*;
        match self {
            Named(x) => x.fields.iter(),
            Unnamed(x) => x.fields.iter(),
            Unit => punct::empty_punctuated_iter(),
        }
    }
    pub fn iter_mut(&mut self) -> punct::IterMut<Field> {
        use Fields::*;
        match self {
            Named(x) => x.fields.iter_mut(),
            Unnamed(x) => x.fields.iter_mut(),
            Unit => punct::empty_punctuated_iter_mut(),
        }
    }
    pub fn len(&self) -> usize {
        use Fields::*;
        match self {
            Named(x) => x.fields.len(),
            Unnamed(x) => x.fields.len(),
            Unit => 0,
        }
    }
    pub fn is_empty(&self) -> bool {
        use Fields::*;
        match self {
            Named(x) => x.fields.is_empty(),
            Unnamed(x) => x.fields.is_empty(),
            Unit => true,
        }
    }
}
impl IntoIterator for Fields {
    type Item = Field;
    type IntoIter = punct::IntoIter<Field>;
    fn into_iter(self) -> Self::IntoIter {
        use Fields::*;
        match self {
            Named(x) => x.fields.into_iter(),
            Unnamed(x) => x.fields.into_iter(),
            Unit => Puncted::<Field, ()>::new().into_iter(),
        }
    }
}
impl<'a> IntoIterator for &'a Fields {
    type Item = &'a Field;
    type IntoIter = punct::Iter<'a, Field>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a> IntoIterator for &'a mut Fields {
    type Item = &'a mut Field;
    type IntoIter = punct::IterMut<'a, Field>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
impl<F: Folder + ?Sized> Fold for Fields {
    fn fold(&self, f: &mut F) {
        use Fields::*;
        match self {
            Named(x) => Named(x.fold(f)),
            Unnamed(x) => Unnamed(x.fold(f)),
            Unit => Unit,
        }
    }
}
impl<H: Hasher> Hash for Fields {
    fn hash(&self, h: &mut H) {
        use Fields::*;
        match self {
            Named(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Unnamed(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Unit => {
                h.write_u8(2u8);
            },
        }
    }
}
impl<V: Visitor + ?Sized> Visit for Fields {
    fn visit(&self, v: &mut V) {
        use Fields::*;
        match self {
            Named(x) => {
                x.visit(v);
            },
            Unnamed(x) => {
                x.visit(v);
            },
            Unit => {},
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        use Fields::*;
        match self {
            Named(x) => {
                x.visit_mut(v);
            },
            Unnamed(x) => {
                x.visit_mut(v);
            },
            Unit => {},
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Named {
    pub brace: tok::Brace,
    pub fields: Puncted<Field, Token![,]>,
}
impl Parse for Named {
    fn parse(s: Stream) -> Res<Self> {
        let y;
        Ok(Named {
            brace: braced!(y in s),
            fields: y.parse_terminated(Field::parse_named, Token![,])?,
        })
    }
}
impl Lower for Named {
    fn lower(&self, s: &mut Stream) {
        self.brace.surround(s, |s| {
            self.fields.lower(s);
        });
    }
}
impl<F: Folder + ?Sized> Fold for Named {
    fn fold(&self, f: &mut F) {
        Named {
            brace: self.brace,
            fields: FoldHelper::lift(self.fields, |x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Named {
    fn hash(&self, h: &mut H) {
        self.fields.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Named {
    fn visit(&self, v: &mut V) {
        for y in Puncted::pairs(&self.fields) {
            let x = y.value();
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for mut y in Puncted::pairs_mut(&mut self.fields) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Unnamed {
    pub parenth: tok::Parenth,
    pub fields: Puncted<Field, Token![,]>,
}
impl Parse for Unnamed {
    fn parse(s: Stream) -> Res<Self> {
        let y;
        Ok(Unnamed {
            parenth: parenthed!(y in s),
            fields: y.parse_terminated(Field::parse_unnamed, Token![,])?,
        })
    }
}
impl Lower for Unnamed {
    fn lower(&self, s: &mut Stream) {
        self.parenth.surround(s, |s| {
            self.fields.lower(s);
        });
    }
}
impl Pretty for Unnamed {
    fn pretty(&self, p: &mut Print) {
        p.word("(");
        p.zerobreak();
        for x in self.fields.iter().delimited() {
            p.field(&x);
            p.trailing_comma(x.is_last);
        }
        p.offset(-INDENT);
        p.word(")");
    }
}
impl<F: Folder + ?Sized> Fold for Unnamed {
    fn fold(&self, f: &mut F) {
        Unnamed {
            parenth: self.parenth,
            fields: FoldHelper::lift(self.fields, |x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Unnamed {
    fn hash(&self, h: &mut H) {
        self.fields.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Unnamed {
    fn visit(&self, v: &mut V) {
        for y in Puncted::pairs(&self.fields) {
            let x = y.value();
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for mut y in Puncted::pairs_mut(&mut self.fields) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Field {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub mut_: Mut,
    pub ident: Option<Ident>,
    pub colon: Option<Token![:]>,
    pub typ: typ::Type,
}
impl Field {
    pub fn parse_named(s: Stream) -> Res<Self> {
        Ok(Field {
            attrs: s.call(attr::Attr::parse_outers)?,
            vis: s.parse()?,
            mut_: Mut::None,
            ident: Some(if s.peek(Token![_]) {
                s.call(Ident::parse_any)
            } else {
                s.parse()
            }?),
            colon: Some(s.parse()?),
            typ: s.parse()?,
        })
    }
    pub fn parse_unnamed(s: Stream) -> Res<Self> {
        Ok(Field {
            attrs: s.call(attr::Attr::parse_outers)?,
            vis: s.parse()?,
            mut_: Mut::None,
            ident: None,
            colon: None,
            typ: s.parse()?,
        })
    }
}
impl Lower for Field {
    fn lower(&self, s: &mut Stream) {
        s.append_all(&self.attrs);
        self.vis.lower(s);
        if let Some(x) = &self.ident {
            x.lower(s);
            ToksOrDefault(&self.colon).lower(s);
        }
        self.typ.lower(s);
    }
}
impl Pretty for Field {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.visibility(&self.vis);
        if let Some(x) = &self.ident {
            p.ident(x);
            p.word(": ");
        }
        p.ty(&self.typ);
    }
}
impl<F: Folder + ?Sized> Fold for Field {
    fn fold(&self, f: &mut F) {
        Field {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vis: self.vis.fold(f),
            mut_: self.mut_.fold(f),
            ident: (self.ident).map(|x| x.fold(f)),
            colon: self.colon,
            typ: self.typ.fold(f),
        }
    }
}
impl<H: Hasher> Hash for Field {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vis.hash(h);
        self.mut_.hash(h);
        self.ident.hash(h);
        self.colon.hash(h);
        self.typ.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Field {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.vis.visit(v);
        &self.mut_.visit(v);
        if let Some(x) = &self.ident {
            x.visit(v);
        }
        &self.typ.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut self.vis.visit_mut(v);
        &mut self.mut_.visit_mut(v);
        if let Some(x) = &mut self.ident {
            x.visit_mut(v);
        }
        &mut self.typ.visit_mut(v);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Mut {
    None,
}
impl<F: Folder + ?Sized> Fold for Mut {
    fn fold(&self, f: &mut F) {
        use Mut::*;
        match self {
            None => None,
        }
    }
}
impl<H: Hasher> Hash for Mut {
    fn hash(&self, h: &mut H) {
        use Mut::*;
        match self {
            None => {
                h.write_u8(0u8);
            },
        }
    }
}
impl<V: Visitor + ?Sized> Visit for Mut {
    fn visit(&self, v: &mut V) {
        use Mut::*;
        match self {
            None => {},
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        use Mut::*;
        match self {
            None => {},
        }
    }
}

pub fn parse_struct(s: Stream) -> Res<(Option<gen::Where>, Fields, Option<Token![;]>)> {
    let mut look = s.look1();
    let mut where_ = None;
    if look.peek(Token![where]) {
        where_ = Some(s.parse()?);
        look = s.look1();
    }
    if where_.is_none() && look.peek(tok::Parenth) {
        let y = s.parse()?;
        look = s.look1();
        if look.peek(Token![where]) {
            where_ = Some(s.parse()?);
            look = s.look1();
        }
        if look.peek(Token![;]) {
            let semi = s.parse()?;
            Ok((where_, Fields::Unnamed(y), Some(semi)))
        } else {
            Err(look.error())
        }
    } else if look.peek(tok::Brace) {
        let y = s.parse()?;
        Ok((where_, Fields::Named(y), None))
    } else if look.peek(Token![;]) {
        let semi = s.parse()?;
        Ok((where_, Fields::Unit, Some(semi)))
    } else {
        Err(look.error())
    }
}
pub fn parse_enum(s: Stream) -> Res<(Option<gen::Where>, tok::Brace, Puncted<Variant, Token![,]>)> {
    let where_ = s.parse()?;
    let y;
    let brace = braced!(y in s);
    let variants = y.parse_terminated(Variant::parse, Token![,])?;
    Ok((where_, brace, variants))
}
pub fn parse_union(s: Stream) -> Res<(Option<gen::Where>, Named)> {
    let where_ = s.parse()?;
    let fields = s.parse()?;
    Ok((where_, fields))
}
