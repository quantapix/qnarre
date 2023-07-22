use super::*;

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
    fn parse_pub(x: Stream) -> Res<Self> {
        let pub_ = x.parse::<Token![pub]>()?;
        if x.peek(tok::Paren) {
            let ahead = x.fork();
            let y;
            let paren = parenthesized!(y in ahead);
            if y.peek(Token![crate]) || y.peek(Token![self]) || y.peek(Token![super]) {
                let path = y.call(Ident::parse_any)?;
                if y.is_empty() {
                    x.advance_to(&ahead);
                    return Ok(Visibility::Restricted(Restricted {
                        pub_,
                        paren,
                        in_: None,
                        path: Box::new(Path::from(path)),
                    }));
                }
            } else if y.peek(Token![in]) {
                let in_: Token![in] = y.parse()?;
                let path = y.call(Path::parse_mod_style)?;
                x.advance_to(&ahead);
                return Ok(Visibility::Restricted(Restricted {
                    pub_,
                    paren,
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

pub struct Restricted {
    pub pub_: Token![pub],
    pub paren: tok::Paren,
    pub in_: Option<Token![in]>,
    pub path: Box<Path>,
}
impl Lower for Restricted {
    fn lower(&self, s: &mut Stream) {
        self.pub_.lower(s);
        self.paren.surround(s, |s| {
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

pub enum Data {
    Enum(Enum),
    Struct(Struct),
    Union(Union),
}

pub struct Enum {
    pub enum_: Token![enum],
    pub brace: tok::Brace,
    pub variants: Puncted<Variant, Token![,]>,
}
pub struct Struct {
    pub struct_: Token![struct],
    pub fields: Fields,
    pub semi: Option<Token![;]>,
}
pub struct Union {
    pub union_: Token![union],
    pub fields: Named,
}

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
        } else if s.peek(tok::Paren) {
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
                for x in xs.named.iter().delimited() {
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

ast_enum_of_structs! {
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

pub struct Unnamed {
    pub paren: tok::Paren,
    pub fields: Puncted<Field, Token![,]>,
}
impl Parse for Unnamed {
    fn parse(s: Stream) -> Res<Self> {
        let y;
        Ok(Unnamed {
            paren: parenthesized!(y in s),
            fields: y.parse_terminated(Field::parse_unnamed, Token![,])?,
        })
    }
}
impl Lower for Unnamed {
    fn lower(&self, s: &mut Stream) {
        self.paren.surround(s, |s| {
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

pub enum Mut {
    None,
}

pub fn parse_struct(s: Stream) -> Res<(Option<gen::Where>, Fields, Option<Token![;]>)> {
    let mut look = s.look1();
    let mut where_ = None;
    if look.peek(Token![where]) {
        where_ = Some(s.parse()?);
        look = s.look1();
    }
    if where_.is_none() && look.peek(tok::Paren) {
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
