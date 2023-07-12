pub struct DeriveInput {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub ident: Ident,
    pub gens: gen::Gens,
    pub data: Data,
}
pub enum Data {
    Enum(Enum),
    Struct(Struct),
    Union(Union),
}
pub struct Enum {
    pub enum_: Token![enum],
    pub brace: tok::Brace,
    pub variants: Punctuated<Variant, Token![,]>,
}
pub struct Struct {
    pub struct_: Token![struct],
    pub fields: Fields,
    pub semi: Option<Token![;]>,
}
pub struct Union {
    pub union_: Token![union],
    pub named: Named,
}
pub struct Variant {
    pub attrs: Vec<attr::Attr>,
    pub ident: Ident,
    pub fields: Fields,
    pub discriminant: Option<(Token![=], Expr)>,
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
            Named(x) => x.field.iter(),
            Unnamed(x) => x.field.iter(),
            Unit => punct::empty_punctuated_iter(),
        }
    }
    pub fn iter_mut(&mut self) -> punct::IterMut<Field> {
        use Fields::*;
        match self {
            Named(x) => x.field.iter_mut(),
            Unnamed(x) => x.field.iter_mut(),
            Unit => punct::empty_punctuated_iter_mut(),
        }
    }
    pub fn len(&self) -> usize {
        use Fields::*;
        match self {
            Named(x) => x.field.len(),
            Unnamed(x) => x.field.len(),
            Unit => 0,
        }
    }
    pub fn is_empty(&self) -> bool {
        use Fields::*;
        match self {
            Named(x) => x.field.is_empty(),
            Unnamed(x) => x.field.is_empty(),
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
            Named(x) => x.field.into_iter(),
            Unnamed(x) => x.field.into_iter(),
            Unit => Punctuated::<Field, ()>::new().into_iter(),
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
    pub field: Punctuated<Field, Token![,]>,
}
pub struct Unnamed {
    pub paren: tok::Paren,
    pub field: Punctuated<Field, Token![,]>,
}
pub struct Field {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub mut_: Mut,
    pub ident: Option<Ident>,
    pub colon: Option<Token![:]>,
    pub typ: typ::Type,
}
pub enum Mut {
    None,
}

impl Parse for Variant {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = x.call(attr::Attr::parse_outer)?;
        let _visibility: Visibility = x.parse()?;
        let ident: Ident = x.parse()?;
        let fields = if x.peek(tok::Brace) {
            Fields::Named(x.parse()?)
        } else if x.peek(tok::Paren) {
            Fields::Unnamed(x.parse()?)
        } else {
            Fields::Unit
        };
        let discriminant = if x.peek(Token![=]) {
            let eq: Token![=] = x.parse()?;
            let discriminant: Expr = x.parse()?;
            Some((eq, discriminant))
        } else {
            None
        };
        Ok(Variant {
            attrs,
            ident,
            fields,
            discriminant,
        })
    }
}
impl Parse for Named {
    fn parse(x: Stream) -> Res<Self> {
        let y;
        Ok(Named {
            brace: braced!(y in x),
            field: y.parse_terminated(Field::parse_named, Token![,])?,
        })
    }
}
impl Parse for Unnamed {
    fn parse(x: Stream) -> Res<Self> {
        let y;
        Ok(Unnamed {
            paren: parenthesized!(y in x),
            field: y.parse_terminated(Field::parse_unnamed, Token![,])?,
        })
    }
}
impl Field {
    pub fn parse_named(x: Stream) -> Res<Self> {
        Ok(Field {
            attrs: x.call(attr::Attr::parse_outer)?,
            vis: x.parse()?,
            mut_: FieldMutability::None,
            ident: Some(if x.peek(Token![_]) {
                x.call(Ident::parse_any)
            } else {
                x.parse()
            }?),
            colon: Some(x.parse()?),
            typ: x.parse()?,
        })
    }
    pub fn parse_unnamed(x: Stream) -> Res<Self> {
        Ok(Field {
            attrs: x.call(attr::Attr::parse_outer)?,
            vis: x.parse()?,
            mut_: FieldMutability::None,
            ident: None,
            colon: None,
            typ: x.parse()?,
        })
    }
}

impl Parse for DeriveInput {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = x.call(attr::Attr::parse_outer)?;
        let vis = x.parse::<Visibility>()?;
        let look = x.lookahead1();
        if look.peek(Token![struct]) {
            let struct_ = x.parse::<Token![struct]>()?;
            let ident = x.parse::<Ident>()?;
            let gens = x.parse::<gen::Gens>()?;
            let (where_, fields, semi) = data_struct(x)?;
            Ok(DeriveInput {
                attrs,
                vis,
                ident,
                gens: gen::Gens { where_, ..gens },
                data: Data::Struct(Struct { struct_, fields, semi }),
            })
        } else if look.peek(Token![enum]) {
            let enum_ = x.parse::<Token![enum]>()?;
            let ident = x.parse::<Ident>()?;
            let gens = x.parse::<gen::Gens>()?;
            let (where_, brace, variants) = data_enum(x)?;
            Ok(DeriveInput {
                attrs,
                vis,
                ident,
                gens: gen::Gens { where_, ..gens },
                data: Data::Enum(Enum { enum_, brace, variants }),
            })
        } else if look.peek(Token![union]) {
            let union_ = x.parse::<Token![union]>()?;
            let ident = x.parse::<Ident>()?;
            let gens = x.parse::<gen::Gens>()?;
            let (where_, named) = data_union(x)?;
            Ok(DeriveInput {
                attrs,
                vis,
                ident,
                gens: gen::Gens { where_, ..gens },
                data: Data::Union(Union { union_, named }),
            })
        } else {
            Err(look.error())
        }
    }
}
pub fn data_struct(x: Stream) -> Res<(Option<gen::Where>, Fields, Option<Token![;]>)> {
    let mut look = x.lookahead1();
    let mut where_ = None;
    if look.peek(Token![where]) {
        where_ = Some(x.parse()?);
        look = x.lookahead1();
    }
    if where_.is_none() && look.peek(tok::Paren) {
        let y = x.parse()?;
        look = x.lookahead1();
        if look.peek(Token![where]) {
            where_ = Some(x.parse()?);
            look = x.lookahead1();
        }
        if look.peek(Token![;]) {
            let semi = x.parse()?;
            Ok((where_, Fields::Unnamed(y), Some(semi)))
        } else {
            Err(look.error())
        }
    } else if look.peek(tok::Brace) {
        let y = x.parse()?;
        Ok((where_, Fields::Named(y), None))
    } else if look.peek(Token![;]) {
        let semi = x.parse()?;
        Ok((where_, Fields::Unit, Some(semi)))
    } else {
        Err(look.error())
    }
}
pub fn data_enum(x: Stream) -> Res<(Option<gen::Where>, tok::Brace, Punctuated<Variant, Token![,]>)> {
    let where_ = x.parse()?;
    let y;
    let brace = braced!(y in x);
    let variants = y.parse_terminated(Variant::parse, Token![,])?;
    Ok((where_, brace, variants))
}
pub fn data_union(x: Stream) -> Res<(Option<gen::Where>, Named)> {
    let where_ = x.parse()?;
    let named = x.parse()?;
    Ok((where_, named))
}

impl ToTokens for Variant {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(&self.attrs);
        self.ident.to_tokens(ys);
        self.fields.to_tokens(ys);
        if let Some((eq, disc)) = &self.discriminant {
            eq.to_tokens(ys);
            disc.to_tokens(ys);
        }
    }
}
impl ToTokens for Named {
    fn to_tokens(&self, ys: &mut Stream) {
        self.brace.surround(ys, |ys| {
            self.field.to_tokens(ys);
        });
    }
}
impl ToTokens for Unnamed {
    fn to_tokens(&self, ys: &mut Stream) {
        self.paren.surround(ys, |ys| {
            self.field.to_tokens(ys);
        });
    }
}
impl ToTokens for Field {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(&self.attrs);
        self.vis.to_tokens(ys);
        if let Some(x) = &self.ident {
            x.to_tokens(ys);
            TokensOrDefault(&self.colon).to_tokens(ys);
        }
        self.typ.to_tokens(ys);
    }
}

impl ToTokens for DeriveInput {
    fn to_tokens(&self, ys: &mut Stream) {
        for x in self.attrs.outer() {
            x.to_tokens(ys);
        }
        self.vis.to_tokens(ys);
        match &self.data {
            Data::Struct(x) => x.struct_.to_tokens(ys),
            Data::Enum(x) => x.enum_.to_tokens(ys),
            Data::Union(x) => x.union_.to_tokens(ys),
        }
        self.ident.to_tokens(ys);
        self.gens.to_tokens(ys);
        match &self.data {
            Data::Struct(data) => match &data.fields {
                Fields::Named(x) => {
                    self.gens.where_.to_tokens(ys);
                    x.to_tokens(ys);
                },
                Fields::Unnamed(x) => {
                    x.to_tokens(ys);
                    self.gens.where_.to_tokens(ys);
                    TokensOrDefault(&data.semi).to_tokens(ys);
                },
                Fields::Unit => {
                    self.gens.where_.to_tokens(ys);
                    TokensOrDefault(&data.semi).to_tokens(ys);
                },
            },
            Data::Enum(x) => {
                self.gens.where_.to_tokens(ys);
                x.brace.surround(ys, |ys| {
                    x.variants.to_tokens(ys);
                });
            },
            Data::Union(x) => {
                self.gens.where_.to_tokens(ys);
                x.named.to_tokens(ys);
            },
        }
    }
}
