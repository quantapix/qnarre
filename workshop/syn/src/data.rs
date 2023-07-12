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
    fn parse(input: Stream) -> Res<Self> {
        let attrs = input.call(attr::Attr::parse_outer)?;
        let _visibility: Visibility = input.parse()?;
        let ident: Ident = input.parse()?;
        let fields = if input.peek(tok::Brace) {
            data::Fields::Named(input.parse()?)
        } else if input.peek(tok::Paren) {
            data::Fields::Unnamed(input.parse()?)
        } else {
            data::Fields::Unit
        };
        let discriminant = if input.peek(Token![=]) {
            let eq: Token![=] = input.parse()?;
            let discriminant: Expr = input.parse()?;
            Some((eq, discriminant))
        } else {
            None
        };
        Ok(data::Variant {
            attrs,
            ident,
            fields,
            discriminant,
        })
    }
}
impl Parse for Named {
    fn parse(input: Stream) -> Res<Self> {
        let content;
        Ok(data::Named {
            brace: braced!(content in input),
            named: content.parse_terminated(Field::parse_named, Token![,])?,
        })
    }
}
impl Parse for Unnamed {
    fn parse(input: Stream) -> Res<Self> {
        let content;
        Ok(data::Unnamed {
            paren: parenthesized!(content in input),
            unnamed: content.parse_terminated(Field::parse_unnamed, Token![,])?,
        })
    }
}
impl data::Field {
    pub fn parse_named(input: Stream) -> Res<Self> {
        Ok(data::Field {
            attrs: input.call(attr::Attr::parse_outer)?,
            vis: input.parse()?,
            mutability: FieldMutability::None,
            ident: Some(if input.peek(Token![_]) {
                input.call(Ident::parse_any)
            } else {
                input.parse()
            }?),
            colon: Some(input.parse()?),
            ty: input.parse()?,
        })
    }
    pub fn parse_unnamed(input: Stream) -> Res<Self> {
        Ok(data::Field {
            attrs: input.call(attr::Attr::parse_outer)?,
            vis: input.parse()?,
            mutability: FieldMutability::None,
            ident: None,
            colon: None,
            ty: input.parse()?,
        })
    }
}

impl Parse for DeriveInput {
    fn parse(input: Stream) -> Res<Self> {
        let attrs = input.call(attr::Attr::parse_outer)?;
        let vis = input.parse::<Visibility>()?;
        let lookahead = input.lookahead1();
        if lookahead.peek(Token![struct]) {
            let struct_ = input.parse::<Token![struct]>()?;
            let ident = input.parse::<Ident>()?;
            let gens = input.parse::<gen::Gens>()?;
            let (where_clause, fields, semi) = data_struct(input)?;
            Ok(DeriveInput {
                attrs,
                vis,
                ident,
                gens: gen::Gens { where_clause, ..gens },
                data: Data::Struct(data::Struct { struct_, fields, semi }),
            })
        } else if lookahead.peek(Token![enum]) {
            let enum_ = input.parse::<Token![enum]>()?;
            let ident = input.parse::<Ident>()?;
            let gens = input.parse::<gen::Gens>()?;
            let (where_clause, brace, variants) = data_enum(input)?;
            Ok(DeriveInput {
                attrs,
                vis,
                ident,
                gens: gen::Gens { where_clause, ..gens },
                data: Data::Enum(data::Enum { enum_, brace, variants }),
            })
        } else if lookahead.peek(Token![union]) {
            let union_ = input.parse::<Token![union]>()?;
            let ident = input.parse::<Ident>()?;
            let gens = input.parse::<gen::Gens>()?;
            let (where_clause, fields) = data_union(input)?;
            Ok(DeriveInput {
                attrs,
                vis,
                ident,
                gens: gen::Gens { where_clause, ..gens },
                data: Data::Union(data::Union { union_, fields }),
            })
        } else {
            Err(lookahead.error())
        }
    }
}
pub fn data_struct(x: Stream) -> Res<(Option<gen::Where>, data::Fields, Option<Token![;]>)> {
    let mut look = x.lookahead1();
    let mut where_clause = None;
    if look.peek(Token![where]) {
        where_clause = Some(x.parse()?);
        look = x.lookahead1();
    }
    if where_clause.is_none() && look.peek(tok::Paren) {
        let fields = x.parse()?;
        look = x.lookahead1();
        if look.peek(Token![where]) {
            where_clause = Some(x.parse()?);
            look = x.lookahead1();
        }
        if look.peek(Token![;]) {
            let semi = x.parse()?;
            Ok((where_clause, data::Fields::Unnamed(fields), Some(semi)))
        } else {
            Err(look.error())
        }
    } else if look.peek(tok::Brace) {
        let fields = x.parse()?;
        Ok((where_clause, data::Fields::Named(fields), None))
    } else if look.peek(Token![;]) {
        let semi = x.parse()?;
        Ok((where_clause, data::Fields::Unit, Some(semi)))
    } else {
        Err(look.error())
    }
}
pub fn data_enum(input: Stream) -> Res<(Option<gen::Where>, tok::Brace, Punctuated<data::Variant, Token![,]>)> {
    let where_clause = input.parse()?;
    let content;
    let brace = braced!(content in input);
    let variants = content.parse_terminated(data::Variant::parse, Token![,])?;
    Ok((where_clause, brace, variants))
}
pub fn data_union(input: Stream) -> Res<(Option<gen::Where>, data::Named)> {
    let where_clause = input.parse()?;
    let fields = input.parse()?;
    Ok((where_clause, fields))
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
            self.named.to_tokens(ys);
        });
    }
}
impl ToTokens for Unnamed {
    fn to_tokens(&self, ys: &mut Stream) {
        self.paren.surround(ys, |ys| {
            self.unnamed.to_tokens(ys);
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
                data::Fields::Named(x) => {
                    self.gens.where_.to_tokens(ys);
                    x.to_tokens(ys);
                },
                data::Fields::Unnamed(x) => {
                    x.to_tokens(ys);
                    self.gens.where_.to_tokens(ys);
                    TokensOrDefault(&data.semi).to_tokens(ys);
                },
                data::Fields::Unit => {
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
                x.fields.to_tokens(ys);
            },
        }
    }
}
