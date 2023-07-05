use super::*;
use crate::punctuated::Punctuated;
ast_struct! {
    pub struct Variant {
        pub attrs: Vec<Attribute>,
        pub ident: Ident,
        pub fields: Fields,
        pub discriminant: Option<(Token![=], Expr)>,
    }
}
ast_enum_of_structs! {
    pub enum Fields {
        Named(FieldsNamed),
        Unnamed(FieldsUnnamed),
        Unit,
    }
}
ast_struct! {
    pub struct FieldsNamed {
        pub brace_token: token::Brace,
        pub named: Punctuated<Field, Token![,]>,
    }
}
ast_struct! {
    pub struct FieldsUnnamed {
        pub paren_token: token::Paren,
        pub unnamed: Punctuated<Field, Token![,]>,
    }
}
impl Fields {
    pub fn iter(&self) -> punctuated::Iter<Field> {
        match self {
            Fields::Unit => crate::punctuated::empty_punctuated_iter(),
            Fields::Named(f) => f.named.iter(),
            Fields::Unnamed(f) => f.unnamed.iter(),
        }
    }
    pub fn iter_mut(&mut self) -> punctuated::IterMut<Field> {
        match self {
            Fields::Unit => crate::punctuated::empty_punctuated_iter_mut(),
            Fields::Named(f) => f.named.iter_mut(),
            Fields::Unnamed(f) => f.unnamed.iter_mut(),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Fields::Unit => 0,
            Fields::Named(f) => f.named.len(),
            Fields::Unnamed(f) => f.unnamed.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        match self {
            Fields::Unit => true,
            Fields::Named(f) => f.named.is_empty(),
            Fields::Unnamed(f) => f.unnamed.is_empty(),
        }
    }
}
impl IntoIterator for Fields {
    type Item = Field;
    type IntoIter = punctuated::IntoIter<Field>;
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Fields::Unit => Punctuated::<Field, ()>::new().into_iter(),
            Fields::Named(f) => f.named.into_iter(),
            Fields::Unnamed(f) => f.unnamed.into_iter(),
        }
    }
}
impl<'a> IntoIterator for &'a Fields {
    type Item = &'a Field;
    type IntoIter = punctuated::Iter<'a, Field>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a> IntoIterator for &'a mut Fields {
    type Item = &'a mut Field;
    type IntoIter = punctuated::IterMut<'a, Field>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
ast_struct! {
    pub struct Field {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub mutability: FieldMutability,
        pub ident: Option<Ident>,
        pub colon_token: Option<Token![:]>,
        pub ty: Type,
    }
}
pub(crate) mod parsing {
    use super::*;
    use crate::ext::IdentExt;
    use crate::parse::{Parse, ParseStream, Result};
    impl Parse for Variant {
        fn parse(input: ParseStream) -> Result<Self> {
            let attrs = input.call(Attribute::parse_outer)?;
            let _visibility: Visibility = input.parse()?;
            let ident: Ident = input.parse()?;
            let fields = if input.peek(token::Brace) {
                Fields::Named(input.parse()?)
            } else if input.peek(token::Paren) {
                Fields::Unnamed(input.parse()?)
            } else {
                Fields::Unit
            };
            let discriminant = if input.peek(Token![=]) {
                let eq_token: Token![=] = input.parse()?;
                let discriminant: Expr = input.parse()?;
                Some((eq_token, discriminant))
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
    impl Parse for FieldsNamed {
        fn parse(input: ParseStream) -> Result<Self> {
            let content;
            Ok(FieldsNamed {
                brace_token: braced!(content in input),
                named: content.parse_terminated(Field::parse_named, Token![,])?,
            })
        }
    }
    impl Parse for FieldsUnnamed {
        fn parse(input: ParseStream) -> Result<Self> {
            let content;
            Ok(FieldsUnnamed {
                paren_token: parenthesized!(content in input),
                unnamed: content.parse_terminated(Field::parse_unnamed, Token![,])?,
            })
        }
    }
    impl Field {
        pub fn parse_named(input: ParseStream) -> Result<Self> {
            Ok(Field {
                attrs: input.call(Attribute::parse_outer)?,
                vis: input.parse()?,
                mutability: FieldMutability::None,
                ident: Some(if input.peek(Token![_]) {
                    input.call(Ident::parse_any)
                } else {
                    input.parse()
                }?),
                colon_token: Some(input.parse()?),
                ty: input.parse()?,
            })
        }
        pub fn parse_unnamed(input: ParseStream) -> Result<Self> {
            Ok(Field {
                attrs: input.call(Attribute::parse_outer)?,
                vis: input.parse()?,
                mutability: FieldMutability::None,
                ident: None,
                colon_token: None,
                ty: input.parse()?,
            })
        }
    }
}
mod printing {
    use super::*;
    use crate::print::TokensOrDefault;
    use proc_macro2::TokenStream;
    use quote::{ToTokens, TokenStreamExt};
    impl ToTokens for Variant {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            tokens.append_all(&self.attrs);
            self.ident.to_tokens(tokens);
            self.fields.to_tokens(tokens);
            if let Some((eq_token, disc)) = &self.discriminant {
                eq_token.to_tokens(tokens);
                disc.to_tokens(tokens);
            }
        }
    }
    impl ToTokens for FieldsNamed {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.brace_token.surround(tokens, |tokens| {
                self.named.to_tokens(tokens);
            });
        }
    }
    impl ToTokens for FieldsUnnamed {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.paren_token.surround(tokens, |tokens| {
                self.unnamed.to_tokens(tokens);
            });
        }
    }
    impl ToTokens for Field {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            tokens.append_all(&self.attrs);
            self.vis.to_tokens(tokens);
            if let Some(ident) = &self.ident {
                ident.to_tokens(tokens);
                TokensOrDefault(&self.colon_token).to_tokens(tokens);
            }
            self.ty.to_tokens(tokens);
        }
    }
}
