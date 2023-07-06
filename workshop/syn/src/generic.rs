use super::*;
use crate::punctuated::{Iter, IterMut, Punctuated};
use proc_macro2::TokenStream;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
ast_struct! {
    pub struct Generics {
        pub lt_token: Option<Token![<]>,
        pub params: Punctuated<GenericParam, Token![,]>,
        pub gt_token: Option<Token![>]>,
        pub where_clause: Option<WhereClause>,
    }
}
ast_enum_of_structs! {
    pub enum GenericParam {
        Lifetime(LifetimeParam),
        Type(TypeParam),
        Const(ConstParam),
    }
}
ast_struct! {
    pub struct LifetimeParam {
        pub attrs: Vec<Attribute>,
        pub lifetime: Lifetime,
        pub colon_token: Option<Token![:]>,
        pub bounds: Punctuated<Lifetime, Token![+]>,
    }
}
ast_struct! {
    pub struct TypeParam {
        pub attrs: Vec<Attribute>,
        pub ident: Ident,
        pub colon_token: Option<Token![:]>,
        pub bounds: Punctuated<TypeParamBound, Token![+]>,
        pub eq_token: Option<Token![=]>,
        pub default: Option<Type>,
    }
}
ast_struct! {
    pub struct ConstParam {
        pub attrs: Vec<Attribute>,
        pub const_token: Token![const],
        pub ident: Ident,
        pub colon_token: Token![:],
        pub ty: Type,
        pub eq_token: Option<Token![=]>,
        pub default: Option<Expr>,
    }
}
impl Default for Generics {
    fn default() -> Self {
        Generics {
            lt_token: None,
            params: Punctuated::new(),
            gt_token: None,
            where_clause: None,
        }
    }
}
impl Generics {
    pub fn lifetimes(&self) -> Lifetimes {
        Lifetimes(self.params.iter())
    }
    pub fn lifetimes_mut(&mut self) -> LifetimesMut {
        LifetimesMut(self.params.iter_mut())
    }
    pub fn type_params(&self) -> TypeParams {
        TypeParams(self.params.iter())
    }
    pub fn type_params_mut(&mut self) -> TypeParamsMut {
        TypeParamsMut(self.params.iter_mut())
    }
    pub fn const_params(&self) -> ConstParams {
        ConstParams(self.params.iter())
    }
    pub fn const_params_mut(&mut self) -> ConstParamsMut {
        ConstParamsMut(self.params.iter_mut())
    }
    pub fn make_where_clause(&mut self) -> &mut WhereClause {
        self.where_clause.get_or_insert_with(|| WhereClause {
            where_token: <Token![where]>::default(),
            predicates: Punctuated::new(),
        })
    }
}
pub struct Lifetimes<'a>(Iter<'a, GenericParam>);
impl<'a> Iterator for Lifetimes<'a> {
    type Item = &'a LifetimeParam;
    fn next(&mut self) -> Option<Self::Item> {
        let next = match self.0.next() {
            Some(item) => item,
            None => return None,
        };
        if let GenericParam::Lifetime(lifetime) = next {
            Some(lifetime)
        } else {
            self.next()
        }
    }
}
pub struct LifetimesMut<'a>(IterMut<'a, GenericParam>);
impl<'a> Iterator for LifetimesMut<'a> {
    type Item = &'a mut LifetimeParam;
    fn next(&mut self) -> Option<Self::Item> {
        let next = match self.0.next() {
            Some(item) => item,
            None => return None,
        };
        if let GenericParam::Lifetime(lifetime) = next {
            Some(lifetime)
        } else {
            self.next()
        }
    }
}
pub struct TypeParams<'a>(Iter<'a, GenericParam>);
impl<'a> Iterator for TypeParams<'a> {
    type Item = &'a TypeParam;
    fn next(&mut self) -> Option<Self::Item> {
        let next = match self.0.next() {
            Some(item) => item,
            None => return None,
        };
        if let GenericParam::Type(type_param) = next {
            Some(type_param)
        } else {
            self.next()
        }
    }
}
pub struct TypeParamsMut<'a>(IterMut<'a, GenericParam>);
impl<'a> Iterator for TypeParamsMut<'a> {
    type Item = &'a mut TypeParam;
    fn next(&mut self) -> Option<Self::Item> {
        let next = match self.0.next() {
            Some(item) => item,
            None => return None,
        };
        if let GenericParam::Type(type_param) = next {
            Some(type_param)
        } else {
            self.next()
        }
    }
}
pub struct ConstParams<'a>(Iter<'a, GenericParam>);
impl<'a> Iterator for ConstParams<'a> {
    type Item = &'a ConstParam;
    fn next(&mut self) -> Option<Self::Item> {
        let next = match self.0.next() {
            Some(item) => item,
            None => return None,
        };
        if let GenericParam::Const(const_param) = next {
            Some(const_param)
        } else {
            self.next()
        }
    }
}
pub struct ConstParamsMut<'a>(IterMut<'a, GenericParam>);
impl<'a> Iterator for ConstParamsMut<'a> {
    type Item = &'a mut ConstParam;
    fn next(&mut self) -> Option<Self::Item> {
        let next = match self.0.next() {
            Some(item) => item,
            None => return None,
        };
        if let GenericParam::Const(const_param) = next {
            Some(const_param)
        } else {
            self.next()
        }
    }
}
pub struct ImplGenerics<'a>(&'a Generics);
pub struct TypeGenerics<'a>(&'a Generics);
pub struct Turbofish<'a>(&'a Generics);
impl Generics {
    pub fn split_for_impl(&self) -> (ImplGenerics, TypeGenerics, Option<&WhereClause>) {
        (ImplGenerics(self), TypeGenerics(self), self.where_clause.as_ref())
    }
}
macro_rules! generics_wrapper_impls {
    ($ty:ident) => {
        impl<'a> Clone for $ty<'a> {
            fn clone(&self) -> Self {
                $ty(self.0)
            }
        }
        impl<'a> Debug for $ty<'a> {
            fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.debug_tuple(stringify!($ty)).field(self.0).finish()
            }
        }
        impl<'a> Eq for $ty<'a> {}
        impl<'a> PartialEq for $ty<'a> {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
        impl<'a> Hash for $ty<'a> {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.0.hash(state);
            }
        }
    };
}
generics_wrapper_impls!(ImplGenerics);
generics_wrapper_impls!(TypeGenerics);
generics_wrapper_impls!(Turbofish);
impl<'a> TypeGenerics<'a> {
    pub fn as_turbofish(&self) -> Turbofish {
        Turbofish(self.0)
    }
}
ast_struct! {
    pub struct BoundLifetimes {
        pub for_token: Token![for],
        pub lt_token: Token![<],
        pub lifetimes: Punctuated<GenericParam, Token![,]>,
        pub gt_token: Token![>],
    }
}
impl Default for BoundLifetimes {
    fn default() -> Self {
        BoundLifetimes {
            for_token: Default::default(),
            lt_token: Default::default(),
            lifetimes: Punctuated::new(),
            gt_token: Default::default(),
        }
    }
}
impl LifetimeParam {
    pub fn new(lifetime: Lifetime) -> Self {
        LifetimeParam {
            attrs: Vec::new(),
            lifetime,
            colon_token: None,
            bounds: Punctuated::new(),
        }
    }
}
impl From<Ident> for TypeParam {
    fn from(ident: Ident) -> Self {
        TypeParam {
            attrs: vec![],
            ident,
            colon_token: None,
            bounds: Punctuated::new(),
            eq_token: None,
            default: None,
        }
    }
}
ast_enum_of_structs! {
    pub enum TypeParamBound {
        Trait(TraitBound),
        Lifetime(Lifetime),
        Verbatim(TokenStream),
    }
}
ast_struct! {
    pub struct TraitBound {
        pub paren_token: Option<token::Paren>,
        pub modifier: TraitBoundModifier,
        pub lifetimes: Option<BoundLifetimes>,
        pub path: Path,
    }
}
ast_enum! {
    pub enum TraitBoundModifier {
        None,
        Maybe(Token![?]),
    }
}
ast_struct! {
    pub struct WhereClause {
        pub where_token: Token![where],
        pub predicates: Punctuated<WherePredicate, Token![,]>,
    }
}
ast_enum_of_structs! {
    pub enum WherePredicate {
        Lifetime(PredicateLifetime),
        Type(PredicateType),
    }
}
ast_struct! {
    pub struct PredicateLifetime {
        pub lifetime: Lifetime,
        pub colon_token: Token![:],
        pub bounds: Punctuated<Lifetime, Token![+]>,
    }
}
ast_struct! {
    pub struct PredicateType {
        pub lifetimes: Option<BoundLifetimes>,
        pub bounded_ty: Type,
        pub colon_token: Token![:],
        pub bounds: Punctuated<TypeParamBound, Token![+]>,
    }
}
pub(crate) mod parsing {
    use super::*;
    use crate::ext::IdentExt;
    use crate::parse::{Parse, ParseStream, Result};

    impl Parse for Generics {
        fn parse(input: ParseStream) -> Result<Self> {
            if !input.peek(Token![<]) {
                return Ok(Generics::default());
            }
            let lt_token: Token![<] = input.parse()?;
            let mut params = Punctuated::new();
            loop {
                if input.peek(Token![>]) {
                    break;
                }
                let attrs = input.call(Attribute::parse_outer)?;
                let lookahead = input.lookahead1();
                if lookahead.peek(Lifetime) {
                    params.push_value(GenericParam::Lifetime(LifetimeParam {
                        attrs,
                        ..input.parse()?
                    }));
                } else if lookahead.peek(Ident) {
                    params.push_value(GenericParam::Type(TypeParam {
                        attrs,
                        ..input.parse()?
                    }));
                } else if lookahead.peek(Token![const]) {
                    params.push_value(GenericParam::Const(ConstParam {
                        attrs,
                        ..input.parse()?
                    }));
                } else if input.peek(Token![_]) {
                    params.push_value(GenericParam::Type(TypeParam {
                        attrs,
                        ident: input.call(Ident::parse_any)?,
                        colon_token: None,
                        bounds: Punctuated::new(),
                        eq_token: None,
                        default: None,
                    }));
                } else {
                    return Err(lookahead.error());
                }
                if input.peek(Token![>]) {
                    break;
                }
                let punct = input.parse()?;
                params.push_punct(punct);
            }
            let gt_token: Token![>] = input.parse()?;
            Ok(Generics {
                lt_token: Some(lt_token),
                params,
                gt_token: Some(gt_token),
                where_clause: None,
            })
        }
    }

    impl Parse for GenericParam {
        fn parse(input: ParseStream) -> Result<Self> {
            let attrs = input.call(Attribute::parse_outer)?;
            let lookahead = input.lookahead1();
            if lookahead.peek(Ident) {
                Ok(GenericParam::Type(TypeParam {
                    attrs,
                    ..input.parse()?
                }))
            } else if lookahead.peek(Lifetime) {
                Ok(GenericParam::Lifetime(LifetimeParam {
                    attrs,
                    ..input.parse()?
                }))
            } else if lookahead.peek(Token![const]) {
                Ok(GenericParam::Const(ConstParam {
                    attrs,
                    ..input.parse()?
                }))
            } else {
                Err(lookahead.error())
            }
        }
    }

    impl Parse for LifetimeParam {
        fn parse(input: ParseStream) -> Result<Self> {
            let has_colon;
            Ok(LifetimeParam {
                attrs: input.call(Attribute::parse_outer)?,
                lifetime: input.parse()?,
                colon_token: {
                    if input.peek(Token![:]) {
                        has_colon = true;
                        Some(input.parse()?)
                    } else {
                        has_colon = false;
                        None
                    }
                },
                bounds: {
                    let mut bounds = Punctuated::new();
                    if has_colon {
                        loop {
                            if input.peek(Token![,]) || input.peek(Token![>]) {
                                break;
                            }
                            let value = input.parse()?;
                            bounds.push_value(value);
                            if !input.peek(Token![+]) {
                                break;
                            }
                            let punct = input.parse()?;
                            bounds.push_punct(punct);
                        }
                    }
                    bounds
                },
            })
        }
    }

    impl Parse for BoundLifetimes {
        fn parse(input: ParseStream) -> Result<Self> {
            Ok(BoundLifetimes {
                for_token: input.parse()?,
                lt_token: input.parse()?,
                lifetimes: {
                    let mut lifetimes = Punctuated::new();
                    while !input.peek(Token![>]) {
                        let attrs = input.call(Attribute::parse_outer)?;
                        let lifetime: Lifetime = input.parse()?;
                        lifetimes.push_value(GenericParam::Lifetime(LifetimeParam {
                            attrs,
                            lifetime,
                            colon_token: None,
                            bounds: Punctuated::new(),
                        }));
                        if input.peek(Token![>]) {
                            break;
                        }
                        lifetimes.push_punct(input.parse()?);
                    }
                    lifetimes
                },
                gt_token: input.parse()?,
            })
        }
    }

    impl Parse for Option<BoundLifetimes> {
        fn parse(input: ParseStream) -> Result<Self> {
            if input.peek(Token![for]) {
                input.parse().map(Some)
            } else {
                Ok(None)
            }
        }
    }

    impl Parse for TypeParam {
        fn parse(input: ParseStream) -> Result<Self> {
            let attrs = input.call(Attribute::parse_outer)?;
            let ident: Ident = input.parse()?;
            let colon_token: Option<Token![:]> = input.parse()?;
            let mut bounds = Punctuated::new();
            if colon_token.is_some() {
                loop {
                    if input.peek(Token![,]) || input.peek(Token![>]) || input.peek(Token![=]) {
                        break;
                    }
                    let value: TypeParamBound = input.parse()?;
                    bounds.push_value(value);
                    if !input.peek(Token![+]) {
                        break;
                    }
                    let punct: Token![+] = input.parse()?;
                    bounds.push_punct(punct);
                }
            }
            let eq_token: Option<Token![=]> = input.parse()?;
            let default = if eq_token.is_some() {
                Some(input.parse::<Type>()?)
            } else {
                None
            };
            Ok(TypeParam {
                attrs,
                ident,
                colon_token,
                bounds,
                eq_token,
                default,
            })
        }
    }

    impl Parse for TypeParamBound {
        fn parse(input: ParseStream) -> Result<Self> {
            if input.peek(Lifetime) {
                return input.parse().map(TypeParamBound::Lifetime);
            }
            let begin = input.fork();
            let content;
            let (paren_token, content) = if input.peek(token::Paren) {
                (Some(parenthesized!(content in input)), &content)
            } else {
                (None, input)
            };
            let is_tilde_const = cfg!(feature = "full") && content.peek(Token![~]) && content.peek2(Token![const]);
            if is_tilde_const {
                content.parse::<Token![~]>()?;
                content.parse::<Token![const]>()?;
            }
            let mut bound: TraitBound = content.parse()?;
            bound.paren_token = paren_token;
            if is_tilde_const {
                Ok(TypeParamBound::Verbatim(verbatim_between(&begin, input)))
            } else {
                Ok(TypeParamBound::Trait(bound))
            }
        }
    }
    impl TypeParamBound {
        pub(crate) fn parse_multiple(input: ParseStream, allow_plus: bool) -> Result<Punctuated<Self, Token![+]>> {
            let mut bounds = Punctuated::new();
            loop {
                bounds.push_value(input.parse()?);
                if !(allow_plus && input.peek(Token![+])) {
                    break;
                }
                bounds.push_punct(input.parse()?);
                if !(input.peek(Ident::peek_any)
                    || input.peek(Token![::])
                    || input.peek(Token![?])
                    || input.peek(Lifetime)
                    || input.peek(token::Paren)
                    || input.peek(Token![~]))
                {
                    break;
                }
            }
            Ok(bounds)
        }
    }

    impl Parse for TraitBound {
        fn parse(input: ParseStream) -> Result<Self> {
            let modifier: TraitBoundModifier = input.parse()?;
            let lifetimes: Option<BoundLifetimes> = input.parse()?;
            let mut path: Path = input.parse()?;
            if path.segments.last().unwrap().arguments.is_empty()
                && (input.peek(token::Paren) || input.peek(Token![::]) && input.peek3(token::Paren))
            {
                input.parse::<Option<Token![::]>>()?;
                let args: ParenthesizedGenericArguments = input.parse()?;
                let parenthesized = PathArguments::Parenthesized(args);
                path.segments.last_mut().unwrap().arguments = parenthesized;
            }
            Ok(TraitBound {
                paren_token: None,
                modifier,
                lifetimes,
                path,
            })
        }
    }

    impl Parse for TraitBoundModifier {
        fn parse(input: ParseStream) -> Result<Self> {
            if input.peek(Token![?]) {
                input.parse().map(TraitBoundModifier::Maybe)
            } else {
                Ok(TraitBoundModifier::None)
            }
        }
    }

    impl Parse for ConstParam {
        fn parse(input: ParseStream) -> Result<Self> {
            let mut default = None;
            Ok(ConstParam {
                attrs: input.call(Attribute::parse_outer)?,
                const_token: input.parse()?,
                ident: input.parse()?,
                colon_token: input.parse()?,
                ty: input.parse()?,
                eq_token: {
                    if input.peek(Token![=]) {
                        let eq_token = input.parse()?;
                        default = Some(path::parsing::const_argument(input)?);
                        Some(eq_token)
                    } else {
                        None
                    }
                },
                default,
            })
        }
    }

    impl Parse for WhereClause {
        fn parse(input: ParseStream) -> Result<Self> {
            Ok(WhereClause {
                where_token: input.parse()?,
                predicates: {
                    let mut predicates = Punctuated::new();
                    loop {
                        if input.is_empty()
                            || input.peek(token::Brace)
                            || input.peek(Token![,])
                            || input.peek(Token![;])
                            || input.peek(Token![:]) && !input.peek(Token![::])
                            || input.peek(Token![=])
                        {
                            break;
                        }
                        let value = input.parse()?;
                        predicates.push_value(value);
                        if !input.peek(Token![,]) {
                            break;
                        }
                        let punct = input.parse()?;
                        predicates.push_punct(punct);
                    }
                    predicates
                },
            })
        }
    }

    impl Parse for Option<WhereClause> {
        fn parse(input: ParseStream) -> Result<Self> {
            if input.peek(Token![where]) {
                input.parse().map(Some)
            } else {
                Ok(None)
            }
        }
    }

    impl Parse for WherePredicate {
        fn parse(input: ParseStream) -> Result<Self> {
            if input.peek(Lifetime) && input.peek2(Token![:]) {
                Ok(WherePredicate::Lifetime(PredicateLifetime {
                    lifetime: input.parse()?,
                    colon_token: input.parse()?,
                    bounds: {
                        let mut bounds = Punctuated::new();
                        loop {
                            if input.is_empty()
                                || input.peek(token::Brace)
                                || input.peek(Token![,])
                                || input.peek(Token![;])
                                || input.peek(Token![:])
                                || input.peek(Token![=])
                            {
                                break;
                            }
                            let value = input.parse()?;
                            bounds.push_value(value);
                            if !input.peek(Token![+]) {
                                break;
                            }
                            let punct = input.parse()?;
                            bounds.push_punct(punct);
                        }
                        bounds
                    },
                }))
            } else {
                Ok(WherePredicate::Type(PredicateType {
                    lifetimes: input.parse()?,
                    bounded_ty: input.parse()?,
                    colon_token: input.parse()?,
                    bounds: {
                        let mut bounds = Punctuated::new();
                        loop {
                            if input.is_empty()
                                || input.peek(token::Brace)
                                || input.peek(Token![,])
                                || input.peek(Token![;])
                                || input.peek(Token![:]) && !input.peek(Token![::])
                                || input.peek(Token![=])
                            {
                                break;
                            }
                            let value = input.parse()?;
                            bounds.push_value(value);
                            if !input.peek(Token![+]) {
                                break;
                            }
                            let punct = input.parse()?;
                            bounds.push_punct(punct);
                        }
                        bounds
                    },
                }))
            }
        }
    }
}
mod printing {
    use super::*;
    use crate::{attr::FilterAttrs, TokensOrDefault};
    use proc_macro2::TokenStream;
    use quote::{ToTokens, TokenStreamExt};
    impl ToTokens for Generics {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            if self.params.is_empty() {
                return;
            }
            TokensOrDefault(&self.lt_token).to_tokens(tokens);
            let mut trailing_or_empty = true;
            for param in self.params.pairs() {
                if let GenericParam::Lifetime(_) = **param.value() {
                    param.to_tokens(tokens);
                    trailing_or_empty = param.punct().is_some();
                }
            }
            for param in self.params.pairs() {
                match param.value() {
                    GenericParam::Type(_) | GenericParam::Const(_) => {
                        if !trailing_or_empty {
                            <Token![,]>::default().to_tokens(tokens);
                            trailing_or_empty = true;
                        }
                        param.to_tokens(tokens);
                    },
                    GenericParam::Lifetime(_) => {},
                }
            }
            TokensOrDefault(&self.gt_token).to_tokens(tokens);
        }
    }
    impl<'a> ToTokens for ImplGenerics<'a> {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            if self.0.params.is_empty() {
                return;
            }
            TokensOrDefault(&self.0.lt_token).to_tokens(tokens);
            let mut trailing_or_empty = true;
            for param in self.0.params.pairs() {
                if let GenericParam::Lifetime(_) = **param.value() {
                    param.to_tokens(tokens);
                    trailing_or_empty = param.punct().is_some();
                }
            }
            for param in self.0.params.pairs() {
                if let GenericParam::Lifetime(_) = **param.value() {
                    continue;
                }
                if !trailing_or_empty {
                    <Token![,]>::default().to_tokens(tokens);
                    trailing_or_empty = true;
                }
                match param.value() {
                    GenericParam::Lifetime(_) => unreachable!(),
                    GenericParam::Type(param) => {
                        tokens.append_all(param.attrs.outer());
                        param.ident.to_tokens(tokens);
                        if !param.bounds.is_empty() {
                            TokensOrDefault(&param.colon_token).to_tokens(tokens);
                            param.bounds.to_tokens(tokens);
                        }
                    },
                    GenericParam::Const(param) => {
                        tokens.append_all(param.attrs.outer());
                        param.const_token.to_tokens(tokens);
                        param.ident.to_tokens(tokens);
                        param.colon_token.to_tokens(tokens);
                        param.ty.to_tokens(tokens);
                    },
                }
                param.punct().to_tokens(tokens);
            }
            TokensOrDefault(&self.0.gt_token).to_tokens(tokens);
        }
    }
    impl<'a> ToTokens for TypeGenerics<'a> {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            if self.0.params.is_empty() {
                return;
            }
            TokensOrDefault(&self.0.lt_token).to_tokens(tokens);
            let mut trailing_or_empty = true;
            for param in self.0.params.pairs() {
                if let GenericParam::Lifetime(def) = *param.value() {
                    def.lifetime.to_tokens(tokens);
                    param.punct().to_tokens(tokens);
                    trailing_or_empty = param.punct().is_some();
                }
            }
            for param in self.0.params.pairs() {
                if let GenericParam::Lifetime(_) = **param.value() {
                    continue;
                }
                if !trailing_or_empty {
                    <Token![,]>::default().to_tokens(tokens);
                    trailing_or_empty = true;
                }
                match param.value() {
                    GenericParam::Lifetime(_) => unreachable!(),
                    GenericParam::Type(param) => {
                        param.ident.to_tokens(tokens);
                    },
                    GenericParam::Const(param) => {
                        param.ident.to_tokens(tokens);
                    },
                }
                param.punct().to_tokens(tokens);
            }
            TokensOrDefault(&self.0.gt_token).to_tokens(tokens);
        }
    }
    impl<'a> ToTokens for Turbofish<'a> {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            if !self.0.params.is_empty() {
                <Token![::]>::default().to_tokens(tokens);
                TypeGenerics(self.0).to_tokens(tokens);
            }
        }
    }
    impl ToTokens for BoundLifetimes {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.for_token.to_tokens(tokens);
            self.lt_token.to_tokens(tokens);
            self.lifetimes.to_tokens(tokens);
            self.gt_token.to_tokens(tokens);
        }
    }
    impl ToTokens for LifetimeParam {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            tokens.append_all(self.attrs.outer());
            self.lifetime.to_tokens(tokens);
            if !self.bounds.is_empty() {
                TokensOrDefault(&self.colon_token).to_tokens(tokens);
                self.bounds.to_tokens(tokens);
            }
        }
    }
    impl ToTokens for TypeParam {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            tokens.append_all(self.attrs.outer());
            self.ident.to_tokens(tokens);
            if !self.bounds.is_empty() {
                TokensOrDefault(&self.colon_token).to_tokens(tokens);
                self.bounds.to_tokens(tokens);
            }
            if let Some(default) = &self.default {
                TokensOrDefault(&self.eq_token).to_tokens(tokens);
                default.to_tokens(tokens);
            }
        }
    }
    impl ToTokens for TraitBound {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let to_tokens = |tokens: &mut TokenStream| {
                self.modifier.to_tokens(tokens);
                self.lifetimes.to_tokens(tokens);
                self.path.to_tokens(tokens);
            };
            match &self.paren_token {
                Some(paren) => paren.surround(tokens, to_tokens),
                None => to_tokens(tokens),
            }
        }
    }
    impl ToTokens for TraitBoundModifier {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            match self {
                TraitBoundModifier::None => {},
                TraitBoundModifier::Maybe(t) => t.to_tokens(tokens),
            }
        }
    }
    impl ToTokens for ConstParam {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            tokens.append_all(self.attrs.outer());
            self.const_token.to_tokens(tokens);
            self.ident.to_tokens(tokens);
            self.colon_token.to_tokens(tokens);
            self.ty.to_tokens(tokens);
            if let Some(default) = &self.default {
                TokensOrDefault(&self.eq_token).to_tokens(tokens);
                default.to_tokens(tokens);
            }
        }
    }
    impl ToTokens for WhereClause {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            if !self.predicates.is_empty() {
                self.where_token.to_tokens(tokens);
                self.predicates.to_tokens(tokens);
            }
        }
    }
    impl ToTokens for PredicateLifetime {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.lifetime.to_tokens(tokens);
            self.colon_token.to_tokens(tokens);
            self.bounds.to_tokens(tokens);
        }
    }
    impl ToTokens for PredicateType {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.lifetimes.to_tokens(tokens);
            self.bounded_ty.to_tokens(tokens);
            self.colon_token.to_tokens(tokens);
            self.bounds.to_tokens(tokens);
        }
    }
}
