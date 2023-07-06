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
pub struct ImplGenerics<'a>(pub &'a Generics);
pub struct TypeGenerics<'a>(pub &'a Generics);
pub struct Turbofish<'a>(pub &'a Generics);
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
