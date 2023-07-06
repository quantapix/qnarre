use super::*;
use crate::punctuated::Punctuated;
ast_struct! {
    pub struct Path {
        pub leading_colon: Option<Token![::]>,
        pub segments: Punctuated<PathSegment, Token![::]>,
    }
}
impl<T> From<T> for Path
where
    T: Into<PathSegment>,
{
    fn from(segment: T) -> Self {
        let mut path = Path {
            leading_colon: None,
            segments: Punctuated::new(),
        };
        path.segments.push_value(segment.into());
        path
    }
}
impl Path {
    pub fn is_ident<I: ?Sized>(&self, ident: &I) -> bool
    where
        Ident: PartialEq<I>,
    {
        match self.get_ident() {
            Some(id) => id == ident,
            None => false,
        }
    }
    pub fn get_ident(&self) -> Option<&Ident> {
        if self.leading_colon.is_none() && self.segments.len() == 1 && self.segments[0].arguments.is_none() {
            Some(&self.segments[0].ident)
        } else {
            None
        }
    }
}
ast_struct! {
    pub struct PathSegment {
        pub ident: Ident,
        pub arguments: PathArguments,
    }
}
impl<T> From<T> for PathSegment
where
    T: Into<Ident>,
{
    fn from(ident: T) -> Self {
        PathSegment {
            ident: ident.into(),
            arguments: PathArguments::None,
        }
    }
}
ast_enum! {
    pub enum PathArguments {
        None,
        AngleBracketed(AngleBracketedGenericArguments),
        Parenthesized(ParenthesizedGenericArguments),
    }
}
impl Default for PathArguments {
    fn default() -> Self {
        PathArguments::None
    }
}
impl PathArguments {
    pub fn is_empty(&self) -> bool {
        match self {
            PathArguments::None => true,
            PathArguments::AngleBracketed(bracketed) => bracketed.args.is_empty(),
            PathArguments::Parenthesized(_) => false,
        }
    }
    pub fn is_none(&self) -> bool {
        match self {
            PathArguments::None => true,
            PathArguments::AngleBracketed(_) | PathArguments::Parenthesized(_) => false,
        }
    }
}
ast_enum! {
    pub enum GenericArgument {
        Lifetime(Lifetime),
        Type(Type),
        Const(Expr),
        AssocType(AssocType),
        AssocConst(AssocConst),
        Constraint(Constraint),
    }
}
ast_struct! {
    pub struct AngleBracketedGenericArguments {
        pub colon2_token: Option<Token![::]>,
        pub lt_token: Token![<],
        pub args: Punctuated<GenericArgument, Token![,]>,
        pub gt_token: Token![>],
    }
}
ast_struct! {
    pub struct AssocType {
        pub ident: Ident,
        pub generics: Option<AngleBracketedGenericArguments>,
        pub eq_token: Token![=],
        pub ty: Type,
    }
}
ast_struct! {
    pub struct AssocConst {
        pub ident: Ident,
        pub generics: Option<AngleBracketedGenericArguments>,
        pub eq_token: Token![=],
        pub value: Expr,
    }
}
ast_struct! {
    pub struct Constraint {
        pub ident: Ident,
        pub generics: Option<AngleBracketedGenericArguments>,
        pub colon_token: Token![:],
        pub bounds: Punctuated<TypeParamBound, Token![+]>,
    }
}
ast_struct! {
    pub struct ParenthesizedGenericArguments {
        pub paren_token: token::Paren,
        pub inputs: Punctuated<Type, Token![,]>,
        pub output: ReturnType,
    }
}
ast_struct! {
    pub struct QSelf {
        pub lt_token: Token![<],
        pub ty: Box<Type>,
        pub position: usize,
        pub as_token: Option<Token![as]>,
        pub gt_token: Token![>],
    }
}
