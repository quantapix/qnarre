use super::*;
use crate::punctuated::Punctuated;
use proc_macro2::TokenStream;
ast_enum_of_structs! {
    pub enum Pat {
        Const(PatConst),
        Ident(PatIdent),
        Lit(PatLit),
        Macro(PatMacro),
        Or(PatOr),
        Paren(PatParen),
        Path(PatPath),
        Range(PatRange),
        Reference(PatReference),
        Rest(PatRest),
        Slice(PatSlice),
        Struct(PatStruct),
        Tuple(PatTuple),
        TupleStruct(PatTupleStruct),
        Type(PatType),
        Verbatim(TokenStream),
        Wild(PatWild),
    }
}
ast_struct! {
    pub struct PatIdent {
        pub attrs: Vec<Attribute>,
        pub by_ref: Option<Token![ref]>,
        pub mutability: Option<Token![mut]>,
        pub ident: Ident,
        pub subpat: Option<(Token![@], Box<Pat>)>,
    }
}
ast_struct! {
    pub struct PatOr {
        pub attrs: Vec<Attribute>,
        pub leading_vert: Option<Token![|]>,
        pub cases: Punctuated<Pat, Token![|]>,
    }
}
ast_struct! {
    pub struct PatParen {
        pub attrs: Vec<Attribute>,
        pub paren_token: token::Paren,
        pub pat: Box<Pat>,
    }
}
ast_struct! {
    pub struct PatReference {
        pub attrs: Vec<Attribute>,
        pub and_token: Token![&],
        pub mutability: Option<Token![mut]>,
        pub pat: Box<Pat>,
    }
}
ast_struct! {
    pub struct PatRest {
        pub attrs: Vec<Attribute>,
        pub dot2_token: Token![..],
    }
}
ast_struct! {
    pub struct PatSlice {
        pub attrs: Vec<Attribute>,
        pub bracket_token: token::Bracket,
        pub elems: Punctuated<Pat, Token![,]>,
    }
}
ast_struct! {
    pub struct PatStruct {
        pub attrs: Vec<Attribute>,
        pub qself: Option<QSelf>,
        pub path: Path,
        pub brace_token: token::Brace,
        pub fields: Punctuated<FieldPat, Token![,]>,
        pub rest: Option<PatRest>,
    }
}
ast_struct! {
    pub struct PatTuple {
        pub attrs: Vec<Attribute>,
        pub paren_token: token::Paren,
        pub elems: Punctuated<Pat, Token![,]>,
    }
}
ast_struct! {
    pub struct PatTupleStruct {
        pub attrs: Vec<Attribute>,
        pub qself: Option<QSelf>,
        pub path: Path,
        pub paren_token: token::Paren,
        pub elems: Punctuated<Pat, Token![,]>,
    }
}
ast_struct! {
    pub struct PatType {
        pub attrs: Vec<Attribute>,
        pub pat: Box<Pat>,
        pub colon_token: Token![:],
        pub ty: Box<Type>,
    }
}
ast_struct! {
    pub struct PatWild {
        pub attrs: Vec<Attribute>,
        pub underscore_token: Token![_],
    }
}
ast_struct! {
    pub struct FieldPat {
        pub attrs: Vec<Attribute>,
        pub member: Member,
        pub colon_token: Option<Token![:]>,
        pub pat: Box<Pat>,
    }
}
