use super::{punctuated::Punctuated, *};
use proc_macro2::TokenStream;

ast_enum_of_structs! {
    pub enum Type {
        Array(TypeArray),
        BareFn(TypeBareFn),
        Group(TypeGroup),
        ImplTrait(TypeImplTrait),
        Infer(TypeInfer),
        Macro(TypeMacro),
        Never(TypeNever),
        Paren(TypeParen),
        Path(TypePath),
        Ptr(TypePtr),
        Reference(TypeReference),
        Slice(TypeSlice),
        TraitObject(TypeTraitObject),
        Tuple(TypeTuple),
        Verbatim(TokenStream),
    }
}
ast_struct! {
    pub struct TypeArray {
        pub bracket_token: token::Bracket,
        pub elem: Box<Type>,
        pub semi_token: Token![;],
        pub len: Expr,
    }
}
ast_struct! {
    pub struct TypeBareFn {
        pub lifetimes: Option<BoundLifetimes>,
        pub unsafety: Option<Token![unsafe]>,
        pub abi: Option<Abi>,
        pub fn_token: Token![fn],
        pub paren_token: token::Paren,
        pub inputs: Punctuated<BareFnArg, Token![,]>,
        pub variadic: Option<BareVariadic>,
        pub output: ReturnType,
    }
}
ast_struct! {
    pub struct TypeGroup {
        pub group_token: token::Group,
        pub elem: Box<Type>,
    }
}
ast_struct! {
    pub struct TypeImplTrait {
        pub impl_token: Token![impl],
        pub bounds: Punctuated<TypeParamBound, Token![+]>,
    }
}
ast_struct! {
    pub struct TypeInfer {
        pub underscore_token: Token![_],
    }
}
ast_struct! {
    pub struct TypeMacro {
        pub mac: Macro,
    }
}
ast_struct! {
    pub struct TypeNever {
        pub bang_token: Token![!],
    }
}
ast_struct! {
    pub struct TypeParen {
        pub paren_token: token::Paren,
        pub elem: Box<Type>,
    }
}
ast_struct! {
    pub struct TypePath {
        pub qself: Option<QSelf>,
        pub path: Path,
    }
}
ast_struct! {
    pub struct TypePtr {
        pub star_token: Token![*],
        pub const_token: Option<Token![const]>,
        pub mutability: Option<Token![mut]>,
        pub elem: Box<Type>,
    }
}
ast_struct! {
    pub struct TypeReference {
        pub and_token: Token![&],
        pub lifetime: Option<Lifetime>,
        pub mutability: Option<Token![mut]>,
        pub elem: Box<Type>,
    }
}
ast_struct! {
    pub struct TypeSlice {
        pub bracket_token: token::Bracket,
        pub elem: Box<Type>,
    }
}
ast_struct! {
    pub struct TypeTraitObject {
        pub dyn_token: Option<Token![dyn]>,
        pub bounds: Punctuated<TypeParamBound, Token![+]>,
    }
}
ast_struct! {
    pub struct TypeTuple {
        pub paren_token: token::Paren,
        pub elems: Punctuated<Type, Token![,]>,
    }
}
ast_struct! {
    pub struct Abi {
        pub extern_token: Token![extern],
        pub name: Option<LitStr>,
    }
}
ast_struct! {
    pub struct BareFnArg {
        pub attrs: Vec<Attribute>,
        pub name: Option<(Ident, Token![:])>,
        pub ty: Type,
    }
}
ast_struct! {
    pub struct BareVariadic {
        pub attrs: Vec<Attribute>,
        pub name: Option<(Ident, Token![:])>,
        pub dots: Token![...],
        pub comma: Option<Token![,]>,
    }
}
ast_enum! {
    pub enum ReturnType {
        Default,
        Type(Token![->], Box<Type>),
    }
}
