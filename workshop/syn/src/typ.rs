pub use pm2::Stream;
ast_enum_of_structs! {
    pub enum Type {
        Array(Array),
        Fn(Fn),
        Group(Group),
        Impl(Impl),
        Infer(Infer),
        Mac(Mac),
        Never(Never),
        Paren(Paren),
        Path(Path),
        Ptr(Ptr),
        Ref(Ref),
        Slice(Slice),
        Trait(Trait),
        Tuple(Tuple),
        Verbatim(Stream),
    }
}
pub struct Array {
    pub bracket: tok::Bracket,
    pub elem: Box<Type>,
    pub semi: Token![;],
    pub len: Expr,
}
pub struct Fn {
    pub lifes: Option<gen::bound::Lifes>,
    pub unsafe_: Option<Token![unsafe]>,
    pub abi: Option<Abi>,
    pub fn_: Token![fn],
    pub paren: tok::Paren,
    pub args: Punctuated<FnArg, Token![,]>,
    pub vari: Option<Variadic>,
    pub ret: Ret,
}
pub struct Group {
    pub group: tok::Group,
    pub elem: Box<Type>,
}
pub struct Impl {
    pub impl_: Token![impl],
    pub bounds: Punctuated<gen::bound::Type, Token![+]>,
}
pub struct Infer {
    pub underscore: Token![_],
}
pub struct Mac {
    pub mac: mac::Mac,
}
pub struct Never {
    pub bang: Token![!],
}
pub struct Paren {
    pub paren: tok::Paren,
    pub elem: Box<Type>,
}
pub struct Path {
    pub qself: Option<QSelf>,
    pub path: Path,
}
pub struct Ptr {
    pub star: Token![*],
    pub const_: Option<Token![const]>,
    pub mut_: Option<Token![mut]>,
    pub elem: Box<Type>,
}
pub struct Ref {
    pub and: Token![&],
    pub life: Option<Lifetime>,
    pub mut_: Option<Token![mut]>,
    pub elem: Box<Type>,
}
pub struct Slice {
    pub bracket: tok::Bracket,
    pub elem: Box<Type>,
}
pub struct Trait {
    pub dyn_: Option<Token![dyn]>,
    pub bounds: Punctuated<gen::bound::Type, Token![+]>,
}
pub struct Tuple {
    pub paren: tok::Paren,
    pub elems: Punctuated<Type, Token![,]>,
}
pub struct Abi {
    pub extern_: Token![extern],
    pub name: Option<lit::Str>,
}
pub struct FnArg {
    pub attrs: Vec<attr::Attr>,
    pub name: Option<(Ident, Token![:])>,
    pub ty: Type,
}
pub struct Variadic {
    pub attrs: Vec<attr::Attr>,
    pub name: Option<(Ident, Token![:])>,
    pub dots: Token![...],
    pub comma: Option<Token![,]>,
}
pub enum Ret {
    Default,
    Type(Token![->], Box<Type>),
}
