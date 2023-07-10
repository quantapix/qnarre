use super::punct::Punctuated;
use proc_macro2::TokenStream;
use std::mem;

ast_enum_of_structs! {
    pub enum Item {
        Const(Const),
        Enum(Enum),
        ExternCrate(ExternCrate),
        Fn(Fn),
        Foreign(Foreign),
        Impl(Impl),
        Macro(Mac),
        Mod(Mod),
        Static(Static),
        Struct(Struct),
        Trait(Trait),
        TraitAlias(TraitAlias),
        Type(Type),
        Union(Union),
        Use(Use),
        Verbatim(TokenStream),
    }
}
impl Item {
    pub(crate) fn replace_attrs(&mut self, y: Vec<attr::Attr>) -> Vec<attr::Attr> {
        match self {
            Const(Const { attrs, .. })
            | Enum(Enum { attrs, .. })
            | ExternCrate(ExternCrate { attrs, .. })
            | Fn(Fn { attrs, .. })
            | Foreign(Foreign { attrs, .. })
            | Impl(Impl { attrs, .. })
            | Macro(Mac { attrs, .. })
            | Mod(Mod { attrs, .. })
            | Static(Static { attrs, .. })
            | Struct(Struct { attrs, .. })
            | Trait(Trait { attrs, .. })
            | TraitAlias(TraitAlias { attrs, .. })
            | Type(Type { attrs, .. })
            | Union(Union { attrs, .. })
            | Use(Use { attrs, .. }) => mem::replace(attrs, y),
            Verbatim(_) => Vec::new(),
        }
    }
}
impl From<DeriveInput> for Item {
    fn from(x: DeriveInput) -> Item {
        match x.data {
            Data::Struct(y) => Item::Struct(Struct {
                attrs: x.attrs,
                vis: x.vis,
                struct_: y.struct_,
                ident: x.ident,
                gens: x.gens,
                fields: y.fields,
                semi: y.semi,
            }),
            Data::Enum(y) => Item::Enum(Enum {
                attrs: x.attrs,
                vis: x.vis,
                enum_: y.enum_,
                ident: x.ident,
                gens: x.gens,
                brace: y.brace,
                elems: y.variants,
            }),
            Data::Union(y) => Item::Union(Union {
                attrs: x.attrs,
                vis: x.vis,
                union_: y.union_,
                ident: x.ident,
                gens: x.gens,
                fields: y.fields,
            }),
        }
    }
}

pub struct Const {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub const_: Token![const],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub colon: Token![:],
    pub typ: Box<ty::Type>,
    pub eq: Token![=],
    pub expr: Box<Expr>,
    pub semi: Token![;],
}

pub struct Enum {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub enum_: Token![enum],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub brace: tok::Brace,
    pub elems: Punctuated<Variant, Token![,]>,
}
impl From<Enum> for DeriveInput {
    fn from(x: Enum) -> DeriveInput {
        DeriveInput {
            attrs: x.attrs,
            vis: x.vis,
            ident: x.ident,
            gens: x.gens,
            data: Data::Enum(DataEnum {
                enum_: x.enum_,
                brace: x.brace,
                variants: x.elems,
            }),
        }
    }
}

pub struct ExternCrate {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub extern_: Token![extern],
    pub crate_: Token![crate],
    pub ident: Ident,
    pub rename: Option<(Token![as], Ident)>,
    pub semi: Token![;],
}
pub struct Fn {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub sig: Sig,
    pub block: Box<Block>,
}
pub struct Foreign {
    pub attrs: Vec<attr::Attr>,
    pub unsafe_: Option<Token![unsafe]>,
    pub abi: Abi,
    pub brace: tok::Brace,
    pub items: Vec<Foreign::Item>,
}
pub struct Impl {
    pub attrs: Vec<attr::Attr>,
    pub default_: Option<Token![default]>,
    pub unsafe_: Option<Token![unsafe]>,
    pub impl_: Token![impl],
    pub gens: gen::Gens,
    pub trait_: Option<(Option<Token![!]>, Path, Token![for])>,
    pub typ: Box<ty::Type>,
    pub brace: tok::Brace,
    pub items: Vec<Impl::Item>,
}
pub struct Mac {
    pub attrs: Vec<attr::Attr>,
    pub ident: Option<Ident>,
    pub mac: Macro,
    pub semi: Option<Token![;]>,
}
pub struct Mod {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub unsafe_: Option<Token![unsafe]>,
    pub mod_: Token![mod],
    pub ident: Ident,
    pub gist: Option<(tok::Brace, Vec<Item>)>,
    pub semi: Option<Token![;]>,
}
pub struct Static {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub static_: Token![static],
    pub mut_: StaticMut,
    pub ident: Ident,
    pub colon: Token![:],
    pub typ: Box<ty::Type>,
    pub eq: Token![=],
    pub expr: Box<Expr>,
    pub semi: Token![;],
}

pub struct Struct {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub struct_: Token![struct],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub fields: Fields,
    pub semi: Option<Token![;]>,
}
impl From<Struct> for DeriveInput {
    fn from(x: Struct) -> DeriveInput {
        DeriveInput {
            attrs: x.attrs,
            vis: x.vis,
            ident: x.ident,
            gens: x.gens,
            data: Data::Struct(DataStruct {
                struct_: x.struct_,
                fields: x.fields,
                semi: x.semi,
            }),
        }
    }
}

pub struct Trait {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub unsafe_: Option<Token![unsafe]>,
    pub auto_: Option<Token![auto]>,
    pub restriction: Option<Impl::Restriction>,
    pub trait_: Token![trait],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub colon: Option<Token![:]>,
    pub supers: Punctuated<gen::bound::Type, Token![+]>,
    pub brace: tok::Brace,
    pub items: Vec<Trait::Item>,
}
pub struct TraitAlias {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub trait_: Token![trait],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub eq: Token![=],
    pub bounds: Punctuated<gen::bound::Type, Token![+]>,
    pub semi: Token![;],
}
pub struct Type {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub type_: Token![type],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub eq: Token![=],
    pub typ: Box<ty::Type>,
    pub semi: Token![;],
}

pub struct Union {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub union_: Token![union],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub fields: FieldsNamed,
}
impl From<Union> for DeriveInput {
    fn from(x: Union) -> DeriveInput {
        DeriveInput {
            attrs: x.attrs,
            vis: x.vis,
            ident: x.ident,
            gens: x.gens,
            data: Data::Union(DataUnion {
                union_: x.union_,
                fields: x.fields,
            }),
        }
    }
}

pub struct Use {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub use_: Token![use],
    pub colon: Option<Token![::]>,
    pub tree: Use::Tree,
    pub semi: Token![;],
}

pub struct Receiver {
    pub attrs: Vec<attr::Attr>,
    pub ref_: Option<(Token![&], Option<Lifetime>)>,
    pub mut_: Option<Token![mut]>,
    pub self_: Token![self],
    pub colon: Option<Token![:]>,
    pub typ: Box<ty::Type>,
}
impl Receiver {
    pub fn lifetime(&self) -> Option<&Lifetime> {
        self.ref_.as_ref()?.1.as_ref()
    }
}

ast_enum_of_structs! {
    pub enum FnArg {
        Receiver(Receiver),
        Typed(patt::Type),
    }
}

pub struct Sig {
    pub constness: Option<Token![const]>,
    pub async_: Option<Token![async]>,
    pub unsafe_: Option<Token![unsafe]>,
    pub abi: Option<Abi>,
    pub fn_: Token![fn],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub paren: tok::Paren,
    pub args: Punctuated<FnArg, Token![,]>,
    pub vari: Option<Variadic>,
    pub ret: ty::Ret,
}
impl Sig {
    pub fn receiver(&self) -> Option<&Receiver> {
        let y = self.args.first()?;
        match y {
            FnArg::Receiver(x) => Some(x),
            FnArg::Typed(_) => None,
        }
    }
}

pub struct Variadic {
    pub attrs: Vec<attr::Attr>,
    pub pat: Option<(Box<patt::Patt>, Token![:])>,
    pub dots: Token![...],
    pub comma: Option<Token![,]>,
}
pub enum StaticMut {
    Mut(Token![mut]),
    None,
}

pub mod Foreign {
    ast_enum_of_structs! {
        pub enum Item {
            Fn(Fn),
            Static(Static),
            Type(Type),
            Macro(Mac),
            Verbatim(TokenStream),
        }
    }
    pub struct Fn {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub sig: Sig,
        pub semi: Token![;],
    }
    pub struct Static {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub static_: Token![static],
        pub mut_: StaticMut,
        pub ident: Ident,
        pub colon: Token![:],
        pub typ: Box<ty::Type>,
        pub semi: Token![;],
    }
    pub struct Type {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub type_: Token![type],
        pub ident: Ident,
        pub gens: gen::Gens,
        pub semi: Token![;],
    }
    pub struct Mac {
        pub attrs: Vec<attr::Attr>,
        pub mac: Macro,
        pub semi: Option<Token![;]>,
    }
}
pub mod Impl {
    ast_enum_of_structs! {
        pub enum Item {
            Const(Const),
            Fn(Fn),
            Type(Type),
            Mac(Mac),
            Verbatim(TokenStream),
        }
    }
    pub struct Const {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub default_: Option<Token![default]>,
        pub const_: Token![const],
        pub ident: Ident,
        pub gens: gen::Gens,
        pub colon: Token![:],
        pub typ: ty::Type,
        pub eq: Token![=],
        pub expr: Expr,
        pub semi: Token![;],
    }
    pub struct Fn {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub default_: Option<Token![default]>,
        pub sig: Sig,
        pub block: Block,
    }
    pub struct Type {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub default_: Option<Token![default]>,
        pub type_: Token![type],
        pub ident: Ident,
        pub gens: gen::Gens,
        pub eq: Token![=],
        pub typ: ty::Type,
        pub semi: Token![;],
    }
    pub struct Mac {
        pub attrs: Vec<attr::Attr>,
        pub mac: Macro,
        pub semi: Option<Token![;]>,
    }
    pub enum Restriction {}
}
pub mod Trait {
    ast_enum_of_structs! {
        pub enum Item {
            Const(Const),
            Fn(Fn),
            Type(Type),
            Mac(Mac),
            Verbatim(TokenStream),
        }
    }
    pub struct Const {
        pub attrs: Vec<attr::Attr>,
        pub const_: Token![const],
        pub ident: Ident,
        pub gens: gen::Gens,
        pub colon: Token![:],
        pub typ: ty::Type,
        pub default: Option<(Token![=], Expr)>,
        pub semi: Token![;],
    }
    pub struct Fn {
        pub attrs: Vec<attr::Attr>,
        pub sig: Sig,
        pub default: Option<Block>,
        pub semi: Option<Token![;]>,
    }
    pub struct Type {
        pub attrs: Vec<attr::Attr>,
        pub type_: Token![type],
        pub ident: Ident,
        pub gens: gen::Gens,
        pub colon: Option<Token![:]>,
        pub bounds: Punctuated<gen::bound::Type, Token![+]>,
        pub default: Option<(Token![=], ty::Type)>,
        pub semi: Token![;],
    }
    pub struct Mac {
        pub attrs: Vec<attr::Attr>,
        pub mac: Macro,
        pub semi: Option<Token![;]>,
    }
}
pub mod Use {
    ast_enum_of_structs! {
        pub enum Tree {
            Path(Path),
            Name(Name),
            Rename(Rename),
            Glob(Glob),
            Group(Group),
        }
    }
    pub struct Path {
        pub ident: Ident,
        pub colon2: Token![::],
        pub tree: Box<Tree>,
    }
    pub struct Name {
        pub ident: Ident,
    }
    pub struct Rename {
        pub ident: Ident,
        pub as_: Token![as],
        pub rename: Ident,
    }
    pub struct Glob {
        pub star: Token![*],
    }
    pub struct Group {
        pub brace: tok::Brace,
        pub elems: Punctuated<Tree, Token![,]>,
    }
}
