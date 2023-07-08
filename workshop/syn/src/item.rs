use super::*;
use crate::punct::Punctuated;
use proc_macro2::TokenStream;
use std::mem;

ast_enum_of_structs! {
    pub enum Item {
        Const(ItemConst),
        Enum(ItemEnum),
        ExternCrate(ItemExternCrate),
        Fn(ItemFn),
        ForeignMod(ItemForeignMod),
        Impl(ItemImpl),
        Macro(ItemMacro),
        Mod(ItemMod),
        Static(ItemStatic),
        Struct(ItemStruct),
        Trait(ItemTrait),
        TraitAlias(ItemTraitAlias),
        Type(ItemType),
        Union(ItemUnion),
        Use(ItemUse),
        Verbatim(TokenStream),
    }
}
ast_struct! {
    pub struct ItemConst {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub const_: Token![const],
        pub ident: Ident,
        pub gens: Generics,
        pub colon: Token![:],
        pub ty: Box<Ty>,
        pub eq: Token![=],
        pub expr: Box<Expr>,
        pub semi: Token![;],
    }
}
ast_struct! {
    pub struct ItemEnum {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub enum_: Token![enum],
        pub ident: Ident,
        pub gens: Generics,
        pub brace: tok::Brace,
        pub variants: Punctuated<Variant, Token![,]>,
    }
}
ast_struct! {
    pub struct ItemExternCrate {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub extern_: Token![extern],
        pub crate_: Token![crate],
        pub ident: Ident,
        pub rename: Option<(Token![as], Ident)>,
        pub semi: Token![;],
    }
}
ast_struct! {
    pub struct ItemFn {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub sig: Signature,
        pub block: Box<Block>,
    }
}
ast_struct! {
    pub struct ItemForeignMod {
        pub attrs: Vec<Attribute>,
        pub unsafe_: Option<Token![unsafe]>,
        pub abi: Abi,
        pub brace: tok::Brace,
        pub items: Vec<ForeignItem>,
    }
}
ast_struct! {
    pub struct ItemImpl {
        pub attrs: Vec<Attribute>,
        pub default_: Option<Token![default]>,
        pub unsafe_: Option<Token![unsafe]>,
        pub impl_: Token![impl],
        pub gens: Generics,
        pub trait_: Option<(Option<Token![!]>, Path, Token![for])>,
        pub self_ty: Box<Ty>,
        pub brace: tok::Brace,
        pub items: Vec<ImplItem>,
    }
}
ast_struct! {
    pub struct ItemMacro {
        pub attrs: Vec<Attribute>,
        pub ident: Option<Ident>,
        pub mac: Macro,
        pub semi: Option<Token![;]>,
    }
}
ast_struct! {
    pub struct ItemMod {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub unsafe_: Option<Token![unsafe]>,
        pub mod_: Token![mod],
        pub ident: Ident,
        pub gist: Option<(tok::Brace, Vec<Item>)>,
        pub semi: Option<Token![;]>,
    }
}
ast_struct! {
    pub struct ItemStatic {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub static_: Token![static],
        pub mut_: StaticMut,
        pub ident: Ident,
        pub colon: Token![:],
        pub ty: Box<Ty>,
        pub eq: Token![=],
        pub expr: Box<Expr>,
        pub semi: Token![;],
    }
}
ast_struct! {
    pub struct ItemStruct {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub struct_: Token![struct],
        pub ident: Ident,
        pub gens: Generics,
        pub fields: Fields,
        pub semi: Option<Token![;]>,
    }
}
ast_struct! {
    pub struct ItemTrait {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub unsafe_: Option<Token![unsafe]>,
        pub auto_: Option<Token![auto]>,
        pub restriction: Option<ImplRestriction>,
        pub trait_: Token![trait],
        pub ident: Ident,
        pub gens: Generics,
        pub colon: Option<Token![:]>,
        pub supertraits: Punctuated<TypeParamBound, Token![+]>,
        pub brace: tok::Brace,
        pub items: Vec<TraitItem>,
    }
}
ast_struct! {
    pub struct ItemTraitAlias {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub trait_: Token![trait],
        pub ident: Ident,
        pub gens: Generics,
        pub eq: Token![=],
        pub bounds: Punctuated<TypeParamBound, Token![+]>,
        pub semi: Token![;],
    }
}
ast_struct! {
    pub struct ItemType {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub type_: Token![type],
        pub ident: Ident,
        pub gens: Generics,
        pub eq: Token![=],
        pub ty: Box<Ty>,
        pub semi: Token![;],
    }
}
ast_struct! {
    pub struct ItemUnion {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub union_: Token![union],
        pub ident: Ident,
        pub gens: Generics,
        pub fields: FieldsNamed,
    }
}
ast_struct! {
    pub struct ItemUse {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub use_: Token![use],
        pub leading_colon: Option<Token![::]>,
        pub tree: UseTree,
        pub semi: Token![;],
    }
}
impl Item {
    pub(crate) fn replace_attrs(&mut self, new: Vec<Attribute>) -> Vec<Attribute> {
        match self {
            Item::Const(ItemConst { attrs, .. })
            | Item::Enum(ItemEnum { attrs, .. })
            | Item::ExternCrate(ItemExternCrate { attrs, .. })
            | Item::Fn(ItemFn { attrs, .. })
            | Item::ForeignMod(ItemForeignMod { attrs, .. })
            | Item::Impl(ItemImpl { attrs, .. })
            | Item::Macro(ItemMacro { attrs, .. })
            | Item::Mod(ItemMod { attrs, .. })
            | Item::Static(ItemStatic { attrs, .. })
            | Item::Struct(ItemStruct { attrs, .. })
            | Item::Trait(ItemTrait { attrs, .. })
            | Item::TraitAlias(ItemTraitAlias { attrs, .. })
            | Item::Type(ItemType { attrs, .. })
            | Item::Union(ItemUnion { attrs, .. })
            | Item::Use(ItemUse { attrs, .. }) => mem::replace(attrs, new),
            Item::Verbatim(_) => Vec::new(),
        }
    }
}
impl From<DeriveInput> for Item {
    fn from(x: DeriveInput) -> Item {
        match x.data {
            Data::Struct(y) => Item::Struct(ItemStruct {
                attrs: x.attrs,
                vis: x.vis,
                struct_: y.struct_,
                ident: x.ident,
                gens: x.gens,
                fields: y.fields,
                semi: y.semi,
            }),
            Data::Enum(y) => Item::Enum(ItemEnum {
                attrs: x.attrs,
                vis: x.vis,
                enum_: y.enum_,
                ident: x.ident,
                gens: x.gens,
                brace: y.brace,
                variants: y.variants,
            }),
            Data::Union(y) => Item::Union(ItemUnion {
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
impl From<ItemStruct> for DeriveInput {
    fn from(x: ItemStruct) -> DeriveInput {
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
impl From<ItemEnum> for DeriveInput {
    fn from(x: ItemEnum) -> DeriveInput {
        DeriveInput {
            attrs: x.attrs,
            vis: x.vis,
            ident: x.ident,
            gens: x.gens,
            data: Data::Enum(DataEnum {
                enum_: x.enum_,
                brace: x.brace,
                variants: x.variants,
            }),
        }
    }
}
impl From<ItemUnion> for DeriveInput {
    fn from(x: ItemUnion) -> DeriveInput {
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
ast_enum_of_structs! {
    pub enum UseTree {
        Path(UsePath),
        Name(UseName),
        Rename(UseRename),
        Glob(UseGlob),
        Group(UseGroup),
    }
}
ast_struct! {
    pub struct UsePath {
        pub ident: Ident,
        pub colon2: Token![::],
        pub tree: Box<UseTree>,
    }
}
ast_struct! {
    pub struct UseName {
        pub ident: Ident,
    }
}
ast_struct! {
    pub struct UseRename {
        pub ident: Ident,
        pub as_: Token![as],
        pub rename: Ident,
    }
}
ast_struct! {
    pub struct UseGlob {
        pub star: Token![*],
    }
}
ast_struct! {
    pub struct UseGroup {
        pub brace: tok::Brace,
        pub items: Punctuated<UseTree, Token![,]>,
    }
}
ast_enum_of_structs! {
    pub enum ForeignItem {
        Fn(ForeignItemFn),
        Static(ForeignItemStatic),
        Type(ForeignItemType),
        Macro(ForeignItemMacro),
        Verbatim(TokenStream),
    }
}
ast_struct! {
    pub struct ForeignItemFn {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub sig: Signature,
        pub semi: Token![;],
    }
}
ast_struct! {
    pub struct ForeignItemStatic {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub static_: Token![static],
        pub mut_: StaticMut,
        pub ident: Ident,
        pub colon: Token![:],
        pub ty: Box<Ty>,
        pub semi: Token![;],
    }
}
ast_struct! {
    pub struct ForeignItemType {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub type_: Token![type],
        pub ident: Ident,
        pub gens: Generics,
        pub semi: Token![;],
    }
}
ast_struct! {
    pub struct ForeignItemMacro {
        pub attrs: Vec<Attribute>,
        pub mac: Macro,
        pub semi: Option<Token![;]>,
    }
}
ast_enum_of_structs! {
    pub enum TraitItem {
        Const(TraitItemConst),
        Fn(TraitItemFn),
        Type(TraitItemType),
        Macro(TraitItemMacro),
        Verbatim(TokenStream),
    }
}
ast_struct! {
    pub struct TraitItemConst {
        pub attrs: Vec<Attribute>,
        pub const_: Token![const],
        pub ident: Ident,
        pub gens: Generics,
        pub colon: Token![:],
        pub ty: Ty,
        pub default: Option<(Token![=], Expr)>,
        pub semi: Token![;],
    }
}
ast_struct! {
    pub struct TraitItemFn {
        pub attrs: Vec<Attribute>,
        pub sig: Signature,
        pub default: Option<Block>,
        pub semi: Option<Token![;]>,
    }
}
ast_struct! {
    pub struct TraitItemType {
        pub attrs: Vec<Attribute>,
        pub type_: Token![type],
        pub ident: Ident,
        pub gens: Generics,
        pub colon: Option<Token![:]>,
        pub bounds: Punctuated<TypeParamBound, Token![+]>,
        pub default: Option<(Token![=], Ty)>,
        pub semi: Token![;],
    }
}
ast_struct! {
    pub struct TraitItemMacro {
        pub attrs: Vec<Attribute>,
        pub mac: Macro,
        pub semi: Option<Token![;]>,
    }
}
ast_enum_of_structs! {
    pub enum ImplItem {
        Const(ImplItemConst),
        Fn(ImplItemFn),
        Type(ImplItemType),
        Macro(ImplItemMacro),
        Verbatim(TokenStream),
    }
}
ast_struct! {
    pub struct ImplItemConst {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub default_: Option<Token![default]>,
        pub const_: Token![const],
        pub ident: Ident,
        pub gens: Generics,
        pub colon: Token![:],
        pub ty: Ty,
        pub eq: Token![=],
        pub expr: Expr,
        pub semi: Token![;],
    }
}
ast_struct! {
    pub struct ImplItemFn {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub default_: Option<Token![default]>,
        pub sig: Signature,
        pub block: Block,
    }
}
ast_struct! {
    pub struct ImplItemType {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub default_: Option<Token![default]>,
        pub type_: Token![type],
        pub ident: Ident,
        pub gens: Generics,
        pub eq: Token![=],
        pub ty: Ty,
        pub semi: Token![;],
    }
}
ast_struct! {
    pub struct ImplItemMacro {
        pub attrs: Vec<Attribute>,
        pub mac: Macro,
        pub semi: Option<Token![;]>,
    }
}
ast_struct! {
    pub struct Signature {
        pub constness: Option<Token![const]>,
        pub async_: Option<Token![async]>,
        pub unsafe_: Option<Token![unsafe]>,
        pub abi: Option<Abi>,
        pub fn_: Token![fn],
        pub ident: Ident,
        pub gens: Generics,
        pub paren: tok::Paren,
        pub args: Punctuated<FnArg, Token![,]>,
        pub vari: Option<Variadic>,
        pub ret: ty::Ret,
    }
}
impl Signature {
    pub fn receiver(&self) -> Option<&Receiver> {
        let arg = self.args.first()?;
        match arg {
            FnArg::Receiver(receiver) => Some(receiver),
            FnArg::Typed(_) => None,
        }
    }
}
ast_enum_of_structs! {
    pub enum FnArg {
        Receiver(Receiver),
        Typed(PatType),
    }
}
ast_struct! {
    pub struct Receiver {
        pub attrs: Vec<Attribute>,
        pub reference: Option<(Token![&], Option<Lifetime>)>,
        pub mut_: Option<Token![mut]>,
        pub self_: Token![self],
        pub colon: Option<Token![:]>,
        pub ty: Box<Ty>,
    }
}
impl Receiver {
    pub fn lifetime(&self) -> Option<&Lifetime> {
        self.reference.as_ref()?.1.as_ref()
    }
}
ast_struct! {
    pub struct Variadic {
        pub attrs: Vec<Attribute>,
        pub pat: Option<(Box<Pat>, Token![:])>,
        pub dots: Token![...],
        pub comma: Option<Token![,]>,
    }
}
ast_enum! {
    pub enum StaticMut {
        Mut(Token![mut]),
        None,
    }
}
ast_enum! {
    pub enum ImplRestriction {}
}
