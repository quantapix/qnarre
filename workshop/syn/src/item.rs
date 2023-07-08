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
        pub const_token: Token![const],
        pub ident: Ident,
        pub generics: Generics,
        pub colon_token: Token![:],
        pub ty: Box<Ty>,
        pub eq_token: Token![=],
        pub expr: Box<Expr>,
        pub semi_token: Token![;],
    }
}
ast_struct! {
    pub struct ItemEnum {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub enum_token: Token![enum],
        pub ident: Ident,
        pub generics: Generics,
        pub brace_token: tok::Brace,
        pub variants: Punctuated<Variant, Token![,]>,
    }
}
ast_struct! {
    pub struct ItemExternCrate {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub extern_token: Token![extern],
        pub crate_token: Token![crate],
        pub ident: Ident,
        pub rename: Option<(Token![as], Ident)>,
        pub semi_token: Token![;],
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
        pub unsafety: Option<Token![unsafe]>,
        pub abi: Abi,
        pub brace_token: tok::Brace,
        pub items: Vec<ForeignItem>,
    }
}
ast_struct! {
    pub struct ItemImpl {
        pub attrs: Vec<Attribute>,
        pub defaultness: Option<Token![default]>,
        pub unsafety: Option<Token![unsafe]>,
        pub impl_token: Token![impl],
        pub generics: Generics,
        pub trait_: Option<(Option<Token![!]>, Path, Token![for])>,
        pub self_ty: Box<Ty>,
        pub brace_token: tok::Brace,
        pub items: Vec<ImplItem>,
    }
}
ast_struct! {
    pub struct ItemMacro {
        pub attrs: Vec<Attribute>,
        pub ident: Option<Ident>,
        pub mac: Macro,
        pub semi_token: Option<Token![;]>,
    }
}
ast_struct! {
    pub struct ItemMod {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub unsafety: Option<Token![unsafe]>,
        pub mod_token: Token![mod],
        pub ident: Ident,
        pub content: Option<(tok::Brace, Vec<Item>)>,
        pub semi: Option<Token![;]>,
    }
}
ast_struct! {
    pub struct ItemStatic {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub static_token: Token![static],
        pub mutability: StaticMutability,
        pub ident: Ident,
        pub colon_token: Token![:],
        pub ty: Box<Ty>,
        pub eq_token: Token![=],
        pub expr: Box<Expr>,
        pub semi_token: Token![;],
    }
}
ast_struct! {
    pub struct ItemStruct {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub struct_token: Token![struct],
        pub ident: Ident,
        pub generics: Generics,
        pub fields: Fields,
        pub semi_token: Option<Token![;]>,
    }
}
ast_struct! {
    pub struct ItemTrait {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub unsafety: Option<Token![unsafe]>,
        pub auto_token: Option<Token![auto]>,
        pub restriction: Option<ImplRestriction>,
        pub trait_token: Token![trait],
        pub ident: Ident,
        pub generics: Generics,
        pub colon_token: Option<Token![:]>,
        pub supertraits: Punctuated<TypeParamBound, Token![+]>,
        pub brace_token: tok::Brace,
        pub items: Vec<TraitItem>,
    }
}
ast_struct! {
    pub struct ItemTraitAlias {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub trait_token: Token![trait],
        pub ident: Ident,
        pub generics: Generics,
        pub eq_token: Token![=],
        pub bounds: Punctuated<TypeParamBound, Token![+]>,
        pub semi_token: Token![;],
    }
}
ast_struct! {
    pub struct ItemType {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub type_token: Token![type],
        pub ident: Ident,
        pub generics: Generics,
        pub eq_token: Token![=],
        pub ty: Box<Ty>,
        pub semi_token: Token![;],
    }
}
ast_struct! {
    pub struct ItemUnion {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub union_token: Token![union],
        pub ident: Ident,
        pub generics: Generics,
        pub fields: FieldsNamed,
    }
}
ast_struct! {
    pub struct ItemUse {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub use_token: Token![use],
        pub leading_colon: Option<Token![::]>,
        pub tree: UseTree,
        pub semi_token: Token![;],
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
    fn from(input: DeriveInput) -> Item {
        match input.data {
            Data::Struct(data) => Item::Struct(ItemStruct {
                attrs: input.attrs,
                vis: input.vis,
                struct_token: data.struct_,
                ident: input.ident,
                generics: input.generics,
                fields: data.fields,
                semi_token: data.semi,
            }),
            Data::Enum(data) => Item::Enum(ItemEnum {
                attrs: input.attrs,
                vis: input.vis,
                enum_token: data.enum_,
                ident: input.ident,
                generics: input.generics,
                brace_token: data.brace,
                variants: data.variants,
            }),
            Data::Union(data) => Item::Union(ItemUnion {
                attrs: input.attrs,
                vis: input.vis,
                union_token: data.union_,
                ident: input.ident,
                generics: input.generics,
                fields: data.fields,
            }),
        }
    }
}
impl From<ItemStruct> for DeriveInput {
    fn from(input: ItemStruct) -> DeriveInput {
        DeriveInput {
            attrs: input.attrs,
            vis: input.vis,
            ident: input.ident,
            generics: input.generics,
            data: Data::Struct(DataStruct {
                struct_: input.struct_token,
                fields: input.fields,
                semi: input.semi_token,
            }),
        }
    }
}
impl From<ItemEnum> for DeriveInput {
    fn from(input: ItemEnum) -> DeriveInput {
        DeriveInput {
            attrs: input.attrs,
            vis: input.vis,
            ident: input.ident,
            generics: input.generics,
            data: Data::Enum(DataEnum {
                enum_: input.enum_token,
                brace: input.brace_token,
                variants: input.variants,
            }),
        }
    }
}
impl From<ItemUnion> for DeriveInput {
    fn from(input: ItemUnion) -> DeriveInput {
        DeriveInput {
            attrs: input.attrs,
            vis: input.vis,
            ident: input.ident,
            generics: input.generics,
            data: Data::Union(DataUnion {
                union_: input.union_token,
                fields: input.fields,
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
        pub colon2_token: Token![::],
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
        pub as_token: Token![as],
        pub rename: Ident,
    }
}
ast_struct! {
    pub struct UseGlob {
        pub star_token: Token![*],
    }
}
ast_struct! {
    pub struct UseGroup {
        pub brace_token: tok::Brace,
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
        pub semi_token: Token![;],
    }
}
ast_struct! {
    pub struct ForeignItemStatic {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub static_token: Token![static],
        pub mutability: StaticMutability,
        pub ident: Ident,
        pub colon_token: Token![:],
        pub ty: Box<Ty>,
        pub semi_token: Token![;],
    }
}
ast_struct! {
    pub struct ForeignItemType {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub type_token: Token![type],
        pub ident: Ident,
        pub generics: Generics,
        pub semi_token: Token![;],
    }
}
ast_struct! {
    pub struct ForeignItemMacro {
        pub attrs: Vec<Attribute>,
        pub mac: Macro,
        pub semi_token: Option<Token![;]>,
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
        pub const_token: Token![const],
        pub ident: Ident,
        pub generics: Generics,
        pub colon_token: Token![:],
        pub ty: Ty,
        pub default: Option<(Token![=], Expr)>,
        pub semi_token: Token![;],
    }
}
ast_struct! {
    pub struct TraitItemFn {
        pub attrs: Vec<Attribute>,
        pub sig: Signature,
        pub default: Option<Block>,
        pub semi_token: Option<Token![;]>,
    }
}
ast_struct! {
    pub struct TraitItemType {
        pub attrs: Vec<Attribute>,
        pub type_token: Token![type],
        pub ident: Ident,
        pub generics: Generics,
        pub colon_token: Option<Token![:]>,
        pub bounds: Punctuated<TypeParamBound, Token![+]>,
        pub default: Option<(Token![=], Ty)>,
        pub semi_token: Token![;],
    }
}
ast_struct! {
    pub struct TraitItemMacro {
        pub attrs: Vec<Attribute>,
        pub mac: Macro,
        pub semi_token: Option<Token![;]>,
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
        pub defaultness: Option<Token![default]>,
        pub const_token: Token![const],
        pub ident: Ident,
        pub generics: Generics,
        pub colon_token: Token![:],
        pub ty: Ty,
        pub eq_token: Token![=],
        pub expr: Expr,
        pub semi_token: Token![;],
    }
}
ast_struct! {
    pub struct ImplItemFn {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub defaultness: Option<Token![default]>,
        pub sig: Signature,
        pub block: Block,
    }
}
ast_struct! {
    pub struct ImplItemType {
        pub attrs: Vec<Attribute>,
        pub vis: Visibility,
        pub defaultness: Option<Token![default]>,
        pub type_token: Token![type],
        pub ident: Ident,
        pub generics: Generics,
        pub eq_token: Token![=],
        pub ty: Ty,
        pub semi_token: Token![;],
    }
}
ast_struct! {
    pub struct ImplItemMacro {
        pub attrs: Vec<Attribute>,
        pub mac: Macro,
        pub semi_token: Option<Token![;]>,
    }
}
ast_struct! {
    pub struct Signature {
        pub constness: Option<Token![const]>,
        pub asyncness: Option<Token![async]>,
        pub unsafety: Option<Token![unsafe]>,
        pub abi: Option<Abi>,
        pub fn_token: Token![fn],
        pub ident: Ident,
        pub generics: Generics,
        pub paren_token: tok::Paren,
        pub inputs: Punctuated<FnArg, Token![,]>,
        pub variadic: Option<Variadic>,
        pub output: ReturnType,
    }
}
impl Signature {
    pub fn receiver(&self) -> Option<&Receiver> {
        let arg = self.inputs.first()?;
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
        pub mutability: Option<Token![mut]>,
        pub self_token: Token![self],
        pub colon_token: Option<Token![:]>,
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
    pub enum StaticMutability {
        Mut(Token![mut]),
        None,
    }
}
ast_enum! {
    pub enum ImplRestriction {}
}
