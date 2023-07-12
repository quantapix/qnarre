pub use pm2::Stream;

ast_enum_of_structs! {
    pub enum Item {
        Const(Const),
        Enum(Enum),
        ExternCrate(ExternCrate),
        Fn(Fn),
        Foreign(Foreign),
        Impl(Impl),
        Mac(Mac),
        Mod(Mod),
        Static(Static),
        Struct(Struct),
        Trait(Trait),
        TraitAlias(TraitAlias),
        Type(Type),
        Union(Union),
        Use(Use),
        Verbatim(Stream),
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
            | Mac(Mac { attrs, .. })
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
impl Parse for Item {
    fn parse(x: Stream) -> Res<Self> {
        let beg = x.fork();
        let attrs = x.call(attr::Attr::parse_outer)?;
        parse_rest_of_item(beg, attrs, x)
    }
}

pub struct Const {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub const_: Token![const],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub colon: Token![:],
    pub typ: Box<typ::Type>,
    pub eq: Token![=],
    pub expr: Box<Expr>,
    pub semi: Token![;],
}
impl Parse for Const {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Const {
            attrs: x.call(attr::Attr::parse_outer)?,
            vis: x.parse()?,
            const_: x.parse()?,
            ident: {
                let look = x.lookahead1();
                if look.peek(Ident) || look.peek(Token![_]) {
                    x.call(Ident::parse_any)?
                } else {
                    return Err(look.error());
                }
            },
            gens: gen::Gens::default(),
            colon: x.parse()?,
            typ: x.parse()?,
            eq: x.parse()?,
            expr: x.parse()?,
            semi: x.parse()?,
        })
    }
}

pub struct Enum {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub enum_: Token![enum],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub brace: tok::Brace,
    pub elems: Punctuated<data::Variant, Token![,]>,
}
impl From<Enum> for DeriveInput {
    fn from(x: Enum) -> DeriveInput {
        DeriveInput {
            attrs: x.attrs,
            vis: x.vis,
            ident: x.ident,
            gens: x.gens,
            data: Data::Enum(data::Enum {
                enum_: x.enum_,
                brace: x.brace,
                variants: x.elems,
            }),
        }
    }
}
impl Parse for Enum {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = x.call(attr::Attr::parse_outer)?;
        let vis = x.parse::<Visibility>()?;
        let enum_ = x.parse::<Token![enum]>()?;
        let ident = x.parse::<Ident>()?;
        let gens = x.parse::<gen::Gens>()?;
        let (where_, brace, elems) = data_enum(x)?;
        Ok(Enum {
            attrs,
            vis,
            enum_,
            ident,
            gens: gen::Gens { where_, ..gens },
            brace,
            elems,
        })
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
impl Parse for ExternCrate {
    fn parse(x: Stream) -> Res<Self> {
        Ok(ExternCrate {
            attrs: x.call(attr::Attr::parse_outer)?,
            vis: x.parse()?,
            extern_: x.parse()?,
            crate_: x.parse()?,
            ident: {
                if x.peek(Token![self]) {
                    x.call(Ident::parse_any)?
                } else {
                    x.parse()?
                }
            },
            rename: {
                if x.peek(Token![as]) {
                    let as_: Token![as] = x.parse()?;
                    let y: Ident = if x.peek(Token![_]) {
                        Ident::from(x.parse::<Token![_]>()?)
                    } else {
                        x.parse()?
                    };
                    Some((as_, y))
                } else {
                    None
                }
            },
            semi: x.parse()?,
        })
    }
}

pub struct Fn {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub sig: Sig,
    pub block: Box<Block>,
}
impl Parse for Fn {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = x.call(attr::Attr::parse_outer)?;
        let vis: Visibility = x.parse()?;
        let sig: Sig = x.parse()?;
        parse_rest_of_fn(x, attrs, vis, sig)
    }
}

pub struct Foreign {
    pub attrs: Vec<attr::Attr>,
    pub unsafe_: Option<Token![unsafe]>,
    pub abi: Abi,
    pub brace: tok::Brace,
    pub items: Vec<Foreign::Item>,
}
impl Parse for Foreign {
    fn parse(x: Stream) -> Res<Self> {
        let mut attrs = x.call(attr::Attr::parse_outer)?;
        let unsafe_: Option<Token![unsafe]> = x.parse()?;
        let abi: Abi = x.parse()?;
        let y;
        let brace = braced!(y in x);
        attr::inner(&y, &mut attrs)?;
        let mut items = Vec::new();
        while !y.is_empty() {
            items.push(y.parse()?);
        }
        Ok(Foreign {
            attrs,
            unsafe_,
            abi,
            brace,
            items,
        })
    }
}

pub struct Impl {
    pub attrs: Vec<attr::Attr>,
    pub default_: Option<Token![default]>,
    pub unsafe_: Option<Token![unsafe]>,
    pub impl_: Token![impl],
    pub gens: gen::Gens,
    pub trait_: Option<(Option<Token![!]>, Path, Token![for])>,
    pub typ: Box<typ::Type>,
    pub brace: tok::Brace,
    pub items: Vec<Impl::Item>,
}
impl Parse for Impl {
    fn parse(x: Stream) -> Res<Self> {
        let verbatim = false;
        parse_impl(x, verbatim).map(Option::unwrap)
    }
}

pub struct Mac {
    pub attrs: Vec<attr::Attr>,
    pub ident: Option<Ident>,
    pub mac: mac::Mac,
    pub semi: Option<Token![;]>,
}
impl Parse for Mac {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = x.call(attr::Attr::parse_outer)?;
        let path = x.call(Path::parse_mod_style)?;
        let bang: Token![!] = x.parse()?;
        let ident: Option<Ident> = if x.peek(Token![try]) {
            x.call(Ident::parse_any).map(Some)
        } else {
            x.parse()
        }?;
        let (delim, toks) = x.call(mac::parse_delim)?;
        let semi: Option<Token![;]> = if !delim.is_brace() { Some(x.parse()?) } else { None };
        Ok(Mac {
            attrs,
            ident,
            mac: mac::Mac {
                path,
                bang,
                delim,
                toks,
            },
            semi,
        })
    }
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
impl Parse for Mod {
    fn parse(x: Stream) -> Res<Self> {
        let mut attrs = x.call(attr::Attr::parse_outer)?;
        let vis: Visibility = x.parse()?;
        let unsafe_: Option<Token![unsafe]> = x.parse()?;
        let mod_: Token![mod] = x.parse()?;
        let ident: Ident = if x.peek(Token![try]) {
            x.call(Ident::parse_any)
        } else {
            x.parse()
        }?;
        let look = x.lookahead1();
        if look.peek(Token![;]) {
            Ok(Mod {
                attrs,
                vis,
                unsafe_,
                mod_,
                ident,
                gist: None,
                semi: Some(x.parse()?),
            })
        } else if look.peek(tok::Brace) {
            let y;
            let brace = braced!(y in x);
            attr::inner(&y, &mut attrs)?;
            let mut items = Vec::new();
            while !y.is_empty() {
                items.push(y.parse()?);
            }
            Ok(Mod {
                attrs,
                vis,
                unsafe_,
                mod_,
                ident,
                gist: Some((brace, items)),
                semi: None,
            })
        } else {
            Err(look.error())
        }
    }
}

pub struct Static {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub static_: Token![static],
    pub mut_: StaticMut,
    pub ident: Ident,
    pub colon: Token![:],
    pub typ: Box<typ::Type>,
    pub eq: Token![=],
    pub expr: Box<Expr>,
    pub semi: Token![;],
}
impl Parse for Static {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Static {
            attrs: x.call(attr::Attr::parse_outer)?,
            vis: x.parse()?,
            static_: x.parse()?,
            mut_: x.parse()?,
            ident: x.parse()?,
            colon: x.parse()?,
            typ: x.parse()?,
            eq: x.parse()?,
            expr: x.parse()?,
            semi: x.parse()?,
        })
    }
}

pub struct Struct {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub struct_: Token![struct],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub fields: data::Fields,
    pub semi: Option<Token![;]>,
}
impl From<Struct> for DeriveInput {
    fn from(x: Struct) -> DeriveInput {
        DeriveInput {
            attrs: x.attrs,
            vis: x.vis,
            ident: x.ident,
            gens: x.gens,
            data: Data::Struct(data::Struct {
                struct_: x.struct_,
                fields: x.fields,
                semi: x.semi,
            }),
        }
    }
}
impl Parse for Struct {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = x.call(attr::Attr::parse_outer)?;
        let vis = x.parse::<Visibility>()?;
        let struct_ = x.parse::<Token![struct]>()?;
        let ident = x.parse::<Ident>()?;
        let gens = x.parse::<gen::Gens>()?;
        let (where_, fields, semi) = data_struct(x)?;
        Ok(Struct {
            attrs,
            vis,
            struct_,
            ident,
            gens: gen::Gens { where_, ..gens },
            fields,
            semi,
        })
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
impl Parse for Trait {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = x.call(attr::Attr::parse_outer)?;
        let vis: Visibility = x.parse()?;
        let unsafe_: Option<Token![unsafe]> = x.parse()?;
        let auto_: Option<Token![auto]> = x.parse()?;
        let trait_: Token![trait] = x.parse()?;
        let ident: Ident = x.parse()?;
        let gens: gen::Gens = x.parse()?;
        parse_rest_of_trait(x, attrs, vis, unsafe_, auto_, trait_, ident, gens)
    }
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
impl Parse for TraitAlias {
    fn parse(x: Stream) -> Res<Self> {
        let (attrs, vis, trait_, ident, gens) = parse_start_of_trait_alias(x)?;
        parse_rest_of_trait_alias(x, attrs, vis, trait_, ident, gens)
    }
}

pub struct Type {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub type_: Token![type],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub eq: Token![=],
    pub typ: Box<typ::Type>,
    pub semi: Token![;],
}
impl Parse for Type {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Type {
            attrs: x.call(attr::Attr::parse_outer)?,
            vis: x.parse()?,
            type_: x.parse()?,
            ident: x.parse()?,
            gens: {
                let mut y: gen::Gens = x.parse()?;
                y.where_ = x.parse()?;
                y
            },
            eq: x.parse()?,
            typ: x.parse()?,
            semi: x.parse()?,
        })
    }
}

pub struct Union {
    pub attrs: Vec<attr::Attr>,
    pub vis: Visibility,
    pub union_: Token![union],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub fields: data::Named,
}
impl From<Union> for DeriveInput {
    fn from(x: Union) -> DeriveInput {
        DeriveInput {
            attrs: x.attrs,
            vis: x.vis,
            ident: x.ident,
            gens: x.gens,
            data: Data::Union(data::Union {
                union_: x.union_,
                fields: x.fields,
            }),
        }
    }
}
impl Parse for Union {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = x.call(attr::Attr::parse_outer)?;
        let vis = x.parse::<Visibility>()?;
        let union_ = x.parse::<Token![union]>()?;
        let ident = x.parse::<Ident>()?;
        let gens = x.parse::<gen::Gens>()?;
        let (where_, fields) = data_union(x)?;
        Ok(Union {
            attrs,
            vis,
            union_,
            ident,
            gens: gen::Gens { where_, ..gens },
            fields,
        })
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
impl Parse for Use {
    fn parse(x: Stream) -> Res<Self> {
        let root = false;
        parse_item_use(x, root).map(Option::unwrap)
    }
}
fn parse_item_use(x: Stream, root: bool) -> Res<Option<Use>> {
    let attrs = x.call(attr::Attr::parse_outer)?;
    let vis: Visibility = x.parse()?;
    let use_: Token![use] = x.parse()?;
    let colon: Option<Token![::]> = x.parse()?;
    let tree = Use::parse_tree(x, root && colon.is_none())?;
    let semi: Token![;] = x.parse()?;
    let tree = match tree {
        Some(x) => x,
        None => return Ok(None),
    };
    Ok(Some(Use {
        attrs,
        vis,
        use_,
        colon,
        tree,
        semi,
    }))
}

pub struct Receiver {
    pub attrs: Vec<attr::Attr>,
    pub ref_: Option<(Token![&], Option<Lifetime>)>,
    pub mut_: Option<Token![mut]>,
    pub self_: Token![self],
    pub colon: Option<Token![:]>,
    pub typ: Box<typ::Type>,
}
impl Receiver {
    pub fn lifetime(&self) -> Option<&Lifetime> {
        self.ref_.as_ref()?.1.as_ref()
    }
}
impl Parse for Receiver {
    fn parse(x: Stream) -> Res<Self> {
        let ref_ = if x.peek(Token![&]) {
            let amp: Token![&] = x.parse()?;
            let life: Option<Lifetime> = x.parse()?;
            Some((amp, life))
        } else {
            None
        };
        let mut_: Option<Token![mut]> = x.parse()?;
        let self_: Token![self] = x.parse()?;
        let colon: Option<Token![:]> = if ref_.is_some() { None } else { x.parse()? };
        let typ: typ::Type = if colon.is_some() {
            x.parse()?
        } else {
            let mut y = typ::Type::Path(typ::Path {
                qself: None,
                path: Path::from(Ident::new("Self", self_.span)),
            });
            if let Some((ampersand, lifetime)) = ref_.as_ref() {
                y = typ::Type::Ref(typ::Ref {
                    and: Token![&](ampersand.span),
                    lifetime: lifetime.clone(),
                    mut_: mut_.as_ref().map(|x| Token![mut](x.span)),
                    elem: Box::new(y),
                });
            }
            y
        };
        Ok(Receiver {
            attrs: Vec::new(),
            ref_,
            mut_,
            self_,
            colon,
            typ: Box::new(typ),
        })
    }
}

ast_enum_of_structs! {
    pub enum FnArg {
        Receiver(Receiver),
        Type(pat::Type),
    }
}
impl Parse for FnArg {
    fn parse(x: Stream) -> Res<Self> {
        let variadic = false;
        let attrs = x.call(attr::Attr::parse_outer)?;
        use FnArgOrVariadic::*;
        match parse_fn_arg_or_variadic(x, attrs, variadic)? {
            FnArg(x) => Ok(x),
            Variadic(_) => unreachable!(),
        }
    }
}
enum FnArgOrVariadic {
    FnArg(FnArg),
    Variadic(Variadic),
}

pub struct Sig {
    pub const_: Option<Token![const]>,
    pub async_: Option<Token![async]>,
    pub unsafe_: Option<Token![unsafe]>,
    pub abi: Option<Abi>,
    pub fn_: Token![fn],
    pub ident: Ident,
    pub gens: gen::Gens,
    pub paren: tok::Paren,
    pub args: Punctuated<FnArg, Token![,]>,
    pub vari: Option<Variadic>,
    pub ret: typ::Ret,
}
impl Sig {
    pub fn receiver(&self) -> Option<&Receiver> {
        let y = self.args.first()?;
        match y {
            FnArg::Receiver(&x) => Some(x),
            FnArg::Type(_) => None,
        }
    }
}
impl Parse for Sig {
    fn parse(x: Stream) -> Res<Self> {
        let const_: Option<Token![const]> = x.parse()?;
        let async_: Option<Token![async]> = x.parse()?;
        let unsafe_: Option<Token![unsafe]> = x.parse()?;
        let abi: Option<Abi> = x.parse()?;
        let fn_: Token![fn] = x.parse()?;
        let ident: Ident = x.parse()?;
        let mut gens: gen::Gens = x.parse()?;
        let y;
        let paren = parenthesized!(y in x);
        let (args, vari) = parse_fn_args(&y)?;
        let ret: typ::Ret = x.parse()?;
        gens.where_ = x.parse()?;
        Ok(Sig {
            const_,
            async_,
            unsafe_,
            abi,
            fn_,
            ident,
            gens,
            paren,
            args,
            vari,
            ret,
        })
    }
}

pub struct Variadic {
    pub attrs: Vec<attr::Attr>,
    pub pat: Option<(Box<pat::Pat>, Token![:])>,
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
            Mac(Mac),
            Verbatim(Stream),
        }
    }
    impl Parse for Item {
        fn parse(x: Stream) -> Res<Self> {
            let beg = x.fork();
            let mut attrs = x.call(attr::Attr::parse_outer)?;
            let ahead = x.fork();
            let vis: Visibility = ahead.parse()?;
            let look = ahead.lookahead1();
            let mut y = if look.peek(Token![fn]) || peek_signature(&ahead) {
                let vis: Visibility = x.parse()?;
                let sig: Sig = x.parse()?;
                if x.peek(tok::Brace) {
                    let y;
                    braced!(y in x);
                    y.call(attr::Attr::parse_inner)?;
                    y.call(Block::parse_within)?;
                    Ok(Item::Verbatim(verbatim_between(&beg, x)))
                } else {
                    Ok(Item::Fn(Fn {
                        attrs: Vec::new(),
                        vis,
                        sig,
                        semi: x.parse()?,
                    }))
                }
            } else if look.peek(Token![static]) {
                let vis = x.parse()?;
                let static_ = x.parse()?;
                let mut_ = x.parse()?;
                let ident = x.parse()?;
                let colon = x.parse()?;
                let typ = x.parse()?;
                if x.peek(Token![=]) {
                    x.parse::<Token![=]>()?;
                    x.parse::<Expr>()?;
                    x.parse::<Token![;]>()?;
                    Ok(Item::Verbatim(verbatim_between(&beg, x)))
                } else {
                    Ok(Item::Static(Static {
                        attrs: Vec::new(),
                        vis,
                        static_,
                        mut_,
                        ident,
                        colon,
                        typ,
                        semi: x.parse()?,
                    }))
                }
            } else if look.peek(Token![type]) {
                parse_foreign_item_type(beg, x)
            } else if vis.is_inherited()
                && (look.peek(Ident)
                    || look.peek(Token![self])
                    || look.peek(Token![super])
                    || look.peek(Token![crate])
                    || look.peek(Token![::]))
            {
                x.parse().map(Item::Mac)
            } else {
                Err(look.error())
            }?;
            let ys = match &mut y {
                Item::Fn(x) => &mut x.attrs,
                Item::Static(x) => &mut x.attrs,
                Item::Type(x) => &mut x.attrs,
                Item::Mac(x) => &mut x.attrs,
                Item::Verbatim(_) => return Ok(y),
            };
            attrs.append(ys);
            *ys = attrs;
            Ok(y)
        }
    }

    pub struct Fn {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub sig: Sig,
        pub semi: Token![;],
    }
    impl Parse for Fn {
        fn parse(x: Stream) -> Res<Self> {
            let attrs = x.call(attr::Attr::parse_outer)?;
            let vis: Visibility = x.parse()?;
            let sig: Sig = x.parse()?;
            let semi: Token![;] = x.parse()?;
            Ok(Fn { attrs, vis, sig, semi })
        }
    }

    pub struct Static {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub static_: Token![static],
        pub mut_: StaticMut,
        pub ident: Ident,
        pub colon: Token![:],
        pub typ: Box<typ::Type>,
        pub semi: Token![;],
    }
    impl Parse for Static {
        fn parse(x: Stream) -> Res<Self> {
            Ok(Static {
                attrs: x.call(attr::Attr::parse_outer)?,
                vis: x.parse()?,
                static_: x.parse()?,
                mut_: x.parse()?,
                ident: x.parse()?,
                colon: x.parse()?,
                typ: x.parse()?,
                semi: x.parse()?,
            })
        }
    }

    pub struct Type {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub type_: Token![type],
        pub ident: Ident,
        pub gens: gen::Gens,
        pub semi: Token![;],
    }
    impl Parse for Type {
        fn parse(x: Stream) -> Res<Self> {
            Ok(Type {
                attrs: x.call(attr::Attr::parse_outer)?,
                vis: x.parse()?,
                type_: x.parse()?,
                ident: x.parse()?,
                gens: {
                    let mut y: gen::Gens = x.parse()?;
                    y.where_ = x.parse()?;
                    y
                },
                semi: x.parse()?,
            })
        }
    }

    pub struct Mac {
        pub attrs: Vec<attr::Attr>,
        pub mac: mac::Mac,
        pub semi: Option<Token![;]>,
    }
    impl Parse for Mac {
        fn parse(x: Stream) -> Res<Self> {
            let attrs = x.call(attr::Attr::parse_outer)?;
            let mac: mac::Mac = x.parse()?;
            let semi: Option<Token![;]> = if mac.delim.is_brace() { None } else { Some(x.parse()?) };
            Ok(Mac { attrs, mac, semi })
        }
    }
}
pub mod Impl {
    ast_enum_of_structs! {
        pub enum Item {
            Const(Const),
            Fn(Fn),
            Type(Type),
            Mac(Mac),
            Verbatim(Stream),
        }
    }
    impl Parse for Item {
        fn parse(x: Stream) -> Res<Self> {
            let beg = x.fork();
            let mut attrs = x.call(attr::Attr::parse_outer)?;
            let ahead = x.fork();
            let vis: Visibility = ahead.parse()?;
            let mut look = ahead.lookahead1();
            let default_ = if look.peek(Token![default]) && !ahead.peek2(Token![!]) {
                let y: Token![default] = ahead.parse()?;
                look = ahead.lookahead1();
                Some(y)
            } else {
                None
            };
            let mut y = if look.peek(Token![fn]) || peek_signature(&ahead) {
                let omitted = true;
                if let Some(x) = parse_impl_item_fn(x, omitted)? {
                    Ok(Item::Fn(x))
                } else {
                    Ok(Item::Verbatim(verbatim_between(&beg, x)))
                }
            } else if look.peek(Token![const]) {
                x.advance_to(&ahead);
                let const_: Token![const] = x.parse()?;
                let look = x.lookahead1();
                let ident = if look.peek(Ident) || look.peek(Token![_]) {
                    x.call(Ident::parse_any)?
                } else {
                    return Err(look.error());
                };
                let colon: Token![:] = x.parse()?;
                let typ: typ::Type = x.parse()?;
                if let Some(eq) = x.parse()? {
                    return Ok(Item::Const(Const {
                        attrs,
                        vis,
                        default_,
                        const_,
                        ident,
                        gens: gen::Gens::default(),
                        colon,
                        typ,
                        eq,
                        expr: x.parse()?,
                        semi: x.parse()?,
                    }));
                } else {
                    x.parse::<Token![;]>()?;
                    return Ok(Item::Verbatim(verbatim_between(&beg, x)));
                }
            } else if look.peek(Token![type]) {
                parse_impl_item_type(beg, x)
            } else if vis.is_inherited()
                && default_.is_none()
                && (look.peek(Ident)
                    || look.peek(Token![self])
                    || look.peek(Token![super])
                    || look.peek(Token![crate])
                    || look.peek(Token![::]))
            {
                x.parse().map(Item::Mac)
            } else {
                Err(look.error())
            }?;
            let ys = match &mut y {
                Item::Const(x) => &mut x.attrs,
                Item::Fn(x) => &mut x.attrs,
                Item::Type(x) => &mut x.attrs,
                Item::Mac(x) => &mut x.attrs,
                Item::Verbatim(_) => return Ok(y),
            };
            attrs.append(ys);
            *ys = attrs;
            Ok(y)
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
        pub typ: typ::Type,
        pub eq: Token![=],
        pub expr: Expr,
        pub semi: Token![;],
    }
    impl Parse for Const {
        fn parse(x: Stream) -> Res<Self> {
            Ok(Const {
                attrs: x.call(attr::Attr::parse_outer)?,
                vis: x.parse()?,
                default_: x.parse()?,
                const_: x.parse()?,
                ident: {
                    let look = x.lookahead1();
                    if look.peek(Ident) || look.peek(Token![_]) {
                        x.call(Ident::parse_any)?
                    } else {
                        return Err(look.error());
                    }
                },
                gens: gen::Gens::default(),
                colon: x.parse()?,
                typ: x.parse()?,
                eq: x.parse()?,
                expr: x.parse()?,
                semi: x.parse()?,
            })
        }
    }

    pub struct Fn {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub default_: Option<Token![default]>,
        pub sig: Sig,
        pub block: Block,
    }
    impl Parse for Fn {
        fn parse(x: Stream) -> Res<Self> {
            let omitted = false;
            parse_impl_item_fn(x, omitted).map(Option::unwrap)
        }
    }

    pub struct Type {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub default_: Option<Token![default]>,
        pub type_: Token![type],
        pub ident: Ident,
        pub gens: gen::Gens,
        pub eq: Token![=],
        pub typ: typ::Type,
        pub semi: Token![;],
    }
    impl Parse for Type {
        fn parse(x: Stream) -> Res<Self> {
            let attrs = x.call(attr::Attr::parse_outer)?;
            let vis: Visibility = x.parse()?;
            let default_: Option<Token![default]> = x.parse()?;
            let type_: Token![type] = x.parse()?;
            let ident: Ident = x.parse()?;
            let mut gens: gen::Gens = x.parse()?;
            let eq: Token![=] = x.parse()?;
            let typ: typ::Type = x.parse()?;
            gens.where_ = x.parse()?;
            let semi: Token![;] = x.parse()?;
            Ok(Type {
                attrs,
                vis,
                default_,
                type_,
                ident,
                gens,
                eq,
                typ,
                semi,
            })
        }
    }

    pub struct Mac {
        pub attrs: Vec<attr::Attr>,
        pub mac: mac::Mac,
        pub semi: Option<Token![;]>,
    }
    impl Parse for Mac {
        fn parse(x: Stream) -> Res<Self> {
            let attrs = x.call(attr::Attr::parse_outer)?;
            let mac: mac::Mac = x.parse()?;
            let semi: Option<Token![;]> = if mac.delim.is_brace() { None } else { Some(x.parse()?) };
            Ok(Mac { attrs, mac, semi })
        }
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
            Verbatim(Stream),
        }
    }
    impl Parse for Item {
        fn parse(x: Stream) -> Res<Self> {
            let beg = x.fork();
            let mut attrs = x.call(attr::Attr::parse_outer)?;
            let vis: Visibility = x.parse()?;
            let default_: Option<Token![default]> = x.parse()?;
            let ahead = x.fork();
            let look = ahead.lookahead1();
            let mut y = if look.peek(Token![fn]) || peek_signature(&ahead) {
                x.parse().map(Item::Fn)
            } else if look.peek(Token![const]) {
                ahead.parse::<Token![const]>()?;
                let look = ahead.lookahead1();
                if look.peek(Ident) || look.peek(Token![_]) {
                    x.parse().map(Item::Const)
                } else if look.peek(Token![async])
                    || look.peek(Token![unsafe])
                    || look.peek(Token![extern])
                    || look.peek(Token![fn])
                {
                    x.parse().map(Item::Fn)
                } else {
                    Err(look.error())
                }
            } else if look.peek(Token![type]) {
                parse_trait_item_type(beg.fork(), x)
            } else if vis.is_inherited()
                && default_.is_none()
                && (look.peek(Ident)
                    || look.peek(Token![self])
                    || look.peek(Token![super])
                    || look.peek(Token![crate])
                    || look.peek(Token![::]))
            {
                x.parse().map(Item::Mac)
            } else {
                Err(look.error())
            }?;
            match (vis, default_) {
                (Visibility::Inherited, None) => {},
                _ => return Ok(Item::Verbatim(verbatim_between(&beg, x))),
            }
            let ys = match &mut y {
                Item::Const(item) => &mut item.attrs,
                Item::Fn(item) => &mut item.attrs,
                Item::Type(item) => &mut item.attrs,
                Item::Mac(item) => &mut item.attrs,
                Item::Verbatim(_) => unreachable!(),
            };
            attrs.append(ys);
            *ys = attrs;
            Ok(y)
        }
    }

    pub struct Const {
        pub attrs: Vec<attr::Attr>,
        pub const_: Token![const],
        pub ident: Ident,
        pub gens: gen::Gens,
        pub colon: Token![:],
        pub typ: typ::Type,
        pub default: Option<(Token![=], Expr)>,
        pub semi: Token![;],
    }
    impl Parse for Const {
        fn parse(x: Stream) -> Res<Self> {
            Ok(Const {
                attrs: x.call(attr::Attr::parse_outer)?,
                const_: x.parse()?,
                ident: {
                    let look = x.lookahead1();
                    if look.peek(Ident) || look.peek(Token![_]) {
                        x.call(Ident::parse_any)?
                    } else {
                        return Err(look.error());
                    }
                },
                gens: gen::Gens::default(),
                colon: x.parse()?,
                typ: x.parse()?,
                default: {
                    if x.peek(Token![=]) {
                        let eq: Token![=] = x.parse()?;
                        let default: Expr = x.parse()?;
                        Some((eq, default))
                    } else {
                        None
                    }
                },
                semi: x.parse()?,
            })
        }
    }

    pub struct Fn {
        pub attrs: Vec<attr::Attr>,
        pub sig: Sig,
        pub default: Option<Block>,
        pub semi: Option<Token![;]>,
    }
    impl Parse for Fn {
        fn parse(x: Stream) -> Res<Self> {
            let mut attrs = x.call(attr::Attr::parse_outer)?;
            let sig: Sig = x.parse()?;
            let look = x.lookahead1();
            let (brace, stmts, semi) = if look.peek(tok::Brace) {
                let y;
                let brace = braced!(y in x);
                attr::inner(&y, &mut attrs)?;
                let stmts = y.call(Block::parse_within)?;
                (Some(brace), stmts, None)
            } else if look.peek(Token![;]) {
                let semi: Token![;] = x.parse()?;
                (None, Vec::new(), Some(semi))
            } else {
                return Err(look.error());
            };
            Ok(Fn {
                attrs,
                sig,
                default: brace.map(|brace| Block { brace, stmts }),
                semi,
            })
        }
    }

    pub struct Type {
        pub attrs: Vec<attr::Attr>,
        pub type_: Token![type],
        pub ident: Ident,
        pub gens: gen::Gens,
        pub colon: Option<Token![:]>,
        pub bounds: Punctuated<gen::bound::Type, Token![+]>,
        pub default: Option<(Token![=], typ::Type)>,
        pub semi: Token![;],
    }
    impl Parse for Type {
        fn parse(x: Stream) -> Res<Self> {
            let attrs = x.call(attr::Attr::parse_outer)?;
            let type_: Token![type] = x.parse()?;
            let ident: Ident = x.parse()?;
            let mut gens: gen::Gens = x.parse()?;
            let (colon, bounds) = FlexibleItemTy::parse_optional_bounds(x)?;
            let default = FlexibleItemTy::parse_optional_definition(x)?;
            gens.where_ = x.parse()?;
            let semi: Token![;] = x.parse()?;
            Ok(Type {
                attrs,
                type_,
                ident,
                gens,
                colon,
                bounds,
                default,
                semi,
            })
        }
    }

    pub struct Mac {
        pub attrs: Vec<attr::Attr>,
        pub mac: mac::Mac,
        pub semi: Option<Token![;]>,
    }
    impl Parse for Mac {
        fn parse(x: Stream) -> Res<Self> {
            let attrs = x.call(attr::Attr::parse_outer)?;
            let mac: mac::Mac = x.parse()?;
            let semi: Option<Token![;]> = if mac.delim.is_brace() { None } else { Some(x.parse()?) };
            Ok(Mac { attrs, mac, semi })
        }
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
    impl Parse for Tree {
        fn parse(x: Stream) -> Res<Tree> {
            let root = false;
            parse_tree(x, root).map(Option::unwrap)
        }
    }
    fn parse_tree(x: Stream, root: bool) -> Res<Option<Tree>> {
        let look = x.lookahead1();
        if look.peek(Ident)
            || look.peek(Token![self])
            || look.peek(Token![super])
            || look.peek(Token![crate])
            || look.peek(Token![try])
        {
            let ident = x.call(Ident::parse_any)?;
            if x.peek(Token![::]) {
                Ok(Some(Tree::Path(Path {
                    ident,
                    colon2: x.parse()?,
                    tree: Box::new(x.parse()?),
                })))
            } else if x.peek(Token![as]) {
                Ok(Some(Tree::Rename(Rename {
                    ident,
                    as_: x.parse()?,
                    rename: {
                        if x.peek(Ident) {
                            x.parse()?
                        } else if x.peek(Token![_]) {
                            Ident::from(x.parse::<Token![_]>()?)
                        } else {
                            return Err(x.error("expected identifier or underscore"));
                        }
                    },
                })))
            } else {
                Ok(Some(Tree::Name(Name { ident })))
            }
        } else if look.peek(Token![*]) {
            Ok(Some(Tree::Glob(Glob { star: x.parse()? })))
        } else if look.peek(tok::Brace) {
            let y;
            let brace = braced!(y in x);
            let mut elems = Punctuated::new();
            let mut has_root = false;
            loop {
                if y.is_empty() {
                    break;
                }
                let starts_with_root = root && y.parse::<Option<Token![::]>>()?.is_some();
                has_root |= starts_with_root;
                match parse_tree(&y, root && !starts_with_root)? {
                    Some(x) => elems.push_value(x),
                    None => has_root = true,
                }
                if y.is_empty() {
                    break;
                }
                let comma: Token![,] = y.parse()?;
                elems.push_punct(comma);
            }
            if has_root {
                Ok(None)
            } else {
                Ok(Some(Tree::Group(Group { brace, elems })))
            }
        } else {
            Err(look.error())
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

enum TypeDefault {
    Optional,
    Disallowed,
}
enum WhereLoc {
    BeforeEq,
    AfterEq,
    Both,
}

struct FlexibleItemType {
    vis: Visibility,
    default_: Option<Token![default]>,
    type_: Token![type],
    ident: Ident,
    gens: gen::Gens,
    colon: Option<Token![:]>,
    bounds: Punctuated<gen::bound::Type, Token![+]>,
    typ: Option<(Token![=], typ::Type)>,
    semi: Token![;],
}
impl FlexibleItemType {
    fn parse(x: Stream, allow_default: TypeDefault, where_loc: WhereLoc) -> Res<Self> {
        let vis: Visibility = x.parse()?;
        let default_: Option<Token![default]> = match allow_default {
            TypeDefault::Optional => x.parse()?,
            TypeDefault::Disallowed => None,
        };
        let type_: Token![type] = x.parse()?;
        let ident: Ident = x.parse()?;
        let mut gens: gen::Gens = x.parse()?;
        let (colon, bounds) = Self::parse_optional_bounds(x)?;
        match where_loc {
            WhereLoc::BeforeEq | WhereLoc::Both => {
                gens.where_ = x.parse()?;
            },
            WhereLoc::AfterEq => {},
        }
        let ty = Self::parse_optional_definition(x)?;
        match where_loc {
            WhereLoc::AfterEq | WhereLoc::Both if gens.where_.is_none() => {
                gens.where_ = x.parse()?;
            },
            _ => {},
        }
        let semi: Token![;] = x.parse()?;
        Ok(FlexibleItemType {
            vis,
            default_,
            type_,
            ident,
            gens,
            colon,
            bounds,
            typ: ty,
            semi,
        })
    }
    fn parse_optional_bounds(x: Stream) -> Res<(Option<Token![:]>, Punctuated<gen::bound::Type, Token![+]>)> {
        let colon: Option<Token![:]> = x.parse()?;
        let mut bounds = Punctuated::new();
        if colon.is_some() {
            loop {
                if x.peek(Token![where]) || x.peek(Token![=]) || x.peek(Token![;]) {
                    break;
                }
                bounds.push_value(x.parse::<gen::bound::Type>()?);
                if x.peek(Token![where]) || x.peek(Token![=]) || x.peek(Token![;]) {
                    break;
                }
                bounds.push_punct(x.parse::<Token![+]>()?);
            }
        }
        Ok((colon, bounds))
    }
    fn parse_optional_definition(x: Stream) -> Res<Option<(Token![=], typ::Type)>> {
        let eq: Option<Token![=]> = x.parse()?;
        if let Some(eq) = eq {
            let y: typ::Type = x.parse()?;
            Ok(Some((eq, y)))
        } else {
            Ok(None)
        }
    }
}

pub fn parse_rest_of_item(begin: Buffer, mut attrs: Vec<attr::Attr>, input: Stream) -> Res<Item> {
    let ahead = input.fork();
    let vis: Visibility = ahead.parse()?;
    let lookahead = ahead.lookahead1();
    let mut item = if lookahead.peek(Token![fn]) || peek_signature(&ahead) {
        let vis: Visibility = input.parse()?;
        let sig: Sig = input.parse()?;
        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
            Ok(Item::Verbatim(verbatim_between(&begin, input)))
        } else {
            parse_rest_of_fn(input, Vec::new(), vis, sig).map(Item::Fn)
        }
    } else if lookahead.peek(Token![extern]) {
        ahead.parse::<Token![extern]>()?;
        let lookahead = ahead.lookahead1();
        if lookahead.peek(Token![crate]) {
            input.parse().map(Item::ExternCrate)
        } else if lookahead.peek(tok::Brace) {
            input.parse().map(Item::Foreign)
        } else if lookahead.peek(lit::Str) {
            ahead.parse::<lit::Str>()?;
            let lookahead = ahead.lookahead1();
            if lookahead.peek(tok::Brace) {
                input.parse().map(Item::Foreign)
            } else {
                Err(lookahead.error())
            }
        } else {
            Err(lookahead.error())
        }
    } else if lookahead.peek(Token![use]) {
        let allow_crate_root_in_path = true;
        match parse_item_use(input, allow_crate_root_in_path)? {
            Some(item_use) => Ok(Item::Use(item_use)),
            None => Ok(Item::Verbatim(verbatim_between(&begin, input))),
        }
    } else if lookahead.peek(Token![static]) {
        let vis = input.parse()?;
        let static_ = input.parse()?;
        let mutability = input.parse()?;
        let ident = input.parse()?;
        if input.peek(Token![=]) {
            input.parse::<Token![=]>()?;
            input.parse::<Expr>()?;
            input.parse::<Token![;]>()?;
            Ok(Item::Verbatim(verbatim_between(&begin, input)))
        } else {
            let colon = input.parse()?;
            let ty = input.parse()?;
            if input.peek(Token![;]) {
                input.parse::<Token![;]>()?;
                Ok(Item::Verbatim(verbatim_between(&begin, input)))
            } else {
                Ok(Item::Static(Static {
                    attrs: Vec::new(),
                    vis,
                    static_,
                    mut_: mutability,
                    ident,
                    colon,
                    typ: ty,
                    eq: input.parse()?,
                    expr: input.parse()?,
                    semi: input.parse()?,
                }))
            }
        }
    } else if lookahead.peek(Token![const]) {
        let vis = input.parse()?;
        let const_: Token![const] = input.parse()?;
        let lookahead = input.lookahead1();
        let ident = if lookahead.peek(Ident) || lookahead.peek(Token![_]) {
            input.call(Ident::parse_any)?
        } else {
            return Err(lookahead.error());
        };
        let colon = input.parse()?;
        let ty = input.parse()?;
        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
            Ok(Item::Verbatim(verbatim_between(&begin, input)))
        } else {
            Ok(Item::Const(Const {
                attrs: Vec::new(),
                vis,
                const_,
                ident,
                gens: gen::Gens::default(),
                colon,
                typ: ty,
                eq: input.parse()?,
                expr: input.parse()?,
                semi: input.parse()?,
            }))
        }
    } else if lookahead.peek(Token![unsafe]) {
        ahead.parse::<Token![unsafe]>()?;
        let lookahead = ahead.lookahead1();
        if lookahead.peek(Token![trait]) || lookahead.peek(Token![auto]) && ahead.peek2(Token![trait]) {
            input.parse().map(Item::Trait)
        } else if lookahead.peek(Token![impl]) {
            let allow_verbatim_impl = true;
            if let Some(item) = parse_impl(input, allow_verbatim_impl)? {
                Ok(Item::Impl(item))
            } else {
                Ok(Item::Verbatim(verbatim_between(&begin, input)))
            }
        } else if lookahead.peek(Token![extern]) {
            input.parse().map(Item::Foreign)
        } else if lookahead.peek(Token![mod]) {
            input.parse().map(Item::Mod)
        } else {
            Err(lookahead.error())
        }
    } else if lookahead.peek(Token![mod]) {
        input.parse().map(Item::Mod)
    } else if lookahead.peek(Token![type]) {
        parse_item_type(begin, input)
    } else if lookahead.peek(Token![struct]) {
        input.parse().map(Item::Struct)
    } else if lookahead.peek(Token![enum]) {
        input.parse().map(Item::Enum)
    } else if lookahead.peek(Token![union]) && ahead.peek2(Ident) {
        input.parse().map(Item::Union)
    } else if lookahead.peek(Token![trait]) {
        input.call(parse_trait_or_trait_alias)
    } else if lookahead.peek(Token![auto]) && ahead.peek2(Token![trait]) {
        input.parse().map(Item::Trait)
    } else if lookahead.peek(Token![impl]) || lookahead.peek(Token![default]) && !ahead.peek2(Token![!]) {
        let allow_verbatim_impl = true;
        if let Some(item) = parse_impl(input, allow_verbatim_impl)? {
            Ok(Item::Impl(item))
        } else {
            Ok(Item::Verbatim(verbatim_between(&begin, input)))
        }
    } else if lookahead.peek(Token![macro]) {
        input.advance_to(&ahead);
        parse_macro2(begin, vis, input)
    } else if vis.is_inherited()
        && (lookahead.peek(Ident)
            || lookahead.peek(Token![self])
            || lookahead.peek(Token![super])
            || lookahead.peek(Token![crate])
            || lookahead.peek(Token![::]))
    {
        input.parse().map(Item::Mac)
    } else {
        Err(lookahead.error())
    }?;
    attrs.extend(item.replace_attrs(Vec::new()));
    item.replace_attrs(attrs);
    Ok(item)
}
fn parse_macro2(begin: Buffer, _vis: Visibility, input: Stream) -> Res<Item> {
    input.parse::<Token![macro]>()?;
    input.parse::<Ident>()?;
    let mut lookahead = input.lookahead1();
    if lookahead.peek(tok::Paren) {
        let paren_content;
        parenthesized!(paren_content in input);
        paren_content.parse::<pm2::Stream>()?;
        lookahead = input.lookahead1();
    }
    if lookahead.peek(tok::Brace) {
        let brace_content;
        braced!(brace_content in input);
        brace_content.parse::<pm2::Stream>()?;
    } else {
        return Err(lookahead.error());
    }
    Ok(Item::Verbatim(verbatim_between(&begin, input)))
}
fn peek_signature(x: Stream) -> bool {
    let y = x.fork();
    y.parse::<Option<Token![const]>>().is_ok()
        && y.parse::<Option<Token![async]>>().is_ok()
        && y.parse::<Option<Token![unsafe]>>().is_ok()
        && y.parse::<Option<Abi>>().is_ok()
        && y.peek(Token![fn])
}
fn parse_rest_of_fn(x: Stream, mut attrs: Vec<attr::Attr>, vis: Visibility, sig: Sig) -> Res<Fn> {
    let y;
    let brace = braced!(y in x);
    attr::inner(&y, &mut attrs)?;
    let stmts = y.call(Block::parse_within)?;
    Ok(Fn {
        attrs,
        vis,
        sig,
        block: Box::new(Block { brace, stmts }),
    })
}
fn parse_fn_arg_or_variadic(x: Stream, attrs: Vec<attr::Attr>, variadic: bool) -> Res<FnArgOrVariadic> {
    let ahead = x.fork();
    if let Ok(mut receiver) = ahead.parse::<Receiver>() {
        x.advance_to(&ahead);
        receiver.attrs = attrs;
        return Ok(FnArgOrVariadic::FnArg(FnArg::Receiver(receiver)));
    }
    if x.peek(Ident) && x.peek2(Token![<]) {
        let span = x.fork().parse::<Ident>()?.span();
        return Ok(FnArgOrVariadic::FnArg(FnArg::Typed(pat::Type {
            attrs,
            pat: Box::new(pat::Pat::Wild(pat::Wild {
                attrs: Vec::new(),
                underscore: Token![_](span),
            })),
            colon: Token![:](span),
            ty: x.parse()?,
        })));
    }
    let pat = Box::new(pat::Pat::parse_single(x)?);
    let colon: Token![:] = x.parse()?;
    if variadic {
        if let Some(dots) = x.parse::<Option<Token![...]>>()? {
            return Ok(FnArgOrVariadic::Variadic(Variadic {
                attrs,
                pat: Some((pat, colon)),
                dots,
                comma: None,
            }));
        }
    }
    Ok(FnArgOrVariadic::FnArg(FnArg::Typed(pat::Type {
        attrs,
        pat,
        colon,
        ty: x.parse()?,
    })))
}
fn parse_fn_args(x: Stream) -> Res<(Punctuated<FnArg, Token![,]>, Option<Variadic>)> {
    let mut ys = Punctuated::new();
    let mut vari = None;
    let mut has_receiver = false;
    while !x.is_empty() {
        let attrs = x.call(attr::Attr::parse_outer)?;
        if let Some(dots) = x.parse::<Option<Token![...]>>()? {
            vari = Some(Variadic {
                attrs,
                pat: None,
                dots,
                comma: if x.is_empty() { None } else { Some(x.parse()?) },
            });
            break;
        }
        let variadic = true;
        let y = match parse_fn_arg_or_variadic(x, attrs, variadic)? {
            FnArgOrVariadic::FnArg(x) => x,
            FnArgOrVariadic::Variadic(x) => {
                vari = Some(Variadic {
                    comma: if x.is_empty() { None } else { Some(x.parse()?) },
                    ..x
                });
                break;
            },
        };
        match &y {
            FnArg::Receiver(x) if has_receiver => {
                return Err(Err::new(x.self_.span, "unexpected second method receiver"));
            },
            FnArg::Receiver(x) if !ys.is_empty() => {
                return Err(Err::new(x.self_.span, "unexpected method receiver"));
            },
            FnArg::Receiver(_) => has_receiver = true,
            FnArg::Typed(_) => {},
        }
        ys.push_value(y);
        if x.is_empty() {
            break;
        }
        let y: Token![,] = x.parse()?;
        ys.push_punct(y);
    }
    Ok((ys, vari))
}
fn parse_foreign_item_type(beg: Buffer, x: Stream) -> Res<Foreign::Item> {
    let FlexibleItemType {
        vis,
        default_: _,
        type_,
        ident,
        gens,
        colon,
        bounds: _,
        typ,
        semi,
    } = FlexibleItemTy::parse(x, TypeDefault::Disallowed, WhereLoc::Both)?;
    if colon.is_some() || typ.is_some() {
        Ok(Foreign::Item::Verbatim(verbatim_between(&beg, x)))
    } else {
        Ok(Foreign::Item::Type(Foreign::Type {
            attrs: Vec::new(),
            vis,
            type_,
            ident,
            gens,
            semi,
        }))
    }
}
fn parse_item_type(beg: Buffer, x: Stream) -> Res<Item> {
    let FlexibleItemType {
        vis,
        default_: _,
        type_,
        ident,
        gens,
        colon,
        bounds: _,
        typ,
        semi,
    } = FlexibleItemTy::parse(x, TypeDefault::Disallowed, WhereLoc::BeforeEq)?;
    let (eq, typ) = match typ {
        Some(x) if colon.is_none() => x,
        _ => return Ok(Item::Verbatim(verbatim_between(&beg, x))),
    };
    Ok(Item::Type(Type {
        attrs: Vec::new(),
        vis,
        type_,
        ident,
        gens,
        eq,
        typ: Box::new(typ),
        semi,
    }))
}
fn parse_trait_or_trait_alias(x: Stream) -> Res<Item> {
    let (attrs, vis, trait_, ident, gens) = parse_start_of_trait_alias(x)?;
    let look = x.lookahead1();
    if look.peek(tok::Brace) || look.peek(Token![:]) || look.peek(Token![where]) {
        let unsafety = None;
        let auto_ = None;
        parse_rest_of_trait(x, attrs, vis, unsafety, auto_, trait_, ident, gens).map(Item::Trait)
    } else if look.peek(Token![=]) {
        parse_rest_of_trait_alias(x, attrs, vis, trait_, ident, gens).map(Item::TraitAlias)
    } else {
        Err(look.error())
    }
}
fn parse_rest_of_trait(
    x: Stream,
    mut attrs: Vec<attr::Attr>,
    vis: Visibility,
    unsafe_: Option<Token![unsafe]>,
    auto_: Option<Token![auto]>,
    trait_: Token![trait],
    ident: Ident,
    mut gens: gen::Gens,
) -> Res<Trait> {
    let colon: Option<Token![:]> = x.parse()?;
    let mut supers = Punctuated::new();
    if colon.is_some() {
        loop {
            if x.peek(Token![where]) || x.peek(tok::Brace) {
                break;
            }
            supers.push_value(x.parse()?);
            if x.peek(Token![where]) || x.peek(tok::Brace) {
                break;
            }
            supers.push_punct(x.parse()?);
        }
    }
    gens.where_ = x.parse()?;
    let y;
    let brace = braced!(y in x);
    attr::inner(&y, &mut attrs)?;
    let mut items = Vec::new();
    while !y.is_empty() {
        items.push(y.parse()?);
    }
    Ok(Trait {
        attrs,
        vis,
        unsafe_,
        auto_,
        restriction: None,
        trait_,
        ident,
        gens,
        colon,
        supers,
        brace,
        items,
    })
}
fn parse_start_of_trait_alias(x: Stream) -> Res<(Vec<attr::Attr>, Visibility, Token![trait], Ident, gen::Gens)> {
    let attrs = x.call(attr::Attr::parse_outer)?;
    let vis: Visibility = x.parse()?;
    let trait_: Token![trait] = x.parse()?;
    let ident: Ident = x.parse()?;
    let gens: gen::Gens = x.parse()?;
    Ok((attrs, vis, trait_, ident, gens))
}
fn parse_rest_of_trait_alias(
    x: Stream,
    attrs: Vec<attr::Attr>,
    vis: Visibility,
    trait_: Token![trait],
    ident: Ident,
    mut gens: gen::Gens,
) -> Res<TraitAlias> {
    let eq: Token![=] = x.parse()?;
    let mut bounds = Punctuated::new();
    loop {
        if x.peek(Token![where]) || x.peek(Token![;]) {
            break;
        }
        bounds.push_value(x.parse()?);
        if x.peek(Token![where]) || x.peek(Token![;]) {
            break;
        }
        bounds.push_punct(x.parse()?);
    }
    gens.where_ = x.parse()?;
    let semi: Token![;] = x.parse()?;
    Ok(TraitAlias {
        attrs,
        vis,
        trait_,
        ident,
        gens,
        eq,
        bounds,
        semi,
    })
}
fn parse_trait_item_type(beg: Buffer, x: Stream) -> Res<Trait::Item> {
    let FlexibleItemType {
        vis,
        default_: _,
        type_,
        ident,
        gens,
        colon,
        bounds,
        typ,
        semi,
    } = FlexibleItemTy::parse(x, TypeDefault::Disallowed, WhereLoc::AfterEq)?;
    if vis.is_some() {
        Ok(Trait::Item::Verbatim(verbatim_between(&beg, x)))
    } else {
        Ok(Trait::Item::Type(Trait::Type {
            attrs: Vec::new(),
            type_,
            ident,
            gens,
            colon,
            bounds,
            default: typ,
            semi,
        }))
    }
}
fn parse_impl(x: Stream, verbatim: bool) -> Res<Option<Impl>> {
    let mut attrs = x.call(attr::Attr::parse_outer)?;
    let has_visibility = verbatim && x.parse::<Visibility>()?.is_some();
    let default_: Option<Token![default]> = x.parse()?;
    let unsafe_: Option<Token![unsafe]> = x.parse()?;
    let impl_: Token![impl] = x.parse()?;
    let has_gens = x.peek(Token![<])
        && (x.peek2(Token![>])
            || x.peek2(Token![#])
            || (x.peek2(Ident) || x.peek2(Lifetime))
                && (x.peek3(Token![:]) || x.peek3(Token![,]) || x.peek3(Token![>]) || x.peek3(Token![=]))
            || x.peek2(Token![const]));
    let mut gens: gen::Gens = if has_gens { x.parse()? } else { gen::Gens::default() };
    let is_const = verbatim && (x.peek(Token![const]) || x.peek(Token![?]) && x.peek2(Token![const]));
    if is_const {
        x.parse::<Option<Token![?]>>()?;
        x.parse::<Token![const]>()?;
    }
    let begin = x.fork();
    let polarity = if x.peek(Token![!]) && !x.peek2(tok::Brace) {
        Some(x.parse::<Token![!]>()?)
    } else {
        None
    };
    let mut first: typ::Type = x.parse()?;
    let self_ty: typ::Type;
    let trait_;
    let is_impl_for = x.peek(Token![for]);
    if is_impl_for {
        let for_: Token![for] = x.parse()?;
        let mut first_ty_ref = &first;
        while let typ::Type::Group(ty) = first_ty_ref {
            first_ty_ref = &ty.elem;
        }
        if let typ::Type::Path(typ::Path { qself: None, .. }) = first_ty_ref {
            while let typ::Type::Group(x) = first {
                first = *x.elem;
            }
            if let typ::Type::Path(typ::Path { qself: None, path }) = first {
                trait_ = Some((polarity, path, for_));
            } else {
                unreachable!();
            }
        } else if !verbatim {
            return Err(Err::new_spanned(first_ty_ref, "expected trait path"));
        } else {
            trait_ = None;
        }
        self_ty = x.parse()?;
    } else {
        trait_ = None;
        self_ty = if polarity.is_none() {
            first
        } else {
            typ::Type::Verbatim(verbatim_between(&begin, x))
        };
    }
    gens.where_ = x.parse()?;
    let y;
    let brace = braced!(y in x);
    attr::inner(&y, &mut attrs)?;
    let mut items = Vec::new();
    while !y.is_empty() {
        items.push(y.parse()?);
    }
    if has_visibility || is_const || is_impl_for && trait_.is_none() {
        Ok(None)
    } else {
        Ok(Some(Impl {
            attrs,
            default_,
            unsafe_,
            impl_,
            gens,
            trait_,
            typ: Box::new(self_ty),
            brace,
            items,
        }))
    }
}
fn parse_impl_item_fn(x: Stream, omitted: bool) -> Res<Option<Impl::Fn>> {
    let mut attrs = x.call(attr::Attr::parse_outer)?;
    let vis: Visibility = x.parse()?;
    let default_: Option<Token![default]> = x.parse()?;
    let sig: Sig = x.parse()?;
    if omitted && x.parse::<Option<Token![;]>>()?.is_some() {
        return Ok(None);
    }
    let y;
    let brace = braced!(y in x);
    attrs.extend(y.call(attr::Attr::parse_inner)?);
    let block = Block {
        brace,
        stmts: y.call(Block::parse_within)?,
    };
    Ok(Some(Impl::Fn {
        attrs,
        vis,
        default_,
        sig,
        block,
    }))
}
fn parse_impl_item_type(begin: Buffer, x: Stream) -> Res<Impl::Item> {
    let FlexibleItemType {
        vis,
        default_,
        type_,
        ident,
        gens,
        colon,
        bounds: _,
        typ,
        semi,
    } = FlexibleItemTy::parse(x, TypeDefault::Optional, WhereLoc::AfterEq)?;
    let (eq, typ) = match typ {
        Some(x) if colon.is_none() => x,
        _ => return Ok(Impl::Item::Verbatim(verbatim_between(&begin, x))),
    };
    Ok(Impl::Item::Type(Impl::Type {
        attrs: Vec::new(),
        vis,
        default_,
        type_,
        ident,
        gens,
        eq,
        typ,
        semi,
    }))
}
