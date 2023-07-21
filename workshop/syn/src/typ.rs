use super::*;

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
        Stream(pm2::Stream),
    }
}
impl Type {
    pub fn without_plus(x: Stream) -> Res<Self> {
        let plus = false;
        let gen = true;
        parse_ambig_typ(x, plus, gen)
    }
}
impl Parse for Type {
    fn parse(x: Stream) -> Res<Self> {
        let plus = true;
        let group_gen = true;
        parse_ambig_typ(x, plus, group_gen)
    }
}

pub struct Array {
    pub bracket: tok::Bracket,
    pub elem: Box<Type>,
    pub semi: Token![;],
    pub len: expr::Expr,
}
impl Parse for Array {
    fn parse(x: Stream) -> Res<Self> {
        let gist;
        Ok(Array {
            bracket: bracketed!(gist in x),
            elem: gist.parse()?,
            semi: gist.parse()?,
            len: gist.parse()?,
        })
    }
}
impl Lower for Array {
    fn lower(&self, s: &mut Stream) {
        self.bracket.surround(s, |s| {
            self.elem.lower(s);
            self.semi.lower(s);
            self.len.lower(s);
        });
    }
}

pub struct Fn {
    pub lifes: Option<gen::bound::Lifes>,
    pub unsafe_: Option<Token![unsafe]>,
    pub abi: Option<Abi>,
    pub fn_: Token![fn],
    pub paren: tok::Paren,
    pub args: Puncted<FnArg, Token![,]>,
    pub vari: Option<Variadic>,
    pub ret: Ret,
}
impl Parse for Fn {
    fn parse(x: Stream) -> Res<Self> {
        let args;
        let mut vari = None;
        Ok(Fn {
            lifes: x.parse()?,
            unsafe_: x.parse()?,
            abi: x.parse()?,
            fn_: x.parse()?,
            paren: parenthesized!(args in x),
            args: {
                let mut ys = Puncted::new();
                while !args.is_empty() {
                    let attrs = args.call(attr::Attr::parse_outers)?;
                    if ys.empty_or_trailing()
                        && (args.peek(Token![...])
                            || args.peek(ident::Ident) && args.peek2(Token![:]) && args.peek3(Token![...]))
                    {
                        vari = Some(parse_variadic(&args, attrs)?);
                        break;
                    }
                    let allow_self = ys.is_empty();
                    let arg = parse_fn_arg(&args, allow_self)?;
                    ys.push_value(FnArg { attrs, ..arg });
                    if args.is_empty() {
                        break;
                    }
                    let comma = args.parse()?;
                    ys.push_punct(comma);
                }
                ys
            },
            vari,
            ret: x.call(Ret::without_plus)?,
        })
    }
}
impl Lower for Fn {
    fn lower(&self, s: &mut Stream) {
        self.lifes.lower(s);
        self.unsafe_.lower(s);
        self.abi.lower(s);
        self.fn_.lower(s);
        self.paren.surround(s, |s| {
            self.args.lower(s);
            if let Some(x) = &self.vari {
                if !self.args.empty_or_trailing() {
                    let s = x.dots.spans[0];
                    Token![,](s).lower(s);
                }
                x.lower(s);
            }
        });
        self.ret.lower(s);
    }
}

pub struct Group {
    pub group: tok::Group,
    pub elem: Box<Type>,
}
impl Parse for Group {
    fn parse(x: Stream) -> Res<Self> {
        let y = parse::parse_group(x)?;
        Ok(Group {
            group: y.tok,
            elem: y.buf.parse()?,
        })
    }
}
impl Lower for Group {
    fn lower(&self, s: &mut Stream) {
        self.group.surround(s, |s| {
            self.elem.lower(s);
        });
    }
}

pub struct Impl {
    pub impl_: Token![impl],
    pub bounds: Puncted<gen::bound::Type, Token![+]>,
}
impl Impl {
    pub fn without_plus(x: Stream) -> Res<Self> {
        let plus = false;
        Self::parse(x, plus)
    }
    pub fn parse(x: Stream, plus: bool) -> Res<Self> {
        let impl_: Token![impl] = x.parse()?;
        let bounds = gen::bound::Type::parse_many(x, plus)?;
        let mut last = None;
        let mut one = false;
        for x in &bounds {
            match x {
                gen::bound::Type::Trait(_) | gen::bound::Type::Stream(_) => {
                    one = true;
                    break;
                },
                gen::bound::Type::Life(x) => {
                    last = Some(x.ident.span());
                },
            }
        }
        if !one {
            let msg = "at least one trait must be specified";
            return Err(err::new2(impl_.span, last.unwrap(), msg));
        }
        Ok(Impl { impl_, bounds })
    }
}
impl Parse for Impl {
    fn parse(x: Stream) -> Res<Self> {
        let plus = true;
        Self::parse(x, plus)
    }
}
impl Lower for Impl {
    fn lower(&self, s: &mut Stream) {
        self.impl_.lower(s);
        self.bounds.lower(s);
    }
}

pub struct Infer {
    pub underscore: Token![_],
}
impl Parse for Infer {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Infer { underscore: x.parse()? })
    }
}
impl Lower for Infer {
    fn lower(&self, s: &mut Stream) {
        self.underscore.lower(s);
    }
}

pub struct Mac {
    pub mac: mac::Mac,
}
impl Parse for Mac {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Mac { mac: x.parse()? })
    }
}
impl Lower for Mac {
    fn lower(&self, s: &mut Stream) {
        self.mac.lower(s);
    }
}

pub struct Never {
    pub bang: Token![!],
}
impl Parse for Never {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Never { bang: x.parse()? })
    }
}
impl Lower for Never {
    fn lower(&self, s: &mut Stream) {
        self.bang.lower(s);
    }
}

pub struct Paren {
    pub paren: tok::Paren,
    pub elem: Box<Type>,
}
impl Paren {
    fn parse(x: Stream, plus: bool) -> Res<Self> {
        let gist;
        Ok(Paren {
            paren: parenthesized!(gist in x),
            elem: Box::new({
                let group_gen = true;
                parse_ambig_typ(&gist, plus, group_gen)?
            }),
        })
    }
}
impl Parse for Paren {
    fn parse(x: Stream) -> Res<Self> {
        let plus = false;
        Self::parse(x, plus)
    }
}
impl Lower for Paren {
    fn lower(&self, s: &mut Stream) {
        self.paren.surround(s, |s| {
            self.elem.lower(s);
        });
    }
}

pub struct Path {
    pub qself: Option<path::QSelf>,
    pub path: Path,
}
impl Parse for Path {
    fn parse(x: Stream) -> Res<Self> {
        let expr_style = false;
        let (qself, path) = path::qpath(x, expr_style)?;
        Ok(Path { qself, path })
    }
}
impl Lower for Path {
    fn lower(&self, s: &mut Stream) {
        path::path_to_tokens(s, &self.qself, &self.path);
    }
}

pub struct Ptr {
    pub star: Token![*],
    pub const_: Option<Token![const]>,
    pub mut_: Option<Token![mut]>,
    pub elem: Box<Type>,
}
impl Parse for Ptr {
    fn parse(x: Stream) -> Res<Self> {
        let star: Token![*] = x.parse()?;
        let look = x.look1();
        let (const_, mut_) = if look.peek(Token![const]) {
            (Some(x.parse()?), None)
        } else if look.peek(Token![mut]) {
            (None, Some(x.parse()?))
        } else {
            return Err(look.err());
        };
        Ok(Ptr {
            star,
            const_,
            mut_,
            elem: Box::new(x.call(Type::without_plus)?),
        })
    }
}
impl Lower for Ptr {
    fn lower(&self, s: &mut Stream) {
        self.star.lower(s);
        match &self.mut_ {
            Some(x) => x.lower(s),
            None => {
                ToksOrDefault(&self.const_).lower(s);
            },
        }
        self.elem.lower(s);
    }
}

pub struct Ref {
    pub and: Token![&],
    pub life: Option<Life>,
    pub mut_: Option<Token![mut]>,
    pub elem: Box<Type>,
}
impl Parse for Ref {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Ref {
            and: x.parse()?,
            life: x.parse()?,
            mut_: x.parse()?,
            elem: Box::new(x.call(Type::without_plus)?),
        })
    }
}
impl Lower for Ref {
    fn lower(&self, s: &mut Stream) {
        self.and.lower(s);
        self.life.lower(s);
        self.mut_.lower(s);
        self.elem.lower(s);
    }
}

pub struct Slice {
    pub bracket: tok::Bracket,
    pub elem: Box<Type>,
}
impl Parse for Slice {
    fn parse(x: Stream) -> Res<Self> {
        let gist;
        Ok(Slice {
            bracket: bracketed!(gist in x),
            elem: gist.parse()?,
        })
    }
}
impl Lower for Slice {
    fn lower(&self, s: &mut Stream) {
        self.bracket.surround(s, |s| {
            self.elem.lower(s);
        });
    }
}

pub struct Trait {
    pub dyn_: Option<Token![dyn]>,
    pub bounds: Puncted<gen::bound::Type, Token![+]>,
}
impl Trait {
    pub fn without_plus(x: Stream) -> Res<Self> {
        let plus = false;
        Self::parse(x, plus)
    }
    pub fn parse(x: Stream, plus: bool) -> Res<Self> {
        let dyn_: Option<Token![dyn]> = x.parse()?;
        let span = match &dyn_ {
            Some(x) => x.span,
            None => x.span(),
        };
        let bounds = Self::parse_bounds(span, x, plus)?;
        Ok(Trait { dyn_, bounds })
    }
    fn parse_bounds(s: pm2::Span, x: Stream, plus: bool) -> Res<Puncted<gen::bound::Type, Token![+]>> {
        let ys = gen::bound::Type::parse_many(x, plus)?;
        let mut last = None;
        let mut one = false;
        for y in &ys {
            match y {
                gen::bound::Type::Trait(_) | gen::bound::Type::Stream(_) => {
                    one = true;
                    break;
                },
                gen::bound::Type::Life(x) => {
                    last = Some(x.ident.span());
                },
            }
        }
        if !one {
            let msg = "at least one trait is required for an object type";
            return Err(err::new2(s, last.unwrap(), msg));
        }
        Ok(ys)
    }
}
impl Parse for Trait {
    fn parse(x: Stream) -> Res<Self> {
        let plus = true;
        Self::parse(x, plus)
    }
}
impl Lower for Trait {
    fn lower(&self, s: &mut Stream) {
        self.dyn_.lower(s);
        self.bounds.lower(s);
    }
}

pub struct Tuple {
    pub paren: tok::Paren,
    pub elems: Puncted<Type, Token![,]>,
}
impl Parse for Tuple {
    fn parse(x: Stream) -> Res<Self> {
        let gist;
        let paren = parenthesized!(gist in x);
        if gist.is_empty() {
            return Ok(Tuple {
                paren,
                elems: Puncted::new(),
            });
        }
        let first: Type = gist.parse()?;
        Ok(Tuple {
            paren,
            elems: {
                let mut ys = Puncted::new();
                ys.push_value(first);
                ys.push_punct(gist.parse()?);
                while !gist.is_empty() {
                    ys.push_value(gist.parse()?);
                    if gist.is_empty() {
                        break;
                    }
                    ys.push_punct(gist.parse()?);
                }
                ys
            },
        })
    }
}
impl Lower for Tuple {
    fn lower(&self, s: &mut Stream) {
        self.paren.surround(s, |s| {
            self.elems.lower(s);
            if self.elems.len() == 1 && !self.elems.trailing_punct() {
                <Token![,]>::default().lower(s);
            }
        });
    }
}

pub struct Abi {
    pub extern_: Token![extern],
    pub name: Option<lit::Str>,
}
impl Parse for Abi {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Abi {
            extern_: x.parse()?,
            name: x.parse()?,
        })
    }
}
impl Parse for Option<Abi> {
    fn parse(x: Stream) -> Res<Self> {
        if x.peek(Token![extern]) {
            x.parse().map(Some)
        } else {
            Ok(None)
        }
    }
}
impl Lower for Abi {
    fn lower(&self, s: &mut Stream) {
        self.extern_.lower(s);
        self.name.lower(s);
    }
}

pub struct FnArg {
    pub attrs: Vec<attr::Attr>,
    pub name: Option<(Ident, Token![:])>,
    pub typ: Type,
}
impl Parse for FnArg {
    fn parse(x: Stream) -> Res<Self> {
        let self_ = false;
        parse_fn_arg(x, self_)
    }
}
impl Lower for FnArg {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        if let Some((name, colon)) = &self.name {
            name.lower(s);
            colon.lower(s);
        }
        self.typ.lower(s);
    }
}

pub struct Variadic {
    pub attrs: Vec<attr::Attr>,
    pub name: Option<(Ident, Token![:])>,
    pub dots: Token![...],
    pub comma: Option<Token![,]>,
}
impl Lower for Variadic {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        if let Some((name, colon)) = &self.name {
            name.lower(s);
            colon.lower(s);
        }
        self.dots.lower(s);
        self.comma.lower(s);
    }
}

pub enum Ret {
    Default,
    Type(Token![->], Box<Type>),
}
impl Ret {
    pub fn without_plus(x: Stream) -> Res<Self> {
        let plus = false;
        Self::parse(x, plus)
    }
    pub fn parse(x: Stream, plus: bool) -> Res<Self> {
        if x.peek(Token![->]) {
            let arrow = x.parse()?;
            let group_gen = true;
            let ty = parse_ambig_typ(x, plus, group_gen)?;
            Ok(Ret::Type(arrow, Box::new(ty)))
        } else {
            Ok(Ret::Default)
        }
    }
}
impl Parse for Ret {
    fn parse(x: Stream) -> Res<Self> {
        let plus = true;
        Self::parse(x, plus)
    }
}
impl Lower for Ret {
    fn lower(&self, s: &mut Stream) {
        match self {
            Ret::Type(arrow, x) => {
                arrow.lower(s);
                x.lower(s);
            },
            Ret::Default => {},
        }
    }
}

pub fn parse_ambig_typ(s: Stream, plus: bool, gen: bool) -> Res<Type> {
    let beg = s.fork();
    if s.peek(tok::Group) {
        let mut y: Group = s.parse()?;
        if s.peek(Token![::]) && s.peek3(Ident::peek_any) {
            if let Type::Path(mut typ) = *y.elem {
                Path::parse_rest(s, &mut typ.path, false)?;
                return Ok(Type::Path(typ));
            } else {
                return Ok(Type::Path(Path {
                    qself: Some(path::QSelf {
                        lt: Token![<](y.group.span),
                        pos: 0,
                        as_: None,
                        gt: Token![>](y.group.span),
                        typ: y.elem,
                    }),
                    path: Path::parse_helper(s, false)?,
                }));
            }
        } else if s.peek(Token![<]) && gen || s.peek(Token![::]) && s.peek3(Token![<]) {
            if let Type::Path(mut ty) = *y.elem {
                let args = &mut ty.path.segs.last_mut().unwrap().args;
                if args.is_none() {
                    *args = path::Args::Angled(s.parse()?);
                    Path::parse_rest(s, &mut ty.path, false)?;
                    return Ok(Type::Path(ty));
                } else {
                    y.elem = Box::new(Type::Path(ty));
                }
            }
        }
        return Ok(Type::Group(y));
    }
    let mut lifes = None::<gen::bound::Lifes>;
    let mut look = s.look1();
    if look.peek(Token![for]) {
        lifes = s.parse()?;
        look = s.look1();
        if !look.peek(ident::Ident)
            && !look.peek(Token![fn])
            && !look.peek(Token![unsafe])
            && !look.peek(Token![extern])
            && !look.peek(Token![super])
            && !look.peek(Token![self])
            && !look.peek(Token![Self])
            && !look.peek(Token![crate])
            || s.peek(Token![dyn])
        {
            return Err(look.err());
        }
    }
    if look.peek(tok::Paren) {
        let gist;
        let paren = parenthesized!(gist in s);
        if gist.is_empty() {
            return Ok(Type::Tuple(Tuple {
                paren,
                elems: Puncted::new(),
            }));
        }
        if gist.peek(Life) {
            return Ok(Type::Paren(Paren {
                paren,
                elem: Box::new(Type::TraitObject(gist.parse()?)),
            }));
        }
        if gist.peek(Token![?]) {
            return Ok(Type::TraitObject(Trait {
                dyn_: None,
                bounds: {
                    let mut ys = Puncted::new();
                    ys.push_value(gen::bound::Type::Trait(gen::bound::Trait {
                        paren: Some(paren),
                        ..gist.parse()?
                    }));
                    while let Some(plus) = s.parse()? {
                        ys.push_punct(plus);
                        ys.push_value(s.parse()?);
                    }
                    ys
                },
            }));
        }
        let mut first: Type = gist.parse()?;
        if gist.peek(Token![,]) {
            return Ok(Type::Tuple(Tuple {
                paren,
                elems: {
                    let mut ys = Puncted::new();
                    ys.push_value(first);
                    ys.push_punct(gist.parse()?);
                    while !gist.is_empty() {
                        ys.push_value(gist.parse()?);
                        if gist.is_empty() {
                            break;
                        }
                        ys.push_punct(gist.parse()?);
                    }
                    ys
                },
            }));
        }
        if plus && s.peek(Token![+]) {
            loop {
                let first = match first {
                    Type::Path(Path { qself: None, path }) => gen::bound::Type::Trait(gen::bound::Trait {
                        paren: Some(paren),
                        modif: gen::bound::Modifier::None,
                        lifes: None,
                        path,
                    }),
                    Type::TraitObject(Trait { dyn_: None, bounds }) => {
                        if bounds.len() > 1 || bounds.trailing_punct() {
                            first = Type::TraitObject(Trait { dyn_: None, bounds });
                            break;
                        }
                        match bounds.into_iter().next().unwrap() {
                            gen::bound::Type::Trait(trait_bound) => gen::bound::Type::Trait(gen::bound::Trait {
                                paren: Some(paren),
                                ..trait_bound
                            }),
                            other @ (gen::bound::Type::Life(_) | gen::bound::Type::Stream(_)) => other,
                        }
                    },
                    _ => break,
                };
                return Ok(Type::TraitObject(Trait {
                    dyn_: None,
                    bounds: {
                        let mut ys = Puncted::new();
                        ys.push_value(first);
                        while let Some(plus) = s.parse()? {
                            ys.push_punct(plus);
                            ys.push_value(s.parse()?);
                        }
                        ys
                    },
                }));
            }
        }
        Ok(Type::Paren(Paren {
            paren,
            elem: Box::new(first),
        }))
    } else if look.peek(Token![fn]) || look.peek(Token![unsafe]) || look.peek(Token![extern]) {
        let mut y: Fn = s.parse()?;
        y.lifes = lifes;
        Ok(Type::Fn(y))
    } else if look.peek(ident::Ident)
        || s.peek(Token![super])
        || s.peek(Token![self])
        || s.peek(Token![Self])
        || s.peek(Token![crate])
        || look.peek(Token![::])
        || look.peek(Token![<])
    {
        let typ: Path = s.parse()?;
        if typ.qself.is_some() {
            return Ok(Type::Path(typ));
        }
        if s.peek(Token![!]) && !s.peek(Token![!=]) && typ.path.is_mod_style() {
            let bang: Token![!] = s.parse()?;
            let (delim, toks) = mac::parse_delim(s)?;
            return Ok(Type::Mac(Mac {
                mac: mac::Mac {
                    path: typ.path,
                    bang,
                    delim,
                    toks,
                },
            }));
        }
        if lifes.is_some() || plus && s.peek(Token![+]) {
            let mut bounds = Puncted::new();
            bounds.push_value(gen::bound::Type::Trait(gen::bound::Trait {
                paren: None,
                modif: gen::bound::Modifier::None,
                lifes,
                path: typ.path,
            }));
            if plus {
                while s.peek(Token![+]) {
                    bounds.push_punct(s.parse()?);
                    if !(s.peek(Ident::peek_any)
                        || s.peek(Token![::])
                        || s.peek(Token![?])
                        || s.peek(Life)
                        || s.peek(tok::Paren))
                    {
                        break;
                    }
                    bounds.push_value(s.parse()?);
                }
            }
            return Ok(Type::TraitObject(Trait { dyn_: None, bounds }));
        }
        Ok(Type::Path(typ))
    } else if look.peek(Token![dyn]) {
        let dyn_: Token![dyn] = s.parse()?;
        let dyn_span = dyn_.span;
        let star: Option<Token![*]> = s.parse()?;
        let bounds = Trait::parse_bounds(dyn_span, s, plus)?;
        return Ok(if star.is_some() {
            Type::Stream(parse::parse_verbatim(&beg, s))
        } else {
            Type::TraitObject(Trait {
                dyn_: Some(dyn_),
                bounds,
            })
        });
    } else if look.peek(tok::Bracket) {
        let gist;
        let bracket = bracketed!(gist in s);
        let elem: Type = gist.parse()?;
        if gist.peek(Token![;]) {
            Ok(Type::Array(Array {
                bracket,
                elem: Box::new(elem),
                semi: gist.parse()?,
                len: gist.parse()?,
            }))
        } else {
            Ok(Type::Slice(Slice {
                bracket,
                elem: Box::new(elem),
            }))
        }
    } else if look.peek(Token![*]) {
        s.parse().map(Type::Ptr)
    } else if look.peek(Token![&]) {
        s.parse().map(Type::Reference)
    } else if look.peek(Token![!]) && !s.peek(Token![=]) {
        s.parse().map(Type::Never)
    } else if look.peek(Token![impl]) {
        Impl::parse(s, plus).map(Type::ImplTrait)
    } else if look.peek(Token![_]) {
        s.parse().map(Type::Infer)
    } else if look.peek(Life) {
        s.parse().map(Type::TraitObject)
    } else {
        Err(look.err())
    }
}
fn parse_fn_arg(s: Stream, self_: bool) -> Res<FnArg> {
    let attrs = s.call(attr::Attr::parse_outers)?;
    let beg = s.fork();
    let has_mut_self = self_ && s.peek(Token![mut]) && s.peek2(Token![self]);
    if has_mut_self {
        s.parse::<Token![mut]>()?;
    }
    let mut has_self = false;
    let mut name = if (s.peek(ident::Ident) || s.peek(Token![_]) || {
        has_self = self_ && s.peek(Token![self]);
        has_self
    }) && s.peek2(Token![:])
        && !s.peek2(Token![::])
    {
        let name = s.call(Ident::parse_any)?;
        let colon: Token![:] = s.parse()?;
        Some((name, colon))
    } else {
        has_self = false;
        None
    };
    let ty = if self_ && !has_self && s.peek(Token![mut]) && s.peek2(Token![self]) {
        s.parse::<Token![mut]>()?;
        s.parse::<Token![self]>()?;
        None
    } else if has_mut_self && name.is_none() {
        s.parse::<Token![self]>()?;
        None
    } else {
        Some(s.parse()?)
    };
    let ty = match ty {
        Some(ty) if !has_mut_self => ty,
        _ => {
            name = None;
            Type::Stream(parse::parse_verbatim(&beg, s))
        },
    };
    Ok(FnArg { attrs, name, typ: ty })
}
fn parse_variadic(s: Stream, attrs: Vec<attr::Attr>) -> Res<Variadic> {
    Ok(Variadic {
        attrs,
        name: if s.peek(ident::Ident) || s.peek(Token![_]) {
            let y = s.call(Ident::parse_any)?;
            let colon: Token![:] = s.parse()?;
            Some((y, colon))
        } else {
            None
        },
        dots: s.parse()?,
        comma: s.parse()?,
    })
}
