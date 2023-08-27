use super::*;

enum_of_structs! {
    pub enum Type {
        Array(Array),
        Fn(Fn),
        Group(Group),
        Impl(Impl),
        Infer(Infer),
        Mac(Mac),
        Never(Never),
        Parenth(Parenth),
        Path(Path),
        Ptr(Ptr),
        Ref(Ref),
        Slice(Slice),
        Trait(Trait),
        Tuple(Tuple),
        Verbatim(Verbatim),
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
    fn parse(s: Stream) -> Res<Self> {
        let plus = true;
        let gen = true;
        parse_ambig_typ(s, plus, gen)
    }
}
impl Clone for Type {
    fn clone(&self) -> Self {
        use Type::*;
        match self {
            Array(x) => Array(x.clone()),
            Fn(x) => Fn(x.clone()),
            Group(x) => Group(x.clone()),
            Impl(x) => Impl(x.clone()),
            Infer(x) => Infer(x.clone()),
            Mac(x) => Mac(x.clone()),
            Never(x) => Never(x.clone()),
            Parenth(x) => Parenth(x.clone()),
            Path(x) => Path(x.clone()),
            Ptr(x) => Ptr(x.clone()),
            Ref(x) => Ref(x.clone()),
            Slice(x) => Slice(x.clone()),
            Trait(x) => Trait(x.clone()),
            Tuple(x) => Tuple(x.clone()),
            Verbatim(x) => Verbatim(x.clone()),
        }
    }
}
impl Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Type::")?;
        use Type::*;
        match self {
            Array(x) => x.debug(f, "Array"),
            Fn(x) => x.debug(f, "Fn"),
            Group(x) => x.debug(f, "Group"),
            Impl(x) => x.debug(f, "ImplTrait"),
            Infer(x) => x.debug(f, "Infer"),
            Mac(x) => x.debug(f, "Macro"),
            Never(x) => x.debug(f, "Never"),
            Parenth(x) => x.debug(f, "Parenth"),
            Path(x) => x.debug(f, "Path"),
            Ptr(x) => x.debug(f, "Ptr"),
            Ref(x) => x.debug(f, "Reference"),
            Slice(x) => x.debug(f, "Slice"),
            Trait(x) => x.debug(f, "TraitObject"),
            Tuple(x) => x.debug(f, "Tuple"),
            Verbatim(x) => {
                let mut y = f.debug_tuple("Stream");
                y.field(x);
                y.finish()
            },
        }
    }
}
impl Eq for Type {}
impl PartialEq for Type {
    fn eq(&self, x: &Self) -> bool {
        use Type::*;
        match (self, x) {
            (Array(x), Array(y)) => x == y,
            (Fn(x), Fn(y)) => x == y,
            (Group(x), Group(y)) => x == y,
            (Impl(x), Impl(y)) => x == y,
            (Infer(x), Infer(y)) => x == y,
            (Mac(x), Mac(y)) => x == y,
            (Never(x), Never(y)) => x == y,
            (Parenth(x), Parenth(y)) => x == y,
            (Path(x), Path(y)) => x == y,
            (Ptr(x), Ptr(y)) => x == y,
            (Ref(x), Ref(y)) => x == y,
            (Slice(x), Slice(y)) => x == y,
            (Trait(x), Trait(y)) => x == y,
            (Tuple(x), Tuple(y)) => x == y,
            (Verbatim(x), Verbatim(y)) => StreamHelper(x) == StreamHelper(y),
            _ => false,
        }
    }
}
impl Pretty for Type {
    fn pretty(&self, p: &mut Print) {
        use Type::*;
        match self {
            Array(x) => x.pretty(p),
            Fn(x) => x.pretty(p),
            Group(x) => x.pretty(p),
            Impl(x) => x.pretty(p),
            Infer(x) => x.pretty(p),
            Mac(x) => x.pretty(p),
            Never(x) => x.pretty(p),
            Parenth(x) => x.pretty(p),
            Path(x) => x.pretty(p),
            Ptr(x) => x.pretty(p),
            Ref(x) => x.pretty(p),
            Slice(x) => x.pretty(p),
            Trait(x) => x.pretty(p),
            Tuple(x) => x.pretty(p),
            Verbatim(x) => x.pretty(p),
        }
    }
}
impl<F: Folder + ?Sized> Fold for Type {
    fn fold(&self, f: &mut F) {
        use Type::*;
        match self {
            Array(x) => Array(x.fold(f)),
            Fn(x) => Fn(x.fold(f)),
            Group(x) => Group(x.fold(f)),
            Impl(x) => Impl(x.fold(f)),
            Infer(x) => Infer(x.fold(f)),
            Mac(x) => Mac(x.fold(f)),
            Never(x) => Never(x.fold(f)),
            Parenth(x) => Parenth(x.fold(f)),
            Path(x) => Path(x.fold(f)),
            Ptr(x) => Ptr(x.fold(f)),
            Ref(x) => Ref(x.fold(f)),
            Slice(x) => Slice(x.fold(f)),
            Trait(x) => Trait(x.fold(f)),
            Tuple(x) => Tuple(x.fold(f)),
            Verbatim(x) => Verbatim(x),
        }
    }
}
impl<H: Hasher> Hash for Type {
    fn hash(&self, h: &mut H) {
        use Type::*;
        match self {
            Array(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Fn(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Group(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Impl(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Infer(x) => {
                h.write_u8(4u8);
                x.hash(h);
            },
            Mac(x) => {
                h.write_u8(5u8);
                x.hash(h);
            },
            Never(x) => {
                h.write_u8(6u8);
                x.hash(h);
            },
            Parenth(x) => {
                h.write_u8(7u8);
                x.hash(h);
            },
            Path(x) => {
                h.write_u8(8u8);
                x.hash(h);
            },
            Ptr(x) => {
                h.write_u8(9u8);
                x.hash(h);
            },
            Ref(x) => {
                h.write_u8(10u8);
                x.hash(h);
            },
            Slice(x) => {
                h.write_u8(11u8);
                x.hash(h);
            },
            Trait(x) => {
                h.write_u8(12u8);
                x.hash(h);
            },
            Tuple(x) => {
                h.write_u8(13u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(14u8);
                StreamHelper(x).hash(h);
            },
        }
    }
}
impl<V: Visitor + ?Sized> Visit for Type {
    fn visit(&self, v: &mut V) {
        use Type::*;
        match self {
            Array(x) => {
                x.visit(v);
            },
            Fn(x) => {
                x.visit(v);
            },
            Group(x) => {
                x.visit(v);
            },
            Impl(x) => {
                x.visit(v);
            },
            Infer(x) => {
                x.visit(v);
            },
            Mac(x) => {
                x.visit(v);
            },
            Never(x) => {
                x.visit(v);
            },
            Parenth(x) => {
                x.visit(v);
            },
            Path(x) => {
                x.visit(v);
            },
            Ptr(x) => {
                x.visit(v);
            },
            Ref(x) => {
                x.visit(v);
            },
            Slice(x) => {
                x.visit(v);
            },
            Trait(x) => {
                x.visit(v);
            },
            Tuple(x) => {
                x.visit(v);
            },
            Verbatim(_) => {},
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        use Type::*;
        match self {
            Array(x) => {
                x.visit_mut(v);
            },
            Fn(x) => {
                x.visit_mut(v);
            },
            Group(x) => {
                x.visit_mut(v);
            },
            Impl(x) => {
                x.visit_mut(v);
            },
            Infer(x) => {
                x.visit_mut(v);
            },
            Mac(x) => {
                x.visit_mut(v);
            },
            Never(x) => {
                x.visit_mut(v);
            },
            Parenth(x) => {
                x.visit_mut(v);
            },
            Path(x) => {
                x.visit_mut(v);
            },
            Ptr(x) => {
                x.visit_mut(v);
            },
            Ref(x) => {
                x.visit_mut(v);
            },
            Slice(x) => {
                x.visit_mut(v);
            },
            Trait(x) => {
                x.visit_mut(v);
            },
            Tuple(x) => {
                x.visit_mut(v);
            },
            Verbatim(_) => {},
        }
    }
}

pub struct Array {
    pub bracket: tok::Bracket,
    pub elem: Box<Type>,
    pub semi: Token![;],
    pub len: expr::Expr,
}
impl Parse for Array {
    fn parse(s: Stream) -> Res<Self> {
        let y;
        Ok(Array {
            bracket: bracketed!(y in s),
            elem: y.parse()?,
            semi: y.parse()?,
            len: y.parse()?,
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
impl Clone for Array {
    fn clone(&self) -> Self {
        Array {
            bracket: self.bracket.clone(),
            elem: self.elem.clone(),
            semi: self.semi.clone(),
            len: self.len.clone(),
        }
    }
}
impl Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Array {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut y = f.debug_struct(x);
                y.field("bracket", &self.bracket);
                y.field("elem", &self.elem);
                y.field("semi", &self.semi);
                y.field("len", &self.len);
                y.finish()
            }
        }
        self.debug(f, "typ::Array")
    }
}
impl Eq for Array {}
impl PartialEq for Array {
    fn eq(&self, x: &Self) -> bool {
        self.elem == x.elem && self.len == x.len
    }
}
impl Pretty for Array {
    fn pretty(&self, p: &mut Print) {
        p.word("[");
        &self.elem.pretty(p);
        p.word("; ");
        &self.len.pretty(p);
        p.word("]");
    }
}
impl<F: Folder + ?Sized> Fold for Array {
    fn fold(&self, f: &mut F) {
        Array {
            bracket: self.bracket,
            elem: Box::new(*self.elem.fold(f)),
            semi: self.semi,
            len: self.len.fold(f),
        }
    }
}
impl<H: Hasher> Hash for Array {
    fn hash(&self, h: &mut H) {
        self.elem.hash(h);
        self.len.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Array {
    fn visit(&self, v: &mut V) {
        &*self.elem.visit(v);
        &self.len.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut *self.elem.visit_mut(v);
        &mut self.len.visit_mut(v);
    }
}

pub struct Fn {
    pub lifes: Option<gen::bound::Lifes>,
    pub unsafe_: Option<Token![unsafe]>,
    pub abi: Option<Abi>,
    pub fn_: Token![fn],
    pub parenth: tok::Parenth,
    pub args: Puncted<FnArg, Token![,]>,
    pub vari: Option<Variadic>,
    pub ret: Ret,
}
impl Parse for Fn {
    fn parse(s: Stream) -> Res<Self> {
        let args;
        let mut vari = None;
        Ok(Fn {
            lifes: s.parse()?,
            unsafe_: s.parse()?,
            abi: s.parse()?,
            fn_: s.parse()?,
            parenth: parenthed!(args in s),
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
                    let y = parse_fn_arg(&args, allow_self)?;
                    ys.push_value(FnArg { attrs, ..y });
                    if args.is_empty() {
                        break;
                    }
                    let y = args.parse()?;
                    ys.push_punct(y);
                }
                ys
            },
            vari,
            ret: s.call(Ret::without_plus)?,
        })
    }
}
impl Lower for Fn {
    fn lower(&self, s: &mut Stream) {
        self.lifes.lower(s);
        self.unsafe_.lower(s);
        self.abi.lower(s);
        self.fn_.lower(s);
        self.parenth.surround(s, |s| {
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
impl Clone for Fn {
    fn clone(&self) -> Self {
        Fn {
            lifes: self.lifes.clone(),
            unsafe_: self.unsafe_.clone(),
            abi: self.abi.clone(),
            fn_: self.fn_.clone(),
            parenth: self.parenth.clone(),
            args: self.args.clone(),
            vari: self.vari.clone(),
            ret: self.ret.clone(),
        }
    }
}
impl Debug for Fn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Fn {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut y = f.debug_struct(x);
                y.field("lifes", &self.lifes);
                y.field("unsafe_", &self.unsafe_);
                y.field("abi", &self.abi);
                y.field("fn_", &self.fn_);
                y.field("parenth", &self.parenth);
                y.field("inputs", &self.args);
                y.field("vari", &self.vari);
                y.field("output", &self.ret);
                y.finish()
            }
        }
        self.debug(f, "typ::Fn")
    }
}
impl Eq for Fn {}
impl PartialEq for Fn {
    fn eq(&self, x: &Self) -> bool {
        self.lifes == x.lifes
            && self.unsafe_ == x.unsafe_
            && self.abi == x.abi
            && self.args == x.args
            && self.vari == x.vari
            && self.ret == x.ret
    }
}
impl Pretty for Fn {
    fn pretty(&self, p: &mut Print) {
        if let Some(x) = &self.lifes {
            x.pretty(p);
        }
        if self.unsafe_.is_some() {
            p.word("unsafe ");
        }
        if let Some(x) = &self.abi {
            x.pretty(p);
        }
        p.word("fn(");
        p.cbox(INDENT);
        p.zerobreak();
        for x in self.args.iter().delimited() {
            &x.pretty(p);
            p.trailing_comma(x.is_last && self.vari.is_none());
        }
        if let Some(x) = &self.vari {
            x.pretty(p);
            p.zerobreak();
        }
        p.offset(-INDENT);
        p.end();
        p.word(")");
        &self.ret.pretty(p)
    }
}
impl<F: Folder + ?Sized> Fold for Fn {
    fn fold(&self, f: &mut F) {
        Fn {
            lifes: (self.lifes).map(|x| x.fold(f)),
            unsafe_: self.unsafe_,
            abi: (self.abi).map(|x| x.fold(f)),
            fn_: self.fn_,
            parenth: self.parenth,
            args: FoldHelper::lift(self.args, |x| x.fold(f)),
            vari: (self.vari).map(|x| x.fold(f)),
            ret: self.ret.fold(f),
        }
    }
}
impl<H: Hasher> Hash for Fn {
    fn hash(&self, h: &mut H) {
        self.lifes.hash(h);
        self.unsafe_.hash(h);
        self.abi.hash(h);
        self.args.hash(h);
        self.vari.hash(h);
        self.ret.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Fn {
    fn visit(&self, v: &mut V) {
        if let Some(x) = &self.lifes {
            x.visit(v);
        }
        if let Some(x) = &self.abi {
            x.visit(v);
        }
        for y in Puncted::pairs(&self.args) {
            let x = y.value();
            x.visit(v);
        }
        if let Some(x) = &self.vari {
            x.visit(v);
        }
        &self.ret.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        if let Some(x) = &mut self.lifes {
            x.visit_mut(v);
        }
        if let Some(x) = &mut self.abi {
            x.visit_mut(v);
        }
        for mut y in Puncted::pairs_mut(&mut self.args) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
        if let Some(x) = &mut self.vari {
            x.visit_mut(v);
        }
        &mut self.ret.visit_mut(v);
    }
}

pub struct Group {
    pub group: tok::Group,
    pub elem: Box<Type>,
}
impl Parse for Group {
    fn parse(s: Stream) -> Res<Self> {
        let y = parse::parse_group(s)?;
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
impl Clone for Group {
    fn clone(&self) -> Self {
        Group {
            group: self.group.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Debug for Group {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Group {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut y = f.debug_struct(x);
                y.field("group", &self.group);
                y.field("elem", &self.elem);
                y.finish()
            }
        }
        self.debug(f, "typ::Group")
    }
}
impl Eq for Group {}
impl PartialEq for Group {
    fn eq(&self, x: &Self) -> bool {
        self.elem == x.elem
    }
}
impl Pretty for Group {
    fn pretty(&self, p: &mut Print) {
        &self.elem.pretty(p);
    }
}
impl<F: Folder + ?Sized> Fold for Group {
    fn fold(&self, f: &mut F) {
        Group {
            group: self.group,
            elem: Box::new(*self.elem.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Group {
    fn hash(&self, h: &mut H) {
        self.elem.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Group {
    fn visit(&self, v: &mut V) {
        &*self.elem.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut *self.elem.visit_mut(v);
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
    pub fn parse(s: Stream, plus: bool) -> Res<Self> {
        let impl_: Token![impl] = s.parse()?;
        let bounds = gen::bound::Type::parse_many(s, plus)?;
        let mut last = None;
        let mut one = false;
        for x in &bounds {
            match x {
                gen::bound::Type::Trait(_) | gen::bound::Type::Verbatim(_) => {
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
    fn parse(s: Stream) -> Res<Self> {
        let plus = true;
        Self::parse(s, plus)
    }
}
impl Lower for Impl {
    fn lower(&self, s: &mut Stream) {
        self.impl_.lower(s);
        self.bounds.lower(s);
    }
}
impl Clone for Impl {
    fn clone(&self) -> Self {
        Impl {
            impl_: self.impl_.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Debug for Impl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Impl {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut y = f.debug_struct(x);
                y.field("impl_", &self.impl_);
                y.field("bounds", &self.bounds);
                y.finish()
            }
        }
        self.debug(f, "typ::Impl")
    }
}
impl Eq for Impl {}
impl PartialEq for Impl {
    fn eq(&self, x: &Self) -> bool {
        self.bounds == x.bounds
    }
}
impl Pretty for Impl {
    fn pretty(&self, p: &mut Print) {
        p.word("impl ");
        for x in self.bounds.iter().delimited() {
            if !x.is_first {
                p.word(" + ");
            }
            &x.pretty(p);
        }
    }
}
impl<F: Folder + ?Sized> Fold for Impl {
    fn fold(&self, f: &mut F) {
        Impl {
            impl_: self.impl_,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Impl {
    fn hash(&self, h: &mut H) {
        self.bounds.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Impl {
    fn visit(&self, v: &mut V) {
        for y in Puncted::pairs(&self.bounds) {
            let x = y.value();
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for mut y in Puncted::pairs_mut(&mut self.bounds) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
    }
}

pub struct Infer {
    pub underscore: Token![_],
}
impl Parse for Infer {
    fn parse(s: Stream) -> Res<Self> {
        Ok(Infer { underscore: s.parse()? })
    }
}
impl Lower for Infer {
    fn lower(&self, s: &mut Stream) {
        self.underscore.lower(s);
    }
}
impl Clone for Infer {
    fn clone(&self) -> Self {
        Infer {
            underscore: self.underscore.clone(),
        }
    }
}
impl Debug for Infer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Infer {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("underscore", &self.underscore);
                f.finish()
            }
        }
        self.debug(f, "typ::Infer")
    }
}
impl Eq for Infer {}
impl PartialEq for Infer {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Pretty for Infer {
    fn pretty(&self, p: &mut Print) {
        let _ = self;
        p.word("_");
    }
}
impl<F: Folder + ?Sized> Fold for Infer {
    fn fold(&self, f: &mut F) {
        Infer {
            underscore: self.underscore,
        }
    }
}
impl<H: Hasher> Hash for Infer {
    fn hash(&self, _: &mut H) {}
}
impl<V: Visitor + ?Sized> Visit for Infer {
    fn visit(&self, v: &mut V) {}
    fn visit_mut(&mut self, v: &mut V) {}
}

pub struct Mac {
    pub mac: mac::Mac,
}
impl Parse for Mac {
    fn parse(s: Stream) -> Res<Self> {
        Ok(Mac { mac: s.parse()? })
    }
}
impl Lower for Mac {
    fn lower(&self, s: &mut Stream) {
        self.mac.lower(s);
    }
}
impl Clone for Mac {
    fn clone(&self) -> Self {
        Mac { mac: self.mac.clone() }
    }
}
impl Debug for Mac {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Mac {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("mac", &self.mac);
                f.finish()
            }
        }
        self.debug(f, "typ::Mac")
    }
}
impl Eq for Mac {}
impl PartialEq for Mac {
    fn eq(&self, x: &Self) -> bool {
        self.mac == x.mac
    }
}
impl Pretty for Mac {
    fn pretty(&self, p: &mut Print) {
        let semi = false;
        &self.mac.pretty_with_args(p, (None, semi));
    }
}
impl<F: Folder + ?Sized> Fold for Mac {
    fn fold(&self, f: &mut F) {
        Mac { mac: self.mac.fold(f) }
    }
}
impl<H: Hasher> Hash for Mac {
    fn hash(&self, h: &mut H) {
        self.mac.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Mac {
    fn visit(&self, v: &mut V) {
        &self.mac.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut self.mac.visit_mut(v);
    }
}

pub struct Never {
    pub bang: Token![!],
}
impl Parse for Never {
    fn parse(s: Stream) -> Res<Self> {
        Ok(Never { bang: s.parse()? })
    }
}
impl Lower for Never {
    fn lower(&self, s: &mut Stream) {
        self.bang.lower(s);
    }
}
impl Clone for Never {
    fn clone(&self) -> Self {
        Never {
            bang: self.bang.clone(),
        }
    }
}
impl Debug for Never {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Never {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("bang", &self.bang);
                f.finish()
            }
        }
        self.debug(f, "typ::Never")
    }
}
impl Eq for Never {}
impl PartialEq for Never {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Pretty for Never {
    fn pretty(&self, p: &mut Print) {
        let _ = self;
        p.word("!");
    }
}
impl<F: Folder + ?Sized> Fold for Never {
    fn fold(&self, f: &mut F) {
        Never { bang: self.bang }
    }
}
impl<H: Hasher> Hash for Never {
    fn hash(&self, h: &mut H) {}
}
impl<V: Visitor + ?Sized> Visit for Never {
    fn visit(&self, v: &mut V) {}
    fn visit_mut(&mut self, v: &mut V) {}
}

pub struct Parenth {
    pub parenth: tok::Parenth,
    pub elem: Box<Type>,
}
impl Parenth {
    fn parse(s: Stream, plus: bool) -> Res<Self> {
        let y;
        Ok(Parenth {
            parenth: parenthed!(y in s),
            elem: Box::new({
                let gen = true;
                parse_ambig_typ(&y, plus, gen)?
            }),
        })
    }
}
impl Parse for Parenth {
    fn parse(s: Stream) -> Res<Self> {
        let plus = false;
        Self::parse(s, plus)
    }
}
impl Lower for Parenth {
    fn lower(&self, s: &mut Stream) {
        self.parenth.surround(s, |s| {
            self.elem.lower(s);
        });
    }
}
impl Clone for Parenth {
    fn clone(&self) -> Self {
        Parenth {
            parenth: self.parenth.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Debug for Parenth {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Parenth {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("parenth", &self.parenth);
                f.field("elem", &self.elem);
                f.finish()
            }
        }
        self.debug(f, "typ::Parenth")
    }
}
impl Eq for Parenth {}
impl PartialEq for Parenth {
    fn eq(&self, x: &Self) -> bool {
        self.elem == x.elem
    }
}
impl Pretty for Parenth {
    fn pretty(&self, p: &mut Print) {
        p.word("(");
        &self.elem.pretty(p);
        p.word(")");
    }
}
impl<F: Folder + ?Sized> Fold for Parenth {
    fn fold(&self, f: &mut F) {
        Parenth {
            parenth: self.parenth,
            elem: Box::new(*self.elem.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Parenth {
    fn hash(&self, h: &mut H) {
        self.elem.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Parenth {
    fn visit(&self, v: &mut V) {
        &*self.elem.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut *self.elem.visit_mut(v);
    }
}

pub struct Path {
    pub qself: Option<path::QSelf>,
    pub path: Path,
}
impl Parse for Path {
    fn parse(s: Stream) -> Res<Self> {
        let style = false;
        let (qself, path) = path::qpath(s, style)?;
        Ok(Path { qself, path })
    }
}
impl Lower for Path {
    fn lower(&self, s: &mut Stream) {
        path::path_lower(s, &self.qself, &self.path);
    }
}
impl Clone for Path {
    fn clone(&self) -> Self {
        Path {
            qself: self.qself.clone(),
            path: self.path.clone(),
        }
    }
}
impl Debug for Path {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Path {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("qself", &self.qself);
                f.field("path", &self.path);
                f.finish()
            }
        }
        self.debug(f, "typ::Path")
    }
}
impl Eq for Path {}
impl PartialEq for Path {
    fn eq(&self, x: &Self) -> bool {
        self.qself == x.qself && self.path == x.path
    }
}
impl Pretty for Path {
    fn pretty(&self, p: &mut Print) {
        &self.path.pretty_qpath(p, &self.qself, path::Kind::Type);
    }
}
impl<F: Folder + ?Sized> Fold for Path {
    fn fold(&self, f: &mut F) {
        Path {
            qself: (self.qself).map(|x| x.fold(f)),
            path: self.path.fold(f),
        }
    }
}
impl<H: Hasher> Hash for Path {
    fn hash(&self, h: &mut H) {
        self.qself.hash(h);
        self.path.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Path {
    fn visit(&self, v: &mut V) {
        if let Some(x) = &self.qself {
            x.visit(v);
        }
        &self.path.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        if let Some(x) = &mut self.qself {
            x.visit_mut(v);
        }
        &mut self.path.visit_mut(v);
    }
}

pub struct Ptr {
    pub star: Token![*],
    pub const_: Option<Token![const]>,
    pub mut_: Option<Token![mut]>,
    pub elem: Box<Type>,
}
impl Parse for Ptr {
    fn parse(s: Stream) -> Res<Self> {
        let star: Token![*] = s.parse()?;
        let look = s.look1();
        let (const_, mut_) = if look.peek(Token![const]) {
            (Some(s.parse()?), None)
        } else if look.peek(Token![mut]) {
            (None, Some(s.parse()?))
        } else {
            return Err(look.err());
        };
        Ok(Ptr {
            star,
            const_,
            mut_,
            elem: Box::new(s.call(Type::without_plus)?),
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
impl Clone for Ptr {
    fn clone(&self) -> Self {
        Ptr {
            star: self.star.clone(),
            const_: self.const_.clone(),
            mut_: self.mut_.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Debug for Ptr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Ptr {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("star", &self.star);
                f.field("const_", &self.const_);
                f.field("mut_", &self.mut_);
                f.field("elem", &self.elem);
                f.finish()
            }
        }
        self.debug(f, "typ::Ptr")
    }
}
impl Eq for Ptr {}
impl PartialEq for Ptr {
    fn eq(&self, x: &Self) -> bool {
        self.const_ == x.const_ && self.mut_ == x.mut_ && self.elem == x.elem
    }
}
impl Pretty for Ptr {
    fn pretty(&self, p: &mut Print) {
        p.word("*");
        if self.mut_.is_some() {
            p.word("mut ");
        } else {
            p.word("const ");
        }
        &self.elem.pretty(p);
    }
}
impl<F: Folder + ?Sized> Fold for Ptr {
    fn fold(&self, f: &mut F) {
        Ptr {
            star: self.star,
            const_: self.const_,
            mut_: self.mut_,
            elem: Box::new(*self.elem.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Ptr {
    fn hash(&self, h: &mut H) {
        self.const_.hash(h);
        self.mut_.hash(h);
        self.elem.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Ptr {
    fn visit(&self, v: &mut V) {
        &*self.elem.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut *self.elem.visit_mut(v);
    }
}

pub struct Ref {
    pub and: Token![&],
    pub life: Option<Life>,
    pub mut_: Option<Token![mut]>,
    pub elem: Box<Type>,
}
impl Parse for Ref {
    fn parse(s: Stream) -> Res<Self> {
        Ok(Ref {
            and: s.parse()?,
            life: s.parse()?,
            mut_: s.parse()?,
            elem: Box::new(s.call(Type::without_plus)?),
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
impl Clone for Ref {
    fn clone(&self) -> Self {
        Ref {
            and: self.and.clone(),
            life: self.life.clone(),
            mut_: self.mut_.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Debug for Ref {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Ref {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("and", &self.and);
                f.field("life", &self.life);
                f.field("mut_", &self.mut_);
                f.field("elem", &self.elem);
                f.finish()
            }
        }
        self.debug(f, "typ::Ref")
    }
}
impl Eq for Ref {}
impl PartialEq for Ref {
    fn eq(&self, x: &Self) -> bool {
        self.life == x.life && self.mut_ == x.mut_ && self.elem == x.elem
    }
}
impl Pretty for Ref {
    fn pretty(&self, p: &mut Print) {
        p.word("&");
        if let Some(x) = &self.life {
            x.pretty(p);
            p.nbsp();
        }
        if self.mut_.is_some() {
            p.word("mut ");
        }
        &self.elem.pretty(p);
    }
}
impl<F: Folder + ?Sized> Fold for Ref {
    fn fold(&self, f: &mut F) {
        Ref {
            and: self.and,
            life: (self.life).map(|x| x.fold(f)),
            mut_: self.mut_,
            elem: Box::new(*self.elem.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Ref {
    fn hash(&self, h: &mut H) {
        self.life.hash(h);
        self.mut_.hash(h);
        self.elem.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Ref {
    fn visit(&self, v: &mut V) {
        if let Some(x) = &self.life {
            x.visit(v);
        }
        &*self.elem.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        if let Some(x) = &mut self.life {
            x.visit_mut(v);
        }
        &mut *self.elem.visit_mut(v);
    }
}

pub struct Slice {
    pub bracket: tok::Bracket,
    pub elem: Box<Type>,
}
impl Parse for Slice {
    fn parse(s: Stream) -> Res<Self> {
        let y;
        Ok(Slice {
            bracket: bracketed!(y in s),
            elem: y.parse()?,
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
impl Clone for Slice {
    fn clone(&self) -> Self {
        Slice {
            bracket: self.bracket.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Debug for Slice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Slice {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("bracket", &self.bracket);
                f.field("elem", &self.elem);
                f.finish()
            }
        }
        self.debug(f, "typ::Slice")
    }
}
impl Eq for Slice {}
impl PartialEq for Slice {
    fn eq(&self, x: &Self) -> bool {
        self.elem == x.elem
    }
}
impl Pretty for Slice {
    fn pretty(&self, p: &mut Print) {
        p.word("[");
        &self.elem.pretty(p);
        p.word("]");
    }
}
impl<F: Folder + ?Sized> Fold for Slice {
    fn fold(&self, f: &mut F) {
        Slice {
            bracket: self.bracket,
            elem: Box::new(*self.elem.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Slice {
    fn hash(&self, h: &mut H) {
        self.elem.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Slice {
    fn visit(&self, v: &mut V) {
        &*self.elem.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut *self.elem.visit_mut(v);
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
    pub fn parse(s: Stream, plus: bool) -> Res<Self> {
        let dyn_: Option<Token![dyn]> = s.parse()?;
        let span = match &dyn_ {
            Some(x) => x.span,
            None => s.span(),
        };
        let bounds = Self::parse_bounds(span, s, plus)?;
        Ok(Trait { dyn_, bounds })
    }
    fn parse_bounds(s: Span, x: Stream, plus: bool) -> Res<Puncted<gen::bound::Type, Token![+]>> {
        let ys = gen::bound::Type::parse_many(x, plus)?;
        let mut last = None;
        let mut one = false;
        for y in &ys {
            match y {
                gen::bound::Type::Trait(_) | gen::bound::Type::Verbatim(_) => {
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
    fn parse(s: Stream) -> Res<Self> {
        let plus = true;
        Self::parse(s, plus)
    }
}
impl Lower for Trait {
    fn lower(&self, s: &mut Stream) {
        self.dyn_.lower(s);
        self.bounds.lower(s);
    }
}
impl Clone for Trait {
    fn clone(&self) -> Self {
        Trait {
            dyn_: self.dyn_.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Debug for Trait {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Trait {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("dyn_", &self.dyn_);
                f.field("bounds", &self.bounds);
                f.finish()
            }
        }
        self.debug(f, "typ::Trait")
    }
}
impl Eq for Trait {}
impl PartialEq for Trait {
    fn eq(&self, x: &Self) -> bool {
        self.dyn_ == x.dyn_ && self.bounds == x.bounds
    }
}
impl Pretty for Trait {
    fn pretty(&self, p: &mut Print) {
        p.word("dyn ");
        for x in self.bounds.iter().delimited() {
            if !x.is_first {
                p.word(" + ");
            }
            &x.pretty(p);
        }
    }
}
impl<F: Folder + ?Sized> Fold for Trait {
    fn fold(&self, f: &mut F) {
        Trait {
            dyn_: self.dyn_,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Trait {
    fn hash(&self, h: &mut H) {
        self.dyn_.hash(h);
        self.bounds.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Trait {
    fn visit(&self, v: &mut V) {
        for y in Puncted::pairs(&self.bounds) {
            let x = y.value();
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for mut y in Puncted::pairs_mut(&mut self.bounds) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
    }
}

pub struct Tuple {
    pub parenth: tok::Parenth,
    pub elems: Puncted<Type, Token![,]>,
}
impl Parse for Tuple {
    fn parse(s: Stream) -> Res<Self> {
        let y;
        let parenth = parenthed!(y in s);
        if y.is_empty() {
            return Ok(Tuple {
                parenth,
                elems: Puncted::new(),
            });
        }
        let first: Type = y.parse()?;
        Ok(Tuple {
            parenth,
            elems: {
                let mut ys = Puncted::new();
                ys.push_value(first);
                ys.push_punct(y.parse()?);
                while !y.is_empty() {
                    ys.push_value(y.parse()?);
                    if y.is_empty() {
                        break;
                    }
                    ys.push_punct(y.parse()?);
                }
                ys
            },
        })
    }
}
impl Lower for Tuple {
    fn lower(&self, s: &mut Stream) {
        self.parenth.surround(s, |s| {
            self.elems.lower(s);
            if self.elems.len() == 1 && !self.elems.trailing_punct() {
                <Token![,]>::default().lower(s);
            }
        });
    }
}
impl Clone for Tuple {
    fn clone(&self) -> Self {
        Tuple {
            parenth: self.parenth.clone(),
            elems: self.elems.clone(),
        }
    }
}
impl Debug for Tuple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Tuple {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("parenth", &self.parenth);
                f.field("elems", &self.elems);
                f.finish()
            }
        }
        self.debug(f, "typ::Tuple")
    }
}
impl Eq for Tuple {}
impl PartialEq for Tuple {
    fn eq(&self, x: &Self) -> bool {
        self.elems == x.elems
    }
}
impl Pretty for Tuple {
    fn pretty(&self, p: &mut Print) {
        p.word("(");
        p.cbox(INDENT);
        p.zerobreak();
        for x in self.elems.iter().delimited() {
            &x.pretty(p);
            if self.elems.len() == 1 {
                p.word(",");
                p.zerobreak();
            } else {
                p.trailing_comma(x.is_last);
            }
        }
        p.offset(-INDENT);
        p.end();
        p.word(")");
    }
}
impl<F: Folder + ?Sized> Fold for Tuple {
    fn fold(&self, f: &mut F) {
        Tuple {
            parenth: self.parenth,
            elems: FoldHelper::lift(self.elems, |x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Tuple {
    fn hash(&self, h: &mut H) {
        self.elems.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Tuple {
    fn visit(&self, v: &mut V) {
        for y in Puncted::pairs(&self.elems) {
            let x = y.value();
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for mut y in Puncted::pairs_mut(&mut self.elems) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
    }
}

pub struct Abi {
    pub extern_: Token![extern],
    pub name: Option<lit::Str>,
}
impl Lower for Abi {
    fn lower(&self, s: &mut Stream) {
        self.extern_.lower(s);
        self.name.lower(s);
    }
}
impl Parse for Abi {
    fn parse(s: Stream) -> Res<Self> {
        Ok(Abi {
            extern_: s.parse()?,
            name: s.parse()?,
        })
    }
}
impl Parse for Option<Abi> {
    fn parse(s: Stream) -> Res<Self> {
        if s.peek(Token![extern]) {
            s.parse().map(Some)
        } else {
            Ok(None)
        }
    }
}
impl Clone for Abi {
    fn clone(&self) -> Self {
        Abi {
            extern_: self.extern_.clone(),
            name: self.name.clone(),
        }
    }
}
impl Debug for Abi {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Abi");
        f.field("extern_", &self.extern_);
        f.field("name", &self.name);
        f.finish()
    }
}
impl Eq for Abi {}
impl PartialEq for Abi {
    fn eq(&self, x: &Self) -> bool {
        self.name == x.name
    }
}
impl Pretty for Abi {
    fn pretty(&self, p: &mut Print) {
        p.word("extern ");
        if let Some(x) = &self.name {
            x.pretty(p);
            p.nbsp();
        }
    }
}
impl<F: Folder + ?Sized> Fold for Abi {
    fn fold(&self, f: &mut F) {
        Abi {
            extern_: self.extern_,
            name: (self.name).map(|x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Abi {
    fn hash(&self, h: &mut H) {
        self.name.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Abi {
    fn visit(&self, v: &mut V) {
        if let Some(x) = &self.name {
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        if let Some(x) = &mut self.name {
            x.visit_mut(v);
        }
    }
}

pub struct FnArg {
    pub attrs: Vec<attr::Attr>,
    pub name: Option<(Ident, Token![:])>,
    pub typ: Type,
}
impl Parse for FnArg {
    fn parse(s: Stream) -> Res<Self> {
        let self_ = false;
        parse_fn_arg(s, self_)
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
impl Clone for FnArg {
    fn clone(&self) -> Self {
        FnArg {
            attrs: self.attrs.clone(),
            name: self.name.clone(),
            typ: self.typ.clone(),
        }
    }
}
impl Debug for FnArg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::FnArg");
        f.field("attrs", &self.attrs);
        f.field("name", &self.name);
        f.field("ty", &self.typ);
        f.finish()
    }
}
impl Eq for FnArg {}
impl PartialEq for FnArg {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.name == x.name && self.typ == x.typ
    }
}
impl Pretty for FnArg {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        if let Some((x, _)) = &self.name {
            x.pretty(p);
            p.word(": ");
        }
        &self.typ.pretty(p);
    }
}
impl<F: Folder + ?Sized> Fold for FnArg {
    fn fold(&self, f: &mut F) {
        FnArg {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            name: (self.name).map(|x| ((x).0.fold(f), (x).1)),
            typ: self.typ.fold(f),
        }
    }
}
impl<H: Hasher> Hash for FnArg {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.name.hash(h);
        self.typ.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for FnArg {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        if let Some(x) = &self.name {
            &(x).0.visit(v);
        }
        &self.typ.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        if let Some(x) = &mut self.name {
            &mut (x).0.visit_mut(v);
        }
        &mut self.typ.visit_mut(v);
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
impl Clone for Variadic {
    fn clone(&self) -> Self {
        Variadic {
            attrs: self.attrs.clone(),
            name: self.name.clone(),
            dots: self.dots.clone(),
            comma: self.comma.clone(),
        }
    }
}
impl Debug for Variadic {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Variadic");
        f.field("attrs", &self.attrs);
        f.field("name", &self.name);
        f.field("dots", &self.dots);
        f.field("comma", &self.comma);
        f.finish()
    }
}
impl Eq for Variadic {}
impl PartialEq for Variadic {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.name == x.name && self.comma == x.comma
    }
}
impl Pretty for Variadic {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        if let Some((x, _)) = &self.name {
            x.pretty(p);
            p.word(": ");
        }
        p.word("...");
    }
}
impl<F: Folder + ?Sized> Fold for Variadic {
    fn fold(&self, f: &mut F) {
        Variadic {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            name: (self.name).map(|x| ((x).0.fold(f), (x).1)),
            dots: self.dots,
            comma: self.comma,
        }
    }
}
impl<H: Hasher> Hash for Variadic {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.name.hash(h);
        self.comma.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Variadic {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        if let Some(x) = &self.name {
            &(x).0.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        if let Some(x) = &mut self.name {
            &mut (x).0.visit_mut(v);
        }
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
    pub fn parse(s: Stream, plus: bool) -> Res<Self> {
        if s.peek(Token![->]) {
            let arrow = s.parse()?;
            let gen = true;
            let y = parse_ambig_typ(s, plus, gen)?;
            Ok(Ret::Type(arrow, Box::new(y)))
        } else {
            Ok(Ret::Default)
        }
    }
}
impl Parse for Ret {
    fn parse(s: Stream) -> Res<Self> {
        let plus = true;
        Self::parse(s, plus)
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
impl Clone for Ret {
    fn clone(&self) -> Self {
        use Ret::*;
        match self {
            Default => Default,
            Type(x, v1) => Type(x.clone(), v1.clone()),
        }
    }
}
impl Debug for Ret {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("typ::Ret::")?;
        use Ret::*;
        match self {
            Default => f.write_str("Default"),
            Type(x, v1) => {
                let mut f = f.debug_tuple("Type");
                f.field(x);
                f.field(v1);
                f.finish()
            },
        }
    }
}
impl Eq for Ret {}
impl PartialEq for Ret {
    fn eq(&self, x: &Self) -> bool {
        use Ret::*;
        match (self, x) {
            (Default, Default) => true,
            (Type(_, x), Type(_, y)) => x == y,
            _ => false,
        }
    }
}
impl Pretty for Ret {
    fn pretty(&self, p: &mut Print) {
        use Ret::*;
        match self {
            Default => {},
            Type(_, x) => {
                p.word(" -> ");
                x.pretty(p);
            },
        }
    }
}
impl<F: Folder + ?Sized> Fold for Ret {
    fn fold(&self, f: &mut F) {
        use Ret::*;
        match self {
            Default => Default,
            Type(x, y) => Type(x, Box::new(*y.fold(f))),
        }
    }
}
impl<H: Hasher> Hash for Ret {
    fn hash(&self, h: &mut H) {
        use Ret::*;
        match self {
            Default => {
                h.write_u8(0u8);
            },
            Type(_, v1) => {
                h.write_u8(1u8);
                v1.hash(h);
            },
        }
    }
}
impl<V: Visitor + ?Sized> Visit for Ret {
    fn visit(&self, v: &mut V) {
        use Ret::*;
        match self {
            Default => {},
            Type(_, x) => {
                &**x.visit(v);
            },
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        use Ret::*;
        match self {
            Default => {},
            Type(_, x) => {
                &mut **x.visit_mut(v);
            },
        }
    }
}

pub struct Verbatim(pub pm2::Stream);
impl Pretty for Verbatim {
    fn pretty(&self, p: &mut Print) {
        enum Type {
            Ellipsis,
            DynStar(DynStar),
            MutSelf(MutSelf),
            NotType(NotType),
        }
        use Type::*;
        struct DynStar {
            bounds: punct::Puncted<gen::bound::Type, Token![+]>,
        }
        struct MutSelf {
            typ: Option<Type>,
        }
        struct NotType {
            inner: Type,
        }
        impl parse::Parse for Type {
            fn parse(s: parse::Stream) -> Res<Self> {
                let look = s.lookahead1();
                if look.peek(Token![dyn]) {
                    s.parse::<Token![dyn]>()?;
                    s.parse::<Token![*]>()?;
                    let bounds = s.parse_terminated(gen::bound::Type::parse, Token![+])?;
                    Ok(DynStar(DynStar { bounds }))
                } else if look.peek(Token![mut]) {
                    s.parse::<Token![mut]>()?;
                    s.parse::<Token![self]>()?;
                    let typ = if s.is_empty() {
                        None
                    } else {
                        s.parse::<Token![:]>()?;
                        let y: Type = s.parse()?;
                        Some(y)
                    };
                    Ok(MutSelf(MutSelf { typ }))
                } else if look.peek(Token![!]) {
                    s.parse::<Token![!]>()?;
                    let inner: Type = s.parse()?;
                    Ok(NotType(NotType { inner }))
                } else if look.peek(Token![...]) {
                    s.parse::<Token![...]>()?;
                    Ok(Ellipsis)
                } else {
                    Err(look.error())
                }
            }
        }
        let y: Type = match parse2(self.clone()) {
            Ok(x) => x,
            Err(_) => unimplemented!("Type::Stream`{}`", self),
        };
        match y {
            Ellipsis => {
                p.word("...");
            },
            DynStar(x) => {
                p.word("dyn* ");
                for x in x.bounds.iter().delimited() {
                    if !x.is_first {
                        p.word(" + ");
                    }
                    &x.pretty(p);
                }
            },
            MutSelf(x) => {
                p.word("mut self");
                if let Some(x) = &x.typ {
                    p.word(": ");
                    x.pretty(p);
                }
            },
            NotType(x) => {
                p.word("!");
                &x.inner.pretty(p);
            },
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
                    *args = path::Args::Angle(s.parse()?);
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
    if look.peek(tok::Parenth) {
        let y;
        let parenth = parenthed!(y in s);
        if y.is_empty() {
            return Ok(Type::Tuple(Tuple {
                parenth,
                elems: Puncted::new(),
            }));
        }
        if y.peek(Life) {
            return Ok(Type::Parenth(Parenth {
                parenth,
                elem: Box::new(Type::TraitObject(y.parse()?)),
            }));
        }
        if y.peek(Token![?]) {
            return Ok(Type::TraitObject(Trait {
                dyn_: None,
                bounds: {
                    let mut ys = Puncted::new();
                    ys.push_value(gen::bound::Type::Trait(gen::bound::Trait {
                        parenth: Some(parenth),
                        ..y.parse()?
                    }));
                    while let Some(x) = s.parse()? {
                        ys.push_punct(x);
                        ys.push_value(s.parse()?);
                    }
                    ys
                },
            }));
        }
        let mut first: Type = y.parse()?;
        if y.peek(Token![,]) {
            return Ok(Type::Tuple(Tuple {
                parenth,
                elems: {
                    let mut ys = Puncted::new();
                    ys.push_value(first);
                    ys.push_punct(y.parse()?);
                    while !y.is_empty() {
                        ys.push_value(y.parse()?);
                        if y.is_empty() {
                            break;
                        }
                        ys.push_punct(y.parse()?);
                    }
                    ys
                },
            }));
        }
        if plus && s.peek(Token![+]) {
            loop {
                let first = match first {
                    Type::Path(Path { qself: None, path }) => gen::bound::Type::Trait(gen::bound::Trait {
                        parenth: Some(parenth),
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
                                parenth: Some(parenth),
                                ..trait_bound
                            }),
                            other @ (gen::bound::Type::Life(_) | gen::bound::Type::Verbatim(_)) => other,
                        }
                    },
                    _ => break,
                };
                return Ok(Type::TraitObject(Trait {
                    dyn_: None,
                    bounds: {
                        let mut ys = Puncted::new();
                        ys.push_value(first);
                        while let Some(x) = s.parse()? {
                            ys.push_punct(x);
                            ys.push_value(s.parse()?);
                        }
                        ys
                    },
                }));
            }
        }
        Ok(Type::Parenth(Parenth {
            parenth,
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
            let (delim, toks) = tok::parse_delim(s)?;
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
                parenth: None,
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
                        || s.peek(tok::Parenth))
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
        let span = dyn_.span;
        let star: Option<Token![*]> = s.parse()?;
        let bounds = Trait::parse_bounds(span, s, plus)?;
        return Ok(if star.is_some() {
            Type::Stream(parse::parse_verbatim(&beg, s))
        } else {
            Type::TraitObject(Trait {
                dyn_: Some(dyn_),
                bounds,
            })
        });
    } else if look.peek(tok::Bracket) {
        let y;
        let bracket = bracketed!(y in s);
        let elem: Type = y.parse()?;
        if y.peek(Token![;]) {
            Ok(Type::Array(Array {
                bracket,
                elem: Box::new(elem),
                semi: y.parse()?,
                len: y.parse()?,
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
