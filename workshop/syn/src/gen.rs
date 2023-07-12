pub mod bound {
    pub use pm2::Stream;
    ast_enum_of_structs! {
        pub enum Type {
            Trait(Trait),
            Lifetime(Lifetime),
            Verbatim(Stream),
        }
    }
    impl Type {
        pub fn parse_multiple(x: Stream, plus: bool) -> Res<Punctuated<Self, Token![+]>> {
            let mut ys = Punctuated::new();
            loop {
                ys.push_value(x.parse()?);
                if !(plus && x.peek(Token![+])) {
                    break;
                }
                ys.push_punct(x.parse()?);
                if !(x.peek(Ident::peek_any)
                    || x.peek(Token![::])
                    || x.peek(Token![?])
                    || x.peek(Lifetime)
                    || x.peek(tok::Paren)
                    || x.peek(Token![~]))
                {
                    break;
                }
            }
            Ok(ys)
        }
    }
    impl Parse for Type {
        fn parse(x: Stream) -> Res<Self> {
            if x.peek(Lifetime) {
                return x.parse().map(Type::Lifetime);
            }
            let beg = x.fork();
            let y;
            let (paren, y) = if x.peek(tok::Paren) {
                (Some(parenthesized!(y in x)), &y)
            } else {
                (None, x)
            };
            let is_tilde_const = y.peek(Token![~]) && y.peek2(Token![const]);
            if is_tilde_const {
                y.parse::<Token![~]>()?;
                y.parse::<Token![const]>()?;
            }
            let mut bound: Trait = y.parse()?;
            bound.paren = paren;
            if is_tilde_const {
                Ok(Type::Verbatim(verbatim_between(&beg, x)))
            } else {
                Ok(Type::Trait(bound))
            }
        }
    }

    pub struct Trait {
        pub paren: Option<tok::Paren>,
        pub modif: Modifier,
        pub lifes: Option<Lifes>,
        pub path: path::Path,
    }
    impl Parse for Trait {
        fn parse(x: Stream) -> Res<Self> {
            let modif: Modifier = x.parse()?;
            let lifes: Option<Lifes> = x.parse()?;
            let mut path: Path = x.parse()?;
            if path.segs.last().unwrap().args.is_empty()
                && (x.peek(tok::Paren) || x.peek(Token![::]) && x.peek3(tok::Paren))
            {
                x.parse::<Option<Token![::]>>()?;
                let args: ParenthesizedArgs = x.parse()?;
                let parenthesized = Args::Parenthesized(args);
                path.segs.last_mut().unwrap().args = parenthesized;
            }
            Ok(Trait {
                paren: None,
                modif,
                lifes,
                path,
            })
        }
    }
    impl ToTokens for Trait {
        fn to_tokens(&self, ys: &mut Stream) {
            let f = |ys: &mut Stream| {
                self.modif.to_tokens(ys);
                self.lifes.to_tokens(ys);
                self.path.to_tokens(ys);
            };
            match &self.paren {
                Some(x) => x.surround(ys, f),
                None => f(ys),
            }
        }
    }

    pub enum Modifier {
        None,
        Maybe(Token![?]),
    }
    impl Parse for Modifier {
        fn parse(x: Stream) -> Res<Self> {
            if x.peek(Token![?]) {
                x.parse().map(Modifier::Maybe)
            } else {
                Ok(Modifier::None)
            }
        }
    }
    impl ToTokens for Modifier {
        fn to_tokens(&self, ys: &mut Stream) {
            use Modifier::*;
            match self {
                None => {},
                Maybe(y) => y.to_tokens(ys),
            }
        }
    }

    pub struct Lifes {
        pub for_: Token![for],
        pub lt: Token![<],
        pub lifes: Punctuated<Param, Token![,]>,
        pub gt: Token![>],
    }
    impl Default for Lifes {
        fn default() -> Self {
            Lifes {
                for_: Default::default(),
                lt: Default::default(),
                lifes: Punctuated::new(),
                gt: Default::default(),
            }
        }
    }
    impl Parse for Lifes {
        fn parse(x: Stream) -> Res<Self> {
            Ok(Lifes {
                for_: x.parse()?,
                lt: x.parse()?,
                lifes: {
                    let mut ys = Punctuated::new();
                    while !x.peek(Token![>]) {
                        let attrs = x.call(attr::Attr::parse_outer)?;
                        let life: Lifetime = x.parse()?;
                        ys.push_value(param::Param::Life(param::Life {
                            attrs,
                            life,
                            colon: None,
                            bounds: Punctuated::new(),
                        }));
                        if x.peek(Token![>]) {
                            break;
                        }
                        ys.push_punct(x.parse()?);
                    }
                    ys
                },
                gt: x.parse()?,
            })
        }
    }
    impl Parse for Option<Lifes> {
        fn parse(x: Stream) -> Res<Self> {
            if x.peek(Token![for]) {
                x.parse().map(Some)
            } else {
                Ok(None)
            }
        }
    }
    impl ToTokens for Lifes {
        fn to_tokens(&self, ys: &mut Stream) {
            self.for_.to_tokens(ys);
            self.lt.to_tokens(ys);
            self.lifes.to_tokens(ys);
            self.gt.to_tokens(ys);
        }
    }
}
pub mod param {
    ast_enum_of_structs! {
        pub enum Param {
            Life(Life),
            Type(Type),
            Const(Const),
        }
    }
    impl Parse for Param {
        fn parse(x: Stream) -> Res<Self> {
            let attrs = x.call(attr::Attr::parse_outer)?;
            let look = x.lookahead1();
            if look.peek(Ident) {
                Ok(Param::Type(Type { attrs, ..x.parse()? }))
            } else if look.peek(Lifetime) {
                Ok(Param::Life(Life { attrs, ..x.parse()? }))
            } else if look.peek(Token![const]) {
                Ok(Param::Const(Const { attrs, ..x.parse()? }))
            } else {
                Err(look.error())
            }
        }
    }

    pub struct Life {
        pub attrs: Vec<attr::Attr>,
        pub life: Lifetime,
        pub colon: Option<Token![:]>,
        pub bounds: Punctuated<Lifetime, Token![+]>,
    }
    impl Life {
        pub fn new(life: Lifetime) -> Self {
            Life {
                attrs: Vec::new(),
                life,
                colon: None,
                bounds: Punctuated::new(),
            }
        }
    }
    impl Parse for Life {
        fn parse(x: Stream) -> Res<Self> {
            let colon;
            Ok(Life {
                attrs: x.call(attr::Attr::parse_outer)?,
                life: x.parse()?,
                colon: {
                    if x.peek(Token![:]) {
                        colon = true;
                        Some(x.parse()?)
                    } else {
                        colon = false;
                        None
                    }
                },
                bounds: {
                    let mut ys = Punctuated::new();
                    if colon {
                        loop {
                            if x.peek(Token![,]) || x.peek(Token![>]) {
                                break;
                            }
                            let value = x.parse()?;
                            ys.push_value(value);
                            if !x.peek(Token![+]) {
                                break;
                            }
                            let punct = x.parse()?;
                            ys.push_punct(punct);
                        }
                    }
                    ys
                },
            })
        }
    }
    impl ToTokens for Life {
        fn to_tokens(&self, ys: &mut Stream) {
            ys.append_all(self.attrs.outer());
            self.life.to_tokens(ys);
            if !self.bounds.is_empty() {
                TokensOrDefault(&self.colon).to_tokens(ys);
                self.bounds.to_tokens(ys);
            }
        }
    }

    pub struct Lifes<'a>(Iter<'a, Param>);
    impl<'a> Iterator for Lifes<'a> {
        type Item = &'a Life;
        fn next(&mut self) -> Option<Self::Item> {
            let y = match self.0.next() {
                Some(x) => x,
                None => return None,
            };
            if let Param::Life(x) = y {
                Some(&x)
            } else {
                self.next()
            }
        }
    }
    pub struct LifesMut<'a>(IterMut<'a, Param>);
    impl<'a> Iterator for LifesMut<'a> {
        type Item = &'a mut Life;
        fn next(&mut self) -> Option<Self::Item> {
            let y = match self.0.next() {
                Some(x) => x,
                None => return None,
            };
            if let Param::Life(x) = y {
                Some(&mut x)
            } else {
                self.next()
            }
        }
    }

    pub struct Type {
        pub attrs: Vec<attr::Attr>,
        pub ident: Ident,
        pub colon: Option<Token![:]>,
        pub bounds: Punctuated<bound::Type, Token![+]>,
        pub eq: Option<Token![=]>,
        pub default: Option<typ::Type>,
    }
    impl From<Ident> for Type {
        fn from(ident: Ident) -> Self {
            Type {
                attrs: vec![],
                ident,
                colon: None,
                bounds: Punctuated::new(),
                eq: None,
                default: None,
            }
        }
    }
    impl Parse for Type {
        fn parse(x: Stream) -> Res<Self> {
            let attrs = x.call(attr::Attr::parse_outer)?;
            let ident: Ident = x.parse()?;
            let colon: Option<Token![:]> = x.parse()?;
            let mut ys = Punctuated::new();
            if colon.is_some() {
                loop {
                    if x.peek(Token![,]) || x.peek(Token![>]) || x.peek(Token![=]) {
                        break;
                    }
                    let y: bound::Type = x.parse()?;
                    ys.push_value(y);
                    if !x.peek(Token![+]) {
                        break;
                    }
                    let y: Token![+] = x.parse()?;
                    ys.push_punct(y);
                }
            }
            let eq: Option<Token![=]> = x.parse()?;
            let default = if eq.is_some() {
                Some(x.parse::<typ::Type>()?)
            } else {
                None
            };
            Ok(Type {
                attrs,
                ident,
                colon,
                bounds: ys,
                eq,
                default,
            })
        }
    }
    impl ToTokens for Type {
        fn to_tokens(&self, ys: &mut Stream) {
            ys.append_all(self.attrs.outer());
            self.ident.to_tokens(ys);
            if !self.bounds.is_empty() {
                TokensOrDefault(&self.colon).to_tokens(ys);
                self.bounds.to_tokens(ys);
            }
            if let Some(y) = &self.default {
                TokensOrDefault(&self.eq).to_tokens(ys);
                y.to_tokens(ys);
            }
        }
    }

    pub struct Types<'a>(Iter<'a, Param>);
    impl<'a> Iterator for Types<'a> {
        type Item = &'a Type;
        fn next(&mut self) -> Option<Self::Item> {
            let y = match self.0.next() {
                Some(x) => x,
                None => return None,
            };
            if let Param::Type(x) = y {
                Some(&x)
            } else {
                self.next()
            }
        }
    }
    pub struct TypesMut<'a>(IterMut<'a, Param>);
    impl<'a> Iterator for TypesMut<'a> {
        type Item = &'a mut Type;
        fn next(&mut self) -> Option<Self::Item> {
            let y = match self.0.next() {
                Some(x) => x,
                None => return None,
            };
            if let Param::Type(x) = y {
                Some(&mut x)
            } else {
                self.next()
            }
        }
    }
    pub struct Const {
        pub attrs: Vec<attr::Attr>,
        pub const_: Token![const],
        pub ident: Ident,
        pub colon: Token![:],
        pub typ: typ::Type,
        pub eq: Option<Token![=]>,
        pub default: Option<Expr>,
    }
    impl Parse for Const {
        fn parse(x: Stream) -> Res<Self> {
            let mut default = None;
            Ok(Const {
                attrs: x.call(attr::Attr::parse_outer)?,
                const_: x.parse()?,
                ident: x.parse()?,
                colon: x.parse()?,
                typ: x.parse()?,
                eq: {
                    if x.peek(Token![=]) {
                        let eq = x.parse()?;
                        default = Some(const_argument(x)?);
                        Some(eq)
                    } else {
                        None
                    }
                },
                default,
            })
        }
    }
    impl ToTokens for Const {
        fn to_tokens(&self, ys: &mut Stream) {
            ys.append_all(self.attrs.outer());
            self.const_.to_tokens(ys);
            self.ident.to_tokens(ys);
            self.colon.to_tokens(ys);
            self.typ.to_tokens(ys);
            if let Some(y) = &self.default {
                TokensOrDefault(&self.eq).to_tokens(ys);
                y.to_tokens(ys);
            }
        }
    }

    pub struct Consts<'a>(Iter<'a, Param>);
    impl<'a> Iterator for Consts<'a> {
        type Item = &'a Const;
        fn next(&mut self) -> Option<Self::Item> {
            let y = match self.0.next() {
                Some(x) => x,
                None => return None,
            };
            if let Param::Const(x) = y {
                Some(&x)
            } else {
                self.next()
            }
        }
    }
    pub struct ConstsMut<'a>(IterMut<'a, Param>);
    impl<'a> Iterator for ConstsMut<'a> {
        type Item = &'a mut Const;
        fn next(&mut self) -> Option<Self::Item> {
            let y = match self.0.next() {
                Some(x) => x,
                None => return None,
            };
            if let Param::Const(x) = y {
                Some(&mut x)
            } else {
                self.next()
            }
        }
    }
}
use param::Param;
pub struct Where {
    pub where_: Token![where],
    pub preds: Punctuated<Where::Pred, Token![,]>,
}
impl Parse for Where {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Where {
            where_: x.parse()?,
            preds: {
                let mut ys = Punctuated::new();
                loop {
                    if x.is_empty()
                        || x.peek(tok::Brace)
                        || x.peek(Token![,])
                        || x.peek(Token![;])
                        || x.peek(Token![:]) && !x.peek(Token![::])
                        || x.peek(Token![=])
                    {
                        break;
                    }
                    let y = x.parse()?;
                    ys.push_value(y);
                    if !x.peek(Token![,]) {
                        break;
                    }
                    let y = x.parse()?;
                    ys.push_punct(y);
                }
                ys
            },
        })
    }
}
impl Parse for Option<Where> {
    fn parse(x: Stream) -> Res<Self> {
        if x.peek(Token![where]) {
            x.parse().map(Some)
        } else {
            Ok(None)
        }
    }
}
impl ToTokens for Where {
    fn to_tokens(&self, ys: &mut Stream) {
        if !self.preds.is_empty() {
            self.where_.to_tokens(ys);
            self.preds.to_tokens(ys);
        }
    }
}

pub mod Where {
    ast_enum_of_structs! {
        pub enum Pred {
            Life(Life),
            Type(Type),
        }
    }
    impl Parse for Pred {
        fn parse(x: Stream) -> Res<Self> {
            if x.peek(Lifetime) && x.peek2(Token![:]) {
                Ok(Pred::Life(Life {
                    life: x.parse()?,
                    colon: x.parse()?,
                    bounds: {
                        let mut ys = Punctuated::new();
                        loop {
                            if x.is_empty()
                                || x.peek(tok::Brace)
                                || x.peek(Token![,])
                                || x.peek(Token![;])
                                || x.peek(Token![:])
                                || x.peek(Token![=])
                            {
                                break;
                            }
                            let y = x.parse()?;
                            ys.push_value(y);
                            if !x.peek(Token![+]) {
                                break;
                            }
                            let y = x.parse()?;
                            ys.push_punct(y);
                        }
                        ys
                    },
                }))
            } else {
                Ok(Pred::Type(Type {
                    lifes: x.parse()?,
                    bounded: x.parse()?,
                    colon: x.parse()?,
                    bounds: {
                        let mut ys = Punctuated::new();
                        loop {
                            if x.is_empty()
                                || x.peek(tok::Brace)
                                || x.peek(Token![,])
                                || x.peek(Token![;])
                                || x.peek(Token![:]) && !x.peek(Token![::])
                                || x.peek(Token![=])
                            {
                                break;
                            }
                            let y = x.parse()?;
                            ys.push_value(y);
                            if !x.peek(Token![+]) {
                                break;
                            }
                            let y = x.parse()?;
                            ys.push_punct(y);
                        }
                        ys
                    },
                }))
            }
        }
    }

    pub struct Life {
        pub life: Lifetime,
        pub colon: Token![:],
        pub bounds: Punctuated<Lifetime, Token![+]>,
    }
    impl ToTokens for Life {
        fn to_tokens(&self, ys: &mut Stream) {
            self.life.to_tokens(ys);
            self.colon.to_tokens(ys);
            self.bounds.to_tokens(ys);
        }
    }

    pub struct Type {
        pub lifes: Option<bound::Lifes>,
        pub bounded: typ::Type,
        pub colon: Token![:],
        pub bounds: Punctuated<bound::Type, Token![+]>,
    }
    impl ToTokens for Type {
        fn to_tokens(&self, ys: &mut Stream) {
            self.lifes.to_tokens(ys);
            self.bounded.to_tokens(ys);
            self.colon.to_tokens(ys);
            self.bounds.to_tokens(ys);
        }
    }
}

pub struct Gens {
    pub lt: Option<Token![<]>,
    pub ps: Punctuated<Param, Token![,]>,
    pub gt: Option<Token![>]>,
    pub where_: Option<Where>,
}
impl Gens {
    pub fn life_ps(&self) -> param::Lifes {
        param::Lifes(self.ps.iter())
    }
    pub fn life_ps_mut(&mut self) -> param::LifesMut {
        param::LifesMut(self.ps.iter_mut())
    }
    pub fn type_ps(&self) -> param::Types {
        param::Types(self.ps.iter())
    }
    pub fn type_ps_mut(&mut self) -> param::TypesMut {
        param::TypesMut(self.ps.iter_mut())
    }
    pub fn const_ps(&self) -> param::Consts {
        param::Consts(self.ps.iter())
    }
    pub fn const_ps_mut(&mut self) -> param::ConstsMut {
        param::ConstsMut(self.ps.iter_mut())
    }
    pub fn make_where_clause(&mut self) -> &mut Where {
        self.where_.get_or_insert_with(|| Where {
            where_: <Token![where]>::default(),
            preds: Punctuated::new(),
        })
    }
    pub fn split_for_impl(&self) -> (Impl, Type, Option<&Where>) {
        (Impl(self), Type(self), self.where_.as_ref())
    }
}
impl Default for Gens {
    fn default() -> Self {
        Gens {
            lt: None,
            ps: Punctuated::new(),
            gt: None,
            where_: None,
        }
    }
}
impl Parse for Gens {
    fn parse(x: Stream) -> Res<Self> {
        if !x.peek(Token![<]) {
            return Ok(Gens::default());
        }
        let lt: Token![<] = x.parse()?;
        let mut ys = Punctuated::new();
        loop {
            if x.peek(Token![>]) {
                break;
            }
            let attrs = x.call(attr::Attr::parse_outer)?;
            let look = x.lookahead1();
            if look.peek(Lifetime) {
                ys.push_value(Param::Life(param::Life { attrs, ..x.parse()? }));
            } else if look.peek(Ident) {
                ys.push_value(Param::Type(param::Type { attrs, ..x.parse()? }));
            } else if look.peek(Token![const]) {
                ys.push_value(Param::Const(param::Const { attrs, ..x.parse()? }));
            } else if x.peek(Token![_]) {
                ys.push_value(Param::Type(param::Type {
                    attrs,
                    ident: x.call(Ident::parse_any)?,
                    colon: None,
                    bounds: Punctuated::new(),
                    eq: None,
                    default: None,
                }));
            } else {
                return Err(look.error());
            }
            if x.peek(Token![>]) {
                break;
            }
            let y = x.parse()?;
            ys.push_punct(y);
        }
        let gt: Token![>] = x.parse()?;
        Ok(Gens {
            lt: Some(lt),
            ps: ys,
            gt: Some(gt),
            where_: None,
        })
    }
}
impl ToTokens for Gens {
    fn to_tokens(&self, ys: &mut Stream) {
        if self.ps.is_empty() {
            return;
        }
        TokensOrDefault(&self.lt).to_tokens(ys);
        let mut trailing_or_empty = true;
        for x in self.ps.pairs() {
            if let Param::Life(_) = **x.value() {
                x.to_tokens(ys);
                trailing_or_empty = x.punct().is_some();
            }
        }
        for x in self.ps.pairs() {
            match x.value() {
                Param::Type(_) | Param::Const(_) => {
                    if !trailing_or_empty {
                        <Token![,]>::default().to_tokens(ys);
                        trailing_or_empty = true;
                    }
                    x.to_tokens(ys);
                },
                Param::Life(_) => {},
            }
        }
        TokensOrDefault(&self.gt).to_tokens(ys);
    }
}

pub struct Impl<'a>(pub &'a Gens);
macro_rules! gens_impls {
    ($ty:ident) => {
        impl<'a> Clone for $ty<'a> {
            fn clone(&self) -> Self {
                $ty(self.0)
            }
        }
        impl<'a> Debug for $ty<'a> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.debug_tuple(stringify!($ty)).field(self.0).finish()
            }
        }
        impl<'a> Eq for $ty<'a> {}
        impl<'a> PartialEq for $ty<'a> {
            fn eq(&self, x: &Self) -> bool {
                self.0 == x.0
            }
        }
        impl<'a> Hash for $ty<'a> {
            fn hash<H: Hasher>(&self, x: &mut H) {
                self.0.hash(x);
            }
        }
    };
}
gens_impls!(Impl);
impl<'a> ToTokens for Impl<'a> {
    fn to_tokens(&self, ys: &mut Stream) {
        if self.0.ps.is_empty() {
            return;
        }
        TokensOrDefault(&self.0.lt).to_tokens(ys);
        let mut trailing_or_empty = true;
        for x in self.0.ps.pairs() {
            if let Param::Life(_) = **x.value() {
                x.to_tokens(ys);
                trailing_or_empty = x.punct().is_some();
            }
        }
        for x in self.0.ps.pairs() {
            if let Param::Life(_) = **x.value() {
                continue;
            }
            if !trailing_or_empty {
                <Token![,]>::default().to_tokens(ys);
                trailing_or_empty = true;
            }
            match x.value() {
                Param::Life(_) => unreachable!(),
                Param::Type(x) => {
                    ys.append_all(x.attrs.outer());
                    x.ident.to_tokens(ys);
                    if !x.bounds.is_empty() {
                        TokensOrDefault(&x.colon).to_tokens(ys);
                        x.bounds.to_tokens(ys);
                    }
                },
                Param::Const(x) => {
                    ys.append_all(x.attrs.outer());
                    x.const_.to_tokens(ys);
                    x.ident.to_tokens(ys);
                    x.colon.to_tokens(ys);
                    x.typ.to_tokens(ys);
                },
            }
            x.punct().to_tokens(ys);
        }
        TokensOrDefault(&self.0.gt).to_tokens(ys);
    }
}

pub struct Type<'a>(pub &'a Gens);
gens_impls!(Type);
impl<'a> ToTokens for Type<'a> {
    fn to_tokens(&self, ys: &mut Stream) {
        if self.0.ps.is_empty() {
            return;
        }
        TokensOrDefault(&self.0.lt).to_tokens(ys);
        let mut trailing_or_empty = true;
        for x in self.0.ps.pairs() {
            if let Param::Life(y) = *x.value() {
                y.life.to_tokens(ys);
                x.punct().to_tokens(ys);
                trailing_or_empty = x.punct().is_some();
            }
        }
        for x in self.0.ps.pairs() {
            if let Param::Life(_) = **x.value() {
                continue;
            }
            if !trailing_or_empty {
                <Token![,]>::default().to_tokens(ys);
                trailing_or_empty = true;
            }
            match x.value() {
                Param::Life(_) => unreachable!(),
                Param::Type(x) => {
                    x.ident.to_tokens(ys);
                },
                Param::Const(x) => {
                    x.ident.to_tokens(ys);
                },
            }
            x.punct().to_tokens(ys);
        }
        TokensOrDefault(&self.0.gt).to_tokens(ys);
    }
}
impl<'a> Type<'a> {
    pub fn as_turbofish(&self) -> Turbofish {
        Turbofish(self.0)
    }
}

pub struct Turbofish<'a>(pub &'a Gens);
gens_impls!(Turbofish);
impl<'a> ToTokens for Turbofish<'a> {
    fn to_tokens(&self, ys: &mut Stream) {
        if !self.0.ps.is_empty() {
            <Token![::]>::default().to_tokens(ys);
            Type(self.0).to_tokens(ys);
        }
    }
}
