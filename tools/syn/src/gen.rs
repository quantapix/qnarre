use super::{
    punct::{Iter, IterMut},
    *,
};

pub mod bound {
    use super::*;
    enum_of_structs! {
        pub enum Type {
            Life(Life),
            Trait(Trait),
            Verbatim(Verbatim),
        }
    }
    impl Type {
        pub fn parse_many(s: Stream, plus: bool) -> Res<Puncted<Self, Token![+]>> {
            let mut ys = Puncted::new();
            loop {
                ys.push_value(s.parse()?);
                if !(plus && s.peek(Token![+])) {
                    break;
                }
                ys.push_punct(s.parse()?);
                if !(s.peek(Ident::peek_any)
                    || s.peek(Token![::])
                    || s.peek(Token![?])
                    || s.peek(Life)
                    || s.peek(tok::Parenth)
                    || s.peek(Token![~]))
                {
                    break;
                }
            }
            Ok(ys)
        }
    }
    impl Parse for Type {
        fn parse(s: Stream) -> Res<Self> {
            if s.peek(Life) {
                return s.parse().map(Type::Life);
            }
            let beg = s.fork();
            let y;
            let (parenth, y) = if s.peek(tok::Parenth) {
                (Some(parenthed!(y in s)), &y)
            } else {
                (None, s)
            };
            let is_tilde_const = y.peek(Token![~]) && y.peek2(Token![const]);
            if is_tilde_const {
                y.parse::<Token![~]>()?;
                y.parse::<Token![const]>()?;
            }
            let mut y: Trait = y.parse()?;
            y.parenth = parenth;
            if is_tilde_const {
                Ok(Type::Verbatim(parse::parse_verbatim(&beg, s)))
            } else {
                Ok(Type::Trait(y))
            }
        }
    }
    impl Pretty for Type {
        fn pretty(&self, p: &mut Print) {
            match self {
                Type::Trait(x) => {
                    let tilde = false;
                    x.pretty_with_args(p, tilde);
                },
                Type::Life(x) => x.pretty(p),
                Type::Verbatim(x) => x.pretty(p),
            }
        }
    }

    pub struct Trait {
        pub parenth: Option<tok::Parenth>,
        pub modif: Modifier,
        pub lifes: Option<Lifes>,
        pub path: Path,
    }
    impl Parse for Trait {
        fn parse(s: Stream) -> Res<Self> {
            let modif: Modifier = s.parse()?;
            let lifes: Option<Lifes> = s.parse()?;
            let mut path: Path = s.parse()?;
            if path.segs.last().unwrap().args.is_empty()
                && (s.peek(tok::Parenth) || s.peek(Token![::]) && s.peek3(tok::Parenth))
            {
                s.parse::<Option<Token![::]>>()?;
                let y: path::Parenthed = s.parse()?;
                let y = path::Args::Parenthed(y);
                path.segs.last_mut().unwrap().args = y;
            }
            Ok(Trait {
                parenth: None,
                modif,
                lifes,
                path,
            })
        }
    }
    impl Lower for Trait {
        fn lower(&self, s: &mut Stream) {
            let y = |s: &mut Stream| {
                self.modif.lower(s);
                self.lifes.lower(s);
                self.path.lower(s);
            };
            match &self.parenth {
                Some(x) => x.surround(s, y),
                None => y(s),
            }
        }
    }
    impl Pretty for Trait {
        fn pretty(&self, p: &mut Print, tilde: bool) {
            if self.parenth.is_some() {
                p.word("(");
            }
            if tilde {
                p.word("~const ");
            }
            &self.modif.pretty(p);
            if let Some(x) = &self.lifes {
                x.pretty(p);
            }
            for x in self.path.segs.iter().delimited() {
                if !x.is_first || self.path.colon.is_some() {
                    p.word("::");
                }
                &x.pretty_with_args(p, path::Kind::Type);
            }
            if self.parenth.is_some() {
                p.word(")");
            }
        }
    }

    pub enum Modifier {
        Maybe(Token![?]),
        None,
    }
    impl Parse for Modifier {
        fn parse(s: Stream) -> Res<Self> {
            if s.peek(Token![?]) {
                s.parse().map(Modifier::Maybe)
            } else {
                Ok(Modifier::None)
            }
        }
    }
    impl Lower for Modifier {
        fn lower(&self, s: &mut Stream) {
            use Modifier::*;
            match self {
                Maybe(x) => x.lower(s),
                None => {},
            }
        }
    }
    impl Pretty for Modifier {
        fn pretty(&self, p: &mut Print) {
            use Modifier::*;
            match self {
                Maybe(_) => p.word("?"),
                None => {},
            }
        }
    }

    pub struct Lifes {
        pub for_: Token![for],
        pub lt: Token![<],
        pub lifes: Puncted<Param, Token![,]>,
        pub gt: Token![>],
    }
    impl Default for Lifes {
        fn default() -> Self {
            Lifes {
                for_: Default::default(),
                lt: Default::default(),
                lifes: Puncted::new(),
                gt: Default::default(),
            }
        }
    }
    impl Parse for Lifes {
        fn parse(s: Stream) -> Res<Self> {
            Ok(Lifes {
                for_: s.parse()?,
                lt: s.parse()?,
                lifes: {
                    let mut ys = Puncted::new();
                    while !s.peek(Token![>]) {
                        let attrs = s.call(attr::Attr::parse_outers)?;
                        let life: Life = s.parse()?;
                        ys.push_value(param::Param::Life(param::Life {
                            attrs,
                            life,
                            colon: None,
                            bounds: Puncted::new(),
                        }));
                        if s.peek(Token![>]) {
                            break;
                        }
                        ys.push_punct(s.parse()?);
                    }
                    ys
                },
                gt: s.parse()?,
            })
        }
    }
    impl Parse for Option<Lifes> {
        fn parse(s: Stream) -> Res<Self> {
            if s.peek(Token![for]) {
                s.parse().map(Some)
            } else {
                Ok(None)
            }
        }
    }
    impl Lower for Lifes {
        fn lower(&self, s: &mut Stream) {
            self.for_.lower(s);
            self.lt.lower(s);
            self.lifes.lower(s);
            self.gt.lower(s);
        }
    }
    impl Pretty for Lifes {
        fn pretty(&self, p: &mut Print) {
            p.word("for<");
            for x in self.lifes.iter().delimited() {
                &x.pretty(p);
                if !x.is_last {
                    p.word(", ");
                }
            }
            p.word("> ");
        }
    }

    pub struct Verbatim(pub pm2::Stream);
    impl Pretty for Verbatim {
        fn pretty(&self, p: &mut Print) {
            enum Type {
                Ellipsis,
                TildeConst(Trait),
            }
            impl parse::Parse for Type {
                fn parse(s: parse::Stream) -> Res<Self> {
                    let y;
                    let (parenth, y) = if s.peek(tok::Parenth) {
                        (Some(parenthed!(y in s)), &y)
                    } else {
                        (None, s)
                    };
                    let look = y.lookahead1();
                    if look.peek(Token![~]) {
                        y.parse::<Token![~]>()?;
                        y.parse::<Token![const]>()?;
                        let mut y: Trait = y.parse()?;
                        y.parenth = parenth;
                        Ok(Type::TildeConst(y))
                    } else if look.peek(Token![...]) {
                        y.parse::<Token![...]>()?;
                        Ok(Type::Ellipsis)
                    } else {
                        Err(look.error())
                    }
                }
            }
            let y: Type = match parse2(self.clone()) {
                Ok(x) => x,
                Err(_) => unimplemented!("TypeParamBound::Verbatim `{}`", self),
            };
            match y {
                Type::Ellipsis => {
                    p.word("...");
                },
                Type::TildeConst(x) => {
                    let tilde = true;
                    &x.pretty_with_args(p, tilde);
                },
            }
        }
    }
}
pub mod param {
    use super::*;
    enum_of_structs! {
        pub enum Param {
            Const(Const),
            Life(Life),
            Type(Type),
        }
    }
    impl Parse for Param {
        fn parse(s: Stream) -> Res<Self> {
            let attrs = s.call(attr::Attr::parse_outers)?;
            let look = s.look1();
            if look.peek(ident::Ident) {
                Ok(Param::Type(Type { attrs, ..s.parse()? }))
            } else if look.peek(Life) {
                Ok(Param::Life(Life { attrs, ..s.parse()? }))
            } else if look.peek(Token![const]) {
                Ok(Param::Const(Const { attrs, ..s.parse()? }))
            } else {
                Err(look.error())
            }
        }
    }
    impl Pretty for Param {
        fn pretty(&self, p: &mut Print) {
            use Param::*;
            match self {
                Const(x) => x.pretty(p),
                Life(x) => x.pretty(p),
                Type(x) => x.pretty(p),
            }
        }
    }

    pub struct Life {
        pub attrs: Vec<attr::Attr>,
        pub life: Life,
        pub colon: Option<Token![:]>,
        pub bounds: Puncted<Life, Token![+]>,
    }
    impl Life {
        pub fn new(life: Life) -> Self {
            Life {
                attrs: Vec::new(),
                life,
                colon: None,
                bounds: Puncted::new(),
            }
        }
    }
    impl Parse for Life {
        fn parse(s: Stream) -> Res<Self> {
            let colon;
            Ok(Life {
                attrs: s.call(attr::Attr::parse_outers)?,
                life: s.parse()?,
                colon: {
                    if s.peek(Token![:]) {
                        colon = true;
                        Some(s.parse()?)
                    } else {
                        colon = false;
                        None
                    }
                },
                bounds: {
                    let mut ys = Puncted::new();
                    if colon {
                        loop {
                            if s.peek(Token![,]) || s.peek(Token![>]) {
                                break;
                            }
                            let y = s.parse()?;
                            ys.push_value(y);
                            if !s.peek(Token![+]) {
                                break;
                            }
                            let y = s.parse()?;
                            ys.push_punct(y);
                        }
                    }
                    ys
                },
            })
        }
    }
    impl Lower for Life {
        fn lower(&self, s: &mut Stream) {
            s.append_all(self.attrs.outers());
            self.life.lower(s);
            if !self.bounds.is_empty() {
                ToksOrDefault(&self.colon).lower(s);
                self.bounds.lower(s);
            }
        }
    }
    impl Pretty for Life {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            &self.life.pretty(p);
            for x in self.bounds.iter().delimited() {
                if x.is_first {
                    p.word(": ");
                } else {
                    p.word(" + ");
                }
                &x.pretty(p);
            }
        }
    }

    pub struct Lifes<'a>(pub Iter<'a, Param>);
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
    pub struct LifesMut<'a>(pub IterMut<'a, Param>);
    impl<'a> Iterator for LifesMut<'a> {
        type Item = &'a mut Life;
        fn next(&mut self) -> Option<Self::Item> {
            let y = match self.0.next() {
                Some(x) => x,
                None => return None,
            };
            if let Param::Life(x) = y {
                Some(x)
            } else {
                self.next()
            }
        }
    }

    pub struct Type {
        pub attrs: Vec<attr::Attr>,
        pub ident: Ident,
        pub colon: Option<Token![:]>,
        pub bounds: Puncted<bound::Type, Token![+]>,
        pub eq: Option<Token![=]>,
        pub default: Option<typ::Type>,
    }
    impl From<Ident> for Type {
        fn from(ident: Ident) -> Self {
            Type {
                attrs: vec![],
                ident,
                colon: None,
                bounds: Puncted::new(),
                eq: None,
                default: None,
            }
        }
    }
    impl Parse for Type {
        fn parse(s: Stream) -> Res<Self> {
            let attrs = s.call(attr::Attr::parse_outers)?;
            let ident: Ident = s.parse()?;
            let colon: Option<Token![:]> = s.parse()?;
            let mut ys = Puncted::new();
            if colon.is_some() {
                loop {
                    if s.peek(Token![,]) || s.peek(Token![>]) || s.peek(Token![=]) {
                        break;
                    }
                    let y: bound::Type = s.parse()?;
                    ys.push_value(y);
                    if !s.peek(Token![+]) {
                        break;
                    }
                    let y: Token![+] = s.parse()?;
                    ys.push_punct(y);
                }
            }
            let eq: Option<Token![=]> = s.parse()?;
            let default = if eq.is_some() {
                Some(s.parse::<typ::Type>()?)
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
    impl Lower for Type {
        fn lower(&self, s: &mut Stream) {
            s.append_all(self.attrs.outers());
            self.ident.lower(s);
            if !self.bounds.is_empty() {
                ToksOrDefault(&self.colon).lower(s);
                self.bounds.lower(s);
            }
            if let Some(y) = &self.default {
                ToksOrDefault(&self.eq).lower(s);
                y.lower(s);
            }
        }
    }
    impl Pretty for Type {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            &self.ident.pretty(p);
            p.ibox(INDENT);
            for x in self.bounds.iter().delimited() {
                if x.is_first {
                    p.word(": ");
                } else {
                    p.space();
                    p.word("+ ");
                }
                &x.pretty(p);
            }
            if let Some(x) = &self.default {
                p.space();
                p.word("= ");
                x.pretty(p);
            }
            p.end();
        }
    }

    pub struct Types<'a>(pub Iter<'a, Param>);
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
    pub struct TypesMut<'a>(pub IterMut<'a, Param>);
    impl<'a> Iterator for TypesMut<'a> {
        type Item = &'a mut Type;
        fn next(&mut self) -> Option<Self::Item> {
            let y = match self.0.next() {
                Some(x) => x,
                None => return None,
            };
            if let Param::Type(x) = y {
                Some(x)
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
        pub default: Option<expr::Expr>,
    }
    impl Parse for Const {
        fn parse(s: Stream) -> Res<Self> {
            let mut default = None;
            Ok(Const {
                attrs: s.call(attr::Attr::parse_outers)?,
                const_: s.parse()?,
                ident: s.parse()?,
                colon: s.parse()?,
                typ: s.parse()?,
                eq: {
                    if s.peek(Token![=]) {
                        let y = s.parse()?;
                        default = Some(path::const_arg(s)?);
                        Some(y)
                    } else {
                        None
                    }
                },
                default,
            })
        }
    }
    impl Lower for Const {
        fn lower(&self, s: &mut Stream) {
            s.append_all(self.attrs.outers());
            self.const_.lower(s);
            self.ident.lower(s);
            self.colon.lower(s);
            self.typ.lower(s);
            if let Some(y) = &self.default {
                ToksOrDefault(&self.eq).lower(s);
                y.lower(s);
            }
        }
    }
    impl Pretty for Const {
        fn pretty(&self, p: &mut Print) {
            p.outer_attrs(&self.attrs);
            p.word("const ");
            &self.ident.pretty(p);
            p.word(": ");
            &self.typ.pretty(p);
            if let Some(x) = &self.default {
                p.word(" = ");
                x.pretty(p);
            }
        }
    }

    pub struct Consts<'a>(pub Iter<'a, Param>);
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
    pub struct ConstsMut<'a>(pub IterMut<'a, Param>);
    impl<'a> Iterator for ConstsMut<'a> {
        type Item = &'a mut Const;
        fn next(&mut self) -> Option<Self::Item> {
            let y = match self.0.next() {
                Some(x) => x,
                None => return None,
            };
            if let Param::Const(x) = y {
                Some(x)
            } else {
                self.next()
            }
        }
    }
}
pub use param::Param;

pub struct Where {
    pub where_: Token![where],
    pub preds: Puncted<where_::Pred, Token![,]>,
}
impl Parse for Where {
    fn parse(s: Stream) -> Res<Self> {
        Ok(Where {
            where_: s.parse()?,
            preds: {
                let mut ys = Puncted::new();
                loop {
                    if s.is_empty()
                        || s.peek(tok::Brace)
                        || s.peek(Token![,])
                        || s.peek(Token![;])
                        || s.peek(Token![:]) && !s.peek(Token![::])
                        || s.peek(Token![=])
                    {
                        break;
                    }
                    let y = s.parse()?;
                    ys.push_value(y);
                    if !s.peek(Token![,]) {
                        break;
                    }
                    let y = s.parse()?;
                    ys.push_punct(y);
                }
                ys
            },
        })
    }
}
impl Parse for Option<Where> {
    fn parse(s: Stream) -> Res<Self> {
        if s.peek(Token![where]) {
            s.parse().map(Some)
        } else {
            Ok(None)
        }
    }
}
impl Lower for Where {
    fn lower(&self, s: &mut Stream) {
        if !self.preds.is_empty() {
            self.where_.lower(s);
            self.preds.lower(s);
        }
    }
}
impl Pretty for Where {
    fn pretty(&self, p: &mut Print, breaks: bool, semi: bool) {
        if breaks {
            p.hardbreak();
            p.offset(-INDENT);
            p.word("where");
            p.hardbreak();
            for x in self.preds.iter().delimited() {
                &x.pretty(p);
                if x.is_last && semi {
                    p.word(";");
                } else {
                    p.word(",");
                    p.hardbreak();
                }
            }
            if !semi {
                p.offset(-INDENT);
            }
        } else {
            p.space();
            p.offset(-INDENT);
            p.word("where");
            p.space();
            for x in self.preds.iter().delimited() {
                &x.pretty(p);
                if x.is_last && semi {
                    p.word(";");
                } else {
                    p.trailing_comma_or_space(x.is_last);
                }
            }
            if !semi {
                p.offset(-INDENT);
            }
        }
    }
}
impl Pretty for Option<Where> {
    fn pretty(&self, p: &mut Print, breaks: bool, semi: bool) {
        let y = match self {
            Some(x) if !x.preds.is_empty() => x,
            _ => {
                if semi {
                    p.word(";");
                } else {
                    p.nbsp();
                }
                return;
            },
        };
        y.pretty_with_args(p, (breaks, semi));
    }
}

impl Print {
    pub fn where_for_body(&mut self, x: &Option<Where>) {
        let breaks = true;
        let semi = false;
        x.pretty_with_args(self, (breaks, semi));
    }
    pub fn where_with_semi(&mut self, x: &Option<Where>) {
        let breaks = true;
        let semi = true;
        x.pretty_with_args(self, (breaks, semi));
    }
    pub fn where_oneline(&mut self, x: &Option<Where>) {
        let breaks = false;
        let semi = false;
        x.pretty_with_args(self, (breaks, semi));
    }
    pub fn where_oneline_with_semi(&mut self, x: &Option<Where>) {
        let breaks = false;
        let semi = true;
        x.pretty_with_args(self, (breaks, semi));
    }
}

pub mod where_ {
    use super::*;
    enum_of_structs! {
        pub enum Pred {
            Life(Life),
            Type(Type),
        }
    }
    impl Parse for Pred {
        fn parse(s: Stream) -> Res<Self> {
            if s.peek(Life) && s.peek2(Token![:]) {
                Ok(Pred::Life(Life {
                    life: s.parse()?,
                    colon: s.parse()?,
                    bounds: {
                        let mut ys = Puncted::new();
                        loop {
                            if s.is_empty()
                                || s.peek(tok::Brace)
                                || s.peek(Token![,])
                                || s.peek(Token![;])
                                || s.peek(Token![:])
                                || s.peek(Token![=])
                            {
                                break;
                            }
                            let y = s.parse()?;
                            ys.push_value(y);
                            if !s.peek(Token![+]) {
                                break;
                            }
                            let y = s.parse()?;
                            ys.push_punct(y);
                        }
                        ys
                    },
                }))
            } else {
                Ok(Pred::Type(Type {
                    lifes: s.parse()?,
                    typ: s.parse()?,
                    colon: s.parse()?,
                    bounds: {
                        let mut ys = Puncted::new();
                        loop {
                            if s.is_empty()
                                || s.peek(tok::Brace)
                                || s.peek(Token![,])
                                || s.peek(Token![;])
                                || s.peek(Token![:]) && !s.peek(Token![::])
                                || s.peek(Token![=])
                            {
                                break;
                            }
                            let y = s.parse()?;
                            ys.push_value(y);
                            if !s.peek(Token![+]) {
                                break;
                            }
                            let y = s.parse()?;
                            ys.push_punct(y);
                        }
                        ys
                    },
                }))
            }
        }
    }
    impl Pretty for Pred {
        fn pretty(&self, p: &mut Print) {
            use Pred::*;
            match self {
                Life(x) => x.pretty(p),
                Type(x) => x.pretty(p),
            }
        }
    }

    pub struct Life {
        pub life: Life,
        pub colon: Token![:],
        pub bounds: Puncted<Life, Token![+]>,
    }
    impl Lower for Life {
        fn lower(&self, s: &mut Stream) {
            self.life.lower(s);
            self.colon.lower(s);
            self.bounds.lower(s);
        }
    }
    impl Pretty for Life {
        fn pretty(&self, p: &mut Print) {
            &self.life.pretty(p);
            p.word(":");
            p.ibox(INDENT);
            for x in self.bounds.iter().delimited() {
                if x.is_first {
                    p.nbsp();
                } else {
                    p.space();
                    p.word("+ ");
                }
                &x.pretty(p);
            }
            p.end();
        }
    }

    pub struct Type {
        pub lifes: Option<bound::Lifes>,
        pub typ: typ::Type,
        pub colon: Token![:],
        pub bounds: Puncted<bound::Type, Token![+]>,
    }
    impl Lower for Type {
        fn lower(&self, s: &mut Stream) {
            self.lifes.lower(s);
            self.typ.lower(s);
            self.colon.lower(s);
            self.bounds.lower(s);
        }
    }
    impl Pretty for Type {
        fn pretty(&self, p: &mut Print) {
            if let Some(x) = &self.lifes {
                x.pretty(p);
            }
            &self.typ.pretty(p);
            p.word(":");
            if self.bounds.len() == 1 {
                p.ibox(0);
            } else {
                p.ibox(INDENT);
            }
            for x in self.bounds.iter().delimited() {
                if x.is_first {
                    p.nbsp();
                } else {
                    p.space();
                    p.word("+ ");
                }
                &x.pretty(p);
            }
            p.end();
        }
    }
}

pub struct Gens {
    pub lt: Option<Token![<]>,
    pub params: Puncted<Param, Token![,]>,
    pub gt: Option<Token![>]>,
    pub where_: Option<Where>,
}
impl Gens {
    pub fn lifes(&self) -> param::Lifes {
        param::Lifes(self.params.iter())
    }
    pub fn lifes_mut(&mut self) -> param::LifesMut {
        param::LifesMut(self.params.iter_mut())
    }
    pub fn types(&self) -> param::Types {
        param::Types(self.params.iter())
    }
    pub fn types_mut(&mut self) -> param::TypesMut {
        param::TypesMut(self.params.iter_mut())
    }
    pub fn consts(&self) -> param::Consts {
        param::Consts(self.params.iter())
    }
    pub fn consts_mut(&mut self) -> param::ConstsMut {
        param::ConstsMut(self.params.iter_mut())
    }
    pub fn make_where(&mut self) -> &mut Where {
        self.where_.get_or_insert_with(|| Where {
            where_: <Token![where]>::default(),
            preds: Puncted::new(),
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
            params: Puncted::new(),
            gt: None,
            where_: None,
        }
    }
}
impl Parse for Gens {
    fn parse(s: Stream) -> Res<Self> {
        if !s.peek(Token![<]) {
            return Ok(Gens::default());
        }
        let lt: Token![<] = s.parse()?;
        let mut ys = Puncted::new();
        loop {
            if s.peek(Token![>]) {
                break;
            }
            let attrs = s.call(attr::Attr::parse_outers)?;
            let look = s.look1();
            if look.peek(Life) {
                ys.push_value(Param::Life(param::Life { attrs, ..s.parse()? }));
            } else if look.peek(ident::Ident) {
                ys.push_value(Param::Type(param::Type { attrs, ..s.parse()? }));
            } else if look.peek(Token![const]) {
                ys.push_value(Param::Const(param::Const { attrs, ..s.parse()? }));
            } else if s.peek(Token![_]) {
                ys.push_value(Param::Type(param::Type {
                    attrs,
                    ident: s.call(Ident::parse_any)?,
                    colon: None,
                    bounds: Puncted::new(),
                    eq: None,
                    default: None,
                }));
            } else {
                return Err(look.error());
            }
            if s.peek(Token![>]) {
                break;
            }
            let y = s.parse()?;
            ys.push_punct(y);
        }
        let gt: Token![>] = s.parse()?;
        Ok(Gens {
            lt: Some(lt),
            params: ys,
            gt: Some(gt),
            where_: None,
        })
    }
}
impl Lower for Gens {
    fn lower(&self, s: &mut Stream) {
        if self.params.is_empty() {
            return;
        }
        ToksOrDefault(&self.lt).lower(s);
        let mut trail_or_empty = true;
        for x in self.params.pairs() {
            if let Param::Life(_) = **x.value() {
                x.lower(s);
                trail_or_empty = x.punct().is_some();
            }
        }
        for x in self.params.pairs() {
            match x.value() {
                Param::Type(_) | Param::Const(_) => {
                    if !trail_or_empty {
                        <Token![,]>::default().lower(s);
                        trail_or_empty = true;
                    }
                    x.lower(s);
                },
                Param::Life(_) => {},
            }
        }
        ToksOrDefault(&self.gt).lower(s);
    }
}
impl Pretty for Gens {
    fn pretty(&self, p: &mut Print) {
        if self.params.is_empty() {
            return;
        }
        p.word("<");
        p.cbox(0);
        p.zerobreak();
        #[derive(Ord, PartialOrd, Eq, PartialEq)]
        enum Group {
            First,
            Second,
        }
        fn group(x: &Param) -> Group {
            use Param::*;
            match x {
                Life(_) => Group::First,
                Type(_) | Const(_) => Group::Second,
            }
        }
        let last = self.params.iter().max_by_key(|x| group(x));
        for g in [Group::First, Group::Second] {
            for x in &self.params {
                if group(x) == g {
                    x.pretty(p);
                    p.trailing_comma(std::ptr::eq(x, last.unwrap()));
                }
            }
        }
        p.offset(-INDENT);
        p.end();
        p.word(">");
    }
}

pub struct Impl<'a>(pub &'a Gens);
macro_rules! gens_impls {
    ($n:ident) => {
        impl<'a> Clone for $n<'a> {
            fn clone(&self) -> Self {
                $n(self.0)
            }
        }
        impl<'a> Debug for $n<'a> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.debug_tuple(stringify!($n)).field(self.0).finish()
            }
        }
        impl<'a> Eq for $n<'a> {}
        impl<'a> PartialEq for $n<'a> {
            fn eq(&self, x: &Self) -> bool {
                self.0 == x.0
            }
        }
        impl<'a> Hash for $n<'a> {
            fn hash<H: Hasher>(&self, x: &mut H) {
                self.0.hash(x);
            }
        }
    };
}
gens_impls!(Impl);
impl<'a> Lower for Impl<'a> {
    fn lower(&self, s: &mut Stream) {
        if self.0.params.is_empty() {
            return;
        }
        ToksOrDefault(&self.0.lt).lower(s);
        let mut trail_or_empty = true;
        for x in self.0.params.pairs() {
            if let Param::Life(_) = **x.value() {
                x.lower(s);
                trail_or_empty = x.punct().is_some();
            }
        }
        for x in self.0.params.pairs() {
            if let Param::Life(_) = **x.value() {
                continue;
            }
            if !trail_or_empty {
                <Token![,]>::default().lower(s);
                trail_or_empty = true;
            }
            match x.value() {
                Param::Life(_) => unreachable!(),
                Param::Type(x) => {
                    s.append_all(x.attrs.outers());
                    x.ident.lower(s);
                    if !x.bounds.is_empty() {
                        ToksOrDefault(&x.colon).lower(s);
                        x.bounds.lower(s);
                    }
                },
                Param::Const(x) => {
                    s.append_all(x.attrs.outers());
                    x.const_.lower(s);
                    x.ident.lower(s);
                    x.colon.lower(s);
                    x.typ.lower(s);
                },
            }
            x.punct().lower(s);
        }
        ToksOrDefault(&self.0.gt).lower(s);
    }
}

pub struct Type<'a>(pub &'a Gens);
gens_impls!(Type);
impl<'a> Lower for Type<'a> {
    fn lower(&self, s: &mut Stream) {
        if self.0.params.is_empty() {
            return;
        }
        ToksOrDefault(&self.0.lt).lower(s);
        let mut trail_or_empty = true;
        for x in self.0.params.pairs() {
            if let Param::Life(y) = *x.value() {
                y.life.lower(s);
                x.punct().lower(s);
                trail_or_empty = x.punct().is_some();
            }
        }
        for x in self.0.params.pairs() {
            if let Param::Life(_) = **x.value() {
                continue;
            }
            if !trail_or_empty {
                <Token![,]>::default().lower(s);
                trail_or_empty = true;
            }
            match x.value() {
                Param::Life(_) => unreachable!(),
                Param::Type(x) => {
                    x.ident.lower(s);
                },
                Param::Const(x) => {
                    x.ident.lower(s);
                },
            }
            x.punct().lower(s);
        }
        ToksOrDefault(&self.0.gt).lower(s);
    }
}
impl<'a> Type<'a> {
    pub fn as_turbofish(&self) -> Turbofish {
        Turbofish(self.0)
    }
}

pub struct Turbofish<'a>(pub &'a Gens);
gens_impls!(Turbofish);
impl<'a> Lower for Turbofish<'a> {
    fn lower(&self, s: &mut Stream) {
        if !self.0.params.is_empty() {
            <Token![::]>::default().lower(s);
            Type(self.0).lower(s);
        }
    }
}
