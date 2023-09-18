use super::{
    punct::{Iter, IterMut},
    *,
};

pub mod bound {
    use super::*;
    enum_of_structs! {
        #[derive(Eq, PartialEq)]
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
            let tilde = y.peek(Token![~]) && y.peek2(Token![const]);
            if tilde {
                y.parse::<Token![~]>()?;
                y.parse::<Token![const]>()?;
            }
            let mut y: Trait = y.parse()?;
            y.parenth = parenth;
            if tilde {
                Ok(Type::Verbatim(parse::parse_verbatim(&beg, s)))
            } else {
                Ok(Type::Trait(y))
            }
        }
    }
    impl Clone for Type {
        fn clone(&self) -> Self {
            use self::Type::*;
            match self {
                Life(x) => Life(x.clone()),
                Trait(x) => Trait(x.clone()),
                Verbatim(x) => Verbatim(x.clone()),
            }
        }
    }
    impl Debug for Type {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("gen::bound::Type::")?;
            use self::Type::*;
            match self {
                Life(x) => x.debug(f, "Life"),
                Trait(x) => {
                    let mut f = f.debug_tuple("Trait");
                    f.field(x);
                    f.finish()
                },
                Verbatim(x) => {
                    let mut f = f.debug_tuple("Verbatim");
                    f.field(x);
                    f.finish()
                },
            }
        }
    }
    impl Pretty for Type {
        fn pretty(&self, p: &mut Print) {
            use self::Type::*;
            match self {
                Life(x) => x.pretty(p),
                Trait(x) => {
                    let tilde = false;
                    x.pretty_with_args(p, tilde);
                },
                Verbatim(x) => x.pretty(p),
            }
        }
    }
    impl<F: Folder + ?Sized> Fold for Type {
        fn fold(&self, f: &mut F) {
            use self::Type::*;
            match self {
                Life(x) => Life(x.fold(f)),
                Trait(x) => Trait(x.fold(f)),
                Verbatim(x) => Verbatim(x),
            }
        }
    }
    impl<H: Hasher> Hash for Type {
        fn hash(&self, h: &mut H) {
            use self::Type::*;
            match self {
                Life(x) => {
                    h.write_u8(1u8);
                    x.hash(h);
                },
                Trait(x) => {
                    h.write_u8(0u8);
                    x.hash(h);
                },
                Verbatim(x) => {
                    h.write_u8(2u8);
                    StreamHelper(x).hash(h);
                },
            }
        }
    }
    impl<V: Visitor + ?Sized> Visit for Type {
        fn visit(&self, v: &mut V) {
            use self::Type::*;
            match self {
                Life(x) => {
                    x.visit(v);
                },
                Trait(x) => {
                    x.visit(v);
                },
                Verbatim(_) => {},
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            use self::Type::*;
            match self {
                Life(x) => {
                    x.visit_mut(v);
                },
                Trait(x) => {
                    x.visit_mut(v);
                },
                Verbatim(_) => {},
            }
        }
    }

    #[derive(Eq, PartialEq)]
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
                let y: path::Parenth = s.parse()?;
                let y = path::Args::Parenth(y);
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
    impl Clone for Trait {
        fn clone(&self) -> Self {
            Trait {
                parenth: self.parenth.clone(),
                modif: self.modif.clone(),
                lifes: self.lifes.clone(),
                path: self.path.clone(),
            }
        }
    }
    impl Debug for Trait {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut f = f.debug_struct("gen::bound::Trait");
            f.field("parenth", &self.parenth);
            f.field("modif", &self.modif);
            f.field("lifes", &self.lifes);
            f.field("path", &self.path);
            f.finish()
        }
    }
    impl Pretty for Trait {
        fn pretty_with_args(&self, p: &mut Print, x: &Option<pretty::Args>) {
            if self.parenth.is_some() {
                p.word("(");
            }
            if pretty::Args::tilde(x) {
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
    impl<F: Folder + ?Sized> Fold for Trait {
        fn fold(&self, f: &mut F) {
            Trait {
                parenth: self.parenth,
                modif: self.modif.fold(f),
                lifes: (self.lifes).map(|x| x.fold(f)),
                path: self.path.fold(f),
            }
        }
    }
    impl<H: Hasher> Hash for Trait {
        fn hash(&self, h: &mut H) {
            self.parenth.hash(h);
            self.modif.hash(h);
            self.lifes.hash(h);
            self.path.hash(h);
        }
    }
    impl<V: Visitor + ?Sized> Visit for Trait {
        fn visit(&self, v: &mut V) {
            &self.modif.visit(v);
            if let Some(x) = &self.lifes {
                x.visit(v);
            }
            &self.path.visit(v);
        }
        fn visit_mut(&mut self, v: &mut V) {
            &mut self.modif.visit_mut(v);
            if let Some(x) = &mut self.lifes {
                x.visit_mut(v);
            }
            &mut self.path.visit_mut(v);
        }
    }

    #[derive(Eq, PartialEq)]
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
    impl Copy for Modifier {}
    impl Clone for Modifier {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl Debug for Modifier {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("gen::bound::Modifier::")?;
            use Modifier::*;
            match self {
                Maybe(x) => {
                    let mut f = f.debug_tuple("Maybe");
                    f.field(x);
                    f.finish()
                },
                None => f.write_str("None"),
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
    impl<F: Folder + ?Sized> Fold for Modifier {
        fn fold(&self, f: &mut F) {
            use Modifier::*;
            match self {
                Maybe(x) => Maybe(x),
                None => None,
            }
        }
    }
    impl<H: Hasher> Hash for Modifier {
        fn hash(&self, h: &mut H) {
            use Modifier::*;
            match self {
                Maybe(_) => {
                    h.write_u8(1u8);
                },
                None => {
                    h.write_u8(0u8);
                },
            }
        }
    }
    impl<V: Visitor + ?Sized> Visit for Modifier {
        fn visit(&self, v: &mut V) {
            use Modifier::*;
            match self {
                Maybe(_) => {},
                None => {},
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            use Modifier::*;
            match self {
                Maybe(_) => {},
                None => {},
            }
        }
    }

    #[derive(Eq, PartialEq)]
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
    impl Clone for Lifes {
        fn clone(&self) -> Self {
            Lifes {
                for_: self.for_.clone(),
                lt: self.lt.clone(),
                lifes: self.lifes.clone(),
                gt: self.gt.clone(),
            }
        }
    }
    impl Debug for Lifes {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut f = f.debug_struct("gen::bound::Lifes");
            f.field("for_", &self.for_);
            f.field("lt", &self.lt);
            f.field("lifes", &self.lifes);
            f.field("gt", &self.gt);
            f.finish()
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
    impl<F: Folder + ?Sized> Fold for Lifes {
        fn fold(&self, f: &mut F) {
            Lifes {
                for_: self.for_,
                lt: self.lt,
                lifes: FoldHelper::lift(self.lifes, |x| x.fold(f)),
                gt: self.gt,
            }
        }
    }
    impl<H: Hasher> Hash for Lifes {
        fn hash(&self, h: &mut H) {
            self.lifes.hash(h);
        }
    }
    impl<V: Visitor + ?Sized> Visit for Lifes {
        fn visit(&self, v: &mut V) {
            for y in Puncted::pairs(&self.lifes) {
                let x = y.value();
                x.visit(v);
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            for mut y in Puncted::pairs_mut(&mut self.lifes) {
                let x = y.value_mut();
                x.visit_mut(v);
            }
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
        #[derive(Eq, PartialEq)]
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
    impl Clone for Param {
        fn clone(&self) -> Self {
            use Param::*;
            match self {
                Const(x) => Const(x.clone()),
                Life(x) => Life(x.clone()),
                Type(x) => Type(x.clone()),
            }
        }
    }
    impl Debug for Param {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("gen::param::Param::")?;
            use Param::*;
            match self {
                Life(x) => {
                    let mut f = f.debug_tuple("Life");
                    f.field(x);
                    f.finish()
                },
                Type(x) => {
                    let mut f = f.debug_tuple("Type");
                    f.field(x);
                    f.finish()
                },
                Const(x) => {
                    let mut f = f.debug_tuple("Const");
                    f.field(x);
                    f.finish()
                },
            }
        }
    }
    impl<H: Hasher> Hash for Param {
        fn hash(&self, h: &mut H) {
            use Param::*;
            match self {
                Life(x) => {
                    h.write_u8(0u8);
                    x.hash(h);
                },
                Type(x) => {
                    h.write_u8(1u8);
                    x.hash(h);
                },
                Const(x) => {
                    h.write_u8(2u8);
                    x.hash(h);
                },
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
    impl<F: Folder + ?Sized> Fold for Param {
        fn fold(&self, f: &mut F) {
            use Param::*;
            match self {
                Const(x) => Const(x.fold(f)),
                Life(x) => Life(x.fold(f)),
                Type(x) => Type(x.fold(f)),
            }
        }
    }
    impl<V: Visitor + ?Sized> Visit for Param {
        fn visit(&self, v: &mut V) {
            use Param::*;
            match self {
                Const(x) => {
                    x.visit(v);
                },
                Life(x) => {
                    x.visit(v);
                },
                Type(x) => {
                    x.visit(v);
                },
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            use Param::*;
            match self {
                Const(x) => {
                    x.visit_mut(v);
                },
                Life(x) => {
                    x.visit_mut(v);
                },
                Type(x) => {
                    x.visit_mut(v);
                },
            }
        }
    }

    #[derive(Eq, PartialEq)]
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
    impl Clone for Life {
        fn clone(&self) -> Self {
            Life {
                attrs: self.attrs.clone(),
                life: self.life.clone(),
                colon: self.colon.clone(),
                bounds: self.bounds.clone(),
            }
        }
    }
    impl Debug for Life {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut f = f.debug_struct("gen::param::Life");
            f.field("attrs", &self.attrs);
            f.field("life", &self.life);
            f.field("colon", &self.colon);
            f.field("bounds", &self.bounds);
            f.finish()
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
    impl<F: Folder + ?Sized> Fold for Life {
        fn fold(&self, f: &mut F) {
            Life {
                attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
                life: self.life.fold(f),
                colon: self.colon,
                bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
            }
        }
    }
    impl<H: Hasher> Hash for Life {
        fn hash(&self, h: &mut H) {
            self.attrs.hash(h);
            self.life.hash(h);
            self.colon.hash(h);
            self.bounds.hash(h);
        }
    }
    impl<V: Visitor + ?Sized> Visit for Life {
        fn visit(&self, v: &mut V) {
            for x in &self.attrs {
                x.visit(v);
            }
            &self.life.visit(v);
            for y in Puncted::pairs(&self.bounds) {
                let x = y.value();
                x.visit(v);
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            for x in &mut self.attrs {
                x.visit_mut(v);
            }
            &mut self.life.visit_mut(v);
            for mut y in Puncted::pairs_mut(&mut self.bounds) {
                let x = y.value_mut();
                x.visit_mut(v);
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

    #[derive(Eq, PartialEq)]
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
    impl Clone for Type {
        fn clone(&self) -> Self {
            Type {
                attrs: self.attrs.clone(),
                ident: self.ident.clone(),
                colon: self.colon.clone(),
                bounds: self.bounds.clone(),
                eq: self.eq.clone(),
                default: self.default.clone(),
            }
        }
    }
    impl Debug for Type {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut f = f.debug_struct("gen::param::Type");
            f.field("attrs", &self.attrs);
            f.field("ident", &self.ident);
            f.field("colon", &self.colon);
            f.field("bounds", &self.bounds);
            f.field("eq", &self.eq);
            f.field("default", &self.default);
            f.finish()
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
    impl<F: Folder + ?Sized> Fold for Type {
        fn fold(&self, f: &mut F) {
            Type {
                attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
                ident: self.ident.fold(f),
                colon: self.colon,
                bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
                eq: self.eq,
                default: (self.default).map(|x| x.fold(f)),
            }
        }
    }
    impl<H: Hasher> Hash for Type {
        fn hash(&self, h: &mut H) {
            self.attrs.hash(h);
            self.ident.hash(h);
            self.colon.hash(h);
            self.bounds.hash(h);
            self.eq.hash(h);
            self.default.hash(h);
        }
    }
    impl<V: Visitor + ?Sized> Visit for Type {
        fn visit(&self, v: &mut V) {
            for x in &self.attrs {
                x.visit(v);
            }
            &self.ident.visit(v);
            for y in Puncted::pairs(&self.bounds) {
                let x = y.value();
                x.visit(v);
            }
            if let Some(x) = &self.default {
                x.visit(v);
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            for x in &mut self.attrs {
                x.visit_mut(v);
            }
            &mut self.ident.visit_mut(v);
            for mut y in Puncted::pairs_mut(&mut self.bounds) {
                let x = y.value_mut();
                x.visit_mut(v);
            }
            if let Some(x) = &mut self.default {
                x.visit_mut(v);
            }
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

    #[derive(Eq, PartialEq)]
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
    impl Clone for Const {
        fn clone(&self) -> Self {
            Const {
                attrs: self.attrs.clone(),
                const_: self.const_.clone(),
                ident: self.ident.clone(),
                colon: self.colon.clone(),
                typ: self.typ.clone(),
                eq: self.eq.clone(),
                default: self.default.clone(),
            }
        }
    }
    impl Debug for Const {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut f = f.debug_struct("gen::param::Const");
            f.field("attrs", &self.attrs);
            f.field("const_", &self.const_);
            f.field("ident", &self.ident);
            f.field("colon", &self.colon);
            f.field("typ", &self.typ);
            f.field("eq", &self.eq);
            f.field("default", &self.default);
            f.finish()
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
    impl<F: Folder + ?Sized> Fold for Const {
        fn fold(&self, f: &mut F) {
            Const {
                attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
                const_: self.const_,
                ident: self.ident.fold(f),
                colon: self.colon,
                typ: self.typ.fold(f),
                eq: self.eq,
                default: (self.default).map(|x| x.fold(f)),
            }
        }
    }
    impl<H: Hasher> Hash for Const {
        fn hash(&self, h: &mut H) {
            self.attrs.hash(h);
            self.ident.hash(h);
            self.typ.hash(h);
            self.eq.hash(h);
            self.default.hash(h);
        }
    }
    impl<V: Visitor + ?Sized> Visit for Const {
        fn visit(&self, v: &mut V) {
            for x in &self.attrs {
                x.visit(v);
            }
            &self.ident.visit(v);
            &self.typ.visit(v);
            if let Some(x) = &self.default {
                x.visit(v);
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            for x in &mut self.attrs {
                x.visit_mut(v);
            }
            &mut self.ident.visit_mut(v);
            &mut self.typ.visit_mut(v);
            if let Some(x) = &mut self.default {
                x.visit_mut(v);
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

#[derive(Eq, PartialEq)]
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
impl Lower for Where {
    fn lower(&self, s: &mut Stream) {
        if !self.preds.is_empty() {
            self.where_.lower(s);
            self.preds.lower(s);
        }
    }
}
impl Clone for Where {
    fn clone(&self) -> Self {
        Where {
            where_: self.where_.clone(),
            preds: self.preds.clone(),
        }
    }
}
impl Debug for Where {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("gen::Where");
        f.field("where_", &self.where_);
        f.field("preds", &self.preds);
        f.finish()
    }
}
impl Pretty for Where {
    fn pretty_with_args(&self, p: &mut Print, x: &Option<pretty::Args>) {
        let Some(breaks, semi) = pretty::Args::breaks_semi(x);
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
impl<F: Folder + ?Sized> Fold for Where {
    fn fold(&self, f: &mut F) {
        Where {
            where_: self.where_,
            preds: FoldHelper::lift(self.preds, |x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Where {
    fn hash(&self, h: &mut H) {
        self.preds.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Where {
    fn visit(&self, v: &mut V) {
        for y in Puncted::pairs(&self.preds) {
            let x = y.value();
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for mut y in Puncted::pairs_mut(&mut self.preds) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
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
impl Pretty for Option<Where> {
    fn pretty_with_args(&self, p: &mut Print, x: &Option<pretty::Args>) {
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
        y.pretty_with_args(p, x);
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
        #[derive(Eq, PartialEq)]
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
    impl Clone for Pred {
        fn clone(&self) -> Self {
            use Pred::*;
            match self {
                Life(x) => Life(x.clone()),
                Type(x) => Type(x.clone()),
            }
        }
    }
    impl Debug for Pred {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("gen::where_::Pred::")?;
            use Pred::*;
            match self {
                Life(x) => {
                    let mut f = f.debug_tuple("Life");
                    f.field(x);
                    f.finish()
                },
                Type(x) => {
                    let mut f = f.debug_tuple("Type");
                    f.field(x);
                    f.finish()
                },
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
    impl<F: Folder + ?Sized> Fold for Pred {
        fn fold(&self, f: &mut F) {
            use Pred::*;
            match self {
                Life(x) => Life(x.fold(f)),
                Type(x) => Type(x.fold(f)),
            }
        }
    }
    impl<H: Hasher> Hash for Pred {
        fn hash(&self, h: &mut H) {
            use Pred::*;
            match self {
                Life(v0) => {
                    h.write_u8(0u8);
                    v0.hash(h);
                },
                Type(v0) => {
                    h.write_u8(1u8);
                    v0.hash(h);
                },
            }
        }
    }
    impl<V: Visitor + ?Sized> Visit for Pred {
        fn visit(&self, v: &mut V) {
            use Pred::*;
            match self {
                Life(x) => {
                    x.visit(v);
                },
                Type(x) => {
                    x.visit(v);
                },
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            use Pred::*;
            match self {
                Life(x) => {
                    x.visit_mut(v);
                },
                Type(x) => {
                    x.visit_mut(v);
                },
            }
        }
    }

    #[derive(Eq, PartialEq)]
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
    impl Clone for Life {
        fn clone(&self) -> Self {
            Life {
                life: self.life.clone(),
                colon: self.colon.clone(),
                bounds: self.bounds.clone(),
            }
        }
    }
    impl Debug for Life {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut f = f.debug_struct("gen::where_::Life");
            f.field("life", &self.life);
            f.field("colon", &self.colon);
            f.field("bounds", &self.bounds);
            f.finish()
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
    impl<F: Folder + ?Sized> Fold for Life {
        fn fold(&self, f: &mut F) {
            Life {
                life: self.life.fold(f),
                colon: self.colon,
                bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
            }
        }
    }
    impl<H: Hasher> Hash for Life {
        fn hash(&self, h: &mut H) {
            self.life.hash(h);
            self.bounds.hash(h);
        }
    }
    impl<V: Visitor + ?Sized> Visit for Life {
        fn visit(&self, v: &mut V) {
            &self.life.visit(v);
            for y in Puncted::pairs(&self.bounds) {
                let x = y.value();
                x.visit(v);
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            &mut self.life.visit_mut(v);
            for mut y in Puncted::pairs_mut(&mut self.bounds) {
                let x = y.value_mut();
                x.visit_mut(v);
            }
        }
    }

    #[derive(Eq, PartialEq)]
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
    impl Clone for Type {
        fn clone(&self) -> Self {
            Type {
                lifes: self.lifes.clone(),
                typ: self.typ.clone(),
                colon: self.colon.clone(),
                bounds: self.bounds.clone(),
            }
        }
    }
    impl Debug for Type {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut f = f.debug_struct("gen::where_::Type");
            f.field("lifes", &self.lifes);
            f.field("typ", &self.typ);
            f.field("colon", &self.colon);
            f.field("bounds", &self.bounds);
            f.finish()
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
    impl<F: Folder + ?Sized> Fold for Type {
        fn fold(&self, f: &mut F) {
            Type {
                lifes: (self.lifes).map(|x| x.fold(f)),
                typ: self.typ.fold(f),
                colon: self.colon,
                bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
            }
        }
    }
    impl<H: Hasher> Hash for Type {
        fn hash(&self, h: &mut H) {
            self.lifes.hash(h);
            self.typ.hash(h);
            self.bounds.hash(h);
        }
    }
    impl<V: Visitor + ?Sized> Visit for Type {
        fn visit(&self, v: &mut V) {
            if let Some(x) = &self.lifes {
                x.visit(v);
            }
            &self.typ.visit(v);
            for y in Puncted::pairs(&self.bounds) {
                let x = y.value();
                x.visit(v);
            }
        }
        fn visit_mut(&mut self, v: &mut V) {
            if let Some(x) = &mut self.lifes {
                x.visit_mut(v);
            }
            &mut self.typ.visit_mut(v);
            for mut y in Puncted::pairs_mut(&mut self.bounds) {
                let x = y.value_mut();
                x.visit_mut(v);
            }
        }
    }
}

#[derive(Eq, PartialEq)]
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
impl Clone for Gens {
    fn clone(&self) -> Self {
        Gens {
            lt: self.lt.clone(),
            params: self.params.clone(),
            gt: self.gt.clone(),
            where_: self.where_.clone(),
        }
    }
}
impl Debug for Gens {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("gen::Gens");
        f.field("lt", &self.lt);
        f.field("params", &self.params);
        f.field("gt", &self.gt);
        f.field("where_", &self.where_);
        f.finish()
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
impl<F: Folder + ?Sized> Fold for Gens {
    fn fold(&self, f: &mut F) {
        Gens {
            lt: self.lt,
            params: FoldHelper::lift(self.params, |x| x.fold(f)),
            gt: self.gt,
            where_: (self.where_).map(|x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Gens {
    fn hash(&self, h: &mut H) {
        self.lt.hash(h);
        self.params.hash(h);
        self.gt.hash(h);
        self.where_.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Gens {
    fn visit(&self, v: &mut V) {
        for y in Puncted::pairs(&self.params) {
            let x = y.value();
            x.visit(v);
        }
        if let Some(x) = &self.where_ {
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for mut y in Puncted::pairs_mut(&mut self.params) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
        if let Some(x) = &mut self.where_ {
            x.visit_mut(v);
        }
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
