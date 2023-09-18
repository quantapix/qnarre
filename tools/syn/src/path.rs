use super::*;

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Expr,
    Simple,
    Type,
}

#[derive(Clone, Eq, PartialEq)]
pub struct Path {
    pub colon: Option<Token![::]>,
    pub segs: Puncted<Segment, Token![::]>,
}
impl Path {
    pub fn is_ident<I: ?Sized>(&self, i: &I) -> bool
    where
        Ident: PartialEq<I>,
    {
        match self.get_ident() {
            Some(x) => x == i,
            None => false,
        }
    }
    pub fn get_ident(&self) -> Option<&Ident> {
        if self.colon.is_none() && self.segs.len() == 1 && self.segs[0].args.is_none() {
            Some(&self.segs[0].ident)
        } else {
            None
        }
    }
    pub fn parse_mod_style(s: Stream) -> Res<Self> {
        Ok(Path {
            colon: s.parse()?,
            segs: {
                let mut ys = Puncted::new();
                loop {
                    if !s.peek(ident::Ident)
                        && !s.peek(Token![super])
                        && !s.peek(Token![self])
                        && !s.peek(Token![Self])
                        && !s.peek(Token![crate])
                    {
                        break;
                    }
                    let y = Ident::parse_any(s)?;
                    ys.push_value(Segment::from(y));
                    if !s.peek(Token![::]) {
                        break;
                    }
                    let y = s.parse()?;
                    ys.push_punct(y);
                }
                if ys.is_empty() {
                    return Err(s.parse::<Ident>().unwrap_err());
                } else if ys.trailing_punct() {
                    return Err(s.error("expected path segment after `::`"));
                }
                ys
            },
        })
    }
    pub fn parse_helper(s: Stream, style: bool) -> Res<Self> {
        let mut y = Path {
            colon: s.parse()?,
            segs: {
                let mut ys = Puncted::new();
                let y = Segment::parse_helper(s, style)?;
                ys.push_value(y);
                ys
            },
        };
        Path::parse_rest(s, &mut y, style)?;
        Ok(y)
    }
    pub fn parse_rest(s: Stream, path: &mut Self, style: bool) -> Res<()> {
        while s.peek(Token![::]) && !s.peek3(tok::Parenth) {
            let y: Token![::] = s.parse()?;
            path.segs.push_punct(y);
            let y = Segment::parse_helper(s, style)?;
            path.segs.push_value(y);
        }
        Ok(())
    }
    pub fn is_mod_style(&self) -> bool {
        self.segs.iter().all(|x| x.args.is_none())
    }
    pub fn pretty_qpath(&self, p: &mut Print, qself: &Option<QSelf>, kind: Kind) {
        let qself = match qself {
            Some(x) => x,
            None => {
                self.pretty_with_args(p, kind);
                return;
            },
        };
        assert!(qself.pos < self.segs.len());
        p.word("<");
        &qself.typ.pretty(p);
        let mut xs = self.segs.iter();
        if qself.pos > 0 {
            p.word(" as ");
            for x in xs.by_ref().take(qself.pos).delimited() {
                if !x.is_first || self.colon.is_some() {
                    p.word("::");
                }
                &x.pretty_with_args(p, Kind::Type);
                if x.is_last {
                    p.word(">");
                }
            }
        } else {
            p.word(">");
        }
        for x in xs {
            p.word("::");
            x.pretty_with_args(p, kind);
        }
    }
}
impl<T> From<T> for Path
where
    T: Into<Segment>,
{
    fn from(x: T) -> Self {
        let mut y = Path {
            colon: None,
            segs: Puncted::new(),
        };
        y.segs.push_value(x.into());
        y
    }
}
impl Parse for Path {
    fn parse(s: Stream) -> Res<Self> {
        Self::parse_helper(s, false)
    }
}
impl Lower for Path {
    fn lower(&self, s: &mut Stream) {
        self.colon.lower(s);
        self.segs.lower(s);
    }
}
impl Debug for Path {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Path {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("colon", &self.colon);
                f.field("segs", &self.segs);
                f.finish()
            }
        }
        self.debug(f, "Path")
    }
}
impl Pretty for Path {
    fn pretty_with_args(&self, p: &mut Print, x: &Option<pretty::Args>) {
        let Some(kind) = pretty::Args::kind(x);
        assert!(!self.segs.is_empty());
        for x in self.segs.iter().delimited() {
            if !x.is_first || self.colon.is_some() {
                p.word("::");
            }
            p.path_segment(&x, kind);
        }
    }
}
impl<F: Folder + ?Sized> Fold for Path {
    fn fold(&self, f: &mut F) {
        Path {
            colon: self.colon,
            segs: FoldHelper::lift(self.segs, |x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Path {
    fn hash(&self, h: &mut H) {
        self.colon.hash(h);
        self.segs.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Path {
    fn visit(&self, v: &mut V) {
        for y in Puncted::pairs(&self.segs) {
            let x = y.value();
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for mut y in Puncted::pairs_mut(&mut self.segs) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Segment {
    pub ident: Ident,
    pub args: Args,
}
impl Segment {
    fn parse_helper(s: Stream, style: bool) -> Res<Self> {
        if s.peek(Token![super]) || s.peek(Token![self]) || s.peek(Token![crate]) || s.peek(Token![try]) {
            let y = s.call(Ident::parse_any)?;
            return Ok(Segment::from(y));
        }
        let ident = if s.peek(Token![Self]) {
            s.call(Ident::parse_any)?
        } else {
            s.parse()?
        };
        if !style && s.peek(Token![<]) && !s.peek(Token![<=]) || s.peek(Token![::]) && s.peek3(Token![<]) {
            Ok(Segment {
                ident,
                args: Args::Angle(s.parse()?),
            })
        } else {
            Ok(Segment::from(ident))
        }
    }
}
impl<T> From<T> for Segment
where
    T: Into<Ident>,
{
    fn from(x: T) -> Self {
        Segment {
            ident: x.into(),
            args: Args::None,
        }
    }
}
impl Parse for Segment {
    fn parse(s: Stream) -> Res<Self> {
        Self::parse_helper(s, false)
    }
}
impl Lower for Segment {
    fn lower(&self, s: &mut Stream) {
        self.ident.lower(s);
        self.args.lower(s);
    }
}
impl Debug for Segment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("path::Segment");
        f.field("ident", &self.ident);
        f.field("args", &self.args);
        f.finish()
    }
}
impl Pretty for Segment {
    fn pretty_with_args(&self, p: &mut Print, x: &Option<pretty::Args>) {
        &self.ident.pretty(p);
        &self.args.pretty_with_args(p, x);
    }
}
impl<F: Folder + ?Sized> Fold for Segment {
    fn fold(&self, f: &mut F) {
        Segment {
            ident: self.ident.fold(f),
            args: self.args.fold(f),
        }
    }
}
impl<H: Hasher> Hash for Segment {
    fn hash(&self, h: &mut H) {
        self.ident.hash(h);
        self.args.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Segment {
    fn visit(&self, v: &mut V) {
        &self.ident.visit(v);
        &self.args.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut self.ident.visit_mut(v);
        &mut self.args.visit_mut(v);
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Args {
    None,
    Angle(Angle),
    Parenth(Parenth),
}
impl Args {
    pub fn is_empty(&self) -> bool {
        use Args::*;
        match self {
            None => true,
            Angle(x) => x.args.is_empty(),
            Parenth(_) => false,
        }
    }
    pub fn is_none(&self) -> bool {
        use Args::*;
        match self {
            None => true,
            Angle(_) | Parenth(_) => false,
        }
    }
}
impl Default for Args {
    fn default() -> Self {
        Args::None
    }
}
impl Lower for Args {
    fn lower(&self, s: &mut Stream) {
        use Args::*;
        match self {
            None => {},
            Angle(x) => {
                x.lower(s);
            },
            Parenth(x) => {
                x.lower(s);
            },
        }
    }
}
impl Debug for path::Args {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("path::Args::")?;
        use path::Args::*;
        match self {
            None => f.write_str("None"),
            Angle(x) => x.debug(f, "Angle"),
            Parenth(x) => x.debug(f, "Parenth"),
        }
    }
}
impl Pretty for Args {
    fn pretty_with_args(&self, p: &mut Print, x: &Option<pretty::Args>) {
        use Args::*;
        match self {
            None => {},
            Angle(x) => {
                x.pretty_with_args(p, x);
            },
            Parenth(x) => {
                x.pretty(p);
            },
        }
    }
}
impl<F: Folder + ?Sized> Fold for path::Args {
    fn fold(&self, f: &mut F) {
        use path::Args::*;
        match self {
            None => Args::None,
            Angle(x) => Angle(x.fold(f)),
            Parenth(x) => Parenth(x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Args {
    fn hash(&self, h: &mut H) {
        use Args::*;
        match self {
            None => {
                h.write_u8(0u8);
            },
            Angle(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Parenth(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
        }
    }
}
impl<V: Visitor + ?Sized> Visit for Args {
    fn visit(&self, v: &mut V) {
        use Args::*;
        match self {
            None => {},
            Angle(x) => {
                x.visit(v);
            },
            Parenth(x) => {
                x.visit(v);
            },
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        use Args::*;
        match self {
            None => {},
            Angle(x) => {
                x.visit_mut(v);
            },
            Parenth(x) => {
                x.visit_mut(v);
            },
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Arg {
    Life(Life),
    Type(typ::Type),
    Const(Expr),
    AssocType(AssocType),
    AssocConst(AssocConst),
    Constraint(Constraint),
}
impl Parse for Arg {
    fn parse(s: Stream) -> Res<Self> {
        if s.peek(Life) && !s.peek2(Token![+]) {
            return Ok(Arg::Life(s.parse()?));
        }
        if s.peek(lit::Lit) || s.peek(tok::Brace) {
            return const_arg(s).map(Arg::Const);
        }
        let mut y: typ::Type = s.parse()?;
        match y {
            typ::Type::Path(mut s)
                if s.qself.is_none()
                    && s.path.colon.is_none()
                    && s.path.segs.len() == 1
                    && match &s.path.segs[0].args {
                        Args::None | Args::AngleBracketed(_) => true,
                        Args::Parenth(_) => false,
                    } =>
            {
                if let Some(eq) = s.parse::<Option<Token![=]>>()? {
                    let seg = s.path.segs.pop().unwrap().into_value();
                    let ident = seg.ident;
                    let args = match seg.args {
                        Args::None => None,
                        Args::Angle(x) => Some(x),
                        Args::Parenth(_) => unreachable!(),
                    };
                    return if s.peek(lit::Lit) || s.peek(tok::Brace) {
                        Ok(Arg::AssocConst(AssocConst {
                            ident,
                            args,
                            eq,
                            val: const_arg(s)?,
                        }))
                    } else {
                        Ok(Arg::AssocType(AssocType {
                            ident,
                            args,
                            eq,
                            typ: s.parse()?,
                        }))
                    };
                }
                if let Some(colon) = s.parse::<Option<Token![:]>>()? {
                    let seg = s.path.segs.pop().unwrap().into_value();
                    return Ok(Arg::Constraint(Constraint {
                        ident: seg.ident,
                        args: match seg.args {
                            Args::None => None,
                            Args::Angle(x) => Some(x),
                            Args::Parenth(_) => unreachable!(),
                        },
                        colon,
                        bounds: {
                            let mut ys = Puncted::new();
                            loop {
                                if s.peek(Token![,]) || s.peek(Token![>]) {
                                    break;
                                }
                                let y: gen::bound::Type = s.parse()?;
                                ys.push_value(y);
                                if !s.peek(Token![+]) {
                                    break;
                                }
                                let y: Token![+] = s.parse()?;
                                ys.push_punct(y);
                            }
                            ys
                        },
                    }));
                }
                y = typ::Type::Path(s);
            },
            _ => {},
        }
        Ok(Arg::Type(y))
    }
}
impl Lower for Arg {
    #[allow(clippy::match_same_arms)]
    fn lower(&self, s: &mut Stream) {
        use Arg::*;
        match self {
            Life(x) => x.lower(s),
            Type(x) => x.lower(s),
            Const(x) => match x {
                Expr::Lit(_) => x.lower(s),
                Expr::Block(_) => x.lower(s),
                _ => tok::Brace::default().surround(s, |s| {
                    x.lower(s);
                }),
            },
            AssocType(x) => x.lower(s),
            AssocConst(x) => x.lower(s),
            Constraint(x) => x.lower(s),
        }
    }
}
impl Debug for Arg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("path::Arg::")?;
        use Arg::*;
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
            AssocType(x) => {
                let mut f = f.debug_tuple("AssocType");
                f.field(x);
                f.finish()
            },
            AssocConst(x) => {
                let mut f = f.debug_tuple("AssocConst");
                f.field(x);
                f.finish()
            },
            Constraint(x) => {
                let mut f = f.debug_tuple("Constraint");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Pretty for Arg {
    fn pretty(&self, p: &mut Print) {
        use Arg::*;
        match self {
            Life(x) => p.lifetime(x),
            Type(x) => p.ty(x),
            Const(x) => match x {
                expr::Expr::Lit(x) => x.pretty(p),
                expr::Expr::Block(x) => x.pretty(p),
                _ => {
                    p.word("{");
                    x.pretty(p);
                    p.word("}");
                },
            },
            AssocType(x) => p.assoc_type(x),
            AssocConst(x) => p.assoc_const(x),
            Constraint(x) => p.constraint(x),
        }
    }
}
impl<F: Folder + ?Sized> Fold for Arg {
    fn fold(&self, f: &mut F) {
        use Arg::*;
        match self {
            AssocConst(x) => AssocConst(x.fold(f)),
            AssocType(x) => AssocType(x.fold(f)),
            Const(x) => Const(x.fold(f)),
            Constraint(x) => Constraint(x.fold(f)),
            Life(x) => Life(x.fold(f)),
            Type(x) => Type(x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Arg {
    fn hash(&self, h: &mut H) {
        use Arg::*;
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
            AssocType(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            AssocConst(x) => {
                h.write_u8(4u8);
                x.hash(h);
            },
            Constraint(x) => {
                h.write_u8(5u8);
                x.hash(h);
            },
        }
    }
}
impl<V: Visitor + ?Sized> Visit for Arg {
    fn visit(&self, v: &mut V) {
        use Arg::*;
        match self {
            Life(x) => {
                x.visit(v);
            },
            Type(x) => {
                x.visit(v);
            },
            Const(x) => {
                x.visit(v);
            },
            AssocType(x) => {
                x.visit(v);
            },
            AssocConst(x) => {
                x.visit(v);
            },
            Constraint(x) => {
                x.visit(v);
            },
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        use Arg::*;
        match self {
            Life(x) => {
                x.visit_mut(v);
            },
            Type(x) => {
                x.visit_mut(v);
            },
            Const(x) => {
                x.visit_mut(v);
            },
            AssocType(x) => {
                x.visit_mut(v);
            },
            AssocConst(x) => {
                x.visit_mut(v);
            },
            Constraint(x) => {
                x.visit_mut(v);
            },
        }
    }
}

pub use expr::Expr;

#[derive(Clone, Eq, PartialEq)]
pub struct Angle {
    pub colon2: Option<Token![::]>,
    pub lt: Token![<],
    pub args: Puncted<Arg, Token![,]>,
    pub gt: Token![>],
}
impl Angle {
    pub fn parse_turbofish(s: Stream) -> Res<Self> {
        let y: Token![::] = s.parse()?;
        Self::do_parse(Some(y), s)
    }
    fn do_parse(colon2: Option<Token![::]>, s: Stream) -> Res<Self> {
        Ok(Angle {
            colon2,
            lt: s.parse()?,
            args: {
                let mut ys = Puncted::new();
                loop {
                    if s.peek(Token![>]) {
                        break;
                    }
                    let y: Arg = s.parse()?;
                    ys.push_value(y);
                    if s.peek(Token![>]) {
                        break;
                    }
                    let y: Token![,] = s.parse()?;
                    ys.push_punct(y);
                }
                ys
            },
            gt: s.parse()?,
        })
    }
}
impl Parse for Angle {
    fn parse(s: Stream) -> Res<Self> {
        let y: Option<Token![::]> = s.parse()?;
        Self::do_parse(y, s)
    }
}
impl Lower for Angle {
    fn lower(&self, s: &mut Stream) {
        self.colon2.lower(s);
        self.lt.lower(s);
        let mut trailing_or_empty = true;
        for x in self.args.pairs() {
            use Arg::*;
            match x.value() {
                Life(_) => {
                    x.lower(s);
                    trailing_or_empty = x.punct().is_some();
                },
                Type(_) | Const(_) | AssocType(_) | AssocConst(_) | Constraint(_) => {},
            }
        }
        for x in self.args.pairs() {
            use Arg::*;
            match x.value() {
                Type(_) | Const(_) | AssocType(_) | AssocConst(_) | Constraint(_) => {
                    if !trailing_or_empty {
                        <Token![,]>::default().lower(s);
                    }
                    x.lower(s);
                    trailing_or_empty = x.punct().is_some();
                },
                Life(_) => {},
            }
        }
        self.gt.lower(s);
    }
}
impl Debug for path::Angle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl path::Angle {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("colon2", &self.colon2);
                f.field("lt", &self.lt);
                f.field("args", &self.args);
                f.field("gt", &self.gt);
                f.finish()
            }
        }
        self.debug(f, "path::Angle")
    }
}
impl Pretty for Angle {
    fn pretty_with_args(&self, p: &mut Print, x: &Option<pretty::Args>) {
        let Some(kind) = pretty::Args::kind(x);
        if self.args.is_empty() || kind == Kind::Simple {
            return;
        }
        if kind == Kind::Expr {
            p.word("::");
        }
        p.word("<");
        p.cbox(INDENT);
        p.zerobreak();
        #[derive(Ord, PartialOrd, Eq, PartialEq)]
        enum Group {
            First,
            Second,
        }
        fn group(x: &Arg) -> Group {
            use Arg::*;
            match x {
                Life(_) => Group::First,
                Type(_) | Const(_) | AssocType(_) | AssocConst(_) | Constraint(_) => Group::Second,
            }
        }
        let last = self.args.iter().max_by_key(|x| group(x));
        for g in [Group::First, Group::Second] {
            for x in &self.args {
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
impl<F: Folder + ?Sized> Fold for path::Angle {
    fn fold(&self, f: &mut F) {
        path::Angle {
            colon2: self.colon2,
            lt: self.lt,
            args: FoldHelper::lift(self.args, |x| x.fold(f)),
            gt: self.gt,
        }
    }
}
impl<H: Hasher> Hash for Angle {
    fn hash(&self, h: &mut H) {
        self.colon2.hash(h);
        self.args.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Angle {
    fn visit<'a>(&self, v: &mut V) {
        for y in Puncted::pairs(&self.args) {
            let x = y.value();
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for mut y in Puncted::pairs_mut(&mut self.args) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Parenth {
    pub parenth: tok::Parenth,
    pub args: Puncted<typ::Type, Token![,]>,
    pub ret: typ::Ret,
}
impl Parse for Parenth {
    fn parse(s: Stream) -> Res<Self> {
        let y;
        Ok(Parenth {
            parenth: parenthed!(y in s),
            args: y.parse_terminated(typ::Type::parse, Token![,])?,
            ret: s.call(typ::Ret::without_plus)?,
        })
    }
}
impl Lower for Parenth {
    fn lower(&self, s: &mut Stream) {
        self.parenth.surround(s, |s| {
            self.args.lower(s);
        });
        self.ret.lower(s);
    }
}
impl Debug for path::Parenth {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl path::Parenth {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("parenth", &self.parenth);
                f.field("args", &self.args);
                f.field("ret", &self.ret);
                f.finish()
            }
        }
        self.debug(f, "path::Parenth")
    }
}
impl Pretty for Parenth {
    fn pretty(&self, p: &mut Print) {
        p.cbox(INDENT);
        p.word("(");
        p.zerobreak();
        for x in self.args.iter().delimited() {
            &x.pretty(p);
            p.trailing_comma(x.is_last);
        }
        p.offset(-INDENT);
        p.word(")");
        &self.ret.pretty(p);
        p.end();
    }
}
impl<F: Folder + ?Sized> Fold for path::Parenth {
    fn fold(&self, f: &mut F) {
        path::Parenth {
            parenth: self.parenth,
            args: FoldHelper::lift(self.args, |x| x.fold(f)),
            ret: self.ret.fold(f),
        }
    }
}
impl<H: Hasher> Hash for Parenth {
    fn hash(&self, h: &mut H) {
        self.args.hash(h);
        self.ret.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Parenth {
    fn visit(&self, v: &mut V) {
        for y in Puncted::pairs(&self.ins) {
            let x = y.value();
            x.visit(v);
        }
        &self.out.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        for mut y in Puncted::pairs_mut(&mut self.ins) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
        &mut self.out.visit_mut(v);
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct AssocType {
    pub ident: Ident,
    pub args: Option<Angle>,
    pub eq: Token![=],
    pub typ: typ::Type,
}
impl Lower for AssocType {
    fn lower(&self, s: &mut Stream) {
        self.ident.lower(s);
        self.args.lower(s);
        self.eq.lower(s);
        self.typ.lower(s);
    }
}
impl Debug for AssocType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("path::AssocType");
        f.field("ident", &self.ident);
        f.field("args", &self.args);
        f.field("eq", &self.eq);
        f.field("typ", &self.typ);
        f.finish()
    }
}
impl Pretty for AssocType {
    fn pretty(&self, p: &mut Print) {
        p.ident(&self.ident);
        if let Some(x) = &self.args {
            x.pretty_with_args(p, Kind::Type);
        }
        p.word(" = ");
        &self.typ.pretty(p);
    }
}
impl<F: Folder + ?Sized> Fold for path::AssocType {
    fn fold(&self, f: &mut F) {
        path::AssocType {
            ident: self.ident.fold(f),
            args: (self.args).map(|x| x.fold(f)),
            eq: self.eq,
            typ: self.typ.fold(f),
        }
    }
}
impl<H: Hasher> Hash for AssocType {
    fn hash(&self, h: &mut H) {
        self.ident.hash(h);
        self.args.hash(h);
        self.typ.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for AssocType {
    fn visit(&self, v: &mut V) {
        &self.ident.visit(v);
        if let Some(x) = &self.args {
            x.visit(v);
        }
        &self.ty.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut self.ident.visit_mut(v);
        if let Some(x) = &mut self.args {
            x.visit_mut(v);
        }
        &mut self.typ.visit_mut(v);
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct AssocConst {
    pub ident: Ident,
    pub args: Option<Angle>,
    pub eq: Token![=],
    pub val: Expr,
}
impl Lower for AssocConst {
    fn lower(&self, s: &mut Stream) {
        self.ident.lower(s);
        self.args.lower(s);
        self.eq.lower(s);
        self.val.lower(s);
    }
}
impl Debug for AssocConst {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("path::AssocConst");
        f.field("ident", &self.ident);
        f.field("args", &self.args);
        f.field("eq", &self.eq);
        f.field("val", &self.val);
        f.finish()
    }
}
impl Pretty for AssocConst {
    fn pretty(&self, p: &mut Print) {
        p.ident(&self.ident);
        if let Some(x) = &self.args {
            x.preetty(p, Kind::Type);
        }
        p.word(" = ");
        &self.val.pretty(p);
    }
}
impl<F: Folder + ?Sized> Fold for path::AssocConst {
    fn fold(&self, f: &mut F) {
        path::AssocConst {
            ident: self.ident.fold(f),
            args: (self.args).map(|x| x.fold(f)),
            eq: self.eq,
            val: self.val.fold(f),
        }
    }
}
impl<H: Hasher> Hash for AssocConst {
    fn hash(&self, h: &mut H) {
        self.ident.hash(h);
        self.args.hash(h);
        self.val.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for AssocConst {
    fn visit(&self, v: &mut V) {
        &self.ident.visit(v);
        if let Some(x) = &self.args {
            x.visit(v);
        }
        &self.val.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut self.ident.visit_mut(v);
        if let Some(x) = &mut self.args {
            x.visit_mut(v);
        }
        &mut self.val.visit_mut(v);
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Constraint {
    pub ident: Ident,
    pub args: Option<Angle>,
    pub colon: Token![:],
    pub bounds: Puncted<gen::bound::Type, Token![+]>,
}
impl Lower for Constraint {
    fn lower(&self, s: &mut Stream) {
        self.ident.lower(s);
        self.args.lower(s);
        self.colon.lower(s);
        self.bounds.lower(s);
    }
}
impl Debug for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Constraint");
        f.field("ident", &self.ident);
        f.field("args", &self.args);
        f.field("colon", &self.colon);
        f.field("bounds", &self.bounds);
        f.finish()
    }
}
impl Pretty for Constraint {
    fn pretty(&self, p: &mut Print) {
        p.ident(&self.ident);
        if let Some(x) = &self.args {
            x.pretty_with_args(p, Kind::Type);
        }
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
        p.end();
    }
}
impl<F: Folder + ?Sized> Fold for Constraint {
    fn fold(&self, f: &mut F) {
        Constraint {
            ident: self.ident.fold(f),
            args: (self.args).map(|x| x.fold(f)),
            colon: self.colon,
            bounds: FoldHelper::lift(self.bounds, |x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Constraint {
    fn hash(&self, h: &mut H) {
        self.ident.hash(h);
        self.args.hash(h);
        self.bounds.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Constraint {
    fn visit(&self, v: &mut V) {
        &self.ident.visit(v);
        if let Some(x) = &self.args {
            x.visit(v);
        }
        for y in Puncted::pairs(&self.bounds) {
            let x = y.value();
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut self.ident.visit_mut(v);
        if let Some(x) = &mut self.args {
            x.visit_mut(v);
        }
        for mut y in Puncted::pairs_mut(&mut self.bounds) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct QSelf {
    pub lt: Token![<],
    pub typ: Box<typ::Type>,
    pub pos: usize,
    pub as_: Option<Token![as]>,
    pub gt: Token![>],
}
impl Debug for QSelf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("QSelf");
        f.field("lt", &self.lt);
        f.field("typ", &self.typ);
        f.field("pos", &self.pos);
        f.field("as_", &self.as_);
        f.field("gt", &self.gt);
        f.finish()
    }
}
impl<F: Folder + ?Sized> Fold for path::QSelf {
    fn fold(&self, f: &mut F) {
        path::QSelf {
            lt: self.lt,
            typ: Box::new(*self.typ.fold(f)),
            pos: self.pos,
            as_: self.as_,
            gt: self.gt,
        }
    }
}
impl<H: Hasher> Hash for QSelf {
    fn hash(&self, h: &mut H) {
        self.typ.hash(h);
        self.pos.hash(h);
        self.as_.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for QSelf {
    fn visit(&self, v: &mut V) {
        &*self.typ.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut *self.typ.visit_mut(v);
    }
}

pub struct DisplayPath<'a>(pub &'a Path);
impl<'a> Display for DisplayPath<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, x) in self.0.segs.iter().enumerate() {
            if i > 0 || self.0.colon.is_some() {
                f.write_str("::")?;
            }
            write!(f, "{}", x.ident)?;
        }
        Ok(())
    }
}

pub fn const_arg(s: Stream) -> Res<Expr> {
    let y = s.look1();
    if s.peek(lit::Lit) {
        let y = s.parse()?;
        return Ok(Expr::Lit(y));
    }
    if s.peek(ident::Ident) {
        let y: Ident = s.parse()?;
        return Ok(Expr::Path(expr::Path {
            attrs: Vec::new(),
            qself: None,
            path: Path::from(y),
        }));
    }
    if s.peek(tok::Brace) {
        let y: expr::Block = s.parse()?;
        return Ok(Expr::Block(y));
    }
    Err(y.err())
}
pub fn qpath(s: Stream, style: bool) -> Res<(Option<QSelf>, Path)> {
    if s.peek(Token![<]) {
        let lt: Token![<] = s.parse()?;
        let this: typ::Type = s.parse()?;
        let y = if s.peek(Token![as]) {
            let as_: Token![as] = s.parse()?;
            let y: Path = s.parse()?;
            Some((as_, y))
        } else {
            None
        };
        let gt: Token![>] = s.parse()?;
        let colon2: Token![::] = s.parse()?;
        let mut rest = Puncted::new();
        loop {
            let path = Segment::parse_helper(s, style)?;
            rest.push_value(path);
            if !s.peek(Token![::]) {
                break;
            }
            let punct: Token![::] = s.parse()?;
            rest.push_punct(punct);
        }
        let (pos, as_, y) = match y {
            Some((as_, mut y)) => {
                let pos = y.segs.len();
                y.segs.push_punct(colon2);
                y.segs.extend(rest.into_pairs());
                (pos, Some(as_), y)
            },
            None => {
                let y = Path {
                    colon: Some(colon2),
                    segs: rest,
                };
                (0, None, y)
            },
        };
        let qself = QSelf {
            lt,
            typ: Box::new(this),
            pos,
            as_,
            gt,
        };
        Ok((Some(qself), y))
    } else {
        let y = Path::parse_helper(s, style)?;
        Ok((None, y))
    }
}
pub fn path_lower(s: &mut Stream, x: &Option<QSelf>, path: &Path) {
    let x = match x {
        Some(x) => x,
        None => {
            path.lower(s);
            return;
        },
    };
    x.lt.lower(s);
    x.typ.lower(s);
    let pos = cmp::min(x.pos, path.segs.len());
    let mut ys = path.segs.pairs();
    if pos > 0 {
        ToksOrDefault(&x.as_).lower(s);
        path.colon.lower(s);
        for (i, y) in ys.by_ref().take(pos).enumerate() {
            if i + 1 == pos {
                y.value().lower(s);
                x.gt.lower(s);
                y.punct().lower(s);
            } else {
                y.lower(s);
            }
        }
    } else {
        x.gt.lower(s);
        path.colon.lower(s);
    }
    for y in ys {
        y.lower(s);
    }
}
