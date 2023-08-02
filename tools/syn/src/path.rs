use super::*;

#[derive(Copy, Clone, PartialEq)]
pub enum Kind {
    Expr,
    Simple,
    Type,
}

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
        while s.peek(Token![::]) && !s.peek3(tok::Paren) {
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
                self.pretty(p, kind);
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
                &x.pretty(p, Kind::Type);
                if x.is_last {
                    p.word(">");
                }
            }
        } else {
            p.word(">");
        }
        for x in xs {
            p.word("::");
            x.pretty(p, kind);
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
impl Pretty for Path {
    fn pretty(&self, p: &mut Print, kind: Kind) {
        assert!(!self.segs.is_empty());
        for x in self.segs.iter().delimited() {
            if !x.is_first || self.colon.is_some() {
                p.word("::");
            }
            p.path_segment(&x, kind);
        }
    }
}

pub struct Segment {
    pub ident: Ident,
    pub args: Args,
}
impl Segment {
    fn parse_helper(x: Stream, expr_style: bool) -> Res<Self> {
        if x.peek(Token![super]) || x.peek(Token![self]) || x.peek(Token![crate]) || x.peek(Token![try]) {
            let y = x.call(Ident::parse_any)?;
            return Ok(Segment::from(y));
        }
        let ident = if x.peek(Token![Self]) {
            x.call(Ident::parse_any)?
        } else {
            x.parse()?
        };
        if !expr_style && x.peek(Token![<]) && !x.peek(Token![<=]) || x.peek(Token![::]) && x.peek3(Token![<]) {
            Ok(Segment {
                ident,
                args: Args::Angled(x.parse()?),
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
impl Pretty for Segment {
    fn pretty(&self, p: &mut Print, kind: Kind) {
        &self.ident.pretty(p);
        &self.args.pretty(p, kind);
    }
}

pub enum Args {
    None,
    Angled(Angled),
    Parenthesized(Parenthesized),
}
impl Args {
    pub fn is_empty(&self) -> bool {
        use Args::*;
        match self {
            None => true,
            Angled(x) => x.args.is_empty(),
            Parenthesized(_) => false,
        }
    }
    pub fn is_none(&self) -> bool {
        use Args::*;
        match self {
            None => true,
            Angled(_) | Parenthesized(_) => false,
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
            Angled(x) => {
                x.lower(s);
            },
            Parenthesized(x) => {
                x.lower(s);
            },
        }
    }
}
impl Pretty for Args {
    fn pretty(&self, p: &mut Print, kind: Kind) {
        use Args::*;
        match self {
            None => {},
            Angled(x) => {
                x.pretty(p, kind);
            },
            Parenthesized(x) => {
                x.pretty(p);
            },
        }
    }
}

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
                        Args::Parenthesized(_) => false,
                    } =>
            {
                if let Some(eq) = s.parse::<Option<Token![=]>>()? {
                    let seg = s.path.segs.pop().unwrap().into_value();
                    let ident = seg.ident;
                    let args = match seg.args {
                        Args::None => None,
                        Args::Angled(x) => Some(x),
                        Args::Parenthesized(_) => unreachable!(),
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
                            Args::Angled(x) => Some(x),
                            Args::Parenthesized(_) => unreachable!(),
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

pub use expr::Expr;

pub struct Angled {
    pub colon2: Option<Token![::]>,
    pub lt: Token![<],
    pub args: Puncted<Arg, Token![,]>,
    pub gt: Token![>],
}
impl Angled {
    pub fn parse_turbofish(s: Stream) -> Res<Self> {
        let y: Token![::] = s.parse()?;
        Self::do_parse(Some(y), s)
    }
    fn do_parse(colon2: Option<Token![::]>, s: Stream) -> Res<Self> {
        Ok(Angled {
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
impl Parse for Angled {
    fn parse(s: Stream) -> Res<Self> {
        let y: Option<Token![::]> = s.parse()?;
        Self::do_parse(y, s)
    }
}
impl Lower for Angled {
    fn lower(&self, s: &mut Stream) {
        self.colon2.lower(s);
        self.lt.lower(s);
        let mut trailing_or_empty = true;
        for x in self.args.pairs() {
            match x.value() {
                Arg::Life(_) => {
                    x.lower(s);
                    trailing_or_empty = x.punct().is_some();
                },
                Arg::Type(_) | Arg::Const(_) | Arg::AssocType(_) | Arg::AssocConst(_) | Arg::Constraint(_) => {},
            }
        }
        for x in self.args.pairs() {
            match x.value() {
                Arg::Type(_) | Arg::Const(_) | Arg::AssocType(_) | Arg::AssocConst(_) | Arg::Constraint(_) => {
                    if !trailing_or_empty {
                        <Token![,]>::default().lower(s);
                    }
                    x.lower(s);
                    trailing_or_empty = x.punct().is_some();
                },
                Arg::Life(_) => {},
            }
        }
        self.gt.lower(s);
    }
}
impl Pretty for Angled {
    fn pretty(&self, p: &mut Print, kind: Kind) {
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

pub struct AssocType {
    pub ident: Ident,
    pub args: Option<Angled>,
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
impl Pretty for AssocType {
    fn pretty(&self, p: &mut Print) {
        p.ident(&self.ident);
        if let Some(x) = &self.args {
            x.pretty(p, Kind::Type);
        }
        p.word(" = ");
        &self.typ.pretty(p);
    }
}

pub struct AssocConst {
    pub ident: Ident,
    pub args: Option<Angled>,
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

pub struct Constraint {
    pub ident: Ident,
    pub args: Option<Angled>,
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
impl Pretty for Constraint {
    fn pretty(&self, p: &mut Print) {
        p.ident(&self.ident);
        if let Some(x) = &self.args {
            x.pretty(p, Kind::Type);
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

pub struct Parenthesized {
    pub paren: tok::Paren,
    pub args: Puncted<typ::Type, Token![,]>,
    pub ret: typ::Ret,
}
impl Parse for Parenthesized {
    fn parse(s: Stream) -> Res<Self> {
        let y;
        Ok(Parenthesized {
            paren: parenthesized!(y in s),
            args: y.parse_terminated(typ::Type::parse, Token![,])?,
            ret: s.call(typ::Ret::without_plus)?,
        })
    }
}
impl Lower for Parenthesized {
    fn lower(&self, s: &mut Stream) {
        self.paren.surround(s, |s| {
            self.args.lower(s);
        });
        self.ret.lower(s);
    }
}
impl Pretty for Parenthesized {
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

pub struct QSelf {
    pub lt: Token![<],
    pub typ: Box<typ::Type>,
    pub pos: usize,
    pub as_: Option<Token![as]>,
    pub gt: Token![>],
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

pub fn const_arg(x: Stream) -> Res<Expr> {
    let y = x.look1();
    if x.peek(lit::Lit) {
        let y = x.parse()?;
        return Ok(Expr::Lit(y));
    }
    if x.peek(ident::Ident) {
        let y: Ident = x.parse()?;
        return Ok(Expr::Path(expr::Path {
            attrs: Vec::new(),
            qself: None,
            path: Path::from(y),
        }));
    }
    if x.peek(tok::Brace) {
        let y: expr::Block = x.parse()?;
        return Ok(Expr::Block(y));
    }
    Err(y.err())
}
pub fn qpath(x: Stream, expr_style: bool) -> Res<(Option<QSelf>, Path)> {
    if x.peek(Token![<]) {
        let lt: Token![<] = x.parse()?;
        let this: typ::Type = x.parse()?;
        let y = if x.peek(Token![as]) {
            let as_: Token![as] = x.parse()?;
            let y: Path = x.parse()?;
            Some((as_, y))
        } else {
            None
        };
        let gt: Token![>] = x.parse()?;
        let colon2: Token![::] = x.parse()?;
        let mut rest = Puncted::new();
        loop {
            let path = Segment::parse_helper(x, expr_style)?;
            rest.push_value(path);
            if !x.peek(Token![::]) {
                break;
            }
            let punct: Token![::] = x.parse()?;
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
        let y = Path::parse_helper(x, expr_style)?;
        Ok((None, y))
    }
}
pub fn path_to_tokens(ys: &mut Stream, qself: &Option<QSelf>, path: &Path) {
    let qself = match qself {
        Some(x) => x,
        None => {
            path.lower(ys);
            return;
        },
    };
    qself.lt.lower(ys);
    qself.typ.lower(ys);
    let pos = cmp::min(qself.pos, path.segs.len());
    let mut segs = path.segs.pairs();
    if pos > 0 {
        ToksOrDefault(&qself.as_).lower(ys);
        path.colon.lower(ys);
        for (i, x) in segs.by_ref().take(pos).enumerate() {
            if i + 1 == pos {
                x.value().lower(ys);
                qself.gt.lower(ys);
                x.punct().lower(ys);
            } else {
                x.lower(ys);
            }
        }
    } else {
        qself.gt.lower(ys);
        path.colon.lower(ys);
    }
    for x in segs {
        x.lower(ys);
    }
}
