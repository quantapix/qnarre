use super::*;

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

pub struct Segment {
    pub ident: Ident,
    pub args: Args,
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

pub enum Args {
    None,
    Angled(AngledArgs),
    Parenthesized(ParenthesizedArgs),
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

pub enum Arg {
    Life(Life),
    Type(typ::Type),
    Const(Expr),
    AssocType(AssocType),
    AssocConst(AssocConst),
    Constraint(Constraint),
}

pub use expr::Expr;

pub struct AngledArgs {
    pub colon2: Option<Token![::]>,
    pub lt: Token![<],
    pub args: Puncted<Arg, Token![,]>,
    pub gt: Token![>],
}
pub struct AssocType {
    pub ident: Ident,
    pub args: Option<AngledArgs>,
    pub eq: Token![=],
    pub typ: typ::Type,
}
pub struct AssocConst {
    pub ident: Ident,
    pub args: Option<AngledArgs>,
    pub eq: Token![=],
    pub val: Expr,
}
pub struct Constraint {
    pub ident: Ident,
    pub args: Option<AngledArgs>,
    pub colon: Token![:],
    pub bounds: Puncted<gen::bound::Type, Token![+]>,
}
pub struct ParenthesizedArgs {
    pub paren: tok::Paren,
    pub args: Puncted<typ::Type, Token![,]>,
    pub ret: typ::Ret,
}
pub struct QSelf {
    pub lt: Token![<],
    pub typ: Box<typ::Type>,
    pub pos: usize,
    pub as_: Option<Token![as]>,
    pub gt: Token![>],
}

impl Path {
    pub fn parse_mod_style(s: Stream) -> Res<Self> {
        Ok(Path {
            colon: s.parse()?,
            segs: {
                let mut ys = Puncted::new();
                loop {
                    if !s.peek(Ident)
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
}
impl Parse for Path {
    fn parse(x: Stream) -> Res<Self> {
        Self::parse_helper(x, false)
    }
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
impl Parse for Segment {
    fn parse(s: Stream) -> Res<Self> {
        Self::parse_helper(s, false)
    }
}
impl AngledArgs {
    pub fn parse_turbofish(s: Stream) -> Res<Self> {
        let y: Token![::] = s.parse()?;
        Self::do_parse(Some(y), s)
    }
    fn do_parse(colon2: Option<Token![::]>, s: Stream) -> Res<Self> {
        Ok(AngledArgs {
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
impl Parse for AngledArgs {
    fn parse(s: Stream) -> Res<Self> {
        let y: Option<Token![::]> = s.parse()?;
        Self::do_parse(y, s)
    }
}
impl Parse for ParenthesizedArgs {
    fn parse(s: Stream) -> Res<Self> {
        let y;
        Ok(ParenthesizedArgs {
            paren: parenthesized!(y in s),
            args: y.parse_terminated(typ::Type::parse, Token![,])?,
            ret: s.call(typ::Ret::without_plus)?,
        })
    }
}
impl Parse for Arg {
    fn parse(x: Stream) -> Res<Self> {
        if x.peek(Life) && !x.peek2(Token![+]) {
            return Ok(Arg::Life(x.parse()?));
        }
        if x.peek(Lit) || x.peek(tok::Brace) {
            return const_argument(x).map(Arg::Const);
        }
        let mut y: typ::Type = x.parse()?;
        match y {
            typ::Type::Path(mut ty)
                if ty.qself.is_none()
                    && ty.path.colon.is_none()
                    && ty.path.segs.len() == 1
                    && match &ty.path.segments[0].arguments {
                        Args::None | Args::AngleBracketed(_) => true,
                        Args::Parenthesized(_) => false,
                    } =>
            {
                if let Some(eq) = x.parse::<Option<Token![=]>>()? {
                    let seg = ty.path.segs.pop().unwrap().into_value();
                    let ident = seg.ident;
                    let gnrs = match seg.args {
                        Args::None => None,
                        Args::Angled(x) => Some(x),
                        Args::Parenthesized(_) => unreachable!(),
                    };
                    return if x.peek(Lit) || x.peek(tok::Brace) {
                        Ok(Arg::AssocConst(AssocConst {
                            ident,
                            args: gnrs,
                            eq,
                            val: const_argument(x)?,
                        }))
                    } else {
                        Ok(Arg::AssocType(AssocType {
                            ident,
                            args: gnrs,
                            eq,
                            typ: x.parse()?,
                        }))
                    };
                }
                if let Some(colon) = x.parse::<Option<Token![:]>>()? {
                    let seg = ty.path.segs.pop().unwrap().into_value();
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
                                if x.peek(Token![,]) || x.peek(Token![>]) {
                                    break;
                                }
                                let y: gen::bound::Type = x.parse()?;
                                ys.push_value(y);
                                if !x.peek(Token![+]) {
                                    break;
                                }
                                let y: Token![+] = x.parse()?;
                                ys.push_punct(y);
                            }
                            ys
                        },
                    }));
                }
                y = typ::Type::Path(ty);
            },
            _ => {},
        }
        Ok(Arg::Type(y))
    }
}
pub fn const_argument(x: Stream) -> Res<Expr> {
    let y = x.look1();
    if x.peek(Lit) {
        let y = x.parse()?;
        return Ok(Expr::Lit(y));
    }
    if x.peek(Ident) {
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

impl ToStream for Path {
    fn to_tokens(&self, ys: &mut Stream) {
        self.colon.to_tokens(ys);
        self.segs.to_tokens(ys);
    }
}
impl ToStream for Segment {
    fn to_tokens(&self, ys: &mut Stream) {
        self.ident.to_tokens(ys);
        self.args.to_tokens(ys);
    }
}
impl ToStream for Args {
    fn to_tokens(&self, ys: &mut Stream) {
        match self {
            Args::None => {},
            Args::Angled(x) => {
                x.to_tokens(ys);
            },
            Args::Parenthesized(x) => {
                x.to_tokens(ys);
            },
        }
    }
}
impl ToStream for Arg {
    #[allow(clippy::match_same_arms)]
    fn to_tokens(&self, ys: &mut Stream) {
        use Arg::*;
        match self {
            Life(x) => x.to_tokens(ys),
            Type(x) => x.to_tokens(ys),
            Const(x) => match x {
                Expr::Lit(_) => x.to_tokens(ys),
                Expr::Block(_) => x.to_tokens(ys),
                _ => tok::Brace::default().surround(ys, |ys| {
                    x.to_tokens(ys);
                }),
            },
            AssocType(x) => x.to_tokens(ys),
            AssocConst(x) => x.to_tokens(ys),
            Constraint(x) => x.to_tokens(ys),
        }
    }
}
impl ToStream for AngledArgs {
    fn to_tokens(&self, ys: &mut Stream) {
        self.colon2.to_tokens(ys);
        self.lt.to_tokens(ys);
        let mut trailing_or_empty = true;
        for x in self.args.pairs() {
            match x.value() {
                Arg::Life(_) => {
                    x.to_tokens(ys);
                    trailing_or_empty = x.punct().is_some();
                },
                Arg::Type(_) | Arg::Const(_) | Arg::AssocType(_) | Arg::AssocConst(_) | Arg::Constraint(_) => {},
            }
        }
        for x in self.args.pairs() {
            match x.value() {
                Arg::Type(_) | Arg::Const(_) | Arg::AssocType(_) | Arg::AssocConst(_) | Arg::Constraint(_) => {
                    if !trailing_or_empty {
                        <Token![,]>::default().to_tokens(ys);
                    }
                    x.to_tokens(ys);
                    trailing_or_empty = x.punct().is_some();
                },
                Arg::Life(_) => {},
            }
        }
        self.gt.to_tokens(ys);
    }
}
impl ToStream for AssocType {
    fn to_tokens(&self, ys: &mut Stream) {
        self.ident.to_tokens(ys);
        self.args.to_tokens(ys);
        self.eq.to_tokens(ys);
        self.typ.to_tokens(ys);
    }
}
impl ToStream for AssocConst {
    fn to_tokens(&self, ys: &mut Stream) {
        self.ident.to_tokens(ys);
        self.args.to_tokens(ys);
        self.eq.to_tokens(ys);
        self.val.to_tokens(ys);
    }
}
impl ToStream for Constraint {
    fn to_tokens(&self, ys: &mut Stream) {
        self.ident.to_tokens(ys);
        self.args.to_tokens(ys);
        self.colon.to_tokens(ys);
        self.bounds.to_tokens(ys);
    }
}
impl ToStream for ParenthesizedArgs {
    fn to_tokens(&self, ys: &mut Stream) {
        self.paren.surround(ys, |ys| {
            self.args.to_tokens(ys);
        });
        self.ret.to_tokens(ys);
    }
}
pub fn path_to_tokens(ys: &mut Stream, qself: &Option<QSelf>, path: &Path) {
    let qself = match qself {
        Some(x) => x,
        None => {
            path.to_tokens(ys);
            return;
        },
    };
    qself.lt.to_tokens(ys);
    qself.typ.to_tokens(ys);
    let pos = cmp::min(qself.pos, path.segs.len());
    let mut segs = path.segs.pairs();
    if pos > 0 {
        TokensOrDefault(&qself.as_).to_tokens(ys);
        path.colon.to_tokens(ys);
        for (i, x) in segs.by_ref().take(pos).enumerate() {
            if i + 1 == pos {
                x.value().to_tokens(ys);
                qself.gt.to_tokens(ys);
                x.punct().to_tokens(ys);
            } else {
                x.to_tokens(ys);
            }
        }
    } else {
        qself.gt.to_tokens(ys);
        path.colon.to_tokens(ys);
    }
    for x in segs {
        x.to_tokens(ys);
    }
}
