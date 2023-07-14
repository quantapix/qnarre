use super::*;
pub use expr::{Const, Lit, Macro as Mac, Path, Range};

ast_enum_of_structs! {
    pub enum Pat {
        Const(Const),
        Ident(Ident),
        Lit(Lit),
        Mac(Mac),
        Or(Or),
        Paren(Paren),
        Path(Path),
        Range(Range),
        Ref(Ref),
        Rest(Rest),
        Slice(Slice),
        Struct(Struct),
        Tuple(Tuple),
        TupleStruct(TupleStruct),
        Type(Type),
        Verbatim(Stream),
        Wild(Wild),
    }
}
pub struct Ident {
    pub attrs: Vec<attr::Attr>,
    pub ref_: Option<Token![ref]>,
    pub mut_: Option<Token![mut]>,
    pub ident: super::Ident,
    pub sub: Option<(Token![@], Box<Pat>)>,
}
pub struct Or {
    pub attrs: Vec<attr::Attr>,
    pub vert: Option<Token![|]>,
    pub cases: Puncted<Pat, Token![|]>,
}
pub struct Paren {
    pub attrs: Vec<attr::Attr>,
    pub paren: tok::Paren,
    pub pat: Box<Pat>,
}
pub struct Ref {
    pub attrs: Vec<attr::Attr>,
    pub and: Token![&],
    pub mut_: Option<Token![mut]>,
    pub pat: Box<Pat>,
}
pub struct Rest {
    pub attrs: Vec<attr::Attr>,
    pub dot2: Token![..],
}
pub struct Slice {
    pub attrs: Vec<attr::Attr>,
    pub bracket: tok::Bracket,
    pub elems: Puncted<Pat, Token![,]>,
}
pub struct Struct {
    pub attrs: Vec<attr::Attr>,
    pub qself: Option<QSelf>,
    pub path: Path,
    pub brace: tok::Brace,
    pub fields: Puncted<Field, Token![,]>,
    pub rest: Option<Rest>,
}
pub struct Tuple {
    pub attrs: Vec<attr::Attr>,
    pub paren: tok::Paren,
    pub elems: Puncted<Pat, Token![,]>,
}
pub struct TupleStruct {
    pub attrs: Vec<attr::Attr>,
    pub qself: Option<QSelf>,
    pub path: Path,
    pub paren: tok::Paren,
    pub elems: Puncted<Pat, Token![,]>,
}
pub struct Type {
    pub attrs: Vec<attr::Attr>,
    pub pat: Box<Pat>,
    pub colon: Token![:],
    pub typ: Box<typ::Type>,
}
pub struct Wild {
    pub attrs: Vec<attr::Attr>,
    pub underscore: Token![_],
}
pub struct Field {
    pub attrs: Vec<attr::Attr>,
    pub member: Member,
    pub colon: Option<Token![:]>,
    pub pat: Box<Pat>,
}

impl Pat {
    pub fn parse_single(x: Stream) -> Res<Self> {
        let begin = x.fork();
        let look = x.look1();
        if look.peek(Ident)
            && (x.peek2(Token![::])
                || x.peek2(Token![!])
                || x.peek2(tok::Brace)
                || x.peek2(tok::Paren)
                || x.peek2(Token![..]))
            || x.peek(Token![self]) && x.peek2(Token![::])
            || look.peek(Token![::])
            || look.peek(Token![<])
            || x.peek(Token![Self])
            || x.peek(Token![super])
            || x.peek(Token![crate])
        {
            path_or_mac_or_struct_or_range(x)
        } else if look.peek(Token![_]) {
            x.call(wild).map(Pat::Wild)
        } else if x.peek(Token![box]) {
            verbatim(begin, x)
        } else if x.peek(Token![-]) || look.peek(Lit) || look.peek(Token![const]) {
            lit_or_range(x)
        } else if look.peek(Token![ref]) || look.peek(Token![mut]) || x.peek(Token![self]) || x.peek(Ident) {
            x.call(ident).map(Pat::Ident)
        } else if look.peek(Token![&]) {
            x.call(ref_).map(Pat::Ref)
        } else if look.peek(tok::Paren) {
            x.call(paren_or_tuple)
        } else if look.peek(tok::Bracket) {
            x.call(slice).map(Pat::Slice)
        } else if look.peek(Token![..]) && !x.peek(Token![...]) {
            range_half_open(x)
        } else if look.peek(Token![const]) {
            x.call(const_).map(Pat::Verbatim)
        } else {
            Err(look.error())
        }
    }
    pub fn parse_multi(x: Stream) -> Res<Self> {
        multi_impl(x, None)
    }
    pub fn parse_with_vert(x: Stream) -> Res<Self> {
        let vert: Option<Token![|]> = x.parse()?;
        multi_impl(x, vert)
    }
}
fn multi_impl(x: Stream, vert: Option<Token![|]>) -> Res<Pat> {
    let mut y = Pat::parse_single(x)?;
    if vert.is_some() || x.peek(Token![|]) && !x.peek(Token![||]) && !x.peek(Token![|=]) {
        let mut cases = Puncted::new();
        cases.push_value(y);
        while x.peek(Token![|]) && !x.peek(Token![||]) && !x.peek(Token![|=]) {
            let punct = x.parse()?;
            cases.push_punct(punct);
            let pat = Pat::parse_single(x)?;
            cases.push_value(pat);
        }
        y = Pat::Or(Or {
            attrs: Vec::new(),
            vert,
            cases,
        });
    }
    Ok(y)
}
fn path_or_mac_or_struct_or_range(x: Stream) -> Res<Pat> {
    let (qself, path) = qpath(x, true)?;
    if qself.is_none() && x.peek(Token![!]) && !x.peek(Token![!=]) && path.is_mod_style() {
        let bang: Token![!] = x.parse()?;
        let (delimiter, tokens) = mac::parse_delim(x)?;
        return Ok(Pat::Macro(expr::Mac {
            attrs: Vec::new(),
            mac: Macro {
                path,
                bang,
                delimiter,
                tokens,
            },
        }));
    }
    if x.peek(tok::Brace) {
        struct_(x, qself, path).map(Pat::Struct)
    } else if x.peek(tok::Paren) {
        tuple_struct(x, qself, path).map(Pat::TupleStruct)
    } else if x.peek(Token![..]) {
        range(x, qself, path)
    } else {
        Ok(Pat::Path(expr::Path {
            attrs: Vec::new(),
            qself,
            path,
        }))
    }
}
fn wild(x: Stream) -> Res<Wild> {
    Ok(Wild {
        attrs: Vec::new(),
        underscore: x.parse()?,
    })
}
fn verbatim(beg: Buffer, x: Stream) -> Res<Pat> {
    x.parse::<Token![box]>()?;
    Pat::parse_single(x)?;
    Ok(Pat::Verbatim(parse::parse_verbatim(&beg, x)))
}
fn ident(x: Stream) -> Res<Ident> {
    Ok(Ident {
        attrs: Vec::new(),
        ref_: x.parse()?,
        mut_: x.parse()?,
        ident: x.call(Ident::parse_any)?,
        sub: {
            if x.peek(Token![@]) {
                let at_: Token![@] = x.parse()?;
                let sub = Pat::parse_single(x)?;
                Some((at_, Box::new(sub)))
            } else {
                None
            }
        },
    })
}
fn tuple_struct(x: Stream, qself: Option<QSelf>, path: Path) -> Res<TupleStruct> {
    let gist;
    let paren = parenthesized!(gist in x);
    let mut elems = Puncted::new();
    while !gist.is_empty() {
        let value = Pat::parse_multi(&gist)?;
        elems.push_value(value);
        if gist.is_empty() {
            break;
        }
        let punct = gist.parse()?;
        elems.push_punct(punct);
    }
    Ok(TupleStruct {
        attrs: Vec::new(),
        qself,
        path,
        paren,
        elems,
    })
}
fn struct_(x: Stream, qself: Option<QSelf>, path: Path) -> Res<Struct> {
    let gist;
    let brace = braced!(gist in x);
    let mut fields = Puncted::new();
    let mut rest = None;
    while !gist.is_empty() {
        let attrs = gist.call(attr::Attr::parse_outer)?;
        if gist.peek(Token![..]) {
            rest = Some(Rest {
                attrs,
                dot2: gist.parse()?,
            });
            break;
        }
        let mut y = gist.call(field)?;
        y.attrs = attrs;
        fields.push_value(y);
        if gist.is_empty() {
            break;
        }
        let y: Token![,] = gist.parse()?;
        fields.push_punct(y);
    }
    Ok(Struct {
        attrs: Vec::new(),
        qself,
        path,
        brace,
        fields,
        rest,
    })
}
impl Member {
    fn is_unnamed(&self) -> bool {
        match self {
            Member::Named(_) => false,
            Member::Unnamed(_) => true,
        }
    }
}
fn field(x: Stream) -> Res<Field> {
    let beg = x.fork();
    let box_: Option<Token![box]> = x.parse()?;
    let ref_: Option<Token![ref]> = x.parse()?;
    let mut_: Option<Token![mut]> = x.parse()?;
    let member = if box_.is_some() || ref_.is_some() || mut_.is_some() {
        x.parse().map(Member::Named)
    } else {
        x.parse()
    }?;
    if box_.is_none() && ref_.is_none() && mut_.is_none() && x.peek(Token![:]) || member.is_unnamed() {
        return Ok(Field {
            attrs: Vec::new(),
            member,
            colon: Some(x.parse()?),
            pat: Box::new(Pat::parse_multi(x)?),
        });
    }
    let ident = match member {
        Member::Named(ident) => ident,
        Member::Unnamed(_) => unreachable!(),
    };
    let pat = if box_.is_some() {
        Pat::Verbatim(parse::parse_verbatim(&beg, x))
    } else {
        Pat::Ident(Ident {
            attrs: Vec::new(),
            ref_,
            mut_,
            ident: ident.clone(),
            sub: None,
        })
    };
    Ok(Field {
        attrs: Vec::new(),
        member: Member::Named(ident),
        colon: None,
        pat: Box::new(pat),
    })
}
fn range(x: Stream, qself: Option<QSelf>, path: Path) -> Res<Pat> {
    let limits = RangeLimits::parse_obsolete(x)?;
    let end = x.call(range_bound)?;
    if let (RangeLimits::Closed(_), None) = (&limits, &end) {
        return Err(x.error("expected range upper bound"));
    }
    Ok(Pat::Range(expr::Range {
        attrs: Vec::new(),
        beg: Some(Box::new(Expr::Path(expr::Path {
            attrs: Vec::new(),
            qself,
            path,
        }))),
        limits,
        end: end.map(RangeBound::into_expr),
    }))
}
fn range_half_open(x: Stream) -> Res<Pat> {
    let limits: RangeLimits = x.parse()?;
    let end = x.call(range_bound)?;
    if end.is_some() {
        Ok(Pat::Range(expr::Range {
            attrs: Vec::new(),
            beg: None,
            limits,
            end: end.map(RangeBound::into_expr),
        }))
    } else {
        match limits {
            RangeLimits::HalfOpen(dot2) => Ok(Pat::Rest(Rest {
                attrs: Vec::new(),
                dot2,
            })),
            RangeLimits::Closed(_) => Err(x.error("expected range upper bound")),
        }
    }
}
fn paren_or_tuple(x: Stream) -> Res<Pat> {
    let gist;
    let paren = parenthesized!(gist in x);
    let mut elems = Puncted::new();
    while !gist.is_empty() {
        let x = Pat::parse_multi(&gist)?;
        if gist.is_empty() {
            if elems.is_empty() && !matches!(x, Pat::Rest(_)) {
                return Ok(Pat::Paren(Paren {
                    attrs: Vec::new(),
                    paren,
                    pat: Box::new(x),
                }));
            }
            elems.push_value(x);
            break;
        }
        elems.push_value(x);
        let punct = gist.parse()?;
        elems.push_punct(punct);
    }
    Ok(Pat::Tuple(Tuple {
        attrs: Vec::new(),
        paren,
        elems,
    }))
}
fn ref_(x: Stream) -> Res<Ref> {
    Ok(Ref {
        attrs: Vec::new(),
        and: x.parse()?,
        mut_: x.parse()?,
        pat: Box::new(Pat::parse_single(x)?),
    })
}
fn lit_or_range(x: Stream) -> Res<Pat> {
    let beg = x.call(range_bound)?.unwrap();
    if x.peek(Token![..]) {
        let limits = RangeLimits::parse_obsolete(x)?;
        let end = x.call(range_bound)?;
        if let (RangeLimits::Closed(_), None) = (&limits, &end) {
            return Err(x.error("expected range upper bound"));
        }
        Ok(Pat::Range(expr::Range {
            attrs: Vec::new(),
            beg: Some(beg.into_expr()),
            limits,
            end: end.map(RangeBound::into_expr),
        }))
    } else {
        Ok(beg.into_pat())
    }
}
enum RangeBound {
    Const(expr::Const),
    Lit(expr::Lit),
    Path(expr::Path),
}
impl RangeBound {
    fn into_expr(self) -> Box<Expr> {
        Box::new(match self {
            RangeBound::Const(x) => Expr::Const(x),
            RangeBound::Lit(x) => Expr::Lit(x),
            RangeBound::Path(x) => Expr::Path(x),
        })
    }
    fn into_pat(self) -> Pat {
        match self {
            RangeBound::Const(x) => Pat::Const(x),
            RangeBound::Lit(x) => Pat::Lit(x),
            RangeBound::Path(x) => Pat::Path(x),
        }
    }
}
fn range_bound(x: Stream) -> Res<Option<RangeBound>> {
    if x.is_empty()
        || x.peek(Token![|])
        || x.peek(Token![=])
        || x.peek(Token![:]) && !x.peek(Token![::])
        || x.peek(Token![,])
        || x.peek(Token![;])
        || x.peek(Token![if])
    {
        return Ok(None);
    }
    let look = x.look1();
    let y = if look.peek(Lit) {
        RangeBound::Lit(x.parse()?)
    } else if look.peek(Ident)
        || look.peek(Token![::])
        || look.peek(Token![<])
        || look.peek(Token![self])
        || look.peek(Token![Self])
        || look.peek(Token![super])
        || look.peek(Token![crate])
    {
        RangeBound::Path(x.parse()?)
    } else if look.peek(Token![const]) {
        RangeBound::Const(x.parse()?)
    } else {
        return Err(look.error());
    };
    Ok(Some(y))
}
fn slice(x: Stream) -> Res<Slice> {
    let gist;
    let bracket = bracketed!(gist in x);
    let mut elems = Puncted::new();
    while !gist.is_empty() {
        let y = Pat::parse_multi(&gist)?;
        match y {
            Pat::Range(x) if x.beg.is_none() || x.end.is_none() => {
                let (start, end) = match x.limits {
                    RangeLimits::HalfOpen(x) => (x.spans[0], x.spans[1]),
                    RangeLimits::Closed(x) => (x.spans[0], x.spans[2]),
                };
                let m = "range pattern is not allowed unparenthesized inside slice pattern";
                return Err(err::new2(start, end, m));
            },
            _ => {},
        }
        elems.push_value(y);
        if gist.is_empty() {
            break;
        }
        let y = gist.parse()?;
        elems.push_punct(y);
    }
    Ok(Slice {
        attrs: Vec::new(),
        bracket,
        elems,
    })
}
fn const_(x: Stream) -> Res<pm2::Stream> {
    let beg = x.fork();
    x.parse::<Token![const]>()?;
    let gist;
    braced!(gist in x);
    gist.call(attr::Attr::parse_inner)?;
    gist.call(Block::parse_within)?;
    Ok(parse::parse_verbatim(&beg, x))
}

impl ToTokens for Ident {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outer());
        self.ref_.to_tokens(ys);
        self.mut_.to_tokens(ys);
        self.ident.to_tokens(ys);
        if let Some((at_, sub)) = &self.subpat {
            at_.to_tokens(ys);
            sub.to_tokens(ys);
        }
    }
}
impl ToTokens for Or {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outer());
        self.vert.to_tokens(ys);
        self.cases.to_tokens(ys);
    }
}
impl ToTokens for Paren {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outer());
        self.paren.surround(ys, |ys| {
            self.pat.to_tokens(ys);
        });
    }
}
impl ToTokens for Ref {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outer());
        self.and.to_tokens(ys);
        self.mut_.to_tokens(ys);
        self.pat.to_tokens(ys);
    }
}
impl ToTokens for Rest {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outer());
        self.dot2.to_tokens(ys);
    }
}
impl ToTokens for Slice {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outer());
        self.bracket.surround(ys, |ys| {
            self.elems.to_tokens(ys);
        });
    }
}
impl ToTokens for Struct {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outer());
        path::lower_path(ys, &self.qself, &self.path);
        self.brace.surround(ys, |ys| {
            self.fields.to_tokens(ys);
            if !self.fields.empty_or_trailing() && self.rest.is_some() {
                <Token![,]>::default().to_tokens(ys);
            }
            self.rest.to_tokens(ys);
        });
    }
}
impl ToTokens for Tuple {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outer());
        self.paren.surround(ys, |ys| {
            self.elems.to_tokens(ys);
        });
    }
}
impl ToTokens for TupleStruct {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outer());
        path::lower_path(ys, &self.qself, &self.path);
        self.paren.surround(ys, |ys| {
            self.elems.to_tokens(ys);
        });
    }
}
impl ToTokens for Type {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outer());
        self.pat.to_tokens(ys);
        self.colon.to_tokens(ys);
        self.typ.to_tokens(ys);
    }
}
impl ToTokens for Wild {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outer());
        self.underscore.to_tokens(ys);
    }
}
impl ToTokens for Field {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outer());
        if let Some(x) = &self.colon {
            self.member.to_tokens(ys);
            x.to_tokens(ys);
        }
        self.pat.to_tokens(ys);
    }
}
