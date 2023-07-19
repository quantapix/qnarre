use super::*;
pub use expr::{Const, Lit, Mac, Member, Path, Range};

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
        Wild(Wild),
        Stream(pm2::Stream),
    }
}
impl Pat {
    pub fn parse_one(s: Stream) -> Res<Self> {
        let beg = s.fork();
        let look = s.look1();
        if look.peek(ident::Ident)
            && (s.peek2(Token![::])
                || s.peek2(Token![!])
                || s.peek2(tok::Brace)
                || s.peek2(tok::Paren)
                || s.peek2(Token![..]))
            || s.peek(Token![self]) && s.peek2(Token![::])
            || look.peek(Token![::])
            || look.peek(Token![<])
            || s.peek(Token![Self])
            || s.peek(Token![super])
            || s.peek(Token![crate])
        {
            parse_path_or_mac_or_struct_or_range(s)
        } else if look.peek(Token![_]) {
            s.call(parse_wild).map(Pat::Wild)
        } else if s.peek(Token![box]) {
            parse_verbatim(beg, s)
        } else if s.peek(Token![-]) || look.peek(lit::Lit) || look.peek(Token![const]) {
            parse_lit_or_range(s)
        } else if look.peek(Token![ref]) || look.peek(Token![mut]) || s.peek(Token![self]) || s.peek(ident::Ident) {
            s.call(parse_ident).map(Pat::Ident)
        } else if look.peek(Token![&]) {
            s.call(parse_ref).map(Pat::Ref)
        } else if look.peek(tok::Paren) {
            s.call(parse_paren_or_tuple)
        } else if look.peek(tok::Bracket) {
            s.call(parse_slice).map(Pat::Slice)
        } else if look.peek(Token![..]) && !s.peek(Token![...]) {
            parse_range_half_open(s)
        } else if look.peek(Token![const]) {
            s.call(parse_const).map(Pat::Stream)
        } else {
            Err(look.error())
        }
    }
    pub fn parse_many(s: Stream) -> Res<Self> {
        parse_many(s, None)
    }
    pub fn parse_with_vert(s: Stream) -> Res<Self> {
        let vert: Option<Token![|]> = s.parse()?;
        parse_many(s, vert)
    }
}

pub struct Ident {
    pub attrs: Vec<attr::Attr>,
    pub ref_: Option<Token![ref]>,
    pub mut_: Option<Token![mut]>,
    pub ident: super::Ident,
    pub sub: Option<(Token![@], Box<Pat>)>,
}
impl ToStream for Ident {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.ref_.to_tokens(ys);
        self.mut_.to_tokens(ys);
        self.ident.to_tokens(ys);
        if let Some((at_, sub)) = &self.sub {
            at_.to_tokens(ys);
            sub.to_tokens(ys);
        }
    }
}

pub struct Or {
    pub attrs: Vec<attr::Attr>,
    pub vert: Option<Token![|]>,
    pub cases: Puncted<Pat, Token![|]>,
}
impl ToStream for Or {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.vert.to_tokens(ys);
        self.cases.to_tokens(ys);
    }
}

pub struct Paren {
    pub attrs: Vec<attr::Attr>,
    pub paren: tok::Paren,
    pub pat: Box<Pat>,
}
impl ToStream for Paren {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.paren.surround(ys, |ys| {
            self.pat.to_tokens(ys);
        });
    }
}

pub struct Ref {
    pub attrs: Vec<attr::Attr>,
    pub and: Token![&],
    pub mut_: Option<Token![mut]>,
    pub pat: Box<Pat>,
}
impl ToStream for Ref {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.and.to_tokens(ys);
        self.mut_.to_tokens(ys);
        self.pat.to_tokens(ys);
    }
}

pub struct Rest {
    pub attrs: Vec<attr::Attr>,
    pub dot2: Token![..],
}
impl ToStream for Rest {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.dot2.to_tokens(ys);
    }
}

pub struct Slice {
    pub attrs: Vec<attr::Attr>,
    pub bracket: tok::Bracket,
    pub elems: Puncted<Pat, Token![,]>,
}
impl ToStream for Slice {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.bracket.surround(ys, |ys| {
            self.elems.to_tokens(ys);
        });
    }
}

pub struct Struct {
    pub attrs: Vec<attr::Attr>,
    pub qself: Option<path::QSelf>,
    pub path: Path,
    pub brace: tok::Brace,
    pub fields: Puncted<Field, Token![,]>,
    pub rest: Option<Rest>,
}
impl ToStream for Struct {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        path::path_to_tokens(ys, &self.qself, &self.path);
        self.brace.surround(ys, |ys| {
            self.fields.to_tokens(ys);
            if !self.fields.empty_or_trailing() && self.rest.is_some() {
                <Token![,]>::default().to_tokens(ys);
            }
            self.rest.to_tokens(ys);
        });
    }
}

pub struct Field {
    pub attrs: Vec<attr::Attr>,
    pub memb: Member,
    pub colon: Option<Token![:]>,
    pub pat: Box<Pat>,
}
impl ToStream for Field {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        if let Some(x) = &self.colon {
            self.memb.to_tokens(ys);
            x.to_tokens(ys);
        }
        self.pat.to_tokens(ys);
    }
}

impl Member {
    fn is_unnamed(&self) -> bool {
        match self {
            Member::Named(_) => false,
            Member::Unnamed(_) => true,
        }
    }
}

pub struct Tuple {
    pub attrs: Vec<attr::Attr>,
    pub paren: tok::Paren,
    pub elems: Puncted<Pat, Token![,]>,
}
impl ToStream for Tuple {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.paren.surround(ys, |ys| {
            self.elems.to_tokens(ys);
        });
    }
}

pub struct TupleStruct {
    pub attrs: Vec<attr::Attr>,
    pub qself: Option<path::QSelf>,
    pub path: Path,
    pub paren: tok::Paren,
    pub elems: Puncted<Pat, Token![,]>,
}
impl ToStream for TupleStruct {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        path::path_to_tokens(ys, &self.qself, &self.path);
        self.paren.surround(ys, |ys| {
            self.elems.to_tokens(ys);
        });
    }
}

pub struct Type {
    pub attrs: Vec<attr::Attr>,
    pub pat: Box<Pat>,
    pub colon: Token![:],
    pub typ: Box<typ::Type>,
}
impl ToStream for Type {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.pat.to_tokens(ys);
        self.colon.to_tokens(ys);
        self.typ.to_tokens(ys);
    }
}

pub struct Wild {
    pub attrs: Vec<attr::Attr>,
    pub underscore: Token![_],
}
impl ToStream for Wild {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(self.attrs.outers());
        self.underscore.to_tokens(ys);
    }
}

enum RangeBound {
    Const(Const),
    Lit(Lit),
    Path(Path),
}
impl RangeBound {
    fn into_expr(self) -> Box<expr::Expr> {
        Box::new(match self {
            RangeBound::Const(x) => expr::Expr::Const(x),
            RangeBound::Lit(x) => expr::Expr::Lit(x),
            RangeBound::Path(x) => expr::Expr::Path(x),
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

fn parse_const(s: Stream) -> Res<pm2::Stream> {
    let beg = s.fork();
    s.parse::<Token![const]>()?;
    let y;
    braced!(y in s);
    y.call(attr::Attr::parse_inners)?;
    y.call(stmt::Block::parse_within)?;
    Ok(parse::parse_verbatim(&beg, s))
}
fn parse_field(s: Stream) -> Res<Field> {
    let beg = s.fork();
    let box_: Option<Token![box]> = s.parse()?;
    let ref_: Option<Token![ref]> = s.parse()?;
    let mut_: Option<Token![mut]> = s.parse()?;
    let memb = if box_.is_some() || ref_.is_some() || mut_.is_some() {
        s.parse().map(Member::Named)
    } else {
        s.parse()
    }?;
    if box_.is_none() && ref_.is_none() && mut_.is_none() && s.peek(Token![:]) || memb.is_unnamed() {
        return Ok(Field {
            attrs: Vec::new(),
            memb,
            colon: Some(s.parse()?),
            pat: Box::new(Pat::parse_many(s)?),
        });
    }
    let ident = match memb {
        Member::Named(x) => x,
        Member::Unnamed(_) => unreachable!(),
    };
    let pat = if box_.is_some() {
        Pat::Stream(parse::parse_verbatim(&beg, s))
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
        memb: Member::Named(ident),
        colon: None,
        pat: Box::new(pat),
    })
}
fn parse_ident(s: Stream) -> Res<Ident> {
    Ok(Ident {
        attrs: Vec::new(),
        ref_: s.parse()?,
        mut_: s.parse()?,
        ident: s.call(Ident::parse_any)?,
        sub: {
            if s.peek(Token![@]) {
                let at_: Token![@] = s.parse()?;
                let y = Pat::parse_one(s)?;
                Some((at_, Box::new(y)))
            } else {
                None
            }
        },
    })
}
fn parse_lit_or_range(s: Stream) -> Res<Pat> {
    let beg = s.call(parse_range_bound)?.unwrap();
    if s.peek(Token![..]) {
        let limits = expr::Limits::parse_obsolete(s)?;
        let end = s.call(parse_range_bound)?;
        if let (expr::Limits::Closed(_), None) = (&limits, &end) {
            return Err(s.error("expected range upper bound"));
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
fn parse_many(s: Stream, vert: Option<Token![|]>) -> Res<Pat> {
    let mut y = Pat::parse_one(s)?;
    if vert.is_some() || s.peek(Token![|]) && !s.peek(Token![||]) && !s.peek(Token![|=]) {
        let mut cases = Puncted::new();
        cases.push_value(y);
        while s.peek(Token![|]) && !s.peek(Token![||]) && !s.peek(Token![|=]) {
            let x = s.parse()?;
            cases.push_punct(x);
            let x = Pat::parse_one(s)?;
            cases.push_value(x);
        }
        y = Pat::Or(Or {
            attrs: Vec::new(),
            vert,
            cases,
        });
    }
    Ok(y)
}
fn parse_path_or_mac_or_struct_or_range(s: Stream) -> Res<Pat> {
    let (qself, path) = path::qpath(s, true)?;
    if qself.is_none() && s.peek(Token![!]) && !s.peek(Token![!=]) && path.is_mod_style() {
        let bang: Token![!] = s.parse()?;
        let (delim, toks) = mac::parse_delim(s)?;
        return Ok(Pat::Mac(expr::Mac {
            attrs: Vec::new(),
            mac: mac::Mac {
                path,
                bang,
                delim,
                toks,
            },
        }));
    }
    if s.peek(tok::Brace) {
        parse_struct(s, qself, path).map(Pat::Struct)
    } else if s.peek(tok::Paren) {
        parse_tuple_struct(s, qself, path).map(Pat::TupleStruct)
    } else if s.peek(Token![..]) {
        parse_range(s, qself, path)
    } else {
        Ok(Pat::Path(expr::Path {
            attrs: Vec::new(),
            qself,
            path,
        }))
    }
}
fn parse_paren_or_tuple(s: Stream) -> Res<Pat> {
    let y;
    let paren = parenthesized!(y in s);
    let mut elems = Puncted::new();
    while !y.is_empty() {
        let x = Pat::parse_many(&y)?;
        if y.is_empty() {
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
        let x = y.parse()?;
        elems.push_punct(x);
    }
    Ok(Pat::Tuple(Tuple {
        attrs: Vec::new(),
        paren,
        elems,
    }))
}
fn parse_range(s: Stream, qself: Option<path::QSelf>, path: Path) -> Res<Pat> {
    let limits = expr::Limits::parse_obsolete(s)?;
    let end = s.call(parse_range_bound)?;
    if let (expr::Limits::Closed(_), None) = (&limits, &end) {
        return Err(s.error("expected range upper bound"));
    }
    Ok(Pat::Range(expr::Range {
        attrs: Vec::new(),
        beg: Some(Box::new(expr::Expr::Path(expr::Path {
            attrs: Vec::new(),
            qself,
            path,
        }))),
        limits,
        end: end.map(RangeBound::into_expr),
    }))
}
fn parse_range_bound(s: Stream) -> Res<Option<RangeBound>> {
    if s.is_empty()
        || s.peek(Token![|])
        || s.peek(Token![=])
        || s.peek(Token![:]) && !s.peek(Token![::])
        || s.peek(Token![,])
        || s.peek(Token![;])
        || s.peek(Token![if])
    {
        return Ok(None);
    }
    let look = s.look1();
    let y = if look.peek(lit::Lit) {
        RangeBound::Lit(s.parse()?)
    } else if look.peek(ident::Ident)
        || look.peek(Token![::])
        || look.peek(Token![<])
        || look.peek(Token![self])
        || look.peek(Token![Self])
        || look.peek(Token![super])
        || look.peek(Token![crate])
    {
        RangeBound::Path(s.parse()?)
    } else if look.peek(Token![const]) {
        RangeBound::Const(s.parse()?)
    } else {
        return Err(look.error());
    };
    Ok(Some(y))
}
fn parse_range_half_open(s: Stream) -> Res<Pat> {
    let limits: expr::Limits = s.parse()?;
    let end = s.call(parse_range_bound)?;
    if end.is_some() {
        Ok(Pat::Range(expr::Range {
            attrs: Vec::new(),
            beg: None,
            limits,
            end: end.map(RangeBound::into_expr),
        }))
    } else {
        match limits {
            expr::Limits::HalfOpen(dot2) => Ok(Pat::Rest(Rest {
                attrs: Vec::new(),
                dot2,
            })),
            expr::Limits::Closed(_) => Err(s.error("expected range upper bound")),
        }
    }
}
fn parse_ref(s: Stream) -> Res<Ref> {
    Ok(Ref {
        attrs: Vec::new(),
        and: s.parse()?,
        mut_: s.parse()?,
        pat: Box::new(Pat::parse_one(s)?),
    })
}
fn parse_slice(s: Stream) -> Res<Slice> {
    let y;
    let bracket = bracketed!(y in s);
    let mut elems = Puncted::new();
    while !y.is_empty() {
        let y = Pat::parse_many(&y)?;
        match y {
            Pat::Range(x) if x.beg.is_none() || x.end.is_none() => {
                let (start, end) = match x.limits {
                    expr::Limits::HalfOpen(x) => (x.spans[0], x.spans[1]),
                    expr::Limits::Closed(x) => (x.spans[0], x.spans[2]),
                };
                let m = "range pattern is not allowed unparenthesized inside slice pattern";
                return Err(err::new2(start, end, m));
            },
            _ => {},
        }
        elems.push_value(y);
        if y.is_empty() {
            break;
        }
        let y = y.parse()?;
        elems.push_punct(y);
    }
    Ok(Slice {
        attrs: Vec::new(),
        bracket,
        elems,
    })
}
fn parse_struct(s: Stream, qself: Option<path::QSelf>, path: Path) -> Res<Struct> {
    let y;
    let brace = braced!(y in s);
    let mut fields = Puncted::new();
    let mut rest = None;
    while !y.is_empty() {
        let attrs = y.call(attr::Attr::parse_outers)?;
        if y.peek(Token![..]) {
            rest = Some(Rest {
                attrs,
                dot2: y.parse()?,
            });
            break;
        }
        let mut y = y.call(parse_field)?;
        y.attrs = attrs;
        fields.push_value(y);
        if y.is_empty() {
            break;
        }
        let y: Token![,] = y.parse()?;
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
fn parse_tuple_struct(s: Stream, qself: Option<path::QSelf>, path: Path) -> Res<TupleStruct> {
    let y;
    let paren = parenthesized!(y in s);
    let mut elems = Puncted::new();
    while !y.is_empty() {
        let x = Pat::parse_many(&y)?;
        elems.push_value(x);
        if y.is_empty() {
            break;
        }
        let x = y.parse()?;
        elems.push_punct(x);
    }
    Ok(TupleStruct {
        attrs: Vec::new(),
        qself,
        path,
        paren,
        elems,
    })
}
fn parse_verbatim(beg: parse::Buffer, s: Stream) -> Res<Pat> {
    s.parse::<Token![box]>()?;
    Pat::parse_one(s)?;
    Ok(Pat::Stream(parse::parse_verbatim(&beg, s)))
}
fn parse_wild(s: Stream) -> Res<Wild> {
    Ok(Wild {
        attrs: Vec::new(),
        underscore: s.parse()?,
    })
}
