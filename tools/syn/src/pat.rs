use super::*;
pub use expr::{Const, Lit, Mac, Member, Path, Range};

ast_enum_of_structs! {
    pub enum Pat {
        Const(Const),
        Ident(Ident),
        Lit(Lit),
        Mac(Mac),
        Or(Or),
        Parenth(Parenth),
        Path(Path),
        Range(Range),
        Ref(Ref),
        Rest(Rest),
        Slice(Slice),
        Struct(Struct),
        Tuple(Tuple),
        TupleStruct(TupleStruct),
        Type(Type),
        Verbatim(Verbatim),
        Wild(Wild),
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
                || s.peek2(tok::Parenth)
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
        } else if look.peek(tok::Parenth) {
            s.call(parse_paren_or_tuple)
        } else if look.peek(tok::Bracket) {
            s.call(parse_slice).map(Pat::Slice)
        } else if look.peek(Token![..]) && !s.peek(Token![...]) {
            parse_range_half_open(s)
        } else if look.peek(Token![const]) {
            s.call(parse_const).map(Pat::Verbatim)
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
impl Pretty for Pat {
    fn pretty(&self, p: &mut Print) {
        use pat::Pat::*;
        match self {
            Const(x) => x.pretty(p),
            Ident(x) => x.pretty(p),
            Lit(x) => x.pretty(p),
            Mac(x) => x.pretty(p),
            Or(x) => x.pretty(p),
            Parenth(x) => x.pretty(p),
            Path(x) => x.pretty(p),
            Range(x) => p.expr_range(x),
            Ref(x) => x.pretty(p),
            Rest(x) => x.pretty(p),
            Slice(x) => x.pretty(p),
            Struct(x) => x.pretty(p),
            Tuple(x) => x.pretty(p),
            TupleStruct(x) => x.pretty(p),
            Type(x) => x.pretty(p),
            Verbatim(x) => x.pretty(p),
            Wild(x) => x.pretty(p),
        }
    }
}

pub struct Ident {
    pub attrs: Vec<attr::Attr>,
    pub ref_: Option<Token![ref]>,
    pub mut_: Option<Token![mut]>,
    pub ident: super::Ident,
    pub sub: Option<(Token![@], Box<Pat>)>,
}
impl Lower for Ident {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.ref_.lower(s);
        self.mut_.lower(s);
        self.ident.lower(s);
        if let Some((at_, sub)) = &self.sub {
            at_.lower(s);
            sub.lower(s);
        }
    }
}
impl Pretty for Ident {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        if self.ref_.is_some() {
            p.word("ref ");
        }
        if self.mut_.is_some() {
            p.word("mut ");
        }
        &self.ident.pretty(p);
        if let Some((_, x)) = &self.sub {
            p.word(" @ ");
            x.pretty(p);
        }
    }
}

pub struct Or {
    pub attrs: Vec<attr::Attr>,
    pub vert: Option<Token![|]>,
    pub cases: Puncted<Pat, Token![|]>,
}
impl Lower for Or {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.vert.lower(s);
        self.cases.lower(s);
    }
}
impl Pretty for Or {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        let mut break_ = false;
        for x in &self.cases {
            match x {
                Pat::Lit(_) | Pat::Wild(_) => {},
                _ => {
                    break_ = true;
                    break;
                },
            }
        }
        if break_ {
            p.cbox(0);
        } else {
            p.ibox(0);
        }
        for x in self.cases.iter().delimited() {
            if !x.is_first {
                p.space();
                p.word("| ");
            }
            &x.pretty(p);
        }
        p.end();
    }
}

pub struct Parenth {
    pub attrs: Vec<attr::Attr>,
    pub parenth: tok::Parenth,
    pub pat: Box<Pat>,
}
impl Lower for Parenth {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.parenth.surround(s, |s| {
            self.pat.lower(s);
        });
    }
}
impl Pretty for Parenth {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("(");
        &self.pat.pretty(p);
        p.word(")");
    }
}

pub struct Ref {
    pub attrs: Vec<attr::Attr>,
    pub and: Token![&],
    pub mut_: Option<Token![mut]>,
    pub pat: Box<Pat>,
}
impl Lower for Ref {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.and.lower(s);
        self.mut_.lower(s);
        self.pat.lower(s);
    }
}
impl Pretty for Ref {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("&");
        if self.mut_.is_some() {
            p.word("mut ");
        }
        &self.pat.pretty(p);
    }
}

pub struct Rest {
    pub attrs: Vec<attr::Attr>,
    pub dot2: Token![..],
}
impl Lower for Rest {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.dot2.lower(s);
    }
}
impl Pretty for Rest {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("..");
    }
}

pub struct Slice {
    pub attrs: Vec<attr::Attr>,
    pub bracket: tok::Bracket,
    pub pats: Puncted<Pat, Token![,]>,
}
impl Lower for Slice {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.bracket.surround(s, |s| {
            self.pats.lower(s);
        });
    }
}
impl Pretty for Slice {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("[");
        for x in self.pats.iter().delimited() {
            &x.pretty(p);
            p.trailing_comma(x.is_last);
        }
        p.word("]");
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
impl Lower for Struct {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        path::path_lower(s, &self.qself, &self.path);
        self.brace.surround(s, |s| {
            self.fields.lower(s);
            if !self.fields.empty_or_trailing() && self.rest.is_some() {
                <Token![,]>::default().lower(s);
            }
            self.rest.lower(s);
        });
    }
}
impl Pretty for Struct {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        &self.path.pretty_with_args(p, path::Kind::Expr);
        p.word(" {");
        p.space_if_nonempty();
        for x in self.fields.iter().delimited() {
            &x.pretty(p);
            p.trailing_comma_or_space(x.is_last && self.rest.is_none());
        }
        if let Some(x) = &self.rest {
            x.pretty(p);
            p.space();
        }
        p.offset(-INDENT);
        p.end();
        p.word("}");
    }
}

pub struct Field {
    pub attrs: Vec<attr::Attr>,
    pub memb: Member,
    pub colon: Option<Token![:]>,
    pub pat: Box<Pat>,
}
impl Lower for Field {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        if let Some(x) = &self.colon {
            self.memb.lower(s);
            x.lower(s);
        }
        self.pat.lower(s);
    }
}
impl Pretty for Field {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        if self.colon.is_some() {
            &self.memb.pretty(p);
            p.word(": ");
        }
        &self.pat.pretty(p);
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
    pub parenth: tok::Parenth,
    pub pats: Puncted<Pat, Token![,]>,
}
impl Lower for Tuple {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.parenth.surround(s, |s| {
            self.pats.lower(s);
        });
    }
}
impl Pretty for Tuple {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("(");
        p.cbox(INDENT);
        p.zerobreak();
        for x in self.pats.iter().delimited() {
            &x.pretty(p);
            if self.pats.len() == 1 {
                if self.pats.trailing_punct() {
                    p.word(",");
                }
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

pub struct TupleStruct {
    pub attrs: Vec<attr::Attr>,
    pub qself: Option<path::QSelf>,
    pub path: Path,
    pub parenth: tok::Parenth,
    pub pats: Puncted<Pat, Token![,]>,
}
impl Lower for TupleStruct {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        path::path_lower(s, &self.qself, &self.path);
        self.parenth.surround(s, |s| {
            self.pats.lower(s);
        });
    }
}
impl Pretty for TupleStruct {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        &self.path.pretty_with_args(p, path::Kind::Expr);
        p.word("(");
        p.cbox(INDENT);
        p.zerobreak();
        for x in self.pats.iter().delimited() {
            &x.pretty(p);
            p.trailing_comma(x.is_last);
        }
        p.offset(-INDENT);
        p.end();
        p.word(")");
    }
}

pub struct Type {
    pub attrs: Vec<attr::Attr>,
    pub pat: Box<Pat>,
    pub colon: Token![:],
    pub typ: Box<typ::Type>,
}
impl Lower for Type {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.pat.lower(s);
        self.colon.lower(s);
        self.typ.lower(s);
    }
}
impl Pretty for Type {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        &self.pat.pretty(p);
        p.word(": ");
        &self.typ.pretty(p);
    }
}

pub struct Wild {
    pub attrs: Vec<attr::Attr>,
    pub underscore: Token![_],
}
impl Lower for Wild {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.underscore.lower(s);
    }
}
impl Pretty for Wild {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("_");
    }
}

pub struct Verbatim(pub pm2::Stream);
impl Pretty for Verbatim {
    fn pretty(&self, p: &mut Print) {
        enum Type {
            Box(Pat),
            Const(Const),
            Ellipsis,
        }
        use Type::*;
        struct Const {
            attrs: Vec<attr::Attr>,
            block: stmt::Block,
        }
        impl parse::Parse for Type {
            fn parse(s: parse::Stream) -> Res<Self> {
                let look = s.lookahead1();
                if look.peek(Token![box]) {
                    s.parse::<Token![box]>()?;
                    let y = pat::Pat::parse_single(s)?;
                    Ok(Box(y))
                } else if look.peek(Token![const]) {
                    s.parse::<Token![const]>()?;
                    let y;
                    let brace = braced!(y in s);
                    let attrs = y.call(attr::Attr::parse_inner)?;
                    let stmts = y.call(stmt::Block::parse_within)?;
                    Ok(Const(Const {
                        attrs,
                        block: stmt::Block { brace, stmts },
                    }))
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
            Err(_) => unimplemented!("Pat::Stream `{}`", self),
        };
        match y {
            Ellipsis => {
                p.word("...");
            },
            Box(x) => {
                p.word("box ");
                &x.pretty(p);
            },
            Const(x) => {
                p.word("const ");
                p.cbox(INDENT);
                p.small_block(&x.block, &x.attrs);
                p.end();
            },
        }
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
        Pat::Verbatim(parse::parse_verbatim(&beg, s))
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
    } else if s.peek(tok::Parenth) {
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
    let parenth = parenthed!(y in s);
    let mut elems = Puncted::new();
    while !y.is_empty() {
        let x = Pat::parse_many(&y)?;
        if y.is_empty() {
            if elems.is_empty() && !matches!(x, Pat::Rest(_)) {
                return Ok(Pat::Parenth(Parenth {
                    attrs: Vec::new(),
                    parenth,
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
        parenth,
        pats: elems,
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
        pats: elems,
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
    let parenth = parenthed!(y in s);
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
        parenth,
        pats: elems,
    })
}
fn parse_verbatim(beg: parse::Buffer, s: Stream) -> Res<Pat> {
    s.parse::<Token![box]>()?;
    Pat::parse_one(s)?;
    Ok(Pat::Verbatim(parse::parse_verbatim(&beg, s)))
}
fn parse_wild(s: Stream) -> Res<Wild> {
    Ok(Wild {
        attrs: Vec::new(),
        underscore: s.parse()?,
    })
}
