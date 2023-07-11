ast_enum_of_structs! {
    pub enum Meta {
        List(List),
        NameValue(NameValue),
        Path(Path),
    }
}
impl Meta {
    pub fn path(&self) -> &Path {
        use Meta::*;
        match self {
            Path(x) => x,
            List(x) => &x.path,
            NameValue(x) => &x.path,
        }
    }
    pub fn require_path_only(&self) -> Res<&Path> {
        use Meta::*;
        let y = match self {
            Path(x) => return Ok(x),
            List(x) => x.delim.span().open(),
            NameValue(x) => x.eq.span,
        };
        Err(Err::new(y, "unexpected token in attribute"))
    }
    pub fn require_list(&self) -> Res<&List> {
        use Meta::*;
        match self {
            List(x) => Ok(x),
            Path(x) => Err(err::new2(
                x.segs.first().unwrap().ident.span(),
                x.segs.last().unwrap().ident.span(),
                format!(
                    "expected attribute arguments in parentheses: `{}(...)`",
                    parsing::DisplayPath(x),
                ),
            )),
            NameValue(x) => Err(Err::new(x.eq.span, "expected `(`")),
        }
    }
    pub fn require_name_value(&self) -> Res<&NameValue> {
        use Meta::*;
        match self {
            NameValue(x) => Ok(x),
            Path(x) => Err(err::new2(
                x.segs.first().unwrap().ident.span(),
                x.segs.last().unwrap().ident.span(),
                format!(
                    "expected a value for this attribute: `{} = ...`",
                    parsing::DisplayPath(x),
                ),
            )),
            List(x) => Err(Err::new(x.delim.span().open(), "expected `=`")),
        }
    }
}
impl Parse for Meta {
    fn parse(x: Stream) -> Res<Self> {
        let y = x.call(Path::parse_mod_style)?;
        parse_after_path(y, x)
    }
}

pub struct List {
    pub path: Path,
    pub delim: tok::Delim,
    pub toks: pm2::Stream,
}
impl List {
    pub fn parse_args<T: Parse>(&self) -> Res<T> {
        self.parse_args_with(T::parse)
    }
    pub fn parse_args_with<T: Parser>(&self, x: T) -> Res<T::Output> {
        let y = self.delim.span().close();
        parse::parse_scoped(x, y, self.toks.clone())
    }
    pub fn parse_nested(&self, x: impl FnMut(Nested) -> Res<()>) -> Res<()> {
        self.parse_args_with(parser(x))
    }
}
impl Parse for List {
    fn parse(x: Stream) -> Res<Self> {
        let y = x.call(Path::parse_mod_style)?;
        parse_list_after_path(y, x)
    }
}
impl ToTokens for List {
    fn to_tokens(&self, ys: &mut pm2::Stream) {
        self.path.to_tokens(ys);
        self.delim.surround(ys, self.toks.clone());
    }
}

pub struct NameValue {
    pub path: Path,
    pub eq: Token![=],
    pub expr: Expr,
}
impl Parse for NameValue {
    fn parse(x: Stream) -> Res<Self> {
        let y = x.call(Path::parse_mod_style)?;
        parse_name_value_after_path(y, x)
    }
}
impl ToTokens for NameValue {
    fn to_tokens(&self, ys: &mut pm2::Stream) {
        self.path.to_tokens(ys);
        self.eq.to_tokens(ys);
        self.expr.to_tokens(ys);
    }
}

pub struct Nested<'a> {
    pub path: Path,
    pub ins: Stream<'a>,
}
impl<'a> Nested<'a> {
    pub fn val(&self) -> Res<Stream<'a>> {
        self.ins.parse::<Token![=]>()?;
        Ok(self.ins)
    }
    pub fn parse(&self, cb: impl FnMut(Nested) -> Res<()>) -> Res<()> {
        let y;
        parenthesized!(y in self.ins);
        parse_nested(&y, cb)
    }
    pub fn err(&self, x: impl Display) -> Err {
        let beg = self.path.segs[0].ident.span();
        let end = self.ins.cursor().prev_span();
        err::new2(beg, end, x)
    }
}

pub fn parser(f: impl FnMut(Nested) -> Res<()>) -> impl Parser<Output = ()> {
    |x: Stream| {
        if x.is_empty() {
            Ok(())
        } else {
            parse_nested(x, f)
        }
    }
}

pub fn parse_nested(ins: Stream, mut f: impl FnMut(Nested) -> Res<()>) -> Res<()> {
    loop {
        let path = ins.call(parse_path)?;
        f(Nested { path, ins })?;
        if ins.is_empty() {
            return Ok(());
        }
        ins.parse::<Token![,]>()?;
        if ins.is_empty() {
            return Ok(());
        }
    }
}
pub fn parse_after_path(p: Path, x: Stream) -> Res<Meta> {
    if x.peek(tok::Paren) || x.peek(tok::Bracket) || x.peek(tok::Brace) {
        parse_list_after_path(p, x).map(Meta::List)
    } else if x.peek(Token![=]) {
        parse_name_value_after_path(p, x).map(Meta::NameValue)
    } else {
        Ok(Meta::Path(p))
    }
}
pub fn parse_list_after_path(path: Path, x: Stream) -> Res<List> {
    let (delim, toks) = mac::parse_delim(x)?;
    Ok(List { path, delim, toks })
}
pub fn parse_name_value_after_path(path: Path, x: Stream) -> Res<NameValue> {
    let eq: Token![=] = x.parse()?;
    let ahead = x.fork();
    let lit: Option<Lit> = ahead.parse()?;
    let expr = if let (Some(lit), true) = (lit, ahead.is_empty()) {
        x.advance_to(&ahead);
        Expr::Lit(expr::Lit { attrs: Vec::new(), lit })
    } else if x.peek(Token![#]) && x.peek2(tok::Bracket) {
        return Err(x.error("unexpected attribute inside of attribute"));
    } else {
        x.parse()?
    };
    Ok(NameValue { path, eq, expr })
}

fn parse_path(x: Stream) -> Res<Path> {
    Ok(Path {
        colon: x.parse()?,
        segs: {
            let mut ys = Punctuated::new();
            if x.peek(Ident::peek_any) {
                let y = Ident::parse_any(x)?;
                ys.push_value(path::Segment::from(y));
            } else if x.is_empty() {
                return Err(x.error("expected nested attribute"));
            } else if x.peek(lit::Lit) {
                return Err(x.error("unexpected literal in nested attribute, expected ident"));
            } else {
                return Err(x.error("unexpected token in nested attribute, expected ident"));
            }
            while x.peek(Token![::]) {
                let y = x.parse()?;
                ys.push_punct(y);
                let y = Ident::parse_any(x)?;
                ys.push_value(path::Segment::from(y));
            }
            ys
        },
    })
}
