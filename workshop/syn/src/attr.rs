pub enum Style {
    Outer,
    Inner(Token![!]),
}

pub struct Attr {
    pub pound: Token![#],
    pub style: Style,
    pub bracket: tok::Bracket,
    pub meta: meta::Meta,
}
impl Attr {
    pub fn path(&self) -> &path::Path {
        self.meta.path()
    }
    pub fn parse_args<T: Parse>(&self) -> Res<T> {
        self.parse_args_with(T::parse)
    }
    pub fn parse_args_with<T: Parser>(&self, p: T) -> Res<T::Output> {
        use meta::Meta::*;
        match &self.meta {
            Path(x) => Err(err::new2(
                x.segs.first().unwrap().ident.span(),
                x.segs.last().unwrap().ident.span(),
                format!(
                    "expected attribute arguments in parentheses: {}[{}(...)]",
                    DisplayAttrStyle(&self.style),
                    DisplayPath(x),
                ),
            )),
            NameValue(x) => Err(Err::new(
                x.eq.span,
                format_args!(
                    "expected parentheses: {}[{}(...)]",
                    DisplayAttrStyle(&self.style),
                    DisplayPath(&meta.path),
                ),
            )),
            List(x) => x.parse_args_with(p),
        }
    }
    pub fn parse_nested(&self, x: impl FnMut(meta::Nested) -> Res<()>) -> Res<()> {
        self.parse_args_with(meta::parser(x))
    }
    pub fn parse_outer(x: Stream) -> Res<Vec<Self>> {
        let mut y = Vec::new();
        while x.peek(Token![#]) {
            y.push(x.call(parse_single_outer)?);
        }
        Ok(y)
    }
    pub fn parse_inner(x: Stream) -> Res<Vec<Self>> {
        let mut y = Vec::new();
        parse_inner(x, &mut y)?;
        Ok(y)
    }
}
impl ToTokens for Attr {
    fn to_tokens(&self, ys: &mut TokenStream) {
        self.pound.to_tokens(ys);
        if let Style::Inner(x) = &self.style {
            x.to_tokens(ys);
        }
        self.bracket.surround(ys, |x| {
            self.meta.to_tokens(x);
        });
    }
}

pub trait Filter<'a> {
    type Ret: Iterator<Item = &'a Attr>;
    fn outer(self) -> Self::Ret;
    fn inner(self) -> Self::Ret;
}
impl<'a> Filter<'a> for &'a [Attr] {
    type Ret = iter::Filter<slice::Iter<'a, Attr>, fn(&&Attr) -> bool>;
    fn outer(self) -> Self::Ret {
        fn is_outer(x: &&Attr) -> bool {
            use Style::*;
            match x.style {
                Outer => true,
                Inner(_) => false,
            }
        }
        self.iter().filter(is_outer)
    }
    fn inner(self) -> Self::Ret {
        fn is_inner(x: &&Attr) -> bool {
            use Style::*;
            match x.style {
                Inner(_) => true,
                Outer => false,
            }
        }
        self.iter().filter(is_inner)
    }
}

pub struct DisplayAttrStyle<'a>(pub &'a Style);
impl<'a> Display for DisplayAttrStyle<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self.0 {
            Style::Outer => "#",
            Style::Inner(_) => "#!",
        })
    }
}

pub fn parse_inner(x: Stream, ys: &mut Vec<Attr>) -> Res<()> {
    while x.peek(Token![#]) && x.peek2(Token![!]) {
        ys.push(x.call(parse_single_inner)?);
    }
    Ok(())
}
pub fn parse_single_inner(x: Stream) -> Res<Attr> {
    let y;
    Ok(Attr {
        pound: x.parse()?,
        style: Style::Inner(x.parse()?),
        bracket: bracketed!(y in x),
        meta: y.parse()?,
    })
}
pub fn parse_single_outer(x: Stream) -> Res<Attr> {
    let y;
    Ok(Attr {
        pound: x.parse()?,
        style: Style::Outer,
        bracket: bracketed!(y in x),
        meta: y.parse()?,
    })
}

pub fn outer_attrs_to_tokens(xs: &[Attr], ys: &mut TokenStream) {
    ys.append_all(xs.outer());
}
fn inner_attrs_to_tokens(xs: &[Attr], ys: &mut TokenStream) {
    ys.append_all(xs.inner());
}
