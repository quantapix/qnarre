use super::*;
use std::{iter, slice};

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
    pub fn path(&self) -> &Path {
        self.meta.path()
    }
    pub fn parse_args<T: Parse>(&self) -> Res<T> {
        self.parse_args_with(T::parse)
    }
    pub fn parse_args_with<T: Parser>(&self, p: T) -> Res<T::Output> {
        use meta::Meta;
        match &self.meta {
            Meta::Path(x) => Err(err::new2(
                x.segs.first().unwrap().ident.span(),
                x.segs.last().unwrap().ident.span(),
                format!(
                    "expected args in parentheses: {}[{}(...)]",
                    DisplayStyle(&self.style),
                    path::DisplayPath(x),
                ),
            )),
            Meta::NameValue(x) => Err(Err::new(
                x.eq.span,
                format_args!(
                    "expected parentheses: {}[{}(...)]",
                    DisplayStyle(&self.style),
                    path::DisplayPath(&x.path),
                ),
            )),
            Meta::List(x) => x.parse_args_with(p),
        }
    }
    pub fn parse_nested(&self, x: impl FnMut(meta::Nested) -> Res<()>) -> Res<()> {
        self.parse_args_with(meta::parser(x))
    }
    pub fn parse_inners(s: Stream) -> Res<Vec<Self>> {
        let mut y = Vec::new();
        parse_inners(s, &mut y)?;
        Ok(y)
    }
    pub fn parse_outers(s: Stream) -> Res<Vec<Self>> {
        let mut y = Vec::new();
        while s.peek(Token![#]) {
            y.push(s.call(parse_one_outer)?);
        }
        Ok(y)
    }
}
impl Lower for Attr {
    fn lower(&self, s: &mut Stream) {
        self.pound.lower(s);
        if let Style::Inner(x) = &self.style {
            x.lower(s);
        }
        self.bracket.surround(s, |s| {
            self.meta.lower(s);
        });
    }
}

pub trait Filter<'a> {
    type Ret: Iterator<Item = &'a Attr>;
    fn inners(self) -> Self::Ret;
    fn outers(self) -> Self::Ret;
}
impl<'a> Filter<'a> for &'a [Attr] {
    type Ret = iter::Filter<slice::Iter<'a, Attr>, fn(&&Attr) -> bool>;
    fn inners(self) -> Self::Ret {
        fn is_inner(x: &&Attr) -> bool {
            use Style::*;
            match x.style {
                Inner(_) => true,
                Outer => false,
            }
        }
        self.iter().filter(is_inner)
    }
    fn outers(self) -> Self::Ret {
        fn is_outer(x: &&Attr) -> bool {
            use Style::*;
            match x.style {
                Outer => true,
                Inner(_) => false,
            }
        }
        self.iter().filter(is_outer)
    }
}

pub struct DisplayStyle<'a>(pub &'a Style);
impl<'a> Display for DisplayStyle<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self.0 {
            Style::Outer => "#",
            Style::Inner(_) => "#!",
        })
    }
}

pub fn parse_inners(s: Stream, ys: &mut Vec<Attr>) -> Res<()> {
    while s.peek(Token![#]) && s.peek2(Token![!]) {
        ys.push(s.call(parse_one_inner)?);
    }
    Ok(())
}
pub fn parse_one_inner(s: Stream) -> Res<Attr> {
    let y;
    Ok(Attr {
        pound: s.parse()?,
        style: Style::Inner(s.parse()?),
        bracket: bracketed!(y in s),
        meta: y.parse()?,
    })
}
pub fn parse_one_outer(s: Stream) -> Res<Attr> {
    let y;
    Ok(Attr {
        pound: s.parse()?,
        style: Style::Outer,
        bracket: bracketed!(y in s),
        meta: y.parse()?,
    })
}

pub fn inners_to_tokens(xs: &[Attr], ys: &mut Stream) {
    ys.append_all(xs.inners());
}
pub fn outers_to_tokens(xs: &[Attr], ys: &mut Stream) {
    ys.append_all(xs.outers());
}
