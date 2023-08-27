use super::*;

impl Parse for Ident {
    fn parse(s: Stream) -> Res<Self> {
        s.step(|x| {
            if let Some((x, rest)) = x.ident() {
                if tok::accept_as_ident(&x) {
                    Ok((x, rest))
                } else {
                    Err(x.err(format_args!("expected identifier, found keyword `{}`", x,)))
                }
            } else {
                Err(x.err("expected identifier"))
            }
        })
    }
}
impl<F: Folder + ?Sized> Fold for Ident {
    fn fold(&self, f: &mut F) {
        let mut y = self;
        let span = self.span().fold(f);
        self.set_span(span);
        y
    }
}
impl<V: Visitor + ?Sized> Visit for Ident {
    fn visit(&self, v: &mut V) {
        &self.span().visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        let mut x = self.span();
        &mut x.visit_mut(v);
        self.set_span(x);
    }
}

pub fn make(x: &str, span: Option<Span>) -> Ident {
    let span = span.unwrap_or_else(Span::call_site);
    maybe_raw(x, span)
}
pub fn maybe_raw(x: &str, span: Span) -> Ident {
    if x.starts_with("r#") {
        Ident::new_raw(&x[2..], span)
    } else {
        Ident::new(x, span)
    }
}

macro_rules! ident_from_tok {
    ($x:ident) => {
        impl From<Token![$x]> for Ident {
            fn from(x: Token![$x]) -> Ident {
                Ident::new(stringify!($x), x.span)
            }
        }
    };
}
ident_from_tok!(self);
ident_from_tok!(Self);
ident_from_tok!(super);
ident_from_tok!(crate);
ident_from_tok!(extern);
impl From<Token![_]> for Ident {
    fn from(x: Token![_]) -> Ident {
        Ident::new("_", x.span)
    }
}
pub fn xid_ok(x: &str) -> bool {
    let mut ys = x.chars();
    let first = ys.next().unwrap();
    if !(first == '_' || unicode_ident::is_xid_start(first)) {
        return false;
    }
    for y in ys {
        if !unicode_ident::is_xid_continue(y) {
            return false;
        }
    }
    true
}

pub struct Life {
    pub apos: Span,
    pub ident: Ident,
}
impl Life {
    pub fn new(x: &str, s: Span) -> Self {
        if !x.starts_with('\'') {
            panic!("life name must start with apos as in \"'a\", got {:?}", x);
        }
        if x == "'" {
            panic!("life name must not be empty");
        }
        if !ident::xid_ok(&x[1..]) {
            panic!("{:?} is not a valid life name", x);
        }
        Life {
            apos: s,
            ident: Ident::new(&x[1..], s),
        }
    }
    pub fn span(&self) -> Span {
        self.apos.join(self.ident.span()).unwrap_or(self.apos)
    }
    pub fn set_span(&mut self, s: Span) {
        self.apos = s;
        self.ident.set_span(s);
    }
}
impl Clone for Life {
    fn clone(&self) -> Self {
        Life {
            apos: self.apos,
            ident: self.ident.clone(),
        }
    }
}
impl Debug for Life {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Life {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("apos", &self.apos);
                f.field("ident", &self.ident);
                f.finish()
            }
        }
        self.debug(f, "ident::Life")
    }
}
impl Display for Life {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "'".fmt(f)?;
        self.ident.fmt(f)
    }
}
impl Eq for Life {}
impl PartialEq for Life {
    fn eq(&self, x: &Life) -> bool {
        self.ident.eq(&x.ident)
    }
}
impl PartialOrd for Life {
    fn partial_cmp(&self, x: &Life) -> Option<Ordering> {
        Some(self.cmp(x))
    }
}
impl Ord for Life {
    fn cmp(&self, x: &Life) -> Ordering {
        self.ident.cmp(&x.ident)
    }
}
impl Parse for Life {
    fn parse(s: Stream) -> Res<Self> {
        s.step(|x| x.life().ok_or_else(|| x.err("expected life")))
    }
}
impl Lower for Life {
    fn lower(&self, s: &mut Stream) {
        let mut y = Punct::new('\'', pm2::Spacing::Joint);
        y.set_span(self.apos);
        s.append(y);
        self.ident.lower(s);
    }
}
impl Pretty for Life {
    fn pretty(&self, p: &mut Print) {
        p.word("'");
        p.ident(&self.ident);
    }
}
impl<F: Folder + ?Sized> Fold for Life {
    fn fold(&self, f: &mut F) {
        Life {
            apos: self.apos.fold(f),
            ident: self.ident.fold(f),
        }
    }
}
impl<H: Hasher> Hash for Life {
    fn hash(&self, x: &mut H) {
        self.ident.hash(x);
    }
}
impl<V: Visitor + ?Sized> Visit for Life {
    fn visit(&self, v: &mut V) {
        &self.apos.visit(v);
        &self.ident.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut self.apos.visit_mut(v);
        &mut self.ident.visit_mut(v);
    }
}

#[allow(non_snake_case)]
pub fn Ident(x: look::Marker) -> Ident {
    match x {}
}
#[allow(non_snake_case)]
pub fn Life(x: look::Marker) -> Life {
    match x {}
}

pub trait IdentExt: Sized {
    fn parse_any(x: parse::Stream) -> Res<Self>;
    fn unraw(&self) -> Ident;
}
impl IdentExt for Ident {
    fn parse_any(x: parse::Stream) -> Res<Self> {
        x.step(|c| match c.ident() {
            Some((ident, rest)) => Ok((ident, rest)),
            None => Err(c.err("expected ident")),
        })
    }
    fn unraw(&self) -> Ident {
        let y = self.to_string();
        if let Some(x) = y.strip_prefix("r#") {
            Ident::new(x, self.span())
        } else {
            self.clone()
        }
    }
}
