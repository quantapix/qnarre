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
impl Visit for Ident {
    fn visit<V>(&self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        &self.span().visit(v);
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
    pub apos: pm2::Span,
    pub ident: Ident,
}
impl Life {
    pub fn new(x: &str, s: pm2::Span) -> Self {
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
    pub fn span(&self) -> pm2::Span {
        self.apos.join(self.ident.span()).unwrap_or(self.apos)
    }
    pub fn set_span(&mut self, s: pm2::Span) {
        self.apos = s;
        self.ident.set_span(s);
    }
}
impl Display for Life {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "'".fmt(f)?;
        self.ident.fmt(f)
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
impl PartialEq for Life {
    fn eq(&self, x: &Life) -> bool {
        self.ident.eq(&x.ident)
    }
}
impl Eq for Life {}
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
impl Hash for Life {
    fn hash<H: Hasher>(&self, x: &mut H) {
        self.ident.hash(x);
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
impl Visit for Life {
    fn visit<V>(&self, v: &mut V)
    where
        V: Visitor + ?Sized,
    {
        &self.apos.visit(v);
        &self.ident.visit(v);
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
