use super::*;

impl Parse for Ident {
    fn parse(x: Stream) -> Res<Self> {
        x.step(|cursor| {
            if let Some((ident, rest)) = cursor.ident() {
                if accept_as_ident(&ident) {
                    Ok((ident, rest))
                } else {
                    Err(cursor.err(format_args!("expected identifier, found keyword `{}`", ident,)))
                }
            } else {
                Err(cursor.err("expected identifier"))
            }
        })
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

pub struct Lifetime {
    pub apos: pm2::Span,
    pub ident: Ident,
}
impl Lifetime {
    pub fn new(x: &str, s: pm2::Span) -> Self {
        if !x.starts_with('\'') {
            panic!("lifetime name must start with apostrophe as in \"'a\", got {:?}", x);
        }
        if x == "'" {
            panic!("lifetime name must not be empty");
        }
        if !ident::xid_ok(&x[1..]) {
            panic!("{:?} is not a valid lifetime name", x);
        }
        Lifetime {
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
impl Display for Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "'".fmt(f)?;
        self.ident.fmt(f)
    }
}
impl Clone for Lifetime {
    fn clone(&self) -> Self {
        Lifetime {
            apos: self.apos,
            ident: self.ident.clone(),
        }
    }
}
impl PartialEq for Lifetime {
    fn eq(&self, x: &Lifetime) -> bool {
        self.ident.eq(&x.ident)
    }
}
impl Eq for Lifetime {}
impl PartialOrd for Lifetime {
    fn partial_cmp(&self, x: &Lifetime) -> Option<Ordering> {
        Some(self.cmp(x))
    }
}
impl Ord for Lifetime {
    fn cmp(&self, x: &Lifetime) -> Ordering {
        self.ident.cmp(&x.ident)
    }
}
impl Hash for Lifetime {
    fn hash<H: Hasher>(&self, x: &mut H) {
        self.ident.hash(x);
    }
}
impl Parse for Lifetime {
    fn parse(x: Stream) -> Res<Self> {
        x.step(|c| c.lifetime().ok_or_else(|| c.err("expected lifetime")))
    }
}
impl ToTokens for Lifetime {
    fn to_tokens(&self, ys: &mut Stream) {
        let mut apos = Punct::new('\'', pm2::Spacing::Joint);
        apos.set_span(self.apos);
        ys.append(apos);
        self.ident.to_tokens(ys);
    }
}

#[allow(non_snake_case)]
pub fn Ident(x: look::Marker) -> Ident {
    match x {}
}
#[allow(non_snake_case)]
pub fn Lifetime(x: look::Marker) -> Lifetime {
    match x {}
}

pub trait IdentExt: Sized + private::Sealed {
    #[allow(non_upper_case_globals)]
    const peek_any: private::PeekFn = private::PeekFn;
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
impl parse::Peek for private::PeekFn {
    type Token = private::IdentAny;
}
impl tok::Custom for private::IdentAny {
    fn peek(x: Cursor) -> bool {
        x.ident().is_some()
    }
    fn display() -> &'static str {
        "identifier"
    }
}
impl look::Sealed for private::PeekFn {}
mod private {
    pub trait Sealed {}
    impl Sealed for Ident {}
    pub struct PeekFn;
    pub struct IdentAny;
    impl Copy for PeekFn {}
    impl Clone for PeekFn {
        fn clone(&self) -> Self {
            *self
        }
    }
}
