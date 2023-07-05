use crate::lookahead;
use proc_macro2::{Ident, Span};
use std::cmp::Ordering;
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};
pub struct Lifetime {
    pub apostrophe: Span,
    pub ident: Ident,
}
impl Lifetime {
    pub fn new(symbol: &str, span: Span) -> Self {
        if !symbol.starts_with('\'') {
            panic!(
                "lifetime name must start with apostrophe as in \"'a\", got {:?}",
                symbol
            );
        }
        if symbol == "'" {
            panic!("lifetime name must not be empty");
        }
        if !crate::ident::xid_ok(&symbol[1..]) {
            panic!("{:?} is not a valid lifetime name", symbol);
        }
        Lifetime {
            apostrophe: span,
            ident: Ident::new(&symbol[1..], span),
        }
    }
    pub fn span(&self) -> Span {
        self.apostrophe.join(self.ident.span()).unwrap_or(self.apostrophe)
    }
    pub fn set_span(&mut self, span: Span) {
        self.apostrophe = span;
        self.ident.set_span(span);
    }
}
impl Display for Lifetime {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        "'".fmt(formatter)?;
        self.ident.fmt(formatter)
    }
}
impl Clone for Lifetime {
    fn clone(&self) -> Self {
        Lifetime {
            apostrophe: self.apostrophe,
            ident: self.ident.clone(),
        }
    }
}
impl PartialEq for Lifetime {
    fn eq(&self, other: &Lifetime) -> bool {
        self.ident.eq(&other.ident)
    }
}
impl Eq for Lifetime {}
impl PartialOrd for Lifetime {
    fn partial_cmp(&self, other: &Lifetime) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Lifetime {
    fn cmp(&self, other: &Lifetime) -> Ordering {
        self.ident.cmp(&other.ident)
    }
}
impl Hash for Lifetime {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.ident.hash(h);
    }
}
#[allow(non_snake_case)]
pub fn Lifetime(marker: lookahead::TokenMarker) -> Lifetime {
    match marker {}
}
pub(crate) mod parsing {
    use super::*;
    use crate::parse::{Parse, ParseStream, Result};

    impl Parse for Lifetime {
        fn parse(input: ParseStream) -> Result<Self> {
            input.step(|cursor| cursor.lifetime().ok_or_else(|| cursor.error("expected lifetime")))
        }
    }
}
mod printing {
    use super::*;
    use proc_macro2::{Punct, Spacing, TokenStream};
    use quote::{ToTokens, TokenStreamExt};
    #[cfg_attr(doc_cfg, doc(cfg(feature = "printing")))]
    impl ToTokens for Lifetime {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let mut apostrophe = Punct::new('\'', Spacing::Joint);
            apostrophe.set_span(self.apostrophe);
            tokens.append(apostrophe);
            self.ident.to_tokens(tokens);
        }
    }
}
