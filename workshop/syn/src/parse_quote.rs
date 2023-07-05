#[macro_export]
macro_rules! parse_quote {
    ($($tt:tt)*) => {
        $crate::__private::parse_quote($crate::__private::quote::quote!($($tt)*))
    };
}
#[macro_export]
macro_rules! parse_quote_spanned {
    ($span:expr=> $($tt:tt)*) => {
        $crate::__private::parse_quote($crate::__private::quote::quote_spanned!($span=> $($tt)*))
    };
}
use crate::parse::{Parse, ParseStream, Parser, Result};
use proc_macro2::TokenStream;
pub fn parse<T: ParseQuote>(token_stream: TokenStream) -> T {
    let parser = T::parse;
    match parser.parse2(token_stream) {
        Ok(t) => t,
        Err(err) => panic!("{}", err),
    }
}
pub trait ParseQuote: Sized {
    fn parse(input: ParseStream) -> Result<Self>;
}
impl<T: Parse> ParseQuote for T {
    fn parse(input: ParseStream) -> Result<Self> {
        <T as Parse>::parse(input)
    }
}
use crate::punctuated::Punctuated;

use crate::{attr, Attribute};
use crate::{Block, Pat, Stmt};

impl ParseQuote for Attribute {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(Token![#]) && input.peek2(Token![!]) {
            attr::parsing::single_parse_inner(input)
        } else {
            attr::parsing::single_parse_outer(input)
        }
    }
}
impl ParseQuote for Pat {
    fn parse(input: ParseStream) -> Result<Self> {
        Pat::parse_multi_with_leading_vert(input)
    }
}
impl ParseQuote for Box<Pat> {
    fn parse(input: ParseStream) -> Result<Self> {
        <Pat as ParseQuote>::parse(input).map(Box::new)
    }
}
impl<T: Parse, P: Parse> ParseQuote for Punctuated<T, P> {
    fn parse(input: ParseStream) -> Result<Self> {
        Self::parse_terminated(input)
    }
}
impl ParseQuote for Vec<Stmt> {
    fn parse(input: ParseStream) -> Result<Self> {
        Block::parse_within(input)
    }
}
