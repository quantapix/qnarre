use crate::buffer::Cursor;
use crate::parse::Peek;
use crate::parse::{ParseStream, Result};
use crate::sealed::lookahead;
use crate::token::CustomToken;
use proc_macro2::Ident;
pub trait IdentExt: Sized + private::Sealed {
    fn parse_any(input: ParseStream) -> Result<Self>;
    #[allow(non_upper_case_globals)]
    const peek_any: private::PeekFn = private::PeekFn;
    fn unraw(&self) -> Ident;
}
impl IdentExt for Ident {
    fn parse_any(input: ParseStream) -> Result<Self> {
        input.step(|cursor| match cursor.ident() {
            Some((ident, rest)) => Ok((ident, rest)),
            None => Err(cursor.error("expected ident")),
        })
    }
    fn unraw(&self) -> Ident {
        let string = self.to_string();
        if let Some(string) = string.strip_prefix("r#") {
            Ident::new(string, self.span())
        } else {
            self.clone()
        }
    }
}
impl Peek for private::PeekFn {
    type Token = private::IdentAny;
}
impl CustomToken for private::IdentAny {
    fn peek(cursor: Cursor) -> bool {
        cursor.ident().is_some()
    }
    fn display() -> &'static str {
        "identifier"
    }
}
impl lookahead::Sealed for private::PeekFn {}
mod private {
    use proc_macro2::Ident;
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
