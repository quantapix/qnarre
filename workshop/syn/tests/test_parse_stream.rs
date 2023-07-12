#![allow(clippy::let_underscore_untyped)]
use syn::ext::IdentExt;
use syn::parse::Stream;
use syn::{Ident, Token};
#[test]
fn test_peek() {
    let _ = |x: Stream| {
        let _ = x.peek(Ident);
        let _ = x.peek(Ident::peek_any);
        let _ = x.peek(Token![::]);
    };
}
