#![allow(clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use syn::{Expr, Item};
#[test]
fn test_async_fn() {
    let input = "async fn process() {}";
    snapshot!(input as Item, @r###"
    Item::Fn {
        vis: Visibility::Inherited,
        sig: Signature {
            async_: Some,
            ident: "process",
            gens: Generics,
            ret: ty::Ret::Default,
        },
        block: Block,
    }
    "###);
}
#[test]
fn test_async_closure() {
    let input = "async || {}";
    snapshot!(input as Expr, @r###"
    Expr::Closure {
        async_: Some,
        ret: ty::Ret::Default,
        body: Expr::Block {
            block: Block,
        },
    }
    "###);
}
