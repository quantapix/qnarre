#![allow(clippy::uninlined_format_args)]
#[macro_use]
mod macros;
use syn::*;
#[test]
fn test_grouping() {
    let tokens: pm2::Stream = pm2::Stream::from_iter(vec![
        pm2::Tree::Literal(pm2::Lit::i32_suffixed(1)),
        pm2::Tree::Punct(Punct::new('+', pm2::Spacing::Alone)),
        pm2::Tree::Group(Group::new(
            pm2::Delim::None,
            pm2::Stream::from_iter(vec![
                pm2::Tree::Literal(pm2::Lit::i32_suffixed(2)),
                pm2::Tree::Punct(Punct::new('+', pm2::Spacing::Alone)),
                pm2::Tree::Literal(pm2::Lit::i32_suffixed(3)),
            ]),
        )),
        pm2::Tree::Punct(Punct::new('*', pm2::Spacing::Alone)),
        pm2::Tree::Literal(pm2::Lit::i32_suffixed(4)),
    ]);
    assert_eq!(tokens.to_string(), "1i32 + 2i32 + 3i32 * 4i32");
    snapshot!(tokens as Expr, @r###"
    Expr::Binary {
        left: Expr::Lit {
            lit: 1i32,
        },
        op: BinOp::Add,
        right: Expr::Binary {
            left: Expr::Group {
                expr: Expr::Binary {
                    left: Expr::Lit {
                        lit: 2i32,
                    },
                    op: BinOp::Add,
                    right: Expr::Lit {
                        lit: 3i32,
                    },
                },
            },
            op: BinOp::Mul,
            right: Expr::Lit {
                lit: 4i32,
            },
        },
    }
    "###);
}
