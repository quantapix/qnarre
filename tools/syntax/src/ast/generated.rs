#[rustfmt::skip]
pub mod nodes;
#[rustfmt::skip]
pub mod tokens;

use crate::{
    AstNode,
    SyntaxKind::{self, *},
    SyntaxNode,
};

pub use nodes::*;

impl AstNode for Stmt {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            LET_STMT | EXPR_STMT => true,
            _ => Item::can_cast(kind),
        }
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            LET_STMT => Stmt::LetStmt(LetStmt { syntax }),
            EXPR_STMT => Stmt::ExprStmt(ExprStmt { syntax }),
            _ => {
                let item = Item::cast(syntax)?;
                Stmt::Item(item)
            },
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            Stmt::LetStmt(it) => &it.syntax,
            Stmt::ExprStmt(it) => &it.syntax,
            Stmt::Item(it) => it.syntax(),
        }
    }
}
