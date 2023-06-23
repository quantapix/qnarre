use crate::ast;

pub fn expr_requires_semi_to_be_stmt(e: &ast::Expr) -> bool {
    !matches!(
        e.kind,
        ast::ExprKind::If(..)
            | ast::ExprKind::Match(..)
            | ast::ExprKind::Block(..)
            | ast::ExprKind::While(..)
            | ast::ExprKind::Loop(..)
            | ast::ExprKind::ForLoop(..)
            | ast::ExprKind::TryBlock(..)
            | ast::ExprKind::ConstBlock(..)
    )
}

pub fn expr_trailing_brace(mut expr: &ast::Expr) -> Option<&ast::Expr> {
    use ast::ExprKind::*;

    loop {
        match &expr.kind {
            AddrOf(_, _, e)
            | Assign(_, e, _)
            | AssignOp(_, _, e)
            | Binary(_, _, e)
            | Break(_, Some(e))
            | Let(_, e, _)
            | Range(_, Some(e), _)
            | Ret(Some(e))
            | Unary(_, e)
            | Yield(Some(e)) => {
                expr = e;
            },
            Closure(closure) => {
                expr = &closure.body;
            },
            Async(..) | Block(..) | ForLoop(..) | If(..) | Loop(..) | Match(..) | Struct(..) | TryBlock(..)
            | While(..) => break Some(expr),
            _ => break None,
        }
    }
}
