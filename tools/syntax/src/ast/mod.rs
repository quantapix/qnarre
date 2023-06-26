pub use self::{
    expr_ext::{ArrayExprKind, BlockModifier, CallableExpr, ElseBranch, LiteralKind},
    generated::{nodes::*, tokens::*},
    node_ext::{
        AttrKind, FieldKind, Macro, NameLike, NameOrNameRef, PathSegmentKind, SelfParamKind, SlicePatComponents,
        StructKind, TraitOrAlias, TypeBoundKind, TypeOrConstParam, VisibilityKind,
    },
    operators::{ArithOp, BinaryOp, CmpOp, LogicOp, Ordering, RangeOp, UnaryOp},
    token_ext::{CommentKind, CommentPlacement, CommentShape, IsString, QuoteOffsets, Radix},
    traits::{
        AttrDocCommentIter, DocCommentIter, HasArgList, HasAttrs, HasDocComments, HasGenericParams, HasLoopBody,
        HasModuleItem, HasName, HasTypeBounds, HasVisibility,
    },
};
use crate::SyntaxKind;
use either::Either;
use std::marker::PhantomData;

pub mod edit {
    use crate::{
        ast::{self, make},
        core::{NodeOrToken, WalkEvent},
        ted,
    };
    use std::{fmt, iter, ops};

    #[derive(Debug, Clone, Copy)]
    pub struct IndentLevel(pub u8);
    impl From<u8> for IndentLevel {
        fn from(level: u8) -> IndentLevel {
            IndentLevel(level)
        }
    }
    impl fmt::Display for IndentLevel {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let spaces = "                                        ";
            let buf;
            let len = self.0 as usize * 4;
            let indent = if len <= spaces.len() {
                &spaces[..len]
            } else {
                buf = " ".repeat(len);
                &buf
            };
            fmt::Display::fmt(indent, f)
        }
    }
    impl ops::Add<u8> for IndentLevel {
        type Output = IndentLevel;
        fn add(self, rhs: u8) -> IndentLevel {
            IndentLevel(self.0 + rhs)
        }
    }
    impl IndentLevel {
        pub fn single() -> IndentLevel {
            IndentLevel(0)
        }
        pub fn is_zero(&self) -> bool {
            self.0 == 0
        }
        pub fn from_element(element: &crate::Elem) -> IndentLevel {
            match element {
                NodeOrToken::Node(it) => IndentLevel::from_node(it),
                NodeOrToken::Token(it) => IndentLevel::from_token(it),
            }
        }
        pub fn from_node(node: &crate::Node) -> IndentLevel {
            match node.first_token() {
                Some(it) => Self::from_token(&it),
                None => IndentLevel(0),
            }
        }
        pub fn from_token(token: &crate::Token) -> IndentLevel {
            for ws in prev_tokens(token.clone()).filter_map(ast::Whitespace::cast) {
                let text = ws.syntax().text();
                if let Some(pos) = text.rfind('\n') {
                    let level = text[pos + 1..].chars().count() / 4;
                    return IndentLevel(level as u8);
                }
            }
            IndentLevel(0)
        }
        pub fn increase_indent(self, node: &crate::Node) {
            let tokens = node.preorder_with_tokens().filter_map(|event| match event {
                WalkEvent::Leave(NodeOrToken::Token(it)) => Some(it),
                _ => None,
            });
            for token in tokens {
                if let Some(ws) = ast::Whitespace::cast(token) {
                    if ws.text().contains('\n') {
                        let new_ws = make::tokens::whitespace(&format!("{}{self}", ws.syntax()));
                        ted::replace(ws.syntax(), &new_ws);
                    }
                }
            }
        }
        pub fn decrease_indent(self, node: &crate::Node) {
            let tokens = node.preorder_with_tokens().filter_map(|event| match event {
                WalkEvent::Leave(NodeOrToken::Token(it)) => Some(it),
                _ => None,
            });
            for token in tokens {
                if let Some(ws) = ast::Whitespace::cast(token) {
                    if ws.text().contains('\n') {
                        let new_ws = make::tokens::whitespace(&ws.syntax().text().replace(&format!("\n{self}"), "\n"));
                        ted::replace(ws.syntax(), &new_ws);
                    }
                }
            }
        }
    }
    fn prev_tokens(token: crate::Token) -> impl Iterator<Item = crate::Token> {
        iter::successors(Some(token), |token| token.prev_token())
    }
    pub trait AstNodeEdit: ast::AstNode + Clone + Sized {
        fn indent_level(&self) -> IndentLevel {
            IndentLevel::from_node(self.syntax())
        }
        #[must_use]
        fn indent(&self, level: IndentLevel) -> Self {
            fn indent_inner(node: &crate::Node, level: IndentLevel) -> crate::Node {
                let res = node.clone_subtree().clone_for_update();
                level.increase_indent(&res);
                res.clone_subtree()
            }
            Self::cast(indent_inner(self.syntax(), level)).unwrap()
        }
        #[must_use]
        fn dedent(&self, level: IndentLevel) -> Self {
            fn dedent_inner(node: &crate::Node, level: IndentLevel) -> crate::Node {
                let res = node.clone_subtree().clone_for_update();
                level.decrease_indent(&res);
                res.clone_subtree()
            }
            Self::cast(dedent_inner(self.syntax(), level)).unwrap()
        }
        #[must_use]
        fn reset_indent(&self) -> Self {
            let level = IndentLevel::from_node(self.syntax());
            self.dedent(level)
        }
    }
    impl<N: ast::AstNode + Clone> AstNodeEdit for N {}
    #[test]
    fn test_increase_indent() {
        let arm_list = {
            let arm = make::match_arm(iter::once(make::wildcard_pat().into()), None, make::expr_unit());
            make::match_arm_list(vec![arm.clone(), arm])
        };
        assert_eq!(
            arm_list.syntax().to_string(),
            "{
    _ => (),
    _ => (),
}"
        );
        let indented = arm_list.indent(IndentLevel(2));
        assert_eq!(
            indented.syntax().to_string(),
            "{
            _ => (),
            _ => (),
        }"
        );
    }
}
pub mod edit_in_place;
mod expr_ext;
mod generated {
    #[rustfmt::skip]
pub mod nodes;
#[rustfmt::skip]
pub mod tokens;
    use crate::{
        ast,
        SyntaxKind::{self, *},
    };
    pub use nodes::*;
    impl ast::AstNode for Stmt {
        fn can_cast(kind: SyntaxKind) -> bool {
            match kind {
                LET_STMT | EXPR_STMT => true,
                _ => Item::can_cast(kind),
            }
        }
        fn cast(syntax: crate::Node) -> Option<Self> {
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
        fn syntax(&self) -> &crate::Node {
            match self {
                Stmt::LetStmt(it) => &it.syntax,
                Stmt::ExprStmt(it) => &it.syntax,
                Stmt::Item(it) => it.syntax(),
            }
        }
    }
}
pub mod make;
mod node_ext;
mod operators {
    use std::fmt;
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub enum RangeOp {
        /// `..`
        Exclusive,
        /// `..=`
        Inclusive,
    }
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub enum UnaryOp {
        /// `*`
        Deref,
        /// `!`
        Not,
        /// `-`
        Neg,
    }
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub enum BinaryOp {
        LogicOp(LogicOp),
        ArithOp(ArithOp),
        CmpOp(CmpOp),
        Assignment { op: Option<ArithOp> },
    }
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub enum LogicOp {
        And,
        Or,
    }
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub enum CmpOp {
        Eq { negated: bool },
        Ord { ordering: Ordering, strict: bool },
    }
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub enum Ordering {
        Less,
        Greater,
    }
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub enum ArithOp {
        Add,
        Mul,
        Sub,
        Div,
        Rem,
        Shl,
        Shr,
        BitXor,
        BitOr,
        BitAnd,
    }
    impl fmt::Display for LogicOp {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let res = match self {
                LogicOp::And => "&&",
                LogicOp::Or => "||",
            };
            f.write_str(res)
        }
    }
    impl fmt::Display for ArithOp {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let res = match self {
                ArithOp::Add => "+",
                ArithOp::Mul => "*",
                ArithOp::Sub => "-",
                ArithOp::Div => "/",
                ArithOp::Rem => "%",
                ArithOp::Shl => "<<",
                ArithOp::Shr => ">>",
                ArithOp::BitXor => "^",
                ArithOp::BitOr => "|",
                ArithOp::BitAnd => "&",
            };
            f.write_str(res)
        }
    }
    impl fmt::Display for CmpOp {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let res = match self {
                CmpOp::Eq { negated: false } => "==",
                CmpOp::Eq { negated: true } => "!=",
                CmpOp::Ord {
                    ordering: Ordering::Less,
                    strict: false,
                } => "<=",
                CmpOp::Ord {
                    ordering: Ordering::Less,
                    strict: true,
                } => "<",
                CmpOp::Ord {
                    ordering: Ordering::Greater,
                    strict: false,
                } => ">=",
                CmpOp::Ord {
                    ordering: Ordering::Greater,
                    strict: true,
                } => ">",
            };
            f.write_str(res)
        }
    }
    impl fmt::Display for BinaryOp {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                BinaryOp::LogicOp(op) => fmt::Display::fmt(op, f),
                BinaryOp::ArithOp(op) => fmt::Display::fmt(op, f),
                BinaryOp::CmpOp(op) => fmt::Display::fmt(op, f),
                BinaryOp::Assignment { op } => {
                    if let Some(op) = op {
                        fmt::Display::fmt(op, f)?;
                    }
                    f.write_str("=")?;
                    Ok(())
                },
            }
        }
    }
}
pub mod prec {
    use crate::{
        ast::{self, BinaryOp, Expr, HasArgList},
        match_ast,
    };
    impl Expr {
        pub fn needs_parens_in(&self, parent: crate::Node) -> bool {
            match_ast! {
                match parent {
                    ast::Expr(e) => self.needs_parens_in_expr(&e),
                    ast::Stmt(e) => self.needs_parens_in_stmt(Some(&e)),
                    ast::StmtList(_) => self.needs_parens_in_stmt(None),
                    ast::ArgList(_) => false,
                    ast::MatchArm(_) => false,
                    _ => false,
                }
            }
        }
        fn needs_parens_in_expr(&self, parent: &Expr) -> bool {
            if parent.child_is_followed_by_a_block() {
                use Expr::*;
                match self {
                    ReturnExpr(e) if e.expr().is_none() => return true,
                    BreakExpr(e) if e.expr().is_none() => return true,
                    YieldExpr(e) if e.expr().is_none() => return true,
                    RangeExpr(e) if matches!(e.end(), Some(BlockExpr(..))) => return true,
                    _ if self.contains_exterior_struct_lit() => return true,
                    _ => {},
                }
            }
            if self.is_ret_like_with_no_value() && parent.is_postfix() {
                return false;
            }
            if self.is_paren_like()
                || parent.is_paren_like()
                || self.is_prefix() && (parent.is_prefix() || !self.is_ordered_before(parent))
                || self.is_postfix() && (parent.is_postfix() || self.is_ordered_before(parent))
            {
                return false;
            }
            let (left, right, inv) = match self.is_ordered_before(parent) {
                true => (self, parent, false),
                false => (parent, self, true),
            };
            let (_, left_right_bp) = left.binding_power();
            let (right_left_bp, _) = right.binding_power();
            (left_right_bp < right_left_bp) ^ inv
        }
        fn needs_parens_in_stmt(&self, stmt: Option<&ast::Stmt>) -> bool {
            use Expr::*;
            let mut innermost = self.clone();
            loop {
                let next = match &innermost {
                    BinExpr(e) => e.lhs(),
                    CallExpr(e) => e.expr(),
                    CastExpr(e) => e.expr(),
                    IndexExpr(e) => e.base(),
                    _ => break,
                };
                if let Some(next) = next {
                    innermost = next;
                    if !innermost.requires_semi_to_be_stmt() {
                        return true;
                    }
                } else {
                    break;
                }
            }
            if let Some(ast::Stmt::LetStmt(e)) = stmt {
                if e.let_else().is_some() {
                    match self {
                        BinExpr(e)
                            if e.op_kind()
                                .map(|op| matches!(op, BinaryOp::LogicOp(_)))
                                .unwrap_or(false) =>
                        {
                            return true
                        },
                        _ if self.clone().trailing_brace().is_some() => return true,
                        _ => {},
                    }
                }
            }
            false
        }
        fn binding_power(&self) -> (u8, u8) {
            use ast::{ArithOp::*, BinaryOp::*, Expr::*, LogicOp::*};
            match self {
                ContinueExpr(_) => (0, 0),
                ClosureExpr(_) | ReturnExpr(_) | YieldExpr(_) | YeetExpr(_) | BreakExpr(_) => (0, 1),
                RangeExpr(_) => (5, 5),
                BinExpr(e) => {
                    let Some(op) = e.op_kind() else { return (0, 0) };
                    match op {
                        Assignment { .. } => (4, 3),
                        LogicOp(op) => match op {
                            Or => (7, 8),
                            And => (9, 10),
                        },
                        CmpOp(_) => (11, 11),
                        ArithOp(op) => match op {
                            BitOr => (13, 14),
                            BitXor => (15, 16),
                            BitAnd => (17, 18),
                            Shl | Shr => (19, 20),
                            Add | Sub => (21, 22),
                            Mul | Div | Rem => (23, 24),
                        },
                    }
                },
                CastExpr(_) => (25, 26),
                BoxExpr(_) | RefExpr(_) | LetExpr(_) | PrefixExpr(_) => (0, 27),
                AwaitExpr(_) | CallExpr(_) | MethodCallExpr(_) | IndexExpr(_) | TryExpr(_) | MacroExpr(_) => (29, 0),
                FieldExpr(_) => (31, 32),
                ArrayExpr(_) | TupleExpr(_) | Literal(_) | PathExpr(_) | ParenExpr(_) | IfExpr(_) | WhileExpr(_)
                | ForExpr(_) | LoopExpr(_) | MatchExpr(_) | BlockExpr(_) | RecordExpr(_) | UnderscoreExpr(_) => (0, 0),
            }
        }
        fn is_paren_like(&self) -> bool {
            matches!(self.binding_power(), (0, 0))
        }
        fn is_prefix(&self) -> bool {
            matches!(self.binding_power(), (0, 1..))
        }
        fn is_postfix(&self) -> bool {
            matches!(self.binding_power(), (1.., 0))
        }
        fn requires_semi_to_be_stmt(&self) -> bool {
            use Expr::*;
            !matches!(
                self,
                IfExpr(..) | MatchExpr(..) | BlockExpr(..) | WhileExpr(..) | LoopExpr(..) | ForExpr(..)
            )
        }
        fn trailing_brace(mut self) -> Option<Expr> {
            use Expr::*;
            loop {
                let rhs = match self {
                    RefExpr(e) => e.expr(),
                    BinExpr(e) => e.rhs(),
                    BoxExpr(e) => e.expr(),
                    BreakExpr(e) => e.expr(),
                    LetExpr(e) => e.expr(),
                    RangeExpr(e) => e.end(),
                    ReturnExpr(e) => e.expr(),
                    PrefixExpr(e) => e.expr(),
                    YieldExpr(e) => e.expr(),
                    ClosureExpr(e) => e.body(),
                    BlockExpr(..) | ForExpr(..) | IfExpr(..) | LoopExpr(..) | MatchExpr(..) | RecordExpr(..)
                    | WhileExpr(..) => break Some(self),
                    _ => break None,
                };
                self = rhs?;
            }
        }
        fn contains_exterior_struct_lit(&self) -> bool {
            return contains_exterior_struct_lit_inner(self).is_some();
            fn contains_exterior_struct_lit_inner(expr: &Expr) -> Option<()> {
                use Expr::*;
                match expr {
                    RecordExpr(..) => Some(()),
                    // X { y: 1 } + X { y: 2 }
                    BinExpr(e) => e
                        .lhs()
                        .as_ref()
                        .and_then(contains_exterior_struct_lit_inner)
                        .or_else(|| e.rhs().as_ref().and_then(contains_exterior_struct_lit_inner)),
                    // `&X { y: 1 }`, `X { y: 1 }.y`, `X { y: 1 }.bar(...)`, etc
                    IndexExpr(e) => contains_exterior_struct_lit_inner(&e.base()?),
                    AwaitExpr(e) => contains_exterior_struct_lit_inner(&e.expr()?),
                    PrefixExpr(e) => contains_exterior_struct_lit_inner(&e.expr()?),
                    CastExpr(e) => contains_exterior_struct_lit_inner(&e.expr()?),
                    FieldExpr(e) => contains_exterior_struct_lit_inner(&e.expr()?),
                    MethodCallExpr(e) => contains_exterior_struct_lit_inner(&e.receiver()?),
                    _ => None,
                }
            }
        }
        fn is_ret_like_with_no_value(&self) -> bool {
            use Expr::*;
            match self {
                ReturnExpr(e) => e.expr().is_none(),
                BreakExpr(e) => e.expr().is_none(),
                ContinueExpr(_) => true,
                YieldExpr(e) => e.expr().is_none(),
                _ => false,
            }
        }
        fn is_ordered_before(&self, other: &Expr) -> bool {
            use Expr::*;
            return order(self) < order(other);
            fn order(this: &Expr) -> TextSize {
                let token = match this {
                    RangeExpr(e) => e.op_token(),
                    BinExpr(e) => e.op_token(),
                    CastExpr(e) => e.as_token(),
                    FieldExpr(e) => e.dot_token(),
                    AwaitExpr(e) => e.dot_token(),
                    BoxExpr(e) => e.box_token(),
                    BreakExpr(e) => e.break_token(),
                    CallExpr(e) => e.arg_list().and_then(|args| args.l_paren_token()),
                    ClosureExpr(e) => e.param_list().and_then(|params| params.l_paren_token()),
                    ContinueExpr(e) => e.continue_token(),
                    IndexExpr(e) => e.l_brack_token(),
                    MethodCallExpr(e) => e.dot_token(),
                    PrefixExpr(e) => e.op_token(),
                    RefExpr(e) => e.amp_token(),
                    ReturnExpr(e) => e.return_token(),
                    TryExpr(e) => e.question_mark_token(),
                    YieldExpr(e) => e.yield_token(),
                    YeetExpr(e) => e.do_token(),
                    LetExpr(e) => e.let_token(),
                    ArrayExpr(_) | TupleExpr(_) | Literal(_) | PathExpr(_) | ParenExpr(_) | IfExpr(_)
                    | WhileExpr(_) | ForExpr(_) | LoopExpr(_) | MatchExpr(_) | BlockExpr(_) | RecordExpr(_)
                    | UnderscoreExpr(_) | MacroExpr(_) => None,
                };
                token
                    .map(|t| t.text_range())
                    .unwrap_or_else(|| this.syntax().text_range())
                    .start()
            }
        }
        fn child_is_followed_by_a_block(&self) -> bool {
            use Expr::*;
            match self {
                ArrayExpr(_) | AwaitExpr(_) | BlockExpr(_) | CallExpr(_) | CastExpr(_) | ClosureExpr(_)
                | FieldExpr(_) | IndexExpr(_) | Literal(_) | LoopExpr(_) | MacroExpr(_) | MethodCallExpr(_)
                | ParenExpr(_) | PathExpr(_) | RecordExpr(_) | TryExpr(_) | TupleExpr(_) | UnderscoreExpr(_) => false,
                BinExpr(_) | RangeExpr(_) | BoxExpr(_) | BreakExpr(_) | ContinueExpr(_) | PrefixExpr(_)
                | RefExpr(_) | ReturnExpr(_) | YieldExpr(_) | YeetExpr(_) | LetExpr(_) => self
                    .syntax()
                    .parent()
                    .and_then(Expr::cast)
                    .map(|e| e.child_is_followed_by_a_block())
                    .unwrap_or(false),
                ForExpr(_) | IfExpr(_) | MatchExpr(_) | WhileExpr(_) => true,
            }
        }
    }
}
mod token_ext;
mod traits {
    use crate::{
        ast::{self, support},
        T, *,
    };
    use either::Either;

    pub trait HasName: ast::AstNode {
        fn name(&self) -> Option<ast::Name> {
            support::child(self.syntax())
        }
    }
    pub trait HasVisibility: ast::AstNode {
        fn visibility(&self) -> Option<ast::Visibility> {
            support::child(self.syntax())
        }
    }
    pub trait HasLoopBody: ast::AstNode {
        fn loop_body(&self) -> Option<ast::BlockExpr> {
            support::child(self.syntax())
        }
        fn label(&self) -> Option<ast::Label> {
            support::child(self.syntax())
        }
    }
    pub trait HasArgList: ast::AstNode {
        fn arg_list(&self) -> Option<ast::ArgList> {
            support::child(self.syntax())
        }
    }
    pub trait HasModuleItem: ast::AstNode {
        fn items(&self) -> AstChildren<ast::Item> {
            support::children(self.syntax())
        }
    }
    pub trait HasGenericParams: ast::AstNode {
        fn generic_param_list(&self) -> Option<ast::GenericParamList> {
            support::child(self.syntax())
        }
        fn where_clause(&self) -> Option<ast::WhereClause> {
            support::child(self.syntax())
        }
    }
    pub trait HasTypeBounds: ast::AstNode {
        fn type_bound_list(&self) -> Option<ast::TypeBoundList> {
            support::child(self.syntax())
        }
        fn colon_token(&self) -> Option<crate::Token> {
            support::token(self.syntax(), T![:])
        }
    }
    pub trait HasAttrs: ast::AstNode {
        fn attrs(&self) -> ast::AstChildren<ast::Attr> {
            support::children(self.syntax())
        }
        fn has_atom_attr(&self, atom: &str) -> bool {
            self.attrs().filter_map(|x| x.as_simple_atom()).any(|x| x == atom)
        }
    }
    pub trait HasDocComments: HasAttrs {
        fn doc_comments(&self) -> DocCommentIter {
            DocCommentIter {
                iter: self.syntax().children_with_tokens(),
            }
        }
        fn doc_comments_and_attrs(&self) -> AttrDocCommentIter {
            AttrDocCommentIter {
                iter: self.syntax().children_with_tokens(),
            }
        }
    }
    impl DocCommentIter {
        pub fn from_syntax_node(syntax_node: &ast::api::Node) -> DocCommentIter {
            DocCommentIter {
                iter: syntax_node.children_with_tokens(),
            }
        }
        #[cfg(test)]
        pub fn doc_comment_text(self) -> Option<String> {
            let docs = itertools::Itertools::join(
                &mut self.filter_map(|comment| comment.doc_comment().map(ToOwned::to_owned)),
                "\n",
            );
            if docs.is_empty() {
                None
            } else {
                Some(docs)
            }
        }
    }
    pub struct DocCommentIter {
        iter: ElemChildren,
    }
    impl Iterator for DocCommentIter {
        type Item = ast::Comment;
        fn next(&mut self) -> Option<ast::Comment> {
            self.iter.by_ref().find_map(|el| {
                el.into_token()
                    .and_then(ast::Comment::cast)
                    .filter(ast::Comment::is_doc)
            })
        }
    }
    pub struct AttrDocCommentIter {
        iter: ElemChildren,
    }
    impl AttrDocCommentIter {
        pub fn from_syntax_node(syntax_node: &ast::api::Node) -> AttrDocCommentIter {
            AttrDocCommentIter {
                iter: syntax_node.children_with_tokens(),
            }
        }
    }
    impl Iterator for AttrDocCommentIter {
        type Item = Either<ast::Attr, ast::Comment>;
        fn next(&mut self) -> Option<Self::Item> {
            self.iter.by_ref().find_map(|el| match el {
                crate::Elem::Node(node) => ast::Attr::cast(node).map(Either::Left),
                crate::Elem::Token(tok) => ast::Comment::cast(tok).filter(ast::Comment::is_doc).map(Either::Right),
            })
        }
    }
    impl<A: HasName, B: HasName> HasName for Either<A, B> {}
}

pub trait AstNode {
    fn can_cast(kind: SyntaxKind) -> bool
    where
        Self: Sized;
    fn cast(syntax: crate::Node) -> Option<Self>
    where
        Self: Sized;
    fn syntax(&self) -> &crate::Node;
    fn clone_for_update(&self) -> Self
    where
        Self: Sized,
    {
        Self::cast(self.syntax().clone_for_update()).unwrap()
    }
    fn clone_subtree(&self) -> Self
    where
        Self: Sized,
    {
        Self::cast(self.syntax().clone_subtree()).unwrap()
    }
}
pub trait AstToken {
    fn can_cast(x: SyntaxKind) -> bool
    where
        Self: Sized;
    fn cast(x: crate::Token) -> Option<Self>
    where
        Self: Sized;
    fn syntax(&self) -> &crate::Token;
    fn text(&self) -> &str {
        self.syntax().text()
    }
}
#[derive(Debug, Clone)]
pub struct AstChildren<N> {
    inner: crate::NodeChildren,
    ph: PhantomData<N>,
}
impl<N> AstChildren<N> {
    fn new(x: &crate::Node) -> Self {
        AstChildren {
            inner: x.children(),
            ph: PhantomData,
        }
    }
}
impl<N: AstNode> Iterator for AstChildren<N> {
    type Item = N;
    fn next(&mut self) -> Option<N> {
        self.inner.find_map(N::cast)
    }
}
impl<L, R> AstNode for Either<L, R>
where
    L: AstNode,
    R: AstNode,
{
    fn can_cast(x: SyntaxKind) -> bool
    where
        Self: Sized,
    {
        L::can_cast(x) || R::can_cast(x)
    }
    fn cast(x: crate::Node) -> Option<Self>
    where
        Self: Sized,
    {
        if L::can_cast(x.kind()) {
            L::cast(x).map(Either::Left)
        } else {
            R::cast(x).map(Either::Right)
        }
    }
    fn syntax(&self) -> &crate::Node {
        self.as_ref().either(L::syntax, R::syntax)
    }
}
impl<L, R> HasAttrs for Either<L, R>
where
    L: HasAttrs,
    R: HasAttrs,
{
}
mod support {
    use crate::{ast, SyntaxKind};
    pub fn child<N: ast::AstNode>(x: &crate::Node) -> Option<N> {
        x.children().find_map(N::cast)
    }
    pub fn children<N: ast::AstNode>(x: &crate::Node) -> ast::AstChildren<N> {
        ast::AstChildren::new(x)
    }
    pub fn token(x: &crate::Node, kind: SyntaxKind) -> Option<crate::Token> {
        x.children_with_tokens()
            .filter_map(|x| x.into_token())
            .find(|x| x.kind() == kind)
    }
}

#[test]
fn assert_ast_is_object_safe() {
    fn _f(_: &dyn AstNode, _: &dyn HasName) {}
}
#[test]
fn test_doc_comment_none() {
    let file = SourceFile::parse(
        r#"
        // non-doc
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert!(module.doc_comments().doc_comment_text().is_none());
}
#[test]
fn test_outer_doc_comment_of_items() {
    let file = SourceFile::parse(
        r#"
        /// doc
        // non-doc
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!(" doc", module.doc_comments().doc_comment_text().unwrap());
}
#[test]
fn test_inner_doc_comment_of_items() {
    let file = SourceFile::parse(
        r#"
        //! doc
        // non-doc
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert!(module.doc_comments().doc_comment_text().is_none());
}
#[test]
fn test_doc_comment_of_statics() {
    let file = SourceFile::parse(
        r#"
        /// Number of levels
        static LEVELS: i32 = 0;
        "#,
    )
    .ok()
    .unwrap();
    let st = file.syntax().descendants().find_map(Static::cast).unwrap();
    assert_eq!(" Number of levels", st.doc_comments().doc_comment_text().unwrap());
}
#[test]
fn test_doc_comment_preserves_indents() {
    let file = SourceFile::parse(
        r#"
        /// doc1
        /// ```
        /// fn foo() {
        ///     // ...
        /// }
        /// ```
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!(
        " doc1\n ```\n fn foo() {\n     // ...\n }\n ```",
        module.doc_comments().doc_comment_text().unwrap()
    );
}
#[test]
fn test_doc_comment_preserves_newlines() {
    let file = SourceFile::parse(
        r#"
        /// this
        /// is
        /// mod
        /// foo
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!(
        " this\n is\n mod\n foo",
        module.doc_comments().doc_comment_text().unwrap()
    );
}
#[test]
fn test_doc_comment_single_line_block_strips_suffix() {
    let file = SourceFile::parse(
        r#"
        /** this is mod foo*/
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!(" this is mod foo", module.doc_comments().doc_comment_text().unwrap());
}
#[test]
fn test_doc_comment_single_line_block_strips_suffix_whitespace() {
    let file = SourceFile::parse(
        r#"
        /** this is mod foo */
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!(" this is mod foo ", module.doc_comments().doc_comment_text().unwrap());
}
#[test]
fn test_doc_comment_multi_line_block_strips_suffix() {
    let file = SourceFile::parse(
        r#"
        /**
        this
        is
        mod foo
        */
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!(
        "\n        this\n        is\n        mod foo\n        ",
        module.doc_comments().doc_comment_text().unwrap()
    );
}
#[test]
fn test_comments_preserve_trailing_whitespace() {
    let file = SourceFile::parse(
        "\n/// Representation of a Realm.   \n/// In the specification these are called Realm Records.\nstruct Realm {}",
    )
    .ok()
    .unwrap();
    let def = file.syntax().descendants().find_map(Struct::cast).unwrap();
    assert_eq!(
        " Representation of a Realm.   \n In the specification these are called Realm Records.",
        def.doc_comments().doc_comment_text().unwrap()
    );
}
#[test]
fn test_four_slash_line_comment() {
    let file = SourceFile::parse(
        r#"
        //// too many slashes to be a doc comment
        /// doc comment
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!(" doc comment", module.doc_comments().doc_comment_text().unwrap());
}
#[test]
fn test_where_predicates() {
    fn assert_bound(text: &str, bound: Option<TypeBound>) {
        assert_eq!(text, bound.unwrap().syntax().text().to_string());
    }
    let file = SourceFile::parse(
        r#"
fn foo()
where
   T: Clone + Copy + Debug + 'static,
   'a: 'b + 'c,
   Iterator::Item: 'a + Debug,
   Iterator::Item: Debug + 'a,
   <T as Iterator>::Item: Debug + 'a,
   for<'a> F: Fn(&'a str)
{}
        "#,
    )
    .ok()
    .unwrap();
    let where_clause = file.syntax().descendants().find_map(WhereClause::cast).unwrap();
    let mut predicates = where_clause.predicates();
    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();
    assert!(pred.for_token().is_none());
    assert!(pred.generic_param_list().is_none());
    assert_eq!("T", pred.ty().unwrap().syntax().text().to_string());
    assert_bound("Clone", bounds.next());
    assert_bound("Copy", bounds.next());
    assert_bound("Debug", bounds.next());
    assert_bound("'static", bounds.next());
    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();
    assert_eq!("'a", pred.lifetime().unwrap().lifetime_ident_token().unwrap().text());
    assert_bound("'b", bounds.next());
    assert_bound("'c", bounds.next());
    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();
    assert_eq!("Iterator::Item", pred.ty().unwrap().syntax().text().to_string());
    assert_bound("'a", bounds.next());
    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();
    assert_eq!("Iterator::Item", pred.ty().unwrap().syntax().text().to_string());
    assert_bound("Debug", bounds.next());
    assert_bound("'a", bounds.next());
    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();
    assert_eq!("<T as Iterator>::Item", pred.ty().unwrap().syntax().text().to_string());
    assert_bound("Debug", bounds.next());
    assert_bound("'a", bounds.next());
    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();
    assert!(pred.for_token().is_some());
    assert_eq!("<'a>", pred.generic_param_list().unwrap().syntax().text().to_string());
    assert_eq!("F", pred.ty().unwrap().syntax().text().to_string());
    assert_bound("Fn(&'a str)", bounds.next());
}
