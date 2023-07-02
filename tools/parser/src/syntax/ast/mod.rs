pub use self::{
    expr::{ArrayExprKind, BlockModifier, CallableExpr, ElseBranch, LiteralKind},
    generated::{nodes::*, tokens::*},
    node::{
        AttrKind, FieldKind, Macro, NameLike, NameOrNameRef, PathSegmentKind, SelfParamKind, SlicePatComponents,
        StructKind, TraitOrAlias, TypeBoundKind, TypeOrConstParam, VisibilityKind,
    },
    operators::{ArithOp, BinaryOp, CmpOp, LogicOp, Ordering, RangeOp, UnaryOp},
    token::{CommentKind, CommentPlacement, CommentShape, IsString, QuoteOffsets, Radix},
    traits::{
        AttrDocCommentIter, DocCommentIter, HasArgList, HasAttrs, HasDocComments, HasGenericParams, HasLoopBody,
        HasModuleItem, HasName, HasTypeBounds, HasVisibility,
    },
};
use crate::{syntax, SyntaxKind};
use either::Either;
use std::marker::PhantomData;

pub mod edit {
    use crate::syntax::{
        self,
        ast::{self, make},
        core::{NodeOrToken, WalkEvent},
        ted,
    };
    use std::{fmt, iter, ops};

    #[derive(Debug, Clone, Copy)]
    pub struct Indent(pub u8);
    impl From<u8> for Indent {
        fn from(x: u8) -> Indent {
            Indent(x)
        }
    }
    impl fmt::Display for Indent {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let spaces = "                                        ";
            let buf;
            let len = self.0 as usize * 4;
            let y = if len <= spaces.len() {
                &spaces[..len]
            } else {
                buf = " ".repeat(len);
                &buf
            };
            fmt::Display::fmt(y, f)
        }
    }
    impl ops::Add<u8> for Indent {
        type Output = Indent;
        fn add(self, x: u8) -> Indent {
            Indent(self.0 + x)
        }
    }
    impl Indent {
        pub fn single() -> Indent {
            Indent(0)
        }
        pub fn is_zero(&self) -> bool {
            self.0 == 0
        }
        pub fn from_elem(x: &syntax::Elem) -> Indent {
            use NodeOrToken::*;
            match x {
                Node(x) => Indent::from_node(x),
                Token(x) => Indent::from_tok(x),
            }
        }
        pub fn from_node(x: &syntax::Node) -> Indent {
            match x.first_token() {
                Some(x) => Self::from_tok(&x),
                None => Indent(0),
            }
        }
        pub fn from_tok(x: &syntax::Token) -> Indent {
            for ws in prev_toks(x.clone()).filter_map(ast::Whitespace::cast) {
                let text = ws.syntax().text();
                if let Some(pos) = text.rfind('\n') {
                    let y = text[pos + 1..].chars().count() / 4;
                    return Indent(y as u8);
                }
            }
            Indent(0)
        }
        pub fn increase(self, x: &syntax::Node) {
            let xs = x.preorder_with_tokens().filter_map(|x| match x {
                WalkEvent::Leave(NodeOrToken::Token(x)) => Some(x),
                _ => None,
            });
            for x in xs {
                if let Some(x) = ast::Whitespace::cast(x) {
                    if x.text().contains('\n') {
                        let y = make::tokens::whitespace(&format!("{}{self}", x.syntax()));
                        ted::replace(x.syntax(), &y);
                    }
                }
            }
        }
        pub fn decrease(self, x: &syntax::Node) {
            let xs = x.preorder_with_tokens().filter_map(|x| match x {
                WalkEvent::Leave(NodeOrToken::Token(x)) => Some(x),
                _ => None,
            });
            for x in xs {
                if let Some(x) = ast::Whitespace::cast(x) {
                    if x.text().contains('\n') {
                        let y = make::tokens::whitespace(&x.syntax().text().replace(&format!("\n{self}"), "\n"));
                        ted::replace(x.syntax(), &y);
                    }
                }
            }
        }
    }
    fn prev_toks(x: syntax::Token) -> impl Iterator<Item = syntax::Token> {
        iter::successors(Some(x), |x| x.prev_token())
    }
    pub trait NodeEdit: ast::Node + Clone + Sized {
        fn indent_level(&self) -> Indent {
            Indent::from_node(self.syntax())
        }
        #[must_use]
        fn indent(&self, x: Indent) -> Self {
            fn inner(n: &syntax::Node, x: Indent) -> syntax::Node {
                let y = n.clone_subtree().clone_for_update();
                x.increase(&y);
                y.clone_subtree()
            }
            Self::cast(inner(self.syntax(), x)).unwrap()
        }
        #[must_use]
        fn dedent(&self, x: Indent) -> Self {
            fn inner(n: &syntax::Node, x: Indent) -> syntax::Node {
                let y = n.clone_subtree().clone_for_update();
                x.decrease(&y);
                y.clone_subtree()
            }
            Self::cast(inner(self.syntax(), x)).unwrap()
        }
        #[must_use]
        fn reset_indent(&self) -> Self {
            let y = Indent::from_node(self.syntax());
            self.dedent(y)
        }
    }
    impl<N: ast::Node + Clone> NodeEdit for N {}
    #[test]
    fn test_increase_indent() {
        let y = {
            let x = make::match_arm(iter::once(make::wildcard_pat().into()), None, make::expr_unit());
            make::match_arm_list(vec![x.clone(), x])
        };
        assert_eq!(
            y.syntax().to_string(),
            "{
    _ => (),
    _ => (),
}"
        );
        let y = y.indent(Indent(2));
        assert_eq!(
            y.syntax().to_string(),
            "{
            _ => (),
            _ => (),
        }"
        );
    }
}
pub mod edit_in_place;
mod expr;
mod generated {
    pub mod nodes;
    pub mod tokens;
    use crate::{
        syntax::{self, ast},
        SyntaxKind::{self, *},
    };
    pub use nodes::*;
    impl ast::Node for Stmt {
        fn can_cast(x: SyntaxKind) -> bool {
            match x {
                LET_STMT | EXPR_STMT => true,
                _ => Item::can_cast(x),
            }
        }
        fn cast(syntax: syntax::Node) -> Option<Self> {
            let y = match syntax.kind() {
                LET_STMT => Stmt::LetStmt(LetStmt { syntax }),
                EXPR_STMT => Stmt::ExprStmt(ExprStmt { syntax }),
                _ => {
                    let x = Item::cast(syntax)?;
                    Stmt::Item(x)
                },
            };
            Some(y)
        }
        fn syntax(&self) -> &syntax::Node {
            match self {
                Stmt::LetStmt(x) => &x.syntax,
                Stmt::ExprStmt(x) => &x.syntax,
                Stmt::Item(x) => x.syntax(),
            }
        }
    }
}
pub mod make;
mod node;
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
        match_ast,
        syntax::{
            self,
            ast::{self, BinaryOp, Expr, HasArgList},
            TextSize,
        },
    };
    impl Expr {
        pub fn needs_parens_in(&self, parent: syntax::Node) -> bool {
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
mod token;
mod traits {
    use crate::{
        syntax::{
            self,
            ast::{self, support},
        },
        T,
    };
    use either::Either;

    pub trait HasName: ast::Node {
        fn name(&self) -> Option<ast::Name> {
            support::child(self.syntax())
        }
    }
    pub trait HasVisibility: ast::Node {
        fn visibility(&self) -> Option<ast::Visibility> {
            support::child(self.syntax())
        }
    }
    pub trait HasLoopBody: ast::Node {
        fn loop_body(&self) -> Option<ast::BlockExpr> {
            support::child(self.syntax())
        }
        fn label(&self) -> Option<ast::Label> {
            support::child(self.syntax())
        }
    }
    pub trait HasArgList: ast::Node {
        fn arg_list(&self) -> Option<ast::ArgList> {
            support::child(self.syntax())
        }
    }
    pub trait HasModuleItem: ast::Node {
        fn items(&self) -> ast::Children<ast::Item> {
            support::children(self.syntax())
        }
    }
    pub trait HasGenericParams: ast::Node {
        fn generic_param_list(&self) -> Option<ast::GenericParamList> {
            support::child(self.syntax())
        }
        fn where_clause(&self) -> Option<ast::WhereClause> {
            support::child(self.syntax())
        }
    }
    pub trait HasTypeBounds: ast::Node {
        fn type_bound_list(&self) -> Option<ast::TypeBoundList> {
            support::child(self.syntax())
        }
        fn colon_token(&self) -> Option<syntax::Token> {
            support::token(self.syntax(), T![:])
        }
    }
    pub trait HasAttrs: ast::Node {
        fn attrs(&self) -> ast::Children<ast::Attr> {
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
    pub struct DocCommentIter {
        iter: syntax::ElemChildren,
    }
    impl DocCommentIter {
        pub fn from_syntax_node(x: &syntax::Node) -> DocCommentIter {
            DocCommentIter {
                iter: x.children_with_tokens(),
            }
        }
        #[cfg(test)]
        pub fn doc_comment_text(self) -> Option<String> {
            let docs =
                itertools::Itertools::join(&mut self.filter_map(|x| x.doc_comment().map(ToOwned::to_owned)), "\n");
            if docs.is_empty() {
                None
            } else {
                Some(docs)
            }
        }
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
        iter: syntax::ElemChildren,
    }
    impl AttrDocCommentIter {
        pub fn from_syntax_node(x: &syntax::Node) -> AttrDocCommentIter {
            AttrDocCommentIter {
                iter: x.children_with_tokens(),
            }
        }
    }
    impl Iterator for AttrDocCommentIter {
        type Item = Either<ast::Attr, ast::Comment>;
        fn next(&mut self) -> Option<Self::Item> {
            self.iter.by_ref().find_map(|el| match el {
                syntax::Elem::Node(node) => ast::Attr::cast(node).map(Either::Left),
                syntax::Elem::Token(tok) => ast::Comment::cast(tok).filter(ast::Comment::is_doc).map(Either::Right),
            })
        }
    }
    impl<A: HasName, B: HasName> HasName for Either<A, B> {}
}

pub trait Node {
    fn can_cast(kind: SyntaxKind) -> bool
    where
        Self: Sized;
    fn cast(syntax: syntax::Node) -> Option<Self>
    where
        Self: Sized;
    fn syntax(&self) -> &syntax::Node;
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
pub trait Token {
    fn can_cast(x: SyntaxKind) -> bool
    where
        Self: Sized;
    fn cast(x: syntax::Token) -> Option<Self>
    where
        Self: Sized;
    fn syntax(&self) -> &syntax::Token;
    fn text(&self) -> &str {
        self.syntax().text()
    }
}
#[derive(Debug, Clone)]
pub struct Children<N> {
    inner: syntax::NodeChildren,
    ph: PhantomData<N>,
}
impl<N> Children<N> {
    fn new(x: &syntax::Node) -> Self {
        Children {
            inner: x.children(),
            ph: PhantomData,
        }
    }
}
impl<N: Node> Iterator for Children<N> {
    type Item = N;
    fn next(&mut self) -> Option<N> {
        self.inner.find_map(N::cast)
    }
}
impl<L, R> Node for Either<L, R>
where
    L: Node,
    R: Node,
{
    fn can_cast(x: SyntaxKind) -> bool
    where
        Self: Sized,
    {
        L::can_cast(x) || R::can_cast(x)
    }
    fn cast(x: syntax::Node) -> Option<Self>
    where
        Self: Sized,
    {
        if L::can_cast(x.kind()) {
            L::cast(x).map(Either::Left)
        } else {
            R::cast(x).map(Either::Right)
        }
    }
    fn syntax(&self) -> &syntax::Node {
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
    use crate::{
        syntax::{self, ast},
        SyntaxKind,
    };
    pub fn child<N: ast::Node>(x: &syntax::Node) -> Option<N> {
        x.children().find_map(N::cast)
    }
    pub fn children<N: ast::Node>(x: &syntax::Node) -> ast::Children<N> {
        ast::Children::new(x)
    }
    pub fn token(x: &syntax::Node, kind: SyntaxKind) -> Option<syntax::Token> {
        x.children_with_tokens()
            .filter_map(|x| x.into_token())
            .find(|x| x.kind() == kind)
    }
}

#[test]
fn assert_ast_is_object_safe() {
    fn _f(_: &dyn Node, _: &dyn HasName) {}
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
