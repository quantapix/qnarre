#![cfg(not(syn_disable_nightly_tests))]
#![cfg(not(miri))]
#![recursion_limit = "1024"]
#![feature(rustc_private)]
#![allow(
    clippy::explicit_deref_methods,
    clippy::let_underscore_untyped,
    clippy::manual_assert,
    clippy::manual_let_else,
    clippy::match_like_matches_macro,
    clippy::match_wildcard_for_single_variants,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]
extern crate rustc_ast;
extern crate rustc_ast_pretty;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_span;
extern crate thin_vec;
use crate::common::eq::SpanlessEq;
use crate::common::parse;
use quote::quote;
use regex::Regex;
use rustc_ast::ast;
use rustc_ast::ptr::P;
use rustc_ast_pretty::pprust;
use rustc_span::edition::Edition;
use std::fs;
use std::path::Path;
use std::process;
use std::sync::atomic::{AtomicUsize, Ordering};
#[macro_use]
mod macros;
#[allow(dead_code)]
mod common;
mod repo;
#[test]
fn test_rustc_precedence() {
    common::rayon_init();
    repo::clone_rust();
    let abort_after = common::abort_after();
    if abort_after == 0 {
        panic!("Skipping all precedence tests");
    }
    let passed = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let edition_regex = Regex::new(r"\b(async|try)[!(]").unwrap();
    repo::for_each_rust_file(|path| {
        let content = fs::read_to_string(path).unwrap();
        let content = edition_regex.replace_all(&content, "_$0");
        let (l_passed, l_failed) = match syn::parse_file(&content) {
            Ok(file) => {
                let edition = repo::edition(path).parse().unwrap();
                let exprs = collect_exprs(file);
                let (l_passed, l_failed) = test_expressions(path, edition, exprs);
                errorf!("=== {}: {} passed | {} failed\n", path.display(), l_passed, l_failed,);
                (l_passed, l_failed)
            },
            Err(msg) => {
                errorf!("\nFAIL {} - syn failed to parse: {}\n", path.display(), msg);
                (0, 1)
            },
        };
        passed.fetch_add(l_passed, Ordering::Relaxed);
        let prev_failed = failed.fetch_add(l_failed, Ordering::Relaxed);
        if prev_failed + l_failed >= abort_after {
            process::exit(1);
        }
    });
    let passed = passed.load(Ordering::Relaxed);
    let failed = failed.load(Ordering::Relaxed);
    errorf!("\n===== Precedence Test Results =====\n");
    errorf!("{} passed | {} failed\n", passed, failed);
    if failed > 0 {
        panic!("{} failures", failed);
    }
}
fn test_expressions(path: &Path, edition: Edition, exprs: Vec<syn::Expr>) -> (usize, usize) {
    let mut passed = 0;
    let mut failed = 0;
    rustc_span::create_session_if_not_set_then(edition, |_| {
        for expr in exprs {
            let raw = quote!(#expr).to_string();
            let librustc_ast = if let Some(e) = librustc_parse_and_rewrite(&raw) {
                e
            } else {
                failed += 1;
                errorf!("\nFAIL {} - librustc failed to parse raw\n", path.display());
                continue;
            };
            let syn_expr = syn_brackets(expr);
            let syn_ast = if let Some(e) = parse::librustc_expr(&quote!(#syn_expr).to_string()) {
                e
            } else {
                failed += 1;
                errorf!("\nFAIL {} - librustc failed to parse bracketed\n", path.display(),);
                continue;
            };
            if SpanlessEq::eq(&syn_ast, &librustc_ast) {
                passed += 1;
            } else {
                failed += 1;
                let syn_program = pprust::expr_to_string(&syn_ast);
                let librustc_program = pprust::expr_to_string(&librustc_ast);
                errorf!(
                    "\nFAIL {}\n{}\nsyn != rustc\n{}\n",
                    path.display(),
                    syn_program,
                    librustc_program,
                );
            }
        }
    });
    (passed, failed)
}
fn librustc_parse_and_rewrite(input: &str) -> Option<P<ast::Expr>> {
    parse::librustc_expr(input).and_then(librustc_brackets)
}
fn librustc_brackets(mut librustc_expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
    use rustc_ast::ast::{
        patt::Patt, ty::Type, Attribute, BinOpKind, Block, BorrowKind, Expr, ExprField, ExprKind, GenericArg,
        GenericBound, Local, LocalKind, Stmt, StmtKind, StructExpr, StructRest, TraitBoundModifier,
    };
    use rustc_ast::mut_visit::{noop_visit_generic_arg, noop_visit_local, noop_visit_param_bound, MutVisitor};
    use rustc_data_structures::flat_map_in_place::FlatMapInPlace;
    use rustc_span::DUMMY_SP;
    use std::mem;
    use std::ops::DerefMut;
    use thin_vec::ThinVec;
    struct BracketsVisitor {
        failed: bool,
    }
    fn flat_map_field<T: MutVisitor>(mut f: ExprField, vis: &mut T) -> Vec<ExprField> {
        if f.is_shorthand {
            noop_visit_expr(&mut f.expr, vis);
        } else {
            vis.visit_expr(&mut f.expr);
        }
        vec![f]
    }
    fn flat_map_stmt<T: MutVisitor>(stmt: Stmt, vis: &mut T) -> Vec<Stmt> {
        let kind = match stmt.kind {
            StmtKind::Expr(mut e) => {
                noop_visit_expr(&mut e, vis);
                StmtKind::Expr(e)
            },
            StmtKind::Semi(mut e) => {
                noop_visit_expr(&mut e, vis);
                StmtKind::Semi(e)
            },
            s => s,
        };
        vec![Stmt { kind, ..stmt }]
    }
    fn noop_visit_expr<T: MutVisitor>(e: &mut Expr, vis: &mut T) {
        use rustc_ast::mut_visit::{noop_visit_expr, visit_attrs};
        match &mut e.kind {
            ExprKind::AddrOf(BorrowKind::Raw, ..) => {},
            ExprKind::Struct(expr) => {
                let StructExpr {
                    qself,
                    path,
                    fields,
                    rest,
                } = expr.deref_mut();
                vis.visit_qself(qself);
                vis.visit_path(path);
                fields.flat_map_in_place(|field| flat_map_field(field, vis));
                if let StructRest::Base(rest) = rest {
                    vis.visit_expr(rest);
                }
                vis.visit_id(&mut e.id);
                vis.visit_span(&mut e.span);
                visit_attrs(&mut e.attrs, vis);
            },
            _ => noop_visit_expr(e, vis),
        }
    }
    impl MutVisitor for BracketsVisitor {
        fn visit_expr(&mut self, e: &mut P<Expr>) {
            noop_visit_expr(e, self);
            match e.kind {
                ExprKind::Block(..) | ExprKind::If(..) | ExprKind::Let(..) => {},
                ExprKind::Binary(binop, ref left, ref right)
                    if match (&left.kind, binop.node, &right.kind) {
                        (ExprKind::Let(..), BinOpKind::And, _) | (_, BinOpKind::And, ExprKind::Let(..)) => true,
                        _ => false,
                    } => {},
                _ => {
                    let inner = mem::replace(
                        e,
                        P(Expr {
                            id: ast::DUMMY_NODE_ID,
                            kind: ExprKind::Err,
                            span: DUMMY_SP,
                            attrs: ThinVec::new(),
                            tokens: None,
                        }),
                    );
                    e.kind = ExprKind::Paren(inner);
                },
            }
        }
        fn visit_generic_arg(&mut self, arg: &mut GenericArg) {
            match arg {
                GenericArg::Const(anon_const) => {
                    if let ExprKind::Block(..) = &mut anon_const.value.kind {
                        noop_visit_expr(&mut anon_const.value, self);
                    }
                },
                _ => noop_visit_generic_arg(arg, self),
            }
        }
        fn visit_param_bound(&mut self, bound: &mut GenericBound) {
            match bound {
                GenericBound::Trait(_, TraitBoundModifier::MaybeConst | TraitBoundModifier::MaybeConstMaybe) => {},
                _ => noop_visit_param_bound(bound, self),
            }
        }
        fn visit_block(&mut self, block: &mut P<Block>) {
            self.visit_id(&mut block.id);
            block.stmts.flat_map_in_place(|stmt| flat_map_stmt(stmt, self));
            self.visit_span(&mut block.span);
        }
        fn visit_local(&mut self, local: &mut P<Local>) {
            match local.kind {
                LocalKind::InitElse(..) => {},
                _ => noop_visit_local(local, self),
            }
        }
        fn visit_pat(&mut self, pat: &mut P<patt::Patt>) {
            let _ = pat;
        }
        fn visit_ty(&mut self, ty: &mut P<ty::Type>) {
            let _ = ty;
        }
        fn visit_attribute(&mut self, attr: &mut Attribute) {
            let _ = attr;
        }
    }
    let mut folder = BracketsVisitor { failed: false };
    folder.visit_expr(&mut librustc_expr);
    if folder.failed {
        None
    } else {
        Some(librustc_expr)
    }
}
fn syn_brackets(syn_expr: syn::Expr) -> syn::Expr {
    use syn::fold::{fold_expr, fold_generic_argument, Fold};
    use syn::{patt::Patt, tok, ty::Type, Arg, BinOp, Expr, ExprParen, MetaNameValue, Stmt};
    struct ParenthesizeEveryExpr;
    fn needs_paren(expr: &Expr) -> bool {
        match expr {
            Expr::Group(_) => unreachable!(),
            Expr::If(_) | Expr::Unsafe(_) | Expr::Block(_) | Expr::Let(_) => false,
            Expr::Binary(bin) => match (&*bin.left, bin.op, &*bin.right) {
                (Expr::Let(_), BinOp::And(_), _) | (_, BinOp::And(_), Expr::Let(_)) => false,
                _ => true,
            },
            _ => true,
        }
    }
    impl Fold for ParenthesizeEveryExpr {
        fn fold_expr(&mut self, expr: Expr) -> Expr {
            if needs_paren(&expr) {
                Expr::Paren(ExprParen {
                    attrs: Vec::new(),
                    expr: Box::new(fold_expr(self, expr)),
                    paren: tok::Paren::default(),
                })
            } else {
                fold_expr(self, expr)
            }
        }
        fn fold_generic_argument(&mut self, arg: Arg) -> Arg {
            match arg {
                Arg::Const(arg) => Arg::Const(match arg {
                    Expr::Block(_) => fold_expr(self, arg),
                    _ => arg,
                }),
                _ => fold_generic_argument(self, arg),
            }
        }
        fn fold_stmt(&mut self, stmt: Stmt) -> Stmt {
            match stmt {
                Stmt::Expr(Expr::Verbatim(_), Some(_)) => stmt,
                Stmt::Expr(e, semi) => Stmt::Expr(fold_expr(self, e), semi),
                s => s,
            }
        }
        fn fold_meta_name_value(&mut self, x: MetaNameValue) -> MetaNameValue {
            x
        }
        fn fold_pat(&mut self, x: patt::Patt) -> patt::Patt {
            x
        }
        fn fold_type(&mut self, x: ty::Type) -> ty::Type {
            x
        }
    }
    let mut folder = ParenthesizeEveryExpr;
    folder.fold_expr(syn_expr)
}
fn collect_exprs(file: syn::File) -> Vec<syn::Expr> {
    use syn::fold::Fold;
    use syn::punct::Punctuated;
    use syn::{patt::Patt, tok, ConstParam, Expr, ExprTuple, Path};
    struct CollectExprs(Vec<Expr>);
    impl Fold for CollectExprs {
        fn fold_expr(&mut self, expr: Expr) -> Expr {
            match expr {
                Expr::Verbatim(_) => {},
                _ => self.0.push(expr),
            }
            Expr::Tuple(ExprTuple {
                attrs: vec![],
                elems: Punctuated::new(),
                paren: tok::Paren::default(),
            })
        }
        fn fold_pat(&mut self, pat: patt::Patt) -> patt::Patt {
            pat
        }
        fn fold_path(&mut self, path: Path) -> Path {
            path
        }
        fn fold_const_param(&mut self, const_param: ConstParam) -> ConstParam {
            const_param
        }
    }
    let mut folder = CollectExprs(vec![]);
    folder.fold_file(file);
    folder.0
}
