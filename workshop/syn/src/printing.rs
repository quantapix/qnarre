use super::*;
use proc_macro2::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream};
use quote::{ToTokens, TokenStreamExt};
use std::cmp;

impl ToTokens for Attribute {
    fn to_tokens(&self, xs: &mut TokenStream) {
        self.pound.to_tokens(xs);
        if let AttrStyle::Inner(x) = &self.style {
            x.to_tokens(xs);
        }
        self.bracket.surround(xs, |x| {
            self.meta.to_tokens(x);
        });
    }
}
impl ToTokens for MetaList {
    fn to_tokens(&self, xs: &mut TokenStream) {
        self.path.to_tokens(xs);
        self.delim.surround(xs, self.toks.clone());
    }
}
impl ToTokens for MetaNameValue {
    fn to_tokens(&self, xs: &mut TokenStream) {
        self.path.to_tokens(xs);
        self.eq.to_tokens(xs);
        self.val.to_tokens(xs);
    }
}

impl ToTokens for Generics {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if self.params.is_empty() {
            return;
        }
        TokensOrDefault(&self.lt).to_tokens(tokens);
        let mut trailing_or_empty = true;
        for param in self.params.pairs() {
            if let GenericParam::Lifetime(_) = **param.value() {
                param.to_tokens(tokens);
                trailing_or_empty = param.punct().is_some();
            }
        }
        for param in self.params.pairs() {
            match param.value() {
                GenericParam::Type(_) | GenericParam::Const(_) => {
                    if !trailing_or_empty {
                        <Token![,]>::default().to_tokens(tokens);
                        trailing_or_empty = true;
                    }
                    param.to_tokens(tokens);
                },
                GenericParam::Lifetime(_) => {},
            }
        }
        TokensOrDefault(&self.gt).to_tokens(tokens);
    }
}
impl<'a> ToTokens for ImplGenerics<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if self.0.params.is_empty() {
            return;
        }
        TokensOrDefault(&self.0.lt).to_tokens(tokens);
        let mut trailing_or_empty = true;
        for param in self.0.params.pairs() {
            if let GenericParam::Lifetime(_) = **param.value() {
                param.to_tokens(tokens);
                trailing_or_empty = param.punct().is_some();
            }
        }
        for param in self.0.params.pairs() {
            if let GenericParam::Lifetime(_) = **param.value() {
                continue;
            }
            if !trailing_or_empty {
                <Token![,]>::default().to_tokens(tokens);
                trailing_or_empty = true;
            }
            match param.value() {
                GenericParam::Lifetime(_) => unreachable!(),
                GenericParam::Type(param) => {
                    tokens.append_all(param.attrs.outer());
                    param.ident.to_tokens(tokens);
                    if !param.bounds.is_empty() {
                        TokensOrDefault(&param.colon).to_tokens(tokens);
                        param.bounds.to_tokens(tokens);
                    }
                },
                GenericParam::Const(param) => {
                    tokens.append_all(param.attrs.outer());
                    param.const_.to_tokens(tokens);
                    param.ident.to_tokens(tokens);
                    param.colon.to_tokens(tokens);
                    param.ty.to_tokens(tokens);
                },
            }
            param.punct().to_tokens(tokens);
        }
        TokensOrDefault(&self.0.gt).to_tokens(tokens);
    }
}
impl<'a> ToTokens for TypeGenerics<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if self.0.params.is_empty() {
            return;
        }
        TokensOrDefault(&self.0.lt).to_tokens(tokens);
        let mut trailing_or_empty = true;
        for param in self.0.params.pairs() {
            if let GenericParam::Lifetime(def) = *param.value() {
                def.lifetime.to_tokens(tokens);
                param.punct().to_tokens(tokens);
                trailing_or_empty = param.punct().is_some();
            }
        }
        for param in self.0.params.pairs() {
            if let GenericParam::Lifetime(_) = **param.value() {
                continue;
            }
            if !trailing_or_empty {
                <Token![,]>::default().to_tokens(tokens);
                trailing_or_empty = true;
            }
            match param.value() {
                GenericParam::Lifetime(_) => unreachable!(),
                GenericParam::Type(param) => {
                    param.ident.to_tokens(tokens);
                },
                GenericParam::Const(param) => {
                    param.ident.to_tokens(tokens);
                },
            }
            param.punct().to_tokens(tokens);
        }
        TokensOrDefault(&self.0.gt).to_tokens(tokens);
    }
}
impl<'a> ToTokens for Turbofish<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if !self.0.params.is_empty() {
            <Token![::]>::default().to_tokens(tokens);
            TypeGenerics(self.0).to_tokens(tokens);
        }
    }
}
impl ToTokens for BoundLifetimes {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.for_.to_tokens(tokens);
        self.lt.to_tokens(tokens);
        self.lifetimes.to_tokens(tokens);
        self.gt.to_tokens(tokens);
    }
}
impl ToTokens for LifetimeParam {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.lifetime.to_tokens(tokens);
        if !self.bounds.is_empty() {
            TokensOrDefault(&self.colon).to_tokens(tokens);
            self.bounds.to_tokens(tokens);
        }
    }
}
impl ToTokens for TypeParam {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.ident.to_tokens(tokens);
        if !self.bounds.is_empty() {
            TokensOrDefault(&self.colon).to_tokens(tokens);
            self.bounds.to_tokens(tokens);
        }
        if let Some(default) = &self.default {
            TokensOrDefault(&self.eq).to_tokens(tokens);
            default.to_tokens(tokens);
        }
    }
}
impl ToTokens for TraitBound {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let to_tokens = |tokens: &mut TokenStream| {
            self.modifier.to_tokens(tokens);
            self.lifetimes.to_tokens(tokens);
            self.path.to_tokens(tokens);
        };
        match &self.paren {
            Some(paren) => paren.surround(tokens, to_tokens),
            None => to_tokens(tokens),
        }
    }
}
impl ToTokens for TraitBoundModifier {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            TraitBoundModifier::None => {},
            TraitBoundModifier::Maybe(t) => t.to_tokens(tokens),
        }
    }
}
impl ToTokens for ConstParam {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.const_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        if let Some(default) = &self.default {
            TokensOrDefault(&self.eq).to_tokens(tokens);
            default.to_tokens(tokens);
        }
    }
}
impl ToTokens for WhereClause {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if !self.preds.is_empty() {
            self.where_.to_tokens(tokens);
            self.preds.to_tokens(tokens);
        }
    }
}
impl ToTokens for PredLifetime {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.lifetime.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.bounds.to_tokens(tokens);
    }
}
impl ToTokens for PredType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.lifetimes.to_tokens(tokens);
        self.bounded_ty.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.bounds.to_tokens(tokens);
    }
}

fn wrap_bare_struct(tokens: &mut TokenStream, e: &Expr) {
    if let Expr::Struct(_) = *e {
        tok::Paren::default().surround(tokens, |tokens| {
            e.to_tokens(tokens);
        });
    } else {
        e.to_tokens(tokens);
    }
}
pub(crate) fn outer_attrs_to_tokens(attrs: &[Attribute], tokens: &mut TokenStream) {
    tokens.append_all(attrs.outer());
}
fn inner_attrs_to_tokens(attrs: &[Attribute], tokens: &mut TokenStream) {
    tokens.append_all(attrs.inner());
}
#[cfg(not(feature = "full"))]
pub(crate) fn outer_attrs_to_tokens(_attrs: &[Attribute], _tokens: &mut TokenStream) {}
impl ToTokens for ExprArray {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.bracket_token.surround(tokens, |tokens| {
            self.elems.to_tokens(tokens);
        });
    }
}
impl ToTokens for ExprAssign {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.left.to_tokens(tokens);
        self.eq_token.to_tokens(tokens);
        self.right.to_tokens(tokens);
    }
}
impl ToTokens for ExprAsync {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.async_token.to_tokens(tokens);
        self.capture.to_tokens(tokens);
        self.block.to_tokens(tokens);
    }
}
impl ToTokens for ExprAwait {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.base.to_tokens(tokens);
        self.dot_token.to_tokens(tokens);
        self.await_token.to_tokens(tokens);
    }
}
impl ToTokens for ExprBinary {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.left.to_tokens(tokens);
        self.op.to_tokens(tokens);
        self.right.to_tokens(tokens);
    }
}
impl ToTokens for ExprBlock {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.label.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for ExprBreak {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.break_token.to_tokens(tokens);
        self.label.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for ExprCall {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.func.to_tokens(tokens);
        self.paren_token.surround(tokens, |tokens| {
            self.args.to_tokens(tokens);
        });
    }
}
impl ToTokens for ExprCast {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.as_token.to_tokens(tokens);
        self.ty.to_tokens(tokens);
    }
}
impl ToTokens for ExprClosure {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.lifetimes.to_tokens(tokens);
        self.constness.to_tokens(tokens);
        self.movability.to_tokens(tokens);
        self.asyncness.to_tokens(tokens);
        self.capture.to_tokens(tokens);
        self.or1_token.to_tokens(tokens);
        self.inputs.to_tokens(tokens);
        self.or2_token.to_tokens(tokens);
        self.output.to_tokens(tokens);
        self.body.to_tokens(tokens);
    }
}
impl ToTokens for ExprConst {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.const_token.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for ExprContinue {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.continue_token.to_tokens(tokens);
        self.label.to_tokens(tokens);
    }
}
impl ToTokens for ExprField {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.base.to_tokens(tokens);
        self.dot_token.to_tokens(tokens);
        self.member.to_tokens(tokens);
    }
}
impl ToTokens for ExprForLoop {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.label.to_tokens(tokens);
        self.for_token.to_tokens(tokens);
        self.pat.to_tokens(tokens);
        self.in_token.to_tokens(tokens);
        wrap_bare_struct(tokens, &self.expr);
        self.body.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.body.stmts);
        });
    }
}
impl ToTokens for ExprGroup {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.group_token.surround(tokens, |tokens| {
            self.expr.to_tokens(tokens);
        });
    }
}
impl ToTokens for ExprIf {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.if_token.to_tokens(tokens);
        wrap_bare_struct(tokens, &self.cond);
        self.then_branch.to_tokens(tokens);
        if let Some((else_token, else_)) = &self.else_branch {
            else_token.to_tokens(tokens);
            match **else_ {
                Expr::If(_) | Expr::Block(_) => else_.to_tokens(tokens),
                _ => tok::Brace::default().surround(tokens, |tokens| else_.to_tokens(tokens)),
            }
        }
    }
}
impl ToTokens for ExprIndex {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.bracket_token.surround(tokens, |tokens| {
            self.index.to_tokens(tokens);
        });
    }
}
impl ToTokens for ExprInfer {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.underscore_token.to_tokens(tokens);
    }
}
impl ToTokens for ExprLet {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.let_token.to_tokens(tokens);
        self.pat.to_tokens(tokens);
        self.eq_token.to_tokens(tokens);
        wrap_bare_struct(tokens, &self.expr);
    }
}
impl ToTokens for ExprLit {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.lit.to_tokens(tokens);
    }
}
impl ToTokens for ExprLoop {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.label.to_tokens(tokens);
        self.loop_token.to_tokens(tokens);
        self.body.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.body.stmts);
        });
    }
}
impl ToTokens for ExprMacro {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.mac.to_tokens(tokens);
    }
}
impl ToTokens for ExprMatch {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.match_token.to_tokens(tokens);
        wrap_bare_struct(tokens, &self.expr);
        self.brace_token.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            for (i, arm) in self.arms.iter().enumerate() {
                arm.to_tokens(tokens);
                let is_last = i == self.arms.len() - 1;
                if !is_last && requires_terminator(&arm.body) && arm.comma.is_none() {
                    <Token![,]>::default().to_tokens(tokens);
                }
            }
        });
    }
}
impl ToTokens for ExprMethodCall {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.receiver.to_tokens(tokens);
        self.dot_token.to_tokens(tokens);
        self.method.to_tokens(tokens);
        self.turbofish.to_tokens(tokens);
        self.paren_token.surround(tokens, |tokens| {
            self.args.to_tokens(tokens);
        });
    }
}
impl ToTokens for ExprParen {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.paren_token.surround(tokens, |tokens| {
            self.expr.to_tokens(tokens);
        });
    }
}
impl ToTokens for ExprPath {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        print_path(tokens, &self.qself, &self.path);
    }
}
impl ToTokens for ExprRange {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.start.to_tokens(tokens);
        self.limits.to_tokens(tokens);
        self.end.to_tokens(tokens);
    }
}
impl ToTokens for ExprReference {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.and_token.to_tokens(tokens);
        self.mutability.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for ExprRepeat {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.bracket_token.surround(tokens, |tokens| {
            self.expr.to_tokens(tokens);
            self.semi_token.to_tokens(tokens);
            self.len.to_tokens(tokens);
        });
    }
}
impl ToTokens for ExprReturn {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.return_token.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for ExprStruct {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        print_path(tokens, &self.qself, &self.path);
        self.brace_token.surround(tokens, |tokens| {
            self.fields.to_tokens(tokens);
            if let Some(dot2_token) = &self.dot2_token {
                dot2_token.to_tokens(tokens);
            } else if self.rest.is_some() {
                Token![..](Span::call_site()).to_tokens(tokens);
            }
            self.rest.to_tokens(tokens);
        });
    }
}
impl ToTokens for ExprTry {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.question_token.to_tokens(tokens);
    }
}
impl ToTokens for ExprTryBlock {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.try_token.to_tokens(tokens);
        self.block.to_tokens(tokens);
    }
}
impl ToTokens for ExprTuple {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.paren_token.surround(tokens, |tokens| {
            self.elems.to_tokens(tokens);
            if self.elems.len() == 1 && !self.elems.trailing_punct() {
                <Token![,]>::default().to_tokens(tokens);
            }
        });
    }
}
impl ToTokens for ExprUnary {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.op.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for ExprUnsafe {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.unsafe_token.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for ExprWhile {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.label.to_tokens(tokens);
        self.while_token.to_tokens(tokens);
        wrap_bare_struct(tokens, &self.cond);
        self.body.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.body.stmts);
        });
    }
}
impl ToTokens for ExprYield {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.yield_token.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for Arm {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(&self.attrs);
        self.pat.to_tokens(tokens);
        if let Some((if_token, guard)) = &self.guard {
            if_token.to_tokens(tokens);
            guard.to_tokens(tokens);
        }
        self.fat_arrow_token.to_tokens(tokens);
        self.body.to_tokens(tokens);
        self.comma.to_tokens(tokens);
    }
}
impl ToTokens for FieldValue {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.member.to_tokens(tokens);
        if let Some(colon_token) = &self.colon_token {
            colon_token.to_tokens(tokens);
            self.expr.to_tokens(tokens);
        }
    }
}
impl ToTokens for Index {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let mut lit = Literal::i64_unsuffixed(i64::from(self.index));
        lit.set_span(self.span);
        tokens.append(lit);
    }
}
impl ToTokens for Label {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.name.to_tokens(tokens);
        self.colon_token.to_tokens(tokens);
    }
}
impl ToTokens for Member {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Member::Named(ident) => ident.to_tokens(tokens),
            Member::Unnamed(index) => index.to_tokens(tokens),
        }
    }
}
impl ToTokens for RangeLimits {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            RangeLimits::HalfOpen(t) => t.to_tokens(tokens),
            RangeLimits::Closed(t) => t.to_tokens(tokens),
        }
    }
}

impl ToTokens for ItemExternCrate {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.extern_token.to_tokens(tokens);
        self.crate_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        if let Some((as_token, rename)) = &self.rename {
            as_token.to_tokens(tokens);
            rename.to_tokens(tokens);
        }
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for ItemUse {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.use_token.to_tokens(tokens);
        self.leading_colon.to_tokens(tokens);
        self.tree.to_tokens(tokens);
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for ItemStatic {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.static_token.to_tokens(tokens);
        self.mutability.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.colon_token.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        self.eq_token.to_tokens(tokens);
        self.expr.to_tokens(tokens);
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for ItemConst {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.const_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.colon_token.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        self.eq_token.to_tokens(tokens);
        self.expr.to_tokens(tokens);
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for ItemFn {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.sig.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            tokens.append_all(self.attrs.inner());
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for ItemMod {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.unsafety.to_tokens(tokens);
        self.mod_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        if let Some((brace, items)) = &self.content {
            brace.surround(tokens, |tokens| {
                tokens.append_all(self.attrs.inner());
                tokens.append_all(items);
            });
        } else {
            TokensOrDefault(&self.semi).to_tokens(tokens);
        }
    }
}
impl ToTokens for ItemForeignMod {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.unsafety.to_tokens(tokens);
        self.abi.to_tokens(tokens);
        self.brace_token.surround(tokens, |tokens| {
            tokens.append_all(self.attrs.inner());
            tokens.append_all(&self.items);
        });
    }
}
impl ToTokens for ItemType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.type_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        self.generics.clause.to_tokens(tokens);
        self.eq_token.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for ItemEnum {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.enum_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        self.generics.clause.to_tokens(tokens);
        self.brace_token.surround(tokens, |tokens| {
            self.variants.to_tokens(tokens);
        });
    }
}
impl ToTokens for ItemStruct {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.struct_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        match &self.fields {
            Fields::Named(fields) => {
                self.generics.clause.to_tokens(tokens);
                fields.to_tokens(tokens);
            },
            Fields::Unnamed(fields) => {
                fields.to_tokens(tokens);
                self.generics.clause.to_tokens(tokens);
                TokensOrDefault(&self.semi_token).to_tokens(tokens);
            },
            Fields::Unit => {
                self.generics.clause.to_tokens(tokens);
                TokensOrDefault(&self.semi_token).to_tokens(tokens);
            },
        }
    }
}
impl ToTokens for ItemUnion {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.union_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        self.generics.clause.to_tokens(tokens);
        self.fields.to_tokens(tokens);
    }
}
impl ToTokens for ItemTrait {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.unsafety.to_tokens(tokens);
        self.auto_token.to_tokens(tokens);
        self.trait_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        if !self.supertraits.is_empty() {
            TokensOrDefault(&self.colon_token).to_tokens(tokens);
            self.supertraits.to_tokens(tokens);
        }
        self.generics.clause.to_tokens(tokens);
        self.brace_token.surround(tokens, |tokens| {
            tokens.append_all(self.attrs.inner());
            tokens.append_all(&self.items);
        });
    }
}
impl ToTokens for ItemTraitAlias {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.trait_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        self.eq_token.to_tokens(tokens);
        self.bounds.to_tokens(tokens);
        self.generics.clause.to_tokens(tokens);
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for ItemImpl {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.defaultness.to_tokens(tokens);
        self.unsafety.to_tokens(tokens);
        self.impl_token.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        if let Some((polarity, path, for_token)) = &self.trait_ {
            polarity.to_tokens(tokens);
            path.to_tokens(tokens);
            for_token.to_tokens(tokens);
        }
        self.self_ty.to_tokens(tokens);
        self.generics.clause.to_tokens(tokens);
        self.brace_token.surround(tokens, |tokens| {
            tokens.append_all(self.attrs.inner());
            tokens.append_all(&self.items);
        });
    }
}
impl ToTokens for ItemMacro {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.mac.path.to_tokens(tokens);
        self.mac.bang.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        match &self.mac.delim {
            MacroDelimiter::Paren(paren) => {
                paren.surround(tokens, |tokens| self.mac.toks.to_tokens(tokens));
            },
            MacroDelimiter::Brace(brace) => {
                brace.surround(tokens, |tokens| self.mac.toks.to_tokens(tokens));
            },
            MacroDelimiter::Bracket(bracket) => {
                bracket.surround(tokens, |tokens| self.mac.toks.to_tokens(tokens));
            },
        }
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for UsePath {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ident.to_tokens(tokens);
        self.colon2_token.to_tokens(tokens);
        self.tree.to_tokens(tokens);
    }
}
impl ToTokens for UseName {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ident.to_tokens(tokens);
    }
}
impl ToTokens for UseRename {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ident.to_tokens(tokens);
        self.as_token.to_tokens(tokens);
        self.rename.to_tokens(tokens);
    }
}
impl ToTokens for UseGlob {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.star_token.to_tokens(tokens);
    }
}
impl ToTokens for UseGroup {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.brace_token.surround(tokens, |tokens| {
            self.items.to_tokens(tokens);
        });
    }
}
impl ToTokens for TraitItemConst {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.const_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.colon_token.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        if let Some((eq_token, default)) = &self.default {
            eq_token.to_tokens(tokens);
            default.to_tokens(tokens);
        }
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for TraitItemFn {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.sig.to_tokens(tokens);
        match &self.default {
            Some(block) => {
                block.brace.surround(tokens, |tokens| {
                    tokens.append_all(self.attrs.inner());
                    tokens.append_all(&block.stmts);
                });
            },
            None => {
                TokensOrDefault(&self.semi_token).to_tokens(tokens);
            },
        }
    }
}
impl ToTokens for TraitItemType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.type_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        if !self.bounds.is_empty() {
            TokensOrDefault(&self.colon_token).to_tokens(tokens);
            self.bounds.to_tokens(tokens);
        }
        if let Some((eq_token, default)) = &self.default {
            eq_token.to_tokens(tokens);
            default.to_tokens(tokens);
        }
        self.generics.clause.to_tokens(tokens);
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for TraitItemMacro {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.mac.to_tokens(tokens);
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for ImplItemConst {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.defaultness.to_tokens(tokens);
        self.const_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.colon_token.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        self.eq_token.to_tokens(tokens);
        self.expr.to_tokens(tokens);
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for ImplItemFn {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.defaultness.to_tokens(tokens);
        self.sig.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            tokens.append_all(self.attrs.inner());
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for ImplItemType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.defaultness.to_tokens(tokens);
        self.type_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        self.eq_token.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        self.generics.clause.to_tokens(tokens);
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for ImplItemMacro {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.mac.to_tokens(tokens);
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for ForeignItemFn {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.sig.to_tokens(tokens);
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for ForeignItemStatic {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.static_token.to_tokens(tokens);
        self.mutability.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.colon_token.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for ForeignItemType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.type_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        self.generics.clause.to_tokens(tokens);
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for ForeignItemMacro {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.mac.to_tokens(tokens);
        self.semi_token.to_tokens(tokens);
    }
}
impl ToTokens for Signature {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.constness.to_tokens(tokens);
        self.asyncness.to_tokens(tokens);
        self.unsafety.to_tokens(tokens);
        self.abi.to_tokens(tokens);
        self.fn_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        self.paren_token.surround(tokens, |tokens| {
            self.inputs.to_tokens(tokens);
            if let Some(variadic) = &self.variadic {
                if !self.inputs.empty_or_trailing() {
                    <Token![,]>::default().to_tokens(tokens);
                }
                variadic.to_tokens(tokens);
            }
        });
        self.output.to_tokens(tokens);
        self.generics.clause.to_tokens(tokens);
    }
}
impl ToTokens for Receiver {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        if let Some((ampersand, lifetime)) = &self.reference {
            ampersand.to_tokens(tokens);
            lifetime.to_tokens(tokens);
        }
        self.mutability.to_tokens(tokens);
        self.self_token.to_tokens(tokens);
        if let Some(colon_token) = &self.colon_token {
            colon_token.to_tokens(tokens);
            self.ty.to_tokens(tokens);
        } else {
            let consistent = match (&self.reference, &self.mutability, &*self.ty) {
                (Some(_), mutability, Type::Reference(ty)) => {
                    mutability.is_some() == ty.mut_.is_some()
                        && match &*ty.elem {
                            Type::Path(ty) => ty.qself.is_none() && ty.path.is_ident("Self"),
                            _ => false,
                        }
                },
                (None, _, Type::Path(ty)) => ty.qself.is_none() && ty.path.is_ident("Self"),
                _ => false,
            };
            if !consistent {
                <Token![:]>::default().to_tokens(tokens);
                self.ty.to_tokens(tokens);
            }
        }
    }
}
impl ToTokens for Variadic {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        if let Some((pat, colon)) = &self.pat {
            pat.to_tokens(tokens);
            colon.to_tokens(tokens);
        }
        self.dots.to_tokens(tokens);
        self.comma.to_tokens(tokens);
    }
}
impl ToTokens for StaticMutability {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            StaticMutability::None => {},
            StaticMutability::Mut(mut_token) => mut_token.to_tokens(tokens),
        }
    }
}

impl ToTokens for Block {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.brace.surround(tokens, |tokens| {
            tokens.append_all(&self.stmts);
        });
    }
}
impl ToTokens for Stmt {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Stmt::Local(local) => local.to_tokens(tokens),
            Stmt::Item(item) => item.to_tokens(tokens),
            Stmt::Expr(expr, semi) => {
                expr.to_tokens(tokens);
                semi.to_tokens(tokens);
            },
            Stmt::Macro(mac) => mac.to_tokens(tokens),
        }
    }
}
impl ToTokens for Local {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.let_.to_tokens(tokens);
        self.pat.to_tokens(tokens);
        if let Some(init) = &self.init {
            init.eq.to_tokens(tokens);
            init.expr.to_tokens(tokens);
            if let Some((else_token, diverge)) = &init.diverge {
                else_token.to_tokens(tokens);
                diverge.to_tokens(tokens);
            }
        }
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for StmtMacro {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.mac.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}

impl ToTokens for Variant {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(&self.attrs);
        self.ident.to_tokens(tokens);
        self.fields.to_tokens(tokens);
        if let Some((eq_token, disc)) = &self.discriminant {
            eq_token.to_tokens(tokens);
            disc.to_tokens(tokens);
        }
    }
}
impl ToTokens for FieldsNamed {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.brace_token.surround(tokens, |tokens| {
            self.named.to_tokens(tokens);
        });
    }
}
impl ToTokens for FieldsUnnamed {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.paren_token.surround(tokens, |tokens| {
            self.unnamed.to_tokens(tokens);
        });
    }
}
impl ToTokens for Field {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(&self.attrs);
        self.vis.to_tokens(tokens);
        if let Some(ident) = &self.ident {
            ident.to_tokens(tokens);
            TokensOrDefault(&self.colon_token).to_tokens(tokens);
        }
        self.ty.to_tokens(tokens);
    }
}

impl ToTokens for DeriveInput {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        for attr in self.attrs.outer() {
            attr.to_tokens(tokens);
        }
        self.vis.to_tokens(tokens);
        match &self.data {
            Data::Struct(d) => d.struct_.to_tokens(tokens),
            Data::Enum(d) => d.enum_.to_tokens(tokens),
            Data::Union(d) => d.union_.to_tokens(tokens),
        }
        self.ident.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        match &self.data {
            Data::Struct(data) => match &data.fields {
                Fields::Named(fields) => {
                    self.generics.clause.to_tokens(tokens);
                    fields.to_tokens(tokens);
                },
                Fields::Unnamed(fields) => {
                    fields.to_tokens(tokens);
                    self.generics.clause.to_tokens(tokens);
                    TokensOrDefault(&data.semi).to_tokens(tokens);
                },
                Fields::Unit => {
                    self.generics.clause.to_tokens(tokens);
                    TokensOrDefault(&data.semi).to_tokens(tokens);
                },
            },
            Data::Enum(data) => {
                self.generics.clause.to_tokens(tokens);
                data.brace.surround(tokens, |tokens| {
                    data.variants.to_tokens(tokens);
                });
            },
            Data::Union(data) => {
                self.generics.clause.to_tokens(tokens);
                data.fields.to_tokens(tokens);
            },
        }
    }
}

impl ToTokens for File {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.inner());
        tokens.append_all(&self.items);
    }
}

impl ToTokens for Lifetime {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let mut apostrophe = Punct::new('\'', Spacing::Joint);
        apostrophe.set_span(self.apostrophe);
        tokens.append(apostrophe);
        self.ident.to_tokens(tokens);
    }
}

impl MacroDelimiter {
    pub fn surround(&self, tokens: &mut TokenStream, inner: TokenStream) {
        let (delim, span) = match self {
            MacroDelimiter::Paren(paren) => (Delimiter::Parenthesis, paren.span),
            MacroDelimiter::Brace(brace) => (Delimiter::Brace, brace.span),
            MacroDelimiter::Bracket(bracket) => (Delimiter::Bracket, bracket.span),
        };
        delim(delim, span.join(), tokens, inner);
    }
}
impl ToTokens for Macro {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.path.to_tokens(tokens);
        self.bang.to_tokens(tokens);
        self.delim.surround(tokens, self.toks.clone());
    }
}

impl ToTokens for BinOp {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            BinOp::Add(t) => t.to_tokens(tokens),
            BinOp::Sub(t) => t.to_tokens(tokens),
            BinOp::Mul(t) => t.to_tokens(tokens),
            BinOp::Div(t) => t.to_tokens(tokens),
            BinOp::Rem(t) => t.to_tokens(tokens),
            BinOp::And(t) => t.to_tokens(tokens),
            BinOp::Or(t) => t.to_tokens(tokens),
            BinOp::BitXor(t) => t.to_tokens(tokens),
            BinOp::BitAnd(t) => t.to_tokens(tokens),
            BinOp::BitOr(t) => t.to_tokens(tokens),
            BinOp::Shl(t) => t.to_tokens(tokens),
            BinOp::Shr(t) => t.to_tokens(tokens),
            BinOp::Eq(t) => t.to_tokens(tokens),
            BinOp::Lt(t) => t.to_tokens(tokens),
            BinOp::Le(t) => t.to_tokens(tokens),
            BinOp::Ne(t) => t.to_tokens(tokens),
            BinOp::Ge(t) => t.to_tokens(tokens),
            BinOp::Gt(t) => t.to_tokens(tokens),
            BinOp::AddAssign(t) => t.to_tokens(tokens),
            BinOp::SubAssign(t) => t.to_tokens(tokens),
            BinOp::MulAssign(t) => t.to_tokens(tokens),
            BinOp::DivAssign(t) => t.to_tokens(tokens),
            BinOp::RemAssign(t) => t.to_tokens(tokens),
            BinOp::BitXorAssign(t) => t.to_tokens(tokens),
            BinOp::BitAndAssign(t) => t.to_tokens(tokens),
            BinOp::BitOrAssign(t) => t.to_tokens(tokens),
            BinOp::ShlAssign(t) => t.to_tokens(tokens),
            BinOp::ShrAssign(t) => t.to_tokens(tokens),
        }
    }
}
impl ToTokens for UnOp {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            UnOp::Deref(t) => t.to_tokens(tokens),
            UnOp::Not(t) => t.to_tokens(tokens),
            UnOp::Neg(t) => t.to_tokens(tokens),
        }
    }
}

impl ToTokens for Visibility {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Visibility::Public(pub_token) => pub_token.to_tokens(tokens),
            Visibility::Restricted(vis_restricted) => vis_restricted.to_tokens(tokens),
            Visibility::Inherited => {},
        }
    }
}
impl ToTokens for VisRestricted {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.pub_.to_tokens(tokens);
        self.paren.surround(tokens, |tokens| {
            self.in_.to_tokens(tokens);
            self.path.to_tokens(tokens);
        });
    }
}

impl ToTokens for lit::Str {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.repr.tok.to_tokens(tokens);
    }
}
impl ToTokens for lit::ByteStr {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.repr.tok.to_tokens(tokens);
    }
}
impl ToTokens for lit::Byte {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.repr.tok.to_tokens(tokens);
    }
}
impl ToTokens for lit::Char {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.repr.tok.to_tokens(tokens);
    }
}
impl ToTokens for lit::Int {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.repr.tok.to_tokens(tokens);
    }
}
impl ToTokens for lit::Float {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.repr.tok.to_tokens(tokens);
    }
}
impl ToTokens for lit::Bool {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append(self.token());
    }
}

impl ToTokens for PatIdent {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.ref_.to_tokens(tokens);
        self.mut_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        if let Some((at_token, subpat)) = &self.subpat {
            at_token.to_tokens(tokens);
            subpat.to_tokens(tokens);
        }
    }
}
impl ToTokens for PatOr {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.leading_vert.to_tokens(tokens);
        self.cases.to_tokens(tokens);
    }
}
impl ToTokens for PatParen {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.paren.surround(tokens, |tokens| {
            self.pat.to_tokens(tokens);
        });
    }
}
impl ToTokens for PatReference {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.and_.to_tokens(tokens);
        self.mutability.to_tokens(tokens);
        self.pat.to_tokens(tokens);
    }
}
impl ToTokens for PatRest {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.dot2.to_tokens(tokens);
    }
}
impl ToTokens for PatSlice {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.bracket.surround(tokens, |tokens| {
            self.elems.to_tokens(tokens);
        });
    }
}
impl ToTokens for PatStruct {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        print_path(tokens, &self.qself, &self.path);
        self.brace.surround(tokens, |tokens| {
            self.fields.to_tokens(tokens);
            if !self.fields.empty_or_trailing() && self.rest.is_some() {
                <Token![,]>::default().to_tokens(tokens);
            }
            self.rest.to_tokens(tokens);
        });
    }
}
impl ToTokens for PatTuple {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.paren.surround(tokens, |tokens| {
            self.elems.to_tokens(tokens);
        });
    }
}
impl ToTokens for PatTupleStruct {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        print_path(tokens, &self.qself, &self.path);
        self.paren.surround(tokens, |tokens| {
            self.elems.to_tokens(tokens);
        });
    }
}
impl ToTokens for PatType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.pat.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.ty.to_tokens(tokens);
    }
}
impl ToTokens for PatWild {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.underscore.to_tokens(tokens);
    }
}
impl ToTokens for FieldPat {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        if let Some(colon_token) = &self.colon {
            self.member.to_tokens(tokens);
            colon_token.to_tokens(tokens);
        }
        self.pat.to_tokens(tokens);
    }
}

impl ToTokens for Path {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.leading_colon.to_tokens(tokens);
        self.segments.to_tokens(tokens);
    }
}
impl ToTokens for PathSegment {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ident.to_tokens(tokens);
        self.arguments.to_tokens(tokens);
    }
}
impl ToTokens for PathArguments {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            PathArguments::None => {},
            PathArguments::AngleBracketed(arguments) => {
                arguments.to_tokens(tokens);
            },
            PathArguments::Parenthesized(arguments) => {
                arguments.to_tokens(tokens);
            },
        }
    }
}
impl ToTokens for GenericArgument {
    #[allow(clippy::match_same_arms)]
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            GenericArgument::Lifetime(lt) => lt.to_tokens(tokens),
            GenericArgument::Type(ty) => ty.to_tokens(tokens),
            GenericArgument::Const(expr) => match expr {
                Expr::Lit(_) => expr.to_tokens(tokens),
                Expr::Block(_) => expr.to_tokens(tokens),
                _ => tok::Brace::default().surround(tokens, |tokens| {
                    expr.to_tokens(tokens);
                }),
            },
            GenericArgument::AssocType(assoc) => assoc.to_tokens(tokens),
            GenericArgument::AssocConst(assoc) => assoc.to_tokens(tokens),
            GenericArgument::Constraint(constraint) => constraint.to_tokens(tokens),
        }
    }
}
impl ToTokens for AngleBracketedGenericArguments {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.colon2_token.to_tokens(tokens);
        self.lt.to_tokens(tokens);
        let mut trailing_or_empty = true;
        for param in self.args.pairs() {
            match param.value() {
                GenericArgument::Lifetime(_) => {
                    param.to_tokens(tokens);
                    trailing_or_empty = param.punct().is_some();
                },
                GenericArgument::Type(_)
                | GenericArgument::Const(_)
                | GenericArgument::AssocType(_)
                | GenericArgument::AssocConst(_)
                | GenericArgument::Constraint(_) => {},
            }
        }
        for param in self.args.pairs() {
            match param.value() {
                GenericArgument::Type(_)
                | GenericArgument::Const(_)
                | GenericArgument::AssocType(_)
                | GenericArgument::AssocConst(_)
                | GenericArgument::Constraint(_) => {
                    if !trailing_or_empty {
                        <Token![,]>::default().to_tokens(tokens);
                    }
                    param.to_tokens(tokens);
                    trailing_or_empty = param.punct().is_some();
                },
                GenericArgument::Lifetime(_) => {},
            }
        }
        self.gt.to_tokens(tokens);
    }
}
impl ToTokens for AssocType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ident.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        self.eq.to_tokens(tokens);
        self.ty.to_tokens(tokens);
    }
}
impl ToTokens for AssocConst {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ident.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        self.eq.to_tokens(tokens);
        self.value.to_tokens(tokens);
    }
}
impl ToTokens for Constraint {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ident.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.bounds.to_tokens(tokens);
    }
}
impl ToTokens for ParenthesizedGenericArguments {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.paren.surround(tokens, |tokens| {
            self.inputs.to_tokens(tokens);
        });
        self.output.to_tokens(tokens);
    }
}
pub(crate) fn print_path(tokens: &mut TokenStream, qself: &Option<QSelf>, path: &Path) {
    let qself = match qself {
        Some(qself) => qself,
        None => {
            path.to_tokens(tokens);
            return;
        },
    };
    qself.lt.to_tokens(tokens);
    qself.ty.to_tokens(tokens);
    let pos = cmp::min(qself.position, path.segments.len());
    let mut segments = path.segments.pairs();
    if pos > 0 {
        TokensOrDefault(&qself.as_).to_tokens(tokens);
        path.leading_colon.to_tokens(tokens);
        for (i, segment) in segments.by_ref().take(pos).enumerate() {
            if i + 1 == pos {
                segment.value().to_tokens(tokens);
                qself.gt_.to_tokens(tokens);
                segment.punct().to_tokens(tokens);
            } else {
                segment.to_tokens(tokens);
            }
        }
    } else {
        qself.gt_.to_tokens(tokens);
        path.leading_colon.to_tokens(tokens);
    }
    for segment in segments {
        segment.to_tokens(tokens);
    }
}

pub fn punct(s: &str, spans: &[Span], tokens: &mut TokenStream) {
    assert_eq!(s.len(), spans.len());
    let mut chars = s.chars();
    let mut spans = spans.iter();
    let ch = chars.next_back().unwrap();
    let span = spans.next_back().unwrap();
    for (ch, span) in chars.zip(spans) {
        let mut op = Punct::new(ch, Spacing::Joint);
        op.set_span(*span);
        tokens.append(op);
    }
    let mut op = Punct::new(ch, Spacing::Alone);
    op.set_span(*span);
    tokens.append(op);
}
pub fn keyword(s: &str, span: Span, tokens: &mut TokenStream) {
    tokens.append(Ident::new(s, span));
}
pub fn delim(delim: Delimiter, span: Span, tokens: &mut TokenStream, inner: TokenStream) {
    let mut g = Group::new(delim, inner);
    g.set_span(span);
    tokens.append(g);
}

impl ToTokens for TypeSlice {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.bracket.surround(tokens, |tokens| {
            self.elem.to_tokens(tokens);
        });
    }
}
impl ToTokens for TypeArray {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.bracket_token.surround(tokens, |tokens| {
            self.elem.to_tokens(tokens);
            self.semi.to_tokens(tokens);
            self.len.to_tokens(tokens);
        });
    }
}
impl ToTokens for TypePtr {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.star.to_tokens(tokens);
        match &self.mut_ {
            Some(tok) => tok.to_tokens(tokens),
            None => {
                TokensOrDefault(&self.const_).to_tokens(tokens);
            },
        }
        self.elem.to_tokens(tokens);
    }
}
impl ToTokens for TypeReference {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.and_.to_tokens(tokens);
        self.lifetime.to_tokens(tokens);
        self.mut_.to_tokens(tokens);
        self.elem.to_tokens(tokens);
    }
}
impl ToTokens for TypeBareFn {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.lifetimes.to_tokens(tokens);
        self.unsafe_.to_tokens(tokens);
        self.abi.to_tokens(tokens);
        self.fn_.to_tokens(tokens);
        self.paren.surround(tokens, |tokens| {
            self.inputs.to_tokens(tokens);
            if let Some(variadic) = &self.variadic {
                if !self.inputs.empty_or_trailing() {
                    let span = variadic.dots.spans[0];
                    Token![,](span).to_tokens(tokens);
                }
                variadic.to_tokens(tokens);
            }
        });
        self.output.to_tokens(tokens);
    }
}
impl ToTokens for TypeNever {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.bang.to_tokens(tokens);
    }
}
impl ToTokens for TypeTuple {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.paren.surround(tokens, |tokens| {
            self.elems.to_tokens(tokens);
            if self.elems.len() == 1 && !self.elems.trailing_punct() {
                <Token![,]>::default().to_tokens(tokens);
            }
        });
    }
}
impl ToTokens for TypePath {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        print_path(tokens, &self.qself, &self.path);
    }
}
impl ToTokens for TypeTraitObject {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.dyn_.to_tokens(tokens);
        self.bounds.to_tokens(tokens);
    }
}
impl ToTokens for TypeImplTrait {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.impl_.to_tokens(tokens);
        self.bounds.to_tokens(tokens);
    }
}
impl ToTokens for TypeGroup {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.group.surround(tokens, |tokens| {
            self.elem.to_tokens(tokens);
        });
    }
}
impl ToTokens for TypeParen {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.paren_token.surround(tokens, |tokens| {
            self.elem.to_tokens(tokens);
        });
    }
}
impl ToTokens for TypeInfer {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.underscore.to_tokens(tokens);
    }
}
impl ToTokens for TypeMacro {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.mac.to_tokens(tokens);
    }
}
impl ToTokens for ReturnType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            ReturnType::Default => {},
            ReturnType::Type(arrow, ty) => {
                arrow.to_tokens(tokens);
                ty.to_tokens(tokens);
            },
        }
    }
}
impl ToTokens for BareFnArg {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        if let Some((name, colon)) = &self.name {
            name.to_tokens(tokens);
            colon.to_tokens(tokens);
        }
        self.ty.to_tokens(tokens);
    }
}
impl ToTokens for BareVariadic {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        if let Some((name, colon)) = &self.name {
            name.to_tokens(tokens);
            colon.to_tokens(tokens);
        }
        self.dots.to_tokens(tokens);
        self.comma.to_tokens(tokens);
    }
}
impl ToTokens for Abi {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.extern_.to_tokens(tokens);
        self.name.to_tokens(tokens);
    }
}
