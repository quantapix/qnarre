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
                def.life.to_tokens(tokens);
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
        self.lifes.to_tokens(tokens);
        self.gt.to_tokens(tokens);
    }
}
impl ToTokens for LifetimeParam {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.life.to_tokens(tokens);
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
            self.lifes.to_tokens(tokens);
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
        self.life.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.bounds.to_tokens(tokens);
    }
}
impl ToTokens for PredType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.lifes.to_tokens(tokens);
        self.bounded.to_tokens(tokens);
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
        self.bracket.surround(tokens, |tokens| {
            self.elems.to_tokens(tokens);
        });
    }
}
impl ToTokens for ExprAssign {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.left.to_tokens(tokens);
        self.eq.to_tokens(tokens);
        self.right.to_tokens(tokens);
    }
}
impl ToTokens for ExprAsync {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.async_.to_tokens(tokens);
        self.capture.to_tokens(tokens);
        self.block.to_tokens(tokens);
    }
}
impl ToTokens for ExprAwait {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.base.to_tokens(tokens);
        self.dot.to_tokens(tokens);
        self.await_.to_tokens(tokens);
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
        self.break_.to_tokens(tokens);
        self.label.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for ExprCall {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.func.to_tokens(tokens);
        self.paren.surround(tokens, |tokens| {
            self.args.to_tokens(tokens);
        });
    }
}
impl ToTokens for ExprCast {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.as_.to_tokens(tokens);
        self.ty.to_tokens(tokens);
    }
}
impl ToTokens for ExprClosure {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.lifes.to_tokens(tokens);
        self.const_.to_tokens(tokens);
        self.static_.to_tokens(tokens);
        self.async_.to_tokens(tokens);
        self.move_.to_tokens(tokens);
        self.or1.to_tokens(tokens);
        self.inputs.to_tokens(tokens);
        self.or2.to_tokens(tokens);
        self.ret.to_tokens(tokens);
        self.body.to_tokens(tokens);
    }
}
impl ToTokens for ExprConst {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.const_.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for ExprContinue {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.continue_.to_tokens(tokens);
        self.label.to_tokens(tokens);
    }
}
impl ToTokens for ExprField {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.base.to_tokens(tokens);
        self.dot.to_tokens(tokens);
        self.member.to_tokens(tokens);
    }
}
impl ToTokens for ExprForLoop {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.label.to_tokens(tokens);
        self.for_.to_tokens(tokens);
        self.pat.to_tokens(tokens);
        self.in_.to_tokens(tokens);
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
        self.group.surround(tokens, |tokens| {
            self.expr.to_tokens(tokens);
        });
    }
}
impl ToTokens for ExprIf {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.if_.to_tokens(tokens);
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
        self.bracket.surround(tokens, |tokens| {
            self.index.to_tokens(tokens);
        });
    }
}
impl ToTokens for ExprInfer {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.underscore.to_tokens(tokens);
    }
}
impl ToTokens for ExprLet {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.let_.to_tokens(tokens);
        self.pat.to_tokens(tokens);
        self.eq.to_tokens(tokens);
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
        self.loop_.to_tokens(tokens);
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
        self.match_.to_tokens(tokens);
        wrap_bare_struct(tokens, &self.expr);
        self.brace.surround(tokens, |tokens| {
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
        self.dot.to_tokens(tokens);
        self.method.to_tokens(tokens);
        self.turbofish.to_tokens(tokens);
        self.paren.surround(tokens, |tokens| {
            self.args.to_tokens(tokens);
        });
    }
}
impl ToTokens for ExprParen {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.paren.surround(tokens, |tokens| {
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
        self.and.to_tokens(tokens);
        self.mut_.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for ExprRepeat {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.bracket.surround(tokens, |tokens| {
            self.expr.to_tokens(tokens);
            self.semi.to_tokens(tokens);
            self.len.to_tokens(tokens);
        });
    }
}
impl ToTokens for ExprReturn {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.return_.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for ExprStruct {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        print_path(tokens, &self.qself, &self.path);
        self.brace.surround(tokens, |tokens| {
            self.fields.to_tokens(tokens);
            if let Some(dot2) = &self.dot2 {
                dot2.to_tokens(tokens);
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
        self.question.to_tokens(tokens);
    }
}
impl ToTokens for ExprTryBlock {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.try_.to_tokens(tokens);
        self.block.to_tokens(tokens);
    }
}
impl ToTokens for ExprTuple {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.paren.surround(tokens, |tokens| {
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
        self.unsafe_.to_tokens(tokens);
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
        self.while_.to_tokens(tokens);
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
        self.yield_.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for Arm {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(&self.attrs);
        self.pat.to_tokens(tokens);
        if let Some((if_, guard)) = &self.guard {
            if_.to_tokens(tokens);
            guard.to_tokens(tokens);
        }
        self.fat_arrow.to_tokens(tokens);
        self.body.to_tokens(tokens);
        self.comma.to_tokens(tokens);
    }
}
impl ToTokens for FieldValue {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.member.to_tokens(tokens);
        if let Some(colon) = &self.colon {
            colon.to_tokens(tokens);
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
        self.colon.to_tokens(tokens);
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
        self.extern_.to_tokens(tokens);
        self.crate_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        if let Some((as_, rename)) = &self.rename {
            as_.to_tokens(tokens);
            rename.to_tokens(tokens);
        }
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for ItemUse {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.use_.to_tokens(tokens);
        self.leading_colon.to_tokens(tokens);
        self.tree.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for ItemStatic {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.static_.to_tokens(tokens);
        self.mut_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        self.eq.to_tokens(tokens);
        self.expr.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for ItemConst {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.const_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        self.eq.to_tokens(tokens);
        self.expr.to_tokens(tokens);
        self.semi.to_tokens(tokens);
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
        self.unsafe_.to_tokens(tokens);
        self.mod_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        if let Some((brace, items)) = &self.gist {
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
        self.unsafe_.to_tokens(tokens);
        self.abi.to_tokens(tokens);
        self.brace.surround(tokens, |tokens| {
            tokens.append_all(self.attrs.inner());
            tokens.append_all(&self.items);
        });
    }
}
impl ToTokens for ItemType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.type.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.gens.to_tokens(tokens);
        self.gens.clause.to_tokens(tokens);
        self.eq.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for ItemEnum {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.enum_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.gens.to_tokens(tokens);
        self.gens.clause.to_tokens(tokens);
        self.brace.surround(tokens, |tokens| {
            self.variants.to_tokens(tokens);
        });
    }
}
impl ToTokens for ItemStruct {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.struct_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.gens.to_tokens(tokens);
        match &self.fields {
            Fields::Named(fields) => {
                self.gens.clause.to_tokens(tokens);
                fields.to_tokens(tokens);
            },
            Fields::Unnamed(fields) => {
                fields.to_tokens(tokens);
                self.gens.clause.to_tokens(tokens);
                TokensOrDefault(&self.semi).to_tokens(tokens);
            },
            Fields::Unit => {
                self.gens.clause.to_tokens(tokens);
                TokensOrDefault(&self.semi).to_tokens(tokens);
            },
        }
    }
}
impl ToTokens for ItemUnion {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.union_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.gens.to_tokens(tokens);
        self.gens.clause.to_tokens(tokens);
        self.fields.to_tokens(tokens);
    }
}
impl ToTokens for ItemTrait {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.unsafe_.to_tokens(tokens);
        self.auto_.to_tokens(tokens);
        self.trait_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.gens.to_tokens(tokens);
        if !self.supertraits.is_empty() {
            TokensOrDefault(&self.colon).to_tokens(tokens);
            self.supertraits.to_tokens(tokens);
        }
        self.gens.clause.to_tokens(tokens);
        self.brace.surround(tokens, |tokens| {
            tokens.append_all(self.attrs.inner());
            tokens.append_all(&self.items);
        });
    }
}
impl ToTokens for ItemTraitAlias {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.trait_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.gens.to_tokens(tokens);
        self.eq.to_tokens(tokens);
        self.bounds.to_tokens(tokens);
        self.gens.clause.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for ItemImpl {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.default_.to_tokens(tokens);
        self.unsafe_.to_tokens(tokens);
        self.impl_.to_tokens(tokens);
        self.gens.to_tokens(tokens);
        if let Some((polarity, path, for_)) = &self.trait_ {
            polarity.to_tokens(tokens);
            path.to_tokens(tokens);
            for_.to_tokens(tokens);
        }
        self.self_ty.to_tokens(tokens);
        self.gens.clause.to_tokens(tokens);
        self.brace.surround(tokens, |tokens| {
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
            MacroDelim::Paren(paren) => {
                paren.surround(tokens, |tokens| self.mac.toks.to_tokens(tokens));
            },
            MacroDelim::Brace(brace) => {
                brace.surround(tokens, |tokens| self.mac.toks.to_tokens(tokens));
            },
            MacroDelim::Bracket(bracket) => {
                bracket.surround(tokens, |tokens| self.mac.toks.to_tokens(tokens));
            },
        }
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for UsePath {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ident.to_tokens(tokens);
        self.colon2.to_tokens(tokens);
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
        self.as_.to_tokens(tokens);
        self.rename.to_tokens(tokens);
    }
}
impl ToTokens for UseGlob {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.star.to_tokens(tokens);
    }
}
impl ToTokens for UseGroup {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.brace.surround(tokens, |tokens| {
            self.items.to_tokens(tokens);
        });
    }
}
impl ToTokens for TraitItemConst {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.const_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        if let Some((eq, default)) = &self.default {
            eq.to_tokens(tokens);
            default.to_tokens(tokens);
        }
        self.semi.to_tokens(tokens);
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
                TokensOrDefault(&self.semi).to_tokens(tokens);
            },
        }
    }
}
impl ToTokens for TraitItemType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.type.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.gens.to_tokens(tokens);
        if !self.bounds.is_empty() {
            TokensOrDefault(&self.colon).to_tokens(tokens);
            self.bounds.to_tokens(tokens);
        }
        if let Some((eq, default)) = &self.default {
            eq.to_tokens(tokens);
            default.to_tokens(tokens);
        }
        self.gens.clause.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for TraitItemMacro {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.mac.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for ImplItemConst {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.default_.to_tokens(tokens);
        self.const_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        self.eq.to_tokens(tokens);
        self.expr.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for ImplItemFn {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.default_.to_tokens(tokens);
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
        self.default_.to_tokens(tokens);
        self.type.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.gens.to_tokens(tokens);
        self.eq.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        self.gens.clause.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for ImplItemMacro {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.mac.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for ForeignItemFn {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.sig.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for ForeignItemStatic {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.static_.to_tokens(tokens);
        self.mut_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.ty.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for ForeignItemType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.type.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.gens.to_tokens(tokens);
        self.gens.clause.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for ForeignItemMacro {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.mac.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for Signature {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.constness.to_tokens(tokens);
        self.async_.to_tokens(tokens);
        self.unsafe_.to_tokens(tokens);
        self.abi.to_tokens(tokens);
        self.fn_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.gens.to_tokens(tokens);
        self.paren.surround(tokens, |tokens| {
            self.args.to_tokens(tokens);
            if let Some(vari) = &self.vari {
                if !self.args.empty_or_trailing() {
                    <Token![,]>::default().to_tokens(tokens);
                }
                vari.to_tokens(tokens);
            }
        });
        self.ret.to_tokens(tokens);
        self.gens.clause.to_tokens(tokens);
    }
}
impl ToTokens for Receiver {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        if let Some((ampersand, lifetime)) = &self.reference {
            ampersand.to_tokens(tokens);
            lifetime.to_tokens(tokens);
        }
        self.mut_.to_tokens(tokens);
        self.self_.to_tokens(tokens);
        if let Some(colon) = &self.colon {
            colon.to_tokens(tokens);
            self.ty.to_tokens(tokens);
        } else {
            let consistent = match (&self.reference, &self.mut_, &*self.ty) {
                (Some(_), mutability, Ty::Ref(ty)) => {
                    mutability.is_some() == ty.mut_.is_some()
                        && match &*ty.elem {
                            Ty::Path(ty) => ty.qself.is_none() && ty.path.is_ident("Self"),
                            _ => false,
                        }
                },
                (None, _, Ty::Path(ty)) => ty.qself.is_none() && ty.path.is_ident("Self"),
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
impl ToTokens for StaticMut {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            StaticMut::None => {},
            StaticMut::Mut(mut_token) => mut_token.to_tokens(tokens),
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
        if let Some((eq, disc)) = &self.discriminant {
            eq.to_tokens(tokens);
            disc.to_tokens(tokens);
        }
    }
}
impl ToTokens for FieldsNamed {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.brace.surround(tokens, |tokens| {
            self.named.to_tokens(tokens);
        });
    }
}
impl ToTokens for FieldsUnnamed {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.paren.surround(tokens, |tokens| {
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
            TokensOrDefault(&self.colon).to_tokens(tokens);
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
        self.gens.to_tokens(tokens);
        match &self.data {
            Data::Struct(data) => match &data.fields {
                Fields::Named(fields) => {
                    self.gens.clause.to_tokens(tokens);
                    fields.to_tokens(tokens);
                },
                Fields::Unnamed(fields) => {
                    fields.to_tokens(tokens);
                    self.gens.clause.to_tokens(tokens);
                    TokensOrDefault(&data.semi).to_tokens(tokens);
                },
                Fields::Unit => {
                    self.gens.clause.to_tokens(tokens);
                    TokensOrDefault(&data.semi).to_tokens(tokens);
                },
            },
            Data::Enum(data) => {
                self.gens.clause.to_tokens(tokens);
                data.brace.surround(tokens, |tokens| {
                    data.variants.to_tokens(tokens);
                });
            },
            Data::Union(data) => {
                self.gens.clause.to_tokens(tokens);
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

impl MacroDelim {
    pub fn surround(&self, tokens: &mut TokenStream, inner: TokenStream) {
        let (delim, span) = match self {
            MacroDelim::Paren(paren) => (Delimiter::Parenthesis, paren.span),
            MacroDelim::Brace(brace) => (Delimiter::Brace, brace.span),
            MacroDelim::Bracket(bracket) => (Delimiter::Bracket, bracket.span),
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
            Visibility::Public(pub_) => pub_.to_tokens(tokens),
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
        self.and.to_tokens(tokens);
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
        if let Some(colon) = &self.colon {
            self.member.to_tokens(tokens);
            colon.to_tokens(tokens);
        }
        self.pat.to_tokens(tokens);
    }
}

mod path {
    use crate::path::*;
    impl ToTokens for Path {
        fn to_tokens(&self, ts: &mut TokenStream) {
            self.colon.to_tokens(ts);
            self.segs.to_tokens(ts);
        }
    }
    impl ToTokens for Segment {
        fn to_tokens(&self, ts: &mut TokenStream) {
            self.ident.to_tokens(ts);
            self.args.to_tokens(ts);
        }
    }
    impl ToTokens for Args {
        fn to_tokens(&self, ts: &mut TokenStream) {
            match self {
                Args::None => {},
                Args::Angled(args) => {
                    args.to_tokens(ts);
                },
                Args::Parenthesized(args) => {
                    args.to_tokens(ts);
                },
            }
        }
    }
    impl ToTokens for Arg {
        #[allow(clippy::match_same_arms)]
        fn to_tokens(&self, ts: &mut TokenStream) {
            match self {
                Arg::Lifetime(x) => x.to_tokens(ts),
                Arg::Type(x) => x.to_tokens(ts),
                Arg::Const(x) => match x {
                    Expr::Lit(_) => x.to_tokens(ts),
                    Expr::Block(_) => x.to_tokens(ts),
                    _ => tok::Brace::default().surround(ts, |xs| {
                        x.to_tokens(xs);
                    }),
                },
                Arg::AssocType(x) => x.to_tokens(ts),
                Arg::AssocConst(x) => x.to_tokens(ts),
                Arg::Constraint(x) => x.to_tokens(ts),
            }
        }
    }
    impl ToTokens for AngledArgs {
        fn to_tokens(&self, ts: &mut TokenStream) {
            self.colon2.to_tokens(ts);
            self.lt.to_tokens(ts);
            let mut trailing_or_empty = true;
            for x in self.args.pairs() {
                match x.value() {
                    Arg::Lifetime(_) => {
                        x.to_tokens(ts);
                        trailing_or_empty = x.punct().is_some();
                    },
                    Arg::Type(_) | Arg::Const(_) | Arg::AssocType(_) | Arg::AssocConst(_) | Arg::Constraint(_) => {},
                }
            }
            for x in self.args.pairs() {
                match x.value() {
                    Arg::Type(_) | Arg::Const(_) | Arg::AssocType(_) | Arg::AssocConst(_) | Arg::Constraint(_) => {
                        if !trailing_or_empty {
                            <Token![,]>::default().to_tokens(ts);
                        }
                        x.to_tokens(ts);
                        trailing_or_empty = x.punct().is_some();
                    },
                    Arg::Lifetime(_) => {},
                }
            }
            self.gt.to_tokens(ts);
        }
    }
    impl ToTokens for AssocType {
        fn to_tokens(&self, ts: &mut TokenStream) {
            self.ident.to_tokens(ts);
            self.args.to_tokens(ts);
            self.eq.to_tokens(ts);
            self.ty.to_tokens(ts);
        }
    }
    impl ToTokens for AssocConst {
        fn to_tokens(&self, ts: &mut TokenStream) {
            self.ident.to_tokens(ts);
            self.args.to_tokens(ts);
            self.eq.to_tokens(ts);
            self.val.to_tokens(ts);
        }
    }
    impl ToTokens for Constraint {
        fn to_tokens(&self, ts: &mut TokenStream) {
            self.ident.to_tokens(ts);
            self.args.to_tokens(ts);
            self.colon.to_tokens(ts);
            self.bounds.to_tokens(ts);
        }
    }
    impl ToTokens for ParenthesizedArgs {
        fn to_tokens(&self, ts: &mut TokenStream) {
            self.paren.surround(ts, |xs| {
                self.args.to_tokens(xs);
            });
            self.ret.to_tokens(ts);
        }
    }
    pub(crate) fn print_path(ts: &mut TokenStream, qself: &Option<QSelf>, path: &Path) {
        let qself = match qself {
            Some(qself) => qself,
            None => {
                path.to_tokens(ts);
                return;
            },
        };
        qself.lt.to_tokens(ts);
        qself.ty.to_tokens(ts);
        let pos = cmp::min(qself.pos, path.segs.len());
        let mut segments = path.segs.pairs();
        if pos > 0 {
            TokensOrDefault(&qself.as_).to_tokens(ts);
            path.colon.to_tokens(ts);
            for (i, segment) in segments.by_ref().take(pos).enumerate() {
                if i + 1 == pos {
                    segment.value().to_tokens(ts);
                    qself.gt.to_tokens(ts);
                    segment.punct().to_tokens(ts);
                } else {
                    segment.to_tokens(ts);
                }
            }
        } else {
            qself.gt.to_tokens(ts);
            path.colon.to_tokens(ts);
        }
        for segment in segments {
            segment.to_tokens(ts);
        }
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
pub fn keyword(x: &str, s: Span, xs: &mut TokenStream) {
    xs.append(Ident::new(x, s));
}
pub fn delim(d: Delimiter, s: Span, xs: &mut TokenStream, inner: TokenStream) {
    let mut g = Group::new(d, inner);
    g.set_span(s);
    xs.append(g);
}

mod ty {
    use crate::ty::*;
    impl ToTokens for Slice {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.bracket.surround(xs, |ys| {
                self.elem.to_tokens(ys);
            });
        }
    }
    impl ToTokens for Array {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.bracket.surround(xs, |ys| {
                self.elem.to_tokens(ys);
                self.semi.to_tokens(ys);
                self.len.to_tokens(ys);
            });
        }
    }
    impl ToTokens for Ptr {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.star.to_tokens(xs);
            match &self.mut_ {
                Some(x) => x.to_tokens(xs),
                None => {
                    TokensOrDefault(&self.const_).to_tokens(xs);
                },
            }
            self.elem.to_tokens(xs);
        }
    }
    impl ToTokens for Ref {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.and.to_tokens(xs);
            self.life.to_tokens(xs);
            self.mut_.to_tokens(xs);
            self.elem.to_tokens(xs);
        }
    }
    impl ToTokens for BareFn {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.lifes.to_tokens(xs);
            self.unsafe_.to_tokens(xs);
            self.abi.to_tokens(xs);
            self.fn_.to_tokens(xs);
            self.paren.surround(xs, |ys| {
                self.args.to_tokens(ys);
                if let Some(x) = &self.vari {
                    if !self.args.empty_or_trailing() {
                        let s = x.dots.spans[0];
                        Token![,](s).to_tokens(ys);
                    }
                    x.to_tokens(ys);
                }
            });
            self.ret.to_tokens(xs);
        }
    }
    impl ToTokens for Never {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.bang.to_tokens(xs);
        }
    }
    impl ToTokens for Tuple {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.paren.surround(xs, |ys| {
                self.elems.to_tokens(ys);
                if self.elems.len() == 1 && !self.elems.trailing_punct() {
                    <Token![,]>::default().to_tokens(ys);
                }
            });
        }
    }
    impl ToTokens for Path {
        fn to_tokens(&self, xs: &mut TokenStream) {
            print_path(xs, &self.qself, &self.path);
        }
    }
    impl ToTokens for TraitObj {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.dyn_.to_tokens(xs);
            self.bounds.to_tokens(xs);
        }
    }
    impl ToTokens for Impl {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.impl_.to_tokens(xs);
            self.bounds.to_tokens(xs);
        }
    }
    impl ToTokens for Group {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.group.surround(xs, |ys| {
                self.elem.to_tokens(ys);
            });
        }
    }
    impl ToTokens for Paren {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.paren.surround(xs, |ys| {
                self.elem.to_tokens(ys);
            });
        }
    }
    impl ToTokens for Infer {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.underscore.to_tokens(xs);
        }
    }
    impl ToTokens for Mac {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.mac.to_tokens(xs);
        }
    }
    impl ToTokens for Ret {
        fn to_tokens(&self, xs: &mut TokenStream) {
            match self {
                Ret::Type(arrow, ty) => {
                    arrow.to_tokens(xs);
                    ty.to_tokens(xs);
                },
                Ret::Default => {},
            }
        }
    }
    impl ToTokens for BareFnArg {
        fn to_tokens(&self, xs: &mut TokenStream) {
            xs.append_all(self.attrs.outer());
            if let Some((name, colon)) = &self.name {
                name.to_tokens(xs);
                colon.to_tokens(xs);
            }
            self.ty.to_tokens(xs);
        }
    }
    impl ToTokens for BareVari {
        fn to_tokens(&self, xs: &mut TokenStream) {
            xs.append_all(self.attrs.outer());
            if let Some((name, colon)) = &self.name {
                name.to_tokens(xs);
                colon.to_tokens(xs);
            }
            self.dots.to_tokens(xs);
            self.comma.to_tokens(xs);
        }
    }
    impl ToTokens for Abi {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.extern_.to_tokens(xs);
            self.name.to_tokens(xs);
        }
    }
}

mod printing {
    use super::*;
    use proc_macro2::TokenStream;
    use quote::{ToTokens, TokenStreamExt};
    impl<T, P> ToTokens for Punctuated<T, P>
    where
        T: ToTokens,
        P: ToTokens,
    {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            tokens.append_all(self.pairs());
        }
    }
    impl<T, P> ToTokens for Pair<T, P>
    where
        T: ToTokens,
        P: ToTokens,
    {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            match self {
                Pair::Punctuated(a, b) => {
                    a.to_tokens(tokens);
                    b.to_tokens(tokens);
                },
                Pair::End(a) => a.to_tokens(tokens),
            }
        }
    }
}
