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
        self.expr.to_tokens(xs);
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
                    param.typ.to_tokens(tokens);
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
            TraitBoundModifier::Maybe(x) => x.to_tokens(tokens),
        }
    }
}
impl ToTokens for ConstParam {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append_all(self.attrs.outer());
        self.const_.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.typ.to_tokens(tokens);
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
impl ToTokens for expr::Array {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.bracket.surround(tokens, |tokens| {
            self.elems.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Assign {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.left.to_tokens(tokens);
        self.eq.to_tokens(tokens);
        self.right.to_tokens(tokens);
    }
}
impl ToTokens for expr::Async {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.async_.to_tokens(tokens);
        self.move_.to_tokens(tokens);
        self.block.to_tokens(tokens);
    }
}
impl ToTokens for expr::Await {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.dot.to_tokens(tokens);
        self.await_.to_tokens(tokens);
    }
}
impl ToTokens for expr::Binary {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.left.to_tokens(tokens);
        self.op.to_tokens(tokens);
        self.right.to_tokens(tokens);
    }
}
impl ToTokens for expr::Block {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.label.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for expr::Break {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.break_.to_tokens(tokens);
        self.label.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for expr::Call {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.func.to_tokens(tokens);
        self.paren.surround(tokens, |tokens| {
            self.args.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Cast {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.as_.to_tokens(tokens);
        self.typ.to_tokens(tokens);
    }
}
impl ToTokens for expr::Closure {
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
impl ToTokens for expr::Const {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.const_.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for expr::Continue {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.continue_.to_tokens(tokens);
        self.label.to_tokens(tokens);
    }
}
impl ToTokens for expr::Field {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.base.to_tokens(tokens);
        self.dot.to_tokens(tokens);
        self.memb.to_tokens(tokens);
    }
}
impl ToTokens for expr::ForLoop {
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
impl ToTokens for expr::Group {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.group.surround(tokens, |tokens| {
            self.expr.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::If {
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
impl ToTokens for expr::Index {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.bracket.surround(tokens, |tokens| {
            self.index.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Infer {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.underscore.to_tokens(tokens);
    }
}
impl ToTokens for expr::Let {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.let_.to_tokens(tokens);
        self.pat.to_tokens(tokens);
        self.eq.to_tokens(tokens);
        wrap_bare_struct(tokens, &self.expr);
    }
}
impl ToTokens for expr::Lit {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.lit.to_tokens(tokens);
    }
}
impl ToTokens for expr::Loop {
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
impl ToTokens for expr::Mac {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.mac.to_tokens(tokens);
    }
}
impl ToTokens for expr::Match {
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
impl ToTokens for expr::MethodCall {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.dot.to_tokens(tokens);
        self.method.to_tokens(tokens);
        self.turbofish.to_tokens(tokens);
        self.paren.surround(tokens, |tokens| {
            self.args.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Paren {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.paren.surround(tokens, |tokens| {
            self.expr.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Path {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        print_path(tokens, &self.qself, &self.path);
    }
}
impl ToTokens for expr::Range {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.beg.to_tokens(tokens);
        self.limits.to_tokens(tokens);
        self.end.to_tokens(tokens);
    }
}
impl ToTokens for expr::Ref {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.and.to_tokens(tokens);
        self.mut_.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for expr::Repeat {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.bracket.surround(tokens, |tokens| {
            self.expr.to_tokens(tokens);
            self.semi.to_tokens(tokens);
            self.len.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Return {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.return_.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for expr::Struct {
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
impl ToTokens for expr::Try {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.question.to_tokens(tokens);
    }
}
impl ToTokens for expr::TryBlock {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.try_.to_tokens(tokens);
        self.block.to_tokens(tokens);
    }
}
impl ToTokens for expr::Tuple {
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
impl ToTokens for expr::Unary {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.op.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for expr::Unsafe {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.unsafe_.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for expr::While {
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
impl ToTokens for expr::Yield {
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
    fn to_tokens(&self, xs: &mut TokenStream) {
        match self {
            RangeLimits::HalfOpen(x) => x.to_tokens(xs),
            RangeLimits::Closed(x) => x.to_tokens(xs),
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
        self.typ.to_tokens(tokens);
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
        self.typ.to_tokens(tokens);
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
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.type_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        self.gens.where_.to_tokens(xs);
        self.eq.to_tokens(xs);
        self.typ.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for ItemEnum {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.enum_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        self.gens.where_.to_tokens(xs);
        self.brace.surround(xs, |ys| {
            self.variants.to_tokens(ys);
        });
    }
}
impl ToTokens for ItemStruct {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.struct_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        match &self.fields {
            Fields::Named(fields) => {
                self.gens.where_.to_tokens(xs);
                fields.to_tokens(xs);
            },
            Fields::Unnamed(fields) => {
                fields.to_tokens(xs);
                self.gens.where_.to_tokens(xs);
                TokensOrDefault(&self.semi).to_tokens(xs);
            },
            Fields::Unit => {
                self.gens.where_.to_tokens(xs);
                TokensOrDefault(&self.semi).to_tokens(xs);
            },
        }
    }
}
impl ToTokens for ItemUnion {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.union_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        self.gens.where_.to_tokens(xs);
        self.fields.to_tokens(xs);
    }
}
impl ToTokens for ItemTrait {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.unsafe_.to_tokens(xs);
        self.auto_.to_tokens(xs);
        self.trait_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        if !self.supertraits.is_empty() {
            TokensOrDefault(&self.colon).to_tokens(xs);
            self.supertraits.to_tokens(xs);
        }
        self.gens.where_.to_tokens(xs);
        self.brace.surround(xs, |ys| {
            ys.append_all(self.attrs.inner());
            ys.append_all(&self.items);
        });
    }
}
impl ToTokens for ItemTraitAlias {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.trait_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        self.eq.to_tokens(xs);
        self.bounds.to_tokens(xs);
        self.gens.where_.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for ItemImpl {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.default_.to_tokens(xs);
        self.unsafe_.to_tokens(xs);
        self.impl_.to_tokens(xs);
        self.gens.to_tokens(xs);
        if let Some((polarity, path, for_)) = &self.trait_ {
            polarity.to_tokens(xs);
            path.to_tokens(xs);
            for_.to_tokens(xs);
        }
        self.typ.to_tokens(xs);
        self.gens.where_.to_tokens(xs);
        self.brace.surround(xs, |ys| {
            ys.append_all(self.attrs.inner());
            ys.append_all(&self.items);
        });
    }
}
impl ToTokens for ItemMacro {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.mac.path.to_tokens(xs);
        self.mac.bang.to_tokens(xs);
        self.ident.to_tokens(xs);
        match &self.mac.delim {
            MacroDelim::Paren(x) => {
                x.surround(xs, |ys| self.mac.toks.to_tokens(ys));
            },
            MacroDelim::Brace(x) => {
                x.surround(xs, |ys| self.mac.toks.to_tokens(ys));
            },
            MacroDelim::Bracket(x) => {
                x.surround(xs, |ys| self.mac.toks.to_tokens(ys));
            },
        }
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for UsePath {
    fn to_tokens(&self, xs: &mut TokenStream) {
        self.ident.to_tokens(xs);
        self.colon2.to_tokens(xs);
        self.tree.to_tokens(xs);
    }
}
impl ToTokens for UseName {
    fn to_tokens(&self, xs: &mut TokenStream) {
        self.ident.to_tokens(xs);
    }
}
impl ToTokens for UseRename {
    fn to_tokens(&self, xs: &mut TokenStream) {
        self.ident.to_tokens(xs);
        self.as_.to_tokens(xs);
        self.rename.to_tokens(xs);
    }
}
impl ToTokens for UseGlob {
    fn to_tokens(&self, xs: &mut TokenStream) {
        self.star.to_tokens(xs);
    }
}
impl ToTokens for UseGroup {
    fn to_tokens(&self, xs: &mut TokenStream) {
        self.brace.surround(xs, |ys| {
            self.items.to_tokens(ys);
        });
    }
}
impl ToTokens for TraitItemConst {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.const_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.colon.to_tokens(xs);
        self.typ.to_tokens(xs);
        if let Some((eq, default)) = &self.default {
            eq.to_tokens(xs);
            default.to_tokens(xs);
        }
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for TraitItemFn {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.sig.to_tokens(xs);
        match &self.default {
            Some(block) => {
                block.brace.surround(xs, |ys| {
                    ys.append_all(self.attrs.inner());
                    ys.append_all(&block.stmts);
                });
            },
            None => {
                TokensOrDefault(&self.semi).to_tokens(xs);
            },
        }
    }
}
impl ToTokens for TraitItemType {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.type_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        if !self.bounds.is_empty() {
            TokensOrDefault(&self.colon).to_tokens(xs);
            self.bounds.to_tokens(xs);
        }
        if let Some((eq, default)) = &self.default {
            eq.to_tokens(xs);
            default.to_tokens(xs);
        }
        self.gens.where_.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for TraitItemMacro {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.mac.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for ImplItemConst {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.default_.to_tokens(xs);
        self.const_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.colon.to_tokens(xs);
        self.typ.to_tokens(xs);
        self.eq.to_tokens(xs);
        self.expr.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for ImplItemFn {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.default_.to_tokens(xs);
        self.sig.to_tokens(xs);
        self.block.brace.surround(xs, |ys| {
            ys.append_all(self.attrs.inner());
            ys.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for ImplItemType {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.default_.to_tokens(xs);
        self.type_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        self.eq.to_tokens(xs);
        self.typ.to_tokens(xs);
        self.gens.where_.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for ImplItemMacro {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.mac.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for ForeignItemFn {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.sig.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for ForeignItemStatic {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.static_.to_tokens(xs);
        self.mut_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.colon.to_tokens(xs);
        self.typ.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for ForeignItemType {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.type_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        self.gens.where_.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for ForeignItemMacro {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        self.mac.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for Signature {
    fn to_tokens(&self, xs: &mut TokenStream) {
        self.constness.to_tokens(xs);
        self.async_.to_tokens(xs);
        self.unsafe_.to_tokens(xs);
        self.abi.to_tokens(xs);
        self.fn_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        self.paren.surround(xs, |ys| {
            self.args.to_tokens(ys);
            if let Some(vari) = &self.vari {
                if !self.args.empty_or_trailing() {
                    <Token![,]>::default().to_tokens(ys);
                }
                vari.to_tokens(ys);
            }
        });
        self.ret.to_tokens(xs);
        self.gens.where_.to_tokens(xs);
    }
}
impl ToTokens for Receiver {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        if let Some((ampersand, lifetime)) = &self.reference {
            ampersand.to_tokens(xs);
            lifetime.to_tokens(xs);
        }
        self.mut_.to_tokens(xs);
        self.self_.to_tokens(xs);
        if let Some(colon) = &self.colon {
            colon.to_tokens(xs);
            self.typ.to_tokens(xs);
        } else {
            let consistent = match (&self.reference, &self.mut_, &*self.typ) {
                (Some(_), mutability, ty::Type::Ref(ty)) => {
                    mutability.is_some() == ty.mut_.is_some()
                        && match &*ty.elem {
                            ty::Type::Path(ty) => ty.qself.is_none() && ty.path.is_ident("Self"),
                            _ => false,
                        }
                },
                (None, _, ty::Type::Path(ty)) => ty.qself.is_none() && ty.path.is_ident("Self"),
                _ => false,
            };
            if !consistent {
                <Token![:]>::default().to_tokens(xs);
                self.typ.to_tokens(xs);
            }
        }
    }
}
impl ToTokens for Variadic {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.outer());
        if let Some((pat, colon)) = &self.pat {
            pat.to_tokens(xs);
            colon.to_tokens(xs);
        }
        self.dots.to_tokens(xs);
        self.comma.to_tokens(xs);
    }
}
impl ToTokens for StaticMut {
    fn to_tokens(&self, xs: &mut TokenStream) {
        match self {
            StaticMut::None => {},
            StaticMut::Mut(mut_token) => mut_token.to_tokens(xs),
        }
    }
}

impl ToTokens for Block {
    fn to_tokens(&self, xs: &mut TokenStream) {
        self.brace.surround(xs, |ys| {
            ys.append_all(&self.stmts);
        });
    }
}
impl ToTokens for stmt::Stmt {
    fn to_tokens(&self, xs: &mut TokenStream) {
        match self {
            stmt::Stmt::stmt::Local(x) => x.to_tokens(xs),
            stmt::Stmt::Item(x) => x.to_tokens(xs),
            stmt::Stmt::Expr(x, semi) => {
                x.to_tokens(xs);
                semi.to_tokens(xs);
            },
            stmt::Stmt::Mac(x) => x.to_tokens(xs),
        }
    }
}
impl ToTokens for stmt::Local {
    fn to_tokens(&self, xs: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, xs);
        self.let_.to_tokens(xs);
        self.pat.to_tokens(xs);
        if let Some(init) = &self.init {
            init.eq.to_tokens(xs);
            init.expr.to_tokens(xs);
            if let Some((else_token, diverge)) = &init.diverge {
                else_token.to_tokens(xs);
                diverge.to_tokens(xs);
            }
        }
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for stmt::Mac {
    fn to_tokens(&self, xs: &mut TokenStream) {
        outer_attrs_to_tokens(&self.attrs, xs);
        self.mac.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}

impl ToTokens for Variant {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(&self.attrs);
        self.ident.to_tokens(xs);
        self.fields.to_tokens(xs);
        if let Some((eq, disc)) = &self.discriminant {
            eq.to_tokens(xs);
            disc.to_tokens(xs);
        }
    }
}
impl ToTokens for FieldsNamed {
    fn to_tokens(&self, xs: &mut TokenStream) {
        self.brace.surround(xs, |ys| {
            self.named.to_tokens(ys);
        });
    }
}
impl ToTokens for FieldsUnnamed {
    fn to_tokens(&self, xs: &mut TokenStream) {
        self.paren.surround(xs, |ys| {
            self.unnamed.to_tokens(ys);
        });
    }
}
impl ToTokens for Field {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(&self.attrs);
        self.vis.to_tokens(xs);
        if let Some(ident) = &self.ident {
            ident.to_tokens(xs);
            TokensOrDefault(&self.colon).to_tokens(xs);
        }
        self.typ.to_tokens(xs);
    }
}

impl ToTokens for DeriveInput {
    fn to_tokens(&self, xs: &mut TokenStream) {
        for attr in self.attrs.outer() {
            attr.to_tokens(xs);
        }
        self.vis.to_tokens(xs);
        match &self.data {
            Data::Struct(x) => x.struct_.to_tokens(xs),
            Data::Enum(x) => x.enum_.to_tokens(xs),
            Data::Union(x) => x.union_.to_tokens(xs),
        }
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        match &self.data {
            Data::Struct(data) => match &data.fields {
                Fields::Named(x) => {
                    self.gens.where_.to_tokens(xs);
                    x.to_tokens(xs);
                },
                Fields::Unnamed(x) => {
                    x.to_tokens(xs);
                    self.gens.where_.to_tokens(xs);
                    TokensOrDefault(&data.semi).to_tokens(xs);
                },
                Fields::Unit => {
                    self.gens.where_.to_tokens(xs);
                    TokensOrDefault(&data.semi).to_tokens(xs);
                },
            },
            Data::Enum(x) => {
                self.gens.where_.to_tokens(xs);
                x.brace.surround(xs, |tokens| {
                    x.variants.to_tokens(tokens);
                });
            },
            Data::Union(x) => {
                self.gens.where_.to_tokens(xs);
                x.fields.to_tokens(xs);
            },
        }
    }
}

impl ToTokens for File {
    fn to_tokens(&self, xs: &mut TokenStream) {
        xs.append_all(self.attrs.inner());
        xs.append_all(&self.items);
    }
}

impl ToTokens for Lifetime {
    fn to_tokens(&self, xs: &mut TokenStream) {
        let mut apostrophe = Punct::new('\'', Spacing::Joint);
        apostrophe.set_span(self.apostrophe);
        xs.append(apostrophe);
        self.ident.to_tokens(xs);
    }
}

impl MacroDelim {
    pub fn surround(&self, xs: &mut TokenStream, inner: TokenStream) {
        let (delim, span) = match self {
            MacroDelim::Paren(x) => (Delimiter::Parenthesis, x.span),
            MacroDelim::Brace(x) => (Delimiter::Brace, x.span),
            MacroDelim::Bracket(x) => (Delimiter::Bracket, x.span),
        };
        delim(delim, span.join(), xs, inner);
    }
}
impl ToTokens for Macro {
    fn to_tokens(&self, xs: &mut TokenStream) {
        self.path.to_tokens(xs);
        self.bang.to_tokens(xs);
        self.delim.surround(xs, self.toks.clone());
    }
}

impl ToTokens for BinOp {
    fn to_tokens(&self, xs: &mut TokenStream) {
        use BinOp::*;
        match self {
            Add(x) => x.to_tokens(xs),
            Sub(x) => x.to_tokens(xs),
            Mul(x) => x.to_tokens(xs),
            Div(x) => x.to_tokens(xs),
            Rem(x) => x.to_tokens(xs),
            And(x) => x.to_tokens(xs),
            Or(x) => x.to_tokens(xs),
            BitXor(x) => x.to_tokens(xs),
            BitAnd(x) => x.to_tokens(xs),
            BitOr(x) => x.to_tokens(xs),
            Shl(x) => x.to_tokens(xs),
            Shr(x) => x.to_tokens(xs),
            Eq(x) => x.to_tokens(xs),
            Lt(x) => x.to_tokens(xs),
            Le(x) => x.to_tokens(xs),
            Ne(x) => x.to_tokens(xs),
            Ge(x) => x.to_tokens(xs),
            Gt(x) => x.to_tokens(xs),
            AddAssign(x) => x.to_tokens(xs),
            SubAssign(x) => x.to_tokens(xs),
            MulAssign(x) => x.to_tokens(xs),
            DivAssign(x) => x.to_tokens(xs),
            RemAssign(x) => x.to_tokens(xs),
            BitXorAssign(x) => x.to_tokens(xs),
            BitAndAssign(x) => x.to_tokens(xs),
            BitOrAssign(x) => x.to_tokens(xs),
            ShlAssign(x) => x.to_tokens(xs),
            ShrAssign(x) => x.to_tokens(xs),
        }
    }
}
impl ToTokens for UnOp {
    fn to_tokens(&self, xs: &mut TokenStream) {
        match self {
            UnOp::Deref(x) => x.to_tokens(xs),
            UnOp::Not(x) => x.to_tokens(xs),
            UnOp::Neg(x) => x.to_tokens(xs),
        }
    }
}

impl ToTokens for Visibility {
    fn to_tokens(&self, xs: &mut TokenStream) {
        match self {
            Visibility::Public(x) => x.to_tokens(xs),
            Visibility::Restricted(x) => x.to_tokens(xs),
            Visibility::Inherited => {},
        }
    }
}
impl ToTokens for VisRestricted {
    fn to_tokens(&self, xs: &mut TokenStream) {
        self.pub_.to_tokens(xs);
        self.paren.surround(xs, |ys| {
            self.in_.to_tokens(ys);
            self.path.to_tokens(ys);
        });
    }
}

mod lit {
    use crate::lit::*;
    use proc_macro2::TokenStream;
    use quote::ToTokens;

    impl ToTokens for Str {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.repr.tok.to_tokens(xs);
        }
    }
    impl ToTokens for ByteStr {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.repr.tok.to_tokens(xs);
        }
    }
    impl ToTokens for Byte {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.repr.tok.to_tokens(xs);
        }
    }
    impl ToTokens for Char {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.repr.tok.to_tokens(xs);
        }
    }
    impl ToTokens for Int {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.repr.tok.to_tokens(xs);
        }
    }
    impl ToTokens for Float {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.repr.tok.to_tokens(xs);
        }
    }
    impl ToTokens for Bool {
        fn to_tokens(&self, xs: &mut TokenStream) {
            xs.append(self.token());
        }
    }
}

mod patt {
    use crate::patt::*;
    use proc_macro2::TokenStream;
    use quote::ToTokens;

    impl ToTokens for Ident {
        fn to_tokens(&self, xs: &mut TokenStream) {
            xs.append_all(self.attrs.outer());
            self.ref_.to_tokens(xs);
            self.mut_.to_tokens(xs);
            self.ident.to_tokens(xs);
            if let Some((at_, sub)) = &self.subpat {
                at_.to_tokens(xs);
                sub.to_tokens(xs);
            }
        }
    }
    impl ToTokens for Or {
        fn to_tokens(&self, xs: &mut TokenStream) {
            xs.append_all(self.attrs.outer());
            self.vert.to_tokens(xs);
            self.cases.to_tokens(xs);
        }
    }
    impl ToTokens for Paren {
        fn to_tokens(&self, xs: &mut TokenStream) {
            xs.append_all(self.attrs.outer());
            self.paren.surround(xs, |ys| {
                self.patt.to_tokens(ys);
            });
        }
    }
    impl ToTokens for Ref {
        fn to_tokens(&self, xs: &mut TokenStream) {
            xs.append_all(self.attrs.outer());
            self.and.to_tokens(xs);
            self.mut_.to_tokens(xs);
            self.patt.to_tokens(xs);
        }
    }
    impl ToTokens for Rest {
        fn to_tokens(&self, xs: &mut TokenStream) {
            xs.append_all(self.attrs.outer());
            self.dot2.to_tokens(xs);
        }
    }
    impl ToTokens for Slice {
        fn to_tokens(&self, xs: &mut TokenStream) {
            xs.append_all(self.attrs.outer());
            self.bracket.surround(xs, |ys| {
                self.elems.to_tokens(ys);
            });
        }
    }
    impl ToTokens for Struct {
        fn to_tokens(&self, xs: &mut TokenStream) {
            xs.append_all(self.attrs.outer());
            print_path(xs, &self.qself, &self.path);
            self.brace.surround(xs, |ys| {
                self.fields.to_tokens(ys);
                if !self.fields.empty_or_trailing() && self.rest.is_some() {
                    <Token![,]>::default().to_tokens(ys);
                }
                self.rest.to_tokens(ys);
            });
        }
    }
    impl ToTokens for Tuple {
        fn to_tokens(&self, xs: &mut TokenStream) {
            xs.append_all(self.attrs.outer());
            self.paren.surround(xs, |ys| {
                self.elems.to_tokens(ys);
            });
        }
    }
    impl ToTokens for TupleStruct {
        fn to_tokens(&self, xs: &mut TokenStream) {
            xs.append_all(self.attrs.outer());
            print_path(xs, &self.qself, &self.path);
            self.paren.surround(xs, |ys| {
                self.elems.to_tokens(ys);
            });
        }
    }
    impl ToTokens for Type {
        fn to_tokens(&self, xs: &mut TokenStream) {
            xs.append_all(self.attrs.outer());
            self.patt.to_tokens(xs);
            self.colon.to_tokens(xs);
            self.typ.to_tokens(xs);
        }
    }
    impl ToTokens for Wild {
        fn to_tokens(&self, xs: &mut TokenStream) {
            xs.append_all(self.attrs.outer());
            self.underscore.to_tokens(xs);
        }
    }
    impl ToTokens for Field {
        fn to_tokens(&self, xs: &mut TokenStream) {
            xs.append_all(self.attrs.outer());
            if let Some(colon) = &self.colon {
                self.member.to_tokens(xs);
                colon.to_tokens(xs);
            }
            self.patt.to_tokens(xs);
        }
    }
}

mod path {
    use crate::path::*;
    use proc_macro2::TokenStream;
    use quote::ToTokens;

    impl ToTokens for Path {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.colon.to_tokens(xs);
            self.segs.to_tokens(xs);
        }
    }
    impl ToTokens for Segment {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.ident.to_tokens(xs);
            self.args.to_tokens(xs);
        }
    }
    impl ToTokens for Args {
        fn to_tokens(&self, xs: &mut TokenStream) {
            match self {
                Args::None => {},
                Args::Angled(args) => {
                    args.to_tokens(xs);
                },
                Args::Parenthesized(args) => {
                    args.to_tokens(xs);
                },
            }
        }
    }
    impl ToTokens for Arg {
        #[allow(clippy::match_same_arms)]
        fn to_tokens(&self, xs: &mut TokenStream) {
            use Arg::*;
            match self {
                Lifetime(x) => x.to_tokens(xs),
                Type(x) => x.to_tokens(xs),
                Const(x) => match x {
                    Expr::Lit(_) => x.to_tokens(xs),
                    Expr::Block(_) => x.to_tokens(xs),
                    _ => tok::Brace::default().surround(xs, |xs| {
                        x.to_tokens(xs);
                    }),
                },
                AssocType(x) => x.to_tokens(xs),
                AssocConst(x) => x.to_tokens(xs),
                Constraint(x) => x.to_tokens(xs),
            }
        }
    }
    impl ToTokens for AngledArgs {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.colon2.to_tokens(xs);
            self.lt.to_tokens(xs);
            let mut trailing_or_empty = true;
            for x in self.args.pairs() {
                match x.value() {
                    Arg::Lifetime(_) => {
                        x.to_tokens(xs);
                        trailing_or_empty = x.punct().is_some();
                    },
                    Arg::Type(_) | Arg::Const(_) | Arg::AssocType(_) | Arg::AssocConst(_) | Arg::Constraint(_) => {},
                }
            }
            for x in self.args.pairs() {
                match x.value() {
                    Arg::Type(_) | Arg::Const(_) | Arg::AssocType(_) | Arg::AssocConst(_) | Arg::Constraint(_) => {
                        if !trailing_or_empty {
                            <Token![,]>::default().to_tokens(xs);
                        }
                        x.to_tokens(xs);
                        trailing_or_empty = x.punct().is_some();
                    },
                    Arg::Lifetime(_) => {},
                }
            }
            self.gt.to_tokens(xs);
        }
    }
    impl ToTokens for AssocType {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.ident.to_tokens(xs);
            self.args.to_tokens(xs);
            self.eq.to_tokens(xs);
            self.typ.to_tokens(xs);
        }
    }
    impl ToTokens for AssocConst {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.ident.to_tokens(xs);
            self.args.to_tokens(xs);
            self.eq.to_tokens(xs);
            self.val.to_tokens(xs);
        }
    }
    impl ToTokens for Constraint {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.ident.to_tokens(xs);
            self.args.to_tokens(xs);
            self.colon.to_tokens(xs);
            self.bounds.to_tokens(xs);
        }
    }
    impl ToTokens for ParenthesizedArgs {
        fn to_tokens(&self, xs: &mut TokenStream) {
            self.paren.surround(xs, |ys| {
                self.args.to_tokens(ys);
            });
            self.ret.to_tokens(xs);
        }
    }
    pub(crate) fn print_path(xs: &mut TokenStream, qself: &Option<QSelf>, path: &Path) {
        let qself = match qself {
            Some(qself) => qself,
            None => {
                path.to_tokens(xs);
                return;
            },
        };
        qself.lt.to_tokens(xs);
        qself.typ.to_tokens(xs);
        let pos = cmp::min(qself.pos, path.segs.len());
        let mut segments = path.segs.pairs();
        if pos > 0 {
            TokensOrDefault(&qself.as_).to_tokens(xs);
            path.colon.to_tokens(xs);
            for (i, segment) in segments.by_ref().take(pos).enumerate() {
                if i + 1 == pos {
                    segment.value().to_tokens(xs);
                    qself.gt.to_tokens(xs);
                    segment.punct().to_tokens(xs);
                } else {
                    segment.to_tokens(xs);
                }
            }
        } else {
            qself.gt.to_tokens(xs);
            path.colon.to_tokens(xs);
        }
        for segment in segments {
            segment.to_tokens(xs);
        }
    }
}

pub fn punct(s: &str, spans: &[Span], xs: &mut TokenStream) {
    assert_eq!(s.len(), spans.len());
    let mut chars = s.chars();
    let mut spans = spans.iter();
    let ch = chars.next_back().unwrap();
    let span = spans.next_back().unwrap();
    for (ch, span) in chars.zip(spans) {
        let mut op = Punct::new(ch, Spacing::Joint);
        op.set_span(*span);
        xs.append(op);
    }
    let mut op = Punct::new(ch, Spacing::Alone);
    op.set_span(*span);
    xs.append(op);
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
    use proc_macro2::TokenStream;
    use quote::ToTokens;

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
