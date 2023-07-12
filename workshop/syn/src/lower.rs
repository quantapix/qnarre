use proc_macro2::Ident;
use quote::{ToTokens, TokenStreamExt};
use std::cmp;

fn wrap_bare_struct(tokens: &mut pm2::Stream, e: &Expr) {
    if let Expr::Struct(_) = *e {
        tok::Paren::default().surround(tokens, |tokens| {
            e.to_tokens(tokens);
        });
    } else {
        e.to_tokens(tokens);
    }
}
impl ToTokens for expr::Array {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.bracket.surround(tokens, |tokens| {
            self.elems.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Assign {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.left.to_tokens(tokens);
        self.eq.to_tokens(tokens);
        self.right.to_tokens(tokens);
    }
}
impl ToTokens for expr::Async {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.async_.to_tokens(tokens);
        self.move_.to_tokens(tokens);
        self.block.to_tokens(tokens);
    }
}
impl ToTokens for expr::Await {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.dot.to_tokens(tokens);
        self.await_.to_tokens(tokens);
    }
}
impl ToTokens for expr::Binary {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.left.to_tokens(tokens);
        self.op.to_tokens(tokens);
        self.right.to_tokens(tokens);
    }
}
impl ToTokens for expr::Block {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.label.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for expr::Break {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.break_.to_tokens(tokens);
        self.label.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for expr::Call {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.func.to_tokens(tokens);
        self.paren.surround(tokens, |tokens| {
            self.args.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Cast {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.as_.to_tokens(tokens);
        self.typ.to_tokens(tokens);
    }
}
impl ToTokens for expr::Closure {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
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
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.const_.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for expr::Continue {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.continue_.to_tokens(tokens);
        self.label.to_tokens(tokens);
    }
}
impl ToTokens for expr::Field {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.base.to_tokens(tokens);
        self.dot.to_tokens(tokens);
        self.memb.to_tokens(tokens);
    }
}
impl ToTokens for expr::ForLoop {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
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
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.group.surround(tokens, |tokens| {
            self.expr.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::If {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
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
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.bracket.surround(tokens, |tokens| {
            self.index.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Infer {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.underscore.to_tokens(tokens);
    }
}
impl ToTokens for expr::Let {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.let_.to_tokens(tokens);
        self.pat.to_tokens(tokens);
        self.eq.to_tokens(tokens);
        wrap_bare_struct(tokens, &self.expr);
    }
}
impl ToTokens for expr::Lit {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.lit.to_tokens(tokens);
    }
}
impl ToTokens for expr::Loop {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
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
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.mac.to_tokens(tokens);
    }
}
impl ToTokens for expr::Match {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
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
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
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
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.paren.surround(tokens, |tokens| {
            self.expr.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Path {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        print_path(tokens, &self.qself, &self.path);
    }
}
impl ToTokens for expr::Range {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.beg.to_tokens(tokens);
        self.limits.to_tokens(tokens);
        self.end.to_tokens(tokens);
    }
}
impl ToTokens for expr::Ref {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.and.to_tokens(tokens);
        self.mut_.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for expr::Repeat {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.bracket.surround(tokens, |tokens| {
            self.expr.to_tokens(tokens);
            self.semi.to_tokens(tokens);
            self.len.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Return {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.return_.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for expr::Struct {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        print_path(tokens, &self.qself, &self.path);
        self.brace.surround(tokens, |tokens| {
            self.fields.to_tokens(tokens);
            if let Some(dot2) = &self.dot2 {
                dot2.to_tokens(tokens);
            } else if self.rest.is_some() {
                Token![..](pm2::Span::call_site()).to_tokens(tokens);
            }
            self.rest.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Try {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.question.to_tokens(tokens);
    }
}
impl ToTokens for expr::TryBlock {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.try_.to_tokens(tokens);
        self.block.to_tokens(tokens);
    }
}
impl ToTokens for expr::Tuple {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
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
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.op.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for expr::Unsafe {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.unsafe_.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for expr::While {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
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
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.yield_.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for Arm {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
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
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.member.to_tokens(tokens);
        if let Some(colon) = &self.colon {
            colon.to_tokens(tokens);
            self.expr.to_tokens(tokens);
        }
    }
}
impl ToTokens for Index {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        let mut lit = pm2::Lit::i64_unsuffixed(i64::from(self.index));
        lit.set_span(self.span);
        tokens.append(lit);
    }
}
impl ToTokens for Label {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        self.name.to_tokens(tokens);
        self.colon.to_tokens(tokens);
    }
}
impl ToTokens for Member {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        match self {
            Member::Named(ident) => ident.to_tokens(tokens),
            Member::Unnamed(index) => index.to_tokens(tokens),
        }
    }
}
impl ToTokens for RangeLimits {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        match self {
            RangeLimits::HalfOpen(x) => x.to_tokens(xs),
            RangeLimits::Closed(x) => x.to_tokens(xs),
        }
    }
}

impl ToTokens for item::ExternCrate {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
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
impl ToTokens for item::Use {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.use_.to_tokens(tokens);
        self.colon.to_tokens(tokens);
        self.tree.to_tokens(tokens);
        self.semi.to_tokens(tokens);
    }
}
impl ToTokens for item::Static {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
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
impl ToTokens for item::Const {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
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
impl ToTokens for item::Fn {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        tokens.append_all(self.attrs.outer());
        self.vis.to_tokens(tokens);
        self.sig.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            tokens.append_all(self.attrs.inner());
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for item::Mod {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
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
impl ToTokens for item::Foreign {
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        tokens.append_all(self.attrs.outer());
        self.unsafe_.to_tokens(tokens);
        self.abi.to_tokens(tokens);
        self.brace.surround(tokens, |tokens| {
            tokens.append_all(self.attrs.inner());
            tokens.append_all(&self.items);
        });
    }
}
impl ToTokens for item::Type {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
impl ToTokens for item::Enum {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.enum_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        self.gens.where_.to_tokens(xs);
        self.brace.surround(xs, |ys| {
            self.elems.to_tokens(ys);
        });
    }
}
impl ToTokens for item::Struct {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.struct_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        match &self.fields {
            data::Fields::Named(fields) => {
                self.gens.where_.to_tokens(xs);
                fields.to_tokens(xs);
            },
            data::Fields::Unnamed(fields) => {
                fields.to_tokens(xs);
                self.gens.where_.to_tokens(xs);
                TokensOrDefault(&self.semi).to_tokens(xs);
            },
            data::Fields::Unit => {
                self.gens.where_.to_tokens(xs);
                TokensOrDefault(&self.semi).to_tokens(xs);
            },
        }
    }
}
impl ToTokens for item::Union {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.union_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        self.gens.where_.to_tokens(xs);
        self.fields.to_tokens(xs);
    }
}
impl ToTokens for item::Trait {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.unsafe_.to_tokens(xs);
        self.auto_.to_tokens(xs);
        self.trait_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        if !self.supers.is_empty() {
            TokensOrDefault(&self.colon).to_tokens(xs);
            self.supers.to_tokens(xs);
        }
        self.gens.where_.to_tokens(xs);
        self.brace.surround(xs, |ys| {
            ys.append_all(self.attrs.inner());
            ys.append_all(&self.items);
        });
    }
}
impl ToTokens for item::TraitAlias {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
impl ToTokens for item::Impl {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
impl ToTokens for item::Mac {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        xs.append_all(self.attrs.outer());
        self.mac.path.to_tokens(xs);
        self.mac.bang.to_tokens(xs);
        self.ident.to_tokens(xs);
        match &self.mac.delim {
            tok::Delim::Paren(x) => {
                x.surround(xs, |ys| self.mac.toks.to_tokens(ys));
            },
            tok::Delim::Brace(x) => {
                x.surround(xs, |ys| self.mac.toks.to_tokens(ys));
            },
            tok::Delim::Bracket(x) => {
                x.surround(xs, |ys| self.mac.toks.to_tokens(ys));
            },
        }
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for item::Use::Path {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        self.ident.to_tokens(xs);
        self.colon2.to_tokens(xs);
        self.tree.to_tokens(xs);
    }
}
impl ToTokens for item::Use::Name {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        self.ident.to_tokens(xs);
    }
}
impl ToTokens for item::Use::Rename {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        self.ident.to_tokens(xs);
        self.as_.to_tokens(xs);
        self.rename.to_tokens(xs);
    }
}
impl ToTokens for item::Use::Glob {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        self.star.to_tokens(xs);
    }
}
impl ToTokens for item::Use::Group {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        self.brace.surround(xs, |ys| {
            self.elems.to_tokens(ys);
        });
    }
}
impl ToTokens for item::Trait::Const {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
impl ToTokens for item::Trait::Fn {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
impl ToTokens for item::Trait::Type {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
impl ToTokens for item::Trait::Mac {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        xs.append_all(self.attrs.outer());
        self.mac.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for item::Impl::Const {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
impl ToTokens for item::Impl::Fn {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
impl ToTokens for item::Impl::Type {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
impl ToTokens for item::Impl::Mac {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        xs.append_all(self.attrs.outer());
        self.mac.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for item::Foreign::Fn {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.sig.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for item::Foreign::Static {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
impl ToTokens for item::Foreign::Type {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        xs.append_all(self.attrs.outer());
        self.vis.to_tokens(xs);
        self.type_.to_tokens(xs);
        self.ident.to_tokens(xs);
        self.gens.to_tokens(xs);
        self.gens.where_.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for item::Foreign::Mac {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        xs.append_all(self.attrs.outer());
        self.mac.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}
impl ToTokens for item::Sig {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
impl ToTokens for item::Receiver {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
                (Some(_), mutability, typ::Type::Ref(ty)) => {
                    mutability.is_some() == ty.mut_.is_some()
                        && match &*ty.elem {
                            typ::Type::Path(ty) => ty.qself.is_none() && ty.path.is_ident("Self"),
                            _ => false,
                        }
                },
                (None, _, typ::Type::Path(ty)) => ty.qself.is_none() && ty.path.is_ident("Self"),
                _ => false,
            };
            if !consistent {
                <Token![:]>::default().to_tokens(xs);
                self.typ.to_tokens(xs);
            }
        }
    }
}
impl ToTokens for item::Variadic {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        match self {
            StaticMut::None => {},
            StaticMut::Mut(mut_token) => mut_token.to_tokens(xs),
        }
    }
}

impl ToTokens for Block {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        self.brace.surround(xs, |ys| {
            ys.append_all(&self.stmts);
        });
    }
}
impl ToTokens for stmt::Stmt {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, xs);
        self.mac.to_tokens(xs);
        self.semi.to_tokens(xs);
    }
}

impl ToTokens for data::Variant {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        xs.append_all(&self.attrs);
        self.ident.to_tokens(xs);
        self.fields.to_tokens(xs);
        if let Some((eq, disc)) = &self.discriminant {
            eq.to_tokens(xs);
            disc.to_tokens(xs);
        }
    }
}
impl ToTokens for data::Named {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        self.brace.surround(xs, |ys| {
            self.named.to_tokens(ys);
        });
    }
}
impl ToTokens for data::Unnamed {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        self.paren.surround(xs, |ys| {
            self.unnamed.to_tokens(ys);
        });
    }
}
impl ToTokens for data::Field {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        xs.append_all(&self.attrs);
        self.vis.to_tokens(xs);
        if let Some(x) = &self.ident {
            x.to_tokens(xs);
            TokensOrDefault(&self.colon).to_tokens(xs);
        }
        self.typ.to_tokens(xs);
    }
}

impl ToTokens for DeriveInput {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        for x in self.attrs.outer() {
            x.to_tokens(xs);
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
                data::Fields::Named(x) => {
                    self.gens.where_.to_tokens(xs);
                    x.to_tokens(xs);
                },
                data::Fields::Unnamed(x) => {
                    x.to_tokens(xs);
                    self.gens.where_.to_tokens(xs);
                    TokensOrDefault(&data.semi).to_tokens(xs);
                },
                data::Fields::Unit => {
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
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        xs.append_all(self.attrs.inner());
        xs.append_all(&self.items);
    }
}

impl ToTokens for Lifetime {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        let mut apostrophe = Punct::new('\'', pm2::Spacing::Joint);
        apostrophe.set_span(self.apostrophe);
        xs.append(apostrophe);
        self.ident.to_tokens(xs);
    }
}

impl tok::Delim {
    pub fn surround(&self, xs: &mut pm2::Stream, inner: pm2::Stream) {
        let (delim, span) = match self {
            tok::Delim::Paren(x) => (pm2::Delim::Parenthesis, x.span),
            tok::Delim::Brace(x) => (pm2::Delim::Brace, x.span),
            tok::Delim::Bracket(x) => (pm2::Delim::Bracket, x.span),
        };
        delim(delim, span.join(), xs, inner);
    }
}
impl ToTokens for mac::Mac {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        self.path.to_tokens(xs);
        self.bang.to_tokens(xs);
        self.delim.surround(xs, self.toks.clone());
    }
}

impl ToTokens for BinOp {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
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
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        match self {
            UnOp::Deref(x) => x.to_tokens(xs),
            UnOp::Not(x) => x.to_tokens(xs),
            UnOp::Neg(x) => x.to_tokens(xs),
        }
    }
}

impl ToTokens for Visibility {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        match self {
            Visibility::Public(x) => x.to_tokens(xs),
            Visibility::Restricted(x) => x.to_tokens(xs),
            Visibility::Inherited => {},
        }
    }
}
impl ToTokens for VisRestricted {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        self.pub_.to_tokens(xs);
        self.paren.surround(xs, |ys| {
            self.in_.to_tokens(ys);
            self.path.to_tokens(ys);
        });
    }
}

mod lit {
    use crate::lit::*;
    use proc_macro2::pm2::Stream;
    use quote::ToTokens;

    impl ToTokens for Str {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.repr.tok.to_tokens(xs);
        }
    }
    impl ToTokens for ByteStr {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.repr.tok.to_tokens(xs);
        }
    }
    impl ToTokens for Byte {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.repr.tok.to_tokens(xs);
        }
    }
    impl ToTokens for Char {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.repr.tok.to_tokens(xs);
        }
    }
    impl ToTokens for Int {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.repr.tok.to_tokens(xs);
        }
    }
    impl ToTokens for Float {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.repr.tok.to_tokens(xs);
        }
    }
    impl ToTokens for Bool {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            xs.append(self.token());
        }
    }
}

mod pat {
    use crate::pat::*;
    use proc_macro2::pm2::Stream;
    use quote::ToTokens;

    impl ToTokens for Ident {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
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
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            xs.append_all(self.attrs.outer());
            self.vert.to_tokens(xs);
            self.cases.to_tokens(xs);
        }
    }
    impl ToTokens for Paren {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            xs.append_all(self.attrs.outer());
            self.paren.surround(xs, |ys| {
                self.pat.to_tokens(ys);
            });
        }
    }
    impl ToTokens for Ref {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            xs.append_all(self.attrs.outer());
            self.and.to_tokens(xs);
            self.mut_.to_tokens(xs);
            self.pat.to_tokens(xs);
        }
    }
    impl ToTokens for Rest {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            xs.append_all(self.attrs.outer());
            self.dot2.to_tokens(xs);
        }
    }
    impl ToTokens for Slice {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            xs.append_all(self.attrs.outer());
            self.bracket.surround(xs, |ys| {
                self.elems.to_tokens(ys);
            });
        }
    }
    impl ToTokens for Struct {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
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
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            xs.append_all(self.attrs.outer());
            self.paren.surround(xs, |ys| {
                self.elems.to_tokens(ys);
            });
        }
    }
    impl ToTokens for TupleStruct {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            xs.append_all(self.attrs.outer());
            print_path(xs, &self.qself, &self.path);
            self.paren.surround(xs, |ys| {
                self.elems.to_tokens(ys);
            });
        }
    }
    impl ToTokens for Type {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            xs.append_all(self.attrs.outer());
            self.pat.to_tokens(xs);
            self.colon.to_tokens(xs);
            self.typ.to_tokens(xs);
        }
    }
    impl ToTokens for Wild {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            xs.append_all(self.attrs.outer());
            self.underscore.to_tokens(xs);
        }
    }
    impl ToTokens for Field {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            xs.append_all(self.attrs.outer());
            if let Some(x) = &self.colon {
                self.member.to_tokens(xs);
                x.to_tokens(xs);
            }
            self.pat.to_tokens(xs);
        }
    }
}

mod path {
    use crate::path::*;
    use proc_macro2::pm2::Stream;
    use quote::ToTokens;

    impl ToTokens for Path {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.colon.to_tokens(xs);
            self.segs.to_tokens(xs);
        }
    }
    impl ToTokens for Segment {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.ident.to_tokens(xs);
            self.args.to_tokens(xs);
        }
    }
    impl ToTokens for Args {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
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
        fn to_tokens(&self, xs: &mut pm2::Stream) {
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
        fn to_tokens(&self, xs: &mut pm2::Stream) {
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
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.ident.to_tokens(xs);
            self.args.to_tokens(xs);
            self.eq.to_tokens(xs);
            self.typ.to_tokens(xs);
        }
    }
    impl ToTokens for AssocConst {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.ident.to_tokens(xs);
            self.args.to_tokens(xs);
            self.eq.to_tokens(xs);
            self.val.to_tokens(xs);
        }
    }
    impl ToTokens for Constraint {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.ident.to_tokens(xs);
            self.args.to_tokens(xs);
            self.colon.to_tokens(xs);
            self.bounds.to_tokens(xs);
        }
    }
    impl ToTokens for ParenthesizedArgs {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.paren.surround(xs, |ys| {
                self.args.to_tokens(ys);
            });
            self.ret.to_tokens(xs);
        }
    }
    pub(crate) fn print_path(xs: &mut pm2::Stream, qself: &Option<QSelf>, path: &Path) {
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

pub fn punct(s: &str, spans: &[pm2::Span], xs: &mut pm2::Stream) {
    assert_eq!(s.len(), spans.len());
    let mut chars = s.chars();
    let mut spans = spans.iter();
    let ch = chars.next_back().unwrap();
    let span = spans.next_back().unwrap();
    for (ch, span) in chars.zip(spans) {
        let mut op = Punct::new(ch, pm2::Spacing::Joint);
        op.set_span(*span);
        xs.append(op);
    }
    let mut op = Punct::new(ch, pm2::Spacing::Alone);
    op.set_span(*span);
    xs.append(op);
}
pub fn keyword(x: &str, s: pm2::Span, xs: &mut pm2::Stream) {
    xs.append(Ident::new(x, s));
}
pub fn delim(d: pm2::Delim, s: pm2::Span, xs: &mut pm2::Stream, inner: pm2::Stream) {
    let mut g = Group::new(d, inner);
    g.set_span(s);
    xs.append(g);
}

mod ty {
    use crate::typ::*;
    use proc_macro2::pm2::Stream;
    use quote::ToTokens;

    impl ToTokens for Slice {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.bracket.surround(xs, |ys| {
                self.elem.to_tokens(ys);
            });
        }
    }
    impl ToTokens for Array {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.bracket.surround(xs, |ys| {
                self.elem.to_tokens(ys);
                self.semi.to_tokens(ys);
                self.len.to_tokens(ys);
            });
        }
    }
    impl ToTokens for Ptr {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
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
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.and.to_tokens(xs);
            self.life.to_tokens(xs);
            self.mut_.to_tokens(xs);
            self.elem.to_tokens(xs);
        }
    }
    impl ToTokens for Fn {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
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
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.bang.to_tokens(xs);
        }
    }
    impl ToTokens for Tuple {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.paren.surround(xs, |ys| {
                self.elems.to_tokens(ys);
                if self.elems.len() == 1 && !self.elems.trailing_punct() {
                    <Token![,]>::default().to_tokens(ys);
                }
            });
        }
    }
    impl ToTokens for Path {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            print_path(xs, &self.qself, &self.path);
        }
    }
    impl ToTokens for Trait {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.dyn_.to_tokens(xs);
            self.bounds.to_tokens(xs);
        }
    }
    impl ToTokens for Impl {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.impl_.to_tokens(xs);
            self.bounds.to_tokens(xs);
        }
    }
    impl ToTokens for Group {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.group.surround(xs, |ys| {
                self.elem.to_tokens(ys);
            });
        }
    }
    impl ToTokens for Paren {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.paren.surround(xs, |ys| {
                self.elem.to_tokens(ys);
            });
        }
    }
    impl ToTokens for Infer {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.underscore.to_tokens(xs);
        }
    }
    impl ToTokens for Mac {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.mac.to_tokens(xs);
        }
    }
    impl ToTokens for Ret {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            match self {
                Ret::Type(arrow, ty) => {
                    arrow.to_tokens(xs);
                    ty.to_tokens(xs);
                },
                Ret::Default => {},
            }
        }
    }
    impl ToTokens for FnArg {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            xs.append_all(self.attrs.outer());
            if let Some((name, colon)) = &self.name {
                name.to_tokens(xs);
                colon.to_tokens(xs);
            }
            self.ty.to_tokens(xs);
        }
    }
    impl ToTokens for Variadic {
        fn to_tokens(&self, xs: &mut pm2::Stream) {
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
        fn to_tokens(&self, xs: &mut pm2::Stream) {
            self.extern_.to_tokens(xs);
            self.name.to_tokens(xs);
        }
    }
}

impl<T, P> ToTokens for Punctuated<T, P>
where
    T: ToTokens,
    P: ToTokens,
{
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        tokens.append_all(self.pairs());
    }
}
impl<T, P> ToTokens for Pair<T, P>
where
    T: ToTokens,
    P: ToTokens,
{
    fn to_tokens(&self, tokens: &mut pm2::Stream) {
        match self {
            Pair::Punctuated(a, b) => {
                a.to_tokens(tokens);
                b.to_tokens(tokens);
            },
            Pair::End(a) => a.to_tokens(tokens),
        }
    }
}
