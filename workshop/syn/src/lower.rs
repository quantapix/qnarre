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
