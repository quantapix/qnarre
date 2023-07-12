use super::{
    ext::IdentExt,
    parse::{discouraged::Speculative, Buffer, Parse, Stream},
    tok::Tok,
};
use proc_macro2::{Ident, Punct};
use std::{
    cmp::Ordering,
    fmt::{self, Display},
};

mod kw {
    crate::custom_keyword!(builtin);
    crate::custom_keyword!(raw);
}

pub struct AllowStruct(bool);
enum Precedence {
    Any,
    Assign,
    Range,
    Or,
    And,
    Compare,
    BitOr,
    BitXor,
    BitAnd,
    Shift,
    Arithmetic,
    Term,
    Cast,
}
impl Precedence {
    fn of(op: &BinOp) -> Self {
        match op {
            BinOp::Add(_) | BinOp::Sub(_) => Precedence::Arithmetic,
            BinOp::Mul(_) | BinOp::Div(_) | BinOp::Rem(_) => Precedence::Term,
            BinOp::And(_) => Precedence::And,
            BinOp::Or(_) => Precedence::Or,
            BinOp::BitXor(_) => Precedence::BitXor,
            BinOp::BitAnd(_) => Precedence::BitAnd,
            BinOp::BitOr(_) => Precedence::BitOr,
            BinOp::Shl(_) | BinOp::Shr(_) => Precedence::Shift,
            BinOp::Eq(_) | BinOp::Lt(_) | BinOp::Le(_) | BinOp::Ne(_) | BinOp::Ge(_) | BinOp::Gt(_) => {
                Precedence::Compare
            },
            BinOp::AddAssign(_)
            | BinOp::SubAssign(_)
            | BinOp::MulAssign(_)
            | BinOp::DivAssign(_)
            | BinOp::RemAssign(_)
            | BinOp::BitXorAssign(_)
            | BinOp::BitAndAssign(_)
            | BinOp::BitOrAssign(_)
            | BinOp::ShlAssign(_)
            | BinOp::ShrAssign(_) => Precedence::Assign,
        }
    }
}

impl Parse for Expr {
    fn parse(x: Stream) -> Res<Self> {
        ambiguous_expr(x, AllowStruct(true))
    }
}
impl Expr {
    pub fn parse_without_eager_brace(x: Stream) -> Res<Expr> {
        ambiguous_expr(x, AllowStruct(false))
    }
}
impl Copy for AllowStruct {}
impl Clone for AllowStruct {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for Precedence {}
impl Clone for Precedence {
    fn clone(&self) -> Self {
        *self
    }
}
impl PartialEq for Precedence {
    fn eq(&self, other: &Self) -> bool {
        *self as u8 == *other as u8
    }
}
impl PartialOrd for Precedence {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let this = *self as u8;
        let other = *other as u8;
        Some(this.cmp(&other))
    }
}

impl Visibility {
    fn is_inherited(&self) -> bool {
        match self {
            Visibility::Inherited => true,
            _ => false,
        }
    }
}
impl tok::Delim {
    pub fn is_brace(&self) -> bool {
        match self {
            tok::Delim::Brace(_) => true,
            tok::Delim::Paren(_) | tok::Delim::Bracket(_) => false,
        }
    }
}
impl Parse for StaticMut {
    fn parse(input: Stream) -> Res<Self> {
        let mut_token: Option<Token![mut]> = input.parse()?;
        Ok(mut_token.map_or(StaticMut::None, StaticMut::Mut))
    }
}

mod parsing {
    struct AllowNoSemi(bool);
    impl Block {
        pub fn parse_within(x: Stream) -> Res<Vec<stmt::Stmt>> {
            let mut ys = Vec::new();
            loop {
                while let semi @ Some(_) = x.parse()? {
                    ys.push(stmt::Stmt::Expr(Expr::Verbatim(pm2::Stream::new()), semi));
                }
                if x.is_empty() {
                    break;
                }
                let stmt = parse_stmt(x, AllowNoSemi(true))?;
                let requires_semicolon = match &stmt {
                    stmt::Stmt::Expr(x, None) => expr::requires_terminator(x),
                    stmt::Stmt::Macro(x) => x.semi.is_none() && !x.mac.delimiter.is_brace(),
                    stmt::Stmt::stmt::Local(_) | stmt::Stmt::Item(_) | stmt::Stmt::Expr(_, Some(_)) => false,
                };
                ys.push(stmt);
                if x.is_empty() {
                    break;
                } else if requires_semicolon {
                    return Err(x.error("unexpected token, expected `;`"));
                }
            }
            Ok(ys)
        }
    }
    impl Parse for Block {
        fn parse(x: Stream) -> Res<Self> {
            let content;
            Ok(Block {
                brace: braced!(content in x),
                stmts: content.call(Block::parse_within)?,
            })
        }
    }
    impl Parse for stmt::Stmt {
        fn parse(x: Stream) -> Res<Self> {
            let allow_nosemi = AllowNoSemi(false);
            parse_stmt(x, allow_nosemi)
        }
    }
    fn parse_stmt(x: Stream, allow_nosemi: AllowNoSemi) -> Res<stmt::Stmt> {
        let begin = x.fork();
        let attrs = x.call(attr::Attr::parse_outer)?;
        let ahead = x.fork();
        let mut is_item_macro = false;
        if let Ok(path) = ahead.call(Path::parse_mod_style) {
            if ahead.peek(Token![!]) {
                if ahead.peek2(Ident) || ahead.peek2(Token![try]) {
                    is_item_macro = true;
                } else if ahead.peek2(tok::Brace) && !(ahead.peek3(Token![.]) || ahead.peek3(Token![?])) {
                    x.advance_to(&ahead);
                    return stmt_mac(x, attrs, path).map(stmt::Stmt::Macro);
                }
            }
        }
        if x.peek(Token![let]) {
            stmt_local(x, attrs).map(stmt::Stmt::stmt::Local)
        } else if x.peek(Token![pub])
            || x.peek(Token![crate]) && !x.peek2(Token![::])
            || x.peek(Token![extern])
            || x.peek(Token![use])
            || x.peek(Token![static])
                && (x.peek2(Token![mut])
                    || x.peek2(Ident) && !(x.peek2(Token![async]) && (x.peek3(Token![move]) || x.peek3(Token![|]))))
            || x.peek(Token![const])
                && !(x.peek2(tok::Brace)
                    || x.peek2(Token![static])
                    || x.peek2(Token![async])
                        && !(x.peek3(Token![unsafe]) || x.peek3(Token![extern]) || x.peek3(Token![fn]))
                    || x.peek2(Token![move])
                    || x.peek2(Token![|]))
            || x.peek(Token![unsafe]) && !x.peek2(tok::Brace)
            || x.peek(Token![async]) && (x.peek2(Token![unsafe]) || x.peek2(Token![extern]) || x.peek2(Token![fn]))
            || x.peek(Token![fn])
            || x.peek(Token![mod])
            || x.peek(Token![type])
            || x.peek(Token![struct])
            || x.peek(Token![enum])
            || x.peek(Token![union]) && x.peek2(Ident)
            || x.peek(Token![auto]) && x.peek2(Token![trait])
            || x.peek(Token![trait])
            || x.peek(Token![default]) && (x.peek2(Token![unsafe]) || x.peek2(Token![impl]))
            || x.peek(Token![impl])
            || x.peek(Token![macro])
            || is_item_macro
        {
            let item = parse_rest_of_item(begin, attrs, x)?;
            Ok(stmt::Stmt::Item(item))
        } else {
            stmt_expr(x, allow_nosemi, attrs)
        }
    }
    fn stmt_mac(x: Stream, attrs: Vec<attr::Attr>, path: Path) -> Res<stmt::Mac> {
        let bang: Token![!] = x.parse()?;
        let (delimiter, tokens) = mac::parse_delim(x)?;
        let semi: Option<Token![;]> = x.parse()?;
        Ok(stmt::Mac {
            attrs,
            mac: Macro {
                path,
                bang,
                delimiter,
                tokens,
            },
            semi,
        })
    }
    fn stmt_local(x: Stream, attrs: Vec<attr::Attr>) -> Res<stmt::Local> {
        let let_: Token![let] = x.parse()?;
        let mut pat = pat::Pat::parse_single(x)?;
        if x.peek(Token![:]) {
            let colon: Token![:] = x.parse()?;
            let ty: Type = x.parse()?;
            pat = pat::Pat::Type(pat::Type {
                attrs: Vec::new(),
                pat: Box::new(pat),
                colon,
                ty: Box::new(ty),
            });
        }
        let init = if let Some(eq) = x.parse()? {
            let eq: Token![=] = eq;
            let expr: Expr = x.parse()?;
            let diverge = if let Some(else_token) = x.parse()? {
                let else_token: Token![else] = else_token;
                let diverge = expr::Block {
                    attrs: Vec::new(),
                    label: None,
                    block: x.parse()?,
                };
                Some((else_token, Box::new(Expr::Block(diverge))))
            } else {
                None
            };
            Some(stmt::LocalInit {
                eq,
                expr: Box::new(expr),
                diverge,
            })
        } else {
            None
        };
        let semi: Token![;] = x.parse()?;
        Ok(stmt::Local {
            attrs,
            let_,
            pat,
            init,
            semi,
        })
    }
    fn stmt_expr(x: Stream, allow_nosemi: AllowNoSemi, mut attrs: Vec<attr::Attr>) -> Res<stmt::Stmt> {
        let mut e = expr_early(x)?;
        let mut attr_target = &mut e;
        loop {
            attr_target = match attr_target {
                Expr::Assign(e) => &mut e.left,
                Expr::Binary(e) => &mut e.left,
                Expr::Cast(e) => &mut e.expr,
                Expr::Array(_)
                | Expr::Async(_)
                | Expr::Await(_)
                | Expr::Block(_)
                | Expr::Break(_)
                | Expr::Call(_)
                | Expr::Closure(_)
                | Expr::Const(_)
                | Expr::Continue(_)
                | Expr::Field(_)
                | Expr::ForLoop(_)
                | Expr::Group(_)
                | Expr::If(_)
                | Expr::Index(_)
                | Expr::Infer(_)
                | Expr::Let(_)
                | Expr::Lit(_)
                | Expr::Loop(_)
                | Expr::Macro(_)
                | Expr::Match(_)
                | Expr::MethodCall(_)
                | Expr::Paren(_)
                | Expr::Path(_)
                | Expr::Range(_)
                | Expr::Reference(_)
                | Expr::Repeat(_)
                | Expr::Return(_)
                | Expr::Struct(_)
                | Expr::Try(_)
                | Expr::TryBlock(_)
                | Expr::Tuple(_)
                | Expr::Unary(_)
                | Expr::Unsafe(_)
                | Expr::While(_)
                | Expr::Yield(_)
                | Expr::Verbatim(_) => break,
            };
        }
        attrs.extend(attr_target.replace_attrs(Vec::new()));
        attr_target.replace_attrs(attrs);
        let semi: Option<Token![;]> = x.parse()?;
        match e {
            Expr::Macro(expr::Mac { attrs, mac }) if semi.is_some() || mac.delimiter.is_brace() => {
                return Ok(stmt::Stmt::Macro(stmt::Mac { attrs, mac, semi }));
            },
            _ => {},
        }
        if semi.is_some() {
            Ok(stmt::Stmt::Expr(e, semi))
        } else if allow_nosemi.0 || !expr::requires_terminator(&e) {
            Ok(stmt::Stmt::Expr(e, None))
        } else {
            Err(x.error("expected semicolon"))
        }
    }
}

mod parsing {
    impl Parse for data::Variant {
        fn parse(input: Stream) -> Res<Self> {
            let attrs = input.call(attr::Attr::parse_outer)?;
            let _visibility: Visibility = input.parse()?;
            let ident: Ident = input.parse()?;
            let fields = if input.peek(tok::Brace) {
                data::Fields::Named(input.parse()?)
            } else if input.peek(tok::Paren) {
                data::Fields::Unnamed(input.parse()?)
            } else {
                data::Fields::Unit
            };
            let discriminant = if input.peek(Token![=]) {
                let eq: Token![=] = input.parse()?;
                let discriminant: Expr = input.parse()?;
                Some((eq, discriminant))
            } else {
                None
            };
            Ok(data::Variant {
                attrs,
                ident,
                fields,
                discriminant,
            })
        }
    }
    impl Parse for data::Named {
        fn parse(input: Stream) -> Res<Self> {
            let content;
            Ok(data::Named {
                brace: braced!(content in input),
                named: content.parse_terminated(Field::parse_named, Token![,])?,
            })
        }
    }
    impl Parse for data::Unnamed {
        fn parse(input: Stream) -> Res<Self> {
            let content;
            Ok(data::Unnamed {
                paren: parenthesized!(content in input),
                unnamed: content.parse_terminated(Field::parse_unnamed, Token![,])?,
            })
        }
    }
    impl data::Field {
        pub fn parse_named(input: Stream) -> Res<Self> {
            Ok(data::Field {
                attrs: input.call(attr::Attr::parse_outer)?,
                vis: input.parse()?,
                mutability: FieldMutability::None,
                ident: Some(if input.peek(Token![_]) {
                    input.call(Ident::parse_any)
                } else {
                    input.parse()
                }?),
                colon: Some(input.parse()?),
                ty: input.parse()?,
            })
        }
        pub fn parse_unnamed(input: Stream) -> Res<Self> {
            Ok(data::Field {
                attrs: input.call(attr::Attr::parse_outer)?,
                vis: input.parse()?,
                mutability: FieldMutability::None,
                ident: None,
                colon: None,
                ty: input.parse()?,
            })
        }
    }
}

mod parsing {
    impl Parse for DeriveInput {
        fn parse(input: Stream) -> Res<Self> {
            let attrs = input.call(attr::Attr::parse_outer)?;
            let vis = input.parse::<Visibility>()?;
            let lookahead = input.lookahead1();
            if lookahead.peek(Token![struct]) {
                let struct_ = input.parse::<Token![struct]>()?;
                let ident = input.parse::<Ident>()?;
                let gens = input.parse::<gen::Gens>()?;
                let (where_clause, fields, semi) = data_struct(input)?;
                Ok(DeriveInput {
                    attrs,
                    vis,
                    ident,
                    gens: gen::Gens { where_clause, ..gens },
                    data: Data::Struct(data::Struct { struct_, fields, semi }),
                })
            } else if lookahead.peek(Token![enum]) {
                let enum_ = input.parse::<Token![enum]>()?;
                let ident = input.parse::<Ident>()?;
                let gens = input.parse::<gen::Gens>()?;
                let (where_clause, brace, variants) = data_enum(input)?;
                Ok(DeriveInput {
                    attrs,
                    vis,
                    ident,
                    gens: gen::Gens { where_clause, ..gens },
                    data: Data::Enum(data::Enum { enum_, brace, variants }),
                })
            } else if lookahead.peek(Token![union]) {
                let union_ = input.parse::<Token![union]>()?;
                let ident = input.parse::<Ident>()?;
                let gens = input.parse::<gen::Gens>()?;
                let (where_clause, fields) = data_union(input)?;
                Ok(DeriveInput {
                    attrs,
                    vis,
                    ident,
                    gens: gen::Gens { where_clause, ..gens },
                    data: Data::Union(data::Union { union_, fields }),
                })
            } else {
                Err(lookahead.error())
            }
        }
    }
    pub fn data_struct(input: Stream) -> Res<(Option<gen::Where>, data::Fields, Option<Token![;]>)> {
        let mut lookahead = input.lookahead1();
        let mut where_clause = None;
        if lookahead.peek(Token![where]) {
            where_clause = Some(input.parse()?);
            lookahead = input.lookahead1();
        }
        if where_clause.is_none() && lookahead.peek(tok::Paren) {
            let fields = input.parse()?;
            lookahead = input.lookahead1();
            if lookahead.peek(Token![where]) {
                where_clause = Some(input.parse()?);
                lookahead = input.lookahead1();
            }
            if lookahead.peek(Token![;]) {
                let semi = input.parse()?;
                Ok((where_clause, data::Fields::Unnamed(fields), Some(semi)))
            } else {
                Err(lookahead.error())
            }
        } else if lookahead.peek(tok::Brace) {
            let fields = input.parse()?;
            Ok((where_clause, data::Fields::Named(fields), None))
        } else if lookahead.peek(Token![;]) {
            let semi = input.parse()?;
            Ok((where_clause, data::Fields::Unit, Some(semi)))
        } else {
            Err(lookahead.error())
        }
    }
    pub fn data_enum(input: Stream) -> Res<(Option<gen::Where>, tok::Brace, Punctuated<data::Variant, Token![,]>)> {
        let where_clause = input.parse()?;
        let content;
        let brace = braced!(content in input);
        let variants = content.parse_terminated(data::Variant::parse, Token![,])?;
        Ok((where_clause, brace, variants))
    }
    pub fn data_union(input: Stream) -> Res<(Option<gen::Where>, data::Named)> {
        let where_clause = input.parse()?;
        let fields = input.parse()?;
        Ok((where_clause, fields))
    }
}

impl Parse for File {
    fn parse(x: Stream) -> Res<Self> {
        Ok(File {
            shebang: None,
            attrs: x.call(attr::Attr::parse_inner)?,
            items: {
                let mut ys = Vec::new();
                while !x.is_empty() {
                    ys.push(x.parse()?);
                }
                ys
            },
        })
    }
}

mod parsing {
    fn accept_as_ident(ident: &Ident) -> bool {
        match ident.to_string().as_str() {
            "_" | "abstract" | "as" | "async" | "await" | "become" | "box" | "break" | "const" | "continue"
            | "crate" | "do" | "dyn" | "else" | "enum" | "extern" | "false" | "final" | "fn" | "for" | "if"
            | "impl" | "in" | "let" | "loop" | "macro" | "match" | "mod" | "move" | "mut" | "override" | "priv"
            | "pub" | "ref" | "return" | "Self" | "self" | "static" | "struct" | "super" | "trait" | "true" | "try"
            | "type" | "typeof" | "unsafe" | "unsized" | "use" | "virtual" | "where" | "while" | "yield" => false,
            _ => true,
        }
    }
    impl Parse for Ident {
        fn parse(input: Stream) -> Res<Self> {
            input.step(|cursor| {
                if let Some((ident, rest)) = cursor.ident() {
                    if accept_as_ident(&ident) {
                        Ok((ident, rest))
                    } else {
                        Err(cursor.error(format_args!("expected identifier, found keyword `{}`", ident,)))
                    }
                } else {
                    Err(cursor.error("expected identifier"))
                }
            })
        }
    }
    impl Token for Ident {
        fn peek(cursor: Cursor) -> bool {
            if let Some((ident, _rest)) = cursor.ident() {
                accept_as_ident(&ident)
            } else {
                false
            }
        }
        fn display() -> &'static str {
            "identifier"
        }
    }
}

mod parsing {
    impl Parse for Lifetime {
        fn parse(input: Stream) -> Res<Self> {
            input.step(|cursor| cursor.lifetime().ok_or_else(|| cursor.error("expected lifetime")))
        }
    }
}

mod parsing {
    impl Parse for Macro {
        fn parse(input: Stream) -> Res<Self> {
            let tokens;
            Ok(Macro {
                path: input.call(Path::parse_mod_style)?,
                bang: input.parse()?,
                delimiter: {
                    let (delimiter, content) = mac::parse_delim(input)?;
                    tokens = content;
                    delimiter
                },
                tokens,
            })
        }
    }
}

mod parsing {
    fn parse_binop(input: Stream) -> Res<BinOp> {
        if input.peek(Token![&&]) {
            input.parse().map(BinOp::And)
        } else if input.peek(Token![||]) {
            input.parse().map(BinOp::Or)
        } else if input.peek(Token![<<]) {
            input.parse().map(BinOp::Shl)
        } else if input.peek(Token![>>]) {
            input.parse().map(BinOp::Shr)
        } else if input.peek(Token![==]) {
            input.parse().map(BinOp::Eq)
        } else if input.peek(Token![<=]) {
            input.parse().map(BinOp::Le)
        } else if input.peek(Token![!=]) {
            input.parse().map(BinOp::Ne)
        } else if input.peek(Token![>=]) {
            input.parse().map(BinOp::Ge)
        } else if input.peek(Token![+]) {
            input.parse().map(BinOp::Add)
        } else if input.peek(Token![-]) {
            input.parse().map(BinOp::Sub)
        } else if input.peek(Token![*]) {
            input.parse().map(BinOp::Mul)
        } else if input.peek(Token![/]) {
            input.parse().map(BinOp::Div)
        } else if input.peek(Token![%]) {
            input.parse().map(BinOp::Rem)
        } else if input.peek(Token![^]) {
            input.parse().map(BinOp::BitXor)
        } else if input.peek(Token![&]) {
            input.parse().map(BinOp::BitAnd)
        } else if input.peek(Token![|]) {
            input.parse().map(BinOp::BitOr)
        } else if input.peek(Token![<]) {
            input.parse().map(BinOp::Lt)
        } else if input.peek(Token![>]) {
            input.parse().map(BinOp::Gt)
        } else {
            Err(input.error("expected binary operator"))
        }
    }
    impl Parse for BinOp {
        #[cfg(not(feature = "full"))]
        fn parse(input: Stream) -> Res<Self> {
            parse_binop(input)
        }
        fn parse(input: Stream) -> Res<Self> {
            if input.peek(Token![+=]) {
                input.parse().map(BinOp::AddAssign)
            } else if input.peek(Token![-=]) {
                input.parse().map(BinOp::SubAssign)
            } else if input.peek(Token![*=]) {
                input.parse().map(BinOp::MulAssign)
            } else if input.peek(Token![/=]) {
                input.parse().map(BinOp::DivAssign)
            } else if input.peek(Token![%=]) {
                input.parse().map(BinOp::RemAssign)
            } else if input.peek(Token![^=]) {
                input.parse().map(BinOp::BitXorAssign)
            } else if input.peek(Token![&=]) {
                input.parse().map(BinOp::BitAndAssign)
            } else if input.peek(Token![|=]) {
                input.parse().map(BinOp::BitOrAssign)
            } else if input.peek(Token![<<=]) {
                input.parse().map(BinOp::ShlAssign)
            } else if input.peek(Token![>>=]) {
                input.parse().map(BinOp::ShrAssign)
            } else {
                parse_binop(input)
            }
        }
    }
    impl Parse for UnOp {
        fn parse(input: Stream) -> Res<Self> {
            let lookahead = input.lookahead1();
            if lookahead.peek(Token![*]) {
                input.parse().map(UnOp::Deref)
            } else if lookahead.peek(Token![!]) {
                input.parse().map(UnOp::Not)
            } else if lookahead.peek(Token![-]) {
                input.parse().map(UnOp::Neg)
            } else {
                Err(lookahead.error())
            }
        }
    }
}

mod parsing {
    impl Parse for Visibility {
        fn parse(input: Stream) -> Res<Self> {
            if input.peek(tok::Group) {
                let ahead = input.fork();
                let group = super::parse_group(&ahead)?;
                if group.content.is_empty() {
                    input.advance_to(&ahead);
                    return Ok(Visibility::Inherited);
                }
            }
            if input.peek(Token![pub]) {
                Self::parse_pub(input)
            } else {
                Ok(Visibility::Inherited)
            }
        }
    }
    impl Visibility {
        fn parse_pub(input: Stream) -> Res<Self> {
            let pub_ = input.parse::<Token![pub]>()?;
            if input.peek(tok::Paren) {
                let ahead = input.fork();
                let content;
                let paren = parenthesized!(content in ahead);
                if content.peek(Token![crate]) || content.peek(Token![self]) || content.peek(Token![super]) {
                    let path = content.call(Ident::parse_any)?;
                    if content.is_empty() {
                        input.advance_to(&ahead);
                        return Ok(Visibility::Restricted(VisRestricted {
                            pub_,
                            paren,
                            in_: None,
                            path: Box::new(Path::from(path)),
                        }));
                    }
                } else if content.peek(Token![in]) {
                    let in_: Token![in] = content.parse()?;
                    let path = content.call(Path::parse_mod_style)?;
                    input.advance_to(&ahead);
                    return Ok(Visibility::Restricted(VisRestricted {
                        pub_,
                        paren,
                        in_: Some(in_),
                        path: Box::new(path),
                    }));
                }
            }
            Ok(Visibility::Public(pub_))
        }
        pub fn is_some(&self) -> bool {
            match self {
                Visibility::Inherited => false,
                _ => true,
            }
        }
    }
}

pub mod parsing {
    pub fn keyword(input: Stream, token: &str) -> Res<pm2::Span> {
        input.step(|cursor| {
            if let Some((ident, rest)) = cursor.ident() {
                if ident == token {
                    return Ok((ident.span(), rest));
                }
            }
            Err(cursor.error(format!("expected `{}`", token)))
        })
    }
    pub fn peek_keyword(cursor: Cursor, token: &str) -> bool {
        if let Some((ident, _rest)) = cursor.ident() {
            ident == token
        } else {
            false
        }
    }
    pub fn punct<const N: usize>(input: Stream, token: &str) -> Res<[pm2::Span; N]> {
        let mut spans = [input.span(); N];
        punct_helper(input, token, &mut spans)?;
        Ok(spans)
    }
    fn punct_helper(input: Stream, token: &str, spans: &mut [pm2::Span]) -> Res<()> {
        input.step(|cursor| {
            let mut cursor = *cursor;
            assert_eq!(token.len(), spans.len());
            for (i, ch) in token.chars().enumerate() {
                match cursor.punct() {
                    Some((punct, rest)) => {
                        spans[i] = punct.span();
                        if punct.as_char() != ch {
                            break;
                        } else if i == token.len() - 1 {
                            return Ok(((), rest));
                        } else if punct.spacing() != pm2::Spacing::Joint {
                            break;
                        }
                        cursor = rest;
                    },
                    None => break,
                }
            }
            Err(Err::new(spans[0], format!("expected `{}`", token)))
        })
    }
    pub fn peek_punct(mut cursor: Cursor, token: &str) -> bool {
        for (i, ch) in token.chars().enumerate() {
            match cursor.punct() {
                Some((punct, rest)) => {
                    if punct.as_char() != ch {
                        break;
                    } else if i == token.len() - 1 {
                        return true;
                    } else if punct.spacing() != pm2::Spacing::Joint {
                        break;
                    }
                    cursor = rest;
                },
                None => break,
            }
        }
        false
    }
}
