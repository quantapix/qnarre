
struct AllowNoSemi(bool);

pub struct Block {
    pub brace: tok::Brace,
    pub stmts: Vec<Stmt>,
}
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
impl ToTokens for Block {
    fn to_tokens(&self, ys: &mut pm2::Stream) {
        self.brace.surround(ys, |ys| {
            ys.append_all(&self.stmts);
        });
    }
}

pub enum Stmt {
    Local(Local),
    Item(Item),
    Expr(Expr, Option<Token![;]>),
    Mac(Mac),
}
impl Parse for stmt::Stmt {
    fn parse(x: Stream) -> Res<Self> {
        let allow_nosemi = AllowNoSemi(false);
        parse_stmt(x, allow_nosemi)
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

pub struct Local {
    pub attrs: Vec<attr::Attr>,
    pub let_: Token![let],
    pub pat: pat::Pat,
    pub init: Option<LocalInit>,
    pub semi: Token![;],
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

pub struct LocalInit {
    pub eq: Token![=],
    pub expr: Box<Expr>,
    pub diverge: Option<(Token![else], Box<Expr>)>,
}
pub struct Mac {
    pub attrs: Vec<attr::Attr>,
    pub mac: Macro,
    pub semi: Option<Token![;]>,
}
impl ToTokens for stmt::Mac {
    fn to_tokens(&self, xs: &mut pm2::Stream) {
        outer_attrs_to_tokens(&self.attrs, xs);
        self.mac.to_tokens(xs);
        self.semi.to_tokens(xs);
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
