use super::*;

struct NoSemi(bool);

pub struct Block {
    pub brace: tok::Brace,
    pub stmts: Vec<Stmt>,
}
impl Block {
    pub fn parse_within(s: Stream) -> Res<Vec<Stmt>> {
        let mut ys = Vec::new();
        loop {
            while let semi @ Some(_) = s.parse()? {
                ys.push(Stmt::Expr(Expr::Verbatim(pm2::Stream::new()), semi));
            }
            if s.is_empty() {
                break;
            }
            let y = parse_stmt(s, NoSemi(true))?;
            let semi = match &y {
                Stmt::Expr(x, None) => expr::requires_terminator(x),
                Stmt::Mac(x) => x.semi.is_none() && !x.mac.delim.is_brace(),
                Stmt::Local(_) | Stmt::Item(_) | Stmt::Expr(_, Some(_)) => false,
            };
            ys.push(y);
            if s.is_empty() {
                break;
            } else if semi {
                return Err(s.error("unexpected token, expected `;`"));
            }
        }
        Ok(ys)
    }
}
impl Parse for Block {
    fn parse(s: Stream) -> Res<Self> {
        let y;
        Ok(Block {
            brace: braced!(y in s),
            stmts: y.call(stmt::Block::parse_within)?,
        })
    }
}
impl ToTokens for Block {
    fn to_tokens(&self, ys: &mut Stream) {
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
impl Parse for Stmt {
    fn parse(x: Stream) -> Res<Self> {
        let nosemi = NoSemi(false);
        parse_stmt(x, nosemi)
    }
}
impl ToTokens for Stmt {
    fn to_tokens(&self, ys: &mut Stream) {
        match self {
            Stmt::Local(x) => x.to_tokens(ys),
            Stmt::Item(x) => x.to_tokens(ys),
            Stmt::Expr(x, semi) => {
                x.to_tokens(ys);
                semi.to_tokens(ys);
            },
            Stmt::Mac(x) => x.to_tokens(ys),
        }
    }
}

pub use expr::Expr;
pub use item::Item;

pub struct Local {
    pub attrs: Vec<attr::Attr>,
    pub let_: Token![let],
    pub pat: pat::Pat,
    pub init: Option<Init>,
    pub semi: Token![;],
}
impl ToTokens for Local {
    fn to_tokens(&self, ys: &mut Stream) {
        attr::outers_to_tokens(&self.attrs, ys);
        self.let_.to_tokens(ys);
        self.pat.to_tokens(ys);
        if let Some(x) = &self.init {
            x.eq.to_tokens(ys);
            x.expr.to_tokens(ys);
            if let Some((else_, diverge)) = &x.diverge {
                else_.to_tokens(ys);
                diverge.to_tokens(ys);
            }
        }
        self.semi.to_tokens(ys);
    }
}

pub struct Init {
    pub eq: Token![=],
    pub expr: Box<Expr>,
    pub diverge: Option<(Token![else], Box<Expr>)>,
}

pub struct Mac {
    pub attrs: Vec<attr::Attr>,
    pub mac: mac::Mac,
    pub semi: Option<Token![;]>,
}
impl ToTokens for Mac {
    fn to_tokens(&self, ys: &mut Stream) {
        attr::outers_to_tokens(&self.attrs, ys);
        self.mac.to_tokens(ys);
        self.semi.to_tokens(ys);
    }
}

fn parse_stmt(s: Stream, nosemi: NoSemi) -> Res<Stmt> {
    let beg = s.fork();
    let attrs = s.call(attr::Attr::parse_outers)?;
    let ahead = s.fork();
    let mut is_mac = false;
    if let Ok(x) = ahead.call(Path::parse_mod_style) {
        if ahead.peek(Token![!]) {
            if ahead.peek2(Ident) || ahead.peek2(Token![try]) {
                is_mac = true;
            } else if ahead.peek2(tok::Brace) && !(ahead.peek3(Token![.]) || ahead.peek3(Token![?])) {
                s.advance_to(&ahead);
                return parse_mac(s, attrs, x).map(Stmt::Mac);
            }
        }
    }
    if s.peek(Token![let]) {
        parse_local(s, attrs).map(Stmt::Local)
    } else if s.peek(Token![pub])
        || s.peek(Token![crate]) && !s.peek2(Token![::])
        || s.peek(Token![extern])
        || s.peek(Token![use])
        || s.peek(Token![static])
            && (s.peek2(Token![mut])
                || s.peek2(Ident) && !(s.peek2(Token![async]) && (s.peek3(Token![move]) || s.peek3(Token![|]))))
        || s.peek(Token![const])
            && !(s.peek2(tok::Brace)
                || s.peek2(Token![static])
                || s.peek2(Token![async])
                    && !(s.peek3(Token![unsafe]) || s.peek3(Token![extern]) || s.peek3(Token![fn]))
                || s.peek2(Token![move])
                || s.peek2(Token![|]))
        || s.peek(Token![unsafe]) && !s.peek2(tok::Brace)
        || s.peek(Token![async]) && (s.peek2(Token![unsafe]) || s.peek2(Token![extern]) || s.peek2(Token![fn]))
        || s.peek(Token![fn])
        || s.peek(Token![mod])
        || s.peek(Token![type])
        || s.peek(Token![struct])
        || s.peek(Token![enum])
        || s.peek(Token![union]) && s.peek2(Ident)
        || s.peek(Token![auto]) && s.peek2(Token![trait])
        || s.peek(Token![trait])
        || s.peek(Token![default]) && (s.peek2(Token![unsafe]) || s.peek2(Token![impl]))
        || s.peek(Token![impl])
        || s.peek(Token![macro])
        || is_mac
    {
        let x = item::parse_rest_of_item(beg, attrs, s)?;
        Ok(Stmt::Item(x))
    } else {
        parse_expr(s, attrs, nosemi)
    }
}
fn parse_mac(s: Stream, attrs: Vec<attr::Attr>, path: Path) -> Res<Mac> {
    let bang: Token![!] = s.parse()?;
    let (delim, toks) = mac::parse_delim(s)?;
    let semi: Option<Token![;]> = s.parse()?;
    Ok(Mac {
        attrs,
        mac: mac::Mac {
            path,
            bang,
            delim,
            toks,
        },
        semi,
    })
}
fn parse_local(s: Stream, attrs: Vec<attr::Attr>) -> Res<Local> {
    let let_: Token![let] = s.parse()?;
    let mut pat = pat::Pat::parse_single(s)?;
    if s.peek(Token![:]) {
        let colon: Token![:] = s.parse()?;
        let typ: Type = s.parse()?;
        pat = pat::Pat::Type(pat::Type {
            attrs: Vec::new(),
            pat: Box::new(pat),
            colon,
            typ: Box::new(typ),
        });
    }
    let init = if let Some(eq) = s.parse()? {
        let eq: Token![=] = eq;
        let expr: Expr = s.parse()?;
        let diverge = if let Some(else_) = s.parse()? {
            let else_: Token![else] = else_;
            let diverge = expr::Block {
                attrs: Vec::new(),
                label: None,
                block: s.parse()?,
            };
            Some((else_, Box::new(Expr::Block(diverge))))
        } else {
            None
        };
        Some(Init {
            eq,
            expr: Box::new(expr),
            diverge,
        })
    } else {
        None
    };
    let semi: Token![;] = s.parse()?;
    Ok(Local {
        attrs,
        let_,
        pat,
        init,
        semi,
    })
}
fn parse_expr(s: Stream, mut attrs: Vec<attr::Attr>, nosemi: NoSemi) -> Res<Stmt> {
    let mut y = expr::expr_early(s)?;
    let mut tgt = &mut y;
    loop {
        tgt = match tgt {
            Expr::Assign(x) => &mut x.left,
            Expr::Binary(x) => &mut x.left,
            Expr::Cast(x) => &mut x.expr,
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
            | Expr::Mac(_)
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
    attrs.extend(tgt.replace_attrs(Vec::new()));
    tgt.replace_attrs(attrs);
    let semi: Option<Token![;]> = s.parse()?;
    match y {
        Expr::Mac(expr::Mac { attrs, mac }) if semi.is_some() || mac.delim.is_brace() => {
            return Ok(Stmt::Mac(Mac { attrs, mac, semi }));
        },
        _ => {},
    }
    if semi.is_some() {
        Ok(Stmt::Expr(y, semi))
    } else if nosemi.0 || !expr::requires_terminator(&y) {
        Ok(Stmt::Expr(y, None))
    } else {
        Err(s.error("expected semicolon"))
    }
}
