use super::*;

struct NoSemi(bool);

pub enum Stmt {
    Expr(Expr, Option<Token![;]>),
    Item(Item),
    Local(Local),
    Mac(Mac),
}
impl Parse for Stmt {
    fn parse(s: Stream) -> Res<Self> {
        let y = NoSemi(false);
        parse_stmt(s, y)
    }
}
impl Lower for Stmt {
    fn lower(&self, s: &mut Stream) {
        use Stmt::*;
        match self {
            Expr(x, semi) => {
                x.lower(s);
                semi.lower(s);
            },
            Item(x) => x.lower(s),
            Local(x) => x.lower(s),
            Mac(x) => x.lower(s),
        }
    }
}
impl Pretty for Stmt {
    fn pretty(&self, p: &mut Print) {
        use Stmt::*;
        match self {
            Expr(x, None) => {
                if x.break_after() {
                    p.ibox(0);
                    x.pretty_beg_line(p, true);
                    if x.add_semi() {
                        p.word(";");
                    }
                    p.end();
                    p.hardbreak();
                } else {
                    x.pretty_beg_line(p, true);
                }
            },
            Expr(x, Some(_)) => {
                if let expr::Expr::Verbatim(x) = x {
                    if x.is_empty() {
                        return;
                    }
                }
                p.ibox(0);
                x.pretty_beg_line(p, true);
                if !x.remove_semi() {
                    p.word(";");
                }
                p.end();
                p.hardbreak();
            },
            Item(x) => x.pretty(p),
            Local(x) => {
                p.outer_attrs(&x.attrs);
                p.ibox(0);
                p.word("let ");
                &x.pat.pretty(p);
                if let Some(x) = &x.init {
                    p.word(" = ");
                    p.neverbreak();
                    &x.expr.pretty(p);
                    if let Some((_, x)) = &x.diverge {
                        p.word(" else ");
                        if let expr::Expr::Block(x) = x.as_ref() {
                            p.small_block(&x.block, &[]);
                        } else {
                            p.word("{");
                            p.space();
                            p.ibox(INDENT);
                            x.pretty(p);
                            p.end();
                            p.space();
                            p.offset(-INDENT);
                            p.word("}");
                        }
                    }
                }
                p.word(";");
                p.end();
                p.hardbreak();
            },
            Mac(x) => {
                p.outer_attrs(&x.attrs);
                let semi = true;
                &x.mac.pretty_with_args(p, (None, semi));
                p.hardbreak();
            },
        }
    }
}
impl<V> Visit for Stmt
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        use Stmt::*;
        match self {
            Local(x) => {
                x.visit(v);
            },
            Item(x) => {
                x.visit(v);
            },
            Expr(x, _) => {
                x.visit(v);
            },
            Mac(x) => {
                x.visit(v);
            },
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        use Stmt::*;
        match self {
            Local(x) => {
                x.visit_mut(v);
            },
            Item(x) => {
                x.visit_mut(v);
            },
            Expr(x, _) => {
                x.visit_mut(v);
            },
            Mac(x) => {
                x.visit_mut(v);
            },
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
impl Lower for Local {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.let_.lower(s);
        self.pat.lower(s);
        if let Some(x) = &self.init {
            x.eq.lower(s);
            x.expr.lower(s);
            if let Some((else_, diverge)) = &x.diverge {
                else_.lower(s);
                diverge.lower(s);
            }
        }
        self.semi.lower(s);
    }
}
impl<V> Visit for Local
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.pat.visit(v);
        if let Some(x) = &self.init {
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut self.pat.visit_mut(v);
        if let Some(x) = &mut self.init {
            x.visit_mut(v);
        }
    }
}

pub struct Mac {
    pub attrs: Vec<attr::Attr>,
    pub mac: mac::Mac,
    pub semi: Option<Token![;]>,
}
impl Lower for Mac {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.mac.lower(s);
        self.semi.lower(s);
    }
}
impl<V> Visit for Mac
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.mac.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut self.mac.visit_mut(v);
    }
}

pub struct Block {
    pub brace: tok::Brace,
    pub stmts: Vec<Stmt>,
}
impl Block {
    pub fn parse_within(s: Stream) -> Res<Vec<Stmt>> {
        let mut ys = Vec::new();
        loop {
            use Stmt::*;
            while let semi @ Some(_) = s.parse()? {
                ys.push(Expr(expr::Expr::Verbatim(pm2::Stream::new()), semi));
            }
            if s.is_empty() {
                break;
            }
            let y = parse_stmt(s, NoSemi(true))?;
            let semi = match &y {
                Expr(x, None) => x.needs_term(),
                Mac(x) => x.semi.is_none() && !x.mac.delim.is_brace(),
                Local(_) | Item(_) | Expr(_, Some(_)) => false,
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
            stmts: y.call(Block::parse_within)?,
        })
    }
}
impl Lower for Block {
    fn lower(&self, s: &mut Stream) {
        self.brace.surround(s, |s| {
            s.append_all(&self.stmts);
        });
    }
}
impl<V> Visit for Block
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        for x in &self.stmts {
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.stmts {
            x.visit_mut(v);
        }
    }
}

pub struct Init {
    pub eq: Token![=],
    pub expr: Box<Expr>,
    pub diverge: Option<(Token![else], Box<Expr>)>,
}
impl<V> Visit for Init
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        &*self.expr.visit(v);
        if let Some(x) = &self.diverge {
            &*(x).1.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut *self.expr.visit_mut(v);
        if let Some(x) = &mut self.diverge {
            &mut *(x).1.visit_mut(v);
        }
    }
}

fn parse_stmt(s: Stream, nosemi: NoSemi) -> Res<Stmt> {
    let beg = s.fork();
    let attrs = s.call(attr::Attr::parse_outers)?;
    let ahead = s.fork();
    let mut is_mac = false;
    if let Ok(x) = ahead.call(Path::parse_mod_style) {
        if ahead.peek(Token![!]) {
            if ahead.peek2(ident::Ident) || ahead.peek2(Token![try]) {
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
                || s.peek2(ident::Ident) && !(s.peek2(Token![async]) && (s.peek3(Token![move]) || s.peek3(Token![|]))))
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
        || s.peek(Token![union]) && s.peek2(ident::Ident)
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
    let mut pat = pat::Pat::parse_one(s)?;
    if s.peek(Token![:]) {
        let colon: Token![:] = s.parse()?;
        let typ: typ::Type = s.parse()?;
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
    use Expr::*;
    loop {
        tgt = match tgt {
            Assign(x) => &mut x.left,
            Binary(x) => &mut x.left,
            Cast(x) => &mut x.expr,
            Array(_) | Async(_) | Await(_) | Block(_) | Break(_) | Call(_) | Closure(_) | Const(_) | Continue(_)
            | Field(_) | For(_) | Group(_) | If(_) | Index(_) | Infer(_) | Let(_) | Lit(_) | Loop(_) | Mac(_)
            | Match(_) | Method(_) | Parenth(_) | Path(_) | Range(_) | Ref(_) | Repeat(_) | Return(_) | Struct(_)
            | Try(_) | TryBlock(_) | Tuple(_) | Unary(_) | Unsafe(_) | While(_) | Yield(_) | Verbatim(_) => break,
        };
    }
    attrs.extend(tgt.replace_attrs(Vec::new()));
    tgt.replace_attrs(attrs);
    let semi: Option<Token![;]> = s.parse()?;
    match y {
        Mac(expr::Mac { attrs, mac }) if semi.is_some() || mac.delim.is_brace() => {
            return Ok(Stmt::Mac(stmt::Mac { attrs, mac, semi }));
        },
        _ => {},
    }
    if semi.is_some() {
        Ok(Stmt::Expr(y, semi))
    } else if nosemi.0 || !&y.needs_term() {
        Ok(Stmt::Expr(y, None))
    } else {
        Err(s.error("expected semicolon"))
    }
}
