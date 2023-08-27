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
impl Clone for Stmt {
    fn clone(&self) -> Self {
        use Stmt::*;
        match self {
            Expr(x, v1) => Expr(x.clone(), v1.clone()),
            Item(x) => Item(x.clone()),
            Local(x) => Local(x.clone()),
            Mac(x) => Mac(x.clone()),
        }
    }
}
impl Debug for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("stmt::Stmt::")?;
        use Stmt::*;
        match self {
            Local(x) => x.debug(f, "Local"),
            Item(x) => {
                let mut f = f.debug_tuple("Item");
                f.field(x);
                f.finish()
            },
            Expr(x, v1) => {
                let mut f = f.debug_tuple("Expr");
                f.field(x);
                f.field(v1);
                f.finish()
            },
            Mac(x) => x.debug(f, "Mac"),
        }
    }
}
impl Eq for Stmt {}
impl PartialEq for Stmt {
    fn eq(&self, x: &Self) -> bool {
        use Stmt::*;
        match (self, x) {
            (Expr(x, self1), Expr(y, other1)) => x == y && self1 == other1,
            (Item(x), Item(y)) => x == y,
            (Local(x), Local(y)) => x == y,
            (Mac(x), Mac(y)) => x == y,
            _ => false,
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
impl<F: Folder + ?Sized> Fold for Stmt {
    fn fold(&self, f: &mut F) {
        use Stmt::*;
        match self {
            Expr(x, y) => Expr(x.fold(f), y),
            Item(x) => Item(x.fold(f)),
            Local(x) => Local(x.fold(f)),
            Mac(x) => Mac(x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Stmt {
    fn hash(&self, h: &mut H) {
        use Stmt::*;
        match self {
            Local(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Item(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Expr(x, v1) => {
                h.write_u8(2u8);
                x.hash(h);
                v1.hash(h);
            },
            Mac(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
        }
    }
}
impl<V: Visitor + ?Sized> Visit for Stmt {
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
impl Clone for Local {
    fn clone(&self) -> Self {
        Local {
            attrs: self.attrs.clone(),
            let_: self.let_.clone(),
            pat: self.pat.clone(),
            init: self.init.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Debug for Local {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Local {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("attrs", &self.attrs);
                f.field("let_", &self.let_);
                f.field("pat", &self.pat);
                f.field("init", &self.init);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "stmt::Local")
    }
}
impl Eq for Local {}
impl PartialEq for Local {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.pat == x.pat && self.init == x.init
    }
}
impl<F: Folder + ?Sized> Fold for Local {
    fn fold(&self, f: &mut F) {
        Local {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            let_: self.let_,
            pat: self.pat.fold(f),
            init: (self.init).map(|x| x.fold(f)),
            semi: self.semi,
        }
    }
}
impl<H: Hasher> Hash for Local {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.init.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Local {
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
impl Clone for Mac {
    fn clone(&self) -> Self {
        Mac {
            attrs: self.attrs.clone(),
            mac: self.mac.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Debug for Mac {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Mac {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("attrs", &self.attrs);
                f.field("mac", &self.mac);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "stmt::Mac")
    }
}
impl Eq for Mac {}
impl PartialEq for Mac {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.mac == x.mac && self.semi == x.semi
    }
}
impl<F: Folder + ?Sized> Fold for Mac {
    fn fold(&self, f: &mut F) {
        Mac {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            mac: self.mac.fold(f),
            semi: self.semi,
        }
    }
}
impl<H: Hasher> Hash for Mac {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.mac.hash(h);
        self.semi.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Mac {
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
impl Clone for Block {
    fn clone(&self) -> Self {
        Block {
            brace: self.brace.clone(),
            stmts: self.stmts.clone(),
        }
    }
}
impl Debug for Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Block");
        f.field("brace", &self.brace);
        f.field("stmts", &self.stmts);
        f.finish()
    }
}
impl Eq for Block {}
impl PartialEq for Block {
    fn eq(&self, x: &Self) -> bool {
        self.stmts == x.stmts
    }
}
impl<F: Folder + ?Sized> Fold for Block {
    fn fold(&self, f: &mut F) {
        Block {
            brace: self.brace,
            stmts: FoldHelper::lift(self.stmts, |x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Block {
    fn hash(&self, h: &mut H) {
        self.stmts.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Block {
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
impl Clone for Init {
    fn clone(&self) -> Self {
        Init {
            eq: self.eq.clone(),
            expr: self.expr.clone(),
            diverge: self.diverge.clone(),
        }
    }
}
impl Debug for Init {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("stmt::Init");
        f.field("eq", &self.eq);
        f.field("expr", &self.expr);
        f.field("diverge", &self.diverge);
        f.finish()
    }
}
impl Eq for Init {}
impl PartialEq for Init {
    fn eq(&self, x: &Self) -> bool {
        self.expr == x.expr && self.diverge == x.diverge
    }
}
impl<F: Folder + ?Sized> Fold for Init {
    fn fold(&self, f: &mut F) {
        Init {
            eq: self.eq,
            expr: Box::new(*self.expr.fold(f)),
            diverge: (self.diverge).map(|x| ((x).0, Box::new(*(x).1.fold(f)))),
        }
    }
}
impl<H: Hasher> Hash for Init {
    fn hash(&self, h: &mut H) {
        self.expr.hash(h);
        self.diverge.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Init {
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
    let (delim, toks) = tok::parse_delim(s)?;
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
    loop {
        use Expr::*;
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
        Expr::Mac(expr::Mac { attrs, mac }) if semi.is_some() || mac.delim.is_brace() => {
            return Ok(Stmt::Mac(Mac { attrs, mac, semi }));
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
