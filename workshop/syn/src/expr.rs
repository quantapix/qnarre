use quote::IdentFragment;

pub use pm2::Stream;

ast_enum_of_structs! {
    pub enum Expr {
        Array(Array),
        Assign(Assign),
        Async(Async),
        Await(Await),
        Binary(Binary),
        Block(Block),
        Break(Break),
        Call(Call),
        Cast(Cast),
        Closure(Closure),
        Const(Const),
        Continue(Continue),
        Field(Field),
        ForLoop(ForLoop),
        Group(Group),
        If(If),
        Index(Index),
        Infer(Infer),
        Let(Let),
        Lit(Lit),
        Loop(Loop),
        Mac(Mac),
        Match(Match),
        MethodCall(MethodCall),
        Paren(Paren),
        Path(Path),
        Range(Range),
        Ref(Ref),
        Repeat(Repeat),
        Return(Return),
        Struct(Struct),
        Try(Try),
        TryBlock(TryBlock),
        Tuple(Tuple),
        Unary(Unary),
        Unsafe(Unsafe),
        Verbatim(Stream),
        While(While),
        Yield(Yield),
    }
}

pub struct Array {
    pub attrs: Vec<attr::Attr>,
    pub bracket: tok::Bracket,
    pub elems: Punctuated<Expr, Token![,]>,
}

pub struct Assign {
    pub attrs: Vec<attr::Attr>,
    pub left: Box<Expr>,
    pub eq: Token![=],
    pub right: Box<Expr>,
}
pub struct Async {
    pub attrs: Vec<attr::Attr>,
    pub async_: Token![async],
    pub move_: Option<Token![move]>,
    pub block: stmt::Block,
}
pub struct Await {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub dot: Token![.],
    pub await_: Token![await],
}
pub struct Binary {
    pub attrs: Vec<attr::Attr>,
    pub left: Box<Expr>,
    pub op: BinOp,
    pub right: Box<Expr>,
}
pub struct Block {
    pub attrs: Vec<attr::Attr>,
    pub label: Option<Label>,
    pub block: stmt::Block,
}
pub struct Break {
    pub attrs: Vec<attr::Attr>,
    pub break_: Token![break],
    pub label: Option<Lifetime>,
    pub expr: Option<Box<Expr>>,
}
pub struct Call {
    pub attrs: Vec<attr::Attr>,
    pub func: Box<Expr>,
    pub paren: tok::Paren,
    pub args: Punctuated<Expr, Token![,]>,
}
pub struct Cast {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub as_: Token![as],
    pub typ: Box<typ::Type>,
}
pub struct Closure {
    pub attrs: Vec<attr::Attr>,
    pub lifes: Option<Bgen::bound::Lifes>,
    pub const_: Option<Token![const]>,
    pub static_: Option<Token![static]>,
    pub async_: Option<Token![async]>,
    pub move_: Option<Token![move]>,
    pub or1: Token![|],
    pub ins: Punctuated<pat::Pat, Token![,]>,
    pub or2: Token![|],
    pub ret: typ::Ret,
    pub body: Box<Expr>,
}
pub struct Const {
    pub attrs: Vec<attr::Attr>,
    pub const_: Token![const],
    pub block: stmt::Block,
}
pub struct Continue {
    pub attrs: Vec<attr::Attr>,
    pub continue_: Token![continue],
    pub label: Option<Lifetime>,
}
pub struct Field {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub dot: Token![.],
    pub memb: Member,
}
pub struct ForLoop {
    pub attrs: Vec<attr::Attr>,
    pub label: Option<Label>,
    pub for_: Token![for],
    pub pat: Box<pat::Pat>,
    pub in_: Token![in],
    pub expr: Box<Expr>,
    pub body: stmt::Block,
}
pub struct Group {
    pub attrs: Vec<attr::Attr>,
    pub group: tok::Group,
    pub expr: Box<Expr>,
}
pub struct If {
    pub attrs: Vec<attr::Attr>,
    pub if_: Token![if],
    pub cond: Box<Expr>,
    pub then_: stmt::Block,
    pub else_: Option<(Token![else], Box<Expr>)>,
}
pub struct Index {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub bracket: tok::Bracket,
    pub idx: Box<Expr>,
}
pub struct Infer {
    pub attrs: Vec<attr::Attr>,
    pub underscore: Token![_],
}
pub struct Let {
    pub attrs: Vec<attr::Attr>,
    pub let_: Token![let],
    pub pat: Box<pat::Pat>,
    pub eq: Token![=],
    pub expr: Box<Expr>,
}
pub struct Lit {
    pub attrs: Vec<attr::Attr>,
    pub lit: lit::Lit,
}
pub struct Loop {
    pub attrs: Vec<attr::Attr>,
    pub label: Option<Label>,
    pub loop_: Token![loop],
    pub body: stmt::Block,
}
pub struct Mac {
    pub attrs: Vec<attr::Attr>,
    pub mac: mac::Mac,
}
pub struct Match {
    pub attrs: Vec<attr::Attr>,
    pub match_: Token![match],
    pub expr: Box<Expr>,
    pub brace: tok::Brace,
    pub arms: Vec<Arm>,
}
pub struct MethodCall {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub dot: Token![.],
    pub method: Ident,
    pub turbofish: Option<AngledArgs>,
    pub paren: tok::Paren,
    pub args: Punctuated<Expr, Token![,]>,
}
pub struct Paren {
    pub attrs: Vec<attr::Attr>,
    pub paren: tok::Paren,
    pub expr: Box<Expr>,
}
pub struct Path {
    pub attrs: Vec<attr::Attr>,
    pub qself: Option<QSelf>,
    pub path: path::Path,
}
pub struct Range {
    pub attrs: Vec<attr::Attr>,
    pub beg: Option<Box<Expr>>,
    pub limits: RangeLimits,
    pub end: Option<Box<Expr>>,
}
pub struct Ref {
    pub attrs: Vec<attr::Attr>,
    pub and: Token![&],
    pub mut_: Option<Token![mut]>,
    pub expr: Box<Expr>,
}
pub struct Repeat {
    pub attrs: Vec<attr::Attr>,
    pub bracket: tok::Bracket,
    pub expr: Box<Expr>,
    pub semi: Token![;],
    pub len: Box<Expr>,
}
pub struct Return {
    pub attrs: Vec<attr::Attr>,
    pub return_: Token![return],
    pub expr: Option<Box<Expr>>,
}
pub struct Struct {
    pub attrs: Vec<attr::Attr>,
    pub qself: Option<QSelf>,
    pub path: path::Path,
    pub brace: tok::Brace,
    pub fields: Punctuated<FieldValue, Token![,]>,
    pub dot2: Option<Token![..]>,
    pub rest: Option<Box<Expr>>,
}
pub struct Try {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub question: Token![?],
}
pub struct TryBlock {
    pub attrs: Vec<attr::Attr>,
    pub try_: Token![try],
    pub block: stmt::Block,
}
pub struct Tuple {
    pub attrs: Vec<attr::Attr>,
    pub paren: tok::Paren,
    pub elems: Punctuated<Expr, Token![,]>,
}
pub struct Unary {
    pub attrs: Vec<attr::Attr>,
    pub op: UnOp,
    pub expr: Box<Expr>,
}
pub struct Unsafe {
    pub attrs: Vec<attr::Attr>,
    pub unsafe_: Token![unsafe],
    pub block: stmt::Block,
}
pub struct While {
    pub attrs: Vec<attr::Attr>,
    pub label: Option<Label>,
    pub while_: Token![while],
    pub cond: Box<Expr>,
    pub block: stmt::Block,
}
pub struct Yield {
    pub attrs: Vec<attr::Attr>,
    pub yield_: Token![yield],
    pub expr: Option<Box<Expr>>,
}

impl Expr {
    pub const DUMMY: Self = Expr::Path(Path {
        attrs: Vec::new(),
        qself: None,
        path: path::Path {
            colon: None,
            segs: Punctuated::new(),
        },
    });
    pub fn replace_attrs(&mut self, y: Vec<attr::Attr>) -> Vec<attr::Attr> {
        use Expr::*;
        match self {
            Array(Array { attrs, .. })
            | Assign(Assign { attrs, .. })
            | Async(Async { attrs, .. })
            | Await(Await { attrs, .. })
            | Binary(Binary { attrs, .. })
            | Block(Block { attrs, .. })
            | Break(Break { attrs, .. })
            | Call(Call { attrs, .. })
            | Cast(Cast { attrs, .. })
            | Closure(Closure { attrs, .. })
            | Const(Const { attrs, .. })
            | Continue(Continue { attrs, .. })
            | Field(Field { attrs, .. })
            | ForLoop(ForLoop { attrs, .. })
            | Group(Group { attrs, .. })
            | If(If { attrs, .. })
            | Index(Index { attrs, .. })
            | Infer(Infer { attrs, .. })
            | Let(Let { attrs, .. })
            | Lit(Lit { attrs, .. })
            | Loop(Loop { attrs, .. })
            | Mac(Mac { attrs, .. })
            | Match(Match { attrs, .. })
            | MethodCall(MethodCall { attrs, .. })
            | Paren(Paren { attrs, .. })
            | Path(Path { attrs, .. })
            | Range(Range { attrs, .. })
            | Ref(Ref { attrs, .. })
            | Repeat(Repeat { attrs, .. })
            | Return(Return { attrs, .. })
            | Struct(Struct { attrs, .. })
            | Try(Try { attrs, .. })
            | TryBlock(TryBlock { attrs, .. })
            | Tuple(Tuple { attrs, .. })
            | Unary(Unary { attrs, .. })
            | Unsafe(Unsafe { attrs, .. })
            | While(While { attrs, .. })
            | Yield(Yield { attrs, .. }) => mem::replace(attrs, y),
            Verbatim(_) => Vec::new(),
        }
    }
}

pub enum Member {
    Named(Ident),
    Unnamed(Idx),
}
impl From<Ident> for Member {
    fn from(x: Ident) -> Member {
        Member::Named(x)
    }
}
impl From<Idx> for Member {
    fn from(x: Idx) -> Member {
        Member::Unnamed(x)
    }
}
impl From<usize> for Member {
    fn from(x: usize) -> Member {
        Member::Unnamed(Idx::from(x))
    }
}
impl Eq for Member {}
impl PartialEq for Member {
    fn eq(&self, y: &Self) -> bool {
        use Member::*;
        match (self, y) {
            (Named(x), Named(y)) => x == y,
            (Unnamed(x), Unnamed(y)) => x == y,
            _ => false,
        }
    }
}
impl Hash for Member {
    fn hash<H: Hasher>(&self, y: &mut H) {
        use Member::*;
        match self {
            Named(x) => x.hash(y),
            Unnamed(x) => x.hash(y),
        }
    }
}
impl IdentFragment for Member {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Member::*;
        match self {
            Named(x) => Display::fmt(x, f),
            Unnamed(x) => Display::fmt(&x.idx, f),
        }
    }
    fn span(&self) -> Option<pm2::Span> {
        use Member::*;
        match self {
            Named(x) => Some(x.span()),
            Unnamed(x) => Some(x.span),
        }
    }
}

pub struct Idx {
    pub idx: u32,
    pub span: pm2::Span,
}
impl From<usize> for Idx {
    fn from(x: usize) -> Idx {
        assert!(x < u32::max_value() as usize);
        Idx {
            idx: x as u32,
            span: pm2::Span::call_site(),
        }
    }
}
impl Eq for Idx {}
impl PartialEq for Idx {
    fn eq(&self, x: &Self) -> bool {
        self.idx == x.idx
    }
}
impl Hash for Idx {
    fn hash<H: Hasher>(&self, y: &mut H) {
        self.idx.hash(y);
    }
}
impl IdentFragment for Idx {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.idx, f)
    }
    fn span(&self) -> Option<pm2::Span> {
        Some(self.span)
    }
}

pub enum BinOp {
    Add(Token![+]),
    Sub(Token![-]),
    Mul(Token![*]),
    Div(Token![/]),
    Rem(Token![%]),
    And(Token![&&]),
    Or(Token![||]),
    BitXor(Token![^]),
    BitAnd(Token![&]),
    BitOr(Token![|]),
    Shl(Token![<<]),
    Shr(Token![>>]),
    Eq(Token![==]),
    Lt(Token![<]),
    Le(Token![<=]),
    Ne(Token![!=]),
    Ge(Token![>=]),
    Gt(Token![>]),
    AddAssign(Token![+=]),
    SubAssign(Token![-=]),
    MulAssign(Token![*=]),
    DivAssign(Token![/=]),
    RemAssign(Token![%=]),
    BitXorAssign(Token![^=]),
    BitAndAssign(Token![&=]),
    BitOrAssign(Token![|=]),
    ShlAssign(Token![<<=]),
    ShrAssign(Token![>>=]),
}
pub enum UnOp {
    Deref(Token![*]),
    Not(Token![!]),
    Neg(Token![-]),
}

pub struct FieldValue {
    pub attrs: Vec<attr::Attr>,
    pub memb: Member,
    pub colon: Option<Token![:]>,
    pub expr: Expr,
}
pub struct Label {
    pub name: Lifetime,
    pub colon: Token![:],
}
pub struct Arm {
    pub attrs: Vec<attr::Attr>,
    pub pat: pat::Pat,
    pub guard: Option<(Token![if], Box<Expr>)>,
    pub fat_arrow: Token![=>],
    pub body: Box<Expr>,
    pub comma: Option<Token![,]>,
}

pub enum RangeLimits {
    HalfOpen(Token![..]),
    Closed(Token![..=]),
}

pub fn requires_terminator(x: &Expr) -> bool {
    match x {
        Expr::If(_)
        | Expr::Match(_)
        | Expr::Block(_) | Expr::Unsafe(_) // both under ExprKind::Block in rustc
        | Expr::While(_)
        | Expr::Loop(_)
        | Expr::ForLoop(_)
        | Expr::TryBlock(_)
        | Expr::Const(_) => false,
        Expr::Array(_)
        | Expr::Assign(_)
        | Expr::Async(_)
        | Expr::Await(_)
        | Expr::Binary(_)
        | Expr::Break(_)
        | Expr::Call(_)
        | Expr::Cast(_)
        | Expr::Closure(_)
        | Expr::Continue(_)
        | Expr::Field(_)
        | Expr::Group(_)
        | Expr::Index(_)
        | Expr::Infer(_)
        | Expr::Let(_)
        | Expr::Lit(_)
        | Expr::Mac(_)
        | Expr::MethodCall(_)
        | Expr::Paren(_)
        | Expr::Path(_)
        | Expr::Range(_)
        | Expr::Ref(_)
        | Expr::Repeat(_)
        | Expr::Return(_)
        | Expr::Struct(_)
        | Expr::Try(_)
        | Expr::Tuple(_)
        | Expr::Unary(_)
        | Expr::Yield(_)
        | Expr::Verbatim(_) => true
    }
}

fn parse_expr(x: Stream, mut lhs: Expr, allow_struct: AllowStruct, base: Precedence) -> Res<Expr> {
    loop {
        let ahead = x.fork();
        if let Some(op) = match ahead.parse::<BinOp>() {
            Ok(op) if Precedence::of(&op) >= base => Some(op),
            _ => None,
        } {
            x.advance_to(&ahead);
            let precedence = Precedence::of(&op);
            let mut rhs = unary_expr(x, allow_struct)?;
            loop {
                let next = peek_precedence(x);
                if next > precedence || next == precedence && precedence == Precedence::Assign {
                    rhs = parse_expr(x, rhs, allow_struct, next)?;
                } else {
                    break;
                }
            }
            lhs = Expr::Binary(expr::Binary {
                attrs: Vec::new(),
                left: Box::new(lhs),
                op,
                right: Box::new(rhs),
            });
        } else if Precedence::Assign >= base && x.peek(Token![=]) && !x.peek(Token![==]) && !x.peek(Token![=>]) {
            let eq: Token![=] = x.parse()?;
            let mut rhs = unary_expr(x, allow_struct)?;
            loop {
                let next = peek_precedence(x);
                if next >= Precedence::Assign {
                    rhs = parse_expr(x, rhs, allow_struct, next)?;
                } else {
                    break;
                }
            }
            lhs = Expr::Assign(expr::Assign {
                attrs: Vec::new(),
                left: Box::new(lhs),
                eq,
                right: Box::new(rhs),
            });
        } else if Precedence::Range >= base && x.peek(Token![..]) {
            let limits: RangeLimits = x.parse()?;
            let rhs = if matches!(limits, RangeLimits::HalfOpen(_))
                && (x.is_empty()
                    || x.peek(Token![,])
                    || x.peek(Token![;])
                    || x.peek(Token![.]) && !x.peek(Token![..])
                    || !allow_struct.0 && x.peek(tok::Brace))
            {
                None
            } else {
                let mut rhs = unary_expr(x, allow_struct)?;
                loop {
                    let next = peek_precedence(x);
                    if next > Precedence::Range {
                        rhs = parse_expr(x, rhs, allow_struct, next)?;
                    } else {
                        break;
                    }
                }
                Some(rhs)
            };
            lhs = Expr::Range(expr::Range {
                attrs: Vec::new(),
                beg: Some(Box::new(lhs)),
                limits,
                end: rhs.map(Box::new),
            });
        } else if Precedence::Cast >= base && x.peek(Token![as]) {
            let as_: Token![as] = x.parse()?;
            let allow_plus = false;
            let group_gen = false;
            let ty = ambig_ty(x, allow_plus, group_gen)?;
            check_cast(x)?;
            lhs = Expr::Cast(expr::Cast {
                attrs: Vec::new(),
                expr: Box::new(lhs),
                as_,
                typ: Box::new(ty),
            });
        } else {
            break;
        }
    }
    Ok(lhs)
}
fn peek_precedence(x: Stream) -> Precedence {
    if let Ok(op) = x.fork().parse() {
        Precedence::of(&op)
    } else if x.peek(Token![=]) && !x.peek(Token![=>]) {
        Precedence::Assign
    } else if x.peek(Token![..]) {
        Precedence::Range
    } else if x.peek(Token![as]) {
        Precedence::Cast
    } else {
        Precedence::Any
    }
}
fn ambiguous_expr(x: Stream, allow_struct: AllowStruct) -> Res<Expr> {
    let lhs = unary_expr(x, allow_struct)?;
    parse_expr(x, lhs, allow_struct, Precedence::Any)
}
fn expr_attrs(x: Stream) -> Res<Vec<attr::Attr>> {
    let mut ys = Vec::new();
    loop {
        if x.peek(tok::Group) {
            let ahead = x.fork();
            let group = super::parse_group(&ahead)?;
            if !group.buf.peek(Token![#]) || group.buf.peek2(Token![!]) {
                break;
            }
            let y = group.buf.call(attr::single_outer)?;
            if !group.buf.is_empty() {
                break;
            }
            ys.push(y);
        } else if x.peek(Token![#]) {
            ys.push(x.call(attr::single_outer)?);
        } else {
            break;
        }
    }
    Ok(ys)
}
fn unary_expr(x: Stream, allow_struct: AllowStruct) -> Res<Expr> {
    let begin = x.fork();
    let attrs = x.call(expr_attrs)?;
    if x.peek(Token![&]) {
        let and: Token![&] = x.parse()?;
        let raw: Option<kw::raw> = if x.peek(kw::raw) && (x.peek2(Token![mut]) || x.peek2(Token![const])) {
            Some(x.parse()?)
        } else {
            None
        };
        let mutability: Option<Token![mut]> = x.parse()?;
        if raw.is_some() && mutability.is_none() {
            x.parse::<Token![const]>()?;
        }
        let expr = Box::new(unary_expr(x, allow_struct)?);
        if raw.is_some() {
            Ok(Expr::Verbatim(verbatim_between(&begin, x)))
        } else {
            Ok(Expr::Reference(expr::Ref {
                attrs,
                and,
                mut_: mutability,
                expr,
            }))
        }
    } else if x.peek(Token![*]) || x.peek(Token![!]) || x.peek(Token![-]) {
        expr_unary(x, attrs, allow_struct).map(Expr::Unary)
    } else {
        trailer_expr(begin, attrs, x, allow_struct)
    }
}
fn trailer_expr(begin: Buffer, mut attrs: Vec<attr::Attr>, x: Stream, allow_struct: AllowStruct) -> Res<Expr> {
    let atom = atom_expr(x, allow_struct)?;
    let mut e = trailer_helper(x, atom)?;
    if let Expr::Verbatim(tokens) = &mut e {
        *tokens = verbatim_between(&begin, x);
    } else {
        let inner_attrs = e.replace_attrs(Vec::new());
        attrs.extend(inner_attrs);
        e.replace_attrs(attrs);
    }
    Ok(e)
}
fn trailer_helper(x: Stream, mut e: Expr) -> Res<Expr> {
    loop {
        if x.peek(tok::Paren) {
            let content;
            e = Expr::Call(expr::Call {
                attrs: Vec::new(),
                func: Box::new(e),
                paren: parenthesized!(content in x),
                args: content.parse_terminated(Expr::parse, Token![,])?,
            });
        } else if x.peek(Token![.])
            && !x.peek(Token![..])
            && match e {
                Expr::Range(_) => false,
                _ => true,
            }
        {
            let mut dot: Token![.] = x.parse()?;
            let float_token: Option<lit::Float> = x.parse()?;
            if let Some(float_token) = float_token {
                if multi_index(&mut e, &mut dot, float_token)? {
                    continue;
                }
            }
            let await_: Option<Token![await]> = x.parse()?;
            if let Some(await_) = await_ {
                e = Expr::Await(expr::Await {
                    attrs: Vec::new(),
                    expr: Box::new(e),
                    dot,
                    await_,
                });
                continue;
            }
            let member: Member = x.parse()?;
            let turbofish = if member.is_named() && x.peek(Token![::]) {
                Some(AngledArgs::parse_turbofish(x)?)
            } else {
                None
            };
            if turbofish.is_some() || x.peek(tok::Paren) {
                if let Member::Named(method) = member {
                    let content;
                    e = Expr::MethodCall(expr::MethodCall {
                        attrs: Vec::new(),
                        expr: Box::new(e),
                        dot,
                        method,
                        turbofish,
                        paren: parenthesized!(content in x),
                        args: content.parse_terminated(Expr::parse, Token![,])?,
                    });
                    continue;
                }
            }
            e = Expr::Field(expr::Field {
                attrs: Vec::new(),
                base: Box::new(e),
                dot,
                memb: member,
            });
        } else if x.peek(tok::Bracket) {
            let content;
            e = Expr::Index(expr::Index {
                attrs: Vec::new(),
                expr: Box::new(e),
                bracket: bracketed!(content in x),
                index: content.parse()?,
            });
        } else if x.peek(Token![?]) {
            e = Expr::Try(expr::Try {
                attrs: Vec::new(),
                expr: Box::new(e),
                question: x.parse()?,
            });
        } else {
            break;
        }
    }
    Ok(e)
}
fn atom_expr(x: Stream, allow_struct: AllowStruct) -> Res<Expr> {
    if x.peek(tok::Group) && !x.peek2(Token![::]) && !x.peek2(Token![!]) && !x.peek2(tok::Brace) {
        x.call(expr_group).map(Expr::Group)
    } else if x.peek(Lit) {
        x.parse().map(Expr::Lit)
    } else if x.peek(Token![async]) && (x.peek2(tok::Brace) || x.peek2(Token![move]) && x.peek3(tok::Brace)) {
        x.parse().map(Expr::Async)
    } else if x.peek(Token![try]) && x.peek2(tok::Brace) {
        x.parse().map(Expr::TryBlock)
    } else if x.peek(Token![|])
        || x.peek(Token![move])
        || x.peek(Token![for]) && x.peek2(Token![<]) && (x.peek3(Lifetime) || x.peek3(Token![>]))
        || x.peek(Token![const]) && !x.peek2(tok::Brace)
        || x.peek(Token![static])
        || x.peek(Token![async]) && (x.peek2(Token![|]) || x.peek2(Token![move]))
    {
        expr_closure(x, allow_struct).map(Expr::Closure)
    } else if x.peek(kw::builtin) && x.peek2(Token![#]) {
        expr_builtin(x)
    } else if x.peek(Ident)
        || x.peek(Token![::])
        || x.peek(Token![<])
        || x.peek(Token![self])
        || x.peek(Token![Self])
        || x.peek(Token![super])
        || x.peek(Token![crate])
        || x.peek(Token![try]) && (x.peek2(Token![!]) || x.peek2(Token![::]))
    {
        path_or_macro_or_struct(x, allow_struct)
    } else if x.peek(tok::Paren) {
        paren_or_tuple(x)
    } else if x.peek(Token![break]) {
        expr_break(x, allow_struct).map(Expr::Break)
    } else if x.peek(Token![continue]) {
        x.parse().map(Expr::Continue)
    } else if x.peek(Token![return]) {
        expr_ret(x, allow_struct).map(Expr::Return)
    } else if x.peek(tok::Bracket) {
        array_or_repeat(x)
    } else if x.peek(Token![let]) {
        x.parse().map(Expr::Let)
    } else if x.peek(Token![if]) {
        x.parse().map(Expr::If)
    } else if x.peek(Token![while]) {
        x.parse().map(Expr::While)
    } else if x.peek(Token![for]) {
        x.parse().map(Expr::ForLoop)
    } else if x.peek(Token![loop]) {
        x.parse().map(Expr::Loop)
    } else if x.peek(Token![match]) {
        x.parse().map(Expr::Match)
    } else if x.peek(Token![yield]) {
        x.parse().map(Expr::Yield)
    } else if x.peek(Token![unsafe]) {
        x.parse().map(Expr::Unsafe)
    } else if x.peek(Token![const]) {
        x.parse().map(Expr::Const)
    } else if x.peek(tok::Brace) {
        x.parse().map(Expr::Block)
    } else if x.peek(Token![..]) {
        expr_range(x, allow_struct).map(Expr::Range)
    } else if x.peek(Token![_]) {
        x.parse().map(Expr::Infer)
    } else if x.peek(Lifetime) {
        let the_label: Label = x.parse()?;
        let mut expr = if x.peek(Token![while]) {
            Expr::While(x.parse()?)
        } else if x.peek(Token![for]) {
            Expr::ForLoop(x.parse()?)
        } else if x.peek(Token![loop]) {
            Expr::Loop(x.parse()?)
        } else if x.peek(tok::Brace) {
            Expr::Block(x.parse()?)
        } else {
            return Err(x.error("expected loop or block expression"));
        };
        match &mut expr {
            Expr::While(expr::While { label, .. })
            | Expr::ForLoop(expr::ForLoop { label, .. })
            | Expr::Loop(expr::Loop { label, .. })
            | Expr::Block(expr::Block { label, .. }) => *label = Some(the_label),
            _ => unreachable!(),
        }
        Ok(expr)
    } else {
        Err(x.error("expected expression"))
    }
}
fn expr_builtin(x: Stream) -> Res<Expr> {
    let begin = x.fork();
    x.parse::<kw::builtin>()?;
    x.parse::<Token![#]>()?;
    x.parse::<Ident>()?;
    let args;
    parenthesized!(args in x);
    args.parse::<pm2::Stream>()?;
    Ok(Expr::Verbatim(verbatim_between(&begin, x)))
}
fn path_or_macro_or_struct(x: Stream, #[cfg(feature = "full")] allow_struct: AllowStruct) -> Res<Expr> {
    let (qself, path) = qpath(x, true)?;
    if qself.is_none() && x.peek(Token![!]) && !x.peek(Token![!=]) && path.is_mod_style() {
        let bang: Token![!] = x.parse()?;
        let (delimiter, tokens) = mac::parse_delim(x)?;
        return Ok(Expr::Macro(expr::Mac {
            attrs: Vec::new(),
            mac: Macro {
                path,
                bang,
                delim: delimiter,
                toks: tokens,
            },
        }));
    }
    if allow_struct.0 && x.peek(tok::Brace) {
        return expr_struct_helper(x, qself, path).map(Expr::Struct);
    }
    Ok(Expr::Path(expr::Path {
        attrs: Vec::new(),
        qself,
        path,
    }))
}
impl Parse for expr::Mac {
    fn parse(x: Stream) -> Res<Self> {
        Ok(expr::Mac {
            attrs: Vec::new(),
            mac: x.parse()?,
        })
    }
}
fn paren_or_tuple(x: Stream) -> Res<Expr> {
    let content;
    let paren = parenthesized!(content in x);
    if content.is_empty() {
        return Ok(Expr::Tuple(expr::Tuple {
            attrs: Vec::new(),
            paren,
            elems: Punctuated::new(),
        }));
    }
    let first: Expr = content.parse()?;
    if content.is_empty() {
        return Ok(Expr::Paren(expr::Paren {
            attrs: Vec::new(),
            paren,
            expr: Box::new(first),
        }));
    }
    let mut elems = Punctuated::new();
    elems.push_value(first);
    while !content.is_empty() {
        let punct = content.parse()?;
        elems.push_punct(punct);
        if content.is_empty() {
            break;
        }
        let value = content.parse()?;
        elems.push_value(value);
    }
    Ok(Expr::Tuple(expr::Tuple {
        attrs: Vec::new(),
        paren,
        elems,
    }))
}
fn array_or_repeat(x: Stream) -> Res<Expr> {
    let content;
    let bracket = bracketed!(content in x);
    if content.is_empty() {
        return Ok(Expr::Array(expr::Array {
            attrs: Vec::new(),
            bracket,
            elems: Punctuated::new(),
        }));
    }
    let first: Expr = content.parse()?;
    if content.is_empty() || content.peek(Token![,]) {
        let mut elems = Punctuated::new();
        elems.push_value(first);
        while !content.is_empty() {
            let punct = content.parse()?;
            elems.push_punct(punct);
            if content.is_empty() {
                break;
            }
            let value = content.parse()?;
            elems.push_value(value);
        }
        Ok(Expr::Array(expr::Array {
            attrs: Vec::new(),
            bracket,
            elems,
        }))
    } else if content.peek(Token![;]) {
        let semi: Token![;] = content.parse()?;
        let len: Expr = content.parse()?;
        Ok(Expr::Repeat(expr::Repeat {
            attrs: Vec::new(),
            bracket,
            expr: Box::new(first),
            semi,
            len: Box::new(len),
        }))
    } else {
        Err(content.error("expected `,` or `;`"))
    }
}
impl Parse for expr::Array {
    fn parse(x: Stream) -> Res<Self> {
        let content;
        let bracket = bracketed!(content in x);
        let mut elems = Punctuated::new();
        while !content.is_empty() {
            let first: Expr = content.parse()?;
            elems.push_value(first);
            if content.is_empty() {
                break;
            }
            let punct = content.parse()?;
            elems.push_punct(punct);
        }
        Ok(expr::Array {
            attrs: Vec::new(),
            bracket,
            elems,
        })
    }
}
impl Parse for expr::Repeat {
    fn parse(x: Stream) -> Res<Self> {
        let content;
        Ok(expr::Repeat {
            bracket: bracketed!(content in x),
            attrs: Vec::new(),
            expr: content.parse()?,
            semi: content.parse()?,
            len: content.parse()?,
        })
    }
}
pub fn expr_early(x: Stream) -> Res<Expr> {
    let mut attrs = x.call(expr_attrs)?;
    let mut expr = if x.peek(Token![if]) {
        Expr::If(x.parse()?)
    } else if x.peek(Token![while]) {
        Expr::While(x.parse()?)
    } else if x.peek(Token![for]) && !(x.peek2(Token![<]) && (x.peek3(Lifetime) || x.peek3(Token![>]))) {
        Expr::ForLoop(x.parse()?)
    } else if x.peek(Token![loop]) {
        Expr::Loop(x.parse()?)
    } else if x.peek(Token![match]) {
        Expr::Match(x.parse()?)
    } else if x.peek(Token![try]) && x.peek2(tok::Brace) {
        Expr::TryBlock(x.parse()?)
    } else if x.peek(Token![unsafe]) {
        Expr::Unsafe(x.parse()?)
    } else if x.peek(Token![const]) && x.peek2(tok::Brace) {
        Expr::Const(x.parse()?)
    } else if x.peek(tok::Brace) {
        Expr::Block(x.parse()?)
    } else {
        let allow_struct = AllowStruct(true);
        let mut expr = unary_expr(x, allow_struct)?;
        attrs.extend(expr.replace_attrs(Vec::new()));
        expr.replace_attrs(attrs);
        return parse_expr(x, expr, allow_struct, Precedence::Any);
    };
    if x.peek(Token![.]) && !x.peek(Token![..]) || x.peek(Token![?]) {
        expr = trailer_helper(x, expr)?;
        attrs.extend(expr.replace_attrs(Vec::new()));
        expr.replace_attrs(attrs);
        let allow_struct = AllowStruct(true);
        return parse_expr(x, expr, allow_struct, Precedence::Any);
    }
    attrs.extend(expr.replace_attrs(Vec::new()));
    expr.replace_attrs(attrs);
    Ok(expr)
}
impl Parse for expr::Lit {
    fn parse(x: Stream) -> Res<Self> {
        Ok(expr::Lit {
            attrs: Vec::new(),
            lit: x.parse()?,
        })
    }
}
fn expr_group(x: Stream) -> Res<expr::Group> {
    let group = super::parse_group(x)?;
    Ok(expr::Group {
        attrs: Vec::new(),
        group: group.token,
        expr: group.buf.parse()?,
    })
}
impl Parse for expr::Paren {
    fn parse(x: Stream) -> Res<Self> {
        expr_paren(x)
    }
}
fn expr_paren(x: Stream) -> Res<expr::Paren> {
    let content;
    Ok(expr::Paren {
        attrs: Vec::new(),
        paren: parenthesized!(content in x),
        expr: content.parse()?,
    })
}
impl Parse for expr::Let {
    fn parse(x: Stream) -> Res<Self> {
        Ok(expr::Let {
            attrs: Vec::new(),
            let_: x.parse()?,
            pat: Box::new(pat::Pat::parse_multi(x)?),
            eq: x.parse()?,
            expr: Box::new({
                let allow_struct = AllowStruct(false);
                let lhs = unary_expr(x, allow_struct)?;
                parse_expr(x, lhs, allow_struct, Precedence::Compare)?
            }),
        })
    }
}
impl Parse for expr::If {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = x.call(attr::Attr::parse_outer)?;
        Ok(expr::If {
            attrs,
            if_: x.parse()?,
            cond: Box::new(x.call(Expr::parse_without_eager_brace)?),
            then_branch: x.parse()?,
            else_branch: {
                if x.peek(Token![else]) {
                    Some(x.call(else_block)?)
                } else {
                    None
                }
            },
        })
    }
}
fn else_block(x: Stream) -> Res<(Token![else], Box<Expr>)> {
    let else_token: Token![else] = x.parse()?;
    let lookahead = x.lookahead1();
    let else_branch = if x.peek(Token![if]) {
        x.parse().map(Expr::If)?
    } else if x.peek(tok::Brace) {
        Expr::Block(expr::Block {
            attrs: Vec::new(),
            label: None,
            block: x.parse()?,
        })
    } else {
        return Err(lookahead.error());
    };
    Ok((else_token, Box::new(else_branch)))
}
impl Parse for expr::Infer {
    fn parse(x: Stream) -> Res<Self> {
        Ok(expr::Infer {
            attrs: x.call(attr::Attr::parse_outer)?,
            underscore: x.parse()?,
        })
    }
}
impl Parse for expr::ForLoop {
    fn parse(x: Stream) -> Res<Self> {
        let mut attrs = x.call(attr::Attr::parse_outer)?;
        let label: Option<Label> = x.parse()?;
        let for_: Token![for] = x.parse()?;
        let pat = pat::Pat::parse_multi(x)?;
        let in_: Token![in] = x.parse()?;
        let expr: Expr = x.call(Expr::parse_without_eager_brace)?;
        let content;
        let brace = braced!(content in x);
        attr::inner(&content, &mut attrs)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(expr::ForLoop {
            attrs,
            label,
            for_,
            pat: Box::new(pat),
            in_,
            expr: Box::new(expr),
            body: Block { brace, stmts },
        })
    }
}
impl Parse for expr::Loop {
    fn parse(x: Stream) -> Res<Self> {
        let mut attrs = x.call(attr::Attr::parse_outer)?;
        let label: Option<Label> = x.parse()?;
        let loop_: Token![loop] = x.parse()?;
        let content;
        let brace = braced!(content in x);
        attr::inner(&content, &mut attrs)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(expr::Loop {
            attrs,
            label,
            loop_,
            body: Block { brace, stmts },
        })
    }
}
impl Parse for expr::Match {
    fn parse(x: Stream) -> Res<Self> {
        let mut attrs = x.call(attr::Attr::parse_outer)?;
        let match_: Token![match] = x.parse()?;
        let expr = Expr::parse_without_eager_brace(x)?;
        let content;
        let brace = braced!(content in x);
        attr::inner(&content, &mut attrs)?;
        let mut arms = Vec::new();
        while !content.is_empty() {
            arms.push(content.call(Arm::parse)?);
        }
        Ok(expr::Match {
            attrs,
            match_,
            expr: Box::new(expr),
            brace,
            arms,
        })
    }
}
macro_rules! impl_by_parsing_expr {
        (
            $(
                $expr_type:ty, $variant:ident, $msg:expr,
            )*
        ) => {
            $(
                impl Parse for $expr_type {
                    fn parse(input: Stream) -> Res<Self> {
                        let mut expr: Expr = input.parse()?;
                        loop {
                            match expr {
                                Expr::$variant(inner) => return Ok(inner),
                                Expr::Group(next) => expr = *next.expr,
                                _ => return Err(Error::new_spanned(expr, $msg)),
                            }
                        }
                    }
                }
            )*
        };
    }
impl_by_parsing_expr! {
    expr::Assign, Assign, "expected assignment expression",
    expr::Await, Await, "expected await expression",
    expr::Binary, Binary, "expected binary operation",
    expr::Call, Call, "expected function call expression",
    expr::Cast, Cast, "expected cast expression",
    expr::Field, Field, "expected struct field access",
    expr::Index, Index, "expected indexing expression",
    expr::MethodCall, MethodCall, "expected method call expression",
    expr::Range, Range, "expected range expression",
    expr::Try, Try, "expected try expression",
    expr::Tuple, Tuple, "expected tuple expression",
}
impl Parse for expr::Unary {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = Vec::new();
        let allow_struct = AllowStruct(true);
        expr_unary(x, attrs, allow_struct)
    }
}
fn expr_unary(x: Stream, attrs: Vec<attr::Attr>, allow_struct: AllowStruct) -> Res<expr::Unary> {
    Ok(expr::Unary {
        attrs,
        op: x.parse()?,
        expr: Box::new(unary_expr(x, allow_struct)?),
    })
}
impl Parse for expr::Closure {
    fn parse(x: Stream) -> Res<Self> {
        let allow_struct = AllowStruct(true);
        expr_closure(x, allow_struct)
    }
}
impl Parse for expr::Ref {
    fn parse(x: Stream) -> Res<Self> {
        let allow_struct = AllowStruct(true);
        Ok(expr::Ref {
            attrs: Vec::new(),
            and: x.parse()?,
            mut_: x.parse()?,
            expr: Box::new(unary_expr(x, allow_struct)?),
        })
    }
}
impl Parse for expr::Break {
    fn parse(input: Stream) -> Res<Self> {
        let allow_struct = AllowStruct(true);
        expr_break(input, allow_struct)
    }
}
impl Parse for expr::Return {
    fn parse(input: Stream) -> Res<Self> {
        let allow_struct = AllowStruct(true);
        expr_ret(input, allow_struct)
    }
}
impl Parse for expr::TryBlock {
    fn parse(input: Stream) -> Res<Self> {
        Ok(expr::TryBlock {
            attrs: Vec::new(),
            try_: input.parse()?,
            block: input.parse()?,
        })
    }
}
impl Parse for expr::Yield {
    fn parse(input: Stream) -> Res<Self> {
        Ok(expr::Yield {
            attrs: Vec::new(),
            yield_: input.parse()?,
            expr: {
                if !input.is_empty() && !input.peek(Token![,]) && !input.peek(Token![;]) {
                    Some(input.parse()?)
                } else {
                    None
                }
            },
        })
    }
}
fn expr_closure(input: Stream, allow_struct: AllowStruct) -> Res<expr::Closure> {
    let lifetimes: Option<Bgen::bound::Lifes> = input.parse()?;
    let constness: Option<Token![const]> = input.parse()?;
    let movability: Option<Token![static]> = input.parse()?;
    let asyncness: Option<Token![async]> = input.parse()?;
    let capture: Option<Token![move]> = input.parse()?;
    let or1: Token![|] = input.parse()?;
    let mut inputs = Punctuated::new();
    loop {
        if input.peek(Token![|]) {
            break;
        }
        let value = closure_arg(input)?;
        inputs.push_value(value);
        if input.peek(Token![|]) {
            break;
        }
        let punct: Token![,] = input.parse()?;
        inputs.push_punct(punct);
    }
    let or2: Token![|] = input.parse()?;
    let (output, body) = if input.peek(Token![->]) {
        let arrow: Token![->] = input.parse()?;
        let typ: typ::Type = input.parse()?;
        let body: Block = input.parse()?;
        let output = typ::Ret::Type(arrow, Box::new(typ));
        let block = Expr::Block(expr::Block {
            attrs: Vec::new(),
            label: None,
            block: body,
        });
        (output, block)
    } else {
        let body = ambiguous_expr(input, allow_struct)?;
        (typ::Ret::Default, body)
    };
    Ok(expr::Closure {
        attrs: Vec::new(),
        lifes: lifetimes,
        const_: constness,
        static_: movability,
        async_: asyncness,
        move_: capture,
        or1,
        inputs,
        or2,
        ret: output,
        body: Box::new(body),
    })
}
impl Parse for expr::Async {
    fn parse(input: Stream) -> Res<Self> {
        Ok(expr::Async {
            attrs: Vec::new(),
            async_: input.parse()?,
            move_: input.parse()?,
            block: input.parse()?,
        })
    }
}
fn closure_arg(input: Stream) -> Res<pat::Pat> {
    let attrs = input.call(attr::Attr::parse_outer)?;
    let mut pat = pat::Pat::parse_single(input)?;
    if input.peek(Token![:]) {
        Ok(pat::Pat::Type(pat::Type {
            attrs,
            pat: Box::new(pat),
            colon: input.parse()?,
            ty: input.parse()?,
        }))
    } else {
        match &mut pat {
            pat::Pat::Const(pat) => pat.attrs = attrs,
            pat::Pat::Ident(pat) => pat.attrs = attrs,
            pat::Pat::Lit(pat) => pat.attrs = attrs,
            pat::Pat::Macro(pat) => pat.attrs = attrs,
            pat::Pat::Or(pat) => pat.attrs = attrs,
            pat::Pat::Paren(pat) => pat.attrs = attrs,
            pat::Pat::Path(pat) => pat.attrs = attrs,
            pat::Pat::Range(pat) => pat.attrs = attrs,
            pat::Pat::Reference(pat) => pat.attrs = attrs,
            pat::Pat::Rest(pat) => pat.attrs = attrs,
            pat::Pat::Slice(pat) => pat.attrs = attrs,
            pat::Pat::Struct(pat) => pat.attrs = attrs,
            pat::Pat::Tuple(pat) => pat.attrs = attrs,
            pat::Pat::TupleStruct(pat) => pat.attrs = attrs,
            pat::Pat::Type(_) => unreachable!(),
            pat::Pat::Verbatim(_) => {},
            pat::Pat::Wild(pat) => pat.attrs = attrs,
        }
        Ok(pat)
    }
}
impl Parse for expr::While {
    fn parse(input: Stream) -> Res<Self> {
        let mut attrs = input.call(attr::Attr::parse_outer)?;
        let label: Option<Label> = input.parse()?;
        let while_: Token![while] = input.parse()?;
        let cond = Expr::parse_without_eager_brace(input)?;
        let content;
        let brace = braced!(content in input);
        attr::inner(&content, &mut attrs)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(expr::While {
            attrs,
            label,
            while_,
            cond: Box::new(cond),
            body: Block { brace, stmts },
        })
    }
}
impl Parse for expr::Const {
    fn parse(input: Stream) -> Res<Self> {
        let const_: Token![const] = input.parse()?;
        let content;
        let brace = braced!(content in input);
        let inner_attrs = content.call(attr::Attr::parse_inner)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(expr::Const {
            attrs: inner_attrs,
            const_,
            block: Block { brace, stmts },
        })
    }
}
impl Parse for Label {
    fn parse(input: Stream) -> Res<Self> {
        Ok(Label {
            name: input.parse()?,
            colon: input.parse()?,
        })
    }
}
impl Parse for Option<Label> {
    fn parse(input: Stream) -> Res<Self> {
        if input.peek(Lifetime) {
            input.parse().map(Some)
        } else {
            Ok(None)
        }
    }
}
impl Parse for expr::Continue {
    fn parse(input: Stream) -> Res<Self> {
        Ok(expr::Continue {
            attrs: Vec::new(),
            continue_: input.parse()?,
            label: input.parse()?,
        })
    }
}
fn expr_break(input: Stream, allow_struct: AllowStruct) -> Res<expr::Break> {
    Ok(expr::Break {
        attrs: Vec::new(),
        break_: input.parse()?,
        label: input.parse()?,
        expr: {
            if input.is_empty()
                || input.peek(Token![,])
                || input.peek(Token![;])
                || !allow_struct.0 && input.peek(tok::Brace)
            {
                None
            } else {
                let expr = ambiguous_expr(input, allow_struct)?;
                Some(Box::new(expr))
            }
        },
    })
}
fn expr_ret(input: Stream, allow_struct: AllowStruct) -> Res<expr::Return> {
    Ok(expr::Return {
        attrs: Vec::new(),
        return_: input.parse()?,
        expr: {
            if input.is_empty() || input.peek(Token![,]) || input.peek(Token![;]) {
                None
            } else {
                let expr = ambiguous_expr(input, allow_struct)?;
                Some(Box::new(expr))
            }
        },
    })
}
impl Parse for FieldValue {
    fn parse(input: Stream) -> Res<Self> {
        let attrs = input.call(attr::Attr::parse_outer)?;
        let member: Member = input.parse()?;
        let (colon, value) = if input.peek(Token![:]) || !member.is_named() {
            let colon: Token![:] = input.parse()?;
            let value: Expr = input.parse()?;
            (Some(colon), value)
        } else if let Member::Named(ident) = &member {
            let value = Expr::Path(expr::Path {
                attrs: Vec::new(),
                qself: None,
                path: Path::from(ident.clone()),
            });
            (None, value)
        } else {
            unreachable!()
        };
        Ok(FieldValue {
            attrs,
            member,
            colon,
            expr: value,
        })
    }
}
impl Parse for expr::Struct {
    fn parse(input: Stream) -> Res<Self> {
        let (qself, path) = qpath(input, true)?;
        expr_struct_helper(input, qself, path)
    }
}
fn expr_struct_helper(input: Stream, qself: Option<QSelf>, path: Path) -> Res<expr::Struct> {
    let content;
    let brace = braced!(content in input);
    let mut fields = Punctuated::new();
    while !content.is_empty() {
        if content.peek(Token![..]) {
            return Ok(expr::Struct {
                attrs: Vec::new(),
                qself,
                path,
                brace,
                fields,
                dot2: Some(content.parse()?),
                rest: if content.is_empty() {
                    None
                } else {
                    Some(Box::new(content.parse()?))
                },
            });
        }
        fields.push(content.parse()?);
        if content.is_empty() {
            break;
        }
        let punct: Token![,] = content.parse()?;
        fields.push_punct(punct);
    }
    Ok(expr::Struct {
        attrs: Vec::new(),
        qself,
        path,
        brace,
        fields,
        dot2: None,
        rest: None,
    })
}
impl Parse for expr::Unsafe {
    fn parse(input: Stream) -> Res<Self> {
        let unsafe_: Token![unsafe] = input.parse()?;
        let content;
        let brace = braced!(content in input);
        let inner_attrs = content.call(attr::Attr::parse_inner)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(expr::Unsafe {
            attrs: inner_attrs,
            unsafe_,
            block: Block { brace, stmts },
        })
    }
}
impl Parse for expr::Block {
    fn parse(input: Stream) -> Res<Self> {
        let mut attrs = input.call(attr::Attr::parse_outer)?;
        let label: Option<Label> = input.parse()?;
        let content;
        let brace = braced!(content in input);
        attr::inner(&content, &mut attrs)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(expr::Block {
            attrs,
            label,
            block: Block { brace, stmts },
        })
    }
}
fn expr_range(input: Stream, allow_struct: AllowStruct) -> Res<expr::Range> {
    let limits: RangeLimits = input.parse()?;
    let end = if matches!(limits, RangeLimits::HalfOpen(_))
        && (input.is_empty()
            || input.peek(Token![,])
            || input.peek(Token![;])
            || input.peek(Token![.]) && !input.peek(Token![..])
            || !allow_struct.0 && input.peek(tok::Brace))
    {
        None
    } else {
        let to = ambiguous_expr(input, allow_struct)?;
        Some(Box::new(to))
    };
    Ok(expr::Range {
        attrs: Vec::new(),
        beg: None,
        limits,
        end,
    })
}
impl Parse for RangeLimits {
    fn parse(input: Stream) -> Res<Self> {
        let lookahead = input.lookahead1();
        let dot_dot = lookahead.peek(Token![..]);
        let dot_dot_eq = dot_dot && lookahead.peek(Token![..=]);
        let dot_dot_dot = dot_dot && input.peek(Token![...]);
        if dot_dot_eq {
            input.parse().map(RangeLimits::Closed)
        } else if dot_dot && !dot_dot_dot {
            input.parse().map(RangeLimits::HalfOpen)
        } else {
            Err(lookahead.error())
        }
    }
}
impl RangeLimits {
    pub fn parse_obsolete(input: Stream) -> Res<Self> {
        let lookahead = input.lookahead1();
        let dot_dot = lookahead.peek(Token![..]);
        let dot_dot_eq = dot_dot && lookahead.peek(Token![..=]);
        let dot_dot_dot = dot_dot && input.peek(Token![...]);
        if dot_dot_eq {
            input.parse().map(RangeLimits::Closed)
        } else if dot_dot_dot {
            let dot3: Token![...] = input.parse()?;
            Ok(RangeLimits::Closed(Token![..=](dot3.spans)))
        } else if dot_dot {
            input.parse().map(RangeLimits::HalfOpen)
        } else {
            Err(lookahead.error())
        }
    }
}
impl Parse for expr::Path {
    fn parse(input: Stream) -> Res<Self> {
        #[cfg(not(feature = "full"))]
        let attrs = Vec::new();
        let attrs = input.call(attr::Attr::parse_outer)?;
        let (qself, path) = qpath(input, true)?;
        Ok(expr::Path { attrs, qself, path })
    }
}
impl Parse for Member {
    fn parse(input: Stream) -> Res<Self> {
        if input.peek(Ident) {
            input.parse().map(Member::Named)
        } else if input.peek(lit::Int) {
            input.parse().map(Member::Unnamed)
        } else {
            Err(input.error("expected identifier or integer"))
        }
    }
}
impl Parse for Arm {
    fn parse(input: Stream) -> Res<Arm> {
        let requires_comma;
        Ok(Arm {
            attrs: input.call(attr::Attr::parse_outer)?,
            pat: pat::Pat::parse_multi(input)?,
            guard: {
                if input.peek(Token![if]) {
                    let if_: Token![if] = input.parse()?;
                    let guard: Expr = input.parse()?;
                    Some((if_, Box::new(guard)))
                } else {
                    None
                }
            },
            fat_arrow: input.parse()?,
            body: {
                let body = input.call(expr_early)?;
                requires_comma = requires_terminator(&body);
                Box::new(body)
            },
            comma: {
                if requires_comma && !input.is_empty() {
                    Some(input.parse()?)
                } else {
                    input.parse()?
                }
            },
        })
    }
}
impl Parse for Index {
    fn parse(input: Stream) -> Res<Self> {
        let lit: lit::Int = input.parse()?;
        if lit.suffix().is_empty() {
            Ok(Index {
                index: lit.base10_digits().parse().map_err(|err| Err::new(lit.span(), err))?,
                span: lit.span(),
            })
        } else {
            Err(Err::new(lit.span(), "expected unsuffixed integer"))
        }
    }
}
fn multi_index(e: &mut Expr, dot: &mut Token![.], float: lit::Float) -> Res<bool> {
    let float_token = float.token();
    let float_span = float_token.span();
    let mut float_repr = float_token.to_string();
    let trailing_dot = float_repr.ends_with('.');
    if trailing_dot {
        float_repr.truncate(float_repr.len() - 1);
    }
    let mut offset = 0;
    for part in float_repr.split('.') {
        let mut index: Index = super::parse_str(part).map_err(|err| Err::new(float_span, err))?;
        let part_end = offset + part.len();
        index.span = float_token.subspan(offset..part_end).unwrap_or(float_span);
        let base = mem::replace(e, Expr::DUMMY);
        *e = Expr::Field(expr::Field {
            attrs: Vec::new(),
            base: Box::new(base),
            dot: Token![.](dot.span),
            memb: Member::Unnamed(index),
        });
        let dot_span = float_token.subspan(part_end..part_end + 1).unwrap_or(float_span);
        *dot = Token![.](dot_span);
        offset = part_end + 1;
    }
    Ok(!trailing_dot)
}
impl Member {
    fn is_named(&self) -> bool {
        match self {
            Member::Named(_) => true,
            Member::Unnamed(_) => false,
        }
    }
}
fn check_cast(input: Stream) -> Res<()> {
    let kind = if input.peek(Token![.]) && !input.peek(Token![..]) {
        if input.peek2(Token![await]) {
            "`.await`"
        } else if input.peek2(Ident) && (input.peek3(tok::Paren) || input.peek3(Token![::])) {
            "a method call"
        } else {
            "a field access"
        }
    } else if input.peek(Token![?]) {
        "`?`"
    } else if input.peek(tok::Bracket) {
        "indexing"
    } else if input.peek(tok::Paren) {
        "a function call"
    } else {
        return Ok(());
    };
    let msg = format!("casts cannot be followed by {}", kind);
    Err(input.error(msg))
}

impl ToTokens for expr::Array {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.bracket.surround(tokens, |tokens| {
            self.elems.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Assign {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.left.to_tokens(tokens);
        self.eq.to_tokens(tokens);
        self.right.to_tokens(tokens);
    }
}
impl ToTokens for expr::Async {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.async_.to_tokens(tokens);
        self.move_.to_tokens(tokens);
        self.block.to_tokens(tokens);
    }
}
impl ToTokens for expr::Await {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.dot.to_tokens(tokens);
        self.await_.to_tokens(tokens);
    }
}
impl ToTokens for expr::Binary {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.left.to_tokens(tokens);
        self.op.to_tokens(tokens);
        self.right.to_tokens(tokens);
    }
}
impl ToTokens for expr::Block {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.label.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for expr::Break {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.break_.to_tokens(tokens);
        self.label.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for expr::Call {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.func.to_tokens(tokens);
        self.paren.surround(tokens, |tokens| {
            self.args.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Cast {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.as_.to_tokens(tokens);
        self.typ.to_tokens(tokens);
    }
}
impl ToTokens for expr::Closure {
    fn to_tokens(&self, tokens: &mut Stream) {
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
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.const_.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for expr::Continue {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.continue_.to_tokens(tokens);
        self.label.to_tokens(tokens);
    }
}
impl ToTokens for expr::Field {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.base.to_tokens(tokens);
        self.dot.to_tokens(tokens);
        self.memb.to_tokens(tokens);
    }
}
impl ToTokens for expr::ForLoop {
    fn to_tokens(&self, tokens: &mut Stream) {
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
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.group.surround(tokens, |tokens| {
            self.expr.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::If {
    fn to_tokens(&self, tokens: &mut Stream) {
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
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.bracket.surround(tokens, |tokens| {
            self.index.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Infer {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.underscore.to_tokens(tokens);
    }
}
impl ToTokens for expr::Let {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.let_.to_tokens(tokens);
        self.pat.to_tokens(tokens);
        self.eq.to_tokens(tokens);
        wrap_bare_struct(tokens, &self.expr);
    }
}
impl ToTokens for expr::Lit {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.lit.to_tokens(tokens);
    }
}
impl ToTokens for expr::Loop {
    fn to_tokens(&self, tokens: &mut Stream) {
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
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.mac.to_tokens(tokens);
    }
}
impl ToTokens for expr::Match {
    fn to_tokens(&self, tokens: &mut Stream) {
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
    fn to_tokens(&self, tokens: &mut Stream) {
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
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.paren.surround(tokens, |tokens| {
            self.expr.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Path {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        print_path(tokens, &self.qself, &self.path);
    }
}
impl ToTokens for expr::Range {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.beg.to_tokens(tokens);
        self.limits.to_tokens(tokens);
        self.end.to_tokens(tokens);
    }
}
impl ToTokens for expr::Ref {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.and.to_tokens(tokens);
        self.mut_.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for expr::Repeat {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.bracket.surround(tokens, |tokens| {
            self.expr.to_tokens(tokens);
            self.semi.to_tokens(tokens);
            self.len.to_tokens(tokens);
        });
    }
}
impl ToTokens for expr::Return {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.return_.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for expr::Struct {
    fn to_tokens(&self, tokens: &mut Stream) {
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
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.expr.to_tokens(tokens);
        self.question.to_tokens(tokens);
    }
}
impl ToTokens for expr::TryBlock {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.try_.to_tokens(tokens);
        self.block.to_tokens(tokens);
    }
}
impl ToTokens for expr::Tuple {
    fn to_tokens(&self, tokens: &mut Stream) {
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
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.op.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
impl ToTokens for expr::Unsafe {
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.unsafe_.to_tokens(tokens);
        self.block.brace.surround(tokens, |tokens| {
            inner_attrs_to_tokens(&self.attrs, tokens);
            tokens.append_all(&self.block.stmts);
        });
    }
}
impl ToTokens for expr::While {
    fn to_tokens(&self, tokens: &mut Stream) {
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
    fn to_tokens(&self, tokens: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, tokens);
        self.yield_.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
