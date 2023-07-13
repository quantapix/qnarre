use super::*;
use quote::IdentFragment;

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
    pub fn parse_without_eager_brace(x: Stream) -> Res<Expr> {
        ambiguous_expr(x, AllowStruct(false))
    }
}
impl Parse for Expr {
    fn parse(x: Stream) -> Res<Self> {
        ambiguous_expr(x, AllowStruct(true))
    }
}

macro_rules! impl_by_parsing_expr {
    (
        $(
            $ty:ty, $v:ident, $m:expr,
        )*
    ) => {
        $(
            impl Parse for $ty {
                fn parse(x: Stream) -> Res<Self> {
                    let mut y: Expr = x.parse()?;
                    loop {
                        match y {
                            Expr::$v(x) => return Ok(x),
                            Expr::Group(x) => expr = *x.expr,
                            _ => return Err(Error::new_spanned(expr, $m)),
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

pub struct Array {
    pub attrs: Vec<attr::Attr>,
    pub bracket: tok::Bracket,
    pub elems: Punctuated<Expr, Token![,]>,
}
impl Parse for Array {
    fn parse(x: Stream) -> Res<Self> {
        let y;
        let bracket = bracketed!(y in x);
        let mut elems = Punctuated::new();
        while !y.is_empty() {
            let first: Expr = y.parse()?;
            elems.push_value(first);
            if y.is_empty() {
                break;
            }
            let punct = y.parse()?;
            elems.push_punct(punct);
        }
        Ok(Array {
            attrs: Vec::new(),
            bracket,
            elems,
        })
    }
}
impl ToTokens for Array {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.bracket.surround(ys, |ys| {
            self.elems.to_tokens(ys);
        });
    }
}

pub struct Assign {
    pub attrs: Vec<attr::Attr>,
    pub left: Box<Expr>,
    pub eq: Token![=],
    pub right: Box<Expr>,
}
impl ToTokens for Assign {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.left.to_tokens(ys);
        self.eq.to_tokens(ys);
        self.right.to_tokens(ys);
    }
}

pub struct Async {
    pub attrs: Vec<attr::Attr>,
    pub async_: Token![async],
    pub move_: Option<Token![move]>,
    pub block: stmt::Block,
}
impl Parse for Async {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Async {
            attrs: Vec::new(),
            async_: x.parse()?,
            move_: x.parse()?,
            block: x.parse()?,
        })
    }
}
impl ToTokens for Async {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.async_.to_tokens(ys);
        self.move_.to_tokens(ys);
        self.block.to_tokens(ys);
    }
}

pub struct Await {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub dot: Token![.],
    pub await_: Token![await],
}
impl ToTokens for Await {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.expr.to_tokens(ys);
        self.dot.to_tokens(ys);
        self.await_.to_tokens(ys);
    }
}

pub struct Binary {
    pub attrs: Vec<attr::Attr>,
    pub left: Box<Expr>,
    pub op: BinOp,
    pub right: Box<Expr>,
}
impl ToTokens for Binary {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.left.to_tokens(ys);
        self.op.to_tokens(ys);
        self.right.to_tokens(ys);
    }
}

pub struct Block {
    pub attrs: Vec<attr::Attr>,
    pub label: Option<Label>,
    pub block: stmt::Block,
}
impl Parse for Block {
    fn parse(x: Stream) -> Res<Self> {
        let mut attrs = x.call(attr::Attr::parse_outer)?;
        let label: Option<Label> = x.parse()?;
        let y;
        let brace = braced!(y in x);
        attr::inner(&y, &mut attrs)?;
        let stmts = y.call(Block::parse_within)?;
        Ok(Block {
            attrs,
            label,
            block: stmt::Block { brace, stmts },
        })
    }
}
impl ToTokens for Block {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.label.to_tokens(ys);
        self.block.brace.surround(ys, |ys| {
            inner_attrs_to_tokens(&self.attrs, ys);
            ys.append_all(&self.block.stmts);
        });
    }
}

pub struct Break {
    pub attrs: Vec<attr::Attr>,
    pub break_: Token![break],
    pub label: Option<Lifetime>,
    pub expr: Option<Box<Expr>>,
}
impl Parse for Break {
    fn parse(x: Stream) -> Res<Self> {
        let allow = AllowStruct(true);
        expr_break(x, allow)
    }
}
impl ToTokens for Break {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.break_.to_tokens(ys);
        self.label.to_tokens(ys);
        self.expr.to_tokens(ys);
    }
}

pub struct Call {
    pub attrs: Vec<attr::Attr>,
    pub func: Box<Expr>,
    pub paren: tok::Paren,
    pub args: Punctuated<Expr, Token![,]>,
}
impl ToTokens for Call {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.func.to_tokens(ys);
        self.paren.surround(ys, |ys| {
            self.args.to_tokens(ys);
        });
    }
}

pub struct Cast {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub as_: Token![as],
    pub typ: Box<typ::Type>,
}
impl ToTokens for Cast {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.expr.to_tokens(ys);
        self.as_.to_tokens(ys);
        self.typ.to_tokens(ys);
    }
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
impl Parse for Closure {
    fn parse(x: Stream) -> Res<Self> {
        let allow = AllowStruct(true);
        expr_closure(x, allow)
    }
}
impl ToTokens for Closure {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.lifes.to_tokens(ys);
        self.const_.to_tokens(ys);
        self.static_.to_tokens(ys);
        self.async_.to_tokens(ys);
        self.move_.to_tokens(ys);
        self.or1.to_tokens(ys);
        self.inputs.to_tokens(ys);
        self.or2.to_tokens(ys);
        self.ret.to_tokens(ys);
        self.body.to_tokens(ys);
    }
}

pub struct Const {
    pub attrs: Vec<attr::Attr>,
    pub const_: Token![const],
    pub block: stmt::Block,
}
impl Parse for Const {
    fn parse(x: Stream) -> Res<Self> {
        let const_: Token![const] = x.parse()?;
        let y;
        let brace = braced!(y in x);
        let attrs = y.call(attr::Attr::parse_inner)?;
        let stmts = y.call(Block::parse_within)?;
        Ok(Const {
            attrs,
            const_,
            block: stmt::Block { brace, stmts },
        })
    }
}
impl ToTokens for Const {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.const_.to_tokens(ys);
        self.block.brace.surround(ys, |ys| {
            inner_attrs_to_tokens(&self.attrs, ys);
            ys.append_all(&self.block.stmts);
        });
    }
}

pub struct Continue {
    pub attrs: Vec<attr::Attr>,
    pub continue_: Token![continue],
    pub label: Option<Lifetime>,
}
impl Parse for Continue {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Continue {
            attrs: Vec::new(),
            continue_: x.parse()?,
            label: x.parse()?,
        })
    }
}
impl ToTokens for Continue {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.continue_.to_tokens(ys);
        self.label.to_tokens(ys);
    }
}

pub struct Field {
    pub attrs: Vec<attr::Attr>,
    pub base: Box<Expr>,
    pub dot: Token![.],
    pub memb: Member,
}
impl ToTokens for Field {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.base.to_tokens(ys);
        self.dot.to_tokens(ys);
        self.memb.to_tokens(ys);
    }
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
impl Parse for ForLoop {
    fn parse(x: Stream) -> Res<Self> {
        let mut attrs = x.call(attr::Attr::parse_outer)?;
        let label: Option<Label> = x.parse()?;
        let for_: Token![for] = x.parse()?;
        let pat = pat::Pat::parse_multi(x)?;
        let in_: Token![in] = x.parse()?;
        let expr: Expr = x.call(Expr::parse_without_eager_brace)?;
        let y;
        let brace = braced!(y in x);
        attr::inner(&y, &mut attrs)?;
        let stmts = y.call(Block::parse_within)?;
        Ok(ForLoop {
            attrs,
            label,
            for_,
            pat: Box::new(pat),
            in_,
            expr: Box::new(expr),
            body: stmt::Block { brace, stmts },
        })
    }
}
impl ToTokens for ForLoop {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.label.to_tokens(ys);
        self.for_.to_tokens(ys);
        self.pat.to_tokens(ys);
        self.in_.to_tokens(ys);
        wrap_bare_struct(ys, &self.expr);
        self.body.brace.surround(ys, |ys| {
            inner_attrs_to_tokens(&self.attrs, ys);
            ys.append_all(&self.body.stmts);
        });
    }
}

pub struct Group {
    pub attrs: Vec<attr::Attr>,
    pub group: tok::Group,
    pub expr: Box<Expr>,
}
impl ToTokens for Group {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.group.surround(ys, |ys| {
            self.expr.to_tokens(ys);
        });
    }
}

pub struct If {
    pub attrs: Vec<attr::Attr>,
    pub if_: Token![if],
    pub cond: Box<Expr>,
    pub then_: stmt::Block,
    pub else_: Option<(Token![else], Box<Expr>)>,
}
impl Parse for If {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = x.call(attr::Attr::parse_outer)?;
        Ok(If {
            attrs,
            if_: x.parse()?,
            cond: Box::new(x.call(Expr::parse_without_eager_brace)?),
            then_: x.parse()?,
            else_: {
                if x.peek(Token![else]) {
                    Some(x.call(else_block)?)
                } else {
                    None
                }
            },
        })
    }
}
impl ToTokens for If {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.if_.to_tokens(ys);
        wrap_bare_struct(ys, &self.cond);
        self.then_branch.to_tokens(ys);
        if let Some((else_, else_)) = &self.else_branch {
            else_.to_tokens(ys);
            match **else_ {
                Expr::If(_) | Expr::Block(_) => else_.to_tokens(ys),
                _ => tok::Brace::default().surround(ys, |ys| else_.to_tokens(ys)),
            }
        }
    }
}

pub struct Index {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub bracket: tok::Bracket,
    pub idx: Box<Expr>,
}
impl Parse for Index {
    fn parse(x: Stream) -> Res<Self> {
        let lit: lit::Int = x.parse()?;
        if lit.suffix().is_empty() {
            Ok(Idx {
                idx: lit.base10_digits().parse().map_err(|x| Err::new(lit.span(), x))?,
                span: lit.span(),
            })
        } else {
            Err(Err::new(lit.span(), "expected unsuffixed integer"))
        }
    }
}
impl ToTokens for Index {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.expr.to_tokens(ys);
        self.bracket.surround(ys, |ys| {
            self.idx.to_tokens(ys);
        });
    }
}

pub struct Infer {
    pub attrs: Vec<attr::Attr>,
    pub underscore: Token![_],
}
impl Parse for Infer {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Infer {
            attrs: x.call(attr::Attr::parse_outer)?,
            underscore: x.parse()?,
        })
    }
}
impl ToTokens for Infer {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.underscore.to_tokens(ys);
    }
}

pub struct Let {
    pub attrs: Vec<attr::Attr>,
    pub let_: Token![let],
    pub pat: Box<pat::Pat>,
    pub eq: Token![=],
    pub expr: Box<Expr>,
}
impl Parse for Let {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Let {
            attrs: Vec::new(),
            let_: x.parse()?,
            pat: Box::new(pat::Pat::parse_multi(x)?),
            eq: x.parse()?,
            expr: Box::new({
                let allow = AllowStruct(false);
                let lhs = unary_expr(x, allow)?;
                parse_expr(x, lhs, allow, Precedence::Compare)?
            }),
        })
    }
}
impl ToTokens for Let {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.let_.to_tokens(ys);
        self.pat.to_tokens(ys);
        self.eq.to_tokens(ys);
        wrap_bare_struct(ys, &self.expr);
    }
}

pub struct Lit {
    pub attrs: Vec<attr::Attr>,
    pub lit: lit::Lit,
}
impl Parse for Lit {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Lit {
            attrs: Vec::new(),
            lit: x.parse()?,
        })
    }
}
impl ToTokens for Lit {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.lit.to_tokens(ys);
    }
}

pub struct Loop {
    pub attrs: Vec<attr::Attr>,
    pub label: Option<Label>,
    pub loop_: Token![loop],
    pub body: stmt::Block,
}
impl Parse for Loop {
    fn parse(x: Stream) -> Res<Self> {
        let mut attrs = x.call(attr::Attr::parse_outer)?;
        let label: Option<Label> = x.parse()?;
        let loop_: Token![loop] = x.parse()?;
        let y;
        let brace = braced!(y in x);
        attr::inner(&y, &mut attrs)?;
        let stmts = y.call(Block::parse_within)?;
        Ok(Loop {
            attrs,
            label,
            loop_,
            body: stmt::Block { brace, stmts },
        })
    }
}
impl ToTokens for Loop {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.label.to_tokens(ys);
        self.loop_.to_tokens(ys);
        self.body.brace.surround(ys, |ys| {
            inner_attrs_to_tokens(&self.attrs, ys);
            ys.append_all(&self.body.stmts);
        });
    }
}

pub struct Mac {
    pub attrs: Vec<attr::Attr>,
    pub mac: mac::Mac,
}
impl Parse for Mac {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Mac {
            attrs: Vec::new(),
            mac: x.parse()?,
        })
    }
}
impl ToTokens for Mac {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.mac.to_tokens(ys);
    }
}

pub struct Match {
    pub attrs: Vec<attr::Attr>,
    pub match_: Token![match],
    pub expr: Box<Expr>,
    pub brace: tok::Brace,
    pub arms: Vec<Arm>,
}
impl Parse for Match {
    fn parse(x: Stream) -> Res<Self> {
        let mut attrs = x.call(attr::Attr::parse_outer)?;
        let match_: Token![match] = x.parse()?;
        let expr = Expr::parse_without_eager_brace(x)?;
        let y;
        let brace = braced!(y in x);
        attr::inner(&y, &mut attrs)?;
        let mut arms = Vec::new();
        while !y.is_empty() {
            arms.push(y.call(Arm::parse)?);
        }
        Ok(Match {
            attrs,
            match_,
            expr: Box::new(expr),
            brace,
            arms,
        })
    }
}
impl ToTokens for Match {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.match_.to_tokens(ys);
        wrap_bare_struct(ys, &self.expr);
        self.brace.surround(ys, |ys| {
            inner_attrs_to_tokens(&self.attrs, ys);
            for (i, arm) in self.arms.iter().enumerate() {
                arm.to_tokens(ys);
                let is_last = i == self.arms.len() - 1;
                if !is_last && requires_terminator(&arm.body) && arm.comma.is_none() {
                    <Token![,]>::default().to_tokens(ys);
                }
            }
        });
    }
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
impl ToTokens for MethodCall {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.expr.to_tokens(ys);
        self.dot.to_tokens(ys);
        self.method.to_tokens(ys);
        self.turbofish.to_tokens(ys);
        self.paren.surround(ys, |ys| {
            self.args.to_tokens(ys);
        });
    }
}

pub struct Paren {
    pub attrs: Vec<attr::Attr>,
    pub paren: tok::Paren,
    pub expr: Box<Expr>,
}
impl Parse for Paren {
    fn parse(x: Stream) -> Res<Self> {
        expr_paren(x)
    }
}
impl ToTokens for Paren {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.paren.surround(ys, |ys| {
            self.expr.to_tokens(ys);
        });
    }
}

pub struct Path {
    pub attrs: Vec<attr::Attr>,
    pub qself: Option<QSelf>,
    pub path: path::Path,
}
impl Parse for Path {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = x.call(attr::Attr::parse_outer)?;
        let (qself, path) = qpath(x, true)?;
        Ok(Path { attrs, qself, path })
    }
}
impl ToTokens for Path {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        print_path(ys, &self.qself, &self.path);
    }
}

pub struct Range {
    pub attrs: Vec<attr::Attr>,
    pub beg: Option<Box<Expr>>,
    pub limits: RangeLimits,
    pub end: Option<Box<Expr>>,
}
impl ToTokens for Range {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.beg.to_tokens(ys);
        self.limits.to_tokens(ys);
        self.end.to_tokens(ys);
    }
}

pub struct Ref {
    pub attrs: Vec<attr::Attr>,
    pub and: Token![&],
    pub mut_: Option<Token![mut]>,
    pub expr: Box<Expr>,
}
impl Parse for Ref {
    fn parse(x: Stream) -> Res<Self> {
        let allow = AllowStruct(true);
        Ok(Ref {
            attrs: Vec::new(),
            and: x.parse()?,
            mut_: x.parse()?,
            expr: Box::new(unary_expr(x, allow)?),
        })
    }
}
impl ToTokens for Ref {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.and.to_tokens(ys);
        self.mut_.to_tokens(ys);
        self.expr.to_tokens(ys);
    }
}

pub struct Repeat {
    pub attrs: Vec<attr::Attr>,
    pub bracket: tok::Bracket,
    pub expr: Box<Expr>,
    pub semi: Token![;],
    pub len: Box<Expr>,
}
impl Parse for Repeat {
    fn parse(x: Stream) -> Res<Self> {
        let y;
        Ok(Repeat {
            bracket: bracketed!(y in x),
            attrs: Vec::new(),
            expr: y.parse()?,
            semi: y.parse()?,
            len: y.parse()?,
        })
    }
}
impl ToTokens for Repeat {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.bracket.surround(ys, |ys| {
            self.expr.to_tokens(ys);
            self.semi.to_tokens(ys);
            self.len.to_tokens(ys);
        });
    }
}

pub struct Return {
    pub attrs: Vec<attr::Attr>,
    pub return_: Token![return],
    pub expr: Option<Box<Expr>>,
}
impl Parse for Return {
    fn parse(x: Stream) -> Res<Self> {
        let allow = AllowStruct(true);
        expr_ret(x, allow)
    }
}
impl ToTokens for Return {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.return_.to_tokens(ys);
        self.expr.to_tokens(ys);
    }
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
impl Parse for Struct {
    fn parse(x: Stream) -> Res<Self> {
        let (qself, path) = qpath(x, true)?;
        expr_struct_helper(x, qself, path)
    }
}
impl ToTokens for Struct {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        print_path(ys, &self.qself, &self.path);
        self.brace.surround(ys, |ys| {
            self.fields.to_tokens(ys);
            if let Some(dot2) = &self.dot2 {
                dot2.to_tokens(ys);
            } else if self.rest.is_some() {
                Token![..](pm2::Span::call_site()).to_tokens(ys);
            }
            self.rest.to_tokens(ys);
        });
    }
}

pub struct Try {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub question: Token![?],
}
impl ToTokens for Try {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.expr.to_tokens(ys);
        self.question.to_tokens(ys);
    }
}

pub struct TryBlock {
    pub attrs: Vec<attr::Attr>,
    pub try_: Token![try],
    pub block: stmt::Block,
}
impl Parse for TryBlock {
    fn parse(x: Stream) -> Res<Self> {
        Ok(TryBlock {
            attrs: Vec::new(),
            try_: x.parse()?,
            block: x.parse()?,
        })
    }
}
impl ToTokens for TryBlock {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.try_.to_tokens(ys);
        self.block.to_tokens(ys);
    }
}

pub struct Tuple {
    pub attrs: Vec<attr::Attr>,
    pub paren: tok::Paren,
    pub elems: Punctuated<Expr, Token![,]>,
}
impl ToTokens for Tuple {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.paren.surround(ys, |ys| {
            self.elems.to_tokens(ys);
            if self.elems.len() == 1 && !self.elems.trailing_punct() {
                <Token![,]>::default().to_tokens(ys);
            }
        });
    }
}

pub struct Unary {
    pub attrs: Vec<attr::Attr>,
    pub op: UnOp,
    pub expr: Box<Expr>,
}
impl Parse for Unary {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = Vec::new();
        let allow = AllowStruct(true);
        expr_unary(x, attrs, allow)
    }
}
impl ToTokens for Unary {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.op.to_tokens(ys);
        self.expr.to_tokens(ys);
    }
}

pub struct Unsafe {
    pub attrs: Vec<attr::Attr>,
    pub unsafe_: Token![unsafe],
    pub block: stmt::Block,
}
impl Parse for Unsafe {
    fn parse(x: Stream) -> Res<Self> {
        let unsafe_: Token![unsafe] = x.parse()?;
        let y;
        let brace = braced!(y in x);
        let attrs = y.call(attr::Attr::parse_inner)?;
        let stmts = y.call(Block::parse_within)?;
        Ok(Unsafe {
            attrs,
            unsafe_,
            block: stmt::Block { brace, stmts },
        })
    }
}
impl ToTokens for Unsafe {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.unsafe_.to_tokens(ys);
        self.block.brace.surround(ys, |ys| {
            inner_attrs_to_tokens(&self.attrs, ys);
            ys.append_all(&self.block.stmts);
        });
    }
}

pub struct While {
    pub attrs: Vec<attr::Attr>,
    pub label: Option<Label>,
    pub while_: Token![while],
    pub cond: Box<Expr>,
    pub block: stmt::Block,
}
impl Parse for While {
    fn parse(x: Stream) -> Res<Self> {
        let mut attrs = x.call(attr::Attr::parse_outer)?;
        let label: Option<Label> = x.parse()?;
        let while_: Token![while] = x.parse()?;
        let cond = Expr::parse_without_eager_brace(x)?;
        let y;
        let brace = braced!(y in x);
        attr::inner(&y, &mut attrs)?;
        let stmts = y.call(Block::parse_within)?;
        Ok(While {
            attrs,
            label,
            while_,
            cond: Box::new(cond),
            block: stmt::Block { brace, stmts },
        })
    }
}
impl ToTokens for While {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.label.to_tokens(ys);
        self.while_.to_tokens(ys);
        wrap_bare_struct(ys, &self.cond);
        self.block.brace.surround(ys, |ys| {
            inner_attrs_to_tokens(&self.attrs, ys);
            ys.append_all(&self.block.stmts);
        });
    }
}

pub struct Yield {
    pub attrs: Vec<attr::Attr>,
    pub yield_: Token![yield],
    pub expr: Option<Box<Expr>>,
}
impl Parse for Yield {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Yield {
            attrs: Vec::new(),
            yield_: x.parse()?,
            expr: {
                if !x.is_empty() && !x.peek(Token![,]) && !x.peek(Token![;]) {
                    Some(x.parse()?)
                } else {
                    None
                }
            },
        })
    }
}
impl ToTokens for Yield {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.yield_.to_tokens(ys);
        self.expr.to_tokens(ys);
    }
}

pub enum Member {
    Named(Ident),
    Unnamed(Idx),
}
impl Member {
    fn is_named(&self) -> bool {
        match self {
            Member::Named(_) => true,
            Member::Unnamed(_) => false,
        }
    }
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
impl Parse for Member {
    fn parse(x: Stream) -> Res<Self> {
        if x.peek(Ident) {
            x.parse().map(Member::Named)
        } else if x.peek(lit::Int) {
            x.parse().map(Member::Unnamed)
        } else {
            Err(x.error("expected identifier or integer"))
        }
    }
}
impl ToTokens for Member {
    fn to_tokens(&self, ys: &mut Stream) {
        match self {
            Member::Named(x) => x.to_tokens(ys),
            Member::Unnamed(x) => x.to_tokens(ys),
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
impl ToTokens for Idx {
    fn to_tokens(&self, ys: &mut Stream) {
        let mut lit = pm2::Lit::i64_unsuffixed(i64::from(self.idx));
        lit.set_span(self.span);
        ys.append(lit);
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
impl Parse for BinOp {
    fn parse(x: Stream) -> Res<Self> {
        if x.peek(Token![+=]) {
            x.parse().map(BinOp::AddAssign)
        } else if x.peek(Token![-=]) {
            x.parse().map(BinOp::SubAssign)
        } else if x.peek(Token![*=]) {
            x.parse().map(BinOp::MulAssign)
        } else if x.peek(Token![/=]) {
            x.parse().map(BinOp::DivAssign)
        } else if x.peek(Token![%=]) {
            x.parse().map(BinOp::RemAssign)
        } else if x.peek(Token![^=]) {
            x.parse().map(BinOp::BitXorAssign)
        } else if x.peek(Token![&=]) {
            x.parse().map(BinOp::BitAndAssign)
        } else if x.peek(Token![|=]) {
            x.parse().map(BinOp::BitOrAssign)
        } else if x.peek(Token![<<=]) {
            x.parse().map(BinOp::ShlAssign)
        } else if x.peek(Token![>>=]) {
            x.parse().map(BinOp::ShrAssign)
        } else {
            parse_binop(x)
        }
    }
}
impl ToTokens for BinOp {
    fn to_tokens(&self, ys: &mut Stream) {
        use BinOp::*;
        match self {
            Add(x) => x.to_tokens(ys),
            Sub(x) => x.to_tokens(ys),
            Mul(x) => x.to_tokens(ys),
            Div(x) => x.to_tokens(ys),
            Rem(x) => x.to_tokens(ys),
            And(x) => x.to_tokens(ys),
            Or(x) => x.to_tokens(ys),
            BitXor(x) => x.to_tokens(ys),
            BitAnd(x) => x.to_tokens(ys),
            BitOr(x) => x.to_tokens(ys),
            Shl(x) => x.to_tokens(ys),
            Shr(x) => x.to_tokens(ys),
            Eq(x) => x.to_tokens(ys),
            Lt(x) => x.to_tokens(ys),
            Le(x) => x.to_tokens(ys),
            Ne(x) => x.to_tokens(ys),
            Ge(x) => x.to_tokens(ys),
            Gt(x) => x.to_tokens(ys),
            AddAssign(x) => x.to_tokens(ys),
            SubAssign(x) => x.to_tokens(ys),
            MulAssign(x) => x.to_tokens(ys),
            DivAssign(x) => x.to_tokens(ys),
            RemAssign(x) => x.to_tokens(ys),
            BitXorAssign(x) => x.to_tokens(ys),
            BitAndAssign(x) => x.to_tokens(ys),
            BitOrAssign(x) => x.to_tokens(ys),
            ShlAssign(x) => x.to_tokens(ys),
            ShrAssign(x) => x.to_tokens(ys),
        }
    }
}

pub enum UnOp {
    Deref(Token![*]),
    Not(Token![!]),
    Neg(Token![-]),
}
impl Parse for UnOp {
    fn parse(x: Stream) -> Res<Self> {
        let look = x.lookahead1();
        if look.peek(Token![*]) {
            x.parse().map(UnOp::Deref)
        } else if look.peek(Token![!]) {
            x.parse().map(UnOp::Not)
        } else if look.peek(Token![-]) {
            x.parse().map(UnOp::Neg)
        } else {
            Err(look.error())
        }
    }
}
impl ToTokens for UnOp {
    fn to_tokens(&self, ys: &mut Stream) {
        match self {
            UnOp::Deref(x) => x.to_tokens(ys),
            UnOp::Not(x) => x.to_tokens(ys),
            UnOp::Neg(x) => x.to_tokens(ys),
        }
    }
}

pub struct FieldValue {
    pub attrs: Vec<attr::Attr>,
    pub memb: Member,
    pub colon: Option<Token![:]>,
    pub val: Expr,
}
impl Parse for FieldValue {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = x.call(attr::Attr::parse_outer)?;
        let memb: Member = x.parse()?;
        let (colon, val) = if x.peek(Token![:]) || !memb.is_named() {
            let colon: Token![:] = x.parse()?;
            let y: Expr = x.parse()?;
            (Some(colon), y)
        } else if let Member::Named(ident) = &memb {
            let y = Expr::Path(expr::Path {
                attrs: Vec::new(),
                qself: None,
                path: Path::from(ident.clone()),
            });
            (None, y)
        } else {
            unreachable!()
        };
        Ok(FieldValue {
            attrs,
            memb,
            colon,
            val,
        })
    }
}
impl ToTokens for FieldValue {
    fn to_tokens(&self, ys: &mut Stream) {
        outer_attrs_to_tokens(&self.attrs, ys);
        self.member.to_tokens(ys);
        if let Some(colon) = &self.colon {
            colon.to_tokens(ys);
            self.val.to_tokens(ys);
        }
    }
}

pub struct Label {
    pub name: Lifetime,
    pub colon: Token![:],
}
impl Parse for Label {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Label {
            name: x.parse()?,
            colon: x.parse()?,
        })
    }
}
impl Parse for Option<Label> {
    fn parse(x: Stream) -> Res<Self> {
        if x.peek(Lifetime) {
            x.parse().map(Some)
        } else {
            Ok(None)
        }
    }
}
impl ToTokens for Label {
    fn to_tokens(&self, ys: &mut Stream) {
        self.name.to_tokens(ys);
        self.colon.to_tokens(ys);
    }
}

pub struct Arm {
    pub attrs: Vec<attr::Attr>,
    pub pat: pat::Pat,
    pub guard: Option<(Token![if], Box<Expr>)>,
    pub fat_arrow: Token![=>],
    pub body: Box<Expr>,
    pub comma: Option<Token![,]>,
}
impl Parse for Arm {
    fn parse(x: Stream) -> Res<Arm> {
        let comma;
        Ok(Arm {
            attrs: x.call(attr::Attr::parse_outer)?,
            pat: pat::Pat::parse_multi(x)?,
            guard: {
                if x.peek(Token![if]) {
                    let if_: Token![if] = x.parse()?;
                    let guard: Expr = x.parse()?;
                    Some((if_, Box::new(guard)))
                } else {
                    None
                }
            },
            fat_arrow: x.parse()?,
            body: {
                let body = x.call(expr_early)?;
                comma = requires_terminator(&body);
                Box::new(body)
            },
            comma: {
                if comma && !x.is_empty() {
                    Some(x.parse()?)
                } else {
                    x.parse()?
                }
            },
        })
    }
}
impl ToTokens for Arm {
    fn to_tokens(&self, ys: &mut Stream) {
        ys.append_all(&self.attrs);
        self.pat.to_tokens(ys);
        if let Some((if_, guard)) = &self.guard {
            if_.to_tokens(ys);
            guard.to_tokens(ys);
        }
        self.fat_arrow.to_tokens(ys);
        self.body.to_tokens(ys);
        self.comma.to_tokens(ys);
    }
}

pub enum RangeLimits {
    HalfOpen(Token![..]),
    Closed(Token![..=]),
}
impl RangeLimits {
    pub fn parse_obsolete(x: Stream) -> Res<Self> {
        let look = x.lookahead1();
        let dot2 = look.peek(Token![..]);
        let dot2_eq = dot2 && look.peek(Token![..=]);
        let dot3 = dot2 && x.peek(Token![...]);
        if dot2_eq {
            x.parse().map(RangeLimits::Closed)
        } else if dot3 {
            let y: Token![...] = x.parse()?;
            Ok(RangeLimits::Closed(Token![..=](y.spans)))
        } else if dot2 {
            x.parse().map(RangeLimits::HalfOpen)
        } else {
            Err(look.error())
        }
    }
}
impl Parse for RangeLimits {
    fn parse(x: Stream) -> Res<Self> {
        let look = x.lookahead1();
        let dot2 = look.peek(Token![..]);
        let dot2_eq = dot2 && look.peek(Token![..=]);
        let dot3 = dot2 && x.peek(Token![...]);
        if dot2_eq {
            x.parse().map(RangeLimits::Closed)
        } else if dot2 && !dot3 {
            x.parse().map(RangeLimits::HalfOpen)
        } else {
            Err(look.error())
        }
    }
}
impl ToTokens for RangeLimits {
    fn to_tokens(&self, ys: &mut Stream) {
        match self {
            RangeLimits::HalfOpen(x) => x.to_tokens(ys),
            RangeLimits::Closed(x) => x.to_tokens(ys),
        }
    }
}

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

pub struct AllowStruct(bool);
impl Copy for AllowStruct {}
impl Clone for AllowStruct {
    fn clone(&self) -> Self {
        *self
    }
}

mod kw {
    crate::custom_keyword!(builtin);
    crate::custom_keyword!(raw);
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
fn parse_expr(x: Stream, mut lhs: Expr, allow: AllowStruct, base: Precedence) -> Res<Expr> {
    loop {
        let ahead = x.fork();
        if let Some(op) = match ahead.parse::<BinOp>() {
            Ok(op) if Precedence::of(&op) >= base => Some(op),
            _ => None,
        } {
            x.advance_to(&ahead);
            let prec = Precedence::of(&op);
            let mut rhs = unary_expr(x, allow)?;
            loop {
                let next = peek_precedence(x);
                if next > prec || next == prec && prec == Precedence::Assign {
                    rhs = parse_expr(x, rhs, allow, next)?;
                } else {
                    break;
                }
            }
            lhs = Expr::Binary(Binary {
                attrs: Vec::new(),
                left: Box::new(lhs),
                op,
                right: Box::new(rhs),
            });
        } else if Precedence::Assign >= base && x.peek(Token![=]) && !x.peek(Token![==]) && !x.peek(Token![=>]) {
            let eq: Token![=] = x.parse()?;
            let mut rhs = unary_expr(x, allow)?;
            loop {
                let next = peek_precedence(x);
                if next >= Precedence::Assign {
                    rhs = parse_expr(x, rhs, allow, next)?;
                } else {
                    break;
                }
            }
            lhs = Expr::Assign(Assign {
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
                    || !allow.0 && x.peek(tok::Brace))
            {
                None
            } else {
                let mut rhs = unary_expr(x, allow)?;
                loop {
                    let next = peek_precedence(x);
                    if next > Precedence::Range {
                        rhs = parse_expr(x, rhs, allow, next)?;
                    } else {
                        break;
                    }
                }
                Some(rhs)
            };
            lhs = Expr::Range(Range {
                attrs: Vec::new(),
                beg: Some(Box::new(lhs)),
                limits,
                end: rhs.map(Box::new),
            });
        } else if Precedence::Cast >= base && x.peek(Token![as]) {
            let as_: Token![as] = x.parse()?;
            let plus = false;
            let gen = false;
            let ty = ambig_ty(x, plus, gen)?;
            check_cast(x)?;
            lhs = Expr::Cast(Cast {
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
fn ambiguous_expr(x: Stream, allow: AllowStruct) -> Res<Expr> {
    let y = unary_expr(x, allow)?;
    parse_expr(x, y, allow, Precedence::Any)
}
fn expr_attrs(x: Stream) -> Res<Vec<attr::Attr>> {
    let mut ys = Vec::new();
    loop {
        if x.peek(tok::Group) {
            let ahead = x.fork();
            let group = parse::parse_group(&ahead)?;
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
fn unary_expr(x: Stream, allow: AllowStruct) -> Res<Expr> {
    let beg = x.fork();
    let attrs = x.call(expr_attrs)?;
    if x.peek(Token![&]) {
        let and: Token![&] = x.parse()?;
        let raw: Option<kw::raw> = if x.peek(kw::raw) && (x.peek2(Token![mut]) || x.peek2(Token![const])) {
            Some(x.parse()?)
        } else {
            None
        };
        let mut_: Option<Token![mut]> = x.parse()?;
        if raw.is_some() && mut_.is_none() {
            x.parse::<Token![const]>()?;
        }
        let expr = Box::new(unary_expr(x, allow)?);
        if raw.is_some() {
            Ok(Expr::Verbatim(verbatim_between(&beg, x)))
        } else {
            Ok(Expr::Reference(Ref { attrs, and, mut_, expr }))
        }
    } else if x.peek(Token![*]) || x.peek(Token![!]) || x.peek(Token![-]) {
        expr_unary(x, attrs, allow).map(Expr::Unary)
    } else {
        trailer_expr(beg, attrs, x, allow)
    }
}
fn trailer_expr(beg: Buffer, mut attrs: Vec<attr::Attr>, x: Stream, allow: AllowStruct) -> Res<Expr> {
    let atom = atom_expr(x, allow)?;
    let mut y = trailer_helper(x, atom)?;
    if let Expr::Verbatim(tokens) = &mut y {
        *tokens = verbatim_between(&beg, x);
    } else {
        let inner_attrs = y.replace_attrs(Vec::new());
        attrs.extend(inner_attrs);
        y.replace_attrs(attrs);
    }
    Ok(y)
}
fn trailer_helper(x: Stream, mut y: Expr) -> Res<Expr> {
    loop {
        if x.peek(tok::Paren) {
            let y;
            y = Expr::Call(Call {
                attrs: Vec::new(),
                func: Box::new(y),
                paren: parenthesized!(y in x),
                args: y.parse_terminated(Expr::parse, Token![,])?,
            });
        } else if x.peek(Token![.])
            && !x.peek(Token![..])
            && match y {
                Expr::Range(_) => false,
                _ => true,
            }
        {
            let mut dot: Token![.] = x.parse()?;
            let float_: Option<lit::Float> = x.parse()?;
            if let Some(y) = float_ {
                if multi_index(&mut y, &mut dot, y)? {
                    continue;
                }
            }
            let await_: Option<Token![await]> = x.parse()?;
            if let Some(await_) = await_ {
                y = Expr::Await(Await {
                    attrs: Vec::new(),
                    expr: Box::new(y),
                    dot,
                    await_,
                });
                continue;
            }
            let memb: Member = x.parse()?;
            let turbofish = if memb.is_named() && x.peek(Token![::]) {
                Some(AngledArgs::parse_turbofish(x)?)
            } else {
                None
            };
            if turbofish.is_some() || x.peek(tok::Paren) {
                if let Member::Named(method) = memb {
                    let y;
                    y = Expr::MethodCall(MethodCall {
                        attrs: Vec::new(),
                        expr: Box::new(y),
                        dot,
                        method,
                        turbofish,
                        paren: parenthesized!(y in x),
                        args: y.parse_terminated(Expr::parse, Token![,])?,
                    });
                    continue;
                }
            }
            y = Expr::Field(Field {
                attrs: Vec::new(),
                base: Box::new(y),
                dot,
                memb,
            });
        } else if x.peek(tok::Bracket) {
            let content;
            y = Expr::Index(Index {
                attrs: Vec::new(),
                expr: Box::new(y),
                bracket: bracketed!(content in x),
                idx: content.parse()?,
            });
        } else if x.peek(Token![?]) {
            y = Expr::Try(Try {
                attrs: Vec::new(),
                expr: Box::new(y),
                question: x.parse()?,
            });
        } else {
            break;
        }
    }
    Ok(y)
}
fn atom_expr(x: Stream, allow: AllowStruct) -> Res<Expr> {
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
        expr_closure(x, allow).map(Expr::Closure)
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
        path_or_macro_or_struct(x, allow)
    } else if x.peek(tok::Paren) {
        paren_or_tuple(x)
    } else if x.peek(Token![break]) {
        expr_break(x, allow).map(Expr::Break)
    } else if x.peek(Token![continue]) {
        x.parse().map(Expr::Continue)
    } else if x.peek(Token![return]) {
        expr_ret(x, allow).map(Expr::Return)
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
        expr_range(x, allow).map(Expr::Range)
    } else if x.peek(Token![_]) {
        x.parse().map(Expr::Infer)
    } else if x.peek(Lifetime) {
        let the_label: Label = x.parse()?;
        let mut y = if x.peek(Token![while]) {
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
        match &mut y {
            Expr::While(While { label, .. })
            | Expr::ForLoop(ForLoop { label, .. })
            | Expr::Loop(Loop { label, .. })
            | Expr::Block(Block { label, .. }) => *label = Some(the_label),
            _ => unreachable!(),
        }
        Ok(y)
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
fn path_or_macro_or_struct(x: Stream, #[cfg(feature = "full")] allow: AllowStruct) -> Res<Expr> {
    let (qself, path) = qpath(x, true)?;
    if qself.is_none() && x.peek(Token![!]) && !x.peek(Token![!=]) && path.is_mod_style() {
        let bang: Token![!] = x.parse()?;
        let (delim, toks) = mac::parse_delim(x)?;
        return Ok(Expr::Macro(Mac {
            attrs: Vec::new(),
            mac: Macro {
                path,
                bang,
                delim,
                toks,
            },
        }));
    }
    if allow.0 && x.peek(tok::Brace) {
        return expr_struct_helper(x, qself, path).map(Expr::Struct);
    }
    Ok(Expr::Path(expr::Path {
        attrs: Vec::new(),
        qself,
        path,
    }))
}
fn paren_or_tuple(x: Stream) -> Res<Expr> {
    let y;
    let paren = parenthesized!(y in x);
    if y.is_empty() {
        return Ok(Expr::Tuple(Tuple {
            attrs: Vec::new(),
            paren,
            elems: Punctuated::new(),
        }));
    }
    let first: Expr = y.parse()?;
    if y.is_empty() {
        return Ok(Expr::Paren(Paren {
            attrs: Vec::new(),
            paren,
            expr: Box::new(first),
        }));
    }
    let mut elems = Punctuated::new();
    elems.push_value(first);
    while !y.is_empty() {
        let punct = y.parse()?;
        elems.push_punct(punct);
        if y.is_empty() {
            break;
        }
        let value = y.parse()?;
        elems.push_value(value);
    }
    Ok(Expr::Tuple(Tuple {
        attrs: Vec::new(),
        paren,
        elems,
    }))
}
fn array_or_repeat(x: Stream) -> Res<Expr> {
    let y;
    let bracket = bracketed!(y in x);
    if y.is_empty() {
        return Ok(Expr::Array(Array {
            attrs: Vec::new(),
            bracket,
            elems: Punctuated::new(),
        }));
    }
    let first: Expr = y.parse()?;
    if y.is_empty() || y.peek(Token![,]) {
        let mut elems = Punctuated::new();
        elems.push_value(first);
        while !y.is_empty() {
            let punct = y.parse()?;
            elems.push_punct(punct);
            if y.is_empty() {
                break;
            }
            let value = y.parse()?;
            elems.push_value(value);
        }
        Ok(Expr::Array(Array {
            attrs: Vec::new(),
            bracket,
            elems,
        }))
    } else if y.peek(Token![;]) {
        let semi: Token![;] = y.parse()?;
        let len: Expr = y.parse()?;
        Ok(Expr::Repeat(Repeat {
            attrs: Vec::new(),
            bracket,
            expr: Box::new(first),
            semi,
            len: Box::new(len),
        }))
    } else {
        Err(y.error("expected `,` or `;`"))
    }
}
pub fn expr_early(x: Stream) -> Res<Expr> {
    let mut attrs = x.call(expr_attrs)?;
    let mut y = if x.peek(Token![if]) {
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
        let allow = AllowStruct(true);
        let mut y = unary_expr(x, allow)?;
        attrs.extend(y.replace_attrs(Vec::new()));
        y.replace_attrs(attrs);
        return parse_expr(x, y, allow, Precedence::Any);
    };
    if x.peek(Token![.]) && !x.peek(Token![..]) || x.peek(Token![?]) {
        y = trailer_helper(x, y)?;
        attrs.extend(y.replace_attrs(Vec::new()));
        y.replace_attrs(attrs);
        let allow_struct = AllowStruct(true);
        return parse_expr(x, y, allow_struct, Precedence::Any);
    }
    attrs.extend(y.replace_attrs(Vec::new()));
    y.replace_attrs(attrs);
    Ok(y)
}
fn expr_group(x: Stream) -> Res<Group> {
    let y = parse::parse_group(x)?;
    Ok(Group {
        attrs: Vec::new(),
        group: y.tok,
        expr: y.buf.parse()?,
    })
}
fn expr_paren(x: Stream) -> Res<Paren> {
    let y;
    Ok(Paren {
        attrs: Vec::new(),
        paren: parenthesized!(y in x),
        expr: y.parse()?,
    })
}
fn else_block(x: Stream) -> Res<(Token![else], Box<Expr>)> {
    let else_: Token![else] = x.parse()?;
    let look = x.lookahead1();
    let branch = if x.peek(Token![if]) {
        x.parse().map(Expr::If)?
    } else if x.peek(tok::Brace) {
        Expr::Block(Block {
            attrs: Vec::new(),
            label: None,
            block: x.parse()?,
        })
    } else {
        return Err(look.error());
    };
    Ok((else_, Box::new(branch)))
}
fn expr_unary(x: Stream, attrs: Vec<attr::Attr>, allow: AllowStruct) -> Res<Unary> {
    Ok(Unary {
        attrs,
        op: x.parse()?,
        expr: Box::new(unary_expr(x, allow)?),
    })
}
fn expr_closure(x: Stream, allow: AllowStruct) -> Res<Closure> {
    let lifes: Option<Bgen::bound::Lifes> = x.parse()?;
    let const_: Option<Token![const]> = x.parse()?;
    let static_: Option<Token![static]> = x.parse()?;
    let async_: Option<Token![async]> = x.parse()?;
    let move_: Option<Token![move]> = x.parse()?;
    let or1: Token![|] = x.parse()?;
    let mut ins = Punctuated::new();
    loop {
        if x.peek(Token![|]) {
            break;
        }
        let value = closure_arg(x)?;
        ins.push_value(value);
        if x.peek(Token![|]) {
            break;
        }
        let punct: Token![,] = x.parse()?;
        ins.push_punct(punct);
    }
    let or2: Token![|] = x.parse()?;
    let (ret, body) = if x.peek(Token![->]) {
        let arrow: Token![->] = x.parse()?;
        let typ: typ::Type = x.parse()?;
        let body: Block = x.parse()?;
        let ret = typ::Ret::Type(arrow, Box::new(typ));
        let body = Expr::Block(Block {
            attrs: Vec::new(),
            label: None,
            block: body,
        });
        (ret, body)
    } else {
        let body = ambiguous_expr(x, allow)?;
        (typ::Ret::Default, body)
    };
    Ok(Closure {
        attrs: Vec::new(),
        lifes,
        const_,
        static_,
        async_,
        move_,
        or1,
        ins,
        or2,
        ret,
        body: Box::new(body),
    })
}
fn closure_arg(x: Stream) -> Res<pat::Pat> {
    let attrs = x.call(attr::Attr::parse_outer)?;
    let mut y = pat::Pat::parse_single(x)?;
    if x.peek(Token![:]) {
        Ok(pat::Pat::Type(pat::Type {
            attrs,
            pat: Box::new(y),
            colon: x.parse()?,
            typ: x.parse()?,
        }))
    } else {
        match &mut y {
            pat::Pat::Const(x) => x.attrs = attrs,
            pat::Pat::Ident(x) => x.attrs = attrs,
            pat::Pat::Lit(x) => x.attrs = attrs,
            pat::Pat::Macro(x) => x.attrs = attrs,
            pat::Pat::Or(x) => x.attrs = attrs,
            pat::Pat::Paren(x) => x.attrs = attrs,
            pat::Pat::Path(x) => x.attrs = attrs,
            pat::Pat::Range(x) => x.attrs = attrs,
            pat::Pat::Reference(x) => x.attrs = attrs,
            pat::Pat::Rest(x) => x.attrs = attrs,
            pat::Pat::Slice(x) => x.attrs = attrs,
            pat::Pat::Struct(x) => x.attrs = attrs,
            pat::Pat::Tuple(x) => x.attrs = attrs,
            pat::Pat::TupleStruct(x) => x.attrs = attrs,
            pat::Pat::Type(_) => unreachable!(),
            pat::Pat::Verbatim(_) => {},
            pat::Pat::Wild(x) => x.attrs = attrs,
        }
        Ok(y)
    }
}
fn expr_break(x: Stream, allow: AllowStruct) -> Res<Break> {
    Ok(Break {
        attrs: Vec::new(),
        break_: x.parse()?,
        label: x.parse()?,
        expr: {
            if x.is_empty() || x.peek(Token![,]) || x.peek(Token![;]) || !allow.0 && x.peek(tok::Brace) {
                None
            } else {
                let x = ambiguous_expr(x, allow)?;
                Some(Box::new(x))
            }
        },
    })
}
fn expr_ret(x: Stream, allow: AllowStruct) -> Res<Return> {
    Ok(Return {
        attrs: Vec::new(),
        return_: x.parse()?,
        expr: {
            if x.is_empty() || x.peek(Token![,]) || x.peek(Token![;]) {
                None
            } else {
                let y = ambiguous_expr(x, allow)?;
                Some(Box::new(y))
            }
        },
    })
}
fn expr_struct_helper(x: Stream, qself: Option<QSelf>, path: Path) -> Res<Struct> {
    let y;
    let brace = braced!(y in x);
    let mut fields = Punctuated::new();
    while !y.is_empty() {
        if y.peek(Token![..]) {
            return Ok(Struct {
                attrs: Vec::new(),
                qself,
                path,
                brace,
                fields,
                dot2: Some(y.parse()?),
                rest: if y.is_empty() { None } else { Some(Box::new(y.parse()?)) },
            });
        }
        fields.push(y.parse()?);
        if y.is_empty() {
            break;
        }
        let punct: Token![,] = y.parse()?;
        fields.push_punct(punct);
    }
    Ok(Struct {
        attrs: Vec::new(),
        qself,
        path,
        brace,
        fields,
        dot2: None,
        rest: None,
    })
}
fn expr_range(x: Stream, allow: AllowStruct) -> Res<Range> {
    let limits: RangeLimits = x.parse()?;
    let end = if matches!(limits, RangeLimits::HalfOpen(_))
        && (x.is_empty()
            || x.peek(Token![,])
            || x.peek(Token![;])
            || x.peek(Token![.]) && !x.peek(Token![..])
            || !allow.0 && x.peek(tok::Brace))
    {
        None
    } else {
        let to = ambiguous_expr(x, allow)?;
        Some(Box::new(to))
    };
    Ok(Range {
        attrs: Vec::new(),
        beg: None,
        limits,
        end,
    })
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
        *e = Expr::Field(Field {
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
fn check_cast(x: Stream) -> Res<()> {
    let kind = if x.peek(Token![.]) && !x.peek(Token![..]) {
        if x.peek2(Token![await]) {
            "`.await`"
        } else if x.peek2(Ident) && (x.peek3(tok::Paren) || x.peek3(Token![::])) {
            "a method call"
        } else {
            "a field access"
        }
    } else if x.peek(Token![?]) {
        "`?`"
    } else if x.peek(tok::Bracket) {
        "indexing"
    } else if x.peek(tok::Paren) {
        "a function call"
    } else {
        return Ok(());
    };
    let msg = format!("casts cannot be followed by {}", kind);
    Err(x.error(msg))
}

fn parse_binop(x: Stream) -> Res<BinOp> {
    if x.peek(Token![&&]) {
        x.parse().map(BinOp::And)
    } else if x.peek(Token![||]) {
        x.parse().map(BinOp::Or)
    } else if x.peek(Token![<<]) {
        x.parse().map(BinOp::Shl)
    } else if x.peek(Token![>>]) {
        x.parse().map(BinOp::Shr)
    } else if x.peek(Token![==]) {
        x.parse().map(BinOp::Eq)
    } else if x.peek(Token![<=]) {
        x.parse().map(BinOp::Le)
    } else if x.peek(Token![!=]) {
        x.parse().map(BinOp::Ne)
    } else if x.peek(Token![>=]) {
        x.parse().map(BinOp::Ge)
    } else if x.peek(Token![+]) {
        x.parse().map(BinOp::Add)
    } else if x.peek(Token![-]) {
        x.parse().map(BinOp::Sub)
    } else if x.peek(Token![*]) {
        x.parse().map(BinOp::Mul)
    } else if x.peek(Token![/]) {
        x.parse().map(BinOp::Div)
    } else if x.peek(Token![%]) {
        x.parse().map(BinOp::Rem)
    } else if x.peek(Token![^]) {
        x.parse().map(BinOp::BitXor)
    } else if x.peek(Token![&]) {
        x.parse().map(BinOp::BitAnd)
    } else if x.peek(Token![|]) {
        x.parse().map(BinOp::BitOr)
    } else if x.peek(Token![<]) {
        x.parse().map(BinOp::Lt)
    } else if x.peek(Token![>]) {
        x.parse().map(BinOp::Gt)
    } else {
        Err(x.error("expected binary operator"))
    }
}
