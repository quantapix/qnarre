use super::*;
use quote::Fragment;

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
        While(While),
        Yield(Yield),
        Verbatim(pm2::Stream),
    }
}
impl Expr {
    pub const DUMMY: Self = Expr::Path(Path {
        attrs: Vec::new(),
        qself: None,
        path: path::Path {
            colon: None,
            segs: Puncted::new(),
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
    pub fn is_blocklike(&self) -> bool {
        use Expr::*;
        match self {
            Array(Array { attrs, .. })
            | Async(Async { attrs, .. })
            | Block(Block { attrs, .. })
            | Closure(Closure { attrs, .. })
            | Const(Const { attrs, .. })
            | Struct(Struct { attrs, .. })
            | TryBlock(TryBlock { attrs, .. })
            | Tuple(Tuple { attrs, .. })
            | Unsafe(Unsafe { attrs, .. }) => !attr::has_outer(attrs),
            Assign(_) | Await(_) | Binary(_) | Break(_) | Call(_) | Cast(_) | Continue(_) | Field(_) | ForLoop(_)
            | Group(_) | If(_) | Index(_) | Infer(_) | Let(_) | Lit(_) | Loop(_) | Mac(_) | Match(_)
            | MethodCall(_) | Paren(_) | Path(_) | Range(_) | Ref(_) | Repeat(_) | Return(_) | Try(_) | Unary(_)
            | Verbatim(_) | While(_) | Yield(_) => false,
        }
    }
    pub fn needs_term(&self) -> bool {
        use Expr::*;
        match self {
            If(_) | Match(_) | Block(_) | Unsafe(_) | While(_) | Loop(_) | ForLoop(_) | TryBlock(_) | Const(_) => false,
            Array(_) | Assign(_) | Async(_) | Await(_) | Binary(_) | Break(_) | Call(_) | Cast(_) | Closure(_)
            | Continue(_) | Field(_) | Group(_) | Index(_) | Infer(_) | Let(_) | Lit(_) | Mac(_) | MethodCall(_)
            | Paren(_) | Path(_) | Range(_) | Ref(_) | Repeat(_) | Return(_) | Struct(_) | Try(_) | Tuple(_)
            | Unary(_) | Yield(_) | Verbatim(_) => true,
        }
    }
    pub fn parseable_as_stmt(&self) -> bool {
        use Expr::*;
        match self {
            Array(_) | Async(_) | Block(_) | Break(_) | Closure(_) | Const(_) | Continue(_) | ForLoop(_) | If(_)
            | Infer(_) | Let(_) | Lit(_) | Loop(_) | Mac(_) | Match(_) | Paren(_) | Path(_) | Ref(_) | Repeat(_)
            | Return(_) | Struct(_) | TryBlock(_) | Tuple(_) | Unary(_) | Unsafe(_) | Verbatim(_) | While(_)
            | Yield(_) => true,
            Assign(x) => &x.left.parseable_as_stmt(),
            Await(x) => &x.expr.parseable_as_stmt(),
            Binary(x) => &x.left.needs_term() && &x.left.parseable_as_stmt(),
            Call(x) => &x.func.needs_term() && &x.func.parseable_as_stmt(),
            Cast(x) => &x.expr.needs_term() && &x.expr.parseable_as_stmt(),
            Field(x) => &x.expr.parseable_as_stmt(),
            Group(x) => &x.expr.parseable_as_stmt(),
            Index(x) => &x.expr.needs_term() && &x.expr.parseable_as_stmt(),
            MethodCall(x) => &x.expr.parseable_as_stmt(),
            Range(x) => match &x.beg {
                Some(x) => x.needs_term() && x.parseable_as_stmt(),
                None => true,
            },
            Try(x) => &x.expr.parseable_as_stmt(),
        }
    }
    fn has_struct_lit(&self) -> bool {
        use Expr::*;
        match self {
            Struct(_) => true,
            Assign(Assign { left, right, .. }) | Binary(Binary { left, right, .. }) => {
                left.has_struct_lit() || right.has_struct_lit()
            },
            Await(Await { expr, .. })
            | Cast(Cast { expr, .. })
            | Field(Field { expr, .. })
            | Index(Index { expr, .. })
            | MethodCall(MethodCall { expr, .. })
            | Ref(Ref { expr, .. })
            | Unary(Unary { expr, .. }) => expr.has_struct_lit(),
            Array(_) | Async(_) | Block(_) | Break(_) | Call(_) | Closure(_) | Const(_) | Continue(_) | ForLoop(_)
            | Group(_) | If(_) | Infer(_) | Let(_) | Lit(_) | Loop(_) | Mac(_) | Match(_) | Paren(_) | Path(_)
            | Range(_) | Repeat(_) | Return(_) | Try(_) | TryBlock(_) | Tuple(_) | Unsafe(_) | &Verbatim(_)
            | While(_) | Yield(_) => false,
        }
    }
    fn is_short_ident(&self) -> bool {
        if let Expr::Path(x) = self {
            return x.attrs.is_empty()
                && x.qself.is_none()
                && x.path
                    .get_ident()
                    .map_or(false, |x| x.to_string().len() as isize <= INDENT);
        }
        false
    }
    fn needs_newline(&self) -> bool {
        use Expr::*;
        match self {
            Array(_)
            | Async(_)
            | Block(_)
            | Break(Break { val: None, .. })
            | Closure(_)
            | Const(_)
            | Continue(_)
            | ForLoop(_)
            | If(_)
            | Infer(_)
            | Lit(_)
            | Loop(_)
            | Mac(_)
            | Match(_)
            | Path(_)
            | Range(Range { end: None, .. })
            | Repeat(_)
            | Return(Return { expr: None, .. })
            | Struct(_)
            | TryBlock(_)
            | Tuple(_)
            | Unsafe(_)
            | Verbatim(_)
            | While(_)
            | Yield(Yield { expr: None, .. }) => false,
            Assign(_) | Await(_) | Binary(_) | Cast(_) | Field(_) | Index(_) | MethodCall(_) => true,
            Break(Break { val: Some(x), .. })
            | Call(Call { func: x, .. })
            | Group(Group { expr: x, .. })
            | Let(Let { expr: x, .. })
            | Paren(Paren { expr: x, .. })
            | Range(Range { end: Some(x), .. })
            | Ref(Ref { expr: x, .. })
            | Return(Return { expr: Some(x), .. })
            | Try(Try { expr: x, .. })
            | Unary(Unary { expr: x, .. })
            | Yield(Yield { expr: Some(x), .. }) => x.needs_newline(),
        }
    }
    pub fn add_semi(&self) -> bool {
        use BinOp::*;
        use Expr::*;
        match self {
            Assign(_) | Break(_) | Continue(_) | Return(_) | Yield(_) => true,
            Binary(x) => match x.op {
                AddAssign(_) | SubAssign(_) | MulAssign(_) | DivAssign(_) | RemAssign(_) | BitXorAssign(_)
                | BitAndAssign(_) | BitOrAssign(_) | ShlAssign(_) | ShrAssign(_) => true,
                Add(_) | Sub(_) | Mul(_) | Div(_) | Rem(_) | And(_) | Or(_) | BitXor(_) | BitAnd(_) | BitOr(_)
                | Shl(_) | Shr(_) | Eq(_) | Lt(_) | Le(_) | Ne(_) | Ge(_) | Gt(_) => false,
            },
            Group(x) => &x.expr.add_semi(),
            Array(_) | Async(_) | Await(_) | Block(_) | Call(_) | Cast(_) | Closure(_) | Const(_) | Field(_)
            | ForLoop(_) | If(_) | Index(_) | Infer(_) | Let(_) | Lit(_) | Loop(_) | Mac(_) | Match(_)
            | MethodCall(_) | Paren(_) | Path(_) | Range(_) | Ref(_) | Repeat(_) | Struct(_) | Try(_) | TryBlock(_)
            | Tuple(_) | Unary(_) | Unsafe(_) | Verbatim(_) | While(_) => false,
        }
    }
    pub fn remove_semi(&self) -> bool {
        use Expr::*;
        match self {
            ForLoop(_) | While(_) => true,
            Group(x) => &x.expr.remove_semi(),
            If(x) => match &x.else_ {
                Some((_, x)) => x.remove_semi(),
                None => true,
            },
            Array(_) | Assign(_) | Async(_) | Await(_) | Binary(_) | Block(_) | Break(_) | Call(_) | Cast(_)
            | Closure(_) | Continue(_) | Const(_) | Field(_) | Index(_) | Infer(_) | Let(_) | Lit(_) | Loop(_)
            | Mac(_) | Match(_) | MethodCall(_) | Paren(_) | Path(_) | Range(_) | Ref(_) | Repeat(_) | Return(_)
            | Struct(_) | Try(_) | TryBlock(_) | Tuple(_) | Unary(_) | Unsafe(_) | Verbatim(_) | Yield(_) => false,
        }
    }
    pub fn break_after(&self) -> bool {
        if let Expr::Group(x) = self {
            if let Expr::Verbatim(x) = x.expr.as_ref() {
                return !x.is_empty();
            }
        }
        true
    }
    fn lower_struct(&self, s: &mut Stream) {
        if let Expr::Struct(_) = *self {
            tok::Paren::default().surround(s, |s| {
                self.lower(s);
            });
        } else {
            self.lower(s);
        }
    }
    fn pretty_struct(&self, p: &mut Print) {
        let paren = self.has_struct_lit();
        if paren {
            p.word("(");
        }
        p.cbox(0);
        self.pretty(p);
        if paren {
            p.word(")");
        }
        if self.needs_newline() {
            p.space();
        } else {
            p.nbsp();
        }
        p.end();
    }
    fn pretty_sub(&self, p: &mut Print, beg_of_line: bool) {
        use Expr::*;
        match self {
            Await(x) => x.pretty_sub(p, beg_of_line),
            Call(x) => x.pretty_sub(p),
            Field(x) => x.pretty_sub(p, beg_of_line),
            Index(x) => x.pretty_sub(p, beg_of_line),
            MethodCall(x) => x.pretty_sub(p, beg_of_line, false),
            Try(x) => x.pretty_sub(p, beg_of_line),
            _ => {
                p.cbox(-INDENT);
                self.pretty(p);
                p.end();
            },
        }
    }
    pub fn pretty_beg_of_line(&self, p: &mut Print, beg_of_line: bool) {
        use Expr::*;
        match self {
            Await(x) => p.expr_await(x, beg_of_line),
            Field(x) => p.expr_field(x, beg_of_line),
            Index(x) => p.expr_index(x, beg_of_line),
            MethodCall(x) => p.expr_method_call(x, beg_of_line),
            Try(x) => p.expr_try(x, beg_of_line),
            _ => p.expr(self),
        }
    }
    fn pretty_zerobreak(&self, p: &mut Print, beg_of_line: bool) {
        if beg_of_line && self.is_short_ident() {
            return;
        }
        p.zerobreak();
    }
}
impl Parse for Expr {
    fn parse(x: Stream) -> Res<Self> {
        ambiguous_expr(x, AllowStruct(true))
    }
}
impl Pretty for Expr {
    fn pretty(&self, p: &mut Print) {
        use Expr::*;
        match self {
            Array(x) => x.pretty(p),
            Assign(x) => x.pretty(p),
            Async(x) => x.pretty(p),
            Await(x) => x.pretty(p, false),
            Binary(x) => x.pretty(p),
            Block(x) => x.pretty(p),
            Break(x) => x.pretty(p),
            Call(x) => x.pretty(p, false),
            Cast(x) => x.pretty(p),
            Closure(x) => x.pretty(p),
            Const(x) => x.pretty(p),
            Continue(x) => x.pretty(p),
            Field(x) => x.pretty(p, false),
            ForLoop(x) => x.pretty(p),
            Group(x) => x.pretty(p),
            If(x) => x.pretty(p),
            Index(x) => x.pretty(p, false),
            Infer(x) => x.pretty(p),
            Let(x) => x.pretty(p),
            Lit(x) => x.pretty(p),
            Loop(x) => x.pretty(p),
            Mac(x) => x.pretty(p),
            Match(x) => x.pretty(p),
            MethodCall(x) => x.pretty(p, false),
            Paren(x) => x.pretty(p),
            Path(x) => x.pretty(p),
            Range(x) => x.pretty(p),
            Ref(x) => x.pretty(p),
            Repeat(x) => x.pretty(p),
            Return(x) => x.pretty(p),
            Struct(x) => x.pretty(p),
            Try(x) => x.pretty(p),
            TryBlock(x) => x.pretty(p),
            Tuple(x) => x.pretty(p),
            Unary(x) => x.pretty(p),
            Unsafe(x) => x.pretty(p),
            Verbatim(x) => x.pretty(p),
            While(x) => x.pretty(p),
            Yield(x) => x.pretty(p),
        }
    }
}

macro_rules! impl_by_parsing_expr {
    (
        $(
            $n:ty, $v:ident, $m:expr,
        )*
    ) => {
        $(
            impl Parse for $n {
                fn parse(x: Stream) -> Res<Self> {
                    let mut y: Expr = x.parse()?;
                    loop {
                        match y {
                            Expr::$v(x) => return Ok(x),
                            Expr::Group(x) => y = *x.expr,
                            _ => return Err(Err::new_spanned(y, $m)),
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
    pub elems: Puncted<Expr, Token![,]>,
}
impl Parse for Array {
    fn parse(x: Stream) -> Res<Self> {
        let y;
        let bracket = bracketed!(y in x);
        let mut elems = Puncted::new();
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
impl Lower for Array {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.bracket.surround(s, |s| {
            self.elems.lower(s);
        });
    }
}
impl Pretty for Array {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("[");
        p.cbox(INDENT);
        p.zerobreak();
        for x in self.elems.iter().delimited() {
            &x.pretty(p);
            p.trailing_comma(x.is_last);
        }
        p.offset(-INDENT);
        p.end();
        p.word("]");
    }
}

pub struct Assign {
    pub attrs: Vec<attr::Attr>,
    pub left: Box<Expr>,
    pub eq: Token![=],
    pub right: Box<Expr>,
}
impl Lower for Assign {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.left.lower(s);
        self.eq.lower(s);
        self.right.lower(s);
    }
}
impl Pretty for Assign {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.ibox(0);
        &self.left.pretty(p);
        p.word(" = ");
        &self.right.pretty(p);
        p.end();
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
impl Lower for Async {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.async_.lower(s);
        self.move_.lower(s);
        self.block.lower(s);
    }
}
impl Pretty for Async {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("async ");
        if self.move_.is_some() {
            p.word("move ");
        }
        p.cbox(INDENT);
        p.small_block(&self.block, &self.attrs);
        p.end();
    }
}

pub struct Await {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub dot: Token![.],
    pub await_: Token![await],
}
impl Await {
    fn pretty_sub(&self, p: &mut Print, beg_of_line: bool) {
        &self.expr.pretty_sub(p, beg_of_line);
        &self.expr.pretty_zerobreak(p, beg_of_line);
        p.word(".await");
    }
}
impl Lower for Await {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.expr.lower(s);
        self.dot.lower(s);
        self.await_.lower(s);
    }
}
impl Pretty for Await {
    fn pretty(&self, p: &mut Print, beg_of_line: bool) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        self.pretty_sub(p, beg_of_line);
        p.end();
    }
}

pub struct Binary {
    pub attrs: Vec<attr::Attr>,
    pub left: Box<Expr>,
    pub op: BinOp,
    pub right: Box<Expr>,
}
impl Lower for Binary {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.left.lower(s);
        self.op.lower(s);
        self.right.lower(s);
    }
}
impl Pretty for Binary {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.ibox(INDENT);
        p.ibox(-INDENT);
        &self.left.pretty(p);
        p.end();
        p.space();
        &self.op.pretty(p);
        p.nbsp();
        &self.right.pretty(p);
        p.end();
    }
}

pub struct Block {
    pub attrs: Vec<attr::Attr>,
    pub label: Option<Label>,
    pub block: stmt::Block,
}
impl Parse for Block {
    fn parse(x: Stream) -> Res<Self> {
        let mut attrs = x.call(attr::Attr::parse_outers)?;
        let label: Option<Label> = x.parse()?;
        let y;
        let brace = braced!(y in x);
        attr::parse_inners(&y, &mut attrs)?;
        let stmts = y.call(stmt::Block::parse_within)?;
        Ok(Block {
            attrs,
            label,
            block: stmt::Block { brace, stmts },
        })
    }
}
impl Lower for Block {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.label.lower(s);
        self.block.brace.surround(s, |s| {
            attr::lower_inners(&self.attrs, s);
            s.append_all(&self.block.stmts);
        });
    }
}
impl Pretty for Block {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        if let Some(x) = &self.label {
            p.label(x);
        }
        p.cbox(INDENT);
        p.small_block(&self.block, &self.attrs);
        p.end();
    }
}

pub struct Break {
    pub attrs: Vec<attr::Attr>,
    pub break_: Token![break],
    pub life: Option<Life>,
    pub val: Option<Box<Expr>>,
}
impl Parse for Break {
    fn parse(x: Stream) -> Res<Self> {
        let allow = AllowStruct(true);
        expr_break(x, allow)
    }
}
impl Lower for Break {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.break_.lower(s);
        self.life.lower(s);
        self.val.lower(s);
    }
}
impl Pretty for Break {
    fn pretty(&self, p: &mut Break) {
        p.outer_attrs(&self.attrs);
        p.word("break");
        if let Some(x) = &self.life {
            p.nbsp();
            p.lifetime(x);
        }
        if let Some(x) = &self.val {
            p.nbsp();
            p.value(x);
        }
    }
}

pub struct Call {
    pub attrs: Vec<attr::Attr>,
    pub func: Box<Expr>,
    pub paren: tok::Paren,
    pub args: Puncted<Expr, Token![,]>,
}
impl Call {
    fn pretty_sub(&self, p: &mut Print) {
        &self.func.pretty_sub(p, false);
        p.word("(");
        p.call_args(&self.args);
        p.word(")");
    }
}
impl Lower for Call {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.func.lower(s);
        self.paren.surround(s, |s| {
            self.args.lower(s);
        });
    }
}
impl Pretty for Call {
    fn pretty(&self, p: &mut Call, beg_of_line: bool) {
        p.outer_attrs(&self.attrs);
        &self.func.pretty_beg_of_line(p, beg_of_line);
        p.word("(");
        p.call_args(&self.args);
        p.word(")");
    }
}

pub struct Cast {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub as_: Token![as],
    pub typ: Box<typ::Type>,
}
impl Lower for Cast {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.expr.lower(s);
        self.as_.lower(s);
        self.typ.lower(s);
    }
}
impl Pretty for Cast {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.ibox(INDENT);
        p.ibox(-INDENT);
        &self.expr.pretty(p);
        p.end();
        p.space();
        p.word("as ");
        p.ty(&self.typ);
        p.end();
    }
}

pub struct Closure {
    pub attrs: Vec<attr::Attr>,
    pub lifes: Option<gen::bound::Lifes>,
    pub const_: Option<Token![const]>,
    pub static_: Option<Token![static]>,
    pub async_: Option<Token![async]>,
    pub move_: Option<Token![move]>,
    pub or1: Token![|],
    pub ins: Puncted<pat::Pat, Token![,]>,
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
impl Lower for Closure {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.lifes.lower(s);
        self.const_.lower(s);
        self.static_.lower(s);
        self.async_.lower(s);
        self.move_.lower(s);
        self.or1.lower(s);
        self.ins.lower(s);
        self.or2.lower(s);
        self.ret.lower(s);
        self.body.lower(s);
    }
}
impl Pretty for Closure {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.ibox(0);
        if let Some(x) = &self.lifes {
            p.bound_lifetimes(x);
        }
        if self.const_.is_some() {
            p.word("const ");
        }
        if self.static_.is_some() {
            p.word("static ");
        }
        if self.async_.is_some() {
            p.word("async ");
        }
        if self.move_.is_some() {
            p.word("move ");
        }
        p.cbox(INDENT);
        p.word("|");
        for x in self.ins.iter().delimited() {
            if x.is_first {
                p.zerobreak();
            }
            p.pat(&x);
            if !x.is_last {
                p.word(",");
                p.space();
            }
        }
        match &self.ret {
            typ::Ret::Default => {
                p.word("|");
                p.space();
                p.offset(-INDENT);
                p.end();
                p.neverbreak();
                let wrap_in_brace = match &*self.body {
                    Expr::Match(Match { attrs, .. }) | Expr::Call(Call { attrs, .. }) => attr::has_outer(attrs),
                    x => !x.is_blocklike(),
                };
                if wrap_in_brace {
                    p.cbox(INDENT);
                    let okay_to_brace = &self.body.parseable_as_stmt();
                    p.scan_break(BreakToken {
                        pre_break: Some(if okay_to_brace { '{' } else { '(' }),
                        ..BreakToken::default()
                    });
                    &self.body.pretty(p);
                    p.scan_break(BreakToken {
                        offset: -INDENT,
                        pre_break: (okay_to_brace && &self.body.add_semi()).then(|| ';'),
                        post_break: Some(if okay_to_brace { '}' } else { ')' }),
                        ..BreakToken::default()
                    });
                    p.end();
                } else {
                    &self.body.pretty(p);
                }
            },
            typ::Ret::Type(_, x) => {
                if !self.ins.is_empty() {
                    p.trailing_comma(true);
                    p.offset(-INDENT);
                }
                p.word("|");
                p.end();
                p.word(" -> ");
                p.ty(x);
                p.nbsp();
                p.neverbreak();
                &self.body.pretty(p);
            },
        }
        p.end();
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
        let attrs = y.call(attr::Attr::parse_inners)?;
        let stmts = y.call(stmt::Block::parse_within)?;
        Ok(Const {
            attrs,
            const_,
            block: stmt::Block { brace, stmts },
        })
    }
}
impl Lower for Const {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.const_.lower(s);
        self.block.brace.surround(s, |s| {
            attr::lower_inners(&self.attrs, s);
            s.append_all(&self.block.stmts);
        });
    }
}
impl Pretty for Const {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("const ");
        p.cbox(INDENT);
        p.small_block(&self.block, &self.attrs);
        p.end();
    }
}

pub struct Continue {
    pub attrs: Vec<attr::Attr>,
    pub continue_: Token![continue],
    pub life: Option<Life>,
}
impl Parse for Continue {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Continue {
            attrs: Vec::new(),
            continue_: x.parse()?,
            life: x.parse()?,
        })
    }
}
impl Lower for Continue {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.continue_.lower(s);
        self.life.lower(s);
    }
}
impl Pretty for Continue {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("continue");
        if let Some(x) = &self.life {
            p.nbsp();
            p.lifetime(x);
        }
    }
}

pub struct Field {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub dot: Token![.],
    pub memb: Member,
}
impl Field {
    fn pretty_sub(&self, p: &mut Print, beg_of_line: bool) {
        &self.expr.pretty_sub(p, beg_of_line);
        &self.expr.pretty_zerobreak(p, beg_of_line);
        p.word(".");
        &self.memb.pretty(p);
    }
}
impl Lower for Field {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.expr.lower(s);
        self.dot.lower(s);
        self.memb.lower(s);
    }
}
impl Pretty for Field {
    fn pretty(&self, p: &mut Print, beg_of_line: bool) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        self.pretty_sub(p, beg_of_line);
        p.end();
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
        let mut attrs = x.call(attr::Attr::parse_outers)?;
        let label: Option<Label> = x.parse()?;
        let for_: Token![for] = x.parse()?;
        let pat = pat::Pat::parse_many(x)?;
        let in_: Token![in] = x.parse()?;
        let expr: Expr = x.call(Expr::parse_without_eager_brace)?;
        let y;
        let brace = braced!(y in x);
        attr::parse_inners(&y, &mut attrs)?;
        let stmts = y.call(stmt::Block::parse_within)?;
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
impl Lower for ForLoop {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.label.lower(s);
        self.for_.lower(s);
        self.pat.lower(s);
        self.in_.lower(s);
        &self.expr.lower_struct(s);
        self.body.brace.surround(s, |s| {
            attr::lower_inners(&self.attrs, s);
            s.append_all(&self.body.stmts);
        });
    }
}
impl Pretty for ForLoop {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.ibox(0);
        if let Some(x) = &self.label {
            p.label(x);
        }
        p.word("for ");
        p.pat(&self.pat);
        p.word(" in ");
        p.neverbreak();
        &self.expr.pretty_struct(p);
        p.word("{");
        p.neverbreak();
        p.cbox(INDENT);
        p.hardbreak_if_nonempty();
        p.inner_attrs(&self.attrs);
        for x in &self.body.stmts {
            p.stmt(x);
        }
        p.offset(-INDENT);
        p.end();
        p.word("}");
        p.end();
    }
}

pub struct Group {
    pub attrs: Vec<attr::Attr>,
    pub group: tok::Group,
    pub expr: Box<Expr>,
}
impl Lower for Group {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.group.surround(s, |s| {
            self.expr.lower(s);
        });
    }
}
impl Pretty for Group {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        &self.expr.pretty(p);
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
        let attrs = x.call(attr::Attr::parse_outers)?;
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
impl Lower for If {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.if_.lower(s);
        &self.cond.lower_struct(s);
        self.then_.lower(s);
        if let Some((else_, x)) = &self.else_ {
            else_.lower(s);
            match **x {
                Expr::If(_) | Expr::Block(_) => x.lower(s),
                _ => tok::Brace::default().surround(s, |s| x.lower(s)),
            }
        }
    }
}
impl Pretty for If {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        p.word("if ");
        p.cbox(-INDENT);
        &self.cond.pretty_struct(p);
        p.end();
        if let Some((_, else_)) = &self.else_ {
            let mut else_ = &**else_;
            p.small_block(&self.then_, &[]);
            loop {
                p.word(" else ");
                match else_ {
                    Expr::If(x) => {
                        p.word("if ");
                        p.cbox(-INDENT);
                        &x.cond.pretty_struct(p);
                        p.end();
                        p.small_block(&x.then_, &[]);
                        if let Some((_, x)) = &x.else_ {
                            else_ = x;
                            continue;
                        }
                    },
                    Expr::Block(x) => {
                        p.small_block(&x.block, &[]);
                    },
                    x => {
                        p.word("{");
                        p.space();
                        p.ibox(INDENT);
                        x.pretty(p);
                        p.end();
                        p.space();
                        p.offset(-INDENT);
                        p.word("}");
                    },
                }
                break;
            }
        } else if self.then_.stmts.is_empty() {
            p.word("{}");
        } else {
            p.word("{");
            p.hardbreak();
            for x in &self.then_.stmts {
                p.stmt(x);
            }
            p.offset(-INDENT);
            p.word("}");
        }
        p.end();
    }
}

pub struct Index {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub bracket: tok::Bracket,
    pub idx: Box<Expr>,
}
impl Index {
    fn pretty_sub(&self, p: &mut Print, beg_of_line: bool) {
        &self.expr.pretty_sub(p, beg_of_line);
        p.word("[");
        &self.idx.pretty(p);
        p.word("]");
    }
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
impl Lower for Index {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.expr.lower(s);
        self.bracket.surround(s, |s| {
            self.idx.lower(s);
        });
    }
}
impl Pretty for Index {
    fn pretty(&self, p: &mut Print, beg_of_line: bool) {
        p.outer_attrs(&self.attrs);
        &self.expr.pretty_beg_of_line(p, beg_of_line);
        p.word("[");
        &self.idx.pretty(p);
        p.word("]");
    }
}

pub struct Infer {
    pub attrs: Vec<attr::Attr>,
    pub underscore: Token![_],
}
impl Parse for Infer {
    fn parse(x: Stream) -> Res<Self> {
        Ok(Infer {
            attrs: x.call(attr::Attr::parse_outers)?,
            underscore: x.parse()?,
        })
    }
}
impl Lower for Infer {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.underscore.lower(s);
    }
}
impl Pretty for Infer {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("_");
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
            pat: Box::new(pat::Pat::parse_many(x)?),
            eq: x.parse()?,
            expr: Box::new({
                let allow = AllowStruct(false);
                let lhs = unary_expr(x, allow)?;
                parse_expr(x, lhs, allow, Precedence::Compare)?
            }),
        })
    }
}
impl Lower for Let {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.let_.lower(s);
        self.pat.lower(s);
        self.eq.lower(s);
        &self.expr.lower_struct(s);
    }
}
impl Pretty for Let {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.ibox(INDENT);
        p.word("let ");
        p.ibox(-INDENT);
        &self.pat.pretty(p);
        p.end();
        p.space();
        p.word("= ");
        let paren = &self.expr.has_struct_lit();
        if paren {
            p.word("(");
        }
        &self.expr.pretty(p);
        if paren {
            p.word(")");
        }
        p.end();
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
impl Lower for Lit {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.lit.lower(s);
    }
}
impl Pretty for Lit {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.lit(&self.lit);
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
        let mut attrs = x.call(attr::Attr::parse_outers)?;
        let label: Option<Label> = x.parse()?;
        let loop_: Token![loop] = x.parse()?;
        let y;
        let brace = braced!(y in x);
        attr::parse_inners(&y, &mut attrs)?;
        let stmts = y.call(stmt::Block::parse_within)?;
        Ok(Loop {
            attrs,
            label,
            loop_,
            body: stmt::Block { brace, stmts },
        })
    }
}
impl Lower for Loop {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.label.lower(s);
        self.loop_.lower(s);
        self.body.brace.surround(s, |s| {
            attr::lower_inners(&self.attrs, s);
            s.append_all(&self.body.stmts);
        });
    }
}
impl Pretty for Loop {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        if let Some(x) = &self.label {
            p.label(x);
        }
        p.word("loop {");
        p.cbox(INDENT);
        p.hardbreak_if_nonempty();
        p.inner_attrs(&self.attrs);
        for x in &self.body.stmts {
            p.stmt(x);
        }
        p.offset(-INDENT);
        p.end();
        p.word("}");
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
impl Lower for Mac {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.mac.lower(s);
    }
}
impl Pretty for Mac {
    fn pretty(&self, p: &mut Mac) {
        p.outer_attrs(&self.attrs);
        let semicolon = false;
        p.mac(&self.mac, None, semicolon);
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
        let mut attrs = x.call(attr::Attr::parse_outers)?;
        let match_: Token![match] = x.parse()?;
        let expr = Expr::parse_without_eager_brace(x)?;
        let y;
        let brace = braced!(y in x);
        attr::parse_inners(&y, &mut attrs)?;
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
impl Lower for Match {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.match_.lower(s);
        &self.expr.lower_struct(s);
        self.brace.surround(s, |s| {
            attr::lower_inners(&self.attrs, s);
            for (i, arm) in self.arms.iter().enumerate() {
                arm.lower(s);
                let is_last = i == self.arms.len() - 1;
                if !is_last && &arm.body.needs_term() && arm.comma.is_none() {
                    <Token![,]>::default().lower(s);
                }
            }
        });
    }
}
impl Pretty for Match {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.ibox(0);
        p.word("match ");
        &self.expr.pretty_struct(p);
        p.word("{");
        p.neverbreak();
        p.cbox(INDENT);
        p.hardbreak_if_nonempty();
        p.inner_attrs(&self.attrs);
        for x in &self.arms {
            p.arm(x);
            p.hardbreak();
        }
        p.offset(-INDENT);
        p.end();
        p.word("}");
        p.end();
    }
}

pub struct MethodCall {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub dot: Token![.],
    pub method: Ident,
    pub turbofish: Option<path::Angled>,
    pub paren: tok::Paren,
    pub args: Puncted<Expr, Token![,]>,
}
impl MethodCall {
    fn pretty_sub(&self, p: &mut Print, beg_of_line: bool, unindent: bool) {
        &self.expr.pretty_sub(p, beg_of_line);
        &self.expr.pretty_zerobreak(p, beg_of_line);
        p.word(".");
        &self.method.pretty(p);
        if let Some(x) = &self.turbofish {
            x.pretty(p, path::Kind::Expr);
        }
        p.cbox(if unindent { -INDENT } else { 0 });
        p.word("(");
        p.call_args(&self.args);
        p.word(")");
        p.end();
    }
}
impl Lower for MethodCall {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.expr.lower(s);
        self.dot.lower(s);
        self.method.lower(s);
        self.turbofish.lower(s);
        self.paren.surround(s, |s| {
            self.args.lower(s);
        });
    }
}
impl Pretty for MethodCall {
    fn pretty(&self, p: &mut Print, beg_of_line: bool) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        let unindent = beg_of_line && &self.expr.is_short_ident();
        self.pretty_sub(p, beg_of_line, unindent);
        p.end();
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
impl Lower for Paren {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.paren.surround(s, |s| {
            self.expr.lower(s);
        });
    }
}
impl Pretty for Paren {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("(");
        &self.expr.pretty(p);
        p.word(")");
    }
}

pub struct Path {
    pub attrs: Vec<attr::Attr>,
    pub qself: Option<path::QSelf>,
    pub path: path::Path,
}
impl Parse for Path {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = x.call(attr::Attr::parse_outers)?;
        let (qself, path) = path::qpath(x, true)?;
        Ok(Path { attrs, qself, path })
    }
}
impl Lower for Path {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        path::path_to_tokens(s, &self.qself, &self.path);
    }
}
impl Pretty for Path {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        &self.path.pretty_qpath(p, &self.qself, path::Kind::Expr);
    }
}

pub struct Range {
    pub attrs: Vec<attr::Attr>,
    pub beg: Option<Box<Expr>>,
    pub limits: Limits,
    pub end: Option<Box<Expr>>,
}
impl Lower for Range {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.beg.lower(s);
        self.limits.lower(s);
        self.end.lower(s);
    }
}
impl Pretty for Range {
    fn pretty(&self, p: &mut Range) {
        p.outer_attrs(&self.attrs);
        if let Some(x) = &self.beg {
            x.pretty(p);
        }
        p.word(match self.limits {
            Limits::HalfOpen(_) => "..",
            Limits::Closed(_) => "..=",
        });
        if let Some(x) = &self.end {
            x.pretty(p);
        }
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
impl Lower for Ref {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.and.lower(s);
        self.mut_.lower(s);
        self.expr.lower(s);
    }
}
impl Pretty for Ref {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("&");
        if self.mut_.is_some() {
            p.word("mut ");
        }
        &self.expr.pretty(p);
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
impl Lower for Repeat {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.bracket.surround(s, |s| {
            self.expr.lower(s);
            self.semi.lower(s);
            self.len.lower(s);
        });
    }
}
impl Pretty for Repeat {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("[");
        &self.expr.pretty(p);
        p.word("; ");
        &self.len.pretty(p);
        p.word("]");
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
impl Lower for Return {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.return_.lower(s);
        self.expr.lower(s);
    }
}
impl Pretty for Return {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("return");
        if let Some(x) = &self.expr {
            p.nbsp();
            x.pretty(p);
        }
    }
}

pub struct Struct {
    pub attrs: Vec<attr::Attr>,
    pub qself: Option<path::QSelf>,
    pub path: path::Path,
    pub brace: tok::Brace,
    pub fields: Puncted<FieldValue, Token![,]>,
    pub dot2: Option<Token![..]>,
    pub rest: Option<Box<Expr>>,
}
impl Parse for Struct {
    fn parse(x: Stream) -> Res<Self> {
        let (qself, path) = path::qpath(x, true)?;
        expr_struct_helper(x, qself, path)
    }
}
impl Lower for Struct {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        path::path_to_tokens(s, &self.qself, &self.path);
        self.brace.surround(s, |s| {
            self.fields.lower(s);
            if let Some(dot2) = &self.dot2 {
                dot2.lower(s);
            } else if self.rest.is_some() {
                Token![..](pm2::Span::call_site()).lower(s);
            }
            self.rest.lower(s);
        });
    }
}
impl Pretty for Struct {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        p.ibox(-INDENT);
        &self.path.pretty_qpath(p, &self.qself, path::Kind::Expr);
        p.end();
        p.word(" {");
        p.space_if_nonempty();
        for x in self.fields.iter().delimited() {
            p.field_value(&x);
            p.trailing_comma_or_space(x.is_last && self.rest.is_none());
        }
        if let Some(x) = &self.rest {
            p.word("..");
            x.pretty(p);
            p.space();
        }
        p.offset(-INDENT);
        p.end_with_max_width(34);
        p.word("}");
    }
}

pub struct Try {
    pub attrs: Vec<attr::Attr>,
    pub expr: Box<Expr>,
    pub question: Token![?],
}
impl Try {
    fn pretty_sub(&self, p: &mut Print, beg_of_line: bool) {
        &self.expr.pretty_sub(p, beg_of_line);
        p.word("?");
    }
}
impl Lower for Try {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.expr.lower(s);
        self.question.lower(s);
    }
}
impl Pretty for Try {
    fn pretty(&self, p: &mut Print, beg_of_line: bool) {
        p.outer_attrs(&self.attrs);
        &self.expr.pretty_beg_of_line(p, beg_of_line);
        p.word("?");
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
impl Lower for TryBlock {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.try_.lower(s);
        self.block.lower(s);
    }
}
impl Pretty for TryBlock {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("try ");
        p.cbox(INDENT);
        p.small_block(&self.block, &self.attrs);
        p.end();
    }
}

pub struct Tuple {
    pub attrs: Vec<attr::Attr>,
    pub paren: tok::Paren,
    pub elems: Puncted<Expr, Token![,]>,
}
impl Lower for Tuple {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.paren.surround(s, |s| {
            self.elems.lower(s);
            if self.elems.len() == 1 && !self.elems.trailing_punct() {
                <Token![,]>::default().lower(s);
            }
        });
    }
}
impl Pretty for Tuple {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("(");
        p.cbox(INDENT);
        p.zerobreak();
        for x in self.elems.iter().delimited() {
            &x.pretty(p);
            if self.elems.len() == 1 {
                p.word(",");
                p.zerobreak();
            } else {
                p.trailing_comma(x.is_last);
            }
        }
        p.offset(-INDENT);
        p.end();
        p.word(")");
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
impl Lower for Unary {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.op.lower(s);
        self.expr.lower(s);
    }
}
impl Pretty for Unary {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.unary_operator(&self.op);
        &self.expr.pretty(p);
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
        let attrs = y.call(attr::Attr::parse_inners)?;
        let stmts = y.call(stmt::Block::parse_within)?;
        Ok(Unsafe {
            attrs,
            unsafe_,
            block: stmt::Block { brace, stmts },
        })
    }
}
impl Lower for Unsafe {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.unsafe_.lower(s);
        self.block.brace.surround(s, |s| {
            attr::lower_inners(&self.attrs, s);
            s.append_all(&self.block.stmts);
        });
    }
}
impl Pretty for Unsafe {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("unsafe ");
        p.cbox(INDENT);
        p.small_block(&self.block, &self.attrs);
        p.end();
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
        let mut attrs = x.call(attr::Attr::parse_outers)?;
        let label: Option<Label> = x.parse()?;
        let while_: Token![while] = x.parse()?;
        let cond = Expr::parse_without_eager_brace(x)?;
        let y;
        let brace = braced!(y in x);
        attr::parse_inners(&y, &mut attrs)?;
        let stmts = y.call(stmt::Block::parse_within)?;
        Ok(While {
            attrs,
            label,
            while_,
            cond: Box::new(cond),
            block: stmt::Block { brace, stmts },
        })
    }
}
impl Lower for While {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.label.lower(s);
        self.while_.lower(s);
        &self.cond.lower_struct(s);
        self.block.brace.surround(s, |s| {
            attr::lower_inners(&self.attrs, s);
            s.append_all(&self.block.stmts);
        });
    }
}
impl Pretty for While {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        if let Some(x) = &self.label {
            p.label(x);
        }
        p.word("while ");
        &self.cond.pretty_struct(p);
        p.word("{");
        p.neverbreak();
        p.cbox(INDENT);
        p.hardbreak_if_nonempty();
        p.inner_attrs(&self.attrs);
        for x in &self.block.stmts {
            p.stmt(x);
        }
        p.offset(-INDENT);
        p.end();
        p.word("}");
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
impl Lower for Yield {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.yield_.lower(s);
        self.expr.lower(s);
    }
}
impl Pretty for Yield {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("yield");
        if let Some(x) = &self.expr {
            p.nbsp();
            x.pretty(p);
        }
    }
}

pub struct Verbatim(pub pm2::Stream);
impl Pretty for Verbatim {
    fn pretty(&self, p: &mut Print) {
        enum Type {
            Empty,
            Ellipsis,
            Builtin(Builtin),
            RawRef(RawRef),
        }
        use Type::*;
        struct Builtin {
            attrs: Vec<attr::Attr>,
            name: Ident,
            args: pm2::Stream,
        }
        struct RawRef {
            attrs: Vec<attr::Attr>,
            mutable: bool,
            expr: Expr,
        }
        mod kw {
            crate::custom_kw!(builtin);
            crate::custom_kw!(raw);
        }
        impl parse::Parse for Type {
            fn parse(s: parse::Stream) -> Res<Self> {
                let ahead = s.fork();
                let attrs = ahead.call(attr::Attr::parse_outer)?;
                let lookahead = ahead.lookahead1();
                if s.is_empty() {
                    Ok(Empty)
                } else if lookahead.peek(kw::builtin) {
                    s.advance_to(&ahead);
                    s.parse::<kw::builtin>()?;
                    s.parse::<Token![#]>()?;
                    let name: Ident = s.parse()?;
                    let args;
                    parenthesized!(args in s);
                    let args: pm2::Stream = args.parse()?;
                    Ok(Builtin(Builtin { attrs, name, args }))
                } else if lookahead.peek(Token![&]) {
                    s.advance_to(&ahead);
                    s.parse::<Token![&]>()?;
                    s.parse::<kw::raw>()?;
                    let mutable = s.parse::<Option<Token![mut]>>()?.is_some();
                    if !mutable {
                        s.parse::<Token![const]>()?;
                    }
                    let expr: Expr = s.parse()?;
                    Ok(RawRef(RawRef { attrs, mutable, expr }))
                } else if lookahead.peek(Token![...]) {
                    s.parse::<Token![...]>()?;
                    Ok(Ellipsis)
                } else {
                    Err(lookahead.error())
                }
            }
        }
        let y: Type = match parse2(self.clone()) {
            Ok(x) => x,
            Err(_) => unimplemented!("Expr::Verbatim `{}`", self),
        };
        match y {
            Empty => {},
            Ellipsis => {
                p.word("...");
            },
            Builtin(x) => {
                p.outer_attrs(&x.attrs);
                p.word("builtin # ");
                &x.name.pretty(p);
                p.word("(");
                if !x.args.is_empty() {
                    p.cbox(INDENT);
                    p.zerobreak();
                    p.ibox(0);
                    p.macro_rules_tokens(x.args, false);
                    p.end();
                    p.zerobreak();
                    p.offset(-INDENT);
                    p.end();
                }
                p.word(")");
            },
            RawRef(x) => {
                p.outer_attrs(&x.attrs);
                p.word("&raw ");
                p.word(if x.mutable { "mut " } else { "const " });
                &x.expr.pretty(p);
            },
        }
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
impl Fragment for Member {
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
        if x.peek(ident::Ident) {
            x.parse().map(Member::Named)
        } else if x.peek(lit::Int) {
            x.parse().map(Member::Unnamed)
        } else {
            Err(x.error("expected identifier or integer"))
        }
    }
}
impl Lower for Member {
    fn lower(&self, s: &mut Stream) {
        match self {
            Member::Named(x) => x.lower(s),
            Member::Unnamed(x) => x.lower(s),
        }
    }
}
impl Pretty for Member {
    fn pretty(&self, p: &mut Print) {
        match self {
            Member::Named(x) => x.pretty(p),
            Member::Unnamed(x) => x.pretty(p),
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
impl Fragment for Idx {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.idx, f)
    }
    fn span(&self) -> Option<pm2::Span> {
        Some(self.span)
    }
}
impl Lower for Idx {
    fn lower(&self, s: &mut Stream) {
        let mut y = pm2::Lit::i64_unsuffixed(i64::from(self.idx));
        y.set_span(self.span);
        s.append(y);
    }
}
impl Pretty for Idx {
    fn pretty(&self, p: &mut Print) {
        p.word(self.idx.to_string());
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
        use BinOp::*;
        if x.peek(Token![+=]) {
            x.parse().map(AddAssign)
        } else if x.peek(Token![-=]) {
            x.parse().map(SubAssign)
        } else if x.peek(Token![*=]) {
            x.parse().map(MulAssign)
        } else if x.peek(Token![/=]) {
            x.parse().map(DivAssign)
        } else if x.peek(Token![%=]) {
            x.parse().map(RemAssign)
        } else if x.peek(Token![^=]) {
            x.parse().map(BitXorAssign)
        } else if x.peek(Token![&=]) {
            x.parse().map(BitAndAssign)
        } else if x.peek(Token![|=]) {
            x.parse().map(BitOrAssign)
        } else if x.peek(Token![<<=]) {
            x.parse().map(ShlAssign)
        } else if x.peek(Token![>>=]) {
            x.parse().map(ShrAssign)
        } else {
            parse_binop(x)
        }
    }
}
impl Lower for BinOp {
    fn lower(&self, s: &mut Stream) {
        use BinOp::*;
        match self {
            Add(x) => x.lower(s),
            Sub(x) => x.lower(s),
            Mul(x) => x.lower(s),
            Div(x) => x.lower(s),
            Rem(x) => x.lower(s),
            And(x) => x.lower(s),
            Or(x) => x.lower(s),
            BitXor(x) => x.lower(s),
            BitAnd(x) => x.lower(s),
            BitOr(x) => x.lower(s),
            Shl(x) => x.lower(s),
            Shr(x) => x.lower(s),
            Eq(x) => x.lower(s),
            Lt(x) => x.lower(s),
            Le(x) => x.lower(s),
            Ne(x) => x.lower(s),
            Ge(x) => x.lower(s),
            Gt(x) => x.lower(s),
            AddAssign(x) => x.lower(s),
            SubAssign(x) => x.lower(s),
            MulAssign(x) => x.lower(s),
            DivAssign(x) => x.lower(s),
            RemAssign(x) => x.lower(s),
            BitXorAssign(x) => x.lower(s),
            BitAndAssign(x) => x.lower(s),
            BitOrAssign(x) => x.lower(s),
            ShlAssign(x) => x.lower(s),
            ShrAssign(x) => x.lower(s),
        }
    }
}
impl Pretty for BinOp {
    fn pretty(&self, p: &mut Print) {
        use BinOp::*;
        p.word(match self {
            Add(_) => "+",
            Sub(_) => "-",
            Mul(_) => "*",
            Div(_) => "/",
            Rem(_) => "%",
            And(_) => "&&",
            Or(_) => "||",
            BitXor(_) => "^",
            BitAnd(_) => "&",
            BitOr(_) => "|",
            Shl(_) => "<<",
            Shr(_) => ">>",
            Eq(_) => "==",
            Lt(_) => "<",
            Le(_) => "<=",
            Ne(_) => "!=",
            Ge(_) => ">=",
            Gt(_) => ">",
            AddAssign(_) => "+=",
            SubAssign(_) => "-=",
            MulAssign(_) => "*=",
            DivAssign(_) => "/=",
            RemAssign(_) => "%=",
            BitXorAssign(_) => "^=",
            BitAndAssign(_) => "&=",
            BitOrAssign(_) => "|=",
            ShlAssign(_) => "<<=",
            ShrAssign(_) => ">>=",
        });
    }
}

pub enum UnOp {
    Deref(Token![*]),
    Not(Token![!]),
    Neg(Token![-]),
}
impl Parse for UnOp {
    fn parse(x: Stream) -> Res<Self> {
        use UnOp::*;
        let look = x.look1();
        if look.peek(Token![*]) {
            x.parse().map(Deref)
        } else if look.peek(Token![!]) {
            x.parse().map(Not)
        } else if look.peek(Token![-]) {
            x.parse().map(Neg)
        } else {
            Err(look.err())
        }
    }
}
impl Lower for UnOp {
    fn lower(&self, s: &mut Stream) {
        use UnOp::*;
        match self {
            Deref(x) => x.lower(s),
            Not(x) => x.lower(s),
            Neg(x) => x.lower(s),
        }
    }
}
impl Pretty for UnOp {
    fn pretty(&self, p: &mut Print) {
        use UnOp::*;
        p.word(match self {
            Deref(_) => "*",
            Not(_) => "!",
            Neg(_) => "-",
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown UnOp"),
        });
    }
}

pub struct FieldValue {
    pub attrs: Vec<attr::Attr>,
    pub memb: Member,
    pub colon: Option<Token![:]>,
    pub expr: Expr,
}
impl Parse for FieldValue {
    fn parse(x: Stream) -> Res<Self> {
        let attrs = x.call(attr::Attr::parse_outers)?;
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
            expr: val,
        })
    }
}
impl Lower for FieldValue {
    fn lower(&self, s: &mut Stream) {
        attr::lower_outers(&self.attrs, s);
        self.memb.lower(s);
        if let Some(colon) = &self.colon {
            colon.lower(s);
            self.expr.lower(s);
        }
    }
}
impl Pretty for FieldValue {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.member(&self.memb);
        if self.colon.is_some() {
            p.word(": ");
            p.ibox(0);
            &self.expr.pretty(p);
            p.end();
        }
    }
}

pub struct Label {
    pub name: Life,
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
        if x.peek(Life) {
            x.parse().map(Some)
        } else {
            Ok(None)
        }
    }
}
impl Lower for Label {
    fn lower(&self, s: &mut Stream) {
        self.name.lower(s);
        self.colon.lower(s);
    }
}
impl Pretty for Label {
    fn pretty(&self, p: &mut Print) {
        p.lifetime(&self.name);
        p.word(": ");
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
            attrs: x.call(attr::Attr::parse_outers)?,
            pat: pat::Pat::parse_many(x)?,
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
                comma = &body.needs_term();
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
impl Lower for Arm {
    fn lower(&self, s: &mut Stream) {
        s.append_all(&self.attrs);
        self.pat.lower(s);
        if let Some((if_, x)) = &self.guard {
            if_.lower(s);
            x.lower(s);
        }
        self.fat_arrow.lower(s);
        self.body.lower(s);
        self.comma.lower(s);
    }
}
impl Pretty for Arm {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.ibox(0);
        p.pat(&self.pat);
        if let Some((_, x)) = &self.guard {
            p.word(" if ");
            x.pretty(p);
        }
        p.word(" =>");
        let empty_block;
        let mut body = &*self.body;
        while let Expr::Block(x) = body {
            if x.attrs.is_empty() && x.label.is_none() {
                let mut stmts = x.block.stmts.iter();
                if let (Some(stmt::Stmt::Expr(inner, None)), None) = (stmts.next(), stmts.next()) {
                    body = inner;
                    continue;
                }
            }
            break;
        }
        if let Expr::Tuple(x) = body {
            if x.elems.is_empty() && x.attrs.is_empty() {
                empty_block = Expr::Block(Block {
                    attrs: Vec::new(),
                    label: None,
                    block: stmt::Block {
                        brace: tok::Brace::default(),
                        stmts: Vec::new(),
                    },
                });
                body = &empty_block;
            }
        }
        if let Expr::Block(x) = body {
            p.nbsp();
            if let Some(x) = &x.label {
                p.label(x);
            }
            p.word("{");
            p.neverbreak();
            p.cbox(INDENT);
            p.hardbreak_if_nonempty();
            p.inner_attrs(&x.attrs);
            for x in &x.block.stmts {
                p.stmt(x);
            }
            p.offset(-INDENT);
            p.end();
            p.word("}");
            p.end();
        } else {
            p.nbsp();
            p.neverbreak();
            p.cbox(INDENT);
            p.scan_break(BreakToken {
                pre_break: Some('{'),
                ..BreakToken::default()
            });
            body.pretty_beg_of_line(p, true);
            p.scan_break(BreakToken {
                offset: -INDENT,
                pre_break: body.add_semi().then(|| ';'),
                post_break: Some('}'),
                no_break: body.needs_term().then(|| ','),
                ..BreakToken::default()
            });
            p.end();
            p.end();
        }
    }
}

pub enum Limits {
    HalfOpen(Token![..]),
    Closed(Token![..=]),
}
impl Limits {
    pub fn parse_obsolete(x: Stream) -> Res<Self> {
        let look = x.look1();
        let dot2 = look.peek(Token![..]);
        let dot2_eq = dot2 && look.peek(Token![..=]);
        let dot3 = dot2 && x.peek(Token![...]);
        if dot2_eq {
            x.parse().map(Limits::Closed)
        } else if dot3 {
            let y: Token![...] = x.parse()?;
            Ok(Limits::Closed(Token![..=](y.spans)))
        } else if dot2 {
            x.parse().map(Limits::HalfOpen)
        } else {
            Err(look.err())
        }
    }
}
impl Parse for Limits {
    fn parse(x: Stream) -> Res<Self> {
        let look = x.look1();
        let dot2 = look.peek(Token![..]);
        let dot2_eq = dot2 && look.peek(Token![..=]);
        let dot3 = dot2 && x.peek(Token![...]);
        if dot2_eq {
            x.parse().map(Limits::Closed)
        } else if dot2 && !dot3 {
            x.parse().map(Limits::HalfOpen)
        } else {
            Err(look.err())
        }
    }
}
impl Lower for Limits {
    fn lower(&self, s: &mut Stream) {
        match self {
            Limits::HalfOpen(x) => x.lower(s),
            Limits::Closed(x) => x.lower(s),
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
        use BinOp::*;
        match op {
            Add(_) | Sub(_) => Precedence::Arithmetic,
            Mul(_) | Div(_) | Rem(_) => Precedence::Term,
            And(_) => Precedence::And,
            Or(_) => Precedence::Or,
            BitXor(_) => Precedence::BitXor,
            BitAnd(_) => Precedence::BitAnd,
            BitOr(_) => Precedence::BitOr,
            Shl(_) | Shr(_) => Precedence::Shift,
            Eq(_) | Lt(_) | Le(_) | Ne(_) | Ge(_) | Gt(_) => Precedence::Compare,
            AddAssign(_) | SubAssign(_) | MulAssign(_) | DivAssign(_) | RemAssign(_) | BitXorAssign(_)
            | BitAndAssign(_) | BitOrAssign(_) | ShlAssign(_) | ShrAssign(_) => Precedence::Assign,
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
    crate::custom_kw!(builtin);
    crate::custom_kw!(raw);
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
            let limits: Limits = x.parse()?;
            let rhs = if matches!(limits, Limits::HalfOpen(_))
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
            let typ = typ::parse_ambig_typ(x, plus, gen)?;
            check_cast(x)?;
            lhs = Expr::Cast(Cast {
                attrs: Vec::new(),
                expr: Box::new(lhs),
                as_,
                typ: Box::new(typ),
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
            let y = group.buf.call(attr::parse_one_outer)?;
            if !group.buf.is_empty() {
                break;
            }
            ys.push(y);
        } else if x.peek(Token![#]) {
            ys.push(x.call(attr::parse_one_outer)?);
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
            Ok(Expr::Verbatim(parse::parse_verbatim(&beg, x)))
        } else {
            Ok(Expr::Reference(Ref { attrs, and, mut_, expr }))
        }
    } else if x.peek(Token![*]) || x.peek(Token![!]) || x.peek(Token![-]) {
        expr_unary(x, attrs, allow).map(Expr::Unary)
    } else {
        trailer_expr(beg, attrs, x, allow)
    }
}
fn trailer_expr(beg: parse::Buffer, mut attrs: Vec<attr::Attr>, x: Stream, allow: AllowStruct) -> Res<Expr> {
    let atom = atom_expr(x, allow)?;
    let mut y = trailer_helper(x, atom)?;
    if let Expr::Verbatim(tokens) = &mut y {
        *tokens = parse::parse_verbatim(&beg, x);
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
                Some(path::Angled::parse_turbofish(x)?)
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
                expr: Box::new(y),
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
    } else if x.peek(lit::Lit) {
        x.parse().map(Expr::Lit)
    } else if x.peek(Token![async]) && (x.peek2(tok::Brace) || x.peek2(Token![move]) && x.peek3(tok::Brace)) {
        x.parse().map(Expr::Async)
    } else if x.peek(Token![try]) && x.peek2(tok::Brace) {
        x.parse().map(Expr::TryBlock)
    } else if x.peek(Token![|])
        || x.peek(Token![move])
        || x.peek(Token![for]) && x.peek2(Token![<]) && (x.peek3(Life) || x.peek3(Token![>]))
        || x.peek(Token![const]) && !x.peek2(tok::Brace)
        || x.peek(Token![static])
        || x.peek(Token![async]) && (x.peek2(Token![|]) || x.peek2(Token![move]))
    {
        expr_closure(x, allow).map(Expr::Closure)
    } else if x.peek(kw::builtin) && x.peek2(Token![#]) {
        expr_builtin(x)
    } else if x.peek(ident::Ident)
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
    } else if x.peek(Life) {
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
    Ok(Expr::Verbatim(parse::parse_verbatim(&begin, x)))
}
fn path_or_macro_or_struct(x: Stream, allow: AllowStruct) -> Res<Expr> {
    let (qself, path) = path::qpath(x, true)?;
    if qself.is_none() && x.peek(Token![!]) && !x.peek(Token![!=]) && path.is_mod_style() {
        let bang: Token![!] = x.parse()?;
        let (delim, toks) = mac::parse_delim(x)?;
        return Ok(Expr::Mac(Mac {
            attrs: Vec::new(),
            mac: mac::Mac {
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
            elems: Puncted::new(),
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
    let mut elems = Puncted::new();
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
            elems: Puncted::new(),
        }));
    }
    let first: Expr = y.parse()?;
    if y.is_empty() || y.peek(Token![,]) {
        let mut elems = Puncted::new();
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
    } else if x.peek(Token![for]) && !(x.peek2(Token![<]) && (x.peek3(Life) || x.peek3(Token![>]))) {
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
    let look = x.look1();
    let branch = if x.peek(Token![if]) {
        x.parse().map(Expr::If)?
    } else if x.peek(tok::Brace) {
        Expr::Block(Block {
            attrs: Vec::new(),
            label: None,
            block: x.parse()?,
        })
    } else {
        return Err(look.err());
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
    let lifes: Option<gen::bound::Lifes> = x.parse()?;
    let const_: Option<Token![const]> = x.parse()?;
    let static_: Option<Token![static]> = x.parse()?;
    let async_: Option<Token![async]> = x.parse()?;
    let move_: Option<Token![move]> = x.parse()?;
    let or1: Token![|] = x.parse()?;
    let mut ins = Puncted::new();
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
    let attrs = x.call(attr::Attr::parse_outers)?;
    let mut y = pat::Pat::parse_one(x)?;
    if x.peek(Token![:]) {
        Ok(pat::Pat::Type(pat::Type {
            attrs,
            pat: Box::new(y),
            colon: x.parse()?,
            typ: x.parse()?,
        }))
    } else {
        use pat::Pat::*;
        match &mut y {
            Const(x) => x.attrs = attrs,
            Ident(x) => x.attrs = attrs,
            Lit(x) => x.attrs = attrs,
            Mac(x) => x.attrs = attrs,
            Or(x) => x.attrs = attrs,
            Paren(x) => x.attrs = attrs,
            Path(x) => x.attrs = attrs,
            Range(x) => x.attrs = attrs,
            Ref(x) => x.attrs = attrs,
            Rest(x) => x.attrs = attrs,
            Slice(x) => x.attrs = attrs,
            Struct(x) => x.attrs = attrs,
            Tuple(x) => x.attrs = attrs,
            TupleStruct(x) => x.attrs = attrs,
            Type(_) => unreachable!(),
            Verbatim(_) => {},
            Wild(x) => x.attrs = attrs,
        }
        Ok(y)
    }
}
fn expr_break(x: Stream, allow: AllowStruct) -> Res<Break> {
    Ok(Break {
        attrs: Vec::new(),
        break_: x.parse()?,
        life: x.parse()?,
        val: {
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
fn expr_struct_helper(x: Stream, qself: Option<path::QSelf>, path: Path) -> Res<Struct> {
    let y;
    let brace = braced!(y in x);
    let mut fields = Puncted::new();
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
    let limits: Limits = x.parse()?;
    let end = if matches!(limits, Limits::HalfOpen(_))
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
        let mut index: Index = parse::parse_str(part).map_err(|err| Err::new(float_span, err))?;
        let part_end = offset + part.len();
        index.span = float_token.subspan(offset..part_end).unwrap_or(float_span);
        let base = mem::replace(e, Expr::DUMMY);
        *e = Expr::Field(Field {
            attrs: Vec::new(),
            expr: Box::new(base),
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
        } else if x.peek2(ident::Ident) && (x.peek3(tok::Paren) || x.peek3(Token![::])) {
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
    use BinOp::*;
    if x.peek(Token![&&]) {
        x.parse().map(And)
    } else if x.peek(Token![||]) {
        x.parse().map(Or)
    } else if x.peek(Token![<<]) {
        x.parse().map(Shl)
    } else if x.peek(Token![>>]) {
        x.parse().map(Shr)
    } else if x.peek(Token![==]) {
        x.parse().map(Eq)
    } else if x.peek(Token![<=]) {
        x.parse().map(Le)
    } else if x.peek(Token![!=]) {
        x.parse().map(Ne)
    } else if x.peek(Token![>=]) {
        x.parse().map(Ge)
    } else if x.peek(Token![+]) {
        x.parse().map(Add)
    } else if x.peek(Token![-]) {
        x.parse().map(Sub)
    } else if x.peek(Token![*]) {
        x.parse().map(Mul)
    } else if x.peek(Token![/]) {
        x.parse().map(Div)
    } else if x.peek(Token![%]) {
        x.parse().map(Rem)
    } else if x.peek(Token![^]) {
        x.parse().map(BitXor)
    } else if x.peek(Token![&]) {
        x.parse().map(BitAnd)
    } else if x.peek(Token![|]) {
        x.parse().map(BitOr)
    } else if x.peek(Token![<]) {
        x.parse().map(Lt)
    } else if x.peek(Token![>]) {
        x.parse().map(Gt)
    } else {
        Err(x.error("expected binary operator"))
    }
}
