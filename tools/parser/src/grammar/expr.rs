use super::*;

mod atom {
    use super::*;
    pub const LITERAL_FIRST: TokenSet = TokenSet::new(&[
        T![true],
        T![false],
        INT_NUMBER,
        FLOAT_NUMBER,
        BYTE,
        CHAR,
        STRING,
        BYTE_STRING,
        C_STRING,
    ]);
    pub fn literal(p: &mut Parser<'_>) -> Option<CompletedMarker> {
        if !p.at_ts(LITERAL_FIRST) {
            return None;
        }
        let m = p.start();
        p.bump_any();
        Some(m.complete(p, LITERAL))
    }
    pub const ATOM_EXPR_FIRST: TokenSet = LITERAL_FIRST.union(path::FIRST).union(TokenSet::new(&[
        T!['('],
        T!['{'],
        T!['['],
        T![|],
        T![async],
        T![box],
        T![break],
        T![const],
        T![continue],
        T![do],
        T![for],
        T![if],
        T![let],
        T![loop],
        T![match],
        T![move],
        T![return],
        T![static],
        T![try],
        T![unsafe],
        T![while],
        T![yield],
        LIFETIME_IDENT,
    ]));
    pub const EXPR_RECOVERY_SET: TokenSet = TokenSet::new(&[T![')'], T![']']]);
    pub fn atom_expr(p: &mut Parser<'_>, r: Restrictions) -> Option<(CompletedMarker, BlockLike)> {
        if let Some(m) = literal(p) {
            return Some((m, BlockLike::NotBlock));
        }
        if path::is_start(p) {
            return Some(path_expr(p, r));
        }
        let la = p.nth(1);
        let done = match p.current() {
            T!['('] => tuple_expr(p),
            T!['['] => array_expr(p),
            T![if] => if_expr(p),
            T![let] => let_expr(p),
            T![_] => {
                let m = p.start();
                p.bump(T![_]);
                m.complete(p, UNDERSCORE_EXPR)
            },
            T![loop] => loop_expr(p, None),
            T![box] => box_expr(p, None),
            T![while] => while_expr(p, None),
            T![try] => try_block_expr(p, None),
            T![match] => match_expr(p),
            T![return] => return_expr(p),
            T![yield] => yield_expr(p),
            T![do] if p.nth_at_contextual_kw(1, T![yeet]) => yeet_expr(p),
            T![continue] => continue_expr(p),
            T![break] => break_expr(p, r),
            LIFETIME_IDENT if la == T![:] => {
                let m = p.start();
                label(p);
                match p.current() {
                    T![loop] => loop_expr(p, Some(m)),
                    T![for] => for_expr(p, Some(m)),
                    T![while] => while_expr(p, Some(m)),
                    T!['{'] => {
                        stmt_list(p);
                        m.complete(p, BLOCK_EXPR)
                    },
                    _ => {
                        p.error("expected a loop or block");
                        m.complete(p, ERROR);
                        return None;
                    },
                }
            },
            T![const] | T![unsafe] | T![async] if la == T!['{'] => {
                let m = p.start();
                p.bump_any();
                stmt_list(p);
                m.complete(p, BLOCK_EXPR)
            },
            T![async] if la == T![move] && p.nth(2) == T!['{'] => {
                let m = p.start();
                p.bump(T![async]);
                p.eat(T![move]);
                stmt_list(p);
                m.complete(p, BLOCK_EXPR)
            },
            T!['{'] => {
                let m = p.start();
                stmt_list(p);
                m.complete(p, BLOCK_EXPR)
            },
            T![const] | T![static] | T![async] | T![move] | T![|] => closure_expr(p),
            T![for] if la == T![<] => closure_expr(p),
            T![for] => for_expr(p, None),
            _ => {
                p.err_and_bump("expected expression");
                return None;
            },
        };
        let blocklike = if BlockLike::is_blocklike(done.kind()) {
            BlockLike::Block
        } else {
            BlockLike::NotBlock
        };
        Some((done, blocklike))
    }
    fn tuple_expr(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T!['(']));
        let m = p.start();
        p.expect(T!['(']);
        let mut saw_comma = false;
        let mut saw_expr = false;
        if p.eat(T![,]) {
            p.error("expected expression");
            saw_comma = true;
        }
        while !p.at(EOF) && !p.at(T![')']) {
            saw_expr = true;
            if expr(p).is_none() {
                break;
            }
            if !p.at(T![')']) {
                saw_comma = true;
                p.expect(T![,]);
            }
        }
        p.expect(T![')']);
        m.complete(p, if saw_expr && !saw_comma { PAREN_EXPR } else { TUPLE_EXPR })
    }
    fn array_expr(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T!['[']));
        let m = p.start();
        let mut n_exprs = 0u32;
        let mut has_semi = false;
        p.bump(T!['[']);
        while !p.at(EOF) && !p.at(T![']']) {
            n_exprs += 1;
            if expr(p).is_none() {
                break;
            }
            if n_exprs == 1 && p.eat(T![;]) {
                has_semi = true;
                continue;
            }
            if has_semi || !p.at(T![']']) && !p.expect(T![,]) {
                break;
            }
        }
        p.expect(T![']']);
        m.complete(p, ARRAY_EXPR)
    }
    fn closure_expr(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(match p.current() {
            T![const] | T![static] | T![async] | T![move] | T![|] => true,
            T![for] => p.nth(1) == T![<],
            _ => false,
        });
        let m = p.start();
        if p.at(T![for]) {
            ty::for_binder(p);
        }
        p.eat(T![const]);
        p.eat(T![static]);
        p.eat(T![async]);
        p.eat(T![move]);
        if !p.at(T![|]) {
            p.error("expected `|`");
            return m.complete(p, CLOSURE_EXPR);
        }
        param::closure(p);
        if opt_ret_type(p) {
            block_expr(p);
        } else if p.at_ts(EXPR_FIRST) {
            expr(p);
        } else {
            p.error("expected expression");
        }
        m.complete(p, CLOSURE_EXPR)
    }
    fn if_expr(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T![if]));
        let m = p.start();
        p.bump(T![if]);
        expr_no_struct(p);
        block_expr(p);
        if p.at(T![else]) {
            p.bump(T![else]);
            if p.at(T![if]) {
                if_expr(p);
            } else {
                block_expr(p);
            }
        }
        m.complete(p, IF_EXPR)
    }
    fn label(p: &mut Parser<'_>) {
        assert!(p.at(LIFETIME_IDENT) && p.nth(1) == T![:]);
        let m = p.start();
        lifetime(p);
        p.bump_any();
        m.complete(p, LABEL);
    }
    fn loop_expr(p: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
        assert!(p.at(T![loop]));
        let m = m.unwrap_or_else(|| p.start());
        p.bump(T![loop]);
        block_expr(p);
        m.complete(p, LOOP_EXPR)
    }
    fn while_expr(p: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
        assert!(p.at(T![while]));
        let m = m.unwrap_or_else(|| p.start());
        p.bump(T![while]);
        expr_no_struct(p);
        block_expr(p);
        m.complete(p, WHILE_EXPR)
    }
    fn for_expr(p: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
        assert!(p.at(T![for]));
        let m = m.unwrap_or_else(|| p.start());
        p.bump(T![for]);
        pattern::pattern(p);
        p.expect(T![in]);
        expr_no_struct(p);
        block_expr(p);
        m.complete(p, FOR_EXPR)
    }
    fn let_expr(p: &mut Parser<'_>) -> CompletedMarker {
        let m = p.start();
        p.bump(T![let]);
        pattern::top(p);
        p.expect(T![=]);
        expr_let(p);
        m.complete(p, LET_EXPR)
    }
    fn match_expr(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T![match]));
        let m = p.start();
        p.bump(T![match]);
        expr_no_struct(p);
        if p.at(T!['{']) {
            match_arm_list(p);
        } else {
            p.error("expected `{`");
        }
        m.complete(p, MATCH_EXPR)
    }
    pub fn match_arm_list(p: &mut Parser<'_>) {
        assert!(p.at(T!['{']));
        let m = p.start();
        p.eat(T!['{']);
        attr::inners(p);
        while !p.at(EOF) && !p.at(T!['}']) {
            if p.at(T!['{']) {
                err_block(p, "expected match arm");
                continue;
            }
            match_arm(p);
        }
        p.expect(T!['}']);
        m.complete(p, MATCH_ARM_LIST);
    }
    fn match_arm(p: &mut Parser<'_>) {
        let m = p.start();
        attr::outers(p);
        pattern::top_r(p, TokenSet::EMPTY);
        if p.at(T![if]) {
            match_guard(p);
        }
        p.expect(T![=>]);
        let blocklike = match expr_stmt(p, None) {
            Some((_, blocklike)) => blocklike,
            None => BlockLike::NotBlock,
        };
        if !p.eat(T![,]) && !blocklike.is_block() && !p.at(T!['}']) {
            p.error("expected `,`");
        }
        m.complete(p, MATCH_ARM);
    }
    fn match_guard(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T![if]));
        let m = p.start();
        p.bump(T![if]);
        expr(p);
        m.complete(p, MATCH_GUARD)
    }
    pub fn block_expr(p: &mut Parser<'_>) {
        if !p.at(T!['{']) {
            p.error("expected a block");
            return;
        }
        let m = p.start();
        stmt_list(p);
        m.complete(p, BLOCK_EXPR);
    }
    fn stmt_list(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T!['{']));
        let m = p.start();
        p.bump(T!['{']);
        expr_block_contents(p);
        p.expect(T!['}']);
        m.complete(p, STMT_LIST)
    }
    fn return_expr(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T![return]));
        let m = p.start();
        p.bump(T![return]);
        if p.at_ts(EXPR_FIRST) {
            expr(p);
        }
        m.complete(p, RETURN_EXPR)
    }
    fn yield_expr(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T![yield]));
        let m = p.start();
        p.bump(T![yield]);
        if p.at_ts(EXPR_FIRST) {
            expr(p);
        }
        m.complete(p, YIELD_EXPR)
    }
    fn yeet_expr(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T![do]));
        assert!(p.nth_at_contextual_kw(1, T![yeet]));
        let m = p.start();
        p.bump(T![do]);
        p.bump_remap(T![yeet]);
        if p.at_ts(EXPR_FIRST) {
            expr(p);
        }
        m.complete(p, YEET_EXPR)
    }
    fn continue_expr(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T![continue]));
        let m = p.start();
        p.bump(T![continue]);
        if p.at(LIFETIME_IDENT) {
            lifetime(p);
        }
        m.complete(p, CONTINUE_EXPR)
    }
    fn break_expr(p: &mut Parser<'_>, r: Restrictions) -> CompletedMarker {
        assert!(p.at(T![break]));
        let m = p.start();
        p.bump(T![break]);
        if p.at(LIFETIME_IDENT) {
            lifetime(p);
        }
        if p.at_ts(EXPR_FIRST) && !(r.forbid_structs && p.at(T!['{'])) {
            expr(p);
        }
        m.complete(p, BREAK_EXPR)
    }
    fn try_block_expr(p: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
        assert!(p.at(T![try]));
        let m = m.unwrap_or_else(|| p.start());
        if p.nth_at(1, T![!]) {
            let macro_call = p.start();
            let path = p.start();
            let segment = p.start();
            let name_ref = p.start();
            p.bump_remap(IDENT);
            name_ref.complete(p, NAME_REF);
            segment.complete(p, PATH_SEGMENT);
            path.complete(p, PATH);
            let _block_like = item::macro_call_after_excl(p);
            macro_call.complete(p, MACRO_CALL);
            return m.complete(p, MACRO_EXPR);
        }
        p.bump(T![try]);
        if p.at(T!['{']) {
            stmt_list(p);
        } else {
            p.error("expected a block");
        }
        m.complete(p, BLOCK_EXPR)
    }
    fn box_expr(p: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
        assert!(p.at(T![box]));
        let m = m.unwrap_or_else(|| p.start());
        p.bump(T![box]);
        if p.at_ts(EXPR_FIRST) {
            expr(p);
        }
        m.complete(p, BOX_EXPR)
    }
    pub fn const_arg(x: &mut Parser<'_>) {
        match x.current() {
            T!['{'] => {
                block_expr(x);
            },
            k if k.is_literal() => {
                literal(x);
            },
            T![true] | T![false] => {
                literal(x);
            },
            T![-] => {
                let y = x.start();
                x.bump(T![-]);
                literal(x);
                y.complete(x, PREFIX_EXPR);
            },
            _ => {
                let y = x.start();
                path::for_use(x);
                y.complete(x, PATH_EXPR);
            },
        }
    }
}
pub use atom::{block_expr, const_arg, literal, match_arm_list, LITERAL_FIRST};

#[derive(PartialEq, Eq)]
pub enum Semicolon {
    Required,
    Optional,
    Forbidden,
}

const EXPR_FIRST: TokenSet = LHS_FIRST;

pub fn expr(p: &mut Parser<'_>) -> Option<CompletedMarker> {
    let r = Restrictions {
        forbid_structs: false,
        prefer_stmt: false,
    };
    expr_bp(p, None, r, 1).map(|(m, _)| m)
}
pub fn expr_stmt(p: &mut Parser<'_>, m: Option<Marker>) -> Option<(CompletedMarker, BlockLike)> {
    let r = Restrictions {
        forbid_structs: false,
        prefer_stmt: true,
    };
    expr_bp(p, m, r, 1)
}
fn expr_no_struct(p: &mut Parser<'_>) {
    let r = Restrictions {
        forbid_structs: true,
        prefer_stmt: false,
    };
    expr_bp(p, None, r, 1);
}
fn expr_let(p: &mut Parser<'_>) {
    let r = Restrictions {
        forbid_structs: true,
        prefer_stmt: false,
    };
    expr_bp(p, None, r, 5);
}
pub fn stmt(p: &mut Parser<'_>, semicolon: Semicolon) {
    if p.eat(T![;]) {
        return;
    }
    let m = p.start();
    attr::outers(p);
    if p.at(T![let]) {
        let_stmt(p, m, semicolon);
        return;
    }
    let m = match item::opt_item(p, m) {
        Ok(()) => return,
        Err(m) => m,
    };
    if !p.at_ts(EXPR_FIRST) {
        p.err_and_bump("expected expression, item or let statement");
        m.abandon(p);
        return;
    }
    if let Some((cm, blocklike)) = expr_stmt(p, Some(m)) {
        if !(p.at(T!['}']) || (semicolon != Semicolon::Required && p.at(EOF))) {
            let m = cm.precede(p);
            match semicolon {
                Semicolon::Required => {
                    if blocklike.is_block() {
                        p.eat(T![;]);
                    } else {
                        p.expect(T![;]);
                    }
                },
                Semicolon::Optional => {
                    p.eat(T![;]);
                },
                Semicolon::Forbidden => (),
            }
            m.complete(p, EXPR_STMT);
        }
    }
    fn let_stmt(p: &mut Parser<'_>, m: Marker, with_semi: Semicolon) {
        p.bump(T![let]);
        pattern::pattern(p);
        if p.at(T![:]) {
            ty::ascription(p);
        }
        let mut expr_after_eq: Option<CompletedMarker> = None;
        if p.eat(T![=]) {
            expr_after_eq = expr::expr(p);
        }
        if p.at(T![else]) {
            if let Some(expr) = expr_after_eq {
                if BlockLike::is_blocklike(expr.kind()) {
                    p.error("right curly brace `}` before `else` in a `let...else` statement not allowed")
                }
            }
            let m = p.start();
            p.bump(T![else]);
            block_expr(p);
            m.complete(p, LET_ELSE);
        }
        match with_semi {
            Semicolon::Forbidden => (),
            Semicolon::Optional => {
                p.eat(T![;]);
            },
            Semicolon::Required => {
                p.expect(T![;]);
            },
        }
        m.complete(p, LET_STMT);
    }
}
pub fn expr_block_contents(p: &mut Parser<'_>) {
    attr::inners(p);
    while !p.at(EOF) && !p.at(T!['}']) {
        stmt(p, Semicolon::Required);
    }
}

#[derive(Clone, Copy)]
struct Restrictions {
    forbid_structs: bool,
    prefer_stmt: bool,
}

enum Associativity {
    Left,
    Right,
}

#[rustfmt::skip]
fn current_op(p: &Parser<'_>) -> (u8, SyntaxKind, Associativity) {
    use Associativity::*;
    const NOT_AN_OP: (u8, SyntaxKind, Associativity) = (0, T![@], Left);
    match p.current() {
        T![|] if p.at(T![||])  => (3,  T![||],  Left),
        T![|] if p.at(T![|=])  => (1,  T![|=],  Right),
        T![|]                  => (6,  T![|],   Left),
        T![>] if p.at(T![>>=]) => (1,  T![>>=], Right),
        T![>] if p.at(T![>>])  => (9,  T![>>],  Left),
        T![>] if p.at(T![>=])  => (5,  T![>=],  Left),
        T![>]                  => (5,  T![>],   Left),
        T![=] if p.at(T![=>])  => NOT_AN_OP,
        T![=] if p.at(T![==])  => (5,  T![==],  Left),
        T![=]                  => (1,  T![=],   Right),
        T![<] if p.at(T![<=])  => (5,  T![<=],  Left),
        T![<] if p.at(T![<<=]) => (1,  T![<<=], Right),
        T![<] if p.at(T![<<])  => (9,  T![<<],  Left),
        T![<]                  => (5,  T![<],   Left),
        T![+] if p.at(T![+=])  => (1,  T![+=],  Right),
        T![+]                  => (10, T![+],   Left),
        T![^] if p.at(T![^=])  => (1,  T![^=],  Right),
        T![^]                  => (7,  T![^],   Left),
        T![%] if p.at(T![%=])  => (1,  T![%=],  Right),
        T![%]                  => (11, T![%],   Left),
        T![&] if p.at(T![&=])  => (1,  T![&=],  Right),
        T![&] if p.at(T![&&])  => (4,  T![&&],  Left),
        T![&]                  => (8,  T![&],   Left),
        T![/] if p.at(T![/=])  => (1,  T![/=],  Right),
        T![/]                  => (11, T![/],   Left),
        T![*] if p.at(T![*=])  => (1,  T![*=],  Right),
        T![*]                  => (11, T![*],   Left),
        T![.] if p.at(T![..=]) => (2,  T![..=], Left),
        T![.] if p.at(T![..])  => (2,  T![..],  Left),
        T![!] if p.at(T![!=])  => (5,  T![!=],  Left),
        T![-] if p.at(T![-=])  => (1,  T![-=],  Right),
        T![-]                  => (10, T![-],   Left),
        T![as]                 => (12, T![as],  Left),
        _                      => NOT_AN_OP
    }
}
fn expr_bp(p: &mut Parser<'_>, m: Option<Marker>, mut r: Restrictions, bp: u8) -> Option<(CompletedMarker, BlockLike)> {
    let m = m.unwrap_or_else(|| {
        let m = p.start();
        attr::outers(p);
        m
    });
    if !p.at_ts(EXPR_FIRST) {
        p.err_recover("expected expression", atom::EXPR_RECOVERY_SET);
        m.abandon(p);
        return None;
    }
    let mut lhs = match lhs(p, r) {
        Some((lhs, blocklike)) => {
            let lhs = lhs.extend_to(p, m);
            if r.prefer_stmt && blocklike.is_block() {
                return Some((lhs, BlockLike::Block));
            }
            lhs
        },
        None => {
            m.abandon(p);
            return None;
        },
    };
    loop {
        let is_range = p.at(T![..]) || p.at(T![..=]);
        let (op_bp, op, associativity) = current_op(p);
        if op_bp < bp {
            break;
        }
        if p.at(T![as]) {
            lhs = cast_expr(p, lhs);
            continue;
        }
        let m = lhs.precede(p);
        p.bump(op);
        r = Restrictions {
            prefer_stmt: false,
            ..r
        };
        if is_range {
            let has_trailing_expression = p.at_ts(EXPR_FIRST) && !(r.forbid_structs && p.at(T!['{']));
            if !has_trailing_expression {
                lhs = m.complete(p, RANGE_EXPR);
                break;
            }
        }
        let op_bp = match associativity {
            Associativity::Left => op_bp + 1,
            Associativity::Right => op_bp,
        };
        expr_bp(
            p,
            None,
            Restrictions {
                prefer_stmt: false,
                ..r
            },
            op_bp,
        );
        lhs = m.complete(p, if is_range { RANGE_EXPR } else { BIN_EXPR });
    }
    Some((lhs, BlockLike::NotBlock))
}
const LHS_FIRST: TokenSet = atom::ATOM_EXPR_FIRST.union(TokenSet::new(&[T![&], T![*], T![!], T![.], T![-], T![_]]));
fn lhs(p: &mut Parser<'_>, r: Restrictions) -> Option<(CompletedMarker, BlockLike)> {
    let m;
    let kind = match p.current() {
        T![&] => {
            m = p.start();
            p.bump(T![&]);
            if p.at_contextual_kw(T![raw]) && (p.nth_at(1, T![mut]) || p.nth_at(1, T![const])) {
                p.bump_remap(T![raw]);
                p.bump_any();
            } else {
                p.eat(T![mut]);
            }
            REF_EXPR
        },
        T![*] | T![!] | T![-] => {
            m = p.start();
            p.bump_any();
            PREFIX_EXPR
        },
        _ => {
            for op in [T![..=], T![..]] {
                if p.at(op) {
                    m = p.start();
                    p.bump(op);
                    if p.at_ts(EXPR_FIRST) && !(r.forbid_structs && p.at(T!['{'])) {
                        expr_bp(p, None, r, 2);
                    }
                    let cm = m.complete(p, RANGE_EXPR);
                    return Some((cm, BlockLike::NotBlock));
                }
            }
            let (lhs, blocklike) = atom::atom_expr(p, r)?;
            let (cm, block_like) = postfix_expr(p, lhs, blocklike, !(r.prefer_stmt && blocklike.is_block()));
            return Some((cm, block_like));
        },
    };
    expr_bp(p, None, r, 255);
    let cm = m.complete(p, kind);
    Some((cm, BlockLike::NotBlock))
}
fn postfix_expr(
    p: &mut Parser<'_>,
    mut lhs: CompletedMarker,
    mut block_like: BlockLike,
    mut allow_calls: bool,
) -> (CompletedMarker, BlockLike) {
    loop {
        lhs = match p.current() {
            T!['('] if allow_calls => call_expr(p, lhs),
            T!['['] if allow_calls => index_expr(p, lhs),
            T![.] => match postfix_dot_expr::<false>(p, lhs) {
                Ok(it) => it,
                Err(it) => {
                    lhs = it;
                    break;
                },
            },
            T![?] => try_expr(p, lhs),
            _ => break,
        };
        allow_calls = true;
        block_like = BlockLike::NotBlock;
    }
    (lhs, block_like)
}
fn postfix_dot_expr<const FLOAT_RECOVERY: bool>(
    p: &mut Parser<'_>,
    lhs: CompletedMarker,
) -> Result<CompletedMarker, CompletedMarker> {
    if !FLOAT_RECOVERY {
        assert!(p.at(T![.]));
    }
    let nth1 = if FLOAT_RECOVERY { 0 } else { 1 };
    let nth2 = if FLOAT_RECOVERY { 1 } else { 2 };
    if p.nth(nth1) == IDENT && (p.nth(nth2) == T!['('] || p.nth_at(nth2, T![::])) {
        return Ok(method_call_expr::<FLOAT_RECOVERY>(p, lhs));
    }
    if p.nth(nth1) == T![await] {
        let m = lhs.precede(p);
        if !FLOAT_RECOVERY {
            p.bump(T![.]);
        }
        p.bump(T![await]);
        return Ok(m.complete(p, AWAIT_EXPR));
    }
    if p.at(T![..=]) || p.at(T![..]) {
        return Err(lhs);
    }
    field_expr::<FLOAT_RECOVERY>(p, lhs)
}
fn call_expr(p: &mut Parser<'_>, lhs: CompletedMarker) -> CompletedMarker {
    assert!(p.at(T!['(']));
    let m = lhs.precede(p);
    arg_list(p);
    m.complete(p, CALL_EXPR)
}
fn index_expr(p: &mut Parser<'_>, lhs: CompletedMarker) -> CompletedMarker {
    assert!(p.at(T!['[']));
    let m = lhs.precede(p);
    p.bump(T!['[']);
    expr(p);
    p.expect(T![']']);
    m.complete(p, INDEX_EXPR)
}
fn method_call_expr<const FLOAT_RECOVERY: bool>(p: &mut Parser<'_>, lhs: CompletedMarker) -> CompletedMarker {
    if FLOAT_RECOVERY {
        assert!(p.nth(0) == IDENT && (p.nth(1) == T!['('] || p.nth_at(1, T![::])));
    } else {
        assert!(p.at(T![.]) && p.nth(1) == IDENT && (p.nth(2) == T!['('] || p.nth_at(2, T![::])));
    }
    let m = lhs.precede(p);
    if !FLOAT_RECOVERY {
        p.bump(T![.]);
    }
    name_ref(p);
    generic::opt_args(p, true);
    if p.at(T!['(']) {
        arg_list(p);
    }
    m.complete(p, METHOD_CALL_EXPR)
}
fn field_expr<const FLOAT_RECOVERY: bool>(
    p: &mut Parser<'_>,
    lhs: CompletedMarker,
) -> Result<CompletedMarker, CompletedMarker> {
    if !FLOAT_RECOVERY {
        assert!(p.at(T![.]));
    }
    let m = lhs.precede(p);
    if !FLOAT_RECOVERY {
        p.bump(T![.]);
    }
    if p.at(IDENT) || p.at(INT_NUMBER) {
        name_ref_or_idx(p);
    } else if p.at(FLOAT_NUMBER) {
        return match p.split_float(m) {
            (true, m) => {
                let lhs = m.complete(p, FIELD_EXPR);
                postfix_dot_expr::<true>(p, lhs)
            },
            (false, m) => Ok(m.complete(p, FIELD_EXPR)),
        };
    } else {
        p.error("expected field name or number");
    }
    Ok(m.complete(p, FIELD_EXPR))
}
fn try_expr(p: &mut Parser<'_>, lhs: CompletedMarker) -> CompletedMarker {
    assert!(p.at(T![?]));
    let m = lhs.precede(p);
    p.bump(T![?]);
    m.complete(p, TRY_EXPR)
}
fn cast_expr(p: &mut Parser<'_>, lhs: CompletedMarker) -> CompletedMarker {
    assert!(p.at(T![as]));
    let m = lhs.precede(p);
    p.bump(T![as]);
    ty::no_bounds(p);
    m.complete(p, CAST_EXPR)
}
fn arg_list(p: &mut Parser<'_>) {
    assert!(p.at(T!['(']));
    let m = p.start();
    delimited(
        p,
        T!['('],
        T![')'],
        T![,],
        EXPR_FIRST.union(attr::FIRST),
        |p: &mut Parser<'_>| expr(p).is_some(),
    );
    m.complete(p, ARG_LIST);
}
fn path_expr(p: &mut Parser<'_>, r: Restrictions) -> (CompletedMarker, BlockLike) {
    assert!(path::is_start(p));
    let m = p.start();
    path::for_expr(p);
    match p.current() {
        T!['{'] if !r.forbid_structs => {
            record_expr_field_list(p);
            (m.complete(p, RECORD_EXPR), BlockLike::NotBlock)
        },
        T![!] if !p.at(T![!=]) => {
            let block_like = item::macro_call_after_excl(p);
            (m.complete(p, MACRO_CALL).precede(p).complete(p, MACRO_EXPR), block_like)
        },
        _ => (m.complete(p, PATH_EXPR), BlockLike::NotBlock),
    }
}
pub fn record_expr_field_list(p: &mut Parser<'_>) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump(T!['{']);
    while !p.at(EOF) && !p.at(T!['}']) {
        let m = p.start();
        attr::outers(p);
        match p.current() {
            IDENT | INT_NUMBER => {
                if p.nth_at(1, T![::]) {
                    m.abandon(p);
                    p.expect(T![..]);
                    expr(p);
                } else {
                    if p.nth_at(1, T![:]) || p.nth_at(1, T![..]) {
                        name_ref_or_idx(p);
                        p.expect(T![:]);
                    }
                    expr(p);
                    m.complete(p, RECORD_EXPR_FIELD);
                }
            },
            T![.] if p.at(T![..]) => {
                m.abandon(p);
                p.bump(T![..]);
                if !p.at(T!['}']) {
                    expr(p);
                }
            },
            T!['{'] => {
                err_block(p, "expected a field");
                m.abandon(p);
            },
            _ => {
                p.err_and_bump("expected identifier");
                m.abandon(p);
            },
        }
        if !p.at(T!['}']) {
            p.expect(T![,]);
        }
    }
    p.expect(T!['}']);
    m.complete(p, RECORD_EXPR_FIELD_LIST);
}
