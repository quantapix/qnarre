use super::*;

mod atom {
    use super::*;
    pub const LIT_FIRST: TokenSet = TokenSet::new(&[
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
    pub fn literal(x: &mut Parser<'_>) -> Option<CompletedMarker> {
        if !x.at_ts(LIT_FIRST) {
            return None;
        }
        let y = x.start();
        x.bump_any();
        Some(y.complete(x, LITERAL))
    }
    pub const ATOM_FIRST: TokenSet = LIT_FIRST.union(path::FIRST).union(TokenSet::new(&[
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
    pub const RECOVERY_SET: TokenSet = TokenSet::new(&[T![')'], T![']']]);
    pub fn atom_expr(x: &mut Parser<'_>, r: Restrictions) -> Option<(CompletedMarker, BlockLike)> {
        if let Some(x) = literal(x) {
            return Some((x, BlockLike::NotBlock));
        }
        if path::is_start(x) {
            return Some(path_expr(x, r));
        }
        let la = x.nth(1);
        let done = match x.current() {
            T!['('] => tuple_expr(x),
            T!['['] => array_expr(x),
            T![if] => if_expr(x),
            T![let] => let_expr(x),
            T![_] => {
                let y = x.start();
                x.bump(T![_]);
                y.complete(x, UNDERSCORE_EXPR)
            },
            T![loop] => loop_expr(x, None),
            T![box] => box_expr(x, None),
            T![while] => while_expr(x, None),
            T![try] => try_block_expr(x, None),
            T![match] => match_expr(x),
            T![return] => return_expr(x),
            T![yield] => yield_expr(x),
            T![do] if x.nth_at_contextual_kw(1, T![yeet]) => yeet_expr(x),
            T![continue] => continue_expr(x),
            T![break] => break_expr(x, r),
            LIFETIME_IDENT if la == T![:] => {
                let y = x.start();
                label(x);
                match x.current() {
                    T![loop] => loop_expr(x, Some(y)),
                    T![for] => for_expr(x, Some(y)),
                    T![while] => while_expr(x, Some(y)),
                    T!['{'] => {
                        stmt_list(x);
                        y.complete(x, BLOCK_EXPR)
                    },
                    _ => {
                        x.error("expected a loop or block");
                        y.complete(x, ERROR);
                        return None;
                    },
                }
            },
            T![const] | T![unsafe] | T![async] if la == T!['{'] => {
                let y = x.start();
                x.bump_any();
                stmt_list(x);
                y.complete(x, BLOCK_EXPR)
            },
            T![async] if la == T![move] && x.nth(2) == T!['{'] => {
                let y = x.start();
                x.bump(T![async]);
                x.eat(T![move]);
                stmt_list(x);
                y.complete(x, BLOCK_EXPR)
            },
            T!['{'] => {
                let y = x.start();
                stmt_list(x);
                y.complete(x, BLOCK_EXPR)
            },
            T![const] | T![static] | T![async] | T![move] | T![|] => closure_expr(x),
            T![for] if la == T![<] => closure_expr(x),
            T![for] => for_expr(x, None),
            _ => {
                x.err_and_bump("expected expression");
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
    fn tuple_expr(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T!['(']));
        let y = x.start();
        x.expect(T!['(']);
        let mut has_comma = false;
        let mut has_expr = false;
        if x.eat(T![,]) {
            x.error("expected expression");
            has_comma = true;
        }
        while !x.at(EOF) && !x.at(T![')']) {
            has_expr = true;
            if expr(x).is_none() {
                break;
            }
            if !x.at(T![')']) {
                has_comma = true;
                x.expect(T![,]);
            }
        }
        x.expect(T![')']);
        y.complete(x, if has_expr && !has_comma { PAREN_EXPR } else { TUPLE_EXPR })
    }
    fn array_expr(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T!['[']));
        let y = x.start();
        let mut n = 0u32;
        let mut has_semi = false;
        x.bump(T!['[']);
        while !x.at(EOF) && !x.at(T![']']) {
            n += 1;
            if expr(x).is_none() {
                break;
            }
            if n == 1 && x.eat(T![;]) {
                has_semi = true;
                continue;
            }
            if has_semi || !x.at(T![']']) && !x.expect(T![,]) {
                break;
            }
        }
        x.expect(T![']']);
        y.complete(x, ARRAY_EXPR)
    }
    fn closure_expr(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(match x.current() {
            T![const] | T![static] | T![async] | T![move] | T![|] => true,
            T![for] => x.nth(1) == T![<],
            _ => false,
        });
        let y = x.start();
        if x.at(T![for]) {
            ty::for_binder(x);
        }
        x.eat(T![const]);
        x.eat(T![static]);
        x.eat(T![async]);
        x.eat(T![move]);
        if !x.at(T![|]) {
            x.error("expected `|`");
            return y.complete(x, CLOSURE_EXPR);
        }
        param::closure(x);
        if is_opt_ret_type(x) {
            block_expr(x);
        } else if x.at_ts(EXPR_FIRST) {
            expr(x);
        } else {
            x.error("expected expression");
        }
        y.complete(x, CLOSURE_EXPR)
    }
    fn if_expr(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T![if]));
        let y = x.start();
        x.bump(T![if]);
        expr_no_struct(x);
        block_expr(x);
        if x.at(T![else]) {
            x.bump(T![else]);
            if x.at(T![if]) {
                if_expr(x);
            } else {
                block_expr(x);
            }
        }
        y.complete(x, IF_EXPR)
    }
    fn label(x: &mut Parser<'_>) {
        assert!(x.at(LIFETIME_IDENT) && x.nth(1) == T![:]);
        let y = x.start();
        lifetime(x);
        x.bump_any();
        y.complete(x, LABEL);
    }
    fn loop_expr(x: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
        assert!(x.at(T![loop]));
        let y = m.unwrap_or_else(|| x.start());
        x.bump(T![loop]);
        block_expr(x);
        y.complete(x, LOOP_EXPR)
    }
    fn while_expr(x: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
        assert!(x.at(T![while]));
        let y = m.unwrap_or_else(|| x.start());
        x.bump(T![while]);
        expr_no_struct(x);
        block_expr(x);
        y.complete(x, WHILE_EXPR)
    }
    fn for_expr(x: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
        assert!(x.at(T![for]));
        let y = m.unwrap_or_else(|| x.start());
        x.bump(T![for]);
        pattern::one(x);
        x.expect(T![in]);
        expr_no_struct(x);
        block_expr(x);
        y.complete(x, FOR_EXPR)
    }
    fn let_expr(x: &mut Parser<'_>) -> CompletedMarker {
        let y = x.start();
        x.bump(T![let]);
        pattern::top(x);
        x.expect(T![=]);
        expr_let(x);
        y.complete(x, LET_EXPR)
    }
    fn match_expr(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T![match]));
        let y = x.start();
        x.bump(T![match]);
        expr_no_struct(x);
        if x.at(T!['{']) {
            match_arm_list(x);
        } else {
            x.error("expected `{`");
        }
        y.complete(x, MATCH_EXPR)
    }
    pub fn match_arm_list(x: &mut Parser<'_>) {
        assert!(x.at(T!['{']));
        let y = x.start();
        x.eat(T!['{']);
        attr::inners(x);
        while !x.at(EOF) && !x.at(T!['}']) {
            if x.at(T!['{']) {
                err_block(x, "expected match arm");
                continue;
            }
            match_arm(x);
        }
        x.expect(T!['}']);
        y.complete(x, MATCH_ARM_LIST);
    }
    fn match_arm(x: &mut Parser<'_>) {
        let y = x.start();
        attr::outers(x);
        pattern::top_r(x, TokenSet::EMPTY);
        if x.at(T![if]) {
            match_guard(x);
        }
        x.expect(T![=>]);
        let blocklike = match expr_stmt(x, None) {
            Some((_, x)) => x,
            None => BlockLike::NotBlock,
        };
        if !x.eat(T![,]) && !blocklike.is_block() && !x.at(T!['}']) {
            x.error("expected `,`");
        }
        y.complete(x, MATCH_ARM);
    }
    fn match_guard(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T![if]));
        let y = x.start();
        x.bump(T![if]);
        expr(x);
        y.complete(x, MATCH_GUARD)
    }
    pub fn block_expr(x: &mut Parser<'_>) {
        if !x.at(T!['{']) {
            x.error("expected a block");
            return;
        }
        let y = x.start();
        stmt_list(x);
        y.complete(x, BLOCK_EXPR);
    }
    fn stmt_list(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T!['{']));
        let y = x.start();
        x.bump(T!['{']);
        expr_block_contents(x);
        x.expect(T!['}']);
        y.complete(x, STMT_LIST)
    }
    fn return_expr(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T![return]));
        let y = x.start();
        x.bump(T![return]);
        if x.at_ts(EXPR_FIRST) {
            expr(x);
        }
        y.complete(x, RETURN_EXPR)
    }
    fn yield_expr(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T![yield]));
        let y = x.start();
        x.bump(T![yield]);
        if x.at_ts(EXPR_FIRST) {
            expr(x);
        }
        y.complete(x, YIELD_EXPR)
    }
    fn yeet_expr(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T![do]));
        assert!(x.nth_at_contextual_kw(1, T![yeet]));
        let y = x.start();
        x.bump(T![do]);
        x.bump_remap(T![yeet]);
        if x.at_ts(EXPR_FIRST) {
            expr(x);
        }
        y.complete(x, YEET_EXPR)
    }
    fn continue_expr(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T![continue]));
        let y = x.start();
        x.bump(T![continue]);
        if x.at(LIFETIME_IDENT) {
            lifetime(x);
        }
        y.complete(x, CONTINUE_EXPR)
    }
    fn break_expr(x: &mut Parser<'_>, r: Restrictions) -> CompletedMarker {
        assert!(x.at(T![break]));
        let y = x.start();
        x.bump(T![break]);
        if x.at(LIFETIME_IDENT) {
            lifetime(x);
        }
        if x.at_ts(EXPR_FIRST) && !(r.no_structs && x.at(T!['{'])) {
            expr(x);
        }
        y.complete(x, BREAK_EXPR)
    }
    fn try_block_expr(x: &mut Parser<'_>, y: Option<Marker>) -> CompletedMarker {
        assert!(x.at(T![try]));
        let y = y.unwrap_or_else(|| x.start());
        if x.nth_at(1, T![!]) {
            let mac_call = x.start();
            let path = x.start();
            let segment = x.start();
            let name_ref = x.start();
            x.bump_remap(IDENT);
            name_ref.complete(x, NAME_REF);
            segment.complete(x, PATH_SEGMENT);
            path.complete(x, PATH);
            let _block_like = item::macro_call_after_excl(x);
            mac_call.complete(x, MACRO_CALL);
            return y.complete(x, MACRO_EXPR);
        }
        x.bump(T![try]);
        if x.at(T!['{']) {
            stmt_list(x);
        } else {
            x.error("expected a block");
        }
        y.complete(x, BLOCK_EXPR)
    }
    fn box_expr(x: &mut Parser<'_>, y: Option<Marker>) -> CompletedMarker {
        assert!(x.at(T![box]));
        let y = y.unwrap_or_else(|| x.start());
        x.bump(T![box]);
        if x.at_ts(EXPR_FIRST) {
            expr(x);
        }
        y.complete(x, BOX_EXPR)
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
pub use atom::{block_expr, const_arg, literal, match_arm_list, LIT_FIRST};

#[derive(PartialEq, Eq)]
pub enum Semicolon {
    Req,
    Opt,
    Forbidden,
}

const EXPR_FIRST: TokenSet = LHS_FIRST;

pub fn expr(x: &mut Parser<'_>) -> Option<CompletedMarker> {
    let r = Restrictions {
        no_structs: false,
        prefer_stmt: false,
    };
    expr_bp(x, None, r, 1).map(|(m, _)| m)
}
pub fn expr_stmt(x: &mut Parser<'_>, y: Option<Marker>) -> Option<(CompletedMarker, BlockLike)> {
    let r = Restrictions {
        no_structs: false,
        prefer_stmt: true,
    };
    expr_bp(x, y, r, 1)
}
fn expr_no_struct(x: &mut Parser<'_>) {
    let r = Restrictions {
        no_structs: true,
        prefer_stmt: false,
    };
    expr_bp(x, None, r, 1);
}
fn expr_let(x: &mut Parser<'_>) {
    let r = Restrictions {
        no_structs: true,
        prefer_stmt: false,
    };
    expr_bp(x, None, r, 5);
}
pub fn stmt(x: &mut Parser<'_>, semi: Semicolon) {
    if x.eat(T![;]) {
        return;
    }
    let y = x.start();
    attr::outers(x);
    if x.at(T![let]) {
        let_stmt(x, y, semi);
        return;
    }
    let y = match item::opt_item(x, y) {
        Ok(()) => return,
        Err(x) => x,
    };
    if !x.at_ts(EXPR_FIRST) {
        x.err_and_bump("expected expression, item or let statement");
        y.abandon(x);
        return;
    }
    if let Some((cm, blocklike)) = expr_stmt(x, Some(y)) {
        use Semicolon::*;
        if !(x.at(T!['}']) || (semi != Req && x.at(EOF))) {
            let y = cm.precede(x);
            match semi {
                Req => {
                    if blocklike.is_block() {
                        x.eat(T![;]);
                    } else {
                        x.expect(T![;]);
                    }
                },
                Opt => {
                    x.eat(T![;]);
                },
                Forbidden => (),
            }
            y.complete(x, EXPR_STMT);
        }
    }
    fn let_stmt(x: &mut Parser<'_>, y: Marker, semi: Semicolon) {
        x.bump(T![let]);
        pattern::one(x);
        if x.at(T![:]) {
            ty::ascription(x);
        }
        let mut after_eq: Option<CompletedMarker> = None;
        if x.eat(T![=]) {
            after_eq = expr::expr(x);
        }
        if x.at(T![else]) {
            if let Some(expr) = after_eq {
                if BlockLike::is_blocklike(expr.kind()) {
                    x.error("right curly brace `}` before `else` in a `let...else` statement not allowed")
                }
            }
            let y = x.start();
            x.bump(T![else]);
            block_expr(x);
            y.complete(x, LET_ELSE);
        }
        use Semicolon::*;
        match semi {
            Forbidden => (),
            Opt => {
                x.eat(T![;]);
            },
            Req => {
                x.expect(T![;]);
            },
        }
        y.complete(x, LET_STMT);
    }
}
pub fn expr_block_contents(x: &mut Parser<'_>) {
    attr::inners(x);
    while !x.at(EOF) && !x.at(T!['}']) {
        stmt(x, Semicolon::Req);
    }
}

#[derive(Clone, Copy)]
struct Restrictions {
    no_structs: bool,
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
fn expr_bp(x: &mut Parser<'_>, y: Option<Marker>, mut r: Restrictions, bp: u8) -> Option<(CompletedMarker, BlockLike)> {
    let y = y.unwrap_or_else(|| {
        let y = x.start();
        attr::outers(x);
        y
    });
    if !x.at_ts(EXPR_FIRST) {
        x.err_recover("expected expression", atom::RECOVERY_SET);
        y.abandon(x);
        return None;
    }
    let mut lhs = match lhs(x, r) {
        Some((lhs, blocklike)) => {
            let lhs = lhs.extend_to(x, y);
            if r.prefer_stmt && blocklike.is_block() {
                return Some((lhs, BlockLike::Block));
            }
            lhs
        },
        None => {
            y.abandon(x);
            return None;
        },
    };
    loop {
        let is_range = x.at(T![..]) || x.at(T![..=]);
        let (op_bp, op, assoc) = current_op(x);
        if op_bp < bp {
            break;
        }
        if x.at(T![as]) {
            lhs = cast_expr(x, lhs);
            continue;
        }
        let y = lhs.precede(x);
        x.bump(op);
        r = Restrictions {
            prefer_stmt: false,
            ..r
        };
        if is_range {
            let has_trailing = x.at_ts(EXPR_FIRST) && !(r.no_structs && x.at(T!['{']));
            if !has_trailing {
                lhs = y.complete(x, RANGE_EXPR);
                break;
            }
        }
        use Associativity::*;
        let op_bp = match assoc {
            Left => op_bp + 1,
            Right => op_bp,
        };
        expr_bp(
            x,
            None,
            Restrictions {
                prefer_stmt: false,
                ..r
            },
            op_bp,
        );
        lhs = y.complete(x, if is_range { RANGE_EXPR } else { BIN_EXPR });
    }
    Some((lhs, BlockLike::NotBlock))
}
const LHS_FIRST: TokenSet = atom::ATOM_FIRST.union(TokenSet::new(&[T![&], T![*], T![!], T![.], T![-], T![_]]));
fn lhs(x: &mut Parser<'_>, r: Restrictions) -> Option<(CompletedMarker, BlockLike)> {
    let y;
    let kind = match x.current() {
        T![&] => {
            y = x.start();
            x.bump(T![&]);
            if x.at_contextual_kw(T![raw]) && (x.nth_at(1, T![mut]) || x.nth_at(1, T![const])) {
                x.bump_remap(T![raw]);
                x.bump_any();
            } else {
                x.eat(T![mut]);
            }
            REF_EXPR
        },
        T![*] | T![!] | T![-] => {
            y = x.start();
            x.bump_any();
            PREFIX_EXPR
        },
        _ => {
            for op in [T![..=], T![..]] {
                if x.at(op) {
                    y = x.start();
                    x.bump(op);
                    if x.at_ts(EXPR_FIRST) && !(r.no_structs && x.at(T!['{'])) {
                        expr_bp(x, None, r, 2);
                    }
                    let cm = y.complete(x, RANGE_EXPR);
                    return Some((cm, BlockLike::NotBlock));
                }
            }
            let (lhs, blocklike) = atom::atom_expr(x, r)?;
            let (cm, block_like) = postfix_expr(x, lhs, blocklike, !(r.prefer_stmt && blocklike.is_block()));
            return Some((cm, block_like));
        },
    };
    expr_bp(x, None, r, 255);
    let cm = y.complete(x, kind);
    Some((cm, BlockLike::NotBlock))
}
fn postfix_expr(
    x: &mut Parser<'_>,
    mut y: CompletedMarker,
    mut blocklike: BlockLike,
    mut calls: bool,
) -> (CompletedMarker, BlockLike) {
    loop {
        y = match x.current() {
            T!['('] if calls => call_expr(x, y),
            T!['['] if calls => index_expr(x, y),
            T![.] => match postfix_dot_expr::<false>(x, y) {
                Ok(x) => x,
                Err(x) => {
                    y = x;
                    break;
                },
            },
            T![?] => try_expr(x, y),
            _ => break,
        };
        calls = true;
        blocklike = BlockLike::NotBlock;
    }
    (y, blocklike)
}
fn postfix_dot_expr<const FLOAT_RECOVERY: bool>(
    x: &mut Parser<'_>,
    y: CompletedMarker,
) -> Result<CompletedMarker, CompletedMarker> {
    if !FLOAT_RECOVERY {
        assert!(x.at(T![.]));
    }
    let nth1 = if FLOAT_RECOVERY { 0 } else { 1 };
    let nth2 = if FLOAT_RECOVERY { 1 } else { 2 };
    if x.nth(nth1) == IDENT && (x.nth(nth2) == T!['('] || x.nth_at(nth2, T![::])) {
        return Ok(method_call_expr::<FLOAT_RECOVERY>(x, y));
    }
    if x.nth(nth1) == T![await] {
        let m = y.precede(x);
        if !FLOAT_RECOVERY {
            x.bump(T![.]);
        }
        x.bump(T![await]);
        return Ok(m.complete(x, AWAIT_EXPR));
    }
    if x.at(T![..=]) || x.at(T![..]) {
        return Err(y);
    }
    field_expr::<FLOAT_RECOVERY>(x, y)
}
fn call_expr(x: &mut Parser<'_>, y: CompletedMarker) -> CompletedMarker {
    assert!(x.at(T!['(']));
    let y = y.precede(x);
    arg_list(x);
    y.complete(x, CALL_EXPR)
}
fn index_expr(x: &mut Parser<'_>, y: CompletedMarker) -> CompletedMarker {
    assert!(x.at(T!['[']));
    let y = y.precede(x);
    x.bump(T!['[']);
    expr(x);
    x.expect(T![']']);
    y.complete(x, INDEX_EXPR)
}
fn method_call_expr<const FLOAT_RECOVERY: bool>(x: &mut Parser<'_>, y: CompletedMarker) -> CompletedMarker {
    if FLOAT_RECOVERY {
        assert!(x.nth(0) == IDENT && (x.nth(1) == T!['('] || x.nth_at(1, T![::])));
    } else {
        assert!(x.at(T![.]) && x.nth(1) == IDENT && (x.nth(2) == T!['('] || x.nth_at(2, T![::])));
    }
    let y = y.precede(x);
    if !FLOAT_RECOVERY {
        x.bump(T![.]);
    }
    name_ref(x);
    generic::opt_args(x, true);
    if x.at(T!['(']) {
        arg_list(x);
    }
    y.complete(x, METHOD_CALL_EXPR)
}
fn field_expr<const FLOAT_RECOVERY: bool>(
    x: &mut Parser<'_>,
    y: CompletedMarker,
) -> Result<CompletedMarker, CompletedMarker> {
    if !FLOAT_RECOVERY {
        assert!(x.at(T![.]));
    }
    let y = y.precede(x);
    if !FLOAT_RECOVERY {
        x.bump(T![.]);
    }
    if x.at(IDENT) || x.at(INT_NUMBER) {
        name_ref_or_idx(x);
    } else if x.at(FLOAT_NUMBER) {
        return match x.split_float(y) {
            (true, y) => {
                let y = y.complete(x, FIELD_EXPR);
                postfix_dot_expr::<true>(x, y)
            },
            (false, y) => Ok(y.complete(x, FIELD_EXPR)),
        };
    } else {
        x.error("expected field name or number");
    }
    Ok(y.complete(x, FIELD_EXPR))
}
fn try_expr(x: &mut Parser<'_>, y: CompletedMarker) -> CompletedMarker {
    assert!(x.at(T![?]));
    let y = y.precede(x);
    x.bump(T![?]);
    y.complete(x, TRY_EXPR)
}
fn cast_expr(x: &mut Parser<'_>, y: CompletedMarker) -> CompletedMarker {
    assert!(x.at(T![as]));
    let y = y.precede(x);
    x.bump(T![as]);
    ty::no_bounds(x);
    y.complete(x, CAST_EXPR)
}
fn arg_list(x: &mut Parser<'_>) {
    assert!(x.at(T!['(']));
    let y = x.start();
    delimited(
        x,
        T!['('],
        T![')'],
        T![,],
        EXPR_FIRST.union(attr::FIRST),
        |x: &mut Parser<'_>| expr(x).is_some(),
    );
    y.complete(x, ARG_LIST);
}
fn path_expr(x: &mut Parser<'_>, r: Restrictions) -> (CompletedMarker, BlockLike) {
    assert!(path::is_start(x));
    let y = x.start();
    path::for_expr(x);
    match x.current() {
        T!['{'] if !r.no_structs => {
            record_expr_field_list(x);
            (y.complete(x, RECORD_EXPR), BlockLike::NotBlock)
        },
        T![!] if !x.at(T![!=]) => {
            let blocklike = item::macro_call_after_excl(x);
            (y.complete(x, MACRO_CALL).precede(x).complete(x, MACRO_EXPR), blocklike)
        },
        _ => (y.complete(x, PATH_EXPR), BlockLike::NotBlock),
    }
}
pub fn record_expr_field_list(x: &mut Parser<'_>) {
    assert!(x.at(T!['{']));
    let y = x.start();
    x.bump(T!['{']);
    while !x.at(EOF) && !x.at(T!['}']) {
        let y = x.start();
        attr::outers(x);
        match x.current() {
            IDENT | INT_NUMBER => {
                if x.nth_at(1, T![::]) {
                    y.abandon(x);
                    x.expect(T![..]);
                    expr(x);
                } else {
                    if x.nth_at(1, T![:]) || x.nth_at(1, T![..]) {
                        name_ref_or_idx(x);
                        x.expect(T![:]);
                    }
                    expr(x);
                    y.complete(x, RECORD_EXPR_FIELD);
                }
            },
            T![.] if x.at(T![..]) => {
                y.abandon(x);
                x.bump(T![..]);
                if !x.at(T!['}']) {
                    expr(x);
                }
            },
            T!['{'] => {
                err_block(x, "expected a field");
                y.abandon(x);
            },
            _ => {
                x.err_and_bump("expected identifier");
                y.abandon(x);
            },
        }
        if !x.at(T!['}']) {
            x.expect(T![,]);
        }
    }
    x.expect(T!['}']);
    y.complete(x, RECORD_EXPR_FIELD_LIST);
}
