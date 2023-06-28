use crate::{
    parser::{CompletedMarker, Marker, Parser},
    SyntaxKind::{self, *},
    TokenSet, T,
};

pub fn reparser(
    x: SyntaxKind,
    first_child: Option<SyntaxKind>,
    parent: Option<SyntaxKind>,
) -> Option<fn(&mut Parser<'_>)> {
    let y = match x {
        BLOCK_EXPR => expr::block_expr,
        RECORD_FIELD_LIST => item::record_field_list,
        RECORD_EXPR_FIELD_LIST => item::record_expr_field_list,
        VARIANT_LIST => item::variant_list,
        MATCH_ARM_LIST => item::match_arm_list,
        USE_TREE_LIST => item::use_tree_list,
        EXTERN_ITEM_LIST => item::extern_item_list,
        TOKEN_TREE if first_child? == T!['{'] => item::token_tree,
        ASSOC_ITEM_LIST => match parent? {
            IMPL | TRAIT => item::assoc_item_list,
            _ => return None,
        },
        ITEM_LIST => item::item_list,
        _ => return None,
    };
    Some(y)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockLike {
    Block,
    NotBlock,
}
impl BlockLike {
    fn is_block(self) -> bool {
        self == BlockLike::Block
    }
    fn is_blocklike(kind: SyntaxKind) -> bool {
        matches!(
            kind,
            BLOCK_EXPR | IF_EXPR | WHILE_EXPR | FOR_EXPR | LOOP_EXPR | MATCH_EXPR
        )
    }
}

const VIS_FIRST: TokenSet = TokenSet::new(&[T![pub], T![crate]]);

fn opt_vis(x: &mut Parser<'_>, in_tuple_field: bool) -> bool {
    match x.current() {
        T![pub] => {
            let y = x.start();
            x.bump(T![pub]);
            if x.at(T!['(']) {
                match x.nth(1) {
                    T![crate] | T![self] | T![super] | T![ident] | T![')'] if x.nth(2) != T![:] => {
                        if !(in_tuple_field && matches!(x.nth(1), T![ident] | T![')'])) {
                            x.bump(T!['(']);
                            path::for_use(x);
                            x.expect(T![')']);
                        }
                    },
                    T![in] => {
                        x.bump(T!['(']);
                        x.bump(T![in]);
                        path::for_use(x);
                        x.expect(T![')']);
                    },
                    _ => {},
                }
            }
            y.complete(x, VISIBILITY);
            true
        },
        T![crate] => {
            if x.nth_at(1, T![::]) {
                return false;
            }
            let y = x.start();
            x.bump(T![crate]);
            y.complete(x, VISIBILITY);
            true
        },
        _ => false,
    }
}
fn opt_rename(x: &mut Parser<'_>) {
    if x.at(T![as]) {
        let y = x.start();
        x.bump(T![as]);
        if !x.eat(T![_]) {
            name(x);
        }
        y.complete(x, RENAME);
    }
}
fn abi(x: &mut Parser<'_>) {
    assert!(x.at(T![extern]));
    let y = x.start();
    x.bump(T![extern]);
    x.eat(STRING);
    y.complete(x, ABI);
}
fn opt_ret_type(x: &mut Parser<'_>) -> bool {
    if x.at(T![->]) {
        let y = x.start();
        x.bump(T![->]);
        ty::no_bounds(x);
        y.complete(x, RET_TYPE);
        true
    } else {
        false
    }
}
fn name_r(x: &mut Parser<'_>, recovery: TokenSet) {
    if x.at(IDENT) {
        let y = x.start();
        x.bump(IDENT);
        y.complete(x, NAME);
    } else {
        x.err_recover("expected a name", recovery);
    }
}
fn name(x: &mut Parser<'_>) {
    name_r(x, TokenSet::EMPTY);
}
fn name_ref(x: &mut Parser<'_>) {
    if x.at(IDENT) {
        let y = x.start();
        x.bump(IDENT);
        y.complete(x, NAME_REF);
    } else {
        x.err_and_bump("expected identifier");
    }
}
fn name_ref_or_idx(x: &mut Parser<'_>) {
    assert!(x.at(IDENT) || x.at(INT_NUMBER));
    let y = x.start();
    x.bump_any();
    y.complete(x, NAME_REF);
}
fn lifetime(x: &mut Parser<'_>) {
    assert!(x.at(LIFETIME_IDENT));
    let y = x.start();
    x.bump(LIFETIME_IDENT);
    y.complete(x, LIFETIME);
}
fn err_block(x: &mut Parser<'_>, msg: &str) {
    assert!(x.at(T!['{']));
    let y = x.start();
    x.error(msg);
    x.bump(T!['{']);
    expr::expr_block_contents(x);
    x.eat(T!['}']);
    y.complete(x, ERROR);
}
fn delimited(
    x: &mut Parser<'_>,
    bra: SyntaxKind,
    ket: SyntaxKind,
    delim: SyntaxKind,
    first_set: TokenSet,
    mut parser: impl FnMut(&mut Parser<'_>) -> bool,
) {
    x.bump(bra);
    while !x.at(ket) && !x.at(EOF) {
        if !parser(x) {
            break;
        }
        if !x.at(delim) {
            if x.at_ts(first_set) {
                x.error(format!("expected {:?}", delim));
            } else {
                break;
            }
        } else {
            x.bump(delim);
        }
    }
    x.expect(ket);
}

mod attr {
    use super::*;
    pub const FIRST: TokenSet = TokenSet::new(&[T![#]]);
    pub fn meta(x: &mut Parser<'_>) {
        let y = x.start();
        path::for_use(x);
        match x.current() {
            T![=] => {
                x.bump(T![=]);
                if expr::expr(x).is_none() {
                    x.error("expected expression");
                }
            },
            T!['('] | T!['['] | T!['{'] => item::token_tree(x),
            _ => {},
        }
        y.complete(x, META);
    }
    pub fn inners(x: &mut Parser<'_>) {
        while x.at(T![#]) && x.nth(1) == T![!] {
            one(x, true);
        }
    }
    pub fn outers(x: &mut Parser<'_>) {
        while x.at(T![#]) {
            one(x, false);
        }
    }
    fn one(x: &mut Parser<'_>, inner: bool) {
        assert!(x.at(T![#]));
        let y = x.start();
        x.bump(T![#]);
        if inner {
            x.bump(T![!]);
        }
        if x.eat(T!['[']) {
            attr::meta(x);
            if !x.eat(T![']']) {
                x.error("expected `]`");
            }
        } else {
            x.error("expected `[`");
        }
        y.complete(x, ATTR);
    }
}
mod generic {
    use super::*;
    const ARG_FIRST: TokenSet = TokenSet::new(&[
        LIFETIME_IDENT,
        IDENT,
        T!['{'],
        T![true],
        T![false],
        T![-],
        INT_NUMBER,
        FLOAT_NUMBER,
        CHAR,
        BYTE,
        STRING,
        BYTE_STRING,
        C_STRING,
    ])
    .union(ty::FIRST);
    pub fn opt_args(x: &mut Parser<'_>, colon_colon_required: bool) {
        let y;
        if x.at(T![::]) && x.nth(2) == T![<] {
            y = x.start();
            x.bump(T![::]);
        } else if !colon_colon_required && x.at(T![<]) && x.nth(1) != T![=] {
            y = x.start();
        } else {
            return;
        }
        delimited(x, T![<], T![>], T![,], ARG_FIRST, one_arg);
        y.complete(x, GENERIC_ARG_LIST);
    }
    fn one_arg(x: &mut Parser<'_>) -> bool {
        match x.current() {
            LIFETIME_IDENT if !x.nth_at(1, T![+]) => lifetime_arg(x),
            T!['{'] | T![true] | T![false] | T![-] => const_arg(x),
            k if k.is_literal() => const_arg(x),
            IDENT if [T![<], T![=], T![:]].contains(&x.nth(1)) && !x.nth_at(1, T![::]) => {
                let y = x.start();
                name_ref(x);
                opt_args(x, false);
                match x.current() {
                    T![=] => {
                        x.bump_any();
                        if ty::FIRST.contains(x.current()) {
                            ty(x);
                        } else {
                            const_arg(x);
                        }
                        y.complete(x, ASSOC_TYPE_ARG);
                    },
                    T![:] if !x.at(T![::]) => {
                        bounds(x);
                        y.complete(x, ASSOC_TYPE_ARG);
                    },
                    _ => {
                        let y = y.complete(x, PATH_SEGMENT).precede(x).complete(x, PATH);
                        let y = path::for_type_qual(x, y);
                        y.precede(x).complete(x, PATH_TYPE).precede(x).complete(x, TYPE_ARG);
                    },
                }
            },
            IDENT if x.nth_at(1, T!['(']) => {
                let y = x.start();
                name_ref(x);
                param::fn_trait(x);
                if x.at(T![:]) && !x.at(T![::]) {
                    bounds(x);
                    y.complete(x, ASSOC_TYPE_ARG);
                } else {
                    opt_ret_type(x);
                    let y = y.complete(x, PATH_SEGMENT).precede(x).complete(x, PATH);
                    let y = path::for_type_qual(x, y);
                    let y = y.precede(x).complete(x, PATH_TYPE);
                    let y = ty::opt_bounds_as_dyn_trait(x, y);
                    y.precede(x).complete(x, TYPE_ARG);
                }
            },
            _ if x.at_ts(ty::FIRST) => type_arg(x),
            _ => return false,
        }
        true
    }
    fn lifetime_arg(x: &mut Parser<'_>) {
        let y = x.start();
        lifetime(x);
        y.complete(x, LIFETIME_ARG);
    }
    fn const_arg(x: &mut Parser<'_>) {
        let y = x.start();
        expr::const_arg(x);
        y.complete(x, CONST_ARG);
    }
    fn type_arg(x: &mut Parser<'_>) {
        let y = x.start();
        ty(x);
        y.complete(x, TYPE_ARG);
    }

    const PARAM_FIRST: TokenSet = TokenSet::new(&[IDENT, LIFETIME_IDENT, T![const]]);
    pub fn opt_params(x: &mut Parser<'_>) {
        if x.at(T![<]) {
            many_params(x);
        }
    }
    fn many_params(x: &mut Parser<'_>) {
        assert!(x.at(T![<]));
        let y = x.start();
        delimited(x, T![<], T![>], T![,], PARAM_FIRST.union(attr::FIRST), |x| {
            let y = x.start();
            attr::outers(x);
            one_param(x, y)
        });
        y.complete(x, GENERIC_PARAM_LIST);
    }
    fn one_param(x: &mut Parser<'_>, m: Marker) -> bool {
        match x.current() {
            LIFETIME_IDENT => lifetime_param(x, m),
            IDENT => type_param(x, m),
            T![const] => const_param(x, m),
            _ => {
                m.abandon(x);
                x.err_and_bump("expected generic parameter");
                return false;
            },
        }
        true
    }
    fn lifetime_param(x: &mut Parser<'_>, m: Marker) {
        assert!(x.at(LIFETIME_IDENT));
        lifetime(x);
        if x.at(T![:]) {
            lifetime_bounds(x);
        }
        m.complete(x, LIFETIME_PARAM);
    }
    fn lifetime_bounds(x: &mut Parser<'_>) {
        assert!(x.at(T![:]));
        x.bump(T![:]);
        while x.at(LIFETIME_IDENT) {
            lifetime(x);
            if !x.eat(T![+]) {
                break;
            }
        }
    }
    fn const_param(x: &mut Parser<'_>, m: Marker) {
        x.bump(T![const]);
        name(x);
        if x.at(T![:]) {
            ty::ascription(x);
        } else {
            x.error("missing type for const parameter");
        }
        if x.at(T![=]) {
            x.bump(T![=]);
            expr::const_arg(x);
        }
        m.complete(x, CONST_PARAM);
    }
    fn type_param(x: &mut Parser<'_>, m: Marker) {
        assert!(x.at(IDENT));
        name(x);
        if x.at(T![:]) {
            bounds(x);
        }
        if x.at(T![=]) {
            x.bump(T![=]);
            ty(x);
        }
        m.complete(x, TYPE_PARAM);
    }

    pub fn bounds(x: &mut Parser<'_>) {
        assert!(x.at(T![:]));
        x.bump(T![:]);
        bounds_no_colon(x);
    }
    pub fn bounds_no_colon(x: &mut Parser<'_>) {
        let y = x.start();
        bounds_with_marker(x, y);
    }
    pub fn bounds_with_marker(x: &mut Parser<'_>, m: Marker) -> CompletedMarker {
        while type_bound(x) {
            if !x.eat(T![+]) {
                break;
            }
        }
        m.complete(x, TYPE_BOUND_LIST)
    }
    fn type_bound(x: &mut Parser<'_>) -> bool {
        let y = x.start();
        let has_paren = x.eat(T!['(']);
        match x.current() {
            LIFETIME_IDENT => lifetime(x),
            T![for] => ty::for_ty(x, false),
            T![?] if x.nth_at(1, T![for]) => {
                x.bump_any();
                ty::for_ty(x, false)
            },
            current => {
                match current {
                    T![?] => x.bump_any(),
                    T![~] => {
                        x.bump_any();
                        x.expect(T![const]);
                    },
                    _ => (),
                }
                if path::is_use_start(x) {
                    ty::path_with_bounds(x, false);
                } else {
                    y.abandon(x);
                    return false;
                }
            },
        }
        if has_paren {
            x.expect(T![')']);
        }
        y.complete(x, TYPE_BOUND);
        true
    }
    pub fn opt_where_clause(x: &mut Parser<'_>) {
        if !x.at(T![where]) {
            return;
        }
        let y = x.start();
        x.bump(T![where]);
        while is_where_pred(x) {
            where_pred(x);
            let comma = x.eat(T![,]);
            match x.current() {
                T!['{'] | T![;] | T![=] => break,
                _ => (),
            }
            if !comma {
                x.error("expected comma");
            }
        }
        y.complete(x, WHERE_CLAUSE);
        fn is_where_pred(x: &mut Parser<'_>) -> bool {
            match x.current() {
                LIFETIME_IDENT => true,
                T![impl] => false,
                token => ty::FIRST.contains(token),
            }
        }
    }
    fn where_pred(x: &mut Parser<'_>) {
        let y = x.start();
        match x.current() {
            LIFETIME_IDENT => {
                lifetime(x);
                if x.at(T![:]) {
                    bounds(x);
                } else {
                    x.error("expected colon");
                }
            },
            T![impl] => {
                x.error("expected lifetime or type");
            },
            _ => {
                if x.at(T![for]) {
                    ty::for_binder(x);
                }
                ty(x);
                if x.at(T![:]) {
                    bounds(x);
                } else {
                    x.error("expected colon");
                }
            },
        }
        y.complete(x, WHERE_PRED);
    }
}
mod param {
    use super::*;
    pub const FIRST: TokenSet = pattern::PATTERN_FIRST.union(ty::FIRST);
    pub fn fn_def(x: &mut Parser<'_>) {
        many(x, Flavor::FnDef);
    }
    pub fn fn_trait(x: &mut Parser<'_>) {
        many(x, Flavor::FnTrait);
    }
    pub fn fn_ptr(x: &mut Parser<'_>) {
        many(x, Flavor::FnPtr);
    }
    pub fn closure(x: &mut Parser<'_>) {
        many(x, Flavor::Closure);
    }
    #[derive(Debug, Clone, Copy)]
    enum Flavor {
        FnDef,
        FnPtr,
        FnTrait,
        Closure,
    }
    fn many(x: &mut Parser<'_>, flavor: Flavor) {
        use Flavor::*;
        let (bra, ket) = match flavor {
            Closure => (T![|], T![|]),
            FnDef | FnTrait | FnPtr => (T!['('], T![')']),
        };
        let y = x.start();
        x.bump(bra);
        let mut marker = None;
        if let FnDef = flavor {
            let y = x.start();
            attr::outers(x);
            match opt_self(x, y) {
                Ok(()) => {},
                Err(x) => marker = Some(x),
            }
        }
        while !x.at(EOF) && !x.at(ket) {
            let y = match marker.take() {
                Some(x) => x,
                None => {
                    let y = x.start();
                    attr::outers(x);
                    y
                },
            };
            if !x.at_ts(param::FIRST.union(attr::FIRST)) {
                x.error("expected value parameter");
                y.abandon(x);
                break;
            }
            one(x, y, flavor);
            if !x.at(T![,]) {
                if x.at_ts(param::FIRST.union(attr::FIRST)) {
                    x.error("expected `,`");
                } else {
                    break;
                }
            } else {
                x.bump(T![,]);
            }
        }
        if let Some(y) = marker {
            y.abandon(x);
        }
        x.expect(ket);
        y.complete(x, PARAM_LIST);
    }
    fn one(x: &mut Parser<'_>, y: Marker, flavor: Flavor) {
        use Flavor::*;
        match flavor {
            FnDef | FnPtr if x.eat(T![...]) => {},
            FnDef => {
                pattern::pattern(x);
                if !variadic(x) {
                    if x.at(T![:]) {
                        ty::ascription(x);
                    } else {
                        x.error("missing type for function parameter");
                    }
                }
            },
            FnTrait => {
                ty(x);
            },
            FnPtr => {
                if (x.at(IDENT) || x.at(UNDERSCORE)) && x.nth(1) == T![:] && !x.nth_at(1, T![::]) {
                    pattern::pattern_single(x);
                    if !variadic(x) {
                        if x.at(T![:]) {
                            ty::ascription(x);
                        } else {
                            x.error("missing type for function parameter");
                        }
                    }
                } else {
                    ty(x);
                }
            },
            Closure => {
                pattern::pattern_single(x);
                if x.at(T![:]) && !x.at(T![::]) {
                    ty::ascription(x);
                }
            },
        }
        y.complete(x, PARAM);
    }
    fn opt_self(x: &mut Parser<'_>, y: Marker) -> Result<(), Marker> {
        if x.at(T![self]) || x.at(T![mut]) && x.nth(1) == T![self] {
            x.eat(T![mut]);
            self_as_name(x);
            if x.at(T![:]) {
                ty::ascription(x);
            }
        } else {
            let la1 = x.nth(1);
            let la2 = x.nth(2);
            let la3 = x.nth(3);
            if !matches!(
                (x.current(), la1, la2, la3),
                (T![&], T![self], _, _)
                    | (T![&], T![mut] | LIFETIME_IDENT, T![self], _)
                    | (T![&], LIFETIME_IDENT, T![mut], T![self])
            ) {
                return Err(y);
            }
            x.bump(T![&]);
            if x.at(LIFETIME_IDENT) {
                lifetime(x);
            }
            x.eat(T![mut]);
            self_as_name(x);
        }
        y.complete(x, SELF_PARAM);
        if !x.at(T![')']) {
            x.expect(T![,]);
        }
        Ok(())
    }
    fn self_as_name(x: &mut Parser<'_>) {
        let y = x.start();
        x.bump(T![self]);
        y.complete(x, NAME);
    }
    fn variadic(x: &mut Parser<'_>) -> bool {
        if x.at(T![:]) && x.nth_at(1, T![...]) {
            x.bump(T![:]);
            x.bump(T![...]);
            true
        } else {
            false
        }
    }
}
mod path {
    use super::*;
    pub const FIRST: TokenSet = TokenSet::new(&[IDENT, T![self], T![super], T![crate], T![Self], T![:], T![<]]);
    pub fn is_start(x: &Parser<'_>) -> bool {
        is_use_start(x) || x.at(T![<]) || x.at(T![Self])
    }
    pub fn is_use_start(x: &Parser<'_>) -> bool {
        match x.current() {
            IDENT | T![self] | T![super] | T![crate] => true,
            T![:] if x.at(T![::]) => true,
            _ => false,
        }
    }
    pub fn for_use(x: &mut Parser<'_>) {
        one(x, Mode::Use);
    }
    pub fn for_type(x: &mut Parser<'_>) {
        one(x, Mode::Type);
    }
    pub fn for_expr(x: &mut Parser<'_>) {
        one(x, Mode::Expr);
    }
    pub fn for_type_qual(x: &mut Parser<'_>, qual: CompletedMarker) -> CompletedMarker {
        for_qual(x, Mode::Type, qual)
    }
    #[derive(Clone, Copy, Eq, PartialEq)]
    enum Mode {
        Use,
        Type,
        Expr,
    }
    fn one(x: &mut Parser<'_>, mode: Mode) {
        let y = x.start();
        segment(x, mode, true);
        let qual = y.complete(x, PATH);
        for_qual(x, mode, qual);
    }
    fn for_qual(x: &mut Parser<'_>, mode: Mode, mut qual: CompletedMarker) -> CompletedMarker {
        loop {
            let use_tree = mode == Mode::Use && matches!(x.nth(2), T![*] | T!['{']);
            if x.at(T![::]) && !use_tree {
                let y = qual.precede(x);
                x.bump(T![::]);
                segment(x, mode, false);
                let y = y.complete(x, PATH);
                qual = y;
            } else {
                return qual;
            }
        }
    }
    const EXPR_SET: TokenSet = item::ITEM_RECOVERY_SET.union(TokenSet::new(&[T![')'], T![,], T![let]]));
    const TYPE_SET: TokenSet = ty::TYPE_RECOVERY_SET;
    fn segment(x: &mut Parser<'_>, mode: Mode, first: bool) {
        let y = x.start();
        if first && x.eat(T![<]) {
            ty(x);
            if x.eat(T![as]) {
                if is_use_start(x) {
                    ty::path(x);
                } else {
                    x.error("expected a trait");
                }
            }
            x.expect(T![>]);
            if !x.at(T![::]) {
                x.error("expected `::`");
            }
        } else {
            let empty = if first {
                x.eat(T![::]);
                false
            } else {
                true
            };
            match x.current() {
                IDENT => {
                    name_ref(x);
                    opt_type_args(x, mode);
                },
                T![self] | T![super] | T![crate] | T![Self] => {
                    let y = x.start();
                    x.bump_any();
                    y.complete(x, NAME_REF);
                },
                _ => {
                    use Mode::*;
                    let rec_set = match mode {
                        Use => item::ITEM_RECOVERY_SET,
                        Type => TYPE_SET,
                        Expr => EXPR_SET,
                    };
                    x.err_recover("expected identifier", rec_set);
                    if empty {
                        y.abandon(x);
                        return;
                    }
                },
            };
        }
        y.complete(x, PATH_SEGMENT);
    }
    fn opt_type_args(x: &mut Parser<'_>, mode: Mode) {
        use Mode::*;
        match mode {
            Use => {},
            Type => {
                if x.at(T![::]) && x.nth_at(2, T!['(']) {
                    x.bump(T![::]);
                }
                if x.at(T!['(']) {
                    param::fn_trait(x);
                    opt_ret_type(x);
                } else {
                    generic::opt_args(x, false);
                }
            },
            Expr => generic::opt_args(x, true),
        }
    }
}
mod expr;
mod item;
mod pattern;

pub mod pre {
    use super::*;
    pub fn vis(x: &mut Parser<'_>) {
        opt_vis(x, false);
    }
    pub fn block(x: &mut Parser<'_>) {
        expr::block_expr(x);
    }
    pub fn stmt(x: &mut Parser<'_>) {
        expr::stmt(x, expr::Semicolon::Forbidden);
    }
    pub fn pat(x: &mut Parser<'_>) {
        pattern::pattern_single(x);
    }
    pub fn pat_top(x: &mut Parser<'_>) {
        pattern::pattern_top(x);
    }
    pub fn ty(x: &mut Parser<'_>) {
        ty(x);
    }
    pub fn expr(x: &mut Parser<'_>) {
        expr::expr(x);
    }
    pub fn path(x: &mut Parser<'_>) {
        path::for_type(x);
    }
    pub fn item(x: &mut Parser<'_>) {
        item::item_or_macro(x, true);
    }
    pub fn meta(x: &mut Parser<'_>) {
        attr::meta(x);
    }
}
pub mod top {
    use super::*;
    pub fn src_file(x: &mut Parser<'_>) {
        let y = x.start();
        x.eat(SHEBANG);
        item::mod_contents(x, false);
        y.complete(x, SOURCE_FILE);
    }
    pub fn mac_stmts(x: &mut Parser<'_>) {
        let y = x.start();
        while !x.at(EOF) {
            expr::stmt(x, expr::Semicolon::Optional);
        }
        y.complete(x, MACRO_STMTS);
    }
    pub fn mac_items(x: &mut Parser<'_>) {
        let y = x.start();
        item::mod_contents(x, false);
        y.complete(x, MACRO_ITEMS);
    }
    pub fn pattern(x: &mut Parser<'_>) {
        let y = x.start();
        pattern::pattern_top(x);
        if x.at(EOF) {
            y.abandon(x);
            return;
        }
        while !x.at(EOF) {
            x.bump_any();
        }
        y.complete(x, ERROR);
    }
    pub fn ty(x: &mut Parser<'_>) {
        let y = x.start();
        ty(x);
        if x.at(EOF) {
            y.abandon(x);
            return;
        }
        while !x.at(EOF) {
            x.bump_any();
        }
        y.complete(x, ERROR);
    }
    pub fn expr(x: &mut Parser<'_>) {
        let y = x.start();
        expr::expr(x);
        if x.at(EOF) {
            y.abandon(x);
            return;
        }
        while !x.at(EOF) {
            x.bump_any();
        }
        y.complete(x, ERROR);
    }
    pub fn meta(x: &mut Parser<'_>) {
        let y = x.start();
        attr::meta(x);
        if x.at(EOF) {
            y.abandon(x);
            return;
        }
        while !x.at(EOF) {
            x.bump_any();
        }
        y.complete(x, ERROR);
    }
}

pub fn ty(x: &mut Parser<'_>) {
    ty::with_bounds(x, true);
}
mod ty {
    use super::*;
    pub const FIRST: TokenSet = path::FIRST.union(TokenSet::new(&[
        T!['('],
        T!['['],
        T![<],
        T![!],
        T![*],
        T![&],
        T![_],
        T![fn],
        T![unsafe],
        T![extern],
        T![for],
        T![impl],
        T![dyn],
        T![Self],
        LIFETIME_IDENT,
    ]));
    pub const TYPE_RECOVERY_SET: TokenSet = TokenSet::new(&[T![')'], T![>], T![,], T![pub]]);
    pub fn no_bounds(x: &mut Parser<'_>) {
        with_bounds(x, false);
    }
    pub fn with_bounds(x: &mut Parser<'_>, bounds: bool) {
        match x.current() {
            T!['('] => paren_or_tuple(x),
            T![!] => never(x),
            T![*] => ptr(x),
            T!['['] => array_or_slice(x),
            T![&] => ref_ty(x),
            T![_] => infer(x),
            T![fn] | T![unsafe] | T![extern] => fn_ptr(x),
            T![for] => for_ty(x, bounds),
            T![impl] => impl_trait(x),
            T![dyn] => dyn_trait(x),
            T![<] => path_with_bounds(x, bounds),
            _ if path::is_start(x) => path_or_mac(x, bounds),
            LIFETIME_IDENT if x.nth_at(1, T![+]) => bare_dyn_trait(x),
            _ => {
                x.err_recover("expected type", TYPE_RECOVERY_SET);
            },
        }
    }
    pub fn ascription(x: &mut Parser<'_>) {
        assert!(x.at(T![:]));
        x.bump(T![:]);
        if x.at(T![=]) {
            x.error("missing type");
            return;
        }
        ty(x);
    }
    fn paren_or_tuple(x: &mut Parser<'_>) {
        assert!(x.at(T!['(']));
        let y = x.start();
        x.bump(T!['(']);
        let mut n: u32 = 0;
        let mut trailing_comma: bool = false;
        while !x.at(EOF) && !x.at(T![')']) {
            n += 1;
            ty(x);
            if x.eat(T![,]) {
                trailing_comma = true;
            } else {
                trailing_comma = false;
                break;
            }
        }
        x.expect(T![')']);
        let kind = if n == 1 && !trailing_comma {
            PAREN_TYPE
        } else {
            TUPLE_TYPE
        };
        y.complete(x, kind);
    }
    fn never(x: &mut Parser<'_>) {
        assert!(x.at(T![!]));
        let y = x.start();
        x.bump(T![!]);
        y.complete(x, NEVER_TYPE);
    }
    fn ptr(x: &mut Parser<'_>) {
        assert!(x.at(T![*]));
        let y = x.start();
        x.bump(T![*]);
        match x.current() {
            T![mut] | T![const] => x.bump_any(),
            _ => {
                x.error(
                    "expected mut or const in raw pointer type \
                     (use `*mut T` or `*const T` as appropriate)",
                );
            },
        };
        no_bounds(x);
        y.complete(x, PTR_TYPE);
    }
    fn array_or_slice(x: &mut Parser<'_>) {
        assert!(x.at(T!['[']));
        let y = x.start();
        x.bump(T!['[']);
        ty(x);
        let kind = match x.current() {
            T![']'] => {
                x.bump(T![']']);
                SLICE_TYPE
            },
            T![;] => {
                x.bump(T![;]);
                let y = x.start();
                expr::expr(x);
                y.complete(x, CONST_ARG);
                x.expect(T![']']);
                ARRAY_TYPE
            },
            _ => {
                x.error("expected `;` or `]`");
                SLICE_TYPE
            },
        };
        y.complete(x, kind);
    }
    fn ref_ty(x: &mut Parser<'_>) {
        assert!(x.at(T![&]));
        let y = x.start();
        x.bump(T![&]);
        if x.at(LIFETIME_IDENT) {
            lifetime(x);
        }
        x.eat(T![mut]);
        no_bounds(x);
        y.complete(x, REF_TYPE);
    }
    fn infer(x: &mut Parser<'_>) {
        assert!(x.at(T![_]));
        let y = x.start();
        x.bump(T![_]);
        y.complete(x, INFER_TYPE);
    }
    fn fn_ptr(x: &mut Parser<'_>) {
        let y = x.start();
        x.eat(T![unsafe]);
        if x.at(T![extern]) {
            abi(x);
        }
        if !x.eat(T![fn]) {
            y.abandon(x);
            x.error("expected `fn`");
            return;
        }
        if x.at(T!['(']) {
            param::fn_ptr(x);
        } else {
            x.error("expected parameters");
        }
        opt_ret_type(x);
        y.complete(x, FN_PTR_TYPE);
    }
    pub fn for_binder(x: &mut Parser<'_>) {
        assert!(x.at(T![for]));
        x.bump(T![for]);
        if x.at(T![<]) {
            generic::opt_params(x);
        } else {
            x.error("expected `<`");
        }
    }
    pub fn for_ty(x: &mut Parser<'_>, bounds: bool) {
        assert!(x.at(T![for]));
        let y = x.start();
        for_binder(x);
        match x.current() {
            T![fn] | T![unsafe] | T![extern] => {},
            _ if path::is_use_start(x) => {},
            _ => {
                x.error("expected a function pointer or path");
            },
        }
        no_bounds(x);
        let completed = y.complete(x, FOR_TYPE);
        if bounds {
            opt_bounds_as_dyn_trait(x, completed);
        }
    }
    fn impl_trait(x: &mut Parser<'_>) {
        assert!(x.at(T![impl]));
        let y = x.start();
        x.bump(T![impl]);
        generic::bounds_no_colon(x);
        y.complete(x, IMPL_TRAIT_TYPE);
    }
    fn dyn_trait(x: &mut Parser<'_>) {
        assert!(x.at(T![dyn]));
        let y = x.start();
        x.bump(T![dyn]);
        generic::bounds_no_colon(x);
        y.complete(x, DYN_TRAIT_TYPE);
    }
    fn bare_dyn_trait(x: &mut Parser<'_>) {
        let y = x.start();
        generic::bounds_no_colon(x);
        y.complete(x, DYN_TRAIT_TYPE);
    }
    pub fn path(x: &mut Parser<'_>) {
        path_with_bounds(x, true);
    }
    pub fn path_with_bounds(x: &mut Parser<'_>, bounds: bool) {
        assert!(path::is_start(x));
        let y = x.start();
        path::for_type(x);
        let path = y.complete(x, PATH_TYPE);
        if bounds {
            opt_bounds_as_dyn_trait(x, path);
        }
    }
    fn path_or_mac(x: &mut Parser<'_>, bounds: bool) {
        assert!(path::is_start(x));
        let r = x.start();
        let m = x.start();
        path::for_type(x);
        let kind = if x.at(T![!]) && !x.at(T![!=]) {
            item::macro_call_after_excl(x);
            m.complete(x, MACRO_CALL);
            MACRO_TYPE
        } else {
            m.abandon(x);
            PATH_TYPE
        };
        let path = r.complete(x, kind);
        if bounds {
            opt_bounds_as_dyn_trait(x, path);
        }
    }
    pub fn opt_bounds_as_dyn_trait(x: &mut Parser<'_>, y: CompletedMarker) -> CompletedMarker {
        assert!(matches!(
            y.kind(),
            SyntaxKind::PATH_TYPE | SyntaxKind::FOR_TYPE | SyntaxKind::MACRO_TYPE
        ));
        if !x.at(T![+]) {
            return y;
        }
        let y = y.precede(x).complete(x, TYPE_BOUND);
        let y = y.precede(x);
        x.eat(T![+]);
        let y = generic::bounds_with_marker(x, y);
        y.precede(x).complete(x, DYN_TRAIT_TYPE)
    }
}
