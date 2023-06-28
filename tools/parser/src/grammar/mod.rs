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
        RECORD_FIELD_LIST => item::record_fields,
        RECORD_EXPR_FIELD_LIST => expr::record_fields,
        VARIANT_LIST => item::variants,
        MATCH_ARM_LIST => expr::match_arms,
        USE_TREE_LIST => item::use_trees,
        EXTERN_ITEM_LIST => item::externs,
        TOKEN_TREE if first_child? == T!['{'] => item::token_tree,
        ASSOC_ITEM_LIST => match parent? {
            IMPL | TRAIT => item::assoc_items,
            _ => return None,
        },
        ITEM_LIST => item::items,
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

fn is_opt_vis(x: &mut Parser<'_>, in_tuple_field: bool) -> bool {
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
fn is_opt_ret_type(x: &mut Parser<'_>) -> bool {
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
    pub fn opt_args(x: &mut Parser<'_>, colons_required: bool) {
        let y;
        if x.at(T![::]) && x.nth(2) == T![<] {
            y = x.start();
            x.bump(T![::]);
        } else if !colons_required && x.at(T![<]) && x.nth(1) != T![=] {
            y = x.start();
        } else {
            return;
        }
        delimited(x, T![<], T![>], T![,], ARG_FIRST, is_arg);
        y.complete(x, GENERIC_ARG_LIST);
    }
    fn is_arg(x: &mut Parser<'_>) -> bool {
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
                    is_opt_ret_type(x);
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
            params(x);
        }
    }
    fn params(x: &mut Parser<'_>) {
        assert!(x.at(T![<]));
        let y = x.start();
        delimited(x, T![<], T![>], T![,], PARAM_FIRST.union(attr::FIRST), |x| {
            let y = x.start();
            attr::outers(x);
            is_param(x, y)
        });
        y.complete(x, GENERIC_PARAM_LIST);
    }
    fn is_param(x: &mut Parser<'_>, m: Marker) -> bool {
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
    pub const FIRST: TokenSet = pattern::FIRST.union(ty::FIRST);
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
    fn many(x: &mut Parser<'_>, f: Flavor) {
        use Flavor::*;
        let (bra, ket) = match f {
            Closure => (T![|], T![|]),
            FnDef | FnTrait | FnPtr => (T!['('], T![')']),
        };
        let y = x.start();
        x.bump(bra);
        let mut marker = None;
        if let FnDef = f {
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
            one(x, y, f);
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
    fn one(x: &mut Parser<'_>, y: Marker, f: Flavor) {
        use Flavor::*;
        match f {
            FnDef | FnPtr if x.eat(T![...]) => {},
            FnDef => {
                pattern::one(x);
                if !is_variadic(x) {
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
                    pattern::single(x);
                    if !is_variadic(x) {
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
                pattern::single(x);
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
    fn is_variadic(x: &mut Parser<'_>) -> bool {
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
    fn one(x: &mut Parser<'_>, m: Mode) {
        let y = x.start();
        segment(x, m, true);
        let qual = y.complete(x, PATH);
        for_qual(x, m, qual);
    }
    fn for_qual(x: &mut Parser<'_>, m: Mode, mut qual: CompletedMarker) -> CompletedMarker {
        loop {
            let use_tree = m == Mode::Use && matches!(x.nth(2), T![*] | T!['{']);
            if x.at(T![::]) && !use_tree {
                let y = qual.precede(x);
                x.bump(T![::]);
                segment(x, m, false);
                let y = y.complete(x, PATH);
                qual = y;
            } else {
                return qual;
            }
        }
    }
    const EXPR_SET: TokenSet = item::RECOVERY_SET.union(TokenSet::new(&[T![')'], T![,], T![let]]));
    const TYPE_SET: TokenSet = ty::RECOVERY_SET;
    fn segment(x: &mut Parser<'_>, m: Mode, first: bool) {
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
                    opt_type_args(x, m);
                },
                T![self] | T![super] | T![crate] | T![Self] => {
                    let y = x.start();
                    x.bump_any();
                    y.complete(x, NAME_REF);
                },
                _ => {
                    use Mode::*;
                    let rec_set = match m {
                        Use => item::RECOVERY_SET,
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
    fn opt_type_args(x: &mut Parser<'_>, m: Mode) {
        use Mode::*;
        match m {
            Use => {},
            Type => {
                if x.at(T![::]) && x.nth_at(2, T!['(']) {
                    x.bump(T![::]);
                }
                if x.at(T!['(']) {
                    param::fn_trait(x);
                    is_opt_ret_type(x);
                } else {
                    generic::opt_args(x, false);
                }
            },
            Expr => generic::opt_args(x, true),
        }
    }
}
mod pattern {
    use super::*;
    pub const FIRST: TokenSet = expr::LIT_FIRST.union(path::FIRST).union(TokenSet::new(&[
        T![box],
        T![ref],
        T![mut],
        T![const],
        T!['('],
        T!['['],
        T![&],
        T![_],
        T![-],
        T![.],
    ]));
    const TOP_FIRST: TokenSet = FIRST.union(TokenSet::new(&[T![|]]));
    const END_FIRST: TokenSet = expr::LIT_FIRST
        .union(path::FIRST)
        .union(TokenSet::new(&[T![-], T![const]]));
    const RECOVERY_SET: TokenSet =
        TokenSet::new(&[T![let], T![if], T![while], T![loop], T![match], T![')'], T![,], T![=]]);
    pub fn one(x: &mut Parser<'_>) {
        one_r(x, RECOVERY_SET);
    }
    pub fn top(x: &mut Parser<'_>) {
        top_r(x, RECOVERY_SET);
    }
    pub fn single(x: &mut Parser<'_>) {
        single_r(x, RECOVERY_SET);
    }
    pub fn top_r(x: &mut Parser<'_>, rec_set: TokenSet) {
        x.eat(T![|]);
        one_r(x, rec_set);
    }
    fn one_r(x: &mut Parser<'_>, rec_set: TokenSet) {
        let y = x.start();
        single_r(x, rec_set);
        if !x.at(T![|]) {
            y.abandon(x);
            return;
        }
        while x.eat(T![|]) {
            single_r(x, rec_set);
        }
        y.complete(x, OR_PAT);
    }
    fn single_r(x: &mut Parser<'_>, rec_set: TokenSet) {
        if x.at(T![..=]) {
            let y = x.start();
            x.bump(T![..=]);
            atom(x, rec_set);
            y.complete(x, RANGE_PAT);
            return;
        }
        if x.at(T![..]) {
            let y = x.start();
            x.bump(T![..]);
            if x.at_ts(END_FIRST) {
                atom(x, rec_set);
                y.complete(x, RANGE_PAT);
            } else {
                y.complete(x, REST_PAT);
            }
            return;
        }
        if let Some(lhs) = atom(x, rec_set) {
            for range_op in [T![...], T![..=], T![..]] {
                if x.at(range_op) {
                    let y = lhs.precede(x);
                    x.bump(range_op);
                    if matches!(
                        x.current(),
                        T![=] | T![,] | T![:] | T![')'] | T!['}'] | T![']'] | T![if]
                    ) {
                    } else {
                        atom(x, rec_set);
                    }
                    y.complete(x, RANGE_PAT);
                    return;
                }
            }
        }
    }
    fn atom(x: &mut Parser<'_>, rec_set: TokenSet) -> Option<CompletedMarker> {
        let y = match x.current() {
            T![box] => box_pat(x),
            T![ref] | T![mut] => ident(x, true),
            T![const] => const_block(x),
            IDENT => match x.nth(1) {
                T!['('] | T!['{'] | T![!] => path_or_macro(x),
                T![:] if x.nth_at(1, T![::]) => path_or_macro(x),
                _ => ident(x, true),
            },
            _ if path::is_start(x) => path_or_macro(x),
            _ if is_literal_start(x) => literal(x),
            T![_] => wildcard(x),
            T![&] => ref_pat(x),
            T!['('] => tuple_pat(x),
            T!['['] => slice(x),
            _ => {
                x.err_recover("expected pattern", rec_set);
                return None;
            },
        };
        Some(y)
    }
    fn is_literal_start(x: &Parser<'_>) -> bool {
        x.at(T![-]) && (x.nth(1) == INT_NUMBER || x.nth(1) == FLOAT_NUMBER) || x.at_ts(expr::LIT_FIRST)
    }
    fn literal(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(is_literal_start(x));
        let y = x.start();
        if x.at(T![-]) {
            x.bump(T![-]);
        }
        expr::literal(x);
        y.complete(x, LITERAL_PAT)
    }
    fn path_or_macro(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(path::is_start(x));
        let y = x.start();
        path::for_expr(x);
        let kind = match x.current() {
            T!['('] => {
                tuple_fields(x);
                TUPLE_STRUCT_PAT
            },
            T!['{'] => {
                record_fields(x);
                RECORD_PAT
            },
            T![!] => {
                item::mac_call_after_excl(x);
                return y.complete(x, MACRO_CALL).precede(x).complete(x, MACRO_PAT);
            },
            _ => PATH_PAT,
        };
        y.complete(x, kind)
    }
    fn tuple_fields(x: &mut Parser<'_>) {
        assert!(x.at(T!['(']));
        x.bump(T!['(']);
        many(x, T![')']);
        x.expect(T![')']);
    }
    fn record_field(x: &mut Parser<'_>) {
        match x.current() {
            IDENT | INT_NUMBER if x.nth(1) == T![:] => {
                name_ref_or_idx(x);
                x.bump(T![:]);
                one(x);
            },
            T![box] => {
                box_pat(x);
            },
            T![ref] | T![mut] | IDENT => {
                ident(x, false);
            },
            _ => {
                x.err_and_bump("expected identifier");
            },
        }
    }
    fn record_fields(x: &mut Parser<'_>) {
        assert!(x.at(T!['{']));
        let y = x.start();
        x.bump(T!['{']);
        while !x.at(EOF) && !x.at(T!['}']) {
            let m = x.start();
            attr::outers(x);
            match x.current() {
                T![.] if x.at(T![..]) => {
                    x.bump(T![..]);
                    m.complete(x, REST_PAT);
                },
                T!['{'] => {
                    err_block(x, "expected ident");
                    m.abandon(x);
                },
                _ => {
                    record_field(x);
                    m.complete(x, RECORD_PAT_FIELD);
                },
            }
            if !x.at(T!['}']) {
                x.expect(T![,]);
            }
        }
        x.expect(T!['}']);
        y.complete(x, RECORD_PAT_FIELD_LIST);
    }
    fn wildcard(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T![_]));
        let y = x.start();
        x.bump(T![_]);
        y.complete(x, WILDCARD_PAT)
    }
    fn ref_pat(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T![&]));
        let y = x.start();
        x.bump(T![&]);
        x.eat(T![mut]);
        single(x);
        y.complete(x, REF_PAT)
    }
    fn tuple_pat(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T!['(']));
        let y = x.start();
        x.bump(T!['(']);
        let mut has_comma = false;
        let mut has_pat = false;
        let mut has_rest = false;
        if x.eat(T![,]) {
            x.error("expected pattern");
            has_comma = true;
        }
        while !x.at(EOF) && !x.at(T![')']) {
            has_pat = true;
            if !x.at_ts(TOP_FIRST) {
                x.error("expected a pattern");
                break;
            }
            has_rest |= x.at(T![..]);
            top(x);
            if !x.at(T![')']) {
                has_comma = true;
                x.expect(T![,]);
            }
        }
        x.expect(T![')']);
        y.complete(
            x,
            if !has_comma && !has_rest && has_pat {
                PAREN_PAT
            } else {
                TUPLE_PAT
            },
        )
    }
    fn slice(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T!['[']));
        let y = x.start();
        x.bump(T!['[']);
        many(x, T![']']);
        x.expect(T![']']);
        y.complete(x, SLICE_PAT)
    }
    fn many(x: &mut Parser<'_>, ket: SyntaxKind) {
        while !x.at(EOF) && !x.at(ket) {
            top(x);
            if !x.at(T![,]) {
                if x.at_ts(TOP_FIRST) {
                    x.error(format!("expected {:?}, got {:?}", T![,], x.current()));
                } else {
                    break;
                }
            } else {
                x.bump(T![,]);
            }
        }
    }
    fn ident(x: &mut Parser<'_>, with_at: bool) -> CompletedMarker {
        assert!(matches!(x.current(), T![ref] | T![mut] | IDENT));
        let y = x.start();
        x.eat(T![ref]);
        x.eat(T![mut]);
        name_r(x, RECOVERY_SET);
        if with_at && x.eat(T![@]) {
            single(x);
        }
        y.complete(x, IDENT_PAT)
    }
    fn box_pat(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T![box]));
        let y = x.start();
        x.bump(T![box]);
        single(x);
        y.complete(x, BOX_PAT)
    }
    fn const_block(x: &mut Parser<'_>) -> CompletedMarker {
        assert!(x.at(T![const]));
        let y = x.start();
        x.bump(T![const]);
        expr::block_expr(x);
        y.complete(x, CONST_BLOCK_PAT)
    }
}

mod expr;
mod item;

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
    pub const RECOVERY_SET: TokenSet = TokenSet::new(&[T![')'], T![>], T![,], T![pub]]);
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
                x.err_recover("expected type", RECOVERY_SET);
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
        is_opt_ret_type(x);
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
            item::mac_call_after_excl(x);
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

pub mod pre {
    use super::*;
    pub fn vis(x: &mut Parser<'_>) {
        is_opt_vis(x, false);
    }
    pub fn block(x: &mut Parser<'_>) {
        expr::block_expr(x);
    }
    pub fn stmt(x: &mut Parser<'_>) {
        expr::stmt(x, expr::Semicolon::Forbidden);
    }
    pub fn pat(x: &mut Parser<'_>) {
        pattern::single(x);
    }
    pub fn pat_top(x: &mut Parser<'_>) {
        pattern::top(x);
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
        item::item_or_mac(x, true);
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
            expr::stmt(x, expr::Semicolon::Opt);
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
        pattern::top(x);
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
