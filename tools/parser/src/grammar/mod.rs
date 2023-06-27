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
        BLOCK_EXPR => exprs::block_expr,
        RECORD_FIELD_LIST => items::record_field_list,
        RECORD_EXPR_FIELD_LIST => items::record_expr_field_list,
        VARIANT_LIST => items::variant_list,
        MATCH_ARM_LIST => items::match_arm_list,
        USE_TREE_LIST => items::use_tree_list,
        EXTERN_ITEM_LIST => items::extern_item_list,
        TOKEN_TREE if first_child? == T!['{'] => items::token_tree,
        ASSOC_ITEM_LIST => match parent? {
            IMPL | TRAIT => items::assoc_item_list,
            _ => return None,
        },
        ITEM_LIST => items::item_list,
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
                            use_path(x);
                            x.expect(T![')']);
                        }
                    },
                    T![in] => {
                        x.bump(T!['(']);
                        x.bump(T![in]);
                        use_path(x);
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
        types::type_no_bounds(x);
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
    exprs::expr_block_contents(x);
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

pub const ATTR_FIRST: TokenSet = TokenSet::new(&[T![#]]);

pub fn meta(x: &mut Parser<'_>) {
    let y = x.start();
    use_path(x);
    match x.current() {
        T![=] => {
            x.bump(T![=]);
            if exprs::expr(x).is_none() {
                x.error("expected expression");
            }
        },
        T!['('] | T!['['] | T!['{'] => items::token_tree(x),
        _ => {},
    }
    y.complete(x, META);
}
pub fn inner_attrs(x: &mut Parser<'_>) {
    while x.at(T![#]) && x.nth(1) == T![!] {
        attr(x, true);
    }
}
pub fn outer_attrs(x: &mut Parser<'_>) {
    while x.at(T![#]) {
        attr(x, false);
    }
}
fn attr(x: &mut Parser<'_>, inner: bool) {
    assert!(x.at(T![#]));
    let y = x.start();
    x.bump(T![#]);
    if inner {
        x.bump(T![!]);
    }
    if x.eat(T!['[']) {
        meta(x);
        if !x.eat(T![']']) {
            x.error("expected `]`");
        }
    } else {
        x.error("expected `[`");
    }
    y.complete(x, ATTR);
}

const GEN_ARG_FIRST: TokenSet = TokenSet::new(&[
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
.union(types::TYPE_FIRST);

pub fn opt_generic_args(x: &mut Parser<'_>, colon_colon_required: bool) {
    let y;
    if x.at(T![::]) && x.nth(2) == T![<] {
        y = x.start();
        x.bump(T![::]);
    } else if !colon_colon_required && x.at(T![<]) && x.nth(1) != T![=] {
        y = x.start();
    } else {
        return;
    }
    delimited(x, T![<], T![>], T![,], GEN_ARG_FIRST, generic_arg);
    y.complete(x, GENERIC_ARG_LIST);
}
fn generic_arg(x: &mut Parser<'_>) -> bool {
    match x.current() {
        LIFETIME_IDENT if !x.nth_at(1, T![+]) => lifetime_arg(x),
        T!['{'] | T![true] | T![false] | T![-] => const_arg(x),
        k if k.is_literal() => const_arg(x),
        IDENT if [T![<], T![=], T![:]].contains(&x.nth(1)) && !x.nth_at(1, T![::]) => {
            let y = x.start();
            name_ref(x);
            opt_generic_args(x, false);
            match x.current() {
                T![=] => {
                    x.bump_any();
                    if types::TYPE_FIRST.contains(x.current()) {
                        types::type_(x);
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
                    let y = type_path_for_qual(x, y);
                    y.precede(x).complete(x, PATH_TYPE).precede(x).complete(x, TYPE_ARG);
                },
            }
        },
        IDENT if x.nth_at(1, T!['(']) => {
            let y = x.start();
            name_ref(x);
            params_fn_trait(x);
            if x.at(T![:]) && !x.at(T![::]) {
                bounds(x);
                y.complete(x, ASSOC_TYPE_ARG);
            } else {
                opt_ret_type(x);
                let y = y.complete(x, PATH_SEGMENT).precede(x).complete(x, PATH);
                let y = type_path_for_qual(x, y);
                let y = y.precede(x).complete(x, PATH_TYPE);
                let y = types::opt_type_bounds_as_dyn_trait_type(x, y);
                y.precede(x).complete(x, TYPE_ARG);
            }
        },
        _ if x.at_ts(types::TYPE_FIRST) => type_arg(x),
        _ => return false,
    }
    true
}
fn lifetime_arg(x: &mut Parser<'_>) {
    let y = x.start();
    lifetime(x);
    y.complete(x, LIFETIME_ARG);
}
pub fn const_arg_expr(x: &mut Parser<'_>) {
    match x.current() {
        T!['{'] => {
            exprs::block_expr(x);
        },
        k if k.is_literal() => {
            exprs::literal(x);
        },
        T![true] | T![false] => {
            exprs::literal(x);
        },
        T![-] => {
            let y = x.start();
            x.bump(T![-]);
            exprs::literal(x);
            y.complete(x, PREFIX_EXPR);
        },
        _ => {
            let y = x.start();
            use_path(x);
            y.complete(x, PATH_EXPR);
        },
    }
}
pub fn const_arg(x: &mut Parser<'_>) {
    let y = x.start();
    const_arg_expr(x);
    y.complete(x, CONST_ARG);
}
fn type_arg(x: &mut Parser<'_>) {
    let y = x.start();
    types::type_(x);
    y.complete(x, TYPE_ARG);
}

const GEN_PARAM_FIRST: TokenSet = TokenSet::new(&[IDENT, LIFETIME_IDENT, T![const]]);

pub fn opt_generic_params(x: &mut Parser<'_>) {
    if x.at(T![<]) {
        generic_params(x);
    }
}
fn generic_params(x: &mut Parser<'_>) {
    assert!(x.at(T![<]));
    let y = x.start();
    delimited(x, T![<], T![>], T![,], GEN_PARAM_FIRST.union(ATTR_FIRST), |x| {
        let y = x.start();
        outer_attrs(x);
        generic_param(x, y)
    });
    y.complete(x, GENERIC_PARAM_LIST);
}
fn generic_param(x: &mut Parser<'_>, m: Marker) -> bool {
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
fn type_param(x: &mut Parser<'_>, m: Marker) {
    assert!(x.at(IDENT));
    name(x);
    if x.at(T![:]) {
        bounds(x);
    }
    if x.at(T![=]) {
        x.bump(T![=]);
        types::type_(x);
    }
    m.complete(x, TYPE_PARAM);
}
fn const_param(x: &mut Parser<'_>, m: Marker) {
    x.bump(T![const]);
    name(x);
    if x.at(T![:]) {
        types::ascription(x);
    } else {
        x.error("missing type for const parameter");
    }
    if x.at(T![=]) {
        x.bump(T![=]);
        const_arg_expr(x);
    }
    m.complete(x, CONST_PARAM);
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
pub fn bounds(x: &mut Parser<'_>) {
    assert!(x.at(T![:]));
    x.bump(T![:]);
    bounds_without_colon(x);
}
pub fn bounds_without_colon(x: &mut Parser<'_>) {
    let y = x.start();
    bounds_without_colon_m(x, y);
}
pub fn bounds_without_colon_m(x: &mut Parser<'_>, m: Marker) -> CompletedMarker {
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
        T![for] => types::for_type(x, false),
        T![?] if x.nth_at(1, T![for]) => {
            x.bump_any();
            types::for_type(x, false)
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
            if is_use_path_start(x) {
                types::path_type_(x, false);
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
            token => types::TYPE_FIRST.contains(token),
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
                types::for_binder(x);
            }
            types::type_(x);
            if x.at(T![:]) {
                bounds(x);
            } else {
                x.error("expected colon");
            }
        },
    }
    y.complete(x, WHERE_PRED);
}

const PARAM_FIRST: TokenSet = patterns::PATTERN_FIRST.union(types::TYPE_FIRST);

pub fn params_fn_def(x: &mut Parser<'_>) {
    list_(x, Flavor::FnDef);
}
pub fn params_fn_trait(x: &mut Parser<'_>) {
    list_(x, Flavor::FnTrait);
}
pub fn params_fn_ptr(x: &mut Parser<'_>) {
    list_(x, Flavor::FnPointer);
}
pub fn params_closure(x: &mut Parser<'_>) {
    list_(x, Flavor::Closure);
}
#[derive(Debug, Clone, Copy)]
enum Flavor {
    FnDef,
    FnTrait,
    FnPointer,
    Closure,
}
fn list_(x: &mut Parser<'_>, flavor: Flavor) {
    use Flavor::*;
    let (bra, ket) = match flavor {
        Closure => (T![|], T![|]),
        FnDef | FnTrait | FnPointer => (T!['('], T![')']),
    };
    let y = x.start();
    x.bump(bra);
    let mut param_marker = None;
    if let FnDef = flavor {
        let m = x.start();
        outer_attrs(x);
        match opt_self_param(x, m) {
            Ok(()) => {},
            Err(m) => param_marker = Some(m),
        }
    }
    while !x.at(EOF) && !x.at(ket) {
        let m = match param_marker.take() {
            Some(m) => m,
            None => {
                let m = x.start();
                outer_attrs(x);
                m
            },
        };
        if !x.at_ts(PARAM_FIRST.union(ATTR_FIRST)) {
            x.error("expected value parameter");
            m.abandon(x);
            break;
        }
        param(x, m, flavor);
        if !x.at(T![,]) {
            if x.at_ts(PARAM_FIRST.union(ATTR_FIRST)) {
                x.error("expected `,`");
            } else {
                break;
            }
        } else {
            x.bump(T![,]);
        }
    }
    if let Some(m) = param_marker {
        m.abandon(x);
    }
    x.expect(ket);
    y.complete(x, PARAM_LIST);
}
fn param(x: &mut Parser<'_>, y: Marker, flavor: Flavor) {
    match flavor {
        Flavor::FnDef | Flavor::FnPointer if x.eat(T![...]) => {},
        Flavor::FnDef => {
            patterns::pattern(x);
            if !variadic_param(x) {
                if x.at(T![:]) {
                    types::ascription(x);
                } else {
                    x.error("missing type for function parameter");
                }
            }
        },
        Flavor::FnTrait => {
            types::type_(x);
        },
        Flavor::FnPointer => {
            if (x.at(IDENT) || x.at(UNDERSCORE)) && x.nth(1) == T![:] && !x.nth_at(1, T![::]) {
                patterns::pattern_single(x);
                if !variadic_param(x) {
                    if x.at(T![:]) {
                        types::ascription(x);
                    } else {
                        x.error("missing type for function parameter");
                    }
                }
            } else {
                types::type_(x);
            }
        },
        Flavor::Closure => {
            patterns::pattern_single(x);
            if x.at(T![:]) && !x.at(T![::]) {
                types::ascription(x);
            }
        },
    }
    y.complete(x, PARAM);
}
fn variadic_param(x: &mut Parser<'_>) -> bool {
    if x.at(T![:]) && x.nth_at(1, T![...]) {
        x.bump(T![:]);
        x.bump(T![...]);
        true
    } else {
        false
    }
}
fn opt_self_param(x: &mut Parser<'_>, y: Marker) -> Result<(), Marker> {
    if x.at(T![self]) || x.at(T![mut]) && x.nth(1) == T![self] {
        x.eat(T![mut]);
        self_as_name(x);
        if x.at(T![:]) {
            types::ascription(x);
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

pub const PATH_FIRST: TokenSet = TokenSet::new(&[IDENT, T![self], T![super], T![crate], T![Self], T![:], T![<]]);

pub fn is_path_start(x: &Parser<'_>) -> bool {
    is_use_path_start(x) || x.at(T![<]) || x.at(T![Self])
}
pub fn is_use_path_start(x: &Parser<'_>) -> bool {
    match x.current() {
        IDENT | T![self] | T![super] | T![crate] => true,
        T![:] if x.at(T![::]) => true,
        _ => false,
    }
}
pub fn use_path(x: &mut Parser<'_>) {
    path(x, Mode::Use);
}
pub fn type_path(x: &mut Parser<'_>) {
    path(x, Mode::Type);
}
pub fn expr_path(x: &mut Parser<'_>) {
    path(x, Mode::Expr);
}
pub fn type_path_for_qual(x: &mut Parser<'_>, qual: CompletedMarker) -> CompletedMarker {
    path_for_qual(x, Mode::Type, qual)
}
#[derive(Clone, Copy, Eq, PartialEq)]
enum Mode {
    Use,
    Type,
    Expr,
}
fn path(x: &mut Parser<'_>, mode: Mode) {
    let y = x.start();
    path_segment(x, mode, true);
    let qual = y.complete(x, PATH);
    path_for_qual(x, mode, qual);
}
fn path_for_qual(x: &mut Parser<'_>, mode: Mode, mut qual: CompletedMarker) -> CompletedMarker {
    loop {
        let use_tree = mode == Mode::Use && matches!(x.nth(2), T![*] | T!['{']);
        if x.at(T![::]) && !use_tree {
            let y = qual.precede(x);
            x.bump(T![::]);
            path_segment(x, mode, false);
            let y = y.complete(x, PATH);
            qual = y;
        } else {
            return qual;
        }
    }
}

const EXPR_PATH_SEG_REC_SET: TokenSet = items::ITEM_RECOVERY_SET.union(TokenSet::new(&[T![')'], T![,], T![let]]));
const TYPE_PATH_SEG_REC_SET: TokenSet = types::TYPE_RECOVERY_SET;

fn path_segment(x: &mut Parser<'_>, mode: Mode, first: bool) {
    let y = x.start();
    if first && x.eat(T![<]) {
        types::type_(x);
        if x.eat(T![as]) {
            if is_use_path_start(x) {
                types::path_type(x);
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
                opt_path_type_args(x, mode);
            },
            T![self] | T![super] | T![crate] | T![Self] => {
                let y = x.start();
                x.bump_any();
                y.complete(x, NAME_REF);
            },
            _ => {
                let rec_set = match mode {
                    Mode::Use => items::ITEM_RECOVERY_SET,
                    Mode::Type => TYPE_PATH_SEG_REC_SET,
                    Mode::Expr => EXPR_PATH_SEG_REC_SET,
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
fn opt_path_type_args(x: &mut Parser<'_>, mode: Mode) {
    match mode {
        Mode::Use => {},
        Mode::Type => {
            if x.at(T![::]) && x.nth_at(2, T!['(']) {
                x.bump(T![::]);
            }
            if x.at(T!['(']) {
                params_fn_trait(x);
                opt_ret_type(x);
            } else {
                opt_generic_args(x, false);
            }
        },
        Mode::Expr => opt_generic_args(x, true),
    }
}

pub mod prefix {
    use super::*;
    pub fn vis(x: &mut Parser<'_>) {
        opt_vis(x, false);
    }
    pub fn block(x: &mut Parser<'_>) {
        exprs::block_expr(x);
    }
    pub fn stmt(x: &mut Parser<'_>) {
        exprs::stmt(x, exprs::Semicolon::Forbidden);
    }
    pub fn pat(x: &mut Parser<'_>) {
        patterns::pattern_single(x);
    }
    pub fn pat_top(x: &mut Parser<'_>) {
        patterns::pattern_top(x);
    }
    pub fn ty(x: &mut Parser<'_>) {
        types::type_(x);
    }
    pub fn expr(x: &mut Parser<'_>) {
        exprs::expr(x);
    }
    pub fn path(x: &mut Parser<'_>) {
        type_path(x);
    }
    pub fn item(x: &mut Parser<'_>) {
        items::item_or_macro(x, true);
    }
    pub fn meta_item(x: &mut Parser<'_>) {
        meta(x);
    }
}
pub mod top {
    use super::*;
    pub fn src_file(x: &mut Parser<'_>) {
        let y = x.start();
        x.eat(SHEBANG);
        items::mod_contents(x, false);
        y.complete(x, SOURCE_FILE);
    }
    pub fn macro_stmts(x: &mut Parser<'_>) {
        let y = x.start();
        while !x.at(EOF) {
            exprs::stmt(x, exprs::Semicolon::Optional);
        }
        y.complete(x, MACRO_STMTS);
    }
    pub fn macro_items(x: &mut Parser<'_>) {
        let y = x.start();
        items::mod_contents(x, false);
        y.complete(x, MACRO_ITEMS);
    }
    pub fn pattern(x: &mut Parser<'_>) {
        let y = x.start();
        patterns::pattern_top(x);
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
        types::type_(x);
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
        exprs::expr(x);
        if x.at(EOF) {
            y.abandon(x);
            return;
        }
        while !x.at(EOF) {
            x.bump_any();
        }
        y.complete(x, ERROR);
    }
    pub fn meta_item(x: &mut Parser<'_>) {
        let y = x.start();
        meta(x);
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

mod exprs;
mod items;
mod patterns;
mod types;
