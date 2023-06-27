use crate::{
    parser::{CompletedMarker, Marker, Parser},
    SyntaxKind::{self, *},
    TokenSet, T,
};

pub fn reparser(
    node: SyntaxKind,
    first_child: Option<SyntaxKind>,
    parent: Option<SyntaxKind>,
) -> Option<fn(&mut Parser<'_>)> {
    let res = match node {
        BLOCK_EXPR => expressions::block_expr,
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
    Some(res)
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

const VISIBILITY_FIRST: TokenSet = TokenSet::new(&[T![pub], T![crate]]);

fn opt_visibility(p: &mut Parser<'_>, in_tuple_field: bool) -> bool {
    match p.current() {
        T![pub] => {
            let m = p.start();
            p.bump(T![pub]);
            if p.at(T!['(']) {
                match p.nth(1) {
                    T![crate] | T![self] | T![super] | T![ident] | T![')'] if p.nth(2) != T![:] => {
                        if !(in_tuple_field && matches!(p.nth(1), T![ident] | T![')'])) {
                            p.bump(T!['(']);
                            paths::use_path(p);
                            p.expect(T![')']);
                        }
                    },
                    T![in] => {
                        p.bump(T!['(']);
                        p.bump(T![in]);
                        paths::use_path(p);
                        p.expect(T![')']);
                    },
                    _ => {},
                }
            }
            m.complete(p, VISIBILITY);
            true
        },
        T![crate] => {
            if p.nth_at(1, T![::]) {
                return false;
            }
            let m = p.start();
            p.bump(T![crate]);
            m.complete(p, VISIBILITY);
            true
        },
        _ => false,
    }
}
fn opt_rename(p: &mut Parser<'_>) {
    if p.at(T![as]) {
        let m = p.start();
        p.bump(T![as]);
        if !p.eat(T![_]) {
            name(p);
        }
        m.complete(p, RENAME);
    }
}
fn abi(p: &mut Parser<'_>) {
    assert!(p.at(T![extern]));
    let abi = p.start();
    p.bump(T![extern]);
    p.eat(STRING);
    abi.complete(p, ABI);
}
fn opt_ret_type(p: &mut Parser<'_>) -> bool {
    if p.at(T![->]) {
        let m = p.start();
        p.bump(T![->]);
        types::type_no_bounds(p);
        m.complete(p, RET_TYPE);
        true
    } else {
        false
    }
}
fn name_r(p: &mut Parser<'_>, recovery: TokenSet) {
    if p.at(IDENT) {
        let m = p.start();
        p.bump(IDENT);
        m.complete(p, NAME);
    } else {
        p.err_recover("expected a name", recovery);
    }
}
fn name(p: &mut Parser<'_>) {
    name_r(p, TokenSet::EMPTY);
}
fn name_ref(p: &mut Parser<'_>) {
    if p.at(IDENT) {
        let m = p.start();
        p.bump(IDENT);
        m.complete(p, NAME_REF);
    } else {
        p.err_and_bump("expected identifier");
    }
}
fn name_ref_or_index(p: &mut Parser<'_>) {
    assert!(p.at(IDENT) || p.at(INT_NUMBER));
    let m = p.start();
    p.bump_any();
    m.complete(p, NAME_REF);
}
fn lifetime(p: &mut Parser<'_>) {
    assert!(p.at(LIFETIME_IDENT));
    let m = p.start();
    p.bump(LIFETIME_IDENT);
    m.complete(p, LIFETIME);
}
fn error_block(p: &mut Parser<'_>, message: &str) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.error(message);
    p.bump(T!['{']);
    expressions::expr_block_contents(p);
    p.eat(T!['}']);
    m.complete(p, ERROR);
}
fn delimited(
    p: &mut Parser<'_>,
    bra: SyntaxKind,
    ket: SyntaxKind,
    delim: SyntaxKind,
    first_set: TokenSet,
    mut parser: impl FnMut(&mut Parser<'_>) -> bool,
) {
    p.bump(bra);
    while !p.at(ket) && !p.at(EOF) {
        if !parser(p) {
            break;
        }
        if !p.at(delim) {
            if p.at_ts(first_set) {
                p.error(format!("expected {:?}", delim));
            } else {
                break;
            }
        } else {
            p.bump(delim);
        }
    }
    p.expect(ket);
}

mod attributes {
    use super::*;
    pub const ATTRIBUTE_FIRST: TokenSet = TokenSet::new(&[T![#]]);
    pub fn inner_attrs(p: &mut Parser<'_>) {
        while p.at(T![#]) && p.nth(1) == T![!] {
            attr(p, true);
        }
    }
    pub fn outer_attrs(p: &mut Parser<'_>) {
        while p.at(T![#]) {
            attr(p, false);
        }
    }
    fn attr(p: &mut Parser<'_>, inner: bool) {
        assert!(p.at(T![#]));
        let attr = p.start();
        p.bump(T![#]);
        if inner {
            p.bump(T![!]);
        }
        if p.eat(T!['[']) {
            meta(p);
            if !p.eat(T![']']) {
                p.error("expected `]`");
            }
        } else {
            p.error("expected `[`");
        }
        attr.complete(p, ATTR);
    }
    pub fn meta(p: &mut Parser<'_>) {
        let meta = p.start();
        paths::use_path(p);
        match p.current() {
            T![=] => {
                p.bump(T![=]);
                if expressions::expr(p).is_none() {
                    p.error("expected expression");
                }
            },
            T!['('] | T!['['] | T!['{'] => items::token_tree(p),
            _ => {},
        }
        meta.complete(p, META);
    }
}
mod expressions;
mod generic_args {
    use super::*;
    pub fn opt_generic_arg_list(p: &mut Parser<'_>, colon_colon_required: bool) {
        let m;
        if p.at(T![::]) && p.nth(2) == T![<] {
            m = p.start();
            p.bump(T![::]);
        } else if !colon_colon_required && p.at(T![<]) && p.nth(1) != T![=] {
            m = p.start();
        } else {
            return;
        }
        delimited(p, T![<], T![>], T![,], GENERIC_ARG_FIRST, generic_arg);
        m.complete(p, GENERIC_ARG_LIST);
    }
    const GENERIC_ARG_FIRST: TokenSet = TokenSet::new(&[
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
    fn generic_arg(p: &mut Parser<'_>) -> bool {
        match p.current() {
            LIFETIME_IDENT if !p.nth_at(1, T![+]) => lifetime_arg(p),
            T!['{'] | T![true] | T![false] | T![-] => const_arg(p),
            k if k.is_literal() => const_arg(p),
            IDENT if [T![<], T![=], T![:]].contains(&p.nth(1)) && !p.nth_at(1, T![::]) => {
                let m = p.start();
                name_ref(p);
                opt_generic_arg_list(p, false);
                match p.current() {
                    T![=] => {
                        p.bump_any();
                        if types::TYPE_FIRST.contains(p.current()) {
                            types::type_(p);
                        } else {
                            const_arg(p);
                        }
                        m.complete(p, ASSOC_TYPE_ARG);
                    },
                    T![:] if !p.at(T![::]) => {
                        generic_params::bounds(p);
                        m.complete(p, ASSOC_TYPE_ARG);
                    },
                    _ => {
                        let m = m.complete(p, PATH_SEGMENT).precede(p).complete(p, PATH);
                        let m = paths::type_path_for_qualifier(p, m);
                        m.precede(p).complete(p, PATH_TYPE).precede(p).complete(p, TYPE_ARG);
                    },
                }
            },
            IDENT if p.nth_at(1, T!['(']) => {
                let m = p.start();
                name_ref(p);
                params::param_list_fn_trait(p);
                if p.at(T![:]) && !p.at(T![::]) {
                    generic_params::bounds(p);
                    m.complete(p, ASSOC_TYPE_ARG);
                } else {
                    opt_ret_type(p);
                    let m = m.complete(p, PATH_SEGMENT).precede(p).complete(p, PATH);
                    let m = paths::type_path_for_qualifier(p, m);
                    let m = m.precede(p).complete(p, PATH_TYPE);
                    let m = types::opt_type_bounds_as_dyn_trait_type(p, m);
                    m.precede(p).complete(p, TYPE_ARG);
                }
            },
            _ if p.at_ts(types::TYPE_FIRST) => type_arg(p),
            _ => return false,
        }
        true
    }
    fn lifetime_arg(p: &mut Parser<'_>) {
        let m = p.start();
        lifetime(p);
        m.complete(p, LIFETIME_ARG);
    }
    pub fn const_arg_expr(p: &mut Parser<'_>) {
        match p.current() {
            T!['{'] => {
                expressions::block_expr(p);
            },
            k if k.is_literal() => {
                expressions::literal(p);
            },
            T![true] | T![false] => {
                expressions::literal(p);
            },
            T![-] => {
                let lm = p.start();
                p.bump(T![-]);
                expressions::literal(p);
                lm.complete(p, PREFIX_EXPR);
            },
            _ => {
                let lm = p.start();
                paths::use_path(p);
                lm.complete(p, PATH_EXPR);
            },
        }
    }
    pub fn const_arg(p: &mut Parser<'_>) {
        let m = p.start();
        const_arg_expr(p);
        m.complete(p, CONST_ARG);
    }
    fn type_arg(p: &mut Parser<'_>) {
        let m = p.start();
        types::type_(p);
        m.complete(p, TYPE_ARG);
    }
}
mod generic_params {
    use super::*;
    use crate::grammar::attributes::ATTRIBUTE_FIRST;
    pub fn opt_generic_param_list(p: &mut Parser<'_>) {
        if p.at(T![<]) {
            generic_param_list(p);
        }
    }
    fn generic_param_list(p: &mut Parser<'_>) {
        assert!(p.at(T![<]));
        let m = p.start();
        delimited(
            p,
            T![<],
            T![>],
            T![,],
            GENERIC_PARAM_FIRST.union(ATTRIBUTE_FIRST),
            |p| {
                let m = p.start();
                attributes::outer_attrs(p);
                generic_param(p, m)
            },
        );
        m.complete(p, GENERIC_PARAM_LIST);
    }
    const GENERIC_PARAM_FIRST: TokenSet = TokenSet::new(&[IDENT, LIFETIME_IDENT, T![const]]);
    fn generic_param(p: &mut Parser<'_>, m: Marker) -> bool {
        match p.current() {
            LIFETIME_IDENT => lifetime_param(p, m),
            IDENT => type_param(p, m),
            T![const] => const_param(p, m),
            _ => {
                m.abandon(p);
                p.err_and_bump("expected generic parameter");
                return false;
            },
        }
        true
    }
    fn lifetime_param(p: &mut Parser<'_>, m: Marker) {
        assert!(p.at(LIFETIME_IDENT));
        lifetime(p);
        if p.at(T![:]) {
            lifetime_bounds(p);
        }
        m.complete(p, LIFETIME_PARAM);
    }
    fn type_param(p: &mut Parser<'_>, m: Marker) {
        assert!(p.at(IDENT));
        name(p);
        if p.at(T![:]) {
            bounds(p);
        }
        if p.at(T![=]) {
            p.bump(T![=]);
            types::type_(p);
        }
        m.complete(p, TYPE_PARAM);
    }
    fn const_param(p: &mut Parser<'_>, m: Marker) {
        p.bump(T![const]);
        name(p);
        if p.at(T![:]) {
            types::ascription(p);
        } else {
            p.error("missing type for const parameter");
        }
        if p.at(T![=]) {
            p.bump(T![=]);
            generic_args::const_arg_expr(p);
        }
        m.complete(p, CONST_PARAM);
    }
    fn lifetime_bounds(p: &mut Parser<'_>) {
        assert!(p.at(T![:]));
        p.bump(T![:]);
        while p.at(LIFETIME_IDENT) {
            lifetime(p);
            if !p.eat(T![+]) {
                break;
            }
        }
    }
    pub fn bounds(p: &mut Parser<'_>) {
        assert!(p.at(T![:]));
        p.bump(T![:]);
        bounds_without_colon(p);
    }
    pub fn bounds_without_colon(p: &mut Parser<'_>) {
        let m = p.start();
        bounds_without_colon_m(p, m);
    }
    pub fn bounds_without_colon_m(p: &mut Parser<'_>, marker: Marker) -> CompletedMarker {
        while type_bound(p) {
            if !p.eat(T![+]) {
                break;
            }
        }
        marker.complete(p, TYPE_BOUND_LIST)
    }
    fn type_bound(p: &mut Parser<'_>) -> bool {
        let m = p.start();
        let has_paren = p.eat(T!['(']);
        match p.current() {
            LIFETIME_IDENT => lifetime(p),
            T![for] => types::for_type(p, false),
            T![?] if p.nth_at(1, T![for]) => {
                p.bump_any();
                types::for_type(p, false)
            },
            current => {
                match current {
                    T![?] => p.bump_any(),
                    T![~] => {
                        p.bump_any();
                        p.expect(T![const]);
                    },
                    _ => (),
                }
                if paths::is_use_path_start(p) {
                    types::path_type_(p, false);
                } else {
                    m.abandon(p);
                    return false;
                }
            },
        }
        if has_paren {
            p.expect(T![')']);
        }
        m.complete(p, TYPE_BOUND);
        true
    }
    pub fn opt_where_clause(p: &mut Parser<'_>) {
        if !p.at(T![where]) {
            return;
        }
        let m = p.start();
        p.bump(T![where]);
        while is_where_predicate(p) {
            where_predicate(p);
            let comma = p.eat(T![,]);
            match p.current() {
                T!['{'] | T![;] | T![=] => break,
                _ => (),
            }
            if !comma {
                p.error("expected comma");
            }
        }
        m.complete(p, WHERE_CLAUSE);
        fn is_where_predicate(p: &mut Parser<'_>) -> bool {
            match p.current() {
                LIFETIME_IDENT => true,
                T![impl] => false,
                token => types::TYPE_FIRST.contains(token),
            }
        }
    }
    fn where_predicate(p: &mut Parser<'_>) {
        let m = p.start();
        match p.current() {
            LIFETIME_IDENT => {
                lifetime(p);
                if p.at(T![:]) {
                    bounds(p);
                } else {
                    p.error("expected colon");
                }
            },
            T![impl] => {
                p.error("expected lifetime or type");
            },
            _ => {
                if p.at(T![for]) {
                    types::for_binder(p);
                }
                types::type_(p);
                if p.at(T![:]) {
                    bounds(p);
                } else {
                    p.error("expected colon");
                }
            },
        }
        m.complete(p, WHERE_PRED);
    }
}
mod items;
mod params {
    use super::*;
    use crate::grammar::attributes::ATTRIBUTE_FIRST;
    pub fn param_list_fn_def(p: &mut Parser<'_>) {
        list_(p, Flavor::FnDef);
    }
    pub fn param_list_fn_trait(p: &mut Parser<'_>) {
        list_(p, Flavor::FnTrait);
    }
    pub fn param_list_fn_ptr(p: &mut Parser<'_>) {
        list_(p, Flavor::FnPointer);
    }
    pub fn param_list_closure(p: &mut Parser<'_>) {
        list_(p, Flavor::Closure);
    }
    #[derive(Debug, Clone, Copy)]
    enum Flavor {
        FnDef,
        FnTrait,
        FnPointer,
        Closure,
    }
    fn list_(p: &mut Parser<'_>, flavor: Flavor) {
        use Flavor::*;
        let (bra, ket) = match flavor {
            Closure => (T![|], T![|]),
            FnDef | FnTrait | FnPointer => (T!['('], T![')']),
        };
        let list_marker = p.start();
        p.bump(bra);
        let mut param_marker = None;
        if let FnDef = flavor {
            let m = p.start();
            attributes::outer_attrs(p);
            match opt_self_param(p, m) {
                Ok(()) => {},
                Err(m) => param_marker = Some(m),
            }
        }
        while !p.at(EOF) && !p.at(ket) {
            let m = match param_marker.take() {
                Some(m) => m,
                None => {
                    let m = p.start();
                    attributes::outer_attrs(p);
                    m
                },
            };
            if !p.at_ts(PARAM_FIRST.union(ATTRIBUTE_FIRST)) {
                p.error("expected value parameter");
                m.abandon(p);
                break;
            }
            param(p, m, flavor);
            if !p.at(T![,]) {
                if p.at_ts(PARAM_FIRST.union(ATTRIBUTE_FIRST)) {
                    p.error("expected `,`");
                } else {
                    break;
                }
            } else {
                p.bump(T![,]);
            }
        }
        if let Some(m) = param_marker {
            m.abandon(p);
        }
        p.expect(ket);
        list_marker.complete(p, PARAM_LIST);
    }
    const PARAM_FIRST: TokenSet = patterns::PATTERN_FIRST.union(types::TYPE_FIRST);
    fn param(p: &mut Parser<'_>, m: Marker, flavor: Flavor) {
        match flavor {
            Flavor::FnDef | Flavor::FnPointer if p.eat(T![...]) => {},
            Flavor::FnDef => {
                patterns::pattern(p);
                if !variadic_param(p) {
                    if p.at(T![:]) {
                        types::ascription(p);
                    } else {
                        p.error("missing type for function parameter");
                    }
                }
            },
            Flavor::FnTrait => {
                types::type_(p);
            },
            Flavor::FnPointer => {
                if (p.at(IDENT) || p.at(UNDERSCORE)) && p.nth(1) == T![:] && !p.nth_at(1, T![::]) {
                    patterns::pattern_single(p);
                    if !variadic_param(p) {
                        if p.at(T![:]) {
                            types::ascription(p);
                        } else {
                            p.error("missing type for function parameter");
                        }
                    }
                } else {
                    types::type_(p);
                }
            },
            Flavor::Closure => {
                patterns::pattern_single(p);
                if p.at(T![:]) && !p.at(T![::]) {
                    types::ascription(p);
                }
            },
        }
        m.complete(p, PARAM);
    }
    fn variadic_param(p: &mut Parser<'_>) -> bool {
        if p.at(T![:]) && p.nth_at(1, T![...]) {
            p.bump(T![:]);
            p.bump(T![...]);
            true
        } else {
            false
        }
    }
    fn opt_self_param(p: &mut Parser<'_>, m: Marker) -> Result<(), Marker> {
        if p.at(T![self]) || p.at(T![mut]) && p.nth(1) == T![self] {
            p.eat(T![mut]);
            self_as_name(p);
            if p.at(T![:]) {
                types::ascription(p);
            }
        } else {
            let la1 = p.nth(1);
            let la2 = p.nth(2);
            let la3 = p.nth(3);
            if !matches!(
                (p.current(), la1, la2, la3),
                (T![&], T![self], _, _)
                    | (T![&], T![mut] | LIFETIME_IDENT, T![self], _)
                    | (T![&], LIFETIME_IDENT, T![mut], T![self])
            ) {
                return Err(m);
            }
            p.bump(T![&]);
            if p.at(LIFETIME_IDENT) {
                lifetime(p);
            }
            p.eat(T![mut]);
            self_as_name(p);
        }
        m.complete(p, SELF_PARAM);
        if !p.at(T![')']) {
            p.expect(T![,]);
        }
        Ok(())
    }
    fn self_as_name(p: &mut Parser<'_>) {
        let m = p.start();
        p.bump(T![self]);
        m.complete(p, NAME);
    }
}
mod paths {
    use super::*;
    pub const PATH_FIRST: TokenSet = TokenSet::new(&[IDENT, T![self], T![super], T![crate], T![Self], T![:], T![<]]);
    pub fn is_path_start(p: &Parser<'_>) -> bool {
        is_use_path_start(p) || p.at(T![<]) || p.at(T![Self])
    }
    pub fn is_use_path_start(p: &Parser<'_>) -> bool {
        match p.current() {
            IDENT | T![self] | T![super] | T![crate] => true,
            T![:] if p.at(T![::]) => true,
            _ => false,
        }
    }
    pub fn use_path(p: &mut Parser<'_>) {
        path(p, Mode::Use);
    }
    pub fn type_path(p: &mut Parser<'_>) {
        path(p, Mode::Type);
    }
    pub fn expr_path(p: &mut Parser<'_>) {
        path(p, Mode::Expr);
    }
    pub fn type_path_for_qualifier(p: &mut Parser<'_>, qual: CompletedMarker) -> CompletedMarker {
        path_for_qualifier(p, Mode::Type, qual)
    }
    #[derive(Clone, Copy, Eq, PartialEq)]
    enum Mode {
        Use,
        Type,
        Expr,
    }
    fn path(p: &mut Parser<'_>, mode: Mode) {
        let path = p.start();
        path_segment(p, mode, true);
        let qual = path.complete(p, PATH);
        path_for_qualifier(p, mode, qual);
    }
    fn path_for_qualifier(p: &mut Parser<'_>, mode: Mode, mut qual: CompletedMarker) -> CompletedMarker {
        loop {
            let use_tree = mode == Mode::Use && matches!(p.nth(2), T![*] | T!['{']);
            if p.at(T![::]) && !use_tree {
                let path = qual.precede(p);
                p.bump(T![::]);
                path_segment(p, mode, false);
                let path = path.complete(p, PATH);
                qual = path;
            } else {
                return qual;
            }
        }
    }
    const EXPR_PATH_SEGMENT_RECOVERY_SET: TokenSet =
        items::ITEM_RECOVERY_SET.union(TokenSet::new(&[T![')'], T![,], T![let]]));
    const TYPE_PATH_SEGMENT_RECOVERY_SET: TokenSet = types::TYPE_RECOVERY_SET;
    fn path_segment(p: &mut Parser<'_>, mode: Mode, first: bool) {
        let m = p.start();
        if first && p.eat(T![<]) {
            types::type_(p);
            if p.eat(T![as]) {
                if is_use_path_start(p) {
                    types::path_type(p);
                } else {
                    p.error("expected a trait");
                }
            }
            p.expect(T![>]);
            if !p.at(T![::]) {
                p.error("expected `::`");
            }
        } else {
            let empty = if first {
                p.eat(T![::]);
                false
            } else {
                true
            };
            match p.current() {
                IDENT => {
                    name_ref(p);
                    opt_path_type_args(p, mode);
                },
                T![self] | T![super] | T![crate] | T![Self] => {
                    let m = p.start();
                    p.bump_any();
                    m.complete(p, NAME_REF);
                },
                _ => {
                    let recover_set = match mode {
                        Mode::Use => items::ITEM_RECOVERY_SET,
                        Mode::Type => TYPE_PATH_SEGMENT_RECOVERY_SET,
                        Mode::Expr => EXPR_PATH_SEGMENT_RECOVERY_SET,
                    };
                    p.err_recover("expected identifier", recover_set);
                    if empty {
                        m.abandon(p);
                        return;
                    }
                },
            };
        }
        m.complete(p, PATH_SEGMENT);
    }
    fn opt_path_type_args(p: &mut Parser<'_>, mode: Mode) {
        match mode {
            Mode::Use => {},
            Mode::Type => {
                if p.at(T![::]) && p.nth_at(2, T!['(']) {
                    p.bump(T![::]);
                }
                if p.at(T!['(']) {
                    params::param_list_fn_trait(p);
                    opt_ret_type(p);
                } else {
                    generic_args::opt_generic_arg_list(p, false);
                }
            },
            Mode::Expr => generic_args::opt_generic_arg_list(p, true),
        }
    }
}
mod patterns;
mod types;
pub mod entry {
    use super::*;
    pub mod prefix {
        use super::*;
        pub fn vis(p: &mut Parser<'_>) {
            opt_visibility(p, false);
        }
        pub fn block(p: &mut Parser<'_>) {
            expressions::block_expr(p);
        }
        pub fn stmt(p: &mut Parser<'_>) {
            expressions::stmt(p, expressions::Semicolon::Forbidden);
        }
        pub fn pat(p: &mut Parser<'_>) {
            patterns::pattern_single(p);
        }
        pub fn pat_top(p: &mut Parser<'_>) {
            patterns::pattern_top(p);
        }
        pub fn ty(p: &mut Parser<'_>) {
            types::type_(p);
        }
        pub fn expr(p: &mut Parser<'_>) {
            expressions::expr(p);
        }
        pub fn path(p: &mut Parser<'_>) {
            paths::type_path(p);
        }
        pub fn item(p: &mut Parser<'_>) {
            items::item_or_macro(p, true);
        }
        pub fn meta_item(p: &mut Parser<'_>) {
            attributes::meta(p);
        }
    }
    pub mod top {
        use super::*;
        pub fn source_file(p: &mut Parser<'_>) {
            let m = p.start();
            p.eat(SHEBANG);
            items::mod_contents(p, false);
            m.complete(p, SOURCE_FILE);
        }
        pub fn macro_stmts(p: &mut Parser<'_>) {
            let m = p.start();
            while !p.at(EOF) {
                expressions::stmt(p, expressions::Semicolon::Optional);
            }
            m.complete(p, MACRO_STMTS);
        }
        pub fn macro_items(p: &mut Parser<'_>) {
            let m = p.start();
            items::mod_contents(p, false);
            m.complete(p, MACRO_ITEMS);
        }
        pub fn pattern(p: &mut Parser<'_>) {
            let m = p.start();
            patterns::pattern_top(p);
            if p.at(EOF) {
                m.abandon(p);
                return;
            }
            while !p.at(EOF) {
                p.bump_any();
            }
            m.complete(p, ERROR);
        }
        pub fn type_(p: &mut Parser<'_>) {
            let m = p.start();
            types::type_(p);
            if p.at(EOF) {
                m.abandon(p);
                return;
            }
            while !p.at(EOF) {
                p.bump_any();
            }
            m.complete(p, ERROR);
        }
        pub fn expr(p: &mut Parser<'_>) {
            let m = p.start();
            expressions::expr(p);
            if p.at(EOF) {
                m.abandon(p);
                return;
            }
            while !p.at(EOF) {
                p.bump_any();
            }
            m.complete(p, ERROR);
        }
        pub fn meta_item(p: &mut Parser<'_>) {
            let m = p.start();
            attributes::meta(p);
            if p.at(EOF) {
                m.abandon(p);
                return;
            }
            while !p.at(EOF) {
                p.bump_any();
            }
            m.complete(p, ERROR);
        }
    }
}
