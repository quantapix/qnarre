use super::*;
pub const TYPE_FIRST: TokenSet = PATH_FIRST.union(TokenSet::new(&[
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
pub fn type_(p: &mut Parser<'_>) {
    type_with_bounds_cond(p, true);
}
pub fn type_no_bounds(p: &mut Parser<'_>) {
    type_with_bounds_cond(p, false);
}
fn type_with_bounds_cond(p: &mut Parser<'_>, allow_bounds: bool) {
    match p.current() {
        T!['('] => paren_or_tuple_type(p),
        T![!] => never_type(p),
        T![*] => ptr_type(p),
        T!['['] => array_or_slice_type(p),
        T![&] => ref_type(p),
        T![_] => infer_type(p),
        T![fn] | T![unsafe] | T![extern] => fn_ptr_type(p),
        T![for] => for_type(p, allow_bounds),
        T![impl] => impl_trait_type(p),
        T![dyn] => dyn_trait_type(p),
        T![<] => path_type_(p, allow_bounds),
        _ if is_path_start(p) => path_or_macro_type_(p, allow_bounds),
        LIFETIME_IDENT if p.nth_at(1, T![+]) => bare_dyn_trait_type(p),
        _ => {
            p.err_recover("expected type", TYPE_RECOVERY_SET);
        },
    }
}
pub fn ascription(p: &mut Parser<'_>) {
    assert!(p.at(T![:]));
    p.bump(T![:]);
    if p.at(T![=]) {
        p.error("missing type");
        return;
    }
    type_(p);
}
fn paren_or_tuple_type(p: &mut Parser<'_>) {
    assert!(p.at(T!['(']));
    let m = p.start();
    p.bump(T!['(']);
    let mut n_types: u32 = 0;
    let mut trailing_comma: bool = false;
    while !p.at(EOF) && !p.at(T![')']) {
        n_types += 1;
        type_(p);
        if p.eat(T![,]) {
            trailing_comma = true;
        } else {
            trailing_comma = false;
            break;
        }
    }
    p.expect(T![')']);
    let kind = if n_types == 1 && !trailing_comma {
        PAREN_TYPE
    } else {
        TUPLE_TYPE
    };
    m.complete(p, kind);
}
fn never_type(p: &mut Parser<'_>) {
    assert!(p.at(T![!]));
    let m = p.start();
    p.bump(T![!]);
    m.complete(p, NEVER_TYPE);
}
fn ptr_type(p: &mut Parser<'_>) {
    assert!(p.at(T![*]));
    let m = p.start();
    p.bump(T![*]);
    match p.current() {
        T![mut] | T![const] => p.bump_any(),
        _ => {
            p.error(
                "expected mut or const in raw pointer type \
                 (use `*mut T` or `*const T` as appropriate)",
            );
        },
    };
    type_no_bounds(p);
    m.complete(p, PTR_TYPE);
}
fn array_or_slice_type(p: &mut Parser<'_>) {
    assert!(p.at(T!['[']));
    let m = p.start();
    p.bump(T!['[']);
    type_(p);
    let kind = match p.current() {
        T![']'] => {
            p.bump(T![']']);
            SLICE_TYPE
        },
        T![;] => {
            p.bump(T![;]);
            let m = p.start();
            exprs::expr(p);
            m.complete(p, CONST_ARG);
            p.expect(T![']']);
            ARRAY_TYPE
        },
        _ => {
            p.error("expected `;` or `]`");
            SLICE_TYPE
        },
    };
    m.complete(p, kind);
}
fn ref_type(p: &mut Parser<'_>) {
    assert!(p.at(T![&]));
    let m = p.start();
    p.bump(T![&]);
    if p.at(LIFETIME_IDENT) {
        lifetime(p);
    }
    p.eat(T![mut]);
    type_no_bounds(p);
    m.complete(p, REF_TYPE);
}
fn infer_type(p: &mut Parser<'_>) {
    assert!(p.at(T![_]));
    let m = p.start();
    p.bump(T![_]);
    m.complete(p, INFER_TYPE);
}
fn fn_ptr_type(p: &mut Parser<'_>) {
    let m = p.start();
    p.eat(T![unsafe]);
    if p.at(T![extern]) {
        abi(p);
    }
    if !p.eat(T![fn]) {
        m.abandon(p);
        p.error("expected `fn`");
        return;
    }
    if p.at(T!['(']) {
        params_fn_ptr(p);
    } else {
        p.error("expected parameters");
    }
    opt_ret_type(p);
    m.complete(p, FN_PTR_TYPE);
}
pub fn for_binder(p: &mut Parser<'_>) {
    assert!(p.at(T![for]));
    p.bump(T![for]);
    if p.at(T![<]) {
        opt_generic_params(p);
    } else {
        p.error("expected `<`");
    }
}
pub fn for_type(p: &mut Parser<'_>, allow_bounds: bool) {
    assert!(p.at(T![for]));
    let m = p.start();
    for_binder(p);
    match p.current() {
        T![fn] | T![unsafe] | T![extern] => {},
        _ if is_use_path_start(p) => {},
        _ => {
            p.error("expected a function pointer or path");
        },
    }
    type_no_bounds(p);
    let completed = m.complete(p, FOR_TYPE);
    if allow_bounds {
        opt_type_bounds_as_dyn_trait_type(p, completed);
    }
}
fn impl_trait_type(p: &mut Parser<'_>) {
    assert!(p.at(T![impl]));
    let m = p.start();
    p.bump(T![impl]);
    bounds_without_colon(p);
    m.complete(p, IMPL_TRAIT_TYPE);
}
fn dyn_trait_type(p: &mut Parser<'_>) {
    assert!(p.at(T![dyn]));
    let m = p.start();
    p.bump(T![dyn]);
    bounds_without_colon(p);
    m.complete(p, DYN_TRAIT_TYPE);
}
fn bare_dyn_trait_type(p: &mut Parser<'_>) {
    let m = p.start();
    bounds_without_colon(p);
    m.complete(p, DYN_TRAIT_TYPE);
}
pub fn path_type(p: &mut Parser<'_>) {
    path_type_(p, true);
}
fn path_or_macro_type_(p: &mut Parser<'_>, allow_bounds: bool) {
    assert!(is_path_start(p));
    let r = p.start();
    let m = p.start();
    type_path(p);
    let kind = if p.at(T![!]) && !p.at(T![!=]) {
        items::macro_call_after_excl(p);
        m.complete(p, MACRO_CALL);
        MACRO_TYPE
    } else {
        m.abandon(p);
        PATH_TYPE
    };
    let path = r.complete(p, kind);
    if allow_bounds {
        opt_type_bounds_as_dyn_trait_type(p, path);
    }
}
pub fn path_type_(p: &mut Parser<'_>, allow_bounds: bool) {
    assert!(is_path_start(p));
    let m = p.start();
    type_path(p);
    let path = m.complete(p, PATH_TYPE);
    if allow_bounds {
        opt_type_bounds_as_dyn_trait_type(p, path);
    }
}
pub fn opt_type_bounds_as_dyn_trait_type(p: &mut Parser<'_>, type_marker: CompletedMarker) -> CompletedMarker {
    assert!(matches!(
        type_marker.kind(),
        SyntaxKind::PATH_TYPE | SyntaxKind::FOR_TYPE | SyntaxKind::MACRO_TYPE
    ));
    if !p.at(T![+]) {
        return type_marker;
    }
    let m = type_marker.precede(p).complete(p, TYPE_BOUND);
    let m = m.precede(p);
    p.eat(T![+]);
    let m = bounds_without_colon_m(p, m);
    m.precede(p).complete(p, DYN_TRAIT_TYPE)
}
