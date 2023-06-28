use super::*;

pub const RECOVERY_SET: TokenSet = TokenSet::new(&[
    T![fn],
    T![struct],
    T![enum],
    T![impl],
    T![trait],
    T![const],
    T![static],
    T![let],
    T![mod],
    T![pub],
    T![crate],
    T![use],
    T![macro],
    T![;],
]);

pub fn item_or_mac(x: &mut Parser<'_>, stop_on_r_curly: bool) {
    let y = x.start();
    attr::outers(x);
    let y = match opt_item(x, y) {
        Ok(()) => {
            if x.at(T![;]) {
                x.err_and_bump(
                    "expected item, found `;`\n\
                     consider removing this semicolon",
                );
            }
            return;
        },
        Err(x) => x,
    };
    if path::is_use_start(x) {
        use BlockLike::*;
        match mac_call(x) {
            Block => (),
            NotBlock => {
                x.expect(T![;]);
            },
        }
        y.complete(x, MACRO_CALL);
        return;
    }
    y.abandon(x);
    match x.current() {
        T!['{'] => err_block(x, "expected an item"),
        T!['}'] if !stop_on_r_curly => {
            let y = x.start();
            x.error("unmatched `}`");
            x.bump(T!['}']);
            y.complete(x, ERROR);
        },
        EOF | T!['}'] => x.error("expected an item"),
        _ => x.err_and_bump("expected an item"),
    }
}
pub fn opt_item(x: &mut Parser<'_>, y: Marker) -> Result<(), Marker> {
    let has_vis = is_opt_vis(x, false);
    let y = match opt_item_no_modifs(x, y) {
        Ok(()) => return Ok(()),
        Err(x) => x,
    };
    let mut has_mods = false;
    let mut has_extern = false;
    if x.at(T![const]) && x.nth(1) != T!['{'] {
        x.eat(T![const]);
        has_mods = true;
    }
    if x.at(T![async]) && !matches!(x.nth(1), T!['{'] | T![move] | T![|]) {
        x.eat(T![async]);
        has_mods = true;
    }
    if x.at(T![unsafe]) && x.nth(1) != T!['{'] {
        x.eat(T![unsafe]);
        has_mods = true;
    }
    if x.at(T![extern]) {
        has_extern = true;
        has_mods = true;
        abi(x);
    }
    if x.at_contextual_kw(T![auto]) && x.nth(1) == T![trait] {
        x.bump_remap(T![auto]);
        has_mods = true;
    }
    if x.at_contextual_kw(T![default]) {
        match x.nth(1) {
            T![fn] | T![type] | T![const] | T![impl] => {
                x.bump_remap(T![default]);
                has_mods = true;
            },
            T![unsafe] if matches!(x.nth(2), T![impl] | T![fn]) => {
                x.bump_remap(T![default]);
                x.bump(T![unsafe]);
                has_mods = true;
            },
            T![async] => {
                let mut maybe_fn = x.nth(2);
                let is_unsafe = if matches!(maybe_fn, T![unsafe]) {
                    maybe_fn = x.nth(3);
                    true
                } else {
                    false
                };
                if matches!(maybe_fn, T![fn]) {
                    x.bump_remap(T![default]);
                    x.bump(T![async]);
                    if is_unsafe {
                        x.bump(T![unsafe]);
                    }
                    has_mods = true;
                }
            },
            _ => (),
        }
    }
    if x.at_contextual_kw(T![existential]) && x.nth(1) == T![type] {
        x.bump_remap(T![existential]);
        has_mods = true;
    }
    match x.current() {
        T![fn] => for_fn(x, y),
        T![const] if x.nth(1) != T!['{'] => for_const(x, y),
        T![trait] => for_trait(x, y),
        T![impl] => for_impl(x, y),
        T![type] => for_type(x, y),
        T!['{'] if has_extern => {
            externs(x);
            y.complete(x, EXTERN_BLOCK);
        },
        _ if has_vis || has_mods => {
            if has_mods {
                x.error("expected existential, fn, trait or impl");
            } else {
                x.error("expected an item");
            }
            y.complete(x, ERROR);
        },
        _ => return Err(y),
    }
    Ok(())
}
fn opt_item_no_modifs(x: &mut Parser<'_>, y: Marker) -> Result<(), Marker> {
    let la = x.nth(1);
    match x.current() {
        T![extern] if la == T![crate] => extern_crate(x, y),
        T![use] => for_use(x, y),
        T![mod] => for_mod(x, y),
        T![type] => for_type(x, y),
        T![struct] => for_struct(x, y),
        T![enum] => for_enum(x, y),
        IDENT if x.at_contextual_kw(T![union]) && x.nth(1) == IDENT => for_union(x, y),
        T![macro] => mac_def(x, y),
        IDENT if x.at_contextual_kw(T![macro_rules]) && x.nth(1) == BANG => mac_rules(x, y),
        T![const] if (la == IDENT || la == T![_] || la == T![mut]) => for_const(x, y),
        T![static] if (la == IDENT || la == T![_] || la == T![mut]) => for_static(x, y),
        _ => return Err(y),
    };
    Ok(())
}
fn extern_crate(x: &mut Parser<'_>, y: Marker) {
    x.bump(T![extern]);
    x.bump(T![crate]);
    if x.at(T![self]) {
        let y = x.start();
        x.bump(T![self]);
        y.complete(x, NAME_REF);
    } else {
        name_ref(x);
    }
    opt_rename(x);
    x.expect(T![;]);
    y.complete(x, EXTERN_CRATE);
}
pub fn for_mod(x: &mut Parser<'_>, y: Marker) {
    x.bump(T![mod]);
    name(x);
    if x.at(T!['{']) {
        items(x);
    } else if !x.eat(T![;]) {
        x.error("expected `;` or `{`");
    }
    y.complete(x, MODULE);
}
fn for_type(x: &mut Parser<'_>, y: Marker) {
    x.bump(T![type]);
    name(x);
    generic::opt_params(x);
    if x.at(T![:]) {
        generic::bounds(x);
    }
    generic::opt_where_clause(x);
    if x.eat(T![=]) {
        ty(x);
    }
    generic::opt_where_clause(x);
    x.expect(T![;]);
    y.complete(x, TYPE_ALIAS);
}
pub fn items(x: &mut Parser<'_>) {
    assert!(x.at(T!['{']));
    let y = x.start();
    x.bump(T!['{']);
    mod_contents(x, true);
    x.expect(T!['}']);
    y.complete(x, ITEM_LIST);
}
pub fn externs(x: &mut Parser<'_>) {
    assert!(x.at(T!['{']));
    let y = x.start();
    x.bump(T!['{']);
    mod_contents(x, true);
    x.expect(T!['}']);
    y.complete(x, EXTERN_ITEM_LIST);
}
fn mac_rules(x: &mut Parser<'_>, y: Marker) {
    assert!(x.at_contextual_kw(T![macro_rules]));
    x.bump_remap(T![macro_rules]);
    x.expect(T![!]);
    if x.at(IDENT) {
        name(x);
    }
    if x.at(T![try]) {
        let y = x.start();
        x.bump_remap(IDENT);
        y.complete(x, NAME);
    }
    match x.current() {
        T!['['] | T!['('] => {
            token_tree(x);
            x.expect(T![;]);
        },
        T!['{'] => token_tree(x),
        _ => x.error("expected `{`, `[`, `(`"),
    }
    y.complete(x, MACRO_RULES);
}
fn mac_def(x: &mut Parser<'_>, y: Marker) {
    x.expect(T![macro]);
    name_r(x, RECOVERY_SET);
    if x.at(T!['{']) {
        token_tree(x);
    } else if x.at(T!['(']) {
        let y = x.start();
        token_tree(x);
        match x.current() {
            T!['{'] | T!['['] | T!['('] => token_tree(x),
            _ => x.error("expected `{`, `[`, `(`"),
        }
        y.complete(x, TOKEN_TREE);
    } else {
        x.error("unmatched `(`");
    }
    y.complete(x, MACRO_DEF);
}
fn for_fn(x: &mut Parser<'_>, y: Marker) {
    x.bump(T![fn]);
    name_r(x, RECOVERY_SET);
    generic::opt_params(x);
    if x.at(T!['(']) {
        param::fn_def(x);
    } else {
        x.error("expected function arguments");
    }
    is_opt_ret_type(x);
    generic::opt_where_clause(x);
    if x.at(T![;]) {
        x.bump(T![;]);
    } else {
        expr::block_expr(x);
    }
    y.complete(x, FN);
}
fn mac_call(x: &mut Parser<'_>) -> BlockLike {
    assert!(path::is_use_start(x));
    path::for_use(x);
    mac_call_after_excl(x)
}
pub fn mac_call_after_excl(x: &mut Parser<'_>) -> BlockLike {
    x.expect(T![!]);
    use BlockLike::*;
    match x.current() {
        T!['{'] => {
            token_tree(x);
            Block
        },
        T!['('] | T!['['] => {
            token_tree(x);
            NotBlock
        },
        _ => {
            x.error("expected `{`, `[`, `(`");
            NotBlock
        },
    }
}
pub fn token_tree(x: &mut Parser<'_>) {
    let closing_kind = match x.current() {
        T!['{'] => T!['}'],
        T!['('] => T![')'],
        T!['['] => T![']'],
        _ => unreachable!(),
    };
    let y = x.start();
    x.bump_any();
    while !x.at(EOF) && !x.at(closing_kind) {
        match x.current() {
            T!['{'] | T!['('] | T!['['] => token_tree(x),
            T!['}'] => {
                x.error("unmatched `}`");
                y.complete(x, TOKEN_TREE);
                return;
            },
            T![')'] | T![']'] => x.err_and_bump("unmatched brace"),
            _ => x.bump_any(),
        }
    }
    x.expect(closing_kind);
    y.complete(x, TOKEN_TREE);
}

pub fn for_struct(x: &mut Parser<'_>, y: Marker) {
    x.bump(T![struct]);
    struct_or_union(x, y, true);
}
pub fn for_union(x: &mut Parser<'_>, y: Marker) {
    assert!(x.at_contextual_kw(T![union]));
    x.bump_remap(T![union]);
    struct_or_union(x, y, false);
}
fn struct_or_union(x: &mut Parser<'_>, y: Marker, is_struct: bool) {
    name_r(x, RECOVERY_SET);
    generic::opt_params(x);
    match x.current() {
        T![where] => {
            generic::opt_where_clause(x);
            match x.current() {
                T![;] => x.bump(T![;]),
                T!['{'] => record_fields(x),
                _ => {
                    x.error("expected `;` or `{`");
                },
            }
        },
        T!['{'] => record_fields(x),
        T![;] if is_struct => {
            x.bump(T![;]);
        },
        T!['('] if is_struct => {
            tuple_fields(x);
            generic::opt_where_clause(x);
            x.expect(T![;]);
        },
        _ => x.error(if is_struct {
            "expected `;`, `{`, or `(`"
        } else {
            "expected `{`"
        }),
    }
    y.complete(x, if is_struct { STRUCT } else { UNION });
}
pub fn for_enum(x: &mut Parser<'_>, y: Marker) {
    x.bump(T![enum]);
    name_r(x, RECOVERY_SET);
    generic::opt_params(x);
    generic::opt_where_clause(x);
    if x.at(T!['{']) {
        variants(x);
    } else {
        x.error("expected `{`");
    }
    y.complete(x, ENUM);
}
pub fn variants(x: &mut Parser<'_>) {
    assert!(x.at(T!['{']));
    let y = x.start();
    x.bump(T!['{']);
    while !x.at(EOF) && !x.at(T!['}']) {
        if x.at(T!['{']) {
            err_block(x, "expected enum variant");
            continue;
        }
        one(x);
        if !x.at(T!['}']) {
            x.expect(T![,]);
        }
    }
    x.expect(T!['}']);
    y.complete(x, VARIANT_LIST);
    fn one(x: &mut Parser<'_>) {
        let y = x.start();
        attr::outers(x);
        if x.at(IDENT) {
            name(x);
            match x.current() {
                T!['{'] => record_fields(x),
                T!['('] => tuple_fields(x),
                _ => (),
            }
            if x.eat(T![=]) {
                expr::expr(x);
            }
            y.complete(x, VARIANT);
        } else {
            y.abandon(x);
            x.err_and_bump("expected enum variant");
        }
    }
}
pub fn record_fields(x: &mut Parser<'_>) {
    assert!(x.at(T!['{']));
    let y = x.start();
    x.bump(T!['{']);
    while !x.at(T!['}']) && !x.at(EOF) {
        if x.at(T!['{']) {
            err_block(x, "expected field");
            continue;
        }
        one(x);
        if !x.at(T!['}']) {
            x.expect(T![,]);
        }
    }
    x.expect(T!['}']);
    y.complete(x, RECORD_FIELD_LIST);
    fn one(x: &mut Parser<'_>) {
        let y = x.start();
        attr::outers(x);
        is_opt_vis(x, false);
        if x.at(IDENT) {
            name(x);
            x.expect(T![:]);
            ty(x);
            y.complete(x, RECORD_FIELD);
        } else {
            y.abandon(x);
            x.err_and_bump("expected field declaration");
        }
    }
}
const FIELD_FIRST: TokenSet = ty::FIRST.union(attr::FIRST).union(VIS_FIRST);
fn tuple_fields(x: &mut Parser<'_>) {
    assert!(x.at(T!['(']));
    let y = x.start();
    delimited(x, T!['('], T![')'], T![,], FIELD_FIRST, |x| {
        let y = x.start();
        attr::outers(x);
        let has_vis = is_opt_vis(x, true);
        if !x.at_ts(ty::FIRST) {
            x.error("expected a type");
            if has_vis {
                y.complete(x, ERROR);
            } else {
                y.abandon(x);
            }
            return false;
        }
        ty(x);
        y.complete(x, TUPLE_FIELD);
        true
    });
    y.complete(x, TUPLE_FIELD_LIST);
}

pub fn for_const(x: &mut Parser<'_>, y: Marker) {
    x.bump(T![const]);
    const_or_static(x, y, true);
}
pub fn for_static(x: &mut Parser<'_>, y: Marker) {
    x.bump(T![static]);
    const_or_static(x, y, false);
}
fn const_or_static(x: &mut Parser<'_>, y: Marker, is_const: bool) {
    x.eat(T![mut]);
    if is_const && x.eat(T![_]) {
    } else {
        name(x);
    }
    if x.at(T![:]) {
        ty::ascription(x);
    } else {
        x.error("missing type for `const` or `static`");
    }
    if x.eat(T![=]) {
        expr::expr(x);
    }
    x.expect(T![;]);
    y.complete(x, if is_const { CONST } else { STATIC });
}

pub fn for_trait(x: &mut Parser<'_>, y: Marker) {
    x.bump(T![trait]);
    name_r(x, RECOVERY_SET);
    generic::opt_params(x);
    if x.eat(T![=]) {
        generic::bounds_no_colon(x);
        generic::opt_where_clause(x);
        x.expect(T![;]);
        y.complete(x, TRAIT_ALIAS);
        return;
    }
    if x.at(T![:]) {
        generic::bounds(x);
    }
    generic::opt_where_clause(x);
    if x.at(T!['{']) {
        assoc_items(x);
    } else {
        x.error("expected `{`");
    }
    y.complete(x, TRAIT);
}
pub fn for_impl(x: &mut Parser<'_>, y: Marker) {
    x.bump(T![impl]);
    if x.at(T![<]) && is_not_qual_path(x) {
        generic::opt_params(x);
    }
    x.eat(T![const]);
    x.eat(T![!]);
    impl_type(x);
    if x.eat(T![for]) {
        impl_type(x);
    }
    generic::opt_where_clause(x);
    if x.at(T!['{']) {
        assoc_items(x);
    } else {
        x.error("expected `{`");
    }
    y.complete(x, IMPL);
}
pub fn assoc_items(x: &mut Parser<'_>) {
    assert!(x.at(T!['{']));
    let y = x.start();
    x.bump(T!['{']);
    attr::inners(x);
    while !x.at(EOF) && !x.at(T!['}']) {
        if x.at(T!['{']) {
            err_block(x, "expected an item");
            continue;
        }
        item_or_mac(x, true);
    }
    x.expect(T!['}']);
    y.complete(x, ASSOC_ITEM_LIST);
}
fn is_not_qual_path(x: &Parser<'_>) -> bool {
    if x.nth(1) == T![#] || x.nth(1) == T![>] || x.nth(1) == T![const] {
        return true;
    }
    (x.nth(1) == LIFETIME_IDENT || x.nth(1) == IDENT)
        && (x.nth(2) == T![>] || x.nth(2) == T![,] || x.nth(2) == T![:] || x.nth(2) == T![=])
}
pub fn impl_type(x: &mut Parser<'_>) {
    if x.at(T![impl]) {
        x.error("expected trait or type");
        return;
    }
    ty(x);
}

pub fn for_use(x: &mut Parser<'_>, y: Marker) {
    x.bump(T![use]);
    use_one(x, true);
    x.expect(T![;]);
    y.complete(x, USE);
}
fn use_one(x: &mut Parser<'_>, top: bool) {
    let y = x.start();
    match x.current() {
        T![*] => x.bump(T![*]),
        T![:] if x.at(T![::]) && x.nth(2) == T![*] => {
            x.bump(T![::]);
            x.bump(T![*]);
        },
        T!['{'] => use_trees(x),
        T![:] if x.at(T![::]) && x.nth(2) == T!['{'] => {
            x.bump(T![::]);
            use_trees(x);
        },
        _ if path::is_use_start(x) => {
            path::for_use(x);
            match x.current() {
                T![as] => opt_rename(x),
                T![:] if x.at(T![::]) => {
                    x.bump(T![::]);
                    match x.current() {
                        T![*] => x.bump(T![*]),
                        T!['{'] => use_trees(x),
                        _ => x.error("expected `{` or `*`"),
                    }
                },
                _ => (),
            }
        },
        _ => {
            y.abandon(x);
            let msg = "expected one of `*`, `::`, `{`, `self`, `super` or an identifier";
            if top {
                x.err_recover(msg, RECOVERY_SET);
            } else {
                x.err_and_bump(msg);
            }
            return;
        },
    }
    y.complete(x, USE_TREE);
}
pub fn use_trees(x: &mut Parser<'_>) {
    assert!(x.at(T!['{']));
    let y = x.start();
    x.bump(T!['{']);
    while !x.at(EOF) && !x.at(T!['}']) {
        use_one(x, false);
        if !x.at(T!['}']) {
            x.expect(T![,]);
        }
    }
    x.expect(T!['}']);
    y.complete(x, USE_TREE_LIST);
}

pub fn mod_contents(x: &mut Parser<'_>, stop_on_r_curly: bool) {
    attr::inners(x);
    while !(x.at(EOF) || (x.at(T!['}']) && stop_on_r_curly)) {
        item_or_mac(x, stop_on_r_curly);
    }
}
