pub use self::{
    adt::{record_field_list, variant_list},
    expr::{match_arm_list, record_expr_field_list},
    traits::assoc_item_list,
    use_item::use_tree_list,
};
use super::*;

mod adt {
    use super::*;
    pub fn strukt(p: &mut Parser<'_>, m: Marker) {
        p.bump(T![struct]);
        struct_or_union(p, m, true);
    }
    pub fn union(p: &mut Parser<'_>, m: Marker) {
        assert!(p.at_contextual_kw(T![union]));
        p.bump_remap(T![union]);
        struct_or_union(p, m, false);
    }
    fn struct_or_union(p: &mut Parser<'_>, m: Marker, is_struct: bool) {
        name_r(p, ITEM_RECOVERY_SET);
        generic::opt_params(p);
        match p.current() {
            T![where] => {
                generic::opt_where_clause(p);
                match p.current() {
                    T![;] => p.bump(T![;]),
                    T!['{'] => record_field_list(p),
                    _ => {
                        p.error("expected `;` or `{`");
                    },
                }
            },
            T!['{'] => record_field_list(p),
            T![;] if is_struct => {
                p.bump(T![;]);
            },
            T!['('] if is_struct => {
                tuple_field_list(p);
                generic::opt_where_clause(p);
                p.expect(T![;]);
            },
            _ => p.error(if is_struct {
                "expected `;`, `{`, or `(`"
            } else {
                "expected `{`"
            }),
        }
        m.complete(p, if is_struct { STRUCT } else { UNION });
    }
    pub fn enum_(p: &mut Parser<'_>, m: Marker) {
        p.bump(T![enum]);
        name_r(p, ITEM_RECOVERY_SET);
        generic::opt_params(p);
        generic::opt_where_clause(p);
        if p.at(T!['{']) {
            variant_list(p);
        } else {
            p.error("expected `{`");
        }
        m.complete(p, ENUM);
    }
    pub fn variant_list(p: &mut Parser<'_>) {
        assert!(p.at(T!['{']));
        let m = p.start();
        p.bump(T!['{']);
        while !p.at(EOF) && !p.at(T!['}']) {
            if p.at(T!['{']) {
                err_block(p, "expected enum variant");
                continue;
            }
            variant(p);
            if !p.at(T!['}']) {
                p.expect(T![,]);
            }
        }
        p.expect(T!['}']);
        m.complete(p, VARIANT_LIST);
        fn variant(p: &mut Parser<'_>) {
            let m = p.start();
            attr::outers(p);
            if p.at(IDENT) {
                name(p);
                match p.current() {
                    T!['{'] => record_field_list(p),
                    T!['('] => tuple_field_list(p),
                    _ => (),
                }
                if p.eat(T![=]) {
                    expr::expr(p);
                }
                m.complete(p, VARIANT);
            } else {
                m.abandon(p);
                p.err_and_bump("expected enum variant");
            }
        }
    }
    pub fn record_field_list(p: &mut Parser<'_>) {
        assert!(p.at(T!['{']));
        let m = p.start();
        p.bump(T!['{']);
        while !p.at(T!['}']) && !p.at(EOF) {
            if p.at(T!['{']) {
                err_block(p, "expected field");
                continue;
            }
            record_field(p);
            if !p.at(T!['}']) {
                p.expect(T![,]);
            }
        }
        p.expect(T!['}']);
        m.complete(p, RECORD_FIELD_LIST);
        fn record_field(p: &mut Parser<'_>) {
            let m = p.start();
            attr::outers(p);
            is_opt_vis(p, false);
            if p.at(IDENT) {
                name(p);
                p.expect(T![:]);
                ty(p);
                m.complete(p, RECORD_FIELD);
            } else {
                m.abandon(p);
                p.err_and_bump("expected field declaration");
            }
        }
    }
    const TUPLE_FIELD_FIRST: TokenSet = ty::FIRST.union(attr::FIRST).union(VIS_FIRST);
    fn tuple_field_list(p: &mut Parser<'_>) {
        assert!(p.at(T!['(']));
        let m = p.start();
        delimited(p, T!['('], T![')'], T![,], TUPLE_FIELD_FIRST, |p| {
            let m = p.start();
            attr::outers(p);
            let has_vis = is_opt_vis(p, true);
            if !p.at_ts(ty::FIRST) {
                p.error("expected a type");
                if has_vis {
                    m.complete(p, ERROR);
                } else {
                    m.abandon(p);
                }
                return false;
            }
            ty(p);
            m.complete(p, TUPLE_FIELD);
            true
        });
        m.complete(p, TUPLE_FIELD_LIST);
    }
}
mod consts {
    use super::*;
    pub fn konst(p: &mut Parser<'_>, m: Marker) {
        p.bump(T![const]);
        const_or_static(p, m, true);
    }
    pub fn static_(p: &mut Parser<'_>, m: Marker) {
        p.bump(T![static]);
        const_or_static(p, m, false);
    }
    fn const_or_static(p: &mut Parser<'_>, m: Marker, is_const: bool) {
        p.eat(T![mut]);
        if is_const && p.eat(T![_]) {
        } else {
            name(p);
        }
        if p.at(T![:]) {
            ty::ascription(p);
        } else {
            p.error("missing type for `const` or `static`");
        }
        if p.eat(T![=]) {
            expr::expr(p);
        }
        p.expect(T![;]);
        m.complete(p, if is_const { CONST } else { STATIC });
    }
}
mod traits {
    use super::*;
    pub fn trait_(p: &mut Parser<'_>, m: Marker) {
        p.bump(T![trait]);
        name_r(p, ITEM_RECOVERY_SET);
        generic::opt_params(p);
        if p.eat(T![=]) {
            generic::bounds_no_colon(p);
            generic::opt_where_clause(p);
            p.expect(T![;]);
            m.complete(p, TRAIT_ALIAS);
            return;
        }
        if p.at(T![:]) {
            generic::bounds(p);
        }
        generic::opt_where_clause(p);
        if p.at(T!['{']) {
            assoc_item_list(p);
        } else {
            p.error("expected `{`");
        }
        m.complete(p, TRAIT);
    }
    pub fn impl_(p: &mut Parser<'_>, m: Marker) {
        p.bump(T![impl]);
        if p.at(T![<]) && not_a_qualified_path(p) {
            generic::opt_params(p);
        }
        p.eat(T![const]);
        p.eat(T![!]);
        impl_type(p);
        if p.eat(T![for]) {
            impl_type(p);
        }
        generic::opt_where_clause(p);
        if p.at(T!['{']) {
            assoc_item_list(p);
        } else {
            p.error("expected `{`");
        }
        m.complete(p, IMPL);
    }
    pub fn assoc_item_list(p: &mut Parser<'_>) {
        assert!(p.at(T!['{']));
        let m = p.start();
        p.bump(T!['{']);
        attr::inners(p);
        while !p.at(EOF) && !p.at(T!['}']) {
            if p.at(T!['{']) {
                err_block(p, "expected an item");
                continue;
            }
            item_or_macro(p, true);
        }
        p.expect(T!['}']);
        m.complete(p, ASSOC_ITEM_LIST);
    }
    fn not_a_qualified_path(p: &Parser<'_>) -> bool {
        if p.nth(1) == T![#] || p.nth(1) == T![>] || p.nth(1) == T![const] {
            return true;
        }
        (p.nth(1) == LIFETIME_IDENT || p.nth(1) == IDENT)
            && (p.nth(2) == T![>] || p.nth(2) == T![,] || p.nth(2) == T![:] || p.nth(2) == T![=])
    }
    pub fn impl_type(p: &mut Parser<'_>) {
        if p.at(T![impl]) {
            p.error("expected trait or type");
            return;
        }
        ty(p);
    }
}
mod use_item {
    use super::*;
    pub fn use_(p: &mut Parser<'_>, m: Marker) {
        p.bump(T![use]);
        use_tree(p, true);
        p.expect(T![;]);
        m.complete(p, USE);
    }
    fn use_tree(p: &mut Parser<'_>, top_level: bool) {
        let m = p.start();
        match p.current() {
            T![*] => p.bump(T![*]),
            T![:] if p.at(T![::]) && p.nth(2) == T![*] => {
                p.bump(T![::]);
                p.bump(T![*]);
            },
            T!['{'] => use_tree_list(p),
            T![:] if p.at(T![::]) && p.nth(2) == T!['{'] => {
                p.bump(T![::]);
                use_tree_list(p);
            },
            _ if path::is_use_start(p) => {
                path::for_use(p);
                match p.current() {
                    T![as] => opt_rename(p),
                    T![:] if p.at(T![::]) => {
                        p.bump(T![::]);
                        match p.current() {
                            T![*] => p.bump(T![*]),
                            T!['{'] => use_tree_list(p),
                            _ => p.error("expected `{` or `*`"),
                        }
                    },
                    _ => (),
                }
            },
            _ => {
                m.abandon(p);
                let msg = "expected one of `*`, `::`, `{`, `self`, `super` or an identifier";
                if top_level {
                    p.err_recover(msg, ITEM_RECOVERY_SET);
                } else {
                    p.err_and_bump(msg);
                }
                return;
            },
        }
        m.complete(p, USE_TREE);
    }
    pub fn use_tree_list(p: &mut Parser<'_>) {
        assert!(p.at(T!['{']));
        let m = p.start();
        p.bump(T!['{']);
        while !p.at(EOF) && !p.at(T!['}']) {
            use_tree(p, false);
            if !p.at(T!['}']) {
                p.expect(T![,]);
            }
        }
        p.expect(T!['}']);
        m.complete(p, USE_TREE_LIST);
    }
}

pub fn mod_contents(p: &mut Parser<'_>, stop_on_r_curly: bool) {
    attr::inners(p);
    while !(p.at(EOF) || (p.at(T!['}']) && stop_on_r_curly)) {
        item_or_macro(p, stop_on_r_curly);
    }
}
pub const ITEM_RECOVERY_SET: TokenSet = TokenSet::new(&[
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
pub fn item_or_macro(p: &mut Parser<'_>, stop_on_r_curly: bool) {
    let m = p.start();
    attr::outers(p);
    let m = match opt_item(p, m) {
        Ok(()) => {
            if p.at(T![;]) {
                p.err_and_bump(
                    "expected item, found `;`\n\
                     consider removing this semicolon",
                );
            }
            return;
        },
        Err(m) => m,
    };
    if path::is_use_start(p) {
        match macro_call(p) {
            BlockLike::Block => (),
            BlockLike::NotBlock => {
                p.expect(T![;]);
            },
        }
        m.complete(p, MACRO_CALL);
        return;
    }
    m.abandon(p);
    match p.current() {
        T!['{'] => err_block(p, "expected an item"),
        T!['}'] if !stop_on_r_curly => {
            let e = p.start();
            p.error("unmatched `}`");
            p.bump(T!['}']);
            e.complete(p, ERROR);
        },
        EOF | T!['}'] => p.error("expected an item"),
        _ => p.err_and_bump("expected an item"),
    }
}
pub fn opt_item(p: &mut Parser<'_>, m: Marker) -> Result<(), Marker> {
    let has_visibility = is_opt_vis(p, false);
    let m = match opt_item_without_modifiers(p, m) {
        Ok(()) => return Ok(()),
        Err(m) => m,
    };
    let mut has_mods = false;
    let mut has_extern = false;
    if p.at(T![const]) && p.nth(1) != T!['{'] {
        p.eat(T![const]);
        has_mods = true;
    }
    if p.at(T![async]) && !matches!(p.nth(1), T!['{'] | T![move] | T![|]) {
        p.eat(T![async]);
        has_mods = true;
    }
    if p.at(T![unsafe]) && p.nth(1) != T!['{'] {
        p.eat(T![unsafe]);
        has_mods = true;
    }
    if p.at(T![extern]) {
        has_extern = true;
        has_mods = true;
        abi(p);
    }
    if p.at_contextual_kw(T![auto]) && p.nth(1) == T![trait] {
        p.bump_remap(T![auto]);
        has_mods = true;
    }
    if p.at_contextual_kw(T![default]) {
        match p.nth(1) {
            T![fn] | T![type] | T![const] | T![impl] => {
                p.bump_remap(T![default]);
                has_mods = true;
            },
            T![unsafe] if matches!(p.nth(2), T![impl] | T![fn]) => {
                p.bump_remap(T![default]);
                p.bump(T![unsafe]);
                has_mods = true;
            },
            T![async] => {
                let mut maybe_fn = p.nth(2);
                let is_unsafe = if matches!(maybe_fn, T![unsafe]) {
                    maybe_fn = p.nth(3);
                    true
                } else {
                    false
                };
                if matches!(maybe_fn, T![fn]) {
                    p.bump_remap(T![default]);
                    p.bump(T![async]);
                    if is_unsafe {
                        p.bump(T![unsafe]);
                    }
                    has_mods = true;
                }
            },
            _ => (),
        }
    }
    if p.at_contextual_kw(T![existential]) && p.nth(1) == T![type] {
        p.bump_remap(T![existential]);
        has_mods = true;
    }
    match p.current() {
        T![fn] => fn_(p, m),
        T![const] if p.nth(1) != T!['{'] => consts::konst(p, m),
        T![trait] => traits::trait_(p, m),
        T![impl] => traits::impl_(p, m),
        T![type] => type_alias(p, m),
        T!['{'] if has_extern => {
            extern_item_list(p);
            m.complete(p, EXTERN_BLOCK);
        },
        _ if has_visibility || has_mods => {
            if has_mods {
                p.error("expected existential, fn, trait or impl");
            } else {
                p.error("expected an item");
            }
            m.complete(p, ERROR);
        },
        _ => return Err(m),
    }
    Ok(())
}
fn opt_item_without_modifiers(p: &mut Parser<'_>, m: Marker) -> Result<(), Marker> {
    let la = p.nth(1);
    match p.current() {
        T![extern] if la == T![crate] => extern_crate(p, m),
        T![use] => use_item::use_(p, m),
        T![mod] => mod_item(p, m),
        T![type] => type_alias(p, m),
        T![struct] => adt::strukt(p, m),
        T![enum] => adt::enum_(p, m),
        IDENT if p.at_contextual_kw(T![union]) && p.nth(1) == IDENT => adt::union(p, m),
        T![macro] => macro_def(p, m),
        IDENT if p.at_contextual_kw(T![macro_rules]) && p.nth(1) == BANG => macro_rules(p, m),
        T![const] if (la == IDENT || la == T![_] || la == T![mut]) => consts::konst(p, m),
        T![static] if (la == IDENT || la == T![_] || la == T![mut]) => consts::static_(p, m),
        _ => return Err(m),
    };
    Ok(())
}
fn extern_crate(p: &mut Parser<'_>, m: Marker) {
    p.bump(T![extern]);
    p.bump(T![crate]);
    if p.at(T![self]) {
        let m = p.start();
        p.bump(T![self]);
        m.complete(p, NAME_REF);
    } else {
        name_ref(p);
    }
    opt_rename(p);
    p.expect(T![;]);
    m.complete(p, EXTERN_CRATE);
}
pub fn mod_item(p: &mut Parser<'_>, m: Marker) {
    p.bump(T![mod]);
    name(p);
    if p.at(T!['{']) {
        item_list(p);
    } else if !p.eat(T![;]) {
        p.error("expected `;` or `{`");
    }
    m.complete(p, MODULE);
}
fn type_alias(p: &mut Parser<'_>, m: Marker) {
    p.bump(T![type]);
    name(p);
    generic::opt_params(p);
    if p.at(T![:]) {
        generic::bounds(p);
    }
    generic::opt_where_clause(p);
    if p.eat(T![=]) {
        ty(p);
    }
    generic::opt_where_clause(p);
    p.expect(T![;]);
    m.complete(p, TYPE_ALIAS);
}
pub fn item_list(p: &mut Parser<'_>) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump(T!['{']);
    mod_contents(p, true);
    p.expect(T!['}']);
    m.complete(p, ITEM_LIST);
}
pub fn extern_item_list(p: &mut Parser<'_>) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump(T!['{']);
    mod_contents(p, true);
    p.expect(T!['}']);
    m.complete(p, EXTERN_ITEM_LIST);
}
fn macro_rules(p: &mut Parser<'_>, m: Marker) {
    assert!(p.at_contextual_kw(T![macro_rules]));
    p.bump_remap(T![macro_rules]);
    p.expect(T![!]);
    if p.at(IDENT) {
        name(p);
    }
    if p.at(T![try]) {
        let m = p.start();
        p.bump_remap(IDENT);
        m.complete(p, NAME);
    }
    match p.current() {
        T!['['] | T!['('] => {
            token_tree(p);
            p.expect(T![;]);
        },
        T!['{'] => token_tree(p),
        _ => p.error("expected `{`, `[`, `(`"),
    }
    m.complete(p, MACRO_RULES);
}
fn macro_def(p: &mut Parser<'_>, m: Marker) {
    p.expect(T![macro]);
    name_r(p, ITEM_RECOVERY_SET);
    if p.at(T!['{']) {
        token_tree(p);
    } else if p.at(T!['(']) {
        let m = p.start();
        token_tree(p);
        match p.current() {
            T!['{'] | T!['['] | T!['('] => token_tree(p),
            _ => p.error("expected `{`, `[`, `(`"),
        }
        m.complete(p, TOKEN_TREE);
    } else {
        p.error("unmatched `(`");
    }
    m.complete(p, MACRO_DEF);
}
fn fn_(p: &mut Parser<'_>, m: Marker) {
    p.bump(T![fn]);
    name_r(p, ITEM_RECOVERY_SET);
    generic::opt_params(p);
    if p.at(T!['(']) {
        param::fn_def(p);
    } else {
        p.error("expected function arguments");
    }
    is_opt_ret_type(p);
    generic::opt_where_clause(p);
    if p.at(T![;]) {
        p.bump(T![;]);
    } else {
        expr::block_expr(p);
    }
    m.complete(p, FN);
}
fn macro_call(p: &mut Parser<'_>) -> BlockLike {
    assert!(path::is_use_start(p));
    path::for_use(p);
    macro_call_after_excl(p)
}
pub fn macro_call_after_excl(p: &mut Parser<'_>) -> BlockLike {
    p.expect(T![!]);
    match p.current() {
        T!['{'] => {
            token_tree(p);
            BlockLike::Block
        },
        T!['('] | T!['['] => {
            token_tree(p);
            BlockLike::NotBlock
        },
        _ => {
            p.error("expected `{`, `[`, `(`");
            BlockLike::NotBlock
        },
    }
}
pub fn token_tree(p: &mut Parser<'_>) {
    let closing_paren_kind = match p.current() {
        T!['{'] => T!['}'],
        T!['('] => T![')'],
        T!['['] => T![']'],
        _ => unreachable!(),
    };
    let m = p.start();
    p.bump_any();
    while !p.at(EOF) && !p.at(closing_paren_kind) {
        match p.current() {
            T!['{'] | T!['('] | T!['['] => token_tree(p),
            T!['}'] => {
                p.error("unmatched `}`");
                m.complete(p, TOKEN_TREE);
                return;
            },
            T![')'] | T![']'] => p.err_and_bump("unmatched brace"),
            _ => p.bump_any(),
        }
    }
    p.expect(closing_paren_kind);
    m.complete(p, TOKEN_TREE);
}
