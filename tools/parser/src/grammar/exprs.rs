use super::*;

mod atom {
    use super::*;
    // test expr_literals
    // fn foo() {
    //     let _ = true;
    //     let _ = false;
    //     let _ = 1;
    //     let _ = 2.0;
    //     let _ = b'a';
    //     let _ = 'b';
    //     let _ = "c";
    //     let _ = r"d";
    //     let _ = b"e";
    //     let _ = br"f";
    //     let _ = c"g";
    //     let _ = cr"h";
    // }
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
    // E.g. for after the break in `if break {}`, this should not match
    pub const ATOM_EXPR_FIRST: TokenSet = LITERAL_FIRST.union(PATH_FIRST).union(TokenSet::new(&[
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
        if is_path_start(p) {
            return Some(path_expr(p, r));
        }
        let la = p.nth(1);
        let done = match p.current() {
            T!['('] => tuple_expr(p),
            T!['['] => array_expr(p),
            T![if] => if_expr(p),
            T![let] => let_expr(p),
            T![_] => {
                // test destructuring_assignment_wildcard_pat
                // fn foo() {
                //     _ = 1;
                //     Some(_) = None;
                // }
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
                    // test labeled_block
                    // fn f() { 'label: {}; }
                    T!['{'] => {
                        stmt_list(p);
                        m.complete(p, BLOCK_EXPR)
                    },
                    _ => {
                        // test_err misplaced_label_err
                        // fn main() {
                        //     'loop: impl
                        // }
                        p.error("expected a loop or block");
                        m.complete(p, ERROR);
                        return None;
                    },
                }
            },
            // test effect_blocks
            // fn f() { unsafe { } }
            // fn f() { const { } }
            // fn f() { async { } }
            // fn f() { async move { } }
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
                // test for_range_from
                // fn foo() {
                //    for x in 0 .. {
                //        break;
                //    }
                // }
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
    // test tuple_expr
    // fn foo() {
    //     ();
    //     (1);
    //     (1,);
    // }
    fn tuple_expr(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T!['(']));
        let m = p.start();
        p.expect(T!['(']);
        let mut saw_comma = false;
        let mut saw_expr = false;
        // test_err tuple_expr_leading_comma
        // fn foo() {
        //     (,);
        // }
        if p.eat(T![,]) {
            p.error("expected expression");
            saw_comma = true;
        }
        while !p.at(EOF) && !p.at(T![')']) {
            saw_expr = true;
            // test tuple_attrs
            // const A: (i64, i64) = (1, #[cfg(test)] 2);
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
    // test array_expr
    // fn foo() {
    //     [];
    //     [1];
    //     [1, 2,];
    //     [1; 2];
    // }
    fn array_expr(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T!['[']));
        let m = p.start();
        let mut n_exprs = 0u32;
        let mut has_semi = false;
        p.bump(T!['[']);
        while !p.at(EOF) && !p.at(T![']']) {
            n_exprs += 1;
            // test array_attrs
            // const A: &[i64] = &[1, #[cfg(test)] 2];
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
    // test lambda_expr
    // fn foo() {
    //     || ();
    //     || -> i32 { 92 };
    //     |x| x;
    //     move |x: i32,| x;
    //     async || {};
    //     move || {};
    //     async move || {};
    //     static || {};
    //     static move || {};
    //     static async || {};
    //     static async move || {};
    //     for<'a> || {};
    //     for<'a> move || {};
    // }
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
        // test const_closure
        // fn main() { let cl = const || _ = 0; }
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
            // test lambda_ret_block
            // fn main() { || -> i32 { 92 }(); }
            block_expr(p);
        } else if p.at_ts(EXPR_FIRST) {
            // test closure_body_underscore_assignment
            // fn main() { || _ = 0; }
            expr(p);
        } else {
            p.error("expected expression");
        }
        m.complete(p, CLOSURE_EXPR)
    }
    // test if_expr
    // fn foo() {
    //     if true {};
    //     if true {} else {};
    //     if true {} else if false {} else {};
    //     if S {};
    //     if { true } { } else { };
    // }
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
    // test label
    // fn foo() {
    //     'a: loop {}
    //     'b: while true {}
    //     'c: for x in () {}
    // }
    fn label(p: &mut Parser<'_>) {
        assert!(p.at(LIFETIME_IDENT) && p.nth(1) == T![:]);
        let m = p.start();
        lifetime(p);
        p.bump_any();
        m.complete(p, LABEL);
    }
    // test loop_expr
    // fn foo() {
    //     loop {};
    // }
    fn loop_expr(p: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
        assert!(p.at(T![loop]));
        let m = m.unwrap_or_else(|| p.start());
        p.bump(T![loop]);
        block_expr(p);
        m.complete(p, LOOP_EXPR)
    }
    // test while_expr
    // fn foo() {
    //     while true {};
    //     while let Some(x) = it.next() {};
    //     while { true } {};
    // }
    fn while_expr(p: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
        assert!(p.at(T![while]));
        let m = m.unwrap_or_else(|| p.start());
        p.bump(T![while]);
        expr_no_struct(p);
        block_expr(p);
        m.complete(p, WHILE_EXPR)
    }
    // test for_expr
    // fn foo() {
    //     for x in [] {};
    // }
    fn for_expr(p: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
        assert!(p.at(T![for]));
        let m = m.unwrap_or_else(|| p.start());
        p.bump(T![for]);
        patterns::pattern(p);
        p.expect(T![in]);
        expr_no_struct(p);
        block_expr(p);
        m.complete(p, FOR_EXPR)
    }
    // test let_expr
    // fn foo() {
    //     if let Some(_) = None && true {}
    //     while 1 == 5 && (let None = None) {}
    // }
    fn let_expr(p: &mut Parser<'_>) -> CompletedMarker {
        let m = p.start();
        p.bump(T![let]);
        patterns::pattern_top(p);
        p.expect(T![=]);
        expr_let(p);
        m.complete(p, LET_EXPR)
    }
    // test match_expr
    // fn foo() {
    //     match () { };
    //     match S {};
    //     match { } { _ => () };
    //     match { S {} } {};
    // }
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
        // test match_arms_inner_attribute
        // fn foo() {
        //     match () {
        //         #![doc("Inner attribute")]
        //         #![doc("Can be")]
        //         #![doc("Stacked")]
        //         _ => (),
        //     }
        // }
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
    // test match_arm
    // fn foo() {
    //     match () {
    //         _ => (),
    //         _ if Test > Test{field: 0} => (),
    //         X | Y if Z => (),
    //         | X | Y if Z => (),
    //         | X => (),
    //     };
    // }
    fn match_arm(p: &mut Parser<'_>) {
        let m = p.start();
        // test match_arms_outer_attributes
        // fn foo() {
        //     match () {
        //         #[cfg(feature = "some")]
        //         _ => (),
        //         #[cfg(feature = "other")]
        //         _ => (),
        //         #[cfg(feature = "many")]
        //         #[cfg(feature = "attributes")]
        //         #[cfg(feature = "before")]
        //         _ => (),
        //     }
        // }
        attr::outers(p);
        patterns::pattern_top_r(p, TokenSet::EMPTY);
        if p.at(T![if]) {
            match_guard(p);
        }
        p.expect(T![=>]);
        let blocklike = match expr_stmt(p, None) {
            Some((_, blocklike)) => blocklike,
            None => BlockLike::NotBlock,
        };
        // test match_arms_commas
        // fn foo() {
        //     match () {
        //         _ => (),
        //         _ => {}
        //         _ => ()
        //     }
        // }
        if !p.eat(T![,]) && !blocklike.is_block() && !p.at(T!['}']) {
            p.error("expected `,`");
        }
        m.complete(p, MATCH_ARM);
    }
    // test match_guard
    // fn foo() {
    //     match () {
    //         _ if foo => (),
    //         _ if let foo = bar => (),
    //     }
    // }
    fn match_guard(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T![if]));
        let m = p.start();
        p.bump(T![if]);
        expr(p);
        m.complete(p, MATCH_GUARD)
    }
    // test block
    // fn a() {}
    // fn b() { let _ = 1; }
    // fn c() { 1; 2; }
    // fn d() { 1; 2 }
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
    // test return_expr
    // fn foo() {
    //     return;
    //     return 92;
    // }
    fn return_expr(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T![return]));
        let m = p.start();
        p.bump(T![return]);
        if p.at_ts(EXPR_FIRST) {
            expr(p);
        }
        m.complete(p, RETURN_EXPR)
    }
    // test yield_expr
    // fn foo() {
    //     yield;
    //     yield 1;
    // }
    fn yield_expr(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T![yield]));
        let m = p.start();
        p.bump(T![yield]);
        if p.at_ts(EXPR_FIRST) {
            expr(p);
        }
        m.complete(p, YIELD_EXPR)
    }
    // test yeet_expr
    // fn foo() {
    //     do yeet;
    //     do yeet 1
    // }
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
    // test continue_expr
    // fn foo() {
    //     loop {
    //         continue;
    //         continue 'l;
    //     }
    // }
    fn continue_expr(p: &mut Parser<'_>) -> CompletedMarker {
        assert!(p.at(T![continue]));
        let m = p.start();
        p.bump(T![continue]);
        if p.at(LIFETIME_IDENT) {
            lifetime(p);
        }
        m.complete(p, CONTINUE_EXPR)
    }
    // test break_expr
    // fn foo() {
    //     loop {
    //         break;
    //         break 'l;
    //         break 92;
    //         break 'l 92;
    //     }
    // }
    fn break_expr(p: &mut Parser<'_>, r: Restrictions) -> CompletedMarker {
        assert!(p.at(T![break]));
        let m = p.start();
        p.bump(T![break]);
        if p.at(LIFETIME_IDENT) {
            lifetime(p);
        }
        // test break_ambiguity
        // fn foo(){
        //     if break {}
        //     while break {}
        //     for i in break {}
        //     match break {}
        // }
        if p.at_ts(EXPR_FIRST) && !(r.forbid_structs && p.at(T!['{'])) {
            expr(p);
        }
        m.complete(p, BREAK_EXPR)
    }
    // test try_block_expr
    // fn foo() {
    //     let _ = try {};
    // }
    fn try_block_expr(p: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
        assert!(p.at(T![try]));
        let m = m.unwrap_or_else(|| p.start());
        // Special-case `try!` as macro.
        // This is a hack until we do proper edition support
        if p.nth_at(1, T![!]) {
            // test try_macro_fallback
            // fn foo() { try!(Ok(())); }
            let macro_call = p.start();
            let path = p.start();
            let path_segment = p.start();
            let name_ref = p.start();
            p.bump_remap(IDENT);
            name_ref.complete(p, NAME_REF);
            path_segment.complete(p, PATH_SEGMENT);
            path.complete(p, PATH);
            let _block_like = items::macro_call_after_excl(p);
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
    // test box_expr
    // fn foo() {
    //     let x = box 1i32;
    //     let y = (box 1i32, box 2i32);
    //     let z = Foo(box 1i32, box 2i32);
    // }
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
                use_path(x);
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
    // test attr_on_expr_stmt
    // fn foo() {
    //     #[A] foo();
    //     #[B] bar!{}
    //     #[C] #[D] {}
    //     #[D] return ();
    // }
    attr::outers(p);
    if p.at(T![let]) {
        let_stmt(p, m, semicolon);
        return;
    }
    // test block_items
    // fn a() { fn b() {} }
    let m = match items::opt_item(p, m) {
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
            // test no_semi_after_block
            // fn foo() {
            //     if true {}
            //     loop {}
            //     match () {}
            //     while true {}
            //     for _ in () {}
            //     {}
            //     {}
            //     macro_rules! test {
            //          () => {}
            //     }
            //     test!{}
            // }
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
    // test let_stmt
    // fn f() { let x: i32 = 92; }
    fn let_stmt(p: &mut Parser<'_>, m: Marker, with_semi: Semicolon) {
        p.bump(T![let]);
        patterns::pattern(p);
        if p.at(T![:]) {
            // test let_stmt_ascription
            // fn f() { let x: i32; }
            ty::ascription(p);
        }
        let mut expr_after_eq: Option<CompletedMarker> = None;
        if p.eat(T![=]) {
            // test let_stmt_init
            // fn f() { let x = 92; }
            expr_after_eq = exprs::expr(p);
        }
        if p.at(T![else]) {
            // test_err let_else_right_curly_brace
            // fn func() { let Some(_) = {Some(1)} else { panic!("h") };}
            if let Some(expr) = expr_after_eq {
                if BlockLike::is_blocklike(expr.kind()) {
                    p.error("right curly brace `}` before `else` in a `let...else` statement not allowed")
                }
            }
            // test let_else
            // fn f() { let Some(x) = opt else { return }; }
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
        // test nocontentexpr
        // fn foo(){
        //     ;;;some_expr();;;;{;;;};;;;Ok(())
        // }
        // test nocontentexpr_after_item
        // fn simple_function() {
        //     enum LocalEnum {
        //         One,
        //         Two,
        //     };
        //     fn f() {};
        //     struct S {};
        // }
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
        // If you update this, remember to update `expr_let()` too.
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
// Parses expression with binding power of at least bp.
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
                // test stmt_bin_expr_ambiguity
                // fn f() {
                //     let _ = {1} & 2;
                //     {1} &2;
                // }
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
        // test as_precedence
        // fn f() { let _ = &1 as *const i32; }
        if p.at(T![as]) {
            lhs = cast_expr(p, lhs);
            continue;
        }
        let m = lhs.precede(p);
        p.bump(op);
        // test binop_resets_statementness
        // fn f() { v = {1}&2; }
        r = Restrictions {
            prefer_stmt: false,
            ..r
        };
        if is_range {
            // test postfix_range
            // fn foo() {
            //     let x = 1..;
            //     match 1.. { _ => () };
            //     match a.b()..S { _ => () };
            // }
            let has_trailing_expression = p.at_ts(EXPR_FIRST) && !(r.forbid_structs && p.at(T!['{']));
            if !has_trailing_expression {
                // no RHS
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
        // test ref_expr
        // fn foo() {
        //     // reference operator
        //     let _ = &1;
        //     let _ = &mut &f();
        //     let _ = &raw;
        //     let _ = &raw.0;
        //     // raw reference operator
        //     let _ = &raw mut foo;
        //     let _ = &raw const foo;
        // }
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
        // test unary_expr
        // fn foo() {
        //     **&1;
        //     !!true;
        //     --1;
        // }
        T![*] | T![!] | T![-] => {
            m = p.start();
            p.bump_any();
            PREFIX_EXPR
        },
        _ => {
            // test full_range_expr
            // fn foo() { xs[..]; }
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
            // test expression_after_block
            // fn foo() {
            //    let mut p = F{x: 5};
            //    {p}.x = 10;
            // }
            let (lhs, blocklike) = atom::atom_expr(p, r)?;
            let (cm, block_like) = postfix_expr(p, lhs, blocklike, !(r.prefer_stmt && blocklike.is_block()));
            return Some((cm, block_like));
        },
    };
    // parse the interior of the unary expression
    expr_bp(p, None, r, 255);
    let cm = m.complete(p, kind);
    Some((cm, BlockLike::NotBlock))
}
fn postfix_expr(
    p: &mut Parser<'_>,
    mut lhs: CompletedMarker,
    // Calls are disallowed if the type is a block and we prefer statements because the call cannot be disambiguated from a tuple
    // E.g. `while true {break}();` is parsed as
    // `while true {break}; ();`
    mut block_like: BlockLike,
    mut allow_calls: bool,
) -> (CompletedMarker, BlockLike) {
    loop {
        lhs = match p.current() {
            // test stmt_postfix_expr_ambiguity
            // fn foo() {
            //     match () {
            //         _ => {}
            //         () => {}
            //         [] => {}
            //     }
            // }
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
    // test await_expr
    // fn foo() {
    //     x.await;
    //     x.0.await;
    //     x.0().await?.hello();
    //     x.0.0.await;
    //     x.0. await;
    // }
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
// test call_expr
// fn foo() {
//     let _ = f();
//     let _ = f()(1)(1, 2,);
//     let _ = f(<Foo>::func());
//     f(<Foo as Trait>::func());
// }
fn call_expr(p: &mut Parser<'_>, lhs: CompletedMarker) -> CompletedMarker {
    assert!(p.at(T!['(']));
    let m = lhs.precede(p);
    arg_list(p);
    m.complete(p, CALL_EXPR)
}
// test index_expr
// fn foo() {
//     x[1][2];
// }
fn index_expr(p: &mut Parser<'_>, lhs: CompletedMarker) -> CompletedMarker {
    assert!(p.at(T!['[']));
    let m = lhs.precede(p);
    p.bump(T!['[']);
    expr(p);
    p.expect(T![']']);
    m.complete(p, INDEX_EXPR)
}
// test method_call_expr
// fn foo() {
//     x.foo();
//     y.bar::<T>(1, 2,);
//     x.0.0.call();
//     x.0. call();
// }
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
// test field_expr
// fn foo() {
//     x.foo;
//     x.0.bar;
//     x.0.1;
//     x.0. bar;
//     x.0();
// }
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
// test try_expr
// fn foo() {
//     x?;
// }
fn try_expr(p: &mut Parser<'_>, lhs: CompletedMarker) -> CompletedMarker {
    assert!(p.at(T![?]));
    let m = lhs.precede(p);
    p.bump(T![?]);
    m.complete(p, TRY_EXPR)
}
// test cast_expr
// fn foo() {
//     82 as i32;
//     81 as i8 + 1;
//     79 as i16 - 1;
//     0x36 as u8 <= 0x37;
// }
fn cast_expr(p: &mut Parser<'_>, lhs: CompletedMarker) -> CompletedMarker {
    assert!(p.at(T![as]));
    let m = lhs.precede(p);
    p.bump(T![as]);
    ty::no_bounds(p);
    m.complete(p, CAST_EXPR)
}
// test_err arg_list_recovery
// fn main() {
//     foo(bar::);
//     foo(bar:);
//     foo(bar+);
// }
fn arg_list(p: &mut Parser<'_>) {
    assert!(p.at(T!['(']));
    let m = p.start();
    // test arg_with_attr
    // fn main() {
    //     foo(#[attr] 92)
    // }
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
// test path_expr
// fn foo() {
//     let _ = a;
//     let _ = a::b;
//     let _ = ::a::<b>;
//     let _ = format!();
// }
fn path_expr(p: &mut Parser<'_>, r: Restrictions) -> (CompletedMarker, BlockLike) {
    assert!(is_path_start(p));
    let m = p.start();
    expr_path(p);
    match p.current() {
        T!['{'] if !r.forbid_structs => {
            record_expr_field_list(p);
            (m.complete(p, RECORD_EXPR), BlockLike::NotBlock)
        },
        T![!] if !p.at(T![!=]) => {
            let block_like = items::macro_call_after_excl(p);
            (m.complete(p, MACRO_CALL).precede(p).complete(p, MACRO_EXPR), block_like)
        },
        _ => (m.complete(p, PATH_EXPR), BlockLike::NotBlock),
    }
}
// test record_lit
// fn foo() {
//     S {};
//     S { x };
//     S { x, y: 32, };
//     S { x, y: 32, ..Default::default() };
//     S { x: ::default() };
//     TupleStruct { 0: 1 };
// }
pub fn record_expr_field_list(p: &mut Parser<'_>) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump(T!['{']);
    while !p.at(EOF) && !p.at(T!['}']) {
        let m = p.start();
        // test record_literal_field_with_attr
        // fn main() {
        //     S { #[cfg(test)] field: 1 }
        // }
        attr::outers(p);
        match p.current() {
            IDENT | INT_NUMBER => {
                // test_err record_literal_missing_ellipsis_recovery
                // fn main() {
                //     S { S::default() }
                // }
                if p.nth_at(1, T![::]) {
                    m.abandon(p);
                    p.expect(T![..]);
                    expr(p);
                } else {
                    // test_err record_literal_before_ellipsis_recovery
                    // fn main() {
                    //     S { field ..S::default() }
                    // }
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
                // test destructuring_assignment_struct_rest_pattern
                // fn foo() {
                //     S { .. } = S {};
                // }
                // We permit `.. }` on the left-hand side of a destructuring assignment.
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
