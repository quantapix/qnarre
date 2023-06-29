use crate::{Lexed, PreEntry, Step};

#[test]
fn vis() {
    check(PreEntry::Vis, "pub fn foo() {}", "pub(crate)");
    check(PreEntry::Vis, "fn foo() {}", "");
    check(PreEntry::Vis, "pub(fn foo() {}", "pub");
    check(PreEntry::Vis, "pub(crate fn foo() {}", "pub(crate");
    check(PreEntry::Vis, "crate fn foo() {}", "crate");
}

#[test]
fn block() {
    check(PreEntry::Block, "{}, 92", "{}");
    check(PreEntry::Block, "{, 92)", "{, 92)");
    check(PreEntry::Block, "()", "");
}

#[test]
fn stmt() {
    check(PreEntry::Stmt, "92; fn", "92");
    check(PreEntry::Stmt, "let _ = 92; 1", "let _ = 92");
    check(PreEntry::Stmt, "pub fn f() {} = 92", "pub fn f() {}");
    check(PreEntry::Stmt, "struct S;;", "struct S;");
    check(PreEntry::Stmt, "fn f() {};", "fn f() {}");
    check(PreEntry::Stmt, ";;;", ";");
    check(PreEntry::Stmt, "+", "+");
    check(PreEntry::Stmt, "@", "@");
    check(PreEntry::Stmt, "loop {} - 1", "loop {}");
}

#[test]
fn pat() {
    check(PreEntry::Pat, "x y", "x");
    check(PreEntry::Pat, "fn f() {}", "fn");
    check(PreEntry::Pat, ".. ..", "..");
}

#[test]
fn ty() {
    check(PreEntry::Ty, "fn() foo", "fn()");
    check(PreEntry::Ty, "Clone + Copy + fn", "Clone + Copy +");
    check(PreEntry::Ty, "struct f", "struct");
}

#[test]
fn expr() {
    check(PreEntry::Expr, "92 92", "92");
    check(PreEntry::Expr, "+1", "+");
    check(PreEntry::Expr, "-1", "-1");
    check(PreEntry::Expr, "fn foo() {}", "fn");
    check(PreEntry::Expr, "#[attr] ()", "#[attr] ()");
    check(PreEntry::Expr, "foo.0", "foo.0");
    check(PreEntry::Expr, "foo.0.1", "foo.0.1");
    check(PreEntry::Expr, "foo.0. foo", "foo.0. foo");
}

#[test]
fn path() {
    check(PreEntry::Path, "foo::bar baz", "foo::bar");
    check(PreEntry::Path, "foo::<> baz", "foo::<>");
    check(PreEntry::Path, "foo<> baz", "foo<>");
    check(PreEntry::Path, "Fn() -> i32?", "Fn() -> i32");
    check(PreEntry::Path, "<_>::foo", "<_>::foo");
}

#[test]
fn item() {
    check(PreEntry::Item, "fn foo() {};", "fn foo() {};");
    check(PreEntry::Item, "#[attr] pub struct S {} 92", "#[attr] pub struct S {}");
    check(PreEntry::Item, "item!{}?", "item!{}");
    check(PreEntry::Item, "????", "?");
}

#[test]
fn meta_item() {
    check(PreEntry::MetaItem, "attr, ", "attr");
    check(
        PreEntry::MetaItem,
        "attr(some token {stream});",
        "attr(some token {stream})",
    );
    check(PreEntry::MetaItem, "path::attr = 2 * 2!", "path::attr = 2 * 2");
}

#[track_caller]
fn check(entry: PreEntry, input: &str, prefix: &str) {
    let lexed = Lexed::new(input);
    let input = lexed.to_input();

    let mut n_tokens = 0;
    for step in entry.parse(&input).iter() {
        match step {
            Step::Token {
                n_input_toks: n_input_tokens,
                ..
            } => n_tokens += n_input_tokens as usize,
            Step::FloatSplit { .. } => n_tokens += 1,
            Step::Enter { .. } | Step::Exit | Step::Error { .. } => (),
        }
    }

    let mut i = 0;
    loop {
        if n_tokens == 0 {
            break;
        }
        if !lexed.kind(i).is_trivia() {
            n_tokens -= 1;
        }
        i += 1;
    }
    let buf = &lexed.as_str()[..lexed.text_start(i)];
    assert_eq!(buf, prefix);
}
