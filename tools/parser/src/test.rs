use crate::{Lexed, TopEntry};
use expect_test::expect_file;
use std::{
    cell::RefCell,
    fmt::Write,
    fs, panic,
    path::{Path, PathBuf},
    sync::Once,
};

#[must_use]
pub struct PanicContext {
    _priv: (),
}
impl PanicContext {
    fn init() {
        let default_hook = panic::take_hook();
        let hook = move |p: &panic::PanicInfo<'_>| {
            with_ctx(|x| {
                if !x.is_empty() {
                    eprintln!("Panic context:");
                    for frame in x.iter() {
                        eprintln!("> {frame}\n");
                    }
                }
                default_hook(p);
            });
        };
        panic::set_hook(Box::new(hook));
    }
}
impl Drop for PanicContext {
    fn drop(&mut self) {
        with_ctx(|x| assert!(x.pop().is_some()));
    }
}
fn with_ctx(f: impl FnOnce(&mut Vec<String>)) {
    thread_local! {
        static CTX: RefCell<Vec<String>> = RefCell::new(Vec::new());
    }
    CTX.with(|x| f(&mut x.borrow_mut()));
}

pub fn enter(ctx: String) -> PanicContext {
    static ONCE: Once = Once::new();
    ONCE.call_once(PanicContext::init);
    with_ctx(|x| x.push(ctx));
    PanicContext { _priv: () }
}

#[test]
fn lex_ok() {
    for x in TestCase::list("lexer/ok") {
        let _guard = enter(format!("{:?}", x.rs));
        let y = lex(&x.text);
        expect_file![x.rast].assert_eq(&y)
    }
}
#[test]
fn lex_err() {
    for x in TestCase::list("lexer/err") {
        let _guard = enter(format!("{:?}", x.rs));
        let y = lex(&x.text);
        expect_file![x.rast].assert_eq(&y)
    }
}
fn lex(x: &str) -> String {
    let x = Lexed::new(x);
    let mut y = String::new();
    for i in 0..x.len() {
        let kind = x.kind(i);
        let text = x.text(i);
        let err = x.err(i);
        let err = err.map(|x| format!(" error: {x}")).unwrap_or_default();
        writeln!(y, "{kind:?} {text:?}{err}").unwrap();
    }
    y
}
#[test]
fn parse_ok() {
    for x in TestCase::list("parser/ok") {
        let _guard = enter(format!("{:?}", x.rs));
        let (y, errs) = parse(TopEntry::SourceFile, &x.text);
        assert!(!errs, "errors in an OK file {}:\n{y}", x.rs.display());
        expect_file![x.rast].assert_eq(&y);
    }
}
#[test]
fn parse_inline_ok() {
    for x in TestCase::list("parser/inline/ok") {
        let _guard = enter(format!("{:?}", x.rs));
        let (y, errs) = parse(TopEntry::SourceFile, &x.text);
        assert!(!errs, "errors in an OK file {}:\n{y}", x.rs.display());
        expect_file![x.rast].assert_eq(&y);
    }
}
#[test]
fn parse_err() {
    for x in TestCase::list("parser/err") {
        let _guard = enter(format!("{:?}", x.rs));
        let (y, errs) = parse(TopEntry::SourceFile, &x.text);
        assert!(errs, "no errors in an ERR file {}:\n{y}", x.rs.display());
        expect_file![x.rast].assert_eq(&y)
    }
}
#[test]
fn parse_inline_err() {
    for x in TestCase::list("parser/inline/err") {
        let _guard = enter(format!("{:?}", x.rs));
        let (y, errs) = parse(TopEntry::SourceFile, &x.text);
        assert!(errs, "no errors in an ERR file {}:\n{y}", x.rs.display());
        expect_file![x.rast].assert_eq(&y)
    }
}

fn parse(entry: TopEntry, text: &str) -> (String, bool) {
    let lexed = Lexed::new(text);
    let input = lexed.to_input();
    let out = entry.parse(&input);
    let mut y = String::new();
    let mut errs = Vec::new();
    let mut indent = String::new();
    let mut depth = 0;
    let mut len = 0;
    use crate::StrStep::*;
    lexed.intersperse_trivia(&out, &mut |step| match step {
        Token { kind, text } => {
            assert!(depth > 0);
            len += text.len();
            writeln!(y, "{indent}{kind:?} {text:?}").unwrap();
        },
        Enter { kind } => {
            assert!(depth > 0 || len == 0);
            depth += 1;
            writeln!(y, "{indent}{kind:?}").unwrap();
            indent.push_str("  ");
        },
        Exit => {
            assert!(depth > 0);
            depth -= 1;
            indent.pop();
            indent.pop();
        },
        crate::StrStep::Error { msg, pos } => {
            assert!(depth > 0);
            errs.push(format!("error {pos}: {msg}\n"))
        },
    });
    assert_eq!(
        len,
        text.len(),
        "didn't parse all text.\nParsed:\n{}\n\nAll:\n{}\n",
        &text[..len],
        text
    );
    for (token, msg) in lexed.errs() {
        let pos = lexed.text_start(token);
        errs.push(format!("error {pos}: {msg}\n"));
    }
    let has_errs = !errs.is_empty();
    for e in errs {
        y.push_str(&e);
    }
    (y, has_errs)
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct TestCase {
    rs: PathBuf,
    rast: PathBuf,
    text: String,
}
impl TestCase {
    fn list(path: &'static str) -> Vec<TestCase> {
        let crate_root_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let test_data_dir = crate_root_dir.join("test_data");
        let dir = test_data_dir.join(path);
        let mut y = Vec::new();
        let read_dir = fs::read_dir(&dir).unwrap_or_else(|err| panic!("can't `read_dir` {}: {err}", dir.display()));
        for file in read_dir {
            let file = file.unwrap();
            let path = file.path();
            if path.extension().unwrap_or_default() == "rs" {
                let rs = path;
                let rast = rs.with_extension("rast");
                let text = fs::read_to_string(&rs).unwrap();
                y.push(TestCase { rs, rast, text });
            }
        }
        y.sort();
        y
    }
}

mod pre {
    use crate::{
        Lexed,
        PreEntry::{self, *},
        Step,
    };
    #[test]
    fn vis() {
        check(Vis, "pub fn foo() {}", "pub(crate)");
        check(Vis, "fn foo() {}", "");
        check(Vis, "pub(fn foo() {}", "pub");
        check(Vis, "pub(crate fn foo() {}", "pub(crate");
        check(Vis, "crate fn foo() {}", "crate");
    }
    #[test]
    fn block() {
        check(Block, "{}, 92", "{}");
        check(Block, "{, 92)", "{, 92)");
        check(Block, "()", "");
    }
    #[test]
    fn stmt() {
        check(Stmt, "92; fn", "92");
        check(Stmt, "let _ = 92; 1", "let _ = 92");
        check(Stmt, "pub fn f() {} = 92", "pub fn f() {}");
        check(Stmt, "struct S;;", "struct S;");
        check(Stmt, "fn f() {};", "fn f() {}");
        check(Stmt, ";;;", ";");
        check(Stmt, "+", "+");
        check(Stmt, "@", "@");
        check(Stmt, "loop {} - 1", "loop {}");
    }
    #[test]
    fn pat() {
        check(Pat, "x y", "x");
        check(Pat, "fn f() {}", "fn");
        check(Pat, ".. ..", "..");
    }
    #[test]
    fn ty() {
        check(Ty, "fn() foo", "fn()");
        check(Ty, "Clone + Copy + fn", "Clone + Copy +");
        check(Ty, "struct f", "struct");
    }
    #[test]
    fn expr() {
        check(Expr, "92 92", "92");
        check(Expr, "+1", "+");
        check(Expr, "-1", "-1");
        check(Expr, "fn foo() {}", "fn");
        check(Expr, "#[attr] ()", "#[attr] ()");
        check(Expr, "foo.0", "foo.0");
        check(Expr, "foo.0.1", "foo.0.1");
        check(Expr, "foo.0. foo", "foo.0. foo");
    }
    #[test]
    fn path() {
        check(Path, "foo::bar baz", "foo::bar");
        check(Path, "foo::<> baz", "foo::<>");
        check(Path, "foo<> baz", "foo<>");
        check(Path, "Fn() -> i32?", "Fn() -> i32");
        check(Path, "<_>::foo", "<_>::foo");
    }
    #[test]
    fn item() {
        check(Item, "fn foo() {};", "fn foo() {};");
        check(Item, "#[attr] pub struct S {} 92", "#[attr] pub struct S {}");
        check(Item, "item!{}?", "item!{}");
        check(Item, "????", "?");
    }
    #[test]
    fn meta_item() {
        check(MetaItem, "attr, ", "attr");
        check(MetaItem, "attr(some token {stream});", "attr(some token {stream})");
        check(MetaItem, "path::attr = 2 * 2!", "path::attr = 2 * 2");
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
}

mod top {
    use crate::TopEntry::{self, *};
    use expect_test::expect;

    #[test]
    fn source_file() {
        check(
            SourceFile,
            "",
            expect![[r#"
        SOURCE_FILE
    "#]],
        );
        check(
            SourceFile,
            "struct S;",
            expect![[r#"
        SOURCE_FILE
          STRUCT
            STRUCT_KW "struct"
            WHITESPACE " "
            NAME
              IDENT "S"
            SEMICOLON ";"
    "#]],
        );
        check(
            SourceFile,
            "@error@",
            expect![[r#"
        SOURCE_FILE
          ERROR
            AT "@"
          MACRO_CALL
            PATH
              PATH_SEGMENT
                NAME_REF
                  IDENT "error"
          ERROR
            AT "@"
        error 0: expected an item
        error 6: expected BANG
        error 6: expected `{`, `[`, `(`
        error 6: expected SEMICOLON
        error 6: expected an item
    "#]],
        );
    }
    #[test]
    fn macro_stmt() {
        check(
            MacroStmts,
            "",
            expect![[r#"
            MACRO_STMTS
        "#]],
        );
        check(
            MacroStmts,
            "#!/usr/bin/rust",
            expect![[r##"
            MACRO_STMTS
              ERROR
                SHEBANG "#!/usr/bin/rust"
            error 0: expected expression, item or let statement
        "##]],
        );
        check(
            MacroStmts,
            "let x = 1 2 struct S;",
            expect![[r#"
            MACRO_STMTS
              LET_STMT
                LET_KW "let"
                WHITESPACE " "
                IDENT_PAT
                  NAME
                    IDENT "x"
                WHITESPACE " "
                EQ "="
                WHITESPACE " "
                LITERAL
                  INT_NUMBER "1"
              WHITESPACE " "
              EXPR_STMT
                LITERAL
                  INT_NUMBER "2"
              WHITESPACE " "
              STRUCT
                STRUCT_KW "struct"
                WHITESPACE " "
                NAME
                  IDENT "S"
                SEMICOLON ";"
        "#]],
        );
    }
    #[test]
    fn macro_items() {
        check(
            MacroItems,
            "",
            expect![[r#"
            MACRO_ITEMS
        "#]],
        );
        check(
            MacroItems,
            "#!/usr/bin/rust",
            expect![[r##"
            MACRO_ITEMS
              ERROR
                SHEBANG "#!/usr/bin/rust"
            error 0: expected an item
        "##]],
        );
        check(
            MacroItems,
            "struct S; foo!{}",
            expect![[r#"
            MACRO_ITEMS
              STRUCT
                STRUCT_KW "struct"
                WHITESPACE " "
                NAME
                  IDENT "S"
                SEMICOLON ";"
              WHITESPACE " "
              MACRO_CALL
                PATH
                  PATH_SEGMENT
                    NAME_REF
                      IDENT "foo"
                BANG "!"
                TOKEN_TREE
                  L_CURLY "{"
                  R_CURLY "}"
        "#]],
        );
    }
    #[test]
    fn macro_pattern() {
        check(
            Pattern,
            "",
            expect![[r#"
            ERROR
            error 0: expected pattern
        "#]],
        );
        check(
            Pattern,
            "Some(_)",
            expect![[r#"
            TUPLE_STRUCT_PAT
              PATH
                PATH_SEGMENT
                  NAME_REF
                    IDENT "Some"
              L_PAREN "("
              WILDCARD_PAT
                UNDERSCORE "_"
              R_PAREN ")"
        "#]],
        );
        check(
            Pattern,
            "None leftover tokens",
            expect![[r#"
            ERROR
              IDENT_PAT
                NAME
                  IDENT "None"
              WHITESPACE " "
              IDENT "leftover"
              WHITESPACE " "
              IDENT "tokens"
        "#]],
        );
        check(
            Pattern,
            "@err",
            expect![[r#"
            ERROR
              ERROR
                AT "@"
              IDENT "err"
            error 0: expected pattern
        "#]],
        );
    }
    #[test]
    fn ty() {
        check(
            Type,
            "",
            expect![[r#"
            ERROR
            error 0: expected type
        "#]],
        );
        check(
            Type,
            "Option<!>",
            expect![[r#"
            PATH_TYPE
              PATH
                PATH_SEGMENT
                  NAME_REF
                    IDENT "Option"
                  GENERIC_ARG_LIST
                    L_ANGLE "<"
                    TYPE_ARG
                      NEVER_TYPE
                        BANG "!"
                    R_ANGLE ">"
        "#]],
        );
        check(
            Type,
            "() () ()",
            expect![[r#"
            ERROR
              TUPLE_TYPE
                L_PAREN "("
                R_PAREN ")"
              WHITESPACE " "
              L_PAREN "("
              R_PAREN ")"
              WHITESPACE " "
              L_PAREN "("
              R_PAREN ")"
        "#]],
        );
        check(
            Type,
            "$$$",
            expect![[r#"
            ERROR
              ERROR
                DOLLAR "$"
              DOLLAR "$"
              DOLLAR "$"
            error 0: expected type
        "#]],
        );
    }
    #[test]
    fn expr() {
        check(
            Expr,
            "",
            expect![[r#"
            ERROR
            error 0: expected expression
        "#]],
        );
        check(
            Expr,
            "2 + 2 == 5",
            expect![[r#"
        BIN_EXPR
          BIN_EXPR
            LITERAL
              INT_NUMBER "2"
            WHITESPACE " "
            PLUS "+"
            WHITESPACE " "
            LITERAL
              INT_NUMBER "2"
          WHITESPACE " "
          EQ2 "=="
          WHITESPACE " "
          LITERAL
            INT_NUMBER "5"
    "#]],
        );
        check(
            Expr,
            "let _ = 0;",
            expect![[r#"
            ERROR
              LET_EXPR
                LET_KW "let"
                WHITESPACE " "
                WILDCARD_PAT
                  UNDERSCORE "_"
                WHITESPACE " "
                EQ "="
                WHITESPACE " "
                LITERAL
                  INT_NUMBER "0"
              SEMICOLON ";"
        "#]],
        );
    }
    #[track_caller]
    fn check(entry: TopEntry, input: &str, expect: expect_test::Expect) {
        let (parsed, _errors) = super::parse(entry, input);
        expect.assert_eq(&parsed)
    }
}
