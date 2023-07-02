use crate::{Lexed, TopEntry};
use expect_test::expect_file;
use std::{
    cell::RefCell,
    fmt::Write,
    fs, panic,
    path::{Path, PathBuf},
    sync::Once,
};

mod srcgen {
    use crate::srcgen;
    use std::{
        collections::HashMap,
        fs, iter,
        path::{Path, PathBuf},
    };

    #[test]
    fn srcgen_parser_tests() {
        let grammar_dir = srcgen::project_root().join(Path::new("crates/parser/src/grammar"));
        let tests = tests_from_dir(&grammar_dir);

        install_tests(&tests.ok, "crates/parser/test_data/parser/inline/ok");
        install_tests(&tests.err, "crates/parser/test_data/parser/inline/err");

        fn install_tests(tests: &HashMap<String, Test>, into: &str) {
            let tests_dir = srcgen::project_root().join(into);
            if !tests_dir.is_dir() {
                fs::create_dir_all(&tests_dir).unwrap();
            }
            let existing = existing_tests(&tests_dir, true);
            for t in existing.keys().filter(|&t| !tests.contains_key(t)) {
                panic!("Test is deleted: {t}");
            }

            let mut new_idx = existing.len() + 1;
            for (name, test) in tests {
                let path = match existing.get(name) {
                    Some((path, _test)) => path.clone(),
                    None => {
                        let file_name = format!("{new_idx:04}_{name}.rs");
                        new_idx += 1;
                        tests_dir.join(file_name)
                    },
                };
                srcgen::ensure_file_contents(&path, &test.text);
            }
        }
    }

    #[derive(Debug)]
    struct Test {
        name: String,
        text: String,
        ok: bool,
    }

    #[derive(Default, Debug)]
    struct Tests {
        ok: HashMap<String, Test>,
        err: HashMap<String, Test>,
    }

    fn collect_tests(s: &str) -> Vec<Test> {
        let mut res = Vec::new();
        for comment_block in srcgen::CommentBlock::extract_untagged(s) {
            let first_line = &comment_block.texts[0];
            let (name, ok) = if let Some(name) = first_line.strip_prefix("test ") {
                (name.to_string(), true)
            } else if let Some(name) = first_line.strip_prefix("test_err ") {
                (name.to_string(), false)
            } else {
                continue;
            };
            let text: String = comment_block.texts[1..]
                .iter()
                .cloned()
                .chain(iter::once(String::new()))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(!text.trim().is_empty() && text.ends_with('\n'));
            res.push(Test { name, text, ok })
        }
        res
    }

    fn tests_from_dir(dir: &Path) -> Tests {
        let mut res = Tests::default();
        for entry in srcgen::list_rust_files(dir) {
            process_file(&mut res, entry.as_path());
        }
        let grammar_rs = dir.parent().unwrap().join("grammar.rs");
        process_file(&mut res, &grammar_rs);
        return res;

        fn process_file(res: &mut Tests, path: &Path) {
            let text = fs::read_to_string(path).unwrap();

            for test in collect_tests(&text) {
                if test.ok {
                    if let Some(old_test) = res.ok.insert(test.name.clone(), test) {
                        panic!("Duplicate test: {}", old_test.name);
                    }
                } else if let Some(old_test) = res.err.insert(test.name.clone(), test) {
                    panic!("Duplicate test: {}", old_test.name);
                }
            }
        }
    }

    fn existing_tests(dir: &Path, ok: bool) -> HashMap<String, (PathBuf, Test)> {
        let mut res = HashMap::default();
        for file in fs::read_dir(dir).unwrap() {
            let file = file.unwrap();
            let path = file.path();
            if path.extension().unwrap_or_default() != "rs" {
                continue;
            }
            let name = {
                let file_name = path.file_name().unwrap().to_str().unwrap();
                file_name[5..file_name.len() - 3].to_string()
            };
            let text = fs::read_to_string(&path).unwrap();
            let test = Test {
                name: name.clone(),
                text,
                ok,
            };
            if let Some(old) = res.insert(name, (path, test)) {
                println!("Duplicate test: {old:?}");
            }
        }
        res
    }
}

pub fn enter(context: String) -> PanicContext {
    static ONCE: Once = Once::new();
    ONCE.call_once(PanicContext::init);
    with_ctx(|ctx| ctx.push(context));
    PanicContext { _priv: () }
}

#[must_use]
pub struct PanicContext {
    _priv: (),
}
impl PanicContext {
    fn init() {
        let default_hook = panic::take_hook();
        let hook = move |panic_info: &panic::PanicInfo<'_>| {
            with_ctx(|ctx| {
                if !ctx.is_empty() {
                    eprintln!("Panic context:");
                    for frame in ctx.iter() {
                        eprintln!("> {frame}\n");
                    }
                }
                default_hook(panic_info);
            });
        };
        panic::set_hook(Box::new(hook));
    }
}
impl Drop for PanicContext {
    fn drop(&mut self) {
        with_ctx(|ctx| assert!(ctx.pop().is_some()));
    }
}
fn with_ctx(f: impl FnOnce(&mut Vec<String>)) {
    thread_local! {
        static CTX: RefCell<Vec<String>> = RefCell::new(Vec::new());
    }
    CTX.with(|ctx| f(&mut ctx.borrow_mut()));
}

#[test]
fn lex_ok() {
    for case in TestCase::list("lexer/ok") {
        let _guard = enter(format!("{:?}", case.rs));
        let actual = lex(&case.text);
        expect_file![case.rast].assert_eq(&actual)
    }
}
#[test]
fn lex_err() {
    for case in TestCase::list("lexer/err") {
        let _guard = enter(format!("{:?}", case.rs));
        let actual = lex(&case.text);
        expect_file![case.rast].assert_eq(&actual)
    }
}
fn lex(text: &str) -> String {
    let lexed = Lexed::new(text);
    let mut res = String::new();
    for i in 0..lexed.len() {
        let kind = lexed.kind(i);
        let text = lexed.text(i);
        let error = lexed.err(i);
        let error = error.map(|err| format!(" error: {err}")).unwrap_or_default();
        writeln!(res, "{kind:?} {text:?}{error}").unwrap();
    }
    res
}
#[test]
fn parse_ok() {
    for case in TestCase::list("parser/ok") {
        let _guard = enter(format!("{:?}", case.rs));
        let (actual, errors) = parse(TopEntry::SourceFile, &case.text);
        assert!(!errors, "errors in an OK file {}:\n{actual}", case.rs.display());
        expect_file![case.rast].assert_eq(&actual);
    }
}
#[test]
fn parse_inline_ok() {
    for case in TestCase::list("parser/inline/ok") {
        let _guard = enter(format!("{:?}", case.rs));
        let (actual, errors) = parse(TopEntry::SourceFile, &case.text);
        assert!(!errors, "errors in an OK file {}:\n{actual}", case.rs.display());
        expect_file![case.rast].assert_eq(&actual);
    }
}
#[test]
fn parse_err() {
    for case in TestCase::list("parser/err") {
        let _guard = enter(format!("{:?}", case.rs));
        let (actual, errors) = parse(TopEntry::SourceFile, &case.text);
        assert!(errors, "no errors in an ERR file {}:\n{actual}", case.rs.display());
        expect_file![case.rast].assert_eq(&actual)
    }
}
#[test]
fn parse_inline_err() {
    for case in TestCase::list("parser/inline/err") {
        let _guard = enter(format!("{:?}", case.rs));
        let (actual, errors) = parse(TopEntry::SourceFile, &case.text);
        assert!(errors, "no errors in an ERR file {}:\n{actual}", case.rs.display());
        expect_file![case.rast].assert_eq(&actual)
    }
}

fn parse(entry: TopEntry, text: &str) -> (String, bool) {
    let lexed = Lexed::new(text);
    let input = lexed.to_input();
    let output = entry.parse(&input);
    let mut buf = String::new();
    let mut errors = Vec::new();
    let mut indent = String::new();
    let mut depth = 0;
    let mut len = 0;
    lexed.intersperse_trivia(&output, &mut |step| match step {
        crate::StrStep::Token { kind, text } => {
            assert!(depth > 0);
            len += text.len();
            writeln!(buf, "{indent}{kind:?} {text:?}").unwrap();
        },
        crate::StrStep::Enter { kind } => {
            assert!(depth > 0 || len == 0);
            depth += 1;
            writeln!(buf, "{indent}{kind:?}").unwrap();
            indent.push_str("  ");
        },
        crate::StrStep::Exit => {
            assert!(depth > 0);
            depth -= 1;
            indent.pop();
            indent.pop();
        },
        crate::StrStep::Error { msg, pos } => {
            assert!(depth > 0);
            errors.push(format!("error {pos}: {msg}\n"))
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
        errors.push(format!("error {pos}: {msg}\n"));
    }
    let has_errors = !errors.is_empty();
    for e in errors {
        buf.push_str(&e);
    }
    (buf, has_errors)
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
        let mut res = Vec::new();
        let read_dir = fs::read_dir(&dir).unwrap_or_else(|err| panic!("can't `read_dir` {}: {err}", dir.display()));
        for file in read_dir {
            let file = file.unwrap();
            let path = file.path();
            if path.extension().unwrap_or_default() == "rs" {
                let rs = path;
                let rast = rs.with_extension("rast");
                let text = fs::read_to_string(&rs).unwrap();
                res.push(TestCase { rs, rast, text });
            }
        }
        res.sort();
        res
    }
}

mod pre {
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
}

mod top {
    use expect_test::expect;

    use crate::TopEntry;

    #[test]
    fn source_file() {
        check(
            TopEntry::SourceFile,
            "",
            expect![[r#"
        SOURCE_FILE
    "#]],
        );

        check(
            TopEntry::SourceFile,
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
            TopEntry::SourceFile,
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
            TopEntry::MacroStmts,
            "",
            expect![[r#"
            MACRO_STMTS
        "#]],
        );
        check(
            TopEntry::MacroStmts,
            "#!/usr/bin/rust",
            expect![[r##"
            MACRO_STMTS
              ERROR
                SHEBANG "#!/usr/bin/rust"
            error 0: expected expression, item or let statement
        "##]],
        );
        check(
            TopEntry::MacroStmts,
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
            TopEntry::MacroItems,
            "",
            expect![[r#"
            MACRO_ITEMS
        "#]],
        );
        check(
            TopEntry::MacroItems,
            "#!/usr/bin/rust",
            expect![[r##"
            MACRO_ITEMS
              ERROR
                SHEBANG "#!/usr/bin/rust"
            error 0: expected an item
        "##]],
        );
        check(
            TopEntry::MacroItems,
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
            TopEntry::Pattern,
            "",
            expect![[r#"
            ERROR
            error 0: expected pattern
        "#]],
        );
        check(
            TopEntry::Pattern,
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
            TopEntry::Pattern,
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
            TopEntry::Pattern,
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
            TopEntry::Type,
            "",
            expect![[r#"
            ERROR
            error 0: expected type
        "#]],
        );

        check(
            TopEntry::Type,
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
            TopEntry::Type,
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
            TopEntry::Type,
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
            TopEntry::Expr,
            "",
            expect![[r#"
            ERROR
            error 0: expected expression
        "#]],
        );
        check(
            TopEntry::Expr,
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
            TopEntry::Expr,
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
