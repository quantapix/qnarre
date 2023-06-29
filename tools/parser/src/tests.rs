mod gen;
mod pre;
mod top;

use crate::{Lexed, TopEntry};
use expect_test::expect_file;
use std::{
    cell::RefCell,
    fmt::Write,
    fs, panic,
    path::{Path, PathBuf},
    sync::Once,
};

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
