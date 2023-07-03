use expect_test::expect_file;
use rayon::prelude::*;
use std::{
    fs,
    path::{Path, PathBuf},
};
use syntax::{ast, fuzz, SourceFile, SyntaxErr};
use test_utils::{bench, bench_fixture, project_root};

#[test]
fn smoke_test() {
    let x = r##"
fn main() {
    println!("Hello, world!")
}
    "##;
    let y = SourceFile::parse(x);
    assert!(y.ok().is_ok());
}
#[test]
fn benchmark_test() {
    if std::env::var("RUN_SLOW_BENCHES").is_err() {
        return;
    }
    let x = bench_fixture::glorious_old_parser();
    let tree = {
        let _ = bench("parsing");
        let y = SourceFile::parse(&x);
        assert!(y.errors.is_empty());
        assert_eq!(y.tree().syntax.text_range().len(), 352474.into());
        y.tree()
    };
    let _ = bench("tree traversal");
    let y = tree
        .syntax()
        .descendants()
        .filter_map(ast::Fn::cast)
        .filter_map(|f| f.name())
        .count();
    assert_eq!(y, 268);
}

#[test]
fn validate_tests() {
    dir_tests(&test_data_dir(), &["parser/validate"], "rast", |x, path| {
        let y = SourceFile::parse(x);
        let errs = y.errors();
        assert_errs_present(errs, path);
        y.debug_dump()
    });
}

#[test]
fn parser_fuzz_tests() {
    for (_, x) in collect_files(&test_data_dir(), &["parser/fuzz-failures"]) {
        fuzz::check_parser(&x)
    }
}

#[test]
fn reparse_fuzz_tests() {
    for (_, x) in collect_files(&test_data_dir(), &["reparse/fuzz-failures"]) {
        let y = fuzz::CheckReparse::from_data(x.as_bytes()).unwrap();
        y.run();
    }
}

#[test]
fn self_hosting_parsing() {
    let crates_dir = project_root().join("crates");
    let mut files = ::sourcegen::list_rust_files(&crates_dir);
    files.retain(|path| !path.components().any(|component| component.as_os_str() == "test_data"));

    assert!(
        files.len() > 100,
        "self_hosting_parsing found too few files - is it running in the right directory?"
    );
    let errors = files
        .into_par_iter()
        .filter_map(|file| {
            let text = read_text(&file);
            match SourceFile::parse(&text).ok() {
                Ok(_) => None,
                Err(err) => Some((file, err)),
            }
        })
        .collect::<Vec<_>>();
    if !errors.is_empty() {
        let errors = errors
            .into_iter()
            .map(|(path, err)| format!("{}: {:?}\n", path.display(), err[0]))
            .collect::<String>();
        panic!("Parsing errors:\n{errors}\n");
    }
}

fn test_data_dir() -> PathBuf {
    project_root().join("crates/syntax/test_data")
}

fn assert_errs_present(xs: &[SyntaxErr], path: &Path) {
    assert!(
        !xs.is_empty(),
        "There should be errors in the file {:?}",
        path.display()
    );
}

fn dir_tests<F>(data: &Path, xs: &[&str], ext: &str, f: F)
where
    F: Fn(&str, &Path) -> String,
{
    for (x, input) in collect_files(data, xs) {
        let y = f(&input, &x);
        let x = x.with_extension(ext);
        expect_file![x].assert_eq(&y)
    }
}

fn collect_files(root: &Path, xs: &[&str]) -> Vec<(PathBuf, String)> {
    xs.iter()
        .flat_map(|x| {
            let y = root.to_owned().join(x);
            files_in_dir(&y).into_iter()
        })
        .map(|x| {
            let y = read_text(&x);
            (x, y)
        })
        .collect()
}

fn files_in_dir(x: &Path) -> Vec<PathBuf> {
    let mut y = Vec::new();
    for x in fs::read_dir(x).unwrap() {
        let x = x.unwrap();
        let x = x.path();
        if x.extension().unwrap_or_default() == "rs" {
            y.push(x);
        }
    }
    y.sort();
    y
}

fn read_text(x: &Path) -> String {
    fs::read_to_string(x)
        .unwrap_or_else(|_| panic!("File at {path:?} should be valid"))
        .replace("\r\n", "\n")
}
