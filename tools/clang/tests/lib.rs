use clang::*;
use libc::c_char;
use std::ptr;

fn parse() {
    unsafe {
        let x = clang_createIndex(0, 0);
        assert!(!x.is_null());
        let y = clang_parseTranslationUnit(
            x,
            "tests/header.h\0".as_ptr() as *const c_char,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            0,
            0,
        );
        assert!(!y.is_null());
    }
}

#[cfg(feature = "runtime")]
#[test]
fn test() {
    load().unwrap();
    let y = get_lib().unwrap();
    println!("{:?} ({:?})", y.version(), y.path());
    parse();
    unload().unwrap();
}

#[cfg(not(feature = "runtime"))]
#[test]
fn test() {
    parse();
}

#[test]
fn test_support() {
    let y = clang::Clang::find(None, &[]).unwrap();
    println!("{:?}", y);
}

#[test]
fn test_support_target() {
    let xs = &["-target".into(), "x86_64-unknown-linux-gnu".into()];
    let y = clang::Clang::find(None, xs).unwrap();
    println!("{:?}", y);
}
