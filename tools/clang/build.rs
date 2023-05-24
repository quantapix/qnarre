#![allow(unused_attributes)]

extern crate glob;

use std::path::Path;

#[macro_use]
#[path = "build/macros.rs"]
pub mod macros;

#[path = "build/common.rs"]
pub mod common;
#[path = "build/dynamic.rs"]
pub mod dynamic;
#[path = "build/static.rs"]
pub mod r#static;

#[cfg(feature = "runtime")]
fn copy(src: &str, dst: &Path) {
    use std::fs::File;
    use std::io::{Read, Write};
    let mut s = String::new();
    File::open(src).unwrap().read_to_string(&mut s).unwrap();
    File::create(dst).unwrap().write_all(s.as_bytes()).unwrap();
}

#[cfg(feature = "runtime")]
fn main() {
    use std::env;
    if cfg!(feature = "static") {
        panic!("`runtime` and `static` features can't be combined");
    }
    let p = env::var("OUT_DIR").unwrap();
    copy("build/macros.rs", &Path::new(&p).join("macros.rs"));
    copy("build/common.rs", &Path::new(&p).join("common.rs"));
    copy("build/dynamic.rs", &Path::new(&p).join("dynamic.rs"));
}

#[cfg(not(feature = "runtime"))]
fn main() {
    if cfg!(feature = "static") {
        r#static::link();
    } else {
        dynamic::link();
    }
    if let Some(x) = common::run_llvm_config(&["--includedir"]) {
        let p = Path::new(x.trim_end());
        println!("cargo:include={}", p.display());
    }
}
