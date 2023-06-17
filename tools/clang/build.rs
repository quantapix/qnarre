extern crate glob;
use std::path::Path;

#[path = "runtime/dynamic.rs"]
pub mod dynamic;
#[path = "runtime/main.rs"]
pub mod main;
#[path = "runtime/static.rs"]
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
    copy("runtime/main.rs", &Path::new(&p).join("main.rs"));
    copy("runtime/dynamic.rs", &Path::new(&p).join("dynamic.rs"));
}

#[cfg(not(feature = "runtime"))]
fn main() {
    if cfg!(feature = "static") {
        r#static::link();
    } else {
        dynamic::link();
    }
    if let Some(x) = main::llvm_config("--includedir") {
        let p = Path::new(x.trim_end());
        println!("cargo:include={}", p.display());
    }
}
