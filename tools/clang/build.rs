extern crate glob;
use std::path::Path;

#[cfg(feature = "runtime")]
fn main() {
    use std::env;
    if cfg!(feature = "static") {
        panic!("`runtime` and `static` features can't be combined");
    }
    fn copy(src: &str, dst: &Path) {
        use std::fs::File;
        use std::io::{Read, Write};
        let mut y = String::new();
        File::open(src).unwrap().read_to_string(&mut y).unwrap();
        File::create(dst).unwrap().write_all(y.as_bytes()).unwrap();
    }
    let y = env::var("OUT_DIR").unwrap();
    copy("src/utils.rs", &Path::new(&y).join("runtime.rs"));
}

#[cfg(not(feature = "runtime"))]
#[path = "src/utils.rs"]
pub mod utils;

#[cfg(not(feature = "runtime"))]
fn main() {
    if cfg!(feature = "static") {
        utils::r#static::link();
    } else {
        utils::dynamic::link();
    }
    if let Some(x) = utils::llvm_config("--includedir") {
        let y = Path::new(x.trim_end());
        println!("cargo:include={}", y.display());
    }
}
