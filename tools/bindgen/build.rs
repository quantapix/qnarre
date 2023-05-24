use std::{
    env,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

fn main() {
    let p = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut y = File::create(Path::new(&p).join("host-target.txt")).unwrap();
    y.write_all(env::var("TARGET").unwrap().as_bytes()).unwrap();

    println!("cargo:rerun-if-env-changed=LLVM_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    println!("cargo:rerun-if-env-changed=LIBCLANG_STATIC_PATH");
    println!("cargo:rerun-if-env-changed=BINDGEN_EXTRA_CLANG_ARGS");
    println!(
        "cargo:rerun-if-env-changed=BINDGEN_EXTRA_CLANG_ARGS_{}",
        std::env::var("TARGET").unwrap()
    );
    println!(
        "cargo:rerun-if-env-changed=BINDGEN_EXTRA_CLANG_ARGS_{}",
        std::env::var("TARGET").unwrap().replace('-', "_")
    );
}
