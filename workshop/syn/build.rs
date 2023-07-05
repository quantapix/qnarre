#![allow(clippy::let_underscore_untyped, clippy::manual_let_else)]
use std::env;
use std::process::Command;
use std::str;
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let compiler = match rustc_version() {
        Some(compiler) => compiler,
        None => return,
    };
    let _ = compiler.minor;
    if !compiler.nightly {
        println!("cargo:rustc-cfg=syn_disable_nightly_tests");
    }
}
struct Compiler {
    minor: u32,
    nightly: bool,
}
fn rustc_version() -> Option<Compiler> {
    let rustc = env::var_os("RUSTC")?;
    let output = Command::new(rustc).arg("--version").output().ok()?;
    let version = str::from_utf8(&output.stdout).ok()?;
    let mut pieces = version.split('.');
    if pieces.next() != Some("rustc 1") {
        return None;
    }
    let minor = pieces.next()?.parse().ok()?;
    let nightly = version.contains("nightly") || version.ends_with("-dev");
    Some(Compiler { minor, nightly })
}
