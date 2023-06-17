extern crate glob;
use glob::Pattern;
use std::path::{Path, PathBuf};

use super::common;

#[cfg(not(feature = "runtime"))]
pub fn link() {
    let cep = common::CmdError::default();
    let dir = find();
    println!("cargo:rustc-link-search=native={}", dir.display());
    for x in clang_libs(dir) {
        println!("cargo:rustc-link-lib=static={}", x);
    }
    let mode = common::llvm_config("--shared-mode").map(|x| x.trim().to_owned());
    let pre = if mode.map_or(false, |x| x == "static") {
        "static="
    } else {
        ""
    };
    println!(
        "cargo:rustc-link-search=native={}",
        common::llvm_config("--libdir").unwrap().trim_end()
    );
    for x in llvm_libs() {
        println!("cargo:rustc-link-lib={}{}", pre, x);
    }
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-flags=-l ffi -l ncursesw -l stdc++ -l z");
    }
    cep.discard();
}

fn find() -> PathBuf {
    let x = "libclang.a";
    let ys = common::search_clang_dirs(&[x.into()], "LIBCLANG_STATIC_PATH");
    if let Some((y, _)) = ys.into_iter().next() {
        y
    } else {
        panic!("could not find any static libraries");
    }
}

fn clang_libs<P: AsRef<Path>>(dir: P) -> Vec<String> {
    let p = Pattern::escape(dir.as_ref().to_str().unwrap());
    let p = Path::new(&p);
    let pat = p.join("libclang*.a").to_str().unwrap().to_owned();
    if let Ok(ys) = glob::glob(&pat) {
        ys.filter_map(|x| x.ok().and_then(|l| lib_name(&l))).collect()
    } else {
        LIBS.iter().map(|l| (*l).to_string()).collect()
    }
}

fn lib_name(path: &Path) -> Option<String> {
    path.file_stem().map(|p| {
        let s = p.to_string_lossy();
        if let Some(x) = s.strip_prefix("lib") {
            x.to_owned()
        } else {
            s.to_string()
        }
    })
}

const LIBS: &[&str] = &[
    "clang",
    "clangAST",
    "clangAnalysis",
    "clangBasic",
    "clangDriver",
    "clangEdit",
    "clangFrontend",
    "clangIndex",
    "clangLex",
    "clangParse",
    "clangRewrite",
    "clangSema",
    "clangSerialization",
];

fn llvm_libs() -> Vec<String> {
    common::llvm_config("--libs")
        .unwrap()
        .split_whitespace()
        .filter_map(|x| {
            if let Some(p) = x.strip_prefix("-l") {
                Some(p.into())
            } else {
                lib_name(Path::new(x))
            }
        })
        .collect()
}
