use std::{
    cell::RefCell,
    collections::HashMap,
    env,
    path::{Path, PathBuf},
    process::Command,
};
extern crate glob;
use glob::{MatchOptions, Pattern};

const LLVM_CONFIG: &str = "/usr/bin/llvm-config-17";

pub fn llvm_config(xs: &str) -> Option<String> {
    #[cfg(test)]
    if let Some(x) = &*MOCK.lock().unwrap() {
        return x(xs);
    }
    let x = env::var("LLVM_CONFIG_PATH").unwrap_or(LLVM_CONFIG.into());
    let x = format!("{} --link-static {}", x, xs);
    let n = "llvm-config";
    let y = match Command::new("sh").arg("-c").arg(&x).output() {
        Ok(x) => x,
        Err(e) => {
            add_err(n, &x, xs, format!("error: {}", e));
            return None;
        },
    };
    if y.status.success() {
        Some(String::from_utf8_lossy(&y.stdout).into_owned())
    } else {
        add_err(n, &x, xs, format!("exit code: {}", y.status));
        None
    }
}

fn sh(exec: &str, args: &[&str]) -> Result<(String, String), String> {
    Command::new(exec)
        .args(args)
        .output()
        .map(|x| {
            let o = String::from_utf8_lossy(&x.stdout).into_owned();
            let e = String::from_utf8_lossy(&x.stderr).into_owned();
            (o, e)
        })
        .map_err(|x| format!("could not run executable `{}`: {}", exec, x))
}

pub fn search_clang_dirs(files: &[String], x: &str) -> Vec<(PathBuf, String)> {
    if let Ok(p) = env::var(x).map(|x| Path::new(&x).to_path_buf()) {
        if let Some(y) = p.parent() {
            let ys = search_dirs(y, files);
            let file = p.file_name().unwrap().to_str().unwrap();
            if ys.iter().any(|(_, x)| x == file) {
                return vec![(y.into(), file.into())];
            }
        }
        return search_dirs(&p, files);
    }
    let mut ys = vec![];
    if let Some(x) = llvm_config("--prefix") {
        let y = Path::new(x.lines().next().unwrap()).to_path_buf();
        ys.extend(search_dirs(&y.join("bin"), files));
        ys.extend(search_dirs(&y.join("lib"), files));
        ys.extend(search_dirs(&y.join("lib64"), files));
    }
    if let Ok(x) = env::var("LD_LIBRARY_PATH") {
        for y in env::split_paths(&x) {
            ys.extend(search_dirs(&y, files));
        }
    }
    let ds: Vec<&str> = DIRS.into();
    let mut opts = MatchOptions::new();
    opts.case_sensitive = false;
    opts.require_literal_separator = true;
    for d in ds.iter() {
        if let Ok(xs) = glob::glob_with(d, opts) {
            for y in xs.filter_map(Result::ok).filter(|x| x.is_dir()) {
                ys.extend(search_dirs(&y, files));
            }
        }
    }
    ys
}

fn search_dirs(dir: &Path, files: &[String]) -> Vec<(PathBuf, String)> {
    search_dir(dir, files)
}

fn search_dir(dir: &Path, files: &[String]) -> Vec<(PathBuf, String)> {
    let p = Pattern::escape(dir.to_str().unwrap());
    let p = Path::new(&p);
    let ys = files.iter().map(|x| p.join(x).to_str().unwrap().to_owned());
    let mut opts = MatchOptions::new();
    opts.require_literal_separator = true;
    ys.map(|x| glob::glob_with(&x, opts))
        .filter_map(Result::ok)
        .flatten()
        .filter_map(|x| {
            let p = x.ok()?;
            let n = p.file_name()?.to_str().unwrap();
            if n.contains("-cpp.") {
                return None;
            }
            Some((p.to_owned(), n.into()))
        })
        .collect::<Vec<_>>()
}

const DIRS: &[&str] = &[
    "/usr/local/llvm*/lib*",
    "/usr/local/lib*/*/*",
    "/usr/local/lib*/*",
    "/usr/local/lib*",
    "/usr/lib*/*/*",
    "/usr/lib*/*",
    "/usr/lib*",
];

thread_local! {
    static CMD_ERRORS: RefCell<HashMap<String, Vec<String>>> = RefCell::default();
}

fn add_err(name: &str, path: &str, args: &str, msg: String) {
    CMD_ERRORS.with(|x| {
        x.borrow_mut().entry(name.into()).or_insert_with(Vec::new).push(format!(
            "couldn't execute `{} {}` (path={}) ({})",
            name, args, path, msg,
        ))
    });
}

#[derive(Default)]
pub struct CmdError {
    discard: bool,
}
impl CmdError {
    #[cfg(not(feature = "runtime"))]
    pub fn discard(mut self) {
        self.discard = true;
    }
}
impl Drop for CmdError {
    fn drop(&mut self) {
        if self.discard {
            return;
        }
        let es = CMD_ERRORS.with(|x| x.borrow().clone());
        if let Some(ys) = es.get("llvm-config") {
            println!(
                "cargo:warning=could not execute `llvm-config` one or more \
                times, if the LLVM_CONFIG_PATH environment variable is set to \
                a full path to valid `llvm-config` executable it will be used \
                to try to find an instance of `libclang` on your system: {}",
                ys.iter().map(|x| format!("\"{}\"", x)).collect::<Vec<_>>().join("\n  "),
            )
        }
    }
}

pub mod dynamic {
    use super::*;
    use std::{
        fs::File,
        io::{self, Error, ErrorKind, Read},
        path::{Path, PathBuf},
    };

    #[cfg(not(feature = "runtime"))]
    pub fn link() {
        let e = CmdError::default();
        let (dir, x) = find(false).unwrap();
        println!("cargo:rustc-link-search={}", dir.display());
        let x = x.trim_start_matches("lib");
        let y = match x.find(".dylib").or_else(|| x.find(".so")) {
            Some(i) => &x[0..i],
            None => x,
        };
        println!("cargo:rustc-link-lib=dylib={}", y);
        e.discard();
    }

    pub fn find(runtime: bool) -> Result<(PathBuf, String), String> {
        search(runtime)?
            .iter()
            .rev()
            .max_by_key(|x| &x.2)
            .cloned()
            .map(|(dir, x, _)| (dir, x))
            .ok_or_else(|| "unreachable".into())
    }

    fn search(runtime: bool) -> Result<Vec<(PathBuf, String, Vec<u32>)>, String> {
        let mut xs = vec!["libclang-*.so".into()];
        if runtime {
            xs.push("libclang-*.so.*".into());
            xs.push("libclang.so.*".into());
        }
        let mut ys = vec![];
        let mut invalid = vec![];
        for (dir, x) in search_clang_dirs(&xs, "LIBCLANG_PATH") {
            let p = dir.join(&x);
            match validate_lib(&p) {
                Ok(()) => {
                    let v = parse_version(&x);
                    ys.push((dir, x, v))
                },
                Err(x) => invalid.push(format!("({}: {})", p.display(), x)),
            }
        }
        if !ys.is_empty() {
            return Ok(ys);
        }
        let y = format!(
            "couldn't find any shared libraries for: [{}] (invalid: [{}])",
            xs.iter().map(|x| format!("'{}'", x)).collect::<Vec<_>>().join(", "),
            invalid.join(", "),
        );
        Err(y)
    }

    fn validate_lib(x: &Path) -> Result<(), String> {
        fn parse_header(x: &Path) -> io::Result<u8> {
            let mut x = File::open(x)?;
            let mut ys = [0; 5];
            x.read_exact(&mut ys)?;
            if ys[..4] == [127, 69, 76, 70] {
                Ok(ys[4])
            } else {
                Err(Error::new(ErrorKind::InvalidData, "invalid ELF header"))
            }
        }
        let y = parse_header(x).map_err(|x| x.to_string())?;
        if y != 2 {
            return Err("invalid ELF class (32-bit)".into());
        }
        Ok(())
    }

    fn parse_version(file: &str) -> Vec<u32> {
        let y = if let Some(x) = file.strip_prefix("libclang.so.") {
            x
        } else if file.starts_with("libclang-") {
            &file[9..file.len() - 3]
        } else {
            return vec![];
        };
        y.split('.').map(|x| x.parse().unwrap_or(0)).collect()
    }
}

mod r#static {
    extern crate glob;
    use glob::Pattern;
    use std::path::{Path, PathBuf};
    #[cfg(not(feature = "runtime"))]
    pub fn link() {
        let cep = super::CmdError::default();
        let dir = find();
        println!("cargo:rustc-link-search=native={}", dir.display());
        for x in clang_libs(dir) {
            println!("cargo:rustc-link-lib=static={}", x);
        }
        let mode = main::llvm_config("--shared-mode").map(|x| x.trim().to_owned());
        let pre = if mode.map_or(false, |x| x == "static") {
            "static="
        } else {
            ""
        };
        println!(
            "cargo:rustc-link-search=native={}",
            main::llvm_config("--libdir").unwrap().trim_end()
        );
        for x in llvm_libs() {
            println!("cargo:rustc-link-lib={}{}", pre, x);
        }
        println!("cargo:rustc-flags=-l ffi -l ncursesw -l stdc++ -l z");
        cep.discard();
    }

    fn find() -> PathBuf {
        let x = "libclang.a";
        let ys = super::search_clang_dirs(&[x.into()], "LIBCLANG_STATIC_PATH");
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
        super::llvm_config("--libs")
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
}

#[cfg(test)]
pub static MOCK: std::sync::Mutex<Option<Box<dyn Fn(&str) -> Option<String> + Send + Sync + 'static>>> =
    std::sync::Mutex::new(None);
