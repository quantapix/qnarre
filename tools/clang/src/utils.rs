use std::{
    cell::RefCell,
    collections::HashMap,
    env,
    path::{Path, PathBuf},
    process::Command,
};
extern crate glob;
use glob::{MatchOptions, Pattern};

pub fn sh_cmd(x: &str, args: &str) -> Option<String> {
    let x = format!("{} {}", x, args);
    let y = match Command::new("sh").arg("-c").arg(&x).output() {
        Ok(x) => x,
        Err(e) => {
            add_err(&x, format!("error: {}", e));
            return None;
        },
    };
    if y.status.success() {
        Some(String::from_utf8_lossy(&y.stdout).into_owned())
    } else {
        add_err(&x, format!("status: {}", y.status));
        None
    }
}

const LLVM_CONFIG: &str = "/usr/bin/llvm-config-17";

pub fn llvm_config(args: &str) -> Option<String> {
    #[cfg(test)]
    if let Some(x) = &*MOCK.lock().unwrap() {
        return x(args);
    }
    let y = env::var("LLVM_CONFIG_PATH").unwrap_or(LLVM_CONFIG.into());
    #[cfg(feature = "static")]
    let y = format!("{} --link-static", y);
    sh_cmd(&y, args)
}

#[cfg(not(feature = "runtime"))]
fn search_for(xs: &[String]) -> Vec<(PathBuf, String)> {
    fn search_path(p: &Path, xs: &[String]) -> Vec<(PathBuf, String)> {
        let p = Pattern::escape(p.to_str().unwrap());
        let p = Path::new(&p);
        let ys = xs.iter().map(|x| p.join(x).to_str().unwrap().to_owned());
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
    let mut ys = vec![];
    if let Some(x) = llvm_config("--prefix") {
        let p = Path::new(x.lines().next().unwrap()).to_path_buf();
        ys.extend(search_path(&p.join("bin"), xs));
        ys.extend(search_path(&p.join("lib"), xs));
        ys.extend(search_path(&p.join("lib64"), xs));
    }
    ys
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

thread_local! {
    static CMD_ERRORS: RefCell<HashMap<String, Vec<String>>> = RefCell::default();
}

fn add_err(cmd: &str, msg: String) {
    CMD_ERRORS.with(|x| {
        x.borrow_mut()
            .entry(cmd.into())
            .or_insert_with(Vec::new)
            .push(format!("couldn't execute `{}` ({})", cmd, msg,))
    });
}

#[cfg(all(not(feature = "runtime"), not(feature = "static")))]
pub mod dynamic {
    use std::{
        fs::File,
        io::{self, Error, ErrorKind, Read},
        path::{Path, PathBuf},
    };

    pub fn link() {
        let e = super::CmdError::default();
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
        for (p, n) in super::search_for(&xs) {
            match validate(&p) {
                Ok(()) => {
                    let v = version(&n);
                    ys.push((p, n, v))
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

    fn validate(x: &Path) -> Result<(), String> {
        fn parse_header(x: &Path) -> io::Result<u8> {
            print!("parse_header {}\n", x.display());
            let mut x = File::open(x)?;
            print!("parse_header opened\n");
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

    fn version(file: &str) -> Vec<u32> {
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

#[cfg(feature = "static")]
pub mod r#static {
    extern crate glob;
    use glob::Pattern;
    use std::path::{Path, PathBuf};
    pub fn link() {
        let cep = super::CmdError::default();
        let dir = find();
        println!("cargo:rustc-link-search=native={}", dir.display());
        for x in clang_libs(dir) {
            println!("cargo:rustc-link-lib=static={}", x);
        }
        let mode = super::llvm_config("--shared-mode").map(|x| x.trim().to_owned());
        let pre = if mode.map_or(false, |x| x == "static") {
            "static="
        } else {
            ""
        };
        println!(
            "cargo:rustc-link-search=native={}",
            super::llvm_config("--libdir").unwrap().trim_end()
        );
        for x in llvm_libs() {
            println!("cargo:rustc-link-lib={}{}", pre, x);
        }
        println!("cargo:rustc-flags=-l ffi -l ncursesw -l stdc++ -l z");
        cep.discard();
    }

    fn find() -> PathBuf {
        let ys = super::search_dirs(&["libclang.a".into()]);
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
