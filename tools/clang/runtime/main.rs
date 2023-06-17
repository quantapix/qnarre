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

pub fn llvm_config(args: &str) -> Option<String> {
    #[cfg(test)]
    if let Some(f) = &*MOCK.lock().unwrap() {
        return f(args);
    }
    let p = env::var("LLVM_CONFIG_PATH").unwrap_or(LLVM_CONFIG.into());
    let p = format!("{} --link-static {}", p, args);
    let n = "llvm-config";
    let y = match Command::new("sh").arg("-c").arg(&p).output() {
        Ok(x) => x,
        Err(x) => {
            add_err(n, &p, args, format!("error: {}", x));
            return None;
        },
    };
    if y.status.success() {
        Some(String::from_utf8_lossy(&y.stdout).into_owned())
    } else {
        add_err(n, &p, args, format!("exit code: {}", y.status));
        None
    }
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
        let p = Path::new(x.lines().next().unwrap()).to_path_buf();
        ys.extend(search_dirs(&p.join("bin"), files));
        ys.extend(search_dirs(&p.join("lib"), files));
        ys.extend(search_dirs(&p.join("lib64"), files));
    }
    if let Ok(x) = env::var("LD_LIBRARY_PATH") {
        for p in env::split_paths(&x) {
            ys.extend(search_dirs(&p, files));
        }
    }
    let ds: Vec<&str> = DIRS.into();
    let ds = if test!() {
        ds.iter().map(|x| x.strip_prefix('/').unwrap_or(x)).collect::<Vec<_>>()
    } else {
        ds
    };
    let mut opts = MatchOptions::new();
    opts.case_sensitive = false;
    opts.require_literal_separator = true;
    for d in ds.iter() {
        if let Ok(ps) = glob::glob_with(d, opts) {
            for p in ps.filter_map(Result::ok).filter(|x| x.is_dir()) {
                ys.extend(search_dirs(&p, files));
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

macro_rules! test {
    () => {
        cfg!(test)
    };
}

#[cfg(test)]
pub static MOCK: std::sync::Mutex<Option<Box<dyn Fn(&str) -> Option<String> + Send + Sync + 'static>>> =
    std::sync::Mutex::new(None);
