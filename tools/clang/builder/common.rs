extern crate glob;

use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use glob::{MatchOptions, Pattern};

thread_local! {
    static CMD_ERRORS: RefCell<HashMap<String, Vec<String>>> = RefCell::default();
}

fn add_cmd_error(name: &str, path: &str, args: &str, msg: String) {
    CMD_ERRORS.with(|x| {
        x.borrow_mut().entry(name.into()).or_insert_with(Vec::new).push(format!(
            "couldn't execute `{} {}` (path={}) ({})",
            name, args, path, msg,
        ))
    });
}

#[derive(Default)]
pub struct CmdErrorPrinter {
    discard: bool,
}

impl CmdErrorPrinter {
    pub fn discard(mut self) {
        self.discard = true;
    }
}

impl Drop for CmdErrorPrinter {
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

const LLVM_CONFIG: &str = "/usr/bin/llvm-config-17";

pub fn llvm_config(args: &str) -> Option<String> {
    let p = env::var("LLVM_CONFIG_PATH").unwrap_or(LLVM_CONFIG.into());
    let p = format!("{} --link-static {}", p, args);
    let n = "llvm-config";
    let y = match Command::new("sh").arg("-c").arg(&p).output() {
        Ok(x) => x,
        Err(x) => {
            add_cmd_error(n, &p, args, format!("error: {}", x));
            return None;
        },
    };
    if y.status.success() {
        Some(String::from_utf8_lossy(&y.stdout).into_owned())
    } else {
        add_cmd_error(n, &p, args, format!("exit code: {}", y.status));
        None
    }
}

/*
#[cfg(test)]
pub static RUN_COMMAND_MOCK: std::sync::Mutex<
    Option<Box<dyn Fn(&str, &str, &[&str]) -> Option<String> + Send + Sync + 'static>>,
> = std::sync::Mutex::new(None);

fn run_command(name: &str, path: &str, args: &[&str]) -> Option<String> {
    #[cfg(test)]
    if let Some(command) = &*RUN_COMMAND_MOCK.lock().unwrap() {
        return command(name, path, args);
    }
}
*/

const DIRS_LINUX: &[&str] = &[
    "/usr/local/llvm*/lib*",
    "/usr/local/lib*/*/*",
    "/usr/local/lib*/*",
    "/usr/local/lib*",
    "/usr/lib*/*/*",
    "/usr/lib*/*",
    "/usr/lib*",
];

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

fn search_dirs(dir: &Path, files: &[String]) -> Vec<(PathBuf, String)> {
    let ys = search_dir(dir, files);
    ys
}

pub fn search_clang_dirs(files: &[String], variable: &str) -> Vec<(PathBuf, String)> {
    if let Ok(p) = env::var(variable).map(|x| Path::new(&x).to_path_buf()) {
        if let Some(parent) = p.parent() {
            let ds = search_dirs(parent, files);
            let file = p.file_name().unwrap().to_str().unwrap();
            if ds.iter().any(|(_, f)| f == file) {
                return vec![(parent.into(), file.into())];
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
    let ds: Vec<&str> = if target_os!("linux") { DIRS_LINUX.into() } else { vec![] };
    let ds = if test!() {
        ds.iter()
            .map(|d| d.strip_prefix('/').or_else(|| d.strip_prefix("C:\\")).unwrap_or(d))
            .collect::<Vec<_>>()
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
