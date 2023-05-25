use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, io};

use glob::{self, Pattern};

use libc::c_int;

use super::CXVersion;

#[derive(Clone, Debug)]
pub struct Clang {
    pub path: PathBuf,
    pub version: Option<CXVersion>,
    pub c_search_paths: Option<Vec<PathBuf>>,
    pub cpp_search_paths: Option<Vec<PathBuf>>,
}

impl Clang {
    fn new(path: impl AsRef<Path>, args: &[String]) -> Self {
        Self {
            path: path.as_ref().into(),
            version: parse_version(path.as_ref()),
            c_search_paths: parse_search_paths(path.as_ref(), "c", args),
            cpp_search_paths: parse_search_paths(path.as_ref(), "c++", args),
        }
    }

    pub fn find(path: Option<&Path>, args: &[String]) -> Option<Clang> {
        if let Ok(path) = env::var("CLANG_PATH") {
            let path = Path::new(&path);
            if path.is_file() && is_executable(path).unwrap_or(false) {
                return Some(Clang::new(path, args));
            }
        }
        let mut target = None;
        for i in 0..args.len() {
            if args[i] == "-target" && i + 1 < args.len() {
                target = Some(&args[i + 1]);
            }
        }
        let mut ys = vec![];
        if let Some(path) = path {
            ys.push(path.into());
        }
        if let Ok(path) = llvm_config(&["--bindir"]) {
            if let Some(y) = path.lines().next() {
                ys.push(y.into());
            }
        }
        if let Ok(path) = env::var("PATH") {
            ys.extend(env::split_paths(&path));
        }
        if let Some(target) = target {
            let default = format!("{}-clang{}", target, env::consts::EXE_SUFFIX);
            let versioned = format!("{}-clang-[0-9]*{}", target, env::consts::EXE_SUFFIX);
            let patterns = &[&default[..], &versioned[..]];
            for y in &ys {
                if let Some(path) = find(y, patterns) {
                    return Some(Clang::new(path, args));
                }
            }
        }
        let default = format!("clang{}", env::consts::EXE_SUFFIX);
        let versioned = format!("clang-[0-9]*{}", env::consts::EXE_SUFFIX);
        let patterns = &[&default[..], &versioned[..]];
        for y in ys {
            if let Some(path) = find(&y, patterns) {
                return Some(Clang::new(path, args));
            }
        }

        None
    }
}

fn parse_version(path: &Path) -> Option<CXVersion> {
    let y = clang(path, &["--version"]).0;
    let start = y.find("version ")? + 8;
    let mut ys = y[start..].split_whitespace().next()?.split('.');
    let major = ys.next().and_then(parse_v_number)?;
    let minor = ys.next().and_then(parse_v_number)?;
    let subminor = ys.next().and_then(parse_v_number).unwrap_or(0);
    Some(CXVersion {
        Major: major,
        Minor: minor,
        Subminor: subminor,
    })
}

fn parse_v_number(x: &str) -> Option<c_int> {
    x.chars()
        .take_while(|x| x.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

fn parse_search_paths(path: &Path, lang: &str, args: &[String]) -> Option<Vec<PathBuf>> {
    let mut xs = vec!["-E", "-x", lang, "-", "-v"];
    xs.extend(args.iter().map(|x| &**x));
    let y = clang(path, &xs).1;
    let start = y.find("#include <...> search starts here:")? + 34;
    let end = y.find("End of search list.")?;
    let ys = y[start..end].replace("(framework directory)", "");
    Some(
        ys.lines()
            .filter(|x| !x.is_empty())
            .map(|x| Path::new(x.trim()).into())
            .collect(),
    )
}

fn clang(path: &Path, args: &[&str]) -> (String, String) {
    run(&path.to_string_lossy(), args).unwrap()
}

const LLVM_CONFIG: &str = "/usr/bin/llvm-config-17";

fn llvm_config(args: &[&str]) -> Result<String, String> {
    let p = env::var("LLVM_CONFIG_PATH").unwrap_or(LLVM_CONFIG.into());
    run(&p, args).map(|(x, _)| x)
}

fn run(exe: &str, args: &[&str]) -> Result<(String, String), String> {
    Command::new(exe)
        .args(args)
        .output()
        .map(|x| {
            let y = String::from_utf8_lossy(&x.stdout).into_owned();
            let err = String::from_utf8_lossy(&x.stderr).into_owned();
            (y, err)
        })
        .map_err(|x| format!("could not run executable `{}`: {}", exe, x))
}

fn find(dir: &Path, patterns: &[&str]) -> Option<PathBuf> {
    let dir = if let Some(x) = dir.to_str() {
        Path::new(&Pattern::escape(x)).to_owned()
    } else {
        return None;
    };
    for p in patterns {
        let p = dir.join(p).to_string_lossy().into_owned();
        if let Some(y) = glob::glob(&p).ok()?.find_map(|x| x.ok()) {
            if y.is_file() && is_executable(&y).unwrap_or(false) {
                return Some(y);
            }
        }
    }

    None
}

#[cfg(unix)]
fn is_executable(path: &Path) -> io::Result<bool> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let y = CString::new(path.as_os_str().as_bytes())?;
    unsafe { Ok(libc::access(y.as_ptr(), libc::X_OK) == 0) }
}
