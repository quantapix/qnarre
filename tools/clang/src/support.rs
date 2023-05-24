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
            let p = Path::new(&path);
            if p.is_file() && is_executable(p).unwrap_or(false) {
                return Some(Clang::new(p, args));
            }
        }
        let mut target = None;
        for i in 0..args.len() {
            if args[i] == "-target" && i + 1 < args.len() {
                target = Some(&args[i + 1]);
            }
        }
        let mut paths = vec![];
        if let Some(path) = path {
            paths.push(path.into());
        }
        if let Ok(path) = run_llvm_config(&["--bindir"]) {
            if let Some(line) = path.lines().next() {
                paths.push(line.into());
            }
        }
        if let Ok(path) = env::var("PATH") {
            paths.extend(env::split_paths(&path));
        }
        if let Some(target) = target {
            let default = format!("{}-clang{}", target, env::consts::EXE_SUFFIX);
            let versioned = format!("{}-clang-[0-9]*{}", target, env::consts::EXE_SUFFIX);
            let patterns = &[&default[..], &versioned[..]];
            for path in &paths {
                if let Some(path) = find(path, patterns) {
                    return Some(Clang::new(path, args));
                }
            }
        }
        let default = format!("clang{}", env::consts::EXE_SUFFIX);
        let versioned = format!("clang-[0-9]*{}", env::consts::EXE_SUFFIX);
        let patterns = &[&default[..], &versioned[..]];
        for path in paths {
            if let Some(path) = find(&path, patterns) {
                return Some(Clang::new(path, args));
            }
        }

        None
    }
}

fn find(directory: &Path, patterns: &[&str]) -> Option<PathBuf> {
    let directory = if let Some(directory) = directory.to_str() {
        Path::new(&Pattern::escape(directory)).to_owned()
    } else {
        return None;
    };
    for pattern in patterns {
        let pattern = directory.join(pattern).to_string_lossy().into_owned();
        if let Some(path) = glob::glob(&pattern).ok()?.filter_map(|p| p.ok()).next() {
            if path.is_file() && is_executable(&path).unwrap_or(false) {
                return Some(path);
            }
        }
    }

    None
}

#[cfg(unix)]
fn is_executable(path: &Path) -> io::Result<bool> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(path.as_os_str().as_bytes())?;
    unsafe { Ok(libc::access(path.as_ptr(), libc::X_OK) == 0) }
}

fn run(executable: &str, arguments: &[&str]) -> Result<(String, String), String> {
    Command::new(executable)
        .args(arguments)
        .output()
        .map(|o| {
            let stdout = String::from_utf8_lossy(&o.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&o.stderr).into_owned();
            (stdout, stderr)
        })
        .map_err(|e| format!("could not run executable `{}`: {}", executable, e))
}

fn run_clang(path: &Path, arguments: &[&str]) -> (String, String) {
    run(&path.to_string_lossy(), arguments).unwrap()
}

fn run_llvm_config(arguments: &[&str]) -> Result<String, String> {
    let config = env::var("LLVM_CONFIG_PATH").unwrap_or_else(|_| "llvm-config".to_string());
    run(&config, arguments).map(|(o, _)| o)
}

fn parse_version_number(number: &str) -> Option<c_int> {
    number
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

fn parse_version(path: &Path) -> Option<CXVersion> {
    let output = run_clang(path, &["--version"]).0;
    let start = output.find("version ")? + 8;
    let mut numbers = output[start..].split_whitespace().next()?.split('.');
    let major = numbers.next().and_then(parse_version_number)?;
    let minor = numbers.next().and_then(parse_version_number)?;
    let subminor = numbers.next().and_then(parse_version_number).unwrap_or(0);
    Some(CXVersion {
        Major: major,
        Minor: minor,
        Subminor: subminor,
    })
}

fn parse_search_paths(path: &Path, language: &str, args: &[String]) -> Option<Vec<PathBuf>> {
    let mut clang_args = vec!["-E", "-x", language, "-", "-v"];
    clang_args.extend(args.iter().map(|s| &**s));
    let output = run_clang(path, &clang_args).1;
    let start = output.find("#include <...> search starts here:")? + 34;
    let end = output.find("End of search list.")?;
    let paths = output[start..end].replace("(framework directory)", "");
    Some(
        paths
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| Path::new(l.trim()).into())
            .collect(),
    )
}
