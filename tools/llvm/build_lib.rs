extern crate cc;
#[macro_use]
extern crate lazy_static;
extern crate regex;
extern crate semver;

use regex::Regex;
use semver::Version;
use std::env;
use std::ffi::OsStr;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::Command;

lazy_static! {
    static ref LLVM_PREFIX: String = format!("LLVM_RS_{}_PREFIX", env!("CARGO_PKG_VERSION_MAJOR"));
    static ref NO_CLEAN_CFLAGS: String = format!("LLVM_RS_{}_NO_CLEAN_CFLAGS", env!("CARGO_PKG_VERSION_MAJOR"));
    static ref FORCE_FFI: String = format!("LLVM_RS_{}_FFI_WORKAROUND", env!("CARGO_PKG_VERSION_MAJOR"));
}

lazy_static! {
    static ref CRATE_VERSION: Version = {
        let v = Version::parse(env!("CARGO_PKG_VERSION")).expect("Invalid crate version");
        Version {
            major: v.major / 10,
            minor: v.major % 10,
            ..v
        }
    };
    static ref LLVM_CONFIG_PATH: Option<PathBuf> = llvm_config_path();
}

fn target_env_is(n: &str) -> bool {
    match env::var_os("CARGO_CFG_TARGET_ENV") {
        Some(s) => s == n,
        None => false,
    }
}

fn target_os_is(n: &str) -> bool {
    match env::var_os("CARGO_CFG_TARGET_OS") {
        Some(s) => s == n,
        None => false,
    }
}

fn llvm_config_path() -> Option<PathBuf> {
    let pre = env::var_os(&*LLVM_PREFIX)
        .map(|p| PathBuf::from(p).join("bin"))
        .unwrap_or_else(PathBuf::new);
    for b in llvm_config_bins() {
        let b = pre.join(b);
        match llvm_version(&b) {
            Ok(ref v) if is_compatible(v) => {
                return Some(b);
            },
            Ok(v) => {
                println!("Found LLVM {} on PATH, need {}.", v, *CRATE_VERSION);
            },
            Err(ref e) if e.kind() == ErrorKind::NotFound => {},
            Err(e) => panic!("PATH has no llvm-config: {}", e),
        }
    }
    None
}

fn llvm_config_bins() -> std::vec::IntoIter<String> {
    let bs = vec!["llvm-config".into(), format!("llvm-config-{}", CRATE_VERSION.major)];
    bs.into_iter()
}

fn is_compatible(v: &Version) -> bool {
    v.major == CRATE_VERSION.major
}

fn llvm_config(arg: &str) -> String {
    try_llvm_config(Some(arg).into_iter()).expect("llvm-config failed")
}

fn try_llvm_config<'a>(xs: impl Iterator<Item = &'a str>) -> io::Result<String> {
    llvm_config_ex(&*LLVM_CONFIG_PATH.clone().unwrap(), xs)
}

fn llvm_config_ex<'a, S: AsRef<OsStr>>(x: S, xs: impl Iterator<Item = &'a str>) -> io::Result<String> {
    Command::new(x).args(xs).output().and_then(|y| {
        if y.status.code() != Some(0) {
            Err(io::Error::new(
                io::ErrorKind::Other,
                format!("error code {:?}", y.status.code()),
            ))
        } else if y.stdout.is_empty() {
            Err(io::Error::new(io::ErrorKind::NotFound, "empty return"))
        } else {
            Ok(String::from_utf8(y.stdout).expect("invalid UTF-8 return"))
        }
    })
}

fn llvm_version<S: AsRef<OsStr>>(x: &S) -> io::Result<Version> {
    let v = llvm_config_ex(x.as_ref(), ["--version"].iter().copied())?;
    let re = Regex::new(r"^(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))??").unwrap();
    let c = match re.captures(&v) {
        Some(c) => c,
        None => {
            panic!("llvm-config returned no version: {}", v);
        },
    };
    let s = match c.name("patch") {
        None => format!("{}.0", &c[0]),
        Some(_) => c[0].to_string(),
    };
    Ok(Version::parse(&s).unwrap())
}

fn get_system_libraries(kind: LibraryKind) -> Vec<String> {
    let link_arg = match kind {
        LibraryKind::Static => "--link-static",
        LibraryKind::Dynamic => "--link-shared",
    };
    try_llvm_config(["--system-libs", link_arg].iter().copied())
        .expect("Surprising failure from llvm-config")
        .split(&[' ', '\n'] as &[char])
        .filter(|s| !s.is_empty())
        .map(|flag| {
            if flag.starts_with("-l") {
                return flag[2..].to_owned();
            }
            let maybe_lib = Path::new(&flag);
            if maybe_lib.is_file() {
                println!("cargo:rustc-link-search={}", maybe_lib.parent().unwrap().display());
                let soname = maybe_lib
                    .file_name()
                    .unwrap()
                    .to_str()
                    .expect("Shared library path must be a valid string");
                let stem = soname
                    .rsplit_once(target_dylib_extension())
                    .expect("Shared library should be a .so file")
                    .0;
                stem.trim_start_matches("lib")
            } else {
                panic!("Unable to parse result of llvm-config --system-libs: was {:?}", flag)
            }
            .to_owned()
        })
        .chain(get_system_libcpp().map(str::to_owned))
        .collect::<Vec<String>>()
}

fn target_dylib_extension() -> &'static str {
    ".so"
}

fn get_system_libcpp() -> Option<&'static str> {
    Some("stdc++")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LibraryKind {
    Static,
    Dynamic,
}

impl LibraryKind {
    pub fn from_is_static(is_static: bool) -> Self {
        if is_static {
            LibraryKind::Static
        } else {
            LibraryKind::Dynamic
        }
    }

    pub fn string(&self) -> &'static str {
        match self {
            LibraryKind::Static => "static",
            LibraryKind::Dynamic => "dylib",
        }
    }
}

fn get_link_libraries(preferences: &LinkingPreferences) -> (LibraryKind, Vec<String>) {
    fn get_link_libraries_impl(is_static: bool) -> std::io::Result<String> {
        let link_arg = if is_static { "--link-static" } else { "--link-shared" };
        try_llvm_config(["--libnames", link_arg].iter().copied())
    }
    fn lib_kind(is_static: bool) -> &'static str {
        if is_static {
            "static"
        } else {
            "shared"
        }
    }
    let is_static = preferences.prefer_static;
    let first_err;
    match get_link_libraries_impl(is_static) {
        Ok(s) => return (LibraryKind::from_is_static(is_static), extract_library(&s, is_static)),
        Err(e) => first_err = e,
    }

    let mut second_err = None;

    if !preferences.force {
        println!(
            "cargo:warning=failed to get {} libraries from llvm-config, falling back to {}.",
            lib_kind(is_static),
            lib_kind(!is_static),
        );
        println!("cargo:warning=error: {}", first_err);

        match get_link_libraries_impl(!is_static) {
            Ok(s) => return (LibraryKind::from_is_static(!is_static), extract_library(&s, !is_static)),
            Err(e) => second_err = Some(e),
        }
    }

    let first_error = format!("linking {} library error: {}", lib_kind(is_static), first_err);
    let second_error = if let Some(err) = second_err {
        format!("\nlinking {} library error: {}", lib_kind(!is_static), err)
    } else {
        String::new()
    };

    panic!(
        "failed to get linking libraries from llvm-config.\n{}{}",
        first_error, second_error
    );
}

fn extract_library(s: &str, is_static: bool) -> Vec<String> {
    s.split(&[' ', '\n'] as &[char])
        .filter(|s| !s.is_empty())
        .map(|name| {
            if is_static {
                if name.ends_with(".a") {
                    &name[3..name.len() - 2]
                } else {
                    panic!("{:?} does not look like a static library name", name)
                }
            } else {
                if name.ends_with(".so") {
                    &name[3..name.len() - 3]
                } else {
                    panic!("{:?} does not look like a shared library name", name)
                }
            }
            .to_string()
        })
        .collect::<Vec<String>>()
}

#[derive(Debug, Clone, Copy)]
struct LinkingPreferences {
    prefer_static: bool,
    force: bool,
}

impl LinkingPreferences {
    fn init() -> LinkingPreferences {
        let prefer_static = cfg!(feature = "prefer-static");
        let prefer_dynamic = cfg!(feature = "prefer-dynamic");
        let force_static = cfg!(feature = "force-static");
        let force_dynamic = cfg!(feature = "force-dynamic");
        if [prefer_static, prefer_dynamic, force_static, force_dynamic]
            .iter()
            .filter(|&&x| x)
            .count()
            > 1
        {
            panic!(
                "Only one of the features `prefer-static`, `prefer-dynamic`, `force-static`, \
                 `force-dynamic` can be enabled at once."
            );
        }
        let force_static = force_static || !(prefer_static || prefer_dynamic || force_dynamic);
        LinkingPreferences {
            prefer_static: force_static || prefer_static,
            force: force_static || force_dynamic,
        }
    }
}

fn get_llvm_cflags() -> String {
    let output = llvm_config("--cflags");
    let no_clean = env::var_os(&*NO_CLEAN_CFLAGS).is_some();
    if no_clean {
        return output;
    }
    llvm_config("--cflags")
        .split(&[' ', '\n'][..])
        .filter(|word| !word.starts_with("-W"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_llvm_debug() -> bool {
    llvm_config("--build-mode").contains("Debug")
}

fn main() {
    println!("cargo:rerun-if-env-changed={}", &*LLVM_PREFIX);
    if let Ok(path) = env::var(&*LLVM_PREFIX) {
        println!("cargo:rerun-if-changed={}", path);
    }
    println!("cargo:rerun-if-env-changed={}", &*NO_CLEAN_CFLAGS);
    println!("cargo:rerun-if-env-changed={}", &*FORCE_FFI);
    if cfg!(feature = "no-llvm-linking") && cfg!(feature = "disable-alltargets-init") {
        return;
    }
    if LLVM_CONFIG_PATH.is_none() {
        println!("cargo:rustc-cfg=LLVM_RS_NOT_FOUND");
        return;
    }
    if !cfg!(feature = "disable-alltargets-init") {
        std::env::set_var("CFLAGS", get_llvm_cflags());
        cc::Build::new().file("wrappers/target.c").compile("targetwrappers");
    }
    if cfg!(feature = "no-llvm-linking") {
        return;
    }
    let libdir = llvm_config("--libdir");
    println!("cargo:config_path={}", LLVM_CONFIG_PATH.clone().unwrap().display()); // will be DEP_LLVM_CONFIG_PATH
    println!("cargo:libdir={}", libdir); // DEP_LLVM_LIBDIR

    let preferences = LinkingPreferences::init();
    println!("cargo:rustc-link-search=native={}", libdir);
    let (kind, libs) = get_link_libraries(&preferences);
    for name in libs {
        println!("cargo:rustc-link-lib={}={}", kind.string(), name);
    }
    let sys_lib_kind = if target_env_is("musl") {
        LibraryKind::Static
    } else {
        LibraryKind::Dynamic
    };
    for name in get_system_libraries(kind) {
        println!("cargo:rustc-link-lib={}={}", sys_lib_kind.string(), name);
    }

    let force_ffi = env::var_os(&*FORCE_FFI).is_some();
    if force_ffi {
        println!("cargo:rustc-link-lib=dylib={}", "ffi");
    }
}
