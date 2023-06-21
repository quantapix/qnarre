#![allow(missing_docs)]
#![allow(non_upper_case_globals, dead_code)]
#![deny(clippy::disallowed_methods)]
#![deny(unused_extern_crates)]
#![recursion_limit = "128"]
#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
#[macro_use]
extern crate quote;

pub mod clang;
mod codegen;
mod ir;
mod opts;

pub use codegen::utils::variation;
use codegen::GenError;
pub use ir::annos::VisibilityKind;
use ir::comment;
pub use ir::func::Abi;
use ir::item::Item;
use ir::{Context, ItemId};
use opts::Opts;
use parse::Error;
pub use regex_set::RegexSet;
use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::env;
use std::ffi::OsStr;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::rc::Rc;
use std::str::FromStr;

type HashMap<K, V> = rustc_hash::FxHashMap<K, V>;
type HashSet<K> = rustc_hash::FxHashSet<K>;

bitflags! {
    #[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct Config: u32 {
        const FNS = 1 << 0;
        const TYPS = 1 << 1;
        const VARS = 1 << 2;
        const METHODS = 1 << 3;
        const CONSTRS = 1 << 4;
        const DESTRS = 1 << 5;
    }
}

impl Config {
    pub fn fns(self) -> bool {
        self.contains(Config::FNS)
    }
    pub fn typs(self) -> bool {
        self.contains(Config::TYPS)
    }
    pub fn vars(self) -> bool {
        self.contains(Config::VARS)
    }
    pub fn methods(self) -> bool {
        self.contains(Config::METHODS)
    }
    pub fn constrs(self) -> bool {
        self.contains(Config::CONSTRS)
    }
    pub fn destrs(self) -> bool {
        self.contains(Config::DESTRS)
    }
}
impl Default for Config {
    fn default() -> Self {
        Config::all()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Formatter {
    None,
    Rustfmt,
    Prettyplease,
}
impl Default for Formatter {
    fn default() -> Self {
        Self::Rustfmt
    }
}
impl FromStr for Formatter {
    type Err = String;
    fn from_str(x: &str) -> Result<Self, Self::Err> {
        match x {
            "none" => Ok(Self::None),
            "rustfmt" => Ok(Self::Rustfmt),
            "prettyplease" => Ok(Self::Prettyplease),
            _ => Err(format!("`{}` is not a valid formatter", x)),
        }
    }
}
impl std::fmt::Display for Formatter {
    fn fmt(&self, x: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let y = match self {
            Self::None => "none",
            Self::Rustfmt => "rustfmt",
            Self::Prettyplease => "prettyplease",
        };
        y.fmt(x)
    }
}

#[derive(Debug, Default, Clone)]
pub struct Builder {
    opts: Opts,
}
impl Builder {
    pub fn generate(mut self) -> Result<Bindings, BindgenError> {
        self.opts.clang_args.extend(get_extra_args(&self.opts.parse_callbacks));
        self.opts.clang_args.extend(
            self.opts.input_headers[..self.opts.input_headers.len().saturating_sub(1)]
                .iter()
                .flat_map(|x| ["-include".into(), x.to_string()]),
        );
        let ys = std::mem::take(&mut self.opts.input_header_contents)
            .into_iter()
            .map(|(name, x)| clang::UnsavedFile::new(name, x))
            .collect::<Vec<_>>();
        Bindings::generate(self.opts, ys)
    }
    pub fn dump_preprocessed_input(&self) -> io::Result<()> {
        let lib = clang_lib::Clang::find(None, &[])
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Cannot find clang executable"))?;
        let mut is_cpp = args_are_cpp(&self.opts.clang_args);
        let mut y = String::new();
        for x in &self.opts.input_headers {
            is_cpp |= file_is_cpp(x);
            y.push_str("#include \"");
            y.push_str(x);
            y.push_str("\"\n");
        }
        for (name, contents) in &self.opts.input_header_contents {
            is_cpp |= file_is_cpp(name);
            y.push_str("#line 0 \"");
            y.push_str(name);
            y.push_str("\"\n");
            y.push_str(contents);
        }
        let p = PathBuf::from(if is_cpp { "__bindgen.cpp" } else { "__bindgen.c" });
        {
            let mut f = File::create(&p)?;
            f.write_all(y.as_bytes())?;
        }
        let mut cmd = Command::new(lib.path);
        cmd.arg("-save-temps")
            .arg("-E")
            .arg("-C")
            .arg("-c")
            .arg(&p)
            .stdout(Stdio::piped());
        for x in &self.opts.clang_args {
            cmd.arg(x);
        }
        for x in get_extra_args(&self.opts.parse_callbacks) {
            cmd.arg(x);
        }
        let mut child = cmd.spawn()?;
        let mut preproc = child.stdout.take().unwrap();
        let mut f = File::create(if is_cpp { "__bindgen.ii" } else { "__bindgen.i" })?;
        io::copy(&mut preproc, &mut f)?;
        if child.wait()?.success() {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "clang exited with non-zero status",
            ))
        }
    }
}
pub fn builder() -> Builder {
    Default::default()
}

impl Opts {
    fn build(&mut self) {
        const REGEX_SETS_LEN: usize = 24;
        let ys: [_; REGEX_SETS_LEN] = [
            &mut self.allowed_vars,
            &mut self.allowed_types,
            &mut self.allowed_fns,
            &mut self.allowed_files,
            &mut self.blocklisted_types,
            &mut self.blocklisted_fns,
            &mut self.blocklisted_items,
            &mut self.blocklisted_files,
            &mut self.opaque_types,
            &mut self.constified_enums,
            &mut self.constified_enum_mods,
            &mut self.rustified_enums,
            &mut self.rustified_non_exhaustive_enums,
            &mut self.type_alias,
            &mut self.new_type_alias,
            &mut self.new_type_alias_deref,
            &mut self.bindgen_wrapper_union,
            &mut self.manually_drop_union,
            &mut self.no_partialeq_types,
            &mut self.no_copy_types,
            &mut self.no_debug_types,
            &mut self.no_default_types,
            &mut self.no_hash_types,
            &mut self.must_use_types,
        ];
        let ms = self.record_matches;
        for y in self.abi_overrides.values_mut().chain(ys) {
            y.build(ms);
        }
    }
    fn last_callback<T>(&self, f: impl Fn(&dyn callbacks::Parse) -> Option<T>) -> Option<T> {
        self.parse_callbacks.iter().filter_map(|x| f(x.as_ref())).last()
    }
    fn all_callbacks<T>(&self, f: impl Fn(&dyn callbacks::Parse) -> Vec<T>) -> Vec<T> {
        self.parse_callbacks.iter().flat_map(|x| f(x.as_ref())).collect()
    }
    fn process_comment(&self, x: &str) -> String {
        let y = comment::preproc(x);
        self.parse_callbacks
            .last()
            .and_then(|x| x.process_comment(&y))
            .unwrap_or(y)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BindgenError {
    FolderAsHeader(PathBuf),
    NoReadPerms(PathBuf),
    NotExist(PathBuf),
    Diagnostic(String),
    Codegen(GenError),
}
impl std::fmt::Display for BindgenError {
    fn fmt(&self, y: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BindgenError::FolderAsHeader(x) => {
                write!(y, "'{}' is a folder", x.display())
            },
            BindgenError::NoReadPerms(x) => {
                write!(y, "insufficient permissions to read '{}'", x.display())
            },
            BindgenError::NotExist(x) => {
                write!(y, "header '{}' does not exist.", x.display())
            },
            BindgenError::Diagnostic(x) => {
                write!(y, "clang diagnosed error: {}", x)
            },
            BindgenError::Codegen(x) => {
                write!(y, "codegen error: {}", x)
            },
        }
    }
}
impl std::error::Error for BindgenError {}

pub const HOST_TARGET: &str = include_str!(concat!(env!("OUT_DIR"), "/host-target.txt"));

#[derive(Debug)]
pub struct Bindings {
    opts: Opts,
    module: proc_macro2::TokenStream,
}
impl Bindings {
    pub fn generate(mut opts: Opts, input_unsaved_files: Vec<clang::UnsavedFile>) -> Result<Bindings, BindgenError> {
        load_libclang();
        opts.build();
        let (effective_target, explicit_target) = find_effective_target(&opts.clang_args);
        let is_host_build = rust_to_clang_target(HOST_TARGET) == effective_target;
        if !explicit_target && !is_host_build {
            opts.clang_args.insert(0, format!("--target={}", effective_target));
        };
        fn detect_include_paths(opts: &mut Opts) {
            if !opts.detect_include_paths {
                return;
            }
            let clang_args_for_clang = {
                let mut last_was_include_prefix = false;
                opts.clang_args
                    .iter()
                    .filter(|x| {
                        if last_was_include_prefix {
                            last_was_include_prefix = false;
                            return false;
                        }
                        let x = &**x;
                        if x == "-I" || x == "--include-directory" {
                            last_was_include_prefix = true;
                            return false;
                        }
                        if x.starts_with("-I") || x.starts_with("--include-directory=") {
                            return false;
                        }
                        true
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            };
            let clang = match clang_lib::Clang::find(None, &clang_args_for_clang) {
                None => return,
                Some(x) => x,
            };
            let is_cpp = args_are_cpp(&opts.clang_args) || opts.input_headers.iter().any(|x| file_is_cpp(x));
            let paths = if is_cpp { clang.cpp_ps } else { clang.c_ps };
            if let Some(paths) = paths {
                for p in paths.into_iter() {
                    if let Ok(p) = p.into_os_string().into_string() {
                        opts.clang_args.push("-isystem".to_owned());
                        opts.clang_args.push(p);
                    }
                }
            }
        }
        detect_include_paths(&mut opts);
        #[cfg(unix)]
        fn can_read(perms: &std::fs::Permissions) -> bool {
            use std::os::unix::fs::PermissionsExt;
            perms.mode() & 0o444 > 0
        }
        if let Some(h) = opts.input_headers.last() {
            let path = Path::new(h);
            if let Ok(md) = std::fs::metadata(path) {
                if md.is_dir() {
                    return Err(BindgenError::FolderAsHeader(path.into()));
                }
                if !can_read(&md.permissions()) {
                    return Err(BindgenError::NoReadPerms(path.into()));
                }
                let h = h.clone();
                opts.clang_args.push(h);
            } else {
                return Err(BindgenError::NotExist(path.into()));
            }
        }
        for (idx, f) in input_unsaved_files.iter().enumerate() {
            if idx != 0 || !opts.input_headers.is_empty() {
                opts.clang_args.push("-include".to_owned());
            }
            opts.clang_args.push(f.name.to_str().unwrap().to_owned())
        }
        let time_phases = opts.time_phases;
        let mut ctx = Context::new(opts, &input_unsaved_files);
        if is_host_build {
            debug_assert_eq!(
                ctx.target_ptr_size(),
                std::mem::size_of::<*mut ()>(),
                "{:?} {:?}",
                effective_target,
                HOST_TARGET
            );
        }
        {
            let _t = timer::Timer::new("parse").with_output(time_phases);
            parse(&mut ctx)?;
        }
        let (module, opts) = codegen::codegen(ctx).map_err(BindgenError::Codegen)?;
        Ok(Bindings { opts, module })
    }
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let y = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(path.as_ref())?;
        self.write(Box::new(y))?;
        Ok(())
    }
    pub fn write<'a>(&self, mut writer: Box<dyn Write + 'a>) -> io::Result<()> {
        if !self.opts.disable_header_comment {
            let version = option_env!("CARGO_PKG_VERSION");
            let header = format!(
                "/* automatically generated by rust-bindgen {} */\n\n",
                version.unwrap_or("(unknown version)")
            );
            writer.write_all(header.as_bytes())?;
        }
        for line in self.opts.raw_lines.iter() {
            writer.write_all(line.as_bytes())?;
            writer.write_all("\n".as_bytes())?;
        }
        if !self.opts.raw_lines.is_empty() {
            writer.write_all("\n".as_bytes())?;
        }
        match self.format_toks(&self.module) {
            Ok(x) => {
                writer.write_all(x.as_bytes())?;
            },
            Err(x) => {
                eprintln!("Failed to run rustfmt: {} (non-fatal, continuing)", x);
                writer.write_all(self.module.to_string().as_bytes())?;
            },
        }
        Ok(())
    }
    fn rustfmt_path(&self) -> io::Result<Cow<PathBuf>> {
        debug_assert!(matches!(self.opts.formatter, Formatter::Rustfmt));
        if let Some(ref p) = self.opts.rustfmt_path {
            return Ok(Cow::Borrowed(p));
        }
        if let Ok(rustfmt) = env::var("RUSTFMT") {
            return Ok(Cow::Owned(rustfmt.into()));
        }
        #[cfg(feature = "which-rustfmt")]
        match which::which("rustfmt") {
            Ok(p) => Ok(Cow::Owned(p)),
            Err(e) => Err(io::Error::new(io::ErrorKind::Other, format!("{}", e))),
        }
        #[cfg(not(feature = "which-rustfmt"))]
        Ok(Cow::Owned("rustfmt".into()))
    }
    fn format_toks(&self, tokens: &proc_macro2::TokenStream) -> io::Result<String> {
        let _t = timer::Timer::new("rustfmt_generated_string").with_output(self.opts.time_phases);
        match self.opts.formatter {
            Formatter::None => return Ok(tokens.to_string()),
            Formatter::Prettyplease => {
                return Ok(prettyplease::unparse(&syn::parse_quote!(#tokens)));
            },
            Formatter::Rustfmt => (),
        }
        let rustfmt = self.rustfmt_path()?;
        let mut cmd = Command::new(&*rustfmt);
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
        if let Some(path) = self.opts.rustfmt_configuration_file.as_ref().and_then(|f| f.to_str()) {
            cmd.args(["--config-path", path]);
        }
        let mut child = cmd.spawn()?;
        let mut child_stdin = child.stdin.take().unwrap();
        let mut child_stdout = child.stdout.take().unwrap();
        let source = tokens.to_string();
        let stdin_handle = ::std::thread::spawn(move || {
            let _ = child_stdin.write_all(source.as_bytes());
            source
        });
        let mut output = vec![];
        io::copy(&mut child_stdout, &mut output)?;
        let status = child.wait()?;
        let source = stdin_handle.join().expect(
            "The thread writing to rustfmt's stdin doesn't do \
             anything that could panic",
        );
        match String::from_utf8(output) {
            Ok(x) => match status.code() {
                Some(0) => Ok(x),
                Some(2) => Err(io::Error::new(
                    io::ErrorKind::Other,
                    "Rustfmt parsing errors.".to_string(),
                )),
                Some(3) => Ok(x),
                _ => Err(io::Error::new(
                    io::ErrorKind::Other,
                    "Internal rustfmt error".to_string(),
                )),
            },
            _ => Ok(source),
        }
    }
}
impl std::fmt::Display for Bindings {
    fn fmt(&self, x: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut bytes = vec![];
        self.write(Box::new(&mut bytes) as Box<dyn Write>)
            .expect("writing to a vec cannot fail");
        x.write_str(std::str::from_utf8(&bytes).expect("we should only write bindings that are valid utf-8"))
    }
}

#[derive(Debug)]
pub struct Version {
    pub parsed: Option<(u32, u32)>,
    pub full: String,
}

#[derive(Debug)]
pub struct CargoCallbacks;
impl callbacks::Parse for CargoCallbacks {
    fn include_file(&self, x: &str) {
        println!("cargo:rerun-if-changed={}", x);
    }
    fn read_env_var(&self, x: &str) {
        println!("cargo:rerun-if-env-changed={}", x);
    }
}

mod deps {
    use std::{collections::BTreeSet, path::PathBuf};
    #[derive(Clone, Debug)]
    pub struct DepfileSpec {
        pub out_mod: String,
        pub dep_path: PathBuf,
    }
    impl DepfileSpec {
        pub fn write(&self, deps: &BTreeSet<String>) -> std::io::Result<()> {
            std::fs::write(&self.dep_path, self.to_string(deps))
        }
        fn to_string(&self, deps: &BTreeSet<String>) -> String {
            let escape = |x: &str| x.replace('\\', "\\\\").replace(' ', "\\ ");
            let mut buf = format!("{}:", escape(&self.out_mod));
            for file in deps {
                buf = format!("{} {}", buf, escape(file));
            }
            buf
        }
    }

    pub mod callbacks {
        pub use crate::ir::analysis::DeriveTrait;
        pub use crate::ir::derive::Resolved as ImplementsTrait;
        pub use crate::ir::enum_ty::{EnumVariantCustom, EnumVariantValue};
        pub use crate::ir::int::IntKind;
        use std::fmt;
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
        pub enum MacroParsing {
            Ignore,
            #[default]
            Default,
        }
        pub trait Parse: fmt::Debug {
            fn will_parse_macro(&self, _name: &str) -> MacroParsing {
                MacroParsing::Default
            }
            fn gen_name_override(&self, _: ItemInfo<'_>) -> Option<String> {
                None
            }
            fn gen_link_name_override(&self, _: ItemInfo<'_>) -> Option<String> {
                None
            }
            fn int_macro(&self, _name: &str, _value: i64) -> Option<IntKind> {
                None
            }
            fn str_macro(&self, _name: &str, _value: &[u8]) {}
            fn fn_macro(&self, _name: &str, _value: &[&[u8]]) {}
            fn enum_variant_behavior(
                &self,
                _name: Option<&str>,
                _orig_name: &str,
                _: EnumVariantValue,
            ) -> Option<EnumVariantCustom> {
                None
            }
            fn enum_variant_name(&self, _name: Option<&str>, _orig_name: &str, _: EnumVariantValue) -> Option<String> {
                None
            }
            fn item_name(&self, _orig_name: &str) -> Option<String> {
                None
            }
            fn include_file(&self, _name: &str) {}
            fn read_env_var(&self, _key: &str) {}
            fn blocklisted_type_implements_trait(&self, _name: &str, _: DeriveTrait) -> Option<ImplementsTrait> {
                None
            }
            fn add_derives(&self, _: &DeriveInfo<'_>) -> Vec<String> {
                vec![]
            }
            fn process_comment(&self, _comment: &str) -> Option<String> {
                None
            }
        }
        #[derive(Debug)]
        #[non_exhaustive]
        pub struct DeriveInfo<'a> {
            pub name: &'a str,
            pub kind: TypeKind,
        }
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum TypeKind {
            Struct,
            Enum,
            Union,
        }
        #[non_exhaustive]
        pub struct ItemInfo<'a> {
            pub name: &'a str,
            pub kind: ItemKind,
        }
        #[non_exhaustive]
        pub enum ItemKind {
            Func,
            Var,
        }
    }
    pub mod parse {
        use crate::clang;
        use crate::ir::{Context, ItemId};
        #[derive(Debug)]
        pub enum Error {
            Recurse,
            Continue,
        }
        #[derive(Debug)]
        pub enum Resolved<T> {
            AlreadyDone(ItemId),
            New(T, Option<clang::Cursor>),
        }
        pub trait SubItem: Sized {
            fn parse(cur: clang::Cursor, ctx: &mut Context) -> Result<Resolved<Self>, Error>;
        }
    }
    mod regex_set {
        use std::cell::Cell;
        #[derive(Clone, Debug, Default)]
        pub struct RegexSet {
            items: Vec<String>,
            matched: Vec<Cell<bool>>,
            set: Option<regex::RegexSet>,
            record_matches: bool,
        }
        impl RegexSet {
            pub fn new() -> RegexSet {
                RegexSet::default()
            }
            pub fn is_empty(&self) -> bool {
                self.items.is_empty()
            }
            pub fn insert<S>(&mut self, x: S)
            where
                S: AsRef<str>,
            {
                self.items.push(x.as_ref().to_owned());
                self.matched.push(Cell::new(false));
                self.set = None;
            }
            pub fn get_items(&self) -> &[String] {
                &self.items[..]
            }
            pub fn unmatched_items(&self) -> impl Iterator<Item = &String> {
                self.items.iter().enumerate().filter_map(move |(i, x)| {
                    if !self.record_matches || self.matched[i].get() {
                        return None;
                    }
                    Some(x)
                })
            }
            #[inline]
            pub fn build(&mut self, record_matches: bool) {
                self.build_inner(record_matches, None)
            }
            fn build_inner(&mut self, record_matches: bool, _name: Option<&'static str>) {
                let items = self.items.iter().map(|x| format!("^({})$", x));
                self.record_matches = record_matches;
                self.set = match regex::RegexSet::new(items) {
                    Ok(x) => Some(x),
                    Err(e) => {
                        warn!("Invalid regex in {:?}: {:?}", self.items, e);
                        None
                    },
                }
            }
            pub fn matches<S>(&self, x: S) -> bool
            where
                S: AsRef<str>,
            {
                let s = x.as_ref();
                let set = match self.set {
                    Some(ref set) => set,
                    None => return false,
                };
                if !self.record_matches {
                    return set.is_match(s);
                }
                let ys = set.matches(s);
                if !ys.matched_any() {
                    return false;
                }
                for i in ys.iter() {
                    self.matched[i].set(true);
                }
                true
            }
        }
    }
    mod timer {
        use std::io::{self, Write};
        use std::time::{Duration, Instant};
        #[derive(Debug)]
        pub struct Timer<'a> {
            output: bool,
            name: &'a str,
            start: Instant,
        }
        impl<'a> Timer<'a> {
            pub fn new(name: &'a str) -> Self {
                Timer {
                    output: true,
                    name,
                    start: Instant::now(),
                }
            }
            pub fn with_output(mut self, x: bool) -> Self {
                self.output = x;
                self
            }
            pub fn elapsed(&self) -> Duration {
                Instant::now() - self.start
            }
            fn print_elapsed(&mut self) {
                if self.output {
                    let d = self.elapsed();
                    let ms = (d.as_secs() as f64) * 1e3 + (d.subsec_nanos() as f64) / 1e6;
                    let e = io::stderr();
                    writeln!(e.lock(), "  time: {:>9.3} ms.\t{}", ms, self.name).expect("should not fail");
                }
            }
        }
        impl<'a> Drop for Timer<'a> {
            fn drop(&mut self) {
                self.print_elapsed();
            }
        }
    }

    #[cfg(feature = "runtime")]
    fn load_libclang() {
        if clang_lib::is_loaded() {
            return;
        }
        lazy_static! {
            static ref LIBCLANG: std::sync::Arc<clang_lib::SharedLib> = {
                clang_lib::load().expect("Unable to find libclang");
                clang_lib::get_lib().expect(
                    "We just loaded libclang and it had better still be \
                 here!",
                )
            };
        }
        clang_lib::set_lib(Some(LIBCLANG.clone()));
    }
    #[cfg(not(feature = "runtime"))]
    fn load_libclang() {}

    fn file_is_cpp(x: &str) -> bool {
        x.ends_with(".hpp") || x.ends_with(".hxx") || x.ends_with(".hh") || x.ends_with(".h++")
    }
    fn args_are_cpp(xs: &[String]) -> bool {
        for x in xs.windows(2) {
            if x[0] == "-xc++" || x[1] == "-xc++" {
                return true;
            }
            if x[0] == "-x" && x[1] == "c++" {
                return true;
            }
            if x[0] == "-include" && file_is_cpp(&x[1]) {
                return true;
            }
        }
        false
    }

    fn get_extra_args(xs: &[Rc<dyn callbacks::Parse>]) -> Vec<String> {
        let ys = match get_target_dependent_env_var(xs, "BINDGEN_EXTRA_CLANG_ARGS") {
            None => return vec![],
            Some(x) => x,
        };
        if let Some(ys) = shlex::split(&ys) {
            return ys;
        }
        vec![ys]
    }

    pub fn clang_version() -> Version {
        load_libclang();
        let raw_v: String = clang::extract_clang_version();
        let split_v: Option<Vec<&str>> = raw_v
            .split_whitespace()
            .find(|x| x.chars().next().map_or(false, |x| x.is_ascii_digit()))
            .map(|x| x.split('.').collect());
        if let Some(v) = split_v {
            if v.len() >= 2 {
                let maybe_major = v[0].parse::<u32>();
                let maybe_minor = v[1].parse::<u32>();
                if let (Ok(major), Ok(minor)) = (maybe_major, maybe_minor) {
                    return Version {
                        parsed: Some((major, minor)),
                        full: raw_v.clone(),
                    };
                }
            }
        };
        Version {
            parsed: None,
            full: raw_v.clone(),
        }
    }
    fn env_var<K: AsRef<str> + AsRef<OsStr>>(
        parse_callbacks: &[Rc<dyn callbacks::Parse>],
        key: K,
    ) -> Result<String, std::env::VarError> {
        for x in parse_callbacks {
            x.read_env_var(key.as_ref());
        }
        std::env::var(key)
    }
    fn get_target_dependent_env_var(xs: &[Rc<dyn callbacks::Parse>], var: &str) -> Option<String> {
        if let Ok(x) = env_var(xs, "TARGET") {
            if let Ok(x) = env_var(xs, format!("{}_{}", var, x)) {
                return Some(x);
            }
            if let Ok(x) = env_var(xs, format!("{}_{}", var, x.replace('-', "_"))) {
                return Some(x);
            }
        }
        env_var(xs, var).ok()
    }

    fn rust_to_clang_target(x: &str) -> String {
        x.to_owned()
    }
    fn find_effective_target(xs: &[String]) -> (String, bool) {
        let mut args = xs.iter();
        while let Some(opt) = args.next() {
            if opt.starts_with("--target=") {
                let mut split = opt.split('=');
                split.next();
                return (split.next().unwrap().to_owned(), true);
            }
            if opt == "-target" {
                if let Some(x) = args.next() {
                    return (x.clone(), true);
                }
            }
        }
        if let Ok(t) = env::var("TARGET") {
            return (rust_to_clang_target(&t), false);
        }
        (rust_to_clang_target(HOST_TARGET), false)
    }
    fn filter_builtins(ctx: &Context, cur: &clang::Cursor) -> bool {
        ctx.opts().builtins || !cur.is_builtin()
    }
    fn parse_one(ctx: &mut Context, cur: clang::Cursor, parent: Option<ItemId>) -> clang_lib::CXChildVisitResult {
        if !filter_builtins(ctx, &cur) {
            return clang_lib::CXChildVisit_Continue;
        }
        match Item::parse(cur, parent, ctx) {
            Ok(..) => {},
            Err(Error::Continue) => {},
            Err(Error::Recurse) => {
                cur.visit(|x| parse_one(ctx, x, parent));
            },
        }
        clang_lib::CXChildVisit_Continue
    }
    fn parse(ctx: &mut Context) -> Result<(), BindgenError> {
        let mut error = None;
        for d in ctx.translation_unit().diags().iter() {
            let msg = d.format();
            let is_err = d.severity() >= clang_lib::CXDiagnostic_Error;
            if is_err {
                let y = error.get_or_insert_with(String::new);
                y.push_str(&msg);
                y.push('\n');
            } else {
                eprintln!("clang diag: {}", msg);
            }
        }
        if let Some(x) = error {
            return Err(BindgenError::Diagnostic(x));
        }
        let cur = ctx.translation_unit().cursor();
        if ctx.opts().emit_ast {
            fn dump_if_not_builtin(cur: &clang::Cursor) -> clang_lib::CXChildVisitResult {
                if !cur.is_builtin() {
                    clang::ast_dump(cur, 0)
                } else {
                    clang_lib::CXChildVisit_Continue
                }
            }
            cur.visit(|x| dump_if_not_builtin(&x));
        }
        let root = ctx.root_mod();
        ctx.with_mod(root, |x| cur.visit(|x2| parse_one(x, x2, None)));
        assert!(ctx.current_mod() == ctx.root_mod(), "How did this happen?");
        Ok(())
    }

    #[test]
    fn commandline_flag_unit_test() {
        let bindings = crate::builder();
        let command_line_flags = bindings.command_line_flags();
        let test_cases = vec![
            "--rust-target",
            "--no-derive-default",
            "--generate",
            "functions,types,vars,methods,constructors,destructors",
        ]
        .iter()
        .map(|&x| x.into())
        .collect::<Vec<String>>();
        assert!(test_cases.iter().all(|x| command_line_flags.contains(x)));
        let bindings = crate::builder()
            .header("input_header")
            .allowlist_type("Distinct_Type")
            .allowlist_fn("safe_function");
        let command_line_flags = bindings.command_line_flags();
        let test_cases = vec![
            "--rust-target",
            "input_header",
            "--no-derive-default",
            "--generate",
            "functions,types,vars,methods,constructors,destructors",
            "--allowlist-type",
            "Distinct_Type",
            "--allowlist-function",
            "safe_function",
        ]
        .iter()
        .map(|&x| x.into())
        .collect::<Vec<String>>();
        println!("{:?}", command_line_flags);
        assert!(test_cases.iter().all(|x| command_line_flags.contains(x)));
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn escaping_depfile() {
            let spec = DepfileSpec {
                out_mod: "Mod Name".to_owned(),
                dep_path: PathBuf::new(),
            };
            let deps: BTreeSet<String> = vec![
                r"/absolute/path".to_owned(),
                r"C:\win\absolute\path".to_owned(),
                r"../relative/path".to_owned(),
                r"..\win\relative\path".to_owned(),
                r"../path/with spaces/in/it".to_owned(),
                r"..\win\path\with spaces\in\it".to_owned(),
                r"path\with/mixed\separators".to_owned(),
            ]
            .into_iter()
            .collect();
            assert_eq!(
                spec.to_string(&deps),
                "Mod\\ Name: \
        ../path/with\\ spaces/in/it \
        ../relative/path \
        ..\\\\win\\\\path\\\\with\\ spaces\\\\in\\\\it \
        ..\\\\win\\\\relative\\\\path \
        /absolute/path \
        C:\\\\win\\\\absolute\\\\path \
        path\\\\with/mixed\\\\separators"
            );
        }
    }
}
