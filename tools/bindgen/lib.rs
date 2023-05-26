#![deny(missing_docs)]
#![deny(unused_extern_crates)]
#![deny(clippy::disallowed_methods)]
#![allow(non_upper_case_globals)]
#![recursion_limit = "128"]

#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate quote;

#[cfg(feature = "logging")]
#[macro_use]
extern crate log;

#[cfg(not(feature = "logging"))]
#[macro_use]
mod log_stubs;

#[macro_use]
mod extra_assertions;

mod codegen;
mod deps;
mod options;
mod time;

pub mod callbacks;

mod clang;
#[cfg(feature = "experimental")]
mod diagnostics;
mod features;
mod ir;
mod parse;
mod regex_set;

pub use codegen::{AliasVariation, EnumVariation, MacroTypeVariation, NonCopyUnionStyle};
pub use features::{RustTarget, LATEST_STABLE_RUST, RUST_TARGET_STRINGS};
pub use ir::annotations::FieldVisibilityKind;
pub use ir::function::Abi;
pub use regex_set::RegexSet;

use codegen::CodegenError;
use features::RustFeatures;
use ir::comment;
use ir::context::{BindgenContext, ItemId};
use ir::item::Item;
use options::BindgenOptions;
use parse::ParseError;

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

pub const DEFAULT_ANON_FIELDS_PREFIX: &str = "__bindgen_anon_";

const DEFAULT_NON_EXTERN_FNS_SUFFIX: &str = "__extern";

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

bitflags! {
    #[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct CodegenConfig: u32 {
        const FUNCTIONS = 1 << 0;
        const TYPES = 1 << 1;
        const VARS = 1 << 2;
        const METHODS = 1 << 3;
        const CONSTRUCTORS = 1 << 4;
        const DESTRUCTORS = 1 << 5;
    }
}

impl CodegenConfig {
    pub fn functions(self) -> bool {
        self.contains(CodegenConfig::FUNCTIONS)
    }

    pub fn types(self) -> bool {
        self.contains(CodegenConfig::TYPES)
    }

    pub fn vars(self) -> bool {
        self.contains(CodegenConfig::VARS)
    }

    pub fn methods(self) -> bool {
        self.contains(CodegenConfig::METHODS)
    }

    pub fn constructors(self) -> bool {
        self.contains(CodegenConfig::CONSTRUCTORS)
    }

    pub fn destructors(self) -> bool {
        self.contains(CodegenConfig::DESTRUCTORS)
    }
}

impl Default for CodegenConfig {
    fn default() -> Self {
        CodegenConfig::all()
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let y = match self {
            Self::None => "none",
            Self::Rustfmt => "rustfmt",
            Self::Prettyplease => "prettyplease",
        };
        y.fmt(f)
    }
}

#[derive(Debug, Default, Clone)]
pub struct Builder {
    options: BindgenOptions,
}

pub fn builder() -> Builder {
    Default::default()
}

fn get_extra_clang_args(xs: &[Rc<dyn callbacks::ParseCallbacks>]) -> Vec<String> {
    let ys = match get_target_dependent_env_var(xs, "BINDGEN_EXTRA_CLANG_ARGS") {
        None => return vec![],
        Some(x) => x,
    };
    if let Some(ys) = shlex::split(&ys) {
        return ys;
    }
    vec![ys]
}

impl Builder {
    pub fn generate(mut self) -> Result<Bindings, BindgenError> {
        self.opts
            .clang_args
            .extend(get_extra_clang_args(&self.opts.parse_callbacks));
        self.opts.clang_args.extend(
            self.opts.input_headers[..self.opts.input_headers.len().saturating_sub(1)]
                .iter()
                .flat_map(|x| ["-include".into(), x.to_string()]),
        );
        let ys = std::mem::take(&mut self.opts.input_header_contents)
            .into_iter()
            .map(|(name, contents)| clang::UnsavedFile::new(name, contents))
            .collect::<Vec<_>>();
        Bindings::generate(self.opts, ys)
    }

    pub fn dump_preprocessed_input(&self) -> io::Result<()> {
        let clang = clang::support::Clang::find(None, &[])
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
        let mut cmd = Command::new(clang.path);
        cmd.arg("-save-temps")
            .arg("-E")
            .arg("-C")
            .arg("-c")
            .arg(&p)
            .stdout(Stdio::piped());
        for x in &self.opts.clang_args {
            cmd.arg(x);
        }
        for x in get_extra_clang_args(&self.opts.parse_callbacks) {
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

impl BindgenOptions {
    fn build(&mut self) {
        const REGEX_SETS_LEN: usize = 27;

        let regex_sets: [_; REGEX_SETS_LEN] = [
            &mut self.allowlisted_vars,
            &mut self.allowlisted_types,
            &mut self.allowlisted_fns,
            &mut self.allowlisted_files,
            &mut self.blocklisted_types,
            &mut self.blocklisted_fns,
            &mut self.blocklisted_items,
            &mut self.blocklisted_files,
            &mut self.opaque_types,
            &mut self.bitfield_enums,
            &mut self.constified_enums,
            &mut self.constified_enum_modules,
            &mut self.newtype_enums,
            &mut self.newtype_global_enums,
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

        let record_matches = self.record_matches;
        //#[cfg(feature = "experimental")]
        {
            let n = REGEX_SETS_LEN + self.abi_overrides.len();
            let names = if self.emit_diagnostics {
                <[&str; REGEX_SETS_LEN]>::into_iter([
                    "--blocklist-type",
                    "--blocklist-function",
                    "--blocklist-item",
                    "--blocklist-file",
                    "--opaque-type",
                    "--allowlist-type",
                    "--allowlist-function",
                    "--allowlist-var",
                    "--allowlist-file",
                    "--bitfield-enum",
                    "--newtype-enum",
                    "--newtype-global-enum",
                    "--rustified-enum",
                    "--rustified-enum-non-exhaustive",
                    "--constified-enum-module",
                    "--constified-enum",
                    "--type-alias",
                    "--new-type-alias",
                    "--new-type-alias-deref",
                    "--bindgen-wrapper-union",
                    "--manually-drop-union",
                    "--no-partialeq",
                    "--no-copy",
                    "--no-debug",
                    "--no-default",
                    "--no-hash",
                    "--must-use",
                ])
                .chain((0..self.abi_overrides.len()).map(|_| "--override-abi"))
                .map(Some)
                .collect()
            } else {
                vec![None; n]
            };
            for (regex_set, name) in self.abi_overrides.values_mut().chain(regex_sets).zip(names) {
                regex_set.build_with_diagnostics(record_matches, name);
            }
        }
        #[cfg(not(feature = "experimental"))]
        for regex_set in self.abi_overrides.values_mut().chain(regex_sets) {
            regex_set.build(record_matches);
        }
        if !self.rust_features.untagged_union {
            self.untagged_union = false;
        }
    }

    pub fn set_rust_target(&mut self, rust_target: RustTarget) {
        self.rust_target = rust_target;

        self.rust_features = rust_target.into();
    }

    pub fn rust_features(&self) -> RustFeatures {
        self.rust_features
    }

    fn last_callback<T>(&self, f: impl Fn(&dyn callbacks::ParseCallbacks) -> Option<T>) -> Option<T> {
        self.parse_callbacks.iter().filter_map(|cb| f(cb.as_ref())).last()
    }

    fn all_callbacks<T>(&self, f: impl Fn(&dyn callbacks::ParseCallbacks) -> Vec<T>) -> Vec<T> {
        self.parse_callbacks.iter().flat_map(|cb| f(cb.as_ref())).collect()
    }

    fn process_comment(&self, comment: &str) -> String {
        let comment = comment::preprocess(comment);
        self.parse_callbacks
            .last()
            .and_then(|cb| cb.process_comment(&comment))
            .unwrap_or(comment)
    }
}

#[cfg(feature = "runtime")]
fn ensure_libclang_is_loaded() {
    if clang::is_loaded() {
        return;
    }

    lazy_static! {
        static ref LIBCLANG: std::sync::Arc<clang::SharedLibrary> = {
            clang::load().expect("Unable to find libclang");
            clang::get_library().expect(
                "We just loaded libclang and it had better still be \
                 here!",
            )
        };
    }

    clang::set_library(Some(LIBCLANG.clone()));
}

#[cfg(not(feature = "runtime"))]
fn ensure_libclang_is_loaded() {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BindgenError {
    FolderAsHeader(PathBuf),
    InsufficientPermissions(PathBuf),
    NotExist(PathBuf),
    ClangDiagnostic(String),
    Codegen(CodegenError),
}

impl std::fmt::Display for BindgenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BindgenError::FolderAsHeader(h) => {
                write!(f, "'{}' is a folder", h.display())
            },
            BindgenError::InsufficientPermissions(h) => {
                write!(f, "insufficient permissions to read '{}'", h.display())
            },
            BindgenError::NotExist(h) => {
                write!(f, "header '{}' does not exist.", h.display())
            },
            BindgenError::ClangDiagnostic(message) => {
                write!(f, "clang diagnosed error: {}", message)
            },
            BindgenError::Codegen(err) => {
                write!(f, "codegen error: {}", err)
            },
        }
    }
}

impl std::error::Error for BindgenError {}

#[derive(Debug)]
pub struct Bindings {
    options: BindgenOptions,
    module: proc_macro2::TokenStream,
}

pub(crate) const HOST_TARGET: &str = include_str!(concat!(env!("OUT_DIR"), "/host-target.txt"));

fn rust_to_clang_target(rust_target: &str) -> String {
    if rust_target.starts_with("aarch64-apple-") {
        let mut clang_target = "arm64-apple-".to_owned();
        clang_target.push_str(rust_target.strip_prefix("aarch64-apple-").unwrap());
        return clang_target;
    } else if rust_target.starts_with("riscv64gc-") {
        let mut clang_target = "riscv64-".to_owned();
        clang_target.push_str(rust_target.strip_prefix("riscv64gc-").unwrap());
        return clang_target;
    } else if rust_target.ends_with("-espidf") {
        let mut clang_target = rust_target.strip_suffix("-espidf").unwrap().to_owned();
        clang_target.push_str("-elf");
        if clang_target.starts_with("riscv32imc-") {
            clang_target = "riscv32-".to_owned() + clang_target.strip_prefix("riscv32imc-").unwrap();
        }
        return clang_target;
    }
    rust_target.to_owned()
}

fn find_effective_target(clang_args: &[String]) -> (String, bool) {
    let mut args = clang_args.iter();
    while let Some(opt) = args.next() {
        if opt.starts_with("--target=") {
            let mut split = opt.split('=');
            split.next();
            return (split.next().unwrap().to_owned(), true);
        }

        if opt == "-target" {
            if let Some(target) = args.next() {
                return (target.clone(), true);
            }
        }
    }

    if let Ok(t) = env::var("TARGET") {
        return (rust_to_clang_target(&t), false);
    }

    (rust_to_clang_target(HOST_TARGET), false)
}

impl Bindings {
    pub(crate) fn generate(
        mut options: BindgenOptions,
        input_unsaved_files: Vec<clang::UnsavedFile>,
    ) -> Result<Bindings, BindgenError> {
        ensure_libclang_is_loaded();

        #[cfg(feature = "runtime")]
        debug!(
            "Generating bindings, libclang at {}",
            clang::get_library().unwrap().path().display()
        );
        #[cfg(not(feature = "runtime"))]
        debug!("Generating bindings, libclang linked");

        options.build();

        let (effective_target, explicit_target) = find_effective_target(&options.clang_args);

        let is_host_build = rust_to_clang_target(HOST_TARGET) == effective_target;

        if !explicit_target && !is_host_build {
            options.clang_args.insert(0, format!("--target={}", effective_target));
        };

        fn detect_include_paths(options: &mut BindgenOptions) {
            if !options.detect_include_paths {
                return;
            }

            let clang_args_for_clang = {
                let mut last_was_include_prefix = false;
                options
                    .clang_args
                    .iter()
                    .filter(|arg| {
                        if last_was_include_prefix {
                            last_was_include_prefix = false;
                            return false;
                        }

                        let arg = &**arg;

                        if arg == "-I" || arg == "--include-directory" {
                            last_was_include_prefix = true;
                            return false;
                        }

                        if arg.starts_with("-I") || arg.starts_with("--include-directory=") {
                            return false;
                        }

                        true
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            };

            debug!("Trying to find clang with flags: {:?}", clang_args_for_clang);

            let clang = match clang::support::Clang::find(None, &clang_args_for_clang) {
                None => return,
                Some(clang) => clang,
            };

            debug!("Found clang: {:?}", clang);

            let is_cpp = args_are_cpp(&options.clang_args) || options.input_headers.iter().any(|h| file_is_cpp(h));

            let search_paths = if is_cpp {
                clang.cpp_search_paths
            } else {
                clang.c_search_paths
            };

            if let Some(search_paths) = search_paths {
                for path in search_paths.into_iter() {
                    if let Ok(path) = path.into_os_string().into_string() {
                        options.clang_args.push("-isystem".to_owned());
                        options.clang_args.push(path);
                    }
                }
            }
        }

        detect_include_paths(&mut options);

        #[cfg(unix)]
        fn can_read(perms: &std::fs::Permissions) -> bool {
            use std::os::unix::fs::PermissionsExt;
            perms.mode() & 0o444 > 0
        }

        #[cfg(not(unix))]
        fn can_read(_: &std::fs::Permissions) -> bool {
            true
        }

        if let Some(h) = options.input_headers.last() {
            let path = Path::new(h);
            if let Ok(md) = std::fs::metadata(path) {
                if md.is_dir() {
                    return Err(BindgenError::FolderAsHeader(path.into()));
                }
                if !can_read(&md.permissions()) {
                    return Err(BindgenError::InsufficientPermissions(path.into()));
                }
                let h = h.clone();
                options.clang_args.push(h);
            } else {
                return Err(BindgenError::NotExist(path.into()));
            }
        }

        for (idx, f) in input_unsaved_files.iter().enumerate() {
            if idx != 0 || !options.input_headers.is_empty() {
                options.clang_args.push("-include".to_owned());
            }
            options.clang_args.push(f.name.to_str().unwrap().to_owned())
        }

        debug!("Fixed-up options: {:?}", options);

        let time_phases = options.time_phases;
        let mut context = BindgenContext::new(options, &input_unsaved_files);

        if is_host_build {
            debug_assert_eq!(
                context.target_pointer_size(),
                std::mem::size_of::<*mut ()>(),
                "{:?} {:?}",
                effective_target,
                HOST_TARGET
            );
        }

        {
            let _t = time::Timer::new("parse").with_output(time_phases);
            parse(&mut context)?;
        }

        let (module, options) = codegen::codegen(context).map_err(BindgenError::Codegen)?;

        Ok(Bindings { options, module })
    }

    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(path.as_ref())?;
        self.write(Box::new(file))?;
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

        match self.format_tokens(&self.module) {
            Ok(formatted_bindings) => {
                writer.write_all(formatted_bindings.as_bytes())?;
            },
            Err(err) => {
                eprintln!("Failed to run rustfmt: {} (non-fatal, continuing)", err);
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

    fn format_tokens(&self, tokens: &proc_macro2::TokenStream) -> io::Result<String> {
        let _t = time::Timer::new("rustfmt_generated_string").with_output(self.opts.time_phases);

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

        if let Some(path) = self
            .options
            .rustfmt_configuration_file
            .as_ref()
            .and_then(|f| f.to_str())
        {
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
            Ok(bindings) => match status.code() {
                Some(0) => Ok(bindings),
                Some(2) => Err(io::Error::new(
                    io::ErrorKind::Other,
                    "Rustfmt parsing errors.".to_string(),
                )),
                Some(3) => {
                    rustfmt_non_fatal_error_diagnostic("Rustfmt could not format some lines", &self.opts);
                    Ok(bindings)
                },
                _ => Err(io::Error::new(
                    io::ErrorKind::Other,
                    "Internal rustfmt error".to_string(),
                )),
            },
            _ => Ok(source),
        }
    }
}

fn rustfmt_non_fatal_error_diagnostic(msg: &str, _options: &BindgenOptions) {
    warn!("{}", msg);

    #[cfg(feature = "experimental")]
    if _options.emit_diagnostics {
        use crate::diagnostics::{Diagnostic, Level};

        Diagnostic::default()
            .with_title(msg, Level::Warn)
            .add_annotation("The bindings will be generated but not formatted.", Level::Note)
            .display();
    }
}

impl std::fmt::Display for Bindings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut bytes = vec![];
        self.write(Box::new(&mut bytes) as Box<dyn Write>)
            .expect("writing to a vec cannot fail");
        f.write_str(std::str::from_utf8(&bytes).expect("we should only write bindings that are valid utf-8"))
    }
}

fn filter_builtins(ctx: &BindgenContext, cursor: &clang::Cursor) -> bool {
    ctx.options().builtins || !cursor.is_builtin()
}

fn parse_one(ctx: &mut BindgenContext, cursor: clang::Cursor, parent: Option<ItemId>) -> clang::CXChildVisitResult {
    if !filter_builtins(ctx, &cursor) {
        return CXChildVisit_Continue;
    }

    use clang::CXChildVisit_Continue;
    match Item::parse(cursor, parent, ctx) {
        Ok(..) => {},
        Err(ParseError::Continue) => {},
        Err(ParseError::Recurse) => {
            cursor.visit(|child| parse_one(ctx, child, parent));
        },
    }
    CXChildVisit_Continue
}

fn parse(context: &mut BindgenContext) -> Result<(), BindgenError> {
    use clang::*;

    let mut error = None;
    for d in context.translation_unit().diags().iter() {
        let msg = d.format();
        let is_err = d.severity() >= CXDiagnostic_Error;
        if is_err {
            let error = error.get_or_insert_with(String::new);
            error.push_str(&msg);
            error.push('\n');
        } else {
            eprintln!("clang diag: {}", msg);
        }
    }

    if let Some(message) = error {
        return Err(BindgenError::ClangDiagnostic(message));
    }

    let cursor = context.translation_unit().cursor();

    if context.options().emit_ast {
        fn dump_if_not_builtin(cur: &clang::Cursor) -> CXChildVisitResult {
            if !cur.is_builtin() {
                clang::ast_dump(cur, 0)
            } else {
                CXChildVisit_Continue
            }
        }
        cursor.visit(|cur| dump_if_not_builtin(&cur));
    }

    let root = context.root_module();
    context.with_module(root, |context| cursor.visit(|cursor| parse_one(context, cursor, None)));

    assert!(
        context.current_module() == context.root_module(),
        "How did this happen?"
    );
    Ok(())
}

#[derive(Debug)]
pub struct ClangVersion {
    pub parsed: Option<(u32, u32)>,
    pub full: String,
}

pub fn clang_version() -> ClangVersion {
    ensure_libclang_is_loaded();

    let raw_v: String = clang::extract_clang_version();
    let split_v: Option<Vec<&str>> = raw_v
        .split_whitespace()
        .find(|t| t.chars().next().map_or(false, |v| v.is_ascii_digit()))
        .map(|v| v.split('.').collect());
    if let Some(v) = split_v {
        if v.len() >= 2 {
            let maybe_major = v[0].parse::<u32>();
            let maybe_minor = v[1].parse::<u32>();
            if let (Ok(major), Ok(minor)) = (maybe_major, maybe_minor) {
                return ClangVersion {
                    parsed: Some((major, minor)),
                    full: raw_v.clone(),
                };
            }
        }
    };
    ClangVersion {
        parsed: None,
        full: raw_v.clone(),
    }
}

fn env_var<K: AsRef<str> + AsRef<OsStr>>(
    parse_callbacks: &[Rc<dyn callbacks::ParseCallbacks>],
    key: K,
) -> Result<String, std::env::VarError> {
    for callback in parse_callbacks {
        callback.read_env_var(key.as_ref());
    }
    std::env::var(key)
}

fn get_target_dependent_env_var(parse_callbacks: &[Rc<dyn callbacks::ParseCallbacks>], var: &str) -> Option<String> {
    if let Ok(target) = env_var(parse_callbacks, "TARGET") {
        if let Ok(v) = env_var(parse_callbacks, format!("{}_{}", var, target)) {
            return Some(v);
        }
        if let Ok(v) = env_var(parse_callbacks, format!("{}_{}", var, target.replace('-', "_"))) {
            return Some(v);
        }
    }

    env_var(parse_callbacks, var).ok()
}

#[derive(Debug)]
pub struct CargoCallbacks;

impl callbacks::ParseCallbacks for CargoCallbacks {
    fn include_file(&self, filename: &str) {
        println!("cargo:rerun-if-changed={}", filename);
    }

    fn read_env_var(&self, key: &str) {
        println!("cargo:rerun-if-env-changed={}", key);
    }
}

#[test]
fn commandline_flag_unit_test_function() {
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
        .allowlist_function("safe_function");

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

#[test]
fn test_rust_to_clang_target() {
    assert_eq!(rust_to_clang_target("aarch64-apple-ios"), "arm64-apple-ios");
}

#[test]
fn test_rust_to_clang_target_riscv() {
    assert_eq!(
        rust_to_clang_target("riscv64gc-unknown-linux-gnu"),
        "riscv64-unknown-linux-gnu"
    )
}

#[test]
fn test_rust_to_clang_target_espidf() {
    assert_eq!(rust_to_clang_target("riscv32imc-esp-espidf"), "riscv32-esp-elf");
    assert_eq!(rust_to_clang_target("xtensa-esp32-espidf"), "xtensa-esp32-elf");
}
