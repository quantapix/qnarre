use crate::callbacks::Parse;
use crate::codegen::utils::variation;
use crate::deps::DepfileSpec;
use crate::regex_set::RegexSet;
use crate::Abi;
use crate::Builder;
use crate::Config;
use crate::Formatter;
use crate::HashMap;
use crate::VisibilityKind;
use std::env;
use std::path::PathBuf;
use std::rc::Rc;

macro_rules! regex_opt {
    ($(#[$attrs:meta])* pub fn $($tokens:tt)*) => {
        $(#[$attrs])*
        pub fn $($tokens)*
    };
}
macro_rules! default {
    () => {
        Default::default()
    };
    ($expr:expr) => {
        $expr
    };
}

trait AsArgs {
    fn as_args(&self, xs: &mut Vec<String>, flag: &str);
}
impl AsArgs for bool {
    fn as_args(&self, xs: &mut Vec<String>, flag: &str) {
        if *self {
            xs.push(flag.to_string());
        }
    }
}
impl AsArgs for RegexSet {
    fn as_args(&self, xs: &mut Vec<String>, flag: &str) {
        for x in self.get_items() {
            xs.extend_from_slice(&[flag.to_owned(), x.clone()]);
        }
    }
}
impl AsArgs for Option<String> {
    fn as_args(&self, xs: &mut Vec<String>, flag: &str) {
        if let Some(x) = self {
            xs.extend_from_slice(&[flag.to_owned(), x.clone()]);
        }
    }
}
impl AsArgs for Option<PathBuf> {
    fn as_args(&self, xs: &mut Vec<String>, flag: &str) {
        if let Some(x) = self {
            xs.extend_from_slice(&[flag.to_owned(), x.display().to_string()]);
        }
    }
}

macro_rules! as_args {
    ($flag:literal) => {
        |x, xs| AsArgs::as_args(x, xs, $flag)
    };
    ($expr:expr) => {
        $expr
    };
}

fn ignore<T>(_: &T, _: &mut Vec<String>) {}

macro_rules! options {
    ($(
        $field:ident: $ty:ty {
            $(default: $default:expr,)?
            methods: {$($methods_tokens:tt)*}$(,)?
            as_args: $as_args:expr$(,)?
        }$(,)?
    )*) => {
        #[derive(Debug, Clone)]
        pub(crate) struct Opts {
            $(pub(crate) $field: $ty,)*
        }
        impl Default for Opts {
            fn default() -> Self {
                Self {
                    $($field: default!($($default)*),)*
                }
            }
        }
        impl Builder {
            pub fn command_line_flags(&self) -> Vec<String> {
                let mut ys = vec![];
                let headers = match self.opts.input_headers.split_last() {
                    Some((header, headers)) => {
                        ys.push(header.clone());
                        headers
                    },
                    None => &[]
                };
                $({
                    let func: fn(&$ty, &mut Vec<String>) = as_args!($as_args);
                    func(&self.opts.$field, &mut ys);
                })*
                ys.push("--".to_owned());
                if !self.opts.clang_args.is_empty() {
                    ys.extend_from_slice(&self.opts.clang_args);
                }
                for x in headers {
                    ys.push("-include".to_owned());
                    ys.push(x.clone());
                }
                ys
            }
            $($($methods_tokens)*)*
        }
    };
}

const DEFAULT_ANON_FIELDS_PREFIX: &str = "__bindgen_anon_";

options! {
    blocklisted_types: RegexSet {
        methods: {
            regex_opt! {
                pub fn blocklist_type<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.blocklisted_types.insert(x);
                    self
                }
            }
        },
        as_args: "--blocklist-type",
    },
    blocklisted_fns: RegexSet {
        methods: {
            regex_opt! {
                pub fn blocklist_fn<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.blocklisted_fns.insert(x);
                    self
                }
            }
        },
        as_args: "--blocklist-function",
    },
    blocklisted_items: RegexSet {
        methods: {
            regex_opt! {
                pub fn blocklist_item<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.blocklisted_items.insert(x);
                    self
                }
            }
        },
        as_args: "--blocklist-item",
    },
    blocklisted_files: RegexSet {
        methods: {
            regex_opt! {
                pub fn blocklist_file<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.blocklisted_files.insert(x);
                    self
                }
            }
        },
        as_args: "--blocklist-file",
    },
    opaque_types: RegexSet {
        methods: {
            regex_opt! {
                pub fn opaque_type<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.opaque_types.insert(x);
                    self
                }
            }
        },
        as_args: "--opaque-type",
    },
    rustfmt_path: Option<PathBuf> {
        methods: {
            pub fn with_rustfmt<P: Into<PathBuf>>(mut self, path: P) -> Self {
                self.opts.rustfmt_path = Some(path.into());
                self
            }
        },
        as_args: ignore,
    },
    depfile: Option<DepfileSpec> {
        methods: {
            pub fn depfile<H: Into<String>, D: Into<PathBuf>>(
                mut self,
                out_mod: H,
                depfile: D,
            ) -> Builder {
                self.opts.depfile = Some(DepfileSpec {
                    out_mod: out_mod.into(),
                    dep_path: depfile.into(),
                });
                self
            }
        },
        as_args: |x, xs| {
            if let Some(x) = x {
                xs.push("--depfile".into());
                xs.push(x.dep_path.display().to_string());
            }
        },
    },
    allowed_types: RegexSet {
        methods: {
            regex_opt! {
                pub fn allowlist_type<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.allowed_types.insert(x);
                    self
                }
            }
        },
        as_args: "--allowlist-type",
    },
    allowed_fns: RegexSet {
        methods: {
            regex_opt! {
                pub fn allowlist_fn<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.allowed_fns.insert(x);
                    self
                }
            }
        },
        as_args: "--allowlist-fn",
    },
    allowed_vars: RegexSet {
        methods: {
            regex_opt! {
                pub fn allowlist_var<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.allowed_vars.insert(x);
                    self
                }
            }
        },
        as_args: "--allowlist-var",
    },
    allowed_files: RegexSet {
        methods: {
            regex_opt! {
                pub fn allowlist_file<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.allowed_files.insert(x);
                    self
                }
            }
        },
        as_args: "--allowlist-file",
    },
    default_enum_style: variation::Enum {
        methods: {
            pub fn default_enum_style(
                mut self,
                arg: variation::Enum,
            ) -> Builder {
                self.opts.default_enum_style = arg;
                self
            }
        },
        as_args: |x, xs| {
            if *x != Default::default() {
                xs.push("--default-enum-style".to_owned());
                xs.push(x.to_string());
            }
        },
    },
    rustified_enums: RegexSet {
        methods: {
            regex_opt! {
                pub fn rustified_enum<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.rustified_enums.insert(x);
                    self
                }
            }
        },
        as_args: "--rustified-enum",
    },
    rustified_non_exhaustive_enums: RegexSet {
        methods: {
            regex_opt! {
                pub fn rustified_non_exhaustive_enum<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.rustified_non_exhaustive_enums.insert(x);
                    self
                }
            }
        },
        as_args: "--rustified-non-exhaustive-enums",
    },
    constified_enum_mods: RegexSet {
        methods: {
            regex_opt! {
                pub fn constified_enum_mod<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.constified_enum_mods.insert(x);
                    self
                }
            }
        },
        as_args: "--constified-enum-mod",
    },
    constified_enums: RegexSet {
        methods: {
            regex_opt! {
                pub fn constified_enum<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.constified_enums.insert(x);
                    self
                }
            }
        },
        as_args: "--constified-enum",
    },
    default_macro_const: variation::MacroType {
        methods: {
            pub fn default_macro_const(mut self, x: variation::MacroType) -> Builder {
                self.opts.default_macro_const = x;
                self
            }
        },
        as_args: |x, xs| {
            if *x != Default::default() {
                xs.push("--default-macro-const-type".to_owned());
                xs.push(x.to_string());
            }
        },
    },
    default_alias_style: variation::Alias {
        methods: {
            pub fn default_alias_style(
                mut self,
                x: variation::Alias,
            ) -> Builder {
                self.opts.default_alias_style = x;
                self
            }
        },
        as_args: |x, xs| {
            if *x != Default::default() {
                xs.push("--default-alias-style".to_owned());
                xs.push(x.to_string());
            }
        },
    },
    type_alias: RegexSet {
        methods: {
            regex_opt! {
                pub fn type_alias<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.type_alias.insert(x);
                    self
                }
            }
        },
        as_args: "--type-alias",
    },
    new_type_alias: RegexSet {
        methods: {
            regex_opt! {
                pub fn new_type_alias<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.new_type_alias.insert(x);
                    self
                }
            }
        },
        as_args: "--new-type-alias",
    },
    new_type_alias_deref: RegexSet {
        methods: {
            regex_opt! {
                pub fn new_type_alias_deref<T: AsRef<str>>(mut self, x: T) -> Builder {
                    self.opts.new_type_alias_deref.insert(x);
                    self
                }
            }
        },
        as_args: "--new-type-alias-deref",
    },
    default_non_copy_union_style: variation::NonCopyUnion {
        methods: {
            pub fn default_non_copy_union_style(mut self, x: variation::NonCopyUnion) -> Self {
                self.opts.default_non_copy_union_style = x;
                self
            }
        },
        as_args: |x, xs| {
            if *x != Default::default() {
                xs.push("--default-non-copy-union-style".to_owned());
                xs.push(x.to_string());
            }
        },
    },
    bindgen_wrapper_union: RegexSet {
        methods: {
            regex_opt! {
                pub fn bindgen_wrapper_union<T: AsRef<str>>(mut self, x: T) -> Self {
                    self.opts.bindgen_wrapper_union.insert(x);
                    self
                }
            }
        },
        as_args: "--bindgen-wrapper-union",
    },
    manually_drop_union: RegexSet {
        methods: {
            regex_opt! {
                pub fn manually_drop_union<T: AsRef<str>>(mut self, x: T) -> Self {
                    self.opts.manually_drop_union.insert(x);
                    self
                }
            }
        },
        as_args: "--manually-drop-union",
    },
    builtins: bool {
        methods: {
            pub fn emit_builtins(mut self) -> Builder {
                self.opts.builtins = true;
                self
            }
        },
        as_args: "--builtins",
    },
    emit_ast: bool {
        methods: {
            pub fn emit_clang_ast(mut self) -> Builder {
                self.opts.emit_ast = true;
                self
            }
        },
        as_args: "--emit-clang-ast",
    },
    emit_ir: bool {
        methods: {
            pub fn emit_ir(mut self) -> Builder {
                self.opts.emit_ir = true;
                self
            }
        },
        as_args: "--emit-ir",
    },
    emit_ir_graphviz: Option<String> {
        methods: {
            pub fn emit_ir_graphviz<T: Into<String>>(mut self, path: T) -> Builder {
                let path = path.into();
                self.opts.emit_ir_graphviz = Some(path);
                self
            }
        },
        as_args: "--emit-ir-graphviz",
    },
    enable_cxx_namespaces: bool {
        methods: {
            pub fn enable_cxx_namespaces(mut self) -> Builder {
                self.opts.enable_cxx_namespaces = true;
                self
            }
        },
        as_args: "--enable-cxx-namespaces",
    },
    enable_fn_attr_detection: bool {
        methods: {
            pub fn enable_fn_attr_detection(mut self) -> Self {
                self.opts.enable_fn_attr_detection = true;
                self
            }
        },
        as_args: "--enable-fn-attr-detection",
    },
    disable_name_namespacing: bool {
        methods: {
            pub fn disable_name_namespacing(mut self) -> Builder {
                self.opts.disable_name_namespacing = true;
                self
            }
        },
        as_args: "--disable-name-namespacing",
    },
    disable_nested_struct_naming: bool {
        methods: {
            pub fn disable_nested_struct_naming(mut self) -> Builder {
                self.opts.disable_nested_struct_naming = true;
                self
            }
        },
        as_args: "--disable-nested-struct-naming",
    },
    disable_header_comment: bool {
        methods: {
            pub fn disable_header_comment(mut self) -> Self {
                self.opts.disable_header_comment = true;
                self
            }
        },
        as_args: "--disable-header-comment",
    },
    layout_tests: bool {
        default: true,
        methods: {
            pub fn layout_tests(mut self, x: bool) -> Self {
                self.opts.layout_tests = x;
                self
            }
        },
        as_args: |x, xs| (!x).as_args(xs, "--no-layout-tests"),
    },
    impl_debug: bool {
        methods: {
            pub fn impl_debug(mut self, x: bool) -> Self {
                self.opts.impl_debug = x;
                self
            }
        },
        as_args: "--impl-debug",
    },
    impl_partialeq: bool {
        methods: {
            pub fn impl_partialeq(mut self, x: bool) -> Self {
                self.opts.impl_partialeq = x;
                self
            }
        },
        as_args: "--impl-partialeq",
    },
    derive_copy: bool {
        default: true,
        methods: {
            pub fn derive_copy(mut self, x: bool) -> Self {
                self.opts.derive_copy = x;
                self
            }
        },
        as_args: |x, xs| (!x).as_args(xs, "--no-derive-copy"),
    },
    derive_debug: bool {
        default: true,
        methods: {
            pub fn derive_debug(mut self, x: bool) -> Self {
                self.opts.derive_debug = x;
                self
            }
        },
        as_args: |x, xs| (!x).as_args(xs, "--no-derive-debug"),
    },
    derive_default: bool {
        methods: {
            pub fn derive_default(mut self, x: bool) -> Self {
                self.opts.derive_default = x;
                self
            }
        },
        as_args: |&x, xs| {
            let x = if x {
                "--with-derive-default"
            } else {
                "--no-derive-default"
            };
            xs.push(x.to_owned());
        },
    },
    derive_hash: bool {
        methods: {
            pub fn derive_hash(mut self, x: bool) -> Self {
                self.opts.derive_hash = x;
                self
            }
        },
        as_args: "--with-derive-hash",
    },
    derive_partialord: bool {
        methods: {
            pub fn derive_partialord(mut self, x: bool) -> Self {
                self.opts.derive_partialord = x;
                if !x {
                    self.opts.derive_ord = false;
                }
                self
            }
        },
        as_args: "--with-derive-partialord",
    },
    derive_ord: bool {
        methods: {
            pub fn derive_ord(mut self, x: bool) -> Self {
                self.opts.derive_ord = x;
                self.opts.derive_partialord = x;
                self
            }
        },
        as_args: "--with-derive-ord",
    },
    derive_partialeq: bool {
        methods: {
            pub fn derive_partialeq(mut self, x: bool) -> Self {
                self.opts.derive_partialeq = x;
                if !x {
                    self.opts.derive_eq = false;
                }
                self
            }
        },
        as_args: "--with-derive-partialeq",
    },
    derive_eq: bool {
        methods: {
            pub fn derive_eq(mut self, x: bool) -> Self {
                self.opts.derive_eq = x;
                if x {
                    self.opts.derive_partialeq = x;
                }
                self
            }
        },
        as_args: "--with-derive-eq",
    },
    use_core: bool {
        methods: {
            pub fn use_core(mut self) -> Builder {
                self.opts.use_core = true;
                self
            }
        },
        as_args: "--use-core",
    },
    ctypes_prefix: Option<String> {
        methods: {
            pub fn ctypes_prefix<T: Into<String>>(mut self, x: T) -> Builder {
                self.opts.ctypes_prefix = Some(x.into());
                self
            }
        },
        as_args: "--ctypes-prefix",
    },
    anon_fields_prefix: String {
        default: DEFAULT_ANON_FIELDS_PREFIX.into(),
        methods: {
            pub fn anon_fields_prefix<T: Into<String>>(mut self, x: T) -> Builder {
                self.opts.anon_fields_prefix = x.into();
                self
            }
        },
        as_args: |x, xs| {
            if x != DEFAULT_ANON_FIELDS_PREFIX {
                xs.push("--anon-fields-prefix".to_owned());
                xs.push(x.clone());
            }
        },
    },
    time_phases: bool {
        methods: {
            pub fn time_phases(mut self, x: bool) -> Self {
                self.opts.time_phases = x;
                self
            }
        },
        as_args: "--time-phases",
    },
    convert_floats: bool {
        default: true,
        methods: {
            pub fn no_convert_floats(mut self) -> Self {
                self.opts.convert_floats = false;
                self
            }
        },
        as_args: |x, xs| (!x).as_args(xs, "--no-convert-floats"),
    },
    raw_lines: Vec<String> {
        methods: {
            pub fn raw_line<T: Into<String>>(mut self, x: T) -> Self {
                self.opts.raw_lines.push(x.into());
                self
            }
        },
        as_args: |lines, xs| {
            for x in lines {
                xs.push("--raw-line".to_owned());
                xs.push(x.clone());
            }
        },
    },
    module_lines: HashMap<String, Vec<String>> {
        methods: {
            pub fn module_raw_line<T, U>(mut self, module: T, line: U) -> Self
            where
                T: Into<String>,
                U: Into<String>,
            {
                self.opts
                    .module_lines
                    .entry(module.into())
                    .or_insert_with(Vec::new)
                    .push(line.into());
                self
            }
        },
        as_args: |lines, xs| {
            for (module, lines) in lines {
                for x in lines.iter() {
                    xs.push("--module-raw-line".to_owned());
                    xs.push(module.clone());
                    xs.push(x.clone());
                }
            }
        },
    },
    input_headers:  Vec<String> {
        methods: {
            pub fn header<T: Into<String>>(mut self, x: T) -> Builder {
                self.opts.input_headers.push(x.into());
                self
            }
        },
        as_args: ignore,
    },
    clang_args: Vec<String> {
        methods: {
            pub fn clang_arg<T: Into<String>>(self, x: T) -> Builder {
                self.clang_args([x.into()])
            }
            pub fn clang_args<I: IntoIterator>(mut self, xs: I) -> Builder
            where
                I::Item: AsRef<str>,
            {
                for x in xs {
                    self.opts.clang_args.push(x.as_ref().to_owned());
                }
                self
            }
        },
        as_args: ignore,
    },
    input_header_contents: Vec<(String, String)> {
        methods: {
            pub fn header_contents(mut self, name: &str, contents: &str) -> Builder {
                let absolute_path = env::current_dir()
                    .expect("Cannot retrieve current directory")
                    .join(name)
                    .to_str()
                    .expect("Cannot convert current directory name to string")
                    .to_owned();
                self.opts
                    .input_header_contents
                    .push((absolute_path, contents.into()));
                self
            }
        },
        as_args: ignore,
    },
    parse_callbacks: Vec<Rc<dyn Parse>> {
        methods: {
            pub fn parse_callbacks(mut self, x: Box<dyn Parse>) -> Self {
                self.opts.parse_callbacks.push(Rc::from(x));
                self
            }
        },
        as_args: |_, _| {
        },
    },
    config: Config {
        default: Config::all(),
        methods: {
            pub fn ignore_fns(mut self) -> Builder {
                self.opts.config.remove(Config::FNS);
                self
            }
            pub fn ignore_methods(mut self) -> Builder {
                self.opts.config.remove(Config::METHODS);
                self
            }
            pub fn with_config(mut self, x: Config) -> Self {
                self.opts.config = x;
                self
            }
        },
        as_args: |x, xs| {
            if !x.fns() {
                xs.push("--ignore-functions".to_owned());
            }
            xs.push("--generate".to_owned());
            let mut ys: Vec<String> = Vec::new();
            if x.fns() {
                ys.push("functions".to_owned());
            }
            if x.typs() {
                ys.push("types".to_owned());
            }
            if x.vars() {
                ys.push("vars".to_owned());
            }
            if x.methods() {
                ys.push("methods".to_owned());
            }
            if x.constrs() {
                ys.push("constructors".to_owned());
            }
            if x.destrs() {
                ys.push("destructors".to_owned());
            }
            xs.push(ys.join(","));
            if !x.methods() {
                xs.push("--ignore-methods".to_owned());
            }
        },
    },
    conservative_inline_namespaces: bool {
        methods: {
            pub fn conservative_inline_namespaces(mut self) -> Builder {
                self.opts.conservative_inline_namespaces = true;
                self
            }
        },
        as_args: "--conservative-inline-namespaces",
    },
    generate_comments: bool {
        default: true,
        methods: {
            pub fn generate_comments(mut self, x: bool) -> Self {
                self.opts.generate_comments = x;
                self
            }
        },
        as_args: |x, xs| (!x).as_args(xs, "--no-doc-comments"),
    },
    allowlist_recursively: bool {
        default: true,
        methods: {
            pub fn allowlist_recursively(mut self, x: bool) -> Self {
                self.opts.allowlist_recursively = x;
                self
            }
        },
        as_args: |x, xs| (!x).as_args(xs, "--no-recursive-allowlist"),
    },
    generate_block: bool {
        methods: {
            pub fn generate_block(mut self, x: bool) -> Self {
                self.opts.generate_block = x;
                self
            }
        },
        as_args: "--generate-block",
    },
    generate_cstr: bool {
        methods: {
            pub fn generate_cstr(mut self, x: bool) -> Self {
                self.opts.generate_cstr = x;
                self
            }
        },
        as_args: "--generate-cstr",
    },
    block_extern_crate: bool {
        methods: {
            pub fn block_extern_crate(mut self, x: bool) -> Self {
                self.opts.block_extern_crate = x;
                self
            }
        },
        as_args: "--block-extern-crate",
    },
    enable_mangling: bool {
        default: true,
        methods: {
            pub fn trust_clang_mangling(mut self, x: bool) -> Self {
                self.opts.enable_mangling = x;
                self
            }
        },
        as_args: |x, xs| (!x).as_args(xs, "--distrust-clang-mangling"),
    },
    detect_include_paths: bool {
        default: true,
        methods: {
            pub fn detect_include_paths(mut self, x: bool) -> Self {
                self.opts.detect_include_paths = x;
                self
            }
        },
        as_args: |x, xs| (!x).as_args(xs, "--no-include-path-detection"),
    },
    fit_macro_const: bool {
        methods: {
            pub fn fit_macro_const(mut self, x: bool) -> Self {
                self.opts.fit_macro_const = x;
                self
            }
        },
        as_args: "--fit-macro-constant-types",
    },
    prepend_enum_name: bool {
        default: true,
        methods: {
            pub fn prepend_enum_name(mut self, x: bool) -> Self {
                self.opts.prepend_enum_name = x;
                self
            }
        },
        as_args: |x, xs| (!x).as_args(xs, "--no-prepend-enum-name"),
    },
    untagged_union: bool {
        default: true,
        methods: {
            pub fn disable_untagged_union(mut self) -> Self {
                self.opts.untagged_union = false;
                self
            }
        }
        as_args: |x, xs| (!x).as_args(xs, "--disable-untagged-union"),
    },
    record_matches: bool {
        default: true,
        methods: {
            pub fn record_matches(mut self, x: bool) -> Self {
                self.opts.record_matches = x;
                self
            }
        },
        as_args: |x, xs| (!x).as_args(xs, "--no-record-matches"),
    },
    size_t_is_usize: bool {
        default: true,
        methods: {
            pub fn size_t_is_usize(mut self, x: bool) -> Self {
                self.opts.size_t_is_usize = x;
                self
            }
        },
        as_args: |x, xs| (!x).as_args(xs, "--no-size_t-is-usize"),
    },
    formatter: Formatter {
        methods: {
            #[deprecated]
            pub fn rustfmt_bindings(mut self, x: bool) -> Self {
                self.opts.formatter = if x {
                    Formatter::Rustfmt
                } else {
                    Formatter::None
                };
                self
            }
            pub fn formatter(mut self, x: Formatter) -> Self {
                self.opts.formatter = x;
                self
            }
        },
        as_args: |x, xs| {
            if *x != Default::default() {
                xs.push("--formatter".to_owned());
                xs.push(x.to_string());
            }
        },
    },
    rustfmt_configuration_file: Option<PathBuf> {
        methods: {
            pub fn rustfmt_configuration_file(mut self, x: Option<PathBuf>) -> Self {
                self = self.formatter(Formatter::Rustfmt);
                self.opts.rustfmt_configuration_file = x;
                self
            }
        },
        as_args: "--rustfmt-configuration-file",
    },
    no_partialeq_types: RegexSet {
        methods: {
            regex_opt! {
                pub fn no_partialeq<T: Into<String>>(mut self, x: T) -> Builder {
                    self.opts.no_partialeq_types.insert(x.into());
                    self
                }
            }
        },
        as_args: "--no-partialeq",
    },
    no_copy_types: RegexSet {
        methods: {
            regex_opt! {
                pub fn no_copy<T: Into<String>>(mut self, x: T) -> Self {
                    self.opts.no_copy_types.insert(x.into());
                    self
                }
            }
        },
        as_args: "--no-copy",
    },
    no_debug_types: RegexSet {
        methods: {
            regex_opt! {
                pub fn no_debug<T: Into<String>>(mut self, x: T) -> Self {
                    self.opts.no_debug_types.insert(x.into());
                    self
                }
            }
        },
        as_args: "--no-debug",
    },
    no_default_types: RegexSet {
        methods: {
            regex_opt! {
                pub fn no_default<T: Into<String>>(mut self, x: T) -> Self {
                    self.opts.no_default_types.insert(x.into());
                    self
                }
            }
        },
        as_args: "--no-default",
    },
    no_hash_types: RegexSet {
        methods: {
            regex_opt! {
                pub fn no_hash<T: Into<String>>(mut self, x: T) -> Builder {
                    self.opts.no_hash_types.insert(x.into());
                    self
                }
            }
        },
        as_args: "--no-hash",
    },
    must_use_types: RegexSet {
        methods: {
            regex_opt! {
                pub fn must_use_type<T: Into<String>>(mut self, x: T) -> Builder {
                    self.opts.must_use_types.insert(x.into());
                    self
                }
            }
        },
        as_args: "--must-use-type",
    },
    array_pointers_in_arguments: bool {
        methods: {
            pub fn array_pointers_in_arguments(mut self, x: bool) -> Self {
                self.opts.array_pointers_in_arguments = x;
                self
            }
        },
        as_args: "--use-array-pointers-in-arguments",
    },
    wasm_import_mod_name: Option<String> {
        methods: {
            pub fn wasm_import_mod_name<T: Into<String>>(
                mut self,
                x: T,
            ) -> Self {
                self.opts.wasm_import_mod_name = Some(x.into());
                self
            }
        },
        as_args: "--wasm-import-mod-name",
    },
    dynamic_library_name: Option<String> {
        methods: {
            pub fn dynamic_library_name<T: Into<String>>(
                mut self,
                x: T,
            ) -> Self {
                self.opts.dynamic_library_name = Some(x.into());
                self
            }
        },
        as_args: "--dynamic-loading",
    },
    dynamic_link_require_all: bool {
        methods: {
            pub fn dynamic_link_require_all(mut self, x: bool) -> Self {
                self.opts.dynamic_link_require_all = x;
                self
            }
        },
        as_args: "--dynamic-link-require-all",
    },
    respect_cxx_access_specs: bool {
        methods: {
            pub fn respect_cxx_access_specs(mut self, x: bool) -> Self {
                self.opts.respect_cxx_access_specs = x;
                self
            }
        },
        as_args: "--respect-cxx-access-specs",
    },
    translate_enum_integer_types: bool {
        methods: {
            pub fn translate_enum_integer_types(mut self, x: bool) -> Self {
                self.opts.translate_enum_integer_types = x;
                self
            }
        },
        as_args: "--translate-enum-integer-types",
    },
    c_naming: bool {
        methods: {
            pub fn c_naming(mut self, x: bool) -> Self {
                self.opts.c_naming = x;
                self
            }
        },
        as_args: "--c-naming",
    },
    force_explicit_padding: bool {
        methods: {
            pub fn explicit_padding(mut self, x: bool) -> Self {
                self.opts.force_explicit_padding = x;
                self
            }
        },
        as_args: "--explicit-padding",
    },
    vtable_generation: bool {
        methods: {
            pub fn vtable_generation(mut self, x: bool) -> Self {
                self.opts.vtable_generation = x;
                self
            }
        },
        as_args: "--vtable-generation",
    },
    sort_semantically: bool {
        methods: {
            pub fn sort_semantically(mut self, x: bool) -> Self {
                self.opts.sort_semantically = x;
                self
            }
        },
        as_args: "--sort-semantically",
    },
    merge_extern_blocks: bool {
        methods: {
            pub fn merge_extern_blocks(mut self, x: bool) -> Self {
                self.opts.merge_extern_blocks = x;
                self
            }
        },
        as_args: "--merge-extern-blocks",
    },
    wrap_unsafe_ops: bool {
        methods: {
            pub fn wrap_unsafe_ops(mut self, x: bool) -> Self {
                self.opts.wrap_unsafe_ops = x;
                self
            }
        },
        as_args: "--wrap-unsafe-ops",
    },
    abi_overrides: HashMap<Abi, RegexSet> {
        methods: {
            regex_opt! {
                pub fn override_abi<T: Into<String>>(mut self, abi: Abi, x: T) -> Self {
                    self.opts
                        .abi_overrides
                        .entry(abi)
                        .or_default()
                        .insert(x.into());
                    self
                }
            }
        },
        as_args: |overrides, xs| {
            for (abi, set) in overrides {
                for x in set.get_items() {
                    xs.push("--override-abi".to_owned());
                    xs.push(format!("{}={}", x, abi));
                }
            }
        },
    },
    default_visibility: VisibilityKind {
        methods: {
            pub fn default_visibility(
                mut self,
                x: VisibilityKind,
            ) -> Self {
                self.opts.default_visibility = x;
                self
            }
        },
        as_args: |x, xs| {
            if *x != Default::default() {
                xs.push("--default-visibility".to_owned());
                xs.push(x.to_string());
            }
        },
    },
}
