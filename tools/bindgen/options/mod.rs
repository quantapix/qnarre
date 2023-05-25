#[macro_use]
mod helpers;
mod as_args;

use crate::callbacks::ParseCallbacks;
use crate::codegen::{AliasVariation, EnumVariation, MacroTypeVariation, NonCopyUnionStyle};
use crate::deps::DepfileSpec;
use crate::features::{RustFeatures, RustTarget};
use crate::regex_set::RegexSet;
use crate::Abi;
use crate::Builder;
use crate::CodegenConfig;
use crate::FieldVisibilityKind;
use crate::Formatter;
use crate::HashMap;
use crate::DEFAULT_ANON_FIELDS_PREFIX;

use std::env;
#[cfg(feature = "experimental")]
use std::path::Path;
use std::path::PathBuf;
use std::rc::Rc;

use as_args::AsArgs;
use helpers::ignore;

macro_rules! options {
    ($(
        $(#[doc = $docs:literal])+
        $field:ident: $ty:ty {
            $(default: $default:expr,)?
            methods: {$($methods_tokens:tt)*}$(,)?
            as_args: $as_args:expr$(,)?
        }$(,)?
    )*) => {
        #[derive(Debug, Clone)]
        pub(crate) struct BindgenOptions {
            $($(#[doc = $docs])* pub(crate) $field: $ty,)*
        }

        impl Default for BindgenOptions {
            fn default() -> Self {
                Self {
                    $($field: default!($($default)*),)*
                }
            }
        }

        impl Builder {
            pub fn command_line_flags(&self) -> Vec<String> {
                let mut args = vec![];

                let headers = match self.options.input_headers.split_last() {
                    Some((header, headers)) => {
                        // The last input header is passed as an argument in the first position.
                        args.push(header.clone());
                        headers
                    },
                    None => &[]
                };

                $({
                    let func: fn(&$ty, &mut Vec<String>) = as_args!($as_args);
                    func(&self.options.$field, &mut args);
                })*

                // Add the `--experimental` flag if `bindgen` is built with the `experimental`
                // feature.
                if cfg!(feature = "experimental") {
                    args.push("--experimental".to_owned());
                }

                // Add all the clang arguments.
                args.push("--".to_owned());

                if !self.options.clang_args.is_empty() {
                    args.extend_from_slice(&self.options.clang_args);
                }

                // We need to pass all but the last header via the `-include` clang argument.
                for header in headers {
                    args.push("-include".to_owned());
                    args.push(header.clone());
                }

                args
            }

            $($($methods_tokens)*)*
        }
    };
}

options! {
    blocklisted_types: RegexSet {
        methods: {
            regex_option! {
                /// Do not generate any bindings for the given type.
                pub fn blocklist_type<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.blocklisted_types.insert(arg);
                    self
                }
            }
        },
        as_args: "--blocklist-type",
    },
    blocklisted_functions: RegexSet {
        methods: {
            regex_option! {
                /// Do not generate any bindings for the given function.
                pub fn blocklist_function<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.blocklisted_functions.insert(arg);
                    self
                }
            }
        },
        as_args: "--blocklist-function",
    },
    blocklisted_items: RegexSet {
        methods: {
            regex_option! {
                /// Do not generate any bindings for the given item, regardless of whether it is a
                /// type, function, module, etc.
                pub fn blocklist_item<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.blocklisted_items.insert(arg);
                    self
                }
            }
        },
        as_args: "--blocklist-item",
    },
    blocklisted_files: RegexSet {
        methods: {
            regex_option! {
                /// Do not generate any bindings for the contents of the given file, regardless of
                /// whether the contents of the file are types, functions, modules, etc.
                pub fn blocklist_file<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.blocklisted_files.insert(arg);
                    self
                }
            }
        },
        as_args: "--blocklist-file",
    },
    opaque_types: RegexSet {
        methods: {
            regex_option! {
                /// Treat the given type as opaque in the generated bindings.
                ///
                /// Opaque in this context means that none of the generated bindings will contain
                /// information about the inner representation of the type and the type itself will
                /// be represented as a chunk of bytes with the alignment and size of the type.
                pub fn opaque_type<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.opaque_types.insert(arg);
                    self
                }
            }
        },
        as_args: "--opaque-type",
    },
    rustfmt_path: Option<PathBuf> {
        methods: {
            pub fn with_rustfmt<P: Into<PathBuf>>(mut self, path: P) -> Self {
                self.options.rustfmt_path = Some(path.into());
                self
            }
        },
        as_args: ignore,
    },
    depfile: Option<DepfileSpec> {
        methods: {
            pub fn depfile<H: Into<String>, D: Into<PathBuf>>(
                mut self,
                output_module: H,
                depfile: D,
            ) -> Builder {
                self.options.depfile = Some(DepfileSpec {
                    output_module: output_module.into(),
                    depfile_path: depfile.into(),
                });
                self
            }
        },
        as_args: |depfile, args| {
            if let Some(depfile) = depfile {
                args.push("--depfile".into());
                args.push(depfile.depfile_path.display().to_string());
            }
        },
    },
    allowlisted_types: RegexSet {
        methods: {
            regex_option! {
                /// Generate bindings for the given type.
                ///
                /// This option is transitive by default. Check the documentation of the
                /// [`Builder::allowlist_recursively`] method for further information.
                pub fn allowlist_type<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.allowlisted_types.insert(arg);
                    self
                }
            }
        },
        as_args: "--allowlist-type",
    },
    allowlisted_functions: RegexSet {
        methods: {
            regex_option! {
                /// Generate bindings for the given function.
                ///
                /// This option is transitive by default. Check the documentation of the
                /// [`Builder::allowlist_recursively`] method for further information.
                pub fn allowlist_function<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.allowlisted_functions.insert(arg);
                    self
                }
            }
        },
        as_args: "--allowlist-function",
    },
    allowlisted_vars: RegexSet {
        methods: {
            regex_option! {
                /// Generate bindings for the given variable.
                ///
                /// This option is transitive by default. Check the documentation of the
                /// [`Builder::allowlist_recursively`] method for further information.
                pub fn allowlist_var<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.allowlisted_vars.insert(arg);
                    self
                }
            }
        },
        as_args: "--allowlist-var",
    },
    allowlisted_files: RegexSet {
        methods: {
            regex_option! {
                /// Generate bindings for the content of the given file.
                ///
                /// This option is transitive by default. Check the documentation of the
                /// [`Builder::allowlist_recursively`] method for further information.
                pub fn allowlist_file<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.allowlisted_files.insert(arg);
                    self
                }
            }
        },
        as_args: "--allowlist-file",
    },
    default_enum_style: EnumVariation {
        methods: {
            pub fn default_enum_style(
                mut self,
                arg: EnumVariation,
            ) -> Builder {
                self.options.default_enum_style = arg;
                self
            }
        },
        as_args: |variation, args| {
            if *variation != Default::default() {
                args.push("--default-enum-style".to_owned());
                args.push(variation.to_string());
            }
        },
    },
    bitfield_enums: RegexSet {
        methods: {
            regex_option! {
                /// Mark the given `enum` as being bitfield-like.
                ///
                /// This is similar to the [`Builder::newtype_enum`] style, but with the bitwise
                /// operators implemented.
                pub fn bitfield_enum<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.bitfield_enums.insert(arg);
                    self
                }
            }
        },
        as_args: "--bitfield-enum",
    },
    newtype_enums: RegexSet {
        methods: {
            regex_option! {
                /// Mark the given `enum` as a newtype.
                ///
                /// This means that an integer newtype will be declared to represent the `enum`
                /// type and its variants will be represented as constants inside of this type's
                /// `impl` block.
                pub fn newtype_enum<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.newtype_enums.insert(arg);
                    self
                }
            }
        },
        as_args: "--newtype-enum",
    },
    newtype_global_enums: RegexSet {
        methods: {
            regex_option! {
                /// Mark the given `enum` as a global newtype.
                ///
                /// This is similar to the [`Builder::newtype_enum`] style, but the constants for
                /// each variant are free constants instead of being declared inside an `impl`
                /// block for the newtype.
                pub fn newtype_global_enum<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.newtype_global_enums.insert(arg);
                    self
                }
            }
        },
        as_args: "--newtype-global-enum",
    },
    rustified_enums: RegexSet {
        methods: {
            regex_option! {
                /// Mark the given `enum` as a Rust `enum`.
                ///
                /// This means that each variant of the `enum` will be represented as a Rust `enum`
                /// variant.
                ///
                /// **Use this with caution**, creating an instance of a Rust `enum` with an
                /// invalid value will cause undefined behaviour. To avoid this, use the
                /// [`Builder::newtype_enum`] style instead.
                pub fn rustified_enum<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.rustified_enums.insert(arg);
                    self
                }
            }
        },
        as_args: "--rustified-enum",
    },
    rustified_non_exhaustive_enums: RegexSet {
        methods: {
            regex_option! {
                /// Mark the given `enum` as a non-exhaustive Rust `enum`.
                ///
                /// This is similar to the [`Builder::rustified_enum`] style, but the `enum` is
                /// tagged with the `#[non_exhaustive]` attribute.
                pub fn rustified_non_exhaustive_enum<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.rustified_non_exhaustive_enums.insert(arg);
                    self
                }
            }
        },
        as_args: "--rustified-non-exhaustive-enums",
    },
    constified_enum_modules: RegexSet {
        methods: {
            regex_option! {
                /// Mark the given `enum` as a module with a set of integer constants.
                pub fn constified_enum_module<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.constified_enum_modules.insert(arg);
                    self
                }
            }
        },
        as_args: "--constified-enum-module",
    },
    constified_enums: RegexSet {
        methods: {
            regex_option! {
                /// Mark the given `enum` as a set o integer constants.
                ///
                /// This is similar to the [`Builder::constified_enum_module`] style, but the
                /// constants are generated in the current module instead of in a new module.
                pub fn constified_enum<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.constified_enums.insert(arg);
                    self
                }
            }
        },
        as_args: "--constified-enum",
    },
    default_macro_constant_type: MacroTypeVariation {
        methods: {
            pub fn default_macro_constant_type(mut self, arg: MacroTypeVariation) -> Builder {
                self.options.default_macro_constant_type = arg;
                self
            }

        },
        as_args: |variation, args| {
            if *variation != Default::default() {
                args.push("--default-macro-constant-type".to_owned());
                args.push(variation.to_string());
            }
        },
    },
    default_alias_style: AliasVariation {
        methods: {
            pub fn default_alias_style(
                mut self,
                arg: AliasVariation,
            ) -> Builder {
                self.options.default_alias_style = arg;
                self
            }
        },
        as_args: |variation, args| {
            if *variation != Default::default() {
                args.push("--default-alias-style".to_owned());
                args.push(variation.to_string());
            }
        },
    },
    type_alias: RegexSet {
        methods: {
            regex_option! {
                /// Mark the given `typedef` as a regular Rust `type` alias.
                ///
                /// This is the default behavior, meaning that this method only comes into effect
                /// if a style different from [`AliasVariation::TypeAlias`] was passed to the
                /// [`Builder::default_alias_style`] method.
                pub fn type_alias<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.type_alias.insert(arg);
                    self
                }
            }
        },
        as_args: "--type-alias",
    },
    new_type_alias: RegexSet {
        methods: {
            regex_option! {
                /// Mark the given `typedef` as a Rust newtype by having the aliased
                /// type be wrapped in a `struct` with `#[repr(transparent)]`.
                ///
                /// This method can be used to enforce stricter type checking.
                pub fn new_type_alias<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.new_type_alias.insert(arg);
                    self
                }
            }
        },
        as_args: "--new-type-alias",
    },
    new_type_alias_deref: RegexSet {
        methods: {
            regex_option! {
                /// Mark the given `typedef` to be generated as a newtype that can be dereferenced.
                ///
                /// This is similar to the [`Builder::new_type_alias`] style, but the newtype
                /// implements `Deref` and `DerefMut` with the aliased type as a target.
                pub fn new_type_alias_deref<T: AsRef<str>>(mut self, arg: T) -> Builder {
                    self.options.new_type_alias_deref.insert(arg);
                    self
                }
            }
        },
        as_args: "--new-type-alias-deref",
    },
    default_non_copy_union_style: NonCopyUnionStyle {
        methods: {
            pub fn default_non_copy_union_style(mut self, arg: NonCopyUnionStyle) -> Self {
                self.options.default_non_copy_union_style = arg;
                self
            }
        },
        as_args: |style, args| {
            if *style != Default::default() {
                args.push("--default-non-copy-union-style".to_owned());
                args.push(style.to_string());
            }
        },
    },
    bindgen_wrapper_union: RegexSet {
        methods: {
            regex_option! {
                /// Mark the given `union` to use a `bindgen`-generated wrapper for its members if at
                /// least one them is not `Copy`.
                ///
                /// This is the default behavior, meaning that this method only comes into effect
                /// if a style different from [`NonCopyUnionStyle::BindgenWrapper`] was passed to
                /// the [`Builder::default_non_copy_union_style`] method.
                pub fn bindgen_wrapper_union<T: AsRef<str>>(mut self, arg: T) -> Self {
                    self.options.bindgen_wrapper_union.insert(arg);
                    self
                }
            }
        },
        as_args: "--bindgen-wrapper-union",
    },
    manually_drop_union: RegexSet {
        methods: {
            regex_option! {
                /// Mark the given `union` to use [`::core::mem::ManuallyDrop`] for its members if
                /// at least one of them is not `Copy`.
                ///
                /// The `ManuallyDrop` type was stabilized in Rust 1.20.0, do not use this option
                /// if your target version is lower than this.
                pub fn manually_drop_union<T: AsRef<str>>(mut self, arg: T) -> Self {
                    self.options.manually_drop_union.insert(arg);
                    self
                }
            }

        },
        as_args: "--manually-drop-union",
    },


    builtins: bool {
        methods: {
            pub fn emit_builtins(mut self) -> Builder {
                self.options.builtins = true;
                self
            }
        },
        as_args: "--builtins",
    },
    emit_ast: bool {
        methods: {
            pub fn emit_clang_ast(mut self) -> Builder {
                self.options.emit_ast = true;
                self
            }
        },
        as_args: "--emit-clang-ast",
    },
    emit_ir: bool {
        methods: {
            pub fn emit_ir(mut self) -> Builder {
                self.options.emit_ir = true;
                self
            }
        },
        as_args: "--emit-ir",
    },
    emit_ir_graphviz: Option<String> {
        methods: {
            pub fn emit_ir_graphviz<T: Into<String>>(mut self, path: T) -> Builder {
                let path = path.into();
                self.options.emit_ir_graphviz = Some(path);
                self
            }
        },
        as_args: "--emit-ir-graphviz",
    },

    enable_cxx_namespaces: bool {
        methods: {
            pub fn enable_cxx_namespaces(mut self) -> Builder {
                self.options.enable_cxx_namespaces = true;
                self
            }
        },
        as_args: "--enable-cxx-namespaces",
    },
    enable_function_attribute_detection: bool {
        methods: {
            pub fn enable_function_attribute_detection(mut self) -> Self {
                self.options.enable_function_attribute_detection = true;
                self
            }

        },
        as_args: "--enable-function-attribute-detection",
    },
    disable_name_namespacing: bool {
        methods: {
            pub fn disable_name_namespacing(mut self) -> Builder {
                self.options.disable_name_namespacing = true;
                self
            }
        },
        as_args: "--disable-name-namespacing",
    },
    disable_nested_struct_naming: bool {
        methods: {
            pub fn disable_nested_struct_naming(mut self) -> Builder {
                self.options.disable_nested_struct_naming = true;
                self
            }
        },
        as_args: "--disable-nested-struct-naming",
    },
    disable_header_comment: bool {
        methods: {
            pub fn disable_header_comment(mut self) -> Self {
                self.options.disable_header_comment = true;
                self
            }

        },
        as_args: "--disable-header-comment",
    },
    layout_tests: bool {
        default: true,
        methods: {
            pub fn layout_tests(mut self, doit: bool) -> Self {
                self.options.layout_tests = doit;
                self
            }
        },
        as_args: |value, args| (!value).as_args(args, "--no-layout-tests"),
    },
    impl_debug: bool {
        methods: {
            pub fn impl_debug(mut self, doit: bool) -> Self {
                self.options.impl_debug = doit;
                self
            }

        },
        as_args: "--impl-debug",
    },
    impl_partialeq: bool {
        methods: {
            pub fn impl_partialeq(mut self, doit: bool) -> Self {
                self.options.impl_partialeq = doit;
                self
            }
        },
        as_args: "--impl-partialeq",
    },
    derive_copy: bool {
        default: true,
        methods: {
            pub fn derive_copy(mut self, doit: bool) -> Self {
                self.options.derive_copy = doit;
                self
            }
        },
        as_args: |value, args| (!value).as_args(args, "--no-derive-copy"),
    },

    derive_debug: bool {
        default: true,
        methods: {
            pub fn derive_debug(mut self, doit: bool) -> Self {
                self.options.derive_debug = doit;
                self
            }
        },
        as_args: |value, args| (!value).as_args(args, "--no-derive-debug"),
    },

    derive_default: bool {
        methods: {
            pub fn derive_default(mut self, doit: bool) -> Self {
                self.options.derive_default = doit;
                self
            }
        },
        as_args: |&value, args| {
            let arg = if value {
                "--with-derive-default"
            } else {
                "--no-derive-default"
            };

            args.push(arg.to_owned());
        },
    },
    derive_hash: bool {
        methods: {
            pub fn derive_hash(mut self, doit: bool) -> Self {
                self.options.derive_hash = doit;
                self
            }
        },
        as_args: "--with-derive-hash",
    },
    derive_partialord: bool {
        methods: {
            pub fn derive_partialord(mut self, doit: bool) -> Self {
                self.options.derive_partialord = doit;
                if !doit {
                    self.options.derive_ord = false;
                }
                self
            }
        },
        as_args: "--with-derive-partialord",
    },
    derive_ord: bool {
        methods: {
            pub fn derive_ord(mut self, doit: bool) -> Self {
                self.options.derive_ord = doit;
                self.options.derive_partialord = doit;
                self
            }
        },
        as_args: "--with-derive-ord",
    },
    derive_partialeq: bool {
        methods: {
            pub fn derive_partialeq(mut self, doit: bool) -> Self {
                self.options.derive_partialeq = doit;
                if !doit {
                    self.options.derive_eq = false;
                }
                self
            }
        },
        as_args: "--with-derive-partialeq",
    },
    derive_eq: bool {
        methods: {
            pub fn derive_eq(mut self, doit: bool) -> Self {
                self.options.derive_eq = doit;
                if doit {
                    self.options.derive_partialeq = doit;
                }
                self
            }
        },
        as_args: "--with-derive-eq",
    },
    use_core: bool {
        methods: {
            pub fn use_core(mut self) -> Builder {
                self.options.use_core = true;
                self
            }

        },
        as_args: "--use-core",
    },
    ctypes_prefix: Option<String> {
        methods: {
            pub fn ctypes_prefix<T: Into<String>>(mut self, prefix: T) -> Builder {
                self.options.ctypes_prefix = Some(prefix.into());
                self
            }
        },
        as_args: "--ctypes-prefix",
    },
    anon_fields_prefix: String {
        default: DEFAULT_ANON_FIELDS_PREFIX.into(),
        methods: {
            pub fn anon_fields_prefix<T: Into<String>>(mut self, prefix: T) -> Builder {
                self.options.anon_fields_prefix = prefix.into();
                self
            }
        },
        as_args: |prefix, args| {
            if prefix != DEFAULT_ANON_FIELDS_PREFIX {
                args.push("--anon-fields-prefix".to_owned());
                args.push(prefix.clone());
            }
        },
    },
    time_phases: bool {
        methods: {
            pub fn time_phases(mut self, doit: bool) -> Self {
                self.options.time_phases = doit;
                self
            }
        },
        as_args: "--time-phases",
    },
    convert_floats: bool {
        default: true,
        methods: {
            pub fn no_convert_floats(mut self) -> Self {
                self.options.convert_floats = false;
                self
            }
        },
        as_args: |value, args| (!value).as_args(args, "--no-convert-floats"),
    },
    raw_lines: Vec<String> {
        methods: {
            pub fn raw_line<T: Into<String>>(mut self, arg: T) -> Self {
                self.options.raw_lines.push(arg.into());
                self
            }
        },
        as_args: |raw_lines, args| {
            for line in raw_lines {
                args.push("--raw-line".to_owned());
                args.push(line.clone());
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
                self.options
                    .module_lines
                    .entry(module.into())
                    .or_insert_with(Vec::new)
                    .push(line.into());
                self
            }
        },
        as_args: |module_lines, args| {
            for (module, lines) in module_lines {
                for line in lines.iter() {
                    args.push("--module-raw-line".to_owned());
                    args.push(module.clone());
                    args.push(line.clone());
                }
            }
        },
    },
    input_headers:  Vec<String> {
        methods: {
            pub fn header<T: Into<String>>(mut self, header: T) -> Builder {
                self.options.input_headers.push(header.into());
                self
            }
        },
        as_args: ignore,
    },
    clang_args: Vec<String> {
        methods: {
            pub fn clang_arg<T: Into<String>>(self, arg: T) -> Builder {
                self.clang_args([arg.into()])
            }

            pub fn clang_args<I: IntoIterator>(mut self, args: I) -> Builder
            where
                I::Item: AsRef<str>,
            {
                for arg in args {
                    self.options.clang_args.push(arg.as_ref().to_owned());
                }
                self
            }
        },
        as_args: ignore,
    },
    input_header_contents: Vec<(String, String)> {
        methods: {
            pub fn header_contents(mut self, name: &str, contents: &str) -> Builder {
                // Apparently clang relies on having virtual FS correspondent to
                // the real one, so we need absolute paths here
                let absolute_path = env::current_dir()
                    .expect("Cannot retrieve current directory")
                    .join(name)
                    .to_str()
                    .expect("Cannot convert current directory name to string")
                    .to_owned();
                self.options
                    .input_header_contents
                    .push((absolute_path, contents.into()));
                self
            }
        },
        as_args: ignore,
    },
    parse_callbacks: Vec<Rc<dyn ParseCallbacks>> {
        methods: {
            pub fn parse_callbacks(mut self, cb: Box<dyn ParseCallbacks>) -> Self {
                self.options.parse_callbacks.push(Rc::from(cb));
                self
            }
        },
        as_args: |_callbacks, _args| {
            #[cfg(feature = "__cli")]
            for cb in _callbacks {
                _args.extend(cb.cli_args());
            }
        },
    },
    codegen_config: CodegenConfig {
        default: CodegenConfig::all(),
        methods: {
            pub fn ignore_functions(mut self) -> Builder {
                self.options.codegen_config.remove(CodegenConfig::FUNCTIONS);
                self
            }

            pub fn ignore_methods(mut self) -> Builder {
                self.options.codegen_config.remove(CodegenConfig::METHODS);
                self
            }

            pub fn with_codegen_config(mut self, config: CodegenConfig) -> Self {
                self.options.codegen_config = config;
                self
            }
        },
        as_args: |codegen_config, args| {
            if !codegen_config.functions() {
                args.push("--ignore-functions".to_owned());
            }

            args.push("--generate".to_owned());

            let mut options: Vec<String> = Vec::new();
            if codegen_config.functions() {
                options.push("functions".to_owned());
            }

            if codegen_config.types() {
                options.push("types".to_owned());
            }

            if codegen_config.vars() {
                options.push("vars".to_owned());
            }

            if codegen_config.methods() {
                options.push("methods".to_owned());
            }

            if codegen_config.constructors() {
                options.push("constructors".to_owned());
            }

            if codegen_config.destructors() {
                options.push("destructors".to_owned());
            }

            args.push(options.join(","));

            if !codegen_config.methods() {
                args.push("--ignore-methods".to_owned());
            }
        },
    },
    conservative_inline_namespaces: bool {
        methods: {
            pub fn conservative_inline_namespaces(mut self) -> Builder {
                self.options.conservative_inline_namespaces = true;
                self
            }
        },
        as_args: "--conservative-inline-namespaces",
    },
    generate_comments: bool {
        default: true,
        methods: {
            pub fn generate_comments(mut self, doit: bool) -> Self {
                self.options.generate_comments = doit;
                self
            }
        },
        as_args: |value, args| (!value).as_args(args, "--no-doc-comments"),
    },
    generate_inline_functions: bool {
        methods: {
            #[cfg_attr(
                features = "experimental",
                doc = "\nCheck the [`Builder::wrap_static_fns`] method for an alternative."
            )]
            pub fn generate_inline_functions(mut self, doit: bool) -> Self {
                self.options.generate_inline_functions = doit;
                self
            }
        },
        as_args: "--generate-inline-functions",
    },
    allowlist_recursively: bool {
        default: true,
        methods: {
            pub fn allowlist_recursively(mut self, doit: bool) -> Self {
                self.options.allowlist_recursively = doit;
                self
            }
        },
        as_args: |value, args| (!value).as_args(args, "--no-recursive-allowlist"),
    },
    objc_extern_crate: bool {
        methods: {
            pub fn objc_extern_crate(mut self, doit: bool) -> Self {
                self.options.objc_extern_crate = doit;
                self
            }
        },
        as_args: "--objc-extern-crate",
    },
    generate_block: bool {
        methods: {
            pub fn generate_block(mut self, doit: bool) -> Self {
                self.options.generate_block = doit;
                self
            }
        },
        as_args: "--generate-block",
    },
    generate_cstr: bool {
        methods: {
            pub fn generate_cstr(mut self, doit: bool) -> Self {
                self.options.generate_cstr = doit;
                self
            }
        },
        as_args: "--generate-cstr",
    },
    block_extern_crate: bool {
        methods: {
            pub fn block_extern_crate(mut self, doit: bool) -> Self {
                self.options.block_extern_crate = doit;
                self
            }
        },
        as_args: "--block-extern-crate",
    },
    enable_mangling: bool {
        default: true,
        methods: {
            pub fn trust_clang_mangling(mut self, doit: bool) -> Self {
                self.options.enable_mangling = doit;
                self
            }

        },
        as_args: |value, args| (!value).as_args(args, "--distrust-clang-mangling"),
    },
    detect_include_paths: bool {
        default: true,
        methods: {
            pub fn detect_include_paths(mut self, doit: bool) -> Self {
                self.options.detect_include_paths = doit;
                self
            }
        },
        as_args: |value, args| (!value).as_args(args, "--no-include-path-detection"),
    },
    fit_macro_constants: bool {
        methods: {
            pub fn fit_macro_constants(mut self, doit: bool) -> Self {
                self.options.fit_macro_constants = doit;
                self
            }
        },
        as_args: "--fit-macro-constant-types",
    },
    prepend_enum_name: bool {
        default: true,
        methods: {
            pub fn prepend_enum_name(mut self, doit: bool) -> Self {
                self.options.prepend_enum_name = doit;
                self
            }
        },
        as_args: |value, args| (!value).as_args(args, "--no-prepend-enum-name"),
    },
    rust_target: RustTarget {
        methods: {
            pub fn rust_target(mut self, rust_target: RustTarget) -> Self {
                self.options.set_rust_target(rust_target);
                self
            }
        },
        as_args: |rust_target, args| {
            args.push("--rust-target".to_owned());
            args.push((*rust_target).into());
        },
    },
    rust_features: RustFeatures {
        default: RustTarget::default().into(),
        methods: {},
        as_args: ignore,
    },
    untagged_union: bool {
        default: true,
        methods: {
            pub fn disable_untagged_union(mut self) -> Self {
                self.options.untagged_union = false;
                self
            }
        }
        as_args: |value, args| (!value).as_args(args, "--disable-untagged-union"),
    },
    record_matches: bool {
        default: true,
        methods: {
            pub fn record_matches(mut self, doit: bool) -> Self {
                self.options.record_matches = doit;
                self
            }

        },
        as_args: |value, args| (!value).as_args(args, "--no-record-matches"),
    },
    size_t_is_usize: bool {
        default: true,
        methods: {
            pub fn size_t_is_usize(mut self, is: bool) -> Self {
                self.options.size_t_is_usize = is;
                self
            }
        },
        as_args: |value, args| (!value).as_args(args, "--no-size_t-is-usize"),
    },
    formatter: Formatter {
        methods: {
            #[deprecated]
            pub fn rustfmt_bindings(mut self, doit: bool) -> Self {
                self.options.formatter = if doit {
                    Formatter::Rustfmt
                } else {
                    Formatter::None
                };
                self
            }

            pub fn formatter(mut self, formatter: Formatter) -> Self {
                self.options.formatter = formatter;
                self
            }
        },
        as_args: |formatter, args| {
            if *formatter != Default::default() {
                args.push("--formatter".to_owned());
                args.push(formatter.to_string());
            }
        },
    },
    rustfmt_configuration_file: Option<PathBuf> {
        methods: {
            pub fn rustfmt_configuration_file(mut self, path: Option<PathBuf>) -> Self {
                self = self.formatter(Formatter::Rustfmt);
                self.options.rustfmt_configuration_file = path;
                self
            }
        },
        as_args: "--rustfmt-configuration-file",
    },
    no_partialeq_types: RegexSet {
        methods: {
            regex_option! {
                /// Do not derive `PartialEq` for a given type.
                pub fn no_partialeq<T: Into<String>>(mut self, arg: T) -> Builder {
                    self.options.no_partialeq_types.insert(arg.into());
                    self
                }
            }
        },
        as_args: "--no-partialeq",
    },
    no_copy_types: RegexSet {
        methods: {
            regex_option! {
                /// Do not derive `Copy` and `Clone` for a given type.
                pub fn no_copy<T: Into<String>>(mut self, arg: T) -> Self {
                    self.options.no_copy_types.insert(arg.into());
                    self
                }
            }
        },
        as_args: "--no-copy",
    },
    no_debug_types: RegexSet {
        methods: {
            regex_option! {
                /// Do not derive `Debug` for a given type.
                pub fn no_debug<T: Into<String>>(mut self, arg: T) -> Self {
                    self.options.no_debug_types.insert(arg.into());
                    self
                }
            }
        },
        as_args: "--no-debug",
    },
    no_default_types: RegexSet {
        methods: {
            regex_option! {
                /// Do not derive or implement `Default` for a given type.
                pub fn no_default<T: Into<String>>(mut self, arg: T) -> Self {
                    self.options.no_default_types.insert(arg.into());
                    self
                }
            }
        },
        as_args: "--no-default",
    },
    no_hash_types: RegexSet {
        methods: {
            regex_option! {
                /// Do not derive `Hash` for a given type.
                pub fn no_hash<T: Into<String>>(mut self, arg: T) -> Builder {
                    self.options.no_hash_types.insert(arg.into());
                    self
                }
            }
        },
        as_args: "--no-hash",
    },
    must_use_types: RegexSet {
        methods: {
            regex_option! {
                /// Annotate the given type with the `#[must_use]` attribute.
                pub fn must_use_type<T: Into<String>>(mut self, arg: T) -> Builder {
                    self.options.must_use_types.insert(arg.into());
                    self
                }
            }
        },
        as_args: "--must-use-type",
    },
    array_pointers_in_arguments: bool {
        methods: {
            pub fn array_pointers_in_arguments(mut self, doit: bool) -> Self {
                self.options.array_pointers_in_arguments = doit;
                self
            }

        },
        as_args: "--use-array-pointers-in-arguments",
    },
    wasm_import_module_name: Option<String> {
        methods: {
            pub fn wasm_import_module_name<T: Into<String>>(
                mut self,
                import_name: T,
            ) -> Self {
                self.options.wasm_import_module_name = Some(import_name.into());
                self
            }
        },
        as_args: "--wasm-import-module-name",
    },
    dynamic_library_name: Option<String> {
        methods: {
            pub fn dynamic_library_name<T: Into<String>>(
                mut self,
                dynamic_library_name: T,
            ) -> Self {
                self.options.dynamic_library_name = Some(dynamic_library_name.into());
                self
            }
        },
        as_args: "--dynamic-loading",
    },
    dynamic_link_require_all: bool {
        methods: {
            pub fn dynamic_link_require_all(mut self, req: bool) -> Self {
                self.options.dynamic_link_require_all = req;
                self
            }
        },
        as_args: "--dynamic-link-require-all",
    },
    respect_cxx_access_specs: bool {
        methods: {
            pub fn respect_cxx_access_specs(mut self, doit: bool) -> Self {
                self.options.respect_cxx_access_specs = doit;
                self
            }

        },
        as_args: "--respect-cxx-access-specs",
    },
    translate_enum_integer_types: bool {
        methods: {
            pub fn translate_enum_integer_types(mut self, doit: bool) -> Self {
                self.options.translate_enum_integer_types = doit;
                self
            }
        },
        as_args: "--translate-enum-integer-types",
    },
    c_naming: bool {
        methods: {
            pub fn c_naming(mut self, doit: bool) -> Self {
                self.options.c_naming = doit;
                self
            }
        },
        as_args: "--c-naming",
    },
    force_explicit_padding: bool {
        methods: {
            pub fn explicit_padding(mut self, doit: bool) -> Self {
                self.options.force_explicit_padding = doit;
                self
            }
        },
        as_args: "--explicit-padding",
    },
    vtable_generation: bool {
        methods: {
            pub fn vtable_generation(mut self, doit: bool) -> Self {
                self.options.vtable_generation = doit;
                self
            }
        },
        as_args: "--vtable-generation",
    },
    sort_semantically: bool {
        methods: {
            pub fn sort_semantically(mut self, doit: bool) -> Self {
                self.options.sort_semantically = doit;
                self
            }
        },
        as_args: "--sort-semantically",
    },
    merge_extern_blocks: bool {
        methods: {
            pub fn merge_extern_blocks(mut self, doit: bool) -> Self {
                self.options.merge_extern_blocks = doit;
                self
            }
        },
        as_args: "--merge-extern-blocks",
    },
    wrap_unsafe_ops: bool {
        methods: {
            pub fn wrap_unsafe_ops(mut self, doit: bool) -> Self {
                self.options.wrap_unsafe_ops = doit;
                self
            }
        },
        as_args: "--wrap-unsafe-ops",
    },
    abi_overrides: HashMap<Abi, RegexSet> {
        methods: {
            regex_option! {
                /// Override the ABI of a given function.
                pub fn override_abi<T: Into<String>>(mut self, abi: Abi, arg: T) -> Self {
                    self.options
                        .abi_overrides
                        .entry(abi)
                        .or_default()
                        .insert(arg.into());
                    self
                }
            }
        },
        as_args: |overrides, args| {
            for (abi, set) in overrides {
                for item in set.get_items() {
                    args.push("--override-abi".to_owned());
                    args.push(format!("{}={}", item, abi));
                }
            }
        },
    },
    wrap_static_fns: bool {
        methods: {
            #[cfg(feature = "experimental")]
            pub fn wrap_static_fns(mut self, doit: bool) -> Self {
                self.options.wrap_static_fns = doit;
                self
            }
        },
        as_args: "--wrap-static-fns",
    },
    wrap_static_fns_suffix: Option<String> {
        methods: {
            #[cfg(feature = "experimental")]
            pub fn wrap_static_fns_suffix<T: AsRef<str>>(mut self, suffix: T) -> Self {
                self.options.wrap_static_fns_suffix = Some(suffix.as_ref().to_owned());
                self
            }
        },
        as_args: "--wrap-static-fns-suffix",
    },
    wrap_static_fns_path: Option<PathBuf> {
        methods: {
            #[cfg(feature = "experimental")]
            pub fn wrap_static_fns_path<T: AsRef<Path>>(mut self, path: T) -> Self {
                self.options.wrap_static_fns_path = Some(path.as_ref().to_owned());
                self
            }
        },
        as_args: "--wrap-static-fns-path",
    },
    default_visibility: FieldVisibilityKind {
        methods: {
            pub fn default_visibility(
                mut self,
                visibility: FieldVisibilityKind,
            ) -> Self {
                self.options.default_visibility = visibility;
                self
            }
        },
        as_args: |visibility, args| {
            if *visibility != Default::default() {
                args.push("--default-visibility".to_owned());
                args.push(visibility.to_string());
            }
        },
    },
    emit_diagnostics: bool {
        methods: {
            #[cfg(feature = "experimental")]
            pub fn emit_diagnostics(mut self) -> Self {
                self.options.emit_diagnostics = true;
                self
            }
        },
        as_args: "--emit-diagnostics",
    }
}
