#[cfg(feature = "runtime")]
macro_rules! link {
    (
        @LOAD:
        $(#[doc=$doc:expr])*
        #[cfg($cfg:meta)]
        fn $name:ident($($pname:ident: $pty:ty), *) $(-> $ret:ty)*
    ) => (
        $(#[doc=$doc])*
        #[cfg($cfg)]
        pub fn $name(library: &mut super::SharedLibrary) {
            let symbol = unsafe { library.library.get(stringify!($name).as_bytes()) }.ok();
            library.functions.$name = match symbol {
                Some(s) => *s,
                None => None,
            };
        }

        #[cfg(not($cfg))]
        pub fn $name(_: &mut super::SharedLibrary) {}
    );

    (
        @LOAD:
        fn $name:ident($($pname:ident: $pty:ty), *) $(-> $ret:ty)*
    ) => (
        link!(@LOAD: #[cfg(feature = "runtime")] fn $name($($pname: $pty), *) $(-> $ret)*);
    );

    (
        $(
            $(#[doc=$doc:expr] #[cfg($cfg:meta)])*
            pub fn $name:ident($($pname:ident: $pty:ty), *) $(-> $ret:ty)*;
        )+
    ) => (
        use std::cell::{RefCell};
        use std::fmt;
        use std::sync::{Arc};
        use std::path::{Path, PathBuf};

        #[allow(missing_docs)]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum Version {
            V17_0 = 170,
        }

        impl fmt::Display for Version {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                use Version::*;
                match self {
                    V17_0 => write!(f, "17.0.x or later"),
                }
            }
        }

        #[derive(Debug, Default)]
        pub struct Functions {
            $(
                $(#[doc=$doc] #[cfg($cfg)])*
                pub $name: Option<unsafe extern fn($($pname: $pty), *) $(-> $ret)*>,
            )+
        }

        #[derive(Debug)]
        pub struct SharedLibrary {
            library: libloading::Library,
            path: PathBuf,
            pub functions: Functions,
        }

        impl SharedLibrary {
            fn new(library: libloading::Library, path: PathBuf) -> Self {
                Self { library, path, functions: Functions::default() }
            }

            pub fn path(&self) -> &Path {
                &self.path
            }

            pub fn version(&self) -> Option<Version> {
                macro_rules! check {
                    ($fn:expr, $version:ident) => {
                        if self.library.get::<unsafe extern fn()>($fn).is_ok() {
                            return Some(Version::$version);
                        }
                    };
                }

                unsafe {
                    check!(b"clang_CXXMethod_isCopyAssignmentOperator", V16_0);
                    check!(b"clang_Cursor_getVarDeclInitializer", V12_0);
                    check!(b"clang_Type_getValueType", V11_0);
                    check!(b"clang_Cursor_isAnonymousRecordDecl", V9_0);
                    check!(b"clang_Cursor_getObjCPropertyGetterName", V8_0);
                    check!(b"clang_File_tryGetRealPathName", V7_0);
                    check!(b"clang_CXIndex_setInvocationEmissionPathOption", V6_0);
                    check!(b"clang_Cursor_isExternalSymbol", V5_0);
                    check!(b"clang_EvalResult_getAsLongLong", V4_0);
                    check!(b"clang_CXXConstructor_isConvertingConstructor", V3_9);
                    check!(b"clang_CXXField_isMutable", V3_8);
                    check!(b"clang_Cursor_getOffsetOfField", V3_7);
                    check!(b"clang_Cursor_getStorageClass", V3_6);
                    check!(b"clang_Type_getNumTemplateArguments", V3_5);
                }

                None
            }
        }

        thread_local!(static LIBRARY: RefCell<Option<Arc<SharedLibrary>>> = RefCell::new(None));

        pub fn is_loaded() -> bool {
            LIBRARY.with(|l| l.borrow().is_some())
        }

        fn with_library<T, F>(f: F) -> Option<T> where F: FnOnce(&SharedLibrary) -> T {
            LIBRARY.with(|l| {
                match l.borrow().as_ref() {
                    Some(library) => Some(f(&library)),
                    _ => None,
                }
            })
        }

        $(
            #[cfg_attr(feature="cargo-clippy", allow(clippy::missing_safety_doc))]
            #[cfg_attr(feature="cargo-clippy", allow(clippy::too_many_arguments))]
            $(#[doc=$doc] #[cfg($cfg)])*
            pub unsafe fn $name($($pname: $pty), *) $(-> $ret)* {
                let f = with_library(|library| {
                    if let Some(function) = library.functions.$name {
                        function
                    } else {
                        panic!(
                            r#"
A `libclang` function was called that is not supported by the loaded `libclang` instance.

    called function = `{0}`
    loaded `libclang` instance = {1}
"#,
                            stringify!($name),
                            library
                                .version()
                                .map(|v| format!("{}", v))
                                .unwrap_or_else(|| "unsupported version".into()),
                        );
                    }
                }).expect("a `libclang` shared library is not loaded on this thread");
                f($($pname), *)
            }

            $(#[doc=$doc] #[cfg($cfg)])*
            pub mod $name {
                pub fn is_loaded() -> bool {
                    super::with_library(|l| l.functions.$name.is_some()).unwrap_or(false)
                }
            }
        )+

        mod load {
            $(link!(@LOAD: $(#[cfg($cfg)])* fn $name($($pname: $pty), *) $(-> $ret)*);)+
        }

        pub fn load_manually() -> Result<SharedLibrary, String> {
            #[allow(dead_code)]
            mod build {
                include!(concat!(env!("OUT_DIR"), "/macros.rs"));
                pub mod common { include!(concat!(env!("OUT_DIR"), "/common.rs")); }
                pub mod dynamic { include!(concat!(env!("OUT_DIR"), "/dynamic.rs")); }
            }

            let (directory, filename) = build::dynamic::find(true)?;
            let path = directory.join(filename);

            unsafe {
                let library = libloading::Library::new(&path).map_err(|e| {
                    format!(
                        "the `libclang` shared library at {} could not be opened: {}",
                        path.display(),
                        e,
                    )
                });

                let mut library = SharedLibrary::new(library?, path);
                $(load::$name(&mut library);)+
                Ok(library)
            }
        }

        #[allow(dead_code)]
        pub fn load() -> Result<(), String> {
            let library = Arc::new(load_manually()?);
            LIBRARY.with(|l| *l.borrow_mut() = Some(library));
            Ok(())
        }

        pub fn unload() -> Result<(), String> {
            let library = set_library(None);
            if library.is_some() {
                Ok(())
            } else {
                Err("a `libclang` shared library is not in use in the current thread".into())
            }
        }

        pub fn get_library() -> Option<Arc<SharedLibrary>> {
            LIBRARY.with(|l| l.borrow_mut().clone())
        }

        pub fn set_library(library: Option<Arc<SharedLibrary>>) -> Option<Arc<SharedLibrary>> {
            LIBRARY.with(|l| mem::replace(&mut *l.borrow_mut(), library))
        }
    )
}

#[cfg(not(feature = "runtime"))]
macro_rules! link {
    (
        $(
            pub fn $name:ident($($pn:ident: $pty:ty), *) $(-> $ret:ty)*;
        )+
    ) => (
        extern {
            $(
                pub fn $name($($pn: $pty), *) $(-> $ret)*;
            )+
        }
        $(
            pub mod $name {
                pub fn is_loaded() -> bool { true }
            }
        )+
    )
}
