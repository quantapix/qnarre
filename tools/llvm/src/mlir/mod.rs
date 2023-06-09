use mlir_lib::*;
use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    error,
    ffi::{c_void, CString},
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
    slice,
    str::{self, Utf8Error},
    sync::RwLock,
};

use crate::{
    ir::{Module, Type, TypeLike},
    Attribute, AttributeLike, Context, Error,
};

mod ctx;
pub mod diag;
pub mod dialect;
pub mod ir;
pub mod mlir_lib;
pub mod pass;
pub mod utils;

pub use self::ctx::{Context, ContextRef};

macro_rules! from_raw_subtypes {
    ($type:ident,) => {};
    ($type:ident, $name:ident $(, $names:ident)* $(,)?) => {
        impl<'c> From<$name<'c>> for $type<'c> {
            fn from(value: $name<'c>) -> Self {
                unsafe { Self::from_raw(value.to_raw()) }
            }
        }
        from_raw_subtypes!($type, $($names,)*);
    };
}

#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    AttributeExpected(&'static str, String),
    BlockArgumentExpected(String),
    ElementExpected {
        r#type: &'static str,
        value: String,
    },
    InvokeFunction,
    OperationResultExpected(String),
    PositionOutOfBounds {
        name: &'static str,
        val: String,
        idx: usize,
    },
    ParsePassPipeline(String),
    RunPass,
    TypeExpected(&'static str, String),
    UnknownDiagnosticSeverity(u32),
    Utf8(Utf8Error),
}
impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::AttributeExpected(r#type, attribute) => {
                write!(f, "{type} attribute expected: {attribute}")
            },
            Self::BlockArgumentExpected(value) => {
                write!(f, "block argument expected: {value}")
            },
            Self::ElementExpected { r#type, value } => {
                write!(f, "element of {type} type expected: {value}")
            },
            Self::InvokeFunction => write!(f, "failed to invoke JIT-compiled function"),
            Self::OperationResultExpected(value) => {
                write!(f, "operation result expected: {value}")
            },
            Self::ParsePassPipeline(message) => {
                write!(f, "failed to parse pass pipeline:\n{}", message)
            },
            Self::PositionOutOfBounds { name, value, index } => {
                write!(f, "{name} position {index} out of bounds: {value}")
            },
            Self::RunPass => write!(f, "failed to run pass"),
            Self::TypeExpected(r#type, actual) => {
                write!(f, "{type} type expected: {actual}")
            },
            Self::UnknownDiagnosticSeverity(severity) => {
                write!(f, "unknown diagnostic severity: {severity}")
            },
            Self::Utf8(error) => {
                write!(f, "{}", error)
            },
        }
    }
}
impl error::Error for Error {}
impl From<Utf8Error> for Error {
    fn from(error: Utf8Error) -> Self {
        Self::Utf8(error)
    }
}

pub struct ExecutionEngine {
    raw: MlirExecutionEngine,
}
impl ExecutionEngine {
    pub fn new(
        module: &Module,
        optimization_level: usize,
        shared_library_paths: &[&str],
        enable_object_dump: bool,
    ) -> Self {
        Self {
            raw: unsafe {
                mlirExecutionEngineCreate(
                    module.to_raw(),
                    optimization_level as i32,
                    shared_library_paths.len() as i32,
                    shared_library_paths
                        .iter()
                        .map(|&string| StringRef::from(string).to_raw())
                        .collect::<Vec<_>>()
                        .as_ptr(),
                    enable_object_dump,
                )
            },
        }
    }
    pub unsafe fn invoke_packed(&self, name: &str, arguments: &mut [*mut ()]) -> Result<(), Error> {
        let result = LogicalResult::from_raw(mlirExecutionEngineInvokePacked(
            self.raw,
            StringRef::from(name).to_raw(),
            arguments.as_mut_ptr() as *mut *mut c_void,
        ));
        if result.is_success() {
            Ok(())
        } else {
            Err(Error::InvokeFunction)
        }
    }
    pub fn dump_to_object_file(&self, path: &str) {
        unsafe { mlirExecutionEngineDumpToObjectFile(self.raw, StringRef::from(path).to_raw()) }
    }
}
impl Drop for ExecutionEngine {
    fn drop(&mut self) {
        unsafe { mlirExecutionEngineDestroy(self.raw) }
    }
}

#[derive(Clone, Copy)]
pub struct Integer<'c> {
    raw: MlirAttribute,
    _context: PhantomData<&'c Context>,
}
impl<'c> Integer<'c> {
    pub fn new(integer: i64, r#type: Type<'c>) -> Self {
        unsafe { Self::from_raw(mlirIntegerAttrGet(r#type.to_raw(), integer)) }
    }
    unsafe fn from_raw(raw: MlirAttribute) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
}
impl<'c> AttributeLike<'c> for Integer<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.raw
    }
}
impl<'c> TryFrom<Attribute<'c>> for Integer<'c> {
    type Error = Error;
    fn try_from(attribute: Attribute<'c>) -> Result<Self, Self::Error> {
        if attribute.is_integer() {
            Ok(unsafe { Self::from_raw(attribute.to_raw()) })
        } else {
            Err(Error::AttributeExpected("integer", format!("{}", attribute)))
        }
    }
}
impl<'c> Display for Integer<'c> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(&Attribute::from(*self), f)
    }
}
impl<'c> Debug for Integer<'c> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LogicalResult {
    raw: MlirLogicalResult,
}
impl LogicalResult {
    pub fn success() -> Self {
        Self {
            raw: MlirLogicalResult { value: 1 },
        }
    }
    pub fn failure() -> Self {
        Self {
            raw: MlirLogicalResult { value: 0 },
        }
    }
    pub fn is_success(&self) -> bool {
        self.raw.value != 0
    }
    #[allow(dead_code)]
    pub fn is_failure(&self) -> bool {
        self.raw.value == 0
    }
    pub fn from_raw(result: MlirLogicalResult) -> Self {
        Self { raw: result }
    }
    pub fn to_raw(self) -> MlirLogicalResult {
        self.raw
    }
}
impl From<bool> for LogicalResult {
    fn from(ok: bool) -> Self {
        if ok {
            Self::success()
        } else {
            Self::failure()
        }
    }
}

static STRING_CACHE: Lazy<RwLock<HashMap<String, CString>>> = Lazy::new(Default::default);

#[derive(Clone, Copy, Debug)]
pub struct StringRef<'a> {
    raw: MlirStringRef,
    _parent: PhantomData<&'a ()>,
}
impl<'a> StringRef<'a> {
    pub fn as_str(&self) -> Result<&'a str, Utf8Error> {
        unsafe {
            let bytes = slice::from_raw_parts(self.raw.data as *mut u8, self.raw.length);
            str::from_utf8(if bytes[bytes.len() - 1] == 0 {
                &bytes[..bytes.len() - 1]
            } else {
                bytes
            })
        }
    }
    pub fn to_raw(self) -> MlirStringRef {
        self.raw
    }
    pub unsafe fn from_raw(string: MlirStringRef) -> Self {
        Self {
            raw: string,
            _parent: Default::default(),
        }
    }
}
impl<'a> PartialEq for StringRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirStringRefEqual(self.raw, other.raw) }
    }
}
impl<'a> Eq for StringRef<'a> {}
impl From<&str> for StringRef<'static> {
    fn from(string: &str) -> Self {
        if !STRING_CACHE.read().unwrap().contains_key(string) {
            STRING_CACHE
                .write()
                .unwrap()
                .insert(string.to_owned(), CString::new(string).unwrap());
        }
        let lock = STRING_CACHE.read().unwrap();
        let string = lock.get(string).unwrap();
        unsafe { Self::from_raw(mlirStringRefCreateFromCString(string.as_ptr())) }
    }
}

#[cfg(test)]
mod tests;
