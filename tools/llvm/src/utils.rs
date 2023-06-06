use libc::{c_char, c_void};
use llvm_lib::core::*;
use llvm_lib::error_handling::*;
use llvm_lib::prelude::LLVMDiagnosticInfoRef;
use llvm_lib::support::LLVMLoadLibraryPermanently;
use llvm_lib::LLVMDiagnosticSeverity;
use std::borrow::Cow;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt::{self, Debug, Display, Formatter};
use std::ops::Deref;

pub struct DiagnosticInfo {
    diagnostic_info: LLVMDiagnosticInfoRef,
}
impl DiagnosticInfo {
    pub unsafe fn new(diagnostic_info: LLVMDiagnosticInfoRef) -> Self {
        DiagnosticInfo { diagnostic_info }
    }
    pub fn get_description(&self) -> *mut ::libc::c_char {
        unsafe { LLVMGetDiagInfoDescription(self.diagnostic_info) }
    }
    pub fn severity_is_error(&self) -> bool {
        unsafe {
            match LLVMGetDiagInfoSeverity(self.diagnostic_info) {
                LLVMDiagnosticSeverity::LLVMDSError => true,
                _ => false,
            }
        }
    }
}

#[derive(Eq)]
pub struct LLVMString {
    pub ptr: *const c_char,
}
impl LLVMString {
    pub unsafe fn new(ptr: *const c_char) -> Self {
        LLVMString { ptr }
    }
    pub fn to_string(&self) -> String {
        (*self).to_string_lossy().into_owned()
    }
    pub fn create_from_c_str(string: &CStr) -> LLVMString {
        unsafe { LLVMString::new(LLVMCreateMessage(string.as_ptr() as *const _)) }
    }
    pub fn create_from_str(string: &str) -> LLVMString {
        debug_assert_eq!(string.as_bytes()[string.as_bytes().len() - 1], 0);

        unsafe { LLVMString::new(LLVMCreateMessage(string.as_ptr() as *const _)) }
    }
}
impl Deref for LLVMString {
    type Target = CStr;
    fn deref(&self) -> &Self::Target {
        unsafe { CStr::from_ptr(self.ptr) }
    }
}
impl Debug for LLVMString {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "{:?}", self.deref())
    }
}
impl Display for LLVMString {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "{:?}", self.deref())
    }
}
impl PartialEq for LLVMString {
    fn eq(&self, other: &LLVMString) -> bool {
        **self == **other
    }
}
impl Error for LLVMString {
    fn description(&self) -> &str {
        self.to_str()
            .expect("Could not convert LLVMString to str (likely invalid unicode)")
    }
    fn cause(&self) -> Option<&dyn Error> {
        None
    }
}
impl Drop for LLVMString {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeMessage(self.ptr as *mut _);
        }
    }
}

#[derive(Eq)]
pub enum LLVMStringOrRaw {
    Owned(LLVMString),
    Borrowed(*const c_char),
}
impl LLVMStringOrRaw {
    pub fn as_str(&self) -> &CStr {
        match self {
            LLVMStringOrRaw::Owned(llvm_string) => llvm_string.deref(),
            LLVMStringOrRaw::Borrowed(ptr) => unsafe { CStr::from_ptr(*ptr) },
        }
    }
}
impl PartialEq for LLVMStringOrRaw {
    fn eq(&self, other: &LLVMStringOrRaw) -> bool {
        self.as_str() == other.as_str()
    }
}

pub unsafe fn shutdown_llvm() {
    use llvm_lib::core::LLVMShutdown;
    LLVMShutdown()
}

pub fn get_llvm_version() -> (u32, u32, u32) {
    let mut major: u32 = 0;
    let mut minor: u32 = 0;
    let mut patch: u32 = 0;
    unsafe { LLVMGetVersion(&mut major, &mut minor, &mut patch) };
    return (major, minor, patch);
}

pub fn load_library_permanently(filename: &str) -> bool {
    let filename = to_c_str(filename);
    unsafe { LLVMLoadLibraryPermanently(filename.as_ptr()) == 1 }
}

pub fn is_multithreaded() -> bool {
    use llvm_lib::core::LLVMIsMultithreaded;
    unsafe { LLVMIsMultithreaded() == 1 }
}

pub fn enable_llvm_pretty_stack_trace() {
    unsafe { LLVMEnablePrettyStackTrace() }
}

pub fn to_c_str<'s>(mut s: &'s str) -> Cow<'s, CStr> {
    if s.is_empty() {
        s = "\0";
    }
    if !s.chars().rev().any(|ch| ch == '\0') {
        return Cow::from(CString::new(s).expect("unreachable since null bytes are checked"));
    }
    unsafe { Cow::from(CStr::from_ptr(s.as_ptr() as *const _)) }
}

pub extern "C" fn get_error_str_diagnostic_handler(x: LLVMDiagnosticInfoRef, void_ptr: *mut c_void) {
    let y = unsafe { DiagnosticInfo::new(x) };
    if y.severity_is_error() {
        let ptr = void_ptr as *mut *mut c_void as *mut *mut ::libc::c_char;
        unsafe {
            *ptr = y.get_description();
        }
    }
}

pub unsafe fn install_fatal_error_handler(x: extern "C" fn(*const ::libc::c_char)) {
    LLVMInstallFatalErrorHandler(Some(x))
}
pub fn reset_fatal_error_handler() {
    unsafe { LLVMResetFatalErrorHandler() }
}

#[test]
fn test_to_c_str() {
    assert!(matches!(to_c_str("my string"), Cow::Owned(_)));
    assert!(matches!(to_c_str("my string\0"), Cow::Borrowed(_)));
}
