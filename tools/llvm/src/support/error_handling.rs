//! This module contains some supplemental functions for dealing with errors.

use libc::c_void;
use llvm_lib::core::{LLVMGetDiagInfoDescription, LLVMGetDiagInfoSeverity};
use llvm_lib::error_handling::{LLVMInstallFatalErrorHandler, LLVMResetFatalErrorHandler};
use llvm_lib::prelude::LLVMDiagnosticInfoRef;
use llvm_lib::LLVMDiagnosticSeverity;

//
pub unsafe fn install_fatal_error_handler(handler: extern "C" fn(*const ::libc::c_char)) {
    LLVMInstallFatalErrorHandler(Some(handler))
}

pub fn reset_fatal_error_handler() {
    unsafe { LLVMResetFatalErrorHandler() }
}

pub(crate) struct DiagnosticInfo {
    diagnostic_info: LLVMDiagnosticInfoRef,
}

impl DiagnosticInfo {
    pub unsafe fn new(diagnostic_info: LLVMDiagnosticInfoRef) -> Self {
        DiagnosticInfo { diagnostic_info }
    }

    pub(crate) fn get_description(&self) -> *mut ::libc::c_char {
        unsafe { LLVMGetDiagInfoDescription(self.diagnostic_info) }
    }

    pub(crate) fn severity_is_error(&self) -> bool {
        unsafe {
            match LLVMGetDiagInfoSeverity(self.diagnostic_info) {
                LLVMDiagnosticSeverity::LLVMDSError => true,
                _ => false,
            }
        }
    }
}

//
pub(crate) extern "C" fn get_error_str_diagnostic_handler(
    diagnostic_info: LLVMDiagnosticInfoRef,
    void_ptr: *mut c_void,
) {
    let diagnostic_info = unsafe { DiagnosticInfo::new(diagnostic_info) };

    if diagnostic_info.severity_is_error() {
        let c_ptr_ptr = void_ptr as *mut *mut c_void as *mut *mut ::libc::c_char;

        unsafe {
            *c_ptr_ptr = diagnostic_info.get_description();
        }
    }
}
