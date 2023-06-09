use mlir_lib::*;
use std::{
    ffi::c_void,
    fmt::{self, Formatter},
    sync::Once,
};

use crate::mlir::{dialect::DialectRegistry, pass, Context, Error, LogicalResult, StringRef};

pub fn register_all_dialects(registry: &DialectRegistry) {
    unsafe { mlirRegisterAllDialects(registry.to_raw()) }
}
pub fn register_all_llvm_translations(context: &Context) {
    unsafe { mlirRegisterAllLLVMTranslations(context.to_raw()) }
}
pub fn register_all_passes() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe { mlirRegisterAllPasses() });
}
pub fn parse_pass_pipeline(manager: pass::OperationPassManager, source: &str) -> Result<(), Error> {
    let mut error_message = None;
    let result = LogicalResult::from_raw(unsafe {
        mlirParsePassPipeline(
            manager.to_raw(),
            StringRef::from(source).to_raw(),
            Some(handle_parse_error),
            &mut error_message as *mut _ as *mut _,
        )
    });
    if result.is_success() {
        Ok(())
    } else {
        Err(Error::ParsePassPipeline(
            error_message.unwrap_or_else(|| "failed to parse error message in UTF-8".into()),
        ))
    }
}

unsafe extern "C" fn handle_parse_error(raw_string: MlirStringRef, data: *mut c_void) {
    let string = StringRef::from_raw(raw_string);
    let data = &mut *(data as *mut Option<String>);
    if let Some(message) = data {
        message.extend(string.as_str())
    } else {
        *data = string.as_str().map(String::from).ok();
    }
}

pub unsafe fn into_raw_array<T>(xs: Vec<T>) -> *mut T {
    xs.leak().as_mut_ptr()
}
pub unsafe extern "C" fn print_callback(string: MlirStringRef, data: *mut c_void) {
    let (formatter, result) = &mut *(data as *mut (&mut Formatter, fmt::Result));
    if result.is_err() {
        return;
    }
    *result = (|| {
        write!(
            formatter,
            "{}",
            StringRef::from_raw(string).as_str().map_err(|_| fmt::Error)?
        )
    })();
}
pub unsafe extern "C" fn print_string_callback(string: MlirStringRef, data: *mut c_void) {
    let (writer, result) = &mut *(data as *mut (String, Result<(), Error>));
    if result.is_err() {
        return;
    }
    *result = (|| {
        writer.push_str(StringRef::from_raw(string).as_str()?);
        Ok(())
    })();
}

pub fn load_all_dialects(context: &Context) {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
}

pub fn create_test_context() -> Context {
    let context = Context::new();
    context.attach_diagnostic_handler(|diagnostic| {
        eprintln!("{}", diagnostic);
        true
    });
    load_all_dialects(&context);
    register_all_llvm_translations(&context);
    context
}
