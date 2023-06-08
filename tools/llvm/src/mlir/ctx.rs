use mlir_lib::*;
use std::{ffi::c_void, marker::PhantomData, mem::transmute, ops::Deref};

use super::{
    diagnostic::{Diagnostic, DiagnosticHandlerId},
    dialect::{Dialect, DialectRegistry},
    logical_result::LogicalResult,
    StringRef,
};

#[derive(Debug)]
pub struct Context {
    raw: MlirContext,
}
impl Context {
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirContextCreate() },
        }
    }
    pub fn registered_dialect_count(&self) -> usize {
        unsafe { mlirContextGetNumRegisteredDialects(self.raw) as usize }
    }
    pub fn loaded_dialect_count(&self) -> usize {
        unsafe { mlirContextGetNumLoadedDialects(self.raw) as usize }
    }
    pub fn get_or_load_dialect(&self, name: &str) -> Dialect {
        unsafe { Dialect::from_raw(mlirContextGetOrLoadDialect(self.raw, StringRef::from(name).to_raw())) }
    }
    pub fn append_dialect_registry(&self, registry: &DialectRegistry) {
        unsafe { mlirContextAppendDialectRegistry(self.raw, registry.to_raw()) }
    }
    pub fn load_all_available_dialects(&self) {
        unsafe { mlirContextLoadAllAvailableDialects(self.raw) }
    }
    pub fn enable_multi_threading(&self, enabled: bool) {
        unsafe { mlirContextEnableMultithreading(self.raw, enabled) }
    }
    pub fn allow_unregistered_dialects(&self) -> bool {
        unsafe { mlirContextGetAllowUnregisteredDialects(self.raw) }
    }
    pub fn set_allow_unregistered_dialects(&self, allowed: bool) {
        unsafe { mlirContextSetAllowUnregisteredDialects(self.raw, allowed) }
    }
    pub fn is_registered_operation(&self, name: &str) -> bool {
        unsafe { mlirContextIsRegisteredOperation(self.raw, StringRef::from(name).to_raw()) }
    }
    pub fn to_raw(&self) -> MlirContext {
        self.raw
    }
    pub fn attach_diagnostic_handler<F: FnMut(Diagnostic) -> bool>(&self, handler: F) -> DiagnosticHandlerId {
        unsafe extern "C" fn handle<F: FnMut(Diagnostic) -> bool>(
            diagnostic: MlirDiagnostic,
            user_data: *mut c_void,
        ) -> MlirLogicalResult {
            LogicalResult::from((*(user_data as *mut F))(Diagnostic::from_raw(diagnostic))).to_raw()
        }
        unsafe extern "C" fn destroy<F: FnMut(Diagnostic) -> bool>(user_data: *mut c_void) {
            drop(Box::from_raw(user_data as *mut F));
        }
        unsafe {
            DiagnosticHandlerId::from_raw(mlirContextAttachDiagnosticHandler(
                self.to_raw(),
                Some(handle::<F>),
                Box::into_raw(Box::new(handler)) as *mut _,
                Some(destroy::<F>),
            ))
        }
    }
    pub fn detach_diagnostic_handler(&self, id: DiagnosticHandlerId) {
        unsafe { mlirContextDetachDiagnosticHandler(self.to_raw(), id.to_raw()) }
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe { mlirContextDestroy(self.raw) };
    }
}
impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirContextEqual(self.raw, other.raw) }
    }
}
impl Eq for Context {}

#[derive(Clone, Copy, Debug)]
pub struct ContextRef<'a> {
    raw: MlirContext,
    _reference: PhantomData<&'a Context>,
}
impl<'a> ContextRef<'a> {
    pub unsafe fn from_raw(raw: MlirContext) -> Self {
        Self {
            raw,
            _reference: Default::default(),
        }
    }
}
impl<'a> Deref for ContextRef<'a> {
    type Target = Context;
    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
    }
}
impl<'a> PartialEq for ContextRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirContextEqual(self.raw, other.raw) }
    }
}
impl<'a> Eq for ContextRef<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn new() {
        Context::new();
    }
    #[test]
    fn registered_dialect_count() {
        let context = Context::new();
        assert_eq!(context.registered_dialect_count(), 1);
    }
    #[test]
    fn loaded_dialect_count() {
        let context = Context::new();
        assert_eq!(context.loaded_dialect_count(), 1);
    }
    #[test]
    fn append_dialect_registry() {
        let context = Context::new();
        context.append_dialect_registry(&DialectRegistry::new());
    }
    #[test]
    fn is_registered_operation() {
        let context = Context::new();
        assert!(context.is_registered_operation("builtin.module"));
    }
    #[test]
    fn is_not_registered_operation() {
        let context = Context::new();
        assert!(!context.is_registered_operation("func.func"));
    }
    #[test]
    fn enable_multi_threading() {
        let context = Context::new();
        context.enable_multi_threading(true);
    }
    #[test]
    fn disable_multi_threading() {
        let context = Context::new();
        context.enable_multi_threading(false);
    }
    #[test]
    fn allow_unregistered_dialects() {
        let context = Context::new();
        assert!(!context.allow_unregistered_dialects());
    }
    #[test]
    fn set_allow_unregistered_dialects() {
        let context = Context::new();
        context.set_allow_unregistered_dialects(true);
        assert!(context.allow_unregistered_dialects());
    }
    #[test]
    fn attach_and_detach_diagnostic_handler() {
        let context = Context::new();
        let id = context.attach_diagnostic_handler(|diagnostic| {
            println!("{}", diagnostic);
            true
        });
        context.detach_diagnostic_handler(id);
    }
}
