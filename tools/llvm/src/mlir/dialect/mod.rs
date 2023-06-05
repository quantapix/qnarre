pub mod arith;
pub mod cf;
pub mod func;
pub mod index;
pub mod llvm;
pub mod memref;
pub mod scf;
use crate::{
    ctx::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{
    mlirDialectEqual, mlirDialectGetContext, mlirDialectGetNamespace, mlirDialectHandleGetNamespace,
    mlirDialectHandleInsertDialect, mlirDialectHandleLoadDialect, mlirDialectHandleRegisterDialect,
    mlirDialectRegistryCreate, mlirDialectRegistryDestroy, mlirGetDialectHandle__async__, mlirGetDialectHandle__cf__,
    mlirGetDialectHandle__func__, mlirGetDialectHandle__gpu__, mlirGetDialectHandle__linalg__,
    mlirGetDialectHandle__llvm__, mlirGetDialectHandle__pdl__, mlirGetDialectHandle__quant__,
    mlirGetDialectHandle__scf__, mlirGetDialectHandle__shape__, mlirGetDialectHandle__sparse_tensor__,
    mlirGetDialectHandle__tensor__, MlirDialect, MlirDialectHandle, MlirDialectRegistry,
};
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug)]
pub struct Dialect<'c> {
    raw: MlirDialect,
    _context: PhantomData<&'c Context>,
}
impl<'c> Dialect<'c> {
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirDialectGetContext(self.raw)) }
    }
    pub fn namespace(&self) -> StringRef {
        unsafe { StringRef::from_raw(mlirDialectGetNamespace(self.raw)) }
    }
    pub unsafe fn from_raw(dialect: MlirDialect) -> Self {
        Self {
            raw: dialect,
            _context: Default::default(),
        }
    }
}
impl<'c> PartialEq for Dialect<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirDialectEqual(self.raw, other.raw) }
    }
}
impl<'c> Eq for Dialect<'c> {}

#[derive(Clone, Copy, Debug)]
pub struct DialectHandle {
    raw: MlirDialectHandle,
}
impl DialectHandle {
    pub fn r#async() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__async__()) }
    }
    pub fn cf() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__cf__()) }
    }
    pub fn func() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__func__()) }
    }
    pub fn gpu() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__gpu__()) }
    }
    pub fn linalg() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__linalg__()) }
    }
    pub fn llvm() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__llvm__()) }
    }
    pub fn pdl() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__pdl__()) }
    }
    pub fn quant() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__quant__()) }
    }
    pub fn scf() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__scf__()) }
    }
    pub fn shape() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__shape__()) }
    }
    pub fn sparse_tensor() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__sparse_tensor__()) }
    }
    pub fn tensor() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__tensor__()) }
    }
    pub fn namespace(&self) -> StringRef {
        unsafe { StringRef::from_raw(mlirDialectHandleGetNamespace(self.raw)) }
    }
    pub fn insert_dialect(&self, registry: &DialectRegistry) {
        unsafe { mlirDialectHandleInsertDialect(self.raw, registry.to_raw()) }
    }
    pub fn load_dialect<'c>(&self, context: &'c Context) -> Dialect<'c> {
        unsafe { Dialect::from_raw(mlirDialectHandleLoadDialect(self.raw, context.to_raw())) }
    }
    pub fn register_dialect(&self, context: &Context) {
        unsafe { mlirDialectHandleRegisterDialect(self.raw, context.to_raw()) }
    }
    pub unsafe fn from_raw(handle: MlirDialectHandle) -> Self {
        Self { raw: handle }
    }
}

#[derive(Debug)]
pub struct DialectRegistry {
    raw: MlirDialectRegistry,
}
impl DialectRegistry {
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirDialectRegistryCreate() },
        }
    }
    pub fn to_raw(&self) -> MlirDialectRegistry {
        self.raw
    }
}
impl Drop for DialectRegistry {
    fn drop(&mut self) {
        unsafe { mlirDialectRegistryDestroy(self.raw) };
    }
}
impl Default for DialectRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equal() {
        let context = Context::new();
        assert_eq!(
            DialectHandle::func().load_dialect(&context),
            DialectHandle::func().load_dialect(&context)
        );
    }
    #[test]
    fn not_equal() {
        let context = Context::new();
        assert_ne!(
            DialectHandle::func().load_dialect(&context),
            DialectHandle::llvm().load_dialect(&context)
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn func() {
        DialectHandle::func();
    }
    #[test]
    fn llvm() {
        DialectHandle::llvm();
    }
    #[test]
    fn namespace() {
        DialectHandle::func().namespace();
    }
    #[test]
    fn insert_dialect() {
        let registry = DialectRegistry::new();
        DialectHandle::func().insert_dialect(&registry);
    }
    #[test]
    fn load_dialect() {
        let context = Context::new();
        DialectHandle::func().load_dialect(&context);
    }
    #[test]
    fn register_dialect() {
        let context = Context::new();
        DialectHandle::func().register_dialect(&context);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ctx::Context, dialect::DialectHandle};
    #[test]
    fn new() {
        DialectRegistry::new();
    }
    #[test]
    fn register_all_dialects() {
        DialectRegistry::new();
    }
    #[test]
    fn register_dialect() {
        let registry = DialectRegistry::new();
        DialectHandle::func().insert_dialect(&registry);
        let context = Context::new();
        let count = context.registered_dialect_count();
        context.append_dialect_registry(&registry);
        assert_eq!(context.registered_dialect_count() - count, 1);
    }
}
