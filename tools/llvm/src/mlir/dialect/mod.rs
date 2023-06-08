use mlir_lib::*;
use std::marker::PhantomData;

use crate::mlir::{Context, ContextRef, StringRef};

pub mod arith;
pub mod cf;
pub mod func;
pub mod index;
pub mod llvm;
pub mod memref;
pub mod scf;

#[derive(Clone, Copy, Debug)]
pub struct Dialect<'c> {
    raw: MlirDialect,
    ctx: PhantomData<&'c Context>,
}
impl<'c> Dialect<'c> {
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirDialectGetContext(self.raw)) }
    }
    pub fn namespace(&self) -> StringRef {
        unsafe { StringRef::from_raw(mlirDialectGetNamespace(self.raw)) }
    }
    pub unsafe fn from_raw(raw: MlirDialect) -> Self {
        Self {
            raw,
            ctx: Default::default(),
        }
    }
}
impl<'c> PartialEq for Dialect<'c> {
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirDialectEqual(self.raw, x.raw) }
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
    pub fn insert_dialect(&self, x: &DialectRegistry) {
        unsafe { mlirDialectHandleInsertDialect(self.raw, x.to_raw()) }
    }
    pub fn load_dialect<'c>(&self, x: &'c Context) -> Dialect<'c> {
        unsafe { Dialect::from_raw(mlirDialectHandleLoadDialect(self.raw, x.to_raw())) }
    }
    pub fn register_dialect(&self, x: &Context) {
        unsafe { mlirDialectHandleRegisterDialect(self.raw, x.to_raw()) }
    }
    pub unsafe fn from_raw(raw: MlirDialectHandle) -> Self {
        Self { raw }
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
        let y = Context::new();
        assert_eq!(
            DialectHandle::func().load_dialect(&y),
            DialectHandle::func().load_dialect(&y)
        );
    }
    #[test]
    fn not_equal() {
        let y = Context::new();
        assert_ne!(
            DialectHandle::func().load_dialect(&y),
            DialectHandle::llvm().load_dialect(&y)
        );
    }
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
        let y = DialectRegistry::new();
        DialectHandle::func().insert_dialect(&y);
    }
    #[test]
    fn load_dialect() {
        let y = Context::new();
        DialectHandle::func().load_dialect(&y);
    }
    #[test]
    fn register_dialect() {
        let y = Context::new();
        DialectHandle::func().register_dialect(&y);
    }
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
        let r = DialectRegistry::new();
        DialectHandle::func().insert_dialect(&r);
        let c = Context::new();
        let n = c.registered_dialect_count();
        c.append_dialect_registry(&r);
        assert_eq!(c.registered_dialect_count() - n, 1);
    }
}
