use mlir_sys::{mlirDialectRegistryCreate, mlirDialectRegistryDestroy, MlirDialectRegistry};

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
    use crate::{context::Context, dialect::DialectHandle};

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
