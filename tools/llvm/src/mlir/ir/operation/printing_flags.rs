use mlir_sys::{
    mlirOpPrintingFlagsCreate, mlirOpPrintingFlagsDestroy, mlirOpPrintingFlagsElideLargeElementsAttrs,
    mlirOpPrintingFlagsEnableDebugInfo, mlirOpPrintingFlagsPrintGenericOpForm, mlirOpPrintingFlagsUseLocalScope,
    MlirOpPrintingFlags,
};

#[derive(Debug)]
pub struct OperationPrintingFlags(MlirOpPrintingFlags);

impl OperationPrintingFlags {
    pub fn new() -> Self {
        Self(unsafe { mlirOpPrintingFlagsCreate() })
    }

    pub fn elide_large_elements_attributes(self, limit: usize) -> Self {
        unsafe { mlirOpPrintingFlagsElideLargeElementsAttrs(self.0, limit as isize) }

        self
    }

    pub fn enable_debug_info(self, enabled: bool, pretty_form: bool) -> Self {
        unsafe { mlirOpPrintingFlagsEnableDebugInfo(self.0, enabled, pretty_form) }

        self
    }

    pub fn print_generic_operation_form(self) -> Self {
        unsafe { mlirOpPrintingFlagsPrintGenericOpForm(self.0) }

        self
    }

    pub fn use_local_scope(self) -> Self {
        unsafe { mlirOpPrintingFlagsUseLocalScope(self.0) }

        self
    }

    pub fn to_raw(&self) -> MlirOpPrintingFlags {
        self.0
    }
}

impl Drop for OperationPrintingFlags {
    fn drop(&mut self) {
        unsafe { mlirOpPrintingFlagsDestroy(self.0) }
    }
}

impl Default for OperationPrintingFlags {
    fn default() -> Self {
        Self::new()
    }
}
