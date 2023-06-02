use mlir_sys::MlirDiagnosticHandlerID;

#[derive(Clone, Copy, Debug)]
pub struct DiagnosticHandlerId {
    raw: MlirDiagnosticHandlerID,
}

impl DiagnosticHandlerId {
    pub unsafe fn from_raw(raw: MlirDiagnosticHandlerID) -> Self {
        Self { raw }
    }

    pub fn to_raw(self) -> MlirDiagnosticHandlerID {
        self.raw
    }
}
