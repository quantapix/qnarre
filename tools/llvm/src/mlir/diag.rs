use crate::Error;
use crate::{ir::Location, utils::print_callback, Error};
use mlir_lib::*;
use std::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
};

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

#[derive(Clone, Copy, Debug)]
pub enum DiagnosticSeverity {
    Error,
    Note,
    Remark,
    Warning,
}
impl TryFrom<u32> for DiagnosticSeverity {
    type Error = Error;
    fn try_from(severity: u32) -> Result<Self, Error> {
        #[allow(non_upper_case_globals)]
        Ok(match severity {
            MlirDiagnosticSeverity_MlirDiagnosticError => Self::Error,
            MlirDiagnosticSeverity_MlirDiagnosticNote => Self::Note,
            MlirDiagnosticSeverity_MlirDiagnosticRemark => Self::Remark,
            MlirDiagnosticSeverity_MlirDiagnosticWarning => Self::Warning,
            _ => return Err(Error::UnknownDiagnosticSeverity(severity)),
        })
    }
}

#[derive(Debug)]
pub struct Diagnostic<'c> {
    raw: MlirDiagnostic,
    phantom: PhantomData<&'c ()>,
}
impl<'c> Diagnostic<'c> {
    pub fn location(&self) -> Location {
        unsafe { Location::from_raw(mlirDiagnosticGetLocation(self.raw)) }
    }
    pub fn severity(&self) -> DiagnosticSeverity {
        DiagnosticSeverity::try_from(unsafe { mlirDiagnosticGetSeverity(self.raw) })
            .unwrap_or_else(|error| unreachable!("{}", error))
    }
    pub fn note_count(&self) -> usize {
        (unsafe { mlirDiagnosticGetNumNotes(self.raw) }) as usize
    }
    pub fn note(&self, index: usize) -> Result<Self, Error> {
        if index < self.note_count() {
            Ok(unsafe { Self::from_raw(mlirDiagnosticGetNote(self.raw, index as isize)) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "diagnostic note",
                value: self.to_string(),
                index,
            })
        }
    }
    pub unsafe fn from_raw(raw: MlirDiagnostic) -> Self {
        Self {
            raw,
            phantom: Default::default(),
        }
    }
}
impl<'a> Display for Diagnostic<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));
        unsafe {
            mlirDiagnosticPrint(self.raw, Some(print_callback), &mut data as *mut _ as *mut c_void);
        }
        data.1
    }
}

#[cfg(test)]
mod tests {
    use crate::{ir::Module, Context};
    #[test]
    fn handle_diagnostic() {
        let mut message = None;
        let context = Context::new();
        context.attach_diagnostic_handler(|diagnostic| {
            message = Some(diagnostic.to_string());
            true
        });
        Module::parse(&context, "foo");
        assert_eq!(
            message.unwrap(),
            "custom op 'foo' is unknown (tried 'builtin.foo' as well)"
        );
    }
}
