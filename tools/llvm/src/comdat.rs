//! A `Comdat` helps resolve linker errors for duplicate sections.

use llvm_lib::comdat::{LLVMComdatSelectionKind, LLVMGetComdatSelectionKind, LLVMSetComdatSelectionKind};
use llvm_lib::prelude::LLVMComdatRef;

#[llvm_enum(LLVMComdatSelectionKind)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComdatSelectionKind {
    #[llvm_variant(LLVMAnyComdatSelectionKind)]
    Any,
    #[llvm_variant(LLVMExactMatchComdatSelectionKind)]
    ExactMatch,
    #[llvm_variant(LLVMLargestComdatSelectionKind)]
    Largest,
    #[llvm_variant(LLVMNoDuplicatesComdatSelectionKind)]
    NoDuplicates,
    #[llvm_variant(LLVMSameSizeComdatSelectionKind)]
    SameSize,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct Comdat(pub(crate) LLVMComdatRef);

impl Comdat {
    pub unsafe fn new(comdat: LLVMComdatRef) -> Self {
        debug_assert!(!comdat.is_null());

        Comdat(comdat)
    }

    pub fn as_mut_ptr(&self) -> LLVMComdatRef {
        self.0
    }

    pub fn get_selection_kind(self) -> ComdatSelectionKind {
        let kind_ptr = unsafe { LLVMGetComdatSelectionKind(self.0) };

        ComdatSelectionKind::new(kind_ptr)
    }

    pub fn set_selection_kind(self, kind: ComdatSelectionKind) {
        unsafe { LLVMSetComdatSelectionKind(self.0, kind.into()) }
    }
}
