#[macro_use]
extern crate inkwell_internals;

#[macro_use]
pub mod support;
pub mod attributes;
pub mod basic_block;
pub mod builder;
pub mod comdat;
pub mod context;
pub mod data_layout;
pub mod debug_info;
pub mod execution_engine;
pub mod intrinsics;
pub mod memory_buffer;
pub mod module;
pub mod object_file;
pub mod passes;
pub mod targets;
pub mod types;
pub mod values;

use llvm_lib::LLVMInlineAsmDialect;
use llvm_lib::{
    LLVMAtomicOrdering, LLVMAtomicRMWBinOp, LLVMDLLStorageClass, LLVMIntPredicate, LLVMRealPredicate,
    LLVMThreadLocalMode, LLVMVisibility,
};

use std::convert::TryFrom;

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct AddressSpace(u32);

impl Default for AddressSpace {
    fn default() -> Self {
        AddressSpace(0)
    }
}

impl From<u16> for AddressSpace {
    fn from(val: u16) -> Self {
        AddressSpace(val as u32)
    }
}

impl TryFrom<u32> for AddressSpace {
    type Error = ();

    fn try_from(val: u32) -> Result<Self, Self::Error> {
        if val < 1 << 24 {
            Ok(AddressSpace(val))
        } else {
            Err(())
        }
    }
}

#[llvm_enum(LLVMIntPredicate)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum IntPredicate {
    #[llvm_variant(LLVMIntEQ)]
    EQ,

    #[llvm_variant(LLVMIntNE)]
    NE,

    #[llvm_variant(LLVMIntUGT)]
    UGT,

    #[llvm_variant(LLVMIntUGE)]
    UGE,

    #[llvm_variant(LLVMIntULT)]
    ULT,

    #[llvm_variant(LLVMIntULE)]
    ULE,

    #[llvm_variant(LLVMIntSGT)]
    SGT,

    #[llvm_variant(LLVMIntSGE)]
    SGE,

    #[llvm_variant(LLVMIntSLT)]
    SLT,

    #[llvm_variant(LLVMIntSLE)]
    SLE,
}

#[llvm_enum(LLVMRealPredicate)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FloatPredicate {
    #[llvm_variant(LLVMRealOEQ)]
    OEQ,

    #[llvm_variant(LLVMRealOGE)]
    OGE,

    #[llvm_variant(LLVMRealOGT)]
    OGT,

    #[llvm_variant(LLVMRealOLE)]
    OLE,

    #[llvm_variant(LLVMRealOLT)]
    OLT,

    #[llvm_variant(LLVMRealONE)]
    ONE,

    #[llvm_variant(LLVMRealORD)]
    ORD,

    #[llvm_variant(LLVMRealPredicateFalse)]
    PredicateFalse,

    #[llvm_variant(LLVMRealPredicateTrue)]
    PredicateTrue,

    #[llvm_variant(LLVMRealUEQ)]
    UEQ,

    #[llvm_variant(LLVMRealUGE)]
    UGE,

    #[llvm_variant(LLVMRealUGT)]
    UGT,

    #[llvm_variant(LLVMRealULE)]
    ULE,

    #[llvm_variant(LLVMRealULT)]
    ULT,

    #[llvm_variant(LLVMRealUNE)]
    UNE,

    #[llvm_variant(LLVMRealUNO)]
    UNO,
}

#[llvm_enum(LLVMAtomicOrdering)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AtomicOrdering {
    #[llvm_variant(LLVMAtomicOrderingNotAtomic)]
    NotAtomic,
    #[llvm_variant(LLVMAtomicOrderingUnordered)]
    Unordered,
    #[llvm_variant(LLVMAtomicOrderingMonotonic)]
    Monotonic,
    #[llvm_variant(LLVMAtomicOrderingAcquire)]
    Acquire,
    #[llvm_variant(LLVMAtomicOrderingRelease)]
    Release,
    #[llvm_variant(LLVMAtomicOrderingAcquireRelease)]
    AcquireRelease,
    #[llvm_variant(LLVMAtomicOrderingSequentiallyConsistent)]
    SequentiallyConsistent,
}

#[llvm_enum(LLVMAtomicRMWBinOp)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AtomicRMWBinOp {
    #[llvm_variant(LLVMAtomicRMWBinOpXchg)]
    Xchg,

    #[llvm_variant(LLVMAtomicRMWBinOpAdd)]
    Add,

    #[llvm_variant(LLVMAtomicRMWBinOpSub)]
    Sub,

    #[llvm_variant(LLVMAtomicRMWBinOpAnd)]
    And,

    #[llvm_variant(LLVMAtomicRMWBinOpNand)]
    Nand,

    #[llvm_variant(LLVMAtomicRMWBinOpOr)]
    Or,

    #[llvm_variant(LLVMAtomicRMWBinOpXor)]
    Xor,

    #[llvm_variant(LLVMAtomicRMWBinOpMax)]
    Max,

    #[llvm_variant(LLVMAtomicRMWBinOpMin)]
    Min,

    #[llvm_variant(LLVMAtomicRMWBinOpUMax)]
    UMax,

    #[llvm_variant(LLVMAtomicRMWBinOpUMin)]
    UMin,

    #[llvm_variant(LLVMAtomicRMWBinOpFAdd)]
    FAdd,

    #[llvm_variant(LLVMAtomicRMWBinOpFSub)]
    FSub,

    #[llvm_variant(LLVMAtomicRMWBinOpFMax)]
    FMax,

    #[llvm_variant(LLVMAtomicRMWBinOpFMin)]
    FMin,
}

#[repr(u32)]
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum OptimizationLevel {
    None = 0,
    Less = 1,
    Default = 2,
    Aggressive = 3,
}

impl Default for OptimizationLevel {
    fn default() -> Self {
        OptimizationLevel::Default
    }
}

#[llvm_enum(LLVMVisibility)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GlobalVisibility {
    #[llvm_variant(LLVMDefaultVisibility)]
    Default,
    #[llvm_variant(LLVMHiddenVisibility)]
    Hidden,
    #[llvm_variant(LLVMProtectedVisibility)]
    Protected,
}

impl Default for GlobalVisibility {
    fn default() -> Self {
        GlobalVisibility::Default
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThreadLocalMode {
    GeneralDynamicTLSModel,
    LocalDynamicTLSModel,
    InitialExecTLSModel,
    LocalExecTLSModel,
}

impl ThreadLocalMode {
    pub(crate) fn new(thread_local_mode: LLVMThreadLocalMode) -> Option<Self> {
        match thread_local_mode {
            LLVMThreadLocalMode::LLVMGeneralDynamicTLSModel => Some(ThreadLocalMode::GeneralDynamicTLSModel),
            LLVMThreadLocalMode::LLVMLocalDynamicTLSModel => Some(ThreadLocalMode::LocalDynamicTLSModel),
            LLVMThreadLocalMode::LLVMInitialExecTLSModel => Some(ThreadLocalMode::InitialExecTLSModel),
            LLVMThreadLocalMode::LLVMLocalExecTLSModel => Some(ThreadLocalMode::LocalExecTLSModel),
            LLVMThreadLocalMode::LLVMNotThreadLocal => None,
        }
    }

    pub(crate) fn as_llvm_mode(self) -> LLVMThreadLocalMode {
        match self {
            ThreadLocalMode::GeneralDynamicTLSModel => LLVMThreadLocalMode::LLVMGeneralDynamicTLSModel,
            ThreadLocalMode::LocalDynamicTLSModel => LLVMThreadLocalMode::LLVMLocalDynamicTLSModel,
            ThreadLocalMode::InitialExecTLSModel => LLVMThreadLocalMode::LLVMInitialExecTLSModel,
            ThreadLocalMode::LocalExecTLSModel => LLVMThreadLocalMode::LLVMLocalExecTLSModel,
        }
    }
}

#[llvm_enum(LLVMDLLStorageClass)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DLLStorageClass {
    #[llvm_variant(LLVMDefaultStorageClass)]
    Default,
    #[llvm_variant(LLVMDLLImportStorageClass)]
    Import,
    #[llvm_variant(LLVMDLLExportStorageClass)]
    Export,
}

impl Default for DLLStorageClass {
    fn default() -> Self {
        DLLStorageClass::Default
    }
}

#[llvm_enum(LLVMInlineAsmDialect)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum InlineAsmDialect {
    #[llvm_variant(LLVMInlineAsmDialectATT)]
    ATT,
    #[llvm_variant(LLVMInlineAsmDialectIntel)]
    Intel,
}
