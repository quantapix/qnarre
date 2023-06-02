#![deny(missing_debug_implementations)]
#![cfg_attr(feature = "nightly", feature(doc_cfg))]

#[macro_use]
extern crate inkwell_internals;

#[macro_use]
pub mod support;
#[deny(missing_docs)]
pub mod attributes;
#[deny(missing_docs)]
pub mod basic_block;
pub mod builder;
#[deny(missing_docs)]
#[cfg(not(any(feature = "llvm4-0", feature = "llvm5-0", feature = "llvm6-0")))]
pub mod comdat;
#[deny(missing_docs)]
pub mod context;
pub mod data_layout;
#[cfg(not(any(feature = "llvm4-0", feature = "llvm5-0", feature = "llvm6-0")))]
pub mod debug_info;
pub mod execution_engine;
pub mod intrinsics;
pub mod memory_buffer;
#[deny(missing_docs)]
pub mod module;
pub mod object_file;
pub mod passes;
pub mod targets;
pub mod types;
pub mod values;

#[cfg(feature = "llvm10-0")]
extern crate llvm_lib_100 as llvm_lib;
#[cfg(feature = "llvm11-0")]
extern crate llvm_lib_110 as llvm_lib;
#[cfg(feature = "llvm12-0")]
extern crate llvm_lib_120 as llvm_lib;
#[cfg(feature = "llvm13-0")]
extern crate llvm_lib_130 as llvm_lib;
#[cfg(feature = "llvm14-0")]
extern crate llvm_lib_140 as llvm_lib;
#[cfg(feature = "llvm15-0")]
extern crate llvm_lib_150 as llvm_lib;
#[cfg(feature = "llvm16-0")]
extern crate llvm_lib_160 as llvm_lib;
#[cfg(feature = "llvm4-0")]
extern crate llvm_lib_40 as llvm_lib;
#[cfg(feature = "llvm5-0")]
extern crate llvm_lib_50 as llvm_lib;
#[cfg(feature = "llvm6-0")]
extern crate llvm_lib_60 as llvm_lib;
#[cfg(feature = "llvm7-0")]
extern crate llvm_lib_70 as llvm_lib;
#[cfg(feature = "llvm8-0")]
extern crate llvm_lib_80 as llvm_lib;
#[cfg(feature = "llvm9-0")]
extern crate llvm_lib_90 as llvm_lib;

use llvm_lib::{
    LLVMAtomicOrdering, LLVMAtomicRMWBinOp, LLVMDLLStorageClass, LLVMIntPredicate, LLVMRealPredicate,
    LLVMThreadLocalMode, LLVMVisibility,
};

#[llvm_versions(7.0..=latest)]
use llvm_lib::LLVMInlineAsmDialect;

use std::convert::TryFrom;

macro_rules! assert_unique_features {
    () => {};
    ($first:tt $(,$rest:tt)*) => {
        $(
            #[cfg(all(feature = $first, feature = $rest))]
            compile_error!(concat!("features \"", $first, "\" and \"", $rest, "\" cannot be used together"));
        )*
        assert_unique_features!($($rest),*);
    }
}

macro_rules! assert_used_features {
    ($($all:tt),*) => {
        #[cfg(not(any($(feature = $all),*)))]
        compile_error!(concat!("One of the LLVM feature flags must be provided: ", $($all, " "),*));
    }
}

macro_rules! assert_unique_used_features {
    ($($all:tt),*) => {
        assert_unique_features!($($all),*);
        assert_used_features!($($all),*);
    }
}

assert_unique_used_features! {"llvm4-0", "llvm5-0", "llvm6-0", "llvm7-0", "llvm8-0", "llvm9-0", "llvm10-0", "llvm11-0", "llvm12-0", "llvm13-0", "llvm14-0", "llvm15-0", "llvm16-0"}

///
///
///
///
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

    #[llvm_versions(10.0..=latest)]
    #[llvm_variant(LLVMAtomicRMWBinOpFAdd)]
    FAdd,

    #[llvm_versions(10.0..=latest)]
    #[llvm_variant(LLVMAtomicRMWBinOpFSub)]
    FSub,

    #[llvm_versions(15.0..=latest)]
    #[llvm_variant(LLVMAtomicRMWBinOpFMax)]
    FMax,

    #[llvm_versions(15.0..=latest)]
    #[llvm_variant(LLVMAtomicRMWBinOpFMin)]
    FMin,
}

///
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

#[llvm_versions(7.0..=latest)]
#[llvm_enum(LLVMInlineAsmDialect)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum InlineAsmDialect {
    #[llvm_variant(LLVMInlineAsmDialectATT)]
    ATT,
    #[llvm_variant(LLVMInlineAsmDialectIntel)]
    Intel,
}
