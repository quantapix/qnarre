#![allow(non_snake_case)]

use super::prelude::*;
use error::LLVMErrorRef;
use target_machine::LLVMTargetMachineRef;

#[derive(Debug)]
pub enum LLVMOpaquePassBuilderOptions {}
pub type LLVMPassBuilderOptionsRef = *mut LLVMOpaquePassBuilderOptions;

extern "C" {
    pub fn LLVMCreatePassBuilderOptions() -> LLVMPassBuilderOptionsRef;
    pub fn LLVMDisposePassBuilderOptions(Options: LLVMPassBuilderOptionsRef);
    pub fn LLVMPassBuilderOptionsSetCallGraphProfile(Options: LLVMPassBuilderOptionsRef, CallGraphProfile: LLVMBool);
    pub fn LLVMPassBuilderOptionsSetDebugLogging(Options: LLVMPassBuilderOptionsRef, DebugLogging: LLVMBool);
    pub fn LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll(Options: LLVMPassBuilderOptionsRef, ForgetAllSCEVInLoopUnroll: LLVMBool);
    pub fn LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap(Options: LLVMPassBuilderOptionsRef, LicmMssaNoAccForPromotionCap: ::libc::c_uint);
    pub fn LLVMPassBuilderOptionsSetLicmMssaOptCap(Options: LLVMPassBuilderOptionsRef, LicmMssaOptCap: ::libc::c_uint);
    pub fn LLVMPassBuilderOptionsSetLoopInterleaving(Options: LLVMPassBuilderOptionsRef, LoopInterleaving: LLVMBool);
    pub fn LLVMPassBuilderOptionsSetLoopUnrolling(Options: LLVMPassBuilderOptionsRef, LoopUnrolling: LLVMBool);
    pub fn LLVMPassBuilderOptionsSetLoopVectorization(Options: LLVMPassBuilderOptionsRef, LoopVectorization: LLVMBool);
    pub fn LLVMPassBuilderOptionsSetMergeFunctions(Options: LLVMPassBuilderOptionsRef, MergeFunctions: LLVMBool);
    pub fn LLVMPassBuilderOptionsSetSLPVectorization(Options: LLVMPassBuilderOptionsRef, SLPVectorization: LLVMBool);
    pub fn LLVMPassBuilderOptionsSetVerifyEach(Options: LLVMPassBuilderOptionsRef, VerifyEach: LLVMBool);
    pub fn LLVMRunPasses(M: LLVMModuleRef, Passes: *const ::libc::c_char, TM: LLVMTargetMachineRef, Options: LLVMPassBuilderOptionsRef) -> LLVMErrorRef;
}

#[derive(Debug)]
pub enum LLVMOpaquePassManagerBuilder {}

pub type LLVMPassManagerBuilderRef = *mut LLVMOpaquePassManagerBuilder;

extern "C" {
    pub fn LLVMPassManagerBuilderCreate() -> LLVMPassManagerBuilderRef;
    pub fn LLVMPassManagerBuilderDispose(PMB: LLVMPassManagerBuilderRef);
    pub fn LLVMPassManagerBuilderPopulateFunctionPassManager(PMB: LLVMPassManagerBuilderRef, PM: LLVMPassManagerRef);
    pub fn LLVMPassManagerBuilderPopulateModulePassManager(PMB: LLVMPassManagerBuilderRef, PM: LLVMPassManagerRef);
    pub fn LLVMPassManagerBuilderSetDisableSimplifyLibCalls(PMB: LLVMPassManagerBuilderRef, Value: LLVMBool);
    pub fn LLVMPassManagerBuilderSetDisableUnitAtATime(PMB: LLVMPassManagerBuilderRef, Value: LLVMBool);
    pub fn LLVMPassManagerBuilderSetDisableUnrollLoops(PMB: LLVMPassManagerBuilderRef, Value: LLVMBool);
    pub fn LLVMPassManagerBuilderSetOptLevel(PMB: LLVMPassManagerBuilderRef, OptLevel: ::libc::c_uint);
    pub fn LLVMPassManagerBuilderSetSizeLevel(PMB: LLVMPassManagerBuilderRef, SizeLevel: ::libc::c_uint);
    pub fn LLVMPassManagerBuilderUseInlinerWithThreshold(PMB: LLVMPassManagerBuilderRef, Threshold: ::libc::c_uint);
}

extern "C" {
    pub fn LLVMAddLowerSwitchPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddPromoteMemoryToRegisterPass(PM: LLVMPassManagerRef);
}

extern "C" {
    pub fn LLVMAddInstructionCombiningPass(PM: LLVMPassManagerRef);
}

extern "C" {
    pub fn LLVMAddAlwaysInlinerPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddCalledValuePropagationPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddConstantMergePass(PM: LLVMPassManagerRef);
    pub fn LLVMAddDeadArgEliminationPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddFunctionAttrsPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddFunctionInliningPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddGlobalDCEPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddGlobalOptimizerPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddInternalizePass(arg1: LLVMPassManagerRef, AllButMain: ::libc::c_uint);
    pub fn LLVMAddInternalizePassWithMustPreservePredicate(PM: LLVMPassManagerRef, Context: *mut ::libc::c_void, MustPreserve: Option<extern "C" fn(LLVMValueRef, *mut ::libc::c_void) -> LLVMBool>);
    pub fn LLVMAddIPSCCPPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddMergeFunctionsPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddStripDeadPrototypesPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddStripSymbolsPass(PM: LLVMPassManagerRef);
}

extern "C" {
    pub fn LLVMAddAggressiveDCEPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddAlignmentFromAssumptionsPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddBasicAliasAnalysisPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddBitTrackingDCEPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddCFGSimplificationPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddCorrelatedValuePropagationPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddDCEPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddDeadStoreEliminationPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddDemoteMemoryToRegisterPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddEarlyCSEMemSSAPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddEarlyCSEPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddGVNPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddIndVarSimplifyPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddInstructionCombiningPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddInstructionSimplifyPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddJumpThreadingPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddLICMPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddLoopDeletionPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddLoopIdiomPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddLoopRerollPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddLoopRotatePass(PM: LLVMPassManagerRef);
    pub fn LLVMAddLoopUnrollAndJamPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddLoopUnrollPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddLowerAtomicPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddLowerConstantIntrinsicsPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddLowerExpectIntrinsicPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddMemCpyOptPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddMergedLoadStoreMotionPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddNewGVNPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddPartiallyInlineLibCallsPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddReassociatePass(PM: LLVMPassManagerRef);
    pub fn LLVMAddScalarizerPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddScalarReplAggregatesPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddScalarReplAggregatesPassSSA(PM: LLVMPassManagerRef);
    pub fn LLVMAddScalarReplAggregatesPassWithThreshold(PM: LLVMPassManagerRef, Threshold: ::libc::c_int);
    pub fn LLVMAddSCCPPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddScopedNoAliasAAPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddSimplifyLibCallsPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddTailCallEliminationPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddTypeBasedAliasAnalysisPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddUnifyFunctionExitNodesPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddVerifierPass(PM: LLVMPassManagerRef);
}

extern "C" {
    pub fn LLVMAddLowerSwitchPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddPromoteMemoryToRegisterPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddAddDiscriminatorsPass(PM: LLVMPassManagerRef);
}

extern "C" {
    pub fn LLVMAddLoopVectorizePass(PM: LLVMPassManagerRef);
    pub fn LLVMAddSLPVectorizePass(PM: LLVMPassManagerRef);
}
