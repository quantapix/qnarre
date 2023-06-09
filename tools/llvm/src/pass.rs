use llvm_lib::core::*;
use llvm_lib::initialization::*;
use llvm_lib::prelude::{LLVMPassManagerRef, LLVMPassRegistryRef};
use llvm_lib::transforms::ipo::*;
use llvm_lib::transforms::pass_builder::*;
use llvm_lib::transforms::pass_manager_builder::*;
use llvm_lib::transforms::scalar::*;
use llvm_lib::transforms::vectorize::*;
use std::borrow::Borrow;
use std::marker::PhantomData;

use crate::val::{AsValueRef, FunctionValue};
use crate::Module;
use crate::OptimizationLevel;

#[derive(Debug)]
pub struct PassManagerBuilder {
    raw: LLVMPassManagerBuilderRef,
}
impl PassManagerBuilder {
    pub unsafe fn new(x: LLVMPassManagerBuilderRef) -> Self {
        assert!(!x.is_null());
        PassManagerBuilder { raw: x }
    }
    pub fn as_mut_ptr(&self) -> LLVMPassManagerBuilderRef {
        self.raw
    }
    pub fn create() -> Self {
        let y = unsafe { LLVMPassManagerBuilderCreate() };
        unsafe { PassManagerBuilder::new(y) }
    }
    pub fn set_optimization_level(&self, x: OptimizationLevel) {
        unsafe { LLVMPassManagerBuilderSetOptLevel(self.raw, x as u32) }
    }
    pub fn set_size_level(&self, size_level: u32) {
        unsafe { LLVMPassManagerBuilderSetSizeLevel(self.raw, size_level) }
    }
    pub fn set_disable_unit_at_a_time(&self, disable: bool) {
        unsafe { LLVMPassManagerBuilderSetDisableUnitAtATime(self.raw, disable as i32) }
    }
    pub fn set_disable_unroll_loops(&self, disable: bool) {
        unsafe { LLVMPassManagerBuilderSetDisableUnrollLoops(self.raw, disable as i32) }
    }
    pub fn set_disable_simplify_lib_calls(&self, disable: bool) {
        unsafe { LLVMPassManagerBuilderSetDisableSimplifyLibCalls(self.raw, disable as i32) }
    }
    pub fn set_inliner_with_threshold(&self, threshold: u32) {
        unsafe { LLVMPassManagerBuilderUseInlinerWithThreshold(self.raw, threshold) }
    }
    pub fn populate_function_pass_manager(&self, x: &PassManager<FunctionValue>) {
        unsafe { LLVMPassManagerBuilderPopulateFunctionPassManager(self.raw, x.raw) }
    }
    pub fn populate_module_pass_manager(&self, x: &PassManager<Module>) {
        unsafe { LLVMPassManagerBuilderPopulateModulePassManager(self.raw, x.raw) }
    }
}
impl Drop for PassManagerBuilder {
    fn drop(&mut self) {
        unsafe { LLVMPassManagerBuilderDispose(self.raw) }
    }
}
pub trait PassManagerSubType {
    type Input;
    unsafe fn create<I: Borrow<Self::Input>>(x: I) -> LLVMPassManagerRef;
    unsafe fn run_in_pass_manager(&self, x: &PassManager<Self>) -> bool
    where
        Self: Sized;
}
impl PassManagerSubType for Module<'_> {
    type Input = ();
    unsafe fn create<I: Borrow<Self::Input>>(_: I) -> LLVMPassManagerRef {
        LLVMCreatePassManager()
    }
    unsafe fn run_in_pass_manager(&self, x: &PassManager<Self>) -> bool {
        LLVMRunPassManager(x.raw, self.module.get()) == 1
    }
}
impl<'ctx> PassManagerSubType for FunctionValue<'ctx> {
    type Input = Module<'ctx>;
    unsafe fn create<I: Borrow<Self::Input>>(x: I) -> LLVMPassManagerRef {
        LLVMCreateFunctionPassManagerForModule(x.borrow().module.get())
    }
    unsafe fn run_in_pass_manager(&self, x: &PassManager<Self>) -> bool {
        LLVMRunFunctionPassManager(x.raw, self.as_value_ref()) == 1
    }
}

#[derive(Debug)]
pub struct PassManager<T> {
    pub raw: LLVMPassManagerRef,
    sub_type: PhantomData<T>,
}
impl PassManager<FunctionValue<'_>> {
    pub fn as_mut_ptr(&self) -> LLVMPassManagerRef {
        self.raw
    }
    pub fn initialize(&self) -> bool {
        unsafe { LLVMInitializeFunctionPassManager(self.raw) == 1 }
    }
    pub fn finalize(&self) -> bool {
        unsafe { LLVMFinalizeFunctionPassManager(self.raw) == 1 }
    }
}
impl<T: PassManagerSubType> PassManager<T> {
    pub unsafe fn new(raw: LLVMPassManagerRef) -> Self {
        assert!(!raw.is_null());
        PassManager {
            raw,
            sub_type: PhantomData,
        }
    }
    pub fn create<I: Borrow<T::Input>>(x: I) -> PassManager<T> {
        let y = unsafe { T::create(x) };
        unsafe { PassManager::new(y) }
    }
    pub fn run_on(&self, x: &T) -> bool {
        unsafe { x.run_in_pass_manager(self) }
    }
    pub fn add_constant_merge_pass(&self) {
        unsafe { LLVMAddConstantMergePass(self.raw) }
    }
    pub fn add_merge_functions_pass(&self) {
        unsafe { LLVMAddMergeFunctionsPass(self.raw) }
    }
    pub fn add_dead_arg_elimination_pass(&self) {
        unsafe { LLVMAddDeadArgEliminationPass(self.raw) }
    }
    pub fn add_function_attrs_pass(&self) {
        unsafe { LLVMAddFunctionAttrsPass(self.raw) }
    }
    pub fn add_function_inlining_pass(&self) {
        unsafe { LLVMAddFunctionInliningPass(self.raw) }
    }
    pub fn add_always_inliner_pass(&self) {
        unsafe { LLVMAddAlwaysInlinerPass(self.raw) }
    }
    pub fn add_global_dce_pass(&self) {
        unsafe { LLVMAddGlobalDCEPass(self.raw) }
    }
    pub fn add_global_optimizer_pass(&self) {
        unsafe { LLVMAddGlobalOptimizerPass(self.raw) }
    }
    pub fn add_ipsccp_pass(&self) {
        unsafe { LLVMAddIPSCCPPass(self.raw) }
    }
    pub fn add_internalize_pass(&self, all_but_main: bool) {
        unsafe { LLVMAddInternalizePass(self.raw, all_but_main as u32) }
    }
    pub fn add_strip_dead_prototypes_pass(&self) {
        unsafe { LLVMAddStripDeadPrototypesPass(self.raw) }
    }
    pub fn add_strip_symbol_pass(&self) {
        unsafe { LLVMAddStripSymbolsPass(self.raw) }
    }
    pub fn add_loop_vectorize_pass(&self) {
        unsafe { LLVMAddLoopVectorizePass(self.raw) }
    }
    pub fn add_slp_vectorize_pass(&self) {
        unsafe { LLVMAddSLPVectorizePass(self.raw) }
    }
    pub fn add_aggressive_dce_pass(&self) {
        unsafe { LLVMAddAggressiveDCEPass(self.raw) }
    }
    pub fn add_bit_tracking_dce_pass(&self) {
        unsafe { LLVMAddBitTrackingDCEPass(self.raw) }
    }
    pub fn add_alignment_from_assumptions_pass(&self) {
        unsafe { LLVMAddAlignmentFromAssumptionsPass(self.raw) }
    }
    pub fn add_cfg_simplification_pass(&self) {
        unsafe { LLVMAddCFGSimplificationPass(self.raw) }
    }
    pub fn add_dead_store_elimination_pass(&self) {
        unsafe { LLVMAddDeadStoreEliminationPass(self.raw) }
    }
    pub fn add_scalarizer_pass(&self) {
        unsafe { LLVMAddScalarizerPass(self.raw) }
    }
    pub fn add_merged_load_store_motion_pass(&self) {
        unsafe { LLVMAddMergedLoadStoreMotionPass(self.raw) }
    }
    pub fn add_gvn_pass(&self) {
        unsafe { LLVMAddGVNPass(self.raw) }
    }
    pub fn add_new_gvn_pass(&self) {
        unsafe { LLVMAddNewGVNPass(self.raw) }
    }
    pub fn add_ind_var_simplify_pass(&self) {
        unsafe { LLVMAddIndVarSimplifyPass(self.raw) }
    }
    pub fn add_instruction_combining_pass(&self) {
        unsafe { LLVMAddInstructionCombiningPass(self.raw) }
    }
    pub fn add_jump_threading_pass(&self) {
        unsafe { LLVMAddJumpThreadingPass(self.raw) }
    }
    pub fn add_licm_pass(&self) {
        unsafe { LLVMAddLICMPass(self.raw) }
    }
    pub fn add_loop_deletion_pass(&self) {
        unsafe { LLVMAddLoopDeletionPass(self.raw) }
    }
    pub fn add_loop_idiom_pass(&self) {
        unsafe { LLVMAddLoopIdiomPass(self.raw) }
    }
    pub fn add_loop_rotate_pass(&self) {
        unsafe { LLVMAddLoopRotatePass(self.raw) }
    }
    pub fn add_loop_reroll_pass(&self) {
        unsafe { LLVMAddLoopRerollPass(self.raw) }
    }
    pub fn add_loop_unroll_pass(&self) {
        unsafe { LLVMAddLoopUnrollPass(self.raw) }
    }
    pub fn add_memcpy_optimize_pass(&self) {
        unsafe { LLVMAddMemCpyOptPass(self.raw) }
    }
    pub fn add_partially_inline_lib_calls_pass(&self) {
        unsafe { LLVMAddPartiallyInlineLibCallsPass(self.raw) }
    }
    pub fn add_lower_switch_pass(&self) {
        unsafe { LLVMAddLowerSwitchPass(self.raw) }
    }
    pub fn add_promote_memory_to_register_pass(&self) {
        unsafe { LLVMAddPromoteMemoryToRegisterPass(self.raw) }
    }
    pub fn add_reassociate_pass(&self) {
        unsafe { LLVMAddReassociatePass(self.raw) }
    }
    pub fn add_sccp_pass(&self) {
        unsafe { LLVMAddSCCPPass(self.raw) }
    }
    pub fn add_scalar_repl_aggregates_pass(&self) {
        unsafe { LLVMAddScalarReplAggregatesPass(self.raw) }
    }
    pub fn add_scalar_repl_aggregates_pass_ssa(&self) {
        unsafe { LLVMAddScalarReplAggregatesPassSSA(self.raw) }
    }
    pub fn add_scalar_repl_aggregates_pass_with_threshold(&self, threshold: i32) {
        unsafe { LLVMAddScalarReplAggregatesPassWithThreshold(self.raw, threshold) }
    }
    pub fn add_simplify_lib_calls_pass(&self) {
        unsafe { LLVMAddSimplifyLibCallsPass(self.raw) }
    }
    pub fn add_tail_call_elimination_pass(&self) {
        unsafe { LLVMAddTailCallEliminationPass(self.raw) }
    }
    pub fn add_instruction_simplify_pass(&self) {
        unsafe { LLVMAddInstructionSimplifyPass(self.raw) }
    }
    pub fn add_demote_memory_to_register_pass(&self) {
        unsafe { LLVMAddDemoteMemoryToRegisterPass(self.raw) }
    }
    pub fn add_verifier_pass(&self) {
        unsafe { LLVMAddVerifierPass(self.raw) }
    }
    pub fn add_correlated_value_propagation_pass(&self) {
        unsafe { LLVMAddCorrelatedValuePropagationPass(self.raw) }
    }
    pub fn add_early_cse_pass(&self) {
        unsafe { LLVMAddEarlyCSEPass(self.raw) }
    }
    pub fn add_early_cse_mem_ssa_pass(&self) {
        unsafe { LLVMAddEarlyCSEMemSSAPass(self.raw) }
    }
    pub fn add_lower_expect_intrinsic_pass(&self) {
        unsafe { LLVMAddLowerExpectIntrinsicPass(self.raw) }
    }
    pub fn add_type_based_alias_analysis_pass(&self) {
        unsafe { LLVMAddTypeBasedAliasAnalysisPass(self.raw) }
    }
    pub fn add_scoped_no_alias_aa_pass(&self) {
        unsafe { LLVMAddScopedNoAliasAAPass(self.raw) }
    }
    pub fn add_basic_alias_analysis_pass(&self) {
        unsafe { LLVMAddBasicAliasAnalysisPass(self.raw) }
    }
    pub fn add_loop_unroll_and_jam_pass(&self) {
        unsafe { LLVMAddLoopUnrollAndJamPass(self.raw) }
    }
}
impl<T> Drop for PassManager<T> {
    fn drop(&mut self) {
        unsafe { LLVMDisposePassManager(self.raw) }
    }
}

#[derive(Debug)]
pub struct PassRegistry {
    raw: LLVMPassRegistryRef,
}
impl PassRegistry {
    pub unsafe fn new(raw: LLVMPassRegistryRef) -> PassRegistry {
        assert!(!raw.is_null());
        PassRegistry { raw }
    }
    pub fn as_mut_ptr(&self) -> LLVMPassRegistryRef {
        self.raw
    }
    pub fn get_global() -> PassRegistry {
        let y = unsafe { LLVMGetGlobalPassRegistry() };
        unsafe { PassRegistry::new(y) }
    }
    pub fn initialize_core(&self) {
        unsafe { LLVMInitializeCore(self.raw) }
    }
    pub fn initialize_transform_utils(&self) {
        unsafe { LLVMInitializeTransformUtils(self.raw) }
    }
    pub fn initialize_scalar_opts(&self) {
        unsafe { LLVMInitializeScalarOpts(self.raw) }
    }
    pub fn initialize_vectorization(&self) {
        unsafe { LLVMInitializeVectorization(self.raw) }
    }
    pub fn initialize_inst_combine(&self) {
        unsafe { LLVMInitializeInstCombine(self.raw) }
    }
    pub fn initialize_ipo(&self) {
        unsafe { LLVMInitializeIPO(self.raw) }
    }
    pub fn initialize_analysis(&self) {
        unsafe { LLVMInitializeAnalysis(self.raw) }
    }
    pub fn initialize_ipa(&self) {
        unsafe { LLVMInitializeIPA(self.raw) }
    }
    pub fn initialize_codegen(&self) {
        unsafe { LLVMInitializeCodeGen(self.raw) }
    }
    pub fn initialize_target(&self) {
        unsafe { LLVMInitializeTarget(self.raw) }
    }
}

#[derive(Debug)]
pub struct PassBuilderOptions {
    pub raw: LLVMPassBuilderOptionsRef,
}
impl PassBuilderOptions {
    pub fn create() -> Self {
        unsafe {
            PassBuilderOptions {
                raw: LLVMCreatePassBuilderOptions(),
            }
        }
    }
    pub fn as_mut_ptr(&self) -> LLVMPassBuilderOptionsRef {
        self.raw
    }
    pub fn set_verify_each(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetVerifyEach(self.raw, value as i32);
        }
    }
    pub fn set_debug_logging(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetDebugLogging(self.raw, value as i32);
        }
    }
    pub fn set_loop_interleaving(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetLoopInterleaving(self.raw, value as i32);
        }
    }
    pub fn set_loop_vectorization(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetLoopVectorization(self.raw, value as i32);
        }
    }
    pub fn set_loop_slp_vectorization(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetSLPVectorization(self.raw, value as i32);
        }
    }
    pub fn set_loop_unrolling(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetLoopUnrolling(self.raw, value as i32);
        }
    }
    pub fn set_forget_all_scev_in_loop_unroll(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll(self.raw, value as i32);
        }
    }
    pub fn set_licm_mssa_opt_cap(&self, value: u32) {
        unsafe {
            LLVMPassBuilderOptionsSetLicmMssaOptCap(self.raw, value);
        }
    }
    pub fn set_licm_mssa_no_acc_for_promotion_cap(&self, value: u32) {
        unsafe {
            LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap(self.raw, value);
        }
    }
    pub fn set_call_graph_profile(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetCallGraphProfile(self.raw, value as i32);
        }
    }
    pub fn set_merge_functions(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetMergeFunctions(self.raw, value as i32);
        }
    }
}
impl Drop for PassBuilderOptions {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposePassBuilderOptions(self.raw);
        }
    }
}
