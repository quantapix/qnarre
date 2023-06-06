use llvm_lib::core::*;
use llvm_lib::initialization::*;
use llvm_lib::prelude::{LLVMPassManagerRef, LLVMPassRegistryRef};
use llvm_lib::transforms::ipo::*;
use llvm_lib::transforms::pass_builder::*;
use llvm_lib::transforms::pass_manager_builder::*;
use llvm_lib::transforms::scalar::*;
use llvm_lib::transforms::vectorize::*;

use crate::module::Module;
use crate::val::{AsValueRef, FunctionValue};
use crate::OptimizationLevel;
use std::borrow::Borrow;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct PassManagerBuilder {
    pass_manager_builder: LLVMPassManagerBuilderRef,
}
impl PassManagerBuilder {
    pub unsafe fn new(pass_manager_builder: LLVMPassManagerBuilderRef) -> Self {
        assert!(!pass_manager_builder.is_null());
        PassManagerBuilder { pass_manager_builder }
    }
    pub fn as_mut_ptr(&self) -> LLVMPassManagerBuilderRef {
        self.pass_manager_builder
    }
    pub fn create() -> Self {
        let pass_manager_builder = unsafe { LLVMPassManagerBuilderCreate() };
        unsafe { PassManagerBuilder::new(pass_manager_builder) }
    }
    pub fn set_optimization_level(&self, opt_level: OptimizationLevel) {
        unsafe { LLVMPassManagerBuilderSetOptLevel(self.pass_manager_builder, opt_level as u32) }
    }
    pub fn set_size_level(&self, size_level: u32) {
        unsafe { LLVMPassManagerBuilderSetSizeLevel(self.pass_manager_builder, size_level) }
    }
    pub fn set_disable_unit_at_a_time(&self, disable: bool) {
        unsafe { LLVMPassManagerBuilderSetDisableUnitAtATime(self.pass_manager_builder, disable as i32) }
    }
    pub fn set_disable_unroll_loops(&self, disable: bool) {
        unsafe { LLVMPassManagerBuilderSetDisableUnrollLoops(self.pass_manager_builder, disable as i32) }
    }
    pub fn set_disable_simplify_lib_calls(&self, disable: bool) {
        unsafe { LLVMPassManagerBuilderSetDisableSimplifyLibCalls(self.pass_manager_builder, disable as i32) }
    }
    pub fn set_inliner_with_threshold(&self, threshold: u32) {
        unsafe { LLVMPassManagerBuilderUseInlinerWithThreshold(self.pass_manager_builder, threshold) }
    }
    pub fn populate_function_pass_manager(&self, pass_manager: &PassManager<FunctionValue>) {
        unsafe {
            LLVMPassManagerBuilderPopulateFunctionPassManager(self.pass_manager_builder, pass_manager.pass_manager)
        }
    }
    pub fn populate_module_pass_manager(&self, pass_manager: &PassManager<Module>) {
        unsafe { LLVMPassManagerBuilderPopulateModulePassManager(self.pass_manager_builder, pass_manager.pass_manager) }
    }
}
impl Drop for PassManagerBuilder {
    fn drop(&mut self) {
        unsafe { LLVMPassManagerBuilderDispose(self.pass_manager_builder) }
    }
}
pub trait PassManagerSubType {
    type Input;
    unsafe fn create<I: Borrow<Self::Input>>(input: I) -> LLVMPassManagerRef;
    unsafe fn run_in_pass_manager(&self, pass_manager: &PassManager<Self>) -> bool
    where
        Self: Sized;
}
impl PassManagerSubType for Module<'_> {
    type Input = ();
    unsafe fn create<I: Borrow<Self::Input>>(_: I) -> LLVMPassManagerRef {
        LLVMCreatePassManager()
    }
    unsafe fn run_in_pass_manager(&self, pass_manager: &PassManager<Self>) -> bool {
        LLVMRunPassManager(pass_manager.pass_manager, self.module.get()) == 1
    }
}
impl<'ctx> PassManagerSubType for FunctionValue<'ctx> {
    type Input = Module<'ctx>;
    unsafe fn create<I: Borrow<Self::Input>>(input: I) -> LLVMPassManagerRef {
        LLVMCreateFunctionPassManagerForModule(input.borrow().module.get())
    }
    unsafe fn run_in_pass_manager(&self, pass_manager: &PassManager<Self>) -> bool {
        LLVMRunFunctionPassManager(pass_manager.pass_manager, self.as_value_ref()) == 1
    }
}

#[derive(Debug)]
pub struct PassManager<T> {
    pub(crate) pass_manager: LLVMPassManagerRef,
    sub_type: PhantomData<T>,
}
impl PassManager<FunctionValue<'_>> {
    pub fn as_mut_ptr(&self) -> LLVMPassManagerRef {
        self.pass_manager
    }
    pub fn initialize(&self) -> bool {
        unsafe { LLVMInitializeFunctionPassManager(self.pass_manager) == 1 }
    }
    pub fn finalize(&self) -> bool {
        unsafe { LLVMFinalizeFunctionPassManager(self.pass_manager) == 1 }
    }
}
impl<T: PassManagerSubType> PassManager<T> {
    pub unsafe fn new(pass_manager: LLVMPassManagerRef) -> Self {
        assert!(!pass_manager.is_null());
        PassManager {
            pass_manager,
            sub_type: PhantomData,
        }
    }
    pub fn create<I: Borrow<T::Input>>(input: I) -> PassManager<T> {
        let pass_manager = unsafe { T::create(input) };
        unsafe { PassManager::new(pass_manager) }
    }
    pub fn run_on(&self, input: &T) -> bool {
        unsafe { input.run_in_pass_manager(self) }
    }
    pub fn add_constant_merge_pass(&self) {
        unsafe { LLVMAddConstantMergePass(self.pass_manager) }
    }
    pub fn add_merge_functions_pass(&self) {
        unsafe { LLVMAddMergeFunctionsPass(self.pass_manager) }
    }
    pub fn add_dead_arg_elimination_pass(&self) {
        unsafe { LLVMAddDeadArgEliminationPass(self.pass_manager) }
    }
    pub fn add_function_attrs_pass(&self) {
        unsafe { LLVMAddFunctionAttrsPass(self.pass_manager) }
    }
    pub fn add_function_inlining_pass(&self) {
        unsafe { LLVMAddFunctionInliningPass(self.pass_manager) }
    }
    pub fn add_always_inliner_pass(&self) {
        unsafe { LLVMAddAlwaysInlinerPass(self.pass_manager) }
    }
    pub fn add_global_dce_pass(&self) {
        unsafe { LLVMAddGlobalDCEPass(self.pass_manager) }
    }
    pub fn add_global_optimizer_pass(&self) {
        unsafe { LLVMAddGlobalOptimizerPass(self.pass_manager) }
    }
    pub fn add_ipsccp_pass(&self) {
        unsafe { LLVMAddIPSCCPPass(self.pass_manager) }
    }
    pub fn add_internalize_pass(&self, all_but_main: bool) {
        unsafe { LLVMAddInternalizePass(self.pass_manager, all_but_main as u32) }
    }
    pub fn add_strip_dead_prototypes_pass(&self) {
        unsafe { LLVMAddStripDeadPrototypesPass(self.pass_manager) }
    }
    pub fn add_strip_symbol_pass(&self) {
        unsafe { LLVMAddStripSymbolsPass(self.pass_manager) }
    }
    #[cfg(feature = "llvm4-0")]
    pub fn add_bb_vectorize_pass(&self) {
        use llvm_lib::transforms::vectorize::LLVMAddBBVectorizePass;
        unsafe { LLVMAddBBVectorizePass(self.pass_manager) }
    }
    pub fn add_loop_vectorize_pass(&self) {
        unsafe { LLVMAddLoopVectorizePass(self.pass_manager) }
    }
    pub fn add_slp_vectorize_pass(&self) {
        unsafe { LLVMAddSLPVectorizePass(self.pass_manager) }
    }
    pub fn add_aggressive_dce_pass(&self) {
        unsafe { LLVMAddAggressiveDCEPass(self.pass_manager) }
    }
    pub fn add_bit_tracking_dce_pass(&self) {
        unsafe { LLVMAddBitTrackingDCEPass(self.pass_manager) }
    }
    pub fn add_alignment_from_assumptions_pass(&self) {
        unsafe { LLVMAddAlignmentFromAssumptionsPass(self.pass_manager) }
    }
    pub fn add_cfg_simplification_pass(&self) {
        unsafe { LLVMAddCFGSimplificationPass(self.pass_manager) }
    }
    pub fn add_dead_store_elimination_pass(&self) {
        unsafe { LLVMAddDeadStoreEliminationPass(self.pass_manager) }
    }
    pub fn add_scalarizer_pass(&self) {
        unsafe { LLVMAddScalarizerPass(self.pass_manager) }
    }
    pub fn add_merged_load_store_motion_pass(&self) {
        unsafe { LLVMAddMergedLoadStoreMotionPass(self.pass_manager) }
    }
    pub fn add_gvn_pass(&self) {
        unsafe { LLVMAddGVNPass(self.pass_manager) }
    }
    pub fn add_new_gvn_pass(&self) {
        use llvm_lib::transforms::scalar::LLVMAddNewGVNPass;
        unsafe { LLVMAddNewGVNPass(self.pass_manager) }
    }
    pub fn add_ind_var_simplify_pass(&self) {
        unsafe { LLVMAddIndVarSimplifyPass(self.pass_manager) }
    }
    pub fn add_instruction_combining_pass(&self) {
        unsafe { LLVMAddInstructionCombiningPass(self.pass_manager) }
    }
    pub fn add_jump_threading_pass(&self) {
        unsafe { LLVMAddJumpThreadingPass(self.pass_manager) }
    }
    pub fn add_licm_pass(&self) {
        unsafe { LLVMAddLICMPass(self.pass_manager) }
    }
    pub fn add_loop_deletion_pass(&self) {
        unsafe { LLVMAddLoopDeletionPass(self.pass_manager) }
    }
    pub fn add_loop_idiom_pass(&self) {
        unsafe { LLVMAddLoopIdiomPass(self.pass_manager) }
    }
    pub fn add_loop_rotate_pass(&self) {
        unsafe { LLVMAddLoopRotatePass(self.pass_manager) }
    }
    pub fn add_loop_reroll_pass(&self) {
        unsafe { LLVMAddLoopRerollPass(self.pass_manager) }
    }
    pub fn add_loop_unroll_pass(&self) {
        unsafe { LLVMAddLoopUnrollPass(self.pass_manager) }
    }
    pub fn add_memcpy_optimize_pass(&self) {
        unsafe { LLVMAddMemCpyOptPass(self.pass_manager) }
    }
    pub fn add_partially_inline_lib_calls_pass(&self) {
        unsafe { LLVMAddPartiallyInlineLibCallsPass(self.pass_manager) }
    }
    pub fn add_lower_switch_pass(&self) {
        use llvm_lib::transforms::util::LLVMAddLowerSwitchPass;
        unsafe { LLVMAddLowerSwitchPass(self.pass_manager) }
    }
    pub fn add_promote_memory_to_register_pass(&self) {
        use llvm_lib::transforms::util::LLVMAddPromoteMemoryToRegisterPass;
        unsafe { LLVMAddPromoteMemoryToRegisterPass(self.pass_manager) }
    }
    pub fn add_reassociate_pass(&self) {
        unsafe { LLVMAddReassociatePass(self.pass_manager) }
    }
    pub fn add_sccp_pass(&self) {
        unsafe { LLVMAddSCCPPass(self.pass_manager) }
    }
    pub fn add_scalar_repl_aggregates_pass(&self) {
        unsafe { LLVMAddScalarReplAggregatesPass(self.pass_manager) }
    }
    pub fn add_scalar_repl_aggregates_pass_ssa(&self) {
        unsafe { LLVMAddScalarReplAggregatesPassSSA(self.pass_manager) }
    }
    pub fn add_scalar_repl_aggregates_pass_with_threshold(&self, threshold: i32) {
        unsafe { LLVMAddScalarReplAggregatesPassWithThreshold(self.pass_manager, threshold) }
    }
    pub fn add_simplify_lib_calls_pass(&self) {
        unsafe { LLVMAddSimplifyLibCallsPass(self.pass_manager) }
    }
    pub fn add_tail_call_elimination_pass(&self) {
        unsafe { LLVMAddTailCallEliminationPass(self.pass_manager) }
    }
    pub fn add_instruction_simplify_pass(&self) {
        unsafe { LLVMAddInstructionSimplifyPass(self.pass_manager) }
    }
    pub fn add_demote_memory_to_register_pass(&self) {
        unsafe { LLVMAddDemoteMemoryToRegisterPass(self.pass_manager) }
    }
    pub fn add_verifier_pass(&self) {
        unsafe { LLVMAddVerifierPass(self.pass_manager) }
    }
    pub fn add_correlated_value_propagation_pass(&self) {
        unsafe { LLVMAddCorrelatedValuePropagationPass(self.pass_manager) }
    }
    pub fn add_early_cse_pass(&self) {
        unsafe { LLVMAddEarlyCSEPass(self.pass_manager) }
    }
    pub fn add_early_cse_mem_ssa_pass(&self) {
        use llvm_lib::transforms::scalar::LLVMAddEarlyCSEMemSSAPass;
        unsafe { LLVMAddEarlyCSEMemSSAPass(self.pass_manager) }
    }
    pub fn add_lower_expect_intrinsic_pass(&self) {
        unsafe { LLVMAddLowerExpectIntrinsicPass(self.pass_manager) }
    }
    pub fn add_type_based_alias_analysis_pass(&self) {
        unsafe { LLVMAddTypeBasedAliasAnalysisPass(self.pass_manager) }
    }
    pub fn add_scoped_no_alias_aa_pass(&self) {
        unsafe { LLVMAddScopedNoAliasAAPass(self.pass_manager) }
    }
    pub fn add_basic_alias_analysis_pass(&self) {
        unsafe { LLVMAddBasicAliasAnalysisPass(self.pass_manager) }
    }
    pub fn add_loop_unroll_and_jam_pass(&self) {
        use llvm_lib::transforms::scalar::LLVMAddLoopUnrollAndJamPass;
        unsafe { LLVMAddLoopUnrollAndJamPass(self.pass_manager) }
    }
}
impl<T> Drop for PassManager<T> {
    fn drop(&mut self) {
        unsafe { LLVMDisposePassManager(self.pass_manager) }
    }
}

#[derive(Debug)]
pub struct PassRegistry {
    pass_registry: LLVMPassRegistryRef,
}
impl PassRegistry {
    pub unsafe fn new(pass_registry: LLVMPassRegistryRef) -> PassRegistry {
        assert!(!pass_registry.is_null());
        PassRegistry { pass_registry }
    }
    pub fn as_mut_ptr(&self) -> LLVMPassRegistryRef {
        self.pass_registry
    }
    pub fn get_global() -> PassRegistry {
        let pass_registry = unsafe { LLVMGetGlobalPassRegistry() };
        unsafe { PassRegistry::new(pass_registry) }
    }
    pub fn initialize_core(&self) {
        unsafe { LLVMInitializeCore(self.pass_registry) }
    }
    pub fn initialize_transform_utils(&self) {
        unsafe { LLVMInitializeTransformUtils(self.pass_registry) }
    }
    pub fn initialize_scalar_opts(&self) {
        unsafe { LLVMInitializeScalarOpts(self.pass_registry) }
    }
    pub fn initialize_vectorization(&self) {
        unsafe { LLVMInitializeVectorization(self.pass_registry) }
    }
    pub fn initialize_inst_combine(&self) {
        unsafe { LLVMInitializeInstCombine(self.pass_registry) }
    }
    pub fn initialize_ipo(&self) {
        unsafe { LLVMInitializeIPO(self.pass_registry) }
    }
    pub fn initialize_analysis(&self) {
        unsafe { LLVMInitializeAnalysis(self.pass_registry) }
    }
    pub fn initialize_ipa(&self) {
        unsafe { LLVMInitializeIPA(self.pass_registry) }
    }
    pub fn initialize_codegen(&self) {
        unsafe { LLVMInitializeCodeGen(self.pass_registry) }
    }
    pub fn initialize_target(&self) {
        unsafe { LLVMInitializeTarget(self.pass_registry) }
    }
}

#[derive(Debug)]
pub struct PassBuilderOptions {
    pub(crate) options_ref: LLVMPassBuilderOptionsRef,
}
impl PassBuilderOptions {
    pub fn create() -> Self {
        unsafe {
            PassBuilderOptions {
                options_ref: LLVMCreatePassBuilderOptions(),
            }
        }
    }
    pub fn as_mut_ptr(&self) -> LLVMPassBuilderOptionsRef {
        self.options_ref
    }
    pub fn set_verify_each(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetVerifyEach(self.options_ref, value as i32);
        }
    }
    pub fn set_debug_logging(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetDebugLogging(self.options_ref, value as i32);
        }
    }
    pub fn set_loop_interleaving(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetLoopInterleaving(self.options_ref, value as i32);
        }
    }
    pub fn set_loop_vectorization(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetLoopVectorization(self.options_ref, value as i32);
        }
    }
    pub fn set_loop_slp_vectorization(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetSLPVectorization(self.options_ref, value as i32);
        }
    }
    pub fn set_loop_unrolling(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetLoopUnrolling(self.options_ref, value as i32);
        }
    }
    pub fn set_forget_all_scev_in_loop_unroll(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll(self.options_ref, value as i32);
        }
    }
    pub fn set_licm_mssa_opt_cap(&self, value: u32) {
        unsafe {
            LLVMPassBuilderOptionsSetLicmMssaOptCap(self.options_ref, value);
        }
    }
    pub fn set_licm_mssa_no_acc_for_promotion_cap(&self, value: u32) {
        unsafe {
            LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap(self.options_ref, value);
        }
    }
    pub fn set_call_graph_profile(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetCallGraphProfile(self.options_ref, value as i32);
        }
    }
    pub fn set_merge_functions(&self, value: bool) {
        unsafe {
            LLVMPassBuilderOptionsSetMergeFunctions(self.options_ref, value as i32);
        }
    }
}
impl Drop for PassBuilderOptions {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposePassBuilderOptions(self.options_ref);
        }
    }
}
