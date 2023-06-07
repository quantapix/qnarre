use crate::{ctx::Context, ir::Module, pass::Pass, Error, LogicalResult, StringRef};
use mlir_lib::*;
use std::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
};

pub struct Pass {
    raw: MlirPass,
}
impl Pass {
    pub unsafe fn from_raw_fn(create_raw: unsafe extern "C" fn() -> MlirPass) -> Self {
        Self {
            raw: unsafe { create_raw() },
        }
    }
    pub fn to_raw(&self) -> MlirPass {
        self.raw
    }
    #[doc(hidden)]
    pub unsafe fn __private_from_raw_fn(create_raw: unsafe extern "C" fn() -> MlirPass) -> Self {
        Self::from_raw_fn(create_raw)
    }
}

pub struct PassManager<'c> {
    raw: MlirPassManager,
    _context: PhantomData<&'c Context>,
}
impl<'c> PassManager<'c> {
    pub fn new(context: &Context) -> Self {
        Self {
            raw: unsafe { mlirPassManagerCreate(context.to_raw()) },
            _context: Default::default(),
        }
    }
    pub fn nested_under(&self, name: &str) -> OperationPassManager {
        unsafe {
            OperationPassManager::from_raw(mlirPassManagerGetNestedUnder(self.raw, StringRef::from(name).to_raw()))
        }
    }
    pub fn add_pass(&self, pass: Pass) {
        unsafe { mlirPassManagerAddOwnedPass(self.raw, pass.to_raw()) }
    }
    pub fn enable_verifier(&self, enabled: bool) {
        unsafe { mlirPassManagerEnableVerifier(self.raw, enabled) }
    }
    pub fn enable_ir_printing(&self) {
        unsafe { mlirPassManagerEnableIRPrinting(self.raw) }
    }
    pub fn run(&self, module: &mut Module) -> Result<(), Error> {
        let result = LogicalResult::from_raw(unsafe { mlirPassManagerRun(self.raw, module.to_raw()) });
        if result.is_success() {
            Ok(())
        } else {
            Err(Error::RunPass)
        }
    }
    pub fn as_operation_pass_manager(&self) -> OperationPassManager {
        unsafe { OperationPassManager::from_raw(mlirPassManagerGetAsOpPassManager(self.raw)) }
    }
}
impl<'c> Drop for PassManager<'c> {
    fn drop(&mut self) {
        unsafe { mlirPassManagerDestroy(self.raw) }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OperationPassManager<'a> {
    raw: MlirOpPassManager,
    _parent: PhantomData<&'a PassManager<'a>>,
}
impl<'a> OperationPassManager<'a> {
    pub fn nested_under(&self, name: &str) -> Self {
        unsafe {
            Self::from_raw(mlirOpPassManagerGetNestedUnder(
                self.raw,
                StringRef::from(name).to_raw(),
            ))
        }
    }
    pub fn add_pass(&self, pass: Pass) {
        unsafe { mlirOpPassManagerAddOwnedPass(self.raw, pass.to_raw()) }
    }
    pub fn to_raw(self) -> MlirOpPassManager {
        self.raw
    }
    pub unsafe fn from_raw(raw: MlirOpPassManager) -> Self {
        Self {
            raw,
            _parent: Default::default(),
        }
    }
}
impl<'a> Display for OperationPassManager<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));
        unsafe extern "C" fn callback(string: MlirStringRef, data: *mut c_void) {
            let data = &mut *(data as *mut (&mut Formatter, fmt::Result));
            let result = (|| -> fmt::Result {
                write!(
                    data.0,
                    "{}",
                    StringRef::from_raw(string).as_str().map_err(|_| fmt::Error)?
                )
            })();
            if data.1.is_ok() {
                data.1 = result;
            }
        }
        unsafe {
            mlirPrintPassPipeline(self.raw, Some(callback), &mut data as *mut _ as *mut c_void);
        }
        data.1
    }
}

melior_macro::async_passes!(
    mlirCreateAsyncAsyncFuncToAsyncRuntime,
    mlirCreateAsyncAsyncParallelFor,
    mlirCreateAsyncAsyncRuntimePolicyBasedRefCounting,
    mlirCreateAsyncAsyncRuntimeRefCounting,
    mlirCreateAsyncAsyncRuntimeRefCountingOpt,
    mlirCreateAsyncAsyncToAsyncRuntime,
);

melior_macro::conversion_passes!(
    mlirCreateConversionArithToLLVMConversionPass,
    mlirCreateConversionConvertAffineForToGPU,
    mlirCreateConversionConvertAffineToStandard,
    mlirCreateConversionConvertAMDGPUToROCDL,
    mlirCreateConversionConvertArithToSPIRV,
    mlirCreateConversionConvertArmNeon2dToIntr,
    mlirCreateConversionConvertAsyncToLLVM,
    mlirCreateConversionConvertBufferizationToMemRef,
    mlirCreateConversionConvertComplexToLibm,
    mlirCreateConversionConvertComplexToLLVM,
    mlirCreateConversionConvertComplexToStandard,
    mlirCreateConversionConvertControlFlowToLLVM,
    mlirCreateConversionConvertControlFlowToSPIRV,
    mlirCreateConversionConvertFuncToLLVM,
    mlirCreateConversionConvertFuncToSPIRV,
    mlirCreateConversionConvertGpuLaunchFuncToVulkanLaunchFunc,
    mlirCreateConversionConvertGpuOpsToNVVMOps,
    mlirCreateConversionConvertGpuOpsToROCDLOps,
    mlirCreateConversionConvertGPUToSPIRV,
    mlirCreateConversionConvertIndexToLLVMPass,
    mlirCreateConversionConvertLinalgToLLVM,
    mlirCreateConversionConvertLinalgToStandard,
    mlirCreateConversionConvertMathToFuncs,
    mlirCreateConversionConvertMathToLibm,
    mlirCreateConversionConvertMathToLLVM,
    mlirCreateConversionConvertMathToSPIRV,
    mlirCreateConversionConvertMemRefToSPIRV,
    mlirCreateConversionConvertNVGPUToNVVM,
    mlirCreateConversionConvertOpenACCToLLVM,
    mlirCreateConversionConvertOpenACCToSCF,
    mlirCreateConversionConvertOpenMPToLLVM,
    mlirCreateConversionConvertParallelLoopToGpu,
    mlirCreateConversionConvertPDLToPDLInterp,
    mlirCreateConversionConvertSCFToOpenMP,
    mlirCreateConversionConvertShapeConstraints,
    mlirCreateConversionConvertShapeToStandard,
    mlirCreateConversionConvertSPIRVToLLVM,
    mlirCreateConversionConvertTensorToLinalg,
    mlirCreateConversionConvertTensorToSPIRV,
    mlirCreateConversionConvertVectorToGPU,
    mlirCreateConversionConvertVectorToLLVM,
    mlirCreateConversionConvertVectorToSCF,
    mlirCreateConversionConvertVectorToSPIRV,
    mlirCreateConversionConvertVulkanLaunchFuncToVulkanCalls,
    mlirCreateConversionGpuToLLVMConversionPass,
    mlirCreateConversionLowerHostCodeToLLVM,
    mlirCreateConversionMapMemRefStorageClass,
    mlirCreateConversionMemRefToLLVMConversionPass,
    mlirCreateConversionReconcileUnrealizedCasts,
    mlirCreateConversionSCFToControlFlow,
    mlirCreateConversionSCFToSPIRV,
    mlirCreateConversionTosaToArith,
    mlirCreateConversionTosaToLinalg,
    mlirCreateConversionTosaToLinalgNamed,
    mlirCreateConversionTosaToSCF,
    mlirCreateConversionTosaToTensor,
);

melior_macro::gpu_passes!(
    mlirCreateGPUGPULowerMemorySpaceAttributesPass,
    mlirCreateGPUGpuAsyncRegionPass,
    mlirCreateGPUGpuKernelOutlining,
    mlirCreateGPUGpuLaunchSinkIndexComputations,
    mlirCreateGPUGpuMapParallelLoopsPass,
);

melior_macro::linalg_passes!(
    mlirCreateLinalgConvertElementwiseToLinalg,
    mlirCreateLinalgLinalgBufferize,
    mlirCreateLinalgLinalgDetensorize,
    mlirCreateLinalgLinalgElementwiseOpFusion,
    mlirCreateLinalgLinalgFoldUnitExtentDims,
    mlirCreateLinalgLinalgGeneralization,
    mlirCreateLinalgLinalgInlineScalarOperands,
    mlirCreateLinalgLinalgLowerToAffineLoops,
    mlirCreateLinalgLinalgLowerToLoops,
    mlirCreateLinalgLinalgLowerToParallelLoops,
    mlirCreateLinalgLinalgNamedOpConversion,
);

melior_macro::sparse_tensor_passes!(
    mlirCreateSparseTensorPostSparsificationRewrite,
    mlirCreateSparseTensorPreSparsificationRewrite,
    mlirCreateSparseTensorSparseBufferRewrite,
    mlirCreateSparseTensorSparseTensorCodegen,
    mlirCreateSparseTensorSparseTensorConversionPass,
    mlirCreateSparseTensorSparseVectorization,
    mlirCreateSparseTensorSparsificationPass,
    mlirCreateSparseTensorStorageSpecifierToLLVM,
);

melior_macro::transform_passes!(
    mlirCreateTransformsCSE,
    mlirCreateTransformsCanonicalizer,
    mlirCreateTransformsControlFlowSink,
    mlirCreateTransformsGenerateRuntimeVerification,
    mlirCreateTransformsInliner,
    mlirCreateTransformsLocationSnapshot,
    mlirCreateTransformsLoopInvariantCodeMotion,
    mlirCreateTransformsPrintOpStats,
    mlirCreateTransformsSCCP,
    mlirCreateTransformsStripDebugInfo,
    mlirCreateTransformsSymbolDCE,
    mlirCreateTransformsSymbolPrivatize,
    mlirCreateTransformsTopologicalSort,
    mlirCreateTransformsViewOpGraph,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::DialectRegistry,
        ir::{Location, Module},
        pass::{self, transform::register_print_op_stats},
        utility::{parse_pass_pipeline, register_all_dialects},
    };
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    fn register_all_upstream_dialects(context: &Context) {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
    }
    #[test]
    fn new() {
        let context = Context::new();
        PassManager::new(&context);
    }
    #[test]
    fn add_pass() {
        let context = Context::new();
        PassManager::new(&context).add_pass(pass::conversion::create_func_to_llvm());
    }
    #[test]
    fn enable_verifier() {
        let context = Context::new();
        PassManager::new(&context).enable_verifier(true);
    }
    #[test]
    fn run() {
        let context = Context::new();
        let manager = PassManager::new(&context);
        manager.add_pass(pass::conversion::create_func_to_llvm());
        manager.run(&mut Module::new(Location::unknown(&context))).unwrap();
    }
    #[test]
    fn run_on_function() {
        let context = Context::new();
        register_all_upstream_dialects(&context);
        let mut module = Module::parse(
            &context,
            indoc!(
                "
                func.func @foo(%arg0 : i32) -> i32 {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }
                "
            ),
        )
        .unwrap();
        let manager = PassManager::new(&context);
        manager.add_pass(pass::transform::create_print_op_stats());
        assert_eq!(manager.run(&mut module), Ok(()));
    }
    #[test]
    fn run_on_function_in_nested_module() {
        let context = Context::new();
        register_all_upstream_dialects(&context);
        let mut module = Module::parse(
            &context,
            indoc!(
                "
                func.func @foo(%arg0 : i32) -> i32 {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }
                module {
                    func.func @bar(%arg0 : f32) -> f32 {
                        %res = arith.addf %arg0, %arg0 : f32
                        return %res : f32
                    }
                }
                "
            ),
        )
        .unwrap();
        let manager = PassManager::new(&context);
        manager
            .nested_under("func.func")
            .add_pass(pass::transform::create_print_op_stats());
        assert_eq!(manager.run(&mut module), Ok(()));
        let manager = PassManager::new(&context);
        manager
            .nested_under("builtin.module")
            .nested_under("func.func")
            .add_pass(pass::transform::create_print_op_stats());
        assert_eq!(manager.run(&mut module), Ok(()));
    }
    #[test]
    fn print_pass_pipeline() {
        let context = Context::new();
        let manager = PassManager::new(&context);
        let function_manager = manager.nested_under("func.func");
        function_manager.add_pass(pass::transform::create_print_op_stats());
        assert_eq!(
            manager.as_operation_pass_manager().to_string(),
            "builtin.module(func.func(print-op-stats{json=false}))"
        );
        assert_eq!(function_manager.to_string(), "func.func(print-op-stats{json=false})");
    }
    #[test]
    fn parse_pass_pipeline_() {
        let context = Context::new();
        let manager = PassManager::new(&context);
        insta::assert_display_snapshot!(parse_pass_pipeline(
            manager.as_operation_pass_manager(),
            "builtin.module(func.func(print-op-stats{json=false}),\
                func.func(print-op-stats{json=false}))"
        )
        .unwrap_err());
        register_print_op_stats();
        assert_eq!(
            parse_pass_pipeline(
                manager.as_operation_pass_manager(),
                "builtin.module(func.func(print-op-stats{json=false}),\
                func.func(print-op-stats{json=false}))"
            ),
            Ok(())
        );
        assert_eq!(
            manager.as_operation_pass_manager().to_string(),
            "builtin.module(func.func(print-op-stats{json=false}),\
            func.func(print-op-stats{json=false}))"
        );
    }
}
