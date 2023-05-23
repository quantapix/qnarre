//! Conversion passes.

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
