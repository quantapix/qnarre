use mlir_lib::*;
use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    error,
    ffi::{c_void, CString},
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
    mem::transmute,
    ops::Deref,
    slice,
    str::{self, Utf8Error},
    sync::{Once, RwLock},
};

use crate::{ir::*, Attribute, AttributeLike};

pub mod dialect;
pub mod ir;
pub mod mlir_lib;

use dialect::{Dialect, DialectRegistry};

#[derive(Debug)]
pub struct Context {
    raw: MlirContext,
}
impl Context {
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirContextCreate() },
        }
    }
    pub fn registered_dialect_count(&self) -> usize {
        unsafe { mlirContextGetNumRegisteredDialects(self.raw) as usize }
    }
    pub fn loaded_dialect_count(&self) -> usize {
        unsafe { mlirContextGetNumLoadedDialects(self.raw) as usize }
    }
    pub fn get_or_load_dialect(&self, name: &str) -> Dialect {
        unsafe { Dialect::from_raw(mlirContextGetOrLoadDialect(self.raw, StringRef::from(name).to_raw())) }
    }
    pub fn append_dialect_registry(&self, r: &DialectRegistry) {
        unsafe { mlirContextAppendDialectRegistry(self.raw, r.to_raw()) }
    }
    pub fn load_all_available_dialects(&self) {
        unsafe { mlirContextLoadAllAvailableDialects(self.raw) }
    }
    pub fn enable_multi_threading(&self, enabled: bool) {
        unsafe { mlirContextEnableMultithreading(self.raw, enabled) }
    }
    pub fn allow_unregistered_dialects(&self) -> bool {
        unsafe { mlirContextGetAllowUnregisteredDialects(self.raw) }
    }
    pub fn set_allow_unregistered_dialects(&self, allowed: bool) {
        unsafe { mlirContextSetAllowUnregisteredDialects(self.raw, allowed) }
    }
    pub fn is_registered_operation(&self, name: &str) -> bool {
        unsafe { mlirContextIsRegisteredOperation(self.raw, StringRef::from(name).to_raw()) }
    }
    pub fn to_raw(&self) -> MlirContext {
        self.raw
    }
    pub fn attach_diagnostic_handler<F: FnMut(Diagnostic) -> bool>(&self, handler: F) -> DiagnosticHandlerId {
        unsafe extern "C" fn handle<F: FnMut(Diagnostic) -> bool>(
            diag: MlirDiagnostic,
            data: *mut c_void,
        ) -> MlirLogicalResult {
            LogicalResult::from((*(data as *mut F))(Diagnostic::from_raw(diag))).to_raw()
        }
        unsafe extern "C" fn destroy<F: FnMut(Diagnostic) -> bool>(data: *mut c_void) {
            drop(Box::from_raw(data as *mut F));
        }
        unsafe {
            DiagnosticHandlerId::from_raw(mlirContextAttachDiagnosticHandler(
                self.to_raw(),
                Some(handle::<F>),
                Box::into_raw(Box::new(handler)) as *mut _,
                Some(destroy::<F>),
            ))
        }
    }
    pub fn detach_diagnostic_handler(&self, id: DiagnosticHandlerId) {
        unsafe { mlirContextDetachDiagnosticHandler(self.to_raw(), id.to_raw()) }
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe { mlirContextDestroy(self.raw) };
    }
}
impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
impl PartialEq for Context {
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirContextEqual(self.raw, x.raw) }
    }
}
impl Eq for Context {}

#[derive(Clone, Copy, Debug)]
pub struct ContextRef<'a> {
    raw: MlirContext,
    _ref: PhantomData<&'a Context>,
}
impl<'a> ContextRef<'a> {
    pub unsafe fn from_raw(raw: MlirContext) -> Self {
        Self {
            raw,
            _ref: Default::default(),
        }
    }
}
impl<'a> Deref for ContextRef<'a> {
    type Target = Context;
    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
    }
}
impl<'a> PartialEq for ContextRef<'a> {
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirContextEqual(self.raw, x.raw) }
    }
}
impl<'a> Eq for ContextRef<'a> {}

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
    pub fn note(&self, idx: usize) -> Result<Self, Error> {
        if idx < self.note_count() {
            Ok(unsafe { Self::from_raw(mlirDiagnosticGetNote(self.raw, idx as isize)) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "diagnostic note",
                value: self.to_string(),
                index: idx,
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
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut y = (f, Ok(()));
        unsafe {
            mlirDiagnosticPrint(self.raw, Some(print_callback), &mut y as *mut _ as *mut c_void);
        }
        y.1
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    AttributeExpected(&'static str, String),
    BlockArgumentExpected(String),
    ElementExpected {
        r#type: &'static str,
        value: String,
    },
    InvokeFunction,
    OperationResultExpected(String),
    PositionOutOfBounds {
        name: &'static str,
        val: String,
        idx: usize,
    },
    ParsePassPipeline(String),
    RunPass,
    TypeExpected(&'static str, String),
    UnknownDiagnosticSeverity(u32),
    Utf8(Utf8Error),
}
impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::AttributeExpected(r#type, attribute) => {
                write!(f, "{type} attribute expected: {attribute}")
            },
            Self::BlockArgumentExpected(value) => {
                write!(f, "block argument expected: {value}")
            },
            Self::ElementExpected { r#type, value } => {
                write!(f, "element of {type} type expected: {value}")
            },
            Self::InvokeFunction => write!(f, "failed to invoke JIT-compiled function"),
            Self::OperationResultExpected(value) => {
                write!(f, "operation result expected: {value}")
            },
            Self::ParsePassPipeline(message) => {
                write!(f, "failed to parse pass pipeline:\n{}", message)
            },
            Self::PositionOutOfBounds { name, value, index } => {
                write!(f, "{name} position {index} out of bounds: {value}")
            },
            Self::RunPass => write!(f, "failed to run pass"),
            Self::TypeExpected(r#type, actual) => {
                write!(f, "{type} type expected: {actual}")
            },
            Self::UnknownDiagnosticSeverity(severity) => {
                write!(f, "unknown diagnostic severity: {severity}")
            },
            Self::Utf8(error) => {
                write!(f, "{}", error)
            },
        }
    }
}
impl error::Error for Error {}
impl From<Utf8Error> for Error {
    fn from(error: Utf8Error) -> Self {
        Self::Utf8(error)
    }
}

pub struct ExecutionEngine {
    raw: MlirExecutionEngine,
}
impl ExecutionEngine {
    pub fn new(
        module: &Module,
        optimization_level: usize,
        shared_library_paths: &[&str],
        enable_object_dump: bool,
    ) -> Self {
        Self {
            raw: unsafe {
                mlirExecutionEngineCreate(
                    module.to_raw(),
                    optimization_level as i32,
                    shared_library_paths.len() as i32,
                    shared_library_paths
                        .iter()
                        .map(|&string| StringRef::from(string).to_raw())
                        .collect::<Vec<_>>()
                        .as_ptr(),
                    enable_object_dump,
                )
            },
        }
    }
    pub unsafe fn invoke_packed(&self, name: &str, arguments: &mut [*mut ()]) -> Result<(), Error> {
        let result = LogicalResult::from_raw(mlirExecutionEngineInvokePacked(
            self.raw,
            StringRef::from(name).to_raw(),
            arguments.as_mut_ptr() as *mut *mut c_void,
        ));
        if result.is_success() {
            Ok(())
        } else {
            Err(Error::InvokeFunction)
        }
    }
    pub fn dump_to_object_file(&self, path: &str) {
        unsafe { mlirExecutionEngineDumpToObjectFile(self.raw, StringRef::from(path).to_raw()) }
    }
}
impl Drop for ExecutionEngine {
    fn drop(&mut self) {
        unsafe { mlirExecutionEngineDestroy(self.raw) }
    }
}

#[derive(Clone, Copy)]
pub struct Integer<'c> {
    raw: MlirAttribute,
    _context: PhantomData<&'c Context>,
}
impl<'c> Integer<'c> {
    pub fn new(integer: i64, r#type: Type<'c>) -> Self {
        unsafe { Self::from_raw(mlirIntegerAttrGet(r#type.to_raw(), integer)) }
    }
    unsafe fn from_raw(raw: MlirAttribute) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
}
impl<'c> AttributeLike<'c> for Integer<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.raw
    }
}
impl<'c> TryFrom<Attribute<'c>> for Integer<'c> {
    type Error = Error;
    fn try_from(attribute: Attribute<'c>) -> Result<Self, Self::Error> {
        if attribute.is_integer() {
            Ok(unsafe { Self::from_raw(attribute.to_raw()) })
        } else {
            Err(Error::AttributeExpected("integer", format!("{}", attribute)))
        }
    }
}
impl<'c> Display for Integer<'c> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(&Attribute::from(*self), f)
    }
}
impl<'c> Debug for Integer<'c> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LogicalResult {
    raw: MlirLogicalResult,
}
impl LogicalResult {
    pub fn success() -> Self {
        Self {
            raw: MlirLogicalResult { value: 1 },
        }
    }
    pub fn failure() -> Self {
        Self {
            raw: MlirLogicalResult { value: 0 },
        }
    }
    pub fn is_success(&self) -> bool {
        self.raw.value != 0
    }
    #[allow(dead_code)]
    pub fn is_failure(&self) -> bool {
        self.raw.value == 0
    }
    pub fn from_raw(result: MlirLogicalResult) -> Self {
        Self { raw: result }
    }
    pub fn to_raw(self) -> MlirLogicalResult {
        self.raw
    }
}
impl From<bool> for LogicalResult {
    fn from(ok: bool) -> Self {
        if ok {
            Self::success()
        } else {
            Self::failure()
        }
    }
}

static STRING_CACHE: Lazy<RwLock<HashMap<String, CString>>> = Lazy::new(Default::default);

#[derive(Clone, Copy, Debug)]
pub struct StringRef<'a> {
    raw: MlirStringRef,
    _parent: PhantomData<&'a ()>,
}
impl<'a> StringRef<'a> {
    pub fn as_str(&self) -> Result<&'a str, Utf8Error> {
        unsafe {
            let bytes = slice::from_raw_parts(self.raw.data as *mut u8, self.raw.length);
            str::from_utf8(if bytes[bytes.len() - 1] == 0 {
                &bytes[..bytes.len() - 1]
            } else {
                bytes
            })
        }
    }
    pub fn to_raw(self) -> MlirStringRef {
        self.raw
    }
    pub unsafe fn from_raw(string: MlirStringRef) -> Self {
        Self {
            raw: string,
            _parent: Default::default(),
        }
    }
}
impl<'a> PartialEq for StringRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirStringRefEqual(self.raw, other.raw) }
    }
}
impl<'a> Eq for StringRef<'a> {}
impl From<&str> for StringRef<'static> {
    fn from(string: &str) -> Self {
        if !STRING_CACHE.read().unwrap().contains_key(string) {
            STRING_CACHE
                .write()
                .unwrap()
                .insert(string.to_owned(), CString::new(string).unwrap());
        }
        let lock = STRING_CACHE.read().unwrap();
        let string = lock.get(string).unwrap();
        unsafe { Self::from_raw(mlirStringRefCreateFromCString(string.as_ptr())) }
    }
}

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
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut y = (f, Ok(()));
        unsafe extern "C" fn callback(string: MlirStringRef, data: *mut c_void) {
            let y = &mut *(data as *mut (&mut Formatter, fmt::Result));
            let result = (|| -> fmt::Result {
                write!(y.0, "{}", StringRef::from_raw(string).as_str().map_err(|_| fmt::Error)?)
            })();
            if y.1.is_ok() {
                y.1 = result;
            }
        }
        unsafe {
            mlirPrintPassPipeline(self.raw, Some(callback), &mut y as *mut _ as *mut c_void);
        }
        y.1
    }
}

macros::async_passes!(
    mlirCreateAsyncAsyncFuncToAsyncRuntime,
    mlirCreateAsyncAsyncParallelFor,
    mlirCreateAsyncAsyncRuntimePolicyBasedRefCounting,
    mlirCreateAsyncAsyncRuntimeRefCounting,
    mlirCreateAsyncAsyncRuntimeRefCountingOpt,
    mlirCreateAsyncAsyncToAsyncRuntime,
);

macros::conversion_passes!(
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

macros::gpu_passes!(
    mlirCreateGPUGPULowerMemorySpaceAttributesPass,
    mlirCreateGPUGpuAsyncRegionPass,
    mlirCreateGPUGpuKernelOutlining,
    mlirCreateGPUGpuLaunchSinkIndexComputations,
    mlirCreateGPUGpuMapParallelLoopsPass,
);

macros::linalg_passes!(
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

macros::sparse_tensor_passes!(
    mlirCreateSparseTensorPostSparsificationRewrite,
    mlirCreateSparseTensorPreSparsificationRewrite,
    mlirCreateSparseTensorSparseBufferRewrite,
    mlirCreateSparseTensorSparseTensorCodegen,
    mlirCreateSparseTensorSparseTensorConversionPass,
    mlirCreateSparseTensorSparseVectorization,
    mlirCreateSparseTensorSparsificationPass,
    mlirCreateSparseTensorStorageSpecifierToLLVM,
);

macros::transform_passes!(
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

pub fn register_all_dialects(registry: &DialectRegistry) {
    unsafe { mlirRegisterAllDialects(registry.to_raw()) }
}
pub fn register_all_llvm_translations(context: &Context) {
    unsafe { mlirRegisterAllLLVMTranslations(context.to_raw()) }
}
pub fn register_all_passes() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe { mlirRegisterAllPasses() });
}
pub fn parse_pass_pipeline(manager: pass::OperationPassManager, source: &str) -> Result<(), Error> {
    let mut error_message = None;
    let result = LogicalResult::from_raw(unsafe {
        mlirParsePassPipeline(
            manager.to_raw(),
            StringRef::from(source).to_raw(),
            Some(handle_parse_error),
            &mut error_message as *mut _ as *mut _,
        )
    });
    if result.is_success() {
        Ok(())
    } else {
        Err(Error::ParsePassPipeline(
            error_message.unwrap_or_else(|| "failed to parse error message in UTF-8".into()),
        ))
    }
}

unsafe extern "C" fn handle_parse_error(raw_string: MlirStringRef, data: *mut c_void) {
    let string = StringRef::from_raw(raw_string);
    let data = &mut *(data as *mut Option<String>);
    if let Some(message) = data {
        message.extend(string.as_str())
    } else {
        *data = string.as_str().map(String::from).ok();
    }
}

pub unsafe fn into_raw_array<T>(xs: Vec<T>) -> *mut T {
    xs.leak().as_mut_ptr()
}
pub unsafe extern "C" fn print_callback(string: MlirStringRef, data: *mut c_void) {
    let (formatter, result) = &mut *(data as *mut (&mut Formatter, fmt::Result));
    if result.is_err() {
        return;
    }
    *result = (|| {
        write!(
            formatter,
            "{}",
            StringRef::from_raw(string).as_str().map_err(|_| fmt::Error)?
        )
    })();
}
pub unsafe extern "C" fn print_string_callback(string: MlirStringRef, data: *mut c_void) {
    let (writer, result) = &mut *(data as *mut (String, Result<(), Error>));
    if result.is_err() {
        return;
    }
    *result = (|| {
        writer.push_str(StringRef::from_raw(string).as_str()?);
        Ok(())
    })();
}

pub fn load_all_dialects(context: &Context) {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
}

pub fn create_test_context() -> Context {
    let context = Context::new();
    context.attach_diagnostic_handler(|diagnostic| {
        eprintln!("{}", diagnostic);
        true
    });
    load_all_dialects(&context);
    register_all_llvm_translations(&context);
    context
}

#[cfg(test)]
mod test;
