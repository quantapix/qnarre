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
    fn try_from(x: u32) -> Result<Self, Error> {
        #[allow(non_upper_case_globals)]
        Ok(match x {
            MlirDiagnosticSeverity_MlirDiagnosticError => Self::Error,
            MlirDiagnosticSeverity_MlirDiagnosticNote => Self::Note,
            MlirDiagnosticSeverity_MlirDiagnosticRemark => Self::Remark,
            MlirDiagnosticSeverity_MlirDiagnosticWarning => Self::Warning,
            _ => return Err(Error::UnknownDiagnosticSeverity(x)),
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
            .unwrap_or_else(|x| unreachable!("{}", x))
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
                val: self.to_string(),
                idx,
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
        typ: &'static str,
        val: String,
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
            Self::AttributeExpected(typ, x) => {
                write!(f, "{type} attribute expected: {x}")
            },
            Self::BlockArgumentExpected(x) => {
                write!(f, "block argument expected: {x}")
            },
            Self::ElementExpected { typ, val } => {
                write!(f, "element of {typ} type expected: {val}")
            },
            Self::InvokeFunction => write!(f, "failed to invoke JIT-compiled function"),
            Self::OperationResultExpected(x) => {
                write!(f, "operation result expected: {x}")
            },
            Self::ParsePassPipeline(x) => {
                write!(f, "failed to parse pass pipeline:\n{}", x)
            },
            Self::PositionOutOfBounds { name, val, idx } => {
                write!(f, "{name} position {idx} out of bounds: {val}")
            },
            Self::RunPass => write!(f, "failed to run pass"),
            Self::TypeExpected(typ, x) => {
                write!(f, "{typ} type expected: {x}")
            },
            Self::UnknownDiagnosticSeverity(x) => {
                write!(f, "unknown diagnostic severity: {x}")
            },
            Self::Utf8(x) => {
                write!(f, "{}", x)
            },
        }
    }
}
impl error::Error for Error {}
impl From<Utf8Error> for Error {
    fn from(x: Utf8Error) -> Self {
        Self::Utf8(x)
    }
}

pub struct ExecutionEngine {
    raw: MlirExecutionEngine,
}
impl ExecutionEngine {
    pub fn new(m: &Module, optim: usize, shared_library_paths: &[&str], enable_object_dump: bool) -> Self {
        Self {
            raw: unsafe {
                mlirExecutionEngineCreate(
                    m.to_raw(),
                    optim as i32,
                    shared_library_paths.len() as i32,
                    shared_library_paths
                        .iter()
                        .map(|&x| StringRef::from(x).to_raw())
                        .collect::<Vec<_>>()
                        .as_ptr(),
                    enable_object_dump,
                )
            },
        }
    }
    pub unsafe fn invoke_packed(&self, name: &str, args: &mut [*mut ()]) -> Result<(), Error> {
        let y = LogicalResult::from_raw(mlirExecutionEngineInvokePacked(
            self.raw,
            StringRef::from(name).to_raw(),
            args.as_mut_ptr() as *mut *mut c_void,
        ));
        if y.is_success() {
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
    pub fn new(x: i64, ty: Type<'c>) -> Self {
        unsafe { Self::from_raw(mlirIntegerAttrGet(ty.to_raw(), x)) }
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
    fn try_from(a: Attribute<'c>) -> Result<Self, Self::Error> {
        if a.is_integer() {
            Ok(unsafe { Self::from_raw(a.to_raw()) })
        } else {
            Err(Error::AttributeExpected("integer", format!("{}", a)))
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
    pub fn from_raw(raw: MlirLogicalResult) -> Self {
        Self { raw }
    }
    pub fn to_raw(self) -> MlirLogicalResult {
        self.raw
    }
}
impl From<bool> for LogicalResult {
    fn from(x: bool) -> Self {
        if x {
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
            let ys = slice::from_raw_parts(self.raw.data as *mut u8, self.raw.length);
            str::from_utf8(if ys[ys.len() - 1] == 0 { &ys[..ys.len() - 1] } else { ys })
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
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirStringRefEqual(self.raw, x.raw) }
    }
}
impl<'a> Eq for StringRef<'a> {}
impl From<&str> for StringRef<'static> {
    fn from(x: &str) -> Self {
        if !STRING_CACHE.read().unwrap().contains_key(x) {
            STRING_CACHE
                .write()
                .unwrap()
                .insert(x.to_owned(), CString::new(x).unwrap());
        }
        let lock = STRING_CACHE.read().unwrap();
        let y = lock.get(x).unwrap();
        unsafe { Self::from_raw(mlirStringRefCreateFromCString(y.as_ptr())) }
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
    _ctx: PhantomData<&'c Context>,
}
impl<'c> PassManager<'c> {
    pub fn new(c: &Context) -> Self {
        Self {
            raw: unsafe { mlirPassManagerCreate(c.to_raw()) },
            _ctx: Default::default(),
        }
    }
    pub fn nested_under(&self, name: &str) -> OperationPassManager {
        unsafe {
            OperationPassManager::from_raw(mlirPassManagerGetNestedUnder(self.raw, StringRef::from(name).to_raw()))
        }
    }
    pub fn add_pass(&self, p: Pass) {
        unsafe { mlirPassManagerAddOwnedPass(self.raw, p.to_raw()) }
    }
    pub fn enable_verifier(&self, enabled: bool) {
        unsafe { mlirPassManagerEnableVerifier(self.raw, enabled) }
    }
    pub fn enable_ir_printing(&self) {
        unsafe { mlirPassManagerEnableIRPrinting(self.raw) }
    }
    pub fn run(&self, m: &mut Module) -> Result<(), Error> {
        let y = LogicalResult::from_raw(unsafe { mlirPassManagerRun(self.raw, m.to_raw()) });
        if y.is_success() {
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
    pub fn add_pass(&self, p: Pass) {
        unsafe { mlirOpPassManagerAddOwnedPass(self.raw, p.to_raw()) }
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

pub fn register_all_dialects(r: &DialectRegistry) {
    unsafe { mlirRegisterAllDialects(r.to_raw()) }
}
pub fn register_all_llvm_translations(c: &Context) {
    unsafe { mlirRegisterAllLLVMTranslations(c.to_raw()) }
}
pub fn register_all_passes() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe { mlirRegisterAllPasses() });
}
pub fn parse_pass_pipeline(m: OperationPassManager, src: &str) -> Result<(), Error> {
    let mut e = None;
    let y = LogicalResult::from_raw(unsafe {
        mlirParsePassPipeline(
            m.to_raw(),
            StringRef::from(src).to_raw(),
            Some(handle_parse_error),
            &mut e as *mut _ as *mut _,
        )
    });
    if y.is_success() {
        Ok(())
    } else {
        Err(Error::ParsePassPipeline(
            e.unwrap_or_else(|| "failed to parse error message in UTF-8".into()),
        ))
    }
}

unsafe extern "C" fn handle_parse_error(raw: MlirStringRef, data: *mut c_void) {
    let string = StringRef::from_raw(raw);
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

pub fn load_all_dialects(c: &Context) {
    let r = DialectRegistry::new();
    register_all_dialects(&r);
    c.append_dialect_registry(&r);
    c.load_all_available_dialects();
}

pub fn create_test_context() -> Context {
    let y = Context::new();
    y.attach_diagnostic_handler(|x| {
        eprintln!("{}", x);
        true
    });
    load_all_dialects(&y);
    register_all_llvm_translations(&y);
    y
}

#[cfg(test)]
mod test;
