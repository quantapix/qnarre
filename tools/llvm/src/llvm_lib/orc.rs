#![allow(non_snake_case)]

use super::prelude::*;
use error::LLVMErrorRef;
use target_machine::LLVMTargetMachineRef;
use {execution_engine::*, *};

// llvm-c/Orc.h

pub type LLVMOrcJITTargetAddress = u64;

pub type LLVMOrcExecutorAddress = u64;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LLVMJITSymbolGenericFlags {
    LLVMJITSymbolGenericFlagsNone = 0,
    LLVMJITSymbolGenericFlagsExported = 1,
    LLVMJITSymbolGenericFlagsWeak = 2,
    LLVMJITSymbolGenericFlagsCallable = 4,
    LLVMJITSymbolGenericFlagsMaterializationSideEffectsOnly = 8,
}

pub type LLVMJITSymbolTargetFlags = u8;

#[repr(C)]
#[derive(Debug)]
pub struct LLVMJITSymbolFlags {
    pub GenericFlags: u8,
    pub TargetFlags: u8,
}

#[repr(C)]
#[derive(Debug)]
pub struct LLVMJITEvaluatedSymbol {
    pub Address: LLVMOrcExecutorAddress,
    pub Flags: LLVMJITSymbolFlags,
}

#[derive(Debug)]
pub enum LLVMOrcOpaqueExecutionSession {}
pub type LLVMOrcExecutionSessionRef = *mut LLVMOrcOpaqueExecutionSession;

pub type LLVMOrcErrorReporterFunction = extern "C" fn(Ctx: *mut ::libc::c_void, Err: LLVMErrorRef);

#[derive(Debug)]
pub enum LLVMOrcOpaqueSymbolStringPool {}
pub type LLVMOrcSymbolStringPoolRef = *mut LLVMOrcOpaqueSymbolStringPool;

#[derive(Debug)]
pub enum LLVMOrcOpaqueSymbolStringPoolEntry {}
pub type LLVMOrcSymbolStringPoolEntryRef = *mut LLVMOrcOpaqueSymbolStringPoolEntry;

#[repr(C)]
#[derive(Debug)]
pub struct LLVMOrcCSymbolFlagsMapPair {
    pub Name: LLVMOrcSymbolStringPoolEntryRef,
    pub Flags: LLVMJITSymbolFlags,
}

pub type LLVMOrcCSymbolFlagsMapPairs = *mut LLVMOrcCSymbolFlagsMapPair;

#[repr(C)]
#[derive(Debug)]
pub struct LLVMOrcCSymbolMapPair {
    pub Name: LLVMOrcSymbolStringPoolEntryRef,
    pub Sym: LLVMJITEvaluatedSymbol,
}

pub type LLVMOrcCSymbolMapPairs = *mut LLVMOrcCSymbolMapPair;

#[repr(C)]
#[derive(Debug)]
pub struct LLVMOrcCSymbolAliasMapEntry {
    pub Name: LLVMOrcSymbolStringPoolEntryRef,
    pub Flags: LLVMJITSymbolFlags,
}

#[repr(C)]
#[derive(Debug)]
pub struct LLVMOrcCSymbolAliasMapPair {
    pub Name: LLVMOrcSymbolStringPoolEntryRef,
    pub Entry: LLVMOrcCSymbolAliasMapEntry,
}

pub type LLVMOrcCSymbolAliasMapPairs = *mut LLVMOrcCSymbolAliasMapPair;

#[derive(Debug)]
pub enum LLVMOrcOpaqueJITDylib {}
pub type LLVMOrcJITDylibRef = *mut LLVMOrcOpaqueJITDylib;

#[repr(C)]
#[derive(Debug)]
pub struct LLVMOrcCSymbolsList {
    pub Symbols: *mut LLVMOrcSymbolStringPoolEntryRef,
    pub Length: ::libc::size_t,
}

#[repr(C)]
#[derive(Debug)]
pub struct LLVMOrcCDependenceMapPair {
    pub JD: LLVMOrcJITDylibRef,
    pub Names: LLVMOrcCSymbolsList,
}

pub type LLVMOrcCDependenceMapPairs = *mut LLVMOrcCDependenceMapPair;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LLVMOrcLookupKind {
    LLVMOrcLookupKindStatic,
    LLVMOrcLookupKindDLSym,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LLVMOrcJITDylibLookupFlags {
    LLVMOrcJITDylibLookupFlagsMatchExportedSymbolsOnly,
    LLVMOrcJITDylibLookupFlagsMatchAllSymbols,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LLVMOrcCJITDylibSearchOrderElement {
    pub JD: LLVMOrcJITDylibRef,
    pub JDLookupFlags: LLVMOrcJITDylibLookupFlags,
}

pub type LLVMOrcCJITDylibSearchOrder = *mut LLVMOrcCJITDylibSearchOrderElement;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LLVMOrcSymbolLookupFlags {
    LLVMOrcSymbolLookupFlagsRequiredSymbol,
    LLVMOrcSymbolLookupFlagsWeaklyReferencedSymbol,
}

#[repr(C)]
#[derive(Debug)]
pub struct LLVMOrcCLookupSetElement {
    pub Name: LLVMOrcSymbolStringPoolEntryRef,
    pub LookupFlags: LLVMOrcSymbolLookupFlags,
}

pub type LLVMOrcCLookupSet = *mut LLVMOrcCLookupSetElement;

#[derive(Debug)]
pub enum LLVMOrcOpaqueMaterializationUnit {}
pub type LLVMOrcMaterializationUnitRef = *mut LLVMOrcOpaqueMaterializationUnit;

#[derive(Debug)]
pub enum LLVMOrcOpaqueMaterializationResponsibility {}
pub type LLVMOrcMaterializationResponsibilityRef = *mut LLVMOrcOpaqueMaterializationResponsibility;

pub type LLVMOrcMaterializationUnitMaterializeFunction =
    extern "C" fn(Ctx: *mut ::libc::c_void, MR: LLVMOrcMaterializationResponsibilityRef);

pub type LLVMOrcMaterializationUnitDiscardFunction =
    extern "C" fn(Ctx: *mut ::libc::c_void, JD: LLVMOrcJITDylibRef, Symbol: LLVMOrcSymbolStringPoolEntryRef);

pub type LLVMOrcMaterializationUnitDestroyFunction = extern "C" fn(Ctx: *mut ::libc::c_void);

#[derive(Debug)]
pub enum LLVMOrcOpaqueResourceTracker {}
pub type LLVMOrcResourceTrackerRef = *mut LLVMOrcOpaqueResourceTracker;

#[derive(Debug)]
pub enum LLVMOrcOpaqueDefinitionGenerator {}
pub type LLVMOrcDefinitionGeneratorRef = *mut LLVMOrcOpaqueDefinitionGenerator;

#[derive(Debug)]
pub enum LLVMOrcOpaqueLookupState {}
pub type LLVMOrcLookupStateRef = *mut LLVMOrcOpaqueLookupState;

pub type LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction = extern "C" fn(
    GeneratorObj: LLVMOrcDefinitionGeneratorRef,
    Ctx: *mut ::libc::c_void,
    LookupState: *mut LLVMOrcLookupStateRef,
    Kind: LLVMOrcLookupKind,
    JD: LLVMOrcJITDylibRef,
    JDLookupFlags: LLVMOrcJITDylibLookupFlags,
    LookupSet: LLVMOrcCLookupSet,
    LookupSetSize: usize,
) -> LLVMErrorRef;

pub type LLVMOrcDisposeCAPIDefinitionGeneratorFunction = extern "C" fn(Ctx: *mut ::libc::c_void);

pub type LLVMOrcSymbolPredicate =
    Option<extern "C" fn(Ctx: *mut ::libc::c_void, Sym: LLVMOrcSymbolStringPoolEntryRef) -> ::libc::c_int>;

#[derive(Debug)]
pub enum LLVMOrcOpaqueThreadSafeContext {}
pub type LLVMOrcThreadSafeContextRef = *mut LLVMOrcOpaqueThreadSafeContext;

#[derive(Debug)]
pub enum LLVMOrcOpaqueThreadSafeModule {}
pub type LLVMOrcThreadSafeModuleRef = *mut LLVMOrcOpaqueThreadSafeModule;

pub type LLVMOrcGenericIRModuleOperationFunction =
    extern "C" fn(Ctx: *mut ::libc::c_void, M: LLVMModuleRef) -> LLVMErrorRef;

#[derive(Debug)]
pub enum LLVMOrcOpaqueJITTargetMachineBuilder {}
pub type LLVMOrcJITTargetMachineBuilderRef = *mut LLVMOrcOpaqueJITTargetMachineBuilder;

#[derive(Debug)]
pub enum LLVMOrcOpaqueObjectLayer {}
pub type LLVMOrcObjectLayerRef = *mut LLVMOrcOpaqueObjectLayer;

#[derive(Debug)]
pub enum LLVMOrcOpaqueObjectLinkingLayer {}
pub type LLVMOrcObjectLinkingLayerRef = *mut LLVMOrcOpaqueObjectLayer;

#[derive(Debug)]
pub enum LLVMOrcOpaqueIRTransformLayer {}
pub type LLVMOrcIRTransformLayerRef = *mut LLVMOrcOpaqueIRTransformLayer;

pub type LLVMOrcIRTransformLayerTransformFunction = extern "C" fn(
    Ctx: *mut ::libc::c_void,
    ModInOut: *mut LLVMOrcThreadSafeModuleRef,
    MR: LLVMOrcMaterializationResponsibilityRef,
) -> LLVMErrorRef;

#[derive(Debug)]
pub enum LLVMOrcOpaqueObjectTransformLayer {}
pub type LLVMOrcObjectTransformLayerRef = *mut LLVMOrcOpaqueObjectTransformLayer;

pub type LLVMOrcObjectTransformLayerTransformFunction =
    extern "C" fn(Ctx: *mut ::libc::c_void, ObjInOut: *mut LLVMMemoryBufferRef) -> LLVMErrorRef;

#[derive(Debug)]
pub enum LLVMOrcOpaqueIndirectStubsManager {}
pub type LLVMOrcIndirectStubsManagerRef = *mut LLVMOrcOpaqueIndirectStubsManager;

#[derive(Debug)]
pub enum LLVMOrcOpaqueLazyCallThroughManager {}
pub type LLVMOrcLazyCallThroughManagerRef = *mut LLVMOrcOpaqueLazyCallThroughManager;

#[derive(Debug)]
pub enum LLVMOrcOpaqueDumpObjects {}
pub type LLVMOrcDumpObjectsRef = *mut LLVMOrcOpaqueDumpObjects;

extern "C" {
    pub fn LLVMOrcExecutionSessionGetSymbolStringPool(ES: LLVMOrcExecutionSessionRef) -> LLVMOrcSymbolStringPoolRef;
    pub fn LLVMOrcExecutionSessionIntern(
        ES: LLVMOrcExecutionSessionRef,
        Name: *const ::libc::c_char,
    ) -> LLVMOrcSymbolStringPoolEntryRef;
    pub fn LLVMOrcExecutionSessionSetErrorReporter(
        ES: LLVMOrcExecutionSessionRef,
        ReportError: LLVMOrcErrorReporterFunction,
        Ctx: *mut ::libc::c_void,
    );
    pub fn LLVMOrcSymbolStringPoolClearDeadEntries(SSP: LLVMOrcSymbolStringPoolRef);
}

pub type LLVMOrcExecutionSessionLookupHandleResultFunction =
    extern "C" fn(Err: LLVMErrorRef, Result: LLVMOrcCSymbolMapPairs, NumPairs: usize, Ctx: *mut ::libc::c_void);

extern "C" {
    pub fn LLVMOrcAbsoluteSymbols(Syms: LLVMOrcCSymbolMapPairs, NumPairs: usize) -> LLVMOrcMaterializationUnitRef;
    pub fn LLVMOrcCreateCustomCAPIDefinitionGenerator(
        F: LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction,
        Ctx: *mut ::libc::c_void,
        Dispose: LLVMOrcDisposeCAPIDefinitionGeneratorFunction,
    ) -> LLVMOrcDefinitionGeneratorRef;
    pub fn LLVMOrcCreateCustomMaterializationUnit(
        Name: *const ::libc::c_char,
        Ctx: *mut ::libc::c_void,
        Syms: LLVMOrcCSymbolFlagsMapPairs,
        NumSyms: ::libc::size_t,
        InitSym: LLVMOrcSymbolStringPoolEntryRef,
        Materialize: LLVMOrcMaterializationUnitMaterializeFunction,
        Discard: LLVMOrcMaterializationUnitDiscardFunction,
        Destroy: LLVMOrcMaterializationUnitDestroyFunction,
    ) -> LLVMOrcMaterializationUnitRef;
    pub fn LLVMOrcCreateDumpObjects(
        DumpDir: *const ::libc::c_char,
        IdentifierOverride: *const ::libc::c_char,
    ) -> LLVMOrcDumpObjectsRef;
    pub fn LLVMOrcCreateDynamicLibrarySearchGeneratorForPath(
        Result: *mut LLVMOrcDefinitionGeneratorRef,
        FileName: *const ::libc::c_char,
        GlobalPrefix: ::libc::c_char,
        Filter: LLVMOrcSymbolPredicate,
        FilterCtx: *mut ::libc::c_void,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
        Result: *mut LLVMOrcDefinitionGeneratorRef,
        GlobalPrefix: ::libc::c_char,
        Filter: LLVMOrcSymbolPredicate,
        FilterCtx: *mut ::libc::c_void,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcCreateLocalIndirectStubsManager(
        TargetTriple: *const ::libc::c_char,
    ) -> LLVMOrcIndirectStubsManagerRef;
    pub fn LLVMOrcCreateLocalLazyCallThroughManager(
        TargetTriple: *const ::libc::c_char,
        ES: LLVMOrcExecutionSessionRef,
        ErrorHandlerAddr: LLVMOrcJITTargetAddress,
        LCTM: *mut LLVMOrcLazyCallThroughManagerRef,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcCreateNewThreadSafeContext() -> LLVMOrcThreadSafeContextRef;
    pub fn LLVMOrcCreateNewThreadSafeModule(
        M: LLVMModuleRef,
        TSCtx: LLVMOrcThreadSafeContextRef,
    ) -> LLVMOrcThreadSafeModuleRef;
    pub fn LLVMOrcCreateStaticLibrarySearchGeneratorForPath(
        Result: *mut LLVMOrcDefinitionGeneratorRef,
        ObjLayer: LLVMOrcObjectLayerRef,
        FileName: *const ::libc::c_char,
        TargetTriple: *const ::libc::c_char,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcDisposeCSymbolFlagsMap(Pairs: LLVMOrcCSymbolFlagsMapPairs);
    pub fn LLVMOrcDisposeDefinitionGenerator(DG: LLVMOrcDefinitionGeneratorRef);
    pub fn LLVMOrcDisposeDumpObjects(DumpObjects: LLVMOrcDumpObjectsRef);
    pub fn LLVMOrcDisposeIndirectStubsManager(ISM: LLVMOrcIndirectStubsManagerRef);
    pub fn LLVMOrcDisposeJITTargetMachineBuilder(JTMB: LLVMOrcJITTargetMachineBuilderRef);
    pub fn LLVMOrcDisposeLazyCallThroughManager(LCTM: LLVMOrcLazyCallThroughManagerRef);
    pub fn LLVMOrcDisposeMaterializationResponsibility(MR: LLVMOrcMaterializationResponsibilityRef);
    pub fn LLVMOrcDisposeMaterializationUnit(MU: LLVMOrcMaterializationUnitRef);
    pub fn LLVMOrcDisposeObjectLayer(ObjLayer: LLVMOrcObjectLayerRef);
    pub fn LLVMOrcDisposeSymbols(Symbols: *mut LLVMOrcSymbolStringPoolEntryRef);
    pub fn LLVMOrcDisposeThreadSafeContext(TSCtx: LLVMOrcThreadSafeContextRef);
    pub fn LLVMOrcDisposeThreadSafeModule(TSM: LLVMOrcThreadSafeModuleRef);
    pub fn LLVMOrcDumpObjects_CallOperator(
        DumpObjects: LLVMOrcDumpObjectsRef,
        ObjBuffer: *mut LLVMMemoryBufferRef,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcExecutionSessionCreateBareJITDylib(
        ES: LLVMOrcExecutionSessionRef,
        Name: *const ::libc::c_char,
    ) -> LLVMOrcJITDylibRef;
    pub fn LLVMOrcExecutionSessionCreateJITDylib(
        ES: LLVMOrcExecutionSessionRef,
        Result_: *mut LLVMOrcJITDylibRef,
        Name: *const ::libc::c_char,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcExecutionSessionGetJITDylibByName(
        ES: LLVMOrcExecutionSessionRef,
        Name: *const ::libc::c_char,
    ) -> LLVMOrcJITDylibRef;
    pub fn LLVMOrcExecutionSessionLookup(
        ES: LLVMOrcExecutionSessionRef,
        K: LLVMOrcLookupKind,
        SearchOrder: LLVMOrcCJITDylibSearchOrder,
        SearchOrderSize: usize,
        Symbols: LLVMOrcCLookupSet,
        SymbolsSize: usize,
        HandleResult: LLVMOrcExecutionSessionLookupHandleResultFunction,
        Ctx: *mut ::libc::c_void,
    );
    pub fn LLVMOrcIRTransformLayerEmit(
        IRTransformLayer: LLVMOrcIRTransformLayerRef,
        MR: LLVMOrcMaterializationResponsibilityRef,
        TSM: LLVMOrcThreadSafeModuleRef,
    );
    pub fn LLVMOrcIRTransformLayerSetTransform(
        IRTransformLayer: LLVMOrcIRTransformLayerRef,
        TransformFunction: LLVMOrcIRTransformLayerTransformFunction,
        Ctx: *mut ::libc::c_void,
    );
    pub fn LLVMOrcJITDylibAddGenerator(JD: LLVMOrcJITDylibRef, DG: LLVMOrcDefinitionGeneratorRef);
    pub fn LLVMOrcJITDylibClear(JD: LLVMOrcJITDylibRef) -> LLVMErrorRef;
    pub fn LLVMOrcJITDylibCreateResourceTracker(JD: LLVMOrcJITDylibRef) -> LLVMOrcResourceTrackerRef;
    pub fn LLVMOrcJITDylibDefine(JD: LLVMOrcJITDylibRef, MU: LLVMOrcMaterializationUnitRef) -> LLVMErrorRef;
    pub fn LLVMOrcJITDylibGetDefaultResourceTracker(JD: LLVMOrcJITDylibRef) -> LLVMOrcResourceTrackerRef;
    pub fn LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(
        TM: LLVMTargetMachineRef,
    ) -> LLVMOrcJITTargetMachineBuilderRef;
    pub fn LLVMOrcJITTargetMachineBuilderDetectHost(Result: *mut LLVMOrcJITTargetMachineBuilderRef) -> LLVMErrorRef;
    pub fn LLVMOrcJITTargetMachineBuilderGetTargetTriple(
        JTMB: LLVMOrcJITTargetMachineBuilderRef,
    ) -> *mut ::libc::c_char;
    pub fn LLVMOrcJITTargetMachineBuilderSetTargetTriple(
        JTMB: LLVMOrcJITTargetMachineBuilderRef,
        TargetTriple: *const ::libc::c_char,
    );
    pub fn LLVMOrcLazyReexports(
        LCTM: LLVMOrcLazyCallThroughManagerRef,
        ISM: LLVMOrcIndirectStubsManagerRef,
        SourceRef: LLVMOrcJITDylibRef,
        CallableAliases: LLVMOrcCSymbolAliasMapPairs,
        NumPairs: ::libc::size_t,
    ) -> LLVMOrcMaterializationUnitRef;
    pub fn LLVMOrcLookupStateContinueLookup(S: LLVMOrcLookupStateRef, Err: LLVMErrorRef);
    pub fn LLVMOrcMaterializationResponsibilityAddDependencies(
        MR: LLVMOrcMaterializationResponsibilityRef,
        Name: LLVMOrcSymbolStringPoolEntryRef,
        Dependencies: LLVMOrcCDependenceMapPairs,
        NumPairs: ::libc::size_t,
    );
    pub fn LLVMOrcMaterializationResponsibilityAddDependenciesForAll(
        MR: LLVMOrcMaterializationResponsibilityRef,
        Dependencies: LLVMOrcCDependenceMapPairs,
        NumPairs: ::libc::size_t,
    );
    pub fn LLVMOrcMaterializationResponsibilityDefineMaterializing(
        MR: LLVMOrcMaterializationResponsibilityRef,
        Pairs: LLVMOrcCSymbolFlagsMapPairs,
        NumPairs: ::libc::size_t,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcMaterializationResponsibilityDelegate(
        MR: LLVMOrcMaterializationResponsibilityRef,
        Symbols: *mut LLVMOrcSymbolStringPoolEntryRef,
        NumSymbols: ::libc::size_t,
        Result: *mut LLVMOrcMaterializationResponsibilityRef,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcMaterializationResponsibilityFailMaterialization(MR: LLVMOrcMaterializationResponsibilityRef);
    pub fn LLVMOrcMaterializationResponsibilityGetExecutionSession(
        MR: LLVMOrcMaterializationResponsibilityRef,
    ) -> LLVMOrcExecutionSessionRef;
    pub fn LLVMOrcMaterializationResponsibilityGetInitializerSymbol(
        MR: LLVMOrcMaterializationResponsibilityRef,
    ) -> LLVMOrcSymbolStringPoolEntryRef;
    pub fn LLVMOrcMaterializationResponsibilityGetRequestedSymbols(
        MR: LLVMOrcMaterializationResponsibilityRef,
        NumSymbols: *mut ::libc::size_t,
    ) -> *mut LLVMOrcSymbolStringPoolEntryRef;
    pub fn LLVMOrcMaterializationResponsibilityGetSymbols(
        MR: LLVMOrcMaterializationResponsibilityRef,
        NumPairs: *mut ::libc::size_t,
    ) -> LLVMOrcCSymbolFlagsMapPairs;
    pub fn LLVMOrcMaterializationResponsibilityGetTargetDylib(
        MR: LLVMOrcMaterializationResponsibilityRef,
    ) -> LLVMOrcJITDylibRef;
    pub fn LLVMOrcMaterializationResponsibilityNotifyEmitted(
        MR: LLVMOrcMaterializationResponsibilityRef,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcMaterializationResponsibilityNotifyResolved(
        MR: LLVMOrcMaterializationResponsibilityRef,
        Symbols: LLVMOrcCSymbolMapPairs,
        NumPairs: ::libc::size_t,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcMaterializationResponsibilityReplace(
        MR: LLVMOrcMaterializationResponsibilityRef,
        MU: LLVMOrcMaterializationUnitRef,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcObjectLayerAddObjectFile(
        ObjLayer: LLVMOrcObjectLayerRef,
        JD: LLVMOrcJITDylibRef,
        ObjBuffer: LLVMMemoryBufferRef,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcObjectLayerAddObjectFileWithRT(
        ObjLayer: LLVMOrcObjectLayerRef,
        RT: LLVMOrcResourceTrackerRef,
        ObjBuffer: LLVMMemoryBufferRef,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcObjectLayerEmit(
        ObjLayer: LLVMOrcObjectLayerRef,
        R: LLVMOrcMaterializationResponsibilityRef,
        ObjBuffer: LLVMMemoryBufferRef,
    );
    pub fn LLVMOrcObjectTransformLayerSetTransform(
        ObjTransformLayer: LLVMOrcObjectTransformLayerRef,
        TransformFunction: LLVMOrcObjectTransformLayerTransformFunction,
        Ctx: *mut ::libc::c_void,
    );
    pub fn LLVMOrcReleaseResourceTracker(RT: LLVMOrcResourceTrackerRef);
    pub fn LLVMOrcReleaseSymbolStringPoolEntry(S: LLVMOrcSymbolStringPoolEntryRef);
    pub fn LLVMOrcResourceTrackerRemove(RT: LLVMOrcResourceTrackerRef) -> LLVMErrorRef;
    pub fn LLVMOrcResourceTrackerTransferTo(SrcRT: LLVMOrcResourceTrackerRef, DstRT: LLVMOrcResourceTrackerRef);
    pub fn LLVMOrcRetainSymbolStringPoolEntry(S: LLVMOrcSymbolStringPoolEntryRef);
    pub fn LLVMOrcSymbolStringPoolEntryStr(S: LLVMOrcSymbolStringPoolEntryRef) -> *const ::libc::c_char;
    pub fn LLVMOrcThreadSafeContextGetContext(TSCtx: LLVMOrcThreadSafeContextRef) -> LLVMContextRef;
    pub fn LLVMOrcThreadSafeModuleWithModuleDo(
        TSM: LLVMOrcThreadSafeModuleRef,
        F: LLVMOrcGenericIRModuleOperationFunction,
        Ctx: *mut ::libc::c_void,
    ) -> LLVMErrorRef;
}

pub type LLVMMemoryManagerCreateContextCallback = extern "C" fn(CtxCtx: *mut ::libc::c_void);
pub type LLVMMemoryManagerNotifyTerminatingCallback = extern "C" fn(CtxCtx: *mut ::libc::c_void);

extern "C" {
    pub fn LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks(
        ES: LLVMOrcExecutionSessionRef,
        CreateContext: LLVMMemoryManagerCreateContextCallback,
        NotifyTerminating: LLVMMemoryManagerNotifyTerminatingCallback,
        AllocateCodeSection: LLVMMemoryManagerAllocateCodeSectionCallback,
        AllocateDataSection: LLVMMemoryManagerAllocateDataSectionCallback,
        FinalizeMemory: LLVMMemoryManagerFinalizeMemoryCallback,
        Destroy: LLVMMemoryManagerDestroyCallback,
    ) -> LLVMOrcObjectLayerRef;
    pub fn LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager(
        ES: LLVMOrcExecutionSessionRef,
    ) -> LLVMOrcObjectLayerRef;
    pub fn LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener(
        RTDyldObjLinkingLayer: LLVMOrcObjectLayerRef,
        Listener: LLVMJITEventListenerRef,
    );
}

pub type LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction = extern "C" fn(
    Ctx: *mut ::libc::c_void,
    ES: LLVMOrcExecutionSessionRef,
    Triple: *const ::libc::c_char,
) -> LLVMOrcObjectLayerRef;

#[derive(Debug)]
pub enum LLVMOrcOpaqueLLJITBuilder {}
pub type LLVMOrcLLJITBuilderRef = *mut LLVMOrcOpaqueLLJITBuilder;

#[derive(Debug)]
pub enum LLVMOrcOpaqueLLJIT {}
pub type LLVMOrcLLJITRef = *mut LLVMOrcOpaqueLLJIT;

extern "C" {
    pub fn LLVMOrcCreateLLJIT(Result: *mut LLVMOrcLLJITRef, Builder: LLVMOrcLLJITBuilderRef) -> LLVMErrorRef;
    pub fn LLVMOrcCreateLLJITBuilder() -> LLVMOrcLLJITBuilderRef;
    pub fn LLVMOrcDisposeLLJIT(J: LLVMOrcLLJITRef) -> LLVMErrorRef;
    pub fn LLVMOrcDisposeLLJITBuilder(Builder: LLVMOrcLLJITBuilderRef);
    pub fn LLVMOrcLLJITAddLLVMIRModule(
        J: LLVMOrcLLJITRef,
        JD: LLVMOrcJITDylibRef,
        TSM: LLVMOrcThreadSafeModuleRef,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcLLJITAddLLVMIRModuleWithRT(
        J: LLVMOrcLLJITRef,
        JD: LLVMOrcResourceTrackerRef,
        TSM: LLVMOrcThreadSafeModuleRef,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcLLJITAddObjectFile(
        J: LLVMOrcLLJITRef,
        JD: LLVMOrcJITDylibRef,
        ObjBuffer: LLVMMemoryBufferRef,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcLLJITAddObjectFileWithRT(
        J: LLVMOrcLLJITRef,
        RT: LLVMOrcResourceTrackerRef,
        ObjBuffer: LLVMMemoryBufferRef,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(
        Builder: LLVMOrcLLJITBuilderRef,
        JTMB: LLVMOrcJITTargetMachineBuilderRef,
    );
    pub fn LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(
        Builder: LLVMOrcLLJITBuilderRef,
        F: LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction,
        Ctx: *mut ::libc::c_void,
    );
    pub fn LLVMOrcLLJITGetDataLayoutStr(J: LLVMOrcLLJITRef) -> *const ::libc::c_char;
    pub fn LLVMOrcLLJITGetExecutionSession(J: LLVMOrcLLJITRef) -> LLVMOrcExecutionSessionRef;
    pub fn LLVMOrcLLJITGetGlobalPrefix(J: LLVMOrcLLJITRef) -> ::libc::c_char;
    pub fn LLVMOrcLLJITGetIRTransformLayer(J: LLVMOrcLLJITRef) -> LLVMOrcIRTransformLayerRef;
    pub fn LLVMOrcLLJITGetMainJITDylib(J: LLVMOrcLLJITRef) -> LLVMOrcJITDylibRef;
    pub fn LLVMOrcLLJITGetObjLinkingLayer(J: LLVMOrcLLJITRef) -> LLVMOrcObjectLayerRef;
    pub fn LLVMOrcLLJITGetObjTransformLayer(J: LLVMOrcLLJITRef) -> LLVMOrcObjectTransformLayerRef;
    pub fn LLVMOrcLLJITGetTripleString(J: LLVMOrcLLJITRef) -> *const ::libc::c_char;
    pub fn LLVMOrcLLJITLookup(
        J: LLVMOrcLLJITRef,
        Result: *mut LLVMOrcExecutorAddress,
        Name: *const ::libc::c_char,
    ) -> LLVMErrorRef;
    pub fn LLVMOrcLLJITMangleAndIntern(
        J: LLVMOrcLLJITRef,
        UnmangledName: *const ::libc::c_char,
    ) -> LLVMOrcSymbolStringPoolEntryRef;
}
