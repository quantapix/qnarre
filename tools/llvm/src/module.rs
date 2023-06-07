use crate::ctx::{AsContextRef, Context, ContextRef};
use crate::debug::{DICompileUnit, DWARFEmissionKind, DWARFSourceLanguage, DebugInfoBuilder};
use crate::pass::PassBuilderOptions;
use crate::target::*;
use crate::typ::*;
use crate::val::*;
use crate::Comdat;
use crate::DataLayout;
use crate::ExecutionEngine;
use crate::MemoryBuffer;
use crate::{to_c_str, LLVMString};
use crate::{AddressSpace, OptimizationLevel};
use llvm_lib::analysis::{LLVMVerifierFailureAction, LLVMVerifyModule};
use llvm_lib::bit_reader::LLVMParseBitcodeInContext;
use llvm_lib::bit_writer::{LLVMWriteBitcodeToFile, LLVMWriteBitcodeToMemoryBuffer};
use llvm_lib::core::*;
use llvm_lib::error::LLVMGetErrorMessage;
use llvm_lib::execution_engine::*;
use llvm_lib::prelude::{LLVMModuleRef, LLVMValueRef};
use llvm_lib::transforms::pass_builder::LLVMRunPasses;
use llvm_lib::LLVMLinkage;
use llvm_lib::LLVMModuleFlagBehavior;
use std::cell::{Cell, Ref, RefCell};
use std::ffi::CStr;
use std::fs::File;
use std::marker::PhantomData;
use std::mem::{forget, MaybeUninit};
use std::path::Path;
use std::ptr;
use std::rc::Rc;

#[llvm_enum(LLVMLinkage)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Linkage {
    #[llvm_variant(LLVMAppendingLinkage)]
    Appending,
    #[llvm_variant(LLVMAvailableExternallyLinkage)]
    AvailableExternally,
    #[llvm_variant(LLVMCommonLinkage)]
    Common,
    #[llvm_variant(LLVMDLLExportLinkage)]
    DLLExport,
    #[llvm_variant(LLVMDLLImportLinkage)]
    DLLImport,
    #[llvm_variant(LLVMExternalLinkage)]
    External,
    #[llvm_variant(LLVMExternalWeakLinkage)]
    ExternalWeak,
    #[llvm_variant(LLVMGhostLinkage)]
    Ghost,
    #[llvm_variant(LLVMInternalLinkage)]
    Internal,
    #[llvm_variant(LLVMLinkerPrivateLinkage)]
    LinkerPrivate,
    #[llvm_variant(LLVMLinkerPrivateWeakLinkage)]
    LinkerPrivateWeak,
    #[llvm_variant(LLVMLinkOnceAnyLinkage)]
    LinkOnceAny,
    #[llvm_variant(LLVMLinkOnceODRAutoHideLinkage)]
    LinkOnceODRAutoHide,
    #[llvm_variant(LLVMLinkOnceODRLinkage)]
    LinkOnceODR,
    #[llvm_variant(LLVMPrivateLinkage)]
    Private,
    #[llvm_variant(LLVMWeakAnyLinkage)]
    WeakAny,
    #[llvm_variant(LLVMWeakODRLinkage)]
    WeakODR,
}
#[derive(Debug, PartialEq, Eq)]

pub struct Module<'ctx> {
    data_layout: RefCell<Option<DataLayout>>,
    pub(crate) module: Cell<LLVMModuleRef>,
    pub(crate) owned_by_ee: RefCell<Option<ExecutionEngine<'ctx>>>,
    _marker: PhantomData<&'ctx Context>,
}
impl<'ctx> Module<'ctx> {
    pub(crate) unsafe fn new(module: LLVMModuleRef) -> Self {
        debug_assert!(!module.is_null());
        Module {
            module: Cell::new(module),
            owned_by_ee: RefCell::new(None),
            data_layout: RefCell::new(Some(Module::get_borrowed_data_layout(module))),
            _marker: PhantomData,
        }
    }
    pub fn as_mut_ptr(&self) -> LLVMModuleRef {
        self.module.get()
    }
    pub fn add_function(&self, name: &str, ty: FunctionType<'ctx>, linkage: Option<Linkage>) -> FunctionValue<'ctx> {
        let c_string = to_c_str(name);
        let fn_value = unsafe {
            FunctionValue::new(LLVMAddFunction(self.module.get(), c_string.as_ptr(), ty.as_type_ref()))
                .expect("add_function should always succeed in adding a new function")
        };
        if let Some(linkage) = linkage {
            fn_value.set_linkage(linkage)
        }
        fn_value
    }
    pub fn get_context(&self) -> ContextRef<'ctx> {
        unsafe { ContextRef::new(LLVMGetModuleContext(self.module.get())) }
    }
    pub fn get_first_function(&self) -> Option<FunctionValue<'ctx>> {
        unsafe { FunctionValue::new(LLVMGetFirstFunction(self.module.get())) }
    }
    pub fn get_last_function(&self) -> Option<FunctionValue<'ctx>> {
        unsafe { FunctionValue::new(LLVMGetLastFunction(self.module.get())) }
    }
    pub fn get_function(&self, name: &str) -> Option<FunctionValue<'ctx>> {
        let c_string = to_c_str(name);
        unsafe { FunctionValue::new(LLVMGetNamedFunction(self.module.get(), c_string.as_ptr())) }
    }
    pub fn get_functions(&self) -> FunctionIterator<'ctx> {
        FunctionIterator::from_module(self)
    }
    pub fn get_struct_type(&self, name: &str) -> Option<StructType<'ctx>> {
        self.get_context().get_struct_type(name)
    }
    pub fn set_triple(&self, triple: &TargetTriple) {
        unsafe { LLVMSetTarget(self.module.get(), triple.as_ptr()) }
    }
    pub fn get_triple(&self) -> TargetTriple {
        let target_str = unsafe { LLVMGetTarget(self.module.get()) };
        unsafe { TargetTriple::new(LLVMString::create_from_c_str(CStr::from_ptr(target_str))) }
    }
    pub fn create_execution_engine(&self) -> Result<ExecutionEngine<'ctx>, LLVMString> {
        Target::initialize_native(&InitializationConfig::default()).map_err(|mut err_string| {
            err_string.push('\0');
            LLVMString::create_from_str(&err_string)
        })?;
        if self.owned_by_ee.borrow().is_some() {
            let string = "This module is already owned by an ExecutionEngine.\0";
            return Err(LLVMString::create_from_str(string));
        }
        let mut execution_engine = MaybeUninit::uninit();
        let mut err_string = MaybeUninit::uninit();
        let code = unsafe {
            LLVMCreateExecutionEngineForModule(
                execution_engine.as_mut_ptr(),
                self.module.get(),
                err_string.as_mut_ptr(),
            )
        };
        if code == 1 {
            unsafe {
                return Err(LLVMString::new(err_string.assume_init()));
            }
        }
        let execution_engine = unsafe { execution_engine.assume_init() };
        let execution_engine = unsafe { ExecutionEngine::new(Rc::new(execution_engine), false) };
        *self.owned_by_ee.borrow_mut() = Some(execution_engine.clone());
        Ok(execution_engine)
    }
    pub fn create_interpreter_execution_engine(&self) -> Result<ExecutionEngine<'ctx>, LLVMString> {
        Target::initialize_native(&InitializationConfig::default()).map_err(|mut err_string| {
            err_string.push('\0');
            LLVMString::create_from_str(&err_string)
        })?;
        if self.owned_by_ee.borrow().is_some() {
            let string = "This module is already owned by an ExecutionEngine.\0";
            return Err(LLVMString::create_from_str(string));
        }
        let mut execution_engine = MaybeUninit::uninit();
        let mut err_string = MaybeUninit::uninit();
        let code = unsafe {
            LLVMCreateInterpreterForModule(
                execution_engine.as_mut_ptr(),
                self.module.get(),
                err_string.as_mut_ptr(),
            )
        };
        if code == 1 {
            unsafe {
                return Err(LLVMString::new(err_string.assume_init()));
            }
        }
        let execution_engine = unsafe { execution_engine.assume_init() };
        let execution_engine = unsafe { ExecutionEngine::new(Rc::new(execution_engine), false) };
        *self.owned_by_ee.borrow_mut() = Some(execution_engine.clone());
        Ok(execution_engine)
    }
    pub fn create_jit_execution_engine(
        &self,
        opt_level: OptimizationLevel,
    ) -> Result<ExecutionEngine<'ctx>, LLVMString> {
        Target::initialize_native(&InitializationConfig::default()).map_err(|mut err_string| {
            err_string.push('\0');
            LLVMString::create_from_str(&err_string)
        })?;
        if self.owned_by_ee.borrow().is_some() {
            let string = "This module is already owned by an ExecutionEngine.\0";
            return Err(LLVMString::create_from_str(string));
        }
        let mut execution_engine = MaybeUninit::uninit();
        let mut err_string = MaybeUninit::uninit();
        let code = unsafe {
            LLVMCreateJITCompilerForModule(
                execution_engine.as_mut_ptr(),
                self.module.get(),
                opt_level as u32,
                err_string.as_mut_ptr(),
            )
        };
        if code == 1 {
            unsafe {
                return Err(LLVMString::new(err_string.assume_init()));
            }
        }
        let execution_engine = unsafe { execution_engine.assume_init() };
        let execution_engine = unsafe { ExecutionEngine::new(Rc::new(execution_engine), true) };
        *self.owned_by_ee.borrow_mut() = Some(execution_engine.clone());
        Ok(execution_engine)
    }
    pub fn add_global<T: BasicType<'ctx>>(
        &self,
        type_: T,
        address_space: Option<AddressSpace>,
        name: &str,
    ) -> GlobalValue<'ctx> {
        let c_string = to_c_str(name);
        let value = unsafe {
            match address_space {
                Some(address_space) => LLVMAddGlobalInAddressSpace(
                    self.module.get(),
                    type_.as_type_ref(),
                    c_string.as_ptr(),
                    address_space.0,
                ),
                None => LLVMAddGlobal(self.module.get(), type_.as_type_ref(), c_string.as_ptr()),
            }
        };
        unsafe { GlobalValue::new(value) }
    }
    pub fn write_bitcode_to_path(&self, path: &Path) -> bool {
        let path_str = path.to_str().expect("Did not find a valid Unicode path string");
        let c_string = to_c_str(path_str);
        unsafe { LLVMWriteBitcodeToFile(self.module.get(), c_string.as_ptr()) == 0 }
    }
    pub fn write_bitcode_to_file(&self, file: &File, should_close: bool, unbuffered: bool) -> bool {
        #[cfg(unix)]
        {
            use llvm_lib::bit_writer::LLVMWriteBitcodeToFD;
            use std::os::unix::io::AsRawFd;
            unsafe {
                LLVMWriteBitcodeToFD(
                    self.module.get(),
                    file.as_raw_fd(),
                    should_close as i32,
                    unbuffered as i32,
                ) == 0
            }
        }
        #[cfg(not(unix))]
        return false;
    }
    pub fn write_bitcode_to_memory(&self) -> MemoryBuffer {
        let memory_buffer = unsafe { LLVMWriteBitcodeToMemoryBuffer(self.module.get()) };
        unsafe { MemoryBuffer::new(memory_buffer) }
    }
    pub fn verify(&self) -> Result<(), LLVMString> {
        let mut err_str = MaybeUninit::uninit();
        let action = LLVMVerifierFailureAction::LLVMReturnStatusAction;
        let code = unsafe { LLVMVerifyModule(self.module.get(), action, err_str.as_mut_ptr()) };
        let err_str = unsafe { err_str.assume_init() };
        if code == 1 && !err_str.is_null() {
            return unsafe { Err(LLVMString::new(err_str)) };
        }
        Ok(())
    }
    fn get_borrowed_data_layout(module: LLVMModuleRef) -> DataLayout {
        let data_layout = unsafe {
            use llvm_lib::core::LLVMGetDataLayoutStr;
            LLVMGetDataLayoutStr(module)
        };
        unsafe { DataLayout::new_borrowed(data_layout) }
    }
    pub fn get_data_layout(&self) -> Ref<DataLayout> {
        Ref::map(self.data_layout.borrow(), |l| {
            l.as_ref().expect("DataLayout should always exist until Drop")
        })
    }
    pub fn set_data_layout(&self, data_layout: &DataLayout) {
        unsafe {
            LLVMSetDataLayout(self.module.get(), data_layout.as_ptr());
        }
        *self.data_layout.borrow_mut() = Some(Module::get_borrowed_data_layout(self.module.get()));
    }
    pub fn print_to_stderr(&self) {
        unsafe {
            LLVMDumpModule(self.module.get());
        }
    }
    pub fn print_to_string(&self) -> LLVMString {
        unsafe { LLVMString::new(LLVMPrintModuleToString(self.module.get())) }
    }
    pub fn print_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), LLVMString> {
        let path_str = path
            .as_ref()
            .to_str()
            .expect("Did not find a valid Unicode path string");
        let path = to_c_str(path_str);
        let mut err_string = MaybeUninit::uninit();
        let return_code = unsafe {
            LLVMPrintModuleToFile(
                self.module.get(),
                path.as_ptr() as *const ::libc::c_char,
                err_string.as_mut_ptr(),
            )
        };
        if return_code == 1 {
            unsafe {
                return Err(LLVMString::new(err_string.assume_init()));
            }
        }
        Ok(())
    }
    pub fn to_string(&self) -> String {
        self.print_to_string().to_string()
    }
    pub fn set_inline_assembly(&self, asm: &str) {
        {
            use llvm_lib::core::LLVMSetModuleInlineAsm2;
            unsafe { LLVMSetModuleInlineAsm2(self.module.get(), asm.as_ptr() as *const ::libc::c_char, asm.len()) }
        }
    }
    pub fn add_global_metadata(&self, key: &str, metadata: &MetadataValue<'ctx>) -> Result<(), &'static str> {
        if !metadata.is_node() {
            return Err("metadata is expected to be a node.");
        }
        let c_string = to_c_str(key);
        unsafe {
            LLVMAddNamedMetadataOperand(self.module.get(), c_string.as_ptr(), metadata.as_value_ref());
        }
        Ok(())
    }
    pub fn get_global_metadata_size(&self, key: &str) -> u32 {
        let c_string = to_c_str(key);
        unsafe { LLVMGetNamedMetadataNumOperands(self.module.get(), c_string.as_ptr()) }
    }
    pub fn get_global_metadata(&self, key: &str) -> Vec<MetadataValue<'ctx>> {
        let c_string = to_c_str(key);
        let count = self.get_global_metadata_size(key) as usize;
        let mut vec: Vec<LLVMValueRef> = Vec::with_capacity(count);
        let ptr = vec.as_mut_ptr();
        unsafe {
            LLVMGetNamedMetadataOperands(self.module.get(), c_string.as_ptr(), ptr);
            vec.set_len(count);
        };
        vec.iter().map(|val| unsafe { MetadataValue::new(*val) }).collect()
    }
    pub fn get_first_global(&self) -> Option<GlobalValue<'ctx>> {
        let value = unsafe { LLVMGetFirstGlobal(self.module.get()) };
        if value.is_null() {
            return None;
        }
        unsafe { Some(GlobalValue::new(value)) }
    }
    pub fn get_last_global(&self) -> Option<GlobalValue<'ctx>> {
        let value = unsafe { LLVMGetLastGlobal(self.module.get()) };
        if value.is_null() {
            return None;
        }
        unsafe { Some(GlobalValue::new(value)) }
    }
    pub fn get_global(&self, name: &str) -> Option<GlobalValue<'ctx>> {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMGetNamedGlobal(self.module.get(), c_string.as_ptr()) };
        if value.is_null() {
            return None;
        }
        unsafe { Some(GlobalValue::new(value)) }
    }
    pub fn get_globals(&self) -> GlobalIterator<'ctx> {
        GlobalIterator::from_module(self)
    }
    pub fn parse_bitcode_from_buffer(
        buffer: &MemoryBuffer,
        context: impl AsContextRef<'ctx>,
    ) -> Result<Self, LLVMString> {
        let mut module = MaybeUninit::uninit();
        let mut err_string = MaybeUninit::uninit();
        #[allow(deprecated)]
        let success = unsafe {
            LLVMParseBitcodeInContext(
                context.as_ctx_ref(),
                buffer.memory_buffer,
                module.as_mut_ptr(),
                err_string.as_mut_ptr(),
            )
        };
        if success != 0 {
            unsafe {
                return Err(LLVMString::new(err_string.assume_init()));
            }
        }
        unsafe { Ok(Module::new(module.assume_init())) }
    }
    pub fn parse_bitcode_from_path<P: AsRef<Path>>(
        path: P,
        context: impl AsContextRef<'ctx>,
    ) -> Result<Self, LLVMString> {
        let buffer = MemoryBuffer::create_from_file(path.as_ref())?;
        Self::parse_bitcode_from_buffer(&buffer, context)
    }
    pub fn get_name(&self) -> &CStr {
        let mut length = 0;
        let cstr_ptr = unsafe { LLVMGetModuleIdentifier(self.module.get(), &mut length) };
        unsafe { CStr::from_ptr(cstr_ptr) }
    }
    pub fn set_name(&self, name: &str) {
        unsafe { LLVMSetModuleIdentifier(self.module.get(), name.as_ptr() as *const ::libc::c_char, name.len()) }
    }
    pub fn get_source_file_name(&self) -> &CStr {
        use llvm_lib::core::LLVMGetSourceFileName;
        let mut len = 0;
        let ptr = unsafe { LLVMGetSourceFileName(self.module.get(), &mut len) };
        unsafe { CStr::from_ptr(ptr) }
    }
    pub fn set_source_file_name(&self, file_name: &str) {
        use llvm_lib::core::LLVMSetSourceFileName;
        unsafe {
            LLVMSetSourceFileName(
                self.module.get(),
                file_name.as_ptr() as *const ::libc::c_char,
                file_name.len(),
            )
        }
    }
    pub fn link_in_module(&self, other: Self) -> Result<(), LLVMString> {
        if other.owned_by_ee.borrow().is_some() {
            let string = "Cannot link a module which is already owned by an ExecutionEngine.\0";
            return Err(LLVMString::create_from_str(string));
        }
        use libc::c_void;
        use llvm_lib::linker::LLVMLinkModules2;
        let context = self.get_context();
        let mut char_ptr: *mut ::libc::c_char = ptr::null_mut();
        let char_ptr_ptr = &mut char_ptr as *mut *mut ::libc::c_char as *mut *mut c_void as *mut c_void;
        context.set_diagnostic_handler(get_error_str_diagnostic_handler, char_ptr_ptr);
        let code = unsafe { LLVMLinkModules2(self.module.get(), other.module.get()) };
        forget(other);
        if code == 1 {
            debug_assert!(!char_ptr.is_null());
            unsafe { Err(LLVMString::new(char_ptr)) }
        } else {
            Ok(())
        }
    }
    pub fn get_or_insert_comdat(&self, name: &str) -> Comdat {
        use llvm_lib::comdat::LLVMGetOrInsertComdat;
        let c_string = to_c_str(name);
        let comdat_ptr = unsafe { LLVMGetOrInsertComdat(self.module.get(), c_string.as_ptr()) };
        unsafe { Comdat::new(comdat_ptr) }
    }
    pub fn get_flag(&self, key: &str) -> Option<MetadataValue<'ctx>> {
        use llvm_lib::core::LLVMMetadataAsValue;
        let flag = unsafe { LLVMGetModuleFlag(self.module.get(), key.as_ptr() as *const ::libc::c_char, key.len()) };
        if flag.is_null() {
            return None;
        }
        let flag_value = unsafe { LLVMMetadataAsValue(LLVMGetModuleContext(self.module.get()), flag) };
        unsafe { Some(MetadataValue::new(flag_value)) }
    }
    pub fn add_metadata_flag(&self, key: &str, behavior: FlagBehavior, flag: MetadataValue<'ctx>) {
        let md = flag.as_metadata_ref();
        unsafe {
            LLVMAddModuleFlag(
                self.module.get(),
                behavior.into(),
                key.as_ptr() as *mut ::libc::c_char,
                key.len(),
                md,
            )
        }
    }
    pub fn add_basic_value_flag<BV: BasicValue<'ctx>>(&self, key: &str, behavior: FlagBehavior, flag: BV) {
        use llvm_lib::core::LLVMValueAsMetadata;
        let md = unsafe { LLVMValueAsMetadata(flag.as_value_ref()) };
        unsafe {
            LLVMAddModuleFlag(
                self.module.get(),
                behavior.into(),
                key.as_ptr() as *mut ::libc::c_char,
                key.len(),
                md,
            )
        }
    }
    pub fn strip_debug_info(&self) -> bool {
        use llvm_lib::debuginfo::LLVMStripModuleDebugInfo;
        unsafe { LLVMStripModuleDebugInfo(self.module.get()) == 1 }
    }
    pub fn get_debug_metadata_version(&self) -> libc::c_uint {
        use llvm_lib::debuginfo::LLVMGetModuleDebugMetadataVersion;
        unsafe { LLVMGetModuleDebugMetadataVersion(self.module.get()) }
    }
    pub fn create_debug_info_builder(
        &self,
        allow_unresolved: bool,
        language: DWARFSourceLanguage,
        filename: &str,
        directory: &str,
        producer: &str,
        is_optimized: bool,
        flags: &str,
        runtime_ver: libc::c_uint,
        split_name: &str,
        kind: DWARFEmissionKind,
        dwo_id: libc::c_uint,
        split_debug_inlining: bool,
        debug_info_for_profiling: bool,
        sysroot: &str,
        sdk: &str,
    ) -> (DebugInfoBuilder<'ctx>, DICompileUnit<'ctx>) {
        DebugInfoBuilder::new(
            self,
            allow_unresolved,
            language,
            filename,
            directory,
            producer,
            is_optimized,
            flags,
            runtime_ver,
            split_name,
            kind,
            dwo_id,
            split_debug_inlining,
            debug_info_for_profiling,
            sysroot,
            sdk,
        )
    }
    pub fn run_passes(
        &self,
        passes: &str,
        machine: &TargetMachine,
        options: PassBuilderOptions,
    ) -> Result<(), LLVMString> {
        unsafe {
            let error = LLVMRunPasses(
                self.module.get(),
                to_c_str(passes).as_ptr(),
                machine.target_machine,
                options.options_ref,
            );
            if error == std::ptr::null_mut() {
                Ok(())
            } else {
                let message = LLVMGetErrorMessage(error);
                Err(LLVMString::new(message as *const libc::c_char))
            }
        }
    }
}
impl Clone for Module<'_> {
    fn clone(&self) -> Self {
        let verify = self.verify();
        assert!(
            verify.is_ok(),
            "Cloning a Module seems to segfault when module is not valid. We are preventing that here. Error: {}",
            verify.unwrap_err()
        );
        unsafe { Module::new(LLVMCloneModule(self.module.get())) }
    }
}
impl Drop for Module<'_> {
    fn drop(&mut self) {
        if self.owned_by_ee.borrow_mut().take().is_none() {
            unsafe {
                LLVMDisposeModule(self.module.get());
            }
        }
    }
}

#[llvm_enum(LLVMModuleFlagBehavior)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FlagBehavior {
    #[llvm_variant(LLVMModuleFlagBehaviorError)]
    Error,
    #[llvm_variant(LLVMModuleFlagBehaviorWarning)]
    Warning,
    #[llvm_variant(LLVMModuleFlagBehaviorRequire)]
    Require,
    #[llvm_variant(LLVMModuleFlagBehaviorOverride)]
    Override,
    #[llvm_variant(LLVMModuleFlagBehaviorAppend)]
    Append,
    #[llvm_variant(LLVMModuleFlagBehaviorAppendUnique)]
    AppendUnique,
}

#[derive(Debug)]
pub struct FunctionIterator<'ctx>(FunctionIteratorInner<'ctx>);

#[derive(Debug)]
enum FunctionIteratorInner<'ctx> {
    Empty,
    Start(FunctionValue<'ctx>),
    Previous(FunctionValue<'ctx>),
}
impl<'ctx> FunctionIterator<'ctx> {
    fn from_module(module: &Module<'ctx>) -> Self {
        use FunctionIteratorInner::*;
        match module.get_first_function() {
            None => Self(Empty),
            Some(first) => Self(Start(first)),
        }
    }
}
impl<'ctx> Iterator for FunctionIterator<'ctx> {
    type Item = FunctionValue<'ctx>;
    fn next(&mut self) -> Option<Self::Item> {
        use FunctionIteratorInner::*;
        match self.0 {
            Empty => None,
            Start(first) => {
                self.0 = Previous(first);
                Some(first)
            },
            Previous(prev) => match prev.get_next_function() {
                Some(current) => {
                    self.0 = Previous(current);
                    Some(current)
                },
                None => None,
            },
        }
    }
}

#[derive(Debug)]
pub struct GlobalIterator<'ctx>(GlobalIteratorInner<'ctx>);

#[derive(Debug)]
enum GlobalIteratorInner<'ctx> {
    Empty,
    Start(GlobalValue<'ctx>),
    Previous(GlobalValue<'ctx>),
}
impl<'ctx> GlobalIterator<'ctx> {
    fn from_module(module: &Module<'ctx>) -> Self {
        use GlobalIteratorInner::*;
        match module.get_first_global() {
            None => Self(Empty),
            Some(first) => Self(Start(first)),
        }
    }
}
impl<'ctx> Iterator for GlobalIterator<'ctx> {
    type Item = GlobalValue<'ctx>;
    fn next(&mut self) -> Option<Self::Item> {
        use GlobalIteratorInner::*;
        match self.0 {
            Empty => None,
            Start(first) => {
                self.0 = Previous(first);
                Some(first)
            },
            Previous(prev) => match prev.get_next_global() {
                Some(current) => {
                    self.0 = Previous(current);
                    Some(current)
                },
                None => None,
            },
        }
    }
}
