use llvm_lib::target::*;
use llvm_lib::target_machine::*;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::default::Default;
use std::ffi::CStr;
use std::fmt;
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr;

use crate::pass::PassManager;
use crate::typ::*;
use crate::val::*;
use crate::AsContextRef;
use crate::*;

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum CodeModel {
    Default,
    JITDefault,
    Small,
    Kernel,
    Medium,
    Large,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum RelocMode {
    Default,
    Static,
    PIC,
    DynamicNoPic,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum FileType {
    Assembly,
    Object,
}
impl FileType {
    fn as_llvm_file_type(&self) -> LLVMCodeGenFileType {
        match *self {
            FileType::Assembly => LLVMCodeGenFileType::LLVMAssemblyFile,
            FileType::Object => LLVMCodeGenFileType::LLVMObjectFile,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InitializationConfig {
    pub asm_parser: bool,
    pub asm_printer: bool,
    pub base: bool,
    pub disassembler: bool,
    pub info: bool,
    pub machine_code: bool,
}
impl Default for InitializationConfig {
    fn default() -> Self {
        InitializationConfig {
            asm_parser: true,
            asm_printer: true,
            base: true,
            disassembler: true,
            info: true,
            machine_code: true,
        }
    }
}

#[derive(Eq)]
pub struct TargetTriple {
    pub val: LLVMString,
}
impl TargetTriple {
    pub unsafe fn new(val: LLVMString) -> TargetTriple {
        TargetTriple { val }
    }
    pub fn create(x: &str) -> TargetTriple {
        let y = to_c_str(x);
        TargetTriple {
            val: LLVMString::create_from_c_str(&y),
        }
    }
    pub fn as_str(&self) -> &CStr {
        unsafe { CStr::from_ptr(self.as_ptr()) }
    }
    pub fn as_ptr(&self) -> *const ::libc::c_char {
        self.val.as_ptr()
    }
}
impl PartialEq for TargetTriple {
    fn eq(&self, x: &TargetTriple) -> bool {
        self.val == x.val
    }
}
impl fmt::Debug for TargetTriple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TargetTriple({:?})", self.val)
    }
}
impl fmt::Display for TargetTriple {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "TargetTriple({:?})", self.val)
    }
}

static TARGET_LOCK: Lazy<RwLock<()>> = Lazy::new(|| RwLock::new(()));

#[derive(Debug, Eq, PartialEq)]
pub struct Target {
    raw: LLVMTargetRef,
}
impl Target {
    pub unsafe fn new(raw: LLVMTargetRef) -> Self {
        assert!(!raw.is_null());
        Target { raw }
    }
    pub fn as_mut_ptr(&self) -> LLVMTargetRef {
        self.raw
    }
    #[cfg(feature = "target-x86")]
    pub fn initialize_x86(config: &InitializationConfig) {
        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeX86Target() };
        }
        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeX86TargetInfo() };
        }
        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeX86AsmPrinter() };
        }
        if config.asm_parser {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeX86AsmParser() };
        }
        if config.disassembler {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeX86Disassembler() };
        }
        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeX86TargetMC() };
        }
    }
    #[cfg(feature = "target-nvptx")]
    pub fn initialize_nvptx(config: &InitializationConfig) {
        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeNVPTXTarget() };
        }
        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeNVPTXTargetInfo() };
        }
        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeNVPTXAsmPrinter() };
        }
        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeNVPTXTargetMC() };
        }
    }
    pub fn initialize_native(config: &InitializationConfig) -> Result<(), String> {
        if config.base {
            let _guard = TARGET_LOCK.write();
            let code = unsafe { LLVM_InitializeNativeTarget() };
            if code == 1 {
                return Err("Unknown error in initializing native target".into());
            }
        }
        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            let code = unsafe { LLVM_InitializeNativeAsmPrinter() };
            if code == 1 {
                return Err("Unknown error in initializing native asm printer".into());
            }
        }
        if config.asm_parser {
            let _guard = TARGET_LOCK.write();
            let code = unsafe { LLVM_InitializeNativeAsmParser() };
            if code == 1 {
                return Err("Unknown error in initializing native asm parser".into());
            }
        }
        if config.disassembler {
            let _guard = TARGET_LOCK.write();
            let code = unsafe { LLVM_InitializeNativeDisassembler() };
            if code == 1 {
                return Err("Unknown error in initializing native disassembler".into());
            }
        }
        Ok(())
    }
    pub fn initialize_all(config: &InitializationConfig) {
        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVM_InitializeAllTargets() };
        }
        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVM_InitializeAllTargetInfos() };
        }
        if config.asm_parser {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVM_InitializeAllAsmParsers() };
        }
        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVM_InitializeAllAsmPrinters() };
        }
        if config.disassembler {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVM_InitializeAllDisassemblers() };
        }
        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVM_InitializeAllTargetMCs() };
        }
    }
    pub fn create_target_machine(
        &self,
        triple: &TargetTriple,
        cpu: &str,
        features: &str,
        level: OptimizationLevel,
        reloc_mode: RelocMode,
        code_model: CodeModel,
    ) -> Option<TargetMachine> {
        let cpu = to_c_str(cpu);
        let features = to_c_str(features);
        let level = match level {
            OptimizationLevel::None => LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
            OptimizationLevel::Less => LLVMCodeGenOptLevel::LLVMCodeGenLevelLess,
            OptimizationLevel::Default => LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
            OptimizationLevel::Aggressive => LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
        };
        let code_model = match code_model {
            CodeModel::Default => LLVMCodeModel::LLVMCodeModelDefault,
            CodeModel::JITDefault => LLVMCodeModel::LLVMCodeModelJITDefault,
            CodeModel::Small => LLVMCodeModel::LLVMCodeModelSmall,
            CodeModel::Kernel => LLVMCodeModel::LLVMCodeModelKernel,
            CodeModel::Medium => LLVMCodeModel::LLVMCodeModelMedium,
            CodeModel::Large => LLVMCodeModel::LLVMCodeModelLarge,
        };
        let reloc_mode = match reloc_mode {
            RelocMode::Default => LLVMRelocMode::LLVMRelocDefault,
            RelocMode::Static => LLVMRelocMode::LLVMRelocStatic,
            RelocMode::PIC => LLVMRelocMode::LLVMRelocPIC,
            RelocMode::DynamicNoPic => LLVMRelocMode::LLVMRelocDynamicNoPic,
        };
        let target_machine = unsafe {
            LLVMCreateTargetMachine(
                self.raw,
                triple.as_ptr(),
                cpu.as_ptr(),
                features.as_ptr(),
                level,
                reloc_mode,
                code_model,
            )
        };
        if target_machine.is_null() {
            return None;
        }
        unsafe { Some(TargetMachine::new(target_machine)) }
    }
    pub fn get_first() -> Option<Self> {
        let y = {
            let _guard = TARGET_LOCK.read();
            unsafe { LLVMGetFirstTarget() }
        };
        if y.is_null() {
            return None;
        }
        unsafe { Some(Target::new(y)) }
    }
    pub fn get_next(&self) -> Option<Self> {
        let y = unsafe { LLVMGetNextTarget(self.raw) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(Target::new(y)) }
    }
    pub fn get_name(&self) -> &CStr {
        unsafe { CStr::from_ptr(LLVMGetTargetName(self.raw)) }
    }
    pub fn get_description(&self) -> &CStr {
        unsafe { CStr::from_ptr(LLVMGetTargetDescription(self.raw)) }
    }
    pub fn from_name(name: &str) -> Option<Self> {
        let y = to_c_str(name);
        Self::from_name_raw(y.as_ptr())
    }
    pub fn from_name_raw(c_string: *const ::libc::c_char) -> Option<Self> {
        let y = {
            let _guard = TARGET_LOCK.read();
            unsafe { LLVMGetTargetFromName(c_string) }
        };
        if y.is_null() {
            return None;
        }
        unsafe { Some(Target::new(y)) }
    }
    pub fn from_triple(triple: &TargetTriple) -> Result<Self, LLVMString> {
        let mut y = ptr::null_mut();
        let mut e = MaybeUninit::uninit();
        let code = {
            let _guard = TARGET_LOCK.read();
            unsafe { LLVMGetTargetFromTriple(triple.as_ptr(), &mut y, e.as_mut_ptr()) }
        };
        if code == 1 {
            unsafe {
                return Err(LLVMString::new(e.assume_init()));
            }
        }
        unsafe { Ok(Target::new(y)) }
    }
    pub fn has_jit(&self) -> bool {
        unsafe { LLVMTargetHasJIT(self.raw) == 1 }
    }
    pub fn has_target_machine(&self) -> bool {
        unsafe { LLVMTargetHasTargetMachine(self.raw) == 1 }
    }
    pub fn has_asm_backend(&self) -> bool {
        unsafe { LLVMTargetHasAsmBackend(self.raw) == 1 }
    }
}

#[derive(Debug)]
pub struct TargetMachine {
    pub raw: LLVMTargetMachineRef,
}
impl TargetMachine {
    pub unsafe fn new(raw: LLVMTargetMachineRef) -> Self {
        assert!(!raw.is_null());
        TargetMachine { raw }
    }
    pub fn as_mut_ptr(&self) -> LLVMTargetMachineRef {
        self.raw
    }
    pub fn get_target(&self) -> Target {
        unsafe { Target::new(LLVMGetTargetMachineTarget(self.raw)) }
    }
    pub fn get_triple(&self) -> TargetTriple {
        let y = unsafe { LLVMString::new(LLVMGetTargetMachineTriple(self.raw)) };
        unsafe { TargetTriple::new(y) }
    }
    pub fn get_default_triple() -> TargetTriple {
        let y = unsafe { LLVMString::new(LLVMGetDefaultTargetTriple()) };
        unsafe { TargetTriple::new(y) }
    }
    pub fn normalize_triple(triple: &TargetTriple) -> TargetTriple {
        let y = unsafe { LLVMString::new(LLVMNormalizeTargetTriple(triple.as_ptr())) };
        unsafe { TargetTriple::new(y) }
    }
    pub fn get_host_cpu_name() -> LLVMString {
        unsafe { LLVMString::new(LLVMGetHostCPUName()) }
    }
    pub fn get_host_cpu_features() -> LLVMString {
        unsafe { LLVMString::new(LLVMGetHostCPUFeatures()) }
    }
    pub fn get_cpu(&self) -> LLVMString {
        unsafe { LLVMString::new(LLVMGetTargetMachineCPU(self.raw)) }
    }
    pub fn get_feature_string(&self) -> &CStr {
        unsafe { CStr::from_ptr(LLVMGetTargetMachineFeatureString(self.raw)) }
    }
    pub fn get_target_data(&self) -> TargetData {
        unsafe { TargetData::new(LLVMCreateTargetDataLayout(self.raw)) }
    }
    pub fn set_asm_verbosity(&self, verbosity: bool) {
        unsafe { LLVMSetTargetMachineAsmVerbosity(self.raw, verbosity as i32) }
    }
    pub fn add_analysis_passes<T>(&self, x: &PassManager<T>) {
        unsafe { LLVMAddAnalysisPasses(self.raw, x.raw) }
    }
    pub fn write_to_memory_buffer(&self, m: &Module, t: FileType) -> Result<MemoryBuffer, LLVMString> {
        let mut y = ptr::null_mut();
        let mut e = MaybeUninit::uninit();
        let code = unsafe {
            let module_ptr = m.module.get();
            let file_type_ptr = t.as_llvm_file_type();
            LLVMTargetMachineEmitToMemoryBuffer(self.raw, module_ptr, file_type_ptr, e.as_mut_ptr(), &mut y)
        };
        if code == 1 {
            unsafe {
                return Err(LLVMString::new(e.assume_init()));
            }
        }
        unsafe { Ok(MemoryBuffer::new(y)) }
    }
    pub fn write_to_file(&self, m: &Module, t: FileType, path: &Path) -> Result<(), LLVMString> {
        let path = path.to_str().expect("Did not find a valid Unicode path string");
        let path_c_string = to_c_str(path);
        let mut e = MaybeUninit::uninit();
        let code = unsafe {
            let module_ptr = m.module.get();
            let path_ptr = path_c_string.as_ptr() as *mut _;
            let file_type_ptr = t.as_llvm_file_type();
            LLVMTargetMachineEmitToFile(self.raw, module_ptr, path_ptr, file_type_ptr, e.as_mut_ptr())
        };
        if code == 1 {
            unsafe {
                return Err(LLVMString::new(e.assume_init()));
            }
        }
        Ok(())
    }
}
impl Drop for TargetMachine {
    fn drop(&mut self) {
        unsafe { LLVMDisposeTargetMachine(self.raw) }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum ByteOrdering {
    BigEndian,
    LittleEndian,
}

#[derive(PartialEq, Eq, Debug)]
pub struct TargetData {
    pub raw: LLVMTargetDataRef,
}
impl TargetData {
    pub unsafe fn new(raw: LLVMTargetDataRef) -> TargetData {
        assert!(!raw.is_null());
        TargetData { raw }
    }
    pub fn as_mut_ptr(&self) -> LLVMTargetDataRef {
        self.raw
    }
    #[deprecated(note = "This method will be removed in the future. Please use Context::ptr_sized_int_type instead.")]
    pub fn ptr_sized_int_type_in_context<'ctx>(
        &self,
        c: impl AsContextRef<'ctx>,
        a: Option<AddressSpace>,
    ) -> IntType<'ctx> {
        let y = match a {
            Some(address_space) => unsafe { LLVMIntPtrTypeForASInContext(c.as_ctx_ref(), self.raw, address_space.0) },
            None => unsafe { LLVMIntPtrTypeInContext(c.as_ctx_ref(), self.raw) },
        };
        unsafe { IntType::new(y) }
    }
    pub fn get_data_layout(&self) -> DataLayout {
        unsafe { DataLayout::new_owned(LLVMCopyStringRepOfTargetData(self.raw)) }
    }
    pub fn get_bit_size(&self, type_: &dyn AnyType) -> u64 {
        unsafe { LLVMSizeOfTypeInBits(self.raw, type_.as_type_ref()) }
    }
    pub fn create(x: &str) -> TargetData {
        let y = to_c_str(x);
        unsafe { TargetData::new(LLVMCreateTargetData(y.as_ptr())) }
    }
    pub fn get_byte_ordering(&self) -> ByteOrdering {
        let y = unsafe { LLVMByteOrder(self.raw) };
        match y {
            LLVMByteOrdering::LLVMBigEndian => ByteOrdering::BigEndian,
            LLVMByteOrdering::LLVMLittleEndian => ByteOrdering::LittleEndian,
        }
    }
    pub fn get_pointer_byte_size(&self, a: Option<AddressSpace>) -> u32 {
        match a {
            Some(x) => unsafe { LLVMPointerSizeForAS(self.raw, x.0) },
            None => unsafe { LLVMPointerSize(self.raw) },
        }
    }
    pub fn get_store_size(&self, x: &dyn AnyType) -> u64 {
        unsafe { LLVMStoreSizeOfType(self.raw, x.as_type_ref()) }
    }
    pub fn get_abi_size(&self, x: &dyn AnyType) -> u64 {
        unsafe { LLVMABISizeOfType(self.raw, x.as_type_ref()) }
    }
    pub fn get_abi_alignment(&self, x: &dyn AnyType) -> u32 {
        unsafe { LLVMABIAlignmentOfType(self.raw, x.as_type_ref()) }
    }
    pub fn get_call_frame_alignment(&self, x: &dyn AnyType) -> u32 {
        unsafe { LLVMCallFrameAlignmentOfType(self.raw, x.as_type_ref()) }
    }
    pub fn get_preferred_alignment(&self, x: &dyn AnyType) -> u32 {
        unsafe { LLVMPreferredAlignmentOfType(self.raw, x.as_type_ref()) }
    }
    pub fn get_preferred_alignment_of_global(&self, x: &GlobalValue) -> u32 {
        unsafe { LLVMPreferredAlignmentOfGlobal(self.raw, x.as_value_ref()) }
    }
    pub fn element_at_offset(&self, x: &StructType, offset: u64) -> u32 {
        unsafe { LLVMElementAtOffset(self.raw, x.as_type_ref(), offset) }
    }
    pub fn offset_of_element(&self, x: &StructType, elem: u32) -> Option<u64> {
        if elem > x.count_fields() - 1 {
            return None;
        }
        unsafe { Some(LLVMOffsetOfElement(self.raw, x.as_type_ref(), elem)) }
    }
}
impl Drop for TargetData {
    fn drop(&mut self) {
        unsafe { LLVMDisposeTargetData(self.raw) }
    }
}
