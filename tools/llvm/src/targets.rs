use llvm_lib::target::{
    LLVMABIAlignmentOfType, LLVMABISizeOfType, LLVMByteOrder, LLVMByteOrdering, LLVMCallFrameAlignmentOfType,
    LLVMCopyStringRepOfTargetData, LLVMCreateTargetData, LLVMDisposeTargetData, LLVMElementAtOffset,
    LLVMIntPtrTypeForASInContext, LLVMIntPtrTypeInContext, LLVMOffsetOfElement, LLVMPointerSize, LLVMPointerSizeForAS,
    LLVMPreferredAlignmentOfGlobal, LLVMPreferredAlignmentOfType, LLVMSizeOfTypeInBits, LLVMStoreSizeOfType,
    LLVMTargetDataRef,
};
#[llvm_versions(4.0..=latest)]
use llvm_lib::target_machine::LLVMCreateTargetDataLayout;
use llvm_lib::target_machine::{
    LLVMAddAnalysisPasses, LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMCreateTargetMachine,
    LLVMDisposeTargetMachine, LLVMGetDefaultTargetTriple, LLVMGetFirstTarget, LLVMGetNextTarget,
    LLVMGetTargetDescription, LLVMGetTargetFromName, LLVMGetTargetFromTriple, LLVMGetTargetMachineCPU,
    LLVMGetTargetMachineFeatureString, LLVMGetTargetMachineTarget, LLVMGetTargetMachineTriple, LLVMGetTargetName,
    LLVMRelocMode, LLVMSetTargetMachineAsmVerbosity, LLVMTargetHasAsmBackend, LLVMTargetHasJIT,
    LLVMTargetHasTargetMachine, LLVMTargetMachineEmitToFile, LLVMTargetMachineEmitToMemoryBuffer, LLVMTargetMachineRef,
    LLVMTargetRef,
};
use once_cell::sync::Lazy;
use parking_lot::RwLock;

use crate::context::AsContextRef;
use crate::data_layout::DataLayout;
use crate::memory_buffer::MemoryBuffer;
use crate::module::Module;
use crate::passes::PassManager;
use crate::support::{to_c_str, LLVMString};
use crate::types::{AnyType, AsTypeRef, IntType, StructType};
use crate::values::{AsValueRef, GlobalValue};
use crate::{AddressSpace, OptimizationLevel};

use std::default::Default;
use std::ffi::CStr;
use std::fmt;
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr;

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
    pub(crate) triple: LLVMString,
}

impl TargetTriple {
    pub unsafe fn new(triple: LLVMString) -> TargetTriple {
        TargetTriple { triple }
    }

    pub fn create(triple: &str) -> TargetTriple {
        let c_string = to_c_str(triple);

        TargetTriple {
            triple: LLVMString::create_from_c_str(&c_string),
        }
    }

    pub fn as_str(&self) -> &CStr {
        unsafe { CStr::from_ptr(self.as_ptr()) }
    }

    pub fn as_ptr(&self) -> *const ::libc::c_char {
        self.triple.as_ptr()
    }
}

impl PartialEq for TargetTriple {
    fn eq(&self, other: &TargetTriple) -> bool {
        self.triple == other.triple
    }
}

impl fmt::Debug for TargetTriple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TargetTriple({:?})", self.triple)
    }
}

impl fmt::Display for TargetTriple {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "TargetTriple({:?})", self.triple)
    }
}

static TARGET_LOCK: Lazy<RwLock<()>> = Lazy::new(|| RwLock::new(()));

#[derive(Debug, Eq, PartialEq)]
pub struct Target {
    target: LLVMTargetRef,
}

impl Target {
    pub unsafe fn new(target: LLVMTargetRef) -> Self {
        assert!(!target.is_null());

        Target { target }
    }

    pub fn as_mut_ptr(&self) -> LLVMTargetRef {
        self.target
    }

    #[cfg(feature = "target-x86")]
    pub fn initialize_x86(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeX86AsmParser, LLVMInitializeX86AsmPrinter, LLVMInitializeX86Disassembler,
            LLVMInitializeX86Target, LLVMInitializeX86TargetInfo, LLVMInitializeX86TargetMC,
        };

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

    #[cfg(feature = "target-arm")]
    pub fn initialize_arm(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeARMAsmParser, LLVMInitializeARMAsmPrinter, LLVMInitializeARMDisassembler,
            LLVMInitializeARMTarget, LLVMInitializeARMTargetInfo, LLVMInitializeARMTargetMC,
        };

        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeARMTarget() };
        }

        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeARMTargetInfo() };
        }

        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeARMAsmPrinter() };
        }

        if config.asm_parser {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeARMAsmParser() };
        }

        if config.disassembler {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeARMDisassembler() };
        }

        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeARMTargetMC() };
        }
    }

    #[cfg(feature = "target-mips")]
    pub fn initialize_mips(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeMipsAsmParser, LLVMInitializeMipsAsmPrinter, LLVMInitializeMipsDisassembler,
            LLVMInitializeMipsTarget, LLVMInitializeMipsTargetInfo, LLVMInitializeMipsTargetMC,
        };

        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeMipsTarget() };
        }

        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeMipsTargetInfo() };
        }

        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeMipsAsmPrinter() };
        }

        if config.asm_parser {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeMipsAsmParser() };
        }

        if config.disassembler {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeMipsDisassembler() };
        }

        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeMipsTargetMC() };
        }
    }

    #[cfg(feature = "target-aarch64")]
    pub fn initialize_aarch64(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeAArch64AsmParser, LLVMInitializeAArch64AsmPrinter, LLVMInitializeAArch64Disassembler,
            LLVMInitializeAArch64Target, LLVMInitializeAArch64TargetInfo, LLVMInitializeAArch64TargetMC,
        };

        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeAArch64Target() };
        }

        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeAArch64TargetInfo() };
        }

        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeAArch64AsmPrinter() };
        }

        if config.asm_parser {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeAArch64AsmParser() };
        }

        if config.disassembler {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeAArch64Disassembler() };
        }

        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeAArch64TargetMC() };
        }
    }

    #[cfg(feature = "target-amdgpu")]
    #[llvm_versions(4.0..=latest)]
    pub fn initialize_amd_gpu(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeAMDGPUAsmParser, LLVMInitializeAMDGPUAsmPrinter, LLVMInitializeAMDGPUTarget,
            LLVMInitializeAMDGPUTargetInfo, LLVMInitializeAMDGPUTargetMC,
        };

        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeAMDGPUTarget() };
        }

        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeAMDGPUTargetInfo() };
        }

        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeAMDGPUAsmPrinter() };
        }

        if config.asm_parser {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeAMDGPUAsmParser() };
        }

        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeAMDGPUTargetMC() };
        }
    }

    #[cfg(feature = "target-systemz")]
    pub fn initialize_system_z(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeSystemZAsmParser, LLVMInitializeSystemZAsmPrinter, LLVMInitializeSystemZDisassembler,
            LLVMInitializeSystemZTarget, LLVMInitializeSystemZTargetInfo, LLVMInitializeSystemZTargetMC,
        };

        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeSystemZTarget() };
        }

        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeSystemZTargetInfo() };
        }

        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeSystemZAsmPrinter() };
        }

        if config.asm_parser {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeSystemZAsmParser() };
        }

        if config.disassembler {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeSystemZDisassembler() };
        }

        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeSystemZTargetMC() };
        }
    }

    #[cfg(feature = "target-hexagon")]
    pub fn initialize_hexagon(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeHexagonAsmPrinter, LLVMInitializeHexagonDisassembler, LLVMInitializeHexagonTarget,
            LLVMInitializeHexagonTargetInfo, LLVMInitializeHexagonTargetMC,
        };

        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeHexagonTarget() };
        }

        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeHexagonTargetInfo() };
        }

        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeHexagonAsmPrinter() };
        }

        if config.disassembler {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeHexagonDisassembler() };
        }

        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeHexagonTargetMC() };
        }
    }

    #[cfg(feature = "target-nvptx")]
    pub fn initialize_nvptx(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeNVPTXAsmPrinter, LLVMInitializeNVPTXTarget, LLVMInitializeNVPTXTargetInfo,
            LLVMInitializeNVPTXTargetMC,
        };

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

    #[cfg(feature = "target-msp430")]
    pub fn initialize_msp430(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeMSP430AsmPrinter, LLVMInitializeMSP430Target, LLVMInitializeMSP430TargetInfo,
            LLVMInitializeMSP430TargetMC,
        };

        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeMSP430Target() };
        }

        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeMSP430TargetInfo() };
        }

        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeMSP430AsmPrinter() };
        }

        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeMSP430TargetMC() };
        }
    }

    #[cfg(feature = "target-xcore")]
    pub fn initialize_x_core(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeXCoreAsmPrinter, LLVMInitializeXCoreDisassembler, LLVMInitializeXCoreTarget,
            LLVMInitializeXCoreTargetInfo, LLVMInitializeXCoreTargetMC,
        };

        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeXCoreTarget() };
        }

        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeXCoreTargetInfo() };
        }

        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeXCoreAsmPrinter() };
        }

        if config.disassembler {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeXCoreDisassembler() };
        }

        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeXCoreTargetMC() };
        }
    }

    #[cfg(feature = "target-powerpc")]
    pub fn initialize_power_pc(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializePowerPCAsmParser, LLVMInitializePowerPCAsmPrinter, LLVMInitializePowerPCDisassembler,
            LLVMInitializePowerPCTarget, LLVMInitializePowerPCTargetInfo, LLVMInitializePowerPCTargetMC,
        };

        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializePowerPCTarget() };
        }

        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializePowerPCTargetInfo() };
        }

        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializePowerPCAsmPrinter() };
        }

        if config.asm_parser {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializePowerPCAsmParser() };
        }

        if config.disassembler {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializePowerPCDisassembler() };
        }

        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializePowerPCTargetMC() };
        }
    }

    #[cfg(feature = "target-sparc")]
    pub fn initialize_sparc(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeSparcAsmParser, LLVMInitializeSparcAsmPrinter, LLVMInitializeSparcDisassembler,
            LLVMInitializeSparcTarget, LLVMInitializeSparcTargetInfo, LLVMInitializeSparcTargetMC,
        };

        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeSparcTarget() };
        }

        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeSparcTargetInfo() };
        }

        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeSparcAsmPrinter() };
        }

        if config.asm_parser {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeSparcAsmParser() };
        }

        if config.disassembler {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeSparcDisassembler() };
        }

        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeSparcTargetMC() };
        }
    }

    #[cfg(feature = "target-bpf")]
    pub fn initialize_bpf(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeBPFAsmPrinter, LLVMInitializeBPFTarget, LLVMInitializeBPFTargetInfo,
            LLVMInitializeBPFTargetMC,
        };

        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeBPFTarget() };
        }

        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeBPFTargetInfo() };
        }

        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeBPFAsmPrinter() };
        }

        if config.disassembler {
            use llvm_lib::target::LLVMInitializeBPFDisassembler;

            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeBPFDisassembler() };
        }

        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeBPFTargetMC() };
        }
    }

    #[cfg(feature = "target-lanai")]
    #[llvm_versions(4.0..=latest)]
    pub fn initialize_lanai(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeLanaiAsmParser, LLVMInitializeLanaiAsmPrinter, LLVMInitializeLanaiDisassembler,
            LLVMInitializeLanaiTarget, LLVMInitializeLanaiTargetInfo, LLVMInitializeLanaiTargetMC,
        };

        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeLanaiTarget() };
        }

        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeLanaiTargetInfo() };
        }

        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeLanaiAsmPrinter() };
        }

        if config.asm_parser {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeLanaiAsmParser() };
        }

        if config.disassembler {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeLanaiDisassembler() };
        }

        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeLanaiTargetMC() };
        }
    }

    #[cfg(feature = "target-riscv")]
    #[llvm_versions(9.0..=latest)]
    pub fn initialize_riscv(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeRISCVAsmParser, LLVMInitializeRISCVAsmPrinter, LLVMInitializeRISCVDisassembler,
            LLVMInitializeRISCVTarget, LLVMInitializeRISCVTargetInfo, LLVMInitializeRISCVTargetMC,
        };

        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeRISCVTarget() };
        }

        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeRISCVTargetInfo() };
        }

        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeRISCVAsmPrinter() };
        }

        if config.asm_parser {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeRISCVAsmParser() };
        }

        if config.disassembler {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeRISCVDisassembler() };
        }

        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeRISCVTargetMC() };
        }
    }

    #[cfg(feature = "target-loongarch")]
    #[llvm_versions(16.0..=latest)]
    pub fn initialize_loongarch(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeLoongArchAsmParser, LLVMInitializeLoongArchAsmPrinter, LLVMInitializeLoongArchDisassembler,
            LLVMInitializeLoongArchTarget, LLVMInitializeLoongArchTargetInfo, LLVMInitializeLoongArchTargetMC,
        };

        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeLoongArchTarget() };
        }

        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeLoongArchTargetInfo() };
        }

        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeLoongArchAsmPrinter() };
        }

        if config.asm_parser {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeLoongArchAsmParser() };
        }

        if config.disassembler {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeLoongArchDisassembler() };
        }

        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeLoongArchTargetMC() };
        }
    }

    #[cfg(feature = "target-webassembly")]
    #[llvm_versions(8.0..=latest)]
    pub fn initialize_webassembly(config: &InitializationConfig) {
        use llvm_lib::target::{
            LLVMInitializeWebAssemblyAsmParser, LLVMInitializeWebAssemblyAsmPrinter,
            LLVMInitializeWebAssemblyDisassembler, LLVMInitializeWebAssemblyTarget,
            LLVMInitializeWebAssemblyTargetInfo, LLVMInitializeWebAssemblyTargetMC,
        };

        if config.base {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeWebAssemblyTarget() };
        }

        if config.info {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeWebAssemblyTargetInfo() };
        }

        if config.asm_printer {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeWebAssemblyAsmPrinter() };
        }

        if config.asm_parser {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeWebAssemblyAsmParser() };
        }

        if config.disassembler {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeWebAssemblyDisassembler() };
        }

        if config.machine_code {
            let _guard = TARGET_LOCK.write();
            unsafe { LLVMInitializeWebAssemblyTargetMC() };
        }
    }

    pub fn initialize_native(config: &InitializationConfig) -> Result<(), String> {
        use llvm_lib::target::{
            LLVM_InitializeNativeAsmParser, LLVM_InitializeNativeAsmPrinter, LLVM_InitializeNativeDisassembler,
            LLVM_InitializeNativeTarget,
        };

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
        use llvm_lib::target::{
            LLVM_InitializeAllAsmParsers, LLVM_InitializeAllAsmPrinters, LLVM_InitializeAllDisassemblers,
            LLVM_InitializeAllTargetInfos, LLVM_InitializeAllTargetMCs, LLVM_InitializeAllTargets,
        };

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
                self.target,
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
        let target = {
            let _guard = TARGET_LOCK.read();
            unsafe { LLVMGetFirstTarget() }
        };

        if target.is_null() {
            return None;
        }

        unsafe { Some(Target::new(target)) }
    }

    pub fn get_next(&self) -> Option<Self> {
        let target = unsafe { LLVMGetNextTarget(self.target) };

        if target.is_null() {
            return None;
        }

        unsafe { Some(Target::new(target)) }
    }

    pub fn get_name(&self) -> &CStr {
        unsafe { CStr::from_ptr(LLVMGetTargetName(self.target)) }
    }

    pub fn get_description(&self) -> &CStr {
        unsafe { CStr::from_ptr(LLVMGetTargetDescription(self.target)) }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        let c_string = to_c_str(name);

        Self::from_name_raw(c_string.as_ptr())
    }

    pub(crate) fn from_name_raw(c_string: *const ::libc::c_char) -> Option<Self> {
        let target = {
            let _guard = TARGET_LOCK.read();
            unsafe { LLVMGetTargetFromName(c_string) }
        };

        if target.is_null() {
            return None;
        }

        unsafe { Some(Target::new(target)) }
    }

    pub fn from_triple(triple: &TargetTriple) -> Result<Self, LLVMString> {
        let mut target = ptr::null_mut();
        let mut err_string = MaybeUninit::uninit();

        let code = {
            let _guard = TARGET_LOCK.read();
            unsafe { LLVMGetTargetFromTriple(triple.as_ptr(), &mut target, err_string.as_mut_ptr()) }
        };

        if code == 1 {
            unsafe {
                return Err(LLVMString::new(err_string.assume_init()));
            }
        }

        unsafe { Ok(Target::new(target)) }
    }

    pub fn has_jit(&self) -> bool {
        unsafe { LLVMTargetHasJIT(self.target) == 1 }
    }

    pub fn has_target_machine(&self) -> bool {
        unsafe { LLVMTargetHasTargetMachine(self.target) == 1 }
    }

    pub fn has_asm_backend(&self) -> bool {
        unsafe { LLVMTargetHasAsmBackend(self.target) == 1 }
    }
}

#[derive(Debug)]
pub struct TargetMachine {
    pub(crate) target_machine: LLVMTargetMachineRef,
}

impl TargetMachine {
    pub unsafe fn new(target_machine: LLVMTargetMachineRef) -> Self {
        assert!(!target_machine.is_null());

        TargetMachine { target_machine }
    }

    pub fn as_mut_ptr(&self) -> LLVMTargetMachineRef {
        self.target_machine
    }

    pub fn get_target(&self) -> Target {
        unsafe { Target::new(LLVMGetTargetMachineTarget(self.target_machine)) }
    }

    pub fn get_triple(&self) -> TargetTriple {
        let str = unsafe { LLVMString::new(LLVMGetTargetMachineTriple(self.target_machine)) };

        unsafe { TargetTriple::new(str) }
    }

    pub fn get_default_triple() -> TargetTriple {
        let llvm_string = unsafe { LLVMString::new(LLVMGetDefaultTargetTriple()) };

        unsafe { TargetTriple::new(llvm_string) }
    }

    #[llvm_versions(7.0..=latest)]
    pub fn normalize_triple(triple: &TargetTriple) -> TargetTriple {
        use llvm_lib::target_machine::LLVMNormalizeTargetTriple;

        let normalized = unsafe { LLVMString::new(LLVMNormalizeTargetTriple(triple.as_ptr())) };

        unsafe { TargetTriple::new(normalized) }
    }

    #[llvm_versions(7.0..=latest)]
    pub fn get_host_cpu_name() -> LLVMString {
        use llvm_lib::target_machine::LLVMGetHostCPUName;

        unsafe { LLVMString::new(LLVMGetHostCPUName()) }
    }

    #[llvm_versions(7.0..=latest)]
    pub fn get_host_cpu_features() -> LLVMString {
        use llvm_lib::target_machine::LLVMGetHostCPUFeatures;

        unsafe { LLVMString::new(LLVMGetHostCPUFeatures()) }
    }

    pub fn get_cpu(&self) -> LLVMString {
        unsafe { LLVMString::new(LLVMGetTargetMachineCPU(self.target_machine)) }
    }

    pub fn get_feature_string(&self) -> &CStr {
        unsafe { CStr::from_ptr(LLVMGetTargetMachineFeatureString(self.target_machine)) }
    }

    #[llvm_versions(4.0..=latest)]
    pub fn get_target_data(&self) -> TargetData {
        unsafe { TargetData::new(LLVMCreateTargetDataLayout(self.target_machine)) }
    }

    pub fn set_asm_verbosity(&self, verbosity: bool) {
        unsafe { LLVMSetTargetMachineAsmVerbosity(self.target_machine, verbosity as i32) }
    }

    pub fn add_analysis_passes<T>(&self, pass_manager: &PassManager<T>) {
        unsafe { LLVMAddAnalysisPasses(self.target_machine, pass_manager.pass_manager) }
    }

    pub fn write_to_memory_buffer(&self, module: &Module, file_type: FileType) -> Result<MemoryBuffer, LLVMString> {
        let mut memory_buffer = ptr::null_mut();
        let mut err_string = MaybeUninit::uninit();
        let return_code = unsafe {
            let module_ptr = module.module.get();
            let file_type_ptr = file_type.as_llvm_file_type();

            LLVMTargetMachineEmitToMemoryBuffer(
                self.target_machine,
                module_ptr,
                file_type_ptr,
                err_string.as_mut_ptr(),
                &mut memory_buffer,
            )
        };

        if return_code == 1 {
            unsafe {
                return Err(LLVMString::new(err_string.assume_init()));
            }
        }

        unsafe { Ok(MemoryBuffer::new(memory_buffer)) }
    }

    pub fn write_to_file(&self, module: &Module, file_type: FileType, path: &Path) -> Result<(), LLVMString> {
        let path = path.to_str().expect("Did not find a valid Unicode path string");
        let path_c_string = to_c_str(path);
        let mut err_string = MaybeUninit::uninit();
        let return_code = unsafe {
            let module_ptr = module.module.get();
            let path_ptr = path_c_string.as_ptr() as *mut _;
            let file_type_ptr = file_type.as_llvm_file_type();

            LLVMTargetMachineEmitToFile(
                self.target_machine,
                module_ptr,
                path_ptr,
                file_type_ptr,
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
}

impl Drop for TargetMachine {
    fn drop(&mut self) {
        unsafe { LLVMDisposeTargetMachine(self.target_machine) }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum ByteOrdering {
    BigEndian,
    LittleEndian,
}

#[derive(PartialEq, Eq, Debug)]
pub struct TargetData {
    pub(crate) target_data: LLVMTargetDataRef,
}

impl TargetData {
    pub unsafe fn new(target_data: LLVMTargetDataRef) -> TargetData {
        assert!(!target_data.is_null());

        TargetData { target_data }
    }

    pub fn as_mut_ptr(&self) -> LLVMTargetDataRef {
        self.target_data
    }

    #[deprecated(note = "This method will be removed in the future. Please use Context::ptr_sized_int_type instead.")]
    pub fn ptr_sized_int_type_in_context<'ctx>(
        &self,
        context: impl AsContextRef<'ctx>,
        address_space: Option<AddressSpace>,
    ) -> IntType<'ctx> {
        let int_type_ptr = match address_space {
            Some(address_space) => unsafe {
                LLVMIntPtrTypeForASInContext(context.as_ctx_ref(), self.target_data, address_space.0)
            },
            None => unsafe { LLVMIntPtrTypeInContext(context.as_ctx_ref(), self.target_data) },
        };

        unsafe { IntType::new(int_type_ptr) }
    }

    pub fn get_data_layout(&self) -> DataLayout {
        unsafe { DataLayout::new_owned(LLVMCopyStringRepOfTargetData(self.target_data)) }
    }

    pub fn get_bit_size(&self, type_: &dyn AnyType) -> u64 {
        unsafe { LLVMSizeOfTypeInBits(self.target_data, type_.as_type_ref()) }
    }

    pub fn create(str_repr: &str) -> TargetData {
        let c_string = to_c_str(str_repr);

        unsafe { TargetData::new(LLVMCreateTargetData(c_string.as_ptr())) }
    }

    pub fn get_byte_ordering(&self) -> ByteOrdering {
        let byte_ordering = unsafe { LLVMByteOrder(self.target_data) };

        match byte_ordering {
            LLVMByteOrdering::LLVMBigEndian => ByteOrdering::BigEndian,
            LLVMByteOrdering::LLVMLittleEndian => ByteOrdering::LittleEndian,
        }
    }

    pub fn get_pointer_byte_size(&self, address_space: Option<AddressSpace>) -> u32 {
        match address_space {
            Some(address_space) => unsafe { LLVMPointerSizeForAS(self.target_data, address_space.0) },
            None => unsafe { LLVMPointerSize(self.target_data) },
        }
    }

    pub fn get_store_size(&self, type_: &dyn AnyType) -> u64 {
        unsafe { LLVMStoreSizeOfType(self.target_data, type_.as_type_ref()) }
    }

    pub fn get_abi_size(&self, type_: &dyn AnyType) -> u64 {
        unsafe { LLVMABISizeOfType(self.target_data, type_.as_type_ref()) }
    }

    pub fn get_abi_alignment(&self, type_: &dyn AnyType) -> u32 {
        unsafe { LLVMABIAlignmentOfType(self.target_data, type_.as_type_ref()) }
    }

    pub fn get_call_frame_alignment(&self, type_: &dyn AnyType) -> u32 {
        unsafe { LLVMCallFrameAlignmentOfType(self.target_data, type_.as_type_ref()) }
    }

    pub fn get_preferred_alignment(&self, type_: &dyn AnyType) -> u32 {
        unsafe { LLVMPreferredAlignmentOfType(self.target_data, type_.as_type_ref()) }
    }

    pub fn get_preferred_alignment_of_global(&self, value: &GlobalValue) -> u32 {
        unsafe { LLVMPreferredAlignmentOfGlobal(self.target_data, value.as_value_ref()) }
    }

    pub fn element_at_offset(&self, struct_type: &StructType, offset: u64) -> u32 {
        unsafe { LLVMElementAtOffset(self.target_data, struct_type.as_type_ref(), offset) }
    }

    pub fn offset_of_element(&self, struct_type: &StructType, element: u32) -> Option<u64> {
        if element > struct_type.count_fields() - 1 {
            return None;
        }

        unsafe {
            Some(LLVMOffsetOfElement(
                self.target_data,
                struct_type.as_type_ref(),
                element,
            ))
        }
    }
}

impl Drop for TargetData {
    fn drop(&mut self) {
        unsafe { LLVMDisposeTargetData(self.target_data) }
    }
}
