#[llvm_versions(8.0..=latest)]
use llvm_lib::core::LLVMGlobalSetMetadata;
#[llvm_versions(4.0..=7.0)]
use llvm_lib::core::{
    LLVMDeleteGlobal, LLVMGetAlignment, LLVMGetDLLStorageClass, LLVMGetInitializer, LLVMGetLinkage, LLVMGetNextGlobal,
    LLVMGetPreviousGlobal, LLVMGetSection, LLVMGetThreadLocalMode, LLVMGetVisibility, LLVMIsDeclaration,
    LLVMIsExternallyInitialized, LLVMIsGlobalConstant, LLVMIsThreadLocal, LLVMSetAlignment, LLVMSetDLLStorageClass,
    LLVMSetExternallyInitialized, LLVMSetGlobalConstant, LLVMSetInitializer, LLVMSetLinkage, LLVMSetSection,
    LLVMSetThreadLocal, LLVMSetThreadLocalMode, LLVMSetVisibility,
};
#[llvm_versions(8.0..=latest)]
use llvm_lib::core::{
    LLVMDeleteGlobal, LLVMGetAlignment, LLVMGetDLLStorageClass, LLVMGetInitializer, LLVMGetLinkage, LLVMGetNextGlobal,
    LLVMGetPreviousGlobal, LLVMGetThreadLocalMode, LLVMGetVisibility, LLVMIsDeclaration, LLVMIsExternallyInitialized,
    LLVMIsGlobalConstant, LLVMIsThreadLocal, LLVMSetAlignment, LLVMSetDLLStorageClass, LLVMSetExternallyInitialized,
    LLVMSetGlobalConstant, LLVMSetInitializer, LLVMSetLinkage, LLVMSetThreadLocal, LLVMSetThreadLocalMode,
    LLVMSetVisibility,
};
#[llvm_versions(7.0..=latest)]
use llvm_lib::core::{LLVMGetUnnamedAddress, LLVMSetUnnamedAddress};
#[llvm_versions(4.0..=6.0)]
use llvm_lib::core::{LLVMHasUnnamedAddr, LLVMSetUnnamedAddr};
use llvm_lib::prelude::LLVMValueRef;
use llvm_lib::LLVMThreadLocalMode;
#[llvm_versions(7.0..=latest)]
use llvm_lib::LLVMUnnamedAddr;

use std::ffi::CStr;
use std::fmt::{self, Display};

#[llvm_versions(7.0..=latest)]
use crate::comdat::Comdat;
use crate::module::Linkage;
use crate::values::traits::AsValueRef;
#[llvm_versions(8.0..=latest)]
use crate::values::MetadataValue;
use crate::values::{BasicValue, BasicValueEnum, PointerValue, Value};
use crate::{DLLStorageClass, GlobalVisibility, ThreadLocalMode};

use super::AnyValue;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct GlobalValue<'ctx> {
    global_value: Value<'ctx>,
}

impl<'ctx> GlobalValue<'ctx> {
    pub(crate) unsafe fn new(value: LLVMValueRef) -> Self {
        assert!(!value.is_null());

        GlobalValue {
            global_value: Value::new(value),
        }
    }

    pub fn get_name(&self) -> &CStr {
        self.global_value.get_name()
    }

    pub fn set_name(&self, name: &str) {
        self.global_value.set_name(name)
    }

    pub fn get_previous_global(self) -> Option<GlobalValue<'ctx>> {
        let value = unsafe { LLVMGetPreviousGlobal(self.as_value_ref()) };

        if value.is_null() {
            return None;
        }

        unsafe { Some(GlobalValue::new(value)) }
    }

    pub fn get_next_global(self) -> Option<GlobalValue<'ctx>> {
        let value = unsafe { LLVMGetNextGlobal(self.as_value_ref()) };

        if value.is_null() {
            return None;
        }

        unsafe { Some(GlobalValue::new(value)) }
    }

    pub fn get_dll_storage_class(self) -> DLLStorageClass {
        let dll_storage_class = unsafe { LLVMGetDLLStorageClass(self.as_value_ref()) };

        DLLStorageClass::new(dll_storage_class)
    }

    pub fn set_dll_storage_class(self, dll_storage_class: DLLStorageClass) {
        unsafe { LLVMSetDLLStorageClass(self.as_value_ref(), dll_storage_class.into()) }
    }

    pub fn get_initializer(self) -> Option<BasicValueEnum<'ctx>> {
        let value = unsafe { LLVMGetInitializer(self.as_value_ref()) };

        if value.is_null() {
            return None;
        }

        unsafe { Some(BasicValueEnum::new(value)) }
    }

    pub fn set_initializer(self, value: &dyn BasicValue<'ctx>) {
        unsafe { LLVMSetInitializer(self.as_value_ref(), value.as_value_ref()) }
    }

    pub fn is_thread_local(self) -> bool {
        unsafe { LLVMIsThreadLocal(self.as_value_ref()) == 1 }
    }

    pub fn set_thread_local(self, is_thread_local: bool) {
        unsafe { LLVMSetThreadLocal(self.as_value_ref(), is_thread_local as i32) }
    }

    pub fn get_thread_local_mode(self) -> Option<ThreadLocalMode> {
        let thread_local_mode = unsafe { LLVMGetThreadLocalMode(self.as_value_ref()) };

        ThreadLocalMode::new(thread_local_mode)
    }

    pub fn set_thread_local_mode(self, thread_local_mode: Option<ThreadLocalMode>) {
        let thread_local_mode = match thread_local_mode {
            Some(mode) => mode.as_llvm_mode(),
            None => LLVMThreadLocalMode::LLVMNotThreadLocal,
        };

        unsafe { LLVMSetThreadLocalMode(self.as_value_ref(), thread_local_mode) }
    }

    pub fn is_declaration(self) -> bool {
        unsafe { LLVMIsDeclaration(self.as_value_ref()) == 1 }
    }

    #[llvm_versions(4.0..=6.0)]
    pub fn has_unnamed_addr(self) -> bool {
        unsafe { LLVMHasUnnamedAddr(self.as_value_ref()) == 1 }
    }

    #[llvm_versions(7.0..=latest)]
    pub fn has_unnamed_addr(self) -> bool {
        unsafe { LLVMGetUnnamedAddress(self.as_value_ref()) == LLVMUnnamedAddr::LLVMGlobalUnnamedAddr }
    }

    #[llvm_versions(4.0..=6.0)]
    pub fn set_unnamed_addr(self, has_unnamed_addr: bool) {
        unsafe { LLVMSetUnnamedAddr(self.as_value_ref(), has_unnamed_addr as i32) }
    }

    #[llvm_versions(7.0..=latest)]
    pub fn set_unnamed_addr(self, has_unnamed_addr: bool) {
        unsafe {
            if has_unnamed_addr {
                LLVMSetUnnamedAddress(self.as_value_ref(), UnnamedAddress::Global.into())
            } else {
                LLVMSetUnnamedAddress(self.as_value_ref(), UnnamedAddress::None.into())
            }
        }
    }

    pub fn is_constant(self) -> bool {
        unsafe { LLVMIsGlobalConstant(self.as_value_ref()) == 1 }
    }

    pub fn set_constant(self, is_constant: bool) {
        unsafe { LLVMSetGlobalConstant(self.as_value_ref(), is_constant as i32) }
    }

    pub fn is_externally_initialized(self) -> bool {
        unsafe { LLVMIsExternallyInitialized(self.as_value_ref()) == 1 }
    }

    pub fn set_externally_initialized(self, externally_initialized: bool) {
        unsafe { LLVMSetExternallyInitialized(self.as_value_ref(), externally_initialized as i32) }
    }

    pub fn set_visibility(self, visibility: GlobalVisibility) {
        unsafe { LLVMSetVisibility(self.as_value_ref(), visibility.into()) }
    }

    pub fn get_visibility(self) -> GlobalVisibility {
        let visibility = unsafe { LLVMGetVisibility(self.as_value_ref()) };

        GlobalVisibility::new(visibility)
    }

    pub fn get_section(&self) -> Option<&CStr> {
        self.global_value.get_section()
    }

    pub fn set_section(self, section: Option<&str>) {
        self.global_value.set_section(section)
    }

    pub unsafe fn delete(self) {
        LLVMDeleteGlobal(self.as_value_ref())
    }

    pub fn as_pointer_value(self) -> PointerValue<'ctx> {
        unsafe { PointerValue::new(self.as_value_ref()) }
    }

    pub fn get_alignment(self) -> u32 {
        unsafe { LLVMGetAlignment(self.as_value_ref()) }
    }

    pub fn set_alignment(self, alignment: u32) {
        unsafe { LLVMSetAlignment(self.as_value_ref(), alignment) }
    }

    #[llvm_versions(8.0..=latest)]
    pub fn set_metadata(self, metadata: MetadataValue<'ctx>, kind_id: u32) {
        unsafe { LLVMGlobalSetMetadata(self.as_value_ref(), kind_id, metadata.as_metadata_ref()) }
    }

    #[llvm_versions(7.0..=latest)]
    pub fn get_comdat(self) -> Option<Comdat> {
        use llvm_lib::comdat::LLVMGetComdat;

        let comdat_ptr = unsafe { LLVMGetComdat(self.as_value_ref()) };

        if comdat_ptr.is_null() {
            return None;
        }

        unsafe { Some(Comdat::new(comdat_ptr)) }
    }

    #[llvm_versions(7.0..=latest)]
    pub fn set_comdat(self, comdat: Comdat) {
        use llvm_lib::comdat::LLVMSetComdat;

        unsafe { LLVMSetComdat(self.as_value_ref(), comdat.0) }
    }

    #[llvm_versions(7.0..=latest)]
    pub fn get_unnamed_address(self) -> UnnamedAddress {
        use llvm_lib::core::LLVMGetUnnamedAddress;

        let unnamed_address = unsafe { LLVMGetUnnamedAddress(self.as_value_ref()) };

        UnnamedAddress::new(unnamed_address)
    }

    #[llvm_versions(7.0..=latest)]
    pub fn set_unnamed_address(self, address: UnnamedAddress) {
        use llvm_lib::core::LLVMSetUnnamedAddress;

        unsafe { LLVMSetUnnamedAddress(self.as_value_ref(), address.into()) }
    }

    pub fn get_linkage(self) -> Linkage {
        unsafe { LLVMGetLinkage(self.as_value_ref()).into() }
    }

    pub fn set_linkage(self, linkage: Linkage) {
        unsafe { LLVMSetLinkage(self.as_value_ref(), linkage.into()) }
    }
}

unsafe impl AsValueRef for GlobalValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.global_value.value
    }
}

impl Display for GlobalValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

#[llvm_versions(7.0..=latest)]
#[llvm_enum(LLVMUnnamedAddr)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum UnnamedAddress {
    #[llvm_variant(LLVMNoUnnamedAddr)]
    None,

    #[llvm_variant(LLVMLocalUnnamedAddr)]
    Local,

    #[llvm_variant(LLVMGlobalUnnamedAddr)]
    Global,
}
