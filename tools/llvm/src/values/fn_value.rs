use llvm_lib::analysis::{LLVMVerifierFailureAction, LLVMVerifyFunction, LLVMViewFunctionCFG, LLVMViewFunctionCFGOnly};
use llvm_lib::core::{
    LLVMAddAttributeAtIndex, LLVMGetAttributeCountAtIndex, LLVMGetEnumAttributeAtIndex, LLVMGetStringAttributeAtIndex,
    LLVMRemoveEnumAttributeAtIndex, LLVMRemoveStringAttributeAtIndex,
};
use llvm_lib::core::{
    LLVMCountBasicBlocks, LLVMCountParams, LLVMDeleteFunction, LLVMGetBasicBlocks, LLVMGetFirstBasicBlock,
    LLVMGetFirstParam, LLVMGetFunctionCallConv, LLVMGetGC, LLVMGetIntrinsicID, LLVMGetLastBasicBlock, LLVMGetLastParam,
    LLVMGetLinkage, LLVMGetNextFunction, LLVMGetNextParam, LLVMGetParam, LLVMGetParams, LLVMGetPreviousFunction,
    LLVMIsAFunction, LLVMIsConstant, LLVMSetFunctionCallConv, LLVMSetGC, LLVMSetLinkage, LLVMSetParamAlignment,
};
use llvm_lib::core::{LLVMGetPersonalityFn, LLVMSetPersonalityFn};
#[llvm_versions(7.0..=latest)]
use llvm_lib::debuginfo::{LLVMGetSubprogram, LLVMSetSubprogram};
use llvm_lib::prelude::{LLVMBasicBlockRef, LLVMValueRef};

use std::ffi::CStr;
use std::fmt::{self, Display};
use std::marker::PhantomData;
use std::mem::forget;

use crate::attributes::{Attribute, AttributeLoc};
use crate::basic_block::BasicBlock;
#[llvm_versions(7.0..=latest)]
use crate::debug_info::DISubprogram;
use crate::module::Linkage;
use crate::support::to_c_str;
use crate::types::FunctionType;
use crate::values::traits::{AnyValue, AsValueRef};
use crate::values::{BasicValueEnum, GlobalValue, Value};

#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub struct FunctionValue<'ctx> {
    fn_value: Value<'ctx>,
}

impl<'ctx> FunctionValue<'ctx> {
    pub(crate) unsafe fn new(value: LLVMValueRef) -> Option<Self> {
        if value.is_null() {
            return None;
        }

        assert!(!LLVMIsAFunction(value).is_null());

        Some(FunctionValue {
            fn_value: Value::new(value),
        })
    }

    pub fn get_linkage(self) -> Linkage {
        unsafe { LLVMGetLinkage(self.as_value_ref()).into() }
    }

    pub fn set_linkage(self, linkage: Linkage) {
        unsafe { LLVMSetLinkage(self.as_value_ref(), linkage.into()) }
    }

    pub fn is_null(self) -> bool {
        self.fn_value.is_null()
    }

    pub fn is_undef(self) -> bool {
        self.fn_value.is_undef()
    }

    pub fn print_to_stderr(self) {
        self.fn_value.print_to_stderr()
    }

    pub fn verify(self, print: bool) -> bool {
        let action = if print {
            LLVMVerifierFailureAction::LLVMPrintMessageAction
        } else {
            LLVMVerifierFailureAction::LLVMReturnStatusAction
        };

        let code = unsafe { LLVMVerifyFunction(self.fn_value.value, action) };

        code != 1
    }

    pub fn get_next_function(self) -> Option<Self> {
        unsafe { FunctionValue::new(LLVMGetNextFunction(self.as_value_ref())) }
    }

    pub fn get_previous_function(self) -> Option<Self> {
        unsafe { FunctionValue::new(LLVMGetPreviousFunction(self.as_value_ref())) }
    }

    pub fn get_first_param(self) -> Option<BasicValueEnum<'ctx>> {
        let param = unsafe { LLVMGetFirstParam(self.as_value_ref()) };

        if param.is_null() {
            return None;
        }

        unsafe { Some(BasicValueEnum::new(param)) }
    }

    pub fn get_last_param(self) -> Option<BasicValueEnum<'ctx>> {
        let param = unsafe { LLVMGetLastParam(self.as_value_ref()) };

        if param.is_null() {
            return None;
        }

        unsafe { Some(BasicValueEnum::new(param)) }
    }

    pub fn get_first_basic_block(self) -> Option<BasicBlock<'ctx>> {
        unsafe { BasicBlock::new(LLVMGetFirstBasicBlock(self.as_value_ref())) }
    }

    pub fn get_nth_param(self, nth: u32) -> Option<BasicValueEnum<'ctx>> {
        let count = self.count_params();

        if nth + 1 > count {
            return None;
        }

        unsafe { Some(BasicValueEnum::new(LLVMGetParam(self.as_value_ref(), nth))) }
    }

    pub fn count_params(self) -> u32 {
        unsafe { LLVMCountParams(self.fn_value.value) }
    }

    pub fn count_basic_blocks(self) -> u32 {
        unsafe { LLVMCountBasicBlocks(self.as_value_ref()) }
    }

    pub fn get_basic_blocks(self) -> Vec<BasicBlock<'ctx>> {
        let count = self.count_basic_blocks();
        let mut raw_vec: Vec<LLVMBasicBlockRef> = Vec::with_capacity(count as usize);
        let ptr = raw_vec.as_mut_ptr();

        forget(raw_vec);

        let raw_vec = unsafe {
            LLVMGetBasicBlocks(self.as_value_ref(), ptr);

            Vec::from_raw_parts(ptr, count as usize, count as usize)
        };

        raw_vec
            .iter()
            .map(|val| unsafe { BasicBlock::new(*val).unwrap() })
            .collect()
    }

    pub fn get_param_iter(self) -> ParamValueIter<'ctx> {
        ParamValueIter {
            param_iter_value: self.fn_value.value,
            start: true,
            _marker: PhantomData,
        }
    }

    pub fn get_params(self) -> Vec<BasicValueEnum<'ctx>> {
        let count = self.count_params();
        let mut raw_vec: Vec<LLVMValueRef> = Vec::with_capacity(count as usize);
        let ptr = raw_vec.as_mut_ptr();

        forget(raw_vec);

        let raw_vec = unsafe {
            LLVMGetParams(self.as_value_ref(), ptr);

            Vec::from_raw_parts(ptr, count as usize, count as usize)
        };

        raw_vec.iter().map(|val| unsafe { BasicValueEnum::new(*val) }).collect()
    }

    pub fn get_last_basic_block(self) -> Option<BasicBlock<'ctx>> {
        unsafe { BasicBlock::new(LLVMGetLastBasicBlock(self.fn_value.value)) }
    }

    pub fn get_name(&self) -> &CStr {
        self.fn_value.get_name()
    }

    pub fn view_function_cfg(self) {
        unsafe { LLVMViewFunctionCFG(self.as_value_ref()) }
    }

    pub fn view_function_cfg_only(self) {
        unsafe { LLVMViewFunctionCFGOnly(self.as_value_ref()) }
    }

    pub unsafe fn delete(self) {
        LLVMDeleteFunction(self.as_value_ref())
    }

    #[llvm_versions(4.0..=7.0)]
    pub fn get_type(self) -> FunctionType<'ctx> {
        use crate::types::PointerType;

        let ptr_type = unsafe { PointerType::new(self.fn_value.get_type()) };

        ptr_type.get_element_type().into_function_type()
    }

    #[llvm_versions(8.0..=latest)]
    pub fn get_type(self) -> FunctionType<'ctx> {
        unsafe { FunctionType::new(llvm_lib::core::LLVMGlobalGetValueType(self.as_value_ref())) }
    }

    pub fn has_personality_function(self) -> bool {
        use llvm_lib::core::LLVMHasPersonalityFn;

        unsafe { LLVMHasPersonalityFn(self.as_value_ref()) == 1 }
    }

    pub fn get_personality_function(self) -> Option<FunctionValue<'ctx>> {
        if !self.has_personality_function() {
            return None;
        }

        unsafe { FunctionValue::new(LLVMGetPersonalityFn(self.as_value_ref())) }
    }

    pub fn set_personality_function(self, personality_fn: FunctionValue<'ctx>) {
        unsafe { LLVMSetPersonalityFn(self.as_value_ref(), personality_fn.as_value_ref()) }
    }

    pub fn get_intrinsic_id(self) -> u32 {
        unsafe { LLVMGetIntrinsicID(self.as_value_ref()) }
    }

    pub fn get_call_conventions(self) -> u32 {
        unsafe { LLVMGetFunctionCallConv(self.as_value_ref()) }
    }

    pub fn set_call_conventions(self, call_conventions: u32) {
        unsafe { LLVMSetFunctionCallConv(self.as_value_ref(), call_conventions) }
    }

    pub fn get_gc(&self) -> &CStr {
        unsafe { CStr::from_ptr(LLVMGetGC(self.as_value_ref())) }
    }

    pub fn set_gc(self, gc: &str) {
        let c_string = to_c_str(gc);

        unsafe { LLVMSetGC(self.as_value_ref(), c_string.as_ptr()) }
    }

    pub fn replace_all_uses_with(self, other: FunctionValue<'ctx>) {
        self.fn_value.replace_all_uses_with(other.as_value_ref())
    }

    pub fn add_attribute(self, loc: AttributeLoc, attribute: Attribute) {
        unsafe { LLVMAddAttributeAtIndex(self.as_value_ref(), loc.get_index(), attribute.attribute) }
    }

    pub fn count_attributes(self, loc: AttributeLoc) -> u32 {
        unsafe { LLVMGetAttributeCountAtIndex(self.as_value_ref(), loc.get_index()) }
    }

    pub fn attributes(self, loc: AttributeLoc) -> Vec<Attribute> {
        use llvm_lib::core::LLVMGetAttributesAtIndex;
        use std::mem::{ManuallyDrop, MaybeUninit};

        let count = self.count_attributes(loc) as usize;

        let mut attribute_refs: Vec<MaybeUninit<Attribute>> = vec![MaybeUninit::uninit(); count];

        unsafe {
            LLVMGetAttributesAtIndex(
                self.as_value_ref(),
                loc.get_index(),
                attribute_refs.as_mut_ptr() as *mut _,
            )
        }

        unsafe {
            let mut attribute_refs = ManuallyDrop::new(attribute_refs);

            Vec::from_raw_parts(
                attribute_refs.as_mut_ptr() as *mut Attribute,
                attribute_refs.len(),
                attribute_refs.capacity(),
            )
        }
    }

    pub fn remove_string_attribute(self, loc: AttributeLoc, key: &str) {
        unsafe {
            LLVMRemoveStringAttributeAtIndex(
                self.as_value_ref(),
                loc.get_index(),
                key.as_ptr() as *const ::libc::c_char,
                key.len() as u32,
            )
        }
    }

    pub fn remove_enum_attribute(self, loc: AttributeLoc, kind_id: u32) {
        unsafe { LLVMRemoveEnumAttributeAtIndex(self.as_value_ref(), loc.get_index(), kind_id) }
    }

    pub fn get_enum_attribute(self, loc: AttributeLoc, kind_id: u32) -> Option<Attribute> {
        let ptr = unsafe { LLVMGetEnumAttributeAtIndex(self.as_value_ref(), loc.get_index(), kind_id) };

        if ptr.is_null() {
            return None;
        }

        unsafe { Some(Attribute::new(ptr)) }
    }

    pub fn get_string_attribute(self, loc: AttributeLoc, key: &str) -> Option<Attribute> {
        let ptr = unsafe {
            LLVMGetStringAttributeAtIndex(
                self.as_value_ref(),
                loc.get_index(),
                key.as_ptr() as *const ::libc::c_char,
                key.len() as u32,
            )
        };

        if ptr.is_null() {
            return None;
        }

        unsafe { Some(Attribute::new(ptr)) }
    }

    pub fn set_param_alignment(self, param_index: u32, alignment: u32) {
        if let Some(param) = self.get_nth_param(param_index) {
            unsafe { LLVMSetParamAlignment(param.as_value_ref(), alignment) }
        }
    }

    pub fn as_global_value(self) -> GlobalValue<'ctx> {
        unsafe { GlobalValue::new(self.as_value_ref()) }
    }

    #[llvm_versions(7.0..=latest)]
    pub fn set_subprogram(self, subprogram: DISubprogram<'ctx>) {
        unsafe { LLVMSetSubprogram(self.as_value_ref(), subprogram.metadata_ref) }
    }

    #[llvm_versions(7.0..=latest)]
    pub fn get_subprogram(self) -> Option<DISubprogram<'ctx>> {
        let metadata_ref = unsafe { LLVMGetSubprogram(self.as_value_ref()) };

        if metadata_ref.is_null() {
            None
        } else {
            Some(DISubprogram {
                metadata_ref,
                _marker: PhantomData,
            })
        }
    }

    pub fn get_section(&self) -> Option<&CStr> {
        self.fn_value.get_section()
    }

    pub fn set_section(self, section: Option<&str>) {
        self.fn_value.set_section(section)
    }
}

unsafe impl AsValueRef for FunctionValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.fn_value.value
    }
}

impl Display for FunctionValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

impl fmt::Debug for FunctionValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let llvm_value = self.print_to_string();
        let llvm_type = self.get_type();
        let name = self.get_name();
        let is_const = unsafe { LLVMIsConstant(self.fn_value.value) == 1 };
        let is_null = self.is_null();

        f.debug_struct("FunctionValue")
            .field("name", &name)
            .field("address", &self.as_value_ref())
            .field("is_const", &is_const)
            .field("is_null", &is_null)
            .field("llvm_value", &llvm_value)
            .field("llvm_type", &llvm_type.print_to_string())
            .finish()
    }
}

#[derive(Debug)]
pub struct ParamValueIter<'ctx> {
    param_iter_value: LLVMValueRef,
    start: bool,
    _marker: PhantomData<&'ctx ()>,
}

impl<'ctx> Iterator for ParamValueIter<'ctx> {
    type Item = BasicValueEnum<'ctx>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start {
            let first_value = unsafe { LLVMGetFirstParam(self.param_iter_value) };

            if first_value.is_null() {
                return None;
            }

            self.start = false;

            self.param_iter_value = first_value;

            return unsafe { Some(Self::Item::new(first_value)) };
        }

        let next_value = unsafe { LLVMGetNextParam(self.param_iter_value) };

        if next_value.is_null() {
            return None;
        }

        self.param_iter_value = next_value;

        unsafe { Some(Self::Item::new(next_value)) }
    }
}
