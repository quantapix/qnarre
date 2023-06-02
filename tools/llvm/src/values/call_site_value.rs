use std::fmt::{self, Display};

use either::Either;
use llvm_lib::core::{
    LLVMGetInstructionCallConv, LLVMGetTypeKind, LLVMIsTailCall, LLVMSetInstrParamAlignment,
    LLVMSetInstructionCallConv, LLVMSetTailCall, LLVMTypeOf,
};
use llvm_lib::prelude::LLVMValueRef;
use llvm_lib::LLVMTypeKind;

use crate::attributes::{Attribute, AttributeLoc};
use crate::values::{AsValueRef, BasicValueEnum, FunctionValue, InstructionValue, Value};

use super::AnyValue;

///
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct CallSiteValue<'ctx>(Value<'ctx>);

impl<'ctx> CallSiteValue<'ctx> {
    pub(crate) unsafe fn new(value: LLVMValueRef) -> Self {
        CallSiteValue(Value::new(value))
    }

    pub fn set_tail_call(self, tail_call: bool) {
        unsafe { LLVMSetTailCall(self.as_value_ref(), tail_call as i32) }
    }

    pub fn is_tail_call(self) -> bool {
        unsafe { LLVMIsTailCall(self.as_value_ref()) == 1 }
    }

    pub fn try_as_basic_value(self) -> Either<BasicValueEnum<'ctx>, InstructionValue<'ctx>> {
        unsafe {
            match LLVMGetTypeKind(LLVMTypeOf(self.as_value_ref())) {
                LLVMTypeKind::LLVMVoidTypeKind => Either::Right(InstructionValue::new(self.as_value_ref())),
                _ => Either::Left(BasicValueEnum::new(self.as_value_ref())),
            }
        }
    }

    pub fn add_attribute(self, loc: AttributeLoc, attribute: Attribute) {
        use llvm_lib::core::LLVMAddCallSiteAttribute;

        unsafe { LLVMAddCallSiteAttribute(self.as_value_ref(), loc.get_index(), attribute.attribute) }
    }

    pub fn get_called_fn_value(self) -> FunctionValue<'ctx> {
        use llvm_lib::core::LLVMGetCalledValue;

        unsafe { FunctionValue::new(LLVMGetCalledValue(self.as_value_ref())).expect("This should never be null?") }
    }

    pub fn count_attributes(self, loc: AttributeLoc) -> u32 {
        use llvm_lib::core::LLVMGetCallSiteAttributeCount;

        unsafe { LLVMGetCallSiteAttributeCount(self.as_value_ref(), loc.get_index()) }
    }

    pub fn attributes(self, loc: AttributeLoc) -> Vec<Attribute> {
        use llvm_lib::core::LLVMGetCallSiteAttributes;
        use std::mem::{ManuallyDrop, MaybeUninit};

        let count = self.count_attributes(loc) as usize;

        let mut attribute_refs: Vec<MaybeUninit<Attribute>> = vec![MaybeUninit::uninit(); count];

        unsafe {
            LLVMGetCallSiteAttributes(
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

    pub fn get_enum_attribute(self, loc: AttributeLoc, kind_id: u32) -> Option<Attribute> {
        use llvm_lib::core::LLVMGetCallSiteEnumAttribute;

        let ptr = unsafe { LLVMGetCallSiteEnumAttribute(self.as_value_ref(), loc.get_index(), kind_id) };

        if ptr.is_null() {
            return None;
        }

        unsafe { Some(Attribute::new(ptr)) }
    }

    pub fn get_string_attribute(self, loc: AttributeLoc, key: &str) -> Option<Attribute> {
        use llvm_lib::core::LLVMGetCallSiteStringAttribute;

        let ptr = unsafe {
            LLVMGetCallSiteStringAttribute(
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

    pub fn remove_enum_attribute(self, loc: AttributeLoc, kind_id: u32) {
        use llvm_lib::core::LLVMRemoveCallSiteEnumAttribute;

        unsafe { LLVMRemoveCallSiteEnumAttribute(self.as_value_ref(), loc.get_index(), kind_id) }
    }

    pub fn remove_string_attribute(self, loc: AttributeLoc, key: &str) {
        use llvm_lib::core::LLVMRemoveCallSiteStringAttribute;

        unsafe {
            LLVMRemoveCallSiteStringAttribute(
                self.as_value_ref(),
                loc.get_index(),
                key.as_ptr() as *const ::libc::c_char,
                key.len() as u32,
            )
        }
    }

    pub fn count_arguments(self) -> u32 {
        use llvm_lib::core::LLVMGetNumArgOperands;

        unsafe { LLVMGetNumArgOperands(self.as_value_ref()) }
    }

    pub fn get_call_convention(self) -> u32 {
        unsafe { LLVMGetInstructionCallConv(self.as_value_ref()) }
    }

    pub fn set_call_convention(self, conv: u32) {
        unsafe { LLVMSetInstructionCallConv(self.as_value_ref(), conv) }
    }

    pub fn set_alignment_attribute(self, loc: AttributeLoc, alignment: u32) {
        assert_eq!(alignment.count_ones(), 1, "Alignment must be a power of two.");

        unsafe { LLVMSetInstrParamAlignment(self.as_value_ref(), loc.get_index(), alignment) }
    }
}

unsafe impl AsValueRef for CallSiteValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.0.value
    }
}

impl Display for CallSiteValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
