use libc::c_void;
use llvm_lib::execution_engine::{
    LLVMCreateGenericValueOfPointer, LLVMDisposeGenericValue, LLVMGenericValueIntWidth, LLVMGenericValueRef,
    LLVMGenericValueToFloat, LLVMGenericValueToInt, LLVMGenericValueToPointer,
};

use crate::types::{AsTypeRef, FloatType};

use std::marker::PhantomData;

#[derive(Debug)]
pub struct GenericValue<'ctx> {
    pub(crate) generic_value: LLVMGenericValueRef,
    _phantom: PhantomData<&'ctx ()>,
}

impl<'ctx> GenericValue<'ctx> {
    pub(crate) unsafe fn new(generic_value: LLVMGenericValueRef) -> Self {
        assert!(!generic_value.is_null());

        GenericValue {
            generic_value,
            _phantom: PhantomData,
        }
    }

    pub fn int_width(self) -> u32 {
        unsafe { LLVMGenericValueIntWidth(self.generic_value) }
    }

    pub unsafe fn create_generic_value_of_pointer<T>(value: &mut T) -> Self {
        let value = LLVMCreateGenericValueOfPointer(value as *mut _ as *mut c_void);

        GenericValue::new(value)
    }

    pub fn as_int(self, is_signed: bool) -> u64 {
        unsafe { LLVMGenericValueToInt(self.generic_value, is_signed as i32) }
    }

    pub fn as_float(self, float_type: &FloatType<'ctx>) -> f64 {
        unsafe { LLVMGenericValueToFloat(float_type.as_type_ref(), self.generic_value) }
    }

    pub unsafe fn into_pointer<T>(self) -> *mut T {
        LLVMGenericValueToPointer(self.generic_value) as *mut T
    }
}

impl Drop for GenericValue<'_> {
    fn drop(&mut self) {
        unsafe { LLVMDisposeGenericValue(self.generic_value) }
    }
}
