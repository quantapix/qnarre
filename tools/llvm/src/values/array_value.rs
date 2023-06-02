use llvm_lib::core::{LLVMGetAsString, LLVMIsAConstantArray, LLVMIsAConstantDataArray, LLVMIsConstantString};
use llvm_lib::prelude::LLVMValueRef;

use std::ffi::CStr;
use std::fmt::{self, Display};

use crate::types::ArrayType;
use crate::values::traits::{AnyValue, AsValueRef};
use crate::values::{InstructionValue, Value};

#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub struct ArrayValue<'ctx> {
    array_value: Value<'ctx>,
}

impl<'ctx> ArrayValue<'ctx> {
    pub(crate) unsafe fn new(value: LLVMValueRef) -> Self {
        assert!(!value.is_null());

        ArrayValue {
            array_value: Value::new(value),
        }
    }

    pub fn get_name(&self) -> &CStr {
        self.array_value.get_name()
    }

    pub fn set_name(&self, name: &str) {
        self.array_value.set_name(name)
    }

    pub fn get_type(self) -> ArrayType<'ctx> {
        unsafe { ArrayType::new(self.array_value.get_type()) }
    }

    pub fn is_null(self) -> bool {
        self.array_value.is_null()
    }

    pub fn is_undef(self) -> bool {
        self.array_value.is_undef()
    }

    pub fn print_to_stderr(self) {
        self.array_value.print_to_stderr()
    }

    pub fn as_instruction(self) -> Option<InstructionValue<'ctx>> {
        self.array_value.as_instruction()
    }

    pub fn replace_all_uses_with(self, other: ArrayValue<'ctx>) {
        self.array_value.replace_all_uses_with(other.as_value_ref())
    }

    pub fn is_const(self) -> bool {
        self.array_value.is_const()
    }

    pub fn is_const_string(self) -> bool {
        unsafe { LLVMIsConstantString(self.as_value_ref()) == 1 }
    }

    pub fn get_string_constant(&self) -> Option<&CStr> {
        let mut len = 0;
        let ptr = unsafe { LLVMGetAsString(self.as_value_ref(), &mut len) };

        if ptr.is_null() {
            None
        } else {
            unsafe { Some(CStr::from_ptr(ptr)) }
        }
    }
}

unsafe impl AsValueRef for ArrayValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.array_value.value
    }
}

impl Display for ArrayValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

impl fmt::Debug for ArrayValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let llvm_value = self.print_to_string();
        let llvm_type = self.get_type();
        let name = self.get_name();
        let is_const = self.is_const();
        let is_null = self.is_null();
        let is_const_array = unsafe { !LLVMIsAConstantArray(self.as_value_ref()).is_null() };
        let is_const_data_array = unsafe { !LLVMIsAConstantDataArray(self.as_value_ref()).is_null() };

        f.debug_struct("ArrayValue")
            .field("name", &name)
            .field("address", &self.as_value_ref())
            .field("is_const", &is_const)
            .field("is_const_array", &is_const_array)
            .field("is_const_data_array", &is_const_data_array)
            .field("is_null", &is_null)
            .field("llvm_value", &llvm_value)
            .field("llvm_type", &llvm_type)
            .finish()
    }
}
