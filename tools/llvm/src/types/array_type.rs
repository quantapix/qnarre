use llvm_lib::core::{LLVMConstArray, LLVMGetArrayLength};
use llvm_lib::prelude::{LLVMTypeRef, LLVMValueRef};

use crate::context::ContextRef;
use crate::support::LLVMString;
use crate::types::enums::BasicMetadataTypeEnum;
use crate::types::traits::AsTypeRef;
use crate::types::{BasicTypeEnum, FunctionType, PointerType, Type};
use crate::values::{ArrayValue, AsValueRef, IntValue};
use crate::AddressSpace;

use std::fmt::{self, Display};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct ArrayType<'ctx> {
    array_type: Type<'ctx>,
}

impl<'ctx> ArrayType<'ctx> {
    pub unsafe fn new(array_type: LLVMTypeRef) -> Self {
        assert!(!array_type.is_null());

        ArrayType {
            array_type: Type::new(array_type),
        }
    }

    pub fn size_of(self) -> Option<IntValue<'ctx>> {
        self.array_type.size_of()
    }

    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.array_type.get_alignment()
    }

    pub fn ptr_type(self, address_space: AddressSpace) -> PointerType<'ctx> {
        self.array_type.ptr_type(address_space)
    }

    pub fn get_context(self) -> ContextRef<'ctx> {
        self.array_type.get_context()
    }

    pub fn fn_type(self, param_types: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.array_type.fn_type(param_types, is_var_args)
    }

    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.array_type.array_type(size)
    }

    pub fn const_array(self, values: &[ArrayValue<'ctx>]) -> ArrayValue<'ctx> {
        let mut values: Vec<LLVMValueRef> = values.iter().map(|val| val.as_value_ref()).collect();
        unsafe {
            ArrayValue::new(LLVMConstArray(
                self.as_type_ref(),
                values.as_mut_ptr(),
                values.len() as u32,
            ))
        }
    }

    pub fn const_zero(self) -> ArrayValue<'ctx> {
        unsafe { ArrayValue::new(self.array_type.const_zero()) }
    }

    pub fn len(self) -> u32 {
        unsafe { LLVMGetArrayLength(self.as_type_ref()) }
    }

    pub fn print_to_string(self) -> LLVMString {
        self.array_type.print_to_string()
    }

    pub fn get_undef(self) -> ArrayValue<'ctx> {
        unsafe { ArrayValue::new(self.array_type.get_undef()) }
    }

    pub fn get_element_type(self) -> BasicTypeEnum<'ctx> {
        self.array_type.get_element_type().to_basic_type_enum()
    }
}

unsafe impl AsTypeRef for ArrayType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.array_type.ty
    }
}

impl Display for ArrayType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
