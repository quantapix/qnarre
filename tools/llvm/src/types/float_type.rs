use llvm_lib::core::{LLVMConstArray, LLVMConstReal, LLVMConstRealOfStringAndSize};
use llvm_lib::execution_engine::LLVMCreateGenericValueOfFloat;
use llvm_lib::prelude::{LLVMTypeRef, LLVMValueRef};

use crate::context::ContextRef;
use crate::support::LLVMString;
use crate::types::enums::BasicMetadataTypeEnum;
use crate::types::traits::AsTypeRef;
use crate::types::{ArrayType, FunctionType, PointerType, Type, VectorType};
use crate::values::{ArrayValue, AsValueRef, FloatValue, GenericValue, IntValue};
use crate::AddressSpace;

use std::fmt::{self, Display};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct FloatType<'ctx> {
    float_type: Type<'ctx>,
}

impl<'ctx> FloatType<'ctx> {
    pub unsafe fn new(float_type: LLVMTypeRef) -> Self {
        assert!(!float_type.is_null());

        FloatType {
            float_type: Type::new(float_type),
        }
    }

    pub fn fn_type(self, param_types: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.float_type.fn_type(param_types, is_var_args)
    }

    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.float_type.array_type(size)
    }

    pub fn vec_type(self, size: u32) -> VectorType<'ctx> {
        self.float_type.vec_type(size)
    }

    pub fn const_float(self, value: f64) -> FloatValue<'ctx> {
        unsafe { FloatValue::new(LLVMConstReal(self.float_type.ty, value)) }
    }

    pub fn const_float_from_string(self, slice: &str) -> FloatValue<'ctx> {
        unsafe {
            FloatValue::new(LLVMConstRealOfStringAndSize(
                self.as_type_ref(),
                slice.as_ptr() as *const ::libc::c_char,
                slice.len() as u32,
            ))
        }
    }

    pub fn const_zero(self) -> FloatValue<'ctx> {
        unsafe { FloatValue::new(self.float_type.const_zero()) }
    }

    pub fn size_of(self) -> IntValue<'ctx> {
        self.float_type.size_of().unwrap()
    }

    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.float_type.get_alignment()
    }

    pub fn get_context(self) -> ContextRef<'ctx> {
        self.float_type.get_context()
    }

    pub fn ptr_type(self, address_space: AddressSpace) -> PointerType<'ctx> {
        self.float_type.ptr_type(address_space)
    }

    pub fn print_to_string(self) -> LLVMString {
        self.float_type.print_to_string()
    }

    pub fn get_undef(&self) -> FloatValue<'ctx> {
        unsafe { FloatValue::new(self.float_type.get_undef()) }
    }

    pub fn create_generic_value(self, value: f64) -> GenericValue<'ctx> {
        unsafe { GenericValue::new(LLVMCreateGenericValueOfFloat(self.as_type_ref(), value)) }
    }

    pub fn const_array(self, values: &[FloatValue<'ctx>]) -> ArrayValue<'ctx> {
        let mut values: Vec<LLVMValueRef> = values.iter().map(|val| val.as_value_ref()).collect();
        unsafe {
            ArrayValue::new(LLVMConstArray(
                self.as_type_ref(),
                values.as_mut_ptr(),
                values.len() as u32,
            ))
        }
    }
}

unsafe impl AsTypeRef for FloatType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.float_type.ty
    }
}

impl Display for FloatType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
