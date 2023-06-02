use llvm_lib::core::{LLVMConstArray, LLVMConstVector, LLVMGetVectorSize};
use llvm_lib::prelude::{LLVMTypeRef, LLVMValueRef};

use crate::context::ContextRef;
use crate::support::LLVMString;
use crate::types::enums::BasicMetadataTypeEnum;
use crate::types::{traits::AsTypeRef, ArrayType, BasicTypeEnum, FunctionType, PointerType, Type};
use crate::values::{ArrayValue, AsValueRef, BasicValue, IntValue, VectorValue};
use crate::AddressSpace;

use std::fmt::{self, Display};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct VectorType<'ctx> {
    vec_type: Type<'ctx>,
}

impl<'ctx> VectorType<'ctx> {
    pub unsafe fn new(vector_type: LLVMTypeRef) -> Self {
        assert!(!vector_type.is_null());

        VectorType {
            vec_type: Type::new(vector_type),
        }
    }

    pub fn size_of(self) -> Option<IntValue<'ctx>> {
        self.vec_type.size_of()
    }

    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.vec_type.get_alignment()
    }

    pub fn get_size(self) -> u32 {
        unsafe { LLVMGetVectorSize(self.as_type_ref()) }
    }

    pub fn const_vector<V: BasicValue<'ctx>>(values: &[V]) -> VectorValue<'ctx> {
        let mut values: Vec<LLVMValueRef> = values.iter().map(|val| val.as_value_ref()).collect();
        unsafe { VectorValue::new(LLVMConstVector(values.as_mut_ptr(), values.len() as u32)) }
    }

    pub fn const_zero(self) -> VectorValue<'ctx> {
        unsafe { VectorValue::new(self.vec_type.const_zero()) }
    }

    pub fn print_to_string(self) -> LLVMString {
        self.vec_type.print_to_string()
    }

    pub fn get_undef(self) -> VectorValue<'ctx> {
        unsafe { VectorValue::new(self.vec_type.get_undef()) }
    }

    pub fn get_element_type(self) -> BasicTypeEnum<'ctx> {
        self.vec_type.get_element_type().to_basic_type_enum()
    }

    pub fn ptr_type(self, address_space: AddressSpace) -> PointerType<'ctx> {
        self.vec_type.ptr_type(address_space)
    }

    pub fn fn_type(self, param_types: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.vec_type.fn_type(param_types, is_var_args)
    }

    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.vec_type.array_type(size)
    }

    pub fn const_array(self, values: &[VectorValue<'ctx>]) -> ArrayValue<'ctx> {
        let mut values: Vec<LLVMValueRef> = values.iter().map(|val| val.as_value_ref()).collect();

        unsafe {
            ArrayValue::new(LLVMConstArray(
                self.as_type_ref(),
                values.as_mut_ptr(),
                values.len() as u32,
            ))
        }
    }

    pub fn get_context(self) -> ContextRef<'ctx> {
        self.vec_type.get_context()
    }
}

unsafe impl AsTypeRef for VectorType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.vec_type.ty
    }
}

impl Display for VectorType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
