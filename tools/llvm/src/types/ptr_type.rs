use llvm_lib::core::{LLVMConstArray, LLVMGetPointerAddressSpace};
use llvm_lib::prelude::{LLVMTypeRef, LLVMValueRef};

use crate::context::ContextRef;
use crate::support::LLVMString;
use crate::types::traits::AsTypeRef;
#[llvm_versions(4.0..=14.0)]
use crate::types::AnyTypeEnum;
use crate::types::{ArrayType, FunctionType, Type, VectorType};
use crate::values::{ArrayValue, AsValueRef, IntValue, PointerValue};
use crate::AddressSpace;

use crate::types::enums::BasicMetadataTypeEnum;
use std::fmt::{self, Display};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct PointerType<'ctx> {
    ptr_type: Type<'ctx>,
}

impl<'ctx> PointerType<'ctx> {
    pub unsafe fn new(ptr_type: LLVMTypeRef) -> Self {
        assert!(!ptr_type.is_null());

        PointerType {
            ptr_type: Type::new(ptr_type),
        }
    }

    pub fn size_of(self) -> IntValue<'ctx> {
        self.ptr_type.size_of().unwrap()
    }

    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.ptr_type.get_alignment()
    }

    pub fn ptr_type(self, address_space: AddressSpace) -> PointerType<'ctx> {
        self.ptr_type.ptr_type(address_space)
    }

    pub fn get_context(self) -> ContextRef<'ctx> {
        self.ptr_type.get_context()
    }

    pub fn fn_type(self, param_types: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.ptr_type.fn_type(param_types, is_var_args)
    }

    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.ptr_type.array_type(size)
    }

    pub fn get_address_space(self) -> AddressSpace {
        let addr_space = unsafe { LLVMGetPointerAddressSpace(self.as_type_ref()) };

        AddressSpace(addr_space)
    }

    pub fn print_to_string(self) -> LLVMString {
        self.ptr_type.print_to_string()
    }

    pub fn const_null(self) -> PointerValue<'ctx> {
        unsafe { PointerValue::new(self.ptr_type.const_zero()) }
    }

    pub fn const_zero(self) -> PointerValue<'ctx> {
        unsafe { PointerValue::new(self.ptr_type.const_zero()) }
    }

    pub fn get_undef(self) -> PointerValue<'ctx> {
        unsafe { PointerValue::new(self.ptr_type.get_undef()) }
    }

    pub fn vec_type(self, size: u32) -> VectorType<'ctx> {
        self.ptr_type.vec_type(size)
    }

    #[llvm_versions(4.0..=14.0)]
    pub fn get_element_type(self) -> AnyTypeEnum<'ctx> {
        self.ptr_type.get_element_type()
    }

    pub fn const_array(self, values: &[PointerValue<'ctx>]) -> ArrayValue<'ctx> {
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

unsafe impl AsTypeRef for PointerType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.ptr_type.ty
    }
}

impl Display for PointerType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
