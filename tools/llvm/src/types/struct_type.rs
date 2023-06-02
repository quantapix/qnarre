use llvm_lib::core::{
    LLVMConstArray, LLVMConstNamedStruct, LLVMCountStructElementTypes, LLVMGetStructElementTypes, LLVMGetStructName,
    LLVMIsOpaqueStruct, LLVMIsPackedStruct, LLVMStructGetTypeAtIndex, LLVMStructSetBody,
};
use llvm_lib::prelude::{LLVMTypeRef, LLVMValueRef};

use std::ffi::CStr;
use std::fmt::{self, Display};
use std::mem::forget;

use crate::context::ContextRef;
use crate::support::LLVMString;
use crate::types::enums::BasicMetadataTypeEnum;
use crate::types::traits::AsTypeRef;
use crate::types::{ArrayType, BasicTypeEnum, FunctionType, PointerType, Type};
use crate::values::{ArrayValue, AsValueRef, BasicValueEnum, IntValue, StructValue};
use crate::AddressSpace;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct StructType<'ctx> {
    struct_type: Type<'ctx>,
}

impl<'ctx> StructType<'ctx> {
    pub unsafe fn new(struct_type: LLVMTypeRef) -> Self {
        assert!(!struct_type.is_null());

        StructType {
            struct_type: Type::new(struct_type),
        }
    }

    pub fn get_field_type_at_index(self, index: u32) -> Option<BasicTypeEnum<'ctx>> {
        if self.is_opaque() {
            return None;
        }

        if index >= self.count_fields() {
            return None;
        }

        unsafe { Some(BasicTypeEnum::new(LLVMStructGetTypeAtIndex(self.as_type_ref(), index))) }
    }

    pub fn const_named_struct(self, values: &[BasicValueEnum<'ctx>]) -> StructValue<'ctx> {
        let mut args: Vec<LLVMValueRef> = values.iter().map(|val| val.as_value_ref()).collect();
        unsafe {
            StructValue::new(LLVMConstNamedStruct(
                self.as_type_ref(),
                args.as_mut_ptr(),
                args.len() as u32,
            ))
        }
    }

    pub fn const_zero(self) -> StructValue<'ctx> {
        unsafe { StructValue::new(self.struct_type.const_zero()) }
    }

    pub fn size_of(self) -> Option<IntValue<'ctx>> {
        self.struct_type.size_of()
    }

    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.struct_type.get_alignment()
    }

    pub fn get_context(self) -> ContextRef<'ctx> {
        self.struct_type.get_context()
    }

    pub fn get_name(&self) -> Option<&CStr> {
        let name = unsafe { LLVMGetStructName(self.as_type_ref()) };

        if name.is_null() {
            return None;
        }

        let c_str = unsafe { CStr::from_ptr(name) };

        Some(c_str)
    }

    pub fn ptr_type(self, address_space: AddressSpace) -> PointerType<'ctx> {
        self.struct_type.ptr_type(address_space)
    }

    pub fn fn_type(self, param_types: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.struct_type.fn_type(param_types, is_var_args)
    }

    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.struct_type.array_type(size)
    }

    pub fn is_packed(self) -> bool {
        unsafe { LLVMIsPackedStruct(self.as_type_ref()) == 1 }
    }

    pub fn is_opaque(self) -> bool {
        unsafe { LLVMIsOpaqueStruct(self.as_type_ref()) == 1 }
    }

    pub fn count_fields(self) -> u32 {
        unsafe { LLVMCountStructElementTypes(self.as_type_ref()) }
    }

    pub fn get_field_types(self) -> Vec<BasicTypeEnum<'ctx>> {
        let count = self.count_fields();
        let mut raw_vec: Vec<LLVMTypeRef> = Vec::with_capacity(count as usize);
        let ptr = raw_vec.as_mut_ptr();

        forget(raw_vec);

        let raw_vec = unsafe {
            LLVMGetStructElementTypes(self.as_type_ref(), ptr);

            Vec::from_raw_parts(ptr, count as usize, count as usize)
        };

        raw_vec.iter().map(|val| unsafe { BasicTypeEnum::new(*val) }).collect()
    }

    pub fn print_to_string(self) -> LLVMString {
        self.struct_type.print_to_string()
    }

    pub fn get_undef(self) -> StructValue<'ctx> {
        unsafe { StructValue::new(self.struct_type.get_undef()) }
    }

    pub fn set_body(self, field_types: &[BasicTypeEnum<'ctx>], packed: bool) -> bool {
        let is_opaque = self.is_opaque();
        let mut field_types: Vec<LLVMTypeRef> = field_types.iter().map(|val| val.as_type_ref()).collect();
        unsafe {
            LLVMStructSetBody(
                self.as_type_ref(),
                field_types.as_mut_ptr(),
                field_types.len() as u32,
                packed as i32,
            );
        }

        is_opaque
    }

    pub fn const_array(self, values: &[StructValue<'ctx>]) -> ArrayValue<'ctx> {
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

unsafe impl AsTypeRef for StructType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.struct_type.ty
    }
}

impl Display for StructType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
