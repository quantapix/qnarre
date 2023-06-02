use llvm_lib::core::{
    LLVMCountParamTypes, LLVMGetParamTypes, LLVMGetReturnType, LLVMGetTypeKind, LLVMIsFunctionVarArg,
};
use llvm_lib::prelude::LLVMTypeRef;
use llvm_lib::LLVMTypeKind;

use std::fmt::{self, Display};
use std::mem::forget;

use crate::context::ContextRef;
use crate::support::LLVMString;
use crate::types::traits::AsTypeRef;
use crate::types::{AnyType, BasicTypeEnum, PointerType, Type};
use crate::AddressSpace;

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct FunctionType<'ctx> {
    fn_type: Type<'ctx>,
}

impl<'ctx> FunctionType<'ctx> {
    pub unsafe fn new(fn_type: LLVMTypeRef) -> Self {
        assert!(!fn_type.is_null());

        FunctionType {
            fn_type: Type::new(fn_type),
        }
    }

    pub fn ptr_type(self, address_space: AddressSpace) -> PointerType<'ctx> {
        self.fn_type.ptr_type(address_space)
    }

    pub fn is_var_arg(self) -> bool {
        unsafe { LLVMIsFunctionVarArg(self.as_type_ref()) != 0 }
    }

    pub fn get_param_types(self) -> Vec<BasicTypeEnum<'ctx>> {
        let count = self.count_param_types();
        let mut raw_vec: Vec<LLVMTypeRef> = Vec::with_capacity(count as usize);
        let ptr = raw_vec.as_mut_ptr();

        forget(raw_vec);

        let raw_vec = unsafe {
            LLVMGetParamTypes(self.as_type_ref(), ptr);

            Vec::from_raw_parts(ptr, count as usize, count as usize)
        };

        raw_vec.iter().map(|val| unsafe { BasicTypeEnum::new(*val) }).collect()
    }

    pub fn count_param_types(self) -> u32 {
        unsafe { LLVMCountParamTypes(self.as_type_ref()) }
    }

    pub fn is_sized(self) -> bool {
        self.fn_type.is_sized()
    }

    pub fn get_context(self) -> ContextRef<'ctx> {
        self.fn_type.get_context()
    }

    pub fn print_to_string(self) -> LLVMString {
        self.fn_type.print_to_string()
    }

    pub fn get_return_type(self) -> Option<BasicTypeEnum<'ctx>> {
        let ty = unsafe { LLVMGetReturnType(self.as_type_ref()) };

        let kind = unsafe { LLVMGetTypeKind(ty) };

        if let LLVMTypeKind::LLVMVoidTypeKind = kind {
            return None;
        }

        unsafe { Some(BasicTypeEnum::new(ty)) }
    }
}

impl fmt::Debug for FunctionType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let llvm_type = self.print_to_string();

        f.debug_struct("FunctionType")
            .field("address", &self.as_type_ref())
            .field("is_var_args", &self.is_var_arg())
            .field("llvm_type", &llvm_type)
            .finish()
    }
}

unsafe impl AsTypeRef for FunctionType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.fn_type.ty
    }
}

impl Display for FunctionType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
