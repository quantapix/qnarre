use llvm_lib::prelude::LLVMTypeRef;

use crate::context::ContextRef;
use crate::support::LLVMString;
use crate::types::enums::BasicMetadataTypeEnum;
use crate::types::traits::AsTypeRef;
use crate::types::{FunctionType, Type};

use std::fmt::{self, Display};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct VoidType<'ctx> {
    void_type: Type<'ctx>,
}

impl<'ctx> VoidType<'ctx> {
    pub unsafe fn new(void_type: LLVMTypeRef) -> Self {
        assert!(!void_type.is_null());

        VoidType {
            void_type: Type::new(void_type),
        }
    }

    pub fn is_sized(self) -> bool {
        self.void_type.is_sized()
    }

    pub fn get_context(self) -> ContextRef<'ctx> {
        self.void_type.get_context()
    }

    pub fn fn_type(self, param_types: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.void_type.fn_type(param_types, is_var_args)
    }

    pub fn print_to_string(self) -> LLVMString {
        self.void_type.print_to_string()
    }
}

unsafe impl AsTypeRef for VoidType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.void_type.ty
    }
}

impl Display for VoidType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
