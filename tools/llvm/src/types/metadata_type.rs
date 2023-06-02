use llvm_lib::prelude::LLVMTypeRef;

use crate::context::ContextRef;
use crate::support::LLVMString;
use crate::types::enums::BasicMetadataTypeEnum;
use crate::types::traits::AsTypeRef;
use crate::types::{FunctionType, Type};

use std::fmt::{self, Display};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct MetadataType<'ctx> {
    metadata_type: Type<'ctx>,
}

impl<'ctx> MetadataType<'ctx> {
    #[llvm_versions(6.0..=latest)]
    pub unsafe fn new(metadata_type: LLVMTypeRef) -> Self {
        assert!(!metadata_type.is_null());

        MetadataType {
            metadata_type: Type::new(metadata_type),
        }
    }

    #[llvm_versions(6.0..=latest)]
    pub fn fn_type(self, param_types: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.metadata_type.fn_type(param_types, is_var_args)
    }

    #[llvm_versions(6.0..=latest)]
    pub fn get_context(self) -> ContextRef<'ctx> {
        self.metadata_type.get_context()
    }

    pub fn print_to_string(self) -> LLVMString {
        self.metadata_type.print_to_string()
    }
}

unsafe impl AsTypeRef for MetadataType<'_> {
    #[llvm_versions(6.0..=latest)]
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.metadata_type.ty
    }

    #[llvm_versions(4.0..=5.0)]
    fn as_type_ref(&self) -> LLVMTypeRef {
        unimplemented!("MetadataType is only available in LLVM > 6.0")
    }
}

impl Display for MetadataType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
