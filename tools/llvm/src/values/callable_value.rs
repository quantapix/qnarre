use either::Either;
use std::convert::TryFrom;
use std::fmt::{self, Display};

use crate::types::AsTypeRef;
use crate::values::AsValueRef;
use crate::values::{AnyValue, FunctionValue, PointerValue};

use llvm_lib::core::{LLVMGetElementType, LLVMGetTypeKind, LLVMTypeOf};
use llvm_lib::prelude::LLVMTypeRef;
use llvm_lib::prelude::LLVMValueRef;
use llvm_lib::LLVMTypeKind;

///
///
///
///
///
///
///
///
///
///
///
///
///
///
///
#[derive(Debug)]
pub struct CallableValue<'ctx>(Either<FunctionValue<'ctx>, PointerValue<'ctx>>);

unsafe impl<'ctx> AsValueRef for CallableValue<'ctx> {
    fn as_value_ref(&self) -> LLVMValueRef {
        use either::Either::*;

        match self.0 {
            Left(function) => function.as_value_ref(),
            Right(pointer) => pointer.as_value_ref(),
        }
    }
}

unsafe impl<'ctx> AnyValue<'ctx> for CallableValue<'ctx> {}

unsafe impl<'ctx> AsTypeRef for CallableValue<'ctx> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        use either::Either::*;

        match self.0 {
            Left(function) => function.get_type().as_type_ref(),
            Right(pointer) => pointer.get_type().get_element_type().as_type_ref(),
        }
    }
}

impl<'ctx> CallableValue<'ctx> {
    #[llvm_versions(4.0..=14.0)]
    pub(crate) fn returns_void(&self) -> bool {
        use llvm_lib::core::LLVMGetReturnType;

        let return_type =
            unsafe { LLVMGetTypeKind(LLVMGetReturnType(LLVMGetElementType(LLVMTypeOf(self.as_value_ref())))) };

        matches!(return_type, LLVMTypeKind::LLVMVoidTypeKind)
    }
}

impl<'ctx> From<FunctionValue<'ctx>> for CallableValue<'ctx> {
    fn from(value: FunctionValue<'ctx>) -> Self {
        Self(Either::Left(value))
    }
}

impl<'ctx> TryFrom<PointerValue<'ctx>> for CallableValue<'ctx> {
    type Error = ();

    fn try_from(value: PointerValue<'ctx>) -> Result<Self, Self::Error> {
        let value_ref = value.as_value_ref();
        let ty_kind = unsafe { LLVMGetTypeKind(LLVMGetElementType(LLVMTypeOf(value_ref))) };
        let is_a_fn_ptr = matches!(ty_kind, LLVMTypeKind::LLVMFunctionTypeKind);

        if is_a_fn_ptr {
            Ok(Self(Either::Right(value)))
        } else {
            Err(())
        }
    }
}

impl Display for CallableValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
