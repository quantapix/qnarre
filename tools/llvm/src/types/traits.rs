use llvm_lib::prelude::LLVMTypeRef;

use std::fmt::Debug;

use crate::support::LLVMString;
use crate::types::enums::{AnyTypeEnum, BasicMetadataTypeEnum, BasicTypeEnum};
use crate::types::{ArrayType, FloatType, FunctionType, IntType, PointerType, StructType, Type, VectorType, VoidType};
use crate::values::{FloatMathValue, FloatValue, IntMathValue, IntValue, PointerMathValue, PointerValue, VectorValue};
use crate::AddressSpace;

pub unsafe trait AsTypeRef {
    fn as_type_ref(&self) -> LLVMTypeRef;
}

macro_rules! trait_type_set {
    ($trait_name:ident: $($args:ident),*) => (
        $(
            unsafe impl<'ctx> $trait_name<'ctx> for $args<'ctx> {}
        )*
    );
}

pub unsafe trait AnyType<'ctx>: AsTypeRef + Debug {
    fn as_any_type_enum(&self) -> AnyTypeEnum<'ctx> {
        unsafe { AnyTypeEnum::new(self.as_type_ref()) }
    }

    fn print_to_string(&self) -> LLVMString {
        unsafe { Type::new(self.as_type_ref()).print_to_string() }
    }
}

pub unsafe trait BasicType<'ctx>: AnyType<'ctx> {
    fn as_basic_type_enum(&self) -> BasicTypeEnum<'ctx> {
        unsafe { BasicTypeEnum::new(self.as_type_ref()) }
    }

    fn fn_type(&self, param_types: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        unsafe { Type::new(self.as_type_ref()).fn_type(param_types, is_var_args) }
    }

    fn is_sized(&self) -> bool {
        unsafe { Type::new(self.as_type_ref()).is_sized() }
    }

    fn size_of(&self) -> Option<IntValue<'ctx>> {
        unsafe { Type::new(self.as_type_ref()).size_of() }
    }

    fn array_type(&self, size: u32) -> ArrayType<'ctx> {
        unsafe { Type::new(self.as_type_ref()).array_type(size) }
    }

    fn ptr_type(&self, address_space: AddressSpace) -> PointerType<'ctx> {
        unsafe { Type::new(self.as_type_ref()).ptr_type(address_space) }
    }
}

pub unsafe trait IntMathType<'ctx>: BasicType<'ctx> {
    type ValueType: IntMathValue<'ctx>;
    type MathConvType: FloatMathType<'ctx>;
    type PtrConvType: PointerMathType<'ctx>;
}

pub unsafe trait FloatMathType<'ctx>: BasicType<'ctx> {
    type ValueType: FloatMathValue<'ctx>;
    type MathConvType: IntMathType<'ctx>;
}

pub unsafe trait PointerMathType<'ctx>: BasicType<'ctx> {
    type ValueType: PointerMathValue<'ctx>;
    type PtrConvType: IntMathType<'ctx>;
}

trait_type_set! {AnyType: AnyTypeEnum, BasicTypeEnum, IntType, FunctionType, FloatType, PointerType, StructType, ArrayType, VoidType, VectorType}
trait_type_set! {BasicType: BasicTypeEnum, IntType, FloatType, PointerType, StructType, ArrayType, VectorType}

unsafe impl<'ctx> IntMathType<'ctx> for IntType<'ctx> {
    type ValueType = IntValue<'ctx>;
    type MathConvType = FloatType<'ctx>;
    type PtrConvType = PointerType<'ctx>;
}

unsafe impl<'ctx> IntMathType<'ctx> for VectorType<'ctx> {
    type ValueType = VectorValue<'ctx>;
    type MathConvType = VectorType<'ctx>;
    type PtrConvType = VectorType<'ctx>;
}

unsafe impl<'ctx> FloatMathType<'ctx> for FloatType<'ctx> {
    type ValueType = FloatValue<'ctx>;
    type MathConvType = IntType<'ctx>;
}

unsafe impl<'ctx> FloatMathType<'ctx> for VectorType<'ctx> {
    type ValueType = VectorValue<'ctx>;
    type MathConvType = VectorType<'ctx>;
}

unsafe impl<'ctx> PointerMathType<'ctx> for PointerType<'ctx> {
    type ValueType = PointerValue<'ctx>;
    type PtrConvType = IntType<'ctx>;
}

unsafe impl<'ctx> PointerMathType<'ctx> for VectorType<'ctx> {
    type ValueType = VectorValue<'ctx>;
    type PtrConvType = VectorType<'ctx>;
}
