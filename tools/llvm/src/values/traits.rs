use llvm_lib::prelude::LLVMValueRef;

use std::fmt::Debug;

use crate::support::LLVMString;
use crate::types::{FloatMathType, FloatType, IntMathType, IntType, PointerMathType, PointerType, VectorType};
use crate::values::{
    AggregateValueEnum, AnyValueEnum, ArrayValue, BasicValueEnum, BasicValueUse, CallSiteValue, FloatValue,
    FunctionValue, GlobalValue, InstructionValue, IntValue, PhiValue, PointerValue, StructValue, Value, VectorValue,
};

use super::{BasicMetadataValueEnum, MetadataValue};

pub unsafe trait AsValueRef {
    fn as_value_ref(&self) -> LLVMValueRef;
}

macro_rules! trait_value_set {
    ($trait_name:ident: $($args:ident),*) => (
        $(
            unsafe impl<'ctx> $trait_name<'ctx> for $args<'ctx> {}
        )*

    );
}

macro_rules! math_trait_value_set {
    ($trait_name:ident: $(($value_type:ident => $base_type:ident)),*) => (
        $(
            unsafe impl<'ctx> $trait_name<'ctx> for $value_type<'ctx> {
                type BaseType = $base_type<'ctx>;
                unsafe fn new(value: LLVMValueRef) -> $value_type<'ctx> {
                    unsafe {
                        $value_type::new(value)
                    }
                }
            }
        )*
    )
}

pub unsafe trait AggregateValue<'ctx>: BasicValue<'ctx> {
    fn as_aggregate_value_enum(&self) -> AggregateValueEnum<'ctx> {
        unsafe { AggregateValueEnum::new(self.as_value_ref()) }
    }

    #[llvm_versions(4.0..=14.0)]
    fn const_extract_value(&self, indexes: &mut [u32]) -> BasicValueEnum<'ctx> {
        use llvm_lib::core::LLVMConstExtractValue;

        unsafe {
            BasicValueEnum::new(LLVMConstExtractValue(
                self.as_value_ref(),
                indexes.as_mut_ptr(),
                indexes.len() as u32,
            ))
        }
    }

    #[llvm_versions(4.0..=14.0)]
    fn const_insert_value<BV: BasicValue<'ctx>>(&self, value: BV, indexes: &mut [u32]) -> BasicValueEnum<'ctx> {
        use llvm_lib::core::LLVMConstInsertValue;

        unsafe {
            BasicValueEnum::new(LLVMConstInsertValue(
                self.as_value_ref(),
                value.as_value_ref(),
                indexes.as_mut_ptr(),
                indexes.len() as u32,
            ))
        }
    }
}

pub unsafe trait BasicValue<'ctx>: AnyValue<'ctx> {
    fn as_basic_value_enum(&self) -> BasicValueEnum<'ctx> {
        unsafe { BasicValueEnum::new(self.as_value_ref()) }
    }

    fn as_instruction_value(&self) -> Option<InstructionValue<'ctx>> {
        let value = unsafe { Value::new(self.as_value_ref()) };

        if !value.is_instruction() {
            return None;
        }

        unsafe { Some(InstructionValue::new(self.as_value_ref())) }
    }

    fn get_first_use(&self) -> Option<BasicValueUse> {
        unsafe { Value::new(self.as_value_ref()).get_first_use() }
    }

    fn set_name(&self, name: &str) {
        unsafe { Value::new(self.as_value_ref()).set_name(name) }
    }
}

pub unsafe trait IntMathValue<'ctx>: BasicValue<'ctx> {
    type BaseType: IntMathType<'ctx>;
    unsafe fn new(value: LLVMValueRef) -> Self;
}

pub unsafe trait FloatMathValue<'ctx>: BasicValue<'ctx> {
    type BaseType: FloatMathType<'ctx>;
    unsafe fn new(value: LLVMValueRef) -> Self;
}

pub unsafe trait PointerMathValue<'ctx>: BasicValue<'ctx> {
    type BaseType: PointerMathType<'ctx>;
    unsafe fn new(value: LLVMValueRef) -> Self;
}

pub unsafe trait AnyValue<'ctx>: AsValueRef + Debug {
    fn as_any_value_enum(&self) -> AnyValueEnum<'ctx> {
        unsafe { AnyValueEnum::new(self.as_value_ref()) }
    }

    fn print_to_string(&self) -> LLVMString {
        unsafe { Value::new(self.as_value_ref()).print_to_string() }
    }
}

trait_value_set! {AggregateValue: ArrayValue, AggregateValueEnum, StructValue}
trait_value_set! {AnyValue: AnyValueEnum, BasicValueEnum, BasicMetadataValueEnum, AggregateValueEnum, ArrayValue, IntValue, FloatValue, GlobalValue, PhiValue, PointerValue, FunctionValue, StructValue, VectorValue, InstructionValue, CallSiteValue, MetadataValue}
trait_value_set! {BasicValue: ArrayValue, BasicValueEnum, AggregateValueEnum, IntValue, FloatValue, GlobalValue, StructValue, PointerValue, VectorValue}
math_trait_value_set! {IntMathValue: (IntValue => IntType), (VectorValue => VectorType), (PointerValue => IntType)}
math_trait_value_set! {FloatMathValue: (FloatValue => FloatType), (VectorValue => VectorType)}
math_trait_value_set! {PointerMathValue: (PointerValue => PointerType), (VectorValue => VectorType)}
