use llvm_lib::core::{
    LLVMConstAllOnes, LLVMConstArray, LLVMConstInt, LLVMConstIntOfArbitraryPrecision, LLVMConstIntOfStringAndSize,
    LLVMGetIntTypeWidth,
};
use llvm_lib::execution_engine::LLVMCreateGenericValueOfInt;
use llvm_lib::prelude::{LLVMTypeRef, LLVMValueRef};

use crate::context::ContextRef;
use crate::support::LLVMString;
use crate::types::traits::AsTypeRef;
use crate::types::{ArrayType, FunctionType, PointerType, Type, VectorType};
use crate::values::{ArrayValue, AsValueRef, GenericValue, IntValue};
use crate::AddressSpace;

use crate::types::enums::BasicMetadataTypeEnum;
use std::convert::TryFrom;
use std::fmt::{self, Display};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StringRadix {
    Binary = 2,
    Octal = 8,
    Decimal = 10,
    Hexadecimal = 16,
    Alphanumeric = 36,
}

impl TryFrom<u8> for StringRadix {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            2 => Ok(StringRadix::Binary),
            8 => Ok(StringRadix::Octal),
            10 => Ok(StringRadix::Decimal),
            16 => Ok(StringRadix::Hexadecimal),
            36 => Ok(StringRadix::Alphanumeric),
            _ => Err(()),
        }
    }
}

impl StringRadix {
    pub fn matches_str(&self, slice: &str) -> bool {
        let slice = slice.strip_prefix(|c| c == '+' || c == '-').unwrap_or(slice);

        if slice.is_empty() {
            return false;
        }

        let mut it = slice.chars();
        match self {
            StringRadix::Binary => it.all(|c| matches!(c, '0'..='1')),
            StringRadix::Octal => it.all(|c| matches!(c, '0'..='7')),
            StringRadix::Decimal => it.all(|c| matches!(c, '0'..='9')),
            StringRadix::Hexadecimal => it.all(|c| matches!(c, '0'..='9' | 'a'..='f' | 'A'..='F')),
            StringRadix::Alphanumeric => it.all(|c| matches!(c, '0'..='9' | 'a'..='z' | 'A'..='Z')),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct IntType<'ctx> {
    int_type: Type<'ctx>,
}

impl<'ctx> IntType<'ctx> {
    pub unsafe fn new(int_type: LLVMTypeRef) -> Self {
        assert!(!int_type.is_null());

        IntType {
            int_type: Type::new(int_type),
        }
    }

    pub fn const_int(self, value: u64, sign_extend: bool) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstInt(self.as_type_ref(), value, sign_extend as i32)) }
    }

    pub fn const_int_from_string(self, slice: &str, radix: StringRadix) -> Option<IntValue<'ctx>> {
        if !radix.matches_str(slice) {
            return None;
        }

        unsafe {
            Some(IntValue::new(LLVMConstIntOfStringAndSize(
                self.as_type_ref(),
                slice.as_ptr() as *const ::libc::c_char,
                slice.len() as u32,
                radix as u8,
            )))
        }
    }

    pub fn const_int_arbitrary_precision(self, words: &[u64]) -> IntValue<'ctx> {
        unsafe {
            IntValue::new(LLVMConstIntOfArbitraryPrecision(
                self.as_type_ref(),
                words.len() as u32,
                words.as_ptr(),
            ))
        }
    }

    pub fn const_all_ones(self) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstAllOnes(self.as_type_ref())) }
    }

    pub fn const_zero(self) -> IntValue<'ctx> {
        unsafe { IntValue::new(self.int_type.const_zero()) }
    }

    pub fn fn_type(self, param_types: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.int_type.fn_type(param_types, is_var_args)
    }

    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.int_type.array_type(size)
    }

    pub fn vec_type(self, size: u32) -> VectorType<'ctx> {
        self.int_type.vec_type(size)
    }

    pub fn get_context(self) -> ContextRef<'ctx> {
        self.int_type.get_context()
    }

    pub fn size_of(self) -> IntValue<'ctx> {
        self.int_type.size_of().unwrap()
    }

    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.int_type.get_alignment()
    }

    pub fn ptr_type(self, address_space: AddressSpace) -> PointerType<'ctx> {
        self.int_type.ptr_type(address_space)
    }

    pub fn get_bit_width(self) -> u32 {
        unsafe { LLVMGetIntTypeWidth(self.as_type_ref()) }
    }

    pub fn print_to_string(self) -> LLVMString {
        self.int_type.print_to_string()
    }

    pub fn get_undef(self) -> IntValue<'ctx> {
        unsafe { IntValue::new(self.int_type.get_undef()) }
    }

    pub fn create_generic_value(self, value: u64, is_signed: bool) -> GenericValue<'ctx> {
        unsafe { GenericValue::new(LLVMCreateGenericValueOfInt(self.as_type_ref(), value, is_signed as i32)) }
    }

    pub fn const_array(self, values: &[IntValue<'ctx>]) -> ArrayValue<'ctx> {
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

unsafe impl AsTypeRef for IntType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.int_type.ty
    }
}

impl Display for IntType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
