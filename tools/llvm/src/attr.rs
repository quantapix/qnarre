//! `Attribute`s are optional modifiers to functions, function parameters, and return types.

use llvm_lib::core::{
    LLVMGetEnumAttributeKind, LLVMGetEnumAttributeKindForName, LLVMGetEnumAttributeValue, LLVMGetLastEnumAttributeKind,
    LLVMGetStringAttributeKind, LLVMGetStringAttributeValue, LLVMIsEnumAttribute, LLVMIsStringAttribute,
};
#[llvm_versions(12.0..=latest)]
use llvm_lib::core::{LLVMGetTypeAttributeValue, LLVMIsTypeAttribute};
use llvm_lib::prelude::LLVMAttributeRef;

use std::ffi::CStr;

#[llvm_versions(12.0..=latest)]
use crate::types::AnyTypeEnum;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attribute {
    pub(crate) attribute: LLVMAttributeRef,
}

impl Attribute {
    pub unsafe fn new(attribute: LLVMAttributeRef) -> Self {
        debug_assert!(!attribute.is_null());

        Attribute { attribute }
    }

    pub fn as_mut_ptr(&self) -> LLVMAttributeRef {
        self.attribute
    }

    pub fn is_enum(self) -> bool {
        unsafe { LLVMIsEnumAttribute(self.attribute) == 1 }
    }

    pub fn is_string(self) -> bool {
        unsafe { LLVMIsStringAttribute(self.attribute) == 1 }
    }

    #[llvm_versions(12.0..=latest)]
    pub fn is_type(self) -> bool {
        unsafe { LLVMIsTypeAttribute(self.attribute) == 1 }
    }

    pub fn get_named_enum_kind_id(name: &str) -> u32 {
        unsafe { LLVMGetEnumAttributeKindForName(name.as_ptr() as *const ::libc::c_char, name.len()) }
    }

    #[llvm_versions(4.0..=11.0)]
    pub fn get_enum_kind_id(self) -> u32 {
        assert!(self.get_enum_kind_id_is_valid()); // FIXME: SubTypes

        unsafe { LLVMGetEnumAttributeKind(self.attribute) }
    }

    #[llvm_versions(12.0..=latest)]
    pub fn get_enum_kind_id(self) -> u32 {
        assert!(self.get_enum_kind_id_is_valid()); // FIXME: SubTypes

        unsafe { LLVMGetEnumAttributeKind(self.attribute) }
    }

    #[llvm_versions(4.0..=11.0)]
    fn get_enum_kind_id_is_valid(self) -> bool {
        self.is_enum()
    }

    #[llvm_versions(12.0..=latest)]
    fn get_enum_kind_id_is_valid(self) -> bool {
        self.is_enum() || self.is_type()
    }

    pub fn get_last_enum_kind_id() -> u32 {
        unsafe { LLVMGetLastEnumAttributeKind() }
    }

    pub fn get_enum_value(self) -> u64 {
        assert!(self.is_enum()); // FIXME: SubTypes

        unsafe { LLVMGetEnumAttributeValue(self.attribute) }
    }

    pub fn get_string_kind_id(&self) -> &CStr {
        assert!(self.is_string()); // FIXME: SubTypes

        let mut length = 0;
        let cstr_ptr = unsafe { LLVMGetStringAttributeKind(self.attribute, &mut length) };

        unsafe { CStr::from_ptr(cstr_ptr) }
    }

    pub fn get_string_value(&self) -> &CStr {
        assert!(self.is_string()); // FIXME: SubTypes

        let mut length = 0;
        let cstr_ptr = unsafe { LLVMGetStringAttributeValue(self.attribute, &mut length) };

        unsafe { CStr::from_ptr(cstr_ptr) }
    }

    #[llvm_versions(12.0..=latest)]
    pub fn get_type_value(&self) -> AnyTypeEnum {
        assert!(self.is_type()); // FIXME: SubTypes

        unsafe { AnyTypeEnum::new(LLVMGetTypeAttributeValue(self.attribute)) }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AttributeLoc {
    Return,
    Param(u32),
    Function,
}

impl AttributeLoc {
    pub(crate) fn get_index(self) -> u32 {
        match self {
            AttributeLoc::Return => 0,
            AttributeLoc::Param(index) => {
                assert!(
                    index <= u32::max_value() - 2,
                    "Param index must be <= u32::max_value() - 2"
                );

                index + 1
            },
            AttributeLoc::Function => u32::max_value(),
        }
    }
}
