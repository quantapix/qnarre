//! A value is an instance of a type.

#[deny(missing_docs)]
mod array_value;
#[deny(missing_docs)]
mod basic_value_use;
#[deny(missing_docs)]
mod call_site_value;
mod enums;
mod float_value;
mod fn_value;
mod generic_value;
mod global_value;
mod instruction_value;
mod int_value;
mod metadata_value;
mod phi_value;
mod ptr_value;
mod struct_value;
mod traits;
mod vec_value;

#[cfg(not(any(feature = "llvm15-0", feature = "llvm16-0")))]
mod callable_value;

#[cfg(not(any(feature = "llvm15-0", feature = "llvm16-0")))]
pub use crate::values::callable_value::CallableValue;

use crate::support::{to_c_str, LLVMString};
pub use crate::values::array_value::ArrayValue;
pub use crate::values::basic_value_use::BasicValueUse;
pub use crate::values::call_site_value::CallSiteValue;
pub use crate::values::enums::{AggregateValueEnum, AnyValueEnum, BasicMetadataValueEnum, BasicValueEnum};
pub use crate::values::float_value::FloatValue;
pub use crate::values::fn_value::FunctionValue;
pub use crate::values::generic_value::GenericValue;
pub use crate::values::global_value::GlobalValue;
#[llvm_versions(7.0..=latest)]
pub use crate::values::global_value::UnnamedAddress;
pub use crate::values::instruction_value::{InstructionOpcode, InstructionValue};
pub use crate::values::int_value::IntValue;
pub use crate::values::metadata_value::{MetadataValue, FIRST_CUSTOM_METADATA_KIND_ID};
pub use crate::values::phi_value::PhiValue;
pub use crate::values::ptr_value::PointerValue;
pub use crate::values::struct_value::StructValue;
pub use crate::values::traits::AsValueRef;
pub use crate::values::traits::{AggregateValue, AnyValue, BasicValue, FloatMathValue, IntMathValue, PointerMathValue};
pub use crate::values::vec_value::VectorValue;

use llvm_lib::core::{
    LLVMDumpValue, LLVMGetFirstUse, LLVMGetSection, LLVMIsAInstruction, LLVMIsConstant, LLVMIsNull, LLVMIsUndef,
    LLVMPrintTypeToString, LLVMPrintValueToString, LLVMReplaceAllUsesWith, LLVMSetSection, LLVMTypeOf,
};
use llvm_lib::prelude::{LLVMTypeRef, LLVMValueRef};

use std::ffi::CStr;
use std::fmt;
use std::marker::PhantomData;

#[derive(PartialEq, Eq, Clone, Copy, Hash)]
struct Value<'ctx> {
    value: LLVMValueRef,
    _marker: PhantomData<&'ctx ()>,
}

impl<'ctx> Value<'ctx> {
    pub(crate) unsafe fn new(value: LLVMValueRef) -> Self {
        debug_assert!(
            !value.is_null(),
            "This should never happen since containing struct should check null ptrs"
        );

        Value {
            value,
            _marker: PhantomData,
        }
    }

    fn is_instruction(self) -> bool {
        unsafe { !LLVMIsAInstruction(self.value).is_null() }
    }

    fn as_instruction(self) -> Option<InstructionValue<'ctx>> {
        if !self.is_instruction() {
            return None;
        }

        unsafe { Some(InstructionValue::new(self.value)) }
    }

    fn is_null(self) -> bool {
        unsafe { LLVMIsNull(self.value) == 1 }
    }

    fn is_const(self) -> bool {
        unsafe { LLVMIsConstant(self.value) == 1 }
    }

    fn set_name(self, name: &str) {
        let c_string = to_c_str(name);

        #[cfg(any(feature = "llvm4-0", feature = "llvm5-0", feature = "llvm6-0"))]
        {
            use llvm_lib::core::LLVMSetValueName;

            unsafe {
                LLVMSetValueName(self.value, c_string.as_ptr());
            }
        }
        #[cfg(not(any(feature = "llvm4-0", feature = "llvm5-0", feature = "llvm6-0")))]
        {
            use llvm_lib::core::LLVMSetValueName2;

            unsafe { LLVMSetValueName2(self.value, c_string.as_ptr(), name.len()) }
        }
    }

    fn get_name(&self) -> &CStr {
        #[cfg(any(feature = "llvm4-0", feature = "llvm5-0", feature = "llvm6-0"))]
        let ptr = unsafe {
            use llvm_lib::core::LLVMGetValueName;

            LLVMGetValueName(self.value)
        };
        #[cfg(not(any(feature = "llvm4-0", feature = "llvm5-0", feature = "llvm6-0")))]
        let ptr = unsafe {
            use llvm_lib::core::LLVMGetValueName2;
            let mut len = 0;

            LLVMGetValueName2(self.value, &mut len)
        };

        unsafe { CStr::from_ptr(ptr) }
    }

    fn is_undef(self) -> bool {
        unsafe { LLVMIsUndef(self.value) == 1 }
    }

    fn get_type(self) -> LLVMTypeRef {
        unsafe { LLVMTypeOf(self.value) }
    }

    fn print_to_string(self) -> LLVMString {
        unsafe { LLVMString::new(LLVMPrintValueToString(self.value)) }
    }

    fn print_to_stderr(self) {
        unsafe { LLVMDumpValue(self.value) }
    }

    fn replace_all_uses_with(self, other: LLVMValueRef) {
        if self.value != other {
            unsafe { LLVMReplaceAllUsesWith(self.value, other) }
        }
    }

    pub fn get_first_use(self) -> Option<BasicValueUse<'ctx>> {
        let use_ = unsafe { LLVMGetFirstUse(self.value) };

        if use_.is_null() {
            return None;
        }

        unsafe { Some(BasicValueUse::new(use_)) }
    }

    pub fn get_section(&self) -> Option<&CStr> {
        let ptr = unsafe { LLVMGetSection(self.value) };

        if ptr.is_null() {
            return None;
        }

        if cfg!(target_os = "macos") {
            let name = unsafe { CStr::from_ptr(ptr) };
            let name_string = name.to_string_lossy();
            let mut chars = name_string.chars();
            if Some(',') == chars.next() {
                Some(unsafe { CStr::from_ptr(ptr.add(1)) })
            } else {
                Some(name)
            }
        } else {
            Some(unsafe { CStr::from_ptr(ptr) })
        }
    }

    fn set_section(self, section: Option<&str>) {
        #[cfg(target_os = "macos")]
        let section = section.map(|s| {
            if s.contains(",") {
                format!("{}", s)
            } else {
                format!(",{}", s)
            }
        });

        let c_string = section.as_deref().map(to_c_str);

        unsafe {
            LLVMSetSection(
                self.value,
                c_string.as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null()),
            )
        }
    }
}

impl fmt::Debug for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let llvm_value = self.print_to_string();
        let llvm_type = unsafe { CStr::from_ptr(LLVMPrintTypeToString(LLVMTypeOf(self.value))) };
        let name = self.get_name();
        let is_const = self.is_const();
        let is_null = self.is_null();
        let is_undef = self.is_undef();

        f.debug_struct("Value")
            .field("name", &name)
            .field("address", &self.value)
            .field("is_const", &is_const)
            .field("is_null", &is_null)
            .field("is_undef", &is_undef)
            .field("llvm_value", &llvm_value)
            .field("llvm_type", &llvm_type)
            .finish()
    }
}

use llvm_lib::core::{LLVMGetAsString, LLVMIsAConstantArray, LLVMIsAConstantDataArray, LLVMIsConstantString};
use llvm_lib::prelude::LLVMValueRef;

use std::ffi::CStr;
use std::fmt::{self, Display};

use crate::types::ArrayType;
use crate::values::traits::{AnyValue, AsValueRef};
use crate::values::{InstructionValue, Value};

#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub struct ArrayValue<'ctx> {
    array_value: Value<'ctx>,
}

impl<'ctx> ArrayValue<'ctx> {
    pub(crate) unsafe fn new(value: LLVMValueRef) -> Self {
        assert!(!value.is_null());

        ArrayValue {
            array_value: Value::new(value),
        }
    }

    pub fn get_name(&self) -> &CStr {
        self.array_value.get_name()
    }

    pub fn set_name(&self, name: &str) {
        self.array_value.set_name(name)
    }

    pub fn get_type(self) -> ArrayType<'ctx> {
        unsafe { ArrayType::new(self.array_value.get_type()) }
    }

    pub fn is_null(self) -> bool {
        self.array_value.is_null()
    }

    pub fn is_undef(self) -> bool {
        self.array_value.is_undef()
    }

    pub fn print_to_stderr(self) {
        self.array_value.print_to_stderr()
    }

    pub fn as_instruction(self) -> Option<InstructionValue<'ctx>> {
        self.array_value.as_instruction()
    }

    pub fn replace_all_uses_with(self, other: ArrayValue<'ctx>) {
        self.array_value.replace_all_uses_with(other.as_value_ref())
    }

    pub fn is_const(self) -> bool {
        self.array_value.is_const()
    }

    pub fn is_const_string(self) -> bool {
        unsafe { LLVMIsConstantString(self.as_value_ref()) == 1 }
    }

    pub fn get_string_constant(&self) -> Option<&CStr> {
        let mut len = 0;
        let ptr = unsafe { LLVMGetAsString(self.as_value_ref(), &mut len) };

        if ptr.is_null() {
            None
        } else {
            unsafe { Some(CStr::from_ptr(ptr)) }
        }
    }
}

unsafe impl AsValueRef for ArrayValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.array_value.value
    }
}

impl Display for ArrayValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

impl fmt::Debug for ArrayValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let llvm_value = self.print_to_string();
        let llvm_type = self.get_type();
        let name = self.get_name();
        let is_const = self.is_const();
        let is_null = self.is_null();
        let is_const_array = unsafe { !LLVMIsAConstantArray(self.as_value_ref()).is_null() };
        let is_const_data_array = unsafe { !LLVMIsAConstantDataArray(self.as_value_ref()).is_null() };

        f.debug_struct("ArrayValue")
            .field("name", &name)
            .field("address", &self.as_value_ref())
            .field("is_const", &is_const)
            .field("is_const_array", &is_const_array)
            .field("is_const_data_array", &is_const_data_array)
            .field("is_null", &is_null)
            .field("llvm_value", &llvm_value)
            .field("llvm_type", &llvm_type)
            .finish()
    }
}

use either::{
    Either,
    Either::{Left, Right},
};
use llvm_lib::core::{LLVMGetNextUse, LLVMGetUsedValue, LLVMGetUser, LLVMIsABasicBlock, LLVMValueAsBasicBlock};
use llvm_lib::prelude::LLVMUseRef;

use std::marker::PhantomData;

use crate::basic_block::BasicBlock;
use crate::values::{AnyValueEnum, BasicValueEnum};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BasicValueUse<'ctx>(LLVMUseRef, PhantomData<&'ctx ()>);

impl<'ctx> BasicValueUse<'ctx> {
    pub(crate) unsafe fn new(use_: LLVMUseRef) -> Self {
        debug_assert!(!use_.is_null());

        BasicValueUse(use_, PhantomData)
    }

    pub fn get_next_use(self) -> Option<Self> {
        let use_ = unsafe { LLVMGetNextUse(self.0) };

        if use_.is_null() {
            return None;
        }

        unsafe { Some(Self::new(use_)) }
    }

    pub fn get_user(self) -> AnyValueEnum<'ctx> {
        unsafe { AnyValueEnum::new(LLVMGetUser(self.0)) }
    }

    pub fn get_used_value(self) -> Either<BasicValueEnum<'ctx>, BasicBlock<'ctx>> {
        let used_value = unsafe { LLVMGetUsedValue(self.0) };

        let is_basic_block = unsafe { !LLVMIsABasicBlock(used_value).is_null() };

        if is_basic_block {
            let bb = unsafe { BasicBlock::new(LLVMValueAsBasicBlock(used_value)) };

            Right(bb.expect("BasicBlock should always be valid"))
        } else {
            unsafe { Left(BasicValueEnum::new(used_value)) }
        }
    }
}

use std::fmt::{self, Display};

use either::Either;
use llvm_lib::core::{
    LLVMGetInstructionCallConv, LLVMGetTypeKind, LLVMIsTailCall, LLVMSetInstrParamAlignment,
    LLVMSetInstructionCallConv, LLVMSetTailCall, LLVMTypeOf,
};
use llvm_lib::prelude::LLVMValueRef;
use llvm_lib::LLVMTypeKind;

use crate::attributes::{Attribute, AttributeLoc};
use crate::values::{AsValueRef, BasicValueEnum, FunctionValue, InstructionValue, Value};

use super::AnyValue;

///
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct CallSiteValue<'ctx>(Value<'ctx>);

impl<'ctx> CallSiteValue<'ctx> {
    pub(crate) unsafe fn new(value: LLVMValueRef) -> Self {
        CallSiteValue(Value::new(value))
    }

    pub fn set_tail_call(self, tail_call: bool) {
        unsafe { LLVMSetTailCall(self.as_value_ref(), tail_call as i32) }
    }

    pub fn is_tail_call(self) -> bool {
        unsafe { LLVMIsTailCall(self.as_value_ref()) == 1 }
    }

    pub fn try_as_basic_value(self) -> Either<BasicValueEnum<'ctx>, InstructionValue<'ctx>> {
        unsafe {
            match LLVMGetTypeKind(LLVMTypeOf(self.as_value_ref())) {
                LLVMTypeKind::LLVMVoidTypeKind => Either::Right(InstructionValue::new(self.as_value_ref())),
                _ => Either::Left(BasicValueEnum::new(self.as_value_ref())),
            }
        }
    }

    pub fn add_attribute(self, loc: AttributeLoc, attribute: Attribute) {
        use llvm_lib::core::LLVMAddCallSiteAttribute;

        unsafe { LLVMAddCallSiteAttribute(self.as_value_ref(), loc.get_index(), attribute.attribute) }
    }

    pub fn get_called_fn_value(self) -> FunctionValue<'ctx> {
        use llvm_lib::core::LLVMGetCalledValue;

        unsafe { FunctionValue::new(LLVMGetCalledValue(self.as_value_ref())).expect("This should never be null?") }
    }

    pub fn count_attributes(self, loc: AttributeLoc) -> u32 {
        use llvm_lib::core::LLVMGetCallSiteAttributeCount;

        unsafe { LLVMGetCallSiteAttributeCount(self.as_value_ref(), loc.get_index()) }
    }

    pub fn attributes(self, loc: AttributeLoc) -> Vec<Attribute> {
        use llvm_lib::core::LLVMGetCallSiteAttributes;
        use std::mem::{ManuallyDrop, MaybeUninit};

        let count = self.count_attributes(loc) as usize;

        let mut attribute_refs: Vec<MaybeUninit<Attribute>> = vec![MaybeUninit::uninit(); count];

        unsafe {
            LLVMGetCallSiteAttributes(
                self.as_value_ref(),
                loc.get_index(),
                attribute_refs.as_mut_ptr() as *mut _,
            )
        }

        unsafe {
            let mut attribute_refs = ManuallyDrop::new(attribute_refs);

            Vec::from_raw_parts(
                attribute_refs.as_mut_ptr() as *mut Attribute,
                attribute_refs.len(),
                attribute_refs.capacity(),
            )
        }
    }

    pub fn get_enum_attribute(self, loc: AttributeLoc, kind_id: u32) -> Option<Attribute> {
        use llvm_lib::core::LLVMGetCallSiteEnumAttribute;

        let ptr = unsafe { LLVMGetCallSiteEnumAttribute(self.as_value_ref(), loc.get_index(), kind_id) };

        if ptr.is_null() {
            return None;
        }

        unsafe { Some(Attribute::new(ptr)) }
    }

    pub fn get_string_attribute(self, loc: AttributeLoc, key: &str) -> Option<Attribute> {
        use llvm_lib::core::LLVMGetCallSiteStringAttribute;

        let ptr = unsafe {
            LLVMGetCallSiteStringAttribute(
                self.as_value_ref(),
                loc.get_index(),
                key.as_ptr() as *const ::libc::c_char,
                key.len() as u32,
            )
        };

        if ptr.is_null() {
            return None;
        }

        unsafe { Some(Attribute::new(ptr)) }
    }

    pub fn remove_enum_attribute(self, loc: AttributeLoc, kind_id: u32) {
        use llvm_lib::core::LLVMRemoveCallSiteEnumAttribute;

        unsafe { LLVMRemoveCallSiteEnumAttribute(self.as_value_ref(), loc.get_index(), kind_id) }
    }

    pub fn remove_string_attribute(self, loc: AttributeLoc, key: &str) {
        use llvm_lib::core::LLVMRemoveCallSiteStringAttribute;

        unsafe {
            LLVMRemoveCallSiteStringAttribute(
                self.as_value_ref(),
                loc.get_index(),
                key.as_ptr() as *const ::libc::c_char,
                key.len() as u32,
            )
        }
    }

    pub fn count_arguments(self) -> u32 {
        use llvm_lib::core::LLVMGetNumArgOperands;

        unsafe { LLVMGetNumArgOperands(self.as_value_ref()) }
    }

    pub fn get_call_convention(self) -> u32 {
        unsafe { LLVMGetInstructionCallConv(self.as_value_ref()) }
    }

    pub fn set_call_convention(self, conv: u32) {
        unsafe { LLVMSetInstructionCallConv(self.as_value_ref(), conv) }
    }

    pub fn set_alignment_attribute(self, loc: AttributeLoc, alignment: u32) {
        assert_eq!(alignment.count_ones(), 1, "Alignment must be a power of two.");

        unsafe { LLVMSetInstrParamAlignment(self.as_value_ref(), loc.get_index(), alignment) }
    }
}

unsafe impl AsValueRef for CallSiteValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.0.value
    }
}

impl Display for CallSiteValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

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

use llvm_lib::core::{LLVMGetTypeKind, LLVMGetValueKind, LLVMIsAInstruction, LLVMTypeOf};
use llvm_lib::prelude::LLVMValueRef;
use llvm_lib::{LLVMTypeKind, LLVMValueKind};

use crate::types::{AnyTypeEnum, BasicTypeEnum};
use crate::values::traits::AsValueRef;
use crate::values::{
    ArrayValue, FloatValue, FunctionValue, InstructionValue, IntValue, MetadataValue, PhiValue, PointerValue,
    StructValue, VectorValue,
};

use std::convert::TryFrom;
use std::fmt::{self, Display};

use super::AnyValue;

macro_rules! enum_value_set {
    ($enum_name:ident: $($args:ident),*) => (
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $enum_name<'ctx> {
            $(
                $args($args<'ctx>),
            )*
        }

        unsafe impl AsValueRef for $enum_name<'_> {
            fn as_value_ref(&self) -> LLVMValueRef {
                match *self {
                    $(
                        $enum_name::$args(ref t) => t.as_value_ref(),
                    )*
                }
            }
        }

        $(
            impl<'ctx> From<$args<'ctx>> for $enum_name<'ctx> {
                fn from(value: $args) -> $enum_name {
                    $enum_name::$args(value)
                }
            }

            impl<'ctx> PartialEq<$args<'ctx>> for $enum_name<'ctx> {
                fn eq(&self, other: &$args<'ctx>) -> bool {
                    self.as_value_ref() == other.as_value_ref()
                }
            }

            impl<'ctx> PartialEq<$enum_name<'ctx>> for $args<'ctx> {
                fn eq(&self, other: &$enum_name<'ctx>) -> bool {
                    self.as_value_ref() == other.as_value_ref()
                }
            }

            impl<'ctx> TryFrom<$enum_name<'ctx>> for $args<'ctx> {
                type Error = ();

                fn try_from(value: $enum_name<'ctx>) -> Result<Self, Self::Error> {
                    match value {
                        $enum_name::$args(ty) => Ok(ty),
                        _ => Err(()),
                    }
                }
            }
        )*
    );
}

enum_value_set! {AggregateValueEnum: ArrayValue, StructValue}
enum_value_set! {AnyValueEnum: ArrayValue, IntValue, FloatValue, PhiValue, FunctionValue, PointerValue, StructValue, VectorValue, InstructionValue, MetadataValue}
enum_value_set! {BasicValueEnum: ArrayValue, IntValue, FloatValue, PointerValue, StructValue, VectorValue}
enum_value_set! {BasicMetadataValueEnum: ArrayValue, IntValue, FloatValue, PointerValue, StructValue, VectorValue, MetadataValue}

impl<'ctx> AnyValueEnum<'ctx> {
    pub(crate) unsafe fn new(value: LLVMValueRef) -> Self {
        match LLVMGetTypeKind(LLVMTypeOf(value)) {
            LLVMTypeKind::LLVMFloatTypeKind
            | LLVMTypeKind::LLVMFP128TypeKind
            | LLVMTypeKind::LLVMDoubleTypeKind
            | LLVMTypeKind::LLVMHalfTypeKind
            | LLVMTypeKind::LLVMX86_FP80TypeKind
            | LLVMTypeKind::LLVMPPC_FP128TypeKind => AnyValueEnum::FloatValue(FloatValue::new(value)),
            LLVMTypeKind::LLVMIntegerTypeKind => AnyValueEnum::IntValue(IntValue::new(value)),
            LLVMTypeKind::LLVMStructTypeKind => AnyValueEnum::StructValue(StructValue::new(value)),
            LLVMTypeKind::LLVMPointerTypeKind => match LLVMGetValueKind(value) {
                LLVMValueKind::LLVMFunctionValueKind => AnyValueEnum::FunctionValue(FunctionValue::new(value).unwrap()),
                _ => AnyValueEnum::PointerValue(PointerValue::new(value)),
            },
            LLVMTypeKind::LLVMArrayTypeKind => AnyValueEnum::ArrayValue(ArrayValue::new(value)),
            LLVMTypeKind::LLVMVectorTypeKind => AnyValueEnum::VectorValue(VectorValue::new(value)),
            LLVMTypeKind::LLVMFunctionTypeKind => AnyValueEnum::FunctionValue(FunctionValue::new(value).unwrap()),
            LLVMTypeKind::LLVMVoidTypeKind => {
                if LLVMIsAInstruction(value).is_null() {
                    panic!("Void value isn't an instruction.");
                }
                AnyValueEnum::InstructionValue(InstructionValue::new(value))
            },
            LLVMTypeKind::LLVMMetadataTypeKind => panic!("Metadata values are not supported as AnyValue's."),
            _ => panic!("The given type is not supported."),
        }
    }

    pub fn get_type(&self) -> AnyTypeEnum<'ctx> {
        unsafe { AnyTypeEnum::new(LLVMTypeOf(self.as_value_ref())) }
    }

    pub fn is_array_value(self) -> bool {
        matches!(self, AnyValueEnum::ArrayValue(_))
    }

    pub fn is_int_value(self) -> bool {
        matches!(self, AnyValueEnum::IntValue(_))
    }

    pub fn is_float_value(self) -> bool {
        matches!(self, AnyValueEnum::FloatValue(_))
    }

    pub fn is_phi_value(self) -> bool {
        matches!(self, AnyValueEnum::PhiValue(_))
    }

    pub fn is_function_value(self) -> bool {
        matches!(self, AnyValueEnum::FunctionValue(_))
    }

    pub fn is_pointer_value(self) -> bool {
        matches!(self, AnyValueEnum::PointerValue(_))
    }

    pub fn is_struct_value(self) -> bool {
        matches!(self, AnyValueEnum::StructValue(_))
    }

    pub fn is_vector_value(self) -> bool {
        matches!(self, AnyValueEnum::VectorValue(_))
    }

    pub fn is_instruction_value(self) -> bool {
        matches!(self, AnyValueEnum::InstructionValue(_))
    }

    pub fn into_array_value(self) -> ArrayValue<'ctx> {
        if let AnyValueEnum::ArrayValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the ArrayValue variant", self)
        }
    }

    pub fn into_int_value(self) -> IntValue<'ctx> {
        if let AnyValueEnum::IntValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the IntValue variant", self)
        }
    }

    pub fn into_float_value(self) -> FloatValue<'ctx> {
        if let AnyValueEnum::FloatValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the FloatValue variant", self)
        }
    }

    pub fn into_phi_value(self) -> PhiValue<'ctx> {
        if let AnyValueEnum::PhiValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the PhiValue variant", self)
        }
    }

    pub fn into_function_value(self) -> FunctionValue<'ctx> {
        if let AnyValueEnum::FunctionValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the FunctionValue variant", self)
        }
    }

    pub fn into_pointer_value(self) -> PointerValue<'ctx> {
        if let AnyValueEnum::PointerValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the PointerValue variant", self)
        }
    }

    pub fn into_struct_value(self) -> StructValue<'ctx> {
        if let AnyValueEnum::StructValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the StructValue variant", self)
        }
    }

    pub fn into_vector_value(self) -> VectorValue<'ctx> {
        if let AnyValueEnum::VectorValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the VectorValue variant", self)
        }
    }

    pub fn into_instruction_value(self) -> InstructionValue<'ctx> {
        if let AnyValueEnum::InstructionValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the InstructionValue variant", self)
        }
    }
}

impl<'ctx> BasicValueEnum<'ctx> {
    pub(crate) unsafe fn new(value: LLVMValueRef) -> Self {
        match LLVMGetTypeKind(LLVMTypeOf(value)) {
            LLVMTypeKind::LLVMFloatTypeKind
            | LLVMTypeKind::LLVMFP128TypeKind
            | LLVMTypeKind::LLVMDoubleTypeKind
            | LLVMTypeKind::LLVMHalfTypeKind
            | LLVMTypeKind::LLVMX86_FP80TypeKind
            | LLVMTypeKind::LLVMPPC_FP128TypeKind => BasicValueEnum::FloatValue(FloatValue::new(value)),
            LLVMTypeKind::LLVMIntegerTypeKind => BasicValueEnum::IntValue(IntValue::new(value)),
            LLVMTypeKind::LLVMStructTypeKind => BasicValueEnum::StructValue(StructValue::new(value)),
            LLVMTypeKind::LLVMPointerTypeKind => BasicValueEnum::PointerValue(PointerValue::new(value)),
            LLVMTypeKind::LLVMArrayTypeKind => BasicValueEnum::ArrayValue(ArrayValue::new(value)),
            LLVMTypeKind::LLVMVectorTypeKind => BasicValueEnum::VectorValue(VectorValue::new(value)),
            _ => unreachable!("The given type is not a basic type."),
        }
    }

    pub fn set_name(&self, name: &str) {
        match self {
            BasicValueEnum::ArrayValue(v) => v.set_name(name),
            BasicValueEnum::IntValue(v) => v.set_name(name),
            BasicValueEnum::FloatValue(v) => v.set_name(name),
            BasicValueEnum::PointerValue(v) => v.set_name(name),
            BasicValueEnum::StructValue(v) => v.set_name(name),
            BasicValueEnum::VectorValue(v) => v.set_name(name),
        }
    }

    pub fn get_type(&self) -> BasicTypeEnum<'ctx> {
        unsafe { BasicTypeEnum::new(LLVMTypeOf(self.as_value_ref())) }
    }

    pub fn is_array_value(self) -> bool {
        matches!(self, BasicValueEnum::ArrayValue(_))
    }

    pub fn is_int_value(self) -> bool {
        matches!(self, BasicValueEnum::IntValue(_))
    }

    pub fn is_float_value(self) -> bool {
        matches!(self, BasicValueEnum::FloatValue(_))
    }

    pub fn is_pointer_value(self) -> bool {
        matches!(self, BasicValueEnum::PointerValue(_))
    }

    pub fn is_struct_value(self) -> bool {
        matches!(self, BasicValueEnum::StructValue(_))
    }

    pub fn is_vector_value(self) -> bool {
        matches!(self, BasicValueEnum::VectorValue(_))
    }

    pub fn into_array_value(self) -> ArrayValue<'ctx> {
        if let BasicValueEnum::ArrayValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the ArrayValue variant", self)
        }
    }

    pub fn into_int_value(self) -> IntValue<'ctx> {
        if let BasicValueEnum::IntValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the IntValue variant", self)
        }
    }

    pub fn into_float_value(self) -> FloatValue<'ctx> {
        if let BasicValueEnum::FloatValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the FloatValue variant", self)
        }
    }

    pub fn into_pointer_value(self) -> PointerValue<'ctx> {
        if let BasicValueEnum::PointerValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected PointerValue variant", self)
        }
    }

    pub fn into_struct_value(self) -> StructValue<'ctx> {
        if let BasicValueEnum::StructValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the StructValue variant", self)
        }
    }

    pub fn into_vector_value(self) -> VectorValue<'ctx> {
        if let BasicValueEnum::VectorValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the VectorValue variant", self)
        }
    }
}

impl<'ctx> AggregateValueEnum<'ctx> {
    pub(crate) unsafe fn new(value: LLVMValueRef) -> Self {
        match LLVMGetTypeKind(LLVMTypeOf(value)) {
            LLVMTypeKind::LLVMArrayTypeKind => AggregateValueEnum::ArrayValue(ArrayValue::new(value)),
            LLVMTypeKind::LLVMStructTypeKind => AggregateValueEnum::StructValue(StructValue::new(value)),
            _ => unreachable!("The given type is not an aggregate type."),
        }
    }

    pub fn is_array_value(self) -> bool {
        matches!(self, AggregateValueEnum::ArrayValue(_))
    }

    pub fn is_struct_value(self) -> bool {
        matches!(self, AggregateValueEnum::StructValue(_))
    }

    pub fn into_array_value(self) -> ArrayValue<'ctx> {
        if let AggregateValueEnum::ArrayValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the ArrayValue variant", self)
        }
    }

    pub fn into_struct_value(self) -> StructValue<'ctx> {
        if let AggregateValueEnum::StructValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the StructValue variant", self)
        }
    }
}

impl<'ctx> BasicMetadataValueEnum<'ctx> {
    pub(crate) unsafe fn new(value: LLVMValueRef) -> Self {
        match LLVMGetTypeKind(LLVMTypeOf(value)) {
            LLVMTypeKind::LLVMFloatTypeKind
            | LLVMTypeKind::LLVMFP128TypeKind
            | LLVMTypeKind::LLVMDoubleTypeKind
            | LLVMTypeKind::LLVMHalfTypeKind
            | LLVMTypeKind::LLVMX86_FP80TypeKind
            | LLVMTypeKind::LLVMPPC_FP128TypeKind => BasicMetadataValueEnum::FloatValue(FloatValue::new(value)),
            LLVMTypeKind::LLVMIntegerTypeKind => BasicMetadataValueEnum::IntValue(IntValue::new(value)),
            LLVMTypeKind::LLVMStructTypeKind => BasicMetadataValueEnum::StructValue(StructValue::new(value)),
            LLVMTypeKind::LLVMPointerTypeKind => BasicMetadataValueEnum::PointerValue(PointerValue::new(value)),
            LLVMTypeKind::LLVMArrayTypeKind => BasicMetadataValueEnum::ArrayValue(ArrayValue::new(value)),
            LLVMTypeKind::LLVMVectorTypeKind => BasicMetadataValueEnum::VectorValue(VectorValue::new(value)),
            LLVMTypeKind::LLVMMetadataTypeKind => BasicMetadataValueEnum::MetadataValue(MetadataValue::new(value)),
            _ => unreachable!("Unsupported type"),
        }
    }

    pub fn is_array_value(self) -> bool {
        matches!(self, BasicMetadataValueEnum::ArrayValue(_))
    }

    pub fn is_int_value(self) -> bool {
        matches!(self, BasicMetadataValueEnum::IntValue(_))
    }

    pub fn is_float_value(self) -> bool {
        matches!(self, BasicMetadataValueEnum::FloatValue(_))
    }

    pub fn is_pointer_value(self) -> bool {
        matches!(self, BasicMetadataValueEnum::PointerValue(_))
    }

    pub fn is_struct_value(self) -> bool {
        matches!(self, BasicMetadataValueEnum::StructValue(_))
    }

    pub fn is_vector_value(self) -> bool {
        matches!(self, BasicMetadataValueEnum::VectorValue(_))
    }

    pub fn is_metadata_value(self) -> bool {
        matches!(self, BasicMetadataValueEnum::MetadataValue(_))
    }

    pub fn into_array_value(self) -> ArrayValue<'ctx> {
        if let BasicMetadataValueEnum::ArrayValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the ArrayValue variant", self)
        }
    }

    pub fn into_int_value(self) -> IntValue<'ctx> {
        if let BasicMetadataValueEnum::IntValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the IntValue variant", self)
        }
    }

    pub fn into_float_value(self) -> FloatValue<'ctx> {
        if let BasicMetadataValueEnum::FloatValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected FloatValue variant", self)
        }
    }

    pub fn into_pointer_value(self) -> PointerValue<'ctx> {
        if let BasicMetadataValueEnum::PointerValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the PointerValue variant", self)
        }
    }

    pub fn into_struct_value(self) -> StructValue<'ctx> {
        if let BasicMetadataValueEnum::StructValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the StructValue variant", self)
        }
    }

    pub fn into_vector_value(self) -> VectorValue<'ctx> {
        if let BasicMetadataValueEnum::VectorValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected the VectorValue variant", self)
        }
    }

    pub fn into_metadata_value(self) -> MetadataValue<'ctx> {
        if let BasicMetadataValueEnum::MetadataValue(v) = self {
            v
        } else {
            panic!("Found {:?} but expected MetaData variant", self)
        }
    }
}

impl<'ctx> From<BasicValueEnum<'ctx>> for AnyValueEnum<'ctx> {
    fn from(value: BasicValueEnum<'ctx>) -> Self {
        unsafe { AnyValueEnum::new(value.as_value_ref()) }
    }
}

impl<'ctx> From<BasicValueEnum<'ctx>> for BasicMetadataValueEnum<'ctx> {
    fn from(value: BasicValueEnum<'ctx>) -> Self {
        unsafe { BasicMetadataValueEnum::new(value.as_value_ref()) }
    }
}

impl<'ctx> TryFrom<AnyValueEnum<'ctx>> for BasicValueEnum<'ctx> {
    type Error = ();

    fn try_from(value: AnyValueEnum<'ctx>) -> Result<Self, Self::Error> {
        use AnyValueEnum::*;
        Ok(match value {
            ArrayValue(av) => av.into(),
            IntValue(iv) => iv.into(),
            FloatValue(fv) => fv.into(),
            PointerValue(pv) => pv.into(),
            StructValue(sv) => sv.into(),
            VectorValue(vv) => vv.into(),
            MetadataValue(_) | PhiValue(_) | FunctionValue(_) | InstructionValue(_) => return Err(()),
        })
    }
}

impl<'ctx> TryFrom<AnyValueEnum<'ctx>> for BasicMetadataValueEnum<'ctx> {
    type Error = ();

    fn try_from(value: AnyValueEnum<'ctx>) -> Result<Self, Self::Error> {
        use AnyValueEnum::*;
        Ok(match value {
            ArrayValue(av) => av.into(),
            IntValue(iv) => iv.into(),
            FloatValue(fv) => fv.into(),
            PointerValue(pv) => pv.into(),
            StructValue(sv) => sv.into(),
            VectorValue(vv) => vv.into(),
            MetadataValue(mv) => mv.into(),
            PhiValue(_) | FunctionValue(_) | InstructionValue(_) => return Err(()),
        })
    }
}

impl<'ctx> TryFrom<BasicMetadataValueEnum<'ctx>> for BasicValueEnum<'ctx> {
    type Error = ();

    fn try_from(value: BasicMetadataValueEnum<'ctx>) -> Result<Self, Self::Error> {
        use BasicMetadataValueEnum::*;
        Ok(match value {
            ArrayValue(av) => av.into(),
            IntValue(iv) => iv.into(),
            FloatValue(fv) => fv.into(),
            PointerValue(pv) => pv.into(),
            StructValue(sv) => sv.into(),
            VectorValue(vv) => vv.into(),
            MetadataValue(_) => return Err(()),
        })
    }
}

impl Display for AggregateValueEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

impl Display for AnyValueEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

impl Display for BasicValueEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

impl Display for BasicMetadataValueEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
