#[deny(missing_docs)]
use llvm_lib::core::LLVMGetTypeKind;
use llvm_lib::core::{
    LLVMAlignOf, LLVMArrayType, LLVMConstNull, LLVMConstPointerNull, LLVMFunctionType, LLVMGetElementType,
    LLVMGetTypeContext, LLVMGetTypeKind, LLVMGetUndef, LLVMPointerType, LLVMPrintTypeToString, LLVMSizeOf,
    LLVMTypeIsSized, LLVMVectorType,
};
use llvm_lib::core::{
    LLVMConstAllOnes, LLVMConstArray, LLVMConstInt, LLVMConstIntOfArbitraryPrecision, LLVMConstIntOfStringAndSize,
    LLVMGetIntTypeWidth,
};
use llvm_lib::core::{
    LLVMConstArray, LLVMConstNamedStruct, LLVMCountStructElementTypes, LLVMGetStructElementTypes, LLVMGetStructName,
    LLVMIsOpaqueStruct, LLVMIsPackedStruct, LLVMStructGetTypeAtIndex, LLVMStructSetBody,
};
use llvm_lib::core::{LLVMConstArray, LLVMConstReal, LLVMConstRealOfStringAndSize};
use llvm_lib::core::{LLVMConstArray, LLVMConstVector, LLVMGetVectorSize};
use llvm_lib::core::{LLVMConstArray, LLVMGetArrayLength};
use llvm_lib::core::{LLVMConstArray, LLVMGetPointerAddressSpace};
use llvm_lib::core::{
    LLVMCountParamTypes, LLVMGetParamTypes, LLVMGetReturnType, LLVMGetTypeKind, LLVMIsFunctionVarArg,
};
use llvm_lib::execution_engine::LLVMCreateGenericValueOfFloat;
use llvm_lib::execution_engine::LLVMCreateGenericValueOfInt;
use llvm_lib::prelude::LLVMTypeRef;
use llvm_lib::prelude::LLVMTypeRef;
use llvm_lib::prelude::LLVMTypeRef;
use llvm_lib::prelude::LLVMTypeRef;
use llvm_lib::prelude::LLVMTypeRef;
use llvm_lib::prelude::{LLVMTypeRef, LLVMValueRef};
use llvm_lib::prelude::{LLVMTypeRef, LLVMValueRef};
use llvm_lib::prelude::{LLVMTypeRef, LLVMValueRef};
use llvm_lib::prelude::{LLVMTypeRef, LLVMValueRef};
use llvm_lib::prelude::{LLVMTypeRef, LLVMValueRef};
use llvm_lib::prelude::{LLVMTypeRef, LLVMValueRef};
use llvm_lib::prelude::{LLVMTypeRef, LLVMValueRef};
use llvm_lib::LLVMTypeKind;
use llvm_lib::LLVMTypeKind;
use llvm_lib::LLVMTypeKind;

use crate::context::ContextRef;
use crate::support::LLVMString;
use crate::types::enums::BasicMetadataTypeEnum;
use crate::types::enums::{AnyTypeEnum, BasicMetadataTypeEnum, BasicTypeEnum};
use crate::types::traits::AsTypeRef;
use crate::types::AnyTypeEnum;
use crate::types::MetadataType;
use crate::types::{traits::AsTypeRef, ArrayType, BasicTypeEnum, FunctionType, PointerType, Type};
use crate::types::{AnyType, BasicTypeEnum, PointerType, Type};
use crate::types::{ArrayType, BasicTypeEnum, FunctionType, PointerType, Type};
use crate::types::{ArrayType, FloatType, FunctionType, IntType, PointerType, StructType, Type, VectorType, VoidType};
use crate::types::{ArrayType, FloatType, FunctionType, IntType, PointerType, StructType, VectorType, VoidType};
use crate::types::{ArrayType, FunctionType, PointerType, Type, VectorType};
use crate::types::{BasicTypeEnum, FunctionType, PointerType, Type};
use crate::types::{FunctionType, Type};
use crate::values::IntValue;
use crate::values::{ArrayValue, AsValueRef, BasicValue, IntValue, VectorValue};
use crate::values::{ArrayValue, AsValueRef, BasicValueEnum, IntValue, StructValue};
use crate::values::{ArrayValue, AsValueRef, FloatValue, GenericValue, IntValue};
use crate::values::{ArrayValue, AsValueRef, GenericValue, IntValue};
use crate::values::{ArrayValue, AsValueRef, IntValue};
use crate::values::{ArrayValue, AsValueRef, IntValue, PointerValue};
use crate::values::{BasicValue, BasicValueEnum, IntValue};
use crate::values::{FloatMathValue, FloatValue, IntMathValue, IntValue, PointerMathValue, PointerValue, VectorValue};
use crate::AddressSpace;
use static_alloc::Bump;
use std::convert::TryFrom;
use std::ffi::CStr;
use std::fmt;
use std::fmt::Debug;
use std::fmt::{self, Display};
use std::marker::PhantomData;
use std::mem::forget;

pub use crate::types::array_type::ArrayType;
pub use crate::types::enums::{AnyTypeEnum, BasicMetadataTypeEnum, BasicTypeEnum};
pub use crate::types::float_type::FloatType;
pub use crate::types::fn_type::FunctionType;
pub use crate::types::int_type::{IntType, StringRadix};
pub use crate::types::metadata_type::MetadataType;
pub use crate::types::ptr_type::PointerType;
pub use crate::types::struct_type::StructType;
pub use crate::types::traits::{AnyType, AsTypeRef, BasicType, FloatMathType, IntMathType, PointerMathType};
pub use crate::types::vec_type::VectorType;
pub use crate::types::void_type::VoidType;

#[derive(PartialEq, Eq, Clone, Copy)]
struct Type<'ctx> {
    ty: LLVMTypeRef,
    _marker: PhantomData<&'ctx ()>,
}
impl<'ctx> Type<'ctx> {
    unsafe fn new(ty: LLVMTypeRef) -> Self {
        assert!(!ty.is_null());
        Type {
            ty,
            _marker: PhantomData,
        }
    }
    fn const_zero(self) -> LLVMValueRef {
        unsafe {
            match LLVMGetTypeKind(self.ty) {
                LLVMTypeKind::LLVMMetadataTypeKind => LLVMConstPointerNull(self.ty),
                _ => LLVMConstNull(self.ty),
            }
        }
    }
    fn ptr_type(self, address_space: AddressSpace) -> PointerType<'ctx> {
        unsafe { PointerType::new(LLVMPointerType(self.ty, address_space.0)) }
    }
    fn vec_type(self, size: u32) -> VectorType<'ctx> {
        assert!(size != 0, "Vectors of size zero are not allowed.");
        unsafe { VectorType::new(LLVMVectorType(self.ty, size)) }
    }
    #[cfg(not(feature = "experimental"))]
    fn fn_type(self, param_types: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        let mut param_types: Vec<LLVMTypeRef> = param_types.iter().map(|val| val.as_type_ref()).collect();
        unsafe {
            FunctionType::new(LLVMFunctionType(
                self.ty,
                param_types.as_mut_ptr(),
                param_types.len() as u32,
                is_var_args as i32,
            ))
        }
    }
    #[cfg(feature = "experimental")]
    fn fn_type(self, param_types: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        let pool: Bump<[usize; 16]> = Bump::uninit();
        let mut pool_start = None;
        for (i, param_type) in param_types.iter().enumerate() {
            let addr = pool.leak(param_type.as_type_ref()).expect("Found more than 16 params");
            if i == 0 {
                pool_start = Some(addr as *mut _);
            }
        }
        unsafe {
            FunctionType::new(LLVMFunctionType(
                self.ty,
                pool_start.unwrap_or(std::ptr::null_mut()),
                param_types.len() as u32,
                is_var_args as i32,
            ))
        }
    }
    fn array_type(self, size: u32) -> ArrayType<'ctx> {
        unsafe { ArrayType::new(LLVMArrayType(self.ty, size)) }
    }
    fn get_undef(self) -> LLVMValueRef {
        unsafe { LLVMGetUndef(self.ty) }
    }
    fn get_alignment(self) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMAlignOf(self.ty)) }
    }
    fn get_context(self) -> ContextRef<'ctx> {
        unsafe { ContextRef::new(LLVMGetTypeContext(self.ty)) }
    }
    fn is_sized(self) -> bool {
        unsafe { LLVMTypeIsSized(self.ty) == 1 }
    }
    fn size_of(self) -> Option<IntValue<'ctx>> {
        if !self.is_sized() {
            return None;
        }
        unsafe { Some(IntValue::new(LLVMSizeOf(self.ty))) }
    }
    fn print_to_string(self) -> LLVMString {
        unsafe { LLVMString::new(LLVMPrintTypeToString(self.ty)) }
    }
    pub fn get_element_type(self) -> AnyTypeEnum<'ctx> {
        unsafe { AnyTypeEnum::new(LLVMGetElementType(self.ty)) }
    }
}
impl fmt::Debug for Type<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let llvm_type = self.print_to_string();
        f.debug_struct("Type")
            .field("address", &self.ty)
            .field("llvm_type", &llvm_type)
            .finish()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct ArrayType<'ctx> {
    array_type: Type<'ctx>,
}
impl<'ctx> ArrayType<'ctx> {
    pub unsafe fn new(array_type: LLVMTypeRef) -> Self {
        assert!(!array_type.is_null());
        ArrayType {
            array_type: Type::new(array_type),
        }
    }
    pub fn size_of(self) -> Option<IntValue<'ctx>> {
        self.array_type.size_of()
    }
    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.array_type.get_alignment()
    }
    pub fn ptr_type(self, address_space: AddressSpace) -> PointerType<'ctx> {
        self.array_type.ptr_type(address_space)
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        self.array_type.get_context()
    }
    pub fn fn_type(self, param_types: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.array_type.fn_type(param_types, is_var_args)
    }
    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.array_type.array_type(size)
    }
    pub fn const_array(self, values: &[ArrayValue<'ctx>]) -> ArrayValue<'ctx> {
        let mut values: Vec<LLVMValueRef> = values.iter().map(|val| val.as_value_ref()).collect();
        unsafe {
            ArrayValue::new(LLVMConstArray(
                self.as_type_ref(),
                values.as_mut_ptr(),
                values.len() as u32,
            ))
        }
    }
    pub fn const_zero(self) -> ArrayValue<'ctx> {
        unsafe { ArrayValue::new(self.array_type.const_zero()) }
    }
    pub fn len(self) -> u32 {
        unsafe { LLVMGetArrayLength(self.as_type_ref()) }
    }
    pub fn print_to_string(self) -> LLVMString {
        self.array_type.print_to_string()
    }
    pub fn get_undef(self) -> ArrayValue<'ctx> {
        unsafe { ArrayValue::new(self.array_type.get_undef()) }
    }
    pub fn get_element_type(self) -> BasicTypeEnum<'ctx> {
        self.array_type.get_element_type().to_basic_type_enum()
    }
}
unsafe impl AsTypeRef for ArrayType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.array_type.ty
    }
}
impl Display for ArrayType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

macro_rules! enum_type_set {
    ($(#[$enum_attrs:meta])* $enum_name:ident: { $($(#[$variant_attrs:meta])* $args:ident,)+ }) => (
        #[derive(Debug, PartialEq, Eq, Clone, Copy)]
        $(#[$enum_attrs])*
        pub enum $enum_name<'ctx> {
            $(
                $(#[$variant_attrs])*
                $args($args<'ctx>),
            )*
        }
        unsafe impl AsTypeRef for $enum_name<'_> {
            fn as_type_ref(&self) -> LLVMTypeRef {
                match *self {
                    $(
                        $enum_name::$args(ref t) => t.as_type_ref(),
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

enum_type_set! {
    AnyTypeEnum: {
        ArrayType,
        FloatType,
        FunctionType,
        IntType,
        PointerType,
        StructType,
        VectorType,
        VoidType,
    }
}
enum_type_set! {
    BasicTypeEnum: {
        ArrayType,
        FloatType,
        IntType,
        PointerType,
        StructType,
        VectorType,
    }
}
enum_type_set! {
    BasicMetadataTypeEnum: {
        ArrayType,
        FloatType,
        IntType,
        PointerType,
        StructType,
        VectorType,
        MetadataType,
    }
}

impl<'ctx> BasicMetadataTypeEnum<'ctx> {
    pub fn into_array_type(self) -> ArrayType<'ctx> {
        if let BasicMetadataTypeEnum::ArrayType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected another variant", self);
        }
    }
    pub fn into_float_type(self) -> FloatType<'ctx> {
        if let BasicMetadataTypeEnum::FloatType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected another variant", self);
        }
    }
    pub fn into_int_type(self) -> IntType<'ctx> {
        if let BasicMetadataTypeEnum::IntType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected another variant", self);
        }
    }
    pub fn into_pointer_type(self) -> PointerType<'ctx> {
        if let BasicMetadataTypeEnum::PointerType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected another variant", self);
        }
    }
    pub fn into_struct_type(self) -> StructType<'ctx> {
        if let BasicMetadataTypeEnum::StructType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected another variant", self);
        }
    }
    pub fn into_vector_type(self) -> VectorType<'ctx> {
        if let BasicMetadataTypeEnum::VectorType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected another variant", self);
        }
    }
    pub fn into_metadata_type(self) -> MetadataType<'ctx> {
        if let BasicMetadataTypeEnum::MetadataType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected another variant", self);
        }
    }
    pub fn is_array_type(self) -> bool {
        matches!(self, BasicMetadataTypeEnum::ArrayType(_))
    }
    pub fn is_float_type(self) -> bool {
        matches!(self, BasicMetadataTypeEnum::FloatType(_))
    }
    pub fn is_int_type(self) -> bool {
        matches!(self, BasicMetadataTypeEnum::IntType(_))
    }
    pub fn is_metadata_type(self) -> bool {
        matches!(self, BasicMetadataTypeEnum::MetadataType(_))
    }
    pub fn is_pointer_type(self) -> bool {
        matches!(self, BasicMetadataTypeEnum::PointerType(_))
    }
    pub fn is_struct_type(self) -> bool {
        matches!(self, BasicMetadataTypeEnum::StructType(_))
    }
    pub fn is_vector_type(self) -> bool {
        matches!(self, BasicMetadataTypeEnum::VectorType(_))
    }
    pub fn print_to_string(self) -> LLVMString {
        match self {
            BasicMetadataTypeEnum::ArrayType(t) => t.print_to_string(),
            BasicMetadataTypeEnum::IntType(t) => t.print_to_string(),
            BasicMetadataTypeEnum::FloatType(t) => t.print_to_string(),
            BasicMetadataTypeEnum::PointerType(t) => t.print_to_string(),
            BasicMetadataTypeEnum::StructType(t) => t.print_to_string(),
            BasicMetadataTypeEnum::VectorType(t) => t.print_to_string(),
            BasicMetadataTypeEnum::MetadataType(t) => t.print_to_string(),
        }
    }
}

impl<'ctx> AnyTypeEnum<'ctx> {
    pub unsafe fn new(type_: LLVMTypeRef) -> Self {
        match LLVMGetTypeKind(type_) {
            LLVMTypeKind::LLVMVoidTypeKind => AnyTypeEnum::VoidType(VoidType::new(type_)),
            LLVMTypeKind::LLVMHalfTypeKind
            | LLVMTypeKind::LLVMFloatTypeKind
            | LLVMTypeKind::LLVMDoubleTypeKind
            | LLVMTypeKind::LLVMX86_FP80TypeKind
            | LLVMTypeKind::LLVMFP128TypeKind
            | LLVMTypeKind::LLVMPPC_FP128TypeKind => AnyTypeEnum::FloatType(FloatType::new(type_)),
            LLVMTypeKind::LLVMBFloatTypeKind => AnyTypeEnum::FloatType(FloatType::new(type_)),
            LLVMTypeKind::LLVMLabelTypeKind => panic!("FIXME: Unsupported type: Label"),
            LLVMTypeKind::LLVMIntegerTypeKind => AnyTypeEnum::IntType(IntType::new(type_)),
            LLVMTypeKind::LLVMFunctionTypeKind => AnyTypeEnum::FunctionType(FunctionType::new(type_)),
            LLVMTypeKind::LLVMStructTypeKind => AnyTypeEnum::StructType(StructType::new(type_)),
            LLVMTypeKind::LLVMArrayTypeKind => AnyTypeEnum::ArrayType(ArrayType::new(type_)),
            LLVMTypeKind::LLVMPointerTypeKind => AnyTypeEnum::PointerType(PointerType::new(type_)),
            LLVMTypeKind::LLVMVectorTypeKind => AnyTypeEnum::VectorType(VectorType::new(type_)),
            LLVMTypeKind::LLVMScalableVectorTypeKind => AnyTypeEnum::VectorType(VectorType::new(type_)),
            LLVMTypeKind::LLVMMetadataTypeKind => panic!("Metadata type is not supported as AnyType."),
            LLVMTypeKind::LLVMX86_MMXTypeKind => panic!("FIXME: Unsupported type: MMX"),
            LLVMTypeKind::LLVMX86_AMXTypeKind => panic!("FIXME: Unsupported type: AMX"),
            LLVMTypeKind::LLVMTokenTypeKind => panic!("FIXME: Unsupported type: Token"),
            LLVMTypeKind::LLVMTargetExtTypeKind => panic!("FIXME: Unsupported type: TargetExt"),
        }
    }
    pub(crate) fn to_basic_type_enum(&self) -> BasicTypeEnum<'ctx> {
        unsafe { BasicTypeEnum::new(self.as_type_ref()) }
    }
    pub fn into_array_type(self) -> ArrayType<'ctx> {
        if let AnyTypeEnum::ArrayType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected the ArrayType variant", self);
        }
    }
    pub fn into_float_type(self) -> FloatType<'ctx> {
        if let AnyTypeEnum::FloatType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected the FloatType variant", self);
        }
    }
    pub fn into_function_type(self) -> FunctionType<'ctx> {
        if let AnyTypeEnum::FunctionType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected the FunctionType variant", self);
        }
    }
    pub fn into_int_type(self) -> IntType<'ctx> {
        if let AnyTypeEnum::IntType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected the IntType variant", self);
        }
    }
    pub fn into_pointer_type(self) -> PointerType<'ctx> {
        if let AnyTypeEnum::PointerType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected the PointerType variant", self);
        }
    }
    pub fn into_struct_type(self) -> StructType<'ctx> {
        if let AnyTypeEnum::StructType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected the StructType variant", self);
        }
    }
    pub fn into_vector_type(self) -> VectorType<'ctx> {
        if let AnyTypeEnum::VectorType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected the VectorType variant", self);
        }
    }
    pub fn into_void_type(self) -> VoidType<'ctx> {
        if let AnyTypeEnum::VoidType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected the VoidType variant", self);
        }
    }
    pub fn is_array_type(self) -> bool {
        matches!(self, AnyTypeEnum::ArrayType(_))
    }
    pub fn is_float_type(self) -> bool {
        matches!(self, AnyTypeEnum::FloatType(_))
    }
    pub fn is_function_type(self) -> bool {
        matches!(self, AnyTypeEnum::FunctionType(_))
    }
    pub fn is_int_type(self) -> bool {
        matches!(self, AnyTypeEnum::IntType(_))
    }
    pub fn is_pointer_type(self) -> bool {
        matches!(self, AnyTypeEnum::PointerType(_))
    }
    pub fn is_struct_type(self) -> bool {
        matches!(self, AnyTypeEnum::StructType(_))
    }
    pub fn is_vector_type(self) -> bool {
        matches!(self, AnyTypeEnum::VectorType(_))
    }
    pub fn is_void_type(self) -> bool {
        matches!(self, AnyTypeEnum::VoidType(_))
    }
    pub fn size_of(&self) -> Option<IntValue<'ctx>> {
        match self {
            AnyTypeEnum::ArrayType(t) => t.size_of(),
            AnyTypeEnum::FloatType(t) => Some(t.size_of()),
            AnyTypeEnum::IntType(t) => Some(t.size_of()),
            AnyTypeEnum::PointerType(t) => Some(t.size_of()),
            AnyTypeEnum::StructType(t) => t.size_of(),
            AnyTypeEnum::VectorType(t) => t.size_of(),
            AnyTypeEnum::VoidType(_) => None,
            AnyTypeEnum::FunctionType(_) => None,
        }
    }
    pub fn print_to_string(self) -> LLVMString {
        match self {
            AnyTypeEnum::ArrayType(t) => t.print_to_string(),
            AnyTypeEnum::FloatType(t) => t.print_to_string(),
            AnyTypeEnum::IntType(t) => t.print_to_string(),
            AnyTypeEnum::PointerType(t) => t.print_to_string(),
            AnyTypeEnum::StructType(t) => t.print_to_string(),
            AnyTypeEnum::VectorType(t) => t.print_to_string(),
            AnyTypeEnum::VoidType(t) => t.print_to_string(),
            AnyTypeEnum::FunctionType(t) => t.print_to_string(),
        }
    }
}
impl<'ctx> BasicTypeEnum<'ctx> {
    pub unsafe fn new(type_: LLVMTypeRef) -> Self {
        match LLVMGetTypeKind(type_) {
            LLVMTypeKind::LLVMHalfTypeKind
            | LLVMTypeKind::LLVMFloatTypeKind
            | LLVMTypeKind::LLVMDoubleTypeKind
            | LLVMTypeKind::LLVMX86_FP80TypeKind
            | LLVMTypeKind::LLVMFP128TypeKind
            | LLVMTypeKind::LLVMPPC_FP128TypeKind => BasicTypeEnum::FloatType(FloatType::new(type_)),
            LLVMTypeKind::LLVMBFloatTypeKind => BasicTypeEnum::FloatType(FloatType::new(type_)),
            LLVMTypeKind::LLVMIntegerTypeKind => BasicTypeEnum::IntType(IntType::new(type_)),
            LLVMTypeKind::LLVMStructTypeKind => BasicTypeEnum::StructType(StructType::new(type_)),
            LLVMTypeKind::LLVMPointerTypeKind => BasicTypeEnum::PointerType(PointerType::new(type_)),
            LLVMTypeKind::LLVMArrayTypeKind => BasicTypeEnum::ArrayType(ArrayType::new(type_)),
            LLVMTypeKind::LLVMVectorTypeKind => BasicTypeEnum::VectorType(VectorType::new(type_)),
            LLVMTypeKind::LLVMScalableVectorTypeKind => BasicTypeEnum::VectorType(VectorType::new(type_)),
            LLVMTypeKind::LLVMMetadataTypeKind => panic!("Unsupported basic type: Metadata"),
            LLVMTypeKind::LLVMX86_MMXTypeKind => panic!("Unsupported basic type: MMX"),
            LLVMTypeKind::LLVMX86_AMXTypeKind => unreachable!("Unsupported basic type: AMX"),
            LLVMTypeKind::LLVMLabelTypeKind => unreachable!("Unsupported basic type: Label"),
            LLVMTypeKind::LLVMVoidTypeKind => unreachable!("Unsupported basic type: VoidType"),
            LLVMTypeKind::LLVMFunctionTypeKind => unreachable!("Unsupported basic type: FunctionType"),
            LLVMTypeKind::LLVMTokenTypeKind => unreachable!("Unsupported basic type: Token"),
            LLVMTypeKind::LLVMTargetExtTypeKind => unreachable!("Unsupported basic type: TargetExt"),
        }
    }
    pub fn into_array_type(self) -> ArrayType<'ctx> {
        if let BasicTypeEnum::ArrayType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected the ArrayType variant", self);
        }
    }
    pub fn into_float_type(self) -> FloatType<'ctx> {
        if let BasicTypeEnum::FloatType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected the FloatType variant", self);
        }
    }
    pub fn into_int_type(self) -> IntType<'ctx> {
        if let BasicTypeEnum::IntType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected the IntType variant", self);
        }
    }
    pub fn into_pointer_type(self) -> PointerType<'ctx> {
        if let BasicTypeEnum::PointerType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected the PointerType variant", self);
        }
    }
    pub fn into_struct_type(self) -> StructType<'ctx> {
        if let BasicTypeEnum::StructType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected the StructType variant", self);
        }
    }
    pub fn into_vector_type(self) -> VectorType<'ctx> {
        if let BasicTypeEnum::VectorType(t) = self {
            t
        } else {
            panic!("Found {:?} but expected the VectorType variant", self);
        }
    }
    pub fn is_array_type(self) -> bool {
        matches!(self, BasicTypeEnum::ArrayType(_))
    }
    pub fn is_float_type(self) -> bool {
        matches!(self, BasicTypeEnum::FloatType(_))
    }
    pub fn is_int_type(self) -> bool {
        matches!(self, BasicTypeEnum::IntType(_))
    }
    pub fn is_pointer_type(self) -> bool {
        matches!(self, BasicTypeEnum::PointerType(_))
    }
    pub fn is_struct_type(self) -> bool {
        matches!(self, BasicTypeEnum::StructType(_))
    }
    pub fn is_vector_type(self) -> bool {
        matches!(self, BasicTypeEnum::VectorType(_))
    }
    pub fn const_zero(self) -> BasicValueEnum<'ctx> {
        match self {
            BasicTypeEnum::ArrayType(ty) => ty.const_zero().as_basic_value_enum(),
            BasicTypeEnum::FloatType(ty) => ty.const_zero().as_basic_value_enum(),
            BasicTypeEnum::IntType(ty) => ty.const_zero().as_basic_value_enum(),
            BasicTypeEnum::PointerType(ty) => ty.const_zero().as_basic_value_enum(),
            BasicTypeEnum::StructType(ty) => ty.const_zero().as_basic_value_enum(),
            BasicTypeEnum::VectorType(ty) => ty.const_zero().as_basic_value_enum(),
        }
    }
    pub fn print_to_string(self) -> LLVMString {
        match self {
            BasicTypeEnum::ArrayType(t) => t.print_to_string(),
            BasicTypeEnum::FloatType(t) => t.print_to_string(),
            BasicTypeEnum::IntType(t) => t.print_to_string(),
            BasicTypeEnum::PointerType(t) => t.print_to_string(),
            BasicTypeEnum::StructType(t) => t.print_to_string(),
            BasicTypeEnum::VectorType(t) => t.print_to_string(),
        }
    }
}
impl<'ctx> TryFrom<AnyTypeEnum<'ctx>> for BasicTypeEnum<'ctx> {
    type Error = ();
    fn try_from(value: AnyTypeEnum<'ctx>) -> Result<Self, Self::Error> {
        use AnyTypeEnum::*;
        Ok(match value {
            ArrayType(at) => at.into(),
            FloatType(ft) => ft.into(),
            IntType(it) => it.into(),
            PointerType(pt) => pt.into(),
            StructType(st) => st.into(),
            VectorType(vt) => vt.into(),
            VoidType(_) | FunctionType(_) => return Err(()),
        })
    }
}
impl<'ctx> TryFrom<AnyTypeEnum<'ctx>> for BasicMetadataTypeEnum<'ctx> {
    type Error = ();
    fn try_from(value: AnyTypeEnum<'ctx>) -> Result<Self, Self::Error> {
        use AnyTypeEnum::*;
        Ok(match value {
            ArrayType(at) => at.into(),
            FloatType(ft) => ft.into(),
            IntType(it) => it.into(),
            PointerType(pt) => pt.into(),
            StructType(st) => st.into(),
            VectorType(vt) => vt.into(),
            VoidType(_) | FunctionType(_) => return Err(()),
        })
    }
}
impl<'ctx> TryFrom<BasicMetadataTypeEnum<'ctx>> for BasicTypeEnum<'ctx> {
    type Error = ();
    fn try_from(value: BasicMetadataTypeEnum<'ctx>) -> Result<Self, Self::Error> {
        use BasicMetadataTypeEnum::*;
        Ok(match value {
            ArrayType(at) => at.into(),
            FloatType(ft) => ft.into(),
            IntType(it) => it.into(),
            PointerType(pt) => pt.into(),
            StructType(st) => st.into(),
            VectorType(vt) => vt.into(),
            MetadataType(_) => return Err(()),
        })
    }
}
impl<'ctx> From<BasicTypeEnum<'ctx>> for BasicMetadataTypeEnum<'ctx> {
    fn from(value: BasicTypeEnum<'ctx>) -> Self {
        use BasicTypeEnum::*;
        match value {
            ArrayType(at) => at.into(),
            FloatType(ft) => ft.into(),
            IntType(it) => it.into(),
            PointerType(pt) => pt.into(),
            StructType(st) => st.into(),
            VectorType(vt) => vt.into(),
        }
    }
}
impl Display for AnyTypeEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
impl Display for BasicTypeEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
impl Display for BasicMetadataTypeEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct FloatType<'ctx> {
    float_type: Type<'ctx>,
}
impl<'ctx> FloatType<'ctx> {
    pub unsafe fn new(float_type: LLVMTypeRef) -> Self {
        assert!(!float_type.is_null());
        FloatType {
            float_type: Type::new(float_type),
        }
    }
    pub fn fn_type(self, param_types: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.float_type.fn_type(param_types, is_var_args)
    }
    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.float_type.array_type(size)
    }
    pub fn vec_type(self, size: u32) -> VectorType<'ctx> {
        self.float_type.vec_type(size)
    }
    pub fn const_float(self, value: f64) -> FloatValue<'ctx> {
        unsafe { FloatValue::new(LLVMConstReal(self.float_type.ty, value)) }
    }
    pub fn const_float_from_string(self, slice: &str) -> FloatValue<'ctx> {
        unsafe {
            FloatValue::new(LLVMConstRealOfStringAndSize(
                self.as_type_ref(),
                slice.as_ptr() as *const ::libc::c_char,
                slice.len() as u32,
            ))
        }
    }
    pub fn const_zero(self) -> FloatValue<'ctx> {
        unsafe { FloatValue::new(self.float_type.const_zero()) }
    }
    pub fn size_of(self) -> IntValue<'ctx> {
        self.float_type.size_of().unwrap()
    }
    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.float_type.get_alignment()
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        self.float_type.get_context()
    }
    pub fn ptr_type(self, address_space: AddressSpace) -> PointerType<'ctx> {
        self.float_type.ptr_type(address_space)
    }
    pub fn print_to_string(self) -> LLVMString {
        self.float_type.print_to_string()
    }
    pub fn get_undef(&self) -> FloatValue<'ctx> {
        unsafe { FloatValue::new(self.float_type.get_undef()) }
    }
    pub fn create_generic_value(self, value: f64) -> GenericValue<'ctx> {
        unsafe { GenericValue::new(LLVMCreateGenericValueOfFloat(self.as_type_ref(), value)) }
    }
    pub fn const_array(self, values: &[FloatValue<'ctx>]) -> ArrayValue<'ctx> {
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
unsafe impl AsTypeRef for FloatType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.float_type.ty
    }
}
impl Display for FloatType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct PointerType<'ctx> {
    ptr_type: Type<'ctx>,
}
impl<'ctx> PointerType<'ctx> {
    pub unsafe fn new(ptr_type: LLVMTypeRef) -> Self {
        assert!(!ptr_type.is_null());
        PointerType {
            ptr_type: Type::new(ptr_type),
        }
    }
    pub fn size_of(self) -> IntValue<'ctx> {
        self.ptr_type.size_of().unwrap()
    }
    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.ptr_type.get_alignment()
    }
    pub fn ptr_type(self, address_space: AddressSpace) -> PointerType<'ctx> {
        self.ptr_type.ptr_type(address_space)
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        self.ptr_type.get_context()
    }
    pub fn fn_type(self, param_types: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.ptr_type.fn_type(param_types, is_var_args)
    }
    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.ptr_type.array_type(size)
    }
    pub fn get_address_space(self) -> AddressSpace {
        let addr_space = unsafe { LLVMGetPointerAddressSpace(self.as_type_ref()) };
        AddressSpace(addr_space)
    }
    pub fn print_to_string(self) -> LLVMString {
        self.ptr_type.print_to_string()
    }
    pub fn const_null(self) -> PointerValue<'ctx> {
        unsafe { PointerValue::new(self.ptr_type.const_zero()) }
    }
    pub fn const_zero(self) -> PointerValue<'ctx> {
        unsafe { PointerValue::new(self.ptr_type.const_zero()) }
    }
    pub fn get_undef(self) -> PointerValue<'ctx> {
        unsafe { PointerValue::new(self.ptr_type.get_undef()) }
    }
    pub fn vec_type(self, size: u32) -> VectorType<'ctx> {
        self.ptr_type.vec_type(size)
    }
    #[llvm_versions(4.0..=14.0)]
    pub fn get_element_type(self) -> AnyTypeEnum<'ctx> {
        self.ptr_type.get_element_type()
    }
    pub fn const_array(self, values: &[PointerValue<'ctx>]) -> ArrayValue<'ctx> {
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
unsafe impl AsTypeRef for PointerType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.ptr_type.ty
    }
}
impl Display for PointerType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct VectorType<'ctx> {
    vec_type: Type<'ctx>,
}
impl<'ctx> VectorType<'ctx> {
    pub unsafe fn new(vector_type: LLVMTypeRef) -> Self {
        assert!(!vector_type.is_null());
        VectorType {
            vec_type: Type::new(vector_type),
        }
    }
    pub fn size_of(self) -> Option<IntValue<'ctx>> {
        self.vec_type.size_of()
    }
    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.vec_type.get_alignment()
    }
    pub fn get_size(self) -> u32 {
        unsafe { LLVMGetVectorSize(self.as_type_ref()) }
    }
    pub fn const_vector<V: BasicValue<'ctx>>(values: &[V]) -> VectorValue<'ctx> {
        let mut values: Vec<LLVMValueRef> = values.iter().map(|val| val.as_value_ref()).collect();
        unsafe { VectorValue::new(LLVMConstVector(values.as_mut_ptr(), values.len() as u32)) }
    }
    pub fn const_zero(self) -> VectorValue<'ctx> {
        unsafe { VectorValue::new(self.vec_type.const_zero()) }
    }
    pub fn print_to_string(self) -> LLVMString {
        self.vec_type.print_to_string()
    }
    pub fn get_undef(self) -> VectorValue<'ctx> {
        unsafe { VectorValue::new(self.vec_type.get_undef()) }
    }
    pub fn get_element_type(self) -> BasicTypeEnum<'ctx> {
        self.vec_type.get_element_type().to_basic_type_enum()
    }
    pub fn ptr_type(self, address_space: AddressSpace) -> PointerType<'ctx> {
        self.vec_type.ptr_type(address_space)
    }
    pub fn fn_type(self, param_types: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.vec_type.fn_type(param_types, is_var_args)
    }
    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.vec_type.array_type(size)
    }
    pub fn const_array(self, values: &[VectorValue<'ctx>]) -> ArrayValue<'ctx> {
        let mut values: Vec<LLVMValueRef> = values.iter().map(|val| val.as_value_ref()).collect();
        unsafe {
            ArrayValue::new(LLVMConstArray(
                self.as_type_ref(),
                values.as_mut_ptr(),
                values.len() as u32,
            ))
        }
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        self.vec_type.get_context()
    }
}
unsafe impl AsTypeRef for VectorType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.vec_type.ty
    }
}
impl Display for VectorType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

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
