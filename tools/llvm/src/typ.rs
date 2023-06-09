#[deny(missing_docs)]
use llvm_lib::core::*;
use llvm_lib::execution_engine::*;
use llvm_lib::prelude::*;
use llvm_lib::LLVMTypeKind;
use static_alloc::Bump;
use std::convert::TryFrom;
use std::ffi::CStr;
use std::fmt;
use std::fmt::Debug;
use std::fmt::{self, Display};
use std::marker::PhantomData;
use std::mem::forget;

use crate::ctx::ContextRef;
use crate::val::*;
use crate::AddressSpace;
use crate::LLVMString;

#[derive(PartialEq, Eq, Clone, Copy)]
struct Type<'ctx> {
    raw: LLVMTypeRef,
    _marker: PhantomData<&'ctx ()>,
}
impl<'ctx> Type<'ctx> {
    unsafe fn new(raw: LLVMTypeRef) -> Self {
        assert!(!raw.is_null());
        Type {
            raw,
            _marker: PhantomData,
        }
    }
    fn const_zero(self) -> LLVMValueRef {
        unsafe {
            match LLVMGetTypeKind(self.raw) {
                LLVMTypeKind::LLVMMetadataTypeKind => LLVMConstPointerNull(self.raw),
                _ => LLVMConstNull(self.raw),
            }
        }
    }
    fn ptr_type(self, x: AddressSpace) -> PointerType<'ctx> {
        unsafe { PointerType::new(LLVMPointerType(self.raw, x.0)) }
    }
    fn vec_type(self, size: u32) -> VectorType<'ctx> {
        assert!(size != 0, "Vectors of size zero are not allowed.");
        unsafe { VectorType::new(LLVMVectorType(self.raw, size)) }
    }
    fn fn_type(self, xs: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        let mut ys: Vec<LLVMTypeRef> = xs.iter().map(|x| x.as_type_ref()).collect();
        unsafe {
            FunctionType::new(LLVMFunctionType(
                self.raw,
                ys.as_mut_ptr(),
                ys.len() as u32,
                is_var_args as i32,
            ))
        }
    }
    fn array_type(self, size: u32) -> ArrayType<'ctx> {
        unsafe { ArrayType::new(LLVMArrayType(self.raw, size)) }
    }
    fn get_undef(self) -> LLVMValueRef {
        unsafe { LLVMGetUndef(self.raw) }
    }
    fn get_alignment(self) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMAlignOf(self.raw)) }
    }
    fn get_context(self) -> ContextRef<'ctx> {
        unsafe { ContextRef::new(LLVMGetTypeContext(self.raw)) }
    }
    fn is_sized(self) -> bool {
        unsafe { LLVMTypeIsSized(self.raw) == 1 }
    }
    fn size_of(self) -> Option<IntValue<'ctx>> {
        if !self.is_sized() {
            return None;
        }
        unsafe { Some(IntValue::new(LLVMSizeOf(self.raw))) }
    }
    fn print_to_string(self) -> LLVMString {
        unsafe { LLVMString::new(LLVMPrintTypeToString(self.raw)) }
    }
    pub fn get_element_type(self) -> AnyTypeEnum<'ctx> {
        unsafe { AnyTypeEnum::new(LLVMGetElementType(self.raw)) }
    }
}
impl fmt::Debug for Type<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ty = self.print_to_string();
        f.debug_struct("Type")
            .field("address", &self.raw)
            .field("llvm_type", &ty)
            .finish()
    }
}

pub unsafe trait AsTypeRef {
    fn as_type_ref(&self) -> LLVMTypeRef;
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct ArrayType<'ctx> {
    typ: Type<'ctx>,
}
impl<'ctx> ArrayType<'ctx> {
    pub unsafe fn new(typ: LLVMTypeRef) -> Self {
        assert!(!typ.is_null());
        ArrayType { typ: Type::new(typ) }
    }
    pub fn size_of(self) -> Option<IntValue<'ctx>> {
        self.typ.size_of()
    }
    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.typ.get_alignment()
    }
    pub fn ptr_type(self, x: AddressSpace) -> PointerType<'ctx> {
        self.typ.ptr_type(x)
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        self.typ.get_context()
    }
    pub fn fn_type(self, xs: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.typ.fn_type(xs, is_var_args)
    }
    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.typ.array_type(size)
    }
    pub fn const_array(self, xs: &[ArrayValue<'ctx>]) -> ArrayValue<'ctx> {
        let mut ys: Vec<LLVMValueRef> = xs.iter().map(|val| val.as_value_ref()).collect();
        unsafe { ArrayValue::new(LLVMConstArray(self.as_type_ref(), ys.as_mut_ptr(), ys.len() as u32)) }
    }
    pub fn const_zero(self) -> ArrayValue<'ctx> {
        unsafe { ArrayValue::new(self.typ.const_zero()) }
    }
    pub fn len(self) -> u32 {
        unsafe { LLVMGetArrayLength(self.as_type_ref()) }
    }
    pub fn print_to_string(self) -> LLVMString {
        self.typ.print_to_string()
    }
    pub fn get_undef(self) -> ArrayValue<'ctx> {
        unsafe { ArrayValue::new(self.typ.get_undef()) }
    }
    pub fn get_element_type(self) -> BasicTypeEnum<'ctx> {
        self.typ.get_element_type().to_basic_type_enum()
    }
}
impl Display for ArrayType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsTypeRef for ArrayType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.typ.raw
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
                        $enum_name::$args(ref x) => x.as_type_ref(),
                    )*
                }
            }
        }
        $(
            impl<'ctx> From<$args<'ctx>> for $enum_name<'ctx> {
                fn from(x: $args) -> $enum_name {
                    $enum_name::$args(x)
                }
            }
            impl<'ctx> TryFrom<$enum_name<'ctx>> for $args<'ctx> {
                type Error = ();
                fn try_from(x: $enum_name<'ctx>) -> Result<Self, Self::Error> {
                    match x {
                        $enum_name::$args(x) => Ok(x),
                        _ => Err(()),
                    }
                }
            }
        )*
    );
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
impl<'ctx> BasicTypeEnum<'ctx> {
    pub unsafe fn new(x: LLVMTypeRef) -> Self {
        match LLVMGetTypeKind(x) {
            LLVMTypeKind::LLVMHalfTypeKind
            | LLVMTypeKind::LLVMFloatTypeKind
            | LLVMTypeKind::LLVMDoubleTypeKind
            | LLVMTypeKind::LLVMX86_FP80TypeKind
            | LLVMTypeKind::LLVMFP128TypeKind
            | LLVMTypeKind::LLVMPPC_FP128TypeKind => BasicTypeEnum::FloatType(FloatType::new(x)),
            LLVMTypeKind::LLVMBFloatTypeKind => BasicTypeEnum::FloatType(FloatType::new(x)),
            LLVMTypeKind::LLVMIntegerTypeKind => BasicTypeEnum::IntType(IntType::new(x)),
            LLVMTypeKind::LLVMStructTypeKind => BasicTypeEnum::StructType(StructType::new(x)),
            LLVMTypeKind::LLVMPointerTypeKind => BasicTypeEnum::PointerType(PointerType::new(x)),
            LLVMTypeKind::LLVMArrayTypeKind => BasicTypeEnum::ArrayType(ArrayType::new(x)),
            LLVMTypeKind::LLVMVectorTypeKind => BasicTypeEnum::VectorType(VectorType::new(x)),
            LLVMTypeKind::LLVMScalableVectorTypeKind => BasicTypeEnum::VectorType(VectorType::new(x)),
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
        if let BasicTypeEnum::ArrayType(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the ArrayType variant", self);
        }
    }
    pub fn into_float_type(self) -> FloatType<'ctx> {
        if let BasicTypeEnum::FloatType(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the FloatType variant", self);
        }
    }
    pub fn into_int_type(self) -> IntType<'ctx> {
        if let BasicTypeEnum::IntType(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the IntType variant", self);
        }
    }
    pub fn into_pointer_type(self) -> PointerType<'ctx> {
        if let BasicTypeEnum::PointerType(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the PointerType variant", self);
        }
    }
    pub fn into_struct_type(self) -> StructType<'ctx> {
        if let BasicTypeEnum::StructType(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the StructType variant", self);
        }
    }
    pub fn into_vector_type(self) -> VectorType<'ctx> {
        if let BasicTypeEnum::VectorType(x) = self {
            x
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
            BasicTypeEnum::ArrayType(x) => x.const_zero().as_basic_value_enum(),
            BasicTypeEnum::FloatType(x) => x.const_zero().as_basic_value_enum(),
            BasicTypeEnum::IntType(x) => x.const_zero().as_basic_value_enum(),
            BasicTypeEnum::PointerType(x) => x.const_zero().as_basic_value_enum(),
            BasicTypeEnum::StructType(x) => x.const_zero().as_basic_value_enum(),
            BasicTypeEnum::VectorType(x) => x.const_zero().as_basic_value_enum(),
        }
    }
    pub fn print_to_string(self) -> LLVMString {
        match self {
            BasicTypeEnum::ArrayType(x) => x.print_to_string(),
            BasicTypeEnum::FloatType(x) => x.print_to_string(),
            BasicTypeEnum::IntType(x) => x.print_to_string(),
            BasicTypeEnum::PointerType(x) => x.print_to_string(),
            BasicTypeEnum::StructType(x) => x.print_to_string(),
            BasicTypeEnum::VectorType(x) => x.print_to_string(),
        }
    }
}
impl<'ctx> TryFrom<AnyTypeEnum<'ctx>> for BasicTypeEnum<'ctx> {
    type Error = ();
    fn try_from(x: AnyTypeEnum<'ctx>) -> Result<Self, Self::Error> {
        use AnyTypeEnum::*;
        Ok(match x {
            ArrayType(x) => x.into(),
            FloatType(x) => x.into(),
            IntType(x) => x.into(),
            PointerType(x) => x.into(),
            StructType(x) => x.into(),
            VectorType(x) => x.into(),
            VoidType(_) | FunctionType(_) => return Err(()),
        })
    }
}
impl<'ctx> TryFrom<BasicMetadataTypeEnum<'ctx>> for BasicTypeEnum<'ctx> {
    type Error = ();
    fn try_from(x: BasicMetadataTypeEnum<'ctx>) -> Result<Self, Self::Error> {
        use BasicMetadataTypeEnum::*;
        Ok(match x {
            ArrayType(x) => x.into(),
            FloatType(x) => x.into(),
            IntType(x) => x.into(),
            PointerType(x) => x.into(),
            StructType(x) => x.into(),
            VectorType(x) => x.into(),
            MetadataType(_) => return Err(()),
        })
    }
}
impl Display for BasicTypeEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
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
impl Display for BasicMetadataTypeEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
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
impl<'ctx> AnyTypeEnum<'ctx> {
    pub unsafe fn new(x: LLVMTypeRef) -> Self {
        match LLVMGetTypeKind(x) {
            LLVMTypeKind::LLVMVoidTypeKind => AnyTypeEnum::VoidType(VoidType::new(x)),
            LLVMTypeKind::LLVMHalfTypeKind
            | LLVMTypeKind::LLVMFloatTypeKind
            | LLVMTypeKind::LLVMDoubleTypeKind
            | LLVMTypeKind::LLVMX86_FP80TypeKind
            | LLVMTypeKind::LLVMFP128TypeKind
            | LLVMTypeKind::LLVMPPC_FP128TypeKind => AnyTypeEnum::FloatType(FloatType::new(x)),
            LLVMTypeKind::LLVMBFloatTypeKind => AnyTypeEnum::FloatType(FloatType::new(x)),
            LLVMTypeKind::LLVMLabelTypeKind => panic!("FIXME: Unsupported type: Label"),
            LLVMTypeKind::LLVMIntegerTypeKind => AnyTypeEnum::IntType(IntType::new(x)),
            LLVMTypeKind::LLVMFunctionTypeKind => AnyTypeEnum::FunctionType(FunctionType::new(x)),
            LLVMTypeKind::LLVMStructTypeKind => AnyTypeEnum::StructType(StructType::new(x)),
            LLVMTypeKind::LLVMArrayTypeKind => AnyTypeEnum::ArrayType(ArrayType::new(x)),
            LLVMTypeKind::LLVMPointerTypeKind => AnyTypeEnum::PointerType(PointerType::new(x)),
            LLVMTypeKind::LLVMVectorTypeKind => AnyTypeEnum::VectorType(VectorType::new(x)),
            LLVMTypeKind::LLVMScalableVectorTypeKind => AnyTypeEnum::VectorType(VectorType::new(x)),
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
impl Display for AnyTypeEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct FloatType<'ctx> {
    typ: Type<'ctx>,
}
impl<'ctx> FloatType<'ctx> {
    pub unsafe fn new(typ: LLVMTypeRef) -> Self {
        assert!(!typ.is_null());
        FloatType { typ: Type::new(typ) }
    }
    pub fn fn_type(self, xs: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.typ.fn_type(xs, is_var_args)
    }
    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.typ.array_type(size)
    }
    pub fn vec_type(self, size: u32) -> VectorType<'ctx> {
        self.typ.vec_type(size)
    }
    pub fn const_float(self, x: f64) -> FloatValue<'ctx> {
        unsafe { FloatValue::new(LLVMConstReal(self.typ.raw, x)) }
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
        unsafe { FloatValue::new(self.typ.const_zero()) }
    }
    pub fn size_of(self) -> IntValue<'ctx> {
        self.typ.size_of().unwrap()
    }
    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.typ.get_alignment()
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        self.typ.get_context()
    }
    pub fn ptr_type(self, x: AddressSpace) -> PointerType<'ctx> {
        self.typ.ptr_type(x)
    }
    pub fn print_to_string(self) -> LLVMString {
        self.typ.print_to_string()
    }
    pub fn get_undef(&self) -> FloatValue<'ctx> {
        unsafe { FloatValue::new(self.typ.get_undef()) }
    }
    pub fn create_generic_value(self, value: f64) -> GenericValue<'ctx> {
        unsafe { GenericValue::new(LLVMCreateGenericValueOfFloat(self.as_type_ref(), value)) }
    }
    pub fn const_array(self, xs: &[FloatValue<'ctx>]) -> ArrayValue<'ctx> {
        let mut ys: Vec<LLVMValueRef> = xs.iter().map(|val| val.as_value_ref()).collect();
        unsafe { ArrayValue::new(LLVMConstArray(self.as_type_ref(), ys.as_mut_ptr(), ys.len() as u32)) }
    }
}
impl Display for FloatType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsTypeRef for FloatType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.typ.raw
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct FunctionType<'ctx> {
    typ: Type<'ctx>,
}
impl<'ctx> FunctionType<'ctx> {
    pub unsafe fn new(typ: LLVMTypeRef) -> Self {
        assert!(!typ.is_null());
        FunctionType { typ: Type::new(typ) }
    }
    pub fn ptr_type(self, x: AddressSpace) -> PointerType<'ctx> {
        self.typ.ptr_type(x)
    }
    pub fn is_var_arg(self) -> bool {
        unsafe { LLVMIsFunctionVarArg(self.as_type_ref()) != 0 }
    }
    pub fn get_param_types(self) -> Vec<BasicTypeEnum<'ctx>> {
        let n = self.count_param_types();
        let mut y: Vec<LLVMTypeRef> = Vec::with_capacity(n as usize);
        let ptr = y.as_mut_ptr();
        forget(y);
        let y = unsafe {
            LLVMGetParamTypes(self.as_type_ref(), ptr);
            Vec::from_raw_parts(ptr, n as usize, n as usize)
        };
        y.iter().map(|x| unsafe { BasicTypeEnum::new(*x) }).collect()
    }
    pub fn count_param_types(self) -> u32 {
        unsafe { LLVMCountParamTypes(self.as_type_ref()) }
    }
    pub fn is_sized(self) -> bool {
        self.typ.is_sized()
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        self.typ.get_context()
    }
    pub fn print_to_string(self) -> LLVMString {
        self.typ.print_to_string()
    }
    pub fn get_return_type(self) -> Option<BasicTypeEnum<'ctx>> {
        let y = unsafe { LLVMGetReturnType(self.as_type_ref()) };
        let kind = unsafe { LLVMGetTypeKind(y) };
        if let LLVMTypeKind::LLVMVoidTypeKind = kind {
            return None;
        }
        unsafe { Some(BasicTypeEnum::new(y)) }
    }
}
impl Display for FunctionType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
impl fmt::Debug for FunctionType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ty = self.print_to_string();
        f.debug_struct("FunctionType")
            .field("address", &self.as_type_ref())
            .field("is_var_args", &self.is_var_arg())
            .field("llvm_type", &ty)
            .finish()
    }
}
unsafe impl AsTypeRef for FunctionType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.typ.raw
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
impl TryFrom<u8> for StringRadix {
    type Error = ();
    fn try_from(x: u8) -> Result<Self, Self::Error> {
        match x {
            2 => Ok(StringRadix::Binary),
            8 => Ok(StringRadix::Octal),
            10 => Ok(StringRadix::Decimal),
            16 => Ok(StringRadix::Hexadecimal),
            36 => Ok(StringRadix::Alphanumeric),
            _ => Err(()),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct IntType<'ctx> {
    typ: Type<'ctx>,
}
impl<'ctx> IntType<'ctx> {
    pub unsafe fn new(typ: LLVMTypeRef) -> Self {
        assert!(!typ.is_null());
        IntType { typ: Type::new(typ) }
    }
    pub fn const_int(self, x: u64, sign_extend: bool) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstInt(self.as_type_ref(), x, sign_extend as i32)) }
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
    pub fn const_int_arbitrary_precision(self, xs: &[u64]) -> IntValue<'ctx> {
        unsafe {
            IntValue::new(LLVMConstIntOfArbitraryPrecision(
                self.as_type_ref(),
                xs.len() as u32,
                xs.as_ptr(),
            ))
        }
    }
    pub fn const_all_ones(self) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstAllOnes(self.as_type_ref())) }
    }
    pub fn const_zero(self) -> IntValue<'ctx> {
        unsafe { IntValue::new(self.typ.const_zero()) }
    }
    pub fn fn_type(self, xs: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.typ.fn_type(xs, is_var_args)
    }
    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.typ.array_type(size)
    }
    pub fn vec_type(self, size: u32) -> VectorType<'ctx> {
        self.typ.vec_type(size)
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        self.typ.get_context()
    }
    pub fn size_of(self) -> IntValue<'ctx> {
        self.typ.size_of().unwrap()
    }
    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.typ.get_alignment()
    }
    pub fn ptr_type(self, x: AddressSpace) -> PointerType<'ctx> {
        self.typ.ptr_type(x)
    }
    pub fn get_bit_width(self) -> u32 {
        unsafe { LLVMGetIntTypeWidth(self.as_type_ref()) }
    }
    pub fn print_to_string(self) -> LLVMString {
        self.typ.print_to_string()
    }
    pub fn get_undef(self) -> IntValue<'ctx> {
        unsafe { IntValue::new(self.typ.get_undef()) }
    }
    pub fn create_generic_value(self, value: u64, is_signed: bool) -> GenericValue<'ctx> {
        unsafe { GenericValue::new(LLVMCreateGenericValueOfInt(self.as_type_ref(), value, is_signed as i32)) }
    }
    pub fn const_array(self, xs: &[IntValue<'ctx>]) -> ArrayValue<'ctx> {
        let mut ys: Vec<LLVMValueRef> = xs.iter().map(|val| val.as_value_ref()).collect();
        unsafe { ArrayValue::new(LLVMConstArray(self.as_type_ref(), ys.as_mut_ptr(), ys.len() as u32)) }
    }
}
impl Display for IntType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsTypeRef for IntType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.typ.raw
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct MetadataType<'ctx> {
    typ: Type<'ctx>,
}
impl<'ctx> MetadataType<'ctx> {
    pub unsafe fn new(typ: LLVMTypeRef) -> Self {
        assert!(!typ.is_null());
        MetadataType { typ: Type::new(typ) }
    }
    pub fn fn_type(self, xs: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.typ.fn_type(xs, is_var_args)
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        self.typ.get_context()
    }
    pub fn print_to_string(self) -> LLVMString {
        self.typ.print_to_string()
    }
}
impl Display for MetadataType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsTypeRef for MetadataType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.typ.raw
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct PointerType<'ctx> {
    typ: Type<'ctx>,
}
impl<'ctx> PointerType<'ctx> {
    pub unsafe fn new(typ: LLVMTypeRef) -> Self {
        assert!(!typ.is_null());
        PointerType { typ: Type::new(typ) }
    }
    pub fn size_of(self) -> IntValue<'ctx> {
        self.typ.size_of().unwrap()
    }
    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.typ.get_alignment()
    }
    pub fn ptr_type(self, x: AddressSpace) -> PointerType<'ctx> {
        self.typ.ptr_type(x)
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        self.typ.get_context()
    }
    pub fn fn_type(self, xs: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.typ.fn_type(xs, is_var_args)
    }
    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.typ.array_type(size)
    }
    pub fn get_address_space(self) -> AddressSpace {
        let y = unsafe { LLVMGetPointerAddressSpace(self.as_type_ref()) };
        AddressSpace(y)
    }
    pub fn print_to_string(self) -> LLVMString {
        self.typ.print_to_string()
    }
    pub fn const_null(self) -> PointerValue<'ctx> {
        unsafe { PointerValue::new(self.typ.const_zero()) }
    }
    pub fn const_zero(self) -> PointerValue<'ctx> {
        unsafe { PointerValue::new(self.typ.const_zero()) }
    }
    pub fn get_undef(self) -> PointerValue<'ctx> {
        unsafe { PointerValue::new(self.typ.get_undef()) }
    }
    pub fn vec_type(self, size: u32) -> VectorType<'ctx> {
        self.typ.vec_type(size)
    }
    pub fn const_array(self, xs: &[PointerValue<'ctx>]) -> ArrayValue<'ctx> {
        let mut ys: Vec<LLVMValueRef> = xs.iter().map(|val| val.as_value_ref()).collect();
        unsafe { ArrayValue::new(LLVMConstArray(self.as_type_ref(), ys.as_mut_ptr(), ys.len() as u32)) }
    }
}
impl Display for PointerType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsTypeRef for PointerType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.typ.raw
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct StructType<'ctx> {
    typ: Type<'ctx>,
}
impl<'ctx> StructType<'ctx> {
    pub unsafe fn new(typ: LLVMTypeRef) -> Self {
        assert!(!typ.is_null());
        StructType { typ: Type::new(typ) }
    }
    pub fn get_field_type_at_index(self, idx: u32) -> Option<BasicTypeEnum<'ctx>> {
        if self.is_opaque() {
            return None;
        }
        if idx >= self.count_fields() {
            return None;
        }
        unsafe { Some(BasicTypeEnum::new(LLVMStructGetTypeAtIndex(self.as_type_ref(), idx))) }
    }
    pub fn const_named_struct(self, xs: &[BasicValueEnum<'ctx>]) -> StructValue<'ctx> {
        let mut ys: Vec<LLVMValueRef> = xs.iter().map(|val| val.as_value_ref()).collect();
        unsafe {
            StructValue::new(LLVMConstNamedStruct(
                self.as_type_ref(),
                ys.as_mut_ptr(),
                ys.len() as u32,
            ))
        }
    }
    pub fn const_zero(self) -> StructValue<'ctx> {
        unsafe { StructValue::new(self.typ.const_zero()) }
    }
    pub fn size_of(self) -> Option<IntValue<'ctx>> {
        self.typ.size_of()
    }
    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.typ.get_alignment()
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        self.typ.get_context()
    }
    pub fn get_name(&self) -> Option<&CStr> {
        let name = unsafe { LLVMGetStructName(self.as_type_ref()) };
        if name.is_null() {
            return None;
        }
        let y = unsafe { CStr::from_ptr(name) };
        Some(y)
    }
    pub fn ptr_type(self, x: AddressSpace) -> PointerType<'ctx> {
        self.typ.ptr_type(x)
    }
    pub fn fn_type(self, xs: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.typ.fn_type(xs, is_var_args)
    }
    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.typ.array_type(size)
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
        let n = self.count_fields();
        let mut y: Vec<LLVMTypeRef> = Vec::with_capacity(n as usize);
        let ptr = y.as_mut_ptr();
        forget(y);
        let y = unsafe {
            LLVMGetStructElementTypes(self.as_type_ref(), ptr);
            Vec::from_raw_parts(ptr, n as usize, n as usize)
        };
        y.iter().map(|val| unsafe { BasicTypeEnum::new(*val) }).collect()
    }
    pub fn print_to_string(self) -> LLVMString {
        self.typ.print_to_string()
    }
    pub fn get_undef(self) -> StructValue<'ctx> {
        unsafe { StructValue::new(self.typ.get_undef()) }
    }
    pub fn set_body(self, xs: &[BasicTypeEnum<'ctx>], packed: bool) -> bool {
        let is_opaque = self.is_opaque();
        let mut ys: Vec<LLVMTypeRef> = xs.iter().map(|val| val.as_type_ref()).collect();
        unsafe {
            LLVMStructSetBody(self.as_type_ref(), ys.as_mut_ptr(), ys.len() as u32, packed as i32);
        }
        is_opaque
    }
    pub fn const_array(self, xs: &[StructValue<'ctx>]) -> ArrayValue<'ctx> {
        let mut ys: Vec<LLVMValueRef> = xs.iter().map(|val| val.as_value_ref()).collect();
        unsafe { ArrayValue::new(LLVMConstArray(self.as_type_ref(), ys.as_mut_ptr(), ys.len() as u32)) }
    }
}
impl Display for StructType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsTypeRef for StructType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.typ.raw
    }
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
    fn fn_type(&self, xs: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        unsafe { Type::new(self.as_type_ref()).fn_type(xs, is_var_args) }
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

macro_rules! trait_type_set {
    ($name:ident: $($args:ident),*) => (
        $(
            unsafe impl<'ctx> $name<'ctx> for $args<'ctx> {}
        )*
    );
}

trait_type_set! {AnyType: AnyTypeEnum, BasicTypeEnum, IntType, FunctionType, FloatType, PointerType, StructType, ArrayType, VoidType, VectorType}
trait_type_set! {BasicType: BasicTypeEnum, IntType, FloatType, PointerType, StructType, ArrayType, VectorType}

pub unsafe trait IntMathType<'ctx>: BasicType<'ctx> {
    type ValueType: IntMathValue<'ctx>;
    type MathConvType: FloatMathType<'ctx>;
    type PtrConvType: PointerMathType<'ctx>;
}
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

pub unsafe trait FloatMathType<'ctx>: BasicType<'ctx> {
    type ValueType: FloatMathValue<'ctx>;
    type MathConvType: IntMathType<'ctx>;
}
unsafe impl<'ctx> FloatMathType<'ctx> for FloatType<'ctx> {
    type ValueType = FloatValue<'ctx>;
    type MathConvType = IntType<'ctx>;
}
unsafe impl<'ctx> FloatMathType<'ctx> for VectorType<'ctx> {
    type ValueType = VectorValue<'ctx>;
    type MathConvType = VectorType<'ctx>;
}

pub unsafe trait PointerMathType<'ctx>: BasicType<'ctx> {
    type ValueType: PointerMathValue<'ctx>;
    type PtrConvType: IntMathType<'ctx>;
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
    typ: Type<'ctx>,
}
impl<'ctx> VectorType<'ctx> {
    pub unsafe fn new(typ: LLVMTypeRef) -> Self {
        assert!(!typ.is_null());
        VectorType { typ: Type::new(typ) }
    }
    pub fn size_of(self) -> Option<IntValue<'ctx>> {
        self.typ.size_of()
    }
    pub fn get_alignment(self) -> IntValue<'ctx> {
        self.typ.get_alignment()
    }
    pub fn get_size(self) -> u32 {
        unsafe { LLVMGetVectorSize(self.as_type_ref()) }
    }
    pub fn const_vector<V: BasicValue<'ctx>>(xs: &[V]) -> VectorValue<'ctx> {
        let mut ys: Vec<LLVMValueRef> = xs.iter().map(|val| val.as_value_ref()).collect();
        unsafe { VectorValue::new(LLVMConstVector(ys.as_mut_ptr(), ys.len() as u32)) }
    }
    pub fn const_zero(self) -> VectorValue<'ctx> {
        unsafe { VectorValue::new(self.typ.const_zero()) }
    }
    pub fn print_to_string(self) -> LLVMString {
        self.typ.print_to_string()
    }
    pub fn get_undef(self) -> VectorValue<'ctx> {
        unsafe { VectorValue::new(self.typ.get_undef()) }
    }
    pub fn get_element_type(self) -> BasicTypeEnum<'ctx> {
        self.typ.get_element_type().to_basic_type_enum()
    }
    pub fn ptr_type(self, x: AddressSpace) -> PointerType<'ctx> {
        self.typ.ptr_type(x)
    }
    pub fn fn_type(self, xs: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.typ.fn_type(xs, is_var_args)
    }
    pub fn array_type(self, size: u32) -> ArrayType<'ctx> {
        self.typ.array_type(size)
    }
    pub fn const_array(self, xs: &[VectorValue<'ctx>]) -> ArrayValue<'ctx> {
        let mut ys: Vec<LLVMValueRef> = xs.iter().map(|val| val.as_value_ref()).collect();
        unsafe { ArrayValue::new(LLVMConstArray(self.as_type_ref(), ys.as_mut_ptr(), ys.len() as u32)) }
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        self.typ.get_context()
    }
}
impl Display for VectorType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsTypeRef for VectorType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.typ.raw
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct VoidType<'ctx> {
    typ: Type<'ctx>,
}
impl<'ctx> VoidType<'ctx> {
    pub unsafe fn new(x: LLVMTypeRef) -> Self {
        assert!(!x.is_null());
        VoidType { typ: Type::new(x) }
    }
    pub fn is_sized(self) -> bool {
        self.typ.is_sized()
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        self.typ.get_context()
    }
    pub fn fn_type(self, xs: &[BasicMetadataTypeEnum<'ctx>], is_var_args: bool) -> FunctionType<'ctx> {
        self.typ.fn_type(xs, is_var_args)
    }
    pub fn print_to_string(self) -> LLVMString {
        self.typ.print_to_string()
    }
}
impl Display for VoidType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsTypeRef for VoidType<'_> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.typ.raw
    }
}
