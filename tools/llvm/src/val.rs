use either::{
    Either,
    Either::{Left, Right},
};
use libc::c_void;
use llvm_lib::{analysis::*, comdat::*, core::*, debuginfo::*, execution_engine::*, prelude::*, *};
use std::{
    convert::TryFrom,
    fmt::{self, Debug},
    marker,
    mem::forget,
    {ffi::CStr, fmt, fmt::Display},
};

use crate::{dbg::DISubprogram, typ::*, *};

#[derive(PartialEq, Eq, Clone, Copy, Hash)]
struct Value<'ctx> {
    raw: LLVMValueRef,
    _marker: marker::PhantomData<&'ctx ()>,
}
impl<'ctx> Value<'ctx> {
    pub unsafe fn new(raw: LLVMValueRef) -> Self {
        debug_assert!(!raw.is_null());
        Value {
            raw,
            _marker: marker::PhantomData,
        }
    }
    fn is_instruction(self) -> bool {
        unsafe { !LLVMIsAInstruction(self.raw).is_null() }
    }
    fn as_instruction(self) -> Option<InstructionValue<'ctx>> {
        if !self.is_instruction() {
            return None;
        }
        unsafe { Some(InstructionValue::new(self.raw)) }
    }
    fn is_null(self) -> bool {
        unsafe { LLVMIsNull(self.raw) == 1 }
    }
    fn is_const(self) -> bool {
        unsafe { LLVMIsConstant(self.raw) == 1 }
    }
    fn set_name(self, name: &str) {
        let y = to_c_str(name);
        {
            unsafe { LLVMSetValueName2(self.raw, y.as_ptr(), name.len()) }
        }
    }
    fn get_name(&self) -> &CStr {
        let y = unsafe {
            let mut len = 0;
            LLVMGetValueName2(self.raw, &mut len)
        };
        unsafe { CStr::from_ptr(y) }
    }
    fn is_undef(self) -> bool {
        unsafe { LLVMIsUndef(self.raw) == 1 }
    }
    fn get_type(self) -> LLVMTypeRef {
        unsafe { LLVMTypeOf(self.raw) }
    }
    fn print_to_string(self) -> LLVMString {
        unsafe { LLVMString::new(LLVMPrintValueToString(self.raw)) }
    }
    fn print_to_stderr(self) {
        unsafe { LLVMDumpValue(self.raw) }
    }
    fn replace_all_uses_with(self, x: LLVMValueRef) {
        if self.raw != x {
            unsafe { LLVMReplaceAllUsesWith(self.raw, x) }
        }
    }
    pub fn get_first_use(self) -> Option<BasicValueUse<'ctx>> {
        let y = unsafe { LLVMGetFirstUse(self.raw) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(BasicValueUse::new(y)) }
    }
    pub fn get_section(&self) -> Option<&CStr> {
        let y = unsafe { LLVMGetSection(self.raw) };
        if y.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(y) })
    }
    fn set_section(self, section: Option<&str>) {
        let y = section.as_deref().map(to_c_str);
        unsafe { LLVMSetSection(self.raw, y.as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null())) }
    }
}
impl fmt::Debug for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let val = self.print_to_string();
        let ty = unsafe { CStr::from_ptr(LLVMPrintTypeToString(LLVMTypeOf(self.raw))) };
        let name = self.get_name();
        let is_const = self.is_const();
        let is_null = self.is_null();
        let is_undef = self.is_undef();
        f.debug_struct("Value")
            .field("name", &name)
            .field("address", &self.raw)
            .field("is_const", &is_const)
            .field("is_null", &is_null)
            .field("is_undef", &is_undef)
            .field("llvm_value", &val)
            .field("llvm_type", &ty)
            .finish()
    }
}

pub unsafe trait AsValueRef {
    fn as_value_ref(&self) -> LLVMValueRef;
}

#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub struct ArrayValue<'ctx> {
    val: Value<'ctx>,
}
impl<'ctx> ArrayValue<'ctx> {
    pub unsafe fn new(x: LLVMValueRef) -> Self {
        assert!(!x.is_null());
        ArrayValue { val: Value::new(x) }
    }
    pub fn get_name(&self) -> &CStr {
        self.val.get_name()
    }
    pub fn set_name(&self, name: &str) {
        self.val.set_name(name)
    }
    pub fn get_type(self) -> ArrayType<'ctx> {
        unsafe { ArrayType::new(self.val.get_type()) }
    }
    pub fn is_null(self) -> bool {
        self.val.is_null()
    }
    pub fn is_undef(self) -> bool {
        self.val.is_undef()
    }
    pub fn print_to_stderr(self) {
        self.val.print_to_stderr()
    }
    pub fn as_instruction(self) -> Option<InstructionValue<'ctx>> {
        self.val.as_instruction()
    }
    pub fn replace_all_uses_with(self, x: ArrayValue<'ctx>) {
        self.val.replace_all_uses_with(x.as_value_ref())
    }
    pub fn is_const(self) -> bool {
        self.val.is_const()
    }
    pub fn is_const_string(self) -> bool {
        unsafe { LLVMIsConstantString(self.as_value_ref()) == 1 }
    }
    pub fn get_string_constant(&self) -> Option<&CStr> {
        let mut len = 0;
        let y = unsafe { LLVMGetAsString(self.as_value_ref(), &mut len) };
        if y.is_null() {
            None
        } else {
            unsafe { Some(CStr::from_ptr(y)) }
        }
    }
}
impl Display for ArrayValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
impl fmt::Debug for ArrayValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let val = self.print_to_string();
        let ty = self.get_type();
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
            .field("llvm_value", &val)
            .field("llvm_type", &ty)
            .finish()
    }
}
unsafe impl AsValueRef for ArrayValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.val.raw
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BasicValueUse<'ctx>(LLVMUseRef, marker::PhantomData<&'ctx ()>);
impl<'ctx> BasicValueUse<'ctx> {
    pub unsafe fn new(x: LLVMUseRef) -> Self {
        debug_assert!(!x.is_null());
        BasicValueUse(x, marker::PhantomData)
    }
    pub fn get_next_use(self) -> Option<Self> {
        let y = unsafe { LLVMGetNextUse(self.0) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(Self::new(y)) }
    }
    pub fn get_user(self) -> AnyValueEnum<'ctx> {
        unsafe { AnyValueEnum::new(LLVMGetUser(self.0)) }
    }
    pub fn get_used_value(self) -> Either<BasicValueEnum<'ctx>, BasicBlock<'ctx>> {
        let y = unsafe { LLVMGetUsedValue(self.0) };
        let is_basic_block = unsafe { !LLVMIsABasicBlock(y).is_null() };
        if is_basic_block {
            let y = unsafe { BasicBlock::new(LLVMValueAsBasicBlock(y)) };
            Right(y.expect("BasicBlock should always be valid"))
        } else {
            unsafe { Left(BasicValueEnum::new(y)) }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct CallSiteValue<'ctx>(Value<'ctx>);
impl<'ctx> CallSiteValue<'ctx> {
    pub unsafe fn new(x: LLVMValueRef) -> Self {
        CallSiteValue(Value::new(x))
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
    pub fn add_attribute(self, loc: AttributeLoc, a: Attribute) {
        unsafe { LLVMAddCallSiteAttribute(self.as_value_ref(), loc.get_index(), a.raw) }
    }
    pub fn get_called_fn_value(self) -> FunctionValue<'ctx> {
        unsafe { FunctionValue::new(LLVMGetCalledValue(self.as_value_ref())).expect("This should never be null?") }
    }
    pub fn count_attributes(self, loc: AttributeLoc) -> u32 {
        unsafe { LLVMGetCallSiteAttributeCount(self.as_value_ref(), loc.get_index()) }
    }
    pub fn attributes(self, loc: AttributeLoc) -> Vec<Attribute> {
        use std::mem::{ManuallyDrop, MaybeUninit};
        let count = self.count_attributes(loc) as usize;
        let mut y: Vec<MaybeUninit<Attribute>> = vec![MaybeUninit::uninit(); count];
        unsafe { LLVMGetCallSiteAttributes(self.as_value_ref(), loc.get_index(), y.as_mut_ptr() as *mut _) }
        unsafe {
            let mut y = ManuallyDrop::new(y);
            Vec::from_raw_parts(y.as_mut_ptr() as *mut Attribute, y.len(), y.capacity())
        }
    }
    pub fn get_enum_attribute(self, loc: AttributeLoc, kind_id: u32) -> Option<Attribute> {
        let y = unsafe { LLVMGetCallSiteEnumAttribute(self.as_value_ref(), loc.get_index(), kind_id) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(Attribute::new(y)) }
    }
    pub fn get_string_attribute(self, loc: AttributeLoc, key: &str) -> Option<Attribute> {
        let y = unsafe {
            LLVMGetCallSiteStringAttribute(
                self.as_value_ref(),
                loc.get_index(),
                key.as_ptr() as *const ::libc::c_char,
                key.len() as u32,
            )
        };
        if y.is_null() {
            return None;
        }
        unsafe { Some(Attribute::new(y)) }
    }
    pub fn remove_enum_attribute(self, loc: AttributeLoc, kind_id: u32) {
        unsafe { LLVMRemoveCallSiteEnumAttribute(self.as_value_ref(), loc.get_index(), kind_id) }
    }
    pub fn remove_string_attribute(self, loc: AttributeLoc, key: &str) {
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
        unsafe { LLVMGetNumArgOperands(self.as_value_ref()) }
    }
    pub fn get_call_convention(self) -> u32 {
        unsafe { LLVMGetInstructionCallConv(self.as_value_ref()) }
    }
    pub fn set_call_convention(self, conv: u32) {
        unsafe { LLVMSetInstructionCallConv(self.as_value_ref(), conv) }
    }
    pub fn set_alignment_attribute(self, loc: AttributeLoc, align: u32) {
        assert_eq!(align.count_ones(), 1, "Alignment must be a power of two.");
        unsafe { LLVMSetInstrParamAlignment(self.as_value_ref(), loc.get_index(), align) }
    }
}
impl Display for CallSiteValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsValueRef for CallSiteValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.0.raw
    }
}

#[derive(Debug)]
pub struct CallableValue<'ctx>(Either<FunctionValue<'ctx>, PointerValue<'ctx>>);
impl<'ctx> CallableValue<'ctx> {}
impl<'ctx> From<FunctionValue<'ctx>> for CallableValue<'ctx> {
    fn from(x: FunctionValue<'ctx>) -> Self {
        Self(Either::Left(x))
    }
}
impl<'ctx> TryFrom<PointerValue<'ctx>> for CallableValue<'ctx> {
    type Error = ();
    fn try_from(x: PointerValue<'ctx>) -> Result<Self, Self::Error> {
        let value_ref = x.as_value_ref();
        let ty_kind = unsafe { LLVMGetTypeKind(LLVMGetElementType(LLVMTypeOf(value_ref))) };
        let is_a_fn_ptr = matches!(ty_kind, LLVMTypeKind::LLVMFunctionTypeKind);
        if is_a_fn_ptr {
            Ok(Self(Either::Right(x)))
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
unsafe impl<'ctx> AsValueRef for CallableValue<'ctx> {
    fn as_value_ref(&self) -> LLVMValueRef {
        use either::Either::*;
        match self.0 {
            Left(x) => x.as_value_ref(),
            Right(x) => x.as_value_ref(),
        }
    }
}
unsafe impl<'ctx> AnyValue<'ctx> for CallableValue<'ctx> {}
unsafe impl<'ctx> AsTypeRef for CallableValue<'ctx> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        use either::Either::*;
        match self.0 {
            Left(x) => x.get_type().as_type_ref(),
            Right(x) => x.get_type().get_element_type().as_type_ref(),
        }
    }
}

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
                        $enum_name::$args(ref x) => x.as_value_ref(),
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
            impl<'ctx> PartialEq<$args<'ctx>> for $enum_name<'ctx> {
                fn eq(&self, x: &$args<'ctx>) -> bool {
                    self.as_value_ref() == x.as_value_ref()
                }
            }
            impl<'ctx> PartialEq<$enum_name<'ctx>> for $args<'ctx> {
                fn eq(&self, x: &$enum_name<'ctx>) -> bool {
                    self.as_value_ref() == x.as_value_ref()
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

enum_value_set! {BasicValueEnum: ArrayValue, IntValue, FloatValue, PointerValue, StructValue, VectorValue}
impl<'ctx> BasicValueEnum<'ctx> {
    pub unsafe fn new(value: LLVMValueRef) -> Self {
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
            BasicValueEnum::ArrayValue(x) => x.set_name(name),
            BasicValueEnum::IntValue(x) => x.set_name(name),
            BasicValueEnum::FloatValue(x) => x.set_name(name),
            BasicValueEnum::PointerValue(x) => x.set_name(name),
            BasicValueEnum::StructValue(x) => x.set_name(name),
            BasicValueEnum::VectorValue(x) => x.set_name(name),
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
        if let BasicValueEnum::ArrayValue(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the ArrayValue variant", self)
        }
    }
    pub fn into_int_value(self) -> IntValue<'ctx> {
        if let BasicValueEnum::IntValue(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the IntValue variant", self)
        }
    }
    pub fn into_float_value(self) -> FloatValue<'ctx> {
        if let BasicValueEnum::FloatValue(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the FloatValue variant", self)
        }
    }
    pub fn into_pointer_value(self) -> PointerValue<'ctx> {
        if let BasicValueEnum::PointerValue(x) = self {
            x
        } else {
            panic!("Found {:?} but expected PointerValue variant", self)
        }
    }
    pub fn into_struct_value(self) -> StructValue<'ctx> {
        if let BasicValueEnum::StructValue(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the StructValue variant", self)
        }
    }
    pub fn into_vector_value(self) -> VectorValue<'ctx> {
        if let BasicValueEnum::VectorValue(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the VectorValue variant", self)
        }
    }
}
impl<'ctx> TryFrom<AnyValueEnum<'ctx>> for BasicValueEnum<'ctx> {
    type Error = ();
    fn try_from(x: AnyValueEnum<'ctx>) -> Result<Self, Self::Error> {
        use AnyValueEnum::*;
        Ok(match x {
            ArrayValue(x) => x.into(),
            IntValue(x) => x.into(),
            FloatValue(x) => x.into(),
            PointerValue(x) => x.into(),
            StructValue(x) => x.into(),
            VectorValue(x) => x.into(),
            MetadataValue(_) | PhiValue(_) | FunctionValue(_) | InstructionValue(_) => return Err(()),
        })
    }
}
impl<'ctx> TryFrom<BasicMetadataValueEnum<'ctx>> for BasicValueEnum<'ctx> {
    type Error = ();
    fn try_from(x: BasicMetadataValueEnum<'ctx>) -> Result<Self, Self::Error> {
        use BasicMetadataValueEnum::*;
        Ok(match x {
            ArrayValue(x) => x.into(),
            IntValue(x) => x.into(),
            FloatValue(x) => x.into(),
            PointerValue(x) => x.into(),
            StructValue(x) => x.into(),
            VectorValue(x) => x.into(),
            MetadataValue(_) => return Err(()),
        })
    }
}
impl Display for BasicValueEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

enum_value_set! {BasicMetadataValueEnum: ArrayValue, IntValue, FloatValue, PointerValue, StructValue, VectorValue, MetadataValue}
impl<'ctx> BasicMetadataValueEnum<'ctx> {
    pub unsafe fn new(x: LLVMValueRef) -> Self {
        match LLVMGetTypeKind(LLVMTypeOf(x)) {
            LLVMTypeKind::LLVMFloatTypeKind
            | LLVMTypeKind::LLVMFP128TypeKind
            | LLVMTypeKind::LLVMDoubleTypeKind
            | LLVMTypeKind::LLVMHalfTypeKind
            | LLVMTypeKind::LLVMX86_FP80TypeKind
            | LLVMTypeKind::LLVMPPC_FP128TypeKind => BasicMetadataValueEnum::FloatValue(FloatValue::new(x)),
            LLVMTypeKind::LLVMIntegerTypeKind => BasicMetadataValueEnum::IntValue(IntValue::new(x)),
            LLVMTypeKind::LLVMStructTypeKind => BasicMetadataValueEnum::StructValue(StructValue::new(x)),
            LLVMTypeKind::LLVMPointerTypeKind => BasicMetadataValueEnum::PointerValue(PointerValue::new(x)),
            LLVMTypeKind::LLVMArrayTypeKind => BasicMetadataValueEnum::ArrayValue(ArrayValue::new(x)),
            LLVMTypeKind::LLVMVectorTypeKind => BasicMetadataValueEnum::VectorValue(VectorValue::new(x)),
            LLVMTypeKind::LLVMMetadataTypeKind => BasicMetadataValueEnum::MetadataValue(MetadataValue::new(x)),
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
        if let BasicMetadataValueEnum::ArrayValue(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the ArrayValue variant", self)
        }
    }
    pub fn into_int_value(self) -> IntValue<'ctx> {
        if let BasicMetadataValueEnum::IntValue(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the IntValue variant", self)
        }
    }
    pub fn into_float_value(self) -> FloatValue<'ctx> {
        if let BasicMetadataValueEnum::FloatValue(x) = self {
            x
        } else {
            panic!("Found {:?} but expected FloatValue variant", self)
        }
    }
    pub fn into_pointer_value(self) -> PointerValue<'ctx> {
        if let BasicMetadataValueEnum::PointerValue(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the PointerValue variant", self)
        }
    }
    pub fn into_struct_value(self) -> StructValue<'ctx> {
        if let BasicMetadataValueEnum::StructValue(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the StructValue variant", self)
        }
    }
    pub fn into_vector_value(self) -> VectorValue<'ctx> {
        if let BasicMetadataValueEnum::VectorValue(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the VectorValue variant", self)
        }
    }
    pub fn into_metadata_value(self) -> MetadataValue<'ctx> {
        if let BasicMetadataValueEnum::MetadataValue(x) = self {
            x
        } else {
            panic!("Found {:?} but expected MetaData variant", self)
        }
    }
}
impl<'ctx> From<BasicValueEnum<'ctx>> for BasicMetadataValueEnum<'ctx> {
    fn from(x: BasicValueEnum<'ctx>) -> Self {
        unsafe { BasicMetadataValueEnum::new(x.as_value_ref()) }
    }
}
impl<'ctx> TryFrom<AnyValueEnum<'ctx>> for BasicMetadataValueEnum<'ctx> {
    type Error = ();
    fn try_from(x: AnyValueEnum<'ctx>) -> Result<Self, Self::Error> {
        use AnyValueEnum::*;
        Ok(match x {
            ArrayValue(x) => x.into(),
            IntValue(x) => x.into(),
            FloatValue(x) => x.into(),
            PointerValue(x) => x.into(),
            StructValue(x) => x.into(),
            VectorValue(x) => x.into(),
            MetadataValue(x) => x.into(),
            PhiValue(_) | FunctionValue(_) | InstructionValue(_) => return Err(()),
        })
    }
}
impl Display for BasicMetadataValueEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

enum_value_set! {AggregateValueEnum: ArrayValue, StructValue}
impl<'ctx> AggregateValueEnum<'ctx> {
    pub unsafe fn new(x: LLVMValueRef) -> Self {
        match LLVMGetTypeKind(LLVMTypeOf(x)) {
            LLVMTypeKind::LLVMArrayTypeKind => AggregateValueEnum::ArrayValue(ArrayValue::new(x)),
            LLVMTypeKind::LLVMStructTypeKind => AggregateValueEnum::StructValue(StructValue::new(x)),
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
        if let AggregateValueEnum::ArrayValue(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the ArrayValue variant", self)
        }
    }
    pub fn into_struct_value(self) -> StructValue<'ctx> {
        if let AggregateValueEnum::StructValue(x) = self {
            x
        } else {
            panic!("Found {:?} but expected the StructValue variant", self)
        }
    }
}
impl Display for AggregateValueEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

enum_value_set! {AnyValueEnum: ArrayValue, IntValue, FloatValue, PhiValue, FunctionValue, PointerValue, StructValue, VectorValue, InstructionValue, MetadataValue}
impl<'ctx> AnyValueEnum<'ctx> {
    pub unsafe fn new(x: LLVMValueRef) -> Self {
        match LLVMGetTypeKind(LLVMTypeOf(x)) {
            LLVMTypeKind::LLVMFloatTypeKind
            | LLVMTypeKind::LLVMFP128TypeKind
            | LLVMTypeKind::LLVMDoubleTypeKind
            | LLVMTypeKind::LLVMHalfTypeKind
            | LLVMTypeKind::LLVMX86_FP80TypeKind
            | LLVMTypeKind::LLVMPPC_FP128TypeKind => AnyValueEnum::FloatValue(FloatValue::new(x)),
            LLVMTypeKind::LLVMIntegerTypeKind => AnyValueEnum::IntValue(IntValue::new(x)),
            LLVMTypeKind::LLVMStructTypeKind => AnyValueEnum::StructValue(StructValue::new(x)),
            LLVMTypeKind::LLVMPointerTypeKind => match LLVMGetValueKind(x) {
                LLVMValueKind::LLVMFunctionValueKind => AnyValueEnum::FunctionValue(FunctionValue::new(x).unwrap()),
                _ => AnyValueEnum::PointerValue(PointerValue::new(x)),
            },
            LLVMTypeKind::LLVMArrayTypeKind => AnyValueEnum::ArrayValue(ArrayValue::new(x)),
            LLVMTypeKind::LLVMVectorTypeKind => AnyValueEnum::VectorValue(VectorValue::new(x)),
            LLVMTypeKind::LLVMFunctionTypeKind => AnyValueEnum::FunctionValue(FunctionValue::new(x).unwrap()),
            LLVMTypeKind::LLVMVoidTypeKind => {
                if LLVMIsAInstruction(x).is_null() {
                    panic!("Void value isn't an instruction.");
                }
                AnyValueEnum::InstructionValue(InstructionValue::new(x))
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
impl<'ctx> From<BasicValueEnum<'ctx>> for AnyValueEnum<'ctx> {
    fn from(x: BasicValueEnum<'ctx>) -> Self {
        unsafe { AnyValueEnum::new(x.as_value_ref()) }
    }
}
impl Display for AnyValueEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct FloatValue<'ctx> {
    val: Value<'ctx>,
}
impl<'ctx> FloatValue<'ctx> {
    pub unsafe fn new(x: LLVMValueRef) -> Self {
        assert!(!x.is_null());
        FloatValue { val: Value::new(x) }
    }
    pub fn get_name(&self) -> &CStr {
        self.val.get_name()
    }
    pub fn set_name(&self, name: &str) {
        self.val.set_name(name)
    }
    pub fn get_type(self) -> FloatType<'ctx> {
        unsafe { FloatType::new(self.val.get_type()) }
    }
    pub fn is_null(self) -> bool {
        self.val.is_null()
    }
    pub fn is_undef(self) -> bool {
        self.val.is_undef()
    }
    pub fn print_to_stderr(self) {
        self.val.print_to_stderr()
    }
    pub fn as_instruction(self) -> Option<InstructionValue<'ctx>> {
        self.val.as_instruction()
    }
    pub fn const_cast(self, float_type: FloatType<'ctx>) -> Self {
        unsafe { FloatValue::new(LLVMConstFPCast(self.as_value_ref(), float_type.as_type_ref())) }
    }
    pub fn const_to_unsigned_int(self, int_type: IntType<'ctx>) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstFPToUI(self.as_value_ref(), int_type.as_type_ref())) }
    }
    pub fn const_to_signed_int(self, int_type: IntType<'ctx>) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstFPToSI(self.as_value_ref(), int_type.as_type_ref())) }
    }
    pub fn const_truncate(self, float_type: FloatType<'ctx>) -> FloatValue<'ctx> {
        unsafe { FloatValue::new(LLVMConstFPTrunc(self.as_value_ref(), float_type.as_type_ref())) }
    }
    pub fn const_extend(self, float_type: FloatType<'ctx>) -> FloatValue<'ctx> {
        unsafe { FloatValue::new(LLVMConstFPExt(self.as_value_ref(), float_type.as_type_ref())) }
    }
    pub fn const_compare(self, op: FloatPredicate, rhs: FloatValue<'ctx>) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstFCmp(op.into(), self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn is_const(self) -> bool {
        self.val.is_const()
    }
    pub fn get_constant(self) -> Option<(f64, bool)> {
        if !self.is_const() {
            return None;
        }
        let mut lossy = 0;
        let constant = unsafe { LLVMConstRealGetDouble(self.as_value_ref(), &mut lossy) };
        Some((constant, lossy == 1))
    }
    pub fn replace_all_uses_with(self, x: FloatValue<'ctx>) {
        self.val.replace_all_uses_with(x.as_value_ref())
    }
}
impl<'ctx> TryFrom<InstructionValue<'ctx>> for FloatValue<'ctx> {
    type Error = ();
    fn try_from(x: InstructionValue) -> Result<Self, Self::Error> {
        if x.get_type().is_float_type() {
            unsafe { Ok(FloatValue::new(x.as_value_ref())) }
        } else {
            Err(())
        }
    }
}
impl Display for FloatValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsValueRef for FloatValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.val.raw
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub struct FunctionValue<'ctx> {
    val: Value<'ctx>,
}
impl<'ctx> FunctionValue<'ctx> {
    pub unsafe fn new(x: LLVMValueRef) -> Option<Self> {
        if x.is_null() {
            return None;
        }
        assert!(!LLVMIsAFunction(x).is_null());
        Some(FunctionValue { val: Value::new(x) })
    }
    pub fn get_linkage(self) -> Linkage {
        unsafe { LLVMGetLinkage(self.as_value_ref()).into() }
    }
    pub fn set_linkage(self, linkage: Linkage) {
        unsafe { LLVMSetLinkage(self.as_value_ref(), linkage.into()) }
    }
    pub fn is_null(self) -> bool {
        self.val.is_null()
    }
    pub fn is_undef(self) -> bool {
        self.val.is_undef()
    }
    pub fn print_to_stderr(self) {
        self.val.print_to_stderr()
    }
    pub fn verify(self, print: bool) -> bool {
        let action = if print {
            LLVMVerifierFailureAction::LLVMPrintMessageAction
        } else {
            LLVMVerifierFailureAction::LLVMReturnStatusAction
        };
        let code = unsafe { LLVMVerifyFunction(self.val.raw, action) };
        code != 1
    }
    pub fn get_next_function(self) -> Option<Self> {
        unsafe { FunctionValue::new(LLVMGetNextFunction(self.as_value_ref())) }
    }
    pub fn get_previous_function(self) -> Option<Self> {
        unsafe { FunctionValue::new(LLVMGetPreviousFunction(self.as_value_ref())) }
    }
    pub fn get_first_param(self) -> Option<BasicValueEnum<'ctx>> {
        let y = unsafe { LLVMGetFirstParam(self.as_value_ref()) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(BasicValueEnum::new(y)) }
    }
    pub fn get_last_param(self) -> Option<BasicValueEnum<'ctx>> {
        let y = unsafe { LLVMGetLastParam(self.as_value_ref()) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(BasicValueEnum::new(y)) }
    }
    pub fn get_first_basic_block(self) -> Option<BasicBlock<'ctx>> {
        unsafe { BasicBlock::new(LLVMGetFirstBasicBlock(self.as_value_ref())) }
    }
    pub fn get_nth_param(self, nth: u32) -> Option<BasicValueEnum<'ctx>> {
        let count = self.count_params();
        if nth + 1 > count {
            return None;
        }
        unsafe { Some(BasicValueEnum::new(LLVMGetParam(self.as_value_ref(), nth))) }
    }
    pub fn count_params(self) -> u32 {
        unsafe { LLVMCountParams(self.val.raw) }
    }
    pub fn count_basic_blocks(self) -> u32 {
        unsafe { LLVMCountBasicBlocks(self.as_value_ref()) }
    }
    pub fn get_basic_blocks(self) -> Vec<BasicBlock<'ctx>> {
        let n = self.count_basic_blocks();
        let mut ys: Vec<LLVMBasicBlockRef> = Vec::with_capacity(n as usize);
        let ptr = ys.as_mut_ptr();
        forget(ys);
        let ys = unsafe {
            LLVMGetBasicBlocks(self.as_value_ref(), ptr);
            Vec::from_raw_parts(ptr, n as usize, n as usize)
        };
        ys.iter().map(|x| unsafe { BasicBlock::new(*x).unwrap() }).collect()
    }
    pub fn get_param_iter(self) -> ParamValueIter<'ctx> {
        ParamValueIter {
            raw: self.val.raw,
            start: true,
            _marker: marker::PhantomData,
        }
    }
    pub fn get_params(self) -> Vec<BasicValueEnum<'ctx>> {
        let n = self.count_params();
        let mut ys: Vec<LLVMValueRef> = Vec::with_capacity(n as usize);
        let ptr = ys.as_mut_ptr();
        forget(ys);
        let y = unsafe {
            LLVMGetParams(self.as_value_ref(), ptr);
            Vec::from_raw_parts(ptr, n as usize, n as usize)
        };
        y.iter().map(|x| unsafe { BasicValueEnum::new(*x) }).collect()
    }
    pub fn get_last_basic_block(self) -> Option<BasicBlock<'ctx>> {
        unsafe { BasicBlock::new(LLVMGetLastBasicBlock(self.val.raw)) }
    }
    pub fn get_name(&self) -> &CStr {
        self.val.get_name()
    }
    pub fn view_function_cfg(self) {
        unsafe { LLVMViewFunctionCFG(self.as_value_ref()) }
    }
    pub fn view_function_cfg_only(self) {
        unsafe { LLVMViewFunctionCFGOnly(self.as_value_ref()) }
    }
    pub unsafe fn delete(self) {
        LLVMDeleteFunction(self.as_value_ref())
    }
    pub fn get_type(self) -> FunctionType<'ctx> {
        unsafe { FunctionType::new(llvm_lib::core::LLVMGlobalGetValueType(self.as_value_ref())) }
    }
    pub fn has_personality_function(self) -> bool {
        unsafe { LLVMHasPersonalityFn(self.as_value_ref()) == 1 }
    }
    pub fn get_personality_function(self) -> Option<FunctionValue<'ctx>> {
        if !self.has_personality_function() {
            return None;
        }
        unsafe { FunctionValue::new(LLVMGetPersonalityFn(self.as_value_ref())) }
    }
    pub fn set_personality_function(self, personality_fn: FunctionValue<'ctx>) {
        unsafe { LLVMSetPersonalityFn(self.as_value_ref(), personality_fn.as_value_ref()) }
    }
    pub fn get_intrinsic_id(self) -> u32 {
        unsafe { LLVMGetIntrinsicID(self.as_value_ref()) }
    }
    pub fn get_call_conventions(self) -> u32 {
        unsafe { LLVMGetFunctionCallConv(self.as_value_ref()) }
    }
    pub fn set_call_conventions(self, call_conventions: u32) {
        unsafe { LLVMSetFunctionCallConv(self.as_value_ref(), call_conventions) }
    }
    pub fn get_gc(&self) -> &CStr {
        unsafe { CStr::from_ptr(LLVMGetGC(self.as_value_ref())) }
    }
    pub fn set_gc(self, gc: &str) {
        let y = to_c_str(gc);
        unsafe { LLVMSetGC(self.as_value_ref(), y.as_ptr()) }
    }
    pub fn replace_all_uses_with(self, x: FunctionValue<'ctx>) {
        self.val.replace_all_uses_with(x.as_value_ref())
    }
    pub fn add_attribute(self, loc: AttributeLoc, a: Attribute) {
        unsafe { LLVMAddAttributeAtIndex(self.as_value_ref(), loc.get_index(), a.raw) }
    }
    pub fn count_attributes(self, loc: AttributeLoc) -> u32 {
        unsafe { LLVMGetAttributeCountAtIndex(self.as_value_ref(), loc.get_index()) }
    }
    pub fn attributes(self, loc: AttributeLoc) -> Vec<Attribute> {
        use std::mem::{ManuallyDrop, MaybeUninit};
        let count = self.count_attributes(loc) as usize;
        let mut y: Vec<MaybeUninit<Attribute>> = vec![MaybeUninit::uninit(); count];
        unsafe { LLVMGetAttributesAtIndex(self.as_value_ref(), loc.get_index(), y.as_mut_ptr() as *mut _) }
        unsafe {
            let mut y = ManuallyDrop::new(y);
            Vec::from_raw_parts(y.as_mut_ptr() as *mut Attribute, y.len(), y.capacity())
        }
    }
    pub fn remove_string_attribute(self, loc: AttributeLoc, key: &str) {
        unsafe {
            LLVMRemoveStringAttributeAtIndex(
                self.as_value_ref(),
                loc.get_index(),
                key.as_ptr() as *const ::libc::c_char,
                key.len() as u32,
            )
        }
    }
    pub fn remove_enum_attribute(self, loc: AttributeLoc, kind_id: u32) {
        unsafe { LLVMRemoveEnumAttributeAtIndex(self.as_value_ref(), loc.get_index(), kind_id) }
    }
    pub fn get_enum_attribute(self, loc: AttributeLoc, kind_id: u32) -> Option<Attribute> {
        let y = unsafe { LLVMGetEnumAttributeAtIndex(self.as_value_ref(), loc.get_index(), kind_id) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(Attribute::new(y)) }
    }
    pub fn get_string_attribute(self, loc: AttributeLoc, key: &str) -> Option<Attribute> {
        let y = unsafe {
            LLVMGetStringAttributeAtIndex(
                self.as_value_ref(),
                loc.get_index(),
                key.as_ptr() as *const ::libc::c_char,
                key.len() as u32,
            )
        };
        if y.is_null() {
            return None;
        }
        unsafe { Some(Attribute::new(y)) }
    }
    pub fn set_param_alignment(self, idx: u32, align: u32) {
        if let Some(x) = self.get_nth_param(idx) {
            unsafe { LLVMSetParamAlignment(x.as_value_ref(), align) }
        }
    }
    pub fn as_global_value(self) -> GlobalValue<'ctx> {
        unsafe { GlobalValue::new(self.as_value_ref()) }
    }
    pub fn set_subprogram(self, x: DISubprogram<'ctx>) {
        unsafe { LLVMSetSubprogram(self.as_value_ref(), x.raw) }
    }
    pub fn get_subprogram(self) -> Option<DISubprogram<'ctx>> {
        let raw = unsafe { LLVMGetSubprogram(self.as_value_ref()) };
        if raw.is_null() {
            None
        } else {
            Some(DISubprogram {
                raw,
                _marker: marker::PhantomData,
            })
        }
    }
    pub fn get_section(&self) -> Option<&CStr> {
        self.val.get_section()
    }
    pub fn set_section(self, section: Option<&str>) {
        self.val.set_section(section)
    }
}
impl Display for FunctionValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
impl fmt::Debug for FunctionValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let val = self.print_to_string();
        let ty = self.get_type();
        let name = self.get_name();
        let is_const = unsafe { LLVMIsConstant(self.val.raw) == 1 };
        let is_null = self.is_null();
        f.debug_struct("FunctionValue")
            .field("name", &name)
            .field("address", &self.as_value_ref())
            .field("is_const", &is_const)
            .field("is_null", &is_null)
            .field("llvm_value", &val)
            .field("llvm_type", &ty.print_to_string())
            .finish()
    }
}
unsafe impl AsValueRef for FunctionValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.val.raw
    }
}

#[derive(Debug)]
pub struct ParamValueIter<'ctx> {
    raw: LLVMValueRef,
    start: bool,
    _marker: marker::PhantomData<&'ctx ()>,
}
impl<'ctx> Iterator for ParamValueIter<'ctx> {
    type Item = BasicValueEnum<'ctx>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.start {
            let first_value = unsafe { LLVMGetFirstParam(self.raw) };
            if first_value.is_null() {
                return None;
            }
            self.start = false;
            self.raw = first_value;
            return unsafe { Some(Self::Item::new(first_value)) };
        }
        let next_value = unsafe { LLVMGetNextParam(self.raw) };
        if next_value.is_null() {
            return None;
        }
        self.raw = next_value;
        unsafe { Some(Self::Item::new(next_value)) }
    }
}

#[derive(Debug)]
pub struct GenericValue<'ctx> {
    pub raw: LLVMGenericValueRef,
    _marker: marker::PhantomData<&'ctx ()>,
}
impl<'ctx> GenericValue<'ctx> {
    pub unsafe fn new(x: LLVMGenericValueRef) -> Self {
        assert!(!x.is_null());
        GenericValue {
            raw: x,
            _marker: marker::PhantomData,
        }
    }
    pub fn int_width(self) -> u32 {
        unsafe { LLVMGenericValueIntWidth(self.raw) }
    }
    pub unsafe fn create_generic_value_of_pointer<T>(x: &mut T) -> Self {
        let x = LLVMCreateGenericValueOfPointer(x as *mut _ as *mut c_void);
        GenericValue::new(x)
    }
    pub fn as_int(self, is_signed: bool) -> u64 {
        unsafe { LLVMGenericValueToInt(self.raw, is_signed as i32) }
    }
    pub fn as_float(self, float_type: &FloatType<'ctx>) -> f64 {
        unsafe { LLVMGenericValueToFloat(float_type.as_type_ref(), self.raw) }
    }
    pub unsafe fn into_pointer<T>(self) -> *mut T {
        LLVMGenericValueToPointer(self.raw) as *mut T
    }
}
impl Drop for GenericValue<'_> {
    fn drop(&mut self) {
        unsafe { LLVMDisposeGenericValue(self.raw) }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct GlobalValue<'ctx> {
    val: Value<'ctx>,
}
impl<'ctx> GlobalValue<'ctx> {
    pub unsafe fn new(x: LLVMValueRef) -> Self {
        assert!(!x.is_null());
        GlobalValue { val: Value::new(x) }
    }
    pub fn get_name(&self) -> &CStr {
        self.val.get_name()
    }
    pub fn set_name(&self, name: &str) {
        self.val.set_name(name)
    }
    pub fn get_previous_global(self) -> Option<GlobalValue<'ctx>> {
        let y = unsafe { LLVMGetPreviousGlobal(self.as_value_ref()) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(GlobalValue::new(y)) }
    }
    pub fn get_next_global(self) -> Option<GlobalValue<'ctx>> {
        let y = unsafe { LLVMGetNextGlobal(self.as_value_ref()) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(GlobalValue::new(y)) }
    }
    pub fn get_dll_storage_class(self) -> DLLStorageClass {
        let y = unsafe { LLVMGetDLLStorageClass(self.as_value_ref()) };
        DLLStorageClass::new(y)
    }
    pub fn set_dll_storage_class(self, x: DLLStorageClass) {
        unsafe { LLVMSetDLLStorageClass(self.as_value_ref(), x.into()) }
    }
    pub fn get_initializer(self) -> Option<BasicValueEnum<'ctx>> {
        let y = unsafe { LLVMGetInitializer(self.as_value_ref()) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(BasicValueEnum::new(y)) }
    }
    pub fn set_initializer(self, x: &dyn BasicValue<'ctx>) {
        unsafe { LLVMSetInitializer(self.as_value_ref(), x.as_value_ref()) }
    }
    pub fn is_thread_local(self) -> bool {
        unsafe { LLVMIsThreadLocal(self.as_value_ref()) == 1 }
    }
    pub fn set_thread_local(self, is_thread_local: bool) {
        unsafe { LLVMSetThreadLocal(self.as_value_ref(), is_thread_local as i32) }
    }
    pub fn get_thread_local_mode(self) -> Option<ThreadLocalMode> {
        let y = unsafe { LLVMGetThreadLocalMode(self.as_value_ref()) };
        ThreadLocalMode::new(y)
    }
    pub fn set_thread_local_mode(self, x: Option<ThreadLocalMode>) {
        let x = match x {
            Some(x) => x.as_llvm_mode(),
            None => LLVMThreadLocalMode::LLVMNotThreadLocal,
        };
        unsafe { LLVMSetThreadLocalMode(self.as_value_ref(), x) }
    }
    pub fn is_declaration(self) -> bool {
        unsafe { LLVMIsDeclaration(self.as_value_ref()) == 1 }
    }
    pub fn has_unnamed_addr(self) -> bool {
        unsafe { LLVMGetUnnamedAddress(self.as_value_ref()) == LLVMUnnamedAddr::LLVMGlobalUnnamedAddr }
    }
    pub fn set_unnamed_addr(self, has_unnamed_addr: bool) {
        unsafe {
            if has_unnamed_addr {
                LLVMSetUnnamedAddress(self.as_value_ref(), UnnamedAddress::Global.into())
            } else {
                LLVMSetUnnamedAddress(self.as_value_ref(), UnnamedAddress::None.into())
            }
        }
    }
    pub fn is_constant(self) -> bool {
        unsafe { LLVMIsGlobalConstant(self.as_value_ref()) == 1 }
    }
    pub fn set_constant(self, is_constant: bool) {
        unsafe { LLVMSetGlobalConstant(self.as_value_ref(), is_constant as i32) }
    }
    pub fn is_externally_initialized(self) -> bool {
        unsafe { LLVMIsExternallyInitialized(self.as_value_ref()) == 1 }
    }
    pub fn set_externally_initialized(self, externally_initialized: bool) {
        unsafe { LLVMSetExternallyInitialized(self.as_value_ref(), externally_initialized as i32) }
    }
    pub fn set_visibility(self, x: GlobalVisibility) {
        unsafe { LLVMSetVisibility(self.as_value_ref(), x.into()) }
    }
    pub fn get_visibility(self) -> GlobalVisibility {
        let y = unsafe { LLVMGetVisibility(self.as_value_ref()) };
        GlobalVisibility::new(y)
    }
    pub fn get_section(&self) -> Option<&CStr> {
        self.val.get_section()
    }
    pub fn set_section(self, section: Option<&str>) {
        self.val.set_section(section)
    }
    pub unsafe fn delete(self) {
        LLVMDeleteGlobal(self.as_value_ref())
    }
    pub fn as_pointer_value(self) -> PointerValue<'ctx> {
        unsafe { PointerValue::new(self.as_value_ref()) }
    }
    pub fn get_alignment(self) -> u32 {
        unsafe { LLVMGetAlignment(self.as_value_ref()) }
    }
    pub fn set_alignment(self, alignment: u32) {
        unsafe { LLVMSetAlignment(self.as_value_ref(), alignment) }
    }
    pub fn set_metadata(self, metadata: MetadataValue<'ctx>, kind_id: u32) {
        unsafe { LLVMGlobalSetMetadata(self.as_value_ref(), kind_id, metadata.as_metadata_ref()) }
    }
    pub fn get_comdat(self) -> Option<Comdat> {
        let y = unsafe { LLVMGetComdat(self.as_value_ref()) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(Comdat::new(y)) }
    }
    pub fn set_comdat(self, x: Comdat) {
        unsafe { LLVMSetComdat(self.as_value_ref(), x.0) }
    }
    pub fn get_unnamed_address(self) -> UnnamedAddress {
        let y = unsafe { LLVMGetUnnamedAddress(self.as_value_ref()) };
        UnnamedAddress::new(y)
    }
    pub fn set_unnamed_address(self, x: UnnamedAddress) {
        unsafe { LLVMSetUnnamedAddress(self.as_value_ref(), x.into()) }
    }
    pub fn get_linkage(self) -> Linkage {
        unsafe { LLVMGetLinkage(self.as_value_ref()).into() }
    }
    pub fn set_linkage(self, x: Linkage) {
        unsafe { LLVMSetLinkage(self.as_value_ref(), x.into()) }
    }
}
impl Display for GlobalValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsValueRef for GlobalValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.val.raw
    }
}

#[llvm_enum(LLVMUnnamedAddr)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum UnnamedAddress {
    #[llvm_variant(LLVMNoUnnamedAddr)]
    None,
    #[llvm_variant(LLVMLocalUnnamedAddr)]
    Local,
    #[llvm_variant(LLVMGlobalUnnamedAddr)]
    Global,
}

#[llvm_enum(LLVMOpcode)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstructionOpcode {
    Add,
    AddrSpaceCast,
    Alloca,
    And,
    AShr,
    AtomicCmpXchg,
    AtomicRMW,
    BitCast,
    Br,
    Call,
    CallBr,
    CatchPad,
    CatchRet,
    CatchSwitch,
    CleanupPad,
    CleanupRet,
    ExtractElement,
    ExtractValue,
    FNeg,
    FAdd,
    FCmp,
    FDiv,
    Fence,
    FMul,
    FPExt,
    FPToSI,
    FPToUI,
    FPTrunc,
    Freeze,
    FRem,
    FSub,
    GetElementPtr,
    ICmp,
    IndirectBr,
    InsertElement,
    InsertValue,
    IntToPtr,
    Invoke,
    LandingPad,
    Load,
    LShr,
    Mul,
    Or,
    #[llvm_variant(LLVMPHI)]
    Phi,
    PtrToInt,
    Resume,
    #[llvm_variant(LLVMRet)]
    Return,
    SDiv,
    Select,
    SExt,
    Shl,
    ShuffleVector,
    SIToFP,
    SRem,
    Store,
    Sub,
    Switch,
    Trunc,
    UDiv,
    UIToFP,
    Unreachable,
    URem,
    UserOp1,
    UserOp2,
    VAArg,
    Xor,
    ZExt,
}

#[derive(Debug, PartialEq, Eq, Copy, Hash)]
pub struct InstructionValue<'ctx> {
    val: Value<'ctx>,
}
impl<'ctx> InstructionValue<'ctx> {
    fn is_a_load_inst(self) -> bool {
        !unsafe { LLVMIsALoadInst(self.as_value_ref()) }.is_null()
    }
    fn is_a_store_inst(self) -> bool {
        !unsafe { LLVMIsAStoreInst(self.as_value_ref()) }.is_null()
    }
    fn is_a_alloca_inst(self) -> bool {
        !unsafe { LLVMIsAAllocaInst(self.as_value_ref()) }.is_null()
    }
    fn is_a_atomicrmw_inst(self) -> bool {
        !unsafe { LLVMIsAAtomicRMWInst(self.as_value_ref()) }.is_null()
    }
    fn is_a_cmpxchg_inst(self) -> bool {
        !unsafe { LLVMIsAAtomicCmpXchgInst(self.as_value_ref()) }.is_null()
    }
    pub unsafe fn new(x: LLVMValueRef) -> Self {
        debug_assert!(!x.is_null());
        let val = Value::new(x);
        debug_assert!(val.is_instruction());
        InstructionValue { val }
    }
    pub fn get_name(&self) -> Option<&CStr> {
        if self.get_type().is_void_type() {
            None
        } else {
            Some(self.val.get_name())
        }
    }
    pub fn get_instruction_with_name(&self, name: &str) -> Option<InstructionValue<'ctx>> {
        if let Some(x) = self.get_name() {
            if x.to_str() == Ok(name) {
                return Some(*self);
            }
        }
        return self.get_next_instruction()?.get_instruction_with_name(name);
    }
    pub fn set_name(&self, name: &str) -> Result<(), &'static str> {
        if self.get_type().is_void_type() {
            Err("Cannot set name of a void-type instruction!")
        } else {
            self.val.set_name(name);
            Ok(())
        }
    }
    pub fn get_type(self) -> AnyTypeEnum<'ctx> {
        unsafe { AnyTypeEnum::new(self.val.get_type()) }
    }
    pub fn get_opcode(self) -> InstructionOpcode {
        let y = unsafe { LLVMGetInstructionOpcode(self.as_value_ref()) };
        InstructionOpcode::new(y)
    }
    pub fn get_previous_instruction(self) -> Option<Self> {
        let y = unsafe { LLVMGetPreviousInstruction(self.as_value_ref()) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(InstructionValue::new(y)) }
    }
    pub fn get_next_instruction(self) -> Option<Self> {
        let y = unsafe { LLVMGetNextInstruction(self.as_value_ref()) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(InstructionValue::new(y)) }
    }
    pub fn erase_from_basic_block(self) {
        unsafe { LLVMInstructionEraseFromParent(self.as_value_ref()) }
    }
    pub fn remove_from_basic_block(self) {
        unsafe { LLVMInstructionRemoveFromParent(self.as_value_ref()) }
    }
    pub fn get_parent(self) -> Option<BasicBlock<'ctx>> {
        unsafe { BasicBlock::new(LLVMGetInstructionParent(self.as_value_ref())) }
    }
    pub fn is_tail_call(self) -> bool {
        if self.get_opcode() == InstructionOpcode::Call {
            unsafe { LLVMIsTailCall(self.as_value_ref()) == 1 }
        } else {
            false
        }
    }
    pub fn replace_all_uses_with(self, x: &InstructionValue<'ctx>) {
        self.val.replace_all_uses_with(x.as_value_ref())
    }
    pub fn get_volatile(self) -> Result<bool, &'static str> {
        if !self.is_a_load_inst() && !self.is_a_store_inst() && !self.is_a_atomicrmw_inst() && !self.is_a_cmpxchg_inst()
        {
            return Err("Value is not a load, store, atomicrmw or cmpxchg.");
        }
        Ok(unsafe { LLVMGetVolatile(self.as_value_ref()) } == 1)
    }
    pub fn set_volatile(self, volatile: bool) -> Result<(), &'static str> {
        if !self.is_a_load_inst() && !self.is_a_store_inst() && !self.is_a_atomicrmw_inst() && !self.is_a_cmpxchg_inst()
        {
            return Err("Value is not a load, store, atomicrmw or cmpxchg.");
        }
        unsafe { LLVMSetVolatile(self.as_value_ref(), volatile as i32) };
        Ok(())
    }
    pub fn get_alignment(self) -> Result<u32, &'static str> {
        if !self.is_a_alloca_inst() && !self.is_a_load_inst() && !self.is_a_store_inst() {
            return Err("Value is not an alloca, load or store.");
        }
        Ok(unsafe { LLVMGetAlignment(self.as_value_ref()) })
    }
    pub fn set_alignment(self, alignment: u32) -> Result<(), &'static str> {
        if !alignment.is_power_of_two() && alignment != 0 {
            return Err("Alignment is not a power of 2!");
        }
        if !self.is_a_alloca_inst() && !self.is_a_load_inst() && !self.is_a_store_inst() {
            return Err("Value is not an alloca, load or store.");
        }
        unsafe { LLVMSetAlignment(self.as_value_ref(), alignment) };
        Ok(())
    }
    pub fn get_atomic_ordering(self) -> Result<AtomicOrdering, &'static str> {
        if !self.is_a_load_inst() && !self.is_a_store_inst() {
            return Err("Value is not a load or store.");
        }
        Ok(unsafe { LLVMGetOrdering(self.as_value_ref()) }.into())
    }
    pub fn set_atomic_ordering(self, ordering: AtomicOrdering) -> Result<(), &'static str> {
        if !self.is_a_load_inst() && !self.is_a_store_inst() {
            return Err("Value is not a load or store instruction.");
        }
        match ordering {
            AtomicOrdering::Release if self.is_a_load_inst() => {
                return Err("The release ordering is not valid on load instructions.")
            },
            AtomicOrdering::AcquireRelease => {
                return Err("The acq_rel ordering is not valid on load or store instructions.")
            },
            AtomicOrdering::Acquire if self.is_a_store_inst() => {
                return Err("The acquire ordering is not valid on store instructions.")
            },
            _ => {},
        };
        unsafe { LLVMSetOrdering(self.as_value_ref(), ordering.into()) };
        Ok(())
    }
    pub fn get_num_operands(self) -> u32 {
        unsafe { LLVMGetNumOperands(self.as_value_ref()) as u32 }
    }
    pub fn get_operand(self, index: u32) -> Option<Either<BasicValueEnum<'ctx>, BasicBlock<'ctx>>> {
        let num_operands = self.get_num_operands();
        if index >= num_operands {
            return None;
        }
        let operand = unsafe { LLVMGetOperand(self.as_value_ref(), index) };
        if operand.is_null() {
            return None;
        }
        let is_basic_block = unsafe { !LLVMIsABasicBlock(operand).is_null() };
        if is_basic_block {
            let bb = unsafe { BasicBlock::new(LLVMValueAsBasicBlock(operand)) };
            Some(Right(bb.expect("BasicBlock should always be valid")))
        } else {
            Some(Left(unsafe { BasicValueEnum::new(operand) }))
        }
    }
    pub fn set_operand<BV: BasicValue<'ctx>>(self, index: u32, val: BV) -> bool {
        let num_operands = self.get_num_operands();
        if index >= num_operands {
            return false;
        }
        unsafe { LLVMSetOperand(self.as_value_ref(), index, val.as_value_ref()) }
        true
    }
    pub fn get_operand_use(self, index: u32) -> Option<BasicValueUse<'ctx>> {
        let num_operands = self.get_num_operands();
        if index >= num_operands {
            return None;
        }
        let use_ = unsafe { LLVMGetOperandUse(self.as_value_ref(), index) };
        if use_.is_null() {
            return None;
        }
        unsafe { Some(BasicValueUse::new(use_)) }
    }
    pub fn get_first_use(self) -> Option<BasicValueUse<'ctx>> {
        self.val.get_first_use()
    }
    pub fn get_icmp_predicate(self) -> Option<IntPredicate> {
        if self.get_opcode() == InstructionOpcode::ICmp {
            let pred = unsafe { LLVMGetICmpPredicate(self.as_value_ref()) };
            Some(IntPredicate::new(pred))
        } else {
            None
        }
    }
    pub fn get_fcmp_predicate(self) -> Option<FloatPredicate> {
        if self.get_opcode() == InstructionOpcode::FCmp {
            let pred = unsafe { LLVMGetFCmpPredicate(self.as_value_ref()) };
            Some(FloatPredicate::new(pred))
        } else {
            None
        }
    }
    pub fn has_metadata(self) -> bool {
        unsafe { LLVMHasMetadata(self.val.raw) == 1 }
    }
    pub fn get_metadata(self, kind_id: u32) -> Option<MetadataValue<'ctx>> {
        let metadata_value = unsafe { LLVMGetMetadata(self.val.raw, kind_id) };
        if metadata_value.is_null() {
            return None;
        }
        unsafe { Some(MetadataValue::new(metadata_value)) }
    }
    pub fn set_metadata(self, metadata: MetadataValue<'ctx>, kind_id: u32) -> Result<(), &'static str> {
        if !metadata.is_node() {
            return Err("metadata is expected to be a node.");
        }
        unsafe {
            LLVMSetMetadata(self.val.raw, kind_id, metadata.as_value_ref());
        }
        Ok(())
    }
}
impl Clone for InstructionValue<'_> {
    fn clone(&self) -> Self {
        unsafe { InstructionValue::new(LLVMInstructionClone(self.as_value_ref())) }
    }
}
impl Display for InstructionValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsValueRef for InstructionValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.val.raw
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct IntValue<'ctx> {
    val: Value<'ctx>,
}
impl<'ctx> IntValue<'ctx> {
    pub unsafe fn new(x: LLVMValueRef) -> Self {
        assert!(!x.is_null());
        IntValue { val: Value::new(x) }
    }
    pub fn get_name(&self) -> &CStr {
        self.val.get_name()
    }
    pub fn set_name(&self, name: &str) {
        self.val.set_name(name)
    }
    pub fn get_type(self) -> IntType<'ctx> {
        unsafe { IntType::new(self.val.get_type()) }
    }
    pub fn is_null(self) -> bool {
        self.val.is_null()
    }
    pub fn is_undef(self) -> bool {
        self.val.is_undef()
    }
    pub fn print_to_stderr(self) {
        self.val.print_to_stderr()
    }
    pub fn as_instruction(self) -> Option<InstructionValue<'ctx>> {
        self.val.as_instruction()
    }
    pub fn const_not(self) -> Self {
        unsafe { IntValue::new(LLVMConstNot(self.as_value_ref())) }
    }
    pub fn const_neg(self) -> Self {
        unsafe { IntValue::new(LLVMConstNeg(self.as_value_ref())) }
    }
    pub fn const_nsw_neg(self) -> Self {
        unsafe { IntValue::new(LLVMConstNSWNeg(self.as_value_ref())) }
    }
    pub fn const_nuw_neg(self) -> Self {
        unsafe { IntValue::new(LLVMConstNUWNeg(self.as_value_ref())) }
    }
    pub fn const_add(self, rhs: IntValue<'ctx>) -> Self {
        unsafe { IntValue::new(LLVMConstAdd(self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_nsw_add(self, rhs: IntValue<'ctx>) -> Self {
        unsafe { IntValue::new(LLVMConstNSWAdd(self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_nuw_add(self, rhs: IntValue<'ctx>) -> Self {
        unsafe { IntValue::new(LLVMConstNUWAdd(self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_sub(self, rhs: IntValue<'ctx>) -> Self {
        unsafe { IntValue::new(LLVMConstSub(self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_nsw_sub(self, rhs: IntValue<'ctx>) -> Self {
        unsafe { IntValue::new(LLVMConstNSWSub(self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_nuw_sub(self, rhs: IntValue<'ctx>) -> Self {
        unsafe { IntValue::new(LLVMConstNUWSub(self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_mul(self, rhs: IntValue<'ctx>) -> Self {
        unsafe { IntValue::new(LLVMConstMul(self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_nsw_mul(self, rhs: IntValue<'ctx>) -> Self {
        unsafe { IntValue::new(LLVMConstNSWMul(self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_nuw_mul(self, rhs: IntValue<'ctx>) -> Self {
        unsafe { IntValue::new(LLVMConstNUWMul(self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_and(self, rhs: IntValue<'ctx>) -> Self {
        unsafe { IntValue::new(LLVMConstAnd(self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_or(self, rhs: IntValue<'ctx>) -> Self {
        unsafe { IntValue::new(LLVMConstOr(self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_xor(self, rhs: IntValue<'ctx>) -> Self {
        unsafe { IntValue::new(LLVMConstXor(self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_cast(self, int_type: IntType<'ctx>, is_signed: bool) -> Self {
        unsafe {
            IntValue::new(LLVMConstIntCast(
                self.as_value_ref(),
                int_type.as_type_ref(),
                is_signed as i32,
            ))
        }
    }
    pub fn const_shl(self, rhs: IntValue<'ctx>) -> Self {
        unsafe { IntValue::new(LLVMConstShl(self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_rshr(self, rhs: IntValue<'ctx>) -> Self {
        unsafe { IntValue::new(LLVMConstLShr(self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_ashr(self, rhs: IntValue<'ctx>) -> Self {
        unsafe { IntValue::new(LLVMConstAShr(self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_unsigned_to_float(self, float_type: FloatType<'ctx>) -> FloatValue<'ctx> {
        unsafe { FloatValue::new(LLVMConstUIToFP(self.as_value_ref(), float_type.as_type_ref())) }
    }
    pub fn const_signed_to_float(self, float_type: FloatType<'ctx>) -> FloatValue<'ctx> {
        unsafe { FloatValue::new(LLVMConstSIToFP(self.as_value_ref(), float_type.as_type_ref())) }
    }
    pub fn const_to_pointer(self, ptr_type: PointerType<'ctx>) -> PointerValue<'ctx> {
        unsafe { PointerValue::new(LLVMConstIntToPtr(self.as_value_ref(), ptr_type.as_type_ref())) }
    }
    pub fn const_truncate(self, int_type: IntType<'ctx>) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstTrunc(self.as_value_ref(), int_type.as_type_ref())) }
    }
    pub fn const_s_extend(self, int_type: IntType<'ctx>) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstSExt(self.as_value_ref(), int_type.as_type_ref())) }
    }
    pub fn const_z_ext(self, int_type: IntType<'ctx>) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstZExt(self.as_value_ref(), int_type.as_type_ref())) }
    }
    pub fn const_truncate_or_bit_cast(self, int_type: IntType<'ctx>) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstTruncOrBitCast(self.as_value_ref(), int_type.as_type_ref())) }
    }
    pub fn const_s_extend_or_bit_cast(self, int_type: IntType<'ctx>) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstSExtOrBitCast(self.as_value_ref(), int_type.as_type_ref())) }
    }
    pub fn const_z_ext_or_bit_cast(self, int_type: IntType<'ctx>) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstZExtOrBitCast(self.as_value_ref(), int_type.as_type_ref())) }
    }
    pub fn const_bit_cast(self, int_type: IntType) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstBitCast(self.as_value_ref(), int_type.as_type_ref())) }
    }
    pub fn const_int_compare(self, op: IntPredicate, rhs: IntValue<'ctx>) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstICmp(op.into(), self.as_value_ref(), rhs.as_value_ref())) }
    }
    pub fn const_select<BV: BasicValue<'ctx>>(self, then: BV, else_: BV) -> BasicValueEnum<'ctx> {
        unsafe {
            BasicValueEnum::new(LLVMConstSelect(
                self.as_value_ref(),
                then.as_value_ref(),
                else_.as_value_ref(),
            ))
        }
    }
    pub fn is_const(self) -> bool {
        self.val.is_const()
    }
    pub fn is_constant_int(self) -> bool {
        !unsafe { LLVMIsAConstantInt(self.as_value_ref()) }.is_null()
    }
    pub fn get_zero_extended_constant(self) -> Option<u64> {
        if !self.is_constant_int() {
            return None;
        }
        if self.get_type().get_bit_width() > 64 {
            return None;
        }
        unsafe { Some(LLVMConstIntGetZExtValue(self.as_value_ref())) }
    }
    pub fn get_sign_extended_constant(self) -> Option<i64> {
        if !self.is_constant_int() {
            return None;
        }
        if self.get_type().get_bit_width() > 64 {
            return None;
        }
        unsafe { Some(LLVMConstIntGetSExtValue(self.as_value_ref())) }
    }
    pub fn replace_all_uses_with(self, x: IntValue<'ctx>) {
        self.val.replace_all_uses_with(x.as_value_ref())
    }
}
impl<'ctx> TryFrom<InstructionValue<'ctx>> for IntValue<'ctx> {
    type Error = ();
    fn try_from(x: InstructionValue) -> Result<Self, Self::Error> {
        if x.get_type().is_int_type() {
            unsafe { Ok(IntValue::new(x.as_value_ref())) }
        } else {
            Err(())
        }
    }
}
impl Display for IntValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsValueRef for IntValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.val.raw
    }
}

pub const FIRST_CUSTOM_METADATA_KIND_ID: u32 = 39;

#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub struct MetadataValue<'ctx> {
    val: Value<'ctx>,
}
impl<'ctx> MetadataValue<'ctx> {
    pub unsafe fn new(x: LLVMValueRef) -> Self {
        assert!(!x.is_null());
        assert!(!LLVMIsAMDNode(x).is_null() || !LLVMIsAMDString(x).is_null());
        MetadataValue { val: Value::new(x) }
    }
    pub fn as_metadata_ref(self) -> LLVMMetadataRef {
        unsafe { LLVMValueAsMetadata(self.as_value_ref()) }
    }
    pub fn get_name(&self) -> &CStr {
        self.val.get_name()
    }
    pub fn is_node(self) -> bool {
        unsafe { LLVMIsAMDNode(self.as_value_ref()) == self.as_value_ref() }
    }
    pub fn is_string(self) -> bool {
        unsafe { LLVMIsAMDString(self.as_value_ref()) == self.as_value_ref() }
    }
    pub fn get_string_value(&self) -> Option<&CStr> {
        if self.is_node() {
            return None;
        }
        let mut len = 0;
        let c_str = unsafe { CStr::from_ptr(LLVMGetMDString(self.as_value_ref(), &mut len)) };
        Some(c_str)
    }
    pub fn get_node_size(self) -> u32 {
        if self.is_string() {
            return 0;
        }
        unsafe { LLVMGetMDNodeNumOperands(self.as_value_ref()) }
    }
    pub fn get_node_values(self) -> Vec<BasicMetadataValueEnum<'ctx>> {
        if self.is_string() {
            return Vec::new();
        }
        let count = self.get_node_size() as usize;
        let mut vec: Vec<LLVMValueRef> = Vec::with_capacity(count);
        let ptr = vec.as_mut_ptr();
        unsafe {
            LLVMGetMDNodeOperands(self.as_value_ref(), ptr);
            vec.set_len(count)
        };
        vec.iter()
            .map(|val| unsafe { BasicMetadataValueEnum::new(*val) })
            .collect()
    }
    pub fn print_to_stderr(self) {
        self.val.print_to_stderr()
    }
    pub fn replace_all_uses_with(self, x: &MetadataValue<'ctx>) {
        self.val.replace_all_uses_with(x.as_value_ref())
    }
}
impl Display for MetadataValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
impl fmt::Debug for MetadataValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut d = f.debug_struct("MetadataValue");
        d.field("address", &self.as_value_ref());
        if self.is_string() {
            d.field("value", &self.get_string_value().unwrap());
        } else {
            d.field("values", &self.get_node_values());
        }
        d.field("repr", &self.print_to_string());
        d.finish()
    }
}
unsafe impl AsValueRef for MetadataValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.val.raw
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct PhiValue<'ctx> {
    val: Value<'ctx>,
}
impl<'ctx> PhiValue<'ctx> {
    pub unsafe fn new(x: LLVMValueRef) -> Self {
        assert!(!x.is_null());
        PhiValue { val: Value::new(x) }
    }
    pub fn add_incoming(self, incoming: &[(&dyn BasicValue<'ctx>, BasicBlock<'ctx>)]) {
        let (mut values, mut basic_blocks): (Vec<LLVMValueRef>, Vec<LLVMBasicBlockRef>) =
            { incoming.iter().map(|&(v, bb)| (v.as_value_ref(), bb.raw)).unzip() };
        unsafe {
            LLVMAddIncoming(
                self.as_value_ref(),
                values.as_mut_ptr(),
                basic_blocks.as_mut_ptr(),
                incoming.len() as u32,
            );
        }
    }
    pub fn count_incoming(self) -> u32 {
        unsafe { LLVMCountIncoming(self.as_value_ref()) }
    }
    pub fn get_incoming(self, idx: u32) -> Option<(BasicValueEnum<'ctx>, BasicBlock<'ctx>)> {
        if idx >= self.count_incoming() {
            return None;
        }
        let basic_block =
            unsafe { BasicBlock::new(LLVMGetIncomingBlock(self.as_value_ref(), idx)).expect("Invalid BasicBlock") };
        let value = unsafe { BasicValueEnum::new(LLVMGetIncomingValue(self.as_value_ref(), idx)) };
        Some((value, basic_block))
    }
    pub fn get_name(&self) -> &CStr {
        self.val.get_name()
    }
    pub fn set_name(self, name: &str) {
        self.val.set_name(name);
    }
    pub fn is_null(self) -> bool {
        self.val.is_null()
    }
    pub fn is_undef(self) -> bool {
        self.val.is_undef()
    }
    pub fn as_instruction(self) -> InstructionValue<'ctx> {
        self.val
            .as_instruction()
            .expect("PhiValue should always be a Phi InstructionValue")
    }
    pub fn replace_all_uses_with(self, x: &PhiValue<'ctx>) {
        self.val.replace_all_uses_with(x.as_value_ref())
    }
    pub fn as_basic_value(self) -> BasicValueEnum<'ctx> {
        unsafe { BasicValueEnum::new(self.as_value_ref()) }
    }
}
impl<'ctx> TryFrom<InstructionValue<'ctx>> for PhiValue<'ctx> {
    type Error = ();
    fn try_from(value: InstructionValue) -> Result<Self, Self::Error> {
        if value.get_opcode() == InstructionOpcode::Phi {
            unsafe { Ok(PhiValue::new(value.as_value_ref())) }
        } else {
            Err(())
        }
    }
}
impl Display for PhiValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsValueRef for PhiValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.val.raw
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct PointerValue<'ctx> {
    val: Value<'ctx>,
}
impl<'ctx> PointerValue<'ctx> {
    pub unsafe fn new(x: LLVMValueRef) -> Self {
        assert!(!x.is_null());
        PointerValue { val: Value::new(x) }
    }
    pub fn get_name(&self) -> &CStr {
        self.val.get_name()
    }
    pub fn set_name(&self, name: &str) {
        self.val.set_name(name)
    }
    pub fn get_type(self) -> PointerType<'ctx> {
        unsafe { PointerType::new(self.val.get_type()) }
    }
    pub fn is_null(self) -> bool {
        self.val.is_null()
    }
    pub fn is_undef(self) -> bool {
        self.val.is_undef()
    }
    pub fn is_const(self) -> bool {
        self.val.is_const()
    }
    pub fn print_to_stderr(self) {
        self.val.print_to_stderr()
    }
    pub fn as_instruction(self) -> Option<InstructionValue<'ctx>> {
        self.val.as_instruction()
    }
    pub unsafe fn const_gep<T: BasicType<'ctx>>(self, ty: T, ordered_indexes: &[IntValue<'ctx>]) -> PointerValue<'ctx> {
        let mut index_values: Vec<LLVMValueRef> = ordered_indexes.iter().map(|val| val.as_value_ref()).collect();
        let value = {
            LLVMConstGEP2(
                ty.as_type_ref(),
                self.as_value_ref(),
                index_values.as_mut_ptr(),
                index_values.len() as u32,
            )
        };
        PointerValue::new(value)
    }
    pub unsafe fn const_in_bounds_gep<T: BasicType<'ctx>>(
        self,
        ty: T,
        ordered_indexes: &[IntValue<'ctx>],
    ) -> PointerValue<'ctx> {
        let mut index_values: Vec<LLVMValueRef> = ordered_indexes.iter().map(|val| val.as_value_ref()).collect();
        let value = {
            LLVMConstInBoundsGEP2(
                ty.as_type_ref(),
                self.as_value_ref(),
                index_values.as_mut_ptr(),
                index_values.len() as u32,
            )
        };
        PointerValue::new(value)
    }
}
impl<'ctx> PointerValue<'ctx> {
    pub fn const_to_int(self, int_type: IntType<'ctx>) -> IntValue<'ctx> {
        unsafe { IntValue::new(LLVMConstPtrToInt(self.as_value_ref(), int_type.as_type_ref())) }
    }
    pub fn const_cast(self, ptr_type: PointerType<'ctx>) -> PointerValue<'ctx> {
        unsafe { PointerValue::new(LLVMConstPointerCast(self.as_value_ref(), ptr_type.as_type_ref())) }
    }
    pub fn const_address_space_cast(self, ptr_type: PointerType<'ctx>) -> PointerValue<'ctx> {
        unsafe { PointerValue::new(LLVMConstAddrSpaceCast(self.as_value_ref(), ptr_type.as_type_ref())) }
    }
    pub fn replace_all_uses_with(self, x: PointerValue<'ctx>) {
        self.val.replace_all_uses_with(x.as_value_ref())
    }
}
impl<'ctx> TryFrom<InstructionValue<'ctx>> for PointerValue<'ctx> {
    type Error = ();
    fn try_from(value: InstructionValue) -> Result<Self, Self::Error> {
        if value.get_type().is_pointer_type() {
            unsafe { Ok(PointerValue::new(value.as_value_ref())) }
        } else {
            Err(())
        }
    }
}
impl Display for PointerValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsValueRef for PointerValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.val.raw
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct StructValue<'ctx> {
    val: Value<'ctx>,
}
impl<'ctx> StructValue<'ctx> {
    pub unsafe fn new(x: LLVMValueRef) -> Self {
        assert!(!x.is_null());
        StructValue { val: Value::new(x) }
    }
    pub fn get_name(&self) -> &CStr {
        self.val.get_name()
    }
    pub fn set_name(&self, name: &str) {
        self.val.set_name(name)
    }
    pub fn get_type(self) -> StructType<'ctx> {
        unsafe { StructType::new(self.val.get_type()) }
    }
    pub fn is_null(self) -> bool {
        self.val.is_null()
    }
    pub fn is_undef(self) -> bool {
        self.val.is_undef()
    }
    pub fn print_to_stderr(self) {
        self.val.print_to_stderr()
    }
    pub fn as_instruction(self) -> Option<InstructionValue<'ctx>> {
        self.val.as_instruction()
    }
    pub fn replace_all_uses_with(self, x: StructValue<'ctx>) {
        self.val.replace_all_uses_with(x.as_value_ref())
    }
}
impl Display for StructValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsValueRef for StructValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.val.raw
    }
}

pub unsafe trait AnyValue<'ctx>: AsValueRef + Debug {
    fn as_any_value_enum(&self) -> AnyValueEnum<'ctx> {
        unsafe { AnyValueEnum::new(self.as_value_ref()) }
    }
    fn print_to_string(&self) -> LLVMString {
        unsafe { Value::new(self.as_value_ref()).print_to_string() }
    }
}
pub unsafe trait BasicValue<'ctx>: AnyValue<'ctx> {
    fn as_basic_value_enum(&self) -> BasicValueEnum<'ctx> {
        unsafe { BasicValueEnum::new(self.as_value_ref()) }
    }
    fn as_instruction_value(&self) -> Option<InstructionValue<'ctx>> {
        let y = unsafe { Value::new(self.as_value_ref()) };
        if !y.is_instruction() {
            return None;
        }
        unsafe { Some(InstructionValue::new(self.as_value_ref())) }
    }
    fn get_first_use(&self) -> Option<BasicValueUse> {
        unsafe { Value::new(self.as_value_ref()).get_first_use() }
    }
    fn set_name(&self, x: &str) {
        unsafe { Value::new(self.as_value_ref()).set_name(x) }
    }
}
pub unsafe trait AggregateValue<'ctx>: BasicValue<'ctx> {
    fn as_aggregate_value_enum(&self) -> AggregateValueEnum<'ctx> {
        unsafe { AggregateValueEnum::new(self.as_value_ref()) }
    }
}

macro_rules! trait_value_set {
    ($name:ident: $($args:ident),*) => (
        $(
            unsafe impl<'ctx> $name<'ctx> for $args<'ctx> {}
        )*
    );
}

trait_value_set! {AnyValue: AnyValueEnum, BasicValueEnum, BasicMetadataValueEnum, AggregateValueEnum, ArrayValue, IntValue, FloatValue, GlobalValue, PhiValue, PointerValue, FunctionValue, StructValue, VectorValue, InstructionValue, CallSiteValue, MetadataValue}
trait_value_set! {BasicValue: ArrayValue, BasicValueEnum, AggregateValueEnum, IntValue, FloatValue, GlobalValue, StructValue, PointerValue, VectorValue}
trait_value_set! {AggregateValue: ArrayValue, AggregateValueEnum, StructValue}

pub unsafe trait IntMathValue<'ctx>: BasicValue<'ctx> {
    type BaseType: IntMathType<'ctx>;
    unsafe fn new(x: LLVMValueRef) -> Self;
}
pub unsafe trait FloatMathValue<'ctx>: BasicValue<'ctx> {
    type BaseType: FloatMathType<'ctx>;
    unsafe fn new(x: LLVMValueRef) -> Self;
}
pub unsafe trait PointerMathValue<'ctx>: BasicValue<'ctx> {
    type BaseType: PointerMathType<'ctx>;
    unsafe fn new(x: LLVMValueRef) -> Self;
}

macro_rules! math_trait_value_set {
    ($name:ident: $(($ty:ident => $base:ident)),*) => (
        $(
            unsafe impl<'ctx> $name<'ctx> for $ty<'ctx> {
                type BaseType = $base<'ctx>;
                unsafe fn new(x: LLVMValueRef) -> $ty<'ctx> {
                    unsafe {
                        $ty::new(x)
                    }
                }
            }
        )*
    )
}

math_trait_value_set! {IntMathValue: (IntValue => IntType), (VectorValue => VectorType), (PointerValue => IntType)}
math_trait_value_set! {FloatMathValue: (FloatValue => FloatType), (VectorValue => VectorType)}
math_trait_value_set! {PointerMathValue: (PointerValue => PointerType), (VectorValue => VectorType)}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct VectorValue<'ctx> {
    val: Value<'ctx>,
}
impl<'ctx> VectorValue<'ctx> {
    pub unsafe fn new(x: LLVMValueRef) -> Self {
        assert!(!x.is_null());
        VectorValue { val: Value::new(x) }
    }
    pub fn is_const(self) -> bool {
        self.val.is_const()
    }
    pub fn is_constant_vector(self) -> bool {
        unsafe { !LLVMIsAConstantVector(self.as_value_ref()).is_null() }
    }
    pub fn is_constant_data_vector(self) -> bool {
        unsafe { !LLVMIsAConstantDataVector(self.as_value_ref()).is_null() }
    }
    pub fn print_to_stderr(self) {
        self.val.print_to_stderr()
    }
    pub fn get_name(&self) -> &CStr {
        self.val.get_name()
    }
    pub fn set_name(&self, name: &str) {
        self.val.set_name(name)
    }
    pub fn get_type(self) -> VectorType<'ctx> {
        unsafe { VectorType::new(self.val.get_type()) }
    }
    pub fn is_null(self) -> bool {
        self.val.is_null()
    }
    pub fn is_undef(self) -> bool {
        self.val.is_undef()
    }
    pub fn as_instruction(self) -> Option<InstructionValue<'ctx>> {
        self.val.as_instruction()
    }
    pub fn const_extract_element(self, index: IntValue<'ctx>) -> BasicValueEnum<'ctx> {
        unsafe { BasicValueEnum::new(LLVMConstExtractElement(self.as_value_ref(), index.as_value_ref())) }
    }
    pub fn const_insert_element<BV: BasicValue<'ctx>>(self, index: IntValue<'ctx>, value: BV) -> BasicValueEnum<'ctx> {
        unsafe {
            BasicValueEnum::new(LLVMConstInsertElement(
                self.as_value_ref(),
                value.as_value_ref(),
                index.as_value_ref(),
            ))
        }
    }
    pub fn replace_all_uses_with(self, x: VectorValue<'ctx>) {
        self.val.replace_all_uses_with(x.as_value_ref())
    }
    pub fn get_element_as_constant(self, index: u32) -> BasicValueEnum<'ctx> {
        unsafe { BasicValueEnum::new(LLVMGetElementAsConstant(self.as_value_ref(), index)) }
    }
    pub fn const_select<BV: BasicValue<'ctx>>(self, then: BV, else_: BV) -> BasicValueEnum<'ctx> {
        unsafe {
            BasicValueEnum::new(LLVMConstSelect(
                self.as_value_ref(),
                then.as_value_ref(),
                else_.as_value_ref(),
            ))
        }
    }
    pub fn const_shuffle_vector(self, right: VectorValue<'ctx>, mask: VectorValue<'ctx>) -> VectorValue<'ctx> {
        unsafe {
            VectorValue::new(LLVMConstShuffleVector(
                self.as_value_ref(),
                right.as_value_ref(),
                mask.as_value_ref(),
            ))
        }
    }
}
impl Display for VectorValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
unsafe impl AsValueRef for VectorValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.val.raw
    }
}
