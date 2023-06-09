use libc::{c_char, c_int, c_void};
use llvm_lib::{
    analysis::*, bit_reader::*, bit_writer::*, comdata::*, core::*, error::*, error_handling::*, execution_engine::*,
    ir_reader::*, object::*, prelude::*, support::*, target::*, transforms::pass_builder::LLVMRunPasses, *,
};
use once_cell::sync::Lazy;
use parking_lot::{Mutex, MutexGuard};
use std::{
    borrow::Cow,
    cell::{Cell, Ref, RefCell},
    convert::TryFrom,
    error::Error,
    ffi::{CStr, CString},
    fmt::{self, Debug, Display, Formatter},
    fs::File,
    marker,
    mem::{forget, size_of, transmute_copy, MaybeUninit},
    ops::Deref,
    path::Path,
    ptr,
    rc::Rc,
    slice, thread_local,
};

pub mod builder;
pub mod dbg;
pub mod llvm_lib;
pub mod mlir;
pub mod pass;
pub mod target;
pub mod typ;
pub mod val;

use crate::builder::Builder;
use crate::dbg::{DICompileUnit, DWARFEmissionKind, DWARFSourceLanguage, DebugInfoBuilder};
use crate::pass::PassBuilderOptions;
use crate::target::*;
use crate::typ::*;
use crate::val::*;
use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attribute {
    pub raw: LLVMAttributeRef,
}
impl Attribute {
    pub unsafe fn new(raw: LLVMAttributeRef) -> Self {
        debug_assert!(!raw.is_null());
        Attribute { raw }
    }
    pub fn as_mut_ptr(&self) -> LLVMAttributeRef {
        self.raw
    }
    pub fn is_enum(self) -> bool {
        unsafe { LLVMIsEnumAttribute(self.raw) == 1 }
    }
    pub fn is_string(self) -> bool {
        unsafe { LLVMIsStringAttribute(self.raw) == 1 }
    }
    pub fn is_type(self) -> bool {
        unsafe { LLVMIsTypeAttribute(self.raw) == 1 }
    }
    pub fn get_named_enum_kind_id(name: &str) -> u32 {
        unsafe { LLVMGetEnumAttributeKindForName(name.as_ptr() as *const ::libc::c_char, name.len()) }
    }
    pub fn get_enum_kind_id(self) -> u32 {
        assert!(self.get_enum_kind_id_is_valid());
        unsafe { LLVMGetEnumAttributeKind(self.raw) }
    }
    fn get_enum_kind_id_is_valid(self) -> bool {
        self.is_enum() || self.is_type()
    }
    pub fn get_last_enum_kind_id() -> u32 {
        unsafe { LLVMGetLastEnumAttributeKind() }
    }
    pub fn get_enum_value(self) -> u64 {
        assert!(self.is_enum());
        unsafe { LLVMGetEnumAttributeValue(self.raw) }
    }
    pub fn get_string_kind_id(&self) -> &CStr {
        assert!(self.is_string());
        let mut len = 0;
        let y = unsafe { LLVMGetStringAttributeKind(self.raw, &mut len) };
        unsafe { CStr::from_ptr(y) }
    }
    pub fn get_string_value(&self) -> &CStr {
        assert!(self.is_string());
        let mut len = 0;
        let y = unsafe { LLVMGetStringAttributeValue(self.raw, &mut len) };
        unsafe { CStr::from_ptr(y) }
    }
    pub fn get_type_value(&self) -> AnyTypeEnum {
        assert!(self.is_type());
        unsafe { AnyTypeEnum::new(LLVMGetTypeAttributeValue(self.raw)) }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AttributeLoc {
    Return,
    Param(u32),
    Function,
}
impl AttributeLoc {
    pub fn get_index(self) -> u32 {
        match self {
            AttributeLoc::Return => 0,
            AttributeLoc::Param(x) => {
                assert!(x <= u32::max_value() - 2);
                x + 1
            },
            AttributeLoc::Function => u32::max_value(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct AddressSpace(u32);
impl Default for AddressSpace {
    fn default() -> Self {
        AddressSpace(0)
    }
}
impl From<u16> for AddressSpace {
    fn from(x: u16) -> Self {
        AddressSpace(x as u32)
    }
}
impl TryFrom<u32> for AddressSpace {
    type Error = ();
    fn try_from(x: u32) -> Result<Self, Self::Error> {
        if x < 1 << 24 {
            Ok(AddressSpace(x))
        } else {
            Err(())
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub struct BasicBlock<'ctx> {
    pub raw: LLVMBasicBlockRef,
    _marker: marker::PhantomData<&'ctx ()>,
}
impl<'ctx> BasicBlock<'ctx> {
    pub unsafe fn new(raw: LLVMBasicBlockRef) -> Option<Self> {
        if raw.is_null() {
            return None;
        }
        assert!(!LLVMIsABasicBlock(raw as LLVMValueRef).is_null());
        Some(BasicBlock {
            raw,
            _marker: marker::PhantomData,
        })
    }
    pub fn as_mut_ptr(&self) -> LLVMBasicBlockRef {
        self.raw
    }
    pub fn get_parent(self) -> Option<FunctionValue<'ctx>> {
        unsafe { FunctionValue::new(LLVMGetBasicBlockParent(self.raw)) }
    }
    pub fn get_previous_basic_block(self) -> Option<BasicBlock<'ctx>> {
        self.get_parent()?;
        unsafe { BasicBlock::new(LLVMGetPreviousBasicBlock(self.raw)) }
    }
    pub fn get_next_basic_block(self) -> Option<BasicBlock<'ctx>> {
        self.get_parent()?;
        unsafe { BasicBlock::new(LLVMGetNextBasicBlock(self.raw)) }
    }
    pub fn move_before(self, x: BasicBlock<'ctx>) -> Result<(), ()> {
        if self.get_parent().is_none() || x.get_parent().is_none() {
            return Err(());
        }
        unsafe { LLVMMoveBasicBlockBefore(self.raw, x.raw) }
        Ok(())
    }
    pub fn move_after(self, x: BasicBlock<'ctx>) -> Result<(), ()> {
        if self.get_parent().is_none() || x.get_parent().is_none() {
            return Err(());
        }
        unsafe { LLVMMoveBasicBlockAfter(self.raw, x.raw) }
        Ok(())
    }
    pub fn get_first_instruction(self) -> Option<InstructionValue<'ctx>> {
        let y = unsafe { LLVMGetFirstInstruction(self.raw) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(InstructionValue::new(y)) }
    }
    pub fn get_last_instruction(self) -> Option<InstructionValue<'ctx>> {
        let y = unsafe { LLVMGetLastInstruction(self.raw) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(InstructionValue::new(y)) }
    }
    pub fn get_instruction_with_name(self, name: &str) -> Option<InstructionValue<'ctx>> {
        let y = self.get_first_instruction()?;
        y.get_instruction_with_name(name)
    }
    pub fn get_terminator(self) -> Option<InstructionValue<'ctx>> {
        let y = unsafe { LLVMGetBasicBlockTerminator(self.raw) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(InstructionValue::new(y)) }
    }
    pub fn remove_from_function(self) -> Result<(), ()> {
        if self.get_parent().is_none() {
            return Err(());
        }
        unsafe { LLVMRemoveBasicBlockFromParent(self.raw) }
        Ok(())
    }
    pub unsafe fn delete(self) -> Result<(), ()> {
        if self.get_parent().is_none() {
            return Err(());
        }
        LLVMDeleteBasicBlock(self.raw);
        Ok(())
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        unsafe { ContextRef::new(LLVMGetTypeContext(LLVMTypeOf(LLVMBasicBlockAsValue(self.raw)))) }
    }
    pub fn get_name(&self) -> &CStr {
        let y = unsafe { LLVMGetBasicBlockName(self.raw) };
        unsafe { CStr::from_ptr(y) }
    }
    pub fn set_name(&self, name: &str) {
        let y = to_c_str(name);
        unsafe { LLVMSetValueName2(LLVMBasicBlockAsValue(self.raw), y.as_ptr(), name.len()) };
    }
    pub fn replace_all_uses_with(self, x: &BasicBlock<'ctx>) {
        let y = unsafe { LLVMBasicBlockAsValue(self.raw) };
        let x = unsafe { LLVMBasicBlockAsValue(x.raw) };
        if y != x {
            unsafe {
                LLVMReplaceAllUsesWith(y, x);
            }
        }
    }
    pub fn get_first_use(self) -> Option<BasicValueUse<'ctx>> {
        let y = unsafe { LLVMGetFirstUse(LLVMBasicBlockAsValue(self.raw)) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(BasicValueUse::new(y)) }
    }
    pub unsafe fn get_address(self) -> Option<PointerValue<'ctx>> {
        let parent = self.get_parent()?;
        self.get_previous_basic_block()?;
        let y = PointerValue::new(LLVMBlockAddress(parent.as_value_ref(), self.raw));
        if y.is_null() {
            return None;
        }
        Some(y)
    }
}
impl fmt::Debug for BasicBlock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let val = unsafe { CStr::from_ptr(LLVMPrintValueToString(self.raw as LLVMValueRef)) };
        let ty = unsafe { CStr::from_ptr(LLVMPrintTypeToString(LLVMTypeOf(self.raw as LLVMValueRef))) };
        let is_const = unsafe { LLVMIsConstant(self.raw as LLVMValueRef) == 1 };
        f.debug_struct("BasicBlock")
            .field("address", &self.raw)
            .field("is_const", &is_const)
            .field("llvm_value", &val)
            .field("llvm_type", &ty)
            .finish()
    }
}

static GLOBAL_CTX: Lazy<Mutex<Context>> = Lazy::new(|| unsafe { Mutex::new(Context::new(LLVMGetGlobalContext())) });

thread_local! {
    pub static GLOBAL_CTX_LOCK: Lazy<MutexGuard<'static, Context>> = Lazy::new(|| {
        GLOBAL_CTX.lock()
    });
}

pub unsafe trait AsContextRef<'ctx> {
    fn as_ctx_ref(&self) -> LLVMContextRef;
}

#[derive(Debug, PartialEq, Eq)]
pub struct Context {
    pub ctx: ContextImpl,
}
impl Context {
    pub unsafe fn new(x: LLVMContextRef) -> Self {
        Context {
            ctx: ContextImpl::new(x),
        }
    }
    pub fn create() -> Self {
        unsafe { Context::new(LLVMContextCreate()) }
    }
    pub unsafe fn get_global<F, R>(func: F) -> R
    where
        F: FnOnce(&Context) -> R,
    {
        GLOBAL_CTX_LOCK.with(|x| func(x))
    }
    #[inline]
    pub fn create_builder(&self) -> Builder {
        self.ctx.create_builder()
    }
    #[inline]
    pub fn create_module(&self, name: &str) -> Module {
        self.ctx.create_module(name)
    }
    #[inline]
    pub fn create_module_from_ir(&self, x: MemoryBuffer) -> Result<Module, LLVMString> {
        self.ctx.create_module_from_ir(x)
    }
    #[inline]
    pub fn create_inline_asm<'ctx>(
        &'ctx self,
        ty: FunctionType<'ctx>,
        assembly: String,
        constraints: String,
        sideeffects: bool,
        alignstack: bool,
        dialect: Option<InlineAsmDialect>,
        can_throw: bool,
    ) -> PointerValue<'ctx> {
        self.ctx
            .create_inline_asm(ty, assembly, constraints, sideeffects, alignstack, dialect, can_throw)
    }
    #[inline]
    pub fn void_type(&self) -> VoidType {
        self.ctx.void_type()
    }
    #[inline]
    pub fn bool_type(&self) -> IntType {
        self.ctx.bool_type()
    }
    #[inline]
    pub fn i8_type(&self) -> IntType {
        self.ctx.i8_type()
    }
    #[inline]
    pub fn i16_type(&self) -> IntType {
        self.ctx.i16_type()
    }
    #[inline]
    pub fn i32_type(&self) -> IntType {
        self.ctx.i32_type()
    }
    #[inline]
    pub fn i64_type(&self) -> IntType {
        self.ctx.i64_type()
    }
    #[inline]
    pub fn i128_type(&self) -> IntType {
        self.ctx.i128_type()
    }
    #[inline]
    pub fn custom_width_int_type(&self, x: u32) -> IntType {
        self.ctx.custom_width_int_type(x)
    }
    #[inline]
    pub fn metadata_type(&self) -> MetadataType {
        self.ctx.metadata_type()
    }
    #[inline]
    pub fn ptr_sized_int_type(&self, target_data: &TargetData, address_space: Option<AddressSpace>) -> IntType {
        self.ctx.ptr_sized_int_type(target_data, address_space)
    }
    #[inline]
    pub fn f16_type(&self) -> FloatType {
        self.ctx.f16_type()
    }
    #[inline]
    pub fn f32_type(&self) -> FloatType {
        self.ctx.f32_type()
    }
    #[inline]
    pub fn f64_type(&self) -> FloatType {
        self.ctx.f64_type()
    }
    #[inline]
    pub fn x86_f80_type(&self) -> FloatType {
        self.ctx.x86_f80_type()
    }
    #[inline]
    pub fn f128_type(&self) -> FloatType {
        self.ctx.f128_type()
    }
    #[inline]
    pub fn ppc_f128_type(&self) -> FloatType {
        self.ctx.ppc_f128_type()
    }
    #[inline]
    pub fn struct_type(&self, field_types: &[BasicTypeEnum], packed: bool) -> StructType {
        self.ctx.struct_type(field_types, packed)
    }
    #[inline]
    pub fn opaque_struct_type(&self, name: &str) -> StructType {
        self.ctx.opaque_struct_type(name)
    }
    #[inline]
    pub fn get_struct_type<'ctx>(&self, name: &str) -> Option<StructType<'ctx>> {
        self.ctx.get_struct_type(name)
    }
    #[inline]
    pub fn const_struct(&self, values: &[BasicValueEnum], packed: bool) -> StructValue {
        self.ctx.const_struct(values, packed)
    }
    #[inline]
    pub fn append_basic_block<'ctx>(&'ctx self, function: FunctionValue<'ctx>, name: &str) -> BasicBlock<'ctx> {
        self.ctx.append_basic_block(function, name)
    }
    #[inline]
    pub fn insert_basic_block_after<'ctx>(&'ctx self, basic_block: BasicBlock<'ctx>, name: &str) -> BasicBlock<'ctx> {
        self.ctx.insert_basic_block_after(basic_block, name)
    }
    #[inline]
    pub fn prepend_basic_block<'ctx>(&'ctx self, basic_block: BasicBlock<'ctx>, name: &str) -> BasicBlock<'ctx> {
        self.ctx.prepend_basic_block(basic_block, name)
    }
    #[inline]
    pub fn metadata_node<'ctx>(&'ctx self, values: &[BasicMetadataValueEnum<'ctx>]) -> MetadataValue<'ctx> {
        self.ctx.metadata_node(values)
    }
    #[inline]
    pub fn metadata_string(&self, string: &str) -> MetadataValue {
        self.ctx.metadata_string(string)
    }
    #[inline]
    pub fn get_kind_id(&self, key: &str) -> u32 {
        self.ctx.get_kind_id(key)
    }
    #[inline]
    pub fn create_enum_attribute(&self, kind_id: u32, val: u64) -> Attribute {
        self.ctx.create_enum_attribute(kind_id, val)
    }
    #[inline]
    pub fn create_string_attribute(&self, key: &str, val: &str) -> Attribute {
        self.ctx.create_string_attribute(key, val)
    }
    #[inline]
    pub fn create_type_attribute(&self, kind_id: u32, type_ref: AnyTypeEnum) -> Attribute {
        self.ctx.create_type_attribute(kind_id, type_ref)
    }
    #[inline]
    pub fn const_string(&self, string: &[u8], null_terminated: bool) -> ArrayValue {
        self.ctx.const_string(string, null_terminated)
    }
    #[allow(dead_code)]
    #[inline]
    pub fn set_diagnostic_handler(
        &self,
        handler: extern "C" fn(LLVMDiagnosticInfoRef, *mut c_void),
        void_ptr: *mut c_void,
    ) {
        self.ctx.set_diagnostic_handler(handler, void_ptr)
    }
}
impl PartialEq<ContextRef<'_>> for Context {
    fn eq(&self, x: &ContextRef<'_>) -> bool {
        self.ctx == x.ctx
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            LLVMContextDispose(self.ctx.0);
        }
    }
}
unsafe impl<'ctx> AsContextRef<'ctx> for &'ctx Context {
    fn as_ctx_ref(&self) -> LLVMContextRef {
        self.ctx.0
    }
}
unsafe impl Send for Context {}

#[derive(Debug, PartialEq, Eq)]
pub struct ContextRef<'ctx> {
    pub ctx: ContextImpl,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> ContextRef<'ctx> {
    pub unsafe fn new(x: LLVMContextRef) -> Self {
        ContextRef {
            ctx: ContextImpl::new(x),
            _marker: marker::PhantomData,
        }
    }
    #[inline]
    pub fn create_builder(&self) -> Builder<'ctx> {
        self.ctx.create_builder()
    }
    #[inline]
    pub fn create_module(&self, name: &str) -> Module<'ctx> {
        self.ctx.create_module(name)
    }
    #[inline]
    pub fn create_module_from_ir(&self, x: MemoryBuffer) -> Result<Module<'ctx>, LLVMString> {
        self.ctx.create_module_from_ir(x)
    }
    #[inline]
    pub fn create_inline_asm(
        &self,
        ty: FunctionType<'ctx>,
        assembly: String,
        constraints: String,
        sideeffects: bool,
        alignstack: bool,
        dialect: Option<InlineAsmDialect>,
        can_throw: bool,
    ) -> PointerValue<'ctx> {
        self.ctx
            .create_inline_asm(ty, assembly, constraints, sideeffects, alignstack, dialect, can_throw)
    }
    #[inline]
    pub fn void_type(&self) -> VoidType<'ctx> {
        self.ctx.void_type()
    }
    #[inline]
    pub fn bool_type(&self) -> IntType<'ctx> {
        self.ctx.bool_type()
    }
    #[inline]
    pub fn i8_type(&self) -> IntType<'ctx> {
        self.ctx.i8_type()
    }
    #[inline]
    pub fn i16_type(&self) -> IntType<'ctx> {
        self.ctx.i16_type()
    }
    #[inline]
    pub fn i32_type(&self) -> IntType<'ctx> {
        self.ctx.i32_type()
    }
    #[inline]
    pub fn i64_type(&self) -> IntType<'ctx> {
        self.ctx.i64_type()
    }
    #[inline]
    pub fn i128_type(&self) -> IntType<'ctx> {
        self.ctx.i128_type()
    }
    #[inline]
    pub fn custom_width_int_type(&self, bits: u32) -> IntType<'ctx> {
        self.ctx.custom_width_int_type(bits)
    }
    #[inline]
    pub fn metadata_type(&self) -> MetadataType<'ctx> {
        self.ctx.metadata_type()
    }
    #[inline]
    pub fn ptr_sized_int_type(&self, target_data: &TargetData, address_space: Option<AddressSpace>) -> IntType<'ctx> {
        self.ctx.ptr_sized_int_type(target_data, address_space)
    }
    #[inline]
    pub fn f16_type(&self) -> FloatType<'ctx> {
        self.ctx.f16_type()
    }
    #[inline]
    pub fn f32_type(&self) -> FloatType<'ctx> {
        self.ctx.f32_type()
    }
    #[inline]
    pub fn f64_type(&self) -> FloatType<'ctx> {
        self.ctx.f64_type()
    }
    #[inline]
    pub fn x86_f80_type(&self) -> FloatType<'ctx> {
        self.ctx.x86_f80_type()
    }
    #[inline]
    pub fn f128_type(&self) -> FloatType<'ctx> {
        self.ctx.f128_type()
    }
    #[inline]
    pub fn ppc_f128_type(&self) -> FloatType<'ctx> {
        self.ctx.ppc_f128_type()
    }
    #[inline]
    pub fn struct_type(&self, field_types: &[BasicTypeEnum<'ctx>], packed: bool) -> StructType<'ctx> {
        self.ctx.struct_type(field_types, packed)
    }
    #[inline]
    pub fn opaque_struct_type(&self, name: &str) -> StructType<'ctx> {
        self.ctx.opaque_struct_type(name)
    }
    #[inline]
    pub fn get_struct_type(&self, name: &str) -> Option<StructType<'ctx>> {
        self.ctx.get_struct_type(name)
    }
    #[inline]
    pub fn const_struct(&self, values: &[BasicValueEnum<'ctx>], packed: bool) -> StructValue<'ctx> {
        self.ctx.const_struct(values, packed)
    }
    #[inline]
    pub fn append_basic_block(&self, function: FunctionValue<'ctx>, name: &str) -> BasicBlock<'ctx> {
        self.ctx.append_basic_block(function, name)
    }
    #[inline]
    pub fn insert_basic_block_after(&self, basic_block: BasicBlock<'ctx>, name: &str) -> BasicBlock<'ctx> {
        self.ctx.insert_basic_block_after(basic_block, name)
    }
    #[inline]
    pub fn prepend_basic_block(&self, basic_block: BasicBlock<'ctx>, name: &str) -> BasicBlock<'ctx> {
        self.ctx.prepend_basic_block(basic_block, name)
    }
    #[inline]
    pub fn metadata_node(&self, values: &[BasicMetadataValueEnum<'ctx>]) -> MetadataValue<'ctx> {
        self.ctx.metadata_node(values)
    }
    #[inline]
    pub fn metadata_string(&self, string: &str) -> MetadataValue<'ctx> {
        self.ctx.metadata_string(string)
    }
    #[inline]
    pub fn get_kind_id(&self, key: &str) -> u32 {
        self.ctx.get_kind_id(key)
    }
    #[inline]
    pub fn create_enum_attribute(&self, kind_id: u32, val: u64) -> Attribute {
        self.ctx.create_enum_attribute(kind_id, val)
    }
    #[inline]
    pub fn create_string_attribute(&self, key: &str, val: &str) -> Attribute {
        self.ctx.create_string_attribute(key, val)
    }
    #[inline]
    pub fn create_type_attribute(&self, kind_id: u32, type_ref: AnyTypeEnum) -> Attribute {
        self.ctx.create_type_attribute(kind_id, type_ref)
    }
    #[inline]
    pub fn const_string(&self, string: &[u8], null_terminated: bool) -> ArrayValue<'ctx> {
        self.ctx.const_string(string, null_terminated)
    }
    #[inline]
    pub fn set_diagnostic_handler(
        &self,
        handler: extern "C" fn(LLVMDiagnosticInfoRef, *mut c_void),
        void_ptr: *mut c_void,
    ) {
        self.ctx.set_diagnostic_handler(handler, void_ptr)
    }
}
impl PartialEq<Context> for ContextRef<'_> {
    fn eq(&self, x: &Context) -> bool {
        self.ctx == x.ctx
    }
}
unsafe impl<'ctx> AsContextRef<'ctx> for ContextRef<'ctx> {
    fn as_ctx_ref(&self) -> LLVMContextRef {
        self.ctx.0
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ContextImpl(pub LLVMContextRef);
impl ContextImpl {
    pub unsafe fn new(x: LLVMContextRef) -> Self {
        assert!(!x.is_null());
        ContextImpl(x)
    }
    fn create_builder<'ctx>(&self) -> Builder<'ctx> {
        unsafe { Builder::new(LLVMCreateBuilderInContext(self.0)) }
    }
    fn create_module<'ctx>(&self, name: &str) -> Module<'ctx> {
        let y = to_c_str(name);
        unsafe { Module::new(LLVMModuleCreateWithNameInContext(y.as_ptr(), self.0)) }
    }
    fn create_module_from_ir<'ctx>(&self, x: MemoryBuffer) -> Result<Module<'ctx>, LLVMString> {
        let mut y = ptr::null_mut();
        let mut e = ptr::null_mut();
        let code = unsafe { LLVMParseIRInContext(self.0, x.raw, &mut y, &mut e) };
        forget(x);
        if code == 0 {
            unsafe {
                return Ok(Module::new(y));
            }
        }
        unsafe { Err(LLVMString::new(e)) }
    }
    fn create_inline_asm<'ctx>(
        &self,
        ty: FunctionType<'ctx>,
        mut assembly: String,
        mut constraints: String,
        sideeffects: bool,
        alignstack: bool,
        dialect: Option<InlineAsmDialect>,
        can_throw: bool,
    ) -> PointerValue<'ctx> {
        let value = unsafe {
            LLVMGetInlineAsm(
                ty.as_type_ref(),
                assembly.as_mut_ptr() as *mut ::libc::c_char,
                assembly.len(),
                constraints.as_mut_ptr() as *mut ::libc::c_char,
                constraints.len(),
                sideeffects as i32,
                alignstack as i32,
                dialect.unwrap_or(InlineAsmDialect::ATT).into(),
                can_throw as i32,
            )
        };
        unsafe { PointerValue::new(value) }
    }
    fn void_type<'ctx>(&self) -> VoidType<'ctx> {
        unsafe { VoidType::new(LLVMVoidTypeInContext(self.0)) }
    }
    fn bool_type<'ctx>(&self) -> IntType<'ctx> {
        unsafe { IntType::new(LLVMInt1TypeInContext(self.0)) }
    }
    fn i8_type<'ctx>(&self) -> IntType<'ctx> {
        unsafe { IntType::new(LLVMInt8TypeInContext(self.0)) }
    }
    fn i16_type<'ctx>(&self) -> IntType<'ctx> {
        unsafe { IntType::new(LLVMInt16TypeInContext(self.0)) }
    }
    fn i32_type<'ctx>(&self) -> IntType<'ctx> {
        unsafe { IntType::new(LLVMInt32TypeInContext(self.0)) }
    }
    fn i64_type<'ctx>(&self) -> IntType<'ctx> {
        unsafe { IntType::new(LLVMInt64TypeInContext(self.0)) }
    }
    fn i128_type<'ctx>(&self) -> IntType<'ctx> {
        self.custom_width_int_type(128)
    }
    fn custom_width_int_type<'ctx>(&self, bits: u32) -> IntType<'ctx> {
        unsafe { IntType::new(LLVMIntTypeInContext(self.0, bits)) }
    }
    fn metadata_type<'ctx>(&self) -> MetadataType<'ctx> {
        unsafe { MetadataType::new(LLVMMetadataTypeInContext(self.0)) }
    }
    fn ptr_sized_int_type<'ctx>(&self, data: &TargetData, x: Option<AddressSpace>) -> IntType<'ctx> {
        let y = match x {
            Some(x) => unsafe { LLVMIntPtrTypeForASInContext(self.0, data.raw, x.0) },
            None => unsafe { LLVMIntPtrTypeInContext(self.0, data.raw) },
        };
        unsafe { IntType::new(y) }
    }
    fn f16_type<'ctx>(&self) -> FloatType<'ctx> {
        unsafe { FloatType::new(LLVMHalfTypeInContext(self.0)) }
    }
    fn f32_type<'ctx>(&self) -> FloatType<'ctx> {
        unsafe { FloatType::new(LLVMFloatTypeInContext(self.0)) }
    }
    fn f64_type<'ctx>(&self) -> FloatType<'ctx> {
        unsafe { FloatType::new(LLVMDoubleTypeInContext(self.0)) }
    }
    fn x86_f80_type<'ctx>(&self) -> FloatType<'ctx> {
        unsafe { FloatType::new(LLVMX86FP80TypeInContext(self.0)) }
    }
    fn f128_type<'ctx>(&self) -> FloatType<'ctx> {
        unsafe { FloatType::new(LLVMFP128TypeInContext(self.0)) }
    }
    fn ppc_f128_type<'ctx>(&self) -> FloatType<'ctx> {
        unsafe { FloatType::new(LLVMPPCFP128TypeInContext(self.0)) }
    }
    fn struct_type<'ctx>(&self, xs: &[BasicTypeEnum], packed: bool) -> StructType<'ctx> {
        let mut ys: Vec<LLVMTypeRef> = xs.iter().map(|val| val.as_type_ref()).collect();
        unsafe {
            StructType::new(LLVMStructTypeInContext(
                self.0,
                ys.as_mut_ptr(),
                ys.len() as u32,
                packed as i32,
            ))
        }
    }
    fn opaque_struct_type<'ctx>(&self, name: &str) -> StructType<'ctx> {
        let y = to_c_str(name);
        unsafe { StructType::new(LLVMStructCreateNamed(self.0, y.as_ptr())) }
    }
    fn get_struct_type<'ctx>(&self, name: &str) -> Option<StructType<'ctx>> {
        let name = to_c_str(name);
        let y = unsafe { LLVMGetTypeByName2(self.0, name.as_ptr()) };
        if y.is_null() {
            return None;
        }
        unsafe { Some(StructType::new(y)) }
    }
    fn const_struct<'ctx>(&self, xs: &[BasicValueEnum], packed: bool) -> StructValue<'ctx> {
        let mut ys: Vec<LLVMValueRef> = xs.iter().map(|x| x.as_value_ref()).collect();
        unsafe {
            StructValue::new(LLVMConstStructInContext(
                self.0,
                ys.as_mut_ptr(),
                ys.len() as u32,
                packed as i32,
            ))
        }
    }
    fn append_basic_block<'ctx>(&self, function: FunctionValue<'ctx>, name: &str) -> BasicBlock<'ctx> {
        let y = to_c_str(name);
        unsafe {
            BasicBlock::new(LLVMAppendBasicBlockInContext(
                self.0,
                function.as_value_ref(),
                y.as_ptr(),
            ))
            .expect("Appending basic block should never fail")
        }
    }
    fn insert_basic_block_after<'ctx>(&self, x: BasicBlock<'ctx>, name: &str) -> BasicBlock<'ctx> {
        match x.get_next_basic_block() {
            Some(x) => self.prepend_basic_block(x, name),
            None => {
                let parent_fn = x.get_parent().unwrap();
                self.append_basic_block(parent_fn, name)
            },
        }
    }
    fn prepend_basic_block<'ctx>(&self, x: BasicBlock<'ctx>, name: &str) -> BasicBlock<'ctx> {
        let c_string = to_c_str(name);
        unsafe {
            BasicBlock::new(LLVMInsertBasicBlockInContext(self.0, x.raw, c_string.as_ptr()))
                .expect("Prepending basic block should never fail")
        }
    }
    fn metadata_node<'ctx>(&self, xs: &[BasicMetadataValueEnum<'ctx>]) -> MetadataValue<'ctx> {
        let mut ys: Vec<LLVMValueRef> = xs.iter().map(|val| val.as_value_ref()).collect();
        unsafe { MetadataValue::new(LLVMMDNodeInContext(self.0, ys.as_mut_ptr(), ys.len() as u32)) }
    }
    fn metadata_string<'ctx>(&self, x: &str) -> MetadataValue<'ctx> {
        let y = to_c_str(x);
        unsafe { MetadataValue::new(LLVMMDStringInContext(self.0, y.as_ptr(), x.len() as u32)) }
    }
    fn get_kind_id(&self, key: &str) -> u32 {
        unsafe { LLVMGetMDKindIDInContext(self.0, key.as_ptr() as *const ::libc::c_char, key.len() as u32) }
    }
    fn create_enum_attribute(&self, kind_id: u32, val: u64) -> Attribute {
        unsafe { Attribute::new(LLVMCreateEnumAttribute(self.0, kind_id, val)) }
    }
    fn create_string_attribute(&self, key: &str, val: &str) -> Attribute {
        unsafe {
            Attribute::new(LLVMCreateStringAttribute(
                self.0,
                key.as_ptr() as *const _,
                key.len() as u32,
                val.as_ptr() as *const _,
                val.len() as u32,
            ))
        }
    }
    fn create_type_attribute(&self, kind_id: u32, type_ref: AnyTypeEnum) -> Attribute {
        unsafe { Attribute::new(LLVMCreateTypeAttribute(self.0, kind_id, type_ref.as_type_ref())) }
    }
    fn const_string<'ctx>(&self, string: &[u8], null_terminated: bool) -> ArrayValue<'ctx> {
        unsafe {
            ArrayValue::new(LLVMConstStringInContext(
                self.0,
                string.as_ptr() as *const ::libc::c_char,
                string.len() as u32,
                !null_terminated as i32,
            ))
        }
    }
    fn set_diagnostic_handler(
        &self,
        handler: extern "C" fn(LLVMDiagnosticInfoRef, *mut c_void),
        void_ptr: *mut c_void,
    ) {
        unsafe { LLVMContextSetDiagnosticHandler(self.0, Some(handler), void_ptr) }
    }
}

#[llvm_enum(LLVMIntPredicate)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum IntPredicate {
    #[llvm_variant(LLVMIntEQ)]
    EQ,
    #[llvm_variant(LLVMIntNE)]
    NE,
    #[llvm_variant(LLVMIntUGT)]
    UGT,
    #[llvm_variant(LLVMIntUGE)]
    UGE,
    #[llvm_variant(LLVMIntULT)]
    ULT,
    #[llvm_variant(LLVMIntULE)]
    ULE,
    #[llvm_variant(LLVMIntSGT)]
    SGT,
    #[llvm_variant(LLVMIntSGE)]
    SGE,
    #[llvm_variant(LLVMIntSLT)]
    SLT,
    #[llvm_variant(LLVMIntSLE)]
    SLE,
}

#[llvm_enum(LLVMRealPredicate)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FloatPredicate {
    #[llvm_variant(LLVMRealOEQ)]
    OEQ,
    #[llvm_variant(LLVMRealOGE)]
    OGE,
    #[llvm_variant(LLVMRealOGT)]
    OGT,
    #[llvm_variant(LLVMRealOLE)]
    OLE,
    #[llvm_variant(LLVMRealOLT)]
    OLT,
    #[llvm_variant(LLVMRealONE)]
    ONE,
    #[llvm_variant(LLVMRealORD)]
    ORD,
    #[llvm_variant(LLVMRealPredicateFalse)]
    PredicateFalse,
    #[llvm_variant(LLVMRealPredicateTrue)]
    PredicateTrue,
    #[llvm_variant(LLVMRealUEQ)]
    UEQ,
    #[llvm_variant(LLVMRealUGE)]
    UGE,
    #[llvm_variant(LLVMRealUGT)]
    UGT,
    #[llvm_variant(LLVMRealULE)]
    ULE,
    #[llvm_variant(LLVMRealULT)]
    ULT,
    #[llvm_variant(LLVMRealUNE)]
    UNE,
    #[llvm_variant(LLVMRealUNO)]
    UNO,
}

#[llvm_enum(LLVMAtomicOrdering)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AtomicOrdering {
    #[llvm_variant(LLVMAtomicOrderingNotAtomic)]
    NotAtomic,
    #[llvm_variant(LLVMAtomicOrderingUnordered)]
    Unordered,
    #[llvm_variant(LLVMAtomicOrderingMonotonic)]
    Monotonic,
    #[llvm_variant(LLVMAtomicOrderingAcquire)]
    Acquire,
    #[llvm_variant(LLVMAtomicOrderingRelease)]
    Release,
    #[llvm_variant(LLVMAtomicOrderingAcquireRelease)]
    AcquireRelease,
    #[llvm_variant(LLVMAtomicOrderingSequentiallyConsistent)]
    SequentiallyConsistent,
}

#[llvm_enum(LLVMAtomicRMWBinOp)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AtomicRMWBinOp {
    #[llvm_variant(LLVMAtomicRMWBinOpXchg)]
    Xchg,
    #[llvm_variant(LLVMAtomicRMWBinOpAdd)]
    Add,
    #[llvm_variant(LLVMAtomicRMWBinOpSub)]
    Sub,
    #[llvm_variant(LLVMAtomicRMWBinOpAnd)]
    And,
    #[llvm_variant(LLVMAtomicRMWBinOpNand)]
    Nand,
    #[llvm_variant(LLVMAtomicRMWBinOpOr)]
    Or,
    #[llvm_variant(LLVMAtomicRMWBinOpXor)]
    Xor,
    #[llvm_variant(LLVMAtomicRMWBinOpMax)]
    Max,
    #[llvm_variant(LLVMAtomicRMWBinOpMin)]
    Min,
    #[llvm_variant(LLVMAtomicRMWBinOpUMax)]
    UMax,
    #[llvm_variant(LLVMAtomicRMWBinOpUMin)]
    UMin,
    #[llvm_variant(LLVMAtomicRMWBinOpFAdd)]
    FAdd,
    #[llvm_variant(LLVMAtomicRMWBinOpFSub)]
    FSub,
    #[llvm_variant(LLVMAtomicRMWBinOpFMax)]
    FMax,
    #[llvm_variant(LLVMAtomicRMWBinOpFMin)]
    FMin,
}

#[repr(u32)]
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum OptimizationLevel {
    None = 0,
    Less = 1,
    Default = 2,
    Aggressive = 3,
}
impl Default for OptimizationLevel {
    fn default() -> Self {
        OptimizationLevel::Default
    }
}

#[llvm_enum(LLVMVisibility)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GlobalVisibility {
    #[llvm_variant(LLVMDefaultVisibility)]
    Default,
    #[llvm_variant(LLVMHiddenVisibility)]
    Hidden,
    #[llvm_variant(LLVMProtectedVisibility)]
    Protected,
}
impl Default for GlobalVisibility {
    fn default() -> Self {
        GlobalVisibility::Default
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThreadLocalMode {
    GeneralDynamicTLSModel,
    LocalDynamicTLSModel,
    InitialExecTLSModel,
    LocalExecTLSModel,
}
impl ThreadLocalMode {
    pub fn new(x: LLVMThreadLocalMode) -> Option<Self> {
        match x {
            LLVMThreadLocalMode::LLVMGeneralDynamicTLSModel => Some(ThreadLocalMode::GeneralDynamicTLSModel),
            LLVMThreadLocalMode::LLVMLocalDynamicTLSModel => Some(ThreadLocalMode::LocalDynamicTLSModel),
            LLVMThreadLocalMode::LLVMInitialExecTLSModel => Some(ThreadLocalMode::InitialExecTLSModel),
            LLVMThreadLocalMode::LLVMLocalExecTLSModel => Some(ThreadLocalMode::LocalExecTLSModel),
            LLVMThreadLocalMode::LLVMNotThreadLocal => None,
        }
    }
    pub fn as_llvm_mode(self) -> LLVMThreadLocalMode {
        match self {
            ThreadLocalMode::GeneralDynamicTLSModel => LLVMThreadLocalMode::LLVMGeneralDynamicTLSModel,
            ThreadLocalMode::LocalDynamicTLSModel => LLVMThreadLocalMode::LLVMLocalDynamicTLSModel,
            ThreadLocalMode::InitialExecTLSModel => LLVMThreadLocalMode::LLVMInitialExecTLSModel,
            ThreadLocalMode::LocalExecTLSModel => LLVMThreadLocalMode::LLVMLocalExecTLSModel,
        }
    }
}

#[llvm_enum(LLVMDLLStorageClass)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DLLStorageClass {
    #[llvm_variant(LLVMDefaultStorageClass)]
    Default,
    #[llvm_variant(LLVMDLLImportStorageClass)]
    Import,
    #[llvm_variant(LLVMDLLExportStorageClass)]
    Export,
}
impl Default for DLLStorageClass {
    fn default() -> Self {
        DLLStorageClass::Default
    }
}

#[llvm_enum(LLVMInlineAsmDialect)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum InlineAsmDialect {
    #[llvm_variant(LLVMInlineAsmDialectATT)]
    ATT,
    #[llvm_variant(LLVMInlineAsmDialectIntel)]
    Intel,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Intrinsic {
    id: u32,
}
impl Intrinsic {
    pub unsafe fn new(id: u32) -> Self {
        Self { id }
    }
    pub fn find(name: &str) -> Option<Self> {
        let y = unsafe { LLVMLookupIntrinsicID(name.as_ptr() as *const ::libc::c_char, name.len()) };
        if y == 0 {
            return None;
        }
        Some(unsafe { Intrinsic::new(y) })
    }
    pub fn is_overloaded(&self) -> bool {
        unsafe { LLVMIntrinsicIsOverloaded(self.id) != 0 }
    }
    pub fn get_declaration<'ctx>(&self, m: &Module<'ctx>, xs: &[BasicTypeEnum]) -> Option<FunctionValue<'ctx>> {
        let mut xs: Vec<LLVMTypeRef> = xs.iter().map(|x| x.as_type_ref()).collect();
        if self.is_overloaded() && xs.is_empty() {
            return None;
        }
        let y = unsafe {
            FunctionValue::new(LLVMGetIntrinsicDeclaration(
                m.module.get(),
                self.id,
                xs.as_mut_ptr(),
                xs.len(),
            ))
        };
        y
    }
}

#[derive(Debug)]
pub struct MemoryBuffer {
    pub raw: LLVMMemoryBufferRef,
}
impl MemoryBuffer {
    pub unsafe fn new(raw: LLVMMemoryBufferRef) -> Self {
        assert!(!raw.is_null());
        MemoryBuffer { raw }
    }
    pub fn as_mut_ptr(&self) -> LLVMMemoryBufferRef {
        self.raw
    }
    pub fn create_from_file(path: &Path) -> Result<Self, LLVMString> {
        let path = to_c_str(path.to_str().expect("Did not find a valid Unicode path string"));
        let mut y = ptr::null_mut();
        let mut e = MaybeUninit::uninit();
        let code = unsafe {
            LLVMCreateMemoryBufferWithContentsOfFile(path.as_ptr() as *const ::libc::c_char, &mut y, e.as_mut_ptr())
        };
        if code == 1 {
            unsafe {
                return Err(LLVMString::new(e.assume_init()));
            }
        }
        unsafe { Ok(MemoryBuffer::new(y)) }
    }
    pub fn create_from_stdin() -> Result<Self, LLVMString> {
        let mut y = ptr::null_mut();
        let mut e = MaybeUninit::uninit();
        let code = unsafe { LLVMCreateMemoryBufferWithSTDIN(&mut y, e.as_mut_ptr()) };
        if code == 1 {
            unsafe {
                return Err(LLVMString::new(e.assume_init()));
            }
        }
        unsafe { Ok(MemoryBuffer::new(y)) }
    }
    pub fn create_from_memory_range(xs: &[u8], name: &str) -> Self {
        let name = to_c_str(name);
        let y = unsafe {
            LLVMCreateMemoryBufferWithMemoryRange(
                xs.as_ptr() as *const ::libc::c_char,
                xs.len(),
                name.as_ptr(),
                false as i32,
            )
        };
        unsafe { MemoryBuffer::new(y) }
    }
    pub fn create_from_memory_range_copy(xs: &[u8], name: &str) -> Self {
        let name = to_c_str(name);
        let y = unsafe {
            LLVMCreateMemoryBufferWithMemoryRangeCopy(xs.as_ptr() as *const ::libc::c_char, xs.len(), name.as_ptr())
        };
        unsafe { MemoryBuffer::new(y) }
    }
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            let y = LLVMGetBufferStart(self.raw);
            slice::from_raw_parts(y as *const _, self.get_size())
        }
    }
    pub fn get_size(&self) -> usize {
        unsafe { LLVMGetBufferSize(self.raw) }
    }
    pub fn create_object_file(self) -> Result<ObjectFile, ()> {
        let y = unsafe { LLVMCreateObjectFile(self.raw) };
        forget(self);
        if y.is_null() {
            return Err(());
        }
        unsafe { Ok(ObjectFile::new(y)) }
    }
}
impl Drop for MemoryBuffer {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeMemoryBuffer(self.raw);
        }
    }
}

#[derive(Debug)]
pub struct ObjectFile {
    raw: LLVMObjectFileRef,
}
impl ObjectFile {
    pub unsafe fn new(raw: LLVMObjectFileRef) -> Self {
        assert!(!raw.is_null());
        ObjectFile { raw }
    }
    pub fn as_mut_ptr(&self) -> LLVMObjectFileRef {
        self.raw
    }
    pub fn get_sections(&self) -> SectionIterator {
        let y = unsafe { LLVMGetSections(self.raw) };
        unsafe { SectionIterator::new(y, self.raw) }
    }
    pub fn get_symbols(&self) -> SymbolIterator {
        let y = unsafe { LLVMGetSymbols(self.raw) };
        unsafe { SymbolIterator::new(y, self.raw) }
    }
}
impl Drop for ObjectFile {
    fn drop(&mut self) {
        unsafe { LLVMDisposeObjectFile(self.raw) }
    }
}

#[derive(Debug)]
pub struct SectionIterator {
    raw: LLVMSectionIteratorRef,
    obj: LLVMObjectFileRef,
    before_first: bool,
}
impl SectionIterator {
    pub unsafe fn new(raw: LLVMSectionIteratorRef, obj: LLVMObjectFileRef) -> Self {
        assert!(!raw.is_null());
        assert!(!obj.is_null());
        SectionIterator {
            raw,
            obj,
            before_first: true,
        }
    }
    pub fn as_mut_ptr(&self) -> (LLVMSectionIteratorRef, LLVMObjectFileRef) {
        (self.raw, self.obj)
    }
}
impl Iterator for SectionIterator {
    type Item = Section;
    fn next(&mut self) -> Option<Self::Item> {
        if self.before_first {
            self.before_first = false;
        } else {
            unsafe {
                LLVMMoveToNextSection(self.raw);
            }
        }
        let at_end = unsafe { LLVMIsSectionIteratorAtEnd(self.obj, self.raw) == 1 };
        if at_end {
            return None;
        }
        Some(unsafe { Section::new(self.raw, self.obj) })
    }
}
impl Drop for SectionIterator {
    fn drop(&mut self) {
        unsafe { LLVMDisposeSectionIterator(self.raw) }
    }
}

#[derive(Debug)]
pub struct Section {
    raw: LLVMSectionIteratorRef,
    obj: LLVMObjectFileRef,
}
impl Section {
    pub unsafe fn new(raw: LLVMSectionIteratorRef, obj: LLVMObjectFileRef) -> Self {
        assert!(!raw.is_null());
        assert!(!obj.is_null());
        Section { raw, obj }
    }
    pub unsafe fn as_mut_ptr(&self) -> (LLVMSectionIteratorRef, LLVMObjectFileRef) {
        (self.raw, self.obj)
    }
    pub fn get_name(&self) -> Option<&CStr> {
        let y = unsafe { LLVMGetSectionName(self.raw) };
        if !y.is_null() {
            Some(unsafe { CStr::from_ptr(y) })
        } else {
            None
        }
    }
    pub fn size(&self) -> u64 {
        unsafe { LLVMGetSectionSize(self.raw) }
    }
    pub fn get_contents(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(LLVMGetSectionContents(self.raw) as *const u8, self.size() as usize) }
    }
    pub fn get_address(&self) -> u64 {
        unsafe { LLVMGetSectionAddress(self.raw) }
    }
    pub fn get_relocations(&self) -> RelocationIterator {
        let y = unsafe { LLVMGetRelocations(self.raw) };
        unsafe { RelocationIterator::new(y, self.raw, self.obj) }
    }
}

#[derive(Debug)]
pub struct RelocationIterator {
    raw: LLVMRelocationIteratorRef,
    sec: LLVMSectionIteratorRef,
    obj: LLVMObjectFileRef,
    before_first: bool,
}
impl RelocationIterator {
    pub unsafe fn new(raw: LLVMRelocationIteratorRef, sec: LLVMSectionIteratorRef, obj: LLVMObjectFileRef) -> Self {
        assert!(!raw.is_null());
        assert!(!sec.is_null());
        assert!(!obj.is_null());
        RelocationIterator {
            raw,
            sec,
            obj,
            before_first: true,
        }
    }
    pub fn as_mut_ptr(&self) -> (LLVMRelocationIteratorRef, LLVMSectionIteratorRef, LLVMObjectFileRef) {
        (self.raw, self.sec, self.obj)
    }
}
impl Iterator for RelocationIterator {
    type Item = Relocation;
    fn next(&mut self) -> Option<Self::Item> {
        if self.before_first {
            self.before_first = false;
        } else {
            unsafe { LLVMMoveToNextRelocation(self.raw) }
        }
        let at_end = unsafe { LLVMIsRelocationIteratorAtEnd(self.sec, self.raw) == 1 };
        if at_end {
            return None;
        }
        Some(unsafe { Relocation::new(self.raw, self.obj) })
    }
}
impl Drop for RelocationIterator {
    fn drop(&mut self) {
        unsafe { LLVMDisposeRelocationIterator(self.raw) }
    }
}

#[derive(Debug)]
pub struct Relocation {
    raw: LLVMRelocationIteratorRef,
    obj: LLVMObjectFileRef,
}
impl Relocation {
    pub unsafe fn new(raw: LLVMRelocationIteratorRef, obj: LLVMObjectFileRef) -> Self {
        assert!(!raw.is_null());
        assert!(!obj.is_null());
        Relocation { raw, obj }
    }
    pub fn as_mut_ptr(&self) -> (LLVMRelocationIteratorRef, LLVMObjectFileRef) {
        (self.raw, self.obj)
    }
    pub fn get_offset(&self) -> u64 {
        unsafe { LLVMGetRelocationOffset(self.raw) }
    }
    pub fn get_symbols(&self) -> SymbolIterator {
        let y = unsafe { LLVMGetRelocationSymbol(self.raw) };
        unsafe { SymbolIterator::new(y, self.obj) }
    }
    pub fn get_type(&self) -> (u64, &CStr) {
        let type_int = unsafe { LLVMGetRelocationType(self.raw) };
        let type_name = unsafe { CStr::from_ptr(LLVMGetRelocationTypeName(self.raw)) };
        (type_int, type_name)
    }
    pub fn get_value(&self) -> &CStr {
        unsafe { CStr::from_ptr(LLVMGetRelocationValueString(self.raw)) }
    }
}

#[derive(Debug)]
pub struct SymbolIterator {
    raw: LLVMSymbolIteratorRef,
    obj: LLVMObjectFileRef,
    before_first: bool,
}
impl SymbolIterator {
    pub unsafe fn new(raw: LLVMSymbolIteratorRef, obj: LLVMObjectFileRef) -> Self {
        assert!(!raw.is_null());
        assert!(!obj.is_null());
        SymbolIterator {
            raw,
            obj,
            before_first: true,
        }
    }
    pub fn as_mut_ptr(&self) -> (LLVMSymbolIteratorRef, LLVMObjectFileRef) {
        (self.raw, self.obj)
    }
}
impl Iterator for SymbolIterator {
    type Item = Symbol;
    fn next(&mut self) -> Option<Self::Item> {
        if self.before_first {
            self.before_first = false;
        } else {
            unsafe { LLVMMoveToNextSymbol(self.raw) }
        }
        let at_end = unsafe { LLVMIsSymbolIteratorAtEnd(self.obj, self.raw) == 1 };
        if at_end {
            return None;
        }
        Some(unsafe { Symbol::new(self.raw) })
    }
}
impl Drop for SymbolIterator {
    fn drop(&mut self) {
        unsafe { LLVMDisposeSymbolIterator(self.raw) }
    }
}

#[derive(Debug)]
pub struct Symbol {
    raw: LLVMSymbolIteratorRef,
}
impl Symbol {
    pub unsafe fn new(raw: LLVMSymbolIteratorRef) -> Self {
        assert!(!raw.is_null());
        Symbol { raw }
    }
    pub fn as_mut_ptr(&self) -> LLVMSymbolIteratorRef {
        self.raw
    }
    pub fn get_name(&self) -> Option<&CStr> {
        let y = unsafe { LLVMGetSymbolName(self.raw) };
        if !y.is_null() {
            Some(unsafe { CStr::from_ptr(y) })
        } else {
            None
        }
    }
    pub fn size(&self) -> u64 {
        unsafe { LLVMGetSymbolSize(self.raw) }
    }
    pub fn get_address(&self) -> u64 {
        unsafe { LLVMGetSymbolAddress(self.raw) }
    }
}

#[derive(Eq)]
pub struct DataLayout {
    pub raw: LLVMStringOrRaw,
}
impl DataLayout {
    pub unsafe fn new_owned(x: *const ::libc::c_char) -> DataLayout {
        debug_assert!(!x.is_null());
        DataLayout {
            raw: LLVMStringOrRaw::Owned(LLVMString::new(x)),
        }
    }
    pub unsafe fn new_borrowed(x: *const ::libc::c_char) -> DataLayout {
        debug_assert!(!x.is_null());
        DataLayout {
            raw: LLVMStringOrRaw::Borrowed(x),
        }
    }
    pub fn as_str(&self) -> &CStr {
        self.raw.as_str()
    }
    pub fn as_ptr(&self) -> *const ::libc::c_char {
        match self.raw {
            LLVMStringOrRaw::Owned(ref x) => x.raw,
            LLVMStringOrRaw::Borrowed(x) => x,
        }
    }
}
impl PartialEq for DataLayout {
    fn eq(&self, x: &DataLayout) -> bool {
        self.as_str() == x.as_str()
    }
}
impl fmt::Debug for DataLayout {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("DataLayout")
            .field("address", &self.as_ptr())
            .field("repr", &self.as_str())
            .finish()
    }
}

#[llvm_enum(LLVMComdatSelectionKind)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComdatSelectionKind {
    #[llvm_variant(LLVMAnyComdatSelectionKind)]
    Any,
    #[llvm_variant(LLVMExactMatchComdatSelectionKind)]
    ExactMatch,
    #[llvm_variant(LLVMLargestComdatSelectionKind)]
    Largest,
    #[llvm_variant(LLVMNoDuplicatesComdatSelectionKind)]
    NoDuplicates,
    #[llvm_variant(LLVMSameSizeComdatSelectionKind)]
    SameSize,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct Comdat(pub LLVMComdatRef);
impl Comdat {
    pub unsafe fn new(comdat: LLVMComdatRef) -> Self {
        debug_assert!(!comdat.is_null());
        Comdat(comdat)
    }
    pub fn as_mut_ptr(&self) -> LLVMComdatRef {
        self.0
    }
    pub fn get_selection_kind(self) -> ComdatSelectionKind {
        let kind_ptr = unsafe { LLVMGetComdatSelectionKind(self.0) };
        ComdatSelectionKind::new(kind_ptr)
    }
    pub fn set_selection_kind(self, kind: ComdatSelectionKind) {
        unsafe { LLVMSetComdatSelectionKind(self.0, kind.into()) }
    }
}

pub struct DiagnosticInfo {
    raw: LLVMDiagnosticInfoRef,
}
impl DiagnosticInfo {
    pub unsafe fn new(raw: LLVMDiagnosticInfoRef) -> Self {
        DiagnosticInfo { raw }
    }
    pub fn get_description(&self) -> *mut ::libc::c_char {
        unsafe { LLVMGetDiagInfoDescription(self.raw) }
    }
    pub fn severity_is_error(&self) -> bool {
        unsafe {
            match LLVMGetDiagInfoSeverity(self.raw) {
                LLVMDiagnosticSeverity::LLVMDSError => true,
                _ => false,
            }
        }
    }
}

#[derive(Eq)]
pub struct LLVMString {
    pub raw: *const c_char,
}
impl LLVMString {
    pub unsafe fn new(raw: *const c_char) -> Self {
        LLVMString { raw }
    }
    pub fn to_string(&self) -> String {
        (*self).to_string_lossy().into_owned()
    }
    pub fn create_from_c_str(x: &CStr) -> LLVMString {
        unsafe { LLVMString::new(LLVMCreateMessage(x.as_ptr() as *const _)) }
    }
    pub fn create_from_str(x: &str) -> LLVMString {
        debug_assert_eq!(x.as_bytes()[x.as_bytes().len() - 1], 0);
        unsafe { LLVMString::new(LLVMCreateMessage(x.as_ptr() as *const _)) }
    }
}
impl Deref for LLVMString {
    type Target = CStr;
    fn deref(&self) -> &Self::Target {
        unsafe { CStr::from_ptr(self.raw) }
    }
}
impl Debug for LLVMString {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "{:?}", self.deref())
    }
}
impl Display for LLVMString {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "{:?}", self.deref())
    }
}
impl PartialEq for LLVMString {
    fn eq(&self, x: &LLVMString) -> bool {
        **self == **x
    }
}
impl Error for LLVMString {
    fn description(&self) -> &str {
        self.to_str()
            .expect("Could not convert LLVMString to str (likely invalid unicode)")
    }
    fn cause(&self) -> Option<&dyn Error> {
        None
    }
}
impl Drop for LLVMString {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeMessage(self.raw as *mut _);
        }
    }
}

#[derive(Eq)]
pub enum LLVMStringOrRaw {
    Owned(LLVMString),
    Borrowed(*const c_char),
}
impl LLVMStringOrRaw {
    pub fn as_str(&self) -> &CStr {
        match self {
            LLVMStringOrRaw::Owned(llvm_string) => llvm_string.deref(),
            LLVMStringOrRaw::Borrowed(ptr) => unsafe { CStr::from_ptr(*ptr) },
        }
    }
}
impl PartialEq for LLVMStringOrRaw {
    fn eq(&self, x: &LLVMStringOrRaw) -> bool {
        self.as_str() == x.as_str()
    }
}

pub unsafe fn shutdown_llvm() {
    use llvm_lib::core::LLVMShutdown;
    LLVMShutdown()
}

pub fn get_llvm_version() -> (u32, u32, u32) {
    let mut major: u32 = 0;
    let mut minor: u32 = 0;
    let mut patch: u32 = 0;
    unsafe { LLVMGetVersion(&mut major, &mut minor, &mut patch) };
    return (major, minor, patch);
}

pub fn load_library_permanently(filename: &str) -> bool {
    let filename = to_c_str(filename);
    unsafe { LLVMLoadLibraryPermanently(filename.as_ptr()) == 1 }
}

pub fn is_multithreaded() -> bool {
    use llvm_lib::core::LLVMIsMultithreaded;
    unsafe { LLVMIsMultithreaded() == 1 }
}

pub fn enable_llvm_pretty_stack_trace() {
    unsafe { LLVMEnablePrettyStackTrace() }
}

pub fn to_c_str<'s>(mut s: &'s str) -> Cow<'s, CStr> {
    if s.is_empty() {
        s = "\0";
    }
    if !s.chars().rev().any(|ch| ch == '\0') {
        return Cow::from(CString::new(s).expect("unreachable since null bytes are checked"));
    }
    unsafe { Cow::from(CStr::from_ptr(s.as_ptr() as *const _)) }
}

pub extern "C" fn get_error_str_diagnostic_handler(x: LLVMDiagnosticInfoRef, void_ptr: *mut c_void) {
    let y = unsafe { DiagnosticInfo::new(x) };
    if y.severity_is_error() {
        let ptr = void_ptr as *mut *mut c_void as *mut *mut ::libc::c_char;
        unsafe {
            *ptr = y.get_description();
        }
    }
}

pub unsafe fn install_fatal_error_handler(x: extern "C" fn(*const ::libc::c_char)) {
    LLVMInstallFatalErrorHandler(Some(x))
}
pub fn reset_fatal_error_handler() {
    unsafe { LLVMResetFatalErrorHandler() }
}

#[test]
fn test_to_c_str() {
    assert!(matches!(to_c_str("my string"), Cow::Owned(_)));
    assert!(matches!(to_c_str("my string\0"), Cow::Borrowed(_)));
}

static EE_INNER_PANIC: &str = "ExecutionEngineInner should exist until Drop";

#[derive(Debug, PartialEq, Eq)]
pub enum FunctionLookupError {
    JITNotEnabled,
    FunctionNotFound, // 404!
}
impl Error for FunctionLookupError {}
impl FunctionLookupError {
    fn as_str(&self) -> &str {
        match self {
            FunctionLookupError::JITNotEnabled => "ExecutionEngine does not have JIT functionality enabled",
            FunctionLookupError::FunctionNotFound => "Function not found in ExecutionEngine",
        }
    }
}
impl Display for FunctionLookupError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "FunctionLookupError({})", self.as_str())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum RemoveModuleError {
    ModuleNotOwned,
    IncorrectModuleOwner,
    LLVMError(LLVMString),
}
impl Error for RemoveModuleError {
    fn description(&self) -> &str {
        self.as_str()
    }
    fn cause(&self) -> Option<&dyn Error> {
        None
    }
}
impl RemoveModuleError {
    fn as_str(&self) -> &str {
        match self {
            RemoveModuleError::ModuleNotOwned => "Module is not owned by an Execution Engine",
            RemoveModuleError::IncorrectModuleOwner => "Module is not owned by this Execution Engine",
            RemoveModuleError::LLVMError(string) => string.to_str().unwrap_or("LLVMError with invalid unicode"),
        }
    }
}
impl Display for RemoveModuleError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "RemoveModuleError({})", self.as_str())
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct ExecutionEngine<'ctx> {
    execution_engine: Option<ExecEngineInner<'ctx>>,
    target_data: Option<TargetData>,
    jit_mode: bool,
}
impl<'ctx> ExecutionEngine<'ctx> {
    pub unsafe fn new(execution_engine: Rc<LLVMExecutionEngineRef>, jit_mode: bool) -> Self {
        assert!(!execution_engine.is_null());
        let target_data = LLVMGetExecutionEngineTargetData(*execution_engine);
        ExecutionEngine {
            execution_engine: Some(ExecEngineInner(execution_engine, marker::PhantomData)),
            target_data: Some(TargetData::new(target_data)),
            jit_mode,
        }
    }
    pub fn as_mut_ptr(&self) -> LLVMExecutionEngineRef {
        self.execution_engine_inner()
    }
    pub fn execution_engine_rc(&self) -> &Rc<LLVMExecutionEngineRef> {
        &self.execution_engine.as_ref().expect(EE_INNER_PANIC).0
    }
    #[inline]
    pub fn execution_engine_inner(&self) -> LLVMExecutionEngineRef {
        **self.execution_engine_rc()
    }
    pub fn link_in_mc_jit() {
        unsafe { LLVMLinkInMCJIT() }
    }
    pub fn link_in_interpreter() {
        unsafe {
            LLVMLinkInInterpreter();
        }
    }
    pub fn add_global_mapping(&self, value: &dyn AnyValue<'ctx>, addr: usize) {
        unsafe { LLVMAddGlobalMapping(self.execution_engine_inner(), value.as_value_ref(), addr as *mut _) }
    }
    pub fn add_module(&self, module: &Module<'ctx>) -> Result<(), ()> {
        unsafe { LLVMAddModule(self.execution_engine_inner(), module.module.get()) }
        if module.owned_by_ee.borrow().is_some() {
            return Err(());
        }
        *module.owned_by_ee.borrow_mut() = Some(self.clone());
        Ok(())
    }
    pub fn remove_module(&self, m: &Module<'ctx>) -> Result<(), RemoveModuleError> {
        match *m.owned_by_ee.borrow() {
            Some(ref x) if x.execution_engine_inner() != self.execution_engine_inner() => {
                return Err(RemoveModuleError::IncorrectModuleOwner)
            },
            None => return Err(RemoveModuleError::ModuleNotOwned),
            _ => (),
        }
        let mut y = MaybeUninit::uninit();
        let mut e = MaybeUninit::uninit();
        let code = unsafe {
            LLVMRemoveModule(
                self.execution_engine_inner(),
                m.module.get(),
                y.as_mut_ptr(),
                e.as_mut_ptr(),
            )
        };
        if code == 1 {
            unsafe {
                return Err(RemoveModuleError::LLVMError(LLVMString::new(e.assume_init())));
            }
        }
        let y = unsafe { y.assume_init() };
        m.module.set(y);
        *m.owned_by_ee.borrow_mut() = None;
        Ok(())
    }
    pub unsafe fn get_function<F>(&self, fn_name: &str) -> Result<JitFunction<'ctx, F>, FunctionLookupError>
    where
        F: UnsafeFunctionPointer,
    {
        if !self.jit_mode {
            return Err(FunctionLookupError::JITNotEnabled);
        }
        let address = self.get_function_address(fn_name)?;
        assert_eq!(
            size_of::<F>(),
            size_of::<usize>(),
            "The type `F` must have the same size as a function pointer"
        );
        let execution_engine = self.execution_engine.as_ref().expect(EE_INNER_PANIC);
        Ok(JitFunction {
            _execution_engine: execution_engine.clone(),
            inner: transmute_copy(&address),
        })
    }
    pub fn get_function_address(&self, fn_name: &str) -> Result<usize, FunctionLookupError> {
        let c_string = to_c_str(fn_name);
        let address = unsafe { LLVMGetFunctionAddress(self.execution_engine_inner(), c_string.as_ptr()) };
        if address == 0 {
            return Err(FunctionLookupError::FunctionNotFound);
        }
        Ok(address as usize)
    }
    pub fn get_target_data(&self) -> &TargetData {
        self.target_data
            .as_ref()
            .expect("TargetData should always exist until Drop")
    }
    pub fn get_function_value(&self, fn_name: &str) -> Result<FunctionValue<'ctx>, FunctionLookupError> {
        if !self.jit_mode {
            return Err(FunctionLookupError::JITNotEnabled);
        }
        let c_string = to_c_str(fn_name);
        let mut function = MaybeUninit::uninit();
        let code = unsafe { LLVMFindFunction(self.execution_engine_inner(), c_string.as_ptr(), function.as_mut_ptr()) };
        if code == 0 {
            return unsafe { FunctionValue::new(function.assume_init()).ok_or(FunctionLookupError::FunctionNotFound) };
        };
        Err(FunctionLookupError::FunctionNotFound)
    }
    pub unsafe fn run_function(
        &self,
        function: FunctionValue<'ctx>,
        args: &[&GenericValue<'ctx>],
    ) -> GenericValue<'ctx> {
        let mut args: Vec<LLVMGenericValueRef> = args.iter().map(|val| val.raw).collect();
        let value = LLVMRunFunction(
            self.execution_engine_inner(),
            function.as_value_ref(),
            args.len() as u32,
            args.as_mut_ptr(),
        ); // REVIEW: usize to u32 ok??
        GenericValue::new(value)
    }
    pub unsafe fn run_function_as_main(&self, function: FunctionValue<'ctx>, args: &[&str]) -> c_int {
        let cstring_args: Vec<_> = args.iter().map(|&arg| to_c_str(arg)).collect();
        let raw_args: Vec<*const _> = cstring_args.iter().map(|arg| arg.as_ptr()).collect();
        let environment_variables = vec![]; // TODO: Support envp. Likely needs to be null terminated
        LLVMRunFunctionAsMain(
            self.execution_engine_inner(),
            function.as_value_ref(),
            raw_args.len() as u32,
            raw_args.as_ptr(),
            environment_variables.as_ptr(),
        ) // REVIEW: usize to u32 cast ok??
    }
    pub fn free_fn_machine_code(&self, function: FunctionValue<'ctx>) {
        unsafe { LLVMFreeMachineCodeForFunction(self.execution_engine_inner(), function.as_value_ref()) }
    }
    pub fn run_static_constructors(&self) {
        unsafe { LLVMRunStaticConstructors(self.execution_engine_inner()) }
    }
    pub fn run_static_destructors(&self) {
        unsafe { LLVMRunStaticDestructors(self.execution_engine_inner()) }
    }
}
impl Drop for ExecutionEngine<'_> {
    fn drop(&mut self) {
        forget(
            self.target_data
                .take()
                .expect("TargetData should always exist until Drop"),
        );
        drop(self.execution_engine.take().expect(EE_INNER_PANIC));
    }
}
impl Clone for ExecutionEngine<'_> {
    fn clone(&self) -> Self {
        let execution_engine_rc = self.execution_engine_rc().clone();
        unsafe { ExecutionEngine::new(execution_engine_rc, self.jit_mode) }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExecEngineInner<'ctx>(Rc<LLVMExecutionEngineRef>, marker::PhantomData<&'ctx Context>);
impl Drop for ExecEngineInner<'_> {
    fn drop(&mut self) {
        if Rc::strong_count(&self.0) == 1 {
            unsafe {
                LLVMDisposeExecutionEngine(*self.0);
            }
        }
    }
}
impl Deref for ExecEngineInner<'_> {
    type Target = LLVMExecutionEngineRef;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

#[derive(Clone)]
pub struct JitFunction<'ctx, F> {
    _execution_engine: ExecEngineInner<'ctx>,
    inner: F,
}
impl<'ctx, F: Copy> JitFunction<'ctx, F> {
    pub unsafe fn into_raw(self) -> F {
        self.inner
    }
    pub unsafe fn as_raw(&self) -> F {
        self.inner
    }
}
impl<F> Debug for JitFunction<'_, F> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_tuple("JitFunction").field(&"<unnamed>").finish()
    }
}

pub trait UnsafeFunctionPointer: private::SealedUnsafeFunctionPointer {}
mod private {
    pub trait SealedUnsafeFunctionPointer: Copy {}
}
impl<F: private::SealedUnsafeFunctionPointer> UnsafeFunctionPointer for F {}

macro_rules! impl_unsafe_fn {
    (@recurse $first:ident $( , $rest:ident )*) => {
        impl_unsafe_fn!($( $rest ),*);
    };
    (@recurse) => {};
    ($( $param:ident ),*) => {
        impl<Output, $( $param ),*> private::SealedUnsafeFunctionPointer for unsafe extern "C" fn($( $param ),*) -> Output {}
        impl<Output, $( $param ),*> JitFunction<'_, unsafe extern "C" fn($( $param ),*) -> Output> {
            #[allow(non_snake_case)]
            #[inline(always)]
            pub unsafe fn call(&self, $( $param: $param ),*) -> Output {
                (self.inner)($( $param ),*)
            }
        }
        impl_unsafe_fn!(@recurse $( $param ),*);
    };
}
impl_unsafe_fn!(A, B, C, D, E, F, G, H, I, J, K, L, M);

#[llvm_enum(LLVMLinkage)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Linkage {
    #[llvm_variant(LLVMAppendingLinkage)]
    Appending,
    #[llvm_variant(LLVMAvailableExternallyLinkage)]
    AvailableExternally,
    #[llvm_variant(LLVMCommonLinkage)]
    Common,
    #[llvm_variant(LLVMDLLExportLinkage)]
    DLLExport,
    #[llvm_variant(LLVMDLLImportLinkage)]
    DLLImport,
    #[llvm_variant(LLVMExternalLinkage)]
    External,
    #[llvm_variant(LLVMExternalWeakLinkage)]
    ExternalWeak,
    #[llvm_variant(LLVMGhostLinkage)]
    Ghost,
    #[llvm_variant(LLVMInternalLinkage)]
    Internal,
    #[llvm_variant(LLVMLinkerPrivateLinkage)]
    LinkerPrivate,
    #[llvm_variant(LLVMLinkerPrivateWeakLinkage)]
    LinkerPrivateWeak,
    #[llvm_variant(LLVMLinkOnceAnyLinkage)]
    LinkOnceAny,
    #[llvm_variant(LLVMLinkOnceODRAutoHideLinkage)]
    LinkOnceODRAutoHide,
    #[llvm_variant(LLVMLinkOnceODRLinkage)]
    LinkOnceODR,
    #[llvm_variant(LLVMPrivateLinkage)]
    Private,
    #[llvm_variant(LLVMWeakAnyLinkage)]
    WeakAny,
    #[llvm_variant(LLVMWeakODRLinkage)]
    WeakODR,
}
#[derive(Debug, PartialEq, Eq)]

pub struct Module<'ctx> {
    data_layout: RefCell<Option<DataLayout>>,
    pub module: Cell<LLVMModuleRef>,
    pub owned_by_ee: RefCell<Option<ExecutionEngine<'ctx>>>,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> Module<'ctx> {
    pub unsafe fn new(module: LLVMModuleRef) -> Self {
        debug_assert!(!module.is_null());
        Module {
            module: Cell::new(module),
            owned_by_ee: RefCell::new(None),
            data_layout: RefCell::new(Some(Module::get_borrowed_data_layout(module))),
            _marker: marker::PhantomData,
        }
    }
    pub fn as_mut_ptr(&self) -> LLVMModuleRef {
        self.module.get()
    }
    pub fn add_function(&self, name: &str, ty: FunctionType<'ctx>, linkage: Option<Linkage>) -> FunctionValue<'ctx> {
        let c_string = to_c_str(name);
        let fn_value = unsafe {
            FunctionValue::new(LLVMAddFunction(self.module.get(), c_string.as_ptr(), ty.as_type_ref()))
                .expect("add_function should always succeed in adding a new function")
        };
        if let Some(linkage) = linkage {
            fn_value.set_linkage(linkage)
        }
        fn_value
    }
    pub fn get_context(&self) -> ContextRef<'ctx> {
        unsafe { ContextRef::new(LLVMGetModuleContext(self.module.get())) }
    }
    pub fn get_first_function(&self) -> Option<FunctionValue<'ctx>> {
        unsafe { FunctionValue::new(LLVMGetFirstFunction(self.module.get())) }
    }
    pub fn get_last_function(&self) -> Option<FunctionValue<'ctx>> {
        unsafe { FunctionValue::new(LLVMGetLastFunction(self.module.get())) }
    }
    pub fn get_function(&self, name: &str) -> Option<FunctionValue<'ctx>> {
        let c_string = to_c_str(name);
        unsafe { FunctionValue::new(LLVMGetNamedFunction(self.module.get(), c_string.as_ptr())) }
    }
    pub fn get_functions(&self) -> FunctionIterator<'ctx> {
        FunctionIterator::from_module(self)
    }
    pub fn get_struct_type(&self, name: &str) -> Option<StructType<'ctx>> {
        self.get_context().get_struct_type(name)
    }
    pub fn set_triple(&self, triple: &TargetTriple) {
        unsafe { LLVMSetTarget(self.module.get(), triple.as_ptr()) }
    }
    pub fn get_triple(&self) -> TargetTriple {
        let target_str = unsafe { LLVMGetTarget(self.module.get()) };
        unsafe { TargetTriple::new(LLVMString::create_from_c_str(CStr::from_ptr(target_str))) }
    }
    pub fn create_execution_engine(&self) -> Result<ExecutionEngine<'ctx>, LLVMString> {
        Target::initialize_native(&InitializationConfig::default()).map_err(|mut x| {
            x.push('\0');
            LLVMString::create_from_str(&x)
        })?;
        if self.owned_by_ee.borrow().is_some() {
            let string = "This module is already owned by an ExecutionEngine.\0";
            return Err(LLVMString::create_from_str(string));
        }
        let mut y = MaybeUninit::uninit();
        let mut e = MaybeUninit::uninit();
        let code = unsafe { LLVMCreateExecutionEngineForModule(y.as_mut_ptr(), self.module.get(), e.as_mut_ptr()) };
        if code == 1 {
            unsafe {
                return Err(LLVMString::new(e.assume_init()));
            }
        }
        let y = unsafe { y.assume_init() };
        let y = unsafe { ExecutionEngine::new(Rc::new(y), false) };
        *self.owned_by_ee.borrow_mut() = Some(y.clone());
        Ok(y)
    }
    pub fn create_interpreter_execution_engine(&self) -> Result<ExecutionEngine<'ctx>, LLVMString> {
        Target::initialize_native(&InitializationConfig::default()).map_err(|mut x| {
            x.push('\0');
            LLVMString::create_from_str(&x)
        })?;
        if self.owned_by_ee.borrow().is_some() {
            let string = "This module is already owned by an ExecutionEngine.\0";
            return Err(LLVMString::create_from_str(string));
        }
        let mut y = MaybeUninit::uninit();
        let mut e = MaybeUninit::uninit();
        let code = unsafe { LLVMCreateInterpreterForModule(y.as_mut_ptr(), self.module.get(), e.as_mut_ptr()) };
        if code == 1 {
            unsafe {
                return Err(LLVMString::new(e.assume_init()));
            }
        }
        let y = unsafe { y.assume_init() };
        let y = unsafe { ExecutionEngine::new(Rc::new(y), false) };
        *self.owned_by_ee.borrow_mut() = Some(y.clone());
        Ok(y)
    }
    pub fn create_jit_execution_engine(
        &self,
        opt_level: OptimizationLevel,
    ) -> Result<ExecutionEngine<'ctx>, LLVMString> {
        Target::initialize_native(&InitializationConfig::default()).map_err(|mut x| {
            x.push('\0');
            LLVMString::create_from_str(&x)
        })?;
        if self.owned_by_ee.borrow().is_some() {
            let string = "This module is already owned by an ExecutionEngine.\0";
            return Err(LLVMString::create_from_str(string));
        }
        let mut y = MaybeUninit::uninit();
        let mut e = MaybeUninit::uninit();
        let code = unsafe {
            LLVMCreateJITCompilerForModule(y.as_mut_ptr(), self.module.get(), opt_level as u32, e.as_mut_ptr())
        };
        if code == 1 {
            unsafe {
                return Err(LLVMString::new(e.assume_init()));
            }
        }
        let y = unsafe { y.assume_init() };
        let y = unsafe { ExecutionEngine::new(Rc::new(y), true) };
        *self.owned_by_ee.borrow_mut() = Some(y.clone());
        Ok(y)
    }
    pub fn add_global<T: BasicType<'ctx>>(
        &self,
        type_: T,
        address_space: Option<AddressSpace>,
        name: &str,
    ) -> GlobalValue<'ctx> {
        let c_string = to_c_str(name);
        let value = unsafe {
            match address_space {
                Some(address_space) => LLVMAddGlobalInAddressSpace(
                    self.module.get(),
                    type_.as_type_ref(),
                    c_string.as_ptr(),
                    address_space.0,
                ),
                None => LLVMAddGlobal(self.module.get(), type_.as_type_ref(), c_string.as_ptr()),
            }
        };
        unsafe { GlobalValue::new(value) }
    }
    pub fn write_bitcode_to_path(&self, path: &Path) -> bool {
        let path_str = path.to_str().expect("Did not find a valid Unicode path string");
        let c_string = to_c_str(path_str);
        unsafe { LLVMWriteBitcodeToFile(self.module.get(), c_string.as_ptr()) == 0 }
    }
    pub fn write_bitcode_to_file(&self, file: &File, should_close: bool, unbuffered: bool) -> bool {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            unsafe {
                LLVMWriteBitcodeToFD(
                    self.module.get(),
                    file.as_raw_fd(),
                    should_close as i32,
                    unbuffered as i32,
                ) == 0
            }
        }
        #[cfg(not(unix))]
        return false;
    }
    pub fn write_bitcode_to_memory(&self) -> MemoryBuffer {
        let memory_buffer = unsafe { LLVMWriteBitcodeToMemoryBuffer(self.module.get()) };
        unsafe { MemoryBuffer::new(memory_buffer) }
    }
    pub fn verify(&self) -> Result<(), LLVMString> {
        let mut err_str = MaybeUninit::uninit();
        let action = LLVMVerifierFailureAction::LLVMReturnStatusAction;
        let code = unsafe { LLVMVerifyModule(self.module.get(), action, err_str.as_mut_ptr()) };
        let err_str = unsafe { err_str.assume_init() };
        if code == 1 && !err_str.is_null() {
            return unsafe { Err(LLVMString::new(err_str)) };
        }
        Ok(())
    }
    fn get_borrowed_data_layout(module: LLVMModuleRef) -> DataLayout {
        let data_layout = unsafe { LLVMGetDataLayoutStr(module) };
        unsafe { DataLayout::new_borrowed(data_layout) }
    }
    pub fn get_data_layout(&self) -> Ref<DataLayout> {
        Ref::map(self.data_layout.borrow(), |l| {
            l.as_ref().expect("DataLayout should always exist until Drop")
        })
    }
    pub fn set_data_layout(&self, data_layout: &DataLayout) {
        unsafe {
            LLVMSetDataLayout(self.module.get(), data_layout.as_ptr());
        }
        *self.data_layout.borrow_mut() = Some(Module::get_borrowed_data_layout(self.module.get()));
    }
    pub fn print_to_stderr(&self) {
        unsafe {
            LLVMDumpModule(self.module.get());
        }
    }
    pub fn print_to_string(&self) -> LLVMString {
        unsafe { LLVMString::new(LLVMPrintModuleToString(self.module.get())) }
    }
    pub fn print_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), LLVMString> {
        let path = path
            .as_ref()
            .to_str()
            .expect("Did not find a valid Unicode path string");
        let path = to_c_str(path);
        let mut e = MaybeUninit::uninit();
        let code = unsafe {
            LLVMPrintModuleToFile(
                self.module.get(),
                path.as_ptr() as *const ::libc::c_char,
                e.as_mut_ptr(),
            )
        };
        if code == 1 {
            unsafe {
                return Err(LLVMString::new(e.assume_init()));
            }
        }
        Ok(())
    }
    pub fn to_string(&self) -> String {
        self.print_to_string().to_string()
    }
    pub fn set_inline_assembly(&self, asm: &str) {
        {
            unsafe { LLVMSetModuleInlineAsm2(self.module.get(), asm.as_ptr() as *const ::libc::c_char, asm.len()) }
        }
    }
    pub fn add_global_metadata(&self, key: &str, metadata: &MetadataValue<'ctx>) -> Result<(), &'static str> {
        if !metadata.is_node() {
            return Err("metadata is expected to be a node.");
        }
        let c_string = to_c_str(key);
        unsafe {
            LLVMAddNamedMetadataOperand(self.module.get(), c_string.as_ptr(), metadata.as_value_ref());
        }
        Ok(())
    }
    pub fn get_global_metadata_size(&self, key: &str) -> u32 {
        let c_string = to_c_str(key);
        unsafe { LLVMGetNamedMetadataNumOperands(self.module.get(), c_string.as_ptr()) }
    }
    pub fn get_global_metadata(&self, key: &str) -> Vec<MetadataValue<'ctx>> {
        let c_string = to_c_str(key);
        let count = self.get_global_metadata_size(key) as usize;
        let mut vec: Vec<LLVMValueRef> = Vec::with_capacity(count);
        let ptr = vec.as_mut_ptr();
        unsafe {
            LLVMGetNamedMetadataOperands(self.module.get(), c_string.as_ptr(), ptr);
            vec.set_len(count);
        };
        vec.iter().map(|val| unsafe { MetadataValue::new(*val) }).collect()
    }
    pub fn get_first_global(&self) -> Option<GlobalValue<'ctx>> {
        let value = unsafe { LLVMGetFirstGlobal(self.module.get()) };
        if value.is_null() {
            return None;
        }
        unsafe { Some(GlobalValue::new(value)) }
    }
    pub fn get_last_global(&self) -> Option<GlobalValue<'ctx>> {
        let value = unsafe { LLVMGetLastGlobal(self.module.get()) };
        if value.is_null() {
            return None;
        }
        unsafe { Some(GlobalValue::new(value)) }
    }
    pub fn get_global(&self, name: &str) -> Option<GlobalValue<'ctx>> {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMGetNamedGlobal(self.module.get(), c_string.as_ptr()) };
        if value.is_null() {
            return None;
        }
        unsafe { Some(GlobalValue::new(value)) }
    }
    pub fn get_globals(&self) -> GlobalIterator<'ctx> {
        GlobalIterator::from_module(self)
    }
    pub fn parse_bitcode_from_buffer(buffer: &MemoryBuffer, c: impl AsContextRef<'ctx>) -> Result<Self, LLVMString> {
        let mut y = MaybeUninit::uninit();
        let mut e = MaybeUninit::uninit();
        #[allow(deprecated)]
        let code = unsafe { LLVMParseBitcodeInContext(c.as_ctx_ref(), buffer.raw, y.as_mut_ptr(), e.as_mut_ptr()) };
        if code != 0 {
            unsafe {
                return Err(LLVMString::new(e.assume_init()));
            }
        }
        unsafe { Ok(Module::new(y.assume_init())) }
    }
    pub fn parse_bitcode_from_path<P: AsRef<Path>>(path: P, c: impl AsContextRef<'ctx>) -> Result<Self, LLVMString> {
        let y = MemoryBuffer::create_from_file(path.as_ref())?;
        Self::parse_bitcode_from_buffer(&y, c)
    }
    pub fn get_name(&self) -> &CStr {
        let mut len = 0;
        let y = unsafe { LLVMGetModuleIdentifier(self.module.get(), &mut len) };
        unsafe { CStr::from_ptr(y) }
    }
    pub fn set_name(&self, name: &str) {
        unsafe { LLVMSetModuleIdentifier(self.module.get(), name.as_ptr() as *const ::libc::c_char, name.len()) }
    }
    pub fn get_source_file_name(&self) -> &CStr {
        let mut len = 0;
        let y = unsafe { LLVMGetSourceFileName(self.module.get(), &mut len) };
        unsafe { CStr::from_ptr(y) }
    }
    pub fn set_source_file_name(&self, file: &str) {
        unsafe { LLVMSetSourceFileName(self.module.get(), file.as_ptr() as *const ::libc::c_char, file.len()) }
    }
    pub fn link_in_module(&self, x: Self) -> Result<(), LLVMString> {
        if x.owned_by_ee.borrow().is_some() {
            let string = "Cannot link a module which is already owned by an ExecutionEngine.\0";
            return Err(LLVMString::create_from_str(string));
        }
        use libc::c_void;
        let context = self.get_context();
        let mut char_ptr: *mut ::libc::c_char = ptr::null_mut();
        let char_ptr_ptr = &mut char_ptr as *mut *mut ::libc::c_char as *mut *mut c_void as *mut c_void;
        context.set_diagnostic_handler(get_error_str_diagnostic_handler, char_ptr_ptr);
        let code = unsafe { LLVMLinkModules2(self.module.get(), x.module.get()) };
        forget(x);
        if code == 1 {
            debug_assert!(!char_ptr.is_null());
            unsafe { Err(LLVMString::new(char_ptr)) }
        } else {
            Ok(())
        }
    }
    pub fn get_or_insert_comdat(&self, name: &str) -> Comdat {
        let c_string = to_c_str(name);
        let comdat_ptr = unsafe { LLVMGetOrInsertComdat(self.module.get(), c_string.as_ptr()) };
        unsafe { Comdat::new(comdat_ptr) }
    }
    pub fn get_flag(&self, key: &str) -> Option<MetadataValue<'ctx>> {
        let flag = unsafe { LLVMGetModuleFlag(self.module.get(), key.as_ptr() as *const ::libc::c_char, key.len()) };
        if flag.is_null() {
            return None;
        }
        let flag_value = unsafe { LLVMMetadataAsValue(LLVMGetModuleContext(self.module.get()), flag) };
        unsafe { Some(MetadataValue::new(flag_value)) }
    }
    pub fn add_metadata_flag(&self, key: &str, behavior: FlagBehavior, flag: MetadataValue<'ctx>) {
        let md = flag.as_metadata_ref();
        unsafe {
            LLVMAddModuleFlag(
                self.module.get(),
                behavior.into(),
                key.as_ptr() as *mut ::libc::c_char,
                key.len(),
                md,
            )
        }
    }
    pub fn add_basic_value_flag<BV: BasicValue<'ctx>>(&self, key: &str, behavior: FlagBehavior, flag: BV) {
        let md = unsafe { LLVMValueAsMetadata(flag.as_value_ref()) };
        unsafe {
            LLVMAddModuleFlag(
                self.module.get(),
                behavior.into(),
                key.as_ptr() as *mut ::libc::c_char,
                key.len(),
                md,
            )
        }
    }
    pub fn strip_debug_info(&self) -> bool {
        unsafe { LLVMStripModuleDebugInfo(self.module.get()) == 1 }
    }
    pub fn get_debug_metadata_version(&self) -> libc::c_uint {
        unsafe { LLVMGetModuleDebugMetadataVersion(self.module.get()) }
    }
    pub fn create_debug_info_builder(
        &self,
        allow_unresolved: bool,
        language: DWARFSourceLanguage,
        filename: &str,
        directory: &str,
        producer: &str,
        is_optimized: bool,
        flags: &str,
        runtime_ver: libc::c_uint,
        split_name: &str,
        kind: DWARFEmissionKind,
        dwo_id: libc::c_uint,
        split_debug_inlining: bool,
        debug_info_for_profiling: bool,
        sysroot: &str,
        sdk: &str,
    ) -> (DebugInfoBuilder<'ctx>, DICompileUnit<'ctx>) {
        DebugInfoBuilder::new(
            self,
            allow_unresolved,
            language,
            filename,
            directory,
            producer,
            is_optimized,
            flags,
            runtime_ver,
            split_name,
            kind,
            dwo_id,
            split_debug_inlining,
            debug_info_for_profiling,
            sysroot,
            sdk,
        )
    }
    pub fn run_passes(
        &self,
        passes: &str,
        machine: &TargetMachine,
        options: PassBuilderOptions,
    ) -> Result<(), LLVMString> {
        unsafe {
            let error = LLVMRunPasses(self.module.get(), to_c_str(passes).as_ptr(), machine.raw, options.raw);
            if error == std::ptr::null_mut() {
                Ok(())
            } else {
                let message = LLVMGetErrorMessage(error);
                Err(LLVMString::new(message as *const libc::c_char))
            }
        }
    }
}
impl Clone for Module<'_> {
    fn clone(&self) -> Self {
        let verify = self.verify();
        assert!(
            verify.is_ok(),
            "Cloning a Module seems to segfault when module is not valid. We are preventing that here. Error: {}",
            verify.unwrap_err()
        );
        unsafe { Module::new(LLVMCloneModule(self.module.get())) }
    }
}
impl Drop for Module<'_> {
    fn drop(&mut self) {
        if self.owned_by_ee.borrow_mut().take().is_none() {
            unsafe {
                LLVMDisposeModule(self.module.get());
            }
        }
    }
}

#[llvm_enum(LLVMModuleFlagBehavior)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FlagBehavior {
    #[llvm_variant(LLVMModuleFlagBehaviorError)]
    Error,
    #[llvm_variant(LLVMModuleFlagBehaviorWarning)]
    Warning,
    #[llvm_variant(LLVMModuleFlagBehaviorRequire)]
    Require,
    #[llvm_variant(LLVMModuleFlagBehaviorOverride)]
    Override,
    #[llvm_variant(LLVMModuleFlagBehaviorAppend)]
    Append,
    #[llvm_variant(LLVMModuleFlagBehaviorAppendUnique)]
    AppendUnique,
}

#[derive(Debug)]
pub struct FunctionIterator<'ctx>(FunctionIteratorInner<'ctx>);

#[derive(Debug)]
enum FunctionIteratorInner<'ctx> {
    Empty,
    Start(FunctionValue<'ctx>),
    Previous(FunctionValue<'ctx>),
}
impl<'ctx> FunctionIterator<'ctx> {
    fn from_module(module: &Module<'ctx>) -> Self {
        use FunctionIteratorInner::*;
        match module.get_first_function() {
            None => Self(Empty),
            Some(first) => Self(Start(first)),
        }
    }
}
impl<'ctx> Iterator for FunctionIterator<'ctx> {
    type Item = FunctionValue<'ctx>;
    fn next(&mut self) -> Option<Self::Item> {
        use FunctionIteratorInner::*;
        match self.0 {
            Empty => None,
            Start(first) => {
                self.0 = Previous(first);
                Some(first)
            },
            Previous(prev) => match prev.get_next_function() {
                Some(current) => {
                    self.0 = Previous(current);
                    Some(current)
                },
                None => None,
            },
        }
    }
}

#[derive(Debug)]
pub struct GlobalIterator<'ctx>(GlobalIteratorInner<'ctx>);

#[derive(Debug)]
enum GlobalIteratorInner<'ctx> {
    Empty,
    Start(GlobalValue<'ctx>),
    Previous(GlobalValue<'ctx>),
}
impl<'ctx> GlobalIterator<'ctx> {
    fn from_module(module: &Module<'ctx>) -> Self {
        use GlobalIteratorInner::*;
        match module.get_first_global() {
            None => Self(Empty),
            Some(first) => Self(Start(first)),
        }
    }
}
impl<'ctx> Iterator for GlobalIterator<'ctx> {
    type Item = GlobalValue<'ctx>;
    fn next(&mut self) -> Option<Self::Item> {
        use GlobalIteratorInner::*;
        match self.0 {
            Empty => None,
            Start(first) => {
                self.0 = Previous(first);
                Some(first)
            },
            Previous(prev) => match prev.get_next_global() {
                Some(current) => {
                    self.0 = Previous(current);
                    Some(current)
                },
                None => None,
            },
        }
    }
}
