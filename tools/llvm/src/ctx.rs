use libc::c_void;
use llvm_lib::core::*;
use llvm_lib::ir_reader::LLVMParseIRInContext;
use llvm_lib::prelude::*;
use llvm_lib::target::*;
use once_cell::sync::Lazy;
use parking_lot::{Mutex, MutexGuard};
use std::marker::PhantomData;
use std::mem::forget;
use std::ptr;
use std::thread_local;

use crate::builder::Builder;
use crate::target::TargetData;
use crate::typ::*;
use crate::val::*;
use crate::*;

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
    _marker: PhantomData<&'ctx Context>,
}
impl<'ctx> ContextRef<'ctx> {
    pub unsafe fn new(x: LLVMContextRef) -> Self {
        ContextRef {
            ctx: ContextImpl::new(x),
            _marker: PhantomData,
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
