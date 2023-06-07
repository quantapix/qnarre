use libc::{c_char, c_int, c_void};
use llvm_lib::comdata::*;
use llvm_lib::core::*;
use llvm_lib::error_handling::*;
use llvm_lib::execution_engine::*;
use llvm_lib::object::*;
use llvm_lib::prelude::*;
use llvm_lib::support::LLVMLoadLibraryPermanently;
use llvm_lib::*;
use std::borrow::Cow;
use std::convert::TryFrom;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt::{self, Debug, Display, Formatter};
use std::marker::PhantomData;
use std::mem::{forget, size_of, transmute_copy, MaybeUninit};
use std::ops::Deref;
use std::path::Path;
use std::ptr;
use std::rc::Rc;
use std::slice;

pub mod builder;
pub mod ctx;
pub mod debug;
pub mod llvm_lib;
pub mod mlir;
pub mod module;
pub mod pass;
pub mod target;
pub mod typ;
pub mod val;

use crate::ctx::{Context, ContextRef};
use crate::module::Module;
use crate::target::TargetData;
use crate::typ::*;
use crate::val::*;
use crate::{to_c_str, LLVMString};

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
    pub fn is_type(self) -> bool {
        unsafe { LLVMIsTypeAttribute(self.attribute) == 1 }
    }
    pub fn get_named_enum_kind_id(name: &str) -> u32 {
        unsafe { LLVMGetEnumAttributeKindForName(name.as_ptr() as *const ::libc::c_char, name.len()) }
    }
    pub fn get_enum_kind_id(self) -> u32 {
        assert!(self.get_enum_kind_id_is_valid()); // FIXME: SubTypes
        unsafe { LLVMGetEnumAttributeKind(self.attribute) }
    }
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

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct AddressSpace(u32);
impl Default for AddressSpace {
    fn default() -> Self {
        AddressSpace(0)
    }
}
impl From<u16> for AddressSpace {
    fn from(val: u16) -> Self {
        AddressSpace(val as u32)
    }
}
impl TryFrom<u32> for AddressSpace {
    type Error = ();
    fn try_from(val: u32) -> Result<Self, Self::Error> {
        if val < 1 << 24 {
            Ok(AddressSpace(val))
        } else {
            Err(())
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub struct BasicBlock<'ctx> {
    pub(crate) basic_block: LLVMBasicBlockRef,
    _marker: PhantomData<&'ctx ()>,
}
impl<'ctx> BasicBlock<'ctx> {
    pub(crate) unsafe fn new(basic_block: LLVMBasicBlockRef) -> Option<Self> {
        if basic_block.is_null() {
            return None;
        }
        assert!(!LLVMIsABasicBlock(basic_block as LLVMValueRef).is_null());
        Some(BasicBlock {
            basic_block,
            _marker: PhantomData,
        })
    }
    pub fn as_mut_ptr(&self) -> LLVMBasicBlockRef {
        self.basic_block
    }
    pub fn get_parent(self) -> Option<FunctionValue<'ctx>> {
        unsafe { FunctionValue::new(LLVMGetBasicBlockParent(self.basic_block)) }
    }
    pub fn get_previous_basic_block(self) -> Option<BasicBlock<'ctx>> {
        self.get_parent()?;
        unsafe { BasicBlock::new(LLVMGetPreviousBasicBlock(self.basic_block)) }
    }
    pub fn get_next_basic_block(self) -> Option<BasicBlock<'ctx>> {
        self.get_parent()?;
        unsafe { BasicBlock::new(LLVMGetNextBasicBlock(self.basic_block)) }
    }
    pub fn move_before(self, basic_block: BasicBlock<'ctx>) -> Result<(), ()> {
        if self.get_parent().is_none() || basic_block.get_parent().is_none() {
            return Err(());
        }
        unsafe { LLVMMoveBasicBlockBefore(self.basic_block, basic_block.basic_block) }
        Ok(())
    }
    pub fn move_after(self, basic_block: BasicBlock<'ctx>) -> Result<(), ()> {
        if self.get_parent().is_none() || basic_block.get_parent().is_none() {
            return Err(());
        }
        unsafe { LLVMMoveBasicBlockAfter(self.basic_block, basic_block.basic_block) }
        Ok(())
    }
    pub fn get_first_instruction(self) -> Option<InstructionValue<'ctx>> {
        let value = unsafe { LLVMGetFirstInstruction(self.basic_block) };
        if value.is_null() {
            return None;
        }
        unsafe { Some(InstructionValue::new(value)) }
    }
    pub fn get_last_instruction(self) -> Option<InstructionValue<'ctx>> {
        let value = unsafe { LLVMGetLastInstruction(self.basic_block) };
        if value.is_null() {
            return None;
        }
        unsafe { Some(InstructionValue::new(value)) }
    }
    pub fn get_instruction_with_name(self, name: &str) -> Option<InstructionValue<'ctx>> {
        let instruction = self.get_first_instruction()?;
        instruction.get_instruction_with_name(name)
    }
    pub fn get_terminator(self) -> Option<InstructionValue<'ctx>> {
        let value = unsafe { LLVMGetBasicBlockTerminator(self.basic_block) };
        if value.is_null() {
            return None;
        }
        unsafe { Some(InstructionValue::new(value)) }
    }
    pub fn remove_from_function(self) -> Result<(), ()> {
        if self.get_parent().is_none() {
            return Err(());
        }
        unsafe { LLVMRemoveBasicBlockFromParent(self.basic_block) }
        Ok(())
    }
    pub unsafe fn delete(self) -> Result<(), ()> {
        if self.get_parent().is_none() {
            return Err(());
        }
        LLVMDeleteBasicBlock(self.basic_block);
        Ok(())
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        unsafe { ContextRef::new(LLVMGetTypeContext(LLVMTypeOf(LLVMBasicBlockAsValue(self.basic_block)))) }
    }
    pub fn get_name(&self) -> &CStr {
        let ptr = unsafe { LLVMGetBasicBlockName(self.basic_block) };
        unsafe { CStr::from_ptr(ptr) }
    }
    pub fn set_name(&self, name: &str) {
        let c_string = to_c_str(name);
        use llvm_lib::core::LLVMSetValueName2;
        unsafe { LLVMSetValueName2(LLVMBasicBlockAsValue(self.basic_block), c_string.as_ptr(), name.len()) };
    }
    pub fn replace_all_uses_with(self, other: &BasicBlock<'ctx>) {
        let value = unsafe { LLVMBasicBlockAsValue(self.basic_block) };
        let other = unsafe { LLVMBasicBlockAsValue(other.basic_block) };
        if value != other {
            unsafe {
                LLVMReplaceAllUsesWith(value, other);
            }
        }
    }
    pub fn get_first_use(self) -> Option<BasicValueUse<'ctx>> {
        let use_ = unsafe { LLVMGetFirstUse(LLVMBasicBlockAsValue(self.basic_block)) };
        if use_.is_null() {
            return None;
        }
        unsafe { Some(BasicValueUse::new(use_)) }
    }
    pub unsafe fn get_address(self) -> Option<PointerValue<'ctx>> {
        let parent = self.get_parent()?;
        self.get_previous_basic_block()?;
        let value = PointerValue::new(LLVMBlockAddress(parent.as_value_ref(), self.basic_block));
        if value.is_null() {
            return None;
        }
        Some(value)
    }
}
impl fmt::Debug for BasicBlock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let llvm_value = unsafe { CStr::from_ptr(LLVMPrintValueToString(self.basic_block as LLVMValueRef)) };
        let llvm_type = unsafe { CStr::from_ptr(LLVMPrintTypeToString(LLVMTypeOf(self.basic_block as LLVMValueRef))) };
        let is_const = unsafe { LLVMIsConstant(self.basic_block as LLVMValueRef) == 1 };
        f.debug_struct("BasicBlock")
            .field("address", &self.basic_block)
            .field("is_const", &is_const)
            .field("llvm_value", &llvm_value)
            .field("llvm_type", &llvm_type)
            .finish()
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
    pub(crate) fn new(thread_local_mode: LLVMThreadLocalMode) -> Option<Self> {
        match thread_local_mode {
            LLVMThreadLocalMode::LLVMGeneralDynamicTLSModel => Some(ThreadLocalMode::GeneralDynamicTLSModel),
            LLVMThreadLocalMode::LLVMLocalDynamicTLSModel => Some(ThreadLocalMode::LocalDynamicTLSModel),
            LLVMThreadLocalMode::LLVMInitialExecTLSModel => Some(ThreadLocalMode::InitialExecTLSModel),
            LLVMThreadLocalMode::LLVMLocalExecTLSModel => Some(ThreadLocalMode::LocalExecTLSModel),
            LLVMThreadLocalMode::LLVMNotThreadLocal => None,
        }
    }
    pub(crate) fn as_llvm_mode(self) -> LLVMThreadLocalMode {
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
        let id = unsafe { LLVMLookupIntrinsicID(name.as_ptr() as *const ::libc::c_char, name.len()) };
        if id == 0 {
            return None;
        }
        Some(unsafe { Intrinsic::new(id) })
    }
    pub fn is_overloaded(&self) -> bool {
        unsafe { LLVMIntrinsicIsOverloaded(self.id) != 0 }
    }
    pub fn get_declaration<'ctx>(
        &self,
        module: &Module<'ctx>,
        param_types: &[BasicTypeEnum],
    ) -> Option<FunctionValue<'ctx>> {
        let mut param_types: Vec<LLVMTypeRef> = param_types.iter().map(|val| val.as_type_ref()).collect();
        if self.is_overloaded() && param_types.is_empty() {
            return None;
        }
        let res = unsafe {
            FunctionValue::new(LLVMGetIntrinsicDeclaration(
                module.module.get(),
                self.id,
                param_types.as_mut_ptr(),
                param_types.len(),
            ))
        };
        res
    }
}

#[derive(Debug)]
pub struct MemoryBuffer {
    pub(crate) memory_buffer: LLVMMemoryBufferRef,
}
impl MemoryBuffer {
    pub unsafe fn new(memory_buffer: LLVMMemoryBufferRef) -> Self {
        assert!(!memory_buffer.is_null());
        MemoryBuffer { memory_buffer }
    }
    pub fn as_mut_ptr(&self) -> LLVMMemoryBufferRef {
        self.memory_buffer
    }
    pub fn create_from_file(path: &Path) -> Result<Self, LLVMString> {
        let path = to_c_str(path.to_str().expect("Did not find a valid Unicode path string"));
        let mut memory_buffer = ptr::null_mut();
        let mut err_string = MaybeUninit::uninit();
        let return_code = unsafe {
            LLVMCreateMemoryBufferWithContentsOfFile(
                path.as_ptr() as *const ::libc::c_char,
                &mut memory_buffer,
                err_string.as_mut_ptr(),
            )
        };
        if return_code == 1 {
            unsafe {
                return Err(LLVMString::new(err_string.assume_init()));
            }
        }
        unsafe { Ok(MemoryBuffer::new(memory_buffer)) }
    }
    pub fn create_from_stdin() -> Result<Self, LLVMString> {
        let mut memory_buffer = ptr::null_mut();
        let mut err_string = MaybeUninit::uninit();
        let return_code = unsafe { LLVMCreateMemoryBufferWithSTDIN(&mut memory_buffer, err_string.as_mut_ptr()) };
        if return_code == 1 {
            unsafe {
                return Err(LLVMString::new(err_string.assume_init()));
            }
        }
        unsafe { Ok(MemoryBuffer::new(memory_buffer)) }
    }
    pub fn create_from_memory_range(input: &[u8], name: &str) -> Self {
        let name_c_string = to_c_str(name);
        let memory_buffer = unsafe {
            LLVMCreateMemoryBufferWithMemoryRange(
                input.as_ptr() as *const ::libc::c_char,
                input.len(),
                name_c_string.as_ptr(),
                false as i32,
            )
        };
        unsafe { MemoryBuffer::new(memory_buffer) }
    }
    pub fn create_from_memory_range_copy(input: &[u8], name: &str) -> Self {
        let name_c_string = to_c_str(name);
        let memory_buffer = unsafe {
            LLVMCreateMemoryBufferWithMemoryRangeCopy(
                input.as_ptr() as *const ::libc::c_char,
                input.len(),
                name_c_string.as_ptr(),
            )
        };
        unsafe { MemoryBuffer::new(memory_buffer) }
    }
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            let start = LLVMGetBufferStart(self.memory_buffer);
            slice::from_raw_parts(start as *const _, self.get_size())
        }
    }
    pub fn get_size(&self) -> usize {
        unsafe { LLVMGetBufferSize(self.memory_buffer) }
    }
    pub fn create_object_file(self) -> Result<ObjectFile, ()> {
        let object_file = unsafe { LLVMCreateObjectFile(self.memory_buffer) };
        forget(self);
        if object_file.is_null() {
            return Err(());
        }
        unsafe { Ok(ObjectFile::new(object_file)) }
    }
}
impl Drop for MemoryBuffer {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeMemoryBuffer(self.memory_buffer);
        }
    }
}

#[derive(Debug)]
pub struct ObjectFile {
    object_file: LLVMObjectFileRef,
}
impl ObjectFile {
    pub unsafe fn new(object_file: LLVMObjectFileRef) -> Self {
        assert!(!object_file.is_null());
        ObjectFile { object_file }
    }
    pub fn as_mut_ptr(&self) -> LLVMObjectFileRef {
        self.object_file
    }
    pub fn get_sections(&self) -> SectionIterator {
        let section_iterator = unsafe { LLVMGetSections(self.object_file) };
        unsafe { SectionIterator::new(section_iterator, self.object_file) }
    }
    pub fn get_symbols(&self) -> SymbolIterator {
        let symbol_iterator = unsafe { LLVMGetSymbols(self.object_file) };
        unsafe { SymbolIterator::new(symbol_iterator, self.object_file) }
    }
}
impl Drop for ObjectFile {
    fn drop(&mut self) {
        unsafe { LLVMDisposeObjectFile(self.object_file) }
    }
}

#[derive(Debug)]
pub struct SectionIterator {
    section_iterator: LLVMSectionIteratorRef,
    object_file: LLVMObjectFileRef,
    before_first: bool,
}
impl SectionIterator {
    pub unsafe fn new(section_iterator: LLVMSectionIteratorRef, object_file: LLVMObjectFileRef) -> Self {
        assert!(!section_iterator.is_null());
        assert!(!object_file.is_null());
        SectionIterator {
            section_iterator,
            object_file,
            before_first: true,
        }
    }
    pub fn as_mut_ptr(&self) -> (LLVMSectionIteratorRef, LLVMObjectFileRef) {
        (self.section_iterator, self.object_file)
    }
}
impl Iterator for SectionIterator {
    type Item = Section;
    fn next(&mut self) -> Option<Self::Item> {
        if self.before_first {
            self.before_first = false;
        } else {
            unsafe {
                LLVMMoveToNextSection(self.section_iterator);
            }
        }
        let at_end = unsafe { LLVMIsSectionIteratorAtEnd(self.object_file, self.section_iterator) == 1 };
        if at_end {
            return None;
        }
        Some(unsafe { Section::new(self.section_iterator, self.object_file) })
    }
}
impl Drop for SectionIterator {
    fn drop(&mut self) {
        unsafe { LLVMDisposeSectionIterator(self.section_iterator) }
    }
}

#[derive(Debug)]
pub struct Section {
    section: LLVMSectionIteratorRef,
    object_file: LLVMObjectFileRef,
}
impl Section {
    pub unsafe fn new(section: LLVMSectionIteratorRef, object_file: LLVMObjectFileRef) -> Self {
        assert!(!section.is_null());
        assert!(!object_file.is_null());
        Section { section, object_file }
    }
    pub unsafe fn as_mut_ptr(&self) -> (LLVMSectionIteratorRef, LLVMObjectFileRef) {
        (self.section, self.object_file)
    }
    pub fn get_name(&self) -> Option<&CStr> {
        let name = unsafe { LLVMGetSectionName(self.section) };
        if !name.is_null() {
            Some(unsafe { CStr::from_ptr(name) })
        } else {
            None
        }
    }
    pub fn size(&self) -> u64 {
        unsafe { LLVMGetSectionSize(self.section) }
    }
    pub fn get_contents(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(LLVMGetSectionContents(self.section) as *const u8, self.size() as usize) }
    }
    pub fn get_address(&self) -> u64 {
        unsafe { LLVMGetSectionAddress(self.section) }
    }
    pub fn get_relocations(&self) -> RelocationIterator {
        let relocation_iterator = unsafe { LLVMGetRelocations(self.section) };
        unsafe { RelocationIterator::new(relocation_iterator, self.section, self.object_file) }
    }
}

#[derive(Debug)]
pub struct RelocationIterator {
    relocation_iterator: LLVMRelocationIteratorRef,
    section_iterator: LLVMSectionIteratorRef,
    object_file: LLVMObjectFileRef,
    before_first: bool,
}
impl RelocationIterator {
    pub unsafe fn new(
        relocation_iterator: LLVMRelocationIteratorRef,
        section_iterator: LLVMSectionIteratorRef,
        object_file: LLVMObjectFileRef,
    ) -> Self {
        assert!(!relocation_iterator.is_null());
        assert!(!section_iterator.is_null());
        assert!(!object_file.is_null());
        RelocationIterator {
            relocation_iterator,
            section_iterator,
            object_file,
            before_first: true,
        }
    }
    pub fn as_mut_ptr(&self) -> (LLVMRelocationIteratorRef, LLVMSectionIteratorRef, LLVMObjectFileRef) {
        (self.relocation_iterator, self.section_iterator, self.object_file)
    }
}
impl Iterator for RelocationIterator {
    type Item = Relocation;
    fn next(&mut self) -> Option<Self::Item> {
        if self.before_first {
            self.before_first = false;
        } else {
            unsafe { LLVMMoveToNextRelocation(self.relocation_iterator) }
        }
        let at_end = unsafe { LLVMIsRelocationIteratorAtEnd(self.section_iterator, self.relocation_iterator) == 1 };
        if at_end {
            return None;
        }
        Some(unsafe { Relocation::new(self.relocation_iterator, self.object_file) })
    }
}
impl Drop for RelocationIterator {
    fn drop(&mut self) {
        unsafe { LLVMDisposeRelocationIterator(self.relocation_iterator) }
    }
}

#[derive(Debug)]
pub struct Relocation {
    relocation: LLVMRelocationIteratorRef,
    object_file: LLVMObjectFileRef,
}
impl Relocation {
    pub unsafe fn new(relocation: LLVMRelocationIteratorRef, object_file: LLVMObjectFileRef) -> Self {
        assert!(!relocation.is_null());
        assert!(!object_file.is_null());
        Relocation {
            relocation,
            object_file,
        }
    }
    pub fn as_mut_ptr(&self) -> (LLVMRelocationIteratorRef, LLVMObjectFileRef) {
        (self.relocation, self.object_file)
    }
    pub fn get_offset(&self) -> u64 {
        unsafe { LLVMGetRelocationOffset(self.relocation) }
    }
    pub fn get_symbols(&self) -> SymbolIterator {
        let symbol_iterator = unsafe { LLVMGetRelocationSymbol(self.relocation) };
        unsafe { SymbolIterator::new(symbol_iterator, self.object_file) }
    }
    pub fn get_type(&self) -> (u64, &CStr) {
        let type_int = unsafe { LLVMGetRelocationType(self.relocation) };
        let type_name = unsafe { CStr::from_ptr(LLVMGetRelocationTypeName(self.relocation)) };
        (type_int, type_name)
    }
    pub fn get_value(&self) -> &CStr {
        unsafe { CStr::from_ptr(LLVMGetRelocationValueString(self.relocation)) }
    }
}

#[derive(Debug)]
pub struct SymbolIterator {
    symbol_iterator: LLVMSymbolIteratorRef,
    object_file: LLVMObjectFileRef,
    before_first: bool,
}
impl SymbolIterator {
    pub unsafe fn new(symbol_iterator: LLVMSymbolIteratorRef, object_file: LLVMObjectFileRef) -> Self {
        assert!(!symbol_iterator.is_null());
        assert!(!object_file.is_null());
        SymbolIterator {
            symbol_iterator,
            object_file,
            before_first: true,
        }
    }
    pub fn as_mut_ptr(&self) -> (LLVMSymbolIteratorRef, LLVMObjectFileRef) {
        (self.symbol_iterator, self.object_file)
    }
}
impl Iterator for SymbolIterator {
    type Item = Symbol;
    fn next(&mut self) -> Option<Self::Item> {
        if self.before_first {
            self.before_first = false;
        } else {
            unsafe { LLVMMoveToNextSymbol(self.symbol_iterator) }
        }
        let at_end = unsafe { LLVMIsSymbolIteratorAtEnd(self.object_file, self.symbol_iterator) == 1 };
        if at_end {
            return None;
        }
        Some(unsafe { Symbol::new(self.symbol_iterator) })
    }
}
impl Drop for SymbolIterator {
    fn drop(&mut self) {
        unsafe { LLVMDisposeSymbolIterator(self.symbol_iterator) }
    }
}

#[derive(Debug)]
pub struct Symbol {
    symbol: LLVMSymbolIteratorRef,
}
impl Symbol {
    pub unsafe fn new(symbol: LLVMSymbolIteratorRef) -> Self {
        assert!(!symbol.is_null());
        Symbol { symbol }
    }
    pub fn as_mut_ptr(&self) -> LLVMSymbolIteratorRef {
        self.symbol
    }
    pub fn get_name(&self) -> Option<&CStr> {
        let name = unsafe { LLVMGetSymbolName(self.symbol) };
        if !name.is_null() {
            Some(unsafe { CStr::from_ptr(name) })
        } else {
            None
        }
    }
    pub fn size(&self) -> u64 {
        unsafe { LLVMGetSymbolSize(self.symbol) }
    }
    pub fn get_address(&self) -> u64 {
        unsafe { LLVMGetSymbolAddress(self.symbol) }
    }
}

#[derive(Eq)]
pub struct DataLayout {
    pub(crate) data_layout: LLVMStringOrRaw,
}
impl DataLayout {
    pub(crate) unsafe fn new_owned(data_layout: *const ::libc::c_char) -> DataLayout {
        debug_assert!(!data_layout.is_null());
        DataLayout {
            data_layout: LLVMStringOrRaw::Owned(LLVMString::new(data_layout)),
        }
    }
    pub(crate) unsafe fn new_borrowed(data_layout: *const ::libc::c_char) -> DataLayout {
        debug_assert!(!data_layout.is_null());
        DataLayout {
            data_layout: LLVMStringOrRaw::Borrowed(data_layout),
        }
    }
    pub fn as_str(&self) -> &CStr {
        self.data_layout.as_str()
    }
    pub fn as_ptr(&self) -> *const ::libc::c_char {
        match self.data_layout {
            LLVMStringOrRaw::Owned(ref llvm_string) => llvm_string.ptr,
            LLVMStringOrRaw::Borrowed(ptr) => ptr,
        }
    }
}
impl PartialEq for DataLayout {
    fn eq(&self, other: &DataLayout) -> bool {
        self.as_str() == other.as_str()
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
pub struct Comdat(pub(crate) LLVMComdatRef);
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
    diagnostic_info: LLVMDiagnosticInfoRef,
}
impl DiagnosticInfo {
    pub unsafe fn new(diagnostic_info: LLVMDiagnosticInfoRef) -> Self {
        DiagnosticInfo { diagnostic_info }
    }
    pub fn get_description(&self) -> *mut ::libc::c_char {
        unsafe { LLVMGetDiagInfoDescription(self.diagnostic_info) }
    }
    pub fn severity_is_error(&self) -> bool {
        unsafe {
            match LLVMGetDiagInfoSeverity(self.diagnostic_info) {
                LLVMDiagnosticSeverity::LLVMDSError => true,
                _ => false,
            }
        }
    }
}

#[derive(Eq)]
pub struct LLVMString {
    pub ptr: *const c_char,
}
impl LLVMString {
    pub unsafe fn new(ptr: *const c_char) -> Self {
        LLVMString { ptr }
    }
    pub fn to_string(&self) -> String {
        (*self).to_string_lossy().into_owned()
    }
    pub fn create_from_c_str(string: &CStr) -> LLVMString {
        unsafe { LLVMString::new(LLVMCreateMessage(string.as_ptr() as *const _)) }
    }
    pub fn create_from_str(string: &str) -> LLVMString {
        debug_assert_eq!(string.as_bytes()[string.as_bytes().len() - 1], 0);

        unsafe { LLVMString::new(LLVMCreateMessage(string.as_ptr() as *const _)) }
    }
}
impl Deref for LLVMString {
    type Target = CStr;
    fn deref(&self) -> &Self::Target {
        unsafe { CStr::from_ptr(self.ptr) }
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
    fn eq(&self, other: &LLVMString) -> bool {
        **self == **other
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
            LLVMDisposeMessage(self.ptr as *mut _);
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
    fn eq(&self, other: &LLVMStringOrRaw) -> bool {
        self.as_str() == other.as_str()
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
            execution_engine: Some(ExecEngineInner(execution_engine, PhantomData)),
            target_data: Some(TargetData::new(target_data)),
            jit_mode,
        }
    }
    pub fn as_mut_ptr(&self) -> LLVMExecutionEngineRef {
        self.execution_engine_inner()
    }
    pub(crate) fn execution_engine_rc(&self) -> &Rc<LLVMExecutionEngineRef> {
        &self.execution_engine.as_ref().expect(EE_INNER_PANIC).0
    }
    #[inline]
    pub(crate) fn execution_engine_inner(&self) -> LLVMExecutionEngineRef {
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
    pub fn remove_module(&self, module: &Module<'ctx>) -> Result<(), RemoveModuleError> {
        match *module.owned_by_ee.borrow() {
            Some(ref ee) if ee.execution_engine_inner() != self.execution_engine_inner() => {
                return Err(RemoveModuleError::IncorrectModuleOwner)
            },
            None => return Err(RemoveModuleError::ModuleNotOwned),
            _ => (),
        }
        let mut new_module = MaybeUninit::uninit();
        let mut err_string = MaybeUninit::uninit();
        let code = unsafe {
            LLVMRemoveModule(
                self.execution_engine_inner(),
                module.module.get(),
                new_module.as_mut_ptr(),
                err_string.as_mut_ptr(),
            )
        };
        if code == 1 {
            unsafe {
                return Err(RemoveModuleError::LLVMError(LLVMString::new(err_string.assume_init())));
            }
        }
        let new_module = unsafe { new_module.assume_init() };
        module.module.set(new_module);
        *module.owned_by_ee.borrow_mut() = None;
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
        let mut args: Vec<LLVMGenericValueRef> = args.iter().map(|val| val.generic_value).collect();
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
struct ExecEngineInner<'ctx>(Rc<LLVMExecutionEngineRef>, PhantomData<&'ctx Context>);
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
