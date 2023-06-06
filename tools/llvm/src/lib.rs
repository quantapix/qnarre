pub mod block;
pub mod builder;
pub mod ctx;
pub mod debug;
pub mod execution_engine;
pub mod llvm_lib;
pub mod mlir;
pub mod module;
pub mod pass;
pub mod target;
pub mod typ;
pub mod utils;
pub mod val;

use llvm_lib::comdata::*;
use llvm_lib::core::*;
use llvm_lib::object::*;
use llvm_lib::prelude::*;
use llvm_lib::*;
use std::convert::TryFrom;
use std::ffi::CStr;
use std::fmt;
use std::mem::{forget, MaybeUninit};
use std::path::Path;
use std::ptr;
use std::slice;

use crate::module::Module;
use crate::typ::{AnyTypeEnum, AsTypeRef, BasicTypeEnum};
use crate::utils::{to_c_str, LLVMString, LLVMString, LLVMStringOrRaw};
use crate::val::FunctionValue;

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
