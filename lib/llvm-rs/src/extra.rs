#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]

use super::prelude::*;

// llvm-c/Analysis.h
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LLVMVerifierFailureAction {
    LLVMAbortProcessAction = 0,
    LLVMPrintMessageAction = 1,
    LLVMReturnStatusAction = 2,
}

extern "C" {
    pub fn LLVMVerifyFunction(Fn: LLVMValueRef, Action: LLVMVerifierFailureAction) -> LLVMBool;
    pub fn LLVMVerifyModule(M: LLVMModuleRef, Action: LLVMVerifierFailureAction, OutMessage: *mut *mut ::libc::c_char) -> LLVMBool;
    pub fn LLVMViewFunctionCFG(Fn: LLVMValueRef);
    pub fn LLVMViewFunctionCFGOnly(Fn: LLVMValueRef);
    // llvm-c/BitReader.h
    pub fn LLVMGetBitcodeModule2(MemBuf: LLVMMemoryBufferRef, OutM: *mut LLVMModuleRef) -> LLVMBool;
    pub fn LLVMGetBitcodeModuleInContext2(ContextRef: LLVMContextRef, MemBuf: LLVMMemoryBufferRef, OutM: *mut LLVMModuleRef) -> LLVMBool;
    pub fn LLVMParseBitcode2(MemBuf: LLVMMemoryBufferRef, OutModule: *mut LLVMModuleRef) -> LLVMBool;
    pub fn LLVMParseBitcodeInContext2(ContextRef: LLVMContextRef, MemBuf: LLVMMemoryBufferRef, OutModule: *mut LLVMModuleRef) -> LLVMBool;
    // llvm-c/BitWriter.h
    pub fn LLVMWriteBitcodeToFD(M: LLVMModuleRef, FD: ::libc::c_int, ShouldClose: ::libc::c_int, Unbuffered: ::libc::c_int) -> ::libc::c_int;
    pub fn LLVMWriteBitcodeToFile(M: LLVMModuleRef, Path: *const ::libc::c_char) -> ::libc::c_int;
    pub fn LLVMWriteBitcodeToMemoryBuffer(M: LLVMModuleRef) -> LLVMMemoryBufferRef;
}

// llvm-c/blake3.h
pub const LLVM_BLAKE3_VERSION_STRING: &str = "1.3.1";
pub const LLVM_BLAKE3_KEY_LEN: usize = 32;
pub const LLVM_BLAKE3_OUT_LEN: usize = 32;
pub const LLVM_BLAKE3_BLOCK_LEN: usize = 64;
pub const LLVM_BLAKE3_CHUNK_LEN: usize = 1024;
pub const LLVM_BLAKE3_MAX_DEPTH: usize = 54;

#[repr(C)]
struct llvm_blake3_chunk_state {
    cv: [u32; 8],
    chunk_counter: u64,
    buf: [u8; LLVM_BLAKE3_BLOCK_LEN],
    buf_len: u8,
    blocks_compressed: u8,
    flags: u8,
}

#[repr(C)]
pub struct llvm_blake3_hasher {
    key: [u32; 8],
    chunk: llvm_blake3_chunk_state,
    cv_stack_len: u8,
    cv_stack: [u8; (LLVM_BLAKE3_MAX_DEPTH + 1) * LLVM_BLAKE3_OUT_LEN],
}

extern "C" {
    pub fn llvm_blake3_hasher_finalize_seek(hasher: *mut llvm_blake3_hasher, seek: u64, out: *mut u8, out_len: usize);
    pub fn llvm_blake3_hasher_finalize(hasher: *mut llvm_blake3_hasher, out: *mut u8, out_len: usize);
    pub fn llvm_blake3_hasher_init_derive_key_raw(hasher: *mut llvm_blake3_hasher, context: *const ::libc::c_char, context_len: usize);
    pub fn llvm_blake3_hasher_init_derive_key(hasher: *mut llvm_blake3_hasher, context: *const ::libc::c_char);
    pub fn llvm_blake3_hasher_init_keyed(hasher: *mut llvm_blake3_hasher, key: *const u8);
    pub fn llvm_blake3_hasher_init(hasher: *mut llvm_blake3_hasher);
    pub fn llvm_blake3_hasher_reset(hasher: *mut llvm_blake3_hasher);
    pub fn llvm_blake3_hasher_update(hasher: *mut llvm_blake3_hasher, input: *const ::libc::c_void, input_len: usize);
    pub fn llvm_blake3_version() -> *const ::libc::c_char;
}

// llvm-c/Comdat.h
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LLVMComdatSelectionKind {
    LLVMAnyComdatSelectionKind,
    LLVMExactMatchComdatSelectionKind,
    LLVMLargestComdatSelectionKind,
    LLVMNoDuplicatesComdatSelectionKind,
    LLVMSameSizeComdatSelectionKind,
}

extern "C" {
    pub fn LLVMGetComdat(V: LLVMValueRef) -> LLVMComdatRef;
    pub fn LLVMGetComdatSelectionKind(C: LLVMComdatRef) -> LLVMComdatSelectionKind;
    pub fn LLVMGetOrInsertComdat(M: LLVMModuleRef, Name: *const ::libc::c_char) -> LLVMComdatRef;
    pub fn LLVMSetComdat(V: LLVMValueRef, C: LLVMComdatRef);
    pub fn LLVMSetComdatSelectionKind(C: LLVMComdatRef, Kind: LLVMComdatSelectionKind);
}

// llvm-c/DebugInfo.h
pub type LLVMDIFlags = ::libc::c_int;

pub const LLVMDIFlagZero: LLVMDIFlags = 0;
pub const LLVMDIFlagPrivate: LLVMDIFlags = 1;
pub const LLVMDIFlagProtected: LLVMDIFlags = 2;
pub const LLVMDIFlagPublic: LLVMDIFlags = 3;
pub const LLVMDIFlagFwdDecl: LLVMDIFlags = 1 << 2;
pub const LLVMDIFlagAppleBlock: LLVMDIFlags = 1 << 3;
pub const LLVMDIFlagReservedBit4: LLVMDIFlags = 1 << 4;
pub const LLVMDIFlagVirtual: LLVMDIFlags = 1 << 5;
pub const LLVMDIFlagArtificial: LLVMDIFlags = 1 << 6;
pub const LLVMDIFlagExplicit: LLVMDIFlags = 1 << 7;
pub const LLVMDIFlagPrototyped: LLVMDIFlags = 1 << 8;
pub const LLVMDIFlagObjcClassComplete: LLVMDIFlags = 1 << 9;
pub const LLVMDIFlagObjectPointer: LLVMDIFlags = 1 << 10;
pub const LLVMDIFlagVector: LLVMDIFlags = 1 << 11;
pub const LLVMDIFlagStaticMember: LLVMDIFlags = 1 << 12;
pub const LLVMDIFlagLValueReference: LLVMDIFlags = 1 << 13;
pub const LLVMDIFlagRValueReference: LLVMDIFlags = 1 << 14;
pub const LLVMDIFlagReserved: LLVMDIFlags = 1 << 15;
pub const LLVMDIFlagSingleInheritance: LLVMDIFlags = 1 << 16;
pub const LLVMDIFlagMultipleInheritance: LLVMDIFlags = 2 << 16;
pub const LLVMDIFlagVirtualInheritance: LLVMDIFlags = 3 << 16;
pub const LLVMDIFlagIntroducedVirtual: LLVMDIFlags = 1 << 18;
pub const LLVMDIFlagBitField: LLVMDIFlags = 1 << 19;
pub const LLVMDIFlagNoReturn: LLVMDIFlags = 1 << 20;
pub const LLVMDIFlagTypePassByValue: LLVMDIFlags = 1 << 22;
pub const LLVMDIFlagTypePassByReference: LLVMDIFlags = 1 << 23;
pub const LLVMDIFlagEnumClass: LLVMDIFlags = 1 << 24;
pub const LLVMDIFlagThunk: LLVMDIFlags = 1 << 25;
pub const LLVMDIFlagNonTrivial: LLVMDIFlags = 1 << 26;
pub const LLVMDIFlagBigendian: LLVMDIFlags = 1 << 27;
pub const LLVMDIFlagLittleEndian: LLVMDIFlags = 1 << 28;
pub const LLVMDIFlagIndirectVirtualBase: LLVMDIFlags = (1 << 2) | (1 << 5);
pub const LLVMDIFlagAccessibility: LLVMDIFlags = LLVMDIFlagProtected | LLVMDIFlagPrivate | LLVMDIFlagPublic;
pub const LLVMDIFlagPtrToMemberRep: LLVMDIFlags = LLVMDIFlagSingleInheritance | LLVMDIFlagMultipleInheritance | LLVMDIFlagVirtualInheritance;

#[repr(C)]
#[derive(Debug)]
pub enum LLVMDWARFSourceLanguage {
    LLVMDWARFSourceLanguageC89,
    LLVMDWARFSourceLanguageC,
    LLVMDWARFSourceLanguageAda83,
    LLVMDWARFSourceLanguageC_plus_plus,
    LLVMDWARFSourceLanguageCobol74,
    LLVMDWARFSourceLanguageCobol85,
    LLVMDWARFSourceLanguageFortran77,
    LLVMDWARFSourceLanguageFortran90,
    LLVMDWARFSourceLanguagePascal83,
    LLVMDWARFSourceLanguageModula2,
    LLVMDWARFSourceLanguageJava,
    LLVMDWARFSourceLanguageC99,
    LLVMDWARFSourceLanguageAda95,
    LLVMDWARFSourceLanguageFortran95,
    LLVMDWARFSourceLanguagePLI,
    LLVMDWARFSourceLanguageObjC,
    LLVMDWARFSourceLanguageObjC_plus_plus,
    LLVMDWARFSourceLanguageUPC,
    LLVMDWARFSourceLanguageD,
    LLVMDWARFSourceLanguagePython,
    LLVMDWARFSourceLanguageOpenCL,
    LLVMDWARFSourceLanguageGo,
    LLVMDWARFSourceLanguageModula3,
    LLVMDWARFSourceLanguageHaskell,
    LLVMDWARFSourceLanguageC_plus_plus_03,
    LLVMDWARFSourceLanguageC_plus_plus_11,
    LLVMDWARFSourceLanguageOCaml,
    LLVMDWARFSourceLanguageRust,
    LLVMDWARFSourceLanguageC11,
    LLVMDWARFSourceLanguageSwift,
    LLVMDWARFSourceLanguageJulia,
    LLVMDWARFSourceLanguageDylan,
    LLVMDWARFSourceLanguageC_plus_plus_14,
    LLVMDWARFSourceLanguageFortran03,
    LLVMDWARFSourceLanguageFortran08,
    LLVMDWARFSourceLanguageRenderScript,
    LLVMDWARFSourceLanguageBLISS,
    LLVMDWARFSourceLanguageKotlin,
    LLVMDWARFSourceLanguageZig,
    LLVMDWARFSourceLanguageCrystal,
    LLVMDWARFSourceLanguageC_plus_plus_17,
    LLVMDWARFSourceLanguageC_plus_plus_20,
    LLVMDWARFSourceLanguageC17,
    LLVMDWARFSourceLanguageFortran18,
    LLVMDWARFSourceLanguageAda2005,
    LLVMDWARFSourceLanguageAda2012,
    LLVMDWARFSourceLanguageMips_Assembler,
    LLVMDWARFSourceLanguageGOOGLE_RenderScript,
    LLVMDWARFSourceLanguageBORLAND_Delphi,
}

#[repr(C)]
#[derive(Debug)]
pub enum LLVMDWARFEmissionKind {
    LLVMDWARFEmissionKindNone = 0,
    LLVMDWARFEmissionKindFull,
    LLVMDWARFEmissionKindLineTablesOnly,
}

#[repr(C)]
#[derive(Debug)]
pub enum LLVMMetadataKind {
    LLVMMDStringMetadataKind,
    LLVMConstantAsMetadataMetadataKind,
    LLVMLocalAsMetadataMetadataKind,
    LLVMDistinctMDOperandPlaceholderMetadataKind,
    LLVMMDTupleMetadataKind,
    LLVMDILocationMetadataKind,
    LLVMDIExpressionMetadataKind,
    LLVMDIGlobalVariableExpressionMetadataKind,
    LLVMGenericDINodeMetadataKind,
    LLVMDISubrangeMetadataKind,
    LLVMDIEnumeratorMetadataKind,
    LLVMDIBasicTypeMetadataKind,
    LLVMDIDerivedTypeMetadataKind,
    LLVMDICompositeTypeMetadataKind,
    LLVMDISubroutineTypeMetadataKind,
    LLVMDIFileMetadataKind,
    LLVMDICompileUnitMetadataKind,
    LLVMDISubprogramMetadataKind,
    LLVMDILexicalBlockMetadataKind,
    LLVMDILexicalBlockFileMetadataKind,
    LLVMDINamespaceMetadataKind,
    LLVMDIModuleMetadataKind,
    LLVMDITemplateTypeParameterMetadataKind,
    LLVMDITemplateValueParameterMetadataKind,
    LLVMDIGlobalVariableMetadataKind,
    LLVMDILocalVariableMetadataKind,
    LLVMDILabelMetadataKind,
    LLVMDIObjCPropertyMetadataKind,
    LLVMDIImportedEntityMetadataKind,
    LLVMDIMacroMetadataKind,
    LLVMDIMacroFileMetadataKind,
    LLVMDICommonBlockMetadataKind,
    LLVMDIStringTypeMetadataKind,
    LLVMDIGenericSubrangeMetadataKind,
    LLVMDIArgListMetadataKind,
    LLVMDIAssignIDMetadataKind,
}

pub type LLVMDWARFTypeEncoding = ::libc::c_uint;

#[repr(C)]
#[derive(Debug)]
pub enum LLVMDWARFMacinfoRecordType {
    LLVMDWARFMacinfoRecordTypeDefine = 0x01,
    LLVMDWARFMacinfoRecordTypeMacro = 0x02,
    LLVMDWARFMacinfoRecordTypeStartFile = 0x03,
    LLVMDWARFMacinfoRecordTypeEndFile = 0x04,
    LLVMDWARFMacinfoRecordTypeVendorExt = 0xff,
}

extern "C" {
    pub fn LLVMCreateDIBuilder(M: LLVMModuleRef) -> LLVMDIBuilderRef;
    pub fn LLVMCreateDIBuilderDisallowUnresolved(M: LLVMModuleRef) -> LLVMDIBuilderRef;
    pub fn LLVMDebugMetadataVersion() -> ::libc::c_uint;
    pub fn LLVMDIBuilderCreateArrayType(Builder: LLVMDIBuilderRef, Size: u64, AlignInBits: u32, Ty: LLVMMetadataRef, Subscripts: *mut LLVMMetadataRef, NumSubscripts: ::libc::c_uint) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateArtificialType(Builder: LLVMDIBuilderRef, Type: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateAutoVariable(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, File: LLVMMetadataRef, LineNo: ::libc::c_uint, Ty: LLVMMetadataRef, AlwaysPreserve: LLVMBool, Flags: LLVMDIFlags, AlignInBits: u32) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateBasicType(Builder: LLVMDIBuilderRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, SizeInBits: u64, Encoding: LLVMDWARFTypeEncoding, Flags: LLVMDIFlags) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateBitFieldMemberType(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, File: LLVMMetadataRef, LineNumber: ::libc::c_uint, SizeInBits: u64, OffsetInBits: u64, StorageOffsetInBits: u64, Flags: LLVMDIFlags, Type: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateClassType(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, File: LLVMMetadataRef, LineNumber: ::libc::c_uint, SizeInBits: u64, AlignInBits: u32, OffsetInBits: u64, Flags: LLVMDIFlags, DerivedFrom: LLVMMetadataRef, Elements: *mut LLVMMetadataRef, NumElements: ::libc::c_uint, VTableHolder: LLVMMetadataRef, TemplateParamsNode: LLVMMetadataRef, UniqueIdentifier: *const ::libc::c_char, UniqueIdentifierLen: ::libc::size_t) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateCompileUnit(Builder: LLVMDIBuilderRef, Lang: LLVMDWARFSourceLanguage, FileRef: LLVMMetadataRef, Producer: *const ::libc::c_char, ProducerLen: ::libc::size_t, isOptimized: LLVMBool, Flags: *const ::libc::c_char, FlagsLen: ::libc::size_t, RuntimeVer: ::libc::c_uint, SplitName: *const ::libc::c_char, SplitNameLen: ::libc::size_t, Kind: LLVMDWARFEmissionKind, DWOId: ::libc::c_uint, SplitDebugInlining: LLVMBool, DebugInfoForProfiling: LLVMBool, SysRoot: *const ::libc::c_char, SysRootLen: ::libc::size_t, SDK: *const ::libc::c_char, SDKLen: ::libc::size_t) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateConstantValueExpression(Builder: LLVMDIBuilderRef, Value: u64) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateDebugLocation(Ctx: LLVMContextRef, Line: ::libc::c_uint, Column: ::libc::c_uint, Scope: LLVMMetadataRef, InlinedAt: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateEnumerationType(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, File: LLVMMetadataRef, LineNumber: ::libc::c_uint, SizeInBits: u64, AlignInBits: u32, Elements: *mut LLVMMetadataRef, NumElements: ::libc::c_uint, ClassTy: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateEnumerator(Builder: LLVMDIBuilderRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, Value: i64, IsUnsigned: LLVMBool) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateExpression(Builder: LLVMDIBuilderRef, Addr: *mut u64, Length: ::libc::size_t) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateFile(Builder: LLVMDIBuilderRef, Filename: *const ::libc::c_char, FilenameLen: ::libc::size_t, Directory: *const ::libc::c_char, DirectoryLen: ::libc::size_t) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateForwardDecl(Builder: LLVMDIBuilderRef, Tag: ::libc::c_uint, Name: *const ::libc::c_char, NameLen: ::libc::size_t, Scope: LLVMMetadataRef, File: LLVMMetadataRef, Line: ::libc::c_uint, RuntimeLang: ::libc::c_uint, SizeInBits: u64, AlignInBits: u32, UniqueIdentifier: *const ::libc::c_char, UniqueIdentifierLen: ::libc::size_t) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateFunction(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, LinkageName: *const ::libc::c_char, LinkageNameLen: ::libc::size_t, File: LLVMMetadataRef, LineNo: ::libc::c_uint, Ty: LLVMMetadataRef, IsLocalToUnit: LLVMBool, IsDefinition: LLVMBool, ScopeLine: ::libc::c_uint, Flags: LLVMDIFlags, IsOptimized: LLVMBool) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateGlobalVariableExpression(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, Linkage: *const ::libc::c_char, LinkLen: ::libc::size_t, File: LLVMMetadataRef, LineNo: ::libc::c_uint, Ty: LLVMMetadataRef, LocalToUnit: LLVMBool, Expr: LLVMMetadataRef, Decl: LLVMMetadataRef, AlignInBits: u32) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateImportedDeclaration(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, Decl: LLVMMetadataRef, File: LLVMMetadataRef, Line: ::libc::c_uint, Name: *const ::libc::c_char, NameLen: ::libc::size_t, Elements: *mut LLVMMetadataRef, NumElements: ::libc::c_uint) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateImportedModuleFromAlias(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, ImportedEntity: LLVMMetadataRef, File: LLVMMetadataRef, Line: ::libc::c_uint, Elements: *mut LLVMMetadataRef, NumElements: ::libc::c_uint) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateImportedModuleFromModule(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, M: LLVMMetadataRef, File: LLVMMetadataRef, Line: ::libc::c_uint, Elements: *mut LLVMMetadataRef, NumElements: ::libc::c_uint) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateImportedModuleFromNamespace(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, NS: LLVMMetadataRef, File: LLVMMetadataRef, Line: ::libc::c_uint) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateInheritance(Builder: LLVMDIBuilderRef, Ty: LLVMMetadataRef, BaseTy: LLVMMetadataRef, BaseOffset: u64, VBPtrOffset: u32, Flags: LLVMDIFlags) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateLexicalBlock(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, File: LLVMMetadataRef, Line: ::libc::c_uint, Column: ::libc::c_uint) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateLexicalBlockFile(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, File: LLVMMetadataRef, Discriminator: ::libc::c_uint) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateMacro(Builder: LLVMDIBuilderRef, ParentMacroFile: LLVMMetadataRef, Line: ::libc::c_uint, RecordType: LLVMDWARFMacinfoRecordType, Name: *const ::libc::c_char, NameLen: usize, Value: *const ::libc::c_char, ValueLen: usize) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateMemberPointerType(Builder: LLVMDIBuilderRef, PointeeType: LLVMMetadataRef, ClassType: LLVMMetadataRef, SizeInBits: u64, AlignInBits: u32, Flags: LLVMDIFlags) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateMemberType(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, File: LLVMMetadataRef, LineNo: ::libc::c_uint, SizeInBits: u64, AlignInBits: u32, OffsetInBits: u64, Flags: LLVMDIFlags, Ty: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateModule(Builder: LLVMDIBuilderRef, ParentScope: LLVMMetadataRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, ConfigMacros: *const ::libc::c_char, ConfigMacrosLen: ::libc::size_t, IncludePath: *const ::libc::c_char, IncludePathLen: ::libc::size_t, APINotesFile: *const ::libc::c_char, APINotesFileLen: ::libc::size_t) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateNameSpace(Builder: LLVMDIBuilderRef, ParentScope: LLVMMetadataRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, ExportSymbols: LLVMBool) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateNullPtrType(Builder: LLVMDIBuilderRef) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateObjCIVar(Builder: LLVMDIBuilderRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, File: LLVMMetadataRef, LineNo: ::libc::c_uint, SizeInBits: u64, AlignInBits: u32, OffsetInBits: u64, Flags: LLVMDIFlags, Ty: LLVMMetadataRef, PropertyNode: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateObjCProperty(Builder: LLVMDIBuilderRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, File: LLVMMetadataRef, LineNo: ::libc::c_uint, GetterName: *const ::libc::c_char, GetterNameLen: ::libc::size_t, SetterName: *const ::libc::c_char, SetterNameLen: ::libc::size_t, PropertyAttributes: ::libc::c_uint, Ty: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateObjectPointerType(Builder: LLVMDIBuilderRef, Type: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateParameterVariable(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, ArgNo: ::libc::c_uint, File: LLVMMetadataRef, LineNo: ::libc::c_uint, Ty: LLVMMetadataRef, AlwaysPreserve: LLVMBool, Flags: LLVMDIFlags) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreatePointerType(Builder: LLVMDIBuilderRef, PointeeTy: LLVMMetadataRef, SizeInBits: u64, AlignInBits: u32, AddressSpace: ::libc::c_uint, Name: *const ::libc::c_char, NameLen: ::libc::size_t) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateQualifiedType(Builder: LLVMDIBuilderRef, Tag: ::libc::c_uint, Type: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateReferenceType(Builder: LLVMDIBuilderRef, Tag: ::libc::c_uint, Type: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateReplaceableCompositeType(Builder: LLVMDIBuilderRef, Tag: ::libc::c_uint, Name: *const ::libc::c_char, NameLen: ::libc::size_t, Scope: LLVMMetadataRef, File: LLVMMetadataRef, Line: ::libc::c_uint, RuntimeLang: ::libc::c_uint, SizeInBits: u64, AlignInBits: u32, Flags: LLVMDIFlags, UniqueIdentifier: *const ::libc::c_char, UniqueIdentifierLen: ::libc::size_t) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateStaticMemberType(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, File: LLVMMetadataRef, LineNumber: ::libc::c_uint, Type: LLVMMetadataRef, Flags: LLVMDIFlags, ConstantVal: LLVMValueRef, AlignInBits: u32) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateStructType(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, File: LLVMMetadataRef, LineNumber: ::libc::c_uint, SizeInBits: u64, AlignInBits: u32, Flags: LLVMDIFlags, DerivedFrom: LLVMMetadataRef, Elements: *mut LLVMMetadataRef, NumElements: ::libc::c_uint, RunTimeLang: ::libc::c_uint, VTableHolder: LLVMMetadataRef, UniqueId: *const ::libc::c_char, UniqueIdLen: ::libc::size_t) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateSubroutineType(Builder: LLVMDIBuilderRef, File: LLVMMetadataRef, ParameterTypes: *mut LLVMMetadataRef, NumParameterTypes: ::libc::c_uint, Flags: LLVMDIFlags) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateTempGlobalVariableFwdDecl(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, Linkage: *const ::libc::c_char, LnkLen: ::libc::size_t, File: LLVMMetadataRef, LineNo: ::libc::c_uint, Ty: LLVMMetadataRef, LocalToUnit: LLVMBool, Decl: LLVMMetadataRef, AlignInBits: u32) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateTempMacroFile(Builder: LLVMDIBuilderRef, ParentMacroFile: LLVMMetadataRef, Line: ::libc::c_uint, File: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateTypedef(Builder: LLVMDIBuilderRef, Type: LLVMMetadataRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, File: LLVMMetadataRef, LineNo: ::libc::c_uint, Scope: LLVMMetadataRef, AlignInBits: u32) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateUnionType(Builder: LLVMDIBuilderRef, Scope: LLVMMetadataRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t, File: LLVMMetadataRef, LineNumber: ::libc::c_uint, SizeInBits: u64, AlignInBits: u32, Flags: LLVMDIFlags, Elements: *mut LLVMMetadataRef, NumElements: ::libc::c_uint, RunTimeLang: ::libc::c_uint, UniqueId: *const ::libc::c_char, UniqueIdLen: ::libc::size_t) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateUnspecifiedType(Builder: LLVMDIBuilderRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderCreateVectorType(Builder: LLVMDIBuilderRef, Size: u64, AlignInBits: u32, Ty: LLVMMetadataRef, Subscripts: *mut LLVMMetadataRef, NumSubscripts: ::libc::c_uint) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderFinalize(Builder: LLVMDIBuilderRef);
    pub fn LLVMDIBuilderFinalizeSubprogram(Builder: LLVMDIBuilderRef, Subprogram: LLVMMetadataRef);
    pub fn LLVMDIBuilderGetOrCreateArray(Builder: LLVMDIBuilderRef, Data: *mut LLVMMetadataRef, NumElements: ::libc::size_t) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderGetOrCreateSubrange(Builder: LLVMDIBuilderRef, LowerBound: i64, Count: i64) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderGetOrCreateTypeArray(Builder: LLVMDIBuilderRef, Data: *mut LLVMMetadataRef, NumElements: ::libc::size_t) -> LLVMMetadataRef;
    pub fn LLVMDIBuilderInsertDbgValueAtEnd(Builder: LLVMDIBuilderRef, Val: LLVMValueRef, VarInfo: LLVMMetadataRef, Expr: LLVMMetadataRef, DebugLoc: LLVMMetadataRef, Block: LLVMBasicBlockRef) -> LLVMValueRef;
    pub fn LLVMDIBuilderInsertDbgValueBefore(Builder: LLVMDIBuilderRef, Val: LLVMValueRef, VarInfo: LLVMMetadataRef, Expr: LLVMMetadataRef, DebugLoc: LLVMMetadataRef, Instr: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMDIBuilderInsertDeclareAtEnd(Builder: LLVMDIBuilderRef, Storage: LLVMValueRef, VarInfo: LLVMMetadataRef, Expr: LLVMMetadataRef, DebugLoc: LLVMMetadataRef, Block: LLVMBasicBlockRef) -> LLVMValueRef;
    pub fn LLVMDIBuilderInsertDeclareBefore(Builder: LLVMDIBuilderRef, Storage: LLVMValueRef, VarInfo: LLVMMetadataRef, Expr: LLVMMetadataRef, DebugLoc: LLVMMetadataRef, Instr: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMDIFileGetDirectory(File: LLVMMetadataRef, Len: *mut ::libc::c_uint) -> *const ::libc::c_char;
    pub fn LLVMDIFileGetFilename(File: LLVMMetadataRef, Len: *mut ::libc::c_uint) -> *const ::libc::c_char;
    pub fn LLVMDIFileGetSource(File: LLVMMetadataRef, Len: *mut ::libc::c_uint) -> *const ::libc::c_char;
    pub fn LLVMDIGlobalVariableExpressionGetExpression(GVE: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDIGlobalVariableExpressionGetVariable(GVE: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDILocationGetColumn(Location: LLVMMetadataRef) -> ::libc::c_uint;
    pub fn LLVMDILocationGetInlinedAt(Location: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDILocationGetLine(Location: LLVMMetadataRef) -> ::libc::c_uint;
    pub fn LLVMDILocationGetScope(Location: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDIScopeGetFile(Scope: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDisposeDIBuilder(Builder: LLVMDIBuilderRef);
    pub fn LLVMDisposeTemporaryMDNode(TempNode: LLVMMetadataRef);
    pub fn LLVMDISubprogramGetLine(Subprogram: LLVMMetadataRef) -> ::libc::c_uint;
    pub fn LLVMDITypeGetAlignInBits(DType: LLVMMetadataRef) -> u32;
    pub fn LLVMDITypeGetFlags(DType: LLVMMetadataRef) -> LLVMDIFlags;
    pub fn LLVMDITypeGetLine(DType: LLVMMetadataRef) -> ::libc::c_uint;
    pub fn LLVMDITypeGetName(DType: LLVMMetadataRef, Length: *mut ::libc::size_t) -> *const ::libc::c_char;
    pub fn LLVMDITypeGetOffsetInBits(DType: LLVMMetadataRef) -> u64;
    pub fn LLVMDITypeGetSizeInBits(DType: LLVMMetadataRef) -> u64;
    pub fn LLVMDIVariableGetFile(Var: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMDIVariableGetLine(Var: LLVMMetadataRef) -> ::libc::c_uint;
    pub fn LLVMDIVariableGetScope(Var: LLVMMetadataRef) -> LLVMMetadataRef;
    pub fn LLVMGetMetadataKind(Metadata: LLVMMetadataRef) -> LLVMMetadataKind;
    pub fn LLVMGetModuleDebugMetadataVersion(Module: LLVMModuleRef) -> ::libc::c_uint;
    pub fn LLVMGetSubprogram(Func: LLVMValueRef) -> LLVMMetadataRef;
    pub fn LLVMInstructionGetDebugLoc(Inst: LLVMValueRef) -> LLVMMetadataRef;
    pub fn LLVMInstructionSetDebugLoc(Inst: LLVMValueRef, Loc: LLVMMetadataRef);
    pub fn LLVMMetadataReplaceAllUsesWith(TempTargetMetadata: LLVMMetadataRef, Replacement: LLVMMetadataRef);
    pub fn LLVMSetSubprogram(Func: LLVMValueRef, SP: LLVMMetadataRef);
    pub fn LLVMStripModuleDebugInfo(Module: LLVMModuleRef) -> LLVMBool;
    pub fn LLVMTemporaryMDNode(Ctx: LLVMContextRef, Data: *mut LLVMMetadataRef, NumElements: ::libc::size_t) -> LLVMMetadataRef;
}

// llvm-c/Disassembler.h

#[derive(Debug)]
pub enum LLVMOpaqueDisasmContext {}

pub type LLVMDisasmContextRef = *mut LLVMOpaqueDisasmContext;
pub type LLVMOpInfoCallback = Option<extern "C" fn(DisInfo: *mut ::libc::c_void, PC: u64, Offset: u64, OpSize: u64, InstSize: u64, TagType: ::libc::c_int, TagBuf: *mut ::libc::c_void) -> ::libc::c_int>;

#[repr(C)]
#[derive(Debug)]
pub struct LLVMOpInfoSymbol1 {
    pub Present: u64,
    pub Name: *const ::libc::c_char,
    pub Value: u64,
}

#[repr(C)]
#[derive(Debug)]
pub struct Struct_LLVMOpInfo1 {
    pub AddSymbol: LLVMOpInfoSymbol1,
    pub SubtractSymbol: LLVMOpInfoSymbol1,
    pub Value: u64,
    pub VariantKind: u64,
}

pub const LLVMDisassembler_VariantKind_None: u64 = 0;
pub const LLVMDisassembler_VariantKind_ARM_HI16: u64 = 1;
pub const LLVMDisassembler_VariantKind_ARM_LO16: u64 = 2;
pub const LLVMDisassembler_VariantKind_ARM64_PAGE: u64 = 1;
pub const LLVMDisassembler_VariantKind_ARM64_PAGEOFF: u64 = 2;
pub const LLVMDisassembler_VariantKind_ARM64_GOTPAGE: u64 = 3;
pub const LLVMDisassembler_VariantKind_ARM64_GOTPAGEOFF: u64 = 4;
pub const LLVMDisassembler_VariantKind_ARM64_TLVP: u64 = 5;
pub const LLVMDisassembler_VariantKind_ARM64_TLVOFF: u64 = 6;

pub const LLVMDisassembler_ReferenceType_InOut_None: u64 = 0;

pub const LLVMDisassembler_ReferenceType_In_Branch: u64 = 1;
pub const LLVMDisassembler_ReferenceType_In_PCrel_Load: u64 = 2;

pub const LLVMDisassembler_ReferenceType_In_ARM64_ADRP: u64 = 0x100000001;
pub const LLVMDisassembler_ReferenceType_In_ARM64_ADDXri: u64 = 0x100000002;
pub const LLVMDisassembler_ReferenceType_In_ARM64_LDRXui: u64 = 0x100000003;
pub const LLVMDisassembler_ReferenceType_In_ARM64_LDRXl: u64 = 0x100000004;
pub const LLVMDisassembler_ReferenceType_In_ARM64_ADR: u64 = 0x100000005;

pub const LLVMDisassembler_ReferenceType_Out_SymbolStub: u64 = 1;
pub const LLVMDisassembler_ReferenceType_Out_LitPool_SymAddr: u64 = 2;
pub const LLVMDisassembler_ReferenceType_Out_LitPool_CstrAddr: u64 = 3;

pub const LLVMDisassembler_ReferenceType_Out_Objc_CFString_Ref: u64 = 4;
pub const LLVMDisassembler_ReferenceType_Out_Objc_Message: u64 = 5;
pub const LLVMDisassembler_ReferenceType_Out_Objc_Message_Ref: u64 = 6;
pub const LLVMDisassembler_ReferenceType_Out_Objc_Selector_Ref: u64 = 7;
pub const LLVMDisassembler_ReferenceType_Out_Objc_Class_Ref: u64 = 8;
pub const LLVMDisassembler_ReferenceType_DeMangled_Name: u64 = 9;

pub const LLVMDisassembler_Option_UseMarkup: u64 = 1;
pub const LLVMDisassembler_Option_PrintImmHex: u64 = 2;
pub const LLVMDisassembler_Option_AsmPrinterVariant: u64 = 4;
pub const LLVMDisassembler_Option_SetInstrComments: u64 = 8;
pub const LLVMDisassembler_Option_PrintLatency: u64 = 16;

pub type LLVMSymbolLookupCallback = Option<extern "C" fn(DisInfo: *mut ::libc::c_void, ReferenceValue: u64, ReferenceType: *mut u64, ReferencePC: u64, ReferenceName: *mut *const ::libc::c_char) -> *const ::libc::c_char>;

extern "C" {
    pub fn LLVMCreateDisasm(TripleName: *const ::libc::c_char, DisInfo: *mut ::libc::c_void, TagType: ::libc::c_int, GetOpInfo: LLVMOpInfoCallback, SymbolLookUp: LLVMSymbolLookupCallback) -> LLVMDisasmContextRef;
    pub fn LLVMCreateDisasmCPU(Triple: *const ::libc::c_char, CPU: *const ::libc::c_char, DisInfo: *mut ::libc::c_void, TagType: ::libc::c_int, GetOpInfo: LLVMOpInfoCallback, SymbolLookUp: LLVMSymbolLookupCallback) -> LLVMDisasmContextRef;
    pub fn LLVMCreateDisasmCPUFeatures(Triple: *const ::libc::c_char, CPU: *const ::libc::c_char, Features: *const ::libc::c_char, DisInfo: *mut ::libc::c_void, TagType: ::libc::c_int, GetOpInfo: LLVMOpInfoCallback, SymbolLookUp: LLVMSymbolLookupCallback) -> LLVMDisasmContextRef;
    pub fn LLVMDisasmDispose(DC: LLVMDisasmContextRef);
    pub fn LLVMDisasmInstruction(DC: LLVMDisasmContextRef, Bytes: *mut u8, BytesSize: u64, PC: u64, OutString: *mut ::libc::c_char, OutStringSize: ::libc::size_t) -> ::libc::size_t;
    pub fn LLVMSetDisasmOptions(DC: LLVMDisasmContextRef, Options: u64) -> ::libc::c_int;
}

// llvm-c/ErrorHandling.h
pub type LLVMFatalErrorHandler = Option<extern "C" fn(Reason: *const ::libc::c_char)>;

extern "C" {
    pub fn LLVMEnablePrettyStackTrace();
    pub fn LLVMInstallFatalErrorHandler(Handler: LLVMFatalErrorHandler);
    pub fn LLVMResetFatalErrorHandler();
}

// llvm-c/Error.h
pub const LLVMErrorSuccess: ::libc::c_int = 0;

#[derive(Debug)]
pub enum LLVMOpaqueError {}

pub type LLVMErrorRef = *mut LLVMOpaqueError;
pub type LLVMErrorTypeId = *const ::libc::c_void;

extern "C" {
    pub fn LLVMGetErrorTypeId(Err: LLVMErrorRef) -> LLVMErrorTypeId;
    pub fn LLVMConsumeError(Err: LLVMErrorRef);
    pub fn LLVMGetErrorMessage(Err: LLVMErrorRef) -> *mut ::libc::c_char;
    pub fn LLVMDisposeErrorMessage(ErrMsg: *mut ::libc::c_char);
    pub fn LLVMGetStringErrorTypeId() -> LLVMErrorTypeId;
    pub fn LLVMCreateStringError(ErrMst: *const ::libc::c_char) -> LLVMErrorRef;
}

// llvm-c/ExecutionEngine.h
#[derive(Debug)]
pub enum LLVMOpaqueGenericValue {}

#[derive(Debug)]
pub enum LLVMOpaqueExecutionEngine {}

#[derive(Debug)]
pub enum LLVMOpaqueMCJITMemoryManager {}

pub type LLVMGenericValueRef = *mut LLVMOpaqueGenericValue;
pub type LLVMExecutionEngineRef = *mut LLVMOpaqueExecutionEngine;
pub type LLVMMCJITMemoryManagerRef = *mut LLVMOpaqueMCJITMemoryManager;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct LLVMMCJITCompilerOptions {
    pub OptLevel: ::libc::c_uint,
    pub CodeModel: LLVMCodeModel,
    pub NoFramePointerElim: LLVMBool,
    pub EnableFastISel: LLVMBool,
    pub MCJMM: LLVMMCJITMemoryManagerRef,
}

pub type LLVMMemoryManagerAllocateCodeSectionCallback = extern "C" fn(Opaque: *mut ::libc::c_void, Size: ::libc::uintptr_t, Alignment: ::libc::c_uint, SectionID: ::libc::c_uint, SectionName: *const ::libc::c_char) -> *mut u8;
pub type LLVMMemoryManagerAllocateDataSectionCallback = extern "C" fn(Opaque: *mut ::libc::c_void, Size: ::libc::uintptr_t, Alignment: ::libc::c_uint, SectionID: ::libc::c_uint, SectionName: *const ::libc::c_char, IsReadOnly: LLVMBool) -> *mut u8;
pub type LLVMMemoryManagerFinalizeMemoryCallback = extern "C" fn(Opaque: *mut ::libc::c_void, ErrMsg: *mut *mut ::libc::c_char) -> LLVMBool;
pub type LLVMMemoryManagerDestroyCallback = Option<extern "C" fn(Opaque: *mut ::libc::c_void)>;

extern "C" {
    pub fn LLVMAddGlobalMapping(EE: LLVMExecutionEngineRef, Global: LLVMValueRef, Addr: *mut ::libc::c_void);
    pub fn LLVMAddModule(EE: LLVMExecutionEngineRef, M: LLVMModuleRef);
    pub fn LLVMCreateExecutionEngineForModule(OutEE: *mut LLVMExecutionEngineRef, M: LLVMModuleRef, OutError: *mut *mut ::libc::c_char) -> LLVMBool;
    pub fn LLVMCreateGDBRegistrationListener() -> LLVMJITEventListenerRef;
    pub fn LLVMCreateGenericValueOfFloat(Ty: LLVMTypeRef, N: ::libc::c_double) -> LLVMGenericValueRef;
    pub fn LLVMCreateGenericValueOfInt(Ty: LLVMTypeRef, N: ::libc::c_ulonglong, IsSigned: LLVMBool) -> LLVMGenericValueRef;
    pub fn LLVMCreateGenericValueOfPointer(P: *mut ::libc::c_void) -> LLVMGenericValueRef;
    pub fn LLVMCreateIntelJITEventListener() -> LLVMJITEventListenerRef;
    pub fn LLVMCreateInterpreterForModule(OutInterp: *mut LLVMExecutionEngineRef, M: LLVMModuleRef, OutError: *mut *mut ::libc::c_char) -> LLVMBool;
    pub fn LLVMCreateJITCompilerForModule(OutJIT: *mut LLVMExecutionEngineRef, M: LLVMModuleRef, OptLevel: ::libc::c_uint, OutError: *mut *mut ::libc::c_char) -> LLVMBool;
    pub fn LLVMCreateMCJITCompilerForModule(OutJIT: *mut LLVMExecutionEngineRef, M: LLVMModuleRef, Options: *mut LLVMMCJITCompilerOptions, SizeOfOptions: ::libc::size_t, OutError: *mut *mut ::libc::c_char) -> LLVMBool;
    pub fn LLVMCreateOProfileJITEventListener() -> LLVMJITEventListenerRef;
    pub fn LLVMCreatePerfJITEventListener() -> LLVMJITEventListenerRef;
    pub fn LLVMCreateSimpleMCJITMemoryManager(Opaque: *mut ::libc::c_void, AllocateCodeSection: LLVMMemoryManagerAllocateCodeSectionCallback, AllocateDataSection: LLVMMemoryManagerAllocateDataSectionCallback, FinalizeMemory: LLVMMemoryManagerFinalizeMemoryCallback, Destroy: LLVMMemoryManagerDestroyCallback) -> LLVMMCJITMemoryManagerRef;
    pub fn LLVMDisposeExecutionEngine(EE: LLVMExecutionEngineRef);
    pub fn LLVMDisposeGenericValue(GenVal: LLVMGenericValueRef);
    pub fn LLVMDisposeMCJITMemoryManager(MM: LLVMMCJITMemoryManagerRef);
    pub fn LLVMExecutionEngineGetErrMsg(EE: LLVMExecutionEngineRef, OutError: *mut *mut ::libc::c_char) -> LLVMBool;
    pub fn LLVMFindFunction(EE: LLVMExecutionEngineRef, Name: *const ::libc::c_char, OutFn: *mut LLVMValueRef) -> LLVMBool;
    pub fn LLVMFreeMachineCodeForFunction(EE: LLVMExecutionEngineRef, F: LLVMValueRef);
    pub fn LLVMGenericValueIntWidth(GenValRef: LLVMGenericValueRef) -> ::libc::c_uint;
    pub fn LLVMGenericValueToFloat(TyRef: LLVMTypeRef, GenVal: LLVMGenericValueRef) -> ::libc::c_double;
    pub fn LLVMGenericValueToInt(GenVal: LLVMGenericValueRef, IsSigned: LLVMBool) -> ::libc::c_ulonglong;
    pub fn LLVMGenericValueToPointer(GenVal: LLVMGenericValueRef) -> *mut ::libc::c_void;
    pub fn LLVMGetExecutionEngineTargetData(EE: LLVMExecutionEngineRef) -> LLVMTargetDataRef;
    pub fn LLVMGetExecutionEngineTargetMachine(EE: LLVMExecutionEngineRef) -> LLVMTargetMachineRef;
    pub fn LLVMGetFunctionAddress(EE: LLVMExecutionEngineRef, Name: *const ::libc::c_char) -> u64;
    pub fn LLVMGetGlobalValueAddress(EE: LLVMExecutionEngineRef, Name: *const ::libc::c_char) -> u64;
    pub fn LLVMGetPointerToGlobal(EE: LLVMExecutionEngineRef, Global: LLVMValueRef) -> *mut ::libc::c_void;
    pub fn LLVMInitializeMCJITCompilerOptions(Options: *mut LLVMMCJITCompilerOptions, SizeOfOptions: ::libc::size_t);
    pub fn LLVMLinkInInterpreter();
    pub fn LLVMLinkInMCJIT();
    pub fn LLVMRecompileAndRelinkFunction(EE: LLVMExecutionEngineRef, Fn: LLVMValueRef) -> *mut ::libc::c_void;
    pub fn LLVMRemoveModule(EE: LLVMExecutionEngineRef, M: LLVMModuleRef, OutMod: *mut LLVMModuleRef, OutError: *mut *mut ::libc::c_char) -> LLVMBool;
    pub fn LLVMRunFunction(EE: LLVMExecutionEngineRef, F: LLVMValueRef, NumArgs: ::libc::c_uint, Args: *mut LLVMGenericValueRef) -> LLVMGenericValueRef;
    pub fn LLVMRunFunctionAsMain(EE: LLVMExecutionEngineRef, F: LLVMValueRef, ArgC: ::libc::c_uint, ArgV: *const *const ::libc::c_char, EnvP: *const *const ::libc::c_char) -> ::libc::c_int;
    pub fn LLVMRunStaticConstructors(EE: LLVMExecutionEngineRef);
    pub fn LLVMRunStaticDestructors(EE: LLVMExecutionEngineRef);
}

// init
extern "C" {
    pub fn LLVMInitializeAnalysis(R: LLVMPassRegistryRef);
    pub fn LLVMInitializeCodeGen(R: LLVMPassRegistryRef);
    pub fn LLVMInitializeCore(R: LLVMPassRegistryRef);
    pub fn LLVMInitializeInstCombine(R: LLVMPassRegistryRef);
    pub fn LLVMInitializeIPA(R: LLVMPassRegistryRef);
    pub fn LLVMInitializeIPO(R: LLVMPassRegistryRef);
    pub fn LLVMInitializeScalarOpts(R: LLVMPassRegistryRef);
    pub fn LLVMInitializeTarget(R: LLVMPassRegistryRef);
    pub fn LLVMInitializeTransformUtils(R: LLVMPassRegistryRef);
    pub fn LLVMInitializeVectorization(R: LLVMPassRegistryRef);
}

// llvm-c/IRReader.h
extern "C" {
    pub fn LLVMParseIRInContext(ContextRef: LLVMContextRef, MemBuf: LLVMMemoryBufferRef, OutM: *mut LLVMModuleRef, OutMessage: *mut *mut ::libc::c_char) -> LLVMBool;
}

// llvm-c/Linker.h
#[repr(C)]
#[derive(Debug)]
pub enum LLVMLinkerMode {
    LLVMLinkerDestroySource = 0,
}

extern "C" {
    pub fn LLVMLinkModules2(Dest: LLVMModuleRef, Src: LLVMModuleRef) -> LLVMBool;
}

// llvm-c/lto.h

pub type lto_bool_t = u8;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum lto_symbol_attributes {
    LTO_SYMBOL_ALIGNMENT_MASK = 31,
    LTO_SYMBOL_PERMISSIONS_MASK = 224,
    LTO_SYMBOL_PERMISSIONS_CODE = 160,
    LTO_SYMBOL_PERMISSIONS_DATA = 192,
    LTO_SYMBOL_PERMISSIONS_RODATA = 128,
    LTO_SYMBOL_DEFINITION_MASK = 1792,
    LTO_SYMBOL_DEFINITION_REGULAR = 256,
    LTO_SYMBOL_DEFINITION_TENTATIVE = 512,
    LTO_SYMBOL_DEFINITION_WEAK = 768,
    LTO_SYMBOL_DEFINITION_UNDEFINED = 1024,
    LTO_SYMBOL_DEFINITION_WEAKUNDEF = 1280,
    LTO_SYMBOL_SCOPE_MASK = 14336,
    LTO_SYMBOL_SCOPE_INTERNAL = 2048,
    LTO_SYMBOL_SCOPE_HIDDEN = 0x1000,
    LTO_SYMBOL_SCOPE_PROTECTED = 0x2000,
    LTO_SYMBOL_SCOPE_DEFAULT = 0x1800,
    LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN = 0x2800,
    LTO_SYMBOL_COMDAT = 0x4000,
    LTO_SYMBOL_ALIAS = 0x8000,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum lto_debug_model {
    LTO_DEBUG_MODEL_NONE = 0,
    LTO_DEBUG_MODEL_DWARF = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum lto_codegen_model {
    LTO_CODEGEN_PIC_MODEL_STATIC = 0,
    LTO_CODEGEN_PIC_MODEL_DYNAMIC = 1,
    LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC = 2,
    LTO_CODEGEN_PIC_MODEL_DEFAULT = 3,
}

#[derive(Debug)]
pub enum LLVMOpaqueLTOModule {}

pub type lto_module_t = *mut LLVMOpaqueLTOModule;

#[derive(Debug)]
pub enum LLVMOpaqueLTOCodeGenerator {}

pub type lto_code_gen_t = *mut LLVMOpaqueLTOCodeGenerator;

#[derive(Debug)]
pub enum LLVMOpaqueThinLTOCodeGenerator {}

pub type thinlto_code_gen_t = *mut LLVMOpaqueThinLTOCodeGenerator;

#[derive(Debug)]
pub enum LLVMOpaqueLTOInput {}

pub type lto_input_t = *mut LLVMOpaqueLTOInput;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum lto_codegen_diagnostic_severity_t {
    LTO_DS_ERROR = 0,
    LTO_DS_WARNING = 1,
    LTO_DS_REMARK = 3,
    LTO_DS_NOTE = 2,
}

pub type lto_diagnostic_handler_t = Option<extern "C" fn(severity: lto_codegen_diagnostic_severity_t, diag: *const ::libc::c_char, ctxt: *mut ::libc::c_void)>;

extern "C" {
    pub fn lto_api_version() -> ::libc::c_uint;
    pub fn lto_codegen_add_module(cg: lto_code_gen_t, _mod: lto_module_t) -> lto_bool_t;
    pub fn lto_codegen_add_must_preserve_symbol(cg: lto_code_gen_t, symbol: *const ::libc::c_char);
    pub fn lto_codegen_compile_optimized(cg: lto_code_gen_t, length: *mut ::libc::size_t) -> *mut ::libc::c_void;
    pub fn lto_codegen_compile_to_file(cg: lto_code_gen_t, name: *mut *const ::libc::c_char) -> lto_bool_t;
    pub fn lto_codegen_compile(cg: lto_code_gen_t, length: *mut ::libc::size_t) -> *const ::libc::c_void;
    pub fn lto_codegen_create_in_local_context() -> lto_code_gen_t;
    pub fn lto_codegen_create() -> lto_code_gen_t;
    pub fn lto_codegen_debug_options_array(cg: lto_code_gen_t, arg2: *const *const ::libc::c_char, number: ::libc::c_int);
    pub fn lto_codegen_debug_options(cg: lto_code_gen_t, arg1: *const ::libc::c_char);
    pub fn lto_codegen_dispose(arg1: lto_code_gen_t);
    pub fn lto_codegen_optimize(cg: lto_code_gen_t) -> lto_bool_t;
    pub fn lto_codegen_set_assembler_args(cg: lto_code_gen_t, args: *mut *const ::libc::c_char, nargs: ::libc::c_int);
    pub fn lto_codegen_set_assembler_path(cg: lto_code_gen_t, path: *const ::libc::c_char);
    pub fn lto_codegen_set_cpu(cg: lto_code_gen_t, cpu: *const ::libc::c_char);
    pub fn lto_codegen_set_debug_model(cg: lto_code_gen_t, arg1: lto_debug_model) -> lto_bool_t;
    pub fn lto_codegen_set_diagnostic_handler(arg1: lto_code_gen_t, arg2: lto_diagnostic_handler_t, arg3: *mut ::libc::c_void);
    pub fn lto_codegen_set_module(cg: lto_code_gen_t, _mod: lto_module_t);
    pub fn lto_codegen_set_pic_model(cg: lto_code_gen_t, arg1: lto_codegen_model) -> lto_bool_t;
    pub fn lto_codegen_set_should_embed_uselists(cg: lto_code_gen_t, ShouldEmbedUselists: lto_bool_t);
    pub fn lto_codegen_set_should_internalize(cg: lto_code_gen_t, ShouldInternalize: lto_bool_t);
    pub fn lto_codegen_write_merged_modules(cg: lto_code_gen_t, path: *const ::libc::c_char) -> lto_bool_t;
    pub fn lto_get_error_message() -> *const ::libc::c_char;
    pub fn lto_get_version() -> *const ::libc::c_char;
    pub fn lto_initialize_disassembler();
    pub fn lto_module_create_from_fd_at_offset(fd: ::libc::c_int, path: *const ::libc::c_char, file_size: ::libc::size_t, map_size: ::libc::size_t, offset: ::libc::off_t) -> lto_module_t;
    pub fn lto_module_create_from_fd(fd: ::libc::c_int, path: *const ::libc::c_char, file_size: ::libc::size_t) -> lto_module_t;
    pub fn lto_module_create_from_memory_with_path(mem: *const ::libc::c_void, length: ::libc::size_t, path: *const ::libc::c_char) -> lto_module_t;
    pub fn lto_module_create_from_memory(mem: *const ::libc::c_void, length: ::libc::size_t) -> lto_module_t;
    pub fn lto_module_create_in_codegen_context(mem: *const ::libc::c_void, length: ::libc::size_t, path: *const ::libc::c_char, cg: lto_code_gen_t) -> lto_module_t;
    pub fn lto_module_create_in_local_context(mem: *const ::libc::c_void, length: ::libc::size_t, path: *const ::libc::c_char) -> lto_module_t;
    pub fn lto_module_create(path: *const ::libc::c_char) -> lto_module_t;
    pub fn lto_module_dispose(_mod: lto_module_t);
    pub fn lto_module_get_linkeropts(_mod: lto_module_t) -> *const ::libc::c_char;
    pub fn lto_module_get_macho_cputype(_mod: lto_module_t, out_cputype: *mut ::libc::c_uint, out_cpusubtype: *mut ::libc::c_uint) -> lto_bool_t;
    pub fn lto_module_get_num_symbols(_mod: lto_module_t) -> ::libc::c_uint;
    pub fn lto_module_get_symbol_attribute(_mod: lto_module_t, index: ::libc::c_uint) -> lto_symbol_attributes;
    pub fn lto_module_get_symbol_name(_mod: lto_module_t, index: ::libc::c_uint) -> *const ::libc::c_char;
    pub fn lto_module_get_target_triple(_mod: lto_module_t) -> *const ::libc::c_char;
    pub fn lto_module_has_ctor_dtor(mod_: lto_module_t) -> lto_bool_t;
    pub fn lto_module_has_objc_category(mem: *const ::libc::c_void, length: ::libc::size_t) -> lto_bool_t;
    pub fn lto_module_is_object_file_for_target(path: *const ::libc::c_char, target_triple_prefix: *const ::libc::c_char) -> lto_bool_t;
    pub fn lto_module_is_object_file_in_memory_for_target(mem: *const ::libc::c_void, length: ::libc::size_t, target_triple_prefix: *const ::libc::c_char) -> lto_bool_t;
    pub fn lto_module_is_object_file_in_memory(mem: *const ::libc::c_void, length: ::libc::size_t) -> lto_bool_t;
    pub fn lto_module_is_object_file(path: *const ::libc::c_char) -> lto_bool_t;
    pub fn lto_module_set_target_triple(_mod: lto_module_t, triple: *const ::libc::c_char);
    pub fn lto_set_debug_options(options: *mut *const ::libc::c_char, number: ::libc::c_int);
}

#[repr(C)]
#[derive(Debug)]
#[allow(non_snake_case)]
pub struct LTOObjectBuffer {
    Buffer: *const ::libc::c_char,
    Size: ::libc::size_t,
}

extern "C" {
    pub fn lto_input_create(buffer: *const ::libc::c_void, buffer_size: ::libc::size_t, path: *const ::libc::c_char) -> lto_input_t;
    pub fn lto_input_dispose(input: lto_input_t);
    pub fn lto_input_get_dependent_library(input: lto_input_t, index: ::libc::size_t, size: *mut ::libc::size_t) -> *const ::libc::c_char;
    pub fn lto_input_get_num_dependent_libraries(input: lto_input_t) -> ::libc::c_uint;
    pub fn lto_module_is_thinlto(module: lto_module_t) -> lto_bool_t;
    pub fn lto_runtime_lib_symbols_list(size: *mut usize) -> *const *const ::libc::c_char;
    pub fn thinlto_codegen_add_cross_referenced_symbol(cg: thinlto_code_gen_t, name: *const ::libc::c_char, length: ::libc::c_int);
    pub fn thinlto_codegen_add_module(cg: thinlto_code_gen_t, identifier: *const ::libc::c_char, data: *const ::libc::c_char, length: ::libc::c_int);
    pub fn thinlto_codegen_add_must_preserve_symbol(cg: thinlto_code_gen_t, name: *const ::libc::c_char, length: ::libc::c_int);
    pub fn thinlto_codegen_disable_codegen(cg: thinlto_code_gen_t, disable: lto_bool_t);
    pub fn thinlto_codegen_dispose(cg: thinlto_code_gen_t);
    pub fn thinlto_codegen_process(cg: thinlto_code_gen_t);
    pub fn thinlto_codegen_set_cache_dir(cg: thinlto_code_gen_t, cache_dir: *const ::libc::c_char);
    pub fn thinlto_codegen_set_cache_entry_expiration(cg: thinlto_code_gen_t, expiration: ::libc::c_uint);
    pub fn thinlto_codegen_set_cache_pruning_interval(cg: thinlto_code_gen_t, interval: ::libc::c_int);
    pub fn thinlto_codegen_set_cache_size_bytes(cg: thinlto_code_gen_t, max_size_bytes: ::libc::c_uint);
    pub fn thinlto_codegen_set_cache_size_files(cg: thinlto_code_gen_t, max_size_files: ::libc::c_uint);
    pub fn thinlto_codegen_set_cache_size_megabytes(cg: thinlto_code_gen_t, max_size_megabytes: ::libc::c_uint);
    pub fn thinlto_codegen_set_codegen_only(cg: thinlto_code_gen_t, codegen_only: lto_bool_t);
    pub fn thinlto_codegen_set_cpu(cg: thinlto_code_gen_t, cpu: *const ::libc::c_char);
    pub fn thinlto_codegen_set_final_cache_size_relative_to_available_space(cg: thinlto_code_gen_t, percentage: ::libc::c_uint);
    pub fn thinlto_codegen_set_pic_model(cg: thinlto_code_gen_t, model: lto_codegen_model) -> lto_bool_t;
    pub fn thinlto_codegen_set_savetemps_dir(cg: thinlto_code_gen_t, save_temps_dir: *const ::libc::c_char);
    pub fn thinlto_create_codegen() -> thinlto_code_gen_t;
    pub fn thinlto_debug_options(options: *const *const ::libc::c_char, number: ::libc::c_int);
    pub fn thinlto_module_get_num_object_files(cg: thinlto_code_gen_t) -> ::libc::c_uint;
    pub fn thinlto_module_get_num_objects(cg: thinlto_code_gen_t) -> ::libc::c_int;
    pub fn thinlto_module_get_object_file(cg: thinlto_code_gen_t, index: ::libc::c_uint) -> *const ::libc::c_char;
    pub fn thinlto_module_get_object(cg: thinlto_code_gen_t, index: ::libc::c_uint) -> LTOObjectBuffer;
    pub fn thinlto_set_generated_objects_dir(cg: thinlto_code_gen_t, save_temps_dir: *const ::libc::c_char);
}

// llvm-c/Object.h
#[derive(Debug)]
pub enum LLVMOpaqueSectionIterator {}

pub type LLVMSectionIteratorRef = *mut LLVMOpaqueSectionIterator;

#[derive(Debug)]
pub enum LLVMOpaqueSymbolIterator {}

pub type LLVMSymbolIteratorRef = *mut LLVMOpaqueSymbolIterator;

#[derive(Debug)]
pub enum LLVMOpaqueRelocationIterator {}

pub type LLVMRelocationIteratorRef = *mut LLVMOpaqueRelocationIterator;

#[derive(Debug)]
pub enum LLVMOpaqueBinary {}

pub type LLVMBinaryRef = *mut LLVMOpaqueBinary;

#[repr(C)]
#[derive(Debug)]
pub enum LLVMBinaryType {
    LLVMBinaryTypeArchive,
    LLVMBinaryTypeMachOUniversalBinary,
    LLVMBinaryTypeCOFFImportFile,
    LLVMBinaryTypeIR,
    LLVMBinaryTypeWinRes,
    LLVMBinaryTypeCOFF,
    LLVMBinaryTypeELF32L,
    LLVMBinaryTypeELF32B,
    LLVMBinaryTypeELF64L,
    LLVMBinaryTypeELF64B,
    LLVMBinaryTypeMachO32L,
    LLVMBinaryTypeMachO32B,
    LLVMBinaryTypeMachO64L,
    LLVMBinaryTypeMachO64B,
    LLVMBinaryTypeWasm,
    LLVMBinaryTypeOffload,
}

extern "C" {
    pub fn LLVMBinaryCopyMemoryBuffer(BR: LLVMBinaryRef) -> LLVMMemoryBufferRef;
    pub fn LLVMBinaryGetType(BR: LLVMBinaryRef) -> LLVMBinaryType;
    pub fn LLVMCreateBinary(MemBuf: LLVMMemoryBufferRef, Context: LLVMContextRef, ErrorMessage: *mut *mut ::libc::c_char) -> LLVMBinaryRef;
    pub fn LLVMDisposeBinary(BR: LLVMBinaryRef);
    pub fn LLVMDisposeRelocationIterator(RI: LLVMRelocationIteratorRef);
    pub fn LLVMDisposeSectionIterator(SI: LLVMSectionIteratorRef);
    pub fn LLVMDisposeSymbolIterator(SI: LLVMSymbolIteratorRef);
    pub fn LLVMGetRelocationOffset(RI: LLVMRelocationIteratorRef) -> u64;
    pub fn LLVMGetRelocations(Section: LLVMSectionIteratorRef) -> LLVMRelocationIteratorRef;
    pub fn LLVMGetRelocationSymbol(RI: LLVMRelocationIteratorRef) -> LLVMSymbolIteratorRef;
    pub fn LLVMGetRelocationType(RI: LLVMRelocationIteratorRef) -> u64;
    pub fn LLVMGetRelocationTypeName(RI: LLVMRelocationIteratorRef) -> *const ::libc::c_char;
    pub fn LLVMGetRelocationValueString(RI: LLVMRelocationIteratorRef) -> *const ::libc::c_char;
    pub fn LLVMGetSectionAddress(SI: LLVMSectionIteratorRef) -> u64;
    pub fn LLVMGetSectionContainsSymbol(SI: LLVMSectionIteratorRef, Sym: LLVMSymbolIteratorRef) -> LLVMBool;
    pub fn LLVMGetSectionContents(SI: LLVMSectionIteratorRef) -> *const ::libc::c_char;
    pub fn LLVMGetSectionName(SI: LLVMSectionIteratorRef) -> *const ::libc::c_char;
    pub fn LLVMGetSectionSize(SI: LLVMSectionIteratorRef) -> u64;
    pub fn LLVMGetSymbolAddress(SI: LLVMSymbolIteratorRef) -> u64;
    pub fn LLVMGetSymbolName(SI: LLVMSymbolIteratorRef) -> *const ::libc::c_char;
    pub fn LLVMGetSymbolSize(SI: LLVMSymbolIteratorRef) -> u64;
    pub fn LLVMIsRelocationIteratorAtEnd(Section: LLVMSectionIteratorRef, RI: LLVMRelocationIteratorRef) -> LLVMBool;
    pub fn LLVMMachOUniversalBinaryCopyObjectForArch(BR: LLVMBinaryRef, Arch: *const ::libc::c_char, ArchLen: ::libc::size_t, ErrorMessage: *mut *mut ::libc::c_char) -> LLVMBinaryRef;
    pub fn LLVMMoveToContainingSection(Sect: LLVMSectionIteratorRef, Sym: LLVMSymbolIteratorRef);
    pub fn LLVMMoveToNextRelocation(RI: LLVMRelocationIteratorRef);
    pub fn LLVMMoveToNextSection(SI: LLVMSectionIteratorRef);
    pub fn LLVMMoveToNextSymbol(SI: LLVMSymbolIteratorRef);
    pub fn LLVMObjectFileCopySectionIterator(BR: LLVMBinaryRef) -> LLVMSectionIteratorRef;
    pub fn LLVMObjectFileCopySymbolIterator(BR: LLVMBinaryRef) -> LLVMSymbolIteratorRef;
    pub fn LLVMObjectFileIsSectionIteratorAtEnd(BR: LLVMBinaryRef, SI: LLVMSectionIteratorRef) -> LLVMBool;
    pub fn LLVMObjectFileIsSymbolIteratorAtEnd(BR: LLVMBinaryRef, SI: LLVMSymbolIteratorRef) -> LLVMBool;
}

// llvm-c/Remarks.h

#[repr(C)]
pub enum LLVMRemarkType {
    LLVMRemarkTypeUnknown,
    LLVMRemarkTypePassed,
    LLVMRemarkTypeMissed,
    LLVMRemarkTypeAnalysis,
    LLVMRemarkTypeAnalysisFPCommute,
    LLVMRemarkTypeAnalysisAliasing,
    LLVMRemarkTypeFailure,
}

pub enum LLVMRemarkOpaqueString {}

pub type LLVMRemarkStringRef = *mut LLVMRemarkOpaqueString;

extern "C" {
    pub fn LLVMRemarkStringGetData(String: LLVMRemarkStringRef) -> *const ::libc::c_char;
    pub fn LLVMRemarkStringGetLen(String: LLVMRemarkStringRef) -> u32;
}

pub enum LLVMRemarkOpaqueDebugLoc {}

pub type LLVMRemarkDebugLocRef = *mut LLVMRemarkOpaqueDebugLoc;

extern "C" {
    pub fn LLVMRemarkDebugLocGetSourceColumn(DL: LLVMRemarkDebugLocRef) -> u32;
    pub fn LLVMRemarkDebugLocGetSourceFilePath(DL: LLVMRemarkDebugLocRef) -> LLVMRemarkStringRef;
    pub fn LLVMRemarkDebugLocGetSourceLine(DL: LLVMRemarkDebugLocRef) -> u32;
}

pub enum LLVMRemarkOpaqueArg {}

pub type LLVMRemarkArgRef = *mut LLVMRemarkOpaqueArg;

extern "C" {
    pub fn LLVMRemarkArgGetDebugLoc(Arg: LLVMRemarkArgRef) -> LLVMRemarkDebugLocRef;
    pub fn LLVMRemarkArgGetKey(Arg: LLVMRemarkArgRef) -> LLVMRemarkStringRef;
    pub fn LLVMRemarkArgGetValue(Arg: LLVMRemarkArgRef) -> LLVMRemarkStringRef;
}

pub enum LLVMRemarkOpaqueEntry {}
pub type LLVMRemarkEntryRef = *mut LLVMRemarkOpaqueEntry;

extern "C" {
    pub fn LLVMRemarkEntryDispose(Remark: LLVMRemarkEntryRef);
    pub fn LLVMRemarkEntryGetDebugLoc(Remark: LLVMRemarkEntryRef) -> LLVMRemarkDebugLocRef;
    pub fn LLVMRemarkEntryGetFirstArg(Remark: LLVMRemarkEntryRef) -> LLVMRemarkArgRef;
    pub fn LLVMRemarkEntryGetFunctionName(Remark: LLVMRemarkEntryRef) -> LLVMRemarkStringRef;
    pub fn LLVMRemarkEntryGetHotness(Remark: LLVMRemarkEntryRef) -> u64;
    pub fn LLVMRemarkEntryGetNextArg(It: LLVMRemarkArgRef, Remark: LLVMRemarkEntryRef) -> LLVMRemarkArgRef;
    pub fn LLVMRemarkEntryGetNumArgs(Remark: LLVMRemarkEntryRef) -> u32;
    pub fn LLVMRemarkEntryGetPassName(Remark: LLVMRemarkEntryRef) -> LLVMRemarkStringRef;
    pub fn LLVMRemarkEntryGetRemarkName(Remark: LLVMRemarkEntryRef) -> LLVMRemarkStringRef;
    pub fn LLVMRemarkEntryGetType(Remark: LLVMRemarkEntryRef) -> LLVMRemarkType;
}

pub enum LLVMRemarkOpaqueParser {}

pub type LLVMRemarkParserRef = *mut LLVMRemarkOpaqueParser;

extern "C" {
    pub fn LLVMRemarkParserCreateBitstream(Buf: *const ::libc::c_void, Size: u64) -> LLVMRemarkParserRef;
    pub fn LLVMRemarkParserCreateYAML(Buf: *const ::libc::c_void, Size: u64) -> LLVMRemarkParserRef;
    pub fn LLVMRemarkParserDispose(Parser: LLVMRemarkParserRef);
    pub fn LLVMRemarkParserGetErrorMessage(Parser: LLVMRemarkParserRef) -> *const ::libc::c_char;
    pub fn LLVMRemarkParserGetNext(Parser: LLVMRemarkParserRef) -> LLVMRemarkEntryRef;
    pub fn LLVMRemarkParserHasError(Parser: LLVMRemarkParserRef) -> LLVMBool;
}

pub const REMARKS_API_VERSION: u32 = 1;

extern "C" {
    pub fn LLVMRemarkVersion() -> u32;
}

// llvm-c/Support.h
extern "C" {
    pub fn LLVMAddSymbol(symbolName: *const ::libc::c_char, symbolValue: *mut ::libc::c_void);
    pub fn LLVMLoadLibraryPermanently(Filename: *const ::libc::c_char) -> LLVMBool;
    pub fn LLVMParseCommandLineOptions(argc: ::libc::c_int, argv: *const *const ::libc::c_char, Overview: *const ::libc::c_char);
    pub fn LLVMSearchForAddressOfSymbol(symbolName: *const ::libc::c_char) -> *mut ::libc::c_void;
}

// llvm-c/TargetMachine.h
#[derive(Debug)]
pub enum LLVMOpaqueTargetMachine {}

pub type LLVMTargetMachineRef = *mut LLVMOpaqueTargetMachine;

#[derive(Debug)]
pub enum LLVMTarget {}

pub type LLVMTargetRef = *mut LLVMTarget;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LLVMCodeGenOptLevel {
    LLVMCodeGenLevelNone = 0,
    LLVMCodeGenLevelLess = 1,
    LLVMCodeGenLevelDefault = 2,
    LLVMCodeGenLevelAggressive = 3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LLVMRelocMode {
    LLVMRelocDefault = 0,
    LLVMRelocStatic = 1,
    LLVMRelocPIC = 2,
    LLVMRelocDynamicNoPic = 3,
    LLVMRelocROPI = 4,
    LLVMRelocRWPI = 5,
    LLVMRelocROPI_RWPI = 6,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LLVMCodeModel {
    LLVMCodeModelDefault = 0,
    LLVMCodeModelJITDefault = 1,
    LLVMCodeModelTiny = 2,
    LLVMCodeModelSmall = 3,
    LLVMCodeModelKernel = 4,
    LLVMCodeModelMedium = 5,
    LLVMCodeModelLarge = 6,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LLVMCodeGenFileType {
    LLVMAssemblyFile = 0,
    LLVMObjectFile = 1,
}

extern "C" {
    pub fn LLVMAddAnalysisPasses(T: LLVMTargetMachineRef, PM: LLVMPassManagerRef);
    pub fn LLVMCreateTargetDataLayout(T: LLVMTargetMachineRef) -> LLVMTargetDataRef;
    pub fn LLVMCreateTargetMachine(T: LLVMTargetRef, Triple: *const ::libc::c_char, CPU: *const ::libc::c_char, Features: *const ::libc::c_char, Level: LLVMCodeGenOptLevel, Reloc: LLVMRelocMode, CodeModel: LLVMCodeModel) -> LLVMTargetMachineRef;
    pub fn LLVMDisposeTargetMachine(T: LLVMTargetMachineRef);
    pub fn LLVMGetDefaultTargetTriple() -> *mut ::libc::c_char;
    pub fn LLVMGetFirstTarget() -> LLVMTargetRef;
    pub fn LLVMGetHostCPUFeatures() -> *mut ::libc::c_char;
    pub fn LLVMGetHostCPUName() -> *mut ::libc::c_char;
    pub fn LLVMGetNextTarget(T: LLVMTargetRef) -> LLVMTargetRef;
    pub fn LLVMGetTargetDescription(T: LLVMTargetRef) -> *const ::libc::c_char;
    pub fn LLVMGetTargetFromName(Name: *const ::libc::c_char) -> LLVMTargetRef;
    pub fn LLVMGetTargetFromTriple(Triple: *const ::libc::c_char, T: *mut LLVMTargetRef, ErrorMessage: *mut *mut ::libc::c_char) -> LLVMBool;
    pub fn LLVMGetTargetMachineCPU(T: LLVMTargetMachineRef) -> *mut ::libc::c_char;
    pub fn LLVMGetTargetMachineFeatureString(T: LLVMTargetMachineRef) -> *mut ::libc::c_char;
    pub fn LLVMGetTargetMachineTarget(T: LLVMTargetMachineRef) -> LLVMTargetRef;
    pub fn LLVMGetTargetMachineTriple(T: LLVMTargetMachineRef) -> *mut ::libc::c_char;
    pub fn LLVMGetTargetName(T: LLVMTargetRef) -> *const ::libc::c_char;
    pub fn LLVMNormalizeTargetTriple(triple: *const ::libc::c_char) -> *mut ::libc::c_char;
    pub fn LLVMSetTargetMachineAsmVerbosity(T: LLVMTargetMachineRef, VerboseAsm: LLVMBool);
    pub fn LLVMTargetHasAsmBackend(T: LLVMTargetRef) -> LLVMBool;
    pub fn LLVMTargetHasJIT(T: LLVMTargetRef) -> LLVMBool;
    pub fn LLVMTargetHasTargetMachine(T: LLVMTargetRef) -> LLVMBool;
    pub fn LLVMTargetMachineEmitToFile(T: LLVMTargetMachineRef, M: LLVMModuleRef, Filename: *mut ::libc::c_char, codegen: LLVMCodeGenFileType, ErrorMessage: *mut *mut ::libc::c_char) -> LLVMBool;
    pub fn LLVMTargetMachineEmitToMemoryBuffer(T: LLVMTargetMachineRef, M: LLVMModuleRef, codegen: LLVMCodeGenFileType, ErrorMessage: *mut *mut ::libc::c_char, OutMemBuf: *mut LLVMMemoryBufferRef) -> LLVMBool;
}

//llvm-c/Target.h
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LLVMByteOrdering {
    LLVMBigEndian = 0,
    LLVMLittleEndian = 1,
}

#[derive(Debug)]
pub enum LLVMOpaqueTargetData {}

pub type LLVMTargetDataRef = *mut LLVMOpaqueTargetData;

#[derive(Debug)]
pub enum LLVMOpaqueTargetLibraryInfotData {}

pub type LLVMTargetLibraryInfoRef = *mut LLVMOpaqueTargetLibraryInfotData;

extern "C" {
    pub fn LLVMInitializeAMDGPUAsmParser();
    pub fn LLVMInitializeAMDGPUAsmPrinter();
    pub fn LLVMInitializeAMDGPUTarget();
    pub fn LLVMInitializeAMDGPUTargetInfo();
    pub fn LLVMInitializeAMDGPUTargetMC();
    pub fn LLVMInitializeNVPTXAsmPrinter();
    pub fn LLVMInitializeNVPTXTarget();
    pub fn LLVMInitializeNVPTXTargetInfo();
    pub fn LLVMInitializeNVPTXTargetMC();
    pub fn LLVMInitializeWebAssemblyAsmParser();
    pub fn LLVMInitializeWebAssemblyAsmPrinter();
    pub fn LLVMInitializeWebAssemblyDisassembler();
    pub fn LLVMInitializeWebAssemblyTarget();
    pub fn LLVMInitializeWebAssemblyTargetInfo();
    pub fn LLVMInitializeWebAssemblyTargetMC();
    pub fn LLVMInitializeX86AsmParser();
    pub fn LLVMInitializeX86AsmPrinter();
    pub fn LLVMInitializeX86Disassembler();
    pub fn LLVMInitializeX86Target();
    pub fn LLVMInitializeX86TargetInfo();
    pub fn LLVMInitializeX86TargetMC();
}

extern "C" {
    pub fn LLVMABIAlignmentOfType(TD: LLVMTargetDataRef, Ty: LLVMTypeRef) -> ::libc::c_uint;
    pub fn LLVMABISizeOfType(TD: LLVMTargetDataRef, Ty: LLVMTypeRef) -> ::libc::c_ulonglong;
    pub fn LLVMAddTargetLibraryInfo(TLI: LLVMTargetLibraryInfoRef, PM: LLVMPassManagerRef);
    pub fn LLVMByteOrder(TD: LLVMTargetDataRef) -> LLVMByteOrdering;
    pub fn LLVMCallFrameAlignmentOfType(TD: LLVMTargetDataRef, Ty: LLVMTypeRef) -> ::libc::c_uint;
    pub fn LLVMCopyStringRepOfTargetData(TD: LLVMTargetDataRef) -> *mut ::libc::c_char;
    pub fn LLVMCreateTargetData(StringRep: *const ::libc::c_char) -> LLVMTargetDataRef;
    pub fn LLVMDisposeTargetData(TD: LLVMTargetDataRef);
    pub fn LLVMElementAtOffset(TD: LLVMTargetDataRef, StructTy: LLVMTypeRef, Offset: ::libc::c_ulonglong) -> ::libc::c_uint;
    pub fn LLVMGetModuleDataLayout(M: LLVMModuleRef) -> LLVMTargetDataRef;
    pub fn LLVMIntPtrType(TD: LLVMTargetDataRef) -> LLVMTypeRef;
    pub fn LLVMIntPtrTypeForAS(TD: LLVMTargetDataRef, AS: ::libc::c_uint) -> LLVMTypeRef;
    pub fn LLVMIntPtrTypeForASInContext(C: LLVMContextRef, TD: LLVMTargetDataRef, AS: ::libc::c_uint) -> LLVMTypeRef;
    pub fn LLVMIntPtrTypeInContext(C: LLVMContextRef, TD: LLVMTargetDataRef) -> LLVMTypeRef;
    pub fn LLVMOffsetOfElement(TD: LLVMTargetDataRef, StructTy: LLVMTypeRef, Element: ::libc::c_uint) -> ::libc::c_ulonglong;
    pub fn LLVMPointerSize(TD: LLVMTargetDataRef) -> ::libc::c_uint;
    pub fn LLVMPointerSizeForAS(TD: LLVMTargetDataRef, AS: ::libc::c_uint) -> ::libc::c_uint;
    pub fn LLVMPreferredAlignmentOfGlobal(TD: LLVMTargetDataRef, GlobalVar: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMPreferredAlignmentOfType(TD: LLVMTargetDataRef, Ty: LLVMTypeRef) -> ::libc::c_uint;
    pub fn LLVMSetModuleDataLayout(M: LLVMModuleRef, R: LLVMTargetDataRef);
    pub fn LLVMSizeOfTypeInBits(TD: LLVMTargetDataRef, Ty: LLVMTypeRef) -> ::libc::c_ulonglong;
    pub fn LLVMStoreSizeOfType(TD: LLVMTargetDataRef, Ty: LLVMTypeRef) -> ::libc::c_ulonglong;
}

extern "C" {
    pub fn LLVM_InitializeAllAsmParsers();
    pub fn LLVM_InitializeAllAsmPrinters();
    pub fn LLVM_InitializeAllDisassemblers();
    pub fn LLVM_InitializeAllTargetInfos();
    pub fn LLVM_InitializeAllTargetMCs();
    pub fn LLVM_InitializeAllTargets();
    pub fn LLVM_InitializeNativeAsmParser() -> LLVMBool;
    pub fn LLVM_InitializeNativeAsmPrinter() -> LLVMBool;
    pub fn LLVM_InitializeNativeDisassembler() -> LLVMBool;
    pub fn LLVM_InitializeNativeTarget() -> LLVMBool;
}
