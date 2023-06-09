use llvm_lib::{core::*, debuginfo::*, prelude::*};
use std::{convert::TryInto, marker, ops::Range};

use crate::val::*;
use crate::*;

#[derive(Debug, PartialEq, Eq)]
pub struct DebugInfoBuilder<'ctx> {
    pub raw: LLVMDIBuilderRef,
    _marker: marker::PhantomData<&'ctx Context>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DIScope<'ctx> {
    raw: LLVMMetadataRef,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> DIScope<'ctx> {
    pub fn as_mut_ptr(&self) -> LLVMMetadataRef {
        self.raw
    }
}

pub trait AsDIScope<'ctx> {
    fn as_debug_info_scope(self) -> DIScope<'ctx>;
}
impl<'ctx> DebugInfoBuilder<'ctx> {
    pub fn new(
        module: &Module,
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
    ) -> (Self, DICompileUnit<'ctx>) {
        let raw = unsafe {
            if allow_unresolved {
                LLVMCreateDIBuilder(module.module.get())
            } else {
                LLVMCreateDIBuilderDisallowUnresolved(module.module.get())
            }
        };
        let builder = DebugInfoBuilder {
            raw,
            _marker: marker::PhantomData,
        };
        let file = builder.create_file(filename, directory);
        let cu = builder.create_compile_unit(
            language,
            file,
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
        );
        (builder, cu)
    }
    pub fn as_mut_ptr(&self) -> LLVMDIBuilderRef {
        self.raw
    }
    fn create_compile_unit(
        &self,
        language: DWARFSourceLanguage,
        file: DIFile<'ctx>,
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
    ) -> DICompileUnit<'ctx> {
        let raw = unsafe {
            {
                LLVMDIBuilderCreateCompileUnit(
                    self.raw,
                    language.into(),
                    file.raw,
                    producer.as_ptr() as _,
                    producer.len(),
                    is_optimized as _,
                    flags.as_ptr() as _,
                    flags.len(),
                    runtime_ver,
                    split_name.as_ptr() as _,
                    split_name.len(),
                    kind.into(),
                    dwo_id,
                    split_debug_inlining as _,
                    debug_info_for_profiling as _,
                    sysroot.as_ptr() as _,
                    sysroot.len(),
                    sdk.as_ptr() as _,
                    sdk.len(),
                )
            }
        };
        DICompileUnit {
            file,
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_function(
        &self,
        scope: DIScope<'ctx>,
        name: &str,
        linkage_name: Option<&str>,
        file: DIFile<'ctx>,
        line_no: u32,
        ditype: DISubroutineType<'ctx>,
        is_local_to_unit: bool,
        is_definition: bool,
        scope_line: u32,
        flags: DIFlags,
        is_optimized: bool,
    ) -> DISubprogram<'ctx> {
        let linkage_name = linkage_name.unwrap_or(name);
        let raw = unsafe {
            LLVMDIBuilderCreateFunction(
                self.raw,
                scope.raw,
                name.as_ptr() as _,
                name.len(),
                linkage_name.as_ptr() as _,
                linkage_name.len(),
                file.raw,
                line_no,
                ditype.raw,
                is_local_to_unit as _,
                is_definition as _,
                scope_line as libc::c_uint,
                flags,
                is_optimized as _,
            )
        };
        DISubprogram {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_lexical_block(
        &self,
        parent_scope: DIScope<'ctx>,
        file: DIFile<'ctx>,
        line: u32,
        column: u32,
    ) -> DILexicalBlock<'ctx> {
        let raw = unsafe {
            LLVMDIBuilderCreateLexicalBlock(
                self.raw,
                parent_scope.raw,
                file.raw,
                line as libc::c_uint,
                column as libc::c_uint,
            )
        };
        DILexicalBlock {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_file(&self, filename: &str, directory: &str) -> DIFile<'ctx> {
        let raw = unsafe {
            LLVMDIBuilderCreateFile(
                self.raw,
                filename.as_ptr() as _,
                filename.len(),
                directory.as_ptr() as _,
                directory.len(),
            )
        };
        DIFile {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_debug_location(
        &self,
        context: impl AsContextRef<'ctx>,
        line: u32,
        column: u32,
        scope: DIScope<'ctx>,
        inlined_at: Option<DILocation<'ctx>>,
    ) -> DILocation<'ctx> {
        let raw = unsafe {
            LLVMDIBuilderCreateDebugLocation(
                context.as_ctx_ref(),
                line,
                column,
                scope.raw,
                inlined_at.map(|l| l.raw).unwrap_or(std::ptr::null_mut()),
            )
        };
        DILocation {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_basic_type(
        &self,
        name: &str,
        size_in_bits: u64,
        encoding: LLVMDWARFTypeEncoding,
        lags: DIFlags,
    ) -> Result<DIBasicType<'ctx>, &'static str> {
        if name.is_empty() {
            return Err("basic types must have names");
        }
        let raw = unsafe {
            LLVMDIBuilderCreateBasicType(self.raw, name.as_ptr() as _, name.len(), size_in_bits, encoding, flags)
        };
        Ok(DIBasicType {
            raw,
            _marker: marker::PhantomData,
        })
    }
    pub fn create_typedef(
        &self,
        ditype: DIType<'ctx>,
        name: &str,
        file: DIFile<'ctx>,
        line_no: u32,
        scope: DIScope<'ctx>,
        align_in_bits: u32,
    ) -> DIDerivedType<'ctx> {
        let raw = unsafe {
            LLVMDIBuilderCreateTypedef(
                self.raw,
                ditype.raw,
                name.as_ptr() as _,
                name.len(),
                file.raw,
                line_no,
                scope.raw,
                align_in_bits,
            )
        };
        DIDerivedType {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_union_type(
        &self,
        scope: DIScope<'ctx>,
        name: &str,
        file: DIFile<'ctx>,
        line_no: u32,
        size_in_bits: u64,
        align_in_bits: u32,
        flags: DIFlags,
        elements: &[DIType<'ctx>],
        runtime_language: u32,
        unique_id: &str,
    ) -> DICompositeType<'ctx> {
        let mut elements: Vec<LLVMMetadataRef> = elements.iter().map(|dt| dt.raw).collect();
        let raw = unsafe {
            LLVMDIBuilderCreateUnionType(
                self.raw,
                scope.raw,
                name.as_ptr() as _,
                name.len(),
                file.raw,
                line_no,
                size_in_bits,
                align_in_bits,
                flags,
                elements.as_mut_ptr(),
                elements.len().try_into().unwrap(),
                runtime_language,
                unique_id.as_ptr() as _,
                unique_id.len(),
            )
        };
        DICompositeType {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_member_type(
        &self,
        scope: DIScope<'ctx>,
        name: &str,
        file: DIFile<'ctx>,
        line_no: libc::c_uint,
        size_in_bits: u64,
        align_in_bits: u32,
        offset_in_bits: u64,
        flags: DIFlags,
        ty: DIType<'ctx>,
    ) -> DIDerivedType<'ctx> {
        let raw = unsafe {
            LLVMDIBuilderCreateMemberType(
                self.raw,
                scope.raw,
                name.as_ptr() as _,
                name.len(),
                file.raw,
                line_no,
                size_in_bits,
                align_in_bits,
                offset_in_bits,
                flags,
                ty.raw,
            )
        };
        DIDerivedType {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_struct_type(
        &self,
        scope: DIScope<'ctx>,
        name: &str,
        file: DIFile<'ctx>,
        line_no: libc::c_uint,
        size_in_bits: u64,
        align_in_bits: u32,
        flags: DIFlags,
        derived_from: Option<DIType<'ctx>>,
        elements: &[DIType<'ctx>],
        runtime_language: libc::c_uint,
        vtable_holder: Option<DIType<'ctx>>,
        unique_id: &str,
    ) -> DICompositeType<'ctx> {
        let mut elements: Vec<LLVMMetadataRef> = elements.iter().map(|dt| dt.raw).collect();
        let derived_from = derived_from.map_or(std::ptr::null_mut(), |dt| dt.raw);
        let vtable_holder = vtable_holder.map_or(std::ptr::null_mut(), |dt| dt.raw);
        let raw = unsafe {
            LLVMDIBuilderCreateStructType(
                self.raw,
                scope.raw,
                name.as_ptr() as _,
                name.len(),
                file.raw,
                line_no,
                size_in_bits,
                align_in_bits,
                flags,
                derived_from,
                elements.as_mut_ptr(),
                elements.len().try_into().unwrap(),
                runtime_language,
                vtable_holder,
                unique_id.as_ptr() as _,
                unique_id.len(),
            )
        };
        DICompositeType {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_subroutine_type(
        &self,
        file: DIFile<'ctx>,
        return_type: Option<DIType<'ctx>>,
        parameter_types: &[DIType<'ctx>],
        flags: DIFlags,
    ) -> DISubroutineType<'ctx> {
        let mut p = vec![return_type.map_or(std::ptr::null_mut(), |t| t.raw)];
        p.append(&mut parameter_types.iter().map(|t| t.raw).collect::<Vec<LLVMMetadataRef>>());
        let raw = unsafe {
            LLVMDIBuilderCreateSubroutineType(self.raw, file.raw, p.as_mut_ptr(), p.len().try_into().unwrap(), flags)
        };
        DISubroutineType {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_pointer_type(
        &self,
        name: &str,
        pointee: DIType<'ctx>,
        size_in_bits: u64,
        align_in_bits: u32,
        address_space: AddressSpace,
    ) -> DIDerivedType<'ctx> {
        let raw = unsafe {
            LLVMDIBuilderCreatePointerType(
                self.raw,
                pointee.raw,
                size_in_bits,
                align_in_bits,
                address_space.0,
                name.as_ptr() as _,
                name.len(),
            )
        };
        DIDerivedType {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_reference_type(&self, pointee: DIType<'ctx>, tag: u32) -> DIDerivedType<'ctx> {
        let raw = unsafe { LLVMDIBuilderCreateReferenceType(self.raw, tag, pointee.raw) };
        DIDerivedType {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_array_type(
        &self,
        inner_type: DIType<'ctx>,
        size_in_bits: u64,
        align_in_bits: u32,
        subscripts: &[Range<i64>],
    ) -> DICompositeType<'ctx> {
        let mut subscripts = subscripts
            .iter()
            .map(|range| {
                let lower = range.start;
                let upper = range.end;
                let subscript_size = upper - lower;
                unsafe { LLVMDIBuilderGetOrCreateSubrange(self.raw, lower, subscript_size) }
            })
            .collect::<Vec<_>>();
        let raw = unsafe {
            LLVMDIBuilderCreateArrayType(
                self.raw,
                size_in_bits,
                align_in_bits,
                inner_type.raw,
                subscripts.as_mut_ptr(),
                subscripts.len().try_into().unwrap(),
            )
        };
        DICompositeType {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_global_variable_expression(
        &self,
        scope: DIScope<'ctx>,
        name: &str,
        linkage: &str,
        file: DIFile<'ctx>,
        line_no: u32,
        ty: DIType<'ctx>,
        local_to_unit: bool,
        expression: Option<DIExpression>,
        declaration: Option<DIScope>,
        align_in_bits: u32,
    ) -> DIGlobalVariableExpression<'ctx> {
        let expression_ptr = expression.map_or(std::ptr::null_mut(), |dt| dt.raw);
        let decl_ptr = declaration.map_or(std::ptr::null_mut(), |dt| dt.raw);
        let raw = unsafe {
            LLVMDIBuilderCreateGlobalVariableExpression(
                self.raw,
                scope.raw,
                name.as_ptr() as _,
                name.len(),
                linkage.as_ptr() as _,
                linkage.len(),
                file.raw,
                line_no,
                ty.raw,
                local_to_unit as _,
                expression_ptr,
                decl_ptr,
                align_in_bits,
            )
        };
        DIGlobalVariableExpression {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_constant_expression(&self, value: i64) -> DIExpression<'ctx> {
        let raw = unsafe { LLVMDIBuilderCreateConstantValueExpression(self.raw, value as _) };
        DIExpression {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_parameter_variable(
        &self,
        scope: DIScope<'ctx>,
        name: &str,
        arg_no: u32,
        file: DIFile<'ctx>,
        line_no: u32,
        ty: DIType<'ctx>,
        always_preserve: bool,
        flags: DIFlags,
    ) -> DILocalVariable<'ctx> {
        let raw = unsafe {
            LLVMDIBuilderCreateParameterVariable(
                self.raw,
                scope.raw,
                name.as_ptr() as _,
                name.len(),
                arg_no,
                file.raw,
                line_no,
                ty.raw,
                always_preserve as _,
                flags,
            )
        };
        DILocalVariable {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_auto_variable(
        &self,
        scope: DIScope<'ctx>,
        name: &str,
        file: DIFile<'ctx>,
        line_no: u32,
        ty: DIType<'ctx>,
        always_preserve: bool,
        flags: DIFlags,
        align_in_bits: u32,
    ) -> DILocalVariable<'ctx> {
        let raw = unsafe {
            LLVMDIBuilderCreateAutoVariable(
                self.raw,
                scope.raw,
                name.as_ptr() as _,
                name.len(),
                file.raw,
                line_no,
                ty.raw,
                always_preserve as _,
                flags,
                align_in_bits,
            )
        };
        DILocalVariable {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn create_namespace(&self, scope: DIScope<'ctx>, name: &str, export_symbols: bool) -> DINamespace<'ctx> {
        let raw = unsafe {
            LLVMDIBuilderCreateNameSpace(self.raw, scope.raw, name.as_ptr() as _, name.len(), export_symbols as _)
        };
        DINamespace {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn insert_declare_before_instruction(
        &self,
        storage: PointerValue<'ctx>,
        var_info: Option<DILocalVariable<'ctx>>,
        expr: Option<DIExpression<'ctx>>,
        debug_loc: DILocation<'ctx>,
        instruction: InstructionValue<'ctx>,
    ) -> InstructionValue<'ctx> {
        let y = unsafe {
            LLVMDIBuilderInsertDeclareBefore(
                self.raw,
                storage.as_value_ref(),
                var_info.map(|v| v.raw).unwrap_or(std::ptr::null_mut()),
                expr.unwrap_or_else(|| self.create_expression(vec![])).raw,
                debug_loc.raw,
                instruction.as_value_ref(),
            )
        };
        unsafe { InstructionValue::new(y) }
    }
    pub fn insert_declare_at_end(
        &self,
        storage: PointerValue<'ctx>,
        var_info: Option<DILocalVariable<'ctx>>,
        expr: Option<DIExpression<'ctx>>,
        debug_loc: DILocation<'ctx>,
        block: BasicBlock<'ctx>,
    ) -> InstructionValue<'ctx> {
        let y = unsafe {
            LLVMDIBuilderInsertDeclareAtEnd(
                self.raw,
                storage.as_value_ref(),
                var_info.map(|v| v.raw).unwrap_or(std::ptr::null_mut()),
                expr.unwrap_or_else(|| self.create_expression(vec![])).raw,
                debug_loc.raw,
                block.raw,
            )
        };
        unsafe { InstructionValue::new(y) }
    }
    pub fn create_expression(&self, mut address_operations: Vec<i64>) -> DIExpression<'ctx> {
        let raw = unsafe {
            LLVMDIBuilderCreateExpression(
                self.raw,
                address_operations.as_mut_ptr() as *mut _,
                address_operations.len(),
            )
        };
        DIExpression {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn insert_dbg_value_before(
        &self,
        value: BasicValueEnum<'ctx>,
        var_info: DILocalVariable<'ctx>,
        expr: Option<DIExpression<'ctx>>,
        debug_loc: DILocation<'ctx>,
        instruction: InstructionValue<'ctx>,
    ) -> InstructionValue<'ctx> {
        let y = unsafe {
            LLVMDIBuilderInsertDbgValueBefore(
                self.raw,
                value.as_value_ref(),
                var_info.raw,
                expr.unwrap_or_else(|| self.create_expression(vec![])).raw,
                debug_loc.raw,
                instruction.as_value_ref(),
            )
        };
        unsafe { InstructionValue::new(y) }
    }
    pub unsafe fn create_placeholder_derived_type(&self, context: impl AsContextRef<'ctx>) -> DIDerivedType<'ctx> {
        let raw = LLVMTemporaryMDNode(context.as_ctx_ref(), std::ptr::null_mut(), 0);
        DIDerivedType {
            raw,
            _marker: marker::PhantomData,
        }
    }
    pub unsafe fn replace_placeholder_derived_type(
        &self,
        placeholder: DIDerivedType<'ctx>,
        other: DIDerivedType<'ctx>,
    ) {
        LLVMMetadataReplaceAllUsesWith(placeholder.raw, other.raw);
    }
    pub fn finalize(&self) {
        unsafe { LLVMDIBuilderFinalize(self.raw) };
    }
}
impl<'ctx> Drop for DebugInfoBuilder<'ctx> {
    fn drop(&mut self) {
        self.finalize();
        unsafe { LLVMDisposeDIBuilder(self.raw) }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DIFile<'ctx> {
    pub raw: LLVMMetadataRef,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> AsDIScope<'ctx> for DIFile<'ctx> {
    fn as_debug_info_scope(self) -> DIScope<'ctx> {
        DIScope {
            raw: self.raw,
            _marker: marker::PhantomData,
        }
    }
}
impl<'ctx> DIFile<'ctx> {
    pub fn as_mut_ptr(&self) -> LLVMMetadataRef {
        self.raw
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DICompileUnit<'ctx> {
    file: DIFile<'ctx>,
    pub raw: LLVMMetadataRef,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> DICompileUnit<'ctx> {
    pub fn get_file(&self) -> DIFile<'ctx> {
        self.file
    }
    pub fn as_mut_ptr(&self) -> LLVMMetadataRef {
        self.raw
    }
}
impl<'ctx> AsDIScope<'ctx> for DICompileUnit<'ctx> {
    fn as_debug_info_scope(self) -> DIScope<'ctx> {
        DIScope {
            raw: self.raw,
            _marker: marker::PhantomData,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DINamespace<'ctx> {
    pub raw: LLVMMetadataRef,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> DINamespace<'ctx> {
    pub fn as_mut_ptr(&self) -> LLVMMetadataRef {
        self.raw
    }
}
impl<'ctx> AsDIScope<'ctx> for DINamespace<'ctx> {
    fn as_debug_info_scope(self) -> DIScope<'ctx> {
        DIScope {
            raw: self.raw,
            _marker: marker::PhantomData,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DISubprogram<'ctx> {
    pub raw: LLVMMetadataRef,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> AsDIScope<'ctx> for DISubprogram<'ctx> {
    fn as_debug_info_scope(self) -> DIScope<'ctx> {
        DIScope {
            raw: self.raw,
            _marker: marker::PhantomData,
        }
    }
}
impl<'ctx> DISubprogram<'ctx> {
    pub fn as_mut_ptr(&self) -> LLVMMetadataRef {
        self.raw
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DIType<'ctx> {
    pub raw: LLVMMetadataRef,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> DIType<'ctx> {
    pub fn get_size_in_bits(&self) -> u64 {
        unsafe { LLVMDITypeGetSizeInBits(self.raw) }
    }
    pub fn get_align_in_bits(&self) -> u32 {
        unsafe { LLVMDITypeGetAlignInBits(self.raw) }
    }
    pub fn get_offset_in_bits(&self) -> u64 {
        unsafe { LLVMDITypeGetOffsetInBits(self.raw) }
    }
    pub fn as_mut_ptr(&self) -> LLVMMetadataRef {
        self.raw
    }
}
impl<'ctx> AsDIScope<'ctx> for DIType<'ctx> {
    fn as_debug_info_scope(self) -> DIScope<'ctx> {
        DIScope {
            raw: self.raw,
            _marker: marker::PhantomData,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DIDerivedType<'ctx> {
    pub raw: LLVMMetadataRef,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> DIDerivedType<'ctx> {
    pub fn as_type(&self) -> DIType<'ctx> {
        DIType {
            raw: self.raw,
            _marker: marker::PhantomData,
        }
    }
}
impl<'ctx> DIDerivedType<'ctx> {
    pub fn as_mut_ptr(&self) -> LLVMMetadataRef {
        self.raw
    }
}
impl<'ctx> AsDIScope<'ctx> for DIDerivedType<'ctx> {
    fn as_debug_info_scope(self) -> DIScope<'ctx> {
        DIScope {
            raw: self.raw,
            _marker: marker::PhantomData,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DIBasicType<'ctx> {
    pub raw: LLVMMetadataRef,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> DIBasicType<'ctx> {
    pub fn as_type(&self) -> DIType<'ctx> {
        DIType {
            raw: self.raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn as_mut_ptr(&self) -> LLVMMetadataRef {
        self.raw
    }
}
impl<'ctx> AsDIScope<'ctx> for DIBasicType<'ctx> {
    fn as_debug_info_scope(self) -> DIScope<'ctx> {
        DIScope {
            raw: self.raw,
            _marker: marker::PhantomData,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DICompositeType<'ctx> {
    pub raw: LLVMMetadataRef,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> DICompositeType<'ctx> {
    pub fn as_type(&self) -> DIType<'ctx> {
        DIType {
            raw: self.raw,
            _marker: marker::PhantomData,
        }
    }
    pub fn as_mut_ptr(&self) -> LLVMMetadataRef {
        self.raw
    }
}
impl<'ctx> AsDIScope<'ctx> for DICompositeType<'ctx> {
    fn as_debug_info_scope(self) -> DIScope<'ctx> {
        DIScope {
            raw: self.raw,
            _marker: marker::PhantomData,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DISubroutineType<'ctx> {
    pub raw: LLVMMetadataRef,
    _marker: marker::PhantomData<&'ctx Context>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DILexicalBlock<'ctx> {
    pub raw: LLVMMetadataRef,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> DILexicalBlock<'ctx> {
    pub fn as_mut_ptr(&self) -> LLVMMetadataRef {
        self.raw
    }
}
impl<'ctx> AsDIScope<'ctx> for DILexicalBlock<'ctx> {
    fn as_debug_info_scope(self) -> DIScope<'ctx> {
        DIScope {
            raw: self.raw,
            _marker: marker::PhantomData,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DILocation<'ctx> {
    pub raw: LLVMMetadataRef,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> DILocation<'ctx> {
    pub fn get_line(&self) -> u32 {
        unsafe { LLVMDILocationGetLine(self.raw) }
    }
    pub fn get_column(&self) -> u32 {
        unsafe { LLVMDILocationGetColumn(self.raw) }
    }
    pub fn get_scope(&self) -> DIScope<'ctx> {
        DIScope {
            raw: unsafe { LLVMDILocationGetScope(self.raw) },
            _marker: marker::PhantomData,
        }
    }
    pub fn as_mut_ptr(&self) -> LLVMMetadataRef {
        self.raw
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DILocalVariable<'ctx> {
    pub raw: LLVMMetadataRef,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> DILocalVariable<'ctx> {
    pub fn as_mut_ptr(&self) -> LLVMMetadataRef {
        self.raw
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DIGlobalVariableExpression<'ctx> {
    pub raw: LLVMMetadataRef,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> DIGlobalVariableExpression<'ctx> {
    pub fn as_metadata_value(&self, context: impl AsContextRef<'ctx>) -> MetadataValue<'ctx> {
        unsafe { MetadataValue::new(LLVMMetadataAsValue(context.as_ctx_ref(), self.raw)) }
    }
    pub fn as_mut_ptr(&self) -> LLVMMetadataRef {
        self.raw
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DIExpression<'ctx> {
    pub raw: LLVMMetadataRef,
    _marker: marker::PhantomData<&'ctx Context>,
}
impl<'ctx> DIExpression<'ctx> {
    pub fn as_mut_ptr(&self) -> LLVMMetadataRef {
        self.raw
    }
}

pub fn debug_metadata_version() -> libc::c_uint {
    unsafe { LLVMDebugMetadataVersion() }
}

mod flags {
    pub use llvm_lib::debuginfo::LLVMDIFlags as DIFlags;
    use llvm_lib::debuginfo::{LLVMDWARFEmissionKind, LLVMDWARFSourceLanguage};
    pub trait DIFlagsConstants {
        const ZERO: Self;
        const PRIVATE: Self;
        const PROTECTED: Self;
        const PUBLIC: Self;
        const FWD_DECL: Self;
        const APPLE_BLOCK: Self;
        const VIRTUAL: Self;
        const ARTIFICIAL: Self;
        const EXPLICIT: Self;
        const PROTOTYPED: Self;
        const OBJC_CLASS_COMPLETE: Self;
        const OBJECT_POINTER: Self;
        const VECTOR: Self;
        const STATIC_MEMBER: Self;
        const LVALUE_REFERENCE: Self;
        const RVALUE_REFERENCE: Self;
        const RESERVED: Self;
        const SINGLE_INHERITANCE: Self;
        const MULTIPLE_INHERITANCE: Self;
        const VIRTUAL_INHERITANCE: Self;
        const INTRODUCED_VIRTUAL: Self;
        const BIT_FIELD: Self;
        const NO_RETURN: Self;
        const TYPE_PASS_BY_VALUE: Self;
        const TYPE_PASS_BY_REFERENCE: Self;
        const THUNK: Self;
        const INDIRECT_VIRTUAL_BASE: Self;
    }
    impl DIFlagsConstants for DIFlags {
        const ZERO: DIFlags = llvm_lib::debuginfo::LLVMDIFlagZero;
        const PRIVATE: DIFlags = llvm_lib::debuginfo::LLVMDIFlagPrivate;
        const PROTECTED: DIFlags = llvm_lib::debuginfo::LLVMDIFlagProtected;
        const PUBLIC: DIFlags = llvm_lib::debuginfo::LLVMDIFlagPublic;
        const FWD_DECL: DIFlags = llvm_lib::debuginfo::LLVMDIFlagFwdDecl;
        const APPLE_BLOCK: DIFlags = llvm_lib::debuginfo::LLVMDIFlagAppleBlock;
        const VIRTUAL: DIFlags = llvm_lib::debuginfo::LLVMDIFlagVirtual;
        const ARTIFICIAL: DIFlags = llvm_lib::debuginfo::LLVMDIFlagArtificial;
        const EXPLICIT: DIFlags = llvm_lib::debuginfo::LLVMDIFlagExplicit;
        const PROTOTYPED: DIFlags = llvm_lib::debuginfo::LLVMDIFlagPrototyped;
        const OBJC_CLASS_COMPLETE: DIFlags = llvm_lib::debuginfo::LLVMDIFlagObjcClassComplete;
        const OBJECT_POINTER: DIFlags = llvm_lib::debuginfo::LLVMDIFlagObjectPointer;
        const VECTOR: DIFlags = llvm_lib::debuginfo::LLVMDIFlagVector;
        const STATIC_MEMBER: DIFlags = llvm_lib::debuginfo::LLVMDIFlagStaticMember;
        const LVALUE_REFERENCE: DIFlags = llvm_lib::debuginfo::LLVMDIFlagLValueReference;
        const RVALUE_REFERENCE: DIFlags = llvm_lib::debuginfo::LLVMDIFlagRValueReference;
        const RESERVED: DIFlags = llvm_lib::debuginfo::LLVMDIFlagReserved;
        const SINGLE_INHERITANCE: DIFlags = llvm_lib::debuginfo::LLVMDIFlagSingleInheritance;
        const MULTIPLE_INHERITANCE: DIFlags = llvm_lib::debuginfo::LLVMDIFlagMultipleInheritance;
        const VIRTUAL_INHERITANCE: DIFlags = llvm_lib::debuginfo::LLVMDIFlagVirtualInheritance;
        const INTRODUCED_VIRTUAL: DIFlags = llvm_lib::debuginfo::LLVMDIFlagIntroducedVirtual;
        const BIT_FIELD: DIFlags = llvm_lib::debuginfo::LLVMDIFlagBitField;
        const NO_RETURN: DIFlags = llvm_lib::debuginfo::LLVMDIFlagNoReturn;
        const TYPE_PASS_BY_VALUE: DIFlags = llvm_lib::debuginfo::LLVMDIFlagTypePassByValue;
        const TYPE_PASS_BY_REFERENCE: DIFlags = llvm_lib::debuginfo::LLVMDIFlagTypePassByReference;
        const THUNK: DIFlags = llvm_lib::debuginfo::LLVMDIFlagThunk;
        const INDIRECT_VIRTUAL_BASE: DIFlags = llvm_lib::debuginfo::LLVMDIFlagIndirectVirtualBase;
    }
    #[llvm_enum(LLVMDWARFEmissionKind)]
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub enum DWARFEmissionKind {
        #[llvm_variant(LLVMDWARFEmissionKindNone)]
        None,
        #[llvm_variant(LLVMDWARFEmissionKindFull)]
        Full,
        #[llvm_variant(LLVMDWARFEmissionKindLineTablesOnly)]
        LineTablesOnly,
    }
    #[llvm_enum(LLVMDWARFSourceLanguage)]
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub enum DWARFSourceLanguage {
        #[llvm_variant(LLVMDWARFSourceLanguageC89)]
        C89,
        #[llvm_variant(LLVMDWARFSourceLanguageC)]
        C,
        #[llvm_variant(LLVMDWARFSourceLanguageAda83)]
        Ada83,
        #[llvm_variant(LLVMDWARFSourceLanguageC_plus_plus)]
        CPlusPlus,
        #[llvm_variant(LLVMDWARFSourceLanguageCobol74)]
        Cobol74,
        #[llvm_variant(LLVMDWARFSourceLanguageCobol85)]
        Cobol85,
        #[llvm_variant(LLVMDWARFSourceLanguageFortran77)]
        Fortran77,
        #[llvm_variant(LLVMDWARFSourceLanguageFortran90)]
        Fortran90,
        #[llvm_variant(LLVMDWARFSourceLanguagePascal83)]
        Pascal83,
        #[llvm_variant(LLVMDWARFSourceLanguageModula2)]
        Modula2,
        #[llvm_variant(LLVMDWARFSourceLanguageJava)]
        Java,
        #[llvm_variant(LLVMDWARFSourceLanguageC99)]
        C99,
        #[llvm_variant(LLVMDWARFSourceLanguageAda95)]
        Ada95,
        #[llvm_variant(LLVMDWARFSourceLanguageFortran95)]
        Fortran95,
        #[llvm_variant(LLVMDWARFSourceLanguagePLI)]
        PLI,
        #[llvm_variant(LLVMDWARFSourceLanguageObjC)]
        ObjC,
        #[llvm_variant(LLVMDWARFSourceLanguageObjC_plus_plus)]
        ObjCPlusPlus,
        #[llvm_variant(LLVMDWARFSourceLanguageUPC)]
        UPC,
        #[llvm_variant(LLVMDWARFSourceLanguageD)]
        D,
        #[llvm_variant(LLVMDWARFSourceLanguagePython)]
        Python,
        #[llvm_variant(LLVMDWARFSourceLanguageOpenCL)]
        OpenCL,
        #[llvm_variant(LLVMDWARFSourceLanguageGo)]
        Go,
        #[llvm_variant(LLVMDWARFSourceLanguageModula3)]
        Modula3,
        #[llvm_variant(LLVMDWARFSourceLanguageHaskell)]
        Haskell,
        #[llvm_variant(LLVMDWARFSourceLanguageC_plus_plus_03)]
        CPlusPlus03,
        #[llvm_variant(LLVMDWARFSourceLanguageC_plus_plus_11)]
        CPlusPlus11,
        #[llvm_variant(LLVMDWARFSourceLanguageOCaml)]
        OCaml,
        #[llvm_variant(LLVMDWARFSourceLanguageRust)]
        Rust,
        #[llvm_variant(LLVMDWARFSourceLanguageC11)]
        C11,
        #[llvm_variant(LLVMDWARFSourceLanguageSwift)]
        Swift,
        #[llvm_variant(LLVMDWARFSourceLanguageJulia)]
        Julia,
        #[llvm_variant(LLVMDWARFSourceLanguageDylan)]
        Dylan,
        #[llvm_variant(LLVMDWARFSourceLanguageC_plus_plus_14)]
        CPlusPlus14,
        #[llvm_variant(LLVMDWARFSourceLanguageFortran03)]
        Fortran03,
        #[llvm_variant(LLVMDWARFSourceLanguageFortran08)]
        Fortran08,
        #[llvm_variant(LLVMDWARFSourceLanguageRenderScript)]
        RenderScript,
        #[llvm_variant(LLVMDWARFSourceLanguageBLISS)]
        BLISS,
        #[llvm_variant(LLVMDWARFSourceLanguageMips_Assembler)]
        MipsAssembler,
        #[llvm_variant(LLVMDWARFSourceLanguageGOOGLE_RenderScript)]
        GOOGLERenderScript,
        #[llvm_variant(LLVMDWARFSourceLanguageBORLAND_Delphi)]
        BORLANDDelphi,
        #[llvm_variant(LLVMDWARFSourceLanguageKotlin)]
        Kotlin,
        #[llvm_variant(LLVMDWARFSourceLanguageZig)]
        Zig,
        #[llvm_variant(LLVMDWARFSourceLanguageCrystal)]
        Crystal,
        #[llvm_variant(LLVMDWARFSourceLanguageC_plus_plus_17)]
        CPlusPlus17,
        #[llvm_variant(LLVMDWARFSourceLanguageC_plus_plus_20)]
        CPlusPlus20,
        #[llvm_variant(LLVMDWARFSourceLanguageC17)]
        C17,
        #[llvm_variant(LLVMDWARFSourceLanguageFortran18)]
        Fortran18,
        #[llvm_variant(LLVMDWARFSourceLanguageAda2005)]
        Ada2005,
        #[llvm_variant(LLVMDWARFSourceLanguageAda2012)]
        Ada2012,
    }
}

pub use flags::*;
