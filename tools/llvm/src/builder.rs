use crate::block::BasicBlock;
use crate::debug::DILocation;
use crate::support::to_c_str;
use crate::typ::*;
use crate::val::*;
use crate::{AtomicOrdering, AtomicRMWBinOp, FloatPredicate, IntPredicate};
use llvm_lib::core::*;
use llvm_lib::prelude::{LLVMBuilderRef, LLVMValueRef};
use std::marker::PhantomData;

#[derive(Debug)]
pub struct Builder<'ctx> {
    builder: LLVMBuilderRef,
    _marker: PhantomData<&'ctx ()>,
}

use crate::ctx::Context;
impl<'ctx> Builder<'ctx> {
    pub unsafe fn new(builder: LLVMBuilderRef) -> Self {
        debug_assert!(!builder.is_null());
        Builder {
            builder,
            _marker: PhantomData,
        }
    }
    pub fn as_mut_ptr(&self) -> LLVMBuilderRef {
        self.builder
    }
    pub fn build_return(&self, value: Option<&dyn BasicValue<'ctx>>) -> InstructionValue<'ctx> {
        let value = unsafe {
            value.map_or_else(
                || LLVMBuildRetVoid(self.builder),
                |value| LLVMBuildRet(self.builder, value.as_value_ref()),
            )
        };
        unsafe { InstructionValue::new(value) }
    }
    pub fn build_aggregate_return(&self, values: &[BasicValueEnum<'ctx>]) -> InstructionValue<'ctx> {
        let mut args: Vec<LLVMValueRef> = values.iter().map(|val| val.as_value_ref()).collect();
        let value = unsafe { LLVMBuildAggregateRet(self.builder, args.as_mut_ptr(), args.len() as u32) };
        unsafe { InstructionValue::new(value) }
    }
    pub fn build_call(
        &self,
        function: FunctionValue<'ctx>,
        args: &[BasicMetadataValueEnum<'ctx>],
        name: &str,
    ) -> CallSiteValue<'ctx> {
        self.build_direct_call(function, args, name)
    }
    pub fn build_direct_call(
        &self,
        function: FunctionValue<'ctx>,
        args: &[BasicMetadataValueEnum<'ctx>],
        name: &str,
    ) -> CallSiteValue<'ctx> {
        self.build_call_help(function.get_type(), function.as_value_ref(), args, name)
    }
    pub fn build_indirect_call(
        &self,
        function_type: FunctionType<'ctx>,
        function_pointer: PointerValue<'ctx>,
        args: &[BasicMetadataValueEnum<'ctx>],
        name: &str,
    ) -> CallSiteValue<'ctx> {
        self.build_call_help(function_type, function_pointer.as_value_ref(), args, name)
    }
    fn build_call_help(
        &self,
        function_type: FunctionType<'ctx>,
        fn_val_ref: LLVMValueRef,
        args: &[BasicMetadataValueEnum<'ctx>],
        name: &str,
    ) -> CallSiteValue<'ctx> {
        let name = match function_type.get_return_type() {
            None => "",
            Some(_) => name,
        };
        let fn_ty_ref = function_type.as_type_ref();
        let c_string = to_c_str(name);
        let mut args: Vec<LLVMValueRef> = args.iter().map(|val| val.as_value_ref()).collect();
        let value = unsafe {
            LLVMBuildCall2(
                self.builder,
                fn_ty_ref,
                fn_val_ref,
                args.as_mut_ptr(),
                args.len() as u32,
                c_string.as_ptr(),
            )
        };
        unsafe { CallSiteValue::new(value) }
    }
    pub fn build_invoke(
        &self,
        function: FunctionValue<'ctx>,
        args: &[BasicValueEnum<'ctx>],
        then_block: BasicBlock<'ctx>,
        catch_block: BasicBlock<'ctx>,
        name: &str,
    ) -> CallSiteValue<'ctx> {
        self.build_direct_invoke(function, args, then_block, catch_block, name)
    }
    pub fn build_direct_invoke(
        &self,
        function: FunctionValue<'ctx>,
        args: &[BasicValueEnum<'ctx>],
        then_block: BasicBlock<'ctx>,
        catch_block: BasicBlock<'ctx>,
        name: &str,
    ) -> CallSiteValue<'ctx> {
        self.build_invoke_help(
            function.get_type(),
            function.as_value_ref(),
            args,
            then_block,
            catch_block,
            name,
        )
    }
    pub fn build_indirect_invoke(
        &self,
        function_type: FunctionType<'ctx>,
        function_pointer: PointerValue<'ctx>,
        args: &[BasicValueEnum<'ctx>],
        then_block: BasicBlock<'ctx>,
        catch_block: BasicBlock<'ctx>,
        name: &str,
    ) -> CallSiteValue<'ctx> {
        self.build_invoke_help(
            function_type,
            function_pointer.as_value_ref(),
            args,
            then_block,
            catch_block,
            name,
        )
    }
    fn build_invoke_help(
        &self,
        fn_ty: FunctionType<'ctx>,
        fn_val_ref: LLVMValueRef,
        args: &[BasicValueEnum<'ctx>],
        then_block: BasicBlock<'ctx>,
        catch_block: BasicBlock<'ctx>,
        name: &str,
    ) -> CallSiteValue<'ctx> {
        let fn_ty_ref = fn_ty.as_type_ref();
        let name = if fn_ty.get_return_type().is_none() { "" } else { name };
        let c_string = to_c_str(name);
        let mut args: Vec<LLVMValueRef> = args.iter().map(|val| val.as_value_ref()).collect();
        let value = unsafe {
            LLVMBuildInvoke2(
                self.builder,
                fn_ty_ref,
                fn_val_ref,
                args.as_mut_ptr(),
                args.len() as u32,
                then_block.basic_block,
                catch_block.basic_block,
                c_string.as_ptr(),
            )
        };
        unsafe { CallSiteValue::new(value) }
    }
    pub fn build_landing_pad<T>(
        &self,
        exception_type: T,
        personality_function: FunctionValue<'ctx>,
        clauses: &[BasicValueEnum<'ctx>],
        is_cleanup: bool,
        name: &str,
    ) -> BasicValueEnum<'ctx>
    where
        T: BasicType<'ctx>,
    {
        let c_string = to_c_str(name);
        let num_clauses = clauses.len() as u32;
        let value = unsafe {
            LLVMBuildLandingPad(
                self.builder,
                exception_type.as_type_ref(),
                personality_function.as_value_ref(),
                num_clauses,
                c_string.as_ptr(),
            )
        };
        for clause in clauses {
            unsafe {
                LLVMAddClause(value, clause.as_value_ref());
            }
        }
        unsafe {
            LLVMSetCleanup(value, is_cleanup as _);
        };
        unsafe { BasicValueEnum::new(value) }
    }
    pub fn build_resume<V: BasicValue<'ctx>>(&self, value: V) -> InstructionValue<'ctx> {
        let val = unsafe { LLVMBuildResume(self.builder, value.as_value_ref()) };
        unsafe { InstructionValue::new(val) }
    }
    pub unsafe fn build_gep<T: BasicType<'ctx>>(
        &self,
        pointee_ty: T,
        ptr: PointerValue<'ctx>,
        ordered_indexes: &[IntValue<'ctx>],
        name: &str,
    ) -> PointerValue<'ctx> {
        let c_string = to_c_str(name);
        let mut index_values: Vec<LLVMValueRef> = ordered_indexes.iter().map(|val| val.as_value_ref()).collect();
        let value = LLVMBuildGEP2(
            self.builder,
            pointee_ty.as_type_ref(),
            ptr.as_value_ref(),
            index_values.as_mut_ptr(),
            index_values.len() as u32,
            c_string.as_ptr(),
        );
        PointerValue::new(value)
    }
    pub unsafe fn build_in_bounds_gep<T: BasicType<'ctx>>(
        &self,
        pointee_ty: T,
        ptr: PointerValue<'ctx>,
        ordered_indexes: &[IntValue<'ctx>],
        name: &str,
    ) -> PointerValue<'ctx> {
        let c_string = to_c_str(name);
        let mut index_values: Vec<LLVMValueRef> = ordered_indexes.iter().map(|val| val.as_value_ref()).collect();
        let value = LLVMBuildInBoundsGEP2(
            self.builder,
            pointee_ty.as_type_ref(),
            ptr.as_value_ref(),
            index_values.as_mut_ptr(),
            index_values.len() as u32,
            c_string.as_ptr(),
        );
        PointerValue::new(value)
    }
    pub fn build_struct_gep<T: BasicType<'ctx>>(
        &self,
        pointee_ty: T,
        ptr: PointerValue<'ctx>,
        index: u32,
        name: &str,
    ) -> Result<PointerValue<'ctx>, ()> {
        let pointee_ty = pointee_ty.as_any_type_enum();
        if !pointee_ty.is_struct_type() {
            return Err(());
        }
        let struct_ty = pointee_ty.into_struct_type();
        if index >= struct_ty.count_fields() {
            return Err(());
        }
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildStructGEP2(
                self.builder,
                pointee_ty.as_type_ref(),
                ptr.as_value_ref(),
                index,
                c_string.as_ptr(),
            )
        };
        unsafe { Ok(PointerValue::new(value)) }
    }
    pub fn build_ptr_diff<T: BasicType<'ctx>>(
        &self,
        pointee_ty: T,
        lhs_ptr: PointerValue<'ctx>,
        rhs_ptr: PointerValue<'ctx>,
        name: &str,
    ) -> IntValue<'ctx> {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildPtrDiff2(
                self.builder,
                pointee_ty.as_type_ref(),
                lhs_ptr.as_value_ref(),
                rhs_ptr.as_value_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { IntValue::new(value) }
    }
    pub fn build_phi<T: BasicType<'ctx>>(&self, type_: T, name: &str) -> PhiValue<'ctx> {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildPhi(self.builder, type_.as_type_ref(), c_string.as_ptr()) };
        unsafe { PhiValue::new(value) }
    }
    pub fn build_store<V: BasicValue<'ctx>>(&self, ptr: PointerValue<'ctx>, value: V) -> InstructionValue<'ctx> {
        let value = unsafe { LLVMBuildStore(self.builder, value.as_value_ref(), ptr.as_value_ref()) };
        unsafe { InstructionValue::new(value) }
    }
    pub fn build_load<T: BasicType<'ctx>>(
        &self,
        pointee_ty: T,
        ptr: PointerValue<'ctx>,
        name: &str,
    ) -> BasicValueEnum<'ctx> {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildLoad2(
                self.builder,
                pointee_ty.as_type_ref(),
                ptr.as_value_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { BasicValueEnum::new(value) }
    }
    pub fn build_alloca<T: BasicType<'ctx>>(&self, ty: T, name: &str) -> PointerValue<'ctx> {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildAlloca(self.builder, ty.as_type_ref(), c_string.as_ptr()) };
        unsafe { PointerValue::new(value) }
    }
    pub fn build_array_alloca<T: BasicType<'ctx>>(
        &self,
        ty: T,
        size: IntValue<'ctx>,
        name: &str,
    ) -> PointerValue<'ctx> {
        let c_string = to_c_str(name);
        let value =
            unsafe { LLVMBuildArrayAlloca(self.builder, ty.as_type_ref(), size.as_value_ref(), c_string.as_ptr()) };
        unsafe { PointerValue::new(value) }
    }
    pub fn build_memcpy(
        &self,
        dest: PointerValue<'ctx>,
        dest_align_bytes: u32,
        src: PointerValue<'ctx>,
        src_align_bytes: u32,
        size: IntValue<'ctx>,
    ) -> Result<PointerValue<'ctx>, &'static str> {
        if !is_alignment_ok(src_align_bytes) {
            return Err("The src_align_bytes argument to build_memcpy was not a power of 2.");
        }
        if !is_alignment_ok(dest_align_bytes) {
            return Err("The dest_align_bytes argument to build_memcpy was not a power of 2.");
        }
        let value = unsafe {
            LLVMBuildMemCpy(
                self.builder,
                dest.as_value_ref(),
                dest_align_bytes,
                src.as_value_ref(),
                src_align_bytes,
                size.as_value_ref(),
            )
        };
        unsafe { Ok(PointerValue::new(value)) }
    }
    pub fn build_memmove(
        &self,
        dest: PointerValue<'ctx>,
        dest_align_bytes: u32,
        src: PointerValue<'ctx>,
        src_align_bytes: u32,
        size: IntValue<'ctx>,
    ) -> Result<PointerValue<'ctx>, &'static str> {
        if !is_alignment_ok(src_align_bytes) {
            return Err("The src_align_bytes argument to build_memmove was not a power of 2 under 2^64.");
        }
        if !is_alignment_ok(dest_align_bytes) {
            return Err("The dest_align_bytes argument to build_memmove was not a power of 2 under 2^64.");
        }
        let value = unsafe {
            LLVMBuildMemMove(
                self.builder,
                dest.as_value_ref(),
                dest_align_bytes,
                src.as_value_ref(),
                src_align_bytes,
                size.as_value_ref(),
            )
        };
        unsafe { Ok(PointerValue::new(value)) }
    }
    pub fn build_memset(
        &self,
        dest: PointerValue<'ctx>,
        dest_align_bytes: u32,
        val: IntValue<'ctx>,
        size: IntValue<'ctx>,
    ) -> Result<PointerValue<'ctx>, &'static str> {
        if !is_alignment_ok(dest_align_bytes) {
            return Err("The src_align_bytes argument to build_memmove was not a power of 2 under 2^64.");
        }
        let value = unsafe {
            LLVMBuildMemSet(
                self.builder,
                dest.as_value_ref(),
                val.as_value_ref(),
                size.as_value_ref(),
                dest_align_bytes,
            )
        };
        unsafe { Ok(PointerValue::new(value)) }
    }
    pub fn build_malloc<T: BasicType<'ctx>>(&self, ty: T, name: &str) -> Result<PointerValue<'ctx>, &'static str> {
        if !ty.is_sized() {
            return Err("Cannot build malloc call for an unsized type");
        }
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildMalloc(self.builder, ty.as_type_ref(), c_string.as_ptr()) };
        unsafe { Ok(PointerValue::new(value)) }
    }
    pub fn build_array_malloc<T: BasicType<'ctx>>(
        &self,
        ty: T,
        size: IntValue<'ctx>,
        name: &str,
    ) -> Result<PointerValue<'ctx>, &'static str> {
        if !ty.is_sized() {
            return Err("Cannot build array malloc call for an unsized type");
        }
        let c_string = to_c_str(name);
        let value =
            unsafe { LLVMBuildArrayMalloc(self.builder, ty.as_type_ref(), size.as_value_ref(), c_string.as_ptr()) };
        unsafe { Ok(PointerValue::new(value)) }
    }
    pub fn build_free(&self, ptr: PointerValue<'ctx>) -> InstructionValue<'ctx> {
        unsafe { InstructionValue::new(LLVMBuildFree(self.builder, ptr.as_value_ref())) }
    }
    pub fn insert_instruction(&self, instruction: &InstructionValue<'ctx>, name: Option<&str>) {
        match name {
            Some(name) => {
                let c_string = to_c_str(name);
                unsafe { LLVMInsertIntoBuilderWithName(self.builder, instruction.as_value_ref(), c_string.as_ptr()) }
            },
            None => unsafe {
                LLVMInsertIntoBuilder(self.builder, instruction.as_value_ref());
            },
        }
    }
    pub fn get_insert_block(&self) -> Option<BasicBlock<'ctx>> {
        unsafe { BasicBlock::new(LLVMGetInsertBlock(self.builder)) }
    }
    pub fn build_int_unsigned_div<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildUDiv(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_signed_div<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildSDiv(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_exact_signed_div<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value =
            unsafe { LLVMBuildExactSDiv(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_unsigned_rem<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildURem(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_signed_rem<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildSRem(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_s_extend<T: IntMathValue<'ctx>>(&self, int_value: T, int_type: T::BaseType, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildSExt(
                self.builder,
                int_value.as_value_ref(),
                int_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { T::new(value) }
    }
    pub fn build_address_space_cast(
        &self,
        ptr_val: PointerValue<'ctx>,
        ptr_type: PointerType<'ctx>,
        name: &str,
    ) -> PointerValue<'ctx> {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildAddrSpaceCast(
                self.builder,
                ptr_val.as_value_ref(),
                ptr_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { PointerValue::new(value) }
    }
    pub fn build_bitcast<T, V>(&self, val: V, ty: T, name: &str) -> BasicValueEnum<'ctx>
    where
        T: BasicType<'ctx>,
        V: BasicValue<'ctx>,
    {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildBitCast(self.builder, val.as_value_ref(), ty.as_type_ref(), c_string.as_ptr()) };
        unsafe { BasicValueEnum::new(value) }
    }
    pub fn build_int_s_extend_or_bit_cast<T: IntMathValue<'ctx>>(
        &self,
        int_value: T,
        int_type: T::BaseType,
        name: &str,
    ) -> T {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildSExtOrBitCast(
                self.builder,
                int_value.as_value_ref(),
                int_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { T::new(value) }
    }
    pub fn build_int_z_extend<T: IntMathValue<'ctx>>(&self, int_value: T, int_type: T::BaseType, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildZExt(
                self.builder,
                int_value.as_value_ref(),
                int_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { T::new(value) }
    }
    pub fn build_int_z_extend_or_bit_cast<T: IntMathValue<'ctx>>(
        &self,
        int_value: T,
        int_type: T::BaseType,
        name: &str,
    ) -> T {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildZExtOrBitCast(
                self.builder,
                int_value.as_value_ref(),
                int_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { T::new(value) }
    }
    pub fn build_int_truncate<T: IntMathValue<'ctx>>(&self, int_value: T, int_type: T::BaseType, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildTrunc(
                self.builder,
                int_value.as_value_ref(),
                int_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { T::new(value) }
    }
    pub fn build_int_truncate_or_bit_cast<T: IntMathValue<'ctx>>(
        &self,
        int_value: T,
        int_type: T::BaseType,
        name: &str,
    ) -> T {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildTruncOrBitCast(
                self.builder,
                int_value.as_value_ref(),
                int_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { T::new(value) }
    }
    pub fn build_float_rem<T: FloatMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildFRem(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_float_to_unsigned_int<T: FloatMathValue<'ctx>>(
        &self,
        float: T,
        int_type: <T::BaseType as FloatMathType<'ctx>>::MathConvType,
        name: &str,
    ) -> <<T::BaseType as FloatMathType<'ctx>>::MathConvType as IntMathType<'ctx>>::ValueType {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildFPToUI(
                self.builder,
                float.as_value_ref(),
                int_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { <<T::BaseType as FloatMathType>::MathConvType as IntMathType>::ValueType::new(value) }
    }
    pub fn build_float_to_signed_int<T: FloatMathValue<'ctx>>(
        &self,
        float: T,
        int_type: <T::BaseType as FloatMathType<'ctx>>::MathConvType,
        name: &str,
    ) -> <<T::BaseType as FloatMathType<'ctx>>::MathConvType as IntMathType<'ctx>>::ValueType {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildFPToSI(
                self.builder,
                float.as_value_ref(),
                int_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { <<T::BaseType as FloatMathType>::MathConvType as IntMathType>::ValueType::new(value) }
    }
    pub fn build_unsigned_int_to_float<T: IntMathValue<'ctx>>(
        &self,
        int: T,
        float_type: <T::BaseType as IntMathType<'ctx>>::MathConvType,
        name: &str,
    ) -> <<T::BaseType as IntMathType<'ctx>>::MathConvType as FloatMathType<'ctx>>::ValueType {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildUIToFP(
                self.builder,
                int.as_value_ref(),
                float_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { <<T::BaseType as IntMathType>::MathConvType as FloatMathType>::ValueType::new(value) }
    }
    pub fn build_signed_int_to_float<T: IntMathValue<'ctx>>(
        &self,
        int: T,
        float_type: <T::BaseType as IntMathType<'ctx>>::MathConvType,
        name: &str,
    ) -> <<T::BaseType as IntMathType<'ctx>>::MathConvType as FloatMathType<'ctx>>::ValueType {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildSIToFP(
                self.builder,
                int.as_value_ref(),
                float_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { <<T::BaseType as IntMathType>::MathConvType as FloatMathType>::ValueType::new(value) }
    }
    pub fn build_float_trunc<T: FloatMathValue<'ctx>>(&self, float: T, float_type: T::BaseType, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildFPTrunc(
                self.builder,
                float.as_value_ref(),
                float_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { T::new(value) }
    }
    pub fn build_float_ext<T: FloatMathValue<'ctx>>(&self, float: T, float_type: T::BaseType, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildFPExt(
                self.builder,
                float.as_value_ref(),
                float_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { T::new(value) }
    }
    pub fn build_float_cast<T: FloatMathValue<'ctx>>(&self, float: T, float_type: T::BaseType, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildFPCast(
                self.builder,
                float.as_value_ref(),
                float_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { T::new(value) }
    }
    pub fn build_int_cast<T: IntMathValue<'ctx>>(&self, int: T, int_type: T::BaseType, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildIntCast(
                self.builder,
                int.as_value_ref(),
                int_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { T::new(value) }
    }
    pub fn build_int_cast_sign_flag<T: IntMathValue<'ctx>>(
        &self,
        int: T,
        int_type: T::BaseType,
        is_signed: bool,
        name: &str,
    ) -> T {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildIntCast2(
                self.builder,
                int.as_value_ref(),
                int_type.as_type_ref(),
                is_signed.into(),
                c_string.as_ptr(),
            )
        };
        unsafe { T::new(value) }
    }
    pub fn build_float_div<T: FloatMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildFDiv(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_add<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildAdd(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_nsw_add<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildNSWAdd(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_nuw_add<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildNUWAdd(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_float_add<T: FloatMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildFAdd(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_xor<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildXor(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_and<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildAnd(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_or<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildOr(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_left_shift<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildShl(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_right_shift<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, sign_extend: bool, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe {
            if sign_extend {
                LLVMBuildAShr(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr())
            } else {
                LLVMBuildLShr(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr())
            }
        };
        unsafe { T::new(value) }
    }
    pub fn build_int_sub<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildSub(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_nsw_sub<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildNSWSub(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_nuw_sub<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildNUWSub(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_float_sub<T: FloatMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildFSub(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_mul<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildMul(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_nsw_mul<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildNSWMul(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_nuw_mul<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildNUWMul(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_float_mul<T: FloatMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildFMul(self.builder, lhs.as_value_ref(), rhs.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_cast<T: BasicType<'ctx>, V: BasicValue<'ctx>>(
        &self,
        op: InstructionOpcode,
        from_value: V,
        to_type: T,
        name: &str,
    ) -> BasicValueEnum<'ctx> {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildCast(
                self.builder,
                op.into(),
                from_value.as_value_ref(),
                to_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { BasicValueEnum::new(value) }
    }
    pub fn build_pointer_cast<T: PointerMathValue<'ctx>>(&self, from: T, to: T::BaseType, name: &str) -> T {
        let c_string = to_c_str(name);
        let value =
            unsafe { LLVMBuildPointerCast(self.builder, from.as_value_ref(), to.as_type_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_compare<T: IntMathValue<'ctx>>(
        &self,
        op: IntPredicate,
        lhs: T,
        rhs: T,
        name: &str,
    ) -> <T::BaseType as IntMathType<'ctx>>::ValueType {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildICmp(
                self.builder,
                op.into(),
                lhs.as_value_ref(),
                rhs.as_value_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { <T::BaseType as IntMathType<'ctx>>::ValueType::new(value) }
    }
    pub fn build_float_compare<T: FloatMathValue<'ctx>>(
        &self,
        op: FloatPredicate,
        lhs: T,
        rhs: T,
        name: &str,
    ) -> <<T::BaseType as FloatMathType<'ctx>>::MathConvType as IntMathType<'ctx>>::ValueType {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildFCmp(
                self.builder,
                op.into(),
                lhs.as_value_ref(),
                rhs.as_value_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { <<T::BaseType as FloatMathType>::MathConvType as IntMathType>::ValueType::new(value) }
    }
    pub fn build_unconditional_branch(&self, destination_block: BasicBlock<'ctx>) -> InstructionValue<'ctx> {
        let value = unsafe { LLVMBuildBr(self.builder, destination_block.basic_block) };
        unsafe { InstructionValue::new(value) }
    }
    pub fn build_conditional_branch(
        &self,
        comparison: IntValue<'ctx>,
        then_block: BasicBlock<'ctx>,
        else_block: BasicBlock<'ctx>,
    ) -> InstructionValue<'ctx> {
        let value = unsafe {
            LLVMBuildCondBr(
                self.builder,
                comparison.as_value_ref(),
                then_block.basic_block,
                else_block.basic_block,
            )
        };
        unsafe { InstructionValue::new(value) }
    }
    pub fn build_indirect_branch<BV: BasicValue<'ctx>>(
        &self,
        address: BV,
        destinations: &[BasicBlock<'ctx>],
    ) -> InstructionValue<'ctx> {
        let value = unsafe { LLVMBuildIndirectBr(self.builder, address.as_value_ref(), destinations.len() as u32) };
        for destination in destinations {
            unsafe { LLVMAddDestination(value, destination.basic_block) }
        }
        unsafe { InstructionValue::new(value) }
    }
    pub fn build_int_neg<T: IntMathValue<'ctx>>(&self, value: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildNeg(self.builder, value.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_nsw_neg<T: IntMathValue<'ctx>>(&self, value: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildNSWNeg(self.builder, value.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_int_nuw_neg<T: IntMathValue<'ctx>>(&self, value: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildNUWNeg(self.builder, value.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_float_neg<T: FloatMathValue<'ctx>>(&self, value: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildFNeg(self.builder, value.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn build_not<T: IntMathValue<'ctx>>(&self, value: T, name: &str) -> T {
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildNot(self.builder, value.as_value_ref(), c_string.as_ptr()) };
        unsafe { T::new(value) }
    }
    pub fn position_at(&self, basic_block: BasicBlock<'ctx>, instruction: &InstructionValue<'ctx>) {
        unsafe { LLVMPositionBuilder(self.builder, basic_block.basic_block, instruction.as_value_ref()) }
    }
    pub fn position_before(&self, instruction: &InstructionValue<'ctx>) {
        unsafe { LLVMPositionBuilderBefore(self.builder, instruction.as_value_ref()) }
    }
    pub fn position_at_end(&self, basic_block: BasicBlock<'ctx>) {
        unsafe {
            LLVMPositionBuilderAtEnd(self.builder, basic_block.basic_block);
        }
    }
    pub fn build_extract_value<AV: AggregateValue<'ctx>>(
        &self,
        agg: AV,
        index: u32,
        name: &str,
    ) -> Option<BasicValueEnum<'ctx>> {
        let size = match agg.as_aggregate_value_enum() {
            AggregateValueEnum::ArrayValue(av) => av.get_type().len(),
            AggregateValueEnum::StructValue(sv) => sv.get_type().count_fields(),
        };
        if index >= size {
            return None;
        }
        let c_string = to_c_str(name);
        let value = unsafe { LLVMBuildExtractValue(self.builder, agg.as_value_ref(), index, c_string.as_ptr()) };
        unsafe { Some(BasicValueEnum::new(value)) }
    }
    pub fn build_insert_value<AV, BV>(
        &self,
        agg: AV,
        value: BV,
        index: u32,
        name: &str,
    ) -> Option<AggregateValueEnum<'ctx>>
    where
        AV: AggregateValue<'ctx>,
        BV: BasicValue<'ctx>,
    {
        let size = match agg.as_aggregate_value_enum() {
            AggregateValueEnum::ArrayValue(av) => av.get_type().len(),
            AggregateValueEnum::StructValue(sv) => sv.get_type().count_fields(),
        };
        if index >= size {
            return None;
        }
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildInsertValue(
                self.builder,
                agg.as_value_ref(),
                value.as_value_ref(),
                index,
                c_string.as_ptr(),
            )
        };
        unsafe { Some(AggregateValueEnum::new(value)) }
    }
    pub fn build_extract_element(
        &self,
        vector: VectorValue<'ctx>,
        index: IntValue<'ctx>,
        name: &str,
    ) -> BasicValueEnum<'ctx> {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildExtractElement(
                self.builder,
                vector.as_value_ref(),
                index.as_value_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { BasicValueEnum::new(value) }
    }
    pub fn build_insert_element<V: BasicValue<'ctx>>(
        &self,
        vector: VectorValue<'ctx>,
        element: V,
        index: IntValue<'ctx>,
        name: &str,
    ) -> VectorValue<'ctx> {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildInsertElement(
                self.builder,
                vector.as_value_ref(),
                element.as_value_ref(),
                index.as_value_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { VectorValue::new(value) }
    }
    pub fn build_unreachable(&self) -> InstructionValue<'ctx> {
        let val = unsafe { LLVMBuildUnreachable(self.builder) };
        unsafe { InstructionValue::new(val) }
    }
    pub fn build_fence(&self, atomic_ordering: AtomicOrdering, num: i32, name: &str) -> InstructionValue<'ctx> {
        let c_string = to_c_str(name);
        let val = unsafe { LLVMBuildFence(self.builder, atomic_ordering.into(), num, c_string.as_ptr()) };
        unsafe { InstructionValue::new(val) }
    }
    pub fn build_is_null<T: PointerMathValue<'ctx>>(
        &self,
        ptr: T,
        name: &str,
    ) -> <<T::BaseType as PointerMathType<'ctx>>::PtrConvType as IntMathType<'ctx>>::ValueType {
        let c_string = to_c_str(name);
        let val = unsafe { LLVMBuildIsNull(self.builder, ptr.as_value_ref(), c_string.as_ptr()) };
        unsafe { <<T::BaseType as PointerMathType>::PtrConvType as IntMathType>::ValueType::new(val) }
    }
    pub fn build_is_not_null<T: PointerMathValue<'ctx>>(
        &self,
        ptr: T,
        name: &str,
    ) -> <<T::BaseType as PointerMathType<'ctx>>::PtrConvType as IntMathType<'ctx>>::ValueType {
        let c_string = to_c_str(name);
        let val = unsafe { LLVMBuildIsNotNull(self.builder, ptr.as_value_ref(), c_string.as_ptr()) };
        unsafe { <<T::BaseType as PointerMathType>::PtrConvType as IntMathType>::ValueType::new(val) }
    }
    pub fn build_int_to_ptr<T: IntMathValue<'ctx>>(
        &self,
        int: T,
        ptr_type: <T::BaseType as IntMathType<'ctx>>::PtrConvType,
        name: &str,
    ) -> <<T::BaseType as IntMathType<'ctx>>::PtrConvType as PointerMathType<'ctx>>::ValueType {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildIntToPtr(
                self.builder,
                int.as_value_ref(),
                ptr_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { <<T::BaseType as IntMathType>::PtrConvType as PointerMathType>::ValueType::new(value) }
    }
    pub fn build_ptr_to_int<T: PointerMathValue<'ctx>>(
        &self,
        ptr: T,
        int_type: <T::BaseType as PointerMathType<'ctx>>::PtrConvType,
        name: &str,
    ) -> <<T::BaseType as PointerMathType<'ctx>>::PtrConvType as IntMathType<'ctx>>::ValueType {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildPtrToInt(
                self.builder,
                ptr.as_value_ref(),
                int_type.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { <<T::BaseType as PointerMathType>::PtrConvType as IntMathType>::ValueType::new(value) }
    }
    pub fn clear_insertion_position(&self) {
        unsafe { LLVMClearInsertionPosition(self.builder) }
    }
    pub fn build_switch(
        &self,
        value: IntValue<'ctx>,
        else_block: BasicBlock<'ctx>,
        cases: &[(IntValue<'ctx>, BasicBlock<'ctx>)],
    ) -> InstructionValue<'ctx> {
        let switch_value = unsafe {
            LLVMBuildSwitch(
                self.builder,
                value.as_value_ref(),
                else_block.basic_block,
                cases.len() as u32,
            )
        };
        for &(value, basic_block) in cases {
            unsafe { LLVMAddCase(switch_value, value.as_value_ref(), basic_block.basic_block) }
        }
        unsafe { InstructionValue::new(switch_value) }
    }
    pub fn build_select<BV: BasicValue<'ctx>, IMV: IntMathValue<'ctx>>(
        &self,
        condition: IMV,
        then: BV,
        else_: BV,
        name: &str,
    ) -> BasicValueEnum<'ctx> {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildSelect(
                self.builder,
                condition.as_value_ref(),
                then.as_value_ref(),
                else_.as_value_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { BasicValueEnum::new(value) }
    }
    pub unsafe fn build_global_string(&self, value: &str, name: &str) -> GlobalValue<'ctx> {
        let c_string_value = to_c_str(value);
        let c_string_name = to_c_str(name);
        let value = LLVMBuildGlobalString(self.builder, c_string_value.as_ptr(), c_string_name.as_ptr());
        GlobalValue::new(value)
    }
    pub fn build_global_string_ptr(&self, value: &str, name: &str) -> GlobalValue<'ctx> {
        let c_string_value = to_c_str(value);
        let c_string_name = to_c_str(name);
        let value = unsafe { LLVMBuildGlobalStringPtr(self.builder, c_string_value.as_ptr(), c_string_name.as_ptr()) };
        unsafe { GlobalValue::new(value) }
    }
    pub fn build_shuffle_vector(
        &self,
        left: VectorValue<'ctx>,
        right: VectorValue<'ctx>,
        mask: VectorValue<'ctx>,
        name: &str,
    ) -> VectorValue<'ctx> {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildShuffleVector(
                self.builder,
                left.as_value_ref(),
                right.as_value_ref(),
                mask.as_value_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { VectorValue::new(value) }
    }
    pub fn build_va_arg<BT: BasicType<'ctx>>(
        &self,
        list: PointerValue<'ctx>,
        type_: BT,
        name: &str,
    ) -> BasicValueEnum<'ctx> {
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildVAArg(
                self.builder,
                list.as_value_ref(),
                type_.as_type_ref(),
                c_string.as_ptr(),
            )
        };
        unsafe { BasicValueEnum::new(value) }
    }
    pub fn build_atomicrmw(
        &self,
        op: AtomicRMWBinOp,
        ptr: PointerValue<'ctx>,
        value: IntValue<'ctx>,
        ordering: AtomicOrdering,
    ) -> Result<IntValue<'ctx>, &'static str> {
        if value.get_type().get_bit_width() < 8 || !value.get_type().get_bit_width().is_power_of_two() {
            return Err("The bitwidth of value must be a power of 2 and greater than 8.");
        }
        let val = unsafe {
            LLVMBuildAtomicRMW(
                self.builder,
                op.into(),
                ptr.as_value_ref(),
                value.as_value_ref(),
                ordering.into(),
                false as i32,
            )
        };
        unsafe { Ok(IntValue::new(val)) }
    }
    pub fn build_cmpxchg<V: BasicValue<'ctx>>(
        &self,
        ptr: PointerValue<'ctx>,
        cmp: V,
        new: V,
        success: AtomicOrdering,
        failure: AtomicOrdering,
    ) -> Result<StructValue<'ctx>, &'static str> {
        let cmp = cmp.as_basic_value_enum();
        let new = new.as_basic_value_enum();
        if cmp.get_type() != new.get_type() {
            return Err("The value to compare against and the value to replace with must have the same type.");
        }
        if !cmp.is_int_value() && !cmp.is_pointer_value() {
            return Err("The values must have pointer or integer type.");
        }
        if success < AtomicOrdering::Monotonic || failure < AtomicOrdering::Monotonic {
            return Err("Both success and failure orderings must be Monotonic or stronger.");
        }
        if failure > success {
            return Err("The failure ordering may not be stronger than the success ordering.");
        }
        if failure == AtomicOrdering::Release || failure == AtomicOrdering::AcquireRelease {
            return Err("The failure ordering may not be release or acquire release.");
        }
        let val = unsafe {
            LLVMBuildAtomicCmpXchg(
                self.builder,
                ptr.as_value_ref(),
                cmp.as_value_ref(),
                new.as_value_ref(),
                success.into(),
                failure.into(),
                false as i32,
            )
        };
        unsafe { Ok(StructValue::new(val)) }
    }
    pub fn set_current_debug_location(&self, location: DILocation<'ctx>) {
        use llvm_lib::core::LLVMSetCurrentDebugLocation2;
        unsafe {
            LLVMSetCurrentDebugLocation2(self.builder, location.metadata_ref);
        }
    }
    pub fn get_current_debug_location(&self) -> Option<DILocation<'ctx>> {
        use llvm_lib::core::LLVMGetCurrentDebugLocation;
        use llvm_lib::core::LLVMValueAsMetadata;
        let metadata_ref = unsafe { LLVMGetCurrentDebugLocation(self.builder) };
        if metadata_ref.is_null() {
            return None;
        }
        Some(DILocation {
            metadata_ref: unsafe { LLVMValueAsMetadata(metadata_ref) },
            _marker: PhantomData,
        })
    }
    pub fn unset_current_debug_location(&self) {
        use llvm_lib::core::LLVMSetCurrentDebugLocation2;
        unsafe {
            LLVMSetCurrentDebugLocation2(self.builder, std::ptr::null_mut());
        }
    }
}
impl Drop for Builder<'_> {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeBuilder(self.builder);
        }
    }
}

fn is_alignment_ok(align: u32) -> bool {
    align > 0 && align.is_power_of_two() && (align as f64).log2() < 64.0
}
