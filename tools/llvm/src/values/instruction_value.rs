use either::{
    Either,
    Either::{Left, Right},
};
use llvm_lib::core::{
    LLVMGetAlignment, LLVMGetFCmpPredicate, LLVMGetICmpPredicate, LLVMGetInstructionOpcode, LLVMGetInstructionParent,
    LLVMGetMetadata, LLVMGetNextInstruction, LLVMGetNumOperands, LLVMGetOperand, LLVMGetOperandUse,
    LLVMGetPreviousInstruction, LLVMGetVolatile, LLVMHasMetadata, LLVMInstructionClone, LLVMInstructionEraseFromParent,
    LLVMInstructionRemoveFromParent, LLVMIsAAllocaInst, LLVMIsABasicBlock, LLVMIsALoadInst, LLVMIsAStoreInst,
    LLVMIsTailCall, LLVMSetAlignment, LLVMSetMetadata, LLVMSetOperand, LLVMSetVolatile, LLVMValueAsBasicBlock,
};
use llvm_lib::core::{LLVMGetOrdering, LLVMSetOrdering};
#[llvm_versions(10.0..=latest)]
use llvm_lib::core::{LLVMIsAAtomicCmpXchgInst, LLVMIsAAtomicRMWInst};
use llvm_lib::prelude::LLVMValueRef;
use llvm_lib::LLVMOpcode;

use std::{ffi::CStr, fmt, fmt::Display};

use crate::values::traits::AsValueRef;
use crate::values::{BasicValue, BasicValueEnum, BasicValueUse, MetadataValue, Value};
use crate::{basic_block::BasicBlock, types::AnyTypeEnum};
use crate::{AtomicOrdering, FloatPredicate, IntPredicate};

use super::AnyValue;

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
    #[llvm_versions(9.0..=latest)]
    CallBr,
    CatchPad,
    CatchRet,
    CatchSwitch,
    CleanupPad,
    CleanupRet,
    ExtractElement,
    ExtractValue,
    #[llvm_versions(8.0..=latest)]
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
    #[llvm_versions(10.0..=latest)]
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
    instruction_value: Value<'ctx>,
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
    #[llvm_versions(10.0..=latest)]
    fn is_a_atomicrmw_inst(self) -> bool {
        !unsafe { LLVMIsAAtomicRMWInst(self.as_value_ref()) }.is_null()
    }
    #[llvm_versions(10.0..=latest)]
    fn is_a_cmpxchg_inst(self) -> bool {
        !unsafe { LLVMIsAAtomicCmpXchgInst(self.as_value_ref()) }.is_null()
    }

    pub(crate) unsafe fn new(instruction_value: LLVMValueRef) -> Self {
        debug_assert!(!instruction_value.is_null());

        let value = Value::new(instruction_value);

        debug_assert!(value.is_instruction());

        InstructionValue {
            instruction_value: value,
        }
    }

    pub fn get_name(&self) -> Option<&CStr> {
        if self.get_type().is_void_type() {
            None
        } else {
            Some(self.instruction_value.get_name())
        }
    }

    pub fn get_instruction_with_name(&self, name: &str) -> Option<InstructionValue<'ctx>> {
        if let Some(ins_name) = self.get_name() {
            if ins_name.to_str() == Ok(name) {
                return Some(*self);
            }
        }
        return self.get_next_instruction()?.get_instruction_with_name(name);
    }

    pub fn set_name(&self, name: &str) -> Result<(), &'static str> {
        if self.get_type().is_void_type() {
            Err("Cannot set name of a void-type instruction!")
        } else {
            self.instruction_value.set_name(name);
            Ok(())
        }
    }

    pub fn get_type(self) -> AnyTypeEnum<'ctx> {
        unsafe { AnyTypeEnum::new(self.instruction_value.get_type()) }
    }

    pub fn get_opcode(self) -> InstructionOpcode {
        let opcode = unsafe { LLVMGetInstructionOpcode(self.as_value_ref()) };

        InstructionOpcode::new(opcode)
    }

    pub fn get_previous_instruction(self) -> Option<Self> {
        let value = unsafe { LLVMGetPreviousInstruction(self.as_value_ref()) };

        if value.is_null() {
            return None;
        }

        unsafe { Some(InstructionValue::new(value)) }
    }

    pub fn get_next_instruction(self) -> Option<Self> {
        let value = unsafe { LLVMGetNextInstruction(self.as_value_ref()) };

        if value.is_null() {
            return None;
        }

        unsafe { Some(InstructionValue::new(value)) }
    }

    pub fn erase_from_basic_block(self) {
        unsafe { LLVMInstructionEraseFromParent(self.as_value_ref()) }
    }

    #[llvm_versions(4.0..=latest)]
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

    pub fn replace_all_uses_with(self, other: &InstructionValue<'ctx>) {
        self.instruction_value.replace_all_uses_with(other.as_value_ref())
    }

    #[llvm_versions(4.0..=9.0)]
    pub fn get_volatile(self) -> Result<bool, &'static str> {
        if !self.is_a_load_inst() && !self.is_a_store_inst() {
            return Err("Value is not a load or store.");
        }
        Ok(unsafe { LLVMGetVolatile(self.as_value_ref()) } == 1)
    }

    #[llvm_versions(10.0..=latest)]
    pub fn get_volatile(self) -> Result<bool, &'static str> {
        if !self.is_a_load_inst() && !self.is_a_store_inst() && !self.is_a_atomicrmw_inst() && !self.is_a_cmpxchg_inst()
        {
            return Err("Value is not a load, store, atomicrmw or cmpxchg.");
        }
        Ok(unsafe { LLVMGetVolatile(self.as_value_ref()) } == 1)
    }

    #[llvm_versions(4.0..=9.0)]
    pub fn set_volatile(self, volatile: bool) -> Result<(), &'static str> {
        if !self.is_a_load_inst() && !self.is_a_store_inst() {
            return Err("Value is not a load or store.");
        }
        Ok(unsafe { LLVMSetVolatile(self.as_value_ref(), volatile as i32) })
    }

    #[llvm_versions(10.0..=latest)]
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
        #[cfg(any(feature = "llvm11-0", feature = "llvm12-0"))]
        {
            if alignment == 0 {
                return Err("Alignment cannot be 0");
            }
        }
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
        self.instruction_value.get_first_use()
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
        unsafe { LLVMHasMetadata(self.instruction_value.value) == 1 }
    }

    pub fn get_metadata(self, kind_id: u32) -> Option<MetadataValue<'ctx>> {
        let metadata_value = unsafe { LLVMGetMetadata(self.instruction_value.value, kind_id) };

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
            LLVMSetMetadata(self.instruction_value.value, kind_id, metadata.as_value_ref());
        }

        Ok(())
    }
}

impl Clone for InstructionValue<'_> {
    fn clone(&self) -> Self {
        unsafe { InstructionValue::new(LLVMInstructionClone(self.as_value_ref())) }
    }
}

unsafe impl AsValueRef for InstructionValue<'_> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.instruction_value.value
    }
}

impl Display for InstructionValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_to_string())
    }
}
