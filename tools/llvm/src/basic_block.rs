//! A `BasicBlock` is a container of instructions.

use llvm_lib::core::{
    LLVMBasicBlockAsValue, LLVMBlockAddress, LLVMDeleteBasicBlock, LLVMGetBasicBlockName, LLVMGetBasicBlockParent,
    LLVMGetBasicBlockTerminator, LLVMGetFirstInstruction, LLVMGetFirstUse, LLVMGetLastInstruction,
    LLVMGetNextBasicBlock, LLVMGetPreviousBasicBlock, LLVMGetTypeContext, LLVMIsABasicBlock, LLVMIsConstant,
    LLVMMoveBasicBlockAfter, LLVMMoveBasicBlockBefore, LLVMPrintTypeToString, LLVMPrintValueToString,
    LLVMRemoveBasicBlockFromParent, LLVMReplaceAllUsesWith, LLVMTypeOf,
};
use llvm_lib::prelude::{LLVMBasicBlockRef, LLVMValueRef};

use crate::ctx::ContextRef;
use crate::support::to_c_str;
use crate::values::{AsValueRef, BasicValueUse, FunctionValue, InstructionValue, PointerValue};

use std::ffi::CStr;
use std::fmt;
use std::marker::PhantomData;

///
///
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

        #[cfg(any(feature = "llvm4-0", feature = "llvm5-0", feature = "llvm6-0"))]
        {
            use llvm_lib::core::LLVMSetValueName;

            unsafe { LLVMSetValueName(LLVMBasicBlockAsValue(self.basic_block), c_string.as_ptr()) };
        }
        #[cfg(not(any(feature = "llvm4-0", feature = "llvm5-0", feature = "llvm6-0")))]
        {
            use llvm_lib::core::LLVMSetValueName2;

            unsafe { LLVMSetValueName2(LLVMBasicBlockAsValue(self.basic_block), c_string.as_ptr(), name.len()) };
        }
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
