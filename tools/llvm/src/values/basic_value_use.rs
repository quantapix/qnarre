use either::{
    Either,
    Either::{Left, Right},
};
use llvm_lib::core::{LLVMGetNextUse, LLVMGetUsedValue, LLVMGetUser, LLVMIsABasicBlock, LLVMValueAsBasicBlock};
use llvm_lib::prelude::LLVMUseRef;

use std::marker::PhantomData;

use crate::basic_block::BasicBlock;
use crate::values::{AnyValueEnum, BasicValueEnum};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BasicValueUse<'ctx>(LLVMUseRef, PhantomData<&'ctx ()>);

impl<'ctx> BasicValueUse<'ctx> {
    pub(crate) unsafe fn new(use_: LLVMUseRef) -> Self {
        debug_assert!(!use_.is_null());

        BasicValueUse(use_, PhantomData)
    }

    pub fn get_next_use(self) -> Option<Self> {
        let use_ = unsafe { LLVMGetNextUse(self.0) };

        if use_.is_null() {
            return None;
        }

        unsafe { Some(Self::new(use_)) }
    }

    pub fn get_user(self) -> AnyValueEnum<'ctx> {
        unsafe { AnyValueEnum::new(LLVMGetUser(self.0)) }
    }

    pub fn get_used_value(self) -> Either<BasicValueEnum<'ctx>, BasicBlock<'ctx>> {
        let used_value = unsafe { LLVMGetUsedValue(self.0) };

        let is_basic_block = unsafe { !LLVMIsABasicBlock(used_value).is_null() };

        if is_basic_block {
            let bb = unsafe { BasicBlock::new(LLVMValueAsBasicBlock(used_value)) };

            Right(bb.expect("BasicBlock should always be valid"))
        } else {
            unsafe { Left(BasicValueEnum::new(used_value)) }
        }
    }
}
