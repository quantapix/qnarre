use llvm_lib::core::{LLVMGetIntrinsicDeclaration, LLVMIntrinsicIsOverloaded, LLVMLookupIntrinsicID};
use llvm_lib::prelude::LLVMTypeRef;

use crate::module::Module;
use crate::typ::{AsTypeRef, BasicTypeEnum};
use crate::val::FunctionValue;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Intrinsic {
    id: u32,
}

///
impl Intrinsic {
    pub(crate) unsafe fn new(id: u32) -> Self {
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
