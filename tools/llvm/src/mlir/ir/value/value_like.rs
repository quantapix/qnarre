use super::Type;
use mlir_sys::{mlirValueDump, mlirValueGetType, mlirValueIsABlockArgument, mlirValueIsAOpResult, MlirValue};

pub trait ValueLike {
    fn to_raw(&self) -> MlirValue;

    fn r#type(&self) -> Type {
        unsafe { Type::from_raw(mlirValueGetType(self.to_raw())) }
    }

    fn is_block_argument(&self) -> bool {
        unsafe { mlirValueIsABlockArgument(self.to_raw()) }
    }

    fn is_operation_result(&self) -> bool {
        unsafe { mlirValueIsAOpResult(self.to_raw()) }
    }

    fn dump(&self) {
        unsafe { mlirValueDump(self.to_raw()) }
    }
}
