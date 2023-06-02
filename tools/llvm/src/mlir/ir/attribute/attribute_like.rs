use crate::{
    ir::{r#type, Type},
    ContextRef,
};
use melior_macro::attribute_check_functions;
use mlir_sys::{
    mlirAttributeDump, mlirAttributeGetContext, mlirAttributeGetType, mlirAttributeGetTypeID, MlirAttribute,
};

pub trait AttributeLike<'c> {
    fn to_raw(&self) -> MlirAttribute;

    fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirAttributeGetContext(self.to_raw())) }
    }

    fn r#type(&self) -> Type {
        unsafe { Type::from_raw(mlirAttributeGetType(self.to_raw())) }
    }

    fn type_id(&self) -> r#type::TypeId {
        unsafe { r#type::TypeId::from_raw(mlirAttributeGetTypeID(self.to_raw())) }
    }

    fn dump(&self) {
        unsafe { mlirAttributeDump(self.to_raw()) }
    }

    attribute_check_functions!(
        mlirAttributeIsAAffineMap,
        mlirAttributeIsAArray,
        mlirAttributeIsABool,
        mlirAttributeIsADenseBoolArray,
        mlirAttributeIsADenseElements,
        mlirAttributeIsADenseF32Array,
        mlirAttributeIsADenseF64Array,
        mlirAttributeIsADenseFPElements,
        mlirAttributeIsADenseI16Array,
        mlirAttributeIsADenseI32Array,
        mlirAttributeIsADenseI64Array,
        mlirAttributeIsADenseI8Array,
        mlirAttributeIsADenseIntElements,
        mlirAttributeIsADictionary,
        mlirAttributeIsAElements,
        mlirAttributeIsAFlatSymbolRef,
        mlirAttributeIsAFloat,
        mlirAttributeIsAInteger,
        mlirAttributeIsAIntegerSet,
        mlirAttributeIsAOpaque,
        mlirAttributeIsASparseElements,
        mlirAttributeIsASparseTensorEncodingAttr,
        mlirAttributeIsAStridedLayout,
        mlirAttributeIsAString,
        mlirAttributeIsASymbolRef,
        mlirAttributeIsAType,
        mlirAttributeIsAUnit,
    );
}
