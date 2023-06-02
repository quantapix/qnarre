use super::{Attribute, AttributeLike};
use crate::{
    ir::{Type, TypeLike},
    Error,
};
use mlir_sys::{mlirTypeAttrGet, mlirTypeAttrGetValue, MlirAttribute};

#[derive(Clone, Copy)]
pub struct TypeAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> TypeAttribute<'c> {
    pub fn new(r#type: Type<'c>) -> Self {
        unsafe { Self::from_raw(mlirTypeAttrGet(r#type.to_raw())) }
    }

    pub fn value(&self) -> Type<'c> {
        unsafe { Type::from_raw(mlirTypeAttrGetValue(self.to_raw())) }
    }
}

attribute_traits!(TypeAttribute, is_type, "type");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn value() {
        let context = Context::new();
        let r#type = Type::index(&context);

        assert_eq!(TypeAttribute::new(r#type).value(), r#type);
    }
}
