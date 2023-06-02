use super::{Attribute, AttributeLike};
use crate::{
    ir::{Type, TypeLike},
    Context, Error,
};
use mlir_sys::{mlirFloatAttrDoubleGet, MlirAttribute};

#[derive(Clone, Copy)]
pub struct FloatAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> FloatAttribute<'c> {
    pub fn new(context: &'c Context, number: f64, r#type: Type<'c>) -> Self {
        unsafe { Self::from_raw(mlirFloatAttrDoubleGet(context.to_raw(), r#type.to_raw(), number)) }
    }
}

attribute_traits!(FloatAttribute, is_float, "float");
