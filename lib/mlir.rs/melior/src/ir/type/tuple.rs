use super::TypeLike;
use crate::{ir::Type, utility::into_raw_array, Context, Error};
use mlir_sys::{mlirTupleTypeGet, mlirTupleTypeGetNumTypes, mlirTupleTypeGetType, MlirType};

/// A tuple type.
#[derive(Clone, Copy, Debug)]
pub struct TupleType<'c> {
    r#type: Type<'c>,
}

impl<'c> TupleType<'c> {
    /// Creates a tuple type.
    pub fn new(context: &'c Context, types: &[Type<'c>]) -> Self {
        unsafe {
            Self::from_raw(mlirTupleTypeGet(
                context.to_raw(),
                types.len() as isize,
                into_raw_array(types.iter().map(|r#type| r#type.to_raw()).collect()),
            ))
        }
    }

    /// Gets a field at a position.
    pub fn r#type(&self, index: usize) -> Result<Type, Error> {
        if index < self.type_count() {
            unsafe {
                Ok(Type::from_raw(mlirTupleTypeGetType(
                    self.r#type.to_raw(),
                    index as isize,
                )))
            }
        } else {
            Err(Error::PositionOutOfBounds {
                name: "tuple field",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Gets a number of fields.
    pub fn type_count(&self) -> usize {
        unsafe { mlirTupleTypeGetNumTypes(self.r#type.to_raw()) as usize }
    }
}

type_traits!(TupleType, is_tuple, "tuple");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn new() {
        let context = Context::new();

        assert_eq!(
            Type::from(TupleType::new(&context, &[])),
            Type::parse(&context, "tuple<>").unwrap()
        );
    }

    #[test]
    fn new_with_field() {
        let context = Context::new();

        assert_eq!(
            Type::from(TupleType::new(&context, &[Type::index(&context)])),
            Type::parse(&context, "tuple<index>").unwrap()
        );
    }

    #[test]
    fn new_with_two_fields() {
        let context = Context::new();
        let r#type = Type::index(&context);

        assert_eq!(
            Type::from(TupleType::new(&context, &[r#type, r#type])),
            Type::parse(&context, "tuple<index,index>").unwrap()
        );
    }

    #[test]
    fn r#type() {
        let context = Context::new();
        let r#type = Type::index(&context);

        assert_eq!(TupleType::new(&context, &[r#type]).r#type(0), Ok(r#type));
    }

    #[test]
    fn type_error() {
        let context = Context::new();
        let tuple = TupleType::new(&context, &[]);

        assert_eq!(
            tuple.r#type(42),
            Err(Error::PositionOutOfBounds {
                name: "tuple field",
                value: tuple.to_string(),
                index: 42
            })
        );
    }

    #[test]
    fn type_count() {
        assert_eq!(TupleType::new(&Context::new(), &[]).type_count(), 0);
    }
}
