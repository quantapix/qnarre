use super::TypeLike;
use crate::{ir::Type, utility::into_raw_array, Context, Error};
use mlir_sys::{
    mlirFunctionTypeGet, mlirFunctionTypeGetInput, mlirFunctionTypeGetNumInputs,
    mlirFunctionTypeGetNumResults, mlirFunctionTypeGetResult, MlirType,
};

/// A function type.
#[derive(Clone, Copy, Debug)]
pub struct FunctionType<'c> {
    r#type: Type<'c>,
}

impl<'c> FunctionType<'c> {
    /// Creates a function type.
    pub fn new(context: &'c Context, inputs: &[Type<'c>], results: &[Type<'c>]) -> Self {
        Self {
            r#type: unsafe {
                Type::from_raw(mlirFunctionTypeGet(
                    context.to_raw(),
                    inputs.len() as isize,
                    into_raw_array(inputs.iter().map(|r#type| r#type.to_raw()).collect()),
                    results.len() as isize,
                    into_raw_array(results.iter().map(|r#type| r#type.to_raw()).collect()),
                ))
            },
        }
    }

    /// Gets an input at a position.
    pub fn input(&self, index: usize) -> Result<Type<'c>, Error> {
        if index < self.input_count() {
            unsafe {
                Ok(Type::from_raw(mlirFunctionTypeGetInput(
                    self.r#type.to_raw(),
                    index as isize,
                )))
            }
        } else {
            Err(Error::PositionOutOfBounds {
                name: "function input",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Gets a result at a position.
    pub fn result(&self, index: usize) -> Result<Type<'c>, Error> {
        if index < self.result_count() {
            unsafe {
                Ok(Type::from_raw(mlirFunctionTypeGetResult(
                    self.r#type.to_raw(),
                    index as isize,
                )))
            }
        } else {
            Err(Error::PositionOutOfBounds {
                name: "function result",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Gets a number of inputs.
    pub fn input_count(&self) -> usize {
        unsafe { mlirFunctionTypeGetNumInputs(self.r#type.to_raw()) as usize }
    }

    /// Gets a number of results.
    pub fn result_count(&self) -> usize {
        unsafe { mlirFunctionTypeGetNumResults(self.r#type.to_raw()) as usize }
    }
}

type_traits!(FunctionType, is_function, "function");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn new() {
        let context = Context::new();
        let integer = Type::index(&context);

        assert_eq!(
            Type::from(FunctionType::new(&context, &[integer, integer], &[integer])),
            Type::parse(&context, "(index, index) -> index").unwrap()
        );
    }

    #[test]
    fn multiple_results() {
        let context = Context::new();
        let integer = Type::index(&context);

        assert_eq!(
            Type::from(FunctionType::new(&context, &[], &[integer, integer])),
            Type::parse(&context, "() -> (index, index)").unwrap()
        );
    }

    #[test]
    fn input() {
        let context = Context::new();
        let integer = Type::index(&context);

        assert_eq!(
            FunctionType::new(&context, &[integer], &[]).input(0),
            Ok(integer)
        );
    }

    #[test]
    fn input_error() {
        let context = Context::new();
        let integer = Type::index(&context);
        let function = FunctionType::new(&context, &[integer], &[]);

        assert_eq!(
            function.input(42),
            Err(Error::PositionOutOfBounds {
                name: "function input",
                value: function.to_string(),
                index: 42
            })
        );
    }

    #[test]
    fn result() {
        let context = Context::new();
        let integer = Type::index(&context);

        assert_eq!(
            FunctionType::new(&context, &[], &[integer]).result(0),
            Ok(integer)
        );
    }

    #[test]
    fn result_error() {
        let context = Context::new();
        let integer = Type::index(&context);
        let function = FunctionType::new(&context, &[], &[integer]);

        assert_eq!(
            function.result(42),
            Err(Error::PositionOutOfBounds {
                name: "function result",
                value: function.to_string(),
                index: 42
            })
        );
    }

    #[test]
    fn input_count() {
        let context = Context::new();
        let integer = Type::index(&context);

        assert_eq!(
            FunctionType::new(&context, &[integer], &[]).input_count(),
            1
        );
    }

    #[test]
    fn result_count() {
        let context = Context::new();
        let integer = Type::index(&context);

        assert_eq!(
            FunctionType::new(&context, &[], &[integer]).result_count(),
            1
        );
    }
}
