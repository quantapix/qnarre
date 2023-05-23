//! Types and type IDs.

#[macro_use]
mod r#macro;
mod function;
pub mod id;
mod integer;
mod mem_ref;
mod ranked_tensor;
mod tuple;
mod type_like;

pub use self::{
    function::FunctionType, id::TypeId, integer::IntegerType, mem_ref::MemRefType,
    ranked_tensor::RankedTensorType, tuple::TupleType, type_like::TypeLike,
};
use super::Location;
use crate::{context::Context, string_ref::StringRef, utility::print_callback};
use mlir_sys::{
    mlirBF16TypeGet, mlirF16TypeGet, mlirF32TypeGet, mlirF64TypeGet, mlirIndexTypeGet,
    mlirNoneTypeGet, mlirTypeEqual, mlirTypeParseGet, mlirTypePrint, mlirVectorTypeGet,
    mlirVectorTypeGetChecked, MlirType,
};
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

/// A type.
// Types are always values but their internal storage is owned by contexts.
#[derive(Clone, Copy)]
pub struct Type<'c> {
    raw: MlirType,
    _context: PhantomData<&'c Context>,
}

impl<'c> Type<'c> {
    /// Parses a type.
    pub fn parse(context: &'c Context, source: &str) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirTypeParseGet(
                context.to_raw(),
                StringRef::from(source).to_raw(),
            ))
        }
    }

    /// Creates a bfloat16 type.
    pub fn bfloat16(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirBF16TypeGet(context.to_raw())) }
    }

    /// Creates a float16 type.
    pub fn float16(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirF16TypeGet(context.to_raw())) }
    }

    /// Creates a float32 type.
    pub fn float32(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirF32TypeGet(context.to_raw())) }
    }

    /// Creates a float64 type.
    pub fn float64(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirF64TypeGet(context.to_raw())) }
    }

    /// Creates an index type.
    pub fn index(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirIndexTypeGet(context.to_raw())) }
    }

    /// Creates a none type.
    pub fn none(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirNoneTypeGet(context.to_raw())) }
    }

    /// Creates a vector type.
    pub fn vector(dimensions: &[u64], r#type: Self) -> Self {
        unsafe {
            Self::from_raw(mlirVectorTypeGet(
                dimensions.len() as isize,
                dimensions.as_ptr() as *const i64,
                r#type.raw,
            ))
        }
    }

    /// Creates a vector type with diagnostics.
    pub fn vector_checked(
        location: Location<'c>,
        dimensions: &[u64],
        r#type: Self,
    ) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirVectorTypeGetChecked(
                location.to_raw(),
                dimensions.len() as isize,
                dimensions.as_ptr() as *const i64,
                r#type.raw,
            ))
        }
    }

    /// Creates a type from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirType) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    /// Creates an optional type from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_option_raw(raw: MlirType) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}

impl<'c> TypeLike<'c> for Type<'c> {
    fn to_raw(&self) -> MlirType {
        self.raw
    }
}

impl<'c> PartialEq for Type<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirTypeEqual(self.raw, other.raw) }
    }
}

impl<'c> Eq for Type<'c> {}

impl<'c> Display for Type<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirTypePrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl<'c> Debug for Type<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        write!(formatter, "Type(")?;
        Display::fmt(self, formatter)?;
        write!(formatter, ")")
    }
}

from_raw_subtypes!(
    Type,
    FunctionType,
    IntegerType,
    MemRefType,
    RankedTensorType,
    TupleType
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Type::parse(&Context::new(), "f32");
    }

    #[test]
    fn integer() {
        let context = Context::new();

        assert_eq!(
            Type::from(IntegerType::new(&context, 42)),
            Type::parse(&context, "i42").unwrap()
        );
    }

    #[test]
    fn index() {
        let context = Context::new();

        assert_eq!(
            Type::index(&context),
            Type::parse(&context, "index").unwrap()
        );
    }

    #[test]
    fn vector() {
        let context = Context::new();

        assert_eq!(
            Type::vector(&[42], Type::float64(&context)),
            Type::parse(&context, "vector<42xf64>").unwrap()
        );
    }

    #[test]
    fn vector_with_invalid_dimension() {
        let context = Context::new();

        assert_eq!(
            Type::vector(&[0], IntegerType::new(&context, 32).into()).to_string(),
            "vector<0xi32>"
        );
    }

    #[test]
    fn vector_checked() {
        let context = Context::new();

        assert_eq!(
            Type::vector_checked(
                Location::unknown(&context),
                &[42],
                IntegerType::new(&context, 32).into()
            ),
            Type::parse(&context, "vector<42xi32>")
        );
    }

    #[test]
    fn vector_checked_fail() {
        let context = Context::new();

        assert_eq!(
            Type::vector_checked(Location::unknown(&context), &[0], Type::index(&context)),
            None
        );
    }

    #[test]
    fn equal() {
        let context = Context::new();

        assert_eq!(Type::index(&context), Type::index(&context));
    }

    #[test]
    fn not_equal() {
        let context = Context::new();

        assert_ne!(Type::index(&context), Type::float64(&context));
    }

    #[test]
    fn display() {
        let context = Context::new();

        assert_eq!(Type::index(&context).to_string(), "index");
    }

    #[test]
    fn debug() {
        let context = Context::new();

        assert_eq!(format!("{:?}", Type::index(&context)), "Type(index)");
    }
}
