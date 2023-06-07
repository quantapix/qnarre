pub use self::{
    function::FunctionType, id::TypeId, integer::IntegerType, mem_ref::MemRefType, ranked_tensor::RankedTensorType,
    tuple::TupleType, type_like::TypeLike,
};
use super::Location;
use super::TypeId;
use super::TypeLike;
use crate::{ctx::Context, ctx::ContextRef, utils::print_callback, StringRef};
use crate::{ir::Type, utils::into_raw_array, Context, Error};
use crate::{
    ir::{AffineMap, Attribute, AttributeLike, Location, Type},
    Error,
};
use mlir_lib::*;
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    hash::{Hash, Hasher},
    marker::PhantomData,
};

#[derive(Clone, Copy)]
pub struct Type<'c> {
    raw: MlirType,
    _context: PhantomData<&'c Context>,
}
impl<'c> Type<'c> {
    pub fn parse(context: &'c Context, source: &str) -> Option<Self> {
        unsafe { Self::from_option_raw(mlirTypeParseGet(context.to_raw(), StringRef::from(source).to_raw())) }
    }
    pub fn bfloat16(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirBF16TypeGet(context.to_raw())) }
    }
    pub fn float16(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirF16TypeGet(context.to_raw())) }
    }
    pub fn float32(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirF32TypeGet(context.to_raw())) }
    }
    pub fn float64(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirF64TypeGet(context.to_raw())) }
    }
    pub fn index(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirIndexTypeGet(context.to_raw())) }
    }
    pub fn none(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirNoneTypeGet(context.to_raw())) }
    }
    pub fn vector(dimensions: &[u64], r#type: Self) -> Self {
        unsafe {
            Self::from_raw(mlirVectorTypeGet(
                dimensions.len() as isize,
                dimensions.as_ptr() as *const i64,
                r#type.raw,
            ))
        }
    }
    pub fn vector_checked(location: Location<'c>, dimensions: &[u64], r#type: Self) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirVectorTypeGetChecked(
                location.to_raw(),
                dimensions.len() as isize,
                dimensions.as_ptr() as *const i64,
                r#type.raw,
            ))
        }
    }
    pub unsafe fn from_raw(raw: MlirType) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
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
            mlirTypePrint(self.raw, Some(print_callback), &mut data as *mut _ as *mut c_void);
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
from_raw_subtypes!(Type, FunctionType, IntegerType, MemRefType, RankedTensorType, TupleType);
macro_rules! type_traits {
    ($name: ident, $is_type: ident, $string: expr) => {
        impl<'c> $name<'c> {
            unsafe fn from_raw(raw: MlirType) -> Self {
                Self {
                    r#type: Type::from_raw(raw),
                }
            }
        }
        impl<'c> TryFrom<crate::ir::r#type::Type<'c>> for $name<'c> {
            type Error = crate::Error;
            fn try_from(r#type: crate::ir::r#type::Type<'c>) -> Result<Self, Self::Error> {
                if r#type.$is_type() {
                    Ok(unsafe { Self::from_raw(r#type.to_raw()) })
                } else {
                    Err(Error::TypeExpected($string, r#type.to_string()))
                }
            }
        }
        impl<'c> crate::ir::r#type::TypeLike<'c> for $name<'c> {
            fn to_raw(&self) -> mlir_lib::MlirType {
                self.r#type.to_raw()
            }
        }
        impl<'c> std::fmt::Display for $name<'c> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.r#type, formatter)
            }
        }
    };
}

#[derive(Clone, Copy, Debug)]
pub struct FunctionType<'c> {
    r#type: Type<'c>,
}
impl<'c> FunctionType<'c> {
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
    pub fn input_count(&self) -> usize {
        unsafe { mlirFunctionTypeGetNumInputs(self.r#type.to_raw()) as usize }
    }
    pub fn result_count(&self) -> usize {
        unsafe { mlirFunctionTypeGetNumResults(self.r#type.to_raw()) as usize }
    }
}
type_traits!(FunctionType, is_function, "function");

#[derive(Clone, Copy, Debug)]
pub struct TypeId {
    raw: MlirTypeID,
}
impl TypeId {
    pub unsafe fn from_raw(raw: MlirTypeID) -> Self {
        Self { raw }
    }
}
impl PartialEq for TypeId {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirTypeIDEqual(self.raw, other.raw) }
    }
}
impl Eq for TypeId {}
impl Hash for TypeId {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        unsafe {
            mlirTypeIDHashValue(self.raw).hash(hasher);
        }
    }
}

#[derive(Debug)]
pub struct Allocator {
    raw: MlirTypeIDAllocator,
}
impl Allocator {
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirTypeIDAllocatorCreate() },
        }
    }
    pub fn allocate_type_id(&mut self) -> TypeId {
        unsafe { TypeId::from_raw(mlirTypeIDAllocatorAllocateTypeID(self.raw)) }
    }
}
impl Drop for Allocator {
    fn drop(&mut self) {
        unsafe { mlirTypeIDAllocatorDestroy(self.raw) }
    }
}
impl Default for Allocator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct IntegerType<'c> {
    r#type: Type<'c>,
}
impl<'c> IntegerType<'c> {
    pub fn new(context: &'c Context, bits: u32) -> Self {
        Self {
            r#type: unsafe { Type::from_raw(mlirIntegerTypeGet(context.to_raw(), bits)) },
        }
    }
    pub fn signed(context: &'c Context, bits: u32) -> Self {
        unsafe { Self::from_raw(mlirIntegerTypeSignedGet(context.to_raw(), bits)) }
    }
    pub fn unsigned(context: &'c Context, bits: u32) -> Self {
        unsafe { Self::from_raw(mlirIntegerTypeUnsignedGet(context.to_raw(), bits)) }
    }
    pub fn width(&self) -> u32 {
        unsafe { mlirIntegerTypeGetWidth(self.to_raw()) }
    }
    pub fn is_signed(&self) -> bool {
        unsafe { mlirIntegerTypeIsSigned(self.to_raw()) }
    }
    pub fn is_signless(&self) -> bool {
        unsafe { mlirIntegerTypeIsSignless(self.to_raw()) }
    }
    pub fn is_unsigned(&self) -> bool {
        unsafe { mlirIntegerTypeIsUnsigned(self.to_raw()) }
    }
}
type_traits!(IntegerType, is_integer, "integer");
#[derive(Clone, Copy, Debug)]
pub struct MemRefType<'c> {
    r#type: Type<'c>,
}
impl<'c> MemRefType<'c> {
    pub fn new(
        r#type: Type<'c>,
        dimensions: &[u64],
        layout: Option<Attribute<'c>>,
        memory_space: Option<Attribute<'c>>,
    ) -> Self {
        unsafe {
            Self::from_raw(mlirMemRefTypeGet(
                r#type.to_raw(),
                dimensions.len() as _,
                dimensions.as_ptr() as *const _,
                layout.unwrap_or_else(|| Attribute::null()).to_raw(),
                memory_space.unwrap_or_else(|| Attribute::null()).to_raw(),
            ))
        }
    }
    pub fn checked(
        location: Location<'c>,
        r#type: Type<'c>,
        dimensions: &[u64],
        layout: Attribute<'c>,
        memory_space: Attribute<'c>,
    ) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirMemRefTypeGetChecked(
                location.to_raw(),
                r#type.to_raw(),
                dimensions.len() as isize,
                dimensions.as_ptr() as *const i64,
                layout.to_raw(),
                memory_space.to_raw(),
            ))
        }
    }
    pub fn layout(&self) -> Attribute<'c> {
        unsafe { Attribute::from_raw(mlirMemRefTypeGetLayout(self.r#type.to_raw())) }
    }
    pub fn affine_map(&self) -> AffineMap<'c> {
        unsafe { AffineMap::from_raw(mlirMemRefTypeGetAffineMap(self.r#type.to_raw())) }
    }
    pub fn memory_space(&self) -> Option<Attribute<'c>> {
        unsafe { Attribute::from_option_raw(mlirMemRefTypeGetMemorySpace(self.r#type.to_raw())) }
    }
    unsafe fn from_option_raw(raw: MlirType) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}
type_traits!(MemRefType, is_mem_ref, "mem ref");

#[derive(Clone, Copy, Debug)]
pub struct RankedTensorType<'c> {
    r#type: Type<'c>,
}
impl<'c> RankedTensorType<'c> {
    pub fn new(dimensions: &[u64], r#type: Type<'c>, encoding: Option<Attribute<'c>>) -> Self {
        unsafe {
            Self::from_raw(mlirRankedTensorTypeGet(
                dimensions.len() as _,
                dimensions.as_ptr() as *const _,
                r#type.to_raw(),
                encoding.unwrap_or_else(|| Attribute::null()).to_raw(),
            ))
        }
    }
    pub fn checked(
        dimensions: &[u64],
        r#type: Type<'c>,
        encoding: Attribute<'c>,
        location: Location<'c>,
    ) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirRankedTensorTypeGetChecked(
                location.to_raw(),
                dimensions.len() as _,
                dimensions.as_ptr() as *const _,
                r#type.to_raw(),
                encoding.to_raw(),
            ))
        }
    }
    pub fn encoding(&self) -> Option<Attribute<'c>> {
        unsafe { Attribute::from_option_raw(mlirRankedTensorTypeGetEncoding(self.r#type.to_raw())) }
    }
    unsafe fn from_option_raw(raw: MlirType) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}
type_traits!(RankedTensorType, is_ranked_tensor, "tensor");

#[derive(Clone, Copy, Debug)]
pub struct TupleType<'c> {
    r#type: Type<'c>,
}
impl<'c> TupleType<'c> {
    pub fn new(context: &'c Context, types: &[Type<'c>]) -> Self {
        unsafe {
            Self::from_raw(mlirTupleTypeGet(
                context.to_raw(),
                types.len() as isize,
                into_raw_array(types.iter().map(|r#type| r#type.to_raw()).collect()),
            ))
        }
    }
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
    pub fn type_count(&self) -> usize {
        unsafe { mlirTupleTypeGetNumTypes(self.r#type.to_raw()) as usize }
    }
}
type_traits!(TupleType, is_tuple, "tuple");

pub trait TypeLike<'c> {
    fn to_raw(&self) -> MlirType;
    fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirTypeGetContext(self.to_raw())) }
    }
    fn id(&self) -> TypeId {
        unsafe { TypeId::from_raw(mlirTypeGetTypeID(self.to_raw())) }
    }
    fn dump(&self) {
        unsafe { mlirTypeDump(self.to_raw()) }
    }
    melior_macro::type_check_functions!(
        mlirTypeIsAAnyQuantizedType,
        mlirTypeIsABF16,
        mlirTypeIsACalibratedQuantizedType,
        mlirTypeIsAComplex,
        mlirTypeIsAF16,
        mlirTypeIsAF32,
        mlirTypeIsAF64,
        mlirTypeIsAFloat8E4M3FN,
        mlirTypeIsAFloat8E5M2,
        mlirTypeIsAFunction,
        mlirTypeIsAIndex,
        mlirTypeIsAInteger,
        mlirTypeIsAMemRef,
        mlirTypeIsANone,
        mlirTypeIsAOpaque,
        mlirTypeIsAPDLAttributeType,
        mlirTypeIsAPDLOperationType,
        mlirTypeIsAPDLRangeType,
        mlirTypeIsAPDLType,
        mlirTypeIsAPDLTypeType,
        mlirTypeIsAPDLValueType,
        mlirTypeIsAQuantizedType,
        mlirTypeIsARankedTensor,
        mlirTypeIsAShaped,
        mlirTypeIsATensor,
        mlirTypeIsATransformAnyOpType,
        mlirTypeIsATransformOperationType,
        mlirTypeIsATuple,
        mlirTypeIsAUniformQuantizedPerAxisType,
        mlirTypeIsAUniformQuantizedType,
        mlirTypeIsAUnrankedMemRef,
        mlirTypeIsAUnrankedTensor,
        mlirTypeIsAVector,
    );
}
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
        assert_eq!(Type::index(&context), Type::parse(&context, "index").unwrap());
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
        assert_eq!(FunctionType::new(&context, &[integer], &[]).input(0), Ok(integer));
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
        assert_eq!(FunctionType::new(&context, &[], &[integer]).result(0), Ok(integer));
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
        assert_eq!(FunctionType::new(&context, &[integer], &[]).input_count(), 1);
    }
    #[test]
    fn result_count() {
        let context = Context::new();
        let integer = Type::index(&context);
        assert_eq!(FunctionType::new(&context, &[], &[integer]).result_count(), 1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn new() {
        Allocator::new();
    }
    #[test]
    fn allocate_type_id() {
        Allocator::new().allocate_type_id();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn new() {
        assert!(IntegerType::new(&Context::new(), 64).is_integer());
    }
    #[test]
    fn signed() {
        assert!(IntegerType::signed(&Context::new(), 64).is_integer());
    }
    #[test]
    fn unsigned() {
        assert!(IntegerType::unsigned(&Context::new(), 64).is_integer());
    }
    #[test]
    fn signed_integer() {
        let context = Context::new();
        assert_eq!(
            Type::from(IntegerType::signed(&context, 42)),
            Type::parse(&context, "si42").unwrap()
        );
    }
    #[test]
    fn unsigned_integer() {
        let context = Context::new();
        assert_eq!(
            Type::from(IntegerType::unsigned(&context, 42)),
            Type::parse(&context, "ui42").unwrap()
        );
    }
    #[test]
    fn get_width() {
        let context = Context::new();
        assert_eq!(IntegerType::new(&context, 64).width(), 64);
    }
    #[test]
    fn check_sign() {
        let context = Context::new();
        let signless = IntegerType::new(&context, 42);
        let signed = IntegerType::signed(&context, 42);
        let unsigned = IntegerType::unsigned(&context, 42);
        assert!(signless.is_signless());
        assert!(!signed.is_signless());
        assert!(!unsigned.is_signless());
        assert!(!signless.is_signed());
        assert!(signed.is_signed());
        assert!(!unsigned.is_signed());
        assert!(!signless.is_unsigned());
        assert!(!signed.is_unsigned());
        assert!(unsigned.is_unsigned());
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;
    #[test]
    fn new() {
        let context = Context::new();
        assert_eq!(
            Type::from(MemRefType::new(Type::float64(&context), &[42], None, None,)),
            Type::parse(&context, "memref<42xf64>").unwrap()
        );
    }
    #[test]
    fn layout() {
        let context = Context::new();
        assert_eq!(
            MemRefType::new(Type::index(&context), &[42, 42], None, None,).layout(),
            Attribute::parse(&context, "affine_map<(d0, d1) -> (d0, d1)>").unwrap(),
        );
    }
    #[test]
    fn affine_map() {
        let context = Context::new();
        assert_eq!(
            MemRefType::new(Type::index(&context), &[42, 42], None, None,)
                .affine_map()
                .to_string(),
            "(d0, d1) -> (d0, d1)"
        );
    }
    #[test]
    fn memory_space() {
        let context = Context::new();
        assert_eq!(
            MemRefType::new(Type::index(&context), &[42, 42], None, None).memory_space(),
            None,
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;
    #[test]
    fn new() {
        let context = Context::new();
        assert_eq!(
            Type::from(RankedTensorType::new(&[42], Type::float64(&context), None)),
            Type::parse(&context, "tensor<42xf64>").unwrap()
        );
    }
    #[test]
    fn encoding() {
        let context = Context::new();
        assert_eq!(
            RankedTensorType::new(&[42, 42], Type::index(&context), None).encoding(),
            None,
        );
    }
}
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            r#type::{FunctionType, IntegerType},
            Type,
        },
        Context,
    };
    #[test]
    fn context() {
        Type::parse(&Context::new(), "i8").unwrap().context();
    }
    #[test]
    fn id() {
        let context = Context::new();
        assert_eq!(Type::index(&context).id(), Type::index(&context).id());
    }
    #[test]
    fn is_integer() {
        let context = Context::new();
        assert!(IntegerType::new(&context, 64).is_integer());
    }
    #[test]
    fn is_index() {
        let context = Context::new();
        assert!(Type::index(&context).is_index());
    }
    #[test]
    fn is_bfloat16() {
        let context = Context::new();
        assert!(FunctionType::new(&context, &[], &[]).is_function());
    }
    #[test]
    fn is_function() {
        let context = Context::new();
        assert!(FunctionType::new(&context, &[], &[]).is_function());
    }
    #[test]
    fn is_vector() {
        let context = Context::new();
        assert!(Type::vector(&[42], Type::index(&context)).is_vector());
    }
    #[test]
    fn dump() {
        Type::index(&Context::new()).dump();
    }
}
