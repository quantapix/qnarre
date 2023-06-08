use macros::attribute_check_functions;
use mlir_lib::*;
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

pub use self::{
    array::ArrayAttribute, attribute_like::AttributeLike, dense_elements::DenseElementsAttribute,
    dense_i32_array::DenseI32ArrayAttribute, dense_i64_array::DenseI64ArrayAttribute,
    flat_symbol_ref::FlatSymbolRefAttribute, float::FloatAttribute, integer::IntegerAttribute, r#type::TypeAttribute,
    string::StringAttribute,
};
use super::{Attribute, AttributeLike};
use crate::ir::Type;
use crate::{ctx::Context, utils::print_callback, StringRef};
use crate::{Context, ContextRef, Error, StringRef};

#[derive(Clone, Copy)]
pub struct Attribute<'c> {
    raw: MlirAttribute,
    _context: PhantomData<&'c Context>,
}
impl<'c> Attribute<'c> {
    pub fn parse(context: &'c Context, source: &str) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirAttributeParseGet(
                context.to_raw(),
                StringRef::from(source).to_raw(),
            ))
        }
    }
    pub fn unit(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirUnitAttrGet(context.to_raw())) }
    }
    pub unsafe fn null() -> Self {
        unsafe { Self::from_raw(mlirAttributeGetNull()) }
    }
    pub unsafe fn from_raw(raw: MlirAttribute) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
    pub unsafe fn from_option_raw(raw: MlirAttribute) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}
impl<'c> AttributeLike<'c> for Attribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.raw
    }
}
impl<'c> PartialEq for Attribute<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirAttributeEqual(self.raw, other.raw) }
    }
}
impl<'c> Eq for Attribute<'c> {}
impl<'c> Display for Attribute<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));
        unsafe {
            mlirAttributePrint(self.raw, Some(print_callback), &mut data as *mut _ as *mut c_void);
        }
        data.1
    }
}
impl<'c> Debug for Attribute<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self, formatter)
    }
}

from_raw_subtypes!(
    Attribute,
    ArrayAttribute,
    DenseElementsAttribute,
    DenseI32ArrayAttribute,
    DenseI64ArrayAttribute,
    FlatSymbolRefAttribute,
    FloatAttribute,
    IntegerAttribute,
    StringAttribute,
    TypeAttribute,
);

macro_rules! attribute_traits {
    ($name: ident, $is_type: ident, $string: expr) => {
        impl<'c> $name<'c> {
            unsafe fn from_raw(raw: MlirAttribute) -> Self {
                Self {
                    attribute: Attribute::from_raw(raw),
                }
            }
        }
        impl<'c> TryFrom<crate::ir::attribute::Attribute<'c>> for $name<'c> {
            type Error = crate::Error;
            fn try_from(attribute: crate::ir::attribute::Attribute<'c>) -> Result<Self, Self::Error> {
                if attribute.$is_type() {
                    Ok(unsafe { Self::from_raw(attribute.to_raw()) })
                } else {
                    Err(Error::AttributeExpected($string, attribute.to_string()))
                }
            }
        }
        impl<'c> crate::ir::attribute::AttributeLike<'c> for $name<'c> {
            fn to_raw(&self) -> mlir_lib::MlirAttribute {
                self.attribute.to_raw()
            }
        }
        impl<'c> std::fmt::Display for $name<'c> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.attribute, formatter)
            }
        }
        impl<'c> std::fmt::Debug for $name<'c> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(self, formatter)
            }
        }
    };
}

#[derive(Clone, Copy)]
pub struct ArrayAttribute<'c> {
    attribute: Attribute<'c>,
}
impl<'c> ArrayAttribute<'c> {
    pub fn new(context: &'c Context, values: &[Attribute<'c>]) -> Self {
        unsafe {
            Self::from_raw(mlirArrayAttrGet(
                context.to_raw(),
                values.len() as isize,
                values.as_ptr() as *const _ as *const _,
            ))
        }
    }
    pub fn len(&self) -> usize {
        (unsafe { mlirArrayAttrGetNumElements(self.attribute.to_raw()) }) as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn element(&self, index: usize) -> Result<Attribute<'c>, Error> {
        if index < self.len() {
            Ok(unsafe { Attribute::from_raw(mlirArrayAttrGetElement(self.attribute.to_raw(), index as isize)) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "array element",
                value: self.to_string(),
                index,
            })
        }
    }
}
attribute_traits!(ArrayAttribute, is_dense_i64_array, "dense i64 array");
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{attribute::IntegerAttribute, r#type::IntegerType, Type};
    #[test]
    fn element() {
        let context = Context::new();
        let r#type = IntegerType::new(&context, 64).into();
        let attributes = [
            IntegerAttribute::new(1, r#type).into(),
            IntegerAttribute::new(2, r#type).into(),
            IntegerAttribute::new(3, r#type).into(),
        ];
        let attribute = ArrayAttribute::new(&context, &attributes);
        assert_eq!(attribute.element(0).unwrap(), attributes[0]);
        assert_eq!(attribute.element(1).unwrap(), attributes[1]);
        assert_eq!(attribute.element(2).unwrap(), attributes[2]);
        assert!(matches!(attribute.element(3), Err(Error::PositionOutOfBounds { .. })));
    }
    #[test]
    fn len() {
        let context = Context::new();
        let attribute = ArrayAttribute::new(&context, &[IntegerAttribute::new(1, Type::index(&context)).into()]);
        assert_eq!(attribute.len(), 1);
    }
}

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

#[derive(Clone, Copy)]
pub struct DenseElementsAttribute<'c> {
    attribute: Attribute<'c>,
}
impl<'c> DenseElementsAttribute<'c> {
    pub fn new(r#type: Type<'c>, values: &[Attribute<'c>]) -> Result<Self, Error> {
        if r#type.is_shaped() {
            Ok(unsafe {
                Self::from_raw(mlirDenseElementsAttrGet(
                    r#type.to_raw(),
                    values.len() as isize,
                    values.as_ptr() as *const _ as *const _,
                ))
            })
        } else {
            Err(Error::TypeExpected("shaped", r#type.to_string()))
        }
    }
    pub fn len(&self) -> usize {
        (unsafe { mlirElementsAttrGetNumElements(self.attribute.to_raw()) }) as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn i32_element(&self, index: usize) -> Result<i32, Error> {
        if !self.is_dense_int_elements() {
            Err(Error::ElementExpected {
                r#type: "integer",
                value: self.to_string(),
            })
        } else if index < self.len() {
            Ok(unsafe { mlirDenseElementsAttrGetInt32Value(self.attribute.to_raw(), index as isize) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
    }
    pub fn i64_element(&self, index: usize) -> Result<i64, Error> {
        if !self.is_dense_int_elements() {
            Err(Error::ElementExpected {
                r#type: "integer",
                value: self.to_string(),
            })
        } else if index < self.len() {
            Ok(unsafe { mlirDenseElementsAttrGetInt64Value(self.attribute.to_raw(), index as isize) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
    }
}
attribute_traits!(DenseElementsAttribute, is_dense_elements, "dense elements");

#[derive(Clone, Copy)]
pub struct DenseI32ArrayAttribute<'c> {
    attribute: Attribute<'c>,
}
impl<'c> DenseI32ArrayAttribute<'c> {
    pub fn new(context: &'c Context, values: &[i32]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseI32ArrayGet(
                context.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }
    pub fn len(&self) -> usize {
        (unsafe { mlirArrayAttrGetNumElements(self.attribute.to_raw()) }) as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn element(&self, index: usize) -> Result<i32, Error> {
        if index < self.len() {
            Ok(unsafe { mlirDenseI32ArrayGetElement(self.attribute.to_raw(), index as isize) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "array element",
                value: self.to_string(),
                index,
            })
        }
    }
}
attribute_traits!(DenseI32ArrayAttribute, is_dense_i32_array, "dense i32 array");

#[derive(Clone, Copy)]
pub struct DenseI64ArrayAttribute<'c> {
    attribute: Attribute<'c>,
}
impl<'c> DenseI64ArrayAttribute<'c> {
    pub fn new(context: &'c Context, values: &[i64]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseI64ArrayGet(
                context.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }
    pub fn len(&self) -> usize {
        (unsafe { mlirArrayAttrGetNumElements(self.attribute.to_raw()) }) as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn element(&self, index: usize) -> Result<i64, Error> {
        if index < self.len() {
            Ok(unsafe { mlirDenseI64ArrayGetElement(self.attribute.to_raw(), index as isize) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "array element",
                value: self.to_string(),
                index,
            })
        }
    }
}
attribute_traits!(DenseI64ArrayAttribute, is_dense_i64_array, "dense i64 array");

#[derive(Clone, Copy)]
pub struct FlatSymbolRefAttribute<'c> {
    attribute: Attribute<'c>,
}
impl<'c> FlatSymbolRefAttribute<'c> {
    pub fn new(context: &'c Context, symbol: &str) -> Self {
        unsafe {
            Self::from_raw(mlirFlatSymbolRefAttrGet(
                context.to_raw(),
                StringRef::from(symbol).to_raw(),
            ))
        }
    }
    pub fn value(&self) -> &str {
        unsafe { StringRef::from_raw(mlirFlatSymbolRefAttrGetValue(self.to_raw())) }
            .as_str()
            .unwrap()
    }
}
attribute_traits!(FlatSymbolRefAttribute, is_flat_symbol_ref, "flat symbol ref");

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

#[derive(Clone, Copy)]
pub struct IntegerAttribute<'c> {
    attribute: Attribute<'c>,
}
impl<'c> IntegerAttribute<'c> {
    pub fn new(integer: i64, r#type: Type<'c>) -> Self {
        unsafe { Self::from_raw(mlirIntegerAttrGet(r#type.to_raw(), integer)) }
    }
}
attribute_traits!(IntegerAttribute, is_integer, "integer");

#[derive(Clone, Copy)]
pub struct StringAttribute<'c> {
    attribute: Attribute<'c>,
}
impl<'c> StringAttribute<'c> {
    pub fn new(context: &'c Context, string: &str) -> Self {
        unsafe { Self::from_raw(mlirStringAttrGet(context.to_raw(), StringRef::from(string).to_raw())) }
    }
}
attribute_traits!(StringAttribute, is_string, "string");

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
    use crate::{
        ir::{attribute::IntegerAttribute, IntegerType, MemRefType, Type, TypeLike},
        Context,
    };
    #[test]
    fn value() {
        let context = Context::new();
        let r#type = Type::index(&context);
        assert_eq!(TypeAttribute::new(r#type).value(), r#type);
    }
    #[test]
    fn new() {
        assert_eq!(FlatSymbolRefAttribute::new(&Context::new(), "foo").value(), "foo");
    }
    #[test]
    fn element() {
        let context = Context::new();
        let attribute = DenseI64ArrayAttribute::new(&context, &[1, 2, 3]);
        assert_eq!(attribute.element(0).unwrap(), 1);
        assert_eq!(attribute.element(1).unwrap(), 2);
        assert_eq!(attribute.element(2).unwrap(), 3);
        assert!(matches!(attribute.element(3), Err(Error::PositionOutOfBounds { .. })));
    }
    #[test]
    fn len() {
        let context = Context::new();
        let attribute = DenseI64ArrayAttribute::new(&context, &[1, 2, 3]);
        assert_eq!(attribute.len(), 3);
    }
    #[test]
    fn element() {
        let context = Context::new();
        let attribute = DenseI32ArrayAttribute::new(&context, &[1, 2, 3]);
        assert_eq!(attribute.element(0).unwrap(), 1);
        assert_eq!(attribute.element(1).unwrap(), 2);
        assert_eq!(attribute.element(2).unwrap(), 3);
        assert!(matches!(attribute.element(3), Err(Error::PositionOutOfBounds { .. })));
    }
    #[test]
    fn len() {
        let context = Context::new();
        let attribute = DenseI32ArrayAttribute::new(&context, &[1, 2, 3]);
        assert_eq!(attribute.len(), 3);
    }
    #[test]
    fn i32_element() {
        let context = Context::new();
        let integer_type = IntegerType::new(&context, 32).into();
        let attribute = DenseElementsAttribute::new(
            MemRefType::new(integer_type, &[3], None, None).into(),
            &[IntegerAttribute::new(42, integer_type).into()],
        )
        .unwrap();
        assert_eq!(attribute.i32_element(0), Ok(42));
        assert_eq!(attribute.i32_element(1), Ok(42));
        assert_eq!(attribute.i32_element(2), Ok(42));
        assert_eq!(
            attribute.i32_element(3),
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: attribute.to_string(),
                index: 3,
            })
        );
    }
    #[test]
    fn i64_element() {
        let context = Context::new();
        let integer_type = IntegerType::new(&context, 64).into();
        let attribute = DenseElementsAttribute::new(
            MemRefType::new(integer_type, &[3], None, None).into(),
            &[IntegerAttribute::new(42, integer_type).into()],
        )
        .unwrap();
        assert_eq!(attribute.i64_element(0), Ok(42));
        assert_eq!(attribute.i64_element(1), Ok(42));
        assert_eq!(attribute.i64_element(2), Ok(42));
        assert_eq!(
            attribute.i64_element(3),
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: attribute.to_string(),
                index: 3,
            })
        );
    }
    #[test]
    fn len() {
        let context = Context::new();
        let integer_type = IntegerType::new(&context, 64).into();
        let attribute = DenseElementsAttribute::new(
            MemRefType::new(integer_type, &[3], None, None).into(),
            &[IntegerAttribute::new(0, integer_type).into()],
        )
        .unwrap();
        assert_eq!(attribute.len(), 3);
    }
    #[test]
    fn parse() {
        for attribute in ["unit", "i32", r#""foo""#] {
            assert!(Attribute::parse(&Context::new(), attribute).is_some());
        }
    }
    #[test]
    fn parse_none() {
        assert!(Attribute::parse(&Context::new(), "z").is_none());
    }
    #[test]
    fn context() {
        Attribute::parse(&Context::new(), "unit").unwrap().context();
    }
    #[test]
    fn r#type() {
        let context = Context::new();
        assert_eq!(
            Attribute::parse(&context, "unit").unwrap().r#type(),
            Type::none(&context)
        );
    }
    #[ignore]
    #[test]
    fn type_id() {
        let context = Context::new();
        assert_eq!(
            Attribute::parse(&context, "42 : index").unwrap().type_id(),
            Type::index(&context).id()
        );
    }
    #[test]
    fn is_array() {
        assert!(Attribute::parse(&Context::new(), "[]").unwrap().is_array());
    }
    #[test]
    fn is_bool() {
        assert!(Attribute::parse(&Context::new(), "false").unwrap().is_bool());
    }
    #[test]
    fn is_dense_elements() {
        assert!(Attribute::parse(&Context::new(), "dense<10> : tensor<2xi8>")
            .unwrap()
            .is_dense_elements());
    }
    #[test]
    fn is_dense_int_elements() {
        assert!(Attribute::parse(&Context::new(), "dense<42> : tensor<42xi8>")
            .unwrap()
            .is_dense_int_elements());
    }
    #[test]
    fn is_dense_fp_elements() {
        assert!(Attribute::parse(&Context::new(), "dense<42.0> : tensor<42xf32>")
            .unwrap()
            .is_dense_fp_elements());
    }
    #[test]
    fn is_elements() {
        assert!(
            Attribute::parse(&Context::new(), "sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>")
                .unwrap()
                .is_elements()
        );
    }
    #[test]
    fn is_integer() {
        assert!(Attribute::parse(&Context::new(), "42").unwrap().is_integer());
    }
    #[test]
    fn is_integer_set() {
        assert!(Attribute::parse(&Context::new(), "affine_set<(d0) : (d0 - 2 >= 0)>")
            .unwrap()
            .is_integer_set());
    }
    #[ignore]
    #[test]
    fn is_opaque() {
        assert!(Attribute::parse(&Context::new(), "#foo<\"bar\">").unwrap().is_opaque());
    }
    #[test]
    fn is_sparse_elements() {
        assert!(
            Attribute::parse(&Context::new(), "sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>")
                .unwrap()
                .is_sparse_elements()
        );
    }
    #[test]
    fn is_string() {
        assert!(Attribute::parse(&Context::new(), "\"foo\"").unwrap().is_string());
    }
    #[test]
    fn is_type() {
        assert!(Attribute::parse(&Context::new(), "index").unwrap().is_type());
    }
    #[test]
    fn is_unit() {
        assert!(Attribute::parse(&Context::new(), "unit").unwrap().is_unit());
    }
    #[test]
    fn is_symbol() {
        assert!(Attribute::parse(&Context::new(), "@foo").unwrap().is_symbol_ref());
    }
    #[test]
    fn equal() {
        let context = Context::new();
        let attribute = Attribute::parse(&context, "unit").unwrap();
        assert_eq!(attribute, attribute);
    }
    #[test]
    fn not_equal() {
        let context = Context::new();
        assert_ne!(
            Attribute::parse(&context, "unit").unwrap(),
            Attribute::parse(&context, "42").unwrap()
        );
    }
    #[test]
    fn display() {
        assert_eq!(Attribute::parse(&Context::new(), "unit").unwrap().to_string(), "unit");
    }
}
