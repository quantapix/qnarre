use super::{
    ctx::{Context, ContextRef},
    ir::{Attribute, AttributeLike},
    utils::{into_raw_array, print_callback},
    StringRef,
};
use mlir_lib::*;
use mlir_lib::*;
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
    mem::{forget, transmute},
    ops::Deref,
};
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
    mem::{forget, transmute},
    ops::Deref,
};

use super::{Location, Operation, OperationRef, RegionRef, Type, TypeLike, Value};
use crate::{
    ctx::Context,
    ir::{BlockRef, Type, TypeLike, ValueLike},
    utils::{into_raw_array, print_callback},
    Error,
};
use crate::{
    ctx::{Context, ContextRef},
    ir::*,
    string_ref::StringRef,
    utils::{into_raw_array, print_callback, print_string_callback},
    Error,
};
use mlir_lib::*;
use std::{
    ffi::c_void,
    fmt::{Debug, Display, Formatter},
    marker::PhantomData,
    ops::Deref,
};

pub use self::{builder::OperationBuilder, printing_flags::OperationPrintingFlags, result::OperationResult};
use super::{BlockRef, Identifier, Operation, RegionRef, Value};
use crate::print_callback;
use core::{
    fmt,
    mem::{forget, transmute},
};
use mlir_lib::*;
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

use super::Type;
use super::{block::BlockArgument, op::OperationResult, Type};

pub use self::{
    attr::{Attribute, AttributeLike},
    block::{Block, BlockRef},
    op::{Operation, OperationRef},
    typ::{Type, TypeLike},
    val::{Value, ValueLike},
};
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

pub use self::{
    function::FunctionType, id::TypeId, integer::IntegerType, mem_ref::MemRefType, ranked_tensor::RankedTensorType,
    tuple::TupleType, type_like::TypeLike,
};
use super::*;

#[derive(Clone, Copy)]
pub struct AffineMap<'c> {
    raw: MlirAffineMap,
    _context: PhantomData<&'c Context>,
}
impl<'c> AffineMap<'c> {
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirAffineMapGetContext(self.raw)) }
    }
    pub fn dump(&self) {
        unsafe { mlirAffineMapDump(self.raw) }
    }
    pub unsafe fn from_raw(raw: MlirAffineMap) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
}
impl<'c> PartialEq for AffineMap<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirAffineMapEqual(self.raw, other.raw) }
    }
}
impl<'c> Eq for AffineMap<'c> {}
impl<'c> Display for AffineMap<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));
        unsafe {
            mlirAffineMapPrint(self.raw, Some(print_callback), &mut data as *mut _ as *mut c_void);
        }
        data.1
    }
}
impl<'c> Debug for AffineMap<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self, formatter)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Identifier<'c> {
    raw: MlirIdentifier,
    _context: PhantomData<&'c Context>,
}
impl<'c> Identifier<'c> {
    pub fn new(context: &Context, name: &str) -> Self {
        unsafe { Self::from_raw(mlirIdentifierGet(context.to_raw(), StringRef::from(name).to_raw())) }
    }
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirIdentifierGetContext(self.raw)) }
    }
    pub fn as_string_ref(&self) -> StringRef {
        unsafe { StringRef::from_raw(mlirIdentifierStr(self.raw)) }
    }
    pub unsafe fn from_raw(raw: MlirIdentifier) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
    pub fn to_raw(self) -> MlirIdentifier {
        self.raw
    }
}
impl<'c> PartialEq for Identifier<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirIdentifierEqual(self.raw, other.raw) }
    }
}
impl<'c> Eq for Identifier<'c> {}

#[derive(Clone, Copy, Debug)]
pub struct Location<'c> {
    raw: MlirLocation,
    _context: PhantomData<&'c Context>,
}
impl<'c> Location<'c> {
    pub fn new(context: &'c Context, filename: &str, line: usize, column: usize) -> Self {
        unsafe {
            Self::from_raw(mlirLocationFileLineColGet(
                context.to_raw(),
                StringRef::from(filename).to_raw(),
                line as u32,
                column as u32,
            ))
        }
    }
    pub fn fused(context: &'c Context, locations: &[Self], attribute: Attribute) -> Self {
        unsafe {
            Self::from_raw(mlirLocationFusedGet(
                context.to_raw(),
                locations.len() as isize,
                into_raw_array(locations.iter().map(|location| location.to_raw()).collect()),
                attribute.to_raw(),
            ))
        }
    }
    pub fn name(context: &'c Context, name: &str, child: Location) -> Self {
        unsafe {
            Self::from_raw(mlirLocationNameGet(
                context.to_raw(),
                StringRef::from(name).to_raw(),
                child.to_raw(),
            ))
        }
    }
    pub fn unknown(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirLocationUnknownGet(context.to_raw())) }
    }
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirLocationGetContext(self.raw)) }
    }
    pub unsafe fn from_raw(raw: MlirLocation) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
    pub fn to_raw(self) -> MlirLocation {
        self.raw
    }
}
impl<'c> PartialEq for Location<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirLocationEqual(self.raw, other.raw) }
    }
}
impl<'c> Display for Location<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));
        unsafe {
            mlirLocationPrint(self.raw, Some(print_callback), &mut data as *mut _ as *mut c_void);
        }
        data.1
    }
}

#[derive(Debug)]
pub struct Module<'c> {
    raw: MlirModule,
    _context: PhantomData<&'c Context>,
}
impl<'c> Module<'c> {
    pub fn new(location: Location) -> Self {
        unsafe { Self::from_raw(mlirModuleCreateEmpty(location.to_raw())) }
    }
    pub fn parse(context: &Context, source: &str) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirModuleCreateParse(
                context.to_raw(),
                StringRef::from(source).to_raw(),
            ))
        }
    }
    pub fn as_operation(&self) -> OperationRef {
        unsafe { OperationRef::from_raw(mlirModuleGetOperation(self.raw)) }
    }
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirModuleGetContext(self.raw)) }
    }
    pub fn body(&self) -> BlockRef {
        unsafe { BlockRef::from_raw(mlirModuleGetBody(self.raw)) }
    }
    pub fn from_operation(operation: Operation) -> Option<Self> {
        unsafe { Self::from_option_raw(mlirModuleFromOperation(operation.into_raw())) }
    }
    pub unsafe fn from_raw(raw: MlirModule) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
    pub unsafe fn from_option_raw(raw: MlirModule) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
    pub fn to_raw(&self) -> MlirModule {
        self.raw
    }
}
impl<'c> Drop for Module<'c> {
    fn drop(&mut self) {
        unsafe { mlirModuleDestroy(self.raw) };
    }
}

#[derive(Debug)]
pub struct Region {
    raw: MlirRegion,
}
impl Region {
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirRegionCreate() },
        }
    }
    pub fn first_block(&self) -> Option<BlockRef> {
        unsafe {
            let block = mlirRegionGetFirstBlock(self.raw);
            if block.ptr.is_null() {
                None
            } else {
                Some(BlockRef::from_raw(block))
            }
        }
    }
    pub fn insert_block_after(&self, one: BlockRef, other: Block) -> BlockRef {
        unsafe {
            let r#ref = BlockRef::from_raw(other.to_raw());
            mlirRegionInsertOwnedBlockAfter(self.raw, one.to_raw(), other.into_raw());
            r#ref
        }
    }
    pub fn insert_block_before(&self, one: BlockRef, other: Block) -> BlockRef {
        unsafe {
            let r#ref = BlockRef::from_raw(other.to_raw());
            mlirRegionInsertOwnedBlockBefore(self.raw, one.to_raw(), other.into_raw());
            r#ref
        }
    }
    pub fn append_block(&self, block: Block) -> BlockRef {
        unsafe {
            let r#ref = BlockRef::from_raw(block.to_raw());
            mlirRegionAppendOwnedBlock(self.raw, block.into_raw());
            r#ref
        }
    }
    pub fn into_raw(self) -> MlirRegion {
        let region = self.raw;
        forget(self);
        region
    }
}
impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for Region {
    fn drop(&mut self) {
        unsafe { mlirRegionDestroy(self.raw) }
    }
}
impl PartialEq for Region {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, other.raw) }
    }
}
impl Eq for Region {}

#[derive(Clone, Copy, Debug)]
pub struct RegionRef<'a> {
    raw: MlirRegion,
    _region: PhantomData<&'a Region>,
}
impl<'a> RegionRef<'a> {
    pub unsafe fn from_raw(raw: MlirRegion) -> Self {
        Self {
            raw,
            _region: Default::default(),
        }
    }
    pub unsafe fn from_option_raw(raw: MlirRegion) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}
impl<'a> Deref for RegionRef<'a> {
    type Target = Region;
    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
    }
}
impl<'a> PartialEq for RegionRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, other.raw) }
    }
}
impl<'a> Eq for RegionRef<'a> {}

use macros::attribute_check_functions;
use mlir_lib::*;
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

use super::{Attribute, AttributeLike};
use crate::mlir::{ir::Type, utils::print_callback, Context, ContextRef, Error, StringRef};

#[derive(Clone, Copy)]
pub struct Attribute<'c> {
    raw: MlirAttribute,
    _ctx: PhantomData<&'c Context>,
}
impl<'c> Attribute<'c> {
    pub fn parse(x: &'c Context, src: &str) -> Option<Self> {
        unsafe { Self::from_option_raw(mlirAttributeParseGet(x.to_raw(), StringRef::from(src).to_raw())) }
    }
    pub fn unit(x: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirUnitAttrGet(x.to_raw())) }
    }
    pub unsafe fn null() -> Self {
        unsafe { Self::from_raw(mlirAttributeGetNull()) }
    }
    pub unsafe fn from_raw(raw: MlirAttribute) -> Self {
        Self {
            raw,
            _ctx: Default::default(),
        }
    }
    pub unsafe fn from_option_raw(x: MlirAttribute) -> Option<Self> {
        if x.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(x))
        }
    }
}
impl<'c> AttributeLike<'c> for Attribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.raw
    }
}
impl<'c> PartialEq for Attribute<'c> {
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirAttributeEqual(self.raw, x.raw) }
    }
}
impl<'c> Eq for Attribute<'c> {}
impl<'c> Display for Attribute<'c> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut y = (f, Ok(()));
        unsafe {
            mlirAttributePrint(self.raw, Some(print_callback), &mut y as *mut _ as *mut c_void);
        }
        y.1
    }
}
impl<'c> Debug for Attribute<'c> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self, f)
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
        impl<'c> TryFrom<crate::mlir::ir::attr::Attribute<'c>> for $name<'c> {
            type Error = crate::Error;
            fn try_from(attribute: crate::mlir::ir::attr::Attribute<'c>) -> Result<Self, Self::Error> {
                if attribute.$is_type() {
                    Ok(unsafe { Self::from_raw(attribute.to_raw()) })
                } else {
                    Err(Error::AttributeExpected($string, attribute.to_string()))
                }
            }
        }
        impl<'c> crate::mlir::ir::attr::AttributeLike<'c> for $name<'c> {
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
    attr: Attribute<'c>,
}
impl<'c> ArrayAttribute<'c> {
    pub fn new(ctx: &'c Context, xs: &[Attribute<'c>]) -> Self {
        unsafe {
            Self::from_raw(mlirArrayAttrGet(
                ctx.to_raw(),
                xs.len() as isize,
                xs.as_ptr() as *const _ as *const _,
            ))
        }
    }
    pub fn len(&self) -> usize {
        (unsafe { mlirArrayAttrGetNumElements(self.attr.to_raw()) }) as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn element(&self, idx: usize) -> Result<Attribute<'c>, Error> {
        if idx < self.len() {
            Ok(unsafe { Attribute::from_raw(mlirArrayAttrGetElement(self.attr.to_raw(), idx as isize)) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "array element",
                val: self.to_string(),
                idx,
            })
        }
    }
}
attribute_traits!(ArrayAttribute, is_dense_i64_array, "dense i64 array");
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlir::ir::{attr::IntegerAttribute, r#type::IntegerType, Type};
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

pub struct Block<'c> {
    raw: MlirBlock,
    _context: PhantomData<&'c Context>,
}
impl<'c> Block<'c> {
    pub fn new(arguments: &[(Type<'c>, Location<'c>)]) -> Self {
        unsafe {
            Self::from_raw(mlirBlockCreate(
                arguments.len() as isize,
                into_raw_array(arguments.iter().map(|(argument, _)| argument.to_raw()).collect()),
                into_raw_array(arguments.iter().map(|(_, location)| location.to_raw()).collect()),
            ))
        }
    }
    pub fn argument(&self, index: usize) -> Result<BlockArgument, Error> {
        unsafe {
            if index < self.argument_count() {
                Ok(BlockArgument::from_raw(mlirBlockGetArgument(self.raw, index as isize)))
            } else {
                Err(Error::PositionOutOfBounds {
                    name: "block argument",
                    value: self.to_string(),
                    index,
                })
            }
        }
    }
    pub fn argument_count(&self) -> usize {
        unsafe { mlirBlockGetNumArguments(self.raw) as usize }
    }
    pub fn first_operation(&self) -> Option<OperationRef> {
        unsafe {
            let operation = mlirBlockGetFirstOperation(self.raw);
            if operation.ptr.is_null() {
                None
            } else {
                Some(OperationRef::from_raw(operation))
            }
        }
    }
    pub fn terminator(&self) -> Option<OperationRef> {
        unsafe { OperationRef::from_option_raw(mlirBlockGetTerminator(self.raw)) }
    }
    pub fn parent_region(&self) -> Option<RegionRef> {
        unsafe { RegionRef::from_option_raw(mlirBlockGetParentRegion(self.raw)) }
    }
    pub fn parent_operation(&self) -> Option<OperationRef> {
        unsafe { OperationRef::from_option_raw(mlirBlockGetParentOperation(self.raw)) }
    }
    pub fn add_argument(&self, r#type: Type<'c>, location: Location<'c>) -> Value {
        unsafe { Value::from_raw(mlirBlockAddArgument(self.raw, r#type.to_raw(), location.to_raw())) }
    }
    pub fn append_operation(&self, operation: Operation) -> OperationRef {
        unsafe {
            let operation = operation.into_raw();
            mlirBlockAppendOwnedOperation(self.raw, operation);
            OperationRef::from_raw(operation)
        }
    }
    pub fn insert_operation(&self, position: usize, operation: Operation) -> OperationRef {
        unsafe {
            let operation = operation.into_raw();
            mlirBlockInsertOwnedOperation(self.raw, position as isize, operation);
            OperationRef::from_raw(operation)
        }
    }
    pub fn insert_operation_after(&self, one: OperationRef, other: Operation) -> OperationRef {
        unsafe {
            let other = other.into_raw();
            mlirBlockInsertOwnedOperationAfter(self.raw, one.to_raw(), other);
            OperationRef::from_raw(other)
        }
    }
    pub fn insert_operation_before(&self, one: OperationRef, other: Operation) -> OperationRef {
        unsafe {
            let other = other.into_raw();
            mlirBlockInsertOwnedOperationBefore(self.raw, one.to_raw(), other);
            OperationRef::from_raw(other)
        }
    }
    pub unsafe fn detach(&self) -> Option<Block> {
        if self.parent_region().is_some() {
            mlirBlockDetach(self.raw);
            Some(Block::from_raw(self.raw))
        } else {
            None
        }
    }
    pub fn next_in_region(&self) -> Option<BlockRef> {
        unsafe { BlockRef::from_option_raw(mlirBlockGetNextInRegion(self.raw)) }
    }
    pub unsafe fn from_raw(raw: MlirBlock) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
    pub fn into_raw(self) -> MlirBlock {
        let block = self.raw;
        forget(self);
        block
    }
    pub fn to_raw(&self) -> MlirBlock {
        self.raw
    }
}
impl<'c> Drop for Block<'c> {
    fn drop(&mut self) {
        unsafe { mlirBlockDestroy(self.raw) };
    }
}
impl<'c> PartialEq for Block<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirBlockEqual(self.raw, other.raw) }
    }
}
impl<'c> Eq for Block<'c> {}
impl<'c> Display for Block<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));
        unsafe {
            mlirBlockPrint(self.raw, Some(print_callback), &mut data as *mut _ as *mut c_void);
        }
        data.1
    }
}
impl<'c> Debug for Block<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        writeln!(formatter, "Block(")?;
        Display::fmt(self, formatter)?;
        write!(formatter, ")")
    }
}

#[derive(Clone, Copy)]
pub struct BlockRef<'a> {
    raw: MlirBlock,
    _reference: PhantomData<&'a Block<'a>>,
}
impl<'c> BlockRef<'c> {
    pub unsafe fn from_raw(raw: MlirBlock) -> Self {
        Self {
            raw,
            _reference: Default::default(),
        }
    }
    pub unsafe fn from_option_raw(raw: MlirBlock) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}
impl<'a> Deref for BlockRef<'a> {
    type Target = Block<'a>;
    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
    }
}
impl<'a> PartialEq for BlockRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirBlockEqual(self.raw, other.raw) }
    }
}
impl<'a> Eq for BlockRef<'a> {}
impl<'a> Display for BlockRef<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self.deref(), formatter)
    }
}
impl<'a> Debug for BlockRef<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Debug::fmt(self.deref(), formatter)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BlockArgument<'a> {
    value: Value<'a>,
}
impl<'a> BlockArgument<'a> {
    pub fn argument_number(&self) -> usize {
        unsafe { mlirBlockArgumentGetArgNumber(self.value.to_raw()) as usize }
    }
    pub fn owner(&self) -> BlockRef {
        unsafe { BlockRef::from_raw(mlirBlockArgumentGetOwner(self.value.to_raw())) }
    }
    pub fn set_type(&self, r#type: Type) {
        unsafe { mlirBlockArgumentSetType(self.value.to_raw(), r#type.to_raw()) }
    }
    pub unsafe fn from_raw(value: MlirValue) -> Self {
        Self {
            value: Value::from_raw(value),
        }
    }
}
impl<'a> ValueLike for BlockArgument<'a> {
    fn to_raw(&self) -> MlirValue {
        self.value.to_raw()
    }
}
impl<'a> Display for BlockArgument<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Value::from(*self).fmt(formatter)
    }
}
impl<'a> TryFrom<Value<'a>> for BlockArgument<'a> {
    type Error = Error;
    fn try_from(value: Value<'a>) -> Result<Self, Self::Error> {
        if value.is_block_argument() {
            Ok(Self { value })
        } else {
            Err(Error::BlockArgumentExpected(value.to_string()))
        }
    }
}

pub struct Operation<'c> {
    raw: MlirOperation,
    _context: PhantomData<&'c Context>,
}
impl<'c> Operation<'c> {
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirOperationGetContext(self.raw)) }
    }
    pub fn name(&self) -> Identifier<'c> {
        unsafe { Identifier::from_raw(mlirOperationGetName(self.raw)) }
    }
    pub fn block(&self) -> Option<BlockRef> {
        unsafe { BlockRef::from_option_raw(mlirOperationGetBlock(self.raw)) }
    }
    pub fn result(&self, index: usize) -> Result<OperationResult, Error> {
        unsafe {
            if index < self.result_count() {
                Ok(OperationResult::from_raw(mlirOperationGetResult(
                    self.raw,
                    index as isize,
                )))
            } else {
                Err(Error::PositionOutOfBounds {
                    name: "operation result",
                    value: self.to_string(),
                    index,
                })
            }
        }
    }
    pub fn result_count(&self) -> usize {
        unsafe { mlirOperationGetNumResults(self.raw) as usize }
    }
    pub fn region(&self, index: usize) -> Result<RegionRef, Error> {
        unsafe {
            if index < self.region_count() {
                Ok(RegionRef::from_raw(mlirOperationGetRegion(self.raw, index as isize)))
            } else {
                Err(Error::PositionOutOfBounds {
                    name: "region",
                    value: self.to_string(),
                    index,
                })
            }
        }
    }
    pub fn region_count(&self) -> usize {
        unsafe { mlirOperationGetNumRegions(self.raw) as usize }
    }
    pub fn next_in_block(&self) -> Option<OperationRef> {
        unsafe {
            let operation = mlirOperationGetNextInBlock(self.raw);
            if operation.ptr.is_null() {
                None
            } else {
                Some(OperationRef::from_raw(operation))
            }
        }
    }
    pub fn verify(&self) -> bool {
        unsafe { mlirOperationVerify(self.raw) }
    }
    pub fn dump(&self) {
        unsafe { mlirOperationDump(self.raw) }
    }
    pub fn to_string_with_flags(&self, flags: OperationPrintingFlags) -> Result<String, Error> {
        let mut data = (String::new(), Ok::<_, Error>(()));
        unsafe {
            mlirOperationPrintWithFlags(
                self.raw,
                flags.to_raw(),
                Some(print_string_callback),
                &mut data as *mut _ as *mut _,
            );
        }
        data.1?;
        Ok(data.0)
    }
    pub unsafe fn from_raw(raw: MlirOperation) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
    pub fn into_raw(self) -> MlirOperation {
        let operation = self.raw;
        forget(self);
        operation
    }
}
impl<'c> Clone for Operation<'c> {
    fn clone(&self) -> Self {
        unsafe { Self::from_raw(mlirOperationClone(self.raw)) }
    }
}
impl<'c> Drop for Operation<'c> {
    fn drop(&mut self) {
        unsafe { mlirOperationDestroy(self.raw) };
    }
}
impl<'c> PartialEq for Operation<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirOperationEqual(self.raw, other.raw) }
    }
}
impl<'c> Eq for Operation<'c> {}
impl<'a> Display for Operation<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));
        unsafe {
            mlirOperationPrint(self.raw, Some(print_callback), &mut data as *mut _ as *mut c_void);
        }
        data.1
    }
}
impl<'c> Debug for Operation<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        writeln!(formatter, "Operation(")?;
        Display::fmt(self, formatter)?;
        write!(formatter, ")")
    }
}

#[derive(Clone, Copy)]
pub struct OperationRef<'a> {
    raw: MlirOperation,
    _reference: PhantomData<&'a Operation<'a>>,
}
impl<'a> OperationRef<'a> {
    pub fn result(self, index: usize) -> Result<OperationResult<'a>, Error> {
        unsafe { transmute(self.deref().result(index)) }
    }
    pub fn to_raw(self) -> MlirOperation {
        self.raw
    }
    pub unsafe fn from_raw(raw: MlirOperation) -> Self {
        Self {
            raw,
            _reference: Default::default(),
        }
    }
    pub unsafe fn from_option_raw(raw: MlirOperation) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}
impl<'a> Deref for OperationRef<'a> {
    type Target = Operation<'a>;
    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
    }
}
impl<'a> PartialEq for OperationRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirOperationEqual(self.raw, other.raw) }
    }
}
impl<'a> Eq for OperationRef<'a> {}
impl<'a> Display for OperationRef<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self.deref(), formatter)
    }
}
impl<'a> Debug for OperationRef<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Debug::fmt(self.deref(), formatter)
    }
}

pub struct OperationBuilder<'c> {
    raw: MlirOperationState,
    _context: PhantomData<&'c Context>,
}
impl<'c> OperationBuilder<'c> {
    pub fn new(name: &str, location: Location<'c>) -> Self {
        Self {
            raw: unsafe { mlirOperationStateGet(StringRef::from(name).to_raw(), location.to_raw()) },
            _context: Default::default(),
        }
    }
    pub fn add_results(mut self, results: &[Type<'c>]) -> Self {
        unsafe {
            mlirOperationStateAddResults(
                &mut self.raw,
                results.len() as isize,
                into_raw_array(results.iter().map(|r#type| r#type.to_raw()).collect()),
            )
        }
        self
    }
    pub fn add_operands(mut self, operands: &[Value]) -> Self {
        unsafe {
            mlirOperationStateAddOperands(
                &mut self.raw,
                operands.len() as isize,
                into_raw_array(operands.iter().map(|value| value.to_raw()).collect()),
            )
        }
        self
    }
    pub fn add_regions(mut self, regions: Vec<Region>) -> Self {
        unsafe {
            mlirOperationStateAddOwnedRegions(
                &mut self.raw,
                regions.len() as isize,
                into_raw_array(regions.into_iter().map(|region| region.into_raw()).collect()),
            )
        }
        self
    }
    pub fn add_successors(mut self, successors: &[&Block<'c>]) -> Self {
        unsafe {
            mlirOperationStateAddSuccessors(
                &mut self.raw,
                successors.len() as isize,
                into_raw_array(successors.iter().map(|block| block.to_raw()).collect()),
            )
        }
        self
    }
    pub fn add_attributes(mut self, attributes: &[(Identifier, Attribute<'c>)]) -> Self {
        unsafe {
            mlirOperationStateAddAttributes(
                &mut self.raw,
                attributes.len() as isize,
                into_raw_array(
                    attributes
                        .iter()
                        .map(|(identifier, attribute)| mlirNamedAttributeGet(identifier.to_raw(), attribute.to_raw()))
                        .collect(),
                ),
            )
        }
        self
    }
    pub fn enable_result_type_inference(mut self) -> Self {
        unsafe { mlirOperationStateEnableResultTypeInference(&mut self.raw) }
        self
    }
    pub fn build(mut self) -> Operation<'c> {
        unsafe { Operation::from_raw(mlirOperationCreate(&mut self.raw)) }
    }
}

#[derive(Debug)]
pub struct OperationPrintingFlags(MlirOpPrintingFlags);
impl OperationPrintingFlags {
    pub fn new() -> Self {
        Self(unsafe { mlirOpPrintingFlagsCreate() })
    }
    pub fn elide_large_elements_attributes(self, limit: usize) -> Self {
        unsafe { mlirOpPrintingFlagsElideLargeElementsAttrs(self.0, limit as isize) }
        self
    }
    pub fn enable_debug_info(self, enabled: bool, pretty_form: bool) -> Self {
        unsafe { mlirOpPrintingFlagsEnableDebugInfo(self.0, enabled, pretty_form) }
        self
    }
    pub fn print_generic_operation_form(self) -> Self {
        unsafe { mlirOpPrintingFlagsPrintGenericOpForm(self.0) }
        self
    }
    pub fn use_local_scope(self) -> Self {
        unsafe { mlirOpPrintingFlagsUseLocalScope(self.0) }
        self
    }
    pub fn to_raw(&self) -> MlirOpPrintingFlags {
        self.0
    }
}
impl Drop for OperationPrintingFlags {
    fn drop(&mut self) {
        unsafe { mlirOpPrintingFlagsDestroy(self.0) }
    }
}
impl Default for OperationPrintingFlags {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OperationResult<'a> {
    value: Value<'a>,
}
impl<'a> OperationResult<'a> {
    pub fn result_number(&self) -> usize {
        unsafe { mlirOpResultGetResultNumber(self.value.to_raw()) as usize }
    }
    pub fn owner(&self) -> OperationRef {
        unsafe { OperationRef::from_raw(mlirOpResultGetOwner(self.value.to_raw())) }
    }
    pub unsafe fn from_raw(value: MlirValue) -> Self {
        Self {
            value: Value::from_raw(value),
        }
    }
}
impl<'a> ValueLike for OperationResult<'a> {
    fn to_raw(&self) -> MlirValue {
        self.value.to_raw()
    }
}
impl<'a> Display for OperationResult<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Value::from(*self).fmt(formatter)
    }
}
impl<'a> TryFrom<Value<'a>> for OperationResult<'a> {
    type Error = Error;
    fn try_from(value: Value<'a>) -> Result<Self, Self::Error> {
        if value.is_operation_result() {
            Ok(Self { value })
        } else {
            Err(Error::OperationResultExpected(value.to_string()))
        }
    }
}

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
    macros::type_check_functions!(
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
        impl<'c> TryFrom<crate::mlir::ir::r#type::Type<'c>> for $name<'c> {
            type Error = crate::Error;
            fn try_from(r#type: crate::mlir::ir::r#type::Type<'c>) -> Result<Self, Self::Error> {
                if r#type.$is_type() {
                    Ok(unsafe { Self::from_raw(r#type.to_raw()) })
                } else {
                    Err(Error::TypeExpected($string, r#type.to_string()))
                }
            }
        }
        impl<'c> crate::mlir::ir::r#type::TypeLike<'c> for $name<'c> {
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

#[derive(Clone, Copy)]
pub struct Value<'a> {
    raw: MlirValue,
    _parent: PhantomData<&'a ()>,
}
impl<'a> Value<'a> {
    pub unsafe fn from_raw(value: MlirValue) -> Self {
        Self {
            raw: value,
            _parent: Default::default(),
        }
    }
}
impl<'a> ValueLike for Value<'a> {
    fn to_raw(&self) -> MlirValue {
        self.raw
    }
}
impl<'a> PartialEq for Value<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirValueEqual(self.raw, other.raw) }
    }
}
impl<'a> Eq for Value<'a> {}
impl<'a> Display for Value<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));
        unsafe {
            mlirValuePrint(self.raw, Some(print_callback), &mut data as *mut _ as *mut c_void);
        }
        data.1
    }
}
impl<'a> Debug for Value<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        writeln!(formatter, "Value(")?;
        Display::fmt(self, formatter)?;
        write!(formatter, ")")
    }
}

from_raw_subtypes!(Value, BlockArgument, OperationResult);

#[cfg(test)]
mod tests;
