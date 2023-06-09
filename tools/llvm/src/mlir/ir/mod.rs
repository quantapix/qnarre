use macros::attribute_check_functions;
use mlir_lib::*;
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::{forget, transmute},
    ops::Deref,
};

use crate::mlir::{
    utils::{into_raw_array, print_callback, print_string_callback},
    Context, ContextRef, Error, StringRef,
};

#[derive(Clone, Copy)]
pub struct AffineMap<'c> {
    raw: MlirAffineMap,
    _ctx: PhantomData<&'c Context>,
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
            _ctx: Default::default(),
        }
    }
}
impl<'c> PartialEq for AffineMap<'c> {
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirAffineMapEqual(self.raw, x.raw) }
    }
}
impl<'c> Eq for AffineMap<'c> {}
impl<'c> Display for AffineMap<'c> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut y = (f, Ok(()));
        unsafe {
            mlirAffineMapPrint(self.raw, Some(print_callback), &mut y as *mut _ as *mut c_void);
        }
        y.1
    }
}
impl<'c> Debug for AffineMap<'c> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Identifier<'c> {
    raw: MlirIdentifier,
    _ctx: PhantomData<&'c Context>,
}
impl<'c> Identifier<'c> {
    pub fn new(c: &Context, name: &str) -> Self {
        unsafe { Self::from_raw(mlirIdentifierGet(c.to_raw(), StringRef::from(name).to_raw())) }
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
            _ctx: Default::default(),
        }
    }
    pub fn to_raw(self) -> MlirIdentifier {
        self.raw
    }
}
impl<'c> PartialEq for Identifier<'c> {
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirIdentifierEqual(self.raw, x.raw) }
    }
}
impl<'c> Eq for Identifier<'c> {}

#[derive(Clone, Copy, Debug)]
pub struct Location<'c> {
    raw: MlirLocation,
    _ctx: PhantomData<&'c Context>,
}
impl<'c> Location<'c> {
    pub fn new(c: &'c Context, file: &str, line: usize, col: usize) -> Self {
        unsafe {
            Self::from_raw(mlirLocationFileLineColGet(
                c.to_raw(),
                StringRef::from(file).to_raw(),
                line as u32,
                col as u32,
            ))
        }
    }
    pub fn fused(c: &'c Context, locs: &[Self], a: Attribute) -> Self {
        unsafe {
            Self::from_raw(mlirLocationFusedGet(
                c.to_raw(),
                locs.len() as isize,
                into_raw_array(locs.iter().map(|x| x.to_raw()).collect()),
                a.to_raw(),
            ))
        }
    }
    pub fn name(c: &'c Context, name: &str, child: Location) -> Self {
        unsafe {
            Self::from_raw(mlirLocationNameGet(
                c.to_raw(),
                StringRef::from(name).to_raw(),
                child.to_raw(),
            ))
        }
    }
    pub fn unknown(c: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirLocationUnknownGet(c.to_raw())) }
    }
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirLocationGetContext(self.raw)) }
    }
    pub unsafe fn from_raw(raw: MlirLocation) -> Self {
        Self {
            raw,
            _ctx: Default::default(),
        }
    }
    pub fn to_raw(self) -> MlirLocation {
        self.raw
    }
}
impl<'c> PartialEq for Location<'c> {
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirLocationEqual(self.raw, x.raw) }
    }
}
impl<'c> Display for Location<'c> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut y = (f, Ok(()));
        unsafe {
            mlirLocationPrint(self.raw, Some(print_callback), &mut y as *mut _ as *mut c_void);
        }
        y.1
    }
}

#[derive(Debug)]
pub struct Module<'c> {
    raw: MlirModule,
    _ctx: PhantomData<&'c Context>,
}
impl<'c> Module<'c> {
    pub fn new(loc: Location) -> Self {
        unsafe { Self::from_raw(mlirModuleCreateEmpty(loc.to_raw())) }
    }
    pub fn parse(c: &Context, source: &str) -> Option<Self> {
        unsafe { Self::from_option_raw(mlirModuleCreateParse(c.to_raw(), StringRef::from(source).to_raw())) }
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
    pub fn from_operation(o: Operation) -> Option<Self> {
        unsafe { Self::from_option_raw(mlirModuleFromOperation(o.into_raw())) }
    }
    pub unsafe fn from_raw(raw: MlirModule) -> Self {
        Self {
            raw,
            _ctx: Default::default(),
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
            let y = mlirRegionGetFirstBlock(self.raw);
            if y.ptr.is_null() {
                None
            } else {
                Some(BlockRef::from_raw(y))
            }
        }
    }
    pub fn insert_block_after(&self, one: BlockRef, other: Block) -> BlockRef {
        unsafe {
            let y = BlockRef::from_raw(other.to_raw());
            mlirRegionInsertOwnedBlockAfter(self.raw, one.to_raw(), other.into_raw());
            y
        }
    }
    pub fn insert_block_before(&self, one: BlockRef, other: Block) -> BlockRef {
        unsafe {
            let y = BlockRef::from_raw(other.to_raw());
            mlirRegionInsertOwnedBlockBefore(self.raw, one.to_raw(), other.into_raw());
            y
        }
    }
    pub fn append_block(&self, block: Block) -> BlockRef {
        unsafe {
            let y = BlockRef::from_raw(block.to_raw());
            mlirRegionAppendOwnedBlock(self.raw, block.into_raw());
            y
        }
    }
    pub fn into_raw(self) -> MlirRegion {
        let y = self.raw;
        forget(self);
        y
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
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, x.raw) }
    }
}
impl Eq for Region {}

#[derive(Clone, Copy, Debug)]
pub struct RegionRef<'a> {
    raw: MlirRegion,
    _reg: PhantomData<&'a Region>,
}
impl<'a> RegionRef<'a> {
    pub unsafe fn from_raw(raw: MlirRegion) -> Self {
        Self {
            raw,
            _reg: Default::default(),
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
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, x.raw) }
    }
}
impl<'a> Eq for RegionRef<'a> {}

pub trait AttributeLike<'c> {
    fn to_raw(&self) -> MlirAttribute;
    fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirAttributeGetContext(self.to_raw())) }
    }
    fn ty(&self) -> Type {
        unsafe { Type::from_raw(mlirAttributeGetType(self.to_raw())) }
    }
    fn type_id(&self) -> TypeId {
        unsafe { TypeId::from_raw(mlirAttributeGetTypeID(self.to_raw())) }
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

macro_rules! attr_traits {
    ($name: ident, $is_type: ident, $string: expr) => {
        impl<'c> $name<'c> {
            unsafe fn from_raw(raw: MlirAttribute) -> Self {
                Self {
                    attr: Attribute::from_raw(raw),
                }
            }
        }
        impl<'c> TryFrom<crate::mlir::ir::Attribute<'c>> for $name<'c> {
            type Error = crate::mlir::Error;
            fn try_from(a: crate::mlir::ir::Attribute<'c>) -> Result<Self, Self::Error> {
                if a.$is_type() {
                    Ok(unsafe { Self::from_raw(a.to_raw()) })
                } else {
                    Err(Error::AttributeExpected($string, a.to_string()))
                }
            }
        }
        impl<'c> crate::mlir::ir::AttributeLike<'c> for $name<'c> {
            fn to_raw(&self) -> mlir_lib::MlirAttribute {
                self.attribute.to_raw()
            }
        }
        impl<'c> std::fmt::Display for $name<'c> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.attribute, f)
            }
        }
        impl<'c> std::fmt::Debug for $name<'c> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(self, f)
            }
        }
    };
}

#[derive(Clone, Copy)]
pub struct ArrayAttribute<'c> {
    attr: Attribute<'c>,
}
impl<'c> ArrayAttribute<'c> {
    pub fn new(c: &'c Context, xs: &[Attribute<'c>]) -> Self {
        unsafe {
            Self::from_raw(mlirArrayAttrGet(
                c.to_raw(),
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
attr_traits!(ArrayAttribute, is_dense_i64_array, "dense i64 array");

#[derive(Clone, Copy)]
pub struct DenseElementsAttribute<'c> {
    attr: Attribute<'c>,
}
impl<'c> DenseElementsAttribute<'c> {
    pub fn new(t: Type<'c>, xs: &[Attribute<'c>]) -> Result<Self, Error> {
        if t.is_shaped() {
            Ok(unsafe {
                Self::from_raw(mlirDenseElementsAttrGet(
                    t.to_raw(),
                    xs.len() as isize,
                    xs.as_ptr() as *const _ as *const _,
                ))
            })
        } else {
            Err(Error::TypeExpected("shaped", t.to_string()))
        }
    }
    pub fn len(&self) -> usize {
        (unsafe { mlirElementsAttrGetNumElements(self.attr.to_raw()) }) as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn i32_element(&self, idx: usize) -> Result<i32, Error> {
        if !self.is_dense_int_elements() {
            Err(Error::ElementExpected {
                ty: "integer",
                value: self.to_string(),
            })
        } else if idx < self.len() {
            Ok(unsafe { mlirDenseElementsAttrGetInt32Value(self.attr.to_raw(), idx as isize) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index: idx,
            })
        }
    }
    pub fn i64_element(&self, idx: usize) -> Result<i64, Error> {
        if !self.is_dense_int_elements() {
            Err(Error::ElementExpected {
                ty: "integer",
                value: self.to_string(),
            })
        } else if idx < self.len() {
            Ok(unsafe { mlirDenseElementsAttrGetInt64Value(self.attr.to_raw(), idx as isize) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index: idx,
            })
        }
    }
}
attr_traits!(DenseElementsAttribute, is_dense_elements, "dense elements");

#[derive(Clone, Copy)]
pub struct DenseI32ArrayAttribute<'c> {
    attr: Attribute<'c>,
}
impl<'c> DenseI32ArrayAttribute<'c> {
    pub fn new(c: &'c Context, values: &[i32]) -> Self {
        unsafe { Self::from_raw(mlirDenseI32ArrayGet(c.to_raw(), values.len() as isize, values.as_ptr())) }
    }
    pub fn len(&self) -> usize {
        (unsafe { mlirArrayAttrGetNumElements(self.attr.to_raw()) }) as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn element(&self, idx: usize) -> Result<i32, Error> {
        if idx < self.len() {
            Ok(unsafe { mlirDenseI32ArrayGetElement(self.attr.to_raw(), idx as isize) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "array element",
                value: self.to_string(),
                index: idx,
            })
        }
    }
}
attr_traits!(DenseI32ArrayAttribute, is_dense_i32_array, "dense i32 array");

#[derive(Clone, Copy)]
pub struct DenseI64ArrayAttribute<'c> {
    attr: Attribute<'c>,
}
impl<'c> DenseI64ArrayAttribute<'c> {
    pub fn new(c: &'c Context, xs: &[i64]) -> Self {
        unsafe { Self::from_raw(mlirDenseI64ArrayGet(c.to_raw(), xs.len() as isize, xs.as_ptr())) }
    }
    pub fn len(&self) -> usize {
        (unsafe { mlirArrayAttrGetNumElements(self.attr.to_raw()) }) as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn element(&self, idx: usize) -> Result<i64, Error> {
        if idx < self.len() {
            Ok(unsafe { mlirDenseI64ArrayGetElement(self.attr.to_raw(), idx as isize) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "array element",
                value: self.to_string(),
                index: idx,
            })
        }
    }
}
attr_traits!(DenseI64ArrayAttribute, is_dense_i64_array, "dense i64 array");

#[derive(Clone, Copy)]
pub struct FlatSymbolRefAttribute<'c> {
    attr: Attribute<'c>,
}
impl<'c> FlatSymbolRefAttribute<'c> {
    pub fn new(c: &'c Context, symbol: &str) -> Self {
        unsafe { Self::from_raw(mlirFlatSymbolRefAttrGet(c.to_raw(), StringRef::from(symbol).to_raw())) }
    }
    pub fn value(&self) -> &str {
        unsafe { StringRef::from_raw(mlirFlatSymbolRefAttrGetValue(self.to_raw())) }
            .as_str()
            .unwrap()
    }
}
attr_traits!(FlatSymbolRefAttribute, is_flat_symbol_ref, "flat symbol ref");

#[derive(Clone, Copy)]
pub struct FloatAttribute<'c> {
    attr: Attribute<'c>,
}
impl<'c> FloatAttribute<'c> {
    pub fn new(c: &'c Context, x: f64, t: Type<'c>) -> Self {
        unsafe { Self::from_raw(mlirFloatAttrDoubleGet(c.to_raw(), t.to_raw(), x)) }
    }
}
attr_traits!(FloatAttribute, is_float, "float");

#[derive(Clone, Copy)]
pub struct IntegerAttribute<'c> {
    attr: Attribute<'c>,
}
impl<'c> IntegerAttribute<'c> {
    pub fn new(x: i64, t: Type<'c>) -> Self {
        unsafe { Self::from_raw(mlirIntegerAttrGet(t.to_raw(), x)) }
    }
}
attr_traits!(IntegerAttribute, is_integer, "integer");

#[derive(Clone, Copy)]
pub struct StringAttribute<'c> {
    attr: Attribute<'c>,
}
impl<'c> StringAttribute<'c> {
    pub fn new(c: &'c Context, x: &str) -> Self {
        unsafe { Self::from_raw(mlirStringAttrGet(c.to_raw(), StringRef::from(x).to_raw())) }
    }
}
attr_traits!(StringAttribute, is_string, "string");

#[derive(Clone, Copy)]
pub struct TypeAttribute<'c> {
    attr: Attribute<'c>,
}
impl<'c> TypeAttribute<'c> {
    pub fn new(t: Type<'c>) -> Self {
        unsafe { Self::from_raw(mlirTypeAttrGet(t.to_raw())) }
    }
    pub fn value(&self) -> Type<'c> {
        unsafe { Type::from_raw(mlirTypeAttrGetValue(self.to_raw())) }
    }
}
attr_traits!(TypeAttribute, is_type, "type");

pub struct Block<'c> {
    raw: MlirBlock,
    _ctx: PhantomData<&'c Context>,
}
impl<'c> Block<'c> {
    pub fn new(xs: &[(Type<'c>, Location<'c>)]) -> Self {
        unsafe {
            Self::from_raw(mlirBlockCreate(
                xs.len() as isize,
                into_raw_array(xs.iter().map(|(x, _)| x.to_raw()).collect()),
                into_raw_array(xs.iter().map(|(_, x)| x.to_raw()).collect()),
            ))
        }
    }
    pub fn argument(&self, idx: usize) -> Result<BlockArgument, Error> {
        unsafe {
            if idx < self.argument_count() {
                Ok(BlockArgument::from_raw(mlirBlockGetArgument(self.raw, idx as isize)))
            } else {
                Err(Error::PositionOutOfBounds {
                    name: "block argument",
                    value: self.to_string(),
                    index: idx,
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
    pub fn add_argument(&self, t: Type<'c>, loc: Location<'c>) -> Value {
        unsafe { Value::from_raw(mlirBlockAddArgument(self.raw, t.to_raw(), loc.to_raw())) }
    }
    pub fn append_operation(&self, o: Operation) -> OperationRef {
        unsafe {
            let y = o.into_raw();
            mlirBlockAppendOwnedOperation(self.raw, y);
            OperationRef::from_raw(y)
        }
    }
    pub fn insert_operation(&self, pos: usize, o: Operation) -> OperationRef {
        unsafe {
            let y = o.into_raw();
            mlirBlockInsertOwnedOperation(self.raw, pos as isize, y);
            OperationRef::from_raw(y)
        }
    }
    pub fn insert_operation_after(&self, one: OperationRef, other: Operation) -> OperationRef {
        unsafe {
            let y = other.into_raw();
            mlirBlockInsertOwnedOperationAfter(self.raw, one.to_raw(), y);
            OperationRef::from_raw(y)
        }
    }
    pub fn insert_operation_before(&self, one: OperationRef, other: Operation) -> OperationRef {
        unsafe {
            let y = other.into_raw();
            mlirBlockInsertOwnedOperationBefore(self.raw, one.to_raw(), y);
            OperationRef::from_raw(y)
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
            _ctx: Default::default(),
        }
    }
    pub fn into_raw(self) -> MlirBlock {
        let y = self.raw;
        forget(self);
        y
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
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirBlockEqual(self.raw, x.raw) }
    }
}
impl<'c> Eq for Block<'c> {}
impl<'c> Display for Block<'c> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut y = (f, Ok(()));
        unsafe {
            mlirBlockPrint(self.raw, Some(print_callback), &mut y as *mut _ as *mut c_void);
        }
        y.1
    }
}
impl<'c> Debug for Block<'c> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "Block(")?;
        Display::fmt(self, f)?;
        write!(f, ")")
    }
}

#[derive(Clone, Copy)]
pub struct BlockRef<'a> {
    raw: MlirBlock,
    _ref: PhantomData<&'a Block<'a>>,
}
impl<'c> BlockRef<'c> {
    pub unsafe fn from_raw(raw: MlirBlock) -> Self {
        Self {
            raw,
            _ref: Default::default(),
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
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirBlockEqual(self.raw, x.raw) }
    }
}
impl<'a> Eq for BlockRef<'a> {}
impl<'a> Display for BlockRef<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self.deref(), f)
    }
}
impl<'a> Debug for BlockRef<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Debug::fmt(self.deref(), f)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BlockArgument<'a> {
    val: Value<'a>,
}
impl<'a> BlockArgument<'a> {
    pub fn argument_number(&self) -> usize {
        unsafe { mlirBlockArgumentGetArgNumber(self.val.to_raw()) as usize }
    }
    pub fn owner(&self) -> BlockRef {
        unsafe { BlockRef::from_raw(mlirBlockArgumentGetOwner(self.val.to_raw())) }
    }
    pub fn set_type(&self, t: Type) {
        unsafe { mlirBlockArgumentSetType(self.val.to_raw(), t.to_raw()) }
    }
    pub unsafe fn from_raw(raw: MlirValue) -> Self {
        Self {
            val: Value::from_raw(raw),
        }
    }
}
impl<'a> ValueLike for BlockArgument<'a> {
    fn to_raw(&self) -> MlirValue {
        self.val.to_raw()
    }
}
impl<'a> Display for BlockArgument<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Value::from(*self).fmt(f)
    }
}
impl<'a> TryFrom<Value<'a>> for BlockArgument<'a> {
    type Error = Error;
    fn try_from(val: Value<'a>) -> Result<Self, Self::Error> {
        if val.is_block_argument() {
            Ok(Self { val })
        } else {
            Err(Error::BlockArgumentExpected(val.to_string()))
        }
    }
}

pub struct Operation<'c> {
    raw: MlirOperation,
    _ctx: PhantomData<&'c Context>,
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
    pub fn result(&self, idx: usize) -> Result<OperationResult, Error> {
        unsafe {
            if idx < self.result_count() {
                Ok(OperationResult::from_raw(mlirOperationGetResult(
                    self.raw,
                    idx as isize,
                )))
            } else {
                Err(Error::PositionOutOfBounds {
                    name: "operation result",
                    value: self.to_string(),
                    index: idx,
                })
            }
        }
    }
    pub fn result_count(&self) -> usize {
        unsafe { mlirOperationGetNumResults(self.raw) as usize }
    }
    pub fn region(&self, idx: usize) -> Result<RegionRef, Error> {
        unsafe {
            if idx < self.region_count() {
                Ok(RegionRef::from_raw(mlirOperationGetRegion(self.raw, idx as isize)))
            } else {
                Err(Error::PositionOutOfBounds {
                    name: "region",
                    value: self.to_string(),
                    index: idx,
                })
            }
        }
    }
    pub fn region_count(&self) -> usize {
        unsafe { mlirOperationGetNumRegions(self.raw) as usize }
    }
    pub fn next_in_block(&self) -> Option<OperationRef> {
        unsafe {
            let y = mlirOperationGetNextInBlock(self.raw);
            if y.ptr.is_null() {
                None
            } else {
                Some(OperationRef::from_raw(y))
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
        let mut y = (String::new(), Ok::<_, Error>(()));
        unsafe {
            mlirOperationPrintWithFlags(
                self.raw,
                flags.to_raw(),
                Some(print_string_callback),
                &mut y as *mut _ as *mut _,
            );
        }
        y.1?;
        Ok(y.0)
    }
    pub unsafe fn from_raw(raw: MlirOperation) -> Self {
        Self {
            raw,
            _ctx: Default::default(),
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
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirOperationEqual(self.raw, x.raw) }
    }
}
impl<'c> Eq for Operation<'c> {}
impl<'a> Display for Operation<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut y = (f, Ok(()));
        unsafe {
            mlirOperationPrint(self.raw, Some(print_callback), &mut y as *mut _ as *mut c_void);
        }
        y.1
    }
}
impl<'c> Debug for Operation<'c> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "Operation(")?;
        Display::fmt(self, f)?;
        write!(f, ")")
    }
}

#[derive(Clone, Copy)]
pub struct OperationRef<'a> {
    raw: MlirOperation,
    _ref: PhantomData<&'a Operation<'a>>,
}
impl<'a> OperationRef<'a> {
    pub fn result(self, idx: usize) -> Result<OperationResult<'a>, Error> {
        unsafe { transmute(self.deref().result(idx)) }
    }
    pub fn to_raw(self) -> MlirOperation {
        self.raw
    }
    pub unsafe fn from_raw(raw: MlirOperation) -> Self {
        Self {
            raw,
            _ref: Default::default(),
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
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirOperationEqual(self.raw, x.raw) }
    }
}
impl<'a> Eq for OperationRef<'a> {}
impl<'a> Display for OperationRef<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self.deref(), f)
    }
}
impl<'a> Debug for OperationRef<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Debug::fmt(self.deref(), f)
    }
}

pub struct OperationBuilder<'c> {
    raw: MlirOperationState,
    _ctx: PhantomData<&'c Context>,
}
impl<'c> OperationBuilder<'c> {
    pub fn new(name: &str, loc: Location<'c>) -> Self {
        Self {
            raw: unsafe { mlirOperationStateGet(StringRef::from(name).to_raw(), loc.to_raw()) },
            _ctx: Default::default(),
        }
    }
    pub fn add_results(mut self, results: &[Type<'c>]) -> Self {
        unsafe {
            mlirOperationStateAddResults(
                &mut self.raw,
                results.len() as isize,
                into_raw_array(results.iter().map(|ty| ty.to_raw()).collect()),
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
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Value::from(*self).fmt(f)
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
    _ctx: PhantomData<&'c Context>,
}
impl<'c> Type<'c> {
    pub fn parse(c: &'c Context, source: &str) -> Option<Self> {
        unsafe { Self::from_option_raw(mlirTypeParseGet(c.to_raw(), StringRef::from(source).to_raw())) }
    }
    pub fn bfloat16(c: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirBF16TypeGet(c.to_raw())) }
    }
    pub fn float16(c: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirF16TypeGet(c.to_raw())) }
    }
    pub fn float32(c: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirF32TypeGet(c.to_raw())) }
    }
    pub fn float64(c: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirF64TypeGet(c.to_raw())) }
    }
    pub fn index(c: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirIndexTypeGet(c.to_raw())) }
    }
    pub fn none(c: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirNoneTypeGet(c.to_raw())) }
    }
    pub fn vector(dims: &[u64], ty: Self) -> Self {
        unsafe {
            Self::from_raw(mlirVectorTypeGet(
                dims.len() as isize,
                dims.as_ptr() as *const i64,
                ty.raw,
            ))
        }
    }
    pub fn vector_checked(loc: Location<'c>, dims: &[u64], ty: Self) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirVectorTypeGetChecked(
                loc.to_raw(),
                dims.len() as isize,
                dims.as_ptr() as *const i64,
                ty.raw,
            ))
        }
    }
    pub unsafe fn from_raw(raw: MlirType) -> Self {
        Self {
            raw,
            _ctx: Default::default(),
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
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirTypeEqual(self.raw, x.raw) }
    }
}
impl<'c> Eq for Type<'c> {}
impl<'c> Display for Type<'c> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut y = (f, Ok(()));
        unsafe {
            mlirTypePrint(self.raw, Some(print_callback), &mut y as *mut _ as *mut c_void);
        }
        y.1
    }
}
impl<'c> Debug for Type<'c> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Type(")?;
        Display::fmt(self, f)?;
        write!(f, ")")
    }
}

from_raw_subtypes!(Type, FunctionType, IntegerType, MemRefType, RankedTensorType, TupleType);

macro_rules! type_traits {
    ($name: ident, $is_type: ident, $string: expr) => {
        impl<'c> $name<'c> {
            unsafe fn from_raw(raw: MlirType) -> Self {
                Self {
                    ty: Type::from_raw(raw),
                }
            }
        }
        impl<'c> TryFrom<crate::mlir::ir::Type<'c>> for $name<'c> {
            type Error = crate::mlir::Error;
            fn try_from(t: crate::mlir::ir::Type<'c>) -> Result<Self, Self::Error> {
                if t.$is_type() {
                    Ok(unsafe { Self::from_raw(t.to_raw()) })
                } else {
                    Err(Error::TypeExpected($string, t.to_string()))
                }
            }
        }
        impl<'c> crate::mlir::ir::TypeLike<'c> for $name<'c> {
            fn to_raw(&self) -> mlir_lib::MlirType {
                self.ty.to_raw()
            }
        }
        impl<'c> std::fmt::Display for $name<'c> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.ty, f)
            }
        }
    };
}

#[derive(Clone, Copy, Debug)]
pub struct FunctionType<'c> {
    ty: Type<'c>,
}
impl<'c> FunctionType<'c> {
    pub fn new(c: &'c Context, inputs: &[Type<'c>], results: &[Type<'c>]) -> Self {
        Self {
            ty: unsafe {
                Type::from_raw(mlirFunctionTypeGet(
                    c.to_raw(),
                    inputs.len() as isize,
                    into_raw_array(inputs.iter().map(|ty| ty.to_raw()).collect()),
                    results.len() as isize,
                    into_raw_array(results.iter().map(|ty| ty.to_raw()).collect()),
                ))
            },
        }
    }
    pub fn input(&self, idx: usize) -> Result<Type<'c>, Error> {
        if idx < self.input_count() {
            unsafe { Ok(Type::from_raw(mlirFunctionTypeGetInput(self.ty.to_raw(), idx as isize))) }
        } else {
            Err(Error::PositionOutOfBounds {
                name: "function input",
                value: self.to_string(),
                index: idx,
            })
        }
    }
    pub fn result(&self, idx: usize) -> Result<Type<'c>, Error> {
        if idx < self.result_count() {
            unsafe {
                Ok(Type::from_raw(mlirFunctionTypeGetResult(
                    self.ty.to_raw(),
                    idx as isize,
                )))
            }
        } else {
            Err(Error::PositionOutOfBounds {
                name: "function result",
                value: self.to_string(),
                index: idx,
            })
        }
    }
    pub fn input_count(&self) -> usize {
        unsafe { mlirFunctionTypeGetNumInputs(self.ty.to_raw()) as usize }
    }
    pub fn result_count(&self) -> usize {
        unsafe { mlirFunctionTypeGetNumResults(self.ty.to_raw()) as usize }
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
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirTypeIDEqual(self.raw, x.raw) }
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
    ty: Type<'c>,
}
impl<'c> IntegerType<'c> {
    pub fn new(c: &'c Context, bits: u32) -> Self {
        Self {
            ty: unsafe { Type::from_raw(mlirIntegerTypeGet(c.to_raw(), bits)) },
        }
    }
    pub fn signed(c: &'c Context, bits: u32) -> Self {
        unsafe { Self::from_raw(mlirIntegerTypeSignedGet(c.to_raw(), bits)) }
    }
    pub fn unsigned(c: &'c Context, bits: u32) -> Self {
        unsafe { Self::from_raw(mlirIntegerTypeUnsignedGet(c.to_raw(), bits)) }
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
    ty: Type<'c>,
}
impl<'c> MemRefType<'c> {
    pub fn new(t: Type<'c>, dims: &[u64], layout: Option<Attribute<'c>>, memory_space: Option<Attribute<'c>>) -> Self {
        unsafe {
            Self::from_raw(mlirMemRefTypeGet(
                t.to_raw(),
                dims.len() as _,
                dims.as_ptr() as *const _,
                layout.unwrap_or_else(|| Attribute::null()).to_raw(),
                memory_space.unwrap_or_else(|| Attribute::null()).to_raw(),
            ))
        }
    }
    pub fn checked(
        loc: Location<'c>,
        t: Type<'c>,
        dims: &[u64],
        layout: Attribute<'c>,
        memory_space: Attribute<'c>,
    ) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirMemRefTypeGetChecked(
                loc.to_raw(),
                t.to_raw(),
                dims.len() as isize,
                dims.as_ptr() as *const i64,
                layout.to_raw(),
                memory_space.to_raw(),
            ))
        }
    }
    pub fn layout(&self) -> Attribute<'c> {
        unsafe { Attribute::from_raw(mlirMemRefTypeGetLayout(self.ty.to_raw())) }
    }
    pub fn affine_map(&self) -> AffineMap<'c> {
        unsafe { AffineMap::from_raw(mlirMemRefTypeGetAffineMap(self.ty.to_raw())) }
    }
    pub fn memory_space(&self) -> Option<Attribute<'c>> {
        unsafe { Attribute::from_option_raw(mlirMemRefTypeGetMemorySpace(self.ty.to_raw())) }
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
    ty: Type<'c>,
}
impl<'c> RankedTensorType<'c> {
    pub fn new(dims: &[u64], t: Type<'c>, encoding: Option<Attribute<'c>>) -> Self {
        unsafe {
            Self::from_raw(mlirRankedTensorTypeGet(
                dims.len() as _,
                dims.as_ptr() as *const _,
                t.to_raw(),
                encoding.unwrap_or_else(|| Attribute::null()).to_raw(),
            ))
        }
    }
    pub fn checked(dims: &[u64], t: Type<'c>, encoding: Attribute<'c>, loc: Location<'c>) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirRankedTensorTypeGetChecked(
                loc.to_raw(),
                dims.len() as _,
                dims.as_ptr() as *const _,
                t.to_raw(),
                encoding.to_raw(),
            ))
        }
    }
    pub fn encoding(&self) -> Option<Attribute<'c>> {
        unsafe { Attribute::from_option_raw(mlirRankedTensorTypeGetEncoding(self.ty.to_raw())) }
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
    ty: Type<'c>,
}
impl<'c> TupleType<'c> {
    pub fn new(c: &'c Context, types: &[Type<'c>]) -> Self {
        unsafe {
            Self::from_raw(mlirTupleTypeGet(
                c.to_raw(),
                types.len() as isize,
                into_raw_array(types.iter().map(|ty| ty.to_raw()).collect()),
            ))
        }
    }
    pub fn ty(&self, idx: usize) -> Result<Type, Error> {
        if idx < self.type_count() {
            unsafe { Ok(Type::from_raw(mlirTupleTypeGetType(self.ty.to_raw(), idx as isize))) }
        } else {
            Err(Error::PositionOutOfBounds {
                name: "tuple field",
                value: self.to_string(),
                index: idx,
            })
        }
    }
    pub fn type_count(&self) -> usize {
        unsafe { mlirTupleTypeGetNumTypes(self.ty.to_raw()) as usize }
    }
}
type_traits!(TupleType, is_tuple, "tuple");

pub trait ValueLike {
    fn to_raw(&self) -> MlirValue;
    fn ty(&self) -> Type {
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
    pub unsafe fn from_raw(raw: MlirValue) -> Self {
        Self {
            raw,
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
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirValueEqual(self.raw, x.raw) }
    }
}
impl<'a> Eq for Value<'a> {}
impl<'a> Display for Value<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut y = (f, Ok(()));
        unsafe {
            mlirValuePrint(self.raw, Some(print_callback), &mut y as *mut _ as *mut c_void);
        }
        y.1
    }
}
impl<'a> Debug for Value<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "Value(")?;
        Display::fmt(self, f)?;
        write!(f, ")")
    }
}

from_raw_subtypes!(Value, BlockArgument, OperationResult);

#[cfg(test)]
mod test;
