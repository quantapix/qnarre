pub mod attr;
pub mod block;
pub mod op;
pub mod typ;
mod val;

pub use self::{
    attr::{Attribute, AttributeLike},
    block::{Block, BlockRef},
    op::{Operation, OperationRef},
    typ::{Type, TypeLike},
    val::{Value, ValueLike},
};
use crate::{
    ctx::{Context, ContextRef},
    ir::{Attribute, AttributeLike},
    string_ref::StringRef,
    utils::{into_raw_array, print_callback},
};
use mlir_lib::*;
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
    mem::{forget, transmute},
    ops::Deref,
};

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn new() {
        Identifier::new(&Context::new(), "foo");
    }
    #[test]
    fn context() {
        Identifier::new(&Context::new(), "foo").context();
    }
    #[test]
    fn equal() {
        let context = Context::new();
        assert_eq!(Identifier::new(&context, "foo"), Identifier::new(&context, "foo"));
    }
    #[test]
    fn not_equal() {
        let context = Context::new();
        assert_ne!(Identifier::new(&context, "foo"), Identifier::new(&context, "bar"));
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::{assert_eq, assert_ne};
    #[test]
    fn new() {
        Location::new(&Context::new(), "foo", 42, 42);
    }
    #[test]
    fn fused() {
        let context = Context::new();
        Location::fused(
            &context,
            &[
                Location::new(&Context::new(), "foo", 1, 1),
                Location::new(&Context::new(), "foo", 2, 2),
            ],
            Attribute::parse(&context, "42").unwrap(),
        );
    }
    #[test]
    fn name() {
        let context = Context::new();
        Location::name(&context, "foo", Location::unknown(&context));
    }
    #[test]
    fn unknown() {
        Location::unknown(&Context::new());
    }
    #[test]
    fn context() {
        Location::new(&Context::new(), "foo", 42, 42).context();
    }
    #[test]
    fn equal() {
        let context = Context::new();
        assert_eq!(Location::unknown(&context), Location::unknown(&context));
        assert_eq!(
            Location::new(&context, "foo", 42, 42),
            Location::new(&context, "foo", 42, 42),
        );
    }
    #[test]
    fn not_equal() {
        let context = Context::new();
        assert_ne!(Location::new(&context, "foo", 42, 42), Location::unknown(&context));
    }
    #[test]
    fn display() {
        let context = Context::new();
        assert_eq!(Location::unknown(&context).to_string(), "loc(unknown)");
        assert_eq!(Location::new(&context, "foo", 42, 42).to_string(), "loc(\"foo\":42:42)");
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{operation::OperationBuilder, Block, Region};
    #[test]
    fn new() {
        Module::new(Location::new(&Context::new(), "foo", 42, 42));
    }
    #[test]
    fn context() {
        Module::new(Location::new(&Context::new(), "foo", 42, 42)).context();
    }
    #[test]
    fn parse() {
        assert!(Module::parse(&Context::new(), "module{}").is_some());
    }
    #[test]
    fn parse_none() {
        assert!(Module::parse(&Context::new(), "module{").is_none());
    }
    #[test]
    fn from_operation() {
        let context = Context::new();
        let region = Region::new();
        region.append_block(Block::new(&[]));
        let module = Module::from_operation(
            OperationBuilder::new("builtin.module", Location::unknown(&context))
                .add_regions(vec![region])
                .build(),
        )
        .unwrap();
        assert!(module.as_operation().verify());
        assert_eq!(module.as_operation().to_string(), "module {\n}\n")
    }
    #[test]
    fn from_operation_fail() {
        let context = Context::new();
        assert!(
            Module::from_operation(OperationBuilder::new("func.func", Location::unknown(&context),).build()).is_none()
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn new() {
        Region::new();
    }
    #[test]
    fn first_block() {
        assert!(Region::new().first_block().is_none());
    }
    #[test]
    fn append_block() {
        let region = Region::new();
        let block = Block::new(&[]);
        region.append_block(block);
        assert!(region.first_block().is_some());
    }
    #[test]
    fn insert_block_after() {
        let region = Region::new();
        let block = region.append_block(Block::new(&[]));
        region.insert_block_after(block, Block::new(&[]));
        assert_eq!(region.first_block(), Some(block));
    }
    #[test]
    fn insert_block_before() {
        let region = Region::new();
        let block = region.append_block(Block::new(&[]));
        let block = region.insert_block_before(block, Block::new(&[]));
        assert_eq!(region.first_block(), Some(block));
    }
    #[test]
    fn equal() {
        let region = Region::new();
        assert_eq!(region, region);
    }
    #[test]
    fn not_equal() {
        assert_ne!(Region::new(), Region::new());
    }
}
