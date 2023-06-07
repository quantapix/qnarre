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
use core::{
    fmt,
    mem::{forget, transmute},
};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ctx::Context,
        ir::{operation::OperationBuilder, Block, Location, Type},
        test::load_all_dialects,
    };
    use pretty_assertions::assert_eq;

    #[test]
    fn new() {
        OperationBuilder::new("foo", Location::unknown(&Context::new())).build();
    }
    #[test]
    fn name() {
        let context = Context::new();
        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context),)
                .build()
                .name(),
            Identifier::new(&context, "foo")
        );
    }
    #[test]
    fn block() {
        let block = Block::new(&[]);
        let operation =
            block.append_operation(OperationBuilder::new("foo", Location::unknown(&Context::new())).build());
        assert_eq!(operation.block().as_deref(), Some(&block));
    }
    #[test]
    fn block_none() {
        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&Context::new()))
                .build()
                .block(),
            None
        );
    }
    #[test]
    fn result_error() {
        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&Context::new()))
                .build()
                .result(0)
                .unwrap_err(),
            Error::PositionOutOfBounds {
                name: "operation result",
                value: "\"foo\"() : () -> ()\n".into(),
                index: 0
            }
        );
    }
    #[test]
    fn region_none() {
        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&Context::new()),)
                .build()
                .region(0),
            Err(Error::PositionOutOfBounds {
                name: "region",
                value: "\"foo\"() : () -> ()\n".into(),
                index: 0
            })
        );
    }
    #[test]
    fn clone() {
        let context = Context::new();
        let operation = OperationBuilder::new("foo", Location::unknown(&context)).build();
        let _ = operation.clone();
    }
    #[test]
    fn display() {
        let context = Context::new();
        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context),)
                .build()
                .to_string(),
            "\"foo\"() : () -> ()\n"
        );
    }
    #[test]
    fn debug() {
        let context = Context::new();
        assert_eq!(
            format!(
                "{:?}",
                OperationBuilder::new("foo", Location::unknown(&context)).build()
            ),
            "Operation(\n\"foo\"() : () -> ()\n)"
        );
    }
    #[test]
    fn to_string_with_flags() {
        let context = Context::new();
        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .to_string_with_flags(
                    OperationPrintingFlags::new()
                        .elide_large_elements_attributes(100)
                        .enable_debug_info(true, true)
                        .print_generic_operation_form()
                        .use_local_scope()
                ),
            Ok("\"foo\"() : () -> () [unknown]".into())
        );
    }
    #[test]
    fn new() {
        OperationBuilder::new("foo", Location::unknown(&Context::new())).build();
    }
    #[test]
    fn add_results() {
        let context = Context::new();
        OperationBuilder::new("foo", Location::unknown(&context))
            .add_results(&[Type::parse(&context, "i1").unwrap()])
            .build();
    }
    #[test]
    fn add_regions() {
        let context = Context::new();
        OperationBuilder::new("foo", Location::unknown(&context))
            .add_regions(vec![Region::new()])
            .build();
    }
    #[test]
    fn add_successors() {
        let context = Context::new();
        OperationBuilder::new("foo", Location::unknown(&context))
            .add_successors(&[&Block::new(&[])])
            .build();
    }
    #[test]
    fn add_attributes() {
        let context = Context::new();
        OperationBuilder::new("foo", Location::unknown(&context))
            .add_attributes(&[(
                Identifier::new(&context, "foo"),
                Attribute::parse(&context, "unit").unwrap(),
            )])
            .build();
    }
    #[test]
    fn enable_result_type_inference() {
        let context = Context::new();
        load_all_dialects(&context);
        let location = Location::unknown(&context);
        let r#type = Type::index(&context);
        let block = Block::new(&[(r#type, location)]);
        let argument = block.argument(0).unwrap().into();
        assert_eq!(
            OperationBuilder::new("arith.addi", location)
                .add_operands(&[argument, argument])
                .enable_result_type_inference()
                .build()
                .result(0)
                .unwrap()
                .r#type(),
            r#type,
        );
    }
    #[test]
    fn result_number() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index").unwrap();
        let operation = OperationBuilder::new("foo", Location::unknown(&context))
            .add_results(&[r#type])
            .build();
        assert_eq!(operation.result(0).unwrap().result_number(), 0);
    }
    #[test]
    fn owner() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index").unwrap();
        let block = Block::new(&[(r#type, Location::unknown(&context))]);
        assert_eq!(&*block.argument(0).unwrap().owner(), &block);
    }
}
