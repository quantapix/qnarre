use mlir_lib::*;
use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    error,
    ffi::{c_void, CString},
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
    slice,
    str::{self, Utf8Error},
    sync::RwLock,
};

use crate::{
    ir::{Module, Type, TypeLike},
    Attribute, AttributeLike, Context, Error,
};

mod ctx;
pub mod diag;
pub mod dialect;
pub mod ir;
pub mod mlir_lib;
pub mod pass;
#[cfg(test)]
mod test;
pub mod utils;

pub use self::ctx::{Context, ContextRef};

macro_rules! from_raw_subtypes {
    ($type:ident,) => {};
    ($type:ident, $name:ident $(, $names:ident)* $(,)?) => {
        impl<'c> From<$name<'c>> for $type<'c> {
            fn from(value: $name<'c>) -> Self {
                unsafe { Self::from_raw(value.to_raw()) }
            }
        }
        from_raw_subtypes!($type, $($names,)*);
    };
}

#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    AttributeExpected(&'static str, String),
    BlockArgumentExpected(String),
    ElementExpected {
        r#type: &'static str,
        value: String,
    },
    InvokeFunction,
    OperationResultExpected(String),
    PositionOutOfBounds {
        name: &'static str,
        val: String,
        idx: usize,
    },
    ParsePassPipeline(String),
    RunPass,
    TypeExpected(&'static str, String),
    UnknownDiagnosticSeverity(u32),
    Utf8(Utf8Error),
}
impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::AttributeExpected(r#type, attribute) => {
                write!(formatter, "{type} attribute expected: {attribute}")
            },
            Self::BlockArgumentExpected(value) => {
                write!(formatter, "block argument expected: {value}")
            },
            Self::ElementExpected { r#type, value } => {
                write!(formatter, "element of {type} type expected: {value}")
            },
            Self::InvokeFunction => write!(formatter, "failed to invoke JIT-compiled function"),
            Self::OperationResultExpected(value) => {
                write!(formatter, "operation result expected: {value}")
            },
            Self::ParsePassPipeline(message) => {
                write!(formatter, "failed to parse pass pipeline:\n{}", message)
            },
            Self::PositionOutOfBounds { name, value, index } => {
                write!(formatter, "{name} position {index} out of bounds: {value}")
            },
            Self::RunPass => write!(formatter, "failed to run pass"),
            Self::TypeExpected(r#type, actual) => {
                write!(formatter, "{type} type expected: {actual}")
            },
            Self::UnknownDiagnosticSeverity(severity) => {
                write!(formatter, "unknown diagnostic severity: {severity}")
            },
            Self::Utf8(error) => {
                write!(formatter, "{}", error)
            },
        }
    }
}
impl error::Error for Error {}
impl From<Utf8Error> for Error {
    fn from(error: Utf8Error) -> Self {
        Self::Utf8(error)
    }
}

pub struct ExecutionEngine {
    raw: MlirExecutionEngine,
}
impl ExecutionEngine {
    pub fn new(
        module: &Module,
        optimization_level: usize,
        shared_library_paths: &[&str],
        enable_object_dump: bool,
    ) -> Self {
        Self {
            raw: unsafe {
                mlirExecutionEngineCreate(
                    module.to_raw(),
                    optimization_level as i32,
                    shared_library_paths.len() as i32,
                    shared_library_paths
                        .iter()
                        .map(|&string| StringRef::from(string).to_raw())
                        .collect::<Vec<_>>()
                        .as_ptr(),
                    enable_object_dump,
                )
            },
        }
    }
    pub unsafe fn invoke_packed(&self, name: &str, arguments: &mut [*mut ()]) -> Result<(), Error> {
        let result = LogicalResult::from_raw(mlirExecutionEngineInvokePacked(
            self.raw,
            StringRef::from(name).to_raw(),
            arguments.as_mut_ptr() as *mut *mut c_void,
        ));
        if result.is_success() {
            Ok(())
        } else {
            Err(Error::InvokeFunction)
        }
    }
    pub fn dump_to_object_file(&self, path: &str) {
        unsafe { mlirExecutionEngineDumpToObjectFile(self.raw, StringRef::from(path).to_raw()) }
    }
}
impl Drop for ExecutionEngine {
    fn drop(&mut self) {
        unsafe { mlirExecutionEngineDestroy(self.raw) }
    }
}

#[derive(Clone, Copy)]
pub struct Integer<'c> {
    raw: MlirAttribute,
    _context: PhantomData<&'c Context>,
}
impl<'c> Integer<'c> {
    pub fn new(integer: i64, r#type: Type<'c>) -> Self {
        unsafe { Self::from_raw(mlirIntegerAttrGet(r#type.to_raw(), integer)) }
    }
    unsafe fn from_raw(raw: MlirAttribute) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
}
impl<'c> AttributeLike<'c> for Integer<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.raw
    }
}
impl<'c> TryFrom<Attribute<'c>> for Integer<'c> {
    type Error = Error;
    fn try_from(attribute: Attribute<'c>) -> Result<Self, Self::Error> {
        if attribute.is_integer() {
            Ok(unsafe { Self::from_raw(attribute.to_raw()) })
        } else {
            Err(Error::AttributeExpected("integer", format!("{}", attribute)))
        }
    }
}
impl<'c> Display for Integer<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(&Attribute::from(*self), formatter)
    }
}
impl<'c> Debug for Integer<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self, formatter)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LogicalResult {
    raw: MlirLogicalResult,
}
impl LogicalResult {
    pub fn success() -> Self {
        Self {
            raw: MlirLogicalResult { value: 1 },
        }
    }
    pub fn failure() -> Self {
        Self {
            raw: MlirLogicalResult { value: 0 },
        }
    }
    pub fn is_success(&self) -> bool {
        self.raw.value != 0
    }
    #[allow(dead_code)]
    pub fn is_failure(&self) -> bool {
        self.raw.value == 0
    }
    pub fn from_raw(result: MlirLogicalResult) -> Self {
        Self { raw: result }
    }
    pub fn to_raw(self) -> MlirLogicalResult {
        self.raw
    }
}
impl From<bool> for LogicalResult {
    fn from(ok: bool) -> Self {
        if ok {
            Self::success()
        } else {
            Self::failure()
        }
    }
}

static STRING_CACHE: Lazy<RwLock<HashMap<String, CString>>> = Lazy::new(Default::default);

#[derive(Clone, Copy, Debug)]
pub struct StringRef<'a> {
    raw: MlirStringRef,
    _parent: PhantomData<&'a ()>,
}
impl<'a> StringRef<'a> {
    pub fn as_str(&self) -> Result<&'a str, Utf8Error> {
        unsafe {
            let bytes = slice::from_raw_parts(self.raw.data as *mut u8, self.raw.length);
            str::from_utf8(if bytes[bytes.len() - 1] == 0 {
                &bytes[..bytes.len() - 1]
            } else {
                bytes
            })
        }
    }
    pub fn to_raw(self) -> MlirStringRef {
        self.raw
    }
    pub unsafe fn from_raw(string: MlirStringRef) -> Self {
        Self {
            raw: string,
            _parent: Default::default(),
        }
    }
}
impl<'a> PartialEq for StringRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirStringRefEqual(self.raw, other.raw) }
    }
}
impl<'a> Eq for StringRef<'a> {}
impl From<&str> for StringRef<'static> {
    fn from(string: &str) -> Self {
        if !STRING_CACHE.read().unwrap().contains_key(string) {
            STRING_CACHE
                .write()
                .unwrap()
                .insert(string.to_owned(), CString::new(string).unwrap());
        }
        let lock = STRING_CACHE.read().unwrap();
        let string = lock.get(string).unwrap();
        unsafe { Self::from_raw(mlirStringRefCreateFromCString(string.as_ptr())) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ctx::Context,
        dialect::{self, arith, func, scf},
        ir::{
            attr::{IntegerAttribute, StringAttribute, TypeAttribute},
            operation::OperationBuilder,
            r#type::{FunctionType, IntegerType},
            Block, Location, Module, Region, Type, Value,
        },
        pass,
        test::{create_test_context, load_all_dialects},
    };
    #[test]
    fn build_module() {
        let context = Context::new();
        let module = Module::new(Location::unknown(&context));
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
    #[test]
    fn build_module_with_dialect() {
        let registry = dialect::DialectRegistry::new();
        let context = Context::new();
        context.append_dialect_registry(&registry);
        let module = Module::new(Location::unknown(&context));
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
    #[test]
    fn build_add() {
        let context = Context::new();
        load_all_dialects(&context);
        let location = Location::unknown(&context);
        let module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let function = {
            let block = Block::new(&[(integer_type, location), (integer_type, location)]);
            let sum = block.append_operation(arith::addi(
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                location,
            ));
            block.append_operation(func::r#return(&[sum.result(0).unwrap().into()], location));
            let region = Region::new();
            region.append_block(block);
            func::func(
                &context,
                StringAttribute::new(&context, "add"),
                TypeAttribute::new(FunctionType::new(&context, &[integer_type, integer_type], &[integer_type]).into()),
                region,
                &[],
                Location::unknown(&context),
            )
        };
        module.body().append_operation(function);
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
    #[test]
    fn build_sum() {
        let context = Context::new();
        load_all_dialects(&context);
        let location = Location::unknown(&context);
        let module = Module::new(location);
        let memref_type = Type::parse(&context, "memref<?xf32>").unwrap();
        let function = {
            let function_block = Block::new(&[(memref_type, location), (memref_type, location)]);
            let index_type = Type::parse(&context, "index").unwrap();
            let zero = function_block.append_operation(arith::constant(
                &context,
                IntegerAttribute::new(0, Type::index(&context)).into(),
                location,
            ));
            let dim = function_block.append_operation(
                OperationBuilder::new("memref.dim", location)
                    .add_operands(&[
                        function_block.argument(0).unwrap().into(),
                        zero.result(0).unwrap().into(),
                    ])
                    .add_results(&[index_type])
                    .build(),
            );
            let loop_block = Block::new(&[(index_type, location)]);
            let one = function_block.append_operation(arith::constant(
                &context,
                IntegerAttribute::new(1, Type::index(&context)).into(),
                location,
            ));
            {
                let f32_type = Type::float32(&context);
                let lhs = loop_block.append_operation(
                    OperationBuilder::new("memref.load", location)
                        .add_operands(&[
                            function_block.argument(0).unwrap().into(),
                            loop_block.argument(0).unwrap().into(),
                        ])
                        .add_results(&[f32_type])
                        .build(),
                );
                let rhs = loop_block.append_operation(
                    OperationBuilder::new("memref.load", location)
                        .add_operands(&[
                            function_block.argument(1).unwrap().into(),
                            loop_block.argument(0).unwrap().into(),
                        ])
                        .add_results(&[f32_type])
                        .build(),
                );
                let add = loop_block.append_operation(arith::addf(
                    lhs.result(0).unwrap().into(),
                    rhs.result(0).unwrap().into(),
                    location,
                ));
                loop_block.append_operation(
                    OperationBuilder::new("memref.store", location)
                        .add_operands(&[
                            add.result(0).unwrap().into(),
                            function_block.argument(0).unwrap().into(),
                            loop_block.argument(0).unwrap().into(),
                        ])
                        .build(),
                );
                loop_block.append_operation(scf::r#yield(&[], location));
            }
            function_block.append_operation(scf::r#for(
                zero.result(0).unwrap().into(),
                dim.result(0).unwrap().into(),
                one.result(0).unwrap().into(),
                {
                    let loop_region = Region::new();
                    loop_region.append_block(loop_block);
                    loop_region
                },
                location,
            ));
            function_block.append_operation(func::r#return(&[], location));
            let function_region = Region::new();
            function_region.append_block(function_block);
            func::func(
                &context,
                StringAttribute::new(&context, "sum"),
                TypeAttribute::new(FunctionType::new(&context, &[memref_type, memref_type], &[]).into()),
                function_region,
                &[],
                Location::unknown(&context),
            )
        };
        module.body().append_operation(function);
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
    #[test]
    fn return_value_from_function() {
        let context = Context::new();
        load_all_dialects(&context);
        let location = Location::unknown(&context);
        let module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        fn compile_add<'a>(context: &Context, block: &'a Block, lhs: Value<'a>, rhs: Value<'a>) -> Value<'a> {
            block
                .append_operation(arith::addi(lhs, rhs, Location::unknown(context)))
                .result(0)
                .unwrap()
                .into()
        }
        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "add"),
            TypeAttribute::new(FunctionType::new(&context, &[integer_type, integer_type], &[integer_type]).into()),
            {
                let block = Block::new(&[(integer_type, location), (integer_type, location)]);
                block.append_operation(func::r#return(
                    &[compile_add(
                        &context,
                        &block,
                        block.argument(0).unwrap().into(),
                        block.argument(1).unwrap().into(),
                    )],
                    location,
                ));
                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            Location::unknown(&context),
        ));
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
    #[test]
    fn invoke_packed() {
        let context = create_test_context();
        let mut module = Module::parse(
            &context,
            r#"
            module {
                func.func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }
            }
            "#,
        )
        .unwrap();
        let pass_manager = pass::PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager
            .nested_under("func.func")
            .add_pass(pass::conversion::create_arith_to_llvm());
        assert_eq!(pass_manager.run(&mut module), Ok(()));
        let engine = ExecutionEngine::new(&module, 2, &[], false);
        let mut argument = 42;
        let mut result = -1;
        assert_eq!(
            unsafe {
                engine.invoke_packed(
                    "add",
                    &mut [&mut argument as *mut i32 as *mut (), &mut result as *mut i32 as *mut ()],
                )
            },
            Ok(())
        );
        assert_eq!(argument, 42);
        assert_eq!(result, 84);
    }
    #[test]
    fn dump_to_object_file() {
        let context = create_test_context();
        let mut module = Module::parse(
            &context,
            r#"
            module {
                func.func @add(%arg0 : i32) -> i32 {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }
            }
            "#,
        )
        .unwrap();
        let pass_manager = pass::PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager
            .nested_under("func.func")
            .add_pass(pass::conversion::create_arith_to_llvm());
        assert_eq!(pass_manager.run(&mut module), Ok(()));
        ExecutionEngine::new(&module, 2, &[], true).dump_to_object_file("/tmp/melior/test.o");
    }
    #[test]
    fn success() {
        assert!(LogicalResult::success().is_success());
    }
    #[test]
    fn failure() {
        assert!(LogicalResult::failure().is_failure());
    }
    #[test]
    fn equal() {
        assert_eq!(StringRef::from("foo"), StringRef::from("foo"));
    }
    #[test]
    fn equal_str() {
        assert_eq!(StringRef::from("foo").as_str().unwrap(), "foo");
    }
    #[test]
    fn not_equal() {
        assert_ne!(StringRef::from("foo"), StringRef::from("bar"));
    }
}
