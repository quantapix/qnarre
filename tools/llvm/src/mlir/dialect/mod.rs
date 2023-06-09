use mlir_lib::*;
use std::marker::PhantomData;

use crate::mlir::{
    ir::{attr::*, op::*, typ::*, *},
    utils::into_raw_array,
    Context, ContextRef, Error, StringRef,
};

#[derive(Clone, Copy, Debug)]
pub struct Dialect<'c> {
    raw: MlirDialect,
    ctx: PhantomData<&'c Context>,
}
impl<'c> Dialect<'c> {
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirDialectGetContext(self.raw)) }
    }
    pub fn namespace(&self) -> StringRef {
        unsafe { StringRef::from_raw(mlirDialectGetNamespace(self.raw)) }
    }
    pub unsafe fn from_raw(raw: MlirDialect) -> Self {
        Self {
            raw,
            ctx: Default::default(),
        }
    }
}
impl<'c> PartialEq for Dialect<'c> {
    fn eq(&self, x: &Self) -> bool {
        unsafe { mlirDialectEqual(self.raw, x.raw) }
    }
}
impl<'c> Eq for Dialect<'c> {}

#[derive(Clone, Copy, Debug)]
pub struct DialectHandle {
    raw: MlirDialectHandle,
}
impl DialectHandle {
    pub fn r#async() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__async__()) }
    }
    pub fn cf() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__cf__()) }
    }
    pub fn func() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__func__()) }
    }
    pub fn gpu() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__gpu__()) }
    }
    pub fn linalg() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__linalg__()) }
    }
    pub fn llvm() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__llvm__()) }
    }
    pub fn pdl() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__pdl__()) }
    }
    pub fn quant() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__quant__()) }
    }
    pub fn scf() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__scf__()) }
    }
    pub fn shape() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__shape__()) }
    }
    pub fn sparse_tensor() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__sparse_tensor__()) }
    }
    pub fn tensor() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__tensor__()) }
    }
    pub fn namespace(&self) -> StringRef {
        unsafe { StringRef::from_raw(mlirDialectHandleGetNamespace(self.raw)) }
    }
    pub fn insert_dialect(&self, x: &DialectRegistry) {
        unsafe { mlirDialectHandleInsertDialect(self.raw, x.to_raw()) }
    }
    pub fn load_dialect<'c>(&self, x: &'c Context) -> Dialect<'c> {
        unsafe { Dialect::from_raw(mlirDialectHandleLoadDialect(self.raw, x.to_raw())) }
    }
    pub fn register_dialect(&self, x: &Context) {
        unsafe { mlirDialectHandleRegisterDialect(self.raw, x.to_raw()) }
    }
    pub unsafe fn from_raw(raw: MlirDialectHandle) -> Self {
        Self { raw }
    }
}

#[derive(Debug)]
pub struct DialectRegistry {
    raw: MlirDialectRegistry,
}
impl DialectRegistry {
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirDialectRegistryCreate() },
        }
    }
    pub fn to_raw(&self) -> MlirDialectRegistry {
        self.raw
    }
}
impl Drop for DialectRegistry {
    fn drop(&mut self) {
        unsafe { mlirDialectRegistryDestroy(self.raw) };
    }
}
impl Default for DialectRegistry {
    fn default() -> Self {
        Self::new()
    }
}

//arith
pub fn constant<'c>(ctx: &'c Context, val: Attribute<'c>, loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("arith.constant", loc)
        .add_attributes(&[(Identifier::new(ctx, "value"), val)])
        .enable_result_type_inference()
        .build()
}
pub enum CmpfPredicate {
    False,
    Oeq,
    Ogt,
    Oge,
    Olt,
    Ole,
    One,
    Ord,
    Ueq,
    Ugt,
    Uge,
    Ult,
    Ule,
    Une,
    Uno,
    True,
}
pub fn cmpf<'c>(ctx: &'c Context, pred: CmpfPredicate, lhs: Value, rhs: Value, loc: Location<'c>) -> Operation<'c> {
    cmp(ctx, "arith.cmpf", pred as i64, lhs, rhs, loc)
}
pub enum CmpiPredicate {
    Eq,
    Ne,
    Slt,
    Sle,
    Sgt,
    Sge,
    Ult,
    Ule,
    Ugt,
    Uge,
}
pub fn cmpi<'c>(ctx: &'c Context, pred: CmpiPredicate, lhs: Value, rhs: Value, loc: Location<'c>) -> Operation<'c> {
    cmp(ctx, "arith.cmpi", pred as i64, lhs, rhs, loc)
}
fn cmp<'c>(ctx: &'c Context, name: &str, pred: i64, lhs: Value, rhs: Value, loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new(name, loc)
        .add_attributes(&[(
            Identifier::new(ctx, "pred"),
            IntegerAttribute::new(pred, IntegerType::new(ctx, 64).into()).into(),
        )])
        .add_operands(&[lhs, rhs])
        .enable_result_type_inference()
        .build()
}
macros::binary_operations!(
    arith,
    [
        addf,
        addi,
        addui_extended,
        andi,
        ceildivsi,
        ceildivui,
        divf,
        divsi,
        divui,
        floordivsi,
        maxf,
        maxsi,
        maxui,
        minf,
        minsi,
        minui,
        mulf,
        muli,
        mulsi_extended,
        mului_extended,
        ori,
        remf,
        remsi,
        remui,
        shli,
        shrsi,
        shrui,
        subf,
        subi,
        xori,
    ]
);
macros::unary_operations!(arith, [negf, truncf]);
macros::typed_unary_operations!(
    arith,
    [
        bitcast,
        extf,
        extsi,
        extui,
        fptosi,
        fptoui,
        index_cast,
        index_castui,
        sitofp,
        trunci,
        uitofp
    ]
);

//cf
pub fn assert<'c>(ctx: &'c Context, argument: Value<'c>, message: &str, loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("cf.assert", loc)
        .add_attributes(&[(Identifier::new(ctx, "msg"), StringAttribute::new(ctx, message).into())])
        .add_operands(&[argument])
        .build()
}
pub fn br<'c>(successor: &Block<'c>, destination_operands: &[Value<'c>], loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("cf.br", loc)
        .add_operands(destination_operands)
        .add_successors(&[successor])
        .build()
}
pub fn cond_br<'c>(
    ctx: &'c Context,
    cond: Value<'c>,
    true_successor: &Block<'c>,
    false_successor: &Block<'c>,
    true_successor_operands: &[Value],
    false_successor_operands: &[Value],
    loc: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("cf.cond_br", loc)
        .add_attributes(&[(
            Identifier::new(ctx, "operand_segment_sizes"),
            DenseI32ArrayAttribute::new(
                ctx,
                &[
                    1,
                    true_successor.argument_count() as i32,
                    false_successor.argument_count() as i32,
                ],
            )
            .into(),
        )])
        .add_operands(
            &[cond]
                .into_iter()
                .chain(true_successor_operands.iter().copied())
                .chain(false_successor_operands.iter().copied())
                .collect::<Vec<_>>(),
        )
        .add_successors(&[true_successor, false_successor])
        .build()
}
pub fn switch<'c>(
    ctx: &'c Context,
    case_values: &[i64],
    flag: Value<'c>,
    flag_type: Type<'c>,
    default_destination: (&Block<'c>, &[Value]),
    case_destinations: &[(&Block<'c>, &[Value])],
    loc: Location<'c>,
) -> Result<Operation<'c>, Error> {
    let (destinations, operands): (Vec<_>, Vec<_>) = [default_destination]
        .into_iter()
        .chain(case_destinations.iter().copied())
        .unzip();
    Ok(OperationBuilder::new("cf.switch", loc)
        .add_attributes(&[
            (
                Identifier::new(ctx, "case_values"),
                DenseElementsAttribute::new(
                    RankedTensorType::new(&[case_values.len() as u64], flag_type, None).into(),
                    &case_values
                        .iter()
                        .map(|value| IntegerAttribute::new(*value, flag_type).into())
                        .collect::<Vec<_>>(),
                )?
                .into(),
            ),
            (
                Identifier::new(ctx, "case_operand_segments"),
                DenseI32ArrayAttribute::new(
                    ctx,
                    &case_destinations
                        .iter()
                        .map(|(_, operands)| operands.len() as i32)
                        .collect::<Vec<_>>(),
                )
                .into(),
            ),
            (
                Identifier::new(ctx, "operand_segment_sizes"),
                DenseI32ArrayAttribute::new(
                    ctx,
                    &[
                        1,
                        default_destination.1.len() as i32,
                        case_destinations
                            .iter()
                            .map(|(_, operands)| operands.len() as i32)
                            .sum(),
                    ],
                )
                .into(),
            ),
        ])
        .add_operands(
            &[flag]
                .into_iter()
                .chain(operands.into_iter().flatten().copied())
                .collect::<Vec<_>>(),
        )
        .add_successors(&destinations)
        .build())
}

//func
pub fn call<'c>(
    ctx: &'c Context,
    function: FlatSymbolRefAttribute<'c>,
    arguments: &[Value],
    loc: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("func.call", loc)
        .add_attributes(&[(Identifier::new(ctx, "callee"), function.into())])
        .add_operands(arguments)
        .build()
}
pub fn call_indirect<'c>(function: Value, arguments: &[Value], loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("func.call_indirect", loc)
        .add_operands(&[function])
        .add_operands(arguments)
        .build()
}
pub fn constant<'c>(
    ctx: &'c Context,
    function: FlatSymbolRefAttribute<'c>,
    r#type: FunctionType<'c>,
    loc: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("func.constant", loc)
        .add_attributes(&[(Identifier::new(ctx, "value"), function.into())])
        .add_results(&[r#type.into()])
        .build()
}
pub fn func<'c>(
    ctx: &'c Context,
    name: StringAttribute<'c>,
    r#type: TypeAttribute<'c>,
    region: Region,
    attributes: &[(Identifier<'c>, Attribute<'c>)],
    loc: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("func.func", loc)
        .add_attributes(&[
            (Identifier::new(ctx, "sym_name"), name.into()),
            (Identifier::new(ctx, "function_type"), r#type.into()),
        ])
        .add_attributes(attributes)
        .add_regions(vec![region])
        .build()
}
pub fn r#return<'c>(operands: &[Value], loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("func.return", loc).add_operands(operands).build()
}

//index
pub fn constant<'c>(ctx: &'c Context, value: IntegerAttribute<'c>, loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("index.constant", loc)
        .add_attributes(&[(Identifier::new(ctx, "value"), value.into())])
        .enable_result_type_inference()
        .build()
}
pub fn cmp<'c>(ctx: &'c Context, pred: CmpiPredicate, lhs: Value, rhs: Value, loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("index.cmp", loc)
        .add_attributes(&[(
            Identifier::new(ctx, "pred"),
            Attribute::parse(
                ctx,
                match pred {
                    CmpiPredicate::Eq => "#index<cmp_predicate eq>",
                    CmpiPredicate::Ne => "#index<cmp_predicate ne>",
                    CmpiPredicate::Slt => "#index<cmp_predicate slt>",
                    CmpiPredicate::Sle => "#index<cmp_predicate sle>",
                    CmpiPredicate::Sgt => "#index<cmp_predicate sgt>",
                    CmpiPredicate::Sge => "#index<cmp_predicate sge>",
                    CmpiPredicate::Ult => "#index<cmp_predicate ult>",
                    CmpiPredicate::Ule => "#index<cmp_predicate ule>",
                    CmpiPredicate::Ugt => "#index<cmp_predicate ugt>",
                    CmpiPredicate::Uge => "#index<cmp_predicate uge>",
                },
            )
            .unwrap(),
        )])
        .add_operands(&[lhs, rhs])
        .enable_result_type_inference()
        .build()
}
macros::binary_operations!(
    index,
    [
        add, and, ceildivs, ceildivu, divs, divu, floordivs, maxs, maxu, mins, minu, mul, or, rems, remu, shl, shrs,
        shru, sub, xor,
    ]
);
macros::typed_unary_operations!(index, [casts, castu]);

// llvm
pub fn array(ty: Type, len: u32) -> Type {
    unsafe { Type::from_raw(mlirLLVMArrayTypeGet(ty.to_raw(), len)) }
}
pub fn function<'c>(result: Type<'c>, arguments: &[Type<'c>], variadic_arguments: bool) -> Type<'c> {
    unsafe {
        Type::from_raw(mlirLLVMFunctionTypeGet(
            result.to_raw(),
            arguments.len() as isize,
            into_raw_array(arguments.iter().map(|argument| argument.to_raw()).collect()),
            variadic_arguments,
        ))
    }
}
pub fn opaque_pointer(ctx: &Context) -> Type {
    Type::parse(ctx, "!llvm.ptr").unwrap()
}
pub fn pointer(ty: Type, address_space: u32) -> Type {
    unsafe { Type::from_raw(mlirLLVMPointerTypeGet(ty.to_raw(), address_space)) }
}
pub fn r#struct<'c>(ctx: &'c Context, fields: &[Type<'c>], packed: bool) -> Type<'c> {
    unsafe {
        Type::from_raw(mlirLLVMStructTypeLiteralGet(
            ctx.to_raw(),
            fields.len() as isize,
            into_raw_array(fields.iter().map(|x| x.to_raw()).collect()),
            packed,
        ))
    }
}
pub fn void(ctx: &Context) -> Type {
    unsafe { Type::from_raw(mlirLLVMVoidTypeGet(ctx.to_raw())) }
}

//memref
pub fn alloc<'c>(
    ctx: &'c Context,
    ty: MemRefType<'c>,
    dynamic_sizes: &[Value],
    symbols: &[Value],
    alignment: Option<IntegerAttribute<'c>>,
    loc: Location<'c>,
) -> Operation<'c> {
    allocate(ctx, "memref.alloc", ty, dynamic_sizes, symbols, alignment, loc)
}
pub fn alloca<'c>(
    ctx: &'c Context,
    ty: MemRefType<'c>,
    dynamic_sizes: &[Value],
    symbols: &[Value],
    alignment: Option<IntegerAttribute<'c>>,
    loc: Location<'c>,
) -> Operation<'c> {
    allocate(ctx, "memref.alloca", ty, dynamic_sizes, symbols, alignment, loc)
}
fn allocate<'c>(
    ctx: &'c Context,
    name: &str,
    ty: MemRefType<'c>,
    dynamic_sizes: &[Value],
    symbols: &[Value],
    alignment: Option<IntegerAttribute<'c>>,
    loc: Location<'c>,
) -> Operation<'c> {
    let mut y = OperationBuilder::new(name, loc);
    y = y.add_attributes(&[(
        Identifier::new(ctx, "operand_segment_sizes"),
        DenseI32ArrayAttribute::new(ctx, &[dynamic_sizes.len() as i32, symbols.len() as i32]).into(),
    )]);
    y = y.add_operands(dynamic_sizes).add_operands(symbols);
    if let Some(alignment) = alignment {
        y = y.add_attributes(&[(Identifier::new(ctx, "alignment"), alignment.into())]);
    }
    y.add_results(&[ty.into()]).build()
}
pub fn cast<'c>(x: Value, ty: MemRefType<'c>, loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("memref.cast", loc)
        .add_operands(&[x])
        .add_results(&[ty.into()])
        .build()
}
pub fn dealloc<'c>(x: Value, loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("memref.dealloc", loc).add_operands(&[x]).build()
}
pub fn dim<'c>(x: Value, i: Value, loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("memref.dim", loc)
        .add_operands(&[x, i])
        .enable_result_type_inference()
        .build()
}
pub fn get_global<'c>(ctx: &'c Context, name: &str, ty: MemRefType<'c>, loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("memref.get_global", loc)
        .add_attributes(&[(
            Identifier::new(ctx, "name"),
            FlatSymbolRefAttribute::new(ctx, name).into(),
        )])
        .add_results(&[ty.into()])
        .build()
}
#[allow(clippy::too_many_arguments)]
pub fn global<'c>(
    ctx: &'c Context,
    name: &str,
    visibility: Option<&str>,
    ty: MemRefType<'c>,
    value: Option<Attribute<'c>>,
    constant: bool,
    alignment: Option<IntegerAttribute<'c>>,
    loc: Location<'c>,
) -> Operation<'c> {
    let mut builder = OperationBuilder::new("memref.global", loc).add_attributes(&[
        (Identifier::new(ctx, "sym_name"), StringAttribute::new(ctx, name).into()),
        (Identifier::new(ctx, "type"), TypeAttribute::new(ty.into()).into()),
        (
            Identifier::new(ctx, "initial_value"),
            value.unwrap_or_else(|| Attribute::unit(ctx)),
        ),
    ]);
    if let Some(visibility) = visibility {
        builder = builder.add_attributes(&[(
            Identifier::new(ctx, "sym_visibility"),
            StringAttribute::new(ctx, visibility).into(),
        )]);
    }
    if constant {
        builder = builder.add_attributes(&[(Identifier::new(ctx, "constant"), Attribute::unit(ctx))]);
    }
    if let Some(alignment) = alignment {
        builder = builder.add_attributes(&[(Identifier::new(ctx, "alignment"), alignment.into())]);
    }
    builder.build()
}
pub fn load<'c>(x: Value, indices: &[Value], loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("memref.load", loc)
        .add_operands(&[x])
        .add_operands(indices)
        .enable_result_type_inference()
        .build()
}
pub fn rank<'c>(x: Value, loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("memref.rank", loc)
        .add_operands(&[x])
        .enable_result_type_inference()
        .build()
}
pub fn store<'c>(x: Value, memref: Value, indices: &[Value], loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("memref.store", loc)
        .add_operands(&[x, memref])
        .add_operands(indices)
        .build()
}
pub fn realloc<'c>(
    ctx: &'c Context,
    value: Value,
    size: Option<Value>,
    ty: MemRefType<'c>,
    alignment: Option<IntegerAttribute<'c>>,
    loc: Location<'c>,
) -> Operation<'c> {
    let mut builder = OperationBuilder::new("memref.realloc", loc)
        .add_operands(&[value])
        .add_results(&[ty.into()]);
    if let Some(size) = size {
        builder = builder.add_operands(&[size]);
    }
    if let Some(alignment) = alignment {
        builder = builder.add_attributes(&[(Identifier::new(ctx, "alignment"), alignment.into())]);
    }
    builder.build()
}

//scf
pub fn cond<'c>(cond: Value<'c>, values: &[Value<'c>], loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("scf.cond", loc)
        .add_operands(&[cond])
        .add_operands(values)
        .build()
}
pub fn execute_region<'c>(result_types: &[Type<'c>], region: Region, loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("scf.execute_region", loc)
        .add_results(result_types)
        .add_regions(vec![region])
        .build()
}
pub fn r#for<'c>(
    start: Value<'c>,
    end: Value<'c>,
    step: Value<'c>,
    region: Region,
    loc: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("scf.for", loc)
        .add_operands(&[start, end, step])
        .add_regions(vec![region])
        .build()
}
pub fn r#if<'c>(
    cond: Value<'c>,
    result_types: &[Type<'c>],
    then_region: Region,
    else_region: Region,
    loc: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("scf.if", loc)
        .add_operands(&[cond])
        .add_results(result_types)
        .add_regions(vec![then_region, else_region])
        .build()
}
pub fn index_switch<'c>(
    ctx: &'c Context,
    cond: Value<'c>,
    result_types: &[Type<'c>],
    cases: DenseI64ArrayAttribute<'c>,
    regions: Vec<Region>,
    loc: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("scf.index_switch", loc)
        .add_operands(&[cond])
        .add_results(result_types)
        .add_attributes(&[(Identifier::new(ctx, "cases"), cases.into())])
        .add_regions(regions)
        .build()
}
pub fn r#while<'c>(
    initial_values: &[Value<'c>],
    result_types: &[Type<'c>],
    before_region: Region,
    after_region: Region,
    loc: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("scf.while", loc)
        .add_operands(initial_values)
        .add_results(result_types)
        .add_regions(vec![before_region, after_region])
        .build()
}
pub fn r#yield<'c>(values: &[Value], loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("scf.yield", loc).add_operands(values).build()
}

#[cfg(test)]
mod test;
