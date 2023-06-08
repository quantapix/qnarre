use crate::mlir::{
    ir::{attr::*, op::OperationBuilder, typ::MemRefType, Attribute, Identifier, Location, Operation, Value},
    Context,
};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::{func, index},
        ir::{
            attr::{DenseElementsAttribute, StringAttribute, TypeAttribute},
            ty::{FunctionType, IntegerType, RankedTensorType},
            Block, Module, Region, Type,
        },
        test::create_test_ctx,
    };
    fn compile_operation(name: &str, ctx: &Context, build_block: impl Fn(&Block)) {
        let loc = Location::unknown(ctx);
        let module = Module::new(loc);
        let function = {
            let block = Block::new(&[]);
            build_block(&block);
            block.append_operation(func::r#return(&[], loc));
            let region = Region::new();
            region.append_block(block);
            func::func(
                ctx,
                StringAttribute::new(ctx, "foo"),
                TypeAttribute::new(FunctionType::new(ctx, &[], &[]).into()),
                region,
                &[],
                Location::unknown(ctx),
            )
        };
        module.body().append_operation(function);
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(name, module.as_operation());
    }
    #[test]
    fn compile_alloc_and_dealloc() {
        let ctx = create_test_ctx();
        let loc = Location::unknown(&ctx);
        compile_operation("alloc", &ctx, |block| {
            let memref = block.append_operation(alloc(
                &ctx,
                MemRefType::new(Type::index(&ctx), &[], None, None),
                &[],
                &[],
                None,
                loc,
            ));
            block.append_operation(dealloc(memref.result(0).unwrap().into(), loc));
        })
    }
    #[test]
    fn compile_alloc_and_realloc() {
        let ctx = create_test_ctx();
        let loc = Location::unknown(&ctx);
        compile_operation("realloc", &ctx, |block| {
            let memref = block.append_operation(alloc(
                &ctx,
                MemRefType::new(Type::index(&ctx), &[8], None, None),
                &[],
                &[],
                None,
                loc,
            ));
            block.append_operation(realloc(
                &ctx,
                memref.result(0).unwrap().into(),
                None,
                MemRefType::new(Type::index(&ctx), &[42], None, None),
                None,
                loc,
            ));
        })
    }
    #[test]
    fn compile_alloca() {
        let ctx = create_test_ctx();
        let loc = Location::unknown(&ctx);
        compile_operation("alloca", &ctx, |block| {
            block.append_operation(alloca(
                &ctx,
                MemRefType::new(Type::index(&ctx), &[], None, None),
                &[],
                &[],
                None,
                loc,
            ));
        })
    }
    #[test]
    fn compile_cast() {
        let ctx = create_test_ctx();
        let loc = Location::unknown(&ctx);
        compile_operation("cast", &ctx, |block| {
            let memref = block.append_operation(alloca(
                &ctx,
                MemRefType::new(Type::float64(&ctx), &[42], None, None),
                &[],
                &[],
                None,
                loc,
            ));
            block.append_operation(cast(
                memref.result(0).unwrap().into(),
                Type::parse(&ctx, "memref<?xf64>").unwrap().try_into().unwrap(),
                loc,
            ));
        })
    }
    #[test]
    fn compile_dim() {
        let ctx = create_test_ctx();
        let loc = Location::unknown(&ctx);
        compile_operation("dim", &ctx, |block| {
            let memref = block.append_operation(alloca(
                &ctx,
                MemRefType::new(Type::index(&ctx), &[1], None, None),
                &[],
                &[],
                None,
                loc,
            ));
            let index = block.append_operation(index::constant(&ctx, IntegerAttribute::new(0, Type::index(&ctx)), loc));
            block.append_operation(dim(
                memref.result(0).unwrap().into(),
                index.result(0).unwrap().into(),
                loc,
            ));
        })
    }
    #[test]
    fn compile_get_global() {
        let ctx = create_test_ctx();
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        let mem_ref_type = MemRefType::new(Type::index(&ctx), &[], None, None);
        module
            .body()
            .append_operation(global(&ctx, "foo", None, mem_ref_type, None, false, None, loc));
        module.body().append_operation(func::func(
            &ctx,
            StringAttribute::new(&ctx, "bar"),
            TypeAttribute::new(FunctionType::new(&ctx, &[], &[]).into()),
            {
                let block = Block::new(&[]);
                block.append_operation(get_global(&ctx, "foo", mem_ref_type, loc));
                block.append_operation(func::r#return(&[], loc));
                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            loc,
        ));
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
    #[test]
    fn compile_global() {
        let ctx = create_test_ctx();
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        module.body().append_operation(global(
            &ctx,
            "foo",
            None,
            MemRefType::new(Type::index(&ctx), &[], None, None),
            None,
            false,
            None,
            loc,
        ));
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
    #[test]
    fn compile_global_with_options() {
        let ctx = create_test_ctx();
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        let ty = IntegerType::new(&ctx, 64).into();
        module.body().append_operation(global(
            &ctx,
            "foo",
            Some("private"),
            MemRefType::new(ty, &[], None, None),
            Some(
                DenseElementsAttribute::new(
                    RankedTensorType::new(&[], ty, None).into(),
                    &[IntegerAttribute::new(42, ty).into()],
                )
                .unwrap()
                .into(),
            ),
            true,
            Some(IntegerAttribute::new(8, IntegerType::new(&ctx, 64).into())),
            loc,
        ));
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
    #[test]
    fn compile_load() {
        let ctx = create_test_ctx();
        let loc = Location::unknown(&ctx);
        compile_operation("load", &ctx, |block| {
            let memref = block.append_operation(alloca(
                &ctx,
                MemRefType::new(Type::index(&ctx), &[], None, None),
                &[],
                &[],
                None,
                loc,
            ));
            block.append_operation(load(memref.result(0).unwrap().into(), &[], loc));
        })
    }
    #[test]
    fn compile_load_with_index() {
        let ctx = create_test_ctx();
        let loc = Location::unknown(&ctx);
        compile_operation("load_with_index", &ctx, |block| {
            let memref = block.append_operation(alloca(
                &ctx,
                MemRefType::new(Type::index(&ctx), &[1], None, None),
                &[],
                &[],
                None,
                loc,
            ));
            let index = block.append_operation(index::constant(&ctx, IntegerAttribute::new(0, Type::index(&ctx)), loc));
            block.append_operation(load(
                memref.result(0).unwrap().into(),
                &[index.result(0).unwrap().into()],
                loc,
            ));
        })
    }
    #[test]
    fn compile_rank() {
        let ctx = create_test_ctx();
        let loc = Location::unknown(&ctx);
        compile_operation("rank", &ctx, |block| {
            let memref = block.append_operation(alloca(
                &ctx,
                MemRefType::new(Type::index(&ctx), &[1], None, None),
                &[],
                &[],
                None,
                loc,
            ));
            block.append_operation(rank(memref.result(0).unwrap().into(), loc));
        })
    }
    #[test]
    fn compile_store() {
        let ctx = create_test_ctx();
        let loc = Location::unknown(&ctx);
        compile_operation("store", &ctx, |block| {
            let memref = block.append_operation(alloca(
                &ctx,
                MemRefType::new(Type::index(&ctx), &[], None, None),
                &[],
                &[],
                None,
                loc,
            ));
            let value =
                block.append_operation(index::constant(&ctx, IntegerAttribute::new(42, Type::index(&ctx)), loc));
            block.append_operation(store(
                value.result(0).unwrap().into(),
                memref.result(0).unwrap().into(),
                &[],
                loc,
            ));
        })
    }
    #[test]
    fn compile_store_with_index() {
        let ctx = create_test_ctx();
        let loc = Location::unknown(&ctx);
        compile_operation("store_with_index", &ctx, |block| {
            let memref = block.append_operation(alloca(
                &ctx,
                MemRefType::new(Type::index(&ctx), &[1], None, None),
                &[],
                &[],
                None,
                loc,
            ));
            let value =
                block.append_operation(index::constant(&ctx, IntegerAttribute::new(42, Type::index(&ctx)), loc));
            let index = block.append_operation(index::constant(&ctx, IntegerAttribute::new(0, Type::index(&ctx)), loc));
            block.append_operation(store(
                value.result(0).unwrap().into(),
                memref.result(0).unwrap().into(),
                &[index.result(0).unwrap().into()],
                loc,
            ));
        })
    }
}
