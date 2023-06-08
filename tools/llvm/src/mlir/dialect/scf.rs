use crate::mlir::{
    ir::{attr::DenseI64ArrayAttribute, op::OperationBuilder, Identifier, Location, Operation, Region, Type, Value},
    Context,
};

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
mod tests {
    use super::*;
    use crate::mlir::{
        dialect::{arith, func},
        ir::{
            attr::{FloatAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
            r#type::{FunctionType, IntegerType, Type},
            Attribute, Block, Module,
        },
        test::load_all_dialects,
        Context,
    };
    #[test]
    fn compile_execute_region() {
        let ctx = Context::new();
        load_all_dialects(&ctx);
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        let index_type = Type::index(&ctx);
        module.body().append_operation(func::func(
            &ctx,
            StringAttribute::new(&ctx, "foo"),
            TypeAttribute::new(FunctionType::new(&ctx, &[], &[]).into()),
            {
                let block = Block::new(&[]);
                block.append_operation(execute_region(
                    &[index_type],
                    {
                        let block = Block::new(&[]);
                        let value = block.append_operation(arith::constant(
                            &ctx,
                            IntegerAttribute::new(0, index_type).into(),
                            loc,
                        ));
                        block.append_operation(r#yield(&[value.result(0).unwrap().into()], loc));
                        let region = Region::new();
                        region.append_block(block);
                        region
                    },
                    loc,
                ));
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
    fn compile_for() {
        let ctx = Context::new();
        load_all_dialects(&ctx);
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        module.body().append_operation(func::func(
            &ctx,
            StringAttribute::new(&ctx, "foo"),
            TypeAttribute::new(FunctionType::new(&ctx, &[], &[]).into()),
            {
                let block = Block::new(&[]);
                let start =
                    block.append_operation(arith::constant(&ctx, Attribute::parse(&ctx, "0 : index").unwrap(), loc));
                let end =
                    block.append_operation(arith::constant(&ctx, Attribute::parse(&ctx, "8 : index").unwrap(), loc));
                let step =
                    block.append_operation(arith::constant(&ctx, Attribute::parse(&ctx, "1 : index").unwrap(), loc));
                block.append_operation(r#for(
                    start.result(0).unwrap().into(),
                    end.result(0).unwrap().into(),
                    step.result(0).unwrap().into(),
                    {
                        let block = Block::new(&[(Type::index(&ctx), loc)]);
                        block.append_operation(r#yield(&[], loc));
                        let region = Region::new();
                        region.append_block(block);
                        region
                    },
                    loc,
                ));
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
    mod r#if {
        use super::*;
        #[test]
        fn compile() {
            let ctx = Context::new();
            load_all_dialects(&ctx);
            let loc = Location::unknown(&ctx);
            let module = Module::new(loc);
            let index_type = Type::index(&ctx);
            module.body().append_operation(func::func(
                &ctx,
                StringAttribute::new(&ctx, "foo"),
                TypeAttribute::new(FunctionType::new(&ctx, &[], &[index_type]).into()),
                {
                    let block = Block::new(&[]);
                    let cond = block.append_operation(arith::constant(
                        &ctx,
                        IntegerAttribute::new(0, IntegerType::new(&ctx, 1).into()).into(),
                        loc,
                    ));
                    let result = block.append_operation(r#if(
                        cond.result(0).unwrap().into(),
                        &[index_type],
                        {
                            let block = Block::new(&[]);
                            let result = block.append_operation(arith::constant(
                                &ctx,
                                IntegerAttribute::new(42, index_type).into(),
                                loc,
                            ));
                            block.append_operation(r#yield(&[result.result(0).unwrap().into()], loc));
                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        {
                            let block = Block::new(&[]);
                            let result = block.append_operation(arith::constant(
                                &ctx,
                                IntegerAttribute::new(13, index_type).into(),
                                loc,
                            ));
                            block.append_operation(r#yield(&[result.result(0).unwrap().into()], loc));
                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        loc,
                    ));
                    block.append_operation(func::r#return(&[result.result(0).unwrap().into()], loc));
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
        fn compile_one_sided() {
            let ctx = Context::new();
            load_all_dialects(&ctx);
            let loc = Location::unknown(&ctx);
            let module = Module::new(loc);
            module.body().append_operation(func::func(
                &ctx,
                StringAttribute::new(&ctx, "foo"),
                TypeAttribute::new(FunctionType::new(&ctx, &[], &[]).into()),
                {
                    let block = Block::new(&[]);
                    let cond = block.append_operation(arith::constant(
                        &ctx,
                        IntegerAttribute::new(0, IntegerType::new(&ctx, 1).into()).into(),
                        loc,
                    ));
                    block.append_operation(r#if(
                        cond.result(0).unwrap().into(),
                        &[],
                        {
                            let block = Block::new(&[]);
                            block.append_operation(r#yield(&[], loc));
                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        Region::new(),
                        loc,
                    ));
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
    }
    #[test]
    fn compile_index_switch() {
        let ctx = Context::new();
        load_all_dialects(&ctx);
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        module.body().append_operation(func::func(
            &ctx,
            StringAttribute::new(&ctx, "foo"),
            TypeAttribute::new(FunctionType::new(&ctx, &[], &[]).into()),
            {
                let block = Block::new(&[]);
                let cond = block.append_operation(arith::constant(
                    &ctx,
                    IntegerAttribute::new(0, Type::index(&ctx)).into(),
                    loc,
                ));
                block.append_operation(index_switch(
                    &ctx,
                    cond.result(0).unwrap().into(),
                    &[],
                    DenseI64ArrayAttribute::new(&ctx, &[0, 1]),
                    vec![
                        {
                            let block = Block::new(&[]);
                            block.append_operation(r#yield(&[], loc));
                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        {
                            let block = Block::new(&[]);
                            block.append_operation(r#yield(&[], loc));
                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        {
                            let block = Block::new(&[]);
                            block.append_operation(r#yield(&[], loc));
                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                    ],
                    loc,
                ));
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
    mod r#while {
        use super::*;
        #[test]
        fn compile() {
            let ctx = Context::new();
            load_all_dialects(&ctx);
            let loc = Location::unknown(&ctx);
            let module = Module::new(loc);
            let index_type = Type::index(&ctx);
            module.body().append_operation(func::func(
                &ctx,
                StringAttribute::new(&ctx, "foo"),
                TypeAttribute::new(FunctionType::new(&ctx, &[], &[]).into()),
                {
                    let block = Block::new(&[]);
                    let initial =
                        block.append_operation(arith::constant(&ctx, IntegerAttribute::new(0, index_type).into(), loc));
                    block.append_operation(r#while(
                        &[initial.result(0).unwrap().into()],
                        &[index_type],
                        {
                            let block = Block::new(&[(index_type, loc)]);
                            let cond = block.append_operation(arith::constant(
                                &ctx,
                                IntegerAttribute::new(0, IntegerType::new(&ctx, 1).into()).into(),
                                loc,
                            ));
                            let result = block.append_operation(arith::constant(
                                &ctx,
                                IntegerAttribute::new(42, Type::index(&ctx)).into(),
                                loc,
                            ));
                            block.append_operation(super::cond(
                                cond.result(0).unwrap().into(),
                                &[result.result(0).unwrap().into()],
                                loc,
                            ));
                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        {
                            let block = Block::new(&[(index_type, loc)]);
                            let result = block.append_operation(arith::constant(
                                &ctx,
                                IntegerAttribute::new(42, index_type).into(),
                                loc,
                            ));
                            block.append_operation(r#yield(&[result.result(0).unwrap().into()], loc));
                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        loc,
                    ));
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
        fn compile_with_different_argument_and_result_types() {
            let ctx = Context::new();
            load_all_dialects(&ctx);
            let loc = Location::unknown(&ctx);
            let module = Module::new(loc);
            let index_type = Type::index(&ctx);
            let float_type = Type::float64(&ctx);
            module.body().append_operation(func::func(
                &ctx,
                StringAttribute::new(&ctx, "foo"),
                TypeAttribute::new(FunctionType::new(&ctx, &[], &[]).into()),
                {
                    let block = Block::new(&[]);
                    let initial =
                        block.append_operation(arith::constant(&ctx, IntegerAttribute::new(0, index_type).into(), loc));
                    block.append_operation(r#while(
                        &[initial.result(0).unwrap().into()],
                        &[float_type],
                        {
                            let block = Block::new(&[(index_type, loc)]);
                            let cond = block.append_operation(arith::constant(
                                &ctx,
                                IntegerAttribute::new(0, IntegerType::new(&ctx, 1).into()).into(),
                                loc,
                            ));
                            let result = block.append_operation(arith::constant(
                                &ctx,
                                FloatAttribute::new(&ctx, 42.0, float_type).into(),
                                loc,
                            ));
                            block.append_operation(super::cond(
                                cond.result(0).unwrap().into(),
                                &[result.result(0).unwrap().into()],
                                loc,
                            ));
                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        {
                            let block = Block::new(&[(float_type, loc)]);
                            let result = block.append_operation(arith::constant(
                                &ctx,
                                IntegerAttribute::new(42, Type::index(&ctx)).into(),
                                loc,
                            ));
                            block.append_operation(r#yield(&[result.result(0).unwrap().into()], loc));
                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        loc,
                    ));
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
        fn compile_with_multiple_arguments_and_results() {
            let ctx = Context::new();
            load_all_dialects(&ctx);
            let loc = Location::unknown(&ctx);
            let module = Module::new(loc);
            let index_type = Type::index(&ctx);
            module.body().append_operation(func::func(
                &ctx,
                StringAttribute::new(&ctx, "foo"),
                TypeAttribute::new(FunctionType::new(&ctx, &[], &[]).into()),
                {
                    let block = Block::new(&[]);
                    let initial =
                        block.append_operation(arith::constant(&ctx, IntegerAttribute::new(0, index_type).into(), loc));
                    block.append_operation(r#while(
                        &[initial.result(0).unwrap().into(), initial.result(0).unwrap().into()],
                        &[index_type, index_type],
                        {
                            let block = Block::new(&[(index_type, loc), (index_type, loc)]);
                            let cond = block.append_operation(arith::constant(
                                &ctx,
                                IntegerAttribute::new(0, IntegerType::new(&ctx, 1).into()).into(),
                                loc,
                            ));
                            let result = block.append_operation(arith::constant(
                                &ctx,
                                IntegerAttribute::new(42, Type::index(&ctx)).into(),
                                loc,
                            ));
                            block.append_operation(super::cond(
                                cond.result(0).unwrap().into(),
                                &[result.result(0).unwrap().into(), result.result(0).unwrap().into()],
                                loc,
                            ));
                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        {
                            let block = Block::new(&[(index_type, loc), (index_type, loc)]);
                            let result = block.append_operation(arith::constant(
                                &ctx,
                                IntegerAttribute::new(42, index_type).into(),
                                loc,
                            ));
                            block.append_operation(r#yield(
                                &[result.result(0).unwrap().into(), result.result(0).unwrap().into()],
                                loc,
                            ));
                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        loc,
                    ));
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
    }
}
