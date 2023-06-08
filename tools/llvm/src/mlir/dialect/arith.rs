use crate::mlir::{
    ir::{
        attr::IntegerAttribute, op::OperationBuilder, typ::IntegerType, Attribute, Identifier, Location, Operation,
        Value,
    },
    Context,
};

pub fn constant<'c>(ctx: &'c Context, value: Attribute<'c>, loc: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("arith.constant", loc)
        .add_attributes(&[(Identifier::new(ctx, "value"), value)])
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlir::{
        dialect::func,
        ir::{
            attr::{StringAttribute, TypeAttribute},
            typ::FunctionType,
            Attribute, Block, Location, Module, Region, Type,
        },
        test::load_all_dialects,
        Context,
    };
    fn create_ctx() -> Context {
        let ctx = Context::new();
        load_all_dialects(&ctx);
        ctx
    }
    fn compile_operation<'c>(
        ctx: &'c Context,
        operation: impl Fn(&Block<'c>) -> Operation<'c>,
        block_argument_types: &[Type<'c>],
        function_type: FunctionType<'c>,
    ) {
        let loc = Location::unknown(ctx);
        let module = Module::new(loc);
        let block = Block::new(
            &block_argument_types
                .iter()
                .map(|&r#type| (r#type, loc))
                .collect::<Vec<_>>(),
        );
        let operation = operation(&block);
        let name = operation.name();
        let name = name.as_string_ref().as_str().unwrap();
        block.append_operation(func::r#return(
            &[block.append_operation(operation).result(0).unwrap().into()],
            loc,
        ));
        let region = Region::new();
        region.append_block(block);
        let function = func::func(
            ctx,
            StringAttribute::new(ctx, "foo"),
            TypeAttribute::new(function_type.into()),
            region,
            &[],
            Location::unknown(ctx),
        );
        module.body().append_operation(function);
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(name, module.as_operation());
    }
    #[test]
    fn compile_constant() {
        let ctx = create_ctx();
        let integer_type = IntegerType::new(&ctx, 64).into();
        compile_operation(
            &ctx,
            |_| {
                constant(
                    &ctx,
                    Attribute::parse(&ctx, "42 : i64").unwrap(),
                    Location::unknown(&ctx),
                )
            },
            &[integer_type],
            FunctionType::new(&ctx, &[integer_type], &[integer_type]),
        );
    }
    #[test]
    fn compile_negf() {
        let ctx = create_ctx();
        let f64_type = Type::float64(&ctx);
        compile_operation(
            &ctx,
            |block| negf(block.argument(0).unwrap().into(), Location::unknown(&ctx)),
            &[Type::float64(&ctx)],
            FunctionType::new(&ctx, &[f64_type], &[f64_type]),
        );
    }
    mod cmp {
        use super::*;
        #[test]
        fn compile_cmpf() {
            let ctx = create_ctx();
            let float_type = Type::float64(&ctx);
            compile_operation(
                &ctx,
                |block| {
                    cmpf(
                        &ctx,
                        CmpfPredicate::Oeq,
                        block.argument(0).unwrap().into(),
                        block.argument(1).unwrap().into(),
                        Location::unknown(&ctx),
                    )
                },
                &[float_type, float_type],
                FunctionType::new(&ctx, &[float_type, float_type], &[IntegerType::new(&ctx, 1).into()]),
            );
        }
        #[test]
        fn compile_cmpi() {
            let ctx = create_ctx();
            let integer_type = IntegerType::new(&ctx, 64).into();
            compile_operation(
                &ctx,
                |block| {
                    cmpi(
                        &ctx,
                        CmpiPredicate::Eq,
                        block.argument(0).unwrap().into(),
                        block.argument(1).unwrap().into(),
                        Location::unknown(&ctx),
                    )
                },
                &[integer_type, integer_type],
                FunctionType::new(&ctx, &[integer_type, integer_type], &[IntegerType::new(&ctx, 1).into()]),
            );
        }
    }
    mod typed_unary {
        use super::*;
        #[test]
        fn compile_bitcast() {
            let ctx = create_ctx();
            let integer_type = IntegerType::new(&ctx, 64).into();
            let float_type = Type::float64(&ctx);
            compile_operation(
                &ctx,
                |block| bitcast(block.argument(0).unwrap().into(), float_type, Location::unknown(&ctx)),
                &[integer_type],
                FunctionType::new(&ctx, &[integer_type], &[float_type]),
            );
        }
        #[test]
        fn compile_extf() {
            let ctx = create_ctx();
            compile_operation(
                &ctx,
                |block| {
                    extf(
                        block.argument(0).unwrap().into(),
                        Type::float64(&ctx),
                        Location::unknown(&ctx),
                    )
                },
                &[Type::float32(&ctx)],
                FunctionType::new(&ctx, &[Type::float32(&ctx)], &[Type::float64(&ctx)]),
            );
        }
        #[test]
        fn compile_extsi() {
            let ctx = create_ctx();
            compile_operation(
                &ctx,
                |block| {
                    extsi(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&ctx, 64).into(),
                        Location::unknown(&ctx),
                    )
                },
                &[IntegerType::new(&ctx, 32).into()],
                FunctionType::new(
                    &ctx,
                    &[IntegerType::new(&ctx, 32).into()],
                    &[IntegerType::new(&ctx, 64).into()],
                ),
            );
        }
        #[test]
        fn compile_extui() {
            let ctx = create_ctx();
            compile_operation(
                &ctx,
                |block| {
                    extui(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&ctx, 64).into(),
                        Location::unknown(&ctx),
                    )
                },
                &[IntegerType::new(&ctx, 32).into()],
                FunctionType::new(
                    &ctx,
                    &[IntegerType::new(&ctx, 32).into()],
                    &[IntegerType::new(&ctx, 64).into()],
                ),
            );
        }
        #[test]
        fn compile_fptosi() {
            let ctx = create_ctx();
            compile_operation(
                &ctx,
                |block| {
                    fptosi(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&ctx, 64).into(),
                        Location::unknown(&ctx),
                    )
                },
                &[Type::float32(&ctx)],
                FunctionType::new(&ctx, &[Type::float32(&ctx)], &[IntegerType::new(&ctx, 64).into()]),
            );
        }
        #[test]
        fn compile_fptoui() {
            let ctx = create_ctx();
            compile_operation(
                &ctx,
                |block| {
                    fptoui(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&ctx, 64).into(),
                        Location::unknown(&ctx),
                    )
                },
                &[Type::float32(&ctx)],
                FunctionType::new(&ctx, &[Type::float32(&ctx)], &[IntegerType::new(&ctx, 64).into()]),
            );
        }
        #[test]
        fn compile_index_cast() {
            let ctx = create_ctx();
            compile_operation(
                &ctx,
                |block| {
                    index_cast(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&ctx, 64).into(),
                        Location::unknown(&ctx),
                    )
                },
                &[Type::index(&ctx)],
                FunctionType::new(&ctx, &[Type::index(&ctx)], &[IntegerType::new(&ctx, 64).into()]),
            );
        }
        #[test]
        fn compile_index_castui() {
            let ctx = create_ctx();
            compile_operation(
                &ctx,
                |block| {
                    index_castui(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&ctx, 64).into(),
                        Location::unknown(&ctx),
                    )
                },
                &[Type::index(&ctx)],
                FunctionType::new(&ctx, &[Type::index(&ctx)], &[IntegerType::new(&ctx, 64).into()]),
            );
        }
        #[test]
        fn compile_sitofp() {
            let ctx = create_ctx();
            compile_operation(
                &ctx,
                |block| {
                    sitofp(
                        block.argument(0).unwrap().into(),
                        Type::float64(&ctx),
                        Location::unknown(&ctx),
                    )
                },
                &[IntegerType::new(&ctx, 32).into()],
                FunctionType::new(&ctx, &[IntegerType::new(&ctx, 32).into()], &[Type::float64(&ctx)]),
            );
        }
        #[test]
        fn compile_trunci() {
            let ctx = create_ctx();
            compile_operation(
                &ctx,
                |block| {
                    trunci(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&ctx, 32).into(),
                        Location::unknown(&ctx),
                    )
                },
                &[IntegerType::new(&ctx, 64).into()],
                FunctionType::new(
                    &ctx,
                    &[IntegerType::new(&ctx, 64).into()],
                    &[IntegerType::new(&ctx, 32).into()],
                ),
            );
        }
        #[test]
        fn compile_uitofp() {
            let ctx = create_ctx();
            compile_operation(
                &ctx,
                |block| {
                    uitofp(
                        block.argument(0).unwrap().into(),
                        Type::float64(&ctx),
                        Location::unknown(&ctx),
                    )
                },
                &[IntegerType::new(&ctx, 32).into()],
                FunctionType::new(&ctx, &[IntegerType::new(&ctx, 32).into()], &[Type::float64(&ctx)]),
            );
        }
    }
    #[test]
    fn compile_addi() {
        let ctx = Context::new();
        load_all_dialects(&ctx);
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        let integer_type = IntegerType::new(&ctx, 64).into();
        let function = {
            let block = Block::new(&[(integer_type, loc), (integer_type, loc)]);
            let sum = block.append_operation(addi(
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                loc,
            ));
            block.append_operation(func::r#return(&[sum.result(0).unwrap().into()], loc));
            let region = Region::new();
            region.append_block(block);
            func::func(
                &ctx,
                StringAttribute::new(&ctx, "foo"),
                TypeAttribute::new(FunctionType::new(&ctx, &[integer_type, integer_type], &[integer_type]).into()),
                region,
                &[],
                Location::unknown(&ctx),
            )
        };
        module.body().append_operation(function);
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
}
