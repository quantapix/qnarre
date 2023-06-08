use super::arith::CmpiPredicate;
use crate::mlir::{
    ir::{attr::IntegerAttribute, op::OperationBuilder, Attribute, Identifier, Location, Operation, Value},
    Context,
};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlir::{
        dialect::func,
        ir::{
            attr::{StringAttribute, TypeAttribute},
            typ::{FunctionType, IntegerType},
            Block, Location, Module, Region, Type,
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
        function_type: FunctionType<'c>,
    ) {
        let loc = Location::unknown(ctx);
        let module = Module::new(loc);
        let block = Block::new(
            &(0..function_type.input_count())
                .map(|index| (function_type.input(index).unwrap(), loc))
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
        let index_type = Type::index(&ctx);
        compile_operation(
            &ctx,
            |_| constant(&ctx, IntegerAttribute::new(42, index_type), Location::unknown(&ctx)),
            FunctionType::new(&ctx, &[index_type], &[index_type]),
        );
    }
    #[test]
    fn compile_cmp() {
        let ctx = create_ctx();
        let index_type = Type::index(&ctx);
        compile_operation(
            &ctx,
            |block| {
                cmp(
                    &ctx,
                    CmpiPredicate::Eq,
                    block.argument(0).unwrap().into(),
                    block.argument(1).unwrap().into(),
                    Location::unknown(&ctx),
                )
            },
            FunctionType::new(&ctx, &[index_type, index_type], &[IntegerType::new(&ctx, 1).into()]),
        );
    }
    mod typed_unary {
        use super::*;
        #[test]
        fn compile_casts() {
            let ctx = create_ctx();
            compile_operation(
                &ctx,
                |block| {
                    casts(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&ctx, 64).into(),
                        Location::unknown(&ctx),
                    )
                },
                FunctionType::new(&ctx, &[Type::index(&ctx)], &[IntegerType::new(&ctx, 64).into()]),
            );
        }
        #[test]
        fn compile_castu() {
            let ctx = create_ctx();
            compile_operation(
                &ctx,
                |block| {
                    castu(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&ctx, 64).into(),
                        Location::unknown(&ctx),
                    )
                },
                FunctionType::new(&ctx, &[Type::index(&ctx)], &[IntegerType::new(&ctx, 64).into()]),
            );
        }
    }
    #[test]
    fn compile_add() {
        let ctx = Context::new();
        load_all_dialects(&ctx);
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        let integer_type = Type::index(&ctx);
        let function = {
            let block = Block::new(&[(integer_type, loc), (integer_type, loc)]);
            let sum = block.append_operation(add(
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
