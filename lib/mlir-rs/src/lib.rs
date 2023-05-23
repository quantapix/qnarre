#![doc = include_str!("../README.md")]

#[macro_use]
mod r#macro;
mod context;
pub mod diagnostic;
pub mod dialect;
mod error;
mod execution_engine;
pub mod ir;
mod logical_result;
pub mod pass;
mod string_ref;
#[cfg(test)]
mod test;
pub mod utility;

pub use self::{
    context::{Context, ContextRef},
    error::Error,
    execution_engine::ExecutionEngine,
    string_ref::StringRef,
};

#[cfg(test)]
mod tests {
    use crate::{
        context::Context,
        dialect::{self, arith, func, scf},
        ir::{
            attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
            operation::OperationBuilder,
            r#type::{FunctionType, IntegerType},
            Block, Location, Module, Region, Type, Value,
        },
        test::load_all_dialects,
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
                TypeAttribute::new(
                    FunctionType::new(&context, &[integer_type, integer_type], &[integer_type])
                        .into(),
                ),
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
                TypeAttribute::new(
                    FunctionType::new(&context, &[memref_type, memref_type], &[]).into(),
                ),
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

        fn compile_add<'a>(
            context: &Context,
            block: &'a Block,
            lhs: Value<'a>,
            rhs: Value<'a>,
        ) -> Value<'a> {
            block
                .append_operation(arith::addi(lhs, rhs, Location::unknown(context)))
                .result(0)
                .unwrap()
                .into()
        }

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "add"),
            TypeAttribute::new(
                FunctionType::new(&context, &[integer_type, integer_type], &[integer_type]).into(),
            ),
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
}
