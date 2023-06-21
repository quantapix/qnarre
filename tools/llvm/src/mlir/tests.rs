#[cfg(test)]
mod mlir {
    use crate::mlir::{
        dialect::{self, arith, func, scf},
        ir::*,
        pass,
        test::{create_test_context, load_all_dialects},
        Context,
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
        let pass_manager = PassManager::new(&context);
        pass_manager.add_pass(conversion::create_func_to_llvm());
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
        let pass_manager = PassManager::new(&context);
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

#[cfg(test)]
mod ctx {
    use crate::mlir::{dialect::DialectRegistry, Context};
    #[test]
    fn new() {
        Context::new();
    }
    #[test]
    fn registered_dialect_count() {
        let context = Context::new();
        assert_eq!(context.registered_dialect_count(), 1);
    }
    #[test]
    fn loaded_dialect_count() {
        let context = Context::new();
        assert_eq!(context.loaded_dialect_count(), 1);
    }
    #[test]
    fn append_dialect_registry() {
        let context = Context::new();
        context.append_dialect_registry(&DialectRegistry::new());
    }
    #[test]
    fn is_registered_operation() {
        let context = Context::new();
        assert!(context.is_registered_operation("builtin.module"));
    }
    #[test]
    fn is_not_registered_operation() {
        let context = Context::new();
        assert!(!context.is_registered_operation("func.func"));
    }
    #[test]
    fn enable_multi_threading() {
        let context = Context::new();
        context.enable_multi_threading(true);
    }
    #[test]
    fn disable_multi_threading() {
        let context = Context::new();
        context.enable_multi_threading(false);
    }
    #[test]
    fn allow_unregistered_dialects() {
        let context = Context::new();
        assert!(!context.allow_unregistered_dialects());
    }
    #[test]
    fn set_allow_unregistered_dialects() {
        let context = Context::new();
        context.set_allow_unregistered_dialects(true);
        assert!(context.allow_unregistered_dialects());
    }
    #[test]
    fn attach_and_detach_diagnostic_handler() {
        let context = Context::new();
        let id = context.attach_diagnostic_handler(|diagnostic| {
            println!("{}", diagnostic);
            true
        });
        context.detach_diagnostic_handler(id);
    }
}

#[cfg(test)]
mod diag {
    use crate::mlir::{ir::Module, Context};
    #[test]
    fn handle_diagnostic() {
        let mut message = None;
        let context = Context::new();
        context.attach_diagnostic_handler(|diagnostic| {
            message = Some(diagnostic.to_string());
            true
        });
        Module::parse(&context, "foo");
        assert_eq!(
            message.unwrap(),
            "custom op 'foo' is unknown (tried 'builtin.foo' as well)"
        );
    }
}

#[cfg(test)]
mod pass {
    use crate::mlir::{
        dialect::DialectRegistry,
        ir::{Location, Module},
        parse_pass_pipeline,
        pass::{self, transform::register_print_op_stats},
        register_all_dialects,
    };
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    fn register_all_upstream_dialects(context: &Context) {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
    }
    #[test]
    fn new() {
        let context = Context::new();
        PassManager::new(&context);
    }
    #[test]
    fn add_pass() {
        let context = Context::new();
        PassManager::new(&context).add_pass(pass::conversion::create_func_to_llvm());
    }
    #[test]
    fn enable_verifier() {
        let context = Context::new();
        PassManager::new(&context).enable_verifier(true);
    }
    #[test]
    fn run() {
        let context = Context::new();
        let manager = PassManager::new(&context);
        manager.add_pass(pass::conversion::create_func_to_llvm());
        manager.run(&mut Module::new(Location::unknown(&context))).unwrap();
    }
    #[test]
    fn run_on_function() {
        let context = Context::new();
        register_all_upstream_dialects(&context);
        let mut module = Module::parse(
            &context,
            indoc!(
                "
                func.func @foo(%arg0 : i32) -> i32 {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }
                "
            ),
        )
        .unwrap();
        let manager = PassManager::new(&context);
        manager.add_pass(pass::transform::create_print_op_stats());
        assert_eq!(manager.run(&mut module), Ok(()));
    }
    #[test]
    fn run_on_function_in_nested_module() {
        let context = Context::new();
        register_all_upstream_dialects(&context);
        let mut module = Module::parse(
            &context,
            indoc!(
                "
                func.func @foo(%arg0 : i32) -> i32 {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }
                module {
                    func.func @bar(%arg0 : f32) -> f32 {
                        %res = arith.addf %arg0, %arg0 : f32
                        return %res : f32
                    }
                }
                "
            ),
        )
        .unwrap();
        let manager = PassManager::new(&context);
        manager
            .nested_under("func.func")
            .add_pass(pass::transform::create_print_op_stats());
        assert_eq!(manager.run(&mut module), Ok(()));
        let manager = PassManager::new(&context);
        manager
            .nested_under("builtin.module")
            .nested_under("func.func")
            .add_pass(pass::transform::create_print_op_stats());
        assert_eq!(manager.run(&mut module), Ok(()));
    }
    #[test]
    fn print_pass_pipeline() {
        let context = Context::new();
        let manager = PassManager::new(&context);
        let function_manager = manager.nested_under("func.func");
        function_manager.add_pass(pass::transform::create_print_op_stats());
        assert_eq!(
            manager.as_operation_pass_manager().to_string(),
            "builtin.module(func.func(print-op-stats{json=false}))"
        );
        assert_eq!(function_manager.to_string(), "func.func(print-op-stats{json=false})");
    }
    #[test]
    fn parse_pass_pipeline_() {
        let context = Context::new();
        let manager = PassManager::new(&context);
        insta::assert_display_snapshot!(parse_pass_pipeline(
            manager.as_operation_pass_manager(),
            "builtin.module(func.func(print-op-stats{json=false}),\
                func.func(print-op-stats{json=false}))"
        )
        .unwrap_err());
        register_print_op_stats();
        assert_eq!(
            parse_pass_pipeline(
                manager.as_operation_pass_manager(),
                "builtin.module(func.func(print-op-stats{json=false}),\
                func.func(print-op-stats{json=false}))"
            ),
            Ok(())
        );
        assert_eq!(
            manager.as_operation_pass_manager().to_string(),
            "builtin.module(func.func(print-op-stats{json=false}),\
            func.func(print-op-stats{json=false}))"
        );
    }
}

#[cfg(test)]
mod utils {
    use crate::mlir::{dialect::DialectRegistry, Context};
    #[test]
    fn register_dialects() {
        let y = DialectRegistry::new();
        register_all_dialects(&y);
    }
    #[test]
    fn register_dialects_twice() {
        let y = DialectRegistry::new();
        register_all_dialects(&y);
        register_all_dialects(&y);
    }
    #[test]
    fn register_llvm_translations() {
        let y = Context::new();
        register_all_llvm_translations(&y);
    }
    #[test]
    fn register_llvm_translations_twice() {
        let y = Context::new();
        register_all_llvm_translations(&y);
        register_all_llvm_translations(&y);
    }
    #[test]
    fn register_passes() {
        register_all_passes();
    }
    #[test]
    fn register_passes_twice() {
        register_all_passes();
        register_all_passes();
    }
    #[test]
    fn register_passes_many_times() {
        for _ in 0..1000 {
            register_all_passes();
        }
    }
}
