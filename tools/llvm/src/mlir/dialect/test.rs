#[cfg(test)]
mod dialect {
    use crate::mlir::*;
    #[test]
    fn equal() {
        let y = Context::new();
        assert_eq!(
            DialectHandle::func().load_dialect(&y),
            DialectHandle::func().load_dialect(&y)
        );
    }
    #[test]
    fn not_equal() {
        let y = Context::new();
        assert_ne!(
            DialectHandle::func().load_dialect(&y),
            DialectHandle::llvm().load_dialect(&y)
        );
    }
    #[test]
    fn func() {
        DialectHandle::func();
    }
    #[test]
    fn llvm() {
        DialectHandle::llvm();
    }
    #[test]
    fn namespace() {
        DialectHandle::func().namespace();
    }
    #[test]
    fn insert_dialect() {
        let y = DialectRegistry::new();
        DialectHandle::func().insert_dialect(&y);
    }
    #[test]
    fn load_dialect() {
        let y = Context::new();
        DialectHandle::func().load_dialect(&y);
    }
    #[test]
    fn register_dialect() {
        let y = Context::new();
        DialectHandle::func().register_dialect(&y);
    }
    #[test]
    fn new() {
        DialectRegistry::new();
    }
    #[test]
    fn register_all_dialects() {
        DialectRegistry::new();
    }
    #[test]
    fn register_dialect() {
        let r = DialectRegistry::new();
        DialectHandle::func().insert_dialect(&r);
        let c = Context::new();
        let n = c.registered_dialect_count();
        c.append_dialect_registry(&r);
        assert_eq!(c.registered_dialect_count() - n, 1);
    }
}

#[cfg(test)]
mod arith {
    use crate::mlir::{
        dialect::func,
        ir::{
            attr::{StringAttribute, TypeAttribute},
            typ::FunctionType,
            Attribute, Block, Location, Module, Region, Type,
        },
        test::load_all_dialects,
        Context, *,
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

#[cfg(test)]
mod cf {
    use crate::mlir::{
        dialect::{
            arith::{self, CmpiPredicate},
            func, index,
        },
        ir::{
            attr::{IntegerAttribute, StringAttribute, TypeAttribute},
            r#type::{FunctionType, IntegerType, Type},
            Block, Module, Region,
        },
        test::load_all_dialects,
        Context, *,
    };
    #[test]
    fn compile_assert() {
        let ctx = Context::new();
        load_all_dialects(&ctx);
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        let bool_type: Type = IntegerType::new(&ctx, 1).into();
        module.body().append_operation(func::func(
            &ctx,
            StringAttribute::new(&ctx, "foo"),
            TypeAttribute::new(FunctionType::new(&ctx, &[], &[]).into()),
            {
                let block = Block::new(&[]);
                let operand = block
                    .append_operation(arith::constant(&ctx, IntegerAttribute::new(1, bool_type).into(), loc))
                    .result(0)
                    .unwrap()
                    .into();
                block.append_operation(assert(&ctx, operand, "assert message", loc));
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
    fn compile_br() {
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
                let dest_block = Block::new(&[(index_type, loc)]);
                let operand = block
                    .append_operation(index::constant(&ctx, IntegerAttribute::new(1, index_type), loc))
                    .result(0)
                    .unwrap();
                block.append_operation(br(&dest_block, &[operand.into()], loc));
                dest_block.append_operation(func::r#return(&[], loc));
                let region = Region::new();
                region.append_block(block);
                region.append_block(dest_block);
                region
            },
            &[],
            loc,
        ));
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
    #[test]
    fn compile_cond_br() {
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
                let true_block = Block::new(&[(index_type, loc)]);
                let false_block = Block::new(&[(index_type, loc)]);
                let operand = block
                    .append_operation(index::constant(&ctx, IntegerAttribute::new(1, index_type), loc))
                    .result(0)
                    .unwrap()
                    .into();
                let cond = block
                    .append_operation(index::cmp(&ctx, CmpiPredicate::Eq, operand, operand, loc))
                    .result(0)
                    .unwrap()
                    .into();
                block.append_operation(cond_br(
                    &ctx,
                    cond,
                    &true_block,
                    &false_block,
                    &[operand],
                    &[operand],
                    loc,
                ));
                true_block.append_operation(func::r#return(&[], loc));
                false_block.append_operation(func::r#return(&[], loc));
                let region = Region::new();
                region.append_block(block);
                region.append_block(true_block);
                region.append_block(false_block);
                region
            },
            &[],
            loc,
        ));
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
    #[test]
    fn compile_switch() {
        let ctx = Context::new();
        load_all_dialects(&ctx);
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        let i32_type: Type = IntegerType::new(&ctx, 32).into();
        module.body().append_operation(func::func(
            &ctx,
            StringAttribute::new(&ctx, "foo"),
            TypeAttribute::new(FunctionType::new(&ctx, &[], &[]).into()),
            {
                let block = Block::new(&[]);
                let default_block = Block::new(&[(i32_type, loc)]);
                let first_block = Block::new(&[(i32_type, loc)]);
                let second_block = Block::new(&[(i32_type, loc)]);
                let operand = block
                    .append_operation(arith::constant(&ctx, IntegerAttribute::new(1, i32_type).into(), loc))
                    .result(0)
                    .unwrap()
                    .into();
                block.append_operation(
                    switch(
                        &ctx,
                        &[0, 1],
                        operand,
                        i32_type,
                        (&default_block, &[operand]),
                        &[(&first_block, &[operand]), (&second_block, &[operand])],
                        loc,
                    )
                    .unwrap(),
                );
                default_block.append_operation(func::r#return(&[], loc));
                first_block.append_operation(func::r#return(&[], loc));
                second_block.append_operation(func::r#return(&[], loc));
                let region = Region::new();
                region.append_block(block);
                region.append_block(default_block);
                region.append_block(first_block);
                region.append_block(second_block);
                region
            },
            &[],
            loc,
        ));
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
}

#[cfg(test)]
mod func {
    use crate::mlir::{
        ir::{Block, Module, Type},
        test::load_all_dialects,
        *,
    };
    #[test]
    fn compile_call() {
        let ctx = Context::new();
        load_all_dialects(&ctx);
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        let function = {
            let block = Block::new(&[]);
            block.append_operation(call(&ctx, FlatSymbolRefAttribute::new(&ctx, "foo"), &[], loc));
            block.append_operation(r#return(&[], loc));
            let region = Region::new();
            region.append_block(block);
            func(
                &ctx,
                StringAttribute::new(&ctx, "foo"),
                TypeAttribute::new(FunctionType::new(&ctx, &[], &[]).into()),
                region,
                &[],
                Location::unknown(&ctx),
            )
        };
        module.body().append_operation(function);
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
    #[test]
    fn compile_call_indirect() {
        let ctx = Context::new();
        load_all_dialects(&ctx);
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        let function = {
            let block = Block::new(&[]);
            let function = block.append_operation(constant(
                &ctx,
                FlatSymbolRefAttribute::new(&ctx, "foo"),
                FunctionType::new(&ctx, &[], &[]),
                loc,
            ));
            block.append_operation(call_indirect(function.result(0).unwrap().into(), &[], loc));
            block.append_operation(r#return(&[], loc));
            let region = Region::new();
            region.append_block(block);
            func(
                &ctx,
                StringAttribute::new(&ctx, "foo"),
                TypeAttribute::new(FunctionType::new(&ctx, &[], &[]).into()),
                region,
                &[],
                Location::unknown(&ctx),
            )
        };
        module.body().append_operation(function);
        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
    #[test]
    fn compile_function() {
        let ctx = Context::new();
        load_all_dialects(&ctx);
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        let integer_type = Type::index(&ctx);
        let function = {
            let block = Block::new(&[(integer_type, loc)]);
            block.append_operation(r#return(&[block.argument(0).unwrap().into()], loc));
            let region = Region::new();
            region.append_block(block);
            func(
                &ctx,
                StringAttribute::new(&ctx, "foo"),
                TypeAttribute::new(FunctionType::new(&ctx, &[integer_type], &[integer_type]).into()),
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

#[cfg(test)]
mod index {
    use crate::mlir::{
        dialect::func,
        ir::{
            attr::{StringAttribute, TypeAttribute},
            typ::{FunctionType, IntegerType},
            Block, Location, Module, Region, Type,
        },
        test::load_all_dialects,
        Context, *,
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

#[cfg(test)]
mod llvm {
    use crate::mlir::{dialect, ir::ty::IntegerType, *};
    fn create_ctx() -> Context {
        let ctx = Context::new();
        dialect::DialectHandle::llvm().register_dialect(&ctx);
        ctx.get_or_load_dialect("llvm");
        ctx
    }
    #[test]
    fn opaque_pointer() {
        let ctx = create_ctx();
        assert_eq!(test::opaque_pointer(&ctx), Type::parse(&ctx, "!llvm.ptr").unwrap());
    }
    #[test]
    fn pointer() {
        let ctx = create_ctx();
        let i32 = IntegerType::new(&ctx, 32).into();
        assert_eq!(test::pointer(i32, 0), Type::parse(&ctx, "!llvm.ptr<i32>").unwrap());
    }
    #[test]
    fn pointer_with_address_space() {
        let ctx = create_ctx();
        let i32 = IntegerType::new(&ctx, 32).into();
        assert_eq!(test::pointer(i32, 4), Type::parse(&ctx, "!llvm.ptr<i32, 4>").unwrap());
    }
    #[test]
    fn void() {
        let ctx = create_ctx();
        assert_eq!(test::void(&ctx), Type::parse(&ctx, "!llvm.void").unwrap());
    }
    #[test]
    fn array() {
        let ctx = create_ctx();
        let i32 = IntegerType::new(&ctx, 32).into();
        assert_eq!(test::array(i32, 4), Type::parse(&ctx, "!llvm.array<4 x i32>").unwrap());
    }
    #[test]
    fn function() {
        let ctx = create_ctx();
        let i8 = IntegerType::new(&ctx, 8).into();
        let i32 = IntegerType::new(&ctx, 32).into();
        let i64 = IntegerType::new(&ctx, 64).into();
        assert_eq!(
            test::function(i8, &[i32, i64], false),
            Type::parse(&ctx, "!llvm.func<i8 (i32, i64)>").unwrap()
        );
    }
    #[test]
    fn r#struct() {
        let ctx = create_ctx();
        let i32 = IntegerType::new(&ctx, 32).into();
        let i64 = IntegerType::new(&ctx, 64).into();
        assert_eq!(
            test::r#struct(&ctx, &[i32, i64], false),
            Type::parse(&ctx, "!llvm.struct<(i32, i64)>").unwrap()
        );
    }
    #[test]
    fn packed_struct() {
        let ctx = create_ctx();
        let i32 = IntegerType::new(&ctx, 32).into();
        let i64 = IntegerType::new(&ctx, 64).into();
        assert_eq!(
            test::r#struct(&ctx, &[i32, i64], true),
            Type::parse(&ctx, "!llvm.struct<packed (i32, i64)>").unwrap()
        );
    }
}

#[cfg(test)]
mod memref {
    use crate::mlir::{
        dialect::{func, index},
        ir::{
            attr::{DenseElementsAttribute, StringAttribute, TypeAttribute},
            ty::{FunctionType, IntegerType, RankedTensorType},
            Block, Module, Region, Type,
        },
        test::create_test_ctx,
        *,
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

#[cfg(test)]
mod scf {
    use crate::mlir::{
        dialect::{arith, func},
        ir::{
            attr::{FloatAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
            r#type::{FunctionType, IntegerType, Type},
            Attribute, Block, Module,
        },
        test::load_all_dialects,
        Context, *,
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
