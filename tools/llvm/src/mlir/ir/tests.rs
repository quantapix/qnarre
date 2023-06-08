#[cfg(test)]
mod ir {
    use crate::mlir::ir::*;
    use pretty_assertions::{assert_eq, assert_ne};
    #[test]
    fn new() {
        Identifier::new(&Context::new(), "foo");
    }
    #[test]
    fn context() {
        Identifier::new(&Context::new(), "foo").context();
    }
    #[test]
    fn equal() {
        let context = Context::new();
        assert_eq!(Identifier::new(&context, "foo"), Identifier::new(&context, "foo"));
    }
    #[test]
    fn not_equal() {
        let context = Context::new();
        assert_ne!(Identifier::new(&context, "foo"), Identifier::new(&context, "bar"));
    }
    #[test]
    fn new() {
        Location::new(&Context::new(), "foo", 42, 42);
    }
    #[test]
    fn fused() {
        let context = Context::new();
        Location::fused(
            &context,
            &[
                Location::new(&Context::new(), "foo", 1, 1),
                Location::new(&Context::new(), "foo", 2, 2),
            ],
            Attribute::parse(&context, "42").unwrap(),
        );
    }
    #[test]
    fn name() {
        let context = Context::new();
        Location::name(&context, "foo", Location::unknown(&context));
    }
    #[test]
    fn unknown() {
        Location::unknown(&Context::new());
    }
    #[test]
    fn context() {
        Location::new(&Context::new(), "foo", 42, 42).context();
    }
    #[test]
    fn equal() {
        let context = Context::new();
        assert_eq!(Location::unknown(&context), Location::unknown(&context));
        assert_eq!(
            Location::new(&context, "foo", 42, 42),
            Location::new(&context, "foo", 42, 42),
        );
    }
    #[test]
    fn not_equal() {
        let context = Context::new();
        assert_ne!(Location::new(&context, "foo", 42, 42), Location::unknown(&context));
    }
    #[test]
    fn display() {
        let context = Context::new();
        assert_eq!(Location::unknown(&context).to_string(), "loc(unknown)");
        assert_eq!(Location::new(&context, "foo", 42, 42).to_string(), "loc(\"foo\":42:42)");
    }
    #[test]
    fn new() {
        Module::new(Location::new(&Context::new(), "foo", 42, 42));
    }
    #[test]
    fn context() {
        Module::new(Location::new(&Context::new(), "foo", 42, 42)).context();
    }
    #[test]
    fn parse() {
        assert!(Module::parse(&Context::new(), "module{}").is_some());
    }
    #[test]
    fn parse_none() {
        assert!(Module::parse(&Context::new(), "module{").is_none());
    }
    #[test]
    fn from_operation() {
        let context = Context::new();
        let region = Region::new();
        region.append_block(Block::new(&[]));
        let module = Module::from_operation(
            OperationBuilder::new("builtin.module", Location::unknown(&context))
                .add_regions(vec![region])
                .build(),
        )
        .unwrap();
        assert!(module.as_operation().verify());
        assert_eq!(module.as_operation().to_string(), "module {\n}\n")
    }
    #[test]
    fn from_operation_fail() {
        let context = Context::new();
        assert!(
            Module::from_operation(OperationBuilder::new("func.func", Location::unknown(&context),).build()).is_none()
        );
    }
    #[test]
    fn new() {
        Region::new();
    }
    #[test]
    fn first_block() {
        assert!(Region::new().first_block().is_none());
    }
    #[test]
    fn append_block() {
        let region = Region::new();
        let block = Block::new(&[]);
        region.append_block(block);
        assert!(region.first_block().is_some());
    }
    #[test]
    fn insert_block_after() {
        let region = Region::new();
        let block = region.append_block(Block::new(&[]));
        region.insert_block_after(block, Block::new(&[]));
        assert_eq!(region.first_block(), Some(block));
    }
    #[test]
    fn insert_block_before() {
        let region = Region::new();
        let block = region.append_block(Block::new(&[]));
        let block = region.insert_block_before(block, Block::new(&[]));
        assert_eq!(region.first_block(), Some(block));
    }
    #[test]
    fn equal() {
        let region = Region::new();
        assert_eq!(region, region);
    }
    #[test]
    fn not_equal() {
        assert_ne!(Region::new(), Region::new());
    }
}

#[cfg(test)]
mod attr {
    use crate::mlir::{ir::*, Context, *};
    #[test]
    fn value() {
        let context = Context::new();
        let r#type = Type::index(&context);
        assert_eq!(TypeAttribute::new(r#type).value(), r#type);
    }
    #[test]
    fn new() {
        assert_eq!(FlatSymbolRefAttribute::new(&Context::new(), "foo").value(), "foo");
    }
    #[test]
    fn element() {
        let context = Context::new();
        let attribute = DenseI64ArrayAttribute::new(&context, &[1, 2, 3]);
        assert_eq!(attribute.element(0).unwrap(), 1);
        assert_eq!(attribute.element(1).unwrap(), 2);
        assert_eq!(attribute.element(2).unwrap(), 3);
        assert!(matches!(attribute.element(3), Err(Error::PositionOutOfBounds { .. })));
    }
    #[test]
    fn len() {
        let context = Context::new();
        let attribute = DenseI64ArrayAttribute::new(&context, &[1, 2, 3]);
        assert_eq!(attribute.len(), 3);
    }
    #[test]
    fn element() {
        let context = Context::new();
        let attribute = DenseI32ArrayAttribute::new(&context, &[1, 2, 3]);
        assert_eq!(attribute.element(0).unwrap(), 1);
        assert_eq!(attribute.element(1).unwrap(), 2);
        assert_eq!(attribute.element(2).unwrap(), 3);
        assert!(matches!(attribute.element(3), Err(Error::PositionOutOfBounds { .. })));
    }
    #[test]
    fn len() {
        let context = Context::new();
        let attribute = DenseI32ArrayAttribute::new(&context, &[1, 2, 3]);
        assert_eq!(attribute.len(), 3);
    }
    #[test]
    fn i32_element() {
        let context = Context::new();
        let integer_type = IntegerType::new(&context, 32).into();
        let attribute = DenseElementsAttribute::new(
            MemRefType::new(integer_type, &[3], None, None).into(),
            &[IntegerAttribute::new(42, integer_type).into()],
        )
        .unwrap();
        assert_eq!(attribute.i32_element(0), Ok(42));
        assert_eq!(attribute.i32_element(1), Ok(42));
        assert_eq!(attribute.i32_element(2), Ok(42));
        assert_eq!(
            attribute.i32_element(3),
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: attribute.to_string(),
                index: 3,
            })
        );
    }
    #[test]
    fn i64_element() {
        let context = Context::new();
        let integer_type = IntegerType::new(&context, 64).into();
        let attribute = DenseElementsAttribute::new(
            MemRefType::new(integer_type, &[3], None, None).into(),
            &[IntegerAttribute::new(42, integer_type).into()],
        )
        .unwrap();
        assert_eq!(attribute.i64_element(0), Ok(42));
        assert_eq!(attribute.i64_element(1), Ok(42));
        assert_eq!(attribute.i64_element(2), Ok(42));
        assert_eq!(
            attribute.i64_element(3),
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: attribute.to_string(),
                index: 3,
            })
        );
    }
    #[test]
    fn len() {
        let context = Context::new();
        let integer_type = IntegerType::new(&context, 64).into();
        let attribute = DenseElementsAttribute::new(
            MemRefType::new(integer_type, &[3], None, None).into(),
            &[IntegerAttribute::new(0, integer_type).into()],
        )
        .unwrap();
        assert_eq!(attribute.len(), 3);
    }
    #[test]
    fn parse() {
        for attribute in ["unit", "i32", r#""foo""#] {
            assert!(Attribute::parse(&Context::new(), attribute).is_some());
        }
    }
    #[test]
    fn parse_none() {
        assert!(Attribute::parse(&Context::new(), "z").is_none());
    }
    #[test]
    fn context() {
        Attribute::parse(&Context::new(), "unit").unwrap().context();
    }
    #[test]
    fn r#type() {
        let context = Context::new();
        assert_eq!(
            Attribute::parse(&context, "unit").unwrap().r#type(),
            Type::none(&context)
        );
    }
    #[ignore]
    #[test]
    fn type_id() {
        let context = Context::new();
        assert_eq!(
            Attribute::parse(&context, "42 : index").unwrap().type_id(),
            Type::index(&context).id()
        );
    }
    #[test]
    fn is_array() {
        assert!(Attribute::parse(&Context::new(), "[]").unwrap().is_array());
    }
    #[test]
    fn is_bool() {
        assert!(Attribute::parse(&Context::new(), "false").unwrap().is_bool());
    }
    #[test]
    fn is_dense_elements() {
        assert!(Attribute::parse(&Context::new(), "dense<10> : tensor<2xi8>")
            .unwrap()
            .is_dense_elements());
    }
    #[test]
    fn is_dense_int_elements() {
        assert!(Attribute::parse(&Context::new(), "dense<42> : tensor<42xi8>")
            .unwrap()
            .is_dense_int_elements());
    }
    #[test]
    fn is_dense_fp_elements() {
        assert!(Attribute::parse(&Context::new(), "dense<42.0> : tensor<42xf32>")
            .unwrap()
            .is_dense_fp_elements());
    }
    #[test]
    fn is_elements() {
        assert!(
            Attribute::parse(&Context::new(), "sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>")
                .unwrap()
                .is_elements()
        );
    }
    #[test]
    fn is_integer() {
        assert!(Attribute::parse(&Context::new(), "42").unwrap().is_integer());
    }
    #[test]
    fn is_integer_set() {
        assert!(Attribute::parse(&Context::new(), "affine_set<(d0) : (d0 - 2 >= 0)>")
            .unwrap()
            .is_integer_set());
    }
    #[ignore]
    #[test]
    fn is_opaque() {
        assert!(Attribute::parse(&Context::new(), "#foo<\"bar\">").unwrap().is_opaque());
    }
    #[test]
    fn is_sparse_elements() {
        assert!(
            Attribute::parse(&Context::new(), "sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>")
                .unwrap()
                .is_sparse_elements()
        );
    }
    #[test]
    fn is_string() {
        assert!(Attribute::parse(&Context::new(), "\"foo\"").unwrap().is_string());
    }
    #[test]
    fn is_type() {
        assert!(Attribute::parse(&Context::new(), "index").unwrap().is_type());
    }
    #[test]
    fn is_unit() {
        assert!(Attribute::parse(&Context::new(), "unit").unwrap().is_unit());
    }
    #[test]
    fn is_symbol() {
        assert!(Attribute::parse(&Context::new(), "@foo").unwrap().is_symbol_ref());
    }
    #[test]
    fn equal() {
        let context = Context::new();
        let attribute = Attribute::parse(&context, "unit").unwrap();
        assert_eq!(attribute, attribute);
    }
    #[test]
    fn not_equal() {
        let context = Context::new();
        assert_ne!(
            Attribute::parse(&context, "unit").unwrap(),
            Attribute::parse(&context, "42").unwrap()
        );
    }
    #[test]
    fn display() {
        assert_eq!(Attribute::parse(&Context::new(), "unit").unwrap().to_string(), "unit");
    }
}

#[cfg(test)]
mod block {
    use crate::mlir::{ir::*, test::load_all_dialects, Context};
    use pretty_assertions::assert_eq;
    #[test]
    fn argument_number() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index").unwrap();
        let block = Block::new(&[(r#type, Location::unknown(&context))]);
        assert_eq!(block.argument(0).unwrap().argument_number(), 0);
    }
    #[test]
    fn owner() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index").unwrap();
        let block = Block::new(&[(r#type, Location::unknown(&context))]);
        assert_eq!(&*block.argument(0).unwrap().owner(), &block);
    }
    #[test]
    fn set_type() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index").unwrap();
        let other_type = Type::parse(&context, "f64").unwrap();
        let block = Block::new(&[(r#type, Location::unknown(&context))]);
        let argument = block.argument(0).unwrap();
        argument.set_type(other_type);
        assert_eq!(argument.r#type(), other_type);
    }
    #[test]
    fn new() {
        Block::new(&[]);
    }
    #[test]
    fn argument() {
        let context = Context::new();
        let r#type = IntegerType::new(&context, 64).into();
        assert_eq!(
            Block::new(&[(r#type, Location::unknown(&context))])
                .argument(0)
                .unwrap()
                .r#type(),
            r#type
        );
    }
    #[test]
    fn argument_error() {
        assert_eq!(
            Block::new(&[]).argument(0).unwrap_err(),
            Error::PositionOutOfBounds {
                name: "block argument",
                value: "<<UNLINKED BLOCK>>\n".into(),
                index: 0,
            }
        );
    }
    #[test]
    fn argument_count() {
        assert_eq!(Block::new(&[]).argument_count(), 0);
    }
    #[test]
    fn parent_region() {
        let region = Region::new();
        let block = region.append_block(Block::new(&[]));
        assert_eq!(block.parent_region().as_deref(), Some(&region));
    }
    #[test]
    fn parent_region_none() {
        let block = Block::new(&[]);
        assert_eq!(block.parent_region(), None);
    }
    #[test]
    fn parent_operation() {
        let context = Context::new();
        let module = Module::new(Location::unknown(&context));
        assert_eq!(module.body().parent_operation(), Some(module.as_operation()));
    }
    #[test]
    fn parent_operation_none() {
        let block = Block::new(&[]);
        assert_eq!(block.parent_operation(), None);
    }
    #[test]
    fn terminator() {
        let context = Context::new();
        load_all_dialects(&context);
        let block = Block::new(&[]);
        let operation =
            block.append_operation(OperationBuilder::new("func.return", Location::unknown(&context)).build());
        assert_eq!(block.terminator(), Some(operation));
    }
    #[test]
    fn terminator_none() {
        assert_eq!(Block::new(&[]).terminator(), None);
    }
    #[test]
    fn first_operation() {
        let context = Context::new();
        let block = Block::new(&[]);
        let operation = block.append_operation(OperationBuilder::new("foo", Location::unknown(&context)).build());
        assert_eq!(block.first_operation(), Some(operation));
    }
    #[test]
    fn first_operation_none() {
        let block = Block::new(&[]);
        assert_eq!(block.first_operation(), None);
    }
    #[test]
    fn append_operation() {
        let context = Context::new();
        let block = Block::new(&[]);
        block.append_operation(OperationBuilder::new("foo", Location::unknown(&context)).build());
    }
    #[test]
    fn insert_operation() {
        let context = Context::new();
        let block = Block::new(&[]);
        block.insert_operation(0, OperationBuilder::new("foo", Location::unknown(&context)).build());
    }
    #[test]
    fn insert_operation_after() {
        let context = Context::new();
        let block = Block::new(&[]);
        let first_operation = block.append_operation(OperationBuilder::new("foo", Location::unknown(&context)).build());
        let second_operation = block.insert_operation_after(
            first_operation,
            OperationBuilder::new("foo", Location::unknown(&context)).build(),
        );
        assert_eq!(block.first_operation(), Some(first_operation));
        assert_eq!(block.first_operation().unwrap().next_in_block(), Some(second_operation));
    }
    #[test]
    fn insert_operation_before() {
        let context = Context::new();
        let block = Block::new(&[]);
        let second_operation =
            block.append_operation(OperationBuilder::new("foo", Location::unknown(&context)).build());
        let first_operation = block.insert_operation_before(
            second_operation,
            OperationBuilder::new("foo", Location::unknown(&context)).build(),
        );
        assert_eq!(block.first_operation(), Some(first_operation));
        assert_eq!(block.first_operation().unwrap().next_in_block(), Some(second_operation));
    }
    #[test]
    fn next_in_region() {
        let region = Region::new();
        let first_block = region.append_block(Block::new(&[]));
        let second_block = region.append_block(Block::new(&[]));
        assert_eq!(first_block.next_in_region(), Some(second_block));
    }
    #[test]
    fn detach() {
        let region = Region::new();
        let block = region.append_block(Block::new(&[]));
        assert_eq!(unsafe { block.detach() }.unwrap().to_string(), "<<UNLINKED BLOCK>>\n");
    }
    #[test]
    fn detach_detached() {
        let block = Block::new(&[]);
        assert!(unsafe { block.detach() }.is_none());
    }
    #[test]
    fn display() {
        assert_eq!(Block::new(&[]).to_string(), "<<UNLINKED BLOCK>>\n");
    }
    #[test]
    fn debug() {
        assert_eq!(format!("{:?}", &Block::new(&[])), "Block(\n<<UNLINKED BLOCK>>\n)");
    }
}

#[cfg(test)]
mod op {
    use crate::mlir::{ir::*, test::load_all_dialects, Context};
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

#[cfg(test)]
mod typ {
    use crate::mlir::{ir::*, Context};
    #[test]
    fn new() {
        Type::parse(&Context::new(), "f32");
    }
    #[test]
    fn integer() {
        let context = Context::new();
        assert_eq!(
            Type::from(IntegerType::new(&context, 42)),
            Type::parse(&context, "i42").unwrap()
        );
    }
    #[test]
    fn index() {
        let context = Context::new();
        assert_eq!(Type::index(&context), Type::parse(&context, "index").unwrap());
    }
    #[test]
    fn vector() {
        let context = Context::new();
        assert_eq!(
            Type::vector(&[42], Type::float64(&context)),
            Type::parse(&context, "vector<42xf64>").unwrap()
        );
    }
    #[test]
    fn vector_with_invalid_dimension() {
        let context = Context::new();
        assert_eq!(
            Type::vector(&[0], IntegerType::new(&context, 32).into()).to_string(),
            "vector<0xi32>"
        );
    }
    #[test]
    fn vector_checked() {
        let context = Context::new();
        assert_eq!(
            Type::vector_checked(
                Location::unknown(&context),
                &[42],
                IntegerType::new(&context, 32).into()
            ),
            Type::parse(&context, "vector<42xi32>")
        );
    }
    #[test]
    fn vector_checked_fail() {
        let context = Context::new();
        assert_eq!(
            Type::vector_checked(Location::unknown(&context), &[0], Type::index(&context)),
            None
        );
    }
    #[test]
    fn equal() {
        let context = Context::new();
        assert_eq!(Type::index(&context), Type::index(&context));
    }
    #[test]
    fn not_equal() {
        let context = Context::new();
        assert_ne!(Type::index(&context), Type::float64(&context));
    }
    #[test]
    fn display() {
        let context = Context::new();
        assert_eq!(Type::index(&context).to_string(), "index");
    }
    #[test]
    fn debug() {
        let context = Context::new();
        assert_eq!(format!("{:?}", Type::index(&context)), "Type(index)");
    }
    #[test]
    fn new() {
        let context = Context::new();
        let integer = Type::index(&context);
        assert_eq!(
            Type::from(FunctionType::new(&context, &[integer, integer], &[integer])),
            Type::parse(&context, "(index, index) -> index").unwrap()
        );
    }
    #[test]
    fn multiple_results() {
        let context = Context::new();
        let integer = Type::index(&context);
        assert_eq!(
            Type::from(FunctionType::new(&context, &[], &[integer, integer])),
            Type::parse(&context, "() -> (index, index)").unwrap()
        );
    }
    #[test]
    fn input() {
        let context = Context::new();
        let integer = Type::index(&context);
        assert_eq!(FunctionType::new(&context, &[integer], &[]).input(0), Ok(integer));
    }
    #[test]
    fn input_error() {
        let context = Context::new();
        let integer = Type::index(&context);
        let function = FunctionType::new(&context, &[integer], &[]);
        assert_eq!(
            function.input(42),
            Err(Error::PositionOutOfBounds {
                name: "function input",
                value: function.to_string(),
                index: 42
            })
        );
    }
    #[test]
    fn result() {
        let context = Context::new();
        let integer = Type::index(&context);
        assert_eq!(FunctionType::new(&context, &[], &[integer]).result(0), Ok(integer));
    }
    #[test]
    fn result_error() {
        let context = Context::new();
        let integer = Type::index(&context);
        let function = FunctionType::new(&context, &[], &[integer]);
        assert_eq!(
            function.result(42),
            Err(Error::PositionOutOfBounds {
                name: "function result",
                value: function.to_string(),
                index: 42
            })
        );
    }
    #[test]
    fn input_count() {
        let context = Context::new();
        let integer = Type::index(&context);
        assert_eq!(FunctionType::new(&context, &[integer], &[]).input_count(), 1);
    }
    #[test]
    fn result_count() {
        let context = Context::new();
        let integer = Type::index(&context);
        assert_eq!(FunctionType::new(&context, &[], &[integer]).result_count(), 1);
    }
    #[test]
    fn new() {
        Allocator::new();
    }
    #[test]
    fn allocate_type_id() {
        Allocator::new().allocate_type_id();
    }
    #[test]
    fn new() {
        assert!(IntegerType::new(&Context::new(), 64).is_integer());
    }
    #[test]
    fn signed() {
        assert!(IntegerType::signed(&Context::new(), 64).is_integer());
    }
    #[test]
    fn unsigned() {
        assert!(IntegerType::unsigned(&Context::new(), 64).is_integer());
    }
    #[test]
    fn signed_integer() {
        let context = Context::new();
        assert_eq!(
            Type::from(IntegerType::signed(&context, 42)),
            Type::parse(&context, "si42").unwrap()
        );
    }
    #[test]
    fn unsigned_integer() {
        let context = Context::new();
        assert_eq!(
            Type::from(IntegerType::unsigned(&context, 42)),
            Type::parse(&context, "ui42").unwrap()
        );
    }
    #[test]
    fn get_width() {
        let context = Context::new();
        assert_eq!(IntegerType::new(&context, 64).width(), 64);
    }
    #[test]
    fn check_sign() {
        let context = Context::new();
        let signless = IntegerType::new(&context, 42);
        let signed = IntegerType::signed(&context, 42);
        let unsigned = IntegerType::unsigned(&context, 42);
        assert!(signless.is_signless());
        assert!(!signed.is_signless());
        assert!(!unsigned.is_signless());
        assert!(!signless.is_signed());
        assert!(signed.is_signed());
        assert!(!unsigned.is_signed());
        assert!(!signless.is_unsigned());
        assert!(!signed.is_unsigned());
        assert!(unsigned.is_unsigned());
    }
    #[test]
    fn new() {
        let context = Context::new();
        assert_eq!(
            Type::from(MemRefType::new(Type::float64(&context), &[42], None, None,)),
            Type::parse(&context, "memref<42xf64>").unwrap()
        );
    }
    #[test]
    fn layout() {
        let context = Context::new();
        assert_eq!(
            MemRefType::new(Type::index(&context), &[42, 42], None, None,).layout(),
            Attribute::parse(&context, "affine_map<(d0, d1) -> (d0, d1)>").unwrap(),
        );
    }
    #[test]
    fn affine_map() {
        let context = Context::new();
        assert_eq!(
            MemRefType::new(Type::index(&context), &[42, 42], None, None,)
                .affine_map()
                .to_string(),
            "(d0, d1) -> (d0, d1)"
        );
    }
    #[test]
    fn memory_space() {
        let context = Context::new();
        assert_eq!(
            MemRefType::new(Type::index(&context), &[42, 42], None, None).memory_space(),
            None,
        );
    }
    #[test]
    fn new() {
        let context = Context::new();
        assert_eq!(
            Type::from(RankedTensorType::new(&[42], Type::float64(&context), None)),
            Type::parse(&context, "tensor<42xf64>").unwrap()
        );
    }
    #[test]
    fn encoding() {
        let context = Context::new();
        assert_eq!(
            RankedTensorType::new(&[42, 42], Type::index(&context), None).encoding(),
            None,
        );
    }
    #[test]
    fn new() {
        let context = Context::new();
        assert_eq!(
            Type::from(TupleType::new(&context, &[])),
            Type::parse(&context, "tuple<>").unwrap()
        );
    }
    #[test]
    fn new_with_field() {
        let context = Context::new();
        assert_eq!(
            Type::from(TupleType::new(&context, &[Type::index(&context)])),
            Type::parse(&context, "tuple<index>").unwrap()
        );
    }
    #[test]
    fn new_with_two_fields() {
        let context = Context::new();
        let r#type = Type::index(&context);
        assert_eq!(
            Type::from(TupleType::new(&context, &[r#type, r#type])),
            Type::parse(&context, "tuple<index,index>").unwrap()
        );
    }
    #[test]
    fn r#type() {
        let context = Context::new();
        let r#type = Type::index(&context);
        assert_eq!(TupleType::new(&context, &[r#type]).r#type(0), Ok(r#type));
    }
    #[test]
    fn type_error() {
        let context = Context::new();
        let tuple = TupleType::new(&context, &[]);
        assert_eq!(
            tuple.r#type(42),
            Err(Error::PositionOutOfBounds {
                name: "tuple field",
                value: tuple.to_string(),
                index: 42
            })
        );
    }
    #[test]
    fn type_count() {
        assert_eq!(TupleType::new(&Context::new(), &[]).type_count(), 0);
    }
    #[test]
    fn context() {
        Type::parse(&Context::new(), "i8").unwrap().context();
    }
    #[test]
    fn id() {
        let context = Context::new();
        assert_eq!(Type::index(&context).id(), Type::index(&context).id());
    }
    #[test]
    fn is_integer() {
        let context = Context::new();
        assert!(IntegerType::new(&context, 64).is_integer());
    }
    #[test]
    fn is_index() {
        let context = Context::new();
        assert!(Type::index(&context).is_index());
    }
    #[test]
    fn is_bfloat16() {
        let context = Context::new();
        assert!(FunctionType::new(&context, &[], &[]).is_function());
    }
    #[test]
    fn is_function() {
        let context = Context::new();
        assert!(FunctionType::new(&context, &[], &[]).is_function());
    }
    #[test]
    fn is_vector() {
        let context = Context::new();
        assert!(Type::vector(&[42], Type::index(&context)).is_vector());
    }
    #[test]
    fn dump() {
        Type::index(&Context::new()).dump();
    }
}

#[cfg(test)]
mod val {
    use crate::mlir::{ir::*, test::load_all_dialects, Context, *};
    #[test]
    fn r#type() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);
        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build();
        assert_eq!(operation.result(0).unwrap().r#type(), index_type);
    }
    #[test]
    fn is_operation_result() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let r#type = Type::index(&context);
        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[r#type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build();
        assert!(operation.result(0).unwrap().is_operation_result());
    }
    #[test]
    fn is_block_argument() {
        let context = Context::new();
        let r#type = Type::index(&context);
        let block = Block::new(&[(r#type, Location::unknown(&context))]);
        assert!(block.argument(0).unwrap().is_block_argument());
    }
    #[test]
    fn dump() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);
        let value = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build();
        value.result(0).unwrap().dump();
    }
    #[test]
    fn equal() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);
        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build();
        let result = Value::from(operation.result(0).unwrap());
        assert_eq!(result, result);
    }
    #[test]
    fn not_equal() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);
        let operation = || {
            OperationBuilder::new("arith.constant", location)
                .add_results(&[index_type])
                .add_attributes(&[(
                    Identifier::new(&context, "value"),
                    Attribute::parse(&context, "0 : index").unwrap(),
                )])
                .build()
        };
        assert_ne!(
            Value::from(operation().result(0).unwrap()),
            operation().result(0).unwrap().into()
        );
    }
    #[test]
    fn display() {
        let context = Context::new();
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);
        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build();
        assert_eq!(
            operation.result(0).unwrap().to_string(),
            "%0 = \"arith.constant\"() {value = 0 : index} : () -> index\n"
        );
    }
    #[test]
    fn display_with_dialect_loaded() {
        let context = Context::new();
        load_all_dialects(&context);
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);
        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build();
        assert_eq!(
            operation.result(0).unwrap().to_string(),
            "%c0 = arith.constant 0 : index\n"
        );
    }
    #[test]
    fn debug() {
        let context = Context::new();
        load_all_dialects(&context);
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);
        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build();
        assert_eq!(
            format!("{:?}", Value::from(operation.result(0).unwrap())),
            "Value(\n%c0 = arith.constant 0 : index\n)"
        );
    }
}
