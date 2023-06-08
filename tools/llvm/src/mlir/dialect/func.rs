use crate::mlir::{
    ir::{
        attr::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        op::OperationBuilder,
        typ::FunctionType,
        Attribute, Identifier, Location, Operation, Region, Value,
    },
    Context,
};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlir::{
        ir::{Block, Module, Type},
        test::load_all_dialects,
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
