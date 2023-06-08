use crate::mlir::{
    ir::{Type, TypeLike},
    utils::into_raw_array,
    Context,
};
use mlir_lib::*;
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlir::{dialect, ir::ty::IntegerType};
    fn create_ctx() -> Context {
        let ctx = Context::new();
        dialect::DialectHandle::llvm().register_dialect(&ctx);
        ctx.get_or_load_dialect("llvm");
        ctx
    }
    #[test]
    fn opaque_pointer() {
        let ctx = create_ctx();
        assert_eq!(super::opaque_pointer(&ctx), Type::parse(&ctx, "!llvm.ptr").unwrap());
    }
    #[test]
    fn pointer() {
        let ctx = create_ctx();
        let i32 = IntegerType::new(&ctx, 32).into();
        assert_eq!(super::pointer(i32, 0), Type::parse(&ctx, "!llvm.ptr<i32>").unwrap());
    }
    #[test]
    fn pointer_with_address_space() {
        let ctx = create_ctx();
        let i32 = IntegerType::new(&ctx, 32).into();
        assert_eq!(super::pointer(i32, 4), Type::parse(&ctx, "!llvm.ptr<i32, 4>").unwrap());
    }
    #[test]
    fn void() {
        let ctx = create_ctx();
        assert_eq!(super::void(&ctx), Type::parse(&ctx, "!llvm.void").unwrap());
    }
    #[test]
    fn array() {
        let ctx = create_ctx();
        let i32 = IntegerType::new(&ctx, 32).into();
        assert_eq!(super::array(i32, 4), Type::parse(&ctx, "!llvm.array<4 x i32>").unwrap());
    }
    #[test]
    fn function() {
        let ctx = create_ctx();
        let i8 = IntegerType::new(&ctx, 8).into();
        let i32 = IntegerType::new(&ctx, 32).into();
        let i64 = IntegerType::new(&ctx, 64).into();
        assert_eq!(
            super::function(i8, &[i32, i64], false),
            Type::parse(&ctx, "!llvm.func<i8 (i32, i64)>").unwrap()
        );
    }
    #[test]
    fn r#struct() {
        let ctx = create_ctx();
        let i32 = IntegerType::new(&ctx, 32).into();
        let i64 = IntegerType::new(&ctx, 64).into();
        assert_eq!(
            super::r#struct(&ctx, &[i32, i64], false),
            Type::parse(&ctx, "!llvm.struct<(i32, i64)>").unwrap()
        );
    }
    #[test]
    fn packed_struct() {
        let ctx = create_ctx();
        let i32 = IntegerType::new(&ctx, 32).into();
        let i64 = IntegerType::new(&ctx, 64).into();
        assert_eq!(
            super::r#struct(&ctx, &[i32, i64], true),
            Type::parse(&ctx, "!llvm.struct<packed (i32, i64)>").unwrap()
        );
    }
}
