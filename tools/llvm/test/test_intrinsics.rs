use inkwell::context::Context;
use inkwell::intrinsics::Intrinsic;

#[llvm_versions(9.0..=latest)]
#[test]
fn test_get_cos() {
    Intrinsic::find("llvm.cos").unwrap();
}

#[llvm_versions(9.0..=latest)]
#[test]
fn test_get_nonexistent() {
    assert!(Intrinsic::find("nonsense").is_none())
}

#[llvm_versions(9.0..=latest)]
#[test]
fn test_get_decl_cos() {
    let cos = Intrinsic::find("llvm.cos").unwrap();

    assert!(cos.is_overloaded());

    let context = Context::create();
    let module = context.create_module("my_module");

    assert!(cos.get_declaration(&module, &[]).is_none());

    let decl = cos.get_declaration(&module, &[context.f32_type().into()]).unwrap();

    assert_eq!(decl.get_name().to_str().unwrap(), "llvm.cos.f32");
}

#[llvm_versions(9.0..=latest)]
#[test]
fn test_get_decl_va_copy() {
    let va_copy = Intrinsic::find("llvm.va_copy").unwrap();

    assert!(!va_copy.is_overloaded());

    let context = Context::create();
    let module = context.create_module("my_module");

    assert!(va_copy.get_declaration(&module, &[]).is_some());

    let decl = va_copy.get_declaration(&module, &[context.f32_type().into()]).unwrap();

    assert_eq!(decl.get_name().to_str().unwrap(), "llvm.va_copy.f32");
}
