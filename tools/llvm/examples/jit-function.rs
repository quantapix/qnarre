use llvm_lib::core::*;
use llvm_lib::execution_engine::*;
use llvm_lib::target::*;
use std::ffi::CStr;
use std::mem;

fn main() {
    unsafe {
        let context = LLVMContextCreate();
        let module = LLVMModuleCreateWithNameInContext(b"sum\0".as_ptr() as *const _, context);
        let builder = LLVMCreateBuilderInContext(context);

        let i64t = LLVMInt64TypeInContext(context);
        let mut argts = [i64t, i64t, i64t];
        let function_type = LLVMFunctionType(i64t, argts.as_mut_ptr(), argts.len() as u32, 0);

        let function = LLVMAddFunction(module, b"sum\0".as_ptr() as *const _, function_type);

        let bb = LLVMAppendBasicBlockInContext(context, function, b"entry\0".as_ptr() as *const _);

        LLVMPositionBuilderAtEnd(builder, bb);

        let x = LLVMGetParam(function, 0);
        let y = LLVMGetParam(function, 1);
        let z = LLVMGetParam(function, 2);

        let sum = LLVMBuildAdd(builder, x, y, b"sum.1\0".as_ptr() as *const _);
        let sum = LLVMBuildAdd(builder, sum, z, b"sum.2\0".as_ptr() as *const _);

        LLVMBuildRet(builder, sum);

        LLVMDisposeBuilder(builder);

        LLVMDumpModule(module);

        LLVMLinkInMCJIT();
        LLVM_InitializeNativeTarget();
        LLVM_InitializeNativeAsmPrinter();

        let ee = {
            let mut ee = mem::MaybeUninit::uninit();
            let mut err = mem::zeroed();

            if LLVMCreateExecutionEngineForModule(ee.as_mut_ptr(), module, &mut err) != 0 {
                assert!(!err.is_null());
                panic!("Failed to create execution engine: {:?}", CStr::from_ptr(err));
            }

            ee.assume_init()
        };

        let addr = LLVMGetFunctionAddress(ee, b"sum\0".as_ptr() as *const _);

        let f: extern "C" fn(u64, u64, u64) -> u64 = mem::transmute(addr);

        let x: u64 = 1;
        let y: u64 = 1;
        let z: u64 = 1;
        let res = f(x, y, z);

        println!("{} + {} + {} = {}", x, y, z, res);

        LLVMDisposeExecutionEngine(ee);
        LLVMContextDispose(context);
    }
}
