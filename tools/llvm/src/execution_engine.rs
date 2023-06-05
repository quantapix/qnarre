use libc::c_int;
use llvm_lib::execution_engine::{
    LLVMAddGlobalMapping, LLVMAddModule, LLVMDisposeExecutionEngine, LLVMExecutionEngineRef, LLVMFindFunction,
    LLVMFreeMachineCodeForFunction, LLVMGenericValueRef, LLVMGetExecutionEngineTargetData, LLVMGetFunctionAddress,
    LLVMLinkInInterpreter, LLVMLinkInMCJIT, LLVMRemoveModule, LLVMRunFunction, LLVMRunFunctionAsMain,
    LLVMRunStaticConstructors, LLVMRunStaticDestructors,
};

use crate::ctx::Context;
use crate::module::Module;
use crate::support::{to_c_str, LLVMString};
use crate::targets::TargetData;
use crate::values::{AnyValue, AsValueRef, FunctionValue, GenericValue};

use std::error::Error;
use std::fmt::{self, Debug, Display, Formatter};
use std::marker::PhantomData;
use std::mem::{forget, size_of, transmute_copy, MaybeUninit};
use std::ops::Deref;
use std::rc::Rc;

static EE_INNER_PANIC: &str = "ExecutionEngineInner should exist until Drop";

#[derive(Debug, PartialEq, Eq)]
pub enum FunctionLookupError {
    JITNotEnabled,
    FunctionNotFound, // 404!
}

impl Error for FunctionLookupError {}

impl FunctionLookupError {
    fn as_str(&self) -> &str {
        match self {
            FunctionLookupError::JITNotEnabled => "ExecutionEngine does not have JIT functionality enabled",
            FunctionLookupError::FunctionNotFound => "Function not found in ExecutionEngine",
        }
    }
}

impl Display for FunctionLookupError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "FunctionLookupError({})", self.as_str())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum RemoveModuleError {
    ModuleNotOwned,
    IncorrectModuleOwner,
    LLVMError(LLVMString),
}

impl Error for RemoveModuleError {
    fn description(&self) -> &str {
        self.as_str()
    }

    fn cause(&self) -> Option<&dyn Error> {
        None
    }
}

impl RemoveModuleError {
    fn as_str(&self) -> &str {
        match self {
            RemoveModuleError::ModuleNotOwned => "Module is not owned by an Execution Engine",
            RemoveModuleError::IncorrectModuleOwner => "Module is not owned by this Execution Engine",
            RemoveModuleError::LLVMError(string) => string.to_str().unwrap_or("LLVMError with invalid unicode"),
        }
    }
}

impl Display for RemoveModuleError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "RemoveModuleError({})", self.as_str())
    }
}

///
///
#[derive(PartialEq, Eq, Debug)]
pub struct ExecutionEngine<'ctx> {
    execution_engine: Option<ExecEngineInner<'ctx>>,
    target_data: Option<TargetData>,
    jit_mode: bool,
}

impl<'ctx> ExecutionEngine<'ctx> {
    pub unsafe fn new(execution_engine: Rc<LLVMExecutionEngineRef>, jit_mode: bool) -> Self {
        assert!(!execution_engine.is_null());

        let target_data = LLVMGetExecutionEngineTargetData(*execution_engine);

        ExecutionEngine {
            execution_engine: Some(ExecEngineInner(execution_engine, PhantomData)),
            target_data: Some(TargetData::new(target_data)),
            jit_mode,
        }
    }

    pub fn as_mut_ptr(&self) -> LLVMExecutionEngineRef {
        self.execution_engine_inner()
    }

    pub(crate) fn execution_engine_rc(&self) -> &Rc<LLVMExecutionEngineRef> {
        &self.execution_engine.as_ref().expect(EE_INNER_PANIC).0
    }

    #[inline]
    pub(crate) fn execution_engine_inner(&self) -> LLVMExecutionEngineRef {
        **self.execution_engine_rc()
    }

    pub fn link_in_mc_jit() {
        unsafe { LLVMLinkInMCJIT() }
    }

    pub fn link_in_interpreter() {
        unsafe {
            LLVMLinkInInterpreter();
        }
    }

    pub fn add_global_mapping(&self, value: &dyn AnyValue<'ctx>, addr: usize) {
        unsafe { LLVMAddGlobalMapping(self.execution_engine_inner(), value.as_value_ref(), addr as *mut _) }
    }

    pub fn add_module(&self, module: &Module<'ctx>) -> Result<(), ()> {
        unsafe { LLVMAddModule(self.execution_engine_inner(), module.module.get()) }

        if module.owned_by_ee.borrow().is_some() {
            return Err(());
        }

        *module.owned_by_ee.borrow_mut() = Some(self.clone());

        Ok(())
    }

    pub fn remove_module(&self, module: &Module<'ctx>) -> Result<(), RemoveModuleError> {
        match *module.owned_by_ee.borrow() {
            Some(ref ee) if ee.execution_engine_inner() != self.execution_engine_inner() => {
                return Err(RemoveModuleError::IncorrectModuleOwner)
            },
            None => return Err(RemoveModuleError::ModuleNotOwned),
            _ => (),
        }

        let mut new_module = MaybeUninit::uninit();
        let mut err_string = MaybeUninit::uninit();

        let code = unsafe {
            LLVMRemoveModule(
                self.execution_engine_inner(),
                module.module.get(),
                new_module.as_mut_ptr(),
                err_string.as_mut_ptr(),
            )
        };

        if code == 1 {
            unsafe {
                return Err(RemoveModuleError::LLVMError(LLVMString::new(err_string.assume_init())));
            }
        }

        let new_module = unsafe { new_module.assume_init() };

        module.module.set(new_module);
        *module.owned_by_ee.borrow_mut() = None;

        Ok(())
    }

    pub unsafe fn get_function<F>(&self, fn_name: &str) -> Result<JitFunction<'ctx, F>, FunctionLookupError>
    where
        F: UnsafeFunctionPointer,
    {
        if !self.jit_mode {
            return Err(FunctionLookupError::JITNotEnabled);
        }

        let address = self.get_function_address(fn_name)?;

        assert_eq!(
            size_of::<F>(),
            size_of::<usize>(),
            "The type `F` must have the same size as a function pointer"
        );

        let execution_engine = self.execution_engine.as_ref().expect(EE_INNER_PANIC);

        Ok(JitFunction {
            _execution_engine: execution_engine.clone(),
            inner: transmute_copy(&address),
        })
    }

    pub fn get_function_address(&self, fn_name: &str) -> Result<usize, FunctionLookupError> {
        #[cfg(any(feature = "llvm5-0", feature = "llvm6-0", feature = "llvm7-0", feature = "llvm8-0"))]
        self.get_function_value(fn_name)?;

        let c_string = to_c_str(fn_name);
        let address = unsafe { LLVMGetFunctionAddress(self.execution_engine_inner(), c_string.as_ptr()) };

        if address == 0 {
            return Err(FunctionLookupError::FunctionNotFound);
        }

        Ok(address as usize)
    }

    pub fn get_target_data(&self) -> &TargetData {
        self.target_data
            .as_ref()
            .expect("TargetData should always exist until Drop")
    }

    pub fn get_function_value(&self, fn_name: &str) -> Result<FunctionValue<'ctx>, FunctionLookupError> {
        if !self.jit_mode {
            return Err(FunctionLookupError::JITNotEnabled);
        }

        let c_string = to_c_str(fn_name);
        let mut function = MaybeUninit::uninit();

        let code = unsafe { LLVMFindFunction(self.execution_engine_inner(), c_string.as_ptr(), function.as_mut_ptr()) };

        if code == 0 {
            return unsafe { FunctionValue::new(function.assume_init()).ok_or(FunctionLookupError::FunctionNotFound) };
        };

        Err(FunctionLookupError::FunctionNotFound)
    }

    pub unsafe fn run_function(
        &self,
        function: FunctionValue<'ctx>,
        args: &[&GenericValue<'ctx>],
    ) -> GenericValue<'ctx> {
        let mut args: Vec<LLVMGenericValueRef> = args.iter().map(|val| val.generic_value).collect();

        let value = LLVMRunFunction(
            self.execution_engine_inner(),
            function.as_value_ref(),
            args.len() as u32,
            args.as_mut_ptr(),
        ); // REVIEW: usize to u32 ok??

        GenericValue::new(value)
    }

    pub unsafe fn run_function_as_main(&self, function: FunctionValue<'ctx>, args: &[&str]) -> c_int {
        let cstring_args: Vec<_> = args.iter().map(|&arg| to_c_str(arg)).collect();
        let raw_args: Vec<*const _> = cstring_args.iter().map(|arg| arg.as_ptr()).collect();

        let environment_variables = vec![]; // TODO: Support envp. Likely needs to be null terminated

        LLVMRunFunctionAsMain(
            self.execution_engine_inner(),
            function.as_value_ref(),
            raw_args.len() as u32,
            raw_args.as_ptr(),
            environment_variables.as_ptr(),
        ) // REVIEW: usize to u32 cast ok??
    }

    pub fn free_fn_machine_code(&self, function: FunctionValue<'ctx>) {
        unsafe { LLVMFreeMachineCodeForFunction(self.execution_engine_inner(), function.as_value_ref()) }
    }

    pub fn run_static_constructors(&self) {
        unsafe { LLVMRunStaticConstructors(self.execution_engine_inner()) }
    }

    pub fn run_static_destructors(&self) {
        unsafe { LLVMRunStaticDestructors(self.execution_engine_inner()) }
    }
}

impl Drop for ExecutionEngine<'_> {
    fn drop(&mut self) {
        forget(
            self.target_data
                .take()
                .expect("TargetData should always exist until Drop"),
        );

        drop(self.execution_engine.take().expect(EE_INNER_PANIC));
    }
}

impl Clone for ExecutionEngine<'_> {
    fn clone(&self) -> Self {
        let execution_engine_rc = self.execution_engine_rc().clone();

        unsafe { ExecutionEngine::new(execution_engine_rc, self.jit_mode) }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExecEngineInner<'ctx>(Rc<LLVMExecutionEngineRef>, PhantomData<&'ctx Context>);

impl Drop for ExecEngineInner<'_> {
    fn drop(&mut self) {
        if Rc::strong_count(&self.0) == 1 {
            unsafe {
                LLVMDisposeExecutionEngine(*self.0);
            }
        }
    }
}

impl Deref for ExecEngineInner<'_> {
    type Target = LLVMExecutionEngineRef;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

#[derive(Clone)]
pub struct JitFunction<'ctx, F> {
    _execution_engine: ExecEngineInner<'ctx>,
    inner: F,
}

impl<'ctx, F: Copy> JitFunction<'ctx, F> {
    pub unsafe fn into_raw(self) -> F {
        self.inner
    }

    pub unsafe fn as_raw(&self) -> F {
        self.inner
    }
}

impl<F> Debug for JitFunction<'_, F> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_tuple("JitFunction").field(&"<unnamed>").finish()
    }
}

pub trait UnsafeFunctionPointer: private::SealedUnsafeFunctionPointer {}

mod private {
    pub trait SealedUnsafeFunctionPointer: Copy {}
}

impl<F: private::SealedUnsafeFunctionPointer> UnsafeFunctionPointer for F {}

macro_rules! impl_unsafe_fn {
    (@recurse $first:ident $( , $rest:ident )*) => {
        impl_unsafe_fn!($( $rest ),*);
    };

    (@recurse) => {};

    ($( $param:ident ),*) => {
        impl<Output, $( $param ),*> private::SealedUnsafeFunctionPointer for unsafe extern "C" fn($( $param ),*) -> Output {}

        impl<Output, $( $param ),*> JitFunction<'_, unsafe extern "C" fn($( $param ),*) -> Output> {
            #[allow(non_snake_case)]
            #[inline(always)]
            pub unsafe fn call(&self, $( $param: $param ),*) -> Output {
                (self.inner)($( $param ),*)
            }
        }

        impl_unsafe_fn!(@recurse $( $param ),*);
    };
}

impl_unsafe_fn!(A, B, C, D, E, F, G, H, I, J, K, L, M);

#[cfg(feature = "experimental")]
pub mod experimental {
    use llvm_lib::error::{LLVMConsumeError, LLVMErrorRef, LLVMErrorTypeId, LLVMGetErrorMessage, LLVMGetErrorTypeId};
    use llvm_lib::orc::{
        LLVMOrcAddEagerlyCompiledIR, LLVMOrcAddLazilyCompiledIR, LLVMOrcCreateInstance, LLVMOrcDisposeInstance,
        LLVMOrcDisposeMangledSymbol, LLVMOrcGetErrorMsg, LLVMOrcGetMangledSymbol, LLVMOrcJITStackRef,
    };

    use crate::module::Module;
    use crate::support::to_c_str;
    use crate::targets::TargetMachine;

    use std::ffi::{CStr, CString};
    use std::mem::MaybeUninit;
    use std::ops::Deref;

    #[derive(Debug)]
    pub struct MangledSymbol(*mut libc::c_char);

    impl Deref for MangledSymbol {
        type Target = CStr;

        fn deref(&self) -> &CStr {
            unsafe { CStr::from_ptr(self.0) }
        }
    }

    impl Drop for MangledSymbol {
        fn drop(&mut self) {
            unsafe { LLVMOrcDisposeMangledSymbol(self.0) }
        }
    }

    #[derive(Debug)]
    pub struct LLVMError(LLVMErrorRef);

    impl LLVMError {
        pub fn get_type_id(&self) -> LLVMErrorTypeId {
            unsafe { LLVMGetErrorTypeId(self.0) }
        }
    }

    impl Deref for LLVMError {
        type Target = CStr;

        fn deref(&self) -> &CStr {
            unsafe {
                CStr::from_ptr(LLVMGetErrorMessage(self.0)) // FIXME: LLVMGetErrorMessage consumes the error, needs LLVMDisposeErrorMessage after
            }
        }
    }

    impl Drop for LLVMError {
        fn drop(&mut self) {
            unsafe { LLVMConsumeError(self.0) }
        }
    }

    #[derive(Debug)]
    pub struct Orc(LLVMOrcJITStackRef);

    impl Orc {
        pub fn create(target_machine: TargetMachine) -> Self {
            let stack_ref = unsafe { LLVMOrcCreateInstance(target_machine.target_machine) };

            Orc(stack_ref)
        }

        pub fn add_compiled_ir<'ctx>(&self, module: &Module<'ctx>, lazily: bool) -> Result<(), ()> {
            Ok(())
        }

        pub fn get_error(&self) -> &CStr {
            let err_str = unsafe { LLVMOrcGetErrorMsg(self.0) };

            if err_str.is_null() {
                panic!("Needs to be optional")
            }

            unsafe { CStr::from_ptr(err_str) }
        }

        pub fn get_mangled_symbol(&self, symbol: &str) -> MangledSymbol {
            let mut mangled_symbol = MaybeUninit::uninit();
            let c_symbol = to_c_str(symbol);

            unsafe { LLVMOrcGetMangledSymbol(self.0, mangled_symbol.as_mut_ptr(), c_symbol.as_ptr()) };

            MangledSymbol(unsafe { mangled_symbol.assume_init() })
        }
    }

    impl Drop for Orc {
        fn drop(&mut self) {
            LLVMError(unsafe { LLVMOrcDisposeInstance(self.0) });
        }
    }

    #[test]
    fn test_mangled_str() {
        use crate::targets::{CodeModel, InitializationConfig, RelocMode, Target};
        use crate::OptimizationLevel;

        Target::initialize_native(&InitializationConfig::default()).unwrap();

        let target_triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&target_triple).unwrap();
        let target_machine = target
            .create_target_machine(
                &target_triple,
                &"",
                &"",
                OptimizationLevel::None,
                RelocMode::Default,
                CodeModel::Default,
            )
            .unwrap();
        let orc = Orc::create(target_machine);

        assert_eq!(orc.get_error().to_str().unwrap(), "");

        let mangled_symbol = orc.get_mangled_symbol("MyStructName");

        assert_eq!(orc.get_error().to_str().unwrap(), "");

        assert_eq!(mangled_symbol.to_str().unwrap(), "MyStructName");
    }
}
