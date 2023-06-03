use super::*;

// llvm-c/Core.h
extern "C" {
    pub fn LLVMAddAlias2(
        M: LLVMModuleRef,
        ValueTy: LLVMTypeRef,
        AddrSpace: ::libc::c_uint,
        Aliasee: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMAddAttributeAtIndex(F: LLVMValueRef, Idx: LLVMAttributeIndex, A: LLVMAttributeRef);
    pub fn LLVMAddCallSiteAttribute(C: LLVMValueRef, Idx: LLVMAttributeIndex, A: LLVMAttributeRef);
    pub fn LLVMAddCase(Switch: LLVMValueRef, OnVal: LLVMValueRef, Dest: LLVMBasicBlockRef);
    pub fn LLVMAddClause(LandingPad: LLVMValueRef, ClauseVal: LLVMValueRef);
    pub fn LLVMAddDestination(IndirectBr: LLVMValueRef, Dest: LLVMBasicBlockRef);
    pub fn LLVMAddFunction(M: LLVMModuleRef, Name: *const ::libc::c_char, FunctionTy: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMAddGlobal(M: LLVMModuleRef, Ty: LLVMTypeRef, Name: *const ::libc::c_char) -> LLVMValueRef;
    pub fn LLVMAddGlobalIFunc(
        M: LLVMModuleRef,
        Name: *const ::libc::c_char,
        NameLen: ::libc::size_t,
        Ty: LLVMTypeRef,
        AddrSpace: ::libc::c_uint,
        Resolver: LLVMValueRef,
    ) -> LLVMValueRef;
    pub fn LLVMAddGlobalInAddressSpace(
        M: LLVMModuleRef,
        Ty: LLVMTypeRef,
        Name: *const ::libc::c_char,
        AddressSpace: ::libc::c_uint,
    ) -> LLVMValueRef;
    pub fn LLVMAddHandler(CatchSwitch: LLVMValueRef, Dest: LLVMBasicBlockRef);
    pub fn LLVMAddIncoming(
        PhiNode: LLVMValueRef,
        IncomingValues: *mut LLVMValueRef,
        IncomingBlocks: *mut LLVMBasicBlockRef,
        Count: ::libc::c_uint,
    );
    pub fn LLVMAddMetadataToInst(Builder: LLVMBuilderRef, Inst: LLVMValueRef);
    pub fn LLVMAddModuleFlag(
        M: LLVMModuleRef,
        Behavior: LLVMModuleFlagBehavior,
        Key: *const ::libc::c_char,
        KeyLen: ::libc::size_t,
        Val: LLVMMetadataRef,
    );
    pub fn LLVMAddNamedMetadataOperand(M: LLVMModuleRef, name: *const ::libc::c_char, Val: LLVMValueRef);
    pub fn LLVMAddTargetDependentFunctionAttr(Fn: LLVMValueRef, A: *const ::libc::c_char, V: *const ::libc::c_char);
    pub fn LLVMAliasGetAliasee(Alias: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMAliasSetAliasee(Alias: LLVMValueRef, Aliasee: LLVMValueRef);
    pub fn LLVMAlignOf(Ty: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMAppendBasicBlock(Fn: LLVMValueRef, Name: *const ::libc::c_char) -> LLVMBasicBlockRef;
    pub fn LLVMAppendBasicBlockInContext(
        C: LLVMContextRef,
        Fn: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMBasicBlockRef;
    pub fn LLVMAppendExistingBasicBlock(Fn: LLVMValueRef, BB: LLVMBasicBlockRef);
    pub fn LLVMAppendModuleInlineAsm(M: LLVMModuleRef, Asm: *const ::libc::c_char, Len: ::libc::size_t);
    pub fn LLVMArrayType(ElementType: LLVMTypeRef, ElementCount: ::libc::c_uint) -> LLVMTypeRef;
    pub fn LLVMBasicBlockAsValue(BB: LLVMBasicBlockRef) -> LLVMValueRef;
    pub fn LLVMBFloatType() -> LLVMTypeRef;
    pub fn LLVMBFloatTypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMBlockAddress(F: LLVMValueRef, BB: LLVMBasicBlockRef) -> LLVMValueRef;
    pub fn LLVMBuildAdd(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildAddrSpaceCast(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildAggregateRet(arg1: LLVMBuilderRef, RetVals: *mut LLVMValueRef, N: ::libc::c_uint) -> LLVMValueRef;
    pub fn LLVMBuildAlloca(arg1: LLVMBuilderRef, Ty: LLVMTypeRef, Name: *const ::libc::c_char) -> LLVMValueRef;
    pub fn LLVMBuildAnd(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildArrayAlloca(
        arg1: LLVMBuilderRef,
        Ty: LLVMTypeRef,
        Val: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildArrayMalloc(
        arg1: LLVMBuilderRef,
        Ty: LLVMTypeRef,
        Val: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildAShr(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildAtomicCmpXchg(
        B: LLVMBuilderRef,
        Ptr: LLVMValueRef,
        Cmp: LLVMValueRef,
        New: LLVMValueRef,
        SuccessOrdering: LLVMAtomicOrdering,
        FailureOrdering: LLVMAtomicOrdering,
        SingleThread: LLVMBool,
    ) -> LLVMValueRef;
    pub fn LLVMBuildAtomicRMW(
        B: LLVMBuilderRef,
        op: LLVMAtomicRMWBinOp,
        PTR: LLVMValueRef,
        Val: LLVMValueRef,
        ordering: LLVMAtomicOrdering,
        singleThread: LLVMBool,
    ) -> LLVMValueRef;
    pub fn LLVMBuildBinOp(
        B: LLVMBuilderRef,
        Op: LLVMOpcode,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildBitCast(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildBr(arg1: LLVMBuilderRef, Dest: LLVMBasicBlockRef) -> LLVMValueRef;
    pub fn LLVMBuildCall2(
        arg1: LLVMBuilderRef,
        arg2: LLVMTypeRef,
        Fn: LLVMValueRef,
        Args: *mut LLVMValueRef,
        NumArgs: ::libc::c_uint,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildCast(
        B: LLVMBuilderRef,
        Op: LLVMOpcode,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildCatchPad(
        B: LLVMBuilderRef,
        ParentPad: LLVMValueRef,
        Args: *mut LLVMValueRef,
        NumArgs: ::libc::c_uint,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildCatchRet(B: LLVMBuilderRef, CatchPad: LLVMValueRef, BB: LLVMBasicBlockRef) -> LLVMValueRef;
    pub fn LLVMBuildCatchSwitch(
        B: LLVMBuilderRef,
        ParentPad: LLVMValueRef,
        UnwindBB: LLVMBasicBlockRef,
        NumHandler: ::libc::c_uint,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildCleanupPad(
        B: LLVMBuilderRef,
        ParentPad: LLVMValueRef,
        Args: *mut LLVMValueRef,
        NumArgs: ::libc::c_uint,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildCleanupRet(B: LLVMBuilderRef, CatchPad: LLVMValueRef, BB: LLVMBasicBlockRef) -> LLVMValueRef;
    pub fn LLVMBuildCondBr(
        arg1: LLVMBuilderRef,
        If: LLVMValueRef,
        Then: LLVMBasicBlockRef,
        Else: LLVMBasicBlockRef,
    ) -> LLVMValueRef;
    pub fn LLVMBuilderGetDefaultFPMathTag(Builder: LLVMBuilderRef) -> LLVMMetadataRef;
    pub fn LLVMBuilderSetDefaultFPMathTag(Builder: LLVMBuilderRef, FPMathTag: LLVMMetadataRef);
    pub fn LLVMBuildExactSDiv(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildExactUDiv(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildExtractElement(
        arg1: LLVMBuilderRef,
        VecVal: LLVMValueRef,
        Index: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildExtractValue(
        arg1: LLVMBuilderRef,
        AggVal: LLVMValueRef,
        Index: ::libc::c_uint,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFAdd(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFCmp(
        arg1: LLVMBuilderRef,
        Op: LLVMRealPredicate,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFDiv(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFence(
        B: LLVMBuilderRef,
        ordering: LLVMAtomicOrdering,
        singleThread: LLVMBool,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFMul(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFNeg(arg1: LLVMBuilderRef, V: LLVMValueRef, Name: *const ::libc::c_char) -> LLVMValueRef;
    pub fn LLVMBuildFPCast(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFPExt(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFPToSI(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFPToUI(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFPTrunc(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFree(arg1: LLVMBuilderRef, PointerVal: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMBuildFreeze(arg1: LLVMBuilderRef, Val: LLVMValueRef, Name: *const ::libc::c_char) -> LLVMValueRef;
    pub fn LLVMBuildFRem(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFSub(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildGEP2(
        B: LLVMBuilderRef,
        Ty: LLVMTypeRef,
        Pointer: LLVMValueRef,
        Indices: *mut LLVMValueRef,
        NumIndices: ::libc::c_uint,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildGlobalString(
        B: LLVMBuilderRef,
        Str: *const ::libc::c_char,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildGlobalStringPtr(
        B: LLVMBuilderRef,
        Str: *const ::libc::c_char,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildICmp(
        arg1: LLVMBuilderRef,
        Op: LLVMIntPredicate,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildInBoundsGEP2(
        B: LLVMBuilderRef,
        Ty: LLVMTypeRef,
        Pointer: LLVMValueRef,
        Indices: *mut LLVMValueRef,
        NumIndices: ::libc::c_uint,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildIndirectBr(B: LLVMBuilderRef, Addr: LLVMValueRef, NumDests: ::libc::c_uint) -> LLVMValueRef;
    pub fn LLVMBuildInsertElement(
        arg1: LLVMBuilderRef,
        VecVal: LLVMValueRef,
        EltVal: LLVMValueRef,
        Index: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildInsertValue(
        arg1: LLVMBuilderRef,
        AggVal: LLVMValueRef,
        EltVal: LLVMValueRef,
        Index: ::libc::c_uint,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildIntCast(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildIntCast2(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        IsSigned: LLVMBool,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildIntToPtr(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildInvoke2(
        arg1: LLVMBuilderRef,
        Ty: LLVMTypeRef,
        Fn: LLVMValueRef,
        Args: *mut LLVMValueRef,
        NumArgs: ::libc::c_uint,
        Then: LLVMBasicBlockRef,
        Catch: LLVMBasicBlockRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildIsNotNull(arg1: LLVMBuilderRef, Val: LLVMValueRef, Name: *const ::libc::c_char) -> LLVMValueRef;
    pub fn LLVMBuildIsNull(arg1: LLVMBuilderRef, Val: LLVMValueRef, Name: *const ::libc::c_char) -> LLVMValueRef;
    pub fn LLVMBuildLandingPad(
        B: LLVMBuilderRef,
        Ty: LLVMTypeRef,
        PersFn: LLVMValueRef,
        NumClauses: ::libc::c_uint,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildLoad2(
        arg1: LLVMBuilderRef,
        Ty: LLVMTypeRef,
        PointerVal: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildLShr(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildMalloc(arg1: LLVMBuilderRef, Ty: LLVMTypeRef, Name: *const ::libc::c_char) -> LLVMValueRef;
    pub fn LLVMBuildMemCpy(
        B: LLVMBuilderRef,
        Dst: LLVMValueRef,
        DstAlign: ::libc::c_uint,
        Src: LLVMValueRef,
        SrcAlign: ::libc::c_uint,
        Size: LLVMValueRef,
    ) -> LLVMValueRef;
    pub fn LLVMBuildMemMove(
        B: LLVMBuilderRef,
        Dst: LLVMValueRef,
        DstAlign: ::libc::c_uint,
        Src: LLVMValueRef,
        SrcAlign: ::libc::c_uint,
        Size: LLVMValueRef,
    ) -> LLVMValueRef;
    pub fn LLVMBuildMemSet(
        B: LLVMBuilderRef,
        Ptr: LLVMValueRef,
        Val: LLVMValueRef,
        Len: LLVMValueRef,
        Align: ::libc::c_uint,
    ) -> LLVMValueRef;
    pub fn LLVMBuildMul(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildNeg(arg1: LLVMBuilderRef, V: LLVMValueRef, Name: *const ::libc::c_char) -> LLVMValueRef;
    pub fn LLVMBuildNot(arg1: LLVMBuilderRef, V: LLVMValueRef, Name: *const ::libc::c_char) -> LLVMValueRef;
    pub fn LLVMBuildNSWAdd(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildNSWMul(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildNSWNeg(B: LLVMBuilderRef, V: LLVMValueRef, Name: *const ::libc::c_char) -> LLVMValueRef;
    pub fn LLVMBuildNSWSub(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildNUWAdd(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildNUWMul(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildNUWNeg(B: LLVMBuilderRef, V: LLVMValueRef, Name: *const ::libc::c_char) -> LLVMValueRef;
    pub fn LLVMBuildNUWSub(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildOr(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildPhi(arg1: LLVMBuilderRef, Ty: LLVMTypeRef, Name: *const ::libc::c_char) -> LLVMValueRef;
    pub fn LLVMBuildPointerCast(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildPtrDiff2(
        arg1: LLVMBuilderRef,
        ElemTy: LLVMTypeRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildPtrToInt(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildResume(B: LLVMBuilderRef, Exn: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMBuildRet(arg1: LLVMBuilderRef, V: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMBuildRetVoid(arg1: LLVMBuilderRef) -> LLVMValueRef;
    pub fn LLVMBuildSDiv(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildSelect(
        arg1: LLVMBuilderRef,
        If: LLVMValueRef,
        Then: LLVMValueRef,
        Else: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildSExt(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildSExtOrBitCast(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildShl(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildShuffleVector(
        arg1: LLVMBuilderRef,
        V1: LLVMValueRef,
        V2: LLVMValueRef,
        Mask: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildSIToFP(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildSRem(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildStore(arg1: LLVMBuilderRef, Val: LLVMValueRef, Ptr: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMBuildStructGEP2(
        B: LLVMBuilderRef,
        Ty: LLVMTypeRef,
        Pointer: LLVMValueRef,
        Idx: ::libc::c_uint,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildSub(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildSwitch(
        arg1: LLVMBuilderRef,
        V: LLVMValueRef,
        Else: LLVMBasicBlockRef,
        NumCases: ::libc::c_uint,
    ) -> LLVMValueRef;
    pub fn LLVMBuildTrunc(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildTruncOrBitCast(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildUDiv(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildUIToFP(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildUnreachable(B: LLVMBuilderRef) -> LLVMValueRef;
    pub fn LLVMBuildURem(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildVAArg(
        arg1: LLVMBuilderRef,
        List: LLVMValueRef,
        Ty: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildXor(
        arg1: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildZExt(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildZExtOrBitCast(
        arg1: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const ::libc::c_char,
    ) -> LLVMValueRef;
    pub fn LLVMClearInsertionPosition(Builder: LLVMBuilderRef);
    pub fn LLVMCloneModule(M: LLVMModuleRef) -> LLVMModuleRef;
    pub fn LLVMConstAdd(LHSConstant: LLVMValueRef, RHSConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstAddrSpaceCast(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstAllOnes(Ty: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstAnd(LHSConstant: LLVMValueRef, RHSConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstArray(
        ElementTy: LLVMTypeRef,
        ConstantVals: *mut LLVMValueRef,
        Length: ::libc::c_uint,
    ) -> LLVMValueRef;
    pub fn LLVMConstAShr(LHSConstant: LLVMValueRef, RHSConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstBitCast(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstExtractElement(VectorConstant: LLVMValueRef, IndexConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstFCmp(
        Predicate: LLVMRealPredicate,
        LHSConstant: LLVMValueRef,
        RHSConstant: LLVMValueRef,
    ) -> LLVMValueRef;
    pub fn LLVMConstFPCast(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstFPExt(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstFPToSI(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstFPToUI(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstFPTrunc(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstGEP2(
        Ty: LLVMTypeRef,
        ConstantVal: LLVMValueRef,
        ConstantIndices: *mut LLVMValueRef,
        NumIndices: ::libc::c_uint,
    ) -> LLVMValueRef;
    pub fn LLVMConstICmp(
        Predicate: LLVMIntPredicate,
        LHSConstant: LLVMValueRef,
        RHSConstant: LLVMValueRef,
    ) -> LLVMValueRef;
    pub fn LLVMConstInBoundsGEP2(
        Ty: LLVMTypeRef,
        ConstantVal: LLVMValueRef,
        ConstantIndices: *mut LLVMValueRef,
        NumIndices: ::libc::c_uint,
    ) -> LLVMValueRef;
    pub fn LLVMConstInsertElement(
        VectorConstant: LLVMValueRef,
        ElementValueConstant: LLVMValueRef,
        IndexConstant: LLVMValueRef,
    ) -> LLVMValueRef;
    pub fn LLVMConstInt(IntTy: LLVMTypeRef, N: ::libc::c_ulonglong, SignExtend: LLVMBool) -> LLVMValueRef;
    pub fn LLVMConstIntCast(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef, isSigned: LLVMBool) -> LLVMValueRef;
    pub fn LLVMConstIntGetSExtValue(ConstantVal: LLVMValueRef) -> ::libc::c_longlong;
    pub fn LLVMConstIntGetZExtValue(ConstantVal: LLVMValueRef) -> ::libc::c_ulonglong;
    pub fn LLVMConstIntOfArbitraryPrecision(
        IntTy: LLVMTypeRef,
        NumWords: ::libc::c_uint,
        Words: *const u64,
    ) -> LLVMValueRef;
    pub fn LLVMConstIntOfString(IntTy: LLVMTypeRef, Text: *const ::libc::c_char, Radix: u8) -> LLVMValueRef;
    pub fn LLVMConstIntOfStringAndSize(
        IntTy: LLVMTypeRef,
        Text: *const ::libc::c_char,
        SLen: ::libc::c_uint,
        Radix: u8,
    ) -> LLVMValueRef;
    pub fn LLVMConstIntToPtr(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstLShr(LHSConstant: LLVMValueRef, RHSConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstMul(LHSConstant: LLVMValueRef, RHSConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstNamedStruct(
        StructTy: LLVMTypeRef,
        ConstantVals: *mut LLVMValueRef,
        Count: ::libc::c_uint,
    ) -> LLVMValueRef;
    pub fn LLVMConstNeg(ConstantVal: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstNot(ConstantVal: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstNSWAdd(LHSConstant: LLVMValueRef, RHSConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstNSWMul(LHSConstant: LLVMValueRef, RHSConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstNSWNeg(ConstantVal: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstNSWSub(LHSConstant: LLVMValueRef, RHSConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstNull(Ty: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstNUWAdd(LHSConstant: LLVMValueRef, RHSConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstNUWMul(LHSConstant: LLVMValueRef, RHSConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstNUWNeg(ConstantVal: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstNUWSub(LHSConstant: LLVMValueRef, RHSConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstOr(LHSConstant: LLVMValueRef, RHSConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstPointerCast(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstPointerNull(Ty: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstPtrToInt(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstReal(RealTy: LLVMTypeRef, N: ::libc::c_double) -> LLVMValueRef;
    pub fn LLVMConstRealGetDouble(ConstantVal: LLVMValueRef, losesInfo: *mut LLVMBool) -> ::libc::c_double;
    pub fn LLVMConstRealOfString(RealTy: LLVMTypeRef, Text: *const ::libc::c_char) -> LLVMValueRef;
    pub fn LLVMConstRealOfStringAndSize(
        RealTy: LLVMTypeRef,
        Text: *const ::libc::c_char,
        SLen: ::libc::c_uint,
    ) -> LLVMValueRef;
    pub fn LLVMConstSelect(
        ConstantCondition: LLVMValueRef,
        ConstantIfTrue: LLVMValueRef,
        ConstantIfFalse: LLVMValueRef,
    ) -> LLVMValueRef;
    pub fn LLVMConstSExt(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstSExtOrBitCast(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstShl(LHSConstant: LLVMValueRef, RHSConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstShuffleVector(
        VectorAConstant: LLVMValueRef,
        VectorBConstant: LLVMValueRef,
        MaskConstant: LLVMValueRef,
    ) -> LLVMValueRef;
    pub fn LLVMConstSIToFP(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstString(
        Str: *const ::libc::c_char,
        Length: ::libc::c_uint,
        DontNullTerminate: LLVMBool,
    ) -> LLVMValueRef;
    pub fn LLVMConstStringInContext(
        C: LLVMContextRef,
        Str: *const ::libc::c_char,
        Length: ::libc::c_uint,
        DontNullTerminate: LLVMBool,
    ) -> LLVMValueRef;
    pub fn LLVMConstStruct(ConstantVals: *mut LLVMValueRef, Count: ::libc::c_uint, Packed: LLVMBool) -> LLVMValueRef;
    pub fn LLVMConstStructInContext(
        C: LLVMContextRef,
        ConstantVals: *mut LLVMValueRef,
        Count: ::libc::c_uint,
        Packed: LLVMBool,
    ) -> LLVMValueRef;
    pub fn LLVMConstSub(LHSConstant: LLVMValueRef, RHSConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstTrunc(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstTruncOrBitCast(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstUIToFP(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstVector(ScalarConstantVals: *mut LLVMValueRef, Size: ::libc::c_uint) -> LLVMValueRef;
    pub fn LLVMConstXor(LHSConstant: LLVMValueRef, RHSConstant: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMConstZExt(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstZExtOrBitCast(ConstantVal: LLVMValueRef, ToType: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMContextCreate() -> LLVMContextRef;
    pub fn LLVMContextDispose(C: LLVMContextRef);
    pub fn LLVMContextGetDiagnosticContext(C: LLVMContextRef) -> *mut ::libc::c_void;
    pub fn LLVMContextGetDiagnosticHandler(C: LLVMContextRef) -> LLVMDiagnosticHandler;
    pub fn LLVMContextSetDiagnosticHandler(
        C: LLVMContextRef,
        Handler: LLVMDiagnosticHandler,
        DiagnosticContext: *mut ::libc::c_void,
    );
    pub fn LLVMContextSetDiscardValueNames(C: LLVMContextRef, Discard: LLVMBool);
    pub fn LLVMContextSetOpaquePointers(C: LLVMContextRef, OpaquePointers: LLVMBool);
    pub fn LLVMContextSetYieldCallback(
        C: LLVMContextRef,
        Callback: LLVMYieldCallback,
        OpaqueHandle: *mut ::libc::c_void,
    );
    pub fn LLVMContextShouldDiscardValueNames(C: LLVMContextRef) -> LLVMBool;
    pub fn LLVMCopyModuleFlagsMetadata(M: LLVMModuleRef, Len: *mut ::libc::size_t) -> *mut LLVMModuleFlagEntry;
    pub fn LLVMCountBasicBlocks(Fn: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMCountIncoming(PhiNode: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMCountParams(Fn: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMCountParamTypes(FunctionTy: LLVMTypeRef) -> ::libc::c_uint;
    pub fn LLVMCountStructElementTypes(StructTy: LLVMTypeRef) -> ::libc::c_uint;
    pub fn LLVMCreateBasicBlockInContext(C: LLVMContextRef, Name: *const ::libc::c_char) -> LLVMBasicBlockRef;
    pub fn LLVMCreateBuilder() -> LLVMBuilderRef;
    pub fn LLVMCreateBuilderInContext(C: LLVMContextRef) -> LLVMBuilderRef;
    pub fn LLVMCreateEnumAttribute(C: LLVMContextRef, KindID: ::libc::c_uint, Val: u64) -> LLVMAttributeRef;
    pub fn LLVMCreateFunctionPassManager(MP: LLVMModuleProviderRef) -> LLVMPassManagerRef;
    pub fn LLVMCreateFunctionPassManagerForModule(M: LLVMModuleRef) -> LLVMPassManagerRef;
    pub fn LLVMCreateMemoryBufferWithContentsOfFile(
        Path: *const ::libc::c_char,
        OutMemBuf: *mut LLVMMemoryBufferRef,
        OutMessage: *mut *mut ::libc::c_char,
    ) -> LLVMBool;
    pub fn LLVMCreateMemoryBufferWithMemoryRange(
        InputData: *const ::libc::c_char,
        InputDataLength: ::libc::size_t,
        BufferName: *const ::libc::c_char,
        RequiresNullTerminator: LLVMBool,
    ) -> LLVMMemoryBufferRef;
    pub fn LLVMCreateMemoryBufferWithMemoryRangeCopy(
        InputData: *const ::libc::c_char,
        InputDataLength: ::libc::size_t,
        BufferName: *const ::libc::c_char,
    ) -> LLVMMemoryBufferRef;
    pub fn LLVMCreateMemoryBufferWithSTDIN(
        OutMemBuf: *mut LLVMMemoryBufferRef,
        OutMessage: *mut *mut ::libc::c_char,
    ) -> LLVMBool;
    pub fn LLVMCreateMessage(Message: *const ::libc::c_char) -> *mut ::libc::c_char;
    pub fn LLVMCreateModuleProviderForExistingModule(M: LLVMModuleRef) -> LLVMModuleProviderRef;
    pub fn LLVMCreatePassManager() -> LLVMPassManagerRef;
    pub fn LLVMCreateStringAttribute(
        C: LLVMContextRef,
        K: *const ::libc::c_char,
        KLength: ::libc::c_uint,
        V: *const ::libc::c_char,
        VLength: ::libc::c_uint,
    ) -> LLVMAttributeRef;
    pub fn LLVMCreateTypeAttribute(
        C: LLVMContextRef,
        KindID: ::libc::c_uint,
        type_ref: LLVMTypeRef,
    ) -> LLVMAttributeRef;
    pub fn LLVMDeleteBasicBlock(BB: LLVMBasicBlockRef);
    pub fn LLVMDeleteFunction(Fn: LLVMValueRef);
    pub fn LLVMDeleteGlobal(GlobalVar: LLVMValueRef);
    pub fn LLVMDeleteInstruction(Inst: LLVMValueRef);
    pub fn LLVMDisposeBuilder(Builder: LLVMBuilderRef);
    pub fn LLVMDisposeMemoryBuffer(MemBuf: LLVMMemoryBufferRef);
    pub fn LLVMDisposeMessage(Message: *mut ::libc::c_char);
    pub fn LLVMDisposeModule(M: LLVMModuleRef);
    pub fn LLVMDisposeModuleFlagsMetadata(Entries: *mut LLVMModuleFlagEntry);
    pub fn LLVMDisposeModuleProvider(M: LLVMModuleProviderRef);
    pub fn LLVMDisposePassManager(PM: LLVMPassManagerRef);
    pub fn LLVMDisposeValueMetadataEntries(Entries: *mut LLVMValueMetadataEntry);
    pub fn LLVMDoubleType() -> LLVMTypeRef;
    pub fn LLVMDoubleTypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMDumpModule(M: LLVMModuleRef);
    pub fn LLVMDumpType(Val: LLVMTypeRef);
    pub fn LLVMDumpValue(Val: LLVMValueRef);
    pub fn LLVMEraseGlobalIFunc(IFunc: LLVMValueRef);
    pub fn LLVMFinalizeFunctionPassManager(FPM: LLVMPassManagerRef) -> LLVMBool;
    pub fn LLVMFloatType() -> LLVMTypeRef;
    pub fn LLVMFloatTypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMFP128Type() -> LLVMTypeRef;
    pub fn LLVMFP128TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMFunctionType(
        ReturnType: LLVMTypeRef,
        ParamTypes: *mut LLVMTypeRef,
        ParamCount: ::libc::c_uint,
        IsVarArg: LLVMBool,
    ) -> LLVMTypeRef;
    pub fn LLVMGetAggregateElement(C: LLVMValueRef, idx: ::libc::c_uint) -> LLVMValueRef;
    pub fn LLVMGetAlignment(V: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMGetAllocatedType(Alloca: LLVMValueRef) -> LLVMTypeRef;
    pub fn LLVMGetArgOperand(Funclet: LLVMValueRef, i: ::libc::c_uint) -> LLVMValueRef;
    pub fn LLVMGetArrayLength(ArrayTy: LLVMTypeRef) -> ::libc::c_uint;
    pub fn LLVMGetAsString(C: LLVMValueRef, Length: *mut ::libc::size_t) -> *const ::libc::c_char;
    pub fn LLVMGetAtomicRMWBinOp(AtomicRMWInst: LLVMValueRef) -> LLVMAtomicRMWBinOp;
    pub fn LLVMGetAttributeCountAtIndex(F: LLVMValueRef, Idx: LLVMAttributeIndex) -> ::libc::c_uint;
    pub fn LLVMGetAttributesAtIndex(F: LLVMValueRef, Idx: LLVMAttributeIndex, Attrs: *mut LLVMAttributeRef);
    pub fn LLVMGetBasicBlockName(BB: LLVMBasicBlockRef) -> *const ::libc::c_char;
    pub fn LLVMGetBasicBlockParent(BB: LLVMBasicBlockRef) -> LLVMValueRef;
    pub fn LLVMGetBasicBlocks(Fn: LLVMValueRef, BasicBlocks: *mut LLVMBasicBlockRef);
    pub fn LLVMGetBasicBlockTerminator(BB: LLVMBasicBlockRef) -> LLVMValueRef;
    pub fn LLVMGetBufferSize(MemBuf: LLVMMemoryBufferRef) -> ::libc::size_t;
    pub fn LLVMGetBufferStart(MemBuf: LLVMMemoryBufferRef) -> *const ::libc::c_char;
    pub fn LLVMGetCalledFunctionType(C: LLVMValueRef) -> LLVMTypeRef;
    pub fn LLVMGetCalledValue(Instr: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetCallSiteAttributeCount(C: LLVMValueRef, Idx: LLVMAttributeIndex) -> ::libc::c_uint;
    pub fn LLVMGetCallSiteAttributes(C: LLVMValueRef, Idx: LLVMAttributeIndex, Attrs: *mut LLVMAttributeRef);
    pub fn LLVMGetCallSiteEnumAttribute(
        C: LLVMValueRef,
        Idx: LLVMAttributeIndex,
        KindID: ::libc::c_uint,
    ) -> LLVMAttributeRef;
    pub fn LLVMGetCallSiteStringAttribute(
        C: LLVMValueRef,
        Idx: LLVMAttributeIndex,
        K: *const ::libc::c_char,
        KLen: ::libc::c_uint,
    ) -> LLVMAttributeRef;
    pub fn LLVMGetCastOpcode(
        arg1: LLVMValueRef,
        SrcIsSigned: LLVMBool,
        DestTy: LLVMTypeRef,
        DestIsSigned: LLVMBool,
    ) -> LLVMOpcode;
    pub fn LLVMGetClause(LandingPad: LLVMValueRef, Idx: ::libc::c_uint) -> LLVMValueRef;
    pub fn LLVMGetCmpXchgFailureOrdering(CmpXchgInst: LLVMValueRef) -> LLVMAtomicOrdering;
    pub fn LLVMGetCmpXchgSuccessOrdering(CmpXchgInst: LLVMValueRef) -> LLVMAtomicOrdering;
    pub fn LLVMGetCondition(Branch: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetConstOpcode(ConstantVal: LLVMValueRef) -> LLVMOpcode;
    pub fn LLVMGetCurrentDebugLocation2(Builder: LLVMBuilderRef) -> LLVMMetadataRef;
    pub fn LLVMGetDataLayoutStr(M: LLVMModuleRef) -> *const ::libc::c_char;
    pub fn LLVMGetDebugLocColumn(Val: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMGetDebugLocDirectory(Val: LLVMValueRef, Length: *mut ::libc::c_uint) -> *const ::libc::c_char;
    pub fn LLVMGetDebugLocFilename(Val: LLVMValueRef, Length: *mut ::libc::c_uint) -> *const ::libc::c_char;
    pub fn LLVMGetDebugLocLine(Val: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMGetDiagInfoDescription(DI: LLVMDiagnosticInfoRef) -> *mut ::libc::c_char;
    pub fn LLVMGetDiagInfoSeverity(DI: LLVMDiagnosticInfoRef) -> LLVMDiagnosticSeverity;
    pub fn LLVMGetDLLStorageClass(Global: LLVMValueRef) -> LLVMDLLStorageClass;
    pub fn LLVMGetElementType(Ty: LLVMTypeRef) -> LLVMTypeRef;
    pub fn LLVMGetEntryBasicBlock(Fn: LLVMValueRef) -> LLVMBasicBlockRef;
    pub fn LLVMGetEnumAttributeAtIndex(
        F: LLVMValueRef,
        Idx: LLVMAttributeIndex,
        KindID: ::libc::c_uint,
    ) -> LLVMAttributeRef;
    pub fn LLVMGetEnumAttributeKind(A: LLVMAttributeRef) -> ::libc::c_uint;
    pub fn LLVMGetEnumAttributeKindForName(Name: *const ::libc::c_char, SLen: ::libc::size_t) -> ::libc::c_uint;
    pub fn LLVMGetEnumAttributeValue(A: LLVMAttributeRef) -> u64;
    pub fn LLVMGetFCmpPredicate(Inst: LLVMValueRef) -> LLVMRealPredicate;
    pub fn LLVMGetFirstBasicBlock(Fn: LLVMValueRef) -> LLVMBasicBlockRef;
    pub fn LLVMGetFirstFunction(M: LLVMModuleRef) -> LLVMValueRef;
    pub fn LLVMGetFirstGlobal(M: LLVMModuleRef) -> LLVMValueRef;
    pub fn LLVMGetFirstGlobalAlias(M: LLVMModuleRef) -> LLVMValueRef;
    pub fn LLVMGetFirstGlobalIFunc(M: LLVMModuleRef) -> LLVMValueRef;
    pub fn LLVMGetFirstInstruction(BB: LLVMBasicBlockRef) -> LLVMValueRef;
    pub fn LLVMGetFirstNamedMetadata(M: LLVMModuleRef) -> LLVMNamedMDNodeRef;
    pub fn LLVMGetFirstParam(Fn: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetFirstUse(Val: LLVMValueRef) -> LLVMUseRef;
    pub fn LLVMGetFunctionCallConv(Fn: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMGetGC(Fn: LLVMValueRef) -> *const ::libc::c_char;
    pub fn LLVMGetGEPSourceElementType(GEP: LLVMValueRef) -> LLVMTypeRef;
    pub fn LLVMGetGlobalContext() -> LLVMContextRef;
    pub fn LLVMGetGlobalIFuncResolver(IFunc: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetGlobalParent(Global: LLVMValueRef) -> LLVMModuleRef;
    pub fn LLVMGetGlobalPassRegistry() -> LLVMPassRegistryRef;
    pub fn LLVMGetHandlers(CatchSwitch: LLVMValueRef, Handlers: *mut LLVMBasicBlockRef);
    pub fn LLVMGetICmpPredicate(Inst: LLVMValueRef) -> LLVMIntPredicate;
    pub fn LLVMGetIncomingBlock(PhiNode: LLVMValueRef, Index: ::libc::c_uint) -> LLVMBasicBlockRef;
    pub fn LLVMGetIncomingValue(PhiNode: LLVMValueRef, Index: ::libc::c_uint) -> LLVMValueRef;
    pub fn LLVMGetIndices(Inst: LLVMValueRef) -> *const ::libc::c_uint;
    pub fn LLVMGetInitializer(GlobalVar: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetInlineAsm(
        Ty: LLVMTypeRef,
        AsmString: *mut ::libc::c_char,
        AsmStringSize: ::libc::size_t,
        Constraints: *mut ::libc::c_char,
        ConstraintsSize: ::libc::size_t,
        HasSideEffects: LLVMBool,
        IsAlignStack: LLVMBool,
        Dialect: LLVMInlineAsmDialect,
        CanThrow: LLVMBool,
    ) -> LLVMValueRef;
    pub fn LLVMGetInsertBlock(Builder: LLVMBuilderRef) -> LLVMBasicBlockRef;
    pub fn LLVMGetInstructionCallConv(Instr: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMGetInstructionOpcode(Inst: LLVMValueRef) -> LLVMOpcode;
    pub fn LLVMGetInstructionParent(Inst: LLVMValueRef) -> LLVMBasicBlockRef;
    pub fn LLVMGetIntrinsicDeclaration(
        Mod: LLVMModuleRef,
        ID: ::libc::c_uint,
        ParamTypes: *mut LLVMTypeRef,
        ParamCount: ::libc::size_t,
    ) -> LLVMValueRef;
    pub fn LLVMGetIntrinsicID(Fn: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMGetIntTypeWidth(IntegerTy: LLVMTypeRef) -> ::libc::c_uint;
    pub fn LLVMGetLastBasicBlock(Fn: LLVMValueRef) -> LLVMBasicBlockRef;
    pub fn LLVMGetLastEnumAttributeKind() -> ::libc::c_uint;
    pub fn LLVMGetLastFunction(M: LLVMModuleRef) -> LLVMValueRef;
    pub fn LLVMGetLastGlobal(M: LLVMModuleRef) -> LLVMValueRef;
    pub fn LLVMGetLastGlobalAlias(M: LLVMModuleRef) -> LLVMValueRef;
    pub fn LLVMGetLastGlobalIFunc(M: LLVMModuleRef) -> LLVMValueRef;
    pub fn LLVMGetLastInstruction(BB: LLVMBasicBlockRef) -> LLVMValueRef;
    pub fn LLVMGetLastNamedMetadata(M: LLVMModuleRef) -> LLVMNamedMDNodeRef;
    pub fn LLVMGetLastParam(Fn: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetLinkage(Global: LLVMValueRef) -> LLVMLinkage;
    pub fn LLVMGetMaskValue(ShuffleVectorInst: LLVMValueRef, Elt: ::libc::c_uint) -> ::libc::c_int;
    pub fn LLVMGetMDKindID(Name: *const ::libc::c_char, SLen: ::libc::c_uint) -> ::libc::c_uint;
    pub fn LLVMGetMDKindIDInContext(
        C: LLVMContextRef,
        Name: *const ::libc::c_char,
        SLen: ::libc::c_uint,
    ) -> ::libc::c_uint;
    pub fn LLVMGetMDNodeNumOperands(V: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMGetMDNodeOperands(V: LLVMValueRef, Dest: *mut LLVMValueRef);
    pub fn LLVMGetMDString(V: LLVMValueRef, Len: *mut ::libc::c_uint) -> *const ::libc::c_char;
    pub fn LLVMGetMetadata(Val: LLVMValueRef, KindID: ::libc::c_uint) -> LLVMValueRef;
    pub fn LLVMGetModuleContext(M: LLVMModuleRef) -> LLVMContextRef;
    pub fn LLVMGetModuleFlag(M: LLVMModuleRef, Key: *const ::libc::c_char, KeyLen: ::libc::size_t) -> LLVMMetadataRef;
    pub fn LLVMGetModuleIdentifier(M: LLVMModuleRef, Len: *mut ::libc::size_t) -> *const ::libc::c_char;
    pub fn LLVMGetModuleInlineAsm(M: LLVMModuleRef, Len: *mut ::libc::size_t) -> *const ::libc::c_char;
    pub fn LLVMGetNamedFunction(M: LLVMModuleRef, Name: *const ::libc::c_char) -> LLVMValueRef;
    pub fn LLVMGetNamedGlobal(M: LLVMModuleRef, Name: *const ::libc::c_char) -> LLVMValueRef;
    pub fn LLVMGetNamedGlobalAlias(
        M: LLVMModuleRef,
        Name: *const ::libc::c_char,
        NameLen: ::libc::size_t,
    ) -> LLVMValueRef;
    pub fn LLVMGetNamedGlobalIFunc(
        M: LLVMModuleRef,
        Name: *const ::libc::c_char,
        NameLen: ::libc::size_t,
    ) -> LLVMValueRef;
    pub fn LLVMGetNamedMetadata(
        M: LLVMModuleRef,
        Name: *const ::libc::c_char,
        NameLen: ::libc::size_t,
    ) -> LLVMNamedMDNodeRef;
    pub fn LLVMGetNamedMetadataName(
        NamedMD: LLVMNamedMDNodeRef,
        NameLen: *const ::libc::size_t,
    ) -> *const ::libc::c_char;
    pub fn LLVMGetNamedMetadataNumOperands(M: LLVMModuleRef, name: *const ::libc::c_char) -> ::libc::c_uint;
    pub fn LLVMGetNamedMetadataOperands(M: LLVMModuleRef, name: *const ::libc::c_char, Dest: *mut LLVMValueRef);
    pub fn LLVMGetNextBasicBlock(BB: LLVMBasicBlockRef) -> LLVMBasicBlockRef;
    pub fn LLVMGetNextFunction(Fn: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetNextGlobal(GlobalVar: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetNextGlobalAlias(GA: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetNextGlobalIFunc(IFunc: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetNextInstruction(Inst: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetNextNamedMetadata(NamedMDNode: LLVMNamedMDNodeRef) -> LLVMNamedMDNodeRef;
    pub fn LLVMGetNextParam(Arg: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetNextUse(U: LLVMUseRef) -> LLVMUseRef;
    pub fn LLVMGetNormalDest(InvokeInst: LLVMValueRef) -> LLVMBasicBlockRef;
    pub fn LLVMGetNumArgOperands(Instr: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMGetNumClauses(LandingPad: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMGetNumContainedTypes(Tp: LLVMTypeRef) -> ::libc::c_uint;
    pub fn LLVMGetNumHandlers(CatchSwitch: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMGetNumIndices(Inst: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMGetNumMaskElements(ShuffleVectorInst: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMGetNumOperands(Val: LLVMValueRef) -> ::libc::c_int;
    pub fn LLVMGetNumSuccessors(Term: LLVMValueRef) -> ::libc::c_uint;
    pub fn LLVMGetOperand(Val: LLVMValueRef, Index: ::libc::c_uint) -> LLVMValueRef;
    pub fn LLVMGetOperandUse(Val: LLVMValueRef, Index: ::libc::c_uint) -> LLVMUseRef;
    pub fn LLVMGetOrdering(MemoryAccessInst: LLVMValueRef) -> LLVMAtomicOrdering;
    pub fn LLVMGetOrInsertNamedMetadata(
        M: LLVMModuleRef,
        Name: *const ::libc::c_char,
        NameLen: ::libc::size_t,
    ) -> LLVMNamedMDNodeRef;
    pub fn LLVMGetParam(Fn: LLVMValueRef, Index: ::libc::c_uint) -> LLVMValueRef;
    pub fn LLVMGetParamParent(Inst: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetParams(Fn: LLVMValueRef, Params: *mut LLVMValueRef);
    pub fn LLVMGetParamTypes(FunctionTy: LLVMTypeRef, Dest: *mut LLVMTypeRef);
    pub fn LLVMGetParentCatchSwitch(CatchPad: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetPersonalityFn(Fn: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetPointerAddressSpace(PointerTy: LLVMTypeRef) -> ::libc::c_uint;
    pub fn LLVMGetPoison(Ty: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMGetPreviousBasicBlock(BB: LLVMBasicBlockRef) -> LLVMBasicBlockRef;
    pub fn LLVMGetPreviousFunction(Fn: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetPreviousGlobal(GlobalVar: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetPreviousGlobalAlias(GA: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetPreviousGlobalIFunc(IFunc: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetPreviousInstruction(Inst: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetPreviousNamedMetadata(NamedMDNode: LLVMNamedMDNodeRef) -> LLVMNamedMDNodeRef;
    pub fn LLVMGetPreviousParam(Arg: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetReturnType(FunctionTy: LLVMTypeRef) -> LLVMTypeRef;
    pub fn LLVMGetSection(Global: LLVMValueRef) -> *const ::libc::c_char;
    pub fn LLVMGetSourceFileName(M: LLVMModuleRef, Len: *mut ::libc::size_t) -> *const ::libc::c_char;
    pub fn LLVMGetStringAttributeAtIndex(
        F: LLVMValueRef,
        Idx: LLVMAttributeIndex,
        K: *const ::libc::c_char,
        KLen: ::libc::c_uint,
    ) -> LLVMAttributeRef;
    pub fn LLVMGetStringAttributeKind(A: LLVMAttributeRef, Length: *mut ::libc::c_uint) -> *const ::libc::c_char;
    pub fn LLVMGetStringAttributeValue(A: LLVMAttributeRef, Length: *mut ::libc::c_uint) -> *const ::libc::c_char;
    pub fn LLVMGetStructElementTypes(StructTy: LLVMTypeRef, Dest: *mut LLVMTypeRef);
    pub fn LLVMGetStructName(Ty: LLVMTypeRef) -> *const ::libc::c_char;
    pub fn LLVMGetSubtypes(Tp: LLVMTypeRef, Arr: *mut LLVMTypeRef);
    pub fn LLVMGetSuccessor(Term: LLVMValueRef, i: ::libc::c_uint) -> LLVMBasicBlockRef;
    pub fn LLVMGetSwitchDefaultDest(SwitchInstr: LLVMValueRef) -> LLVMBasicBlockRef;
    pub fn LLVMGetTarget(M: LLVMModuleRef) -> *const ::libc::c_char;
    pub fn LLVMGetThreadLocalMode(GlobalVar: LLVMValueRef) -> LLVMThreadLocalMode;
    pub fn LLVMGetTypeAttributeValue(A: LLVMAttributeRef) -> LLVMTypeRef;
    pub fn LLVMGetTypeByName2(C: LLVMContextRef, Name: *const ::libc::c_char) -> LLVMTypeRef;
    pub fn LLVMGetTypeContext(Ty: LLVMTypeRef) -> LLVMContextRef;
    pub fn LLVMGetTypeKind(Ty: LLVMTypeRef) -> LLVMTypeKind;
    pub fn LLVMGetUndef(Ty: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMGetUndefMaskElem() -> ::libc::c_int;
    pub fn LLVMGetUnnamedAddress(Global: LLVMValueRef) -> LLVMUnnamedAddr;
    pub fn LLVMGetUnwindDest(InvokeInst: LLVMValueRef) -> LLVMBasicBlockRef;
    pub fn LLVMGetUsedValue(U: LLVMUseRef) -> LLVMValueRef;
    pub fn LLVMGetUser(U: LLVMUseRef) -> LLVMValueRef;
    pub fn LLVMGetValueKind(Val: LLVMValueRef) -> LLVMValueKind;
    pub fn LLVMGetValueName2(Val: LLVMValueRef, Length: *mut ::libc::size_t) -> *const ::libc::c_char;
    pub fn LLVMGetVectorSize(VectorTy: LLVMTypeRef) -> ::libc::c_uint;
    pub fn LLVMGetVersion(Major: *mut ::libc::c_uint, Minor: *mut ::libc::c_uint, Patch: *mut ::libc::c_uint);
    pub fn LLVMGetVisibility(Global: LLVMValueRef) -> LLVMVisibility;
    pub fn LLVMGetVolatile(MemoryAccessInst: LLVMValueRef) -> LLVMBool;
    pub fn LLVMGetWeak(CmpXchgInst: LLVMValueRef) -> LLVMBool;
    pub fn LLVMGlobalClearMetadata(Global: LLVMValueRef);
    pub fn LLVMGlobalCopyAllMetadata(
        Value: LLVMValueRef,
        NumEntries: *mut ::libc::size_t,
    ) -> *mut LLVMValueMetadataEntry;
    pub fn LLVMGlobalEraseMetadata(Global: LLVMValueRef, Kind: ::libc::c_uint);
    pub fn LLVMGlobalGetValueType(Global: LLVMValueRef) -> LLVMTypeRef;
    pub fn LLVMGlobalSetMetadata(Global: LLVMValueRef, Kind: ::libc::c_uint, MD: LLVMMetadataRef);
    pub fn LLVMHalfType() -> LLVMTypeRef;
    pub fn LLVMHalfTypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMHasMetadata(Val: LLVMValueRef) -> ::libc::c_int;
    pub fn LLVMHasPersonalityFn(Fn: LLVMValueRef) -> LLVMBool;
    pub fn LLVMInitializeFunctionPassManager(FPM: LLVMPassManagerRef) -> LLVMBool;
    pub fn LLVMInsertBasicBlock(InsertBeforeBB: LLVMBasicBlockRef, Name: *const ::libc::c_char) -> LLVMBasicBlockRef;
    pub fn LLVMInsertBasicBlockInContext(
        C: LLVMContextRef,
        BB: LLVMBasicBlockRef,
        Name: *const ::libc::c_char,
    ) -> LLVMBasicBlockRef;
    pub fn LLVMInsertExistingBasicBlockAfterInsertBlock(Builder: LLVMBuilderRef, BB: LLVMBasicBlockRef);
    pub fn LLVMInsertIntoBuilder(Builder: LLVMBuilderRef, Instr: LLVMValueRef);
    pub fn LLVMInsertIntoBuilderWithName(Builder: LLVMBuilderRef, Instr: LLVMValueRef, Name: *const ::libc::c_char);
    pub fn LLVMInstructionClone(Inst: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMInstructionEraseFromParent(Inst: LLVMValueRef);
    pub fn LLVMInstructionGetAllMetadataOtherThanDebugLoc(
        Instr: LLVMValueRef,
        NumEntries: *mut ::libc::size_t,
    ) -> *mut LLVMValueMetadataEntry;
    pub fn LLVMInstructionRemoveFromParent(Inst: LLVMValueRef);
    pub fn LLVMInt128Type() -> LLVMTypeRef;
    pub fn LLVMInt128TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMInt16Type() -> LLVMTypeRef;
    pub fn LLVMInt16TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMInt1Type() -> LLVMTypeRef;
    pub fn LLVMInt1TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMInt32Type() -> LLVMTypeRef;
    pub fn LLVMInt32TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMInt64Type() -> LLVMTypeRef;
    pub fn LLVMInt64TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMInt8Type() -> LLVMTypeRef;
    pub fn LLVMInt8TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMIntrinsicCopyOverloadedName2(
        Mod: LLVMModuleRef,
        ID: ::libc::c_uint,
        ParamTypes: *mut LLVMTypeRef,
        ParamCount: ::libc::size_t,
        NameLength: *mut ::libc::size_t,
    ) -> *const ::libc::c_char;
    pub fn LLVMIntrinsicGetName(ID: ::libc::c_uint, NameLength: *mut ::libc::size_t) -> *const ::libc::c_char;
    pub fn LLVMIntrinsicGetType(
        Ctx: LLVMContextRef,
        ID: ::libc::c_uint,
        ParamTypes: *mut LLVMTypeRef,
        ParamCount: ::libc::size_t,
    ) -> LLVMTypeRef;
    pub fn LLVMIntrinsicIsOverloaded(ID: ::libc::c_uint) -> LLVMBool;
    pub fn LLVMIntType(NumBits: ::libc::c_uint) -> LLVMTypeRef;
    pub fn LLVMIntTypeInContext(C: LLVMContextRef, NumBits: ::libc::c_uint) -> LLVMTypeRef;
    pub fn LLVMIsAAddrSpaceCastInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAAllocaInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAArgument(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAAtomicCmpXchgInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAAtomicRMWInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsABasicBlock(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsABinaryOperator(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsABitCastInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsABlockAddress(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsABranchInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsACallBrInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsACallInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsACastInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsACatchPadInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsACatchReturnInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsACatchSwitchInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsACleanupPadInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsACleanupReturnInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsACmpInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAConstant(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAConstantAggregateZero(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAConstantArray(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAConstantDataArray(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAConstantDataSequential(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAConstantDataVector(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAConstantExpr(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAConstantFP(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAConstantInt(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAConstantPointerNull(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAConstantStruct(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAConstantTokenNone(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAConstantVector(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsADbgDeclareInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsADbgInfoIntrinsic(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsADbgLabelInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsADbgVariableIntrinsic(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAExtractElementInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAExtractValueInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAFCmpInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAFenceInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAFPExtInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAFPToSIInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAFPToUIInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAFPTruncInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAFreezeInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAFuncletPadInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAFunction(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAGetElementPtrInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAGlobalAlias(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAGlobalIFunc(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAGlobalObject(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAGlobalValue(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAGlobalVariable(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAICmpInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAIndirectBrInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAInlineAsm(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAInsertElementInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAInsertValueInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAInstruction(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAIntrinsicInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAIntToPtrInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAInvokeInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsALandingPadInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsALoadInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAMDNode(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAMDString(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAMemCpyInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAMemIntrinsic(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAMemMoveInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAMemSetInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAPHINode(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAPoisonValue(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAPtrToIntInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAResumeInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAReturnInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsASelectInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsASExtInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAShuffleVectorInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsASIToFPInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAStoreInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsASwitchInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsATerminatorInst(Inst: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAtomicSingleThread(AtomicInst: LLVMValueRef) -> LLVMBool;
    pub fn LLVMIsATruncInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAUIToFPInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAUnaryInstruction(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAUnaryOperator(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAUndefValue(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAUnreachableInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAUser(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAVAArgInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsAZExtInst(Val: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMIsCleanup(LandingPad: LLVMValueRef) -> LLVMBool;
    pub fn LLVMIsConditional(Branch: LLVMValueRef) -> LLVMBool;
    pub fn LLVMIsConstant(Val: LLVMValueRef) -> LLVMBool;
    pub fn LLVMIsConstantString(c: LLVMValueRef) -> LLVMBool;
    pub fn LLVMIsDeclaration(Global: LLVMValueRef) -> LLVMBool;
    pub fn LLVMIsEnumAttribute(A: LLVMAttributeRef) -> LLVMBool;
    pub fn LLVMIsExternallyInitialized(GlobalVar: LLVMValueRef) -> LLVMBool;
    pub fn LLVMIsFunctionVarArg(FunctionTy: LLVMTypeRef) -> LLVMBool;
    pub fn LLVMIsGlobalConstant(GlobalVar: LLVMValueRef) -> LLVMBool;
    pub fn LLVMIsInBounds(GEP: LLVMValueRef) -> LLVMBool;
    pub fn LLVMIsLiteralStruct(StructTy: LLVMTypeRef) -> LLVMBool;
    pub fn LLVMIsMultithreaded() -> LLVMBool;
    pub fn LLVMIsNull(Val: LLVMValueRef) -> LLVMBool;
    pub fn LLVMIsOpaqueStruct(StructTy: LLVMTypeRef) -> LLVMBool;
    pub fn LLVMIsPackedStruct(StructTy: LLVMTypeRef) -> LLVMBool;
    pub fn LLVMIsPoison(Val: LLVMValueRef) -> LLVMBool;
    pub fn LLVMIsStringAttribute(A: LLVMAttributeRef) -> LLVMBool;
    pub fn LLVMIsTailCall(CallInst: LLVMValueRef) -> LLVMBool;
    pub fn LLVMIsThreadLocal(GlobalVar: LLVMValueRef) -> LLVMBool;
    pub fn LLVMIsTypeAttribute(A: LLVMAttributeRef) -> LLVMBool;
    pub fn LLVMIsUndef(Val: LLVMValueRef) -> LLVMBool;
    pub fn LLVMLabelType() -> LLVMTypeRef;
    pub fn LLVMLabelTypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMLookupIntrinsicID(Name: *const ::libc::c_char, NameLen: ::libc::size_t) -> ::libc::c_uint;
    pub fn LLVMMDNodeInContext2(C: LLVMContextRef, MDs: *mut LLVMMetadataRef, Count: ::libc::size_t)
        -> LLVMMetadataRef;
    pub fn LLVMMDStringInContext2(
        C: LLVMContextRef,
        Str: *const ::libc::c_char,
        SLen: ::libc::size_t,
    ) -> LLVMMetadataRef;
    pub fn LLVMMetadataAsValue(C: LLVMContextRef, MD: LLVMMetadataRef) -> LLVMValueRef;
    pub fn LLVMMetadataTypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMModuleCreateWithName(ModuleID: *const ::libc::c_char) -> LLVMModuleRef;
    pub fn LLVMModuleCreateWithNameInContext(ModuleID: *const ::libc::c_char, C: LLVMContextRef) -> LLVMModuleRef;
    pub fn LLVMModuleFlagEntriesGetFlagBehavior(
        Entries: *mut LLVMModuleFlagEntry,
        Index: ::libc::c_uint,
    ) -> LLVMModuleFlagBehavior;
    pub fn LLVMModuleFlagEntriesGetKey(
        Entries: *mut LLVMModuleFlagEntry,
        Index: ::libc::c_uint,
        Len: *mut ::libc::size_t,
    ) -> *const ::libc::c_char;
    pub fn LLVMModuleFlagEntriesGetMetadata(
        Entries: *mut LLVMModuleFlagEntry,
        Index: ::libc::c_uint,
    ) -> LLVMMetadataRef;
    pub fn LLVMMoveBasicBlockAfter(BB: LLVMBasicBlockRef, MovePos: LLVMBasicBlockRef);
    pub fn LLVMMoveBasicBlockBefore(BB: LLVMBasicBlockRef, MovePos: LLVMBasicBlockRef);
    pub fn LLVMPointerType(ElementType: LLVMTypeRef, AddressSpace: ::libc::c_uint) -> LLVMTypeRef;
    pub fn LLVMPointerTypeInContext(C: LLVMContextRef, AddressSpace: ::libc::c_uint) -> LLVMTypeRef;
    pub fn LLVMPointerTypeIsOpaque(Ty: LLVMTypeRef) -> LLVMBool;
    pub fn LLVMPositionBuilder(Builder: LLVMBuilderRef, Block: LLVMBasicBlockRef, Instr: LLVMValueRef);
    pub fn LLVMPositionBuilderAtEnd(Builder: LLVMBuilderRef, Block: LLVMBasicBlockRef);
    pub fn LLVMPositionBuilderBefore(Builder: LLVMBuilderRef, Instr: LLVMValueRef);
    pub fn LLVMPPCFP128Type() -> LLVMTypeRef;
    pub fn LLVMPPCFP128TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMPrintModuleToFile(
        M: LLVMModuleRef,
        Filename: *const ::libc::c_char,
        ErrorMessage: *mut *mut ::libc::c_char,
    ) -> LLVMBool;
    pub fn LLVMPrintModuleToString(M: LLVMModuleRef) -> *mut ::libc::c_char;
    pub fn LLVMPrintTypeToString(Val: LLVMTypeRef) -> *mut ::libc::c_char;
    pub fn LLVMPrintValueToString(Val: LLVMValueRef) -> *mut ::libc::c_char;
    pub fn LLVMRemoveBasicBlockFromParent(BB: LLVMBasicBlockRef);
    pub fn LLVMRemoveCallSiteEnumAttribute(C: LLVMValueRef, Idx: LLVMAttributeIndex, KindID: ::libc::c_uint);
    pub fn LLVMRemoveCallSiteStringAttribute(
        C: LLVMValueRef,
        Idx: LLVMAttributeIndex,
        K: *const ::libc::c_char,
        KLen: ::libc::c_uint,
    );
    pub fn LLVMRemoveEnumAttributeAtIndex(F: LLVMValueRef, Idx: LLVMAttributeIndex, KindID: ::libc::c_uint);
    pub fn LLVMRemoveGlobalIFunc(IFunc: LLVMValueRef);
    pub fn LLVMRemoveStringAttributeAtIndex(
        F: LLVMValueRef,
        Idx: LLVMAttributeIndex,
        K: *const ::libc::c_char,
        KLen: ::libc::c_uint,
    );
    pub fn LLVMReplaceAllUsesWith(OldVal: LLVMValueRef, NewVal: LLVMValueRef);
    pub fn LLVMRunFunctionPassManager(FPM: LLVMPassManagerRef, F: LLVMValueRef) -> LLVMBool;
    pub fn LLVMRunPassManager(PM: LLVMPassManagerRef, M: LLVMModuleRef) -> LLVMBool;
    pub fn LLVMScalableVectorType(ElementType: LLVMTypeRef, ElementCount: ::libc::c_uint) -> LLVMTypeRef;
    pub fn LLVMSetAlignment(V: LLVMValueRef, Bytes: ::libc::c_uint);
    pub fn LLVMSetArgOperand(Funclet: LLVMValueRef, i: ::libc::c_uint, value: LLVMValueRef);
    pub fn LLVMSetAtomicRMWBinOp(AtomicRMWInst: LLVMValueRef, BinOp: LLVMAtomicRMWBinOp);
    pub fn LLVMSetAtomicSingleThread(AtomicInst: LLVMValueRef, SingleThread: LLVMBool);
    pub fn LLVMSetCleanup(LandingPad: LLVMValueRef, Val: LLVMBool);
    pub fn LLVMSetCmpXchgFailureOrdering(CmpXchgInst: LLVMValueRef, Ordering: LLVMAtomicOrdering);
    pub fn LLVMSetCmpXchgSuccessOrdering(CmpXchgInst: LLVMValueRef, Ordering: LLVMAtomicOrdering);
    pub fn LLVMSetCondition(Branch: LLVMValueRef, Cond: LLVMValueRef);
    pub fn LLVMSetCurrentDebugLocation2(Builder: LLVMBuilderRef, Loc: LLVMMetadataRef);
    pub fn LLVMSetDataLayout(M: LLVMModuleRef, DataLayoutStr: *const ::libc::c_char);
    pub fn LLVMSetDLLStorageClass(Global: LLVMValueRef, Class: LLVMDLLStorageClass);
    pub fn LLVMSetExternallyInitialized(GlobalVar: LLVMValueRef, IsExtInit: LLVMBool);
    pub fn LLVMSetFunctionCallConv(Fn: LLVMValueRef, CC: ::libc::c_uint);
    pub fn LLVMSetGC(Fn: LLVMValueRef, Name: *const ::libc::c_char);
    pub fn LLVMSetGlobalConstant(GlobalVar: LLVMValueRef, IsConstant: LLVMBool);
    pub fn LLVMSetGlobalIFuncResolver(IFunc: LLVMValueRef, Resolver: LLVMValueRef);
    pub fn LLVMSetInitializer(GlobalVar: LLVMValueRef, ConstantVal: LLVMValueRef);
    pub fn LLVMSetInstrParamAlignment(Instr: LLVMValueRef, Idx: LLVMAttributeIndex, Align: ::libc::c_uint);
    pub fn LLVMSetInstructionCallConv(Instr: LLVMValueRef, CC: ::libc::c_uint);
    pub fn LLVMSetIsInBounds(GEP: LLVMValueRef, InBounds: LLVMBool);
    pub fn LLVMSetLinkage(Global: LLVMValueRef, Linkage: LLVMLinkage);
    pub fn LLVMSetMetadata(Val: LLVMValueRef, KindID: ::libc::c_uint, Node: LLVMValueRef);
    pub fn LLVMSetModuleIdentifier(M: LLVMModuleRef, Ident: *const ::libc::c_char, Len: ::libc::size_t);
    pub fn LLVMSetModuleInlineAsm2(M: LLVMModuleRef, Asm: *const ::libc::c_char, Len: ::libc::size_t);
    pub fn LLVMSetNormalDest(InvokeInst: LLVMValueRef, B: LLVMBasicBlockRef);
    pub fn LLVMSetOperand(User: LLVMValueRef, Index: ::libc::c_uint, Val: LLVMValueRef);
    pub fn LLVMSetOrdering(MemoryAccessInst: LLVMValueRef, Ordering: LLVMAtomicOrdering);
    pub fn LLVMSetParamAlignment(Arg: LLVMValueRef, Align: ::libc::c_uint);
    pub fn LLVMSetParentCatchSwitch(CatchPad: LLVMValueRef, CatchSwitch: LLVMValueRef);
    pub fn LLVMSetPersonalityFn(Fn: LLVMValueRef, PersonalityFn: LLVMValueRef);
    pub fn LLVMSetSection(Global: LLVMValueRef, Section: *const ::libc::c_char);
    pub fn LLVMSetSourceFileName(M: LLVMModuleRef, Name: *const ::libc::c_char, Len: ::libc::size_t);
    pub fn LLVMSetSuccessor(Term: LLVMValueRef, i: ::libc::c_uint, block: LLVMBasicBlockRef);
    pub fn LLVMSetTailCall(CallInst: LLVMValueRef, IsTailCall: LLVMBool);
    pub fn LLVMSetTarget(M: LLVMModuleRef, Triple: *const ::libc::c_char);
    pub fn LLVMSetThreadLocal(GlobalVar: LLVMValueRef, IsThreadLocal: LLVMBool);
    pub fn LLVMSetThreadLocalMode(GlobalVar: LLVMValueRef, Mode: LLVMThreadLocalMode);
    pub fn LLVMSetUnnamedAddress(Global: LLVMValueRef, UnnamedAddr: LLVMUnnamedAddr);
    pub fn LLVMSetUnwindDest(InvokeInst: LLVMValueRef, B: LLVMBasicBlockRef);
    pub fn LLVMSetValueName2(Val: LLVMValueRef, Name: *const ::libc::c_char, NameLen: ::libc::size_t);
    pub fn LLVMSetVisibility(Global: LLVMValueRef, Viz: LLVMVisibility);
    pub fn LLVMSetVolatile(MemoryAccessInst: LLVMValueRef, IsVolatile: LLVMBool);
    pub fn LLVMSetWeak(CmpXchgInst: LLVMValueRef, IsWeak: LLVMBool);
    pub fn LLVMShutdown();
    pub fn LLVMSizeOf(Ty: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMStartMultithreaded() -> LLVMBool;
    pub fn LLVMStopMultithreaded();
    pub fn LLVMStructCreateNamed(C: LLVMContextRef, Name: *const ::libc::c_char) -> LLVMTypeRef;
    pub fn LLVMStructGetTypeAtIndex(StructTy: LLVMTypeRef, i: ::libc::c_uint) -> LLVMTypeRef;
    pub fn LLVMStructSetBody(
        StructTy: LLVMTypeRef,
        ElementTypes: *mut LLVMTypeRef,
        ElementCount: ::libc::c_uint,
        Packed: LLVMBool,
    );
    pub fn LLVMStructType(
        ElementTypes: *mut LLVMTypeRef,
        ElementCount: ::libc::c_uint,
        Packed: LLVMBool,
    ) -> LLVMTypeRef;
    pub fn LLVMStructTypeInContext(
        C: LLVMContextRef,
        ElementTypes: *mut LLVMTypeRef,
        ElementCount: ::libc::c_uint,
        Packed: LLVMBool,
    ) -> LLVMTypeRef;
    pub fn LLVMTargetExtTypeInContext(
        C: LLVMContextRef,
        Name: *const ::libc::c_char,
        TypeParams: *mut LLVMTypeRef,
        TypeParamCount: ::libc::c_uint,
        IntParams: *mut ::libc::c_uint,
        IntParamCount: ::libc::c_uint,
    ) -> LLVMTypeRef;
    pub fn LLVMTokenTypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMTypeIsSized(Ty: LLVMTypeRef) -> LLVMBool;
    pub fn LLVMTypeOf(Val: LLVMValueRef) -> LLVMTypeRef;
    pub fn LLVMValueAsBasicBlock(Val: LLVMValueRef) -> LLVMBasicBlockRef;
    pub fn LLVMValueAsMetadata(Val: LLVMValueRef) -> LLVMMetadataRef;
    pub fn LLVMValueIsBasicBlock(Val: LLVMValueRef) -> LLVMBool;
    pub fn LLVMValueMetadataEntriesGetKind(
        Entries: *mut LLVMValueMetadataEntry,
        Index: ::libc::c_uint,
    ) -> ::libc::c_uint;
    pub fn LLVMValueMetadataEntriesGetMetadata(
        Entries: *mut LLVMValueMetadataEntry,
        Index: ::libc::c_uint,
    ) -> LLVMMetadataRef;
    pub fn LLVMVectorType(ElementType: LLVMTypeRef, ElementCount: ::libc::c_uint) -> LLVMTypeRef;
    pub fn LLVMVoidType() -> LLVMTypeRef;
    pub fn LLVMVoidTypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMX86AMXType() -> LLVMTypeRef;
    pub fn LLVMX86AMXTypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMX86FP80Type() -> LLVMTypeRef;
    pub fn LLVMX86FP80TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMX86MMXType() -> LLVMTypeRef;
    pub fn LLVMX86MMXTypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
}
