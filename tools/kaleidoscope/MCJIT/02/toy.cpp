#define MINIMAL_STDERR_OUTPUT

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
cl::opt<std::string> InputIR(
    "input-IR",
    cl::desc("Specify the name of an IR file to load for function definitions"),
    cl::value_desc("input IR file name"));

cl::opt<bool> UseObjectCache("use-object-cache",
                             cl::desc("Enable use of the MCJIT object caching"),
                             cl::init(false));
} // namespace

std::string GenerateUniqueName(const char *root) {
  static int i = 0;
  char s[16];
  sprintf(s, "%s%d", root, i++);
  std::string S = s;
  return S;
}

std::string MakeLegalFunctionName(std::string Name) {
  std::string NewName;
  if (!Name.length())
    return GenerateUniqueName("anon_func_");
  NewName = Name;
  if (NewName.find_first_of("0123456789") == 0) {
    NewName.insert(0, 1, 'n');
  }

  std::string legal_elements =
      "_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  size_t pos;
  while ((pos = NewName.find_first_not_of(legal_elements)) !=
         std::string::npos) {
    char old_c = NewName.at(pos);
    char new_str[16];
    sprintf(new_str, "%d", (int)old_c);
    NewName = NewName.replace(pos, 1, new_str);
  }
  return NewName;
}

class MCJITObjectCache : public ObjectCache {
public:
  MCJITObjectCache() {
    // Set IR cache directory
    sys::fs::current_path(CacheDir);
    sys::path::append(CacheDir, "toy_object_cache");
  }

  virtual ~MCJITObjectCache() {}

  virtual void notifyObjectCompiled(const Module *M, const MemoryBuffer *Obj) {
    // Get the ModuleID
    const std::string ModuleID = M->getModuleIdentifier();

    // If we've flagged this as an IR file, cache it
    if (0 == ModuleID.compare(0, 3, "IR:")) {
      std::string IRFileName = ModuleID.substr(3);
      SmallString<128> IRCacheFile = CacheDir;
      sys::path::append(IRCacheFile, IRFileName);
      if (!sys::fs::exists(CacheDir.str()) &&
          sys::fs::create_directory(CacheDir.str())) {
        fprintf(stderr, "Unable to create cache directory\n");
        return;
      }
      std::string ErrStr;
      raw_fd_ostream IRObjectFile(IRCacheFile.c_str(), ErrStr,
                                  raw_fd_ostream::F_Binary);
      IRObjectFile << Obj->getBuffer();
    }
  }

  // MCJIT will call this function before compiling any module
  // MCJIT takes ownership of both the MemoryBuffer object and the memory
  // to which it refers.
  virtual MemoryBuffer *getObject(const Module *M) {
    // Get the ModuleID
    const std::string ModuleID = M->getModuleIdentifier();

    // If we've flagged this as an IR file, cache it
    if (0 == ModuleID.compare(0, 3, "IR:")) {
      std::string IRFileName = ModuleID.substr(3);
      SmallString<128> IRCacheFile = CacheDir;
      sys::path::append(IRCacheFile, IRFileName);
      if (!sys::fs::exists(IRCacheFile.str())) {
        // This file isn't in our cache
        return NULL;
      }
      std::unique_ptr<MemoryBuffer> IRObjectBuffer;
      MemoryBuffer::getFile(IRCacheFile.c_str(), IRObjectBuffer, -1, false);
      // MCJIT will want to write into this buffer, and we don't want that
      // because the file has probably just been mmapped.  Instead we make
      // a copy.  The filed-based buffer will be released when it goes
      // out of scope.
      return MemoryBuffer::getMemBufferCopy(IRObjectBuffer->getBuffer());
    }

    return NULL;
  }

private:
  SmallString<128> CacheDir;
};

class MCJITHelper {
public:
  MCJITHelper(LLVMContext &C) : Context(C), OpenModule(NULL) {}
  ~MCJITHelper();

  Function *getFunction(const std::string FnName);
  Module *getModuleForNewFunction();
  void *getPointerToFunction(Function *F);
  void *getPointerToNamedFunction(const std::string &Name);
  ExecutionEngine *compileModule(Module *M);
  void closeCurrentModule();
  void addModule(Module *M);
  void dump();

private:
  typedef std::vector<Module *> ModuleVector;

  LLVMContext &Context;
  Module *OpenModule;
  ModuleVector Modules;
  std::map<Module *, ExecutionEngine *> EngineMap;
  MCJITObjectCache OurObjectCache;
};

class HelpingMemoryManager : public SectionMemoryManager {
  HelpingMemoryManager(const HelpingMemoryManager &) = delete;
  void operator=(const HelpingMemoryManager &) = delete;

public:
  HelpingMemoryManager(MCJITHelper *Helper) : MasterHelper(Helper) {}
  virtual ~HelpingMemoryManager() {}

  /// This method returns the address of the specified function.
  /// Our implementation will attempt to find functions in other
  /// modules associated with the MCJITHelper to cross link functions
  /// from one generated module to another.
  ///
  /// If \p AbortOnFailure is false and no function with the given name is
  /// found, this function returns a null pointer. Otherwise, it prints a
  /// message to stderr and aborts.
  virtual void *getPointerToNamedFunction(const std::string &Name,
                                          bool AbortOnFailure = true);

private:
  MCJITHelper *MasterHelper;
};

void *HelpingMemoryManager::getPointerToNamedFunction(const std::string &Name,
                                                      bool AbortOnFailure) {
  // Try the standard symbol resolution first, but ask it not to abort.
  void *pfn = SectionMemoryManager::getPointerToNamedFunction(Name, false);
  if (pfn)
    return pfn;

  pfn = MasterHelper->getPointerToNamedFunction(Name);
  if (!pfn && AbortOnFailure)
    report_fatal_error("Program used external function '" + Name +
                       "' which could not be resolved!");
  return pfn;
}

MCJITHelper::~MCJITHelper() {
  // Walk the vector of modules.
  ModuleVector::iterator it, end;
  for (it = Modules.begin(), end = Modules.end(); it != end; ++it) {
    // See if we have an execution engine for this module.
    std::map<Module *, ExecutionEngine *>::iterator mapIt = EngineMap.find(*it);
    // If we have an EE, the EE owns the module so just delete the EE.
    if (mapIt != EngineMap.end()) {
      delete mapIt->second;
    } else {
      // Otherwise, we still own the module.  Delete it now.
      delete *it;
    }
  }
}

Function *MCJITHelper::getFunction(const std::string FnName) {
  ModuleVector::iterator begin = Modules.begin();
  ModuleVector::iterator end = Modules.end();
  ModuleVector::iterator it;
  for (it = begin; it != end; ++it) {
    Function *F = (*it)->getFunction(FnName);
    if (F) {
      if (*it == OpenModule)
        return F;

      assert(OpenModule != NULL);

      // This function is in a module that has already been JITed.
      // We need to generate a new prototype for external linkage.
      Function *PF = OpenModule->getFunction(FnName);
      if (PF && !PF->empty()) {
        ErrorF("redefinition of function across modules");
        return 0;
      }

      // If we don't have a prototype yet, create one.
      if (!PF)
        PF = Function::Create(F->getFunctionType(), Function::ExternalLinkage,
                              FnName, OpenModule);
      return PF;
    }
  }
  return NULL;
}

Module *MCJITHelper::getModuleForNewFunction() {
  // If we have a Module that hasn't been JITed, use that.
  if (OpenModule)
    return OpenModule;

  // Otherwise create a new Module.
  std::string ModName = GenerateUniqueName("mcjit_module_");
  Module *M = new Module(ModName, Context);
  Modules.push_back(M);
  OpenModule = M;
  return M;
}

void *MCJITHelper::getPointerToFunction(Function *F) {
  // Look for this function in an existing module
  ModuleVector::iterator begin = Modules.begin();
  ModuleVector::iterator end = Modules.end();
  ModuleVector::iterator it;
  std::string FnName = F->getName();
  for (it = begin; it != end; ++it) {
    Function *MF = (*it)->getFunction(FnName);
    if (MF == F) {
      std::map<Module *, ExecutionEngine *>::iterator eeIt =
          EngineMap.find(*it);
      if (eeIt != EngineMap.end()) {
        void *P = eeIt->second->getPointerToFunction(F);
        if (P)
          return P;
      } else {
        ExecutionEngine *EE = compileModule(*it);
        void *P = EE->getPointerToFunction(F);
        if (P)
          return P;
      }
    }
  }
  return NULL;
}

void MCJITHelper::closeCurrentModule() { OpenModule = NULL; }

ExecutionEngine *MCJITHelper::compileModule(Module *M) {
  if (M == OpenModule)
    closeCurrentModule();

  std::string ErrStr;
  ExecutionEngine *NewEngine =
      EngineBuilder(M)
          .setErrorStr(&ErrStr)
          .setMCJITMemoryManager(new HelpingMemoryManager(this))
          .create();
  if (!NewEngine) {
    fprintf(stderr, "Could not create ExecutionEngine: %s\n", ErrStr.c_str());
    exit(1);
  }

  if (UseObjectCache)
    NewEngine->setObjectCache(&OurObjectCache);

  // Get the ModuleID so we can identify IR input files
  const std::string ModuleID = M->getModuleIdentifier();

  // If we've flagged this as an IR file, it doesn't need function passes run.
  if (0 != ModuleID.compare(0, 3, "IR:")) {
    // Create a function pass manager for this engine
    FunctionPassManager *FPM = new FunctionPassManager(M);

    // Set up the optimizer pipeline.  Start with registering info about how the
    // target lays out data structures.
    FPM->add(new DataLayout(*NewEngine->getDataLayout()));
    // Provide basic AliasAnalysis support for GVN.
    FPM->add(createBasicAliasAnalysisPass());
    // Promote allocas to registers.
    FPM->add(createPromoteMemoryToRegisterPass());
    // Do simple "peephole" optimizations and bit-twiddling optzns.
    FPM->add(createInstructionCombiningPass());
    // Reassociate expressions.
    FPM->add(createReassociatePass());
    // Eliminate Common SubExpressions.
    FPM->add(createGVNPass());
    // Simplify the control flow graph (deleting unreachable blocks, etc).
    FPM->add(createCFGSimplificationPass());
    FPM->doInitialization();

    // For each function in the module
    Module::iterator it;
    Module::iterator end = M->end();
    for (it = M->begin(); it != end; ++it) {
      // Run the FPM on this function
      FPM->run(*it);
    }

    // We don't need this anymore
    delete FPM;
  }

  // Store this engine
  EngineMap[M] = NewEngine;
  NewEngine->finalizeObject();

  return NewEngine;
}

void *MCJITHelper::getPointerToNamedFunction(const std::string &Name) {
  // Look for the functions in our modules, compiling only as necessary
  ModuleVector::iterator begin = Modules.begin();
  ModuleVector::iterator end = Modules.end();
  ModuleVector::iterator it;
  for (it = begin; it != end; ++it) {
    Function *F = (*it)->getFunction(Name);
    if (F && !F->empty()) {
      std::map<Module *, ExecutionEngine *>::iterator eeIt =
          EngineMap.find(*it);
      if (eeIt != EngineMap.end()) {
        void *P = eeIt->second->getPointerToFunction(F);
        if (P)
          return P;
      } else {
        ExecutionEngine *EE = compileModule(*it);
        void *P = EE->getPointerToFunction(F);
        if (P)
          return P;
      }
    }
  }
  return NULL;
}

void MCJITHelper::addModule(Module *M) { Modules.push_back(M); }

void MCJITHelper::dump() {
  ModuleVector::iterator begin = Modules.begin();
  ModuleVector::iterator end = Modules.end();
  ModuleVector::iterator it;
  for (it = begin; it != end; ++it)
    (*it)->dump();
}

static MCJITHelper *TheHelper;
static LLVMContext TheContext;
static IRBuilder<> Builder(TheContext);
static std::map<std::string, AllocaInst *> NamedValues;

Value *ErrorV(const char *Str) {
  Error(Str);
  return 0;
}

Value *UnaryExprAST::Codegen() {
  Value *OperandV = Operand->Codegen();
  if (OperandV == 0)
    return 0;
#ifdef USE_MCJIT
  Function *F = TheHelper->getFunction(
      MakeLegalFunctionName(std::string("unary") + Opcode));
#else
  Function *F = TheModule->getFunction(std::string("unary") + Opcode);
#endif
  if (F == 0)
    return ErrorV("Unknown unary operator");
  return Builder.CreateCall(F, OperandV, "unop");
}

Value *BinaryExprAST::Codegen() {
  if (Op == '=') {
    VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS);
    if (!LHSE)
      return ErrorV("destination of '=' must be a variable");
    Value *Val = RHS->Codegen();
    if (Val == 0)
      return 0;
    Value *Variable = NamedValues[LHSE->getName()];
    if (Variable == 0)
      return ErrorV("Unknown variable name");
    Builder.CreateStore(Val, Variable);
    return Val;
  }
  Value *L = LHS->Codegen();
  Value *R = RHS->Codegen();
  if (L == 0 || R == 0)
    return 0;
  switch (Op) {
  case '+':
    return Builder.CreateFAdd(L, R, "addtmp");
  case '-':
    return Builder.CreateFSub(L, R, "subtmp");
  case '*':
    return Builder.CreateFMul(L, R, "multmp");
  case '/':
    return Builder.CreateFDiv(L, R, "divtmp");
  case '<':
    L = Builder.CreateFCmpULT(L, R, "cmptmp");
    return Builder.CreateUIToFP(L, Type::getDoubleTy(TheContext), "booltmp");
  default:
    break;
  }
  Function *F =
      TheHelper->getFunction(MakeLegalFunctionName(std::string("binary") + Op));
  assert(F && "binary operator not found!");
  Value *Ops[] = {L, R};
  return Builder.CreateCall(F, Ops, "binop");
}

Value *CallExprAST::Codegen() {
  Function *CalleeF = TheHelper->getFunction(Callee);
  if (CalleeF == 0)
    return ErrorV("Unknown function referenced");
  if (CalleeF->arg_size() != Args.size())
    return ErrorV("Incorrect # arguments passed");
  std::vector<Value *> ArgsV;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    ArgsV.push_back(Args[i]->Codegen());
    if (ArgsV.back() == 0)
      return 0;
  }
  return Builder.CreateCall(CalleeF, ArgsV, "calltmp");
}

Value *IfExprAST::Codegen() {
  Value *CondV = Cond->Codegen();
  if (CondV == 0)
    return 0;
  CondV = Builder.CreateFCmpONE(
      CondV, ConstantFP::get(TheContext, APFloat(0.0)), "ifcond");
  Function *TheFunction = Builder.GetInsertBlock()->getParent();
  BasicBlock *ThenBB = BasicBlock::Create(TheContext, "then", TheFunction);
  BasicBlock *ElseBB = BasicBlock::Create(TheContext, "else");
  BasicBlock *MergeBB = BasicBlock::Create(TheContext, "ifcont");
  Builder.CreateCondBr(CondV, ThenBB, ElseBB);
  Builder.SetInsertPoint(ThenBB);
  Value *ThenV = Then->Codegen();
  if (ThenV == 0)
    return 0;
  Builder.CreateBr(MergeBB);
  ThenBB = Builder.GetInsertBlock();
  TheFunction->insert(TheFunction->end(), ElseBB);
  Builder.SetInsertPoint(ElseBB);
  Value *ElseV = Else->Codegen();
  if (ElseV == 0)
    return 0;
  Builder.CreateBr(MergeBB);
  ElseBB = Builder.GetInsertBlock();
  TheFunction->insert(TheFunction->end(), MergeBB);
  Builder.SetInsertPoint(MergeBB);
  PHINode *PN = Builder.CreatePHI(Type::getDoubleTy(TheContext), 2, "iftmp");
  PN->addIncoming(ThenV, ThenBB);
  PN->addIncoming(ElseV, ElseBB);
  return PN;
}

Value *ForExprAST::Codegen() {
  // Output this as:
  //   var = alloca double
  //   ...
  //   start = startexpr
  //   store start -> var
  //   goto loop
  // loop:
  //   ...
  //   bodyexpr
  //   ...
  // loopend:
  //   step = stepexpr
  //   endcond = endexpr
  //
  //   curvar = load var
  //   nextvar = curvar + step
  //   store nextvar -> var
  //   br endcond, loop, endloop
  // outloop:
  Function *TheFunction = Builder.GetInsertBlock()->getParent();
  AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);
  Value *StartVal = Start->Codegen();
  if (StartVal == 0)
    return 0;
  Builder.CreateStore(StartVal, Alloca);
  BasicBlock *LoopBB = BasicBlock::Create(TheContext, "loop", TheFunction);
  Builder.CreateBr(LoopBB);
  Builder.SetInsertPoint(LoopBB);
  AllocaInst *OldVal = NamedValues[VarName];
  NamedValues[VarName] = Alloca;
  if (Body->Codegen() == 0)
    return 0;
  Value *StepVal;
  if (Step) {
    StepVal = Step->Codegen();
    if (StepVal == 0)
      return 0;
  } else {
    StepVal = ConstantFP::get(TheContext, APFloat(1.0));
  }
  Value *EndCond = End->Codegen();
  if (EndCond == 0)
    return EndCond;
  Value *CurVar = Builder.CreateLoad(Alloca, VarName.c_str());
  Value *NextVar = Builder.CreateFAdd(CurVar, StepVal, "nextvar");
  Builder.CreateStore(NextVar, Alloca);
  EndCond = Builder.CreateFCmpONE(
      EndCond, ConstantFP::get(TheContext, APFloat(0.0)), "loopcond");
  BasicBlock *AfterBB =
      BasicBlock::Create(TheContext, "afterloop", TheFunction);
  Builder.CreateCondBr(EndCond, LoopBB, AfterBB);
  Builder.SetInsertPoint(AfterBB);
  if (OldVal)
    NamedValues[VarName] = OldVal;
  else
    NamedValues.erase(VarName);
  return Constant::getNullValue(Type::getDoubleTy(TheContext));
}

Value *VarExprAST::Codegen() {
  std::vector<AllocaInst *> OldBindings;
  Function *TheFunction = Builder.GetInsertBlock()->getParent();
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second;
    Value *InitVal;
    if (Init) {
      InitVal = Init->Codegen();
      if (InitVal == 0)
        return 0;
    } else {
      InitVal = ConstantFP::get(TheContext, APFloat(0.0));
    }
    AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);
    Builder.CreateStore(InitVal, Alloca);
    OldBindings.push_back(NamedValues[VarName]);
    NamedValues[VarName] = Alloca;
  }
  Value *BodyVal = Body->Codegen();
  if (BodyVal == 0)
    return 0;
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
    NamedValues[VarNames[i].first] = OldBindings[i];
  return BodyVal;
}

Function *PrototypeAST::Codegen() {
  std::vector<Type *> Doubles(Args.size(), Type::getDoubleTy(TheContext));
  FunctionType *FT =
      FunctionType::get(Type::getDoubleTy(TheContext), Doubles, false);
  std::string FnName = MakeLegalFunctionName(Name);
  Module *M = TheHelper->getModuleForNewFunction();
  Function *F = Function::Create(FT, Function::ExternalLinkage, FnName, M);
  if (F->getName() != FnName) {
    F->eraseFromParent();
    F = M->getFunction(Name);
    if (!F->empty()) {
      ErrorF("redefinition of function");
      return 0;
    }
    if (F->arg_size() != Args.size()) {
      ErrorF("redefinition of function with different # args");
      return 0;
    }
  }
  unsigned Idx = 0;
  for (Function::arg_iterator AI = F->arg_begin(); Idx != Args.size();
       ++AI, ++Idx)
    AI->setName(Args[Idx]);
  return F;
}

void PrototypeAST::CreateArgumentAllocas(Function *F) {
  Function::arg_iterator AI = F->arg_begin();
  for (unsigned Idx = 0, e = Args.size(); Idx != e; ++Idx, ++AI) {
    AllocaInst *Alloca = CreateEntryBlockAlloca(F, Args[Idx]);
    Builder.CreateStore(AI, Alloca);
    NamedValues[Args[Idx]] = Alloca;
  }
}

Function *FunctionAST::Codegen() {
  NamedValues.clear();
  Function *TheFunction = Proto->Codegen();
  if (TheFunction == 0)
    return 0;
  if (Proto->isBinaryOp())
    BinopPrecedence[Proto->getOperatorName()] = Proto->getBinaryPrecedence();
  BasicBlock *BB = BasicBlock::Create(TheContext, "entry", TheFunction);
  Builder.SetInsertPoint(BB);
  Proto->CreateArgumentAllocas(TheFunction);
  if (Value *RetVal = Body->Codegen()) {
    Builder.CreateRet(RetVal);
    verifyFunction(*TheFunction);
    // TheFPM->run(*TheFunction);
    return TheFunction;
  }
  TheFunction->eraseFromParent();
  if (Proto->isBinaryOp())
    BinopPrecedence.erase(Proto->getOperatorName());
  return 0;
}

static void HandleDefinition() {
  if (FunctionAST *F = ParseDefinition()) {
    TheHelper->closeCurrentModule();
    if (Function *LF = F->Codegen()) {
#ifndef MINIMAL_STDERR_OUTPUT
      fprintf(stderr, "Read function definition:");
      LF->print(errs());
      fprintf(stderr, "\n");
#endif
    }
  } else {
    getNextToken();
  }
}

static void HandleExtern() {
  if (PrototypeAST *P = ParseExtern()) {
    if (Function *F = P->Codegen()) {
#ifndef MINIMAL_STDERR_OUTPUT
      fprintf(stderr, "Read extern: ");
      F->print(errs());
      fprintf(stderr, "\n");
#endif
    }
  } else {
    getNextToken();
  }
}

static void HandleTopLevelExpression() {
  if (FunctionAST *F = ParseTopLevelExpr()) {
    if (Function *LF = F->Codegen()) {
      void *FPtr = TheHelper->getPointerToFunction(LF);
      double (*FP)() = (double (*)())(intptr_t)FPtr;
#ifdef MINIMAL_STDERR_OUTPUT
      FP();
#else
      fprintf(stderr, "Evaluated to %f\n", FP());
#endif
    }
  } else {
    getNextToken();
  }
}

Module *parseInputIR(std::string InputFile) {
  SMDiagnostic Err;
  Module *M = ParseIRFile(InputFile, Err, TheContext);
  if (!M) {
    Err.print("IR parsing failed: ", errs());
    return NULL;
  }
  char ModID[256];
  sprintf(ModID, "IR:%s", InputFile.c_str());
  M->setModuleIdentifier(ModID);
  TheHelper->addModule(M);
  return M;
}

int main(int argc, char **argv) {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();
  LLVMContext &Context = TheContext;

  cl::ParseCommandLineOptions(argc, argv, "Kaleidoscope example program\n");

  BinopPrecedence['='] = 2;
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['/'] = 40;
  BinopPrecedence['*'] = 40;

#ifndef MINIMAL_STDERR_OUTPUT
  fprintf(stderr, "ready> ");
#endif
  getNextToken();

  TheHelper = new MCJITHelper(Context);

  if (!InputIR.empty()) {
    parseInputIR(InputIR);
  }

  MainLoop();

#ifndef MINIMAL_STDERR_OUTPUT
  TheHelper->print(errs());
#endif

  return 0;
}
