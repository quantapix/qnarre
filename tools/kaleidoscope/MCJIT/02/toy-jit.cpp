#define MINIMAL_STDERR_OUTPUT

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static Module *TheModule;
static FunctionPassManager *TheFPM;
static LLVMContext TheContext;
static IRBuilder<> Builder(TheContext);
static std::map<std::string, AllocaInst *> NamedValues;

static ExecutionEngine *TheExecutionEngine;

static void HandleTopLevelExpression() {
  if (FunctionAST *F = ParseTopLevelExpr()) {
    if (Function *LF = F->Codegen()) {
      void *FPtr = TheExecutionEngine->getPointerToFunction(LF);
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

int main(int argc, char **argv) {
  InitializeNativeTarget();
  LLVMContext &Context = TheContext;

  cl::ParseCommandLineOptions(argc, argv, "Kaleidoscope example program\n");

  BinopPrecedence['='] = 2;
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['/'] = 40;
  BinopPrecedence['*'] = 40; // highest.

  if (!InputIR.empty()) {
    TheModule = parseInputIR(InputIR);
  } else {
    TheModule = new Module("my cool jit", Context);
  }

  std::string ErrStr;
  TheExecutionEngine = EngineBuilder(TheModule).setErrorStr(&ErrStr).create();
  if (!TheExecutionEngine) {
    fprintf(stderr, "Could not create ExecutionEngine: %s\n", ErrStr.c_str());
    exit(1);
  }
  FunctionPassManager OurFPM(TheModule);
  OurFPM.add(new DataLayout(*TheExecutionEngine->getDataLayout()));
  OurFPM.add(createBasicAliasAnalysisPass());
  OurFPM.add(createPromoteMemoryToRegisterPass());
  OurFPM.add(createInstructionCombiningPass());
  OurFPM.add(createReassociatePass());
  OurFPM.add(createGVNPass());
  OurFPM.add(createCFGSimplificationPass());
  OurFPM.doInitialization();
  TheFPM = &OurFPM;

#ifndef MINIMAL_STDERR_OUTPUT
  fprintf(stderr, "ready> ");
#endif
  getNextToken();
  MainLoop();

  TheFPM = 0;
#if !defined(MINIMAL_STDERR_OUTPUT) || defined(DUMP_FINAL_MODULE)
  TheModule->dump();
#endif
  return 0;
}
