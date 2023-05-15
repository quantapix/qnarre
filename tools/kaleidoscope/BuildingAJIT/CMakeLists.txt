set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  InstCombine
  native
  Object
  OrcJIT
  RuntimeDyld
  ScalarOpts
  Support
  TransformUtils
  )

add_kaleidoscope_chapter(BuildingAJIT toy.cpp)

export_executable_symbols(BuildingAJIT)
