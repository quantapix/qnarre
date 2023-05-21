cmake_minimum_required(VERSION 3.22)

if(POLICY CMP0116)
# Introduced in cmake 3.20
# https://cmake.org/cmake/help/latest/policy/CMP0116.html
  cmake_policy(SET CMP0116 OLD)
endif()

include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

project(triton)
include(CTest)

# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(TRITON_BUILD_TUTORIALS "Build C++ Triton tutorials" ON)
option(TRITON_BUILD_PYTHON_MODULE "Build Python Triton bindings" ON)

set(CMAKE_C_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")
set(CMAKE_CXX_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

find_library(TERMINFO_LIBRARY tinfo)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PYBIND11_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__STDC_FORMAT_MACROS  -fPIC -std=gnu++17 -fvisibility=hidden -fvisibility-inlines-hidden")

set(LLVM_LDFLAGS "-L${LLVM_LIBRARY_DIR}")
set(LLVM_LIBRARIES
  libLLVMAggressiveInstCombine.a
  libLLVMAMDGPUAsmParser.a
  libLLVMAMDGPUCodeGen.a
  libLLVMAMDGPUDesc.a
  libLLVMAMDGPUDisassembler.a
  libLLVMAMDGPUInfo.a
  libLLVMAMDGPUUtils.a
  libLLVMAnalysis.a
  libLLVMAnalysis.a
  libLLVMAnalysis.a
  libLLVMAsmParser.a
  libLLVMAsmPrinter.a
  libLLVMBinaryFormat.a
  libLLVMBitReader.a
  libLLVMBitstreamReader.a
  libLLVMBitWriter.a
  libLLVMCodeGen.a
  libLLVMCore.a
  libLLVMCoroutines.a
  libLLVMDebugInfoCodeView.a
  libLLVMDebugInfoDWARF.a
  libLLVMDebugInfoMSF.a
  libLLVMDemangle.a
  libLLVMFrontendOpenMP.a
  libLLVMGlobalISel.a
  libLLVMInstCombine.a
  libLLVMInstrumentation.a
  libLLVMipo.a
  libLLVMipo.a
  libLLVMIRReader.a
  libLLVMLinker.a
  libLLVMMC.a
  libLLVMMCDisassembler.a
  libLLVMMCParser.a
  libLLVMMIRParser.a
  libLLVMNVPTXCodeGen.a
  libLLVMNVPTXDesc.a
  libLLVMNVPTXInfo.a
  libLLVMObjCARCOpts.a
  libLLVMObject.a
  libLLVMPasses.a
  libLLVMProfileData.a
  libLLVMRemarks.a
  libLLVMScalarOpts.a
  libLLVMScalarOpts.a
  libLLVMSelectionDAG.a
  libLLVMSupport.a
  libLLVMTarget.a
  libLLVMTextAPI.a
  libLLVMTransformUtils.a
  libLLVMTransformUtils.a
  libLLVMTransformUtils.a
  libLLVMVectorize.a
)
set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)

if(TRITON_BUILD_PYTHON_MODULE)
  message(STATUS "Adding Python module")
  set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/src)
  set(PYTHON_SRC ${PYTHON_SRC_PATH}/main.cc ${PYTHON_SRC_PATH}/triton.cc)
  include_directories("." ${PYTHON_SRC_PATH})
  if(PYTHON_INCLUDE_DIRS)
    include_directories(${PYTHON_INCLUDE_DIRS})
  else()
    message(STATUS "FINDING...")
    find_package(Python3 COMPONENTS Interpreter Development)
    include_directories(${Python3_INCLUDE_DIRS})
    link_directories(${Python3_LIBRARY_DIRS})
    link_libraries(${Python3_LIBRARIES})
    add_link_options(${Python3_LINK_OPTIONS})
  endif()
endif()

message(STATUS "FINDING2...")
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen) # required by AddMLIR
include(AddLLVM)
include(AddMLIR)

# Disable warnings that show up in external code (gtest;pybind11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default")

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include) # Tablegen'd files

# link_directories(${LLVM_LIBRARY_DIR})
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)

# find_package(PythonLibs REQUIRED)
set(TRITON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

if(TRITON_BUILD_PYTHON_MODULE)
  add_library(triton SHARED ${PYTHON_SRC})
  set(TRITON_LIBRARIES
    TritonAnalysis
    TritonTransforms
    TritonGPUTransforms
    TritonLLVMIR
    TritonPTX
    TritonHSACO
    ${dialect_libs}
    ${conversion_libs}
    # optimizations
    MLIRPass
    MLIRTransforms
    MLIRLLVMDialect
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRExecutionEngine
    MLIRMathToLLVM
    MLIRNVVMToLLVMIRTranslation
    MLIRROCDLToLLVMIRTranslation
    MLIRIR
  )
  target_link_libraries(triton ${LLVM_LIBRARIES} z
    ${TRITON_LIBRARIES}
  )
  link_libraries(stdc++fs)
  target_link_options(triton PRIVATE ${LLVM_LDFLAGS})
endif()

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs,ALL")

if(TRITON_BUILD_PYTHON_MODULE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  target_link_libraries(triton ${CUTLASS_LIBRARIES} ${PYTHON_LDFLAGS})
endif()

add_subdirectory(test)
add_subdirectory(unittest)
