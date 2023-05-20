#!/bin/bash

set -e -x

CURRENT="$(pwd)"

BUILD="$CURRENT/build"
SOURCE="$CURRENT/triton"

ARGS="  \
        -DCMAKE_BUILD_TYPE=TritonRelBuildWithAsserts \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$BUILD/triton/out \
        -DCMAKE_LINKER=lld \
        -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
        -DLLVM_ENABLE_ASSERTIONS=True \
        -DLLVM_EXTERNAL_LIT=$BUILD/llvm/bin/llvm-lit \
        -DLLVM_INCLUDE_DIRS=$BUILD/llvm/out/include \
        -DLLVM_LIBRARY_DIR=$BUILD/llvm/out/lib \
        -DPYBIND11_INCLUDE_DIR=$BUILD/.env/lib64/python3.11/site-packages/pybind11/include \
        -DPython3_EXECUTABLE=$BUILD/.env/bin/python \
        -DTRITON_BUILD_PYTHON_MODULE=ON \
        -DTRITON_BUILD_TUTORIALS=OFF \
"
#        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#        -DCMAKE_BUILD_TYPE=Debug \

mkdir -p "$BUILD/triton"
pushd "$BUILD"
VIRTUAL_ENV="$BUILD/.env"
export VIRTUAL_ENV
cd triton
cmake -G Ninja -S "$SOURCE" $ARGS
ninja install
popd
