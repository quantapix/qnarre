#!/bin/bash

set -e -x

CURRENT="$(pwd)"
SOURCE="$CURRENT"

BUILD="$CURRENT/build"
mkdir -p "$BUILD/standalone"
mkdir -p "$BUILD/toy"

ARGS="  \
        -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_LINKER=lld \
        -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
        -DLLVM_EXTERNAL_LIT=$BUILD/llvm/bin/llvm-lit \
        -DMLIR_DIR=$BUILD/llvm/out/lib/cmake/mlir \
"
#        -DPython_FIND_VIRTUALENV=ONLY \
#        -DLLVM_MINIMUM_PYTHON_VERSION=3.11 \

pushd "$BUILD"
VIRTUAL_ENV="$BUILD/.env"
export VIRTUAL_ENV

pushd standalone
cmake -G Ninja -S "$SOURCE/standalone" $ARGS
cmake --build . --target check-standalone
popd

pushd toy
cmake -G Ninja -S "$SOURCE/toy" $ARGS
cmake --build .
popd

popd
