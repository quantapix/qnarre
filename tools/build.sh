#!/bin/bash

set -e -x

CURRENT="$(pwd)"

BUILD="$CURRENT/build"
SOURCE="$CURRENT/standalone"

ARGS="  \
        -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_LINKER=lld \
        -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
        -DLLVM_EXTERNAL_LIT=$BUILD/llvm/bin/llvm-lit \
        -DMLIR_DIR=$BUILD/llvm/out/lib/cmake/mlir \
"

mkdir -p "$BUILD/standalone"
pushd "$BUILD/standalone"
cmake -G Ninja -S "$SOURCE" $ARGS
cmake --build . --target check-standalone
popd
