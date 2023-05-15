#!/bin/bash

set -e -x

CURRENT="$(pwd)"
SOURCE="$CURRENT"

BUILD="$CURRENT/build"
mkdir -p "$BUILD/{kaleidoscope,toy,standalone}"

ARGS="  \
        -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_LINKER=lld \
        -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
        -DLLVM_EXTERNAL_LIT=$BUILD/llvm/bin/llvm-lit \
        -DMLIR_DIR=$BUILD/llvm/out/lib/cmake/mlir \
"

pushd "$BUILD"
VIRTUAL_ENV="$BUILD/.env"
export VIRTUAL_ENV

pushd kaleidoscope
cmake -G Ninja -S "$SOURCE/kaleidoscope" $ARGS
cmake --build . --target Kaleidoscope-Ch2
popd

pushd toy
cmake -G Ninja -S "$SOURCE/toy" $ARGS
cmake --build . --target toyc-ch1
popd

pushd standalone
cmake -G Ninja -S "$SOURCE/standalone" $ARGS
cmake --build . --target check-standalone
popd

popd