#!/bin/bash

set -e -x

CURRENT="$(pwd)"

BUILD="$CURRENT/build"
SOURCE="$CURRENT/../dev/llvm-project"

ARGS="  \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_INSTALL_PREFIX=$BUILD/llvm/out \
        -DCMAKE_LINKER=lld \
        -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
        -DLLVM_ENABLE_ASSERTIONS=True \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_INSTALL_UTILS=ON \
        -DLLVM_TARGETS_TO_BUILD=X86;NVPTX \
"
#        -DCMAKE_BUILD_TYPE=Release \
#        -DCMAKE_BUILD_TYPE=Debug \

# rm -rf "$BUILD/llvm"
mkdir -p "$BUILD/llvm"
pushd "$BUILD/llvm"
# cmake "$SOURCE/llvm" $ARGS
cmake -G Ninja -S "$SOURCE/llvm" $ARGS
# num_jobs=40
# make -j${num_jobs} install
ninja install
popd
