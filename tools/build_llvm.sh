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
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
"
#        -DCMAKE_BUILD_TYPE=Release \
#        -DCMAKE_BUILD_TYPE=Debug \

mkdir -p "$BUILD/llvm"
pushd "$BUILD"
if [ ! -e .env ]; then
    python3.11 -m venv .env
fi
.env/bin/pip install -U pip wheel setuptools
.env/bin/pip install -r "$SOURCE/mlir/python/requirements.txt"
VIRTUAL_ENV="$BUILD/.env"
export VIRTUAL_ENV
cd llvm
cmake -G Ninja -S "$SOURCE/llvm" $ARGS
ninja install
popd