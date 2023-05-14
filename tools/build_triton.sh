#!/bin/bash

set -e -x

CURRENT="$(pwd)"

BUILD="$CURRENT/build"
SOURCE="$CURRENT/../lib/triton"

ARGS="  \
        -DCMAKE_BUILD_TYPE=TritonRelBuildWithAsserts \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$BUILD/triton/out \
        -DCMAKE_LINKER=lld \
        -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
        -DLLVM_ENABLE_ASSERTIONS=True \
        -DLLVM_ENABLE_WERROR=ON \
        -DLLVM_EXTERNAL_LIT=$BUILD/llvm/bin/llvm-lit \
        -DLLVM_INCLUDE_DIRS=$BUILD/llvm/out/include \
        -DLLVM_LIBRARY_DIR=$BUILD/llvm/out/lib \
        -DPYBIND11_INCLUDE_DIR=$BUILD/.env/lib64/python3.11/site-packages/pybind11/include \
        -DPYTHON_INCLUDE_DIRS=/usr/include/python3.11 \
        -DPython3_EXECUTABLE:FILEPATH=./.env/bin/python \
        -DTRITON_BUILD_PYTHON_MODULE=ON \
        -DTRITON_BUILD_TUTORIALS=OFF \
"
#        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#        -DCMAKE_BUILD_TYPE=Debug \

# rm -rf "$BUILD"
mkdir -p "$BUILD/triton"
pushd "$BUILD"
if [ ! -e .env ]; then
    python3.11 -m venv .env
fi
.env/bin/pip install -U pip wheel setuptools pybind11
popd
pushd "$BUILD/triton"
# cmake "$SOURCE" $ARGS
cmake -G Ninja -S "$SOURCE" $ARGS
# num_jobs=40
# make -j${num_jobs} install
ninja install
popd
