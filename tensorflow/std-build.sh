#!/bin/bash
# Copyright 2019 Quantapix Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# ./bazel-0.26 --user

set -x -e -u -o pipefail

build() {
    if [ "$1" ]; then
        rm -rf std.build std.install
        (cd upstream || exit
         git reset --hard
         git clean -xfd)
    fi
    if [ ! -e std.build ]; then
        mkdir std.build std.install
    fi
    (cd std.build || exit
     if [ ! -e .qpx_config ]; then
         (cd ../upstream || exit
          export CC_OPT_FLAGS="-pipe -fstack-protector-strong -fno-plt"
          export CC_OPT_FLAGS="-march=native $CC_OPT_FLAGS"
          PYTHON_BIN_PATH="$(which python)"
          export PYTHON_BIN_PATH
          export TF_DOWNLOAD_CLANG=0
          export TF_ENABLE_XLA=1
          export TF_IGNORE_MAX_BAZEL_VERSION=1
          export TF_NEED_AWS=0
          export TF_NEED_CUDA=0
          export TF_NEED_GCP=0
          export TF_NEED_GDR=0
          export TF_NEED_HDFS=0
          export TF_NEED_IGNITE=0
          export TF_NEED_JEMALLOC=1
          export TF_NEED_KAFKA=0
          export TF_NEED_MPI=0
          export TF_NEED_NGRAPH=0
          export TF_NEED_OPENCL=0
          export TF_NEED_OPENCL_SYCL=0
          export TF_NEED_ROCM=0
          export TF_NEED_S3=0
          export TF_NEED_TENSORRT=0
          export TF_NEED_VERBS=0
          export TF_SET_ANDROID_WORKSPACE=0
          export USE_DEFAULT_PYTHON_LIB_PATH=1
          if [ "$2" ]; then
              export TF_NEED_GDR=1
              export TF_NEED_VERBS=1
              export TF_NEED_CUDA=1
              export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
              export HOST_CXX_COMPILER_PATH=/usr/bin/gcc
              export TF_CUDA_CLANG=0
              # export CLANG_CUDA_COMPILER_PATH=/usr/bin/clang
              NVENV_PATH=/home/qpix/clone/qnarre_new/.nvenv
              export TF_CUDA_PATHS=$NVENV_PATH/cublas,$NVENV_PATH/cuda,$NVENV_PATH/nccl
              export TF_CUDA_VERSION=$($NVENV_PATH/cuda/bin/nvcc --version | sed -n 's/^.*release \(.*\),.*/\1/p')
              export TF_CUDNN_VERSION=$(sed -n 's/^#define CUDNN_MAJOR\s*\(.*\).*/\1/p' $NVENV_PATH/cuda/include/cudnn.h)
              export TF_NCCL_VERSION=$(sed -n 's/^#define NCCL_MAJOR\s*\(.*\).*/\1/p' $NVENV_PATH/nccl/include/nccl.h)
              export TF_NEED_TENSORRT=0
              export TENSORRT_INSTALL_PATH=$NVENV_PATH/tensorrt
              export TF_TENSORRT_VERSION=$(sed -n 's/^#define NV_TENSORRT_MAJOR\s*\(.*\).*/\1/p' $TENSORRT_INSTALL_PATH/include/NvInfer.h)
              export TF_CUDA_COMPUTE_CAPABILITIES=6.1,7.0
              export LD_LIBRARY_PATH=$NVENV_PATH/cublas/lib64:$NVENV_PATH/cuda/lib64:$NVENV_PATH/extras/CUPTI/lib64:$NVENV_PATH/nccl/lib:$NVENV_PATH/tensorrt/lib
          else
              # export CC_OPT_FLAGS="-march=native $CC_OPT_FLAGS"
              export TF_NEED_GDR=0
              export TF_NEED_VERBS=0
              export TF_NEED_CUDA=0
          fi
          ./configure)
         touch .qpx_config
     else
         echo "*** SKIPPING CONFIG... ***"
     fi
     (cd ../upstream || exit
      if [ "$2" ]; then
          bazel build --config=v2 --config=opt \
                --config=noaws --config=nohdfs \
                --config=nokafka --config=noignite \
                --incompatible_no_support_tools_in_action_inputs=false \
                //tensorflow/tools/pip_package:build_pip_package
      else
          bazel build --config=v2 --config=opt --config=mkl \
                --config=noaws --config=nohdfs \
                --config=nokafka --config=noignite \
                --incompatible_no_support_tools_in_action_inputs=false \
                //tensorflow/tools/pip_package:build_pip_package
      fi
      bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag ../std.install)
    )
    (cd ../tensorboard || exit
     bazel build //tensorboard/pip_package:build_pip_package
     bazel-bin/tensorboard/pip_package/build_pip_package ../std.install
    )
    (cd std.install || exit
     pip install -I tf_nightly-*.whl)
}

show_usage() {
    echo "Usage: $(basename "$0") [-c] [-g]"
}

main() {
    local OPTIND=1
    local CFG=
    local GPU=

    while getopts "cgh" opt; do
	      case $opt in
	          c) CFG=true;;
	          g) GPU=true;;
	          *) show_usage; return 1;;
	      esac
    done
    shift $((OPTIND-1))

	  build "$CFG" "$GPU"
}

main "$@"
