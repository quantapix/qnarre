#!/bin/bash

set -x

set -e -u -o pipefail

# python -m ipykernel install --user --name qenv --display-name "Python (.qenv)"
# jupyter notebook --no-browser --port=8889
# ssh -N -f -L localhost:8888:localhost:8889 remote_user@remote_host

# export PATH=/home/qpix/clone/qnarre_new/.nvold/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/home/qpix/clone/qnarre_new/.nvold/cuda/lib64:$LD_LIBRARY_PATH
# export TF_XLA_FLAGS=--tf_xla_cpu_global_jit

nvenv() {
    mkdir .nvenv
    (cd nvidia
     sh ./cuda_* --extract="$OLDPWD"/.nvenv --override)
    mv .nvenv/cuda-toolkit .nvenv/cuda
    mv .nvenv/cuda-samples .nvenv/cuda/samples
    (cd nvidia
     tar xf cudnn_* -C "$OLDPWD"/.nvenv
     tar xf nccl_* -C "$OLDPWD"/.nvenv
     tar xf tensorrt_* -C "$OLDPWD"/.nvenv)
    mv .nvenv/nccl_* .nvenv/nccl
    mv .nvenv/TensorRT-* .nvenv/tensorrt
    # (cd /usr/local
    #  sudo ln -s "$OLDPWD"/.nvenv/cublas cublas
    #  sudo ln -s "$OLDPWD"/.nvenv/cuda cuda
    #  sudo ln -s "$OLDPWD"/.nvenv/nccl nccl
    #  sudo ln -s "$OLDPWD"/.nvenv/tensorrt tensorrt)
}

keras() {
    (cd keras/upstream
     ../../.qenv/bin/pip install -e .
    )
}

spacy() {
    (cd spacy/blis
     ../../.qenv/bin/pip install -r requirements.txt
     ../../.qenv/bin/python setup.py build_ext -i -j "$(nproc)"
     ../../.qenv/bin/pip install -e .
     ../../.qenv/bin/python -m pytest thinc/
    )
    (cd spacy/thinc
     ../../.qenv/bin/pip install -r requirements.txt
     ../../.qenv/bin/python setup.py build_ext -i -j "$(nproc)"
     ../../.qenv/bin/pip install -e .
     ../../.qenv/bin/python -m pytest thinc/
    )
    (cd spacy/upstream
     ../../.qenv/bin/pip install -r requirements.txt
     ../../.qenv/bin/python setup.py build_ext -i -j "$(nproc)"
     ../../.qenv/bin/pip install -e .
     ../../.qenv/bin/python -m pytest spacy/
    )
    (cd spacy/stanford
     ../../.qenv/bin/pip install -r requirements.txt
     ../../.qenv/bin/python setup.py build_ext -i -j "$(nproc)"
     ../../.qenv/bin/pip install -e .
    )
}

spacy_en() {
    .qenv/bin/python -m spacy download en_core_web_sm
    .qenv/bin/python -m spacy download en_core_web_md
    .qenv/bin/python -m spacy download en_core_web_lg
    .qenv/bin/python -m spacy download en_vectors_web_lg
}

tflow() {
    if ls tensorflow/std.install/tf_nightly-*.whl 1> /dev/null 2>&1; then
	      .qenv/bin/pip install -I tensorflow/std.install/tf_nightly-*.whl
    else
        if $1; then
            .qenv/bin/pip install -U tf-nightly-gpu-2.0-preview
        else
            .qenv/bin/pip install -U tf-nightly-2.0-preview
        fi
        .qenv/bin/pip install -U tfp-nightly
        .qenv/bin/pip install -U tb-nightly
    fi
}

ptorch() {
    .qenv/bin/pip install -U torchvision_nightly
    if $1; then
        .qenv/bin/pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html
    else
        .qenv/bin/pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
    fi
}


show_usage() {
    echo "Usage: $(basename "$0") [-c] [-g]"
}

main() {
    local OPTIND=1
    local CLEAN=false
    local GPU=false

    while getopts "cgh" opt; do
	      case $opt in
	          c) CLEAN=true;;
	          g) GPU=true;;
	          *) show_usage; return 1;;
	      esac
    done
    shift $((OPTIND-1))

    if "$CLEAN"; then
	      rm -rf .nvenv .qenv
    fi

    if [ ! -e .nvenv ]; then
        if "$GPU"; then
	          nvenv
        fi
    fi

    if [ ! -e .qenv ]; then
	      python3.7 -m venv .qenv
    fi
    .qenv/bin/pip install -U pip wheel setuptools pytest
    .qenv/bin/pip install -U pycodestyle pylint pyyaml
    .qenv/bin/pip install -U flake8 autopep8 jedi yapf
    .qenv/bin/pip install -U regex cytoolz joblib scikit-learn pandas matplotlib
    .qenv/bin/pip install -U gin-config  sympy gym pypng spacy-nightly
    # .qenv/bin/pip install -U dash dash-html-components dash-core-components
    # .qenv/bin/pip install -U dash-table dash-daq
    .qenv/bin/pip install -U awscli
    .qenv/bin/pip install -U jupyter

	  # keras
    # spacy
    # spacy_en
    # tflow "$GPU"
    # ptorch "$GPU"
}

main "$@"
