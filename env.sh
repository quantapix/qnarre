#!/bin/bash

# set -xeu -o pipefail

# python -m ipykernel install --user --name qenv --display-name "Python (.env)"
# jupyter notebook --no-browser --port=8889
# ssh -N -f -L localhost:8888:localhost:8889 remote_user@remote_host

# jupyter nbconvert --to script

# export PATH=/home/qpix/clone/qnarre_new/.nvold/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/home/qpix/clone/qnarre_new/.nvold/cuda/lib64:$LD_LIBRARY_PATH
# export TF_XLA_FLAGS=--tf_xla_cpu_global_jit

spacy() {
    if $1; then
        .env/bin/pip install spacy[lookups,transformers,cuda118]
    else
        .env/bin/pip install spacy[lookups,transformers]
    fi
    .env/bin/python -m spacy download en_core_web_sm
    .env/bin/python -m spacy download en_core_web_md
    .env/bin/python -m spacy download en_core_web_lg
    # .env/bin/python -m spacy download en_vectors_web_lg
}

ptorch() {
    if $1; then
        .env/bin/pip install -U cupy-cuda11x
        .env/bin/pip install --pre torch torchaudio torchtext torchvision --index-url https://download.pytorch.org/whl/nightly/cu118
    else
        .env/bin/pip install --pre torch torchaudio torchtext torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
    fi
    .env/bin/pip install -U tensorboard torch-tb-profiler
    # .env/bin/pip install -U captum
}

show_usage() {
    echo "Usage: $(basename "$0") [-c] [-g]"
}

main() {
    local OPTIND=1
    local GPU=false
    local CLEAN=false

    while getopts "cgh" opt; do
        case $opt in
        c) CLEAN=true ;;
        g) GPU=true ;;
        *)
            show_usage
            return 1
            ;;
        esac
    done
    shift $((OPTIND - 1))

    if "$CLEAN"; then
        rm -rf .env
    fi

    if [ ! -e .env ]; then
        python3.11 -m venv .env
    fi

    .env/bin/pip install -U pip wheel setuptools pytest black
    # .env/bin/pip install -U packaging requests opt_einsum
    .env/bin/pip install -U numpy pandas matplotlib scipy scikit-learn nltk
    # .env/bin/pip install -U keras_preprocessing --no-deps
    ptorch "$GPU"
    # spacy "$GPU"
    .env/bin/pip install -U jupyter seaborn awscli
    .env/bin/pip install -U pyarrow sentencepiece
    .env/bin/pip install -U tokenizers accelerate datasets transformers evaluate
    # .env/bin/pip install -U regex cytoolz joblib
    # .env/bin/pip install -U pycodestyle pylint pyyaml
    # .env/bin/pip install -U flake8 autopep8 jedi yapf
    # .env/bin/pip install -U gin-config sympy gym pypng spacy-nightly

    .env/bin/python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
}

main "$@"
