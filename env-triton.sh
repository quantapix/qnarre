#!/bin/bash

# set -xeu -o pipefail

show_usage() {
    echo "Usage: $(basename "$0") [-c]"
}

main() {
    local OPTIND=1
    local GPU=false
    local CLEAN=false

    while getopts "cgh" opt; do
        case $opt in
        c) CLEAN=true ;;
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
    .env/bin/pip install -U cmake jupyter nvidia-ml-py3
    .env/bin/pip install -U cupy-cuda12x
    .env/bin/pip install --pre torch torchtext --index-url https://download.pytorch.org/whl/nightly/cu121
    (cd ./dev/triton/python/
        ../../../.env/bin/pip install -e '.[tests,tutorials]'
    )
}

main "$@"
