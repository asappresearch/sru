#!/bin/bash

# This tests for backward compatibility to BASE_VERSION of sru:
# - can a model saved with BASE_VERSION be loaded into current version?
# - will both models give the same outputs, given the same inputs? (this is
#.  not strictly necesssary for backward compatbility, but if they do give
#   the same outputs for given inputs, this is a strong indication of
#.  backwards compatbility)

# assumptions:
# - virtualenv is available

set -e
set -x

BASE_VERSION=2.3.5
CURRENT_VERSION=$(git rev-parse HEAD)

echo CURRENT: ${CURRENT_VERSION}
echo BASE: ${BASE_VERSION}

git clone -b ${BASE_VERSION} . ../${BASE_VERSION}

(
    cd ../${BASE_VERSION}
    virtualenv -p $(which python) .venv
    source .venv/bin/activate
    pip install -q torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    pip install -q -e ./
    python ../project/test/test_regression_1.py \
        --out-outputs ../project/outputs.pt \
        --out-model ../project/model.pt
)

source .venv/bin/activate
    python test/test_regression_2.py \
        --in-outputs outputs.pt \
        --in-model model.pt
