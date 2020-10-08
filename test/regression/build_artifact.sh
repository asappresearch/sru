#!/bin/bash
"""
this is designed to be run offline, not in CCIE.

Assumptions:
- you are using pyenv, and pyenv is activated
- you have an installed python version of 3.7.7, in pyenv
- you do not have a pyenv called __tmp
- this repo folder contains .git folder
- you are running in linux
- you have a /tmp folder

usage:

build_artifacts.sh [SRU version]

eg:

build_artifacts.sh 2.3.5
"""
set -e
set -x

# we will use this version to create virtualenv(s)
PYENV_BASE=3.7.7

# This is the version of SRU that we will create an artifact for
SRU_BASELINE_VERSION=$1
if [[ x${SRU_BASELINE_VERSION} == x ]]; then {
    set +x
    echo
    echo Usage: build_artifacts.sh [SRU version]
    echo 
    exit 1
} fi

PROJECT_DIR=$PWD
TMP=/tmp

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

if pyenv activate __tmp; then {
    pyenv virtualenv-delete -f __tmp
} fi
pyenv virtualenv 3.7.7 __tmp
pyenv activate __tmp

ARTIFACTS_DIR=${PROJECT_DIR}/test/regression/artifacts
if [[ ! -d ${ARTIFACTS_DIR} ]]; then {
    mkdir -p ${ARTIFACTS_DIR}
} fi

if [[ -d $TMP/2.3.5 ]]; then {
    rm -Rf $TMP/2.3.5
} fi
git clone . $TMP/2.3.5 -b ${SRU_BASELINE_VERSION}

(
    cd $TMP/2.3.5

    pip install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    pip install .
    python ${PROJECT_DIR}/test/regression/build_artifact.py \
        --out-artifact ${ARTIFACTS_DIR}/${SRU_BASELINE_VERSION}.pt
)


rm -Rf $TMP/2.3.5
pyenv deactivate
pyenv virtualenv-delete -f __tmp

set +x
echo
echo Now add ${ARTIFACTS_DIR}/${SRU_BASELINE_VERSION}.pt to git
echo
