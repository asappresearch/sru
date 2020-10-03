#!/bin/bash

# This script requires to be run as either:
# - torch is cpu install, or
# - torch is cuda install, and CUDA toolkit is present

set -e
set -x

python test/test_ts_cpp.py > py_out.txt
cd sru/csrc/
if [[ -d build ]]; then {
    rm -Rf build
} fi
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; import os.path; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))')" ..
make -j
cd ../../../
sru/csrc/build/example_app sru_ts.pt > cpp_out.txt
diff cpp_out.txt py_out.txt
