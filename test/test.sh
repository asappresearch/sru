#!/bin/bash

# This script requires to be run as either:
# - torch is cpu install, or
# - torch is cuda install, and CUDA toolkit is present

set -e
set -x

cd sru/csrc/
if [[ -d build ]]; then {
    rm -Rf build
} fi
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; import os.path; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))')" ..
make -j
cd ../../../

python test/test_ts_sru_cpp.py > py_out.txt
sru/csrc/build/example_app sru_ts.pt > cpp_out.txt
diff cpp_out.txt py_out.txt

python test/test_ts_sru_cpp.py --normalize-after > py_out.txt
sru/csrc/build/example_app sru_ts.pt > cpp_out.txt
diff cpp_out.txt py_out.txt

python test/test_ts_srupp_cpp.py > py_srupp_out.txt
sru/csrc/build/example_app srupp_ts.pt > cpp_srupp_out.txt
diff cpp_srupp_out.txt py_srupp_out.txt

python test/test_ts_srupp_cpp.py --normalize-after > py_srupp_out.txt
sru/csrc/build/example_app srupp_ts.pt > cpp_srupp_out.txt
diff cpp_srupp_out.txt py_srupp_out.txt
