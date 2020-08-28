set -e
export USE_CUDA=0
cd sru/csrc/
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; import os.path; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))')" ..
make -j
./example_app
