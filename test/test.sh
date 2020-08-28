set -e
python test/test_ts_cpp.py > py_out.txt
cd sru/csrc/
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; import os.path; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))')" ..
make -j
cd ../../../
sru/csrc/build/example_app test/sru_ts.pt > cpp_out.txt
diff cpp_out.txt py_out.txt
