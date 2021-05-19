
# SRU C source code
This directory contains the `.cpp` and `.cu` implementations of SRU elementwise recurrence operator.
It also gives a toy test example of using SRU in C++, namely the `main_test_cpp.cpp` and
`CMakeLists.txt`.


## How to compile and run the C++ test example
The example is written based on the tutorial:
https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html 

From the `csrc` directory, compile the code using `cmake`:
```
$ mkdir build
$ cd build 
$ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" -DGLIBCXX_USE_CXX11_ABI="$(python -c 'import torch; print(1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0)')" ..
$ make -j
```
Note that `torch.utils.cmake_prefix_path` is not available in earlier versions of pytorch such as
1.3.1, you have to manually set it to `[path to your pytorch installation]/python3.7/site-packages/torch/share/cmake/Torch`


After compilation, you should be able to see and run `example_app` binary, which needs an exported
torchscript SRU model as input:
```
$ ./example_app <path to an exported torchscript SRU model>
$ ./example_app <path to an exported torchscript SRU model> cuda
```

## Save and load a torchscript SRU model
In Python:
```
import torch
import sru

model = sru.SRU(4, 4)
torchscript_model = torch.jit.script(model)
torchscript_model.save("example_model.pt")
```

Test loading the model:
```
$ ./example_app <path to the model>/example_model.pt
$ ./example_app <path to the model>/example_model.pt cuda
```
