## Linux

Please check [pytorch](https://pytorch.org), you can select pytorch version, os, package, python version and cuda version, and the website will give you the command to run.

Configuration I've tried is:

```
Pytorch Build: State(1.0)
OS: Linux
Package: Conda
Language: Python 3.6
CUDA: 8.0
```

And the command is
```
conda install pytorch torchvision cuda80 -c pytorch
```

If your system is Linux, this is the right way to install pytorch.

## Mac OS

For Mac OS, the situation is a bit complicated.

```
# MacOS Binaries dont support CUDA, install from source if CUDA is needed
```

Our code need CUDA, so you need to install pytorch from source code.

Please refer to [pytorch cuda on mac](https://www.cs.rochester.edu/u/kautz/Installing-Pytorch-Cuda-on-Macbook.html)

