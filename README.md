
## About

SRU is a recurrent unit that can run over 10 times faster than cuDNN LSTM, without loss of accuracy tested on many tasks.

<p align="center">
<img width=650 src="imgs/speed.png"><br>
<i>Average processing time of cuDNN LSTM, conv2d and SRU, tested on GTX 1070</i>
</p>

<br>

## Requirements
 - [PyTorch](http://pytorch.org/)
 - [CuPy](https://cupy.chainer.org/)
 - [pynvrtc](https://github.com/NVIDIA/pynvrtc)
 
CuPy and pynvrtc needed to compile the CUDA code into a callable function at runtime.

<br>

## Contributors
-  **Tao Lei** (tao@asapp.com)
-  **Yu Zhang** (yzhang87@csail.mit.edu)
