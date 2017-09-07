
## About

**SRU** is a recurrent unit that can run over 10 times faster than cuDNN LSTM, without loss of accuracy tested on many tasks. 

<p align="center">
<img width=620 src="imgs/speed.png"><br>
<i>Average processing time of LSTM, conv2d and SRU, tested on GTX 1070</i><br>
</p>

<br>

For example, the figures above and below present the processing time of a single mini-batch and the training time for sentence-level classification tasks. SRU achieves 10 to 16 times speed-up compared to LSTM, and operates as fast as (or faster than) word-level convolutional model (CNNs).

<p align="center">
<img width=550 src="imgs/classification.png"><br>
<i>Training time (x-axis) vs valid accuracies (y-axis) on classification benchmarks</i><br>
</p>

<br>

## Requirements
 - [PyTorch](http://pytorch.org/)
 - [CuPy](https://cupy.chainer.org/)
 - [pynvrtc](https://github.com/NVIDIA/pynvrtc)
 
CuPy and pynvrtc needed to compile the CUDA code into a callable function at runtime.

<br>

## Examples
 - [classification](/classification/)
 - [question answering (SQuAD)](/DrQA/)
 - [language modelling on PTB](/language_model/)
 - machine translation
 - speech recognition
 
<br>

## Contributors
-  **Tao Lei** (tao@asapp.com)
-  **Yu Zhang** (yzhang87@csail.mit.edu)
