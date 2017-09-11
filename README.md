
## About

**SRU** is a recurrent unit that can run over 10 times faster than cuDNN LSTM, without loss of accuracy tested on many tasks. 
<p align="center">
<img width=620 src="imgs/speed.png"><br>
<i>Average processing time of LSTM, conv2d and SRU, tested on GTX 1070</i><br>
</p>
For example, the figure above presents the processing time of a single mini-batch of 32 samples. SRU achieves 10 to 16 times speed-up compared to LSTM, and operates as fast as (or faster than) word-level convolution using conv2d. 

### Reference:
[Training RNNs as Fast as CNNs](https://arxiv.org/abs/1709.02755)
```
@article{lei2017sru,
  title={Training RNNs as Fast as CNNs},
  author={Lei, Tao and Zhang, Yu},
  journal={arXiv preprint arXiv:1709.02755},
  year={2017}
}
```

## Requirements
 - **GPU and CUDA are required**
 - [PyTorch](http://pytorch.org/)
 - [CuPy](https://cupy.chainer.org/)
 - [pynvrtc](https://github.com/NVIDIA/pynvrtc)
 
CuPy and pynvrtc needed to compile the CUDA code into a callable function at runtime.

<br>

## Examples
The usage of SRU is the similar to `nn.LSTM`. 
```python
import torch
from cuda_functional import SRU, SRUCell

# input has length 20, batch size 32 and dimension 128
x = torch.FloatTensor(20, 32, 128).cuda()

input_size, hidden_size = 128, 128

rnn = SRU(input_size, hidden_size,
    num_layers = 2,          # number of stacking RNN layers
    dropout = 0.0,           # dropout applied between RNN layers
    rnn_dropout = 0.0,       # variational dropout applied on linear transformation
    use_tanh = 1,            # use tanh or identity activation
    bidirectional = False    # bidirectional RNN ?
)

output, hidden = rnn(x)      # forward pass

# output is (length, batch size, hidden size * number of directions)
# hidden is (layers, batch size, hidden size * number of directions)

```
<br>

 - [classification](/classification/)
 - [question answering (SQuAD)](/DrQA/)
 - [language modelling on PTB](/language_model/)
 - machine translation
 - speech recognition
 
<br>

## Contributors
-  **Tao Lei** (tao@asapp.com)
-  **Yu Zhang** (yzhang87@csail.mit.edu)

<br>

## To-do
  - [ ] ReLU activation
  - [ ] Layer normalization + residual to compare with highway connection (current version)
  
