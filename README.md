
### COMP 551 - PROJECT 4 - Reproducible Machine Learning

In this project we attempt to reproduce the results of the following paper:
 
#### Reference:
[Training RNNs as Fast as CNNs](https://arxiv.org/abs/1709.02755)
```
@article{lei2017sru,
  title={Training RNNs as Fast as CNNs},
  author={Lei, Tao and Zhang, Yu},
  journal={arXiv preprint arXiv:1709.02755},
  year={2017}
}
```

<br>

## Progress

- [x] Setup GCE instance for training
  - [X] Obtain GCP approval for additional GPUs
- [X] Reproduce author's SRU implementation
- [X] Reproduce classification model
- [X] Reproduce question answering model
- [X] Reproduce langauge model
- [A] Reproduce speech model

X - Complete
A - Attempted but failed to reproduce

## Requirements
 - **GPU and CUDA 8 are required**
 - [PyTorch](http://pytorch.org/)
 - [CuPy](https://cupy.chainer.org/)
 - [pynvrtc](https://github.com/NVIDIA/pynvrtc)
 
Install requirements via `pip install -r requirements.txt`. CuPy and pynvrtc needed to compile the CUDA code into a callable function at runtime. Only single GPU training is supported. 

<br>

Check individual tests for steps taken to reproduce the results

  
