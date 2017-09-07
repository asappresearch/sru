DrQA
---

A pytorch implementation of the ACL 2017 paper [Reading Wikipedia to Answer Open-Domain Questions](http://www-cs.stanford.edu/people/danqi/papers/acl2017.pdf) (DrQA). The code is based on [Runqi](https://hitvoice.github.io/about/)'s implementation (https://github.com/hitvoice/DrQA).

## Requirements
- python >=3.5 
- pytorch 0.2.0
- numpy
- pandas
- msgpack
- spacy 1.x
- cupy
- pynvrtc

## Quick Start
### Setup
- make sure python 3 and pip is installed.
- install [pytorch](http://pytorch.org/) matched with your OS, python and cuda versions.
- install the remaining requirements via `pip install -r requirements.txt`
- download the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) datafile, GloVe word vectors and Spacy English language models using `bash download.sh`.

### Train

```bash
# prepare the data
python prepro.py

# make sure CUDA lib path can be found, e.g.:
export LD_LIBRARY_PATH=/usr/local/cuda/lib64

# specify the path to find SRU implementation, e.g.:
export PYTHONPATH=../../sru/

# train for 50 epoches with batchsize 32
python train.py -e 50 -bs 32
```

## Results
||EM|F1|Time used in RNN|Total time/epoch|
|---|---|---|---|---|
|LSTM (original paper)|69.5|78.8|~523s|~700s|
|SRU (this version)|**70.3**|**79.5**|**~88s**|**~200s**|

Tested on GeForce GTX 1070.

### Credits
Author of the Document Reader model: [Danqi Chen](https://github.com/danqi).

Author of the original Pytorch implementation: [Runqi Yang](https://hitvoice.github.io/about/). 

Most of the pytorch model code is borrowed from [Facebook/ParlAI](https://github.com/facebookresearch/ParlAI/) under a BSD-3 license.
