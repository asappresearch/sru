DrQA
---

A pytorch implementation of the ACL 2017 paper [Reading Wikipedia to Answer Open-Domain Questions](http://www-cs.stanford.edu/people/danqi/papers/acl2017.pdf) (DrQA). The code is based on [Runqi](https://hitvoice.github.io/about/)'s implementation (https://github.com/hitvoice/DrQA).

## Requirements
- python >=3.5 
- numpy
- pandas
- msgpack
- spacy 1.x

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

# make sure CUDA lib path and SRU can be found. if not, try:
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PYTHONPATH=/path_to_sru_repo/sru/

# train for 100 epoches with batchsize 32
python train.py -e 100 -bs 32
```

## Results
||EM|Time used in RNN|Total time/epoch|
|---|---|---|---|
|LSTM (original paper)|69.5|~316s|~431s|
|SRU (v1, 6 layer)|**~71.1**|**~100s**|**~201s**|
|SRU (this version, 6 layer)|**~71.4**|**~100s**|**~201s**|

Tested on GeForce GTX 1070.

### Credits
Author of the Document Reader model: [Danqi Chen](https://github.com/danqi).

Author of the original Pytorch implementation: [Runqi Yang](https://hitvoice.github.io/about/). 

Most of the pytorch model code is borrowed from [Facebook/ParlAI](https://github.com/facebookresearch/ParlAI/) under a BSD-3 license.
