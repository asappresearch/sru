DrQA
---

A pytorch implementation of the ACL 2017 paper [Reading Wikipedia to Answer Open-Domain Questions](http://www-cs.stanford.edu/people/danqi/papers/acl2017.pdf) (DrQA). The code is based on [Runqi](https://hitvoice.github.io/about/)'s implementation (https://github.com/hitvoice/DrQA).

## Steps taken to reproduce classification:

1. Install CUDA 8.0 following [NVIDIA Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
2. Install Python 3.5 or newer.
3. Install [pytorch](http://pytorch.org/) matched with your OS, python and cuda versions.
4. Clone source repo: `git clone https://github.com/taolei87/sru`
5. Install requirements: `pip install -r sru/requirements.txt`.
6. (If using Linux) Make sure wget is intalled and use download the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) datafile, GloVe word vectors and Spacy English language models using `bash download.sh`.
7. (If NOT using Linux) Make sure to download the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) datafile, GloVe word vectors and Spacy English language models.
8. Update document reader model from https://github.com/hitvoice/DrQA. Already done in this repo as of December 2017.
9. Export required paths: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64 && export PYTHONPATH=./sru`
10. Run to train the model example: python3 train.py -e 50 -bs 32 --save_last_only
11. Check the drqa/layers.py to train using different models (SRU / LSTM)

## Requirements
- python >=3.5 
- pytorch 0.2.0
- numpy
- pandas
- msgpack
- spacy 1.x
- cupy
- pynvrtc

### Credits
Author of the Document Reader model: [Danqi Chen](https://github.com/danqi).

Author of the original Pytorch implementation: [Runqi Yang](https://hitvoice.github.io/about/). 

Most of the pytorch model code is borrowed from [Facebook/ParlAI](https://github.com/facebookresearch/ParlAI/) under a BSD-3 license.
