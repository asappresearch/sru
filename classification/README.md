
All work belongs to original author : https://github.com/taolei87/sru

We detail here the steps taken to reproduce their results. train.py is based on the train_classifier.py and is aimed to automate the training on all datasets for all CNN, LSTM and SRU based RNN models.

## Steps taken to reproduce classification:

1. Install CUDA 8.0 following [NVIDIA Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
2. Install Anaconda for Python 2.7: `conda create -n py27 python=2.7 anaconda`.
3. Activate new conda environment: `source activate py27`
4. Clone source repo: `git clone https://github.com/taolei87/sru`
5. Install requirements: `pip install -r sru/requirements.txt`.
6. Install PyTorch: `conda install pytorch torchvision cuda80 -c soumith`
7. Download the dataset from  https://github.com/harvardnlp/sent-conv-torch/tree/master/data
8. Download a pre-trained word embedding such as word2vec from https://github.com/mmihaltz/word2vec-GoogleNews-vectors or lexVec from https://github.com/alexandres/lexvec. We used LexVec in our project and the authors of the original paper used Google word2vec (in text format, not binary)
9. Export required paths: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64 && export PYTHONPATH=./sru`
10. Run the classification example: python train.py --path PATH_TO_DATASET --embedding PATH_TO_WORD_EMBEDDING --max_epoch 10 --cv 0

  
  ### Credits
  
  All work belongs to original author : https://github.com/taolei87/sru
  
  Part of the code (such as text preprocessing) is taken from https://github.com/harvardnlp/sent-conv-torch
  
  CNN model is the implementation of [(Kim, 2014)](http://arxiv.org/abs/1408.5882), following
   - torch / lua version: https://github.com/yoonkim/CNN_sentence
   - pytorch version: https://github.com/Shawn1993/cnn-text-classification-pytorch
  
  
  
