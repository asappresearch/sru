
Code used for sentence classification tasks. We evaluate CNN, LSTM and SRU on 6 benchmarks. Example learning curves:

<p align="center">
<img width=550 src="../imgs/classification.png"><br>
<i>Training time (x-axis) vs valid accuracies (y-axis) on classification benchmarks</i><br>
</p>

## How to run
  - Download the datasets from [harvardnlp/sent-conv-torch/data](https://github.com/harvardnlp/sent-conv-torch/tree/master/data)
  
  - Download pre-trained word embeddings such as [word2vec](https://code.google.com/p/word2vec/); make it into text format
  
  - Make sure CUDA library path and `cuda_functional.py` is available to python. For example,
  ```python
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64
    export PYTHONPATH=../../sru/
  ```
  
  - Run **train_classifier.py** and get the results.
  ```
    python train_classifier.py --help           # see all running options
  
    python train_classifier.py --dataset mr     # which dataset (mr, subj, cr, sst, trec, mpqa) 
          --path data_directory                 # path to the data directory
          --embedding google_word2vec.txt       # path to pre-trained embeddings
          --cv 0                                # 10-fold cross-validation, use split 0 as the test set
  ```
  
  <br>
  
  ### Credits
  
  Part of the code (such as text preprocessing) is taken from https://github.com/harvardnlp/sent-conv-torch
  
  CNN model is the implementation of [(Kim, 2014)](http://arxiv.org/abs/1408.5882), following
   - torch / lua version: https://github.com/yoonkim/CNN_sentence
   - pytorch version: https://github.com/Shawn1993/cnn-text-classification-pytorch
  
  
  
