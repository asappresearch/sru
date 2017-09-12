
Code used for language modeling. 
In the experiments, we used identity activation `--use_tanh 0` and set highway gate bias to `-3`.
These choices are found to produce better results.

## How to run
  - Download the dataset from https://github.com/yoonkim/lstm-char-cnn/tree/master/data/ptb
  
  - Make sure CUDA library path and `cuda_functional.py` is available to python. For example,
  ```python
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64
    export PYTHONPATH=../../sru/
  ```
  
  - Run **train_lm.py** and get the results.
  ```
    python train_lm.py --help               # see all running options
  
    python train_lm.py --train train.txt    # run with default options, 6 SRU layers  
      --dev valid.txt  --test test.txt
  ```
  
  - Additional options for runs in the paper
  ```
    --depth 5 --d 980                       # 5 SRU layers (with 24m param budget)
    --depth 4 --d 1060                      # 4 SRU layers
    --depth 3 --d 1180 --bias -1            # 3 SRU layers, highway bias -1
    --depth 2 --d 950 --lstm                # 2 LSTM layers
  ```
