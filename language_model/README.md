
Code used for language modeling. 

## Train Enwik8
  - Download the dataset from http://mattmahoney.net/dc/enwik8.zip and unzip the file
  
  - Run **train_enwik8.py** and get the results.
  ```
    python train_enwik8.py --help           # see all running options
  
    python train_enwik8.py -data enwik8 -log tensorboard_log_directory
      --noam --lr 3
  ```
  
  - Options for runs reported in the paper
  ```
    --depth 8 --d 1312                      #  8 layers (with 37m param budget)
    --depth 10 --d 1152                     # 10 layers (with 37m param budget)

    --depth 8 --d 3056 --n_proj 512 --dropout 0.3
                                            # 8 layers (with 47m param budget)

    --depth 12 --d 2048 --n_proj 512 --dropout 0.3 --unroll 256 --batch_size 64 --n_e 256
                                            # 12 layers (with 47m param budget)
  ```

## Train PTB
  - Download the dataset from https://github.com/yoonkim/lstm-char-cnn/tree/master/data/ptb
  
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
