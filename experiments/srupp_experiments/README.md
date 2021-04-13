## About
This folder contains the experimental code of SRU++ [tech report](https://arxiv.org/pdf/2102.12459):
```
@article{lei2021srupp,
  title={When Attention Meets Fast Recurrence: Training Language Models with Reduced Compute},
  author={Tao Lei},
  journal={arXiv preprint arXiv:2102.12459},
  year={2021}
}
```
<br>

## Pretrained models
|  Name  |  Params  |  Hidden size <br> (n_d, n_proj) |  Size ratio <br> (n_d : n_proj) | Test result <br> (BPC / PPL) |  Downloadable link  |
| :---- | :----: | :----: | :----: | :----: | :---- |
| Enwik8 base | 108M | 3072, 768 | 4 | 0.974 |  |
| Enwik8 large | 195M | 6016, 752 | 8 | 0.953 |  |
| | | | | |
| Wiki-103 base| 148M | 3072, 768 | 4 | 18.3 |  |
| Wiki-103 large | 234M | 5952, 744 | 8 | 17.1 |  |
| Wiki-103 large <br> 2 attention layers (k=5) | 225M | 5952, 744 | 8 | 17.3 |  |
| | | | | |
| Billion-word base | 328M | 4096, 1024 | 4 | 25.1 |  |
| Billion-word large <br> 2 attention layers (k=5) | 467M | 7616, 1024 | 7.4 | 23.5 |  |

<br>

## Reproduce our results

### Data preparation
- Enwik8: download the dataset (http://mattmahoney.net/dc/enwik8.zip) and unzip it to get the `enwik8` text file.
- Wiki-103: download and unzip the dataset using [the script](https://github.com/kimiyoung/transformer-xl/blob/master/getdata.sh#L18-L27) of Transformer-XL repo.
- Billion Word: download and unzip the dataset using [the script](https://github.com/kimiyoung/transformer-xl/blob/master/getdata.sh#L71-L87) of Transformer-XL repo.

For Wiki-103 and Billion Word datasets, you need to run `prepare_corpus.py` to compute and save the vocabulary before training a model:
```
# Wiki-103:
python prepare_corpus.py --dataset wt103 --datadir wiki103_datadir

# Billion Word:
python prepare_corpus.py --dataset lm1b --datadir lm1b_datadir
```
<br>

### Enwik8
(1) Train & eval base model with 108M parameters:
```
# train using 8 GPUs and mixed precision
python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234
       train_enwik8.py --log tensorboard_log_dir_for_train_run
                       --data enwik8_file
                       --save base_model
                       --layer_norm
                       --fp16

# evaluate with max attention window size 3072
python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234
       train_enwik8.py --log tensorboard_log_dir_for_test_run
                       --data enwik8_file
                       --load base_model.pt
                       --layer_norm
                       --max_iter 0
                       --eval_unroll_size 3072
```
(2) Train & eval large model with 191M parameters:
```
# train using 8 GPUs and mixed precision
python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234
       train_enwik8.py --log tensorboard_log_dir_for_train_run
                       --data enwik8_file
                       --save large_model
                       --n_d 4096
                       --n_proj 1024
                       --dropout 0.32
                       --attn_dropout 0.32
                       --layer_norm
                       --batch_size 8
                       --lr 0.0004
                       --fp16
                       
# evaluate with max attention window size 3072                    
python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234
       train_enwik8.py --log tensorboard_log_dir_for_test_run
                       --data enwik8_file
                       --load large_model.pt
                       --n_d 4096
                       --n_proj 1024
                       --layer_norm
                       --max_iter 0
                       --eval_unroll_size 3072
```
<br>

### Wiki-103
(1) Train & eval base model with 148M parameters:
```
# train using 8 GPUs and mixed precision
python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234
       train_wt103.py --log tensorboard_log_dir_for_train_run
                      --data wiki103_datadir
                      --save base_model
                      --layer_norm
                      --fp16

# evaluate with max attention window size 2560
python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234
       train_wt103.py --log tensorboard_log_dir_for_test_run
                      --data wiki103_datadir
                      --load base_model.pt
                      --layer_norm
                      --max_iter 0
                      --eval_unroll_size 2560
```
(2) Train & eval large model with 232M parameters:
```
# train using 8 GPUs and mixed precision
python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234
       train_wt103.py --log tensorboard_log_dir_for_train_run
                      --data wiki103_datadir
                      --save large_model
                      --n_d 4096
                      --n_proj 1024
                      --dropout 0.2
                      --attn_dropout 0.2
                      --emb_dropout 0.2
                      --layer_norm
                      --unroll_size 1024
                      --fp16

# evaluate with max attention window size 2560
python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234
       train_wt103.py --log tensorboard_log_dir_for_test_run
                      --data wiki103_datadir
                      --load large_model.pt
                      --n_d 4096
                      --n_proj 1024
                      --layer_norm
                      --max_iter 0
                      --eval_unroll_size 2560
```
<br>

### Billion Word
Train and eval the base model with 328M parameters:
```
# train using 8 GPUs and mixed precision
python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234
       train_lm1b.py --log tensorboard_log_dir_for_train_run
                     --data lm1b_datadir
                     --save base_model
                     --layer_norm
                     --fp16
                     
# evaluate with max attention window size 96
python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234
       train_lm1b.py --log tensorboard_log_dir_for_test_run
                     --data lm1b_datadir
                     --load base_model.pt
                     --layer_norm
                     --max_iter 0
                     --eval_unroll_size 96
```
