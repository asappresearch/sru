import sys
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter

from sru import SRUpp
from radam import RAdam


class Model(nn.Module):
    def __init__(self, words, args):
        super(Model, self).__init__()
        self.args = args
        self.n_d = args.n_d
        self.depth = args.depth
        self.drop = nn.Dropout(args.dropout)
        self.embedding_layer = nn.Embedding(len(words), self.n_d)
        self.n_V = len(words)
        self.rnn = SRUpp(
            self.n_d,
            self.n_d,
            args.n_proj,
            num_layers=args.depth,
            highway_bias=args.bias,
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
            num_heads=args.attn_heads,
            layer_norm=args.layer_norm,
            attention_every_n_layers=args.attn_every_n_layers,
        )
        self.output_layer = nn.Linear(self.n_d, self.n_V)
        self.init_weights()

    def init_weights(self):
        params = list(self.embedding_layer.parameters()) + list(self.output_layer.parameters())
        new_init_version = self.args.layer_norm
        for p in params:
            if p.dim() > 1:  # matrix
                # keep old init version for reproducibility
                val = (3.0/self.n_d)**0.5 if new_init_version else (3.0/p.size(0))**0.5
                p.data.uniform_(-val, val)
            else:
                p.data.zero_()

    def set_mask(self, mem_length, same_length=False):
        ones = self.embedding_layer.weight.new_ones(mem_length * 2, mem_length * 2)
        if same_length:
            ''' example: mem_length = 3
                0 1 1 1 1 1
                0 0 1 1 1 1
                0 0 0 1 1 1
                0 0 0 0 1 1
                1 0 0 0 0 1
                1 1 0 0 0 0
            '''
            self.attn_mask = (torch.triu(ones, diagonal=1) +
                              torch.tril(ones, diagonal=-1-mem_length)) * -10000.0
        else:
            self.attn_mask = torch.triu(ones, diagonal=1) * -10000.0

    def forward(self, x, y, hidden, memory):
        input_len = x.size(0)
        mem_len = 0 if memory[0] is None else memory[0].size(0)
        attn_mask = self.attn_mask[mem_len:mem_len + input_len, :mem_len + input_len]
        attn_mask = attn_mask.to(x)
        emb = self.drop(self.embedding_layer(x))
        output, hidden, memory_dict = self.rnn(
            emb, hidden,
            memory=memory,
            attn_mask=attn_mask
        )
        memory = memory_dict['saved_inputs']
        output = self.drop(output)
        output = self.output_layer(output)
        output = output.view(-1, output.size(2))
        loss = F.cross_entropy(output, y.view(-1), reduction='none')
        loss = loss.view(y.size(0), y.size(1))
        return loss, hidden, memory

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        zeros = weight.new(self.depth, batch_size, self.n_d).zero_()
        return zeros


def read_corpus(path, num_test_symbols=5000000):
    raw_data = open(path).read()
    raw_data = np.fromstring(raw_data, dtype=np.uint8)
    unique, data = np.unique(raw_data, return_inverse=True)
    train_data = data[: -2 * num_test_symbols]
    valid_data = data[-2 * num_test_symbols: -num_test_symbols]
    test_data = data[-num_test_symbols:]
    return train_data, valid_data, test_data, unique


def create_batches(data_ids, batch_size, n_nodes=1, rank=0, device='cpu'):
    N = len(data_ids)
    L = (N-1) // (batch_size * n_nodes) * batch_size * n_nodes
    x = np.copy(data_ids[:L].reshape(n_nodes, batch_size, -1)[rank].T)
    y = np.copy(data_ids[1:L+1].reshape(n_nodes, batch_size, -1)[rank].T)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    x, y = x.contiguous(), y.contiguous()
    x, y = x.to(device), y.to(device)
    return x, y


def calc_norm(lis):
    l2_sum = sum(x.norm()**2 for x in lis)
    return l2_sum**0.5


def eval_model(model, valid):
    with torch.no_grad():
        model.eval()
        args = model.args
        batch_size = valid[0].size(1)
        total_loss = 0.0
        unroll_size = args.eval_unroll_size or args.unroll_size
        model.set_mask(unroll_size, same_length=True)
        hidden = model.init_hidden(batch_size)
        memory = [None] * args.depth
        N = (len(valid[0])-1)//unroll_size + 1
        for i in range(N):
            x = valid[0][i*unroll_size:(i+1)*unroll_size]
            y = valid[1][i*unroll_size:(i+1)*unroll_size]
            loss, hidden, memory = model(x, y, hidden, memory=memory)
            total_loss += loss.sum()
        avg_loss = total_loss.item() / valid[1].numel()
        ppl = np.exp(avg_loss)
        model.train()
        model.set_mask(args.unroll_size, same_length=False)
        return ppl, avg_loss


def copy_model(model):
    states = model.state_dict()
    for k in states:
        v = states[k]
        states[k] = v.clone().cpu()
    return states


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):

    if args.local_rank == 0:
        log_path = "{}_{}".format(args.log, random.randint(1, 100))
        train_writer = SummaryWriter(log_dir=log_path+"/train")
        dev_writer = SummaryWriter(log_dir=log_path+"/dev")

    # set up distributed training
    set_seed(args.seed)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.device = device
    local_rank = args.local_rank
    n_nodes = torch.distributed.get_world_size()

    train, dev, test, words = read_corpus(args.data)
    dev_, test_ = dev, test
    unroll_size = args.unroll_size
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size or args.batch_size
    train = create_batches(train, batch_size,
                           n_nodes=n_nodes,
                           rank=local_rank,
                           device=device)
    if local_rank == 0:
        dev = create_batches(dev, eval_batch_size, device=device)
    N = (len(train[0])-1)//unroll_size + 1

    model = Model(words, args)
    if args.load:
        model.load_state_dict(torch.load(args.load))
    model.to(device)
    model.set_mask(unroll_size)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = RAdam(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        args.max_iter,
    )

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    model_ = model
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        dim=1,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    if local_rank == 0:
        print(model_)
        print("vocab size: {}".format(model_.n_V))
        print("num of mini-batches per epoch: {}".format(N))
        print("num of training iters: {}".format(args.max_iter))
        num_params = sum(x.numel() for x in parameters)
        print("num of parameters: {}".format(num_params))
        print("num of processes / world size: {}".format(n_nodes))

    nbatch = 0
    niter = 0
    best_dev = 1e+8
    checkpoint = None
    total_loss = []
    start_time = time.time()

    while niter < args.max_iter:
        model.train()
        hidden = model_.init_hidden(batch_size)
        memory = [None] * args.depth

        for i in range(N):
            if niter >= args.max_iter:
                break
            nbatch += 1
            x = train[0][i*unroll_size:(i+1)*unroll_size]
            y = train[1][i*unroll_size:(i+1)*unroll_size]

            # language model forward and backward
            if not args.fp16:
                loss, hidden, memory = model(x, y, hidden, memory=memory)
                loss = loss.mean()
                (loss / args.update_param_freq).backward()
            else:
                with torch.cuda.amp.autocast():
                    loss, hidden, memory = model(x, y, hidden, memory=memory)
                    loss = loss.mean()
                    scaler.scale(loss / args.update_param_freq).backward()

            hidden.detach_()
            memory = [m.detach_() if m is not None else None for m in memory]
            loss.detach_()
            if len(total_loss) < N:
                total_loss.append(loss)
            else:
                total_loss[i-1] = loss

            #  perform gradient decent every few number of backward()
            if nbatch % args.update_param_freq == 0:
                # learning rate warmup
                if niter < args.warmup_steps:
                    lr_t = args.lr * (niter + 1) / args.warmup_steps
                    optimizer.param_groups[0]['lr'] = lr_t

                # unscale gradient for clipping
                if args.clip_grad > 0 and args.fp16:
                    scaler.unscale_(optimizer)

                # log training stats
                if local_rank == 0 and niter % args.log_period == 0:
                    with torch.no_grad():
                        loss = loss.item()
                        num_processed = niter % args.eval_period + 1
                        eps = num_processed / (time.time() - start_time)
                        sys.stderr.write("\r{:.2f} eta={:.1f}m / {:.1f}m".format(
                            loss,
                            (args.eval_period - num_processed) / eps / 60.0,
                            args.eval_period / eps / 60.0,
                        ))
                        train_writer.add_scalar('loss/lm_loss', loss, niter)
                        train_writer.add_scalar('loss/avg_loss',
                                                sum(total_loss).item()/len(total_loss),
                                                niter)
                        train_writer.add_scalar(
                            'parameter_norm',
                            calc_norm([x.data for x in parameters]),
                            niter
                        )
                        train_writer.add_scalar(
                            'gradient_norm',
                            calc_norm([x.grad for x in parameters if x.grad is not None]),
                            niter
                        )

                # gradient clipping after the gradient is logged
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad)

                # gradient descent
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                niter += 1

                if local_rank == 0 and (niter == args.max_iter or niter % args.eval_period == 0):
                    dev_ppl, dev_loss = eval_model(model_, dev)
                    dev_writer.add_scalar('loss/lm_loss', dev_loss, niter)
                    dev_writer.add_scalar('loss/avg_loss', dev_loss, niter)
                    dev_writer.add_scalar('bpc', np.log2(dev_ppl), niter)
                    elapsed_time = (time.time() - start_time) / 60.0
                    start_time = time.time()
                    sys.stdout.write(
                        "\rnum_batches={}  lr={:.6f}  train_loss={:.4f}  dev_loss={:.4f}"
                        "  dev_bpc={:.2f}\t[{:.1f}m]\n\n".format(
                            nbatch,
                            optimizer.param_groups[0]['lr'],
                            loss,
                            dev_loss,
                            np.log2(dev_ppl),
                            elapsed_time
                        )
                    )
                    sys.stdout.flush()
                    if args.save and dev_ppl < best_dev:
                        best_dev = dev_ppl
                        checkpoint = copy_model(model_)
                        torch.save(checkpoint, "{}.pt".format(args.save))

    if local_rank == 0:
        train_writer.close()
        dev_writer.close()
        if checkpoint is not None:
            model_.load_state_dict(checkpoint)
            model_.to(device)
        dev = create_batches(dev_, 1, device=device)
        test = create_batches(test_, 1, device=device)
        dev_ppl, dev_loss = eval_model(model_, dev)
        test_ppl, test_loss = eval_model(model_, test)
        sys.stdout.write("dev_bpc={:.3f}  test_bpc={:.3f}\n".format(
            np.log2(dev_ppl), np.log2(test_ppl)
        ))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    # load and save options
    argparser.add_argument("--log", type=str, required=True)
    argparser.add_argument("--data", type=str, required=True, help="training file")
    argparser.add_argument("--save", type=str, default="")
    argparser.add_argument("--load", type=str, default="")

    # model configs
    argparser.add_argument("--n_d", "--d", type=int, default=3072)
    argparser.add_argument("--n_proj", type=int, default=768)
    argparser.add_argument("--depth", type=int, default=10)
    argparser.add_argument("--layer_norm", action="store_true")
    argparser.add_argument("--dropout", type=float, default=0.22, help="dropout probability")
    argparser.add_argument("--attn_dropout", type=float, default=0.22)
    argparser.add_argument("--attn_heads", type=int, default=1)
    argparser.add_argument("--attn_every_n_layers", type=int, default=1)
    argparser.add_argument("--bias", type=float, default=-2,
                           help="intial bias of highway gates")

    # training configs
    argparser.add_argument("--seed", type=int, default=1234, help="random seed")
    argparser.add_argument("--fp16", action="store_true")
    argparser.add_argument("--warmup_steps", type=int, default=16000)
    argparser.add_argument("--batch_size", "--batch", type=int, default=4)
    argparser.add_argument("--eval_batch_size", type=int, default=16)
    argparser.add_argument("--update_param_freq", type=int, default=1)
    argparser.add_argument("--unroll_size", type=int, default=1024)
    argparser.add_argument("--eval_unroll_size", type=int, default=0)
    argparser.add_argument("--max_iter", type=int, default=400000)
    argparser.add_argument("--lr", type=float, default=0.0003)
    argparser.add_argument("--weight_decay", type=float, default=0.1)
    argparser.add_argument("--clip_grad", type=float, default=1.0)
    argparser.add_argument("--log_period", type=int, default=200)
    argparser.add_argument("--eval_period", type=int, default=8000)
    argparser.add_argument("--eval_period", type=int, default=8000)

    # distributed data parallel local_rank
    argparser.add_argument("--local_rank", type=int, default=0)

    args = argparser.parse_args()
    if args.local_rank == 0:
        print(args)
    main(args)
