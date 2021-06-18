import sys
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter

from radam import RAdam
from sru import SRUpp

from embedding import AdaptiveEmbedding, AdaptiveLogSoftmax
from data_utils import get_lm_corpus


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.cutoffs = [59998, 159998]
        self.n_V = args.n_token
        self.n_e = args.n_e or args.n_proj
        self.n_d = args.n_d
        self.depth = args.depth
        self.drop = nn.Dropout(args.dropout)
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
            attn_layer_norm=True,
            attention_every_n_layers=args.attn_every_n_layers,
        )
        self.output_layer = AdaptiveLogSoftmax(
            self.n_V,
            self.n_e,
            self.n_d,
            self.cutoffs,
            div_val=args.div_val,
            dropout=args.emb_dropout,
            keep_order=False
        )
        shared_embs = [layer.weight for layer in self.output_layer.out_layers]
        shared_projs = None  # do not share projections in lm1b benchmark, following prior work
        self.embedding_layer = AdaptiveEmbedding(
            self.n_V,
            self.n_e,
            self.n_d,
            self.cutoffs,
            div_val=args.div_val,
            dropout=args.emb_dropout,
            emb_weights=shared_embs,
            proj_weights=shared_projs,
            scale_emb=not args.layer_norm,
        )
        self.init_weights()

    def init_weights(self):
        def _init_range(m):
            for p in m.parameters():
                if p.dim() > 1:
                    p.data.uniform_(-init_range, init_range)
                else:
                    p.data.zero_()

        def _init_xavier(m):
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    p.data.zero_()
        init_range = (3.0 / self.n_d) ** 0.5
        _init_range(self.output_layer.out_layers)
        _init_xavier(self.output_layer.out_projs)
        _init_xavier(self.embedding_layer.emb_projs)

    def set_mask(self, mem_length, same_length=False):
        weight = next(self.parameters())
        ones = weight.new_ones(mem_length * 2, mem_length * 2)
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
        output = output.view(-1, output.size(2))
        loss = self.output_layer(output, y.view(-1))
        loss = loss.view(y.size(0), -1)
        return loss, hidden, memory

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        zeros = weight.new(self.depth, batch_size, self.n_d).zero_()
        return zeros


def calc_norm(lis):
    l2_sum = sum(x.norm()**2 for x in lis)
    return l2_sum**0.5


def eval_model(model, valid):
    with torch.no_grad():
        args = model.args
        batch_size = args.eval_batch_size or args.batch_size
        unroll_size = args.eval_unroll_size or args.unroll_size
        model.set_mask(unroll_size, same_length=True)
        memory = [None] * args.depth
        model.eval()
        hidden = model.init_hidden(batch_size)
        total_loss = 0.0
        total_tok = 0.0
        for x, y, seq_len in valid:
            if torch.is_autocast_enabled():
                torch.clear_autocast_cache()
            loss, hidden, memory = model(x, y, hidden, memory=memory)
            total_loss += loss.sum()
            total_tok += y.numel()
        avg_loss = total_loss.item() / total_tok
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

    corpus = get_lm_corpus(args.data, 'lm1b')
    n_token = args.n_token = len(corpus.vocab)
    args.eval_batch_size = args.eval_batch_size or args.batch_size
    args.eval_unroll_size = args.eval_unroll_size or args.unroll_size
    unroll_size = args.unroll_size
    eval_unroll_size = args.eval_unroll_size
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    train = corpus.get_distributed_iterator('train', batch_size,
                                            unroll_size, n_nodes=n_nodes,
                                            rank=local_rank, device=device)
    dev = corpus.get_iterator('valid', eval_batch_size, eval_unroll_size, device=device)

    model = Model(args)
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
        broadcast_buffers=False,
    )

    if local_rank == 0:
        print(model)
        print("vocab size: {}".format(n_token))
        print("num of training iters: {}".format(args.max_iter))
        num_params = sum(x.numel() for x in parameters if x.requires_grad)
        print("num of parameters: {}".format(num_params))
        print("num of processes / world size: {}".format(n_nodes))

    nbatch = 0
    niter = 0
    best_dev = 1e+8
    checkpoint = None
    total_loss = []
    model.train()
    start_time = time.time()
    hidden = model_.init_hidden(batch_size)
    memory = [None] * args.depth

    for x, y, seq_len in train:
        if niter == args.max_iter:
            break
        nbatch += 1

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
        memory = [m.detach() if m is not None else None for m in memory]
        loss.detach_()
        if len(total_loss) < args.eval_period:
            total_loss.append(loss)
        else:
            total_loss[nbatch % args.eval_period] = loss

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
                    sys.stderr.write("\r{:.2f} eta={:.1f}m / {:.1f}m    ".format(
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
                torch.cuda.empty_cache()
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    dev_ppl, dev_loss = eval_model(model_, dev)
                dev_writer.add_scalar('loss/lm_loss', dev_loss, niter)
                dev_writer.add_scalar('loss/avg_loss', dev_loss, niter)
                dev_writer.add_scalar('loss/ppl', dev_ppl, niter)
                dev_writer.add_scalar('ppl', dev_ppl, niter)
                elapsed_time = (time.time() - start_time) / 60.0
                start_time = time.time()
                sys.stdout.write(
                    "\rnum_iters={}  lr={:.6f}  train_loss={:.4f}  dev_loss={:.4f}"
                    "  dev_bpc={:.2f}\t[{:.1f}m]\n\n".format(
                        niter,
                        optimizer.param_groups[0]['lr'],
                        loss,
                        dev_loss,
                        dev_ppl,
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
        args.eval_batch_size = 1
        dev = corpus.get_iterator('valid', 1, eval_unroll_size, device=device)
        test = corpus.get_iterator('test', 1, eval_unroll_size, device=device)
        dev_ppl, dev_loss = eval_model(model_, dev)
        test_ppl, test_loss = eval_model(model_, test)
        sys.stdout.write("dev_ppl={:.3f}  test_ppl={:.3f}\n".format(
            dev_ppl, test_ppl
        ))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    # load and save options
    argparser.add_argument("--log", type=str, required=True)
    argparser.add_argument("--data", type=str, required=True, help="training file")
    argparser.add_argument("--save", type=str, default="")
    argparser.add_argument("--load", type=str, default="")

    # model configs
    argparser.add_argument("--n_d", "--d", type=int, default=4096)
    argparser.add_argument("--n_proj", type=int, default=1024)
    argparser.add_argument("--n_e", type=int, default=1024,
                           help="inner dimension used in adaptive emb / softmax")
    argparser.add_argument("--div_val", type=float, default=4)
    argparser.add_argument("--depth", type=int, default=10)
    argparser.add_argument("--layer_norm", action="store_true")
    argparser.add_argument("--dropout", type=float, default=0.05, help="dropout probability")
    argparser.add_argument("--emb_dropout", type=float, default=0)
    argparser.add_argument("--attn_dropout", type=float, default=0)
    argparser.add_argument("--attn_heads", type=int, default=1)
    argparser.add_argument("--attn_every_n_layers", type=int, default=1)
    argparser.add_argument("--bias", type=float, default=-2,
                           help="intial bias of highway gates")

    # training configs
    argparser.add_argument("--seed", type=int, default=1234, help="random seed")
    argparser.add_argument("--fp16", action="store_true")
    argparser.add_argument("--warmup_steps", type=int, default=16000)
    argparser.add_argument("--batch_size", "--batch", type=int, default=128)
    argparser.add_argument("--eval_batch_size", type=int, default=8)
    argparser.add_argument("--update_param_freq", type=int, default=1)
    argparser.add_argument("--unroll_size", type=int, default=64)
    argparser.add_argument("--eval_unroll_size", type=int, default=64)
    argparser.add_argument("--max_iter", type=int, default=800000)
    argparser.add_argument("--lr", type=float, default=0.0002)
    argparser.add_argument("--weight_decay", type=float, default=0.1)
    argparser.add_argument("--clip_grad", type=float, default=1.0)
    argparser.add_argument("--log_period", type=int, default=200)
    argparser.add_argument("--eval_period", type=int, default=8000)

    # distributed data parallel local_rank
    argparser.add_argument("--local_rank", type=int, default=0)

    args = argparser.parse_args()
    if args.local_rank == 0:
        print(args)
    main(args)
