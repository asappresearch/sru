import sys
import os
import argparse
import time
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import sru

def read_corpus(path, num_test_symbols=5000000):
    raw_data = open(path).read()
    raw_data = np.fromstring(raw_data, dtype=np.uint8)
    unique, data = np.unique(raw_data, return_inverse=True)
    train_data = data[: -2 * num_test_symbols]
    valid_data = data[-2 * num_test_symbols: -num_test_symbols]
    test_data = data[-num_test_symbols:]
    return train_data, valid_data, test_data, unique

def create_batches(data_ids, batch_size):
    N = len(data_ids)
    L = ((N-1) // batch_size) * batch_size
    x = np.copy(data_ids[:L].reshape(batch_size,-1).T)
    y = np.copy(data_ids[1:L+1].reshape(batch_size,-1).T)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    x, y = x.contiguous(), y.contiguous()
    x, y = x.cuda(), y.cuda()
    return x, y

class HardConcrete(nn.Module):
    def __init__(self, n_in, init_mean=0.5, temperature=1.0, stretch=0.1):
        super(HardConcrete, self).__init__()
        self.n_in = n_in
        self.limit_l = -stretch
        self.limit_r = 1.0+stretch
        self.log_alpha = nn.Parameter(torch.Tensor(n_in))
        self.beta = temperature
        self.init_mean = init_mean
        self.compiled_mask = None
        self.reset_parameters()

    def reset_parameters(self):
        self.log_alpha.data.normal_(math.log(1 - self.init_mean)
                - math.log(self.init_mean), 1e-2)

    def constrain_parameters(self, val):
        self.log_alpha.data.clamp_(min=-val, max=val)

    def l0_norm(self):
        bias = -self.beta * math.log(-self.limit_l/self.limit_r)
        return (self.log_alpha + bias).sigmoid().sum()

    def compile_mask(self, renormalize=True):
        # expected number of 0s
        expected_num_zeros = self.n_in - self.l0_norm().item()
        num_zeros = min(int(expected_num_zeros), self.n_in-1)
        #print(num_zeros)

        # approx expected value of each mask variable z; magic number 0.8
        soft_mask = F.sigmoid(self.log_alpha*0.8)

        # prune small values
        _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
        if renormalize:
            fraction = 1.0 - soft_mask[indices].sum()/(soft_mask.sum() + 1e-9)
            soft_mask /= fraction
        soft_mask[indices] = 0.
        self.compiled_mask = soft_mask

    def reset_mask(self):
        self.compiled_mask = None

    def forward(self, eps=1e-6, mode=4):
        # mode = 1: always sample
        # mode = 2: hard sigmoid
        # mode = 3: compiled mask (normalize)
        # mode = 4: compiled mask (not normalized)
        if self.training or (mode == 1):
            u = self.log_alpha.new(self.n_in).uniform_(eps, 1-eps)
            s = F.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) /
                    self.beta)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
            return s.clamp(min=0., max=1.)
        elif mode == 2:
            #s = F.sigmoid(self.log_alpha / self.beta)
            s = F.sigmoid(self.log_alpha)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
            return s.clamp(min=0., max=1.)
        else:
            if self.compiled_mask is None:
                self.compile_mask(renormalize=(mode == 3))
            return self.compiled_mask

class Model(nn.Module):
    def __init__(self, words, args):
        super(Model, self).__init__()
        self.args = args
        if args.n_e:
            self.n_e = args.n_e
        else:
            self.n_e = len(words) if len(words) < args.n_d else args.n_d
        self.n_d = args.n_d
        self.depth = args.depth
        self.drop = nn.Dropout(args.dropout)
        self.embedding_layer = nn.Embedding(len(words), self.n_e)
        self.n_V = len(words)
        if args.lstm:
            self.rnn = nn.LSTM(self.n_e, self.n_d,
                self.depth,
                dropout = args.dropout
            )
        else:
            self.rnn = sru.SRU(self.n_e, self.n_d, self.depth,
                dropout = args.dropout,# if not args.prune else 0.,
                n_proj = args.n_proj,
                #use_tanh = 0,
                highway_bias = args.bias,
                layer_norm = args.layer_norm
            )
        self.output_layer = nn.Linear(self.n_d, self.n_V)
        self.mask_layers = None
        if args.prune:
            self.mask_layers = nn.ModuleList([ HardConcrete(
                n_in = r.n_proj if r.n_proj else r.n_in,
                init_mean = args.prune_mean,
                stretch = args.prune_stretch)
                for r in self.rnn.rnn_lst ])

        self.init_weights()

    def init_weights(self, val_range=None):
        #val_range = val_range or (3.0/self.n_d)**0.5
        params = list(self.embedding_layer.parameters()) + list(self.output_layer.parameters()) \
                + (list(self.rnn.parameters()) if self.args.lstm else [])
        for p in params:
            if p.dim() > 1:  # matrix
                val = val_range or (3.0/p.size(0))**0.5
                p.data.uniform_(-val, val)
            else:
                p.data.zero_()

    def forward(self, x, hidden, use_mask=True, mode=2):
        emb = self.drop(self.embedding_layer(x))
        if not use_mask:
            masks = None
            output, hidden = self.rnn(emb, hidden)
        else:
            masks = [ layer(mode=mode) for layer in self.mask_layers ]
            h0 = [ h.squeeze(0) for h in hidden.chunk(self.depth, 0) ]
            lst_ht = []
            prev = emb
            for i, r in enumerate(self.rnn.rnn_lst):
                prev, ht = r(prev, h0[i], dim_mask=masks[i])
                lst_ht.append(ht)
            output, hidden = prev, torch.stack(lst_ht)
        output = self.drop(output)
        output = output.view(-1, output.size(2))
        output = self.output_layer(output)
        return output, hidden, masks

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        zeros = Variable(weight.new(self.depth, batch_size, self.n_d).zero_())
        if self.args.lstm:
            return (zeros, zeros)
        else:
            return zeros

def calc_norm(lis):
    l2_sum = sum(x.norm()**2 for x in lis)
    return l2_sum**0.5

def reset_hidden(hidden, p=0.01, lstm=False):
    w = hidden.data if not lstm else hidden[0].data  # (depth, batch_size, d)
    bs = w.size(1)
    mask = Variable(w.new(1, bs, 1).bernoulli_(1-p))
    if not lstm:
        return hidden*mask
    else:
        return (hidden[0]*mask, hidden[1]*mask)

def eval_model(model, valid, use_mask, mode=2):
    with torch.no_grad():
        for layer in model.mask_layers:
            layer.reset_mask()
        model.eval()
        args = model.args
        batch_size = valid[0].size(1)
        total_loss = 0.0
        unroll_size = args.unroll_size
        criterion = nn.CrossEntropyLoss(size_average=False)
        hidden = model.init_hidden(batch_size)
        N = (len(valid[0])-1)//unroll_size + 1
        for i in range(N):
            x = valid[0][i*unroll_size:(i+1)*unroll_size]
            y = valid[1][i*unroll_size:(i+1)*unroll_size].view(-1)
            if args.lstm:
                hidden[0].detach_()
                hidden[1].detach_()
            else:
                hidden.detach_()
            output, hidden, masks = model(x, hidden, use_mask=use_mask, mode=mode)
            loss = criterion(output, y)
            total_loss += loss.item()  # loss.data[0]
        avg_loss = total_loss / valid[1].numel()
        ppl = np.exp(avg_loss)
        if use_mask:
            sparsity = sum(x.le(1e-6).sum().item() for x in masks) \
                    / sum(x.numel() for x in masks)
        else:
            sparsity = 0
        model.train()
        return ppl, avg_loss, sparsity

def copy_model(model):
    states = model.state_dict()
    for k in states:
        v = states[k]
        states[k] = v.clone().cpu()
    return states

def clip_hardconcrete_param(model, value):
    if model.mask_layers:
        for layer in model.mask_layers:
            layer.constrain_parameters(val=value)

def main(args):
    train, dev, test, words  = read_corpus(args.data)
    log_path = "{}_{}".format(args.log, random.randint(1,100))
    train_writer = SummaryWriter(log_dir=log_path+"/train")
    dev_writer = SummaryWriter(log_dir=log_path+"/dev")

    model = Model(words, args)
    if args.load:
        mask_layers = model.mask_layers
        model.mask_layers = None
        model.load_state_dict(torch.load(args.load))
        model.mask_layers = mask_layers
    model.cuda()
    print (model)
    sys.stdout.write("vocab size: {}\n".format(
        model.n_V
    ))
    sys.stdout.write("num of parameters: {}\n".format(
        sum(x.numel() for x in model.parameters() if x.requires_grad)
    ))
    sys.stdout.write("\n")

    dev_, test_ = dev, test
    train = create_batches(train, args.batch_size)
    dev = create_batches(dev, args.batch_size)
    test = create_batches(test, args.batch_size)
    lr = 1.0 if not args.noam else 1.0/(args.n_d**0.5)/(args.warmup_steps**1.5)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = lr * args.lr,
        weight_decay = args.weight_decay
    )
    lambda_ = nn.Parameter(torch.tensor(0.).cuda())
    optimizer_max = optim.Adam(
        [lambda_],
        lr = lr,
        weight_decay = 0
    )
    optimizer_max.param_groups[0]['lr'] = -lr * args.prune_lr
    print(lambda_)

    plis = [ p for p in model.parameters() if p.requires_grad ]
    niter = 1
    unchanged = 0
    best_dev = 1e+8
    unroll_size = args.unroll_size
    batch_size = args.batch_size
    N = (len(train[0])-1)//unroll_size + 1
    criterion = nn.CrossEntropyLoss()
    if not args.prune:
        args.prune_start_epoch = args.max_epoch
        mask_cnt = 0
    else:
        mask_cnt = sum(layer.n_in for layer in model.mask_layers)

    for epoch in range(args.max_epoch):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        hidden = model.init_hidden(batch_size)
        use_mask = epoch >= args.prune_start_epoch

        for i in range(N):
            x = train[0][i*unroll_size:(i+1)*unroll_size]
            y = train[1][i*unroll_size:(i+1)*unroll_size].view(-1)
            x, y =  Variable(x), Variable(y)
            if args.lstm:
                hidden[0].detach_()
                hidden[1].detach_()
            else:
                hidden.detach_()

            model.zero_grad()
            optimizer_max.zero_grad()
            output, hidden, masks = model(x, hidden, use_mask=use_mask)
            loss = criterion(output, y)
            loss.backward()
            loss = loss.item()
            lagrangian_loss = 0
            if use_mask:
                if args.prune_warmup == 0:
                    target_sparsity = args.prune_sparsity
                else:
                    prune_warmup = max(1, args.prune_warmup)
                    target_sparsity = args.prune_sparsity * min(1.0, niter / prune_warmup)
                l0_norm = 0
                for layer in model.mask_layers:
                    l0_norm = l0_norm + layer.l0_norm()
                expected_sparsity = 1. - (l0_norm / mask_cnt)
                lagrangian_loss = lambda_ * (expected_sparsity - target_sparsity) #**2
                #lagrangian_loss = lambda_ * (mask_cnt - l0_norm - target_sparsity * mask_cnt)
                lagrangian_loss.backward()
                expected_sparsity = expected_sparsity.item()
                #expected_sparsity = 1. - (l0_norm.item() / mask_cnt)
                lagrangian_loss = lagrangian_loss.item()
                optimizer_max.step()
            elif model.mask_layers:
                model.mask_layers.zero_grad()

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            optimizer.step()

            if args.prune_clipping > 0:
                clip_hardconcrete_param(model, args.prune_clipping)

            if (niter - 1) % 100 == 0:
                if use_mask:
                    sparsity = sum(x.le(1e-6).sum().item() for x in masks)/mask_cnt
                    train_writer.add_scalar('sparsity',
                        sparsity,
                        niter
                    )
                    train_writer.add_scalar('expected_sparsity',
                        expected_sparsity,
                        niter
                    )
                    train_writer.add_scalar('target_sparsity',
                        target_sparsity,
                        niter
                    )
                    train_writer.add_scalar('lagrangian_loss',
                        lagrangian_loss,
                        niter
                    )
                    train_writer.add_scalar('lambda',
                        lambda_.item(),
                        niter
                    )
                    if (niter - 1) % 5000 == 0:
                        for index, mask in enumerate(masks):
                            train_writer.add_histogram(
                                'mask/{}'.format(index),
                                mask,
                                niter,
                                bins='sqrt',
                            )
                        for index, layer in enumerate(model.mask_layers):
                            train_writer.add_histogram(
                                'log_alpha/{}'.format(index),
                                layer.log_alpha,
                                niter,
                                bins='sqrt',
                            )
                sys.stderr.write("\r{:.4f} {:.2f}/{:.2f} {:.2f}".format(
                    loss,
                    lagrangian_loss,
                    expected_sparsity,
                    sparsity
                ))
                train_writer.add_scalar('loss', loss, niter)
                train_writer.add_scalar('loss_tot', loss + lagrangian_loss, niter)
                train_writer.add_scalar('pnorm',
                    calc_norm([ x.data for x in plis ]),
                    niter
                )
                train_writer.add_scalar('gnorm',
                    calc_norm([ x.grad for x in plis if x.grad is not None]),
                    niter
                )

            if niter%args.log_period == 0 or i == N - 1:
                elapsed_time = (time.time()-start_time)/60.0
                dev_ppl, dev_loss, sparsity = eval_model(model, dev,
                        use_mask=use_mask, mode=4)
                sys.stdout.write("\rIter={}  lr={:.5f}  train_loss={:.4f}  dev_loss={:.4f}"
                        "  dev_bpc={:.2f}  sparsity={:.2f}\teta={:.1f}m\t[{:.1f}m]\n".format(
                    niter,
                    optimizer.param_groups[0]['lr'],
                    loss,
                    dev_loss,
                    np.log2(dev_ppl),
                    sparsity,
                    elapsed_time*N/(i+1),
                    elapsed_time
                ))
                if dev_ppl < best_dev:
                    best_dev = dev_ppl
                    checkpoint = copy_model(model)
                sys.stdout.write("\n")
                sys.stdout.flush()
                dev_writer.add_scalar('loss', dev_loss, niter)
                dev_writer.add_scalar('bpc/topk_unnorm', np.log2(dev_ppl), niter)
                dev_writer.add_scalar('sparsity/topk_unnorm', sparsity, niter)
                #if use_mask:
                    #dev_ppl, dev_loss, sparsity = eval_model(model, dev,
                    #        use_mask=use_mask, mode=1)
                    #dev_writer.add_scalar('bpc/sample', np.log2(dev_ppl), niter)
                    #dev_writer.add_scalar('sparsity/sample', sparsity, niter)
                    #dev_ppl, dev_loss, sparsity = eval_model(model, dev,
                    #        use_mask=use_mask, mode=3)
                    #dev_writer.add_scalar('bpc/topk_norm', np.log2(dev_ppl), niter)
                    #dev_writer.add_scalar('sparsity/topk_norm', sparsity, niter)
                    #dev_ppl, dev_loss, sparsity = eval_model(model, dev,
                    #        use_mask=use_mask, mode=4)
                    #dev_writer.add_scalar('bpc/topk_unnorm', np.log2(dev_ppl), niter)
                    #dev_writer.add_scalar('sparsity/topk_unnorm', sparsity, niter)
            niter += 1
            if args.noam:
                if niter >= args.warmup_steps:
                    lr = 1.0/(args.n_d**0.5)/(niter**0.5)
                else:
                    lr = (1.0/(args.n_d**0.5)/(args.warmup_steps**1.5))*niter
                optimizer.param_groups[0]['lr'] = lr * args.lr
                optimizer_max.param_groups[0]['lr'] = -lr  * args.prune_lr


        if args.save and (epoch + 1) % 10 == 0:
            torch.save(checkpoint, "{}.{}.pt".format(
                args.save,
                epoch + 1
            ))

    train_writer.close()
    dev_writer.close()

    model.load_state_dict(checkpoint)
    model.cuda()
    dev = create_batches(dev_, 1)
    test = create_batches(test_, 1)
    dev_ppl, dev_loss, sparsity  = eval_model(model, dev, use_mask=use_mask, mode=4)
    test_ppl, test_loss, sparsity = eval_model(model, test, use_mask=use_mask, mode=4)
    sys.stdout.write("dev_bpc={:.3f}  test_bpc={:.3f}  sparsity={:.2f}\n".format(
        np.log2(dev_ppl), np.log2(test_ppl), sparsity
    ))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--log", type=str, required=True)
    argparser.add_argument("--noam", action="store_true")
    argparser.add_argument("--warmup_steps", type=int, default=32000)
    argparser.add_argument("--layer_norm", action="store_true")
    argparser.add_argument("--lstm", action="store_true")
    argparser.add_argument("--data", type=str, required=True, help="training file")
    argparser.add_argument("--batch_size", "--batch", type=int, default=128)
    argparser.add_argument("--unroll_size", type=int, default=100)
    argparser.add_argument("--max_epoch", type=int, default=100)
    argparser.add_argument("--n_e", type=int, default=0)
    argparser.add_argument("--n_d", "--d", type=int, default=1024)
    argparser.add_argument("--n_proj", type=int, default=0)
    argparser.add_argument("--dropout", type=float, default=0.2,
        help="dropout probability"
    )
    argparser.add_argument("--bias", type=float, default=-3,
        help="intial bias of highway gates",
    )
    argparser.add_argument("--depth", type=int, default=6)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--weight_decay", type=float, default=1e-7)
    argparser.add_argument("--clip_grad", type=float, default=0.3)
    argparser.add_argument("--log_period", type=int, default=1000000)
    argparser.add_argument("--save", type=str, default="")
    argparser.add_argument("--load", type=str, default="")

    argparser.add_argument("--prune", action="store_true")
    argparser.add_argument("--prune_lr", type=float, default=3)
    argparser.add_argument("--prune_warmup", type=int, default=0)
    argparser.add_argument("--prune_sparsity", type=float, default=0.)
    argparser.add_argument("--prune_stretch", type=float, default=0.1)
    argparser.add_argument("--prune_mean", type=float, default=0.5)
    argparser.add_argument("--prune_clipping", type=float, default=0)
    argparser.add_argument("--prune_start_epoch", type=int, default=0)

    args = argparser.parse_args()
    print (args)
    main(args)
