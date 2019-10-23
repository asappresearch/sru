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
                dropout = args.dropout,
                n_proj = args.n_proj,
                #use_tanh = 0,
                highway_bias = args.bias,
                layer_norm = args.layer_norm
            )
        self.output_layer = nn.Linear(self.n_d, self.n_V)
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

    def forward(self, x, hidden):
        emb = self.drop(self.embedding_layer(x))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        output = output.view(-1, output.size(2))
        output = self.output_layer(output)
        return output, hidden

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

def eval_model(model, valid):
    with torch.no_grad():
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
            x, y = Variable(x, volatile=True), Variable(y, volatile=True)
            if args.lstm:
                hidden[0].detach_()
                hidden[1].detach_()
            else:
                hidden.detach_()
            #hidden = (Variable(hidden[0].data), Variable(hidden[1].data)) if args.lstm \
            #    else Variable(hidden.data)
            output, hidden = model(x, hidden)
            loss = criterion(output, y)
            total_loss += loss.item()  # loss.data[0]
        avg_loss = total_loss / valid[1].numel()
        ppl = np.exp(avg_loss)
        model.train()
    return ppl, avg_loss

def copy_model(model):
    states = model.state_dict()
    for k in states:
        v = states[k]
        states[k] = v.clone()
    return states

def main(args):
    train, dev, test, words  = read_corpus(args.data)
    log_path = "{}_{}".format(args.log, random.randint(1,100))
    train_writer = SummaryWriter(log_dir=log_path+"/train")
    dev_writer = SummaryWriter(log_dir=log_path+"/dev")

    model = Model(words, args)
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
    lr = args.lr if not args.noam else args.lr/(args.n_d**0.5)/(args.warmup_steps**1.5)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = lr,
        weight_decay = args.weight_decay
    )

    plis = [ p for p in model.parameters() if p.requires_grad ]
    niter = 1
    unchanged = 0
    best_dev = 1e+8
    unroll_size = args.unroll_size
    batch_size = args.batch_size
    N = (len(train[0])-1)//unroll_size + 1
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.max_epoch):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        hidden = model.init_hidden(batch_size)

        for i in range(N):
            x = train[0][i*unroll_size:(i+1)*unroll_size]
            y = train[1][i*unroll_size:(i+1)*unroll_size].view(-1)
            x, y =  Variable(x), Variable(y)
            if args.lstm:
                hidden[0].detach_()
                hidden[1].detach_()
            else:
                hidden.detach_()
            #hidden = reset_hidden(hidden, lstm=args.lstm)
            #hidden = (Variable(hidden[0].data), Variable(hidden[1].data)) if args.lstm \
            #    else Variable(hidden.data)

            model.zero_grad()
            output, hidden = model(x, hidden)
            loss = criterion(output, y)
            loss.backward()

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            optimizer.step()

            if (niter - 1) % 100 == 0:
                #sys.stdout.write("\r{}".format(niter))
                #sys.stdout.flush()
                #train_writer.add_scalar('loss', loss.data[0], niter)
                train_writer.add_scalar('loss', loss.item(), niter)
                train_writer.add_scalar('pnorm',
                    calc_norm([ x.data for x in plis ]),
                    niter
                )
                train_writer.add_scalar('gnorm',
                    calc_norm([ x.grad for x in plis ]),
                    niter
                )

            if niter % args.log_period == 0 or i == N - 1:
                elapsed_time = (time.time()-start_time)/60.0
                dev_ppl, dev_loss = eval_model(model, dev)
                sys.stdout.write("\rIter={}  lr={:.5f}  train_loss={:.4f}  dev_loss={:.4f}"
                        "  dev_bpc={:.2f}\teta={:.1f}m\t[{:.1f}m]\n".format(
                    niter,
                    optimizer.param_groups[0]['lr'],
                    loss.item(),  # loss.data[0],
                    dev_loss,
                    np.log2(dev_ppl),
                    elapsed_time*N/(i+1),
                    elapsed_time
                ))
                if dev_ppl < best_dev:
                    best_dev = dev_ppl
                    checkpoint = copy_model(model)
                sys.stdout.write("\n")
                sys.stdout.flush()
                dev_writer.add_scalar('loss', dev_loss, niter)
                dev_writer.add_scalar('bpc', np.log2(dev_ppl), niter)

            niter += 1
            if args.noam:
                if niter >= args.warmup_steps:
                    lr = args.lr/(args.n_d**0.5)/(niter**0.5)
                else:
                    lr = (args.lr/(args.n_d**0.5)/(args.warmup_steps**1.5))*niter
                optimizer.param_groups[0]['lr'] = lr

    train_writer.close()
    dev_writer.close()

    model.load_state_dict(checkpoint)
    dev = create_batches(dev_, 1)
    test = create_batches(test_, 1)
    dev_ppl, dev_loss = eval_model(model, dev)
    test_ppl, test_loss = eval_model(model, test)
    sys.stdout.write("dev_bpc={:.3f}  test_bpc={:.3f}\n".format(
        np.log2(dev_ppl), np.log2(test_ppl)
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
    argparser.add_argument("--log_period", type=int, default=100000)

    args = argparser.parse_args()
    print (args)
    main(args)
