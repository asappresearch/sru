import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from sru_functional import SRUCell

T = 10000

def reset_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def test_fast(L, B, N, train=False):
    reset_seed()
    a = Variable(torch.randn(L, B, N).float().cuda()*0.1)
    c = Variable(torch.zeros(B, N).float().cuda())
    cell = SRUCell(N, N, dropout=0, use_tanh=False).cuda()
    if train: cell.train()
    torch.cuda.synchronize()
    start = time.time()
    tot = 0
    for i in range(T):
        out = cell(a, c)
        tot += out[0].data[-1,-1,-1]
        if train:
            cell.zero_grad()
            out[0].mean().backward()
    torch.cuda.synchronize()
    print ("test_fast: {:.6f}".format(
        (time.time()-start)/T
    ))

def test_lstm(L, B, N, train=False):
    reset_seed()
    a = Variable(torch.randn(L, B, N).float().cuda()*0.1)
    h = Variable(torch.zeros(1, B, N).float().cuda())
    cell = nn.LSTM(N, N, dropout=0.0).cuda()
    if train: cell.train()
    torch.cuda.synchronize()
    start = time.time()
    tot = 0
    for i in range(T):
        out = cell(a, (h,h))
        tot += out[0].data[-1,-1,-1]
        if train:
            cell.zero_grad()
            out[0].mean().backward()
    torch.cuda.synchronize()
    print ("test_lstm: {:.6f}".format(
        (time.time()-start)/T
    ))

def test_conv(L, B, N, k=2, train=False):
    reset_seed()
    a = Variable(torch.randn(B, 1, L, N).float().cuda()*0.1)
    conv = nn.Conv2d(1, N, (k, N)).cuda()
    if train: conv.train()
    torch.cuda.synchronize()
    start = time.time()
    tot = 0
    for i in range(T):
        out = conv(a)
        tot += out.data[-1,-1,-1,-1]
        if train:
            conv.zero_grad()
            out.mean().backward()
    torch.cuda.synchronize()
    print (("test_conv: {:.6f}").format(
        (time.time()-start)/T
    ))

def test(L, N, D, train=False):
    print (L, N, D)
    test_fast(L, N, D, train)
    test_conv(L, N, D, k=2, train=train)
    test_conv(L, N, D, k=3, train=train)
    test_lstm(L, N, D, train)

if __name__=="__main__":
    test(32, 32, 256)
    test(128, 32, 512)

