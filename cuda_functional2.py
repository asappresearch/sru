#from builtins import bytes
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple


tmp_ = torch.rand(1,1).cuda()

SRU_CODE = """
extern "C" {

    __forceinline__ __device__ float sigmoidf(float x)
    {
        return 1.f / (1.f + expf(-x));
    }

    __forceinline__ __device__ float reluf(float x)
    {
        return (x > 0.f) ? x : 0.f;
    }

    __global__ void wcumsum_fwd(const float * __restrict__ x,
                            const float * __restrict__ weight,
                            const float * __restrict__ init,
                            const int len, const int batch, const int d,
                            const int reverse,
                            float * __restrict__ sum)
    {
        int ncols = batch*d;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        const float *xp = x + col;
        const float *wp = weight + col;
        float *sp = sum + col;
        float cur = *(init + col);

        if (reverse) {
            xp += (len-1)*ncols;
            wp += (len-1)*ncols;
            sp += (len-1)*ncols;
        }

        const int ncols_ = reverse ? -ncols : ncols;

        for (int cnt = 0; cnt < len; ++cnt)
        {
            cur = cur*(*wp) + (*xp);
            *sp = cur;
            xp += ncols_;
            wp += ncols_;
            sp += ncols_;
        }
    }

    __global__ void wcumsum_bwd(const float * __restrict__ sum,
                            const float * __restrict__ weight,
                            const float * __restrict__ init,
                            const int len, const int batch, const int d,
                            const int reverse,
                            const float * __restrict__ grad_sum,
                            float * __restrict__ grad_x,
                            float * __restrict__ grad_weight,
                            float * __restrict__ grad_init)
    {
        int ncols = batch*d;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        const float *gsp = grad_sum + ((len-1)*ncols+col);
        const float *sp = sum + ((len-1)*ncols+col);
        const float *wp = weight + ((len-1)*ncols+col);
        float *gxp = grad_x + ((len-1)*ncols+col);
        float *gwp = grad_weight + ((len-1)*ncols+col);
        float cur = 0;

        if (reverse) {
            gsp -= (len-1)*ncols;
            sp  -= (len-1)*ncols;
            wp  -= (len-1)*ncols;
            gxp -= (len-1)*ncols;
            gwp -= (len-1)*ncols;
        }

        const int ncols_ = reverse ? -ncols : ncols;

        for (int cnt = 0; cnt < len; ++cnt)
        {
            const float prev = (cnt < len-1) ? (*(sp-ncols_)) : (*(init+col));
            cur += (*gsp);
            *gxp = cur;
            *gwp = cur*prev;
            cur = cur*(*wp);
            sp -= ncols_;
            wp -= ncols_;
            gsp -= ncols_;
            gxp -= ncols_;
            gwp -= ncols_;
        }
        *(grad_init + col) = cur;
    }
}
"""

SRU_PROG = Program(SRU_CODE.encode('utf-8'), 'sru_prog.cu'.encode('utf-8'))
SRU_PTX = SRU_PROG.compile()
SRU_MOD = function.Module()
SRU_MOD.load(bytes(SRU_PTX.encode()))
WSUM_FWD_FUNC = SRU_MOD.get_function('wcumsum_fwd')
WSUM_BWD_FUNC = SRU_MOD.get_function('wcumsum_bwd')

Stream = namedtuple('Stream', ['ptr'])
SRU_STREAM = Stream(ptr=torch.cuda.current_stream().cuda_stream)

class WeightedCumsum(Function):
    def __init__(self, reverse=0):
        super(WeightedCumsum, self).__init__()
        self.reverse = reverse

    def forward(self, x, weight, init=None):
        # only implement a tensor version
        assert x.dim() == 3
        length = x.size(0)
        batch = x.size(-2)
        d = x.size(-1)
        ncols = batch*d
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)/thread_per_block+1

        init_ = x.new(ncols).zero_() if init is None else init
        size = (length, batch, d)
        cum_sum = x.new(*size)

        FUNC = WSUM_FWD_FUNC
        FUNC(args=[
            x.contiguous().data_ptr(),
            weight.contiguous().data_ptr(),
            init_.contiguous().data_ptr(),
            length,
            batch,
            d,
            self.reverse,
            cum_sum.data_ptr()],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=SRU_STREAM
        )

        self.save_for_backward(x, weight, init, cum_sum)
        return cum_sum

    def backward(self, grad_sum):
        x, weight, init, cum_sum = self.saved_tensors
        length = x.size(0)
        batch = x.size(-2)
        d = x.size(-1)
        ncols = batch*d
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)/thread_per_block+1

        init_ = x.new(ncols).zero_() if init is None else init
        size = (length, batch, d)
        grad_x = x.new(*size)
        grad_weight = x.new(*size)
        grad_init = x.new(batch, d)

        FUNC = WSUM_BWD_FUNC
        FUNC(args=[
            cum_sum.data_ptr(),
            weight.contiguous().data_ptr(),
            init_.contiguous().data_ptr(),
            length,
            batch,
            d,
            self.reverse,
            grad_sum.contiguous().data_ptr(),
            grad_x.data_ptr(),
            grad_weight.data_ptr(),
            grad_init.data_ptr()],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=SRU_STREAM
        )

        return grad_x, grad_weight, grad_init

'''
    Re-impelement Q-RNN with filter width = 1.
    A highway gate is added for comparison with original SRU.
'''
class SRUCell(nn.Module):
    def __init__(self, n_in, n_out, dropout=0, rnn_dropout=0,
                bidirectional=False, use_tanh=1, use_relu=0):
        assert use_relu == 0
        assert use_tanh == 1
        #assert bidirectional == False

        super(SRUCell, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.bidirectional = bidirectional

        out_size = n_out*2 if bidirectional else n_out
        k = 4 if n_in != out_size else 3
        k = k*2 if bidirectional else k
        self.k = k
        self.weight = nn.Parameter(torch.Tensor(
            n_in,
            n_out*k
        ))
        self.bias = nn.Parameter(torch.Tensor(
            n_out*6 if bidirectional else n_out*3
        ))
        self.init_weight()

    def init_weight(self):
        val_range = (3.0/self.n_in)**0.5
        self.weight.data.uniform_(-val_range, val_range)
        self.bias.data.zero_()

    def forward(self, input, c0=None):
        assert input.dim() == 3
        length = input.size(0)
        n_in, n_out = self.n_in, self.n_out
        batch = input.size(-2)
        if c0 is None:
            c0 = Variable(input.data.new(
                batch, n_out if not self.bidirectional else n_out*2
            ).zero_())

        if self.training and (self.rnn_dropout>0):
            mask = self.get_dropout_mask_((batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)
        u = x_2d.mm(self.weight).view(length, batch, -1)  # (length, batch, n_out*k)
        wx = u.chunk(self.k, 2)
        assert len(wx) == self.k
        b = self.bias.chunk(6 if self.bidirectional else 3, 0)

        out_size = n_out*2 if self.bidirectional else n_out

        if not self.bidirectional:
            tilde_x = F.tanh(wx[0] + b[0])
            forget = F.sigmoid(wx[1] + b[1])
            reset = F.sigmoid(wx[2] + b[2])
            c = WeightedCumsum()((1-forget)*tilde_x, forget, c0)
            highway_x = input if n_in == out_size else wx[3]
        else:
            tilde_x_1 = F.tanh(wx[0] + b[0])
            forget_1 = F.sigmoid(wx[1] + b[1])
            reset_1 = F.sigmoid(wx[2] + b[2])
            tilde_x_2 = F.tanh(wx[3] + b[3])
            forget_2 = F.sigmoid(wx[4] + b[4])
            reset_2 = F.sigmoid(wx[5] + b[5])
            c_1 = WeightedCumsum(reverse=0)((1-forget_1)*tilde_x_1, forget_1, c0[:,:n_out])
            c_2 = WeightedCumsum(reverse=1)((1-forget_2)*tilde_x_2, forget_2, c0[:,n_out:])
            # (length, batch, n_out*2)
            c = torch.cat((c_1, c_2), 2)
            reset = torch.cat((reset_1, reset_2), 2)
            highway_x = input if n_in == out_size else torch.cat((wx[6], wx[7]), 2)

        if self.training and (self.dropout>0):
            mask_h = self.get_dropout_mask_((batch, out_size), self.dropout)
            pre_h = c * mask_h.expand_as(c)
        else:
            pre_h = c
        h = pre_h*reset + highway_x*(1-reset)

        return h, c[-1]

    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))


class SRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0, rnn_dropout=0,
                bidirectional=False, use_tanh=1, use_relu=0):
        super(SRU, self).__init__()
        self.n_in = input_size
        self.n_out = hidden_size
        self.depth = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.out_size = hidden_size*2 if bidirectional else hidden_size

        for i in range(num_layers):
            l = SRUCell(
                n_in = self.n_in if i==0 else self.out_size,
                n_out = self.n_out,
                dropout = dropout if i+1 != num_layers else 0,
                rnn_dropout = rnn_dropout,
                bidirectional = bidirectional,
                use_tanh = use_tanh,
                use_relu = use_relu,
            )
            self.rnn_lst.append(l)

    def forward(self, input, c0=None, return_hidden=True):
        assert input.dim() == 3 # (len, batch, n_in)
        dir_ = 2 if self.bidirectional else 1
        if c0 is None:
            zeros = Variable(input.data.new(
                input.size(1), self.n_out*dir_
            ).zero_())
            c0 = [ zeros for i in range(self.depth) ]
        else:
            assert c0.dim() == 3    # (depth, batch, n_out*dir_)
            c0 = [ x.squeeze(0) for x in c0.chunk(self.depth, 0) ]

        prevx = input
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prevx, c0[i])
            prevx = h
            lstc.append(c)

        if return_hidden:
            return prevx, torch.stack(lstc)
        else:
            return prevx


