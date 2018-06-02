#from builtins import bytes
import sys
import time
import math
#import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple

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

    __forceinline__ __device__ float seluf(float x)
    {
        return 1.0507009873554804934193349852946f * (
            (x > 0.f) ? x : 1.6732632423543772848170429916717f * (expf(x)-1.f)
        );
    }

    __forceinline__ __device__ float calc_activation(int type, float x)
    {
        switch (type) {
            case 0:
                return x;
            case 1:
                return tanh(x);
            case 2:
                return reluf(x);
            case 3:
                return seluf(x);
        }
        return x;
    }

    __forceinline__ __device__ float calc_grad_activation(int type, float x)
    {
        switch (type) {
            case 0:
                return 1.f;
            case 1:
                return 1.f-x*x;
            case 2:
                return (x > 0.f) ? 1.f : 0.f;
            case 3:
                return (x > 0.f) ? 1.0507009873554804934193349852946f :
                    x + 1.7580993408473766f;
        }
        return 1.f;
    }

    __global__ void sru_fwd(const float * __restrict__ u, const float * __restrict__ x,
                            const float * __restrict__ bias, const float * __restrict__ init,
                            const float * __restrict__ mask_h,
                            const int len, const int batch, const int d, const int k,
                            float * __restrict__ h, float * __restrict__ c,
                            const int activation_type)
    {
        assert ((k == 3) || (x == NULL));

        int ncols = batch*d;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;

        const float bias1 = *(bias + (col%d));
        const float bias2 = *(bias + (col%d) + d);
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float cur = *(init + col);

        const float *up = u + (col*k);
        const float *xp = (k == 3) ? (x + col) : (up + 3);
        float *cp = c + col;
        float *hp = h + col;

        for (int row = 0; row < len; ++row)
        {
            float g1 = sigmoidf((*(up+1))+bias1);
            float g2 = sigmoidf((*(up+2))+bias2);
            cur = (cur-(*up))*g1 + (*up);
            *cp = cur;
            float val = calc_activation(activation_type, cur);
            *hp = (val*mask-(*xp))*g2 + (*xp);
            up += ncols_u;
            xp += ncols_x;
            cp += ncols;
            hp += ncols;
        }
    }

    __global__ void sru_bwd(const float * __restrict__ u, const float * __restrict__ x,
                            const float * __restrict__ bias, const float * __restrict__ init,
                            const float * __restrict__ mask_h, const float * __restrict__ c,
                            const float * __restrict__ grad_h, const float * __restrict__ grad_last,
                            const int len, const int batch, const int d, const int k,
                            float * __restrict__ grad_u, float * __restrict__ grad_x,
                            float * __restrict__ grad_bias, float * __restrict__ grad_init,
                            int activation_type)
    {
        assert((k == 3) || (x == NULL));
        assert((k == 3) || (grad_x == NULL));

        int ncols = batch*d;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;

        const float bias1 = *(bias + (col%d));
        const float bias2 = *(bias + (col%d) + d);
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float gbias1 = 0;
        float gbias2 = 0;
        float cur = *(grad_last + col);

        const float *up = u + (col*k) + (len-1)*ncols_u;
        const float *xp = (k == 3) ? (x + col + (len-1)*ncols) : (up + 3);
        const float *cp = c + col + (len-1)*ncols;

        const float *ghp = grad_h + col + (len-1)*ncols;
        float *gup = grad_u + (col*k) + (len-1)*ncols_u;
        float *gxp = (k == 3) ? (grad_x + col + (len-1)*ncols) : (gup + 3);

        for (int row = len-1; row >= 0; --row)
        {
            const float g1 = sigmoidf((*(up+1))+bias1);
            const float g2 = sigmoidf((*(up+2))+bias2);

            const float c_val = calc_activation(activation_type, *cp);

            const float x_val = *xp;
            const float u_val = *up;
            const float prev_c_val = (row>0) ? (*(cp-ncols)) : (*(init+col));

            const float gh_val = *ghp;

            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + g0*(1-g1) = (c'-g0)*g1 + g0

            // grad wrt x
            *gxp = gh_val*(1-g2);

            // grad wrt g2, u2 and bias2
            float gg2 = gh_val*(c_val*mask-x_val)*(g2*(1-g2));
            *(gup+2) = gg2;
            gbias2 += gg2;

            // grad wrt c
            const float tmp = g2*calc_grad_activation(activation_type, c_val);
            const float gc = gh_val*mask*tmp + cur;

            // grad wrt u0
            *gup = gc*(1-g1);

            // grad wrt g1, u1, and bias1
            float gg1 = gc*(prev_c_val-u_val)*(g1*(1-g1));
            *(gup+1) = gg1;
            gbias1 += gg1;

            // grad wrt c'
            cur = gc*g1;

            up -= ncols_u;
            xp -= ncols_x;
            cp -= ncols;
            gup -= ncols_u;
            gxp -= ncols_x;
            ghp -= ncols;
        }

        int bias_idx = col % d;
        atomicAdd(grad_bias + bias_idx, gbias1);
        atomicAdd(grad_bias + bias_idx + d, gbias2);
        *(grad_init +col) = cur;
    }

    __global__ void sru_bi_fwd(const float * __restrict__ u, const float * __restrict__ x,
                            const float * __restrict__ bias, const float * __restrict__ init,
                            const float * __restrict__ mask_h,
                            const int len, const int batch, const int d, const int k,
                            float * __restrict__ h, float * __restrict__ c,
                            const int activation_type)
    {
        assert ((k == 3) || (x == NULL));
        assert ((k == 3) || (k == 4));

        int ncols = batch*d*2;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float cur = *(init + col);

        const int d2 = d*2;
        const bool flip = (col%d2) >= d;

        const float bias1 = *(bias + (col%d2));
        const float bias2 = *(bias + (col%d2) + d2);
        const float *up = u + (col*k);
        const float *xp = (k == 3) ? (x + col) : (up + 3);
        float *cp = c + col;
        float *hp = h + col;

        if (flip) {
            up += (len-1)*ncols_u;
            xp += (len-1)*ncols_x;
            cp += (len-1)*ncols;
            hp += (len-1)*ncols;
        }

        int ncols_u_ = flip ? -ncols_u : ncols_u;
        int ncols_x_ = flip ? -ncols_x : ncols_x;
        int ncols_ = flip ? -ncols : ncols;

        for (int cnt = 0; cnt < len; ++cnt)
        {
            float g1 = sigmoidf((*(up+1))+bias1);
            float g2 = sigmoidf((*(up+2))+bias2);
            cur = (cur-(*up))*g1 + (*up);
            *cp = cur;
            float val = calc_activation(activation_type, cur);
            *hp = (val*mask-(*xp))*g2 + (*xp);
            up += ncols_u_;
            xp += ncols_x_;
            cp += ncols_;
            hp += ncols_;
        }

    }

    __global__ void sru_bi_bwd(const float * __restrict__ u, const float * __restrict__ x,
                            const float * __restrict__ bias, const float * __restrict__ init,
                            const float * __restrict__ mask_h, const float * __restrict__ c,
                            const float * __restrict__ grad_h, const float * __restrict__ grad_last,
                            const int len, const int batch, const int d, const int k,
                            float * __restrict__ grad_u, float * __restrict__ grad_x,
                            float * __restrict__ grad_bias, float * __restrict__ grad_init,
                            int activation_type)
    {
        assert((k == 3) || (x == NULL));
        assert((k == 3) || (grad_x == NULL));
        assert((k == 3) || (k == 4));

        int ncols = batch*d*2;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;

        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float gbias1 = 0;
        float gbias2 = 0;
        float cur = *(grad_last + col);

        const int d2 = d*2;
        const bool flip = ((col%d2) >= d);

        const float bias1 = *(bias + (col%d2));
        const float bias2 = *(bias + (col%d2) + d2);
        const float *up = u + (col*k);
        const float *xp = (k == 3) ? (x + col) : (up + 3);
        const float *cp = c + col;
        const float *ghp = grad_h + col;
        float *gup = grad_u + (col*k);
        float *gxp = (k == 3) ? (grad_x + col) : (gup + 3);

        if (!flip) {
            up += (len-1)*ncols_u;
            xp += (len-1)*ncols_x;
            cp += (len-1)*ncols;
            ghp += (len-1)*ncols;
            gup += (len-1)*ncols_u;
            gxp += (len-1)*ncols_x;
        }

        int ncols_u_ = flip ? -ncols_u : ncols_u;
        int ncols_x_ = flip ? -ncols_x : ncols_x;
        int ncols_ = flip ? -ncols : ncols;

        for (int cnt = 0; cnt < len; ++cnt)
        {
            const float g1 = sigmoidf((*(up+1))+bias1);
            const float g2 = sigmoidf((*(up+2))+bias2);

            const float c_val = calc_activation(activation_type, *cp);
            const float x_val = *xp;
            const float u_val = *up;
            const float prev_c_val = (cnt<len-1) ? (*(cp-ncols_)) : (*(init+col));

            const float gh_val = *ghp;

            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + g0*(1-g1) = (c'-g0)*g1 + g0

            // grad wrt x
            *gxp = gh_val*(1-g2);

            // grad wrt g2, u2 and bias2
            float gg2 = gh_val*(c_val*mask-x_val)*(g2*(1-g2));
            *(gup+2) = gg2;
            gbias2 += gg2;

            // grad wrt c
            const float tmp = g2*calc_grad_activation(activation_type, c_val);
            const float gc = gh_val*mask*tmp + cur;

            // grad wrt u0
            *gup = gc*(1-g1);

            // grad wrt g1, u1, and bias1
            float gg1 = gc*(prev_c_val-u_val)*(g1*(1-g1));
            *(gup+1) = gg1;
            gbias1 += gg1;

            // grad wrt c'
            cur = gc*g1;

            up -= ncols_u_;
            xp -= ncols_x_;
            cp -= ncols_;
            gup -= ncols_u_;
            gxp -= ncols_x_;
            ghp -= ncols_;
        }

        int bias_idx = col % d2;
        atomicAdd(grad_bias + bias_idx, gbias1);
        atomicAdd(grad_bias + bias_idx + d2, gbias2);
        *(grad_init +col) = cur;
    }
}
"""


class SRU_Compute_GPU(Function):

    _SRU_PROG = Program(SRU_CODE.encode('utf-8'), 'sru_prog.cu'.encode())
    _SRU_PTX = _SRU_PROG.compile()
    _DEVICE2FUNC = {}

    def __init__(self, activation_type, d_out, bidirectional=False, scale_x=1):
        super(SRU_Compute_GPU, self).__init__()
        self.activation_type = activation_type
        self.d_out = d_out
        self.bidirectional = bidirectional
        self.scale_x = scale_x

    def compile_functions(self):
        device = torch.cuda.current_device()
        print ('SRU loaded for gpu {}'.format(device))
        mod = function.Module()
        mod.load(bytes(self._SRU_PTX.encode()))
        fwd_func = mod.get_function('sru_fwd')
        bwd_func = mod.get_function('sru_bwd')
        bifwd_func = mod.get_function('sru_bi_fwd')
        bibwd_func = mod.get_function('sru_bi_bwd')

        Stream = namedtuple('Stream', ['ptr'])
        current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)

        self._DEVICE2FUNC[device] = (current_stream, fwd_func,
            bifwd_func, bwd_func, bibwd_func
        )
        return current_stream, fwd_func, bifwd_func, bwd_func, bibwd_func

    def get_functions(self):
        res = self._DEVICE2FUNC.get(torch.cuda.current_device(), None)
        return res if res else self.compile_functions()

    def forward(self, u, x, bias, init=None, mask_h=None):
        bidir = 2 if self.bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k//2 if self.bidirectional else k
        ncols = batch*d*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        init_ = x.new(ncols).zero_() if init is None else init
        size = (length, batch, d*bidir) if x.dim() == 3 else (batch, d*bidir)
        c = x.new(*size)
        h = x.new(*size)

        scale_x = self.scale_x
        if k_ == 3:
            x_ptr = x.contiguous()*scale_x if scale_x != 1 else x.contiguous()
            x_ptr = x_ptr.data_ptr()
        else:
            x_ptr = 0

        stream, fwd_func, bifwd_func, _, _ = self.get_functions()
        FUNC = fwd_func if not self.bidirectional else bifwd_func
        FUNC(args=[
            u.contiguous().data_ptr(),
            x_ptr,
            bias.data_ptr(),
            init_.contiguous().data_ptr(),
            mask_h.data_ptr() if mask_h is not None else 0,
            length,
            batch,
            d,
            k_,
            h.data_ptr(),
            c.data_ptr(),
            self.activation_type],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )

        self.save_for_backward(u, x, bias, init, mask_h)
        self.intermediate = c
        if x.dim() == 2:
            last_hidden = c
        elif self.bidirectional:
            last_hidden = torch.cat((c[-1,:,:d], c[0,:,d:]), dim=1)
        else:
            last_hidden = c[-1]
        return h, last_hidden

    def backward(self, grad_h, grad_last):
        bidir = 2 if self.bidirectional else 1
        u, x, bias, init, mask_h = self.saved_tensors
        c = self.intermediate
        scale_x = self.scale_x
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k//2 if self.bidirectional else k
        ncols = batch*d*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_bias = x.new(2*d*bidir).zero_()
        grad_init = x.new(batch, d*bidir)

        # For DEBUG
        #size = (length, batch, x.size(-1)) if x.dim() == 3 else (batch, x.size(-1))
        #grad_x = x.new(*x.size()) if k_ == 3 else x.new(*size).zero_()

        # Normal use
        grad_x = x.new(*x.size()) if k_ == 3 else None

        if k_ == 3:
            x_ptr = x.contiguous()*scale_x if scale_x != 1 else x.contiguous()
            x_ptr = x_ptr.data_ptr()
        else:
            x_ptr = 0

        stream, _, _, bwd_func, bibwd_func = self.get_functions()
        FUNC = bwd_func if not self.bidirectional else bibwd_func
        FUNC(args=[
            u.contiguous().data_ptr(),
            x_ptr,
            bias.data_ptr(),
            init_.contiguous().data_ptr(),
            mask_h.data_ptr() if mask_h is not None else 0,
            c.data_ptr(),
            grad_h.contiguous().data_ptr(),
            grad_last.contiguous().data_ptr(),
            length,
            batch,
            d,
            k_,
            grad_u.data_ptr(),
            grad_x.data_ptr() if k_ == 3 else 0,
            grad_bias.data_ptr(),
            grad_init.data_ptr(),
            self.activation_type],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=stream
        )

        if k_ == 3 and scale_x != 1:
            grad_x.mul_(scale_x)
        return grad_u, grad_x, grad_bias, grad_init, None


def SRU_Compute_CPU(activation_type, d, bidirectional=False, scale_x=1):
    """CPU version of the core SRU computation.

    Has the same interface as SRU_Compute_GPU() but is a regular Python function
    instead of a torch.autograd.Function because we don't implement backward()
    explicitly.
    """
    def sru_compute_cpu(u, x, bias, init=None, mask_h=None):
        bidir = 2 if bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        k = u.size(-1) // d // bidir

        if mask_h is None:
            mask_h = 1

        u = u.view(length, batch, bidir, d, k)

        x_tilde = u[..., 0]

        forget_bias, reset_bias = bias.view(2, bidir, d)
        forget = (u[..., 1] + forget_bias).sigmoid()
        reset = (u[..., 2] + reset_bias).sigmoid()

        if k == 3:
            x_prime = x.view(length, batch, bidir, d)
            x_prime = x_prime*scale_x if scale_x != 1 else x_prime
        else:
            x_prime = u[..., 3]

        h = Variable(x.data.new(length, batch, bidir, d))

        if init is None:
            c_init = Variable(x.data.new(batch, bidir, d).zero_())
        else:
            c_init = init.view(batch, bidir, d)

        c_final = []
        for di in range(bidir):
            if di == 0:
                time_seq = range(length)
            else:
                time_seq = range(length - 1, -1, -1)

            c_prev = c_init[:, di, :]
            for t in time_seq:
                c_t = (c_prev - x_tilde[t, :, di, :]) * forget[t, :, di, :] + x_tilde[t, :, di, :]
                c_prev = c_t

                if activation_type == 0:
                    g_c_t = c_t
                elif activation_type == 1:
                    g_c_t = c_t.tanh()
                elif activation_type == 2:
                    g_c_t = nn.functional.relu(c_t)
                else:
                    assert False, 'Activation type must be 0, 1, or 2, not {}'.format(activation_type)

                h[t, :, di, :] = (g_c_t * mask_h - x_prime[t, :, di, :]) * reset[t, :, di, :] + x_prime[t, :, di, :]

            c_final.append(c_t)

        return h.view(length, batch, -1), torch.stack(c_final, dim=1).view(batch, -1)

    return sru_compute_cpu


class SRUCell(nn.Module):
    def __init__(self, n_in, n_out, dropout=0, rnn_dropout=0,
                 bidirectional=False, use_tanh=1, use_relu=0, use_selu=0,
                 weight_norm=False, layer_norm=False, highway_bias=0, index=-1,
                 rescale=True):
        super(SRUCell, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.rescale = rescale
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.weight_norm = weight_norm
        self.layer_norm = layer_norm
        self.highway_bias = highway_bias
        self.index = index
        self.activation_type = 0
        if use_tanh:
            self.activation_type = 1
        elif use_relu:
            self.activation_type = 2
        elif use_selu:
            self.activation_type = 3

        out_size = n_out*2 if bidirectional else n_out
        k = 4 if n_in != out_size else 3
        self.k = k
        self.size_per_dir = n_out*k
        self.weight = nn.Parameter(torch.Tensor(
            n_in,
            self.size_per_dir*2 if bidirectional else self.size_per_dir
        ))
        self.bias = nn.Parameter(torch.Tensor(
            n_out*4 if bidirectional else n_out*2
        ))
        self.init_weight()

    def init_weight(self):
        # initialize weights such that E[w_ij]=0 and Var[w_ij]=1/d
        val_range = (3.0/self.n_in)**0.5
        self.weight.data.uniform_(-val_range, val_range)

        # initialize bias
        self.bias.data.zero_()
        bias_val, n_out = self.highway_bias, self.n_out
        if self.bidirectional:
            self.bias.data[n_out*2:].zero_().add_(bias_val)
        else:
            self.bias.data[n_out:].zero_().add_(bias_val)

        self.scale_x = 1
        if not self.rescale:
            return
        self.scale_x = (1+math.exp(bias_val)*2)**0.5

        # re-scale weights in case there's dropout and / or layer normalization
        w = self.weight.data.view(self.n_in, -1, self.n_out, self.k)
        if self.dropout>0:
            w[:,:,:,0].mul_((1-self.dropout)**0.5)
        if self.rnn_dropout>0:
            w.mul_((1-self.rnn_dropout)**0.5)
        if self.layer_norm:
            w[:,:,:,1].mul_(0.1)
            w[:,:,:,2].mul_(0.1)
        if self.k == 4:
            w[:,:,:,3].mul_(self.scale_x)

        # re-parameterize when weight normalization is enabled
        if self.weight_norm:
            self.init_weight_norm()

    def init_weight_norm(self):
        weight = self.weight.data
        g = weight.norm(2, 0)
        self.gain = nn.Parameter(g)

    def apply_weight_norm(self, eps=0):
        wnorm = self.weight.norm(2, 0)#, keepdim=True)
        return self.gain.expand_as(self.weight).mul(
            self.weight / (wnorm.expand_as(self.weight) + eps)
        )

    def set_bias(self, bias_val=0):
        sys.stderr.write("\nWARNING: set_bias() is deprecated. use `highway_bias` option"
            " in SRUCell() constructor.\n"
        )
        self.highway_bias = bias_val
        self.init_weight()
        #n_out = self.n_out
        #if self.bidirectional:
        #    self.bias.data[n_out*2:].zero_().add_(bias_val)
        #else:
        #    self.bias.data[n_out:].zero_().add_(bias_val)

    def forward(self, input, c0=None):
        assert input.dim() == 2 or input.dim() == 3
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
        weight = self.weight if not self.weight_norm else self.apply_weight_norm()
        u = x_2d.mm(weight)

        if input.is_cuda:
            SRU_Compute = SRU_Compute_GPU(self.activation_type, n_out, self.bidirectional, self.scale_x)
        else:
            SRU_Compute = SRU_Compute_CPU(self.activation_type, n_out, self.bidirectional, self.scale_x)

        if self.training and (self.dropout>0):
            bidir = 2 if self.bidirectional else 1
            mask_h = self.get_dropout_mask_((batch, n_out*bidir), self.dropout)
            return SRU_Compute(u, input, self.bias, c0, mask_h)
        else:
            return SRU_Compute(u, input, self.bias, c0)

    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))


class SRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0,
                 rnn_dropout=0, bidirectional=False, use_tanh=1, use_relu=0,
                 use_selu=0, weight_norm=False, layer_norm=False,
                 highway_bias=0, rescale=True):
        super(SRU, self).__init__()
        self.n_in = input_size
        self.n_out = hidden_size
        self.depth = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = nn.ModuleList()
        self.ln_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.use_layer_norm = layer_norm
        self.use_wieght_norm = weight_norm
        self.out_size = hidden_size*2 if bidirectional else hidden_size
        if use_tanh + use_relu + use_selu > 1:
            sys.stderr.write("\nWARNING: More than one activation enabled in SRU"
                " (tanh: {}  relu: {}  selu: {})\n".format(use_tanh, use_relu, use_selu)
            )

        for i in range(num_layers):
            l = SRUCell(
                n_in = self.n_in if i==0 else self.out_size,
                n_out = self.n_out,
                dropout = dropout if i+1 != num_layers else 0,
                rnn_dropout = rnn_dropout,
                bidirectional = bidirectional,
                use_tanh = use_tanh,
                use_relu = use_relu,
                use_selu = use_selu,
                weight_norm = weight_norm,
                layer_norm = layer_norm,
                highway_bias = highway_bias,
                index = i+1,
                rescale=rescale
            )
            self.rnn_lst.append(l)
            if layer_norm:
                self.ln_lst.append(LayerNorm(self.n_out))

    def set_bias(self, bias_val=0):
        for l in self.rnn_lst:
            l.set_bias(bias_val)

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
            prevx = self.ln_lst[i](h) if self.use_layer_norm else h
            lstc.append(c)

        if return_hidden:
            return prevx, torch.stack(lstc)
        else:
            return prevx


class LayerNorm(nn.Module):
    '''
    Layer normalization module modified from:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/UtilClass.py
    '''

    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(self, x):
        if x.size(-1) == 1:
            return x
        mu = torch.mean(x, dim=-1)
        sigma = torch.std(x, dim=-1, unbiased=False)
        # HACK. PyTorch is changing behavior
        if mu.dim() == x.dim()-1:
            mu = mu.unsqueeze(mu.dim())
            sigma = sigma.unsqueeze(sigma.dim())
        output = (x - mu.expand_as(x)) / (sigma.expand_as(x) + self.eps)
        output = output.mul(self.a.expand_as(output)) \
            + self.b.expand_as(output)
        return output
