import os
import sys
import warnings
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
if torch.cuda.device_count() > 0:
    try:
        from .cuda_functional import SRU_Compute_GPU
    except:
        from cuda_functional import SRU_Compute_GPU

from torch.utils.cpp_extension import load

cpu_source = os.path.join(os.path.dirname(__file__), "sru_cpu_impl.cpp")
sru_cpu_impl = load(name="sru_cpu_impl", sources=[cpu_source])


def SRU_Compute_CPU(activation_type,
                    d,
                    bidirectional=False,
                    has_skip_term=True,
                    scale_x=1,
                    mask_pad=None):
    """CPU version of the core SRU computation.

    Has the same interface as SRU_Compute_GPU() but is a regular Python function
    instead of a torch.autograd.Function because we don't implement backward()
    explicitly.

    Args:
        activation_type (int) : 0 (identity), 1 (tanh), 2 (ReLU) or 3 (SeLU).
        d (int) : the dimensionality of the hidden layer
            of the `SRUCell`. This is not named very well; it is
            nonetheless named as such to maintain consistency with
            the GPU compute-kernel constructor.
        bidirectional (bool) : whether or not to use bidirectional `SRUCell`s.
            Default: False.
        has_skip_term (bool) : whether or not to include `(1-r[t])*x[t]` skip-term in h[t]
        scale_x (float) : scaling constant on the highway connection
    """

    def sru_compute_cpu(u, x, weight_c, bias, init=None, mask_c=None):
        """
        An SRU is a recurrent neural network cell comprised of 5 equations, described
        in "Simple Recurrent Units for Highly Parallelizable Recurrence."

        The first 3 of these equations each require a matrix-multiply component,
        i.e. the input vector x_t dotted with a weight matrix W_i, where i is in
        {0, 1, 2}.

        As each weight matrix W is dotted with the same input x_t, we can fuse these
        computations into a single matrix-multiply, i.e. `x_t <dot> stack([W_0, W_1, W_2])`.
        We call the result of this computation `U`.

        sru_compute_cpu() accepts 'u' and 'x' (along with a tensor of biases,
        an initial memory cell `c0`, and an optional dropout mask) and computes
        equations (3) - (7). It returns a tensor containing all `t` hidden states
        (where `t` is the number of elements in our input sequence) and the final
        memory cell `c_T`.
        """

        bidir = 2 if bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        k = u.size(-1) // d // bidir

        if not torch.is_grad_enabled():
            assert mask_c is None
            cpu_forward = sru_cpu_impl.cpu_bi_forward if bidirectional else \
                          sru_cpu_impl.cpu_forward
            mask_pad_ = torch.FloatTensor() if mask_pad is None else mask_pad.float()
            return cpu_forward(
                u,
                x.contiguous(),
                weight_c,
                bias,
                init,
                mask_pad_,
                length,
                batch,
                d,
                k,
                activation_type,
                has_skip_term,
                scale_x
            )
        else:
            warnings.warn("Running SRU on CPU with grad_enabled=True. Are you sure?")

        mask_pad_ = mask_pad.view(length, batch, 1).float() if mask_pad is not None else mask_pad
        u = u.view(length, batch, bidir, d, k)
        forget_wc, reset_wc = weight_c.view(2, bidir, d)
        forget_bias, reset_bias = bias.view(2, bidir, d)

        if not has_skip_term:
            x_prime = None
        elif k == 3:
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

            mask_c_ = 1 if mask_c is None else mask_c.view(batch, bidir, d)[:, di, :]
            c_prev = c_init[:, di, :]
            fb, rb = forget_bias[di], reset_bias[di]
            fw, rw = forget_wc[di].expand(batch, d), reset_wc[di].expand(batch, d)
            u0 = u[:, :, di, :, 0].chunk(length)
            u1 = (u[:, :, di, :, 1] + fb).chunk(length)
            u2 = (u[:, :, di, :, 2] + rb).chunk(length)
            xp = x_prime[:, :, di, :].chunk(length)
            for t in time_seq:
                forget_t = (u1[t] + c_prev*fw).sigmoid()
                reset_t = (u2[t] + c_prev*rw).sigmoid()
                c_t = u0[t] + (c_prev - u0[t]) * forget_t
                if mask_pad_ is not None:
                    c_t = c_t * (1-mask_pad_[t]) + c_prev * mask_pad_[t]
                c_prev = c_t

                if activation_type == 0:
                    g_c_t = c_t
                elif activation_type == 1:
                    g_c_t = c_t.tanh()
                elif activation_type == 2:
                    g_c_t = nn.functional.relu(c_t)
                else:
                    raise ValueError('Activation type must be 0, 1, or 2, not {}'.format(activation_type))

                if x_prime is not None:
                    h_t = xp[t] + (g_c_t * mask_c_ - xp[t]) * reset_t
                else:
                    h_t = g_c_t * mask_c_ * reset_t
                if mask_pad_ is not None:
                    h_t = h_t * (1-mask_pad_[t])
                h[t, :, di, :] = h_t

            c_final.append(c_t)
        return h.view(length, batch, -1), torch.stack(c_final, dim=1).view(batch, -1)

    return sru_compute_cpu


class SRUCell(nn.Module):
    """
    An SRU cell, i.e. a single recurrent neural network cell,
    as per `LSTMCell`, `GRUCell` and `RNNCell` in PyTorch.

    Args:
        n_in (int) : the number of dimensions in a single
            input sequence element. For example, if the input sequence
            is a sequence of word embeddings, `input_size` is the
            dimensionality of a single word embedding, e.g. 300.
        n_out (int) : the dimensionality of the hidden state
            of this cell.
        dropout (float) : a number between 0.0 and 1.0. The amount of dropout
            applied to `g(c_t)` internally in this cell.
        rnn_dropout (float) : the amount of dropout applied to the input of
            this cell.
        use_tanh (bool) : use tanh activation
        use_relu (bool) : use relu activation
        use_selu (bool) : use selu activation
        weight_norm (bool) : whether applyweight normalization on self.weight
        is_input_normalized (bool) : whether the input is normalized (e.g. batch norm / layer norm)
        bidirectional (bool) : whether or not to employ a bidirectional cell.
        index (int) : index of this cell when multiple layers are stacked in SRU()
    """

    def __init__(self,
                 n_in,
                 n_out,
                 dropout=0,
                 rnn_dropout=0,
                 bidirectional=False,
                 n_proj=0,
                 use_tanh=False,
                 use_relu=False,
                 use_selu=False,
                 weight_norm=False,
                 is_input_normalized=False,
                 highway_bias=0,
                 has_skip_term=True,
                 rescale=True,
                 v1=False):

        if weight_norm and n_proj > 0:
            raise ValueError(
                "Weight norm is not supported with projection enabled"
            )

        super(SRUCell, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.weight_norm = weight_norm
        self.is_input_normalized = is_input_normalized
        self.highway_bias = highway_bias
        self.has_skip_term=has_skip_term
        self.activation_type = 0
        self.v1 = v1
        self.rescale = rescale
        if use_tanh:
            self.activation_type = 1
            self.activation = 'tanh'
        elif use_relu:
            self.activation_type = 2
            self.activation = 'ReLU'
        elif use_selu:
            self.activation_type = 3
            self.activation = 'SeLU'

        self.n_proj = 0
        if n_proj > 0 and n_proj < n_in and n_proj < n_out:
            self.n_proj = n_proj

        out_size = n_out*2 if bidirectional else n_out
        k = 4 if has_skip_term and n_in != out_size else 3
        self.k = k
        self.size_per_dir = n_out*k
        if self.n_proj == 0:
            self.weight = nn.Parameter(torch.Tensor(
                n_in,
                self.size_per_dir*2 if bidirectional else self.size_per_dir
            ))
        else:
            self.weight_proj = nn.Parameter(torch.Tensor(n_in, self.n_proj))
            self.weight = nn.Parameter(torch.Tensor(
                self.n_proj,
                self.size_per_dir*2 if bidirectional else self.size_per_dir
            ))
        self.weight_c = nn.Parameter(torch.Tensor(
            n_out*4 if bidirectional else n_out*2
        ))
        self.bias = nn.Parameter(torch.Tensor(
            n_out*4 if bidirectional else n_out*2
        ))
        self.scale_x = nn.Parameter(torch.ones(1), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Properly initialize the weights of SRU, following the same recipe as:
            Xavier init:  http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            Kaiming init: https://arxiv.org/abs/1502.01852

        """
        # initialize weights such that E[w_ij]=0 and Var[w_ij]=1/d
        d = self.weight.size(0)
        val_range = (3.0/d)**0.5
        self.weight.data.uniform_(-val_range, val_range)
        w = self.weight.data.view(d, -1, self.n_out, self.k)
        if self.n_proj > 0:
            val_range_2 = (3.0/self.weight_proj.size(0))**0.5
            self.weight_proj.data.uniform_(-val_range_2, val_range_2)

        # initialize bias
        self.bias.data.zero_()
        bias_val, n_out = self.highway_bias, self.n_out
        if self.bidirectional:
            self.bias.data[n_out*2:].zero_().add_(bias_val)
        else:
            self.bias.data[n_out:].zero_().add_(bias_val)

        if not self.v1:
            # intialize weight_c such that E[w]=0 and Var[w]=1
            self.weight_c.data.uniform_(-3.0**0.5, 3.0**0.5)

            # rescale weight_c and the weight of sigmoid gates with a factor of sqrt(0.5)
            w[:, :, :, 1].mul_(0.5**0.5)
            w[:, :, :, 2].mul_(0.5**0.5)
            self.weight_c.data.mul_(0.5**0.5)
        else:
            self.weight_c.data.zero_()
            self.weight_c.requires_grad = False

        self.scale_x.data[0] = 1
        if not self.rescale:
            return
        # scalar used to properly scale the highway output
        scale_val = (1+math.exp(bias_val)*2)**0.5
        self.scale_x.data[0] = scale_val
        if self.k == 4:
            w[:, :, :, 3].mul_(scale_val)

        # re-scale weights for dropout and normalized input for better gradient flow
        if self.dropout > 0:
            w[:, :, :, 0].mul_((1-self.dropout)**0.5)
        if self.rnn_dropout > 0:
            w.mul_((1-self.rnn_dropout)**0.5)
        if self.is_input_normalized:
            w[:, :, :, 1].mul_(0.1)
            w[:, :, :, 2].mul_(0.1)
            self.weight_c.data.mul_(0.1)

        # re-parameterize when weight normalization is enabled
        if self.weight_norm:
            self.reset_weight_norm()

    def reset_weight_norm(self):
        weight = self.weight.data
        g = weight.norm(2, 0)
        self.gain = nn.Parameter(g)

    def apply_weight_norm(self, eps=0):
        wnorm = self.weight.norm(2, 0)  # , keepdim=True)
        return self.gain.expand_as(self.weight).mul(
            self.weight / (wnorm.expand_as(self.weight) + eps)
        )

    def forward(self, input, c0=None, mask_pad=None, return_proj=False):
        """
        This method computes `U`. In addition, it computes the remaining components
        in `SRU_Compute_GPU` or `SRU_Compute_CPU` and return the results.
        """

        if input.dim() != 2 and input.dim() != 3:
            raise ValueError("Input must be 2 or 3 dimensional")
        n_in, n_out = self.n_in, self.n_out
        batch = input.size(-2)
        if c0 is None:
            c0 = Variable(input.data.new(
                batch, n_out if not self.bidirectional else n_out*2
            ).zero_())

        # apply dropout for multiplication
        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        # compute U
        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)
        weight = self.weight if not self.weight_norm else self.apply_weight_norm()
        if self.n_proj > 0:
            x_projected = x_2d.mm(self.weight_proj)  # down-proj to n_proj
            u = x_projected.mm(weight)
        else:
            u = x_2d.mm(weight)

        # get the scaling constant; scale_x is a scalar
        scale_val = self.scale_x.data[0].item()

        # Pytorch Function() doesn't accept NoneType in forward() call.
        # So we put mask_pad as class attribute as a work around
        SRU_Compute_Class = SRU_Compute_GPU if input.is_cuda else SRU_Compute_CPU
        SRU_Compute = SRU_Compute_Class(
            self.activation_type, n_out, self.bidirectional, self.has_skip_term,
            scale_val, mask_pad
        )

        # compute dropout mask for states c[]
        if self.training and (self.dropout > 0):
            bidir = 2 if self.bidirectional else 1
            mask_c = self.get_dropout_mask_((batch, n_out*bidir), self.dropout)
            h, c = SRU_Compute(u, input, self.weight_c, self.bias, c0, mask_c)
        else:
            h, c = SRU_Compute(u, input, self.weight_c, self.bias, c0)

        if return_proj:
            x_projected = x_projected.view(-1, batch, self.n_proj) if self.n_proj else input
            return h, c, x_projected
        else:
            return h, c

    def get_dropout_mask_(self, size, p):
        """
        Composes the dropout mask for the `SRUCell`.
        """
        w = self.weight.data
        return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))

    def extra_repr(self):
        s = "{n_in}, {n_out}"
        if self.n_proj > 0:
            s += ", n_proj={n_proj}"
        if self.dropout > 0:
            s += ", dropout={dropout}"
        if self.rnn_dropout > 0:
            s += ", rnn_dropout={rnn_dropout}"
        if self.bidirectional:
            s += ", bidirectional={bidirectional}"
        if self.highway_bias != 0:
            s += ", highway_bias={highway_bias}"
        if self.weight_norm:
            s += ", weight_norm={weight_norm}"
        if self.activation_type != 0:
            s += ", activation={activation}"
        if self.v1:
            s += ", v1={v1}"
        s += ", rescale={rescale}"
        if not self.has_skip_term:
            s += ", has_skip_term={has_skip_term}"
        return s.format(**self.__dict__)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.extra_repr())


class SRU(nn.Module):
    """
    PyTorch SRU model. In effect, simply wraps an arbitrary number of
    contiguous `SRUCell`s, and returns the matrix and hidden states ,
    as well as final memory cell (`c_t`), from the last of these `SRUCell`s.

    Args:
        input_size (int) : the number of dimensions in a single
            input sequence element. For example, if the input sequence
            is a sequence of word embeddings, `input_size` is the
            dimensionality of a single word embedding, e.g. 300.
        hidden_size (int) : the dimensionality of the hidden state
            of the SRU cell.
        num_layers (int) : number of `SRUCell`s to use in the model.
        dropout (float) : a number between 0.0 and 1.0. The amount of dropout
            applied to `g(c_t)` internally in each `SRUCell`.
        rnn_dropout (float) : the amount of dropout applied to the input of
            each `SRUCell`.
        use_tanh (bool) : use tanh activation
        use_relu (bool) : use relu activation
        use_selu (bool) : use selu activation
        weight_norm (bool) : whether or not to use weight normalization
        layer_norm (bool) : whether or not to use layer normalization on the output of each layer
        bidirectional (bool) : whether or not to use bidirectional `SRUCell`s.
        is_input_normalized (bool) : whether the input is normalized (e.g. batch norm / layer norm)
        highway_bias (float) : initial bias of the highway gate, typicially <= 0
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=2,
                 dropout=0,
                 rnn_dropout=0,
                 bidirectional=False,
                 n_proj=0,
                 use_tanh=False,
                 use_relu=False,
                 use_selu=False,
                 weight_norm=False,
                 layer_norm=False,
                 is_input_normalized=False,
                 highway_bias=0,
                 has_skip_term=True,
                 rescale=True,
                 v1=False):

        super(SRU, self).__init__()
        self.n_in = input_size
        self.n_out = hidden_size
        self.depth = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.n_proj = n_proj
        self.rnn_lst = nn.ModuleList()
        self.ln_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.use_layer_norm = layer_norm
        self.use_weight_norm = weight_norm
        self.has_skip_term = has_skip_term
        self.out_size = hidden_size*2 if bidirectional else hidden_size
        if use_tanh + use_relu + use_selu > 1:
            raise ValueError(
                "More than one activation enabled in SRU"
                " (tanh: {}  relu: {}  selu: {})\n".format(use_tanh, use_relu, use_selu)
            )

        for i in range(num_layers):
            l = SRUCell(
                n_in=self.n_in if i == 0 else self.out_size,
                n_out=self.n_out,
                dropout=dropout if i+1 != num_layers else 0,
                rnn_dropout=rnn_dropout,
                bidirectional=bidirectional,
                n_proj=n_proj,
                use_tanh=use_tanh,
                use_relu=use_relu,
                use_selu=use_selu,
                weight_norm=weight_norm,
                is_input_normalized=is_input_normalized or (i > 0 and self.use_layer_norm),
                highway_bias=highway_bias,
                has_skip_term=has_skip_term,
                rescale=rescale,
                v1=v1
            )
            self.rnn_lst.append(l)
            if layer_norm:
                self.ln_lst.append(LayerNorm(self.out_size))

    def set_bias(self, bias_val=0):
        for l in self.rnn_lst:
            l.set_bias(bias_val)

    def forward(self, input, c0=None, mask_pad=None, return_hidden=True):
        """
        Feeds `input` forward through `num_layers` `SRUCell`s, where `num_layers`
        is a parameter on the constructor of this class.
        """

        # The dimensions of `input` should be: `(sequence_length, batch_size, input_size)`.
        if input.dim() != 3:
            raise ValueError("There must be 3 dimensions for (len, batch, n_in)")
        dir_ = 2 if self.bidirectional else 1
        if c0 is None:
            zeros = Variable(input.data.new(
                input.size(1), self.n_out*dir_
            ).zero_())
            c0 = [ zeros for i in range(self.depth) ]
        else:
            # The dimensions of `input` should be: `(num_layers, batch_size, hidden_size * dir_)`.
            if c0.dim() != 3:
                raise ValueError("There must be 3 dimensions for (depth, batch, n_out*dir_)")
            c0 = [ x.squeeze(0) for x in c0.chunk(self.depth, 0) ]

        prevx = input
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prevx, c0[i], mask_pad=mask_pad)
            prevx = self.ln_lst[i](h) if self.use_layer_norm else h
            lstc.append(c)

        if return_hidden:
            return prevx, torch.stack(lstc)
        else:
            return prevx

    def reset_parameters(self):
        for rnn in self.rnn_lst:
            rnn.reset_parameters()
        for ln in self.ln_lst:
            ln.reset_parameters()


class LayerNorm(nn.Module):
    """
    Layer normalization (https://arxiv.org/abs/1607.06450)

    Module modified from:
      https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/UtilClass.py
    """

    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(self, x):
        """
        Apply layer normalization on the input x
        """
        if x.size(-1) == 1:
            return x
        mean = x.mean(-1, keepdim=True)
        # compute the std. ideally should use std = x.std(-1, keepdim=True)
        # but there is a bug in pytorch: https://github.com/pytorch/pytorch/issues/4320
        var = x.var(-1, keepdim=True)
        std = (var + self.eps)**0.5
        return self.a * (x - mean) / (std) + self.b

    def reset_parameters(self):
        self.a.data[:] = 1.0
        self.b.data[:] = 0.0

    def extra_repr(self):
        return "{}, eps={}".format(self.a.numel(), self.eps)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.extra_repr())
