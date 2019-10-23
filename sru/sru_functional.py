import os
import sys
import copy
import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

SRU_CPU_kernel = None
SRU_GPU_kernel = None

# load C++ implementation for CPU computation
def _lazy_load_cpu_kernel():
    global SRU_CPU_kernel
    if SRU_CPU_kernel is not None:
        return SRU_CPU_kernel
    try:
        from torch.utils.cpp_extension import load
        cpu_source = os.path.join(os.path.dirname(__file__), "sru_cpu_impl.cpp")
        SRU_CPU_kernel = load(
            name="sru_cpu_impl",
            sources=[cpu_source],
            extra_cflags=['-O3'],
            verbose=False
        )
    except:
        # use Python version instead
        SRU_CPU_kernel = False
    return SRU_CPU_kernel

# load C++ implementation for GPU computation
def _lazy_load_cuda_kernel():
    try:
        from .cuda_functional import SRU_Compute_GPU
    except:
        from cuda_functional import SRU_Compute_GPU
    return SRU_Compute_GPU

def SRU_CPU_class(activation_type,
                    d,
                    bidirectional=False,
                    has_skip_term=True,
                    scale_x=None,
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

        sru_cpu_impl = _lazy_load_cpu_kernel()
        if (sru_cpu_impl is not None) and (sru_cpu_impl != False):
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
                    scale_x.item() if scale_x is not None else 1.0
                )
            else:
                warnings.warn("Running SRU on CPU with grad_enabled=True. Are you sure?")
        else:
            warnings.warn("C++ kernel for SRU CPU inference was not loaded. "
                          "Use Python version instead.")

        mask_pad_ = mask_pad.view(length, batch, 1).float() if mask_pad is not None else mask_pad
        u = u.view(length, batch, bidir, d, k)
        forget_wc, reset_wc = weight_c.view(2, bidir, d)
        forget_bias, reset_bias = bias.view(2, bidir, d)

        if not has_skip_term:
            x_prime = None
        elif k == 3:
            x_prime = x.view(length, batch, bidir, d)
            x_prime = x_prime*scale_x if scale_x is not None else x_prime
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
            if x_prime is not None:
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
        input_size (int) : the number of dimensions in a single
            input sequence element. For example, if the input sequence
            is a sequence of word embeddings, `input_size` is the
            dimensionality of a single word embedding, e.g. 300.
        hidden_size (int) : the dimensionality of the hidden state
            of this cell.
        dropout (float) : a number between 0.0 and 1.0. The amount of dropout
            applied to `g(c_t)` internally in this cell.
        rnn_dropout (float) : the amount of dropout applied to the input of
            this cell.
        use_tanh (bool) : use tanh activation
        is_input_normalized (bool) : whether the input is normalized (e.g. batch norm / layer norm)
        bidirectional (bool) : whether or not to employ a bidirectional cell.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 dropout=0,
                 rnn_dropout=0,
                 bidirectional=False,
                 n_proj=0,
                 use_tanh=0,
                 #is_input_normalized=False,
                 highway_bias=0,
                 has_skip_term=True,
                 layer_norm=False,
                 rescale=True,
                 v1=False,
                 custom_u=None):

        super(SRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  # hidden size per direction
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.bidirectional = bidirectional
        #self.is_input_normalized = is_input_normalized
        self.has_skip_term = has_skip_term
        self.highway_bias = highway_bias
        self.v1 = v1
        self.rescale = rescale
        self.activation_type = 0
        self.activation = 'none'
        self.custom_u = custom_u
        if use_tanh:
            self.activation_type = 1
            self.activation = 'tanh'

        # projection dimension
        self.projection_size = 0
        if n_proj > 0 and n_proj < self.input_size and n_proj < self.output_size:
            self.projection_size = n_proj

        # number of sub-matrices used in SRU
        self.num_matrices = 3
        project_first = False
        if has_skip_term and self.input_size != self.output_size:
            if self.custom_u is not None:
                # If a custom u or v  is provided, project before u and v
                project_first = True
            else:
                self.num_matrices = 4

        if project_first:
            input_size = self.output_size
            self.input_to_hidden = nn.Linear(self.input_size, self.output_size)
        else:
            input_size = self.input_size
            self.input_to_hidden = None

        # make parameters
        if self.projection_size == 0:
            self.weight = nn.Parameter(torch.Tensor(
                input_size,
                self.output_size * self.num_matrices
            ))
        else:
            self.weight_proj = nn.Parameter(torch.Tensor(input_size, self.projection_size))
            self.weight = nn.Parameter(torch.Tensor(
                self.projection_size,
                self.output_size * self.num_matrices
            ))
        self.weight_c = nn.Parameter(torch.Tensor(self.output_size * 2))
        self.bias = nn.Parameter(torch.Tensor(self.output_size * 2))

        # scaling constant used in highway connections when rescale=True
        self.register_buffer('scale_x', torch.FloatTensor([0]))

        if layer_norm:
            # Use the true input_size here
            self.layer_norm = nn.LayerNorm(self.input_size)
        else:
            self.layer_norm = None
        self.reset_parameters()

    def reset_parameters(self):
        """
        Properly initialize the weights of SRU, following the same recipe as:
            Xavier init:  http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            Kaiming init: https://arxiv.org/abs/1502.01852

        """
        # initialize weights such that E[w_ij]=0 and Var[w_ij]=1/d
        d = self.weight.size(0)
        val_range = (3.0 / d)**0.5
        self.weight.data.uniform_(-val_range, val_range)
        if self.input_to_hidden is not None:
            val_range = (3.0 / self.input_to_hidden.weight.size(0))**0.5
            self.input_to_hidden.weight.data.uniform_(-val_range, val_range)
        if self.projection_size > 0:
            val_range = (3.0 / self.weight_proj.size(0))**0.5
            self.weight_proj.data.uniform_(-val_range, val_range)

        # initialize bias
        self.bias.data.zero_()
        bias_val, output_size = self.highway_bias, self.output_size
        self.bias.data[output_size:].zero_().add_(bias_val)

        # projection matrix as a tensor of size:
        #    (input_size, bidirection, hidden_size, num_matrices)
        w = self.weight.data.view(d, -1, self.hidden_size, self.num_matrices)
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

        # re-scale weights for dropout and normalized input for better gradient flow
        if self.dropout > 0:
            w[:, :, :, 0].mul_((1 - self.dropout)**0.5)
        if self.rnn_dropout > 0:
            w.mul_((1 - self.rnn_dropout)**0.5)

        # making weights smaller when layer norm is used. need more tests
        if self.layer_norm:
            w.mul_(0.1)
            #self.weight_c.data.mul_(0.25)

        self.scale_x.data[0] = 1
        if not (self.rescale and self.has_skip_term):
            return
        # scalar used to properly scale the highway output
        scale_val = (1 + math.exp(bias_val) * 2)**0.5
        self.scale_x.data[0] = scale_val
        if self.num_matrices == 4:
            w[:, :, :, 3].mul_(scale_val)

    def forward(self, input, c0=None, mask_pad=None):
        """
        This method computes `U`. In addition, it computes the remaining components
        in `SRU_Compute_GPU` or `SRU_Compute_CPU` and return the results.
        """

        if input.dim() != 2 and input.dim() != 3:
            raise ValueError("Input must be 2 or 3 dimensional")

        input_size, hidden_size = self.input_size, self.hidden_size
        batch_size = input.size(-2)
        if c0 is None:
            c0 = Variable(input.data.new(
                batch_size, self.output_size
            ).zero_())

        # apply layer norm before activation (i.e. before SRU computation)
        residual = input
        if self.layer_norm:
            input = self.layer_norm(input)

        # project if needed
        if self.input_to_hidden is not None:
            input = self.input_to_hidden(input)

        # apply dropout for multiplication
        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((batch_size, input_size), self.rnn_dropout)
            input = input * mask.expand_as(input)

        # compute U that's (length, batch_size, output_size, num_matrices)
        if self.custom_u is not None:
            U = self.custom_u(input)
        else:
            U = self.compute_U(input)

        # get the scaling constant; scale_x is a scalar
        scale_val = self.scale_x if self.rescale else None

        # Pytorch Function() doesn't accept NoneType in forward() call.
        # So we put mask_pad as class attribute as a work around
        if input.is_cuda:
            SRU_Compute = _lazy_load_cuda_kernel()(
                self.activation_type,
                hidden_size,
                self.bidirectional,
                self.has_skip_term,
                scale_val,
                mask_pad
            )
        else:
            SRU_Compute = SRU_CPU_class(
                self.activation_type,
                hidden_size,
                self.bidirectional,
                self.has_skip_term,
                scale_val,
                mask_pad
            )

        if self.training and (self.dropout > 0):
            bidir = 2 if self.bidirectional else 1
            mask_c = self.get_dropout_mask_((batch_size, self.output_size), self.dropout)
            h, c = SRU_Compute(U, residual, self.weight_c, self.bias, c0, mask_c)
        else:
            h, c = SRU_Compute(U, residual, self.weight_c, self.bias, c0)

        return h, c

    def compute_U(self, input):
        """
        SRU performs grouped matrix multiplication to transform
        the input (length, batch_size, input_size) into a tensor
        U of size (length, batch_size, output_size, num_matrices)
        """
        # collapse (length, batch_size) into one dimension if necessary
        x = input if input.dim() == 2 else input.contiguous().view(-1, self.input_size)
        if self.projection_size > 0:
            x_projected = x.mm(self.weight_proj)
            U = x_projected.mm(self.weight)
        else:
            U = x.mm(self.weight)
        return U

    def get_dropout_mask_(self, size, p):
        """
        Composes the dropout mask for the `SRUCell`.
        """
        w = self.weight.data
        return Variable(w.new(*size).bernoulli_(1 - p).div_(1 - p))

    def extra_repr(self):
        s = "{input_size}, {hidden_size}"
        if self.projection_size > 0:
            s += ", projection_size={projection_size}"
        if self.dropout > 0:
            s += ", dropout={dropout}"
        if self.rnn_dropout > 0:
            s += ", rnn_dropout={rnn_dropout}"
        if self.bidirectional:
            s += ", bidirectional={bidirectional}"
        if self.highway_bias != 0:
            s += ", highway_bias={highway_bias}"
        if self.activation_type != 0:
            s += ", activation={activation}"
        if self.v1:
            s += ", v1={v1}"
        s += ", rescale={rescale}"
        if not self.has_skip_term:
            s += ", has_skip_term={has_skip_term}"
        if self.layer_norm:
            s += ", layer_norm=True"
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
        layer_norm (bool) : whether or not to use layer normalization on the output of each layer
        bidirectional (bool) : whether or not to use bidirectional `SRUCell`s.
        is_input_normalized (bool) : whether the input is normalized (e.g. batch norm / layer norm)
        highway_bias (float) : initial bias of the highway gate, typicially <= 0
        nn_rnn_compatible_return (bool) : set to True to change the layout of returned state to match
            that of pytorch nn.RNN, ie (num_layers * num_directions, batch, hidden_size)
            (this will be slower, but can make SRU a dropin replacement for nn.RNN and nn.GRU)
        custom_u (nn.Module) : use a custom module to compute the U matrix given the input.
            The module must take as input a tensor of shape (seq_len, batch_size, hidden_size) and
            return a tensor of shape (seq_len, batch_size, hidden_size * 3)
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=2,
                 dropout=0,
                 rnn_dropout=0,
                 bidirectional=False,
                 projection_size=0,
                 use_tanh=False,
                 layer_norm=False,
                 #is_input_normalized=False,
                 highway_bias=0,
                 has_skip_term=True,
                 rescale=False,
                 v1=False,
                 nn_rnn_compatible_return=False,
                 custom_u=None):

        super(SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.projection_size = projection_size
        self.rnn_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.use_layer_norm = layer_norm
        self.has_skip_term = has_skip_term
        self.num_directions = 2 if bidirectional else 1
        self.nn_rnn_compatible_return = nn_rnn_compatible_return

        for i in range(num_layers):
            l = SRUCell(
                self.input_size if i == 0 else self.output_size,
                self.hidden_size,
                dropout=dropout if i + 1 != num_layers else 0,
                rnn_dropout=rnn_dropout,
                bidirectional=bidirectional,
                n_proj=projection_size,
                use_tanh=use_tanh,
                #is_input_normalized=is_input_normalized or (i > 0 and self.use_layer_norm),
                layer_norm=layer_norm,
                highway_bias=highway_bias,
                has_skip_term=has_skip_term,
                rescale=rescale,
                v1=v1,
                custom_u=copy.deepcopy(custom_u) if custom_u is not None else None
            )
            self.rnn_lst.append(l)

    def forward(self, input, c0=None, mask_pad=None):
        """
        Feeds `input` forward through `num_layers` `SRUCell`s, where `num_layers`
        is a parameter on the constructor of this class.

        parameters:
        - input (FloatTensor): (sequence_length, batch_size, input_size)
        - c0 (FloatTensor): (num_layers, batch_size, hidden_size * num_directions)
        - mask_pad (ByteTensor): (sequence_length, batch_size): set to 1 to ignore the value at that position

        input can be packed, which will lead to worse execution speed, but is compatible with many usages
        of nn.RNN.

        Return:
        - prevx: output: FloatTensor, (sequence_length, batch_size, num_directions * hidden_size)
        - lstc_stack: state:
            (FloatTensor): (num_layers, batch_size, num_directions * hidden_size) if not nn_rnn_compatible_return, else
            (FloatTensor): (num_layers * num_directions, batch, hidden_size)
        """

        # unpack packed, if input is packed. packing and then unpacking will be slower than not packing
        # at all, but makes SRU usage compatible with nn.RNN usage
        input_packed = isinstance(input, nn.utils.rnn.PackedSequence)
        if input_packed:
            input, lengths = nn.utils.rnn.pad_packed_sequence(input)
            max_length = lengths.max().item()
            mask_pad = torch.ByteTensor([[0] * l + [1] * (max_length - l) for l in lengths.tolist()])
            mask_pad = mask_pad.to(input.device).transpose(0, 1).contiguous()

        # The dimensions of `input` should be: `(sequence_length, batch_size, input_size)`.
        if input.dim() != 3:
            raise ValueError("There must be 3 dimensions for (length, batch_size, input_size)")

        if c0 is None:
            zeros = Variable(input.data.new(
                input.size(1), self.output_size
            ).zero_())
            c0 = [ zeros for i in range(self.num_layers) ]
        else:
            # The dimensions of `c0` should be: `(num_layers, batch_size, hidden_size * dir_)`.
            if c0.dim() != 3:
                raise ValueError("There must be 3 dimensions for (num_layers, batch_size, output_size)")
            c0 = [ x.squeeze(0) for x in c0.chunk(self.num_layers, 0) ]

        prevx = input
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prevx, c0[i], mask_pad=mask_pad)
            prevx = h
            lstc.append(c)

        if input_packed:
            prevx = nn.utils.rnn.pack_padded_sequence(prevx, lengths, enforce_sorted=False)

        lstc_stack = torch.stack(lstc)
        if self.nn_rnn_compatible_return:
            batch_size = input.size(1)
            lstc_stack = lstc_stack.view(self.num_layers, batch_size, self.num_directions, self.output_size)
            lstc_stack = lstc_stack.transpose(1, 2).contiguous()
            lstc_stack = lstc_stack.view(self.num_layers * self.num_directions, batch_size, self.output_size)
        return prevx, lstc_stack

    def reset_parameters(self):
        for rnn in self.rnn_lst:
            rnn.reset_parameters()

