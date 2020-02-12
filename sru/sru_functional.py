import os
import sys
import copy
import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SRU_Compute_CPU():
    """CPU version of the core SRU computation.

    Has the same interface as SRU_Compute_GPU() but is a regular Python function
    instead of a torch.autograd.Function because we don't implement backward()
    explicitly.
    """

    @staticmethod
    def apply(u, x, weight_c, bias,
              init,
              activation_type,
              d,
              bidirectional,
              has_skip_term,
              scale_x,
              mask_c=None,
              mask_pad=None):
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

        is_custom = len(weight_c.size()) > 1

        sru_cpu_impl = _lazy_load_cpu_kernel()
        if (sru_cpu_impl is not None) and (sru_cpu_impl != False):
            if not torch.is_grad_enabled():
                assert mask_c is None
                cpu_forward = sru_cpu_impl.cpu_bi_forward if bidirectional else \
                              sru_cpu_impl.cpu_forward
                mask_pad_ = torch.FloatTensor() if mask_pad is None else mask_pad.float()
                return cpu_forward(
                    u.contiguous(),
                    x.contiguous(),
                    weight_c.contiguous(),
                    bias,
                    init,
                    mask_pad_,
                    length,
                    batch,
                    d,
                    k,
                    activation_type,
                    has_skip_term,
                    scale_x.item() if scale_x is not None else 1.0,
                    is_custom
                )
            else:
                warnings.warn("Running SRU on CPU with grad_enabled=True. Are you sure?")
        else:
            warnings.warn("C++ kernel for SRU CPU inference was not loaded. "
                          "Use Python version instead.")

        mask_pad_ = mask_pad.view(length, batch, 1).float() if mask_pad is not None else mask_pad
        u = u.contiguous().view(length, batch, bidir, d, k)

        if is_custom:
            weight_c = weight_c.view(length, batch, bidir, d, 2)
            forget_wc = weight_c[..., 0]
            reset_wc = weight_c[..., 1]
        else:
            forget_wc, reset_wc = weight_c.view(2, bidir, d)

        forget_bias, reset_bias = bias.view(2, bidir, d)

        if not has_skip_term:
            x_prime = None
        elif k == 3:
            x_prime = x.view(length, batch, bidir, d)
            x_prime = x_prime * scale_x if scale_x is not None else x_prime
        else:
            x_prime = u[..., 3]

        h = x.new_zeros(length, batch, bidir, d)

        if init is None:
            c_init = x.new_zeros(size=(batch, bidir, d))
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
            if is_custom:
                fw = forget_wc[:, :, di, :].chunk(length)
                rw = reset_wc[:, :, di, :].chunk(length)
            else:
                fw = forget_wc[di].expand(batch, d)
                rw = reset_wc[di].expand(batch, d)
            u0 = u[:, :, di, :, 0].chunk(length)
            u1 = (u[:, :, di, :, 1] + fb).chunk(length)
            u2 = (u[:, :, di, :, 2] + rb).chunk(length)
            if x_prime is not None:
                xp = x_prime[:, :, di, :].chunk(length)

            for t in time_seq:
                if is_custom:
                    forget_t = (u1[t] + c_prev*fw[t]).sigmoid()
                    reset_t = (u2[t] + c_prev*rw[t]).sigmoid()
                else:
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
                 highway_bias=0,
                 has_skip_term=True,
                 layer_norm=False,
                 rescale=True,
                 v1=False,
                 custom_m=None):

        super(SRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  # hidden size per direction
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.has_skip_term = has_skip_term
        self.highway_bias = highway_bias
        self.v1 = v1
        self.rescale = rescale
        self.activation_type = 0
        self.activation = 'none'
        self.custom_m = custom_m
        if use_tanh:
            self.activation_type = 1
            self.activation = 'tanh'

        # projection dimension
        self.projection_size = 0
        if n_proj > 0 and n_proj < self.input_size and n_proj < self.output_size:
            self.projection_size = n_proj

        # number of sub-matrices used in SRU
        self.num_matrices = 3
        if has_skip_term and self.input_size != self.output_size:
            self.num_matrices = 4

        # make parameters
        if self.custom_m is None:
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
        self.weight_c = nn.Parameter(torch.Tensor(2 * self.output_size))
        self.bias = nn.Parameter(torch.Tensor(2 * self.output_size))

        # scaling constant used in highway connections when rescale=True
        self.register_buffer('scale_x', torch.FloatTensor([0]))

        if layer_norm:
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
        # initialize bias and scaling constant
        self.bias.data.zero_()
        bias_val, output_size = self.highway_bias, self.output_size
        self.bias.data[output_size:].zero_().add_(bias_val)
        self.scale_x.data[0] = 1
        if self.rescale and self.has_skip_term:
            # scalar used to properly scale the highway output
            scale_val = (1 + math.exp(bias_val) * 2)**0.5
            self.scale_x.data[0] = scale_val

        if self.custom_m is None:
            # initialize weights such that E[w_ij]=0 and Var[w_ij]=1/d
            d = self.weight.size(0)
            val_range = (3.0 / d)**0.5
            self.weight.data.uniform_(-val_range, val_range)
            if self.projection_size > 0:
                val_range = (3.0 / self.weight_proj.size(0))**0.5
                self.weight_proj.data.uniform_(-val_range, val_range)

            # projection matrix as a tensor of size:
            #    (input_size, bidirection, hidden_size, num_matrices)
            w = self.weight.data.view(d, -1, self.hidden_size, self.num_matrices)

            # re-scale weights for dropout and normalized input for better gradient flow
            if self.dropout > 0:
                w[:, :, :, 0].mul_((1 - self.dropout)**0.5)
            if self.rnn_dropout > 0:
                w.mul_((1 - self.rnn_dropout)**0.5)

            # making weights smaller when layer norm is used. need more tests
            if self.layer_norm:
                w.mul_(0.1)
                #self.weight_c.data.mul_(0.25)

            # properly scale the highway output
            if self.rescale and self.has_skip_term and self.num_matrices == 4:
                scale_val = (1 + math.exp(bias_val) * 2)**0.5
                w[:, :, :, 3].mul_(scale_val)

        if not self.v1:
            # intialize weight_c such that E[w]=0 and Var[w]=1
            self.weight_c.data.uniform_(-3.0**0.5, 3.0**0.5)

            # rescale weight_c and the weight of sigmoid gates with a factor of sqrt(0.5)
            if self.custom_m is None:
                w[:, :, :, 1].mul_(0.5**0.5)
                w[:, :, :, 2].mul_(0.5**0.5)
            self.weight_c.data.mul_(0.5**0.5)
        else:
            self.weight_c.data.zero_()
            self.weight_c.requires_grad = False

    def forward(self, input, c0=None, mask_pad=None, **kwargs):
        """
        This method computes `U`. In addition, it computes the remaining components
        in `SRU_Compute_GPU` or `SRU_Compute_CPU` and return the results.
        """

        if input.dim() != 2 and input.dim() != 3:
            raise ValueError("Input must be 2 or 3 dimensional")

        input_size, hidden_size = self.input_size, self.hidden_size
        batch_size = input.size(-2)
        if c0 is None:
            c0 = input.new_zeros(batch_size, self.output_size)

        # apply layer norm before activation (i.e. before SRU computation)
        residual = input
        if self.layer_norm:
            input = self.layer_norm(input)

        # apply dropout for multiplication
        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((batch_size, input.size(-1)), self.rnn_dropout)
            input = input * mask.expand_as(input)

        # compute U, V
        #   U is (length, batch_size, output_size * num_matrices)
        #   V is (output_size*2,) or (length, batch_size, output_size * 2) if provided
        if self.custom_m is None:
            U = self.compute_U(input)
            V = self.weight_c
        else:
            ret = self.custom_m(input, c0=c0, mask_pad=mask_pad, **kwargs)
            if isinstance(ret, tuple) or isinstance(ret, list):
                if len(ret) > 2:
                    raise Exception("Custom module must return 1 or 2 tensors but got {}.".format(
                        len(ret)
                    ))
                U, V = ret[0], ret[1] + self.weight_c
            else:
                U, V = ret, self.weight_c

            if U.size(-1) != self.output_size * self.num_matrices:
                raise ValueError("U must have a last dimension of {} but got {}.".format(
                    self.output_size * self.num_matrices,
                    U.size(-1)
                ))
            if V.size(-1) != self.output_size * 2:
                raise ValueError("V must have a last dimension of {} but got {}.".format(
                    self.output_size * 2,
                    V.size(-1)
                ))

        # get the scaling constant; scale_x is a scalar
        scale_val = self.scale_x if self.rescale else None

        # get dropout mask
        if self.training and (self.dropout > 0):
            mask_c = self.get_dropout_mask_((batch_size, self.output_size), self.dropout)
        else:
            mask_c = None

        SRU_Compute = _lazy_load_cuda_kernel() if input.is_cuda else SRU_Compute_CPU
        h, c = SRU_Compute.apply(U, residual, V, self.bias, c0,
                                 self.activation_type,
                                 hidden_size,
                                 self.bidirectional,
                                 self.has_skip_term,
                                 scale_val,
                                 mask_c,
                                 mask_pad)
        return h, c

    def compute_U(self, input):
        """
        SRU performs grouped matrix multiplication to transform
        the input (length, batch_size, input_size) into a tensor
        U of size (length * batch_size, output_size * num_matrices)
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
        b = self.bias.data
        return b.new(*size).bernoulli_(1 - p).div_(1 - p)

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
        if self.custom_m is not None:
           s += ",\n  custom_m=" + str(self.custom_m)
        return s.format(**self.__dict__)

    def __repr__(self):
        s = self.extra_repr()
        if len(s.split('\n')) == 1:
            return "{}({})".format(self.__class__.__name__, s)
        else:
            return "{}({}\n)".format(self.__class__.__name__, s)


class SRU(nn.Module):
    """
    PyTorch SRU model. In effect, simply wraps an arbitrary number of contiguous `SRUCell`s, and
    returns the matrix and hidden states , as well as final memory cell (`c_t`), from the last of
    these `SRUCell`s.

    Args:
        input_size (int) : the number of dimensions in a single input sequence element. For example,
            if the input sequence is a sequence of word embeddings, `input_size` is the dimensionality
            of a single word embedding, e.g. 300.
        hidden_size (int) : the dimensionality of the hidden state of the SRU cell.
        num_layers (int) : number of `SRUCell`s to use in the model.
        dropout (float) : a number between 0.0 and 1.0. The amount of dropout applied to `g(c_t)`
            internally in each `SRUCell`.
        rnn_dropout (float) : the amount of dropout applied to the input of each `SRUCell`.
        use_tanh (bool) : use tanh activation
        layer_norm (bool) : whether or not to use layer normalization on the output of each layer
        bidirectional (bool) : whether or not to use bidirectional `SRUCell`s.
        highway_bias (float) : initial bias of the highway gate, typicially <= 0
        nn_rnn_compatible_return (bool) : set to True to change the layout of returned state to
            match that of pytorch nn.RNN, ie (num_layers * num_directions, batch, hidden_size)
            (this will be slower, but can make SRU a drop-in replacement for nn.RNN and nn.GRU)
        custom_m (nn.Module or List[nn.Module]) : use a custom module to compute the U matrix (and V
            matrix) given the input. The module must take as input a tensor of shape (seq_len,
            batch_size, hidden_size).
            It returns a tensor U of shape (seq_len, batch_size, hidden_size * 3), or one optional
            tensor V of shape (seq_len, batch_size, hidden_size * 2).
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
                 highway_bias=0,
                 has_skip_term=True,
                 rescale=False,
                 v1=False,
                 nn_rnn_compatible_return=False,
                 custom_m=None,
                 proj_input_to_hidden_first=False):

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
        if proj_input_to_hidden_first and input_size != output_size:
            first_layer_input_size = output_size
            self.input_to_hidden = nn.Linear(input_size, self.output_size, bias=False)
        else:
            first_layer_input_size = input_size
            self.input_to_hidden = None

        for i in range(num_layers):
            # get custom modules when provided
            custom_m_i = None
            if custom_m is not None:
                custom_m_i = custom_m[i] if isinstance(custom_m, list) else copy.deepcopy(custom_m)
            # create the i-th SRU layer
            l = SRUCell(
                first_layer_input_size if i == 0 else self.output_size,
                self.hidden_size,
                dropout=dropout if i + 1 != num_layers else 0,
                rnn_dropout=rnn_dropout,
                bidirectional=bidirectional,
                n_proj=projection_size,
                use_tanh=use_tanh,
                layer_norm=layer_norm,
                highway_bias=highway_bias,
                has_skip_term=has_skip_term,
                rescale=rescale,
                v1=v1,
                custom_m=custom_m_i
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
            zeros = input.data.new(
                input.size(1), self.output_size
            ).zero_()
            c0 = [ zeros for i in range(self.num_layers) ]
        else:
            # The dimensions of `c0` should be: `(num_layers, batch_size, hidden_size * dir_)`.
            if c0.dim() != 3:
                raise ValueError("There must be 3 dimensions for (num_layers, batch_size, output_size)")
            c0 = [ x.squeeze(0) for x in c0.chunk(self.num_layers, 0) ]

        prevx = input if self.input_to_hidden is None else self.input_to_hidden(input)
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

    def make_backward_compatible(self):
        self.nn_rnn_compatible_return = getattr(self, 'nn_rnn_compatible_return', False)

        # version <= 2.1.7
        if hasattr(self, 'n_in'):
            if len(self.ln_lst):
                raise Exception("Layer norm is not backward compatible for sru<=2.1.7")
            if self.use_weight_norm:
                raise Exception("Weight norm removed in sru>=2.1.9")
            self.input_size = self.n_in
            self.hidden_size = self.n_out
            self.output_size = self.out_size
            self.num_layers = self.depth
            self.projection_size = self.n_proj
            self.use_layer_norm = False
            for cell in self.rnn_lst:
                cell.input_size = cell.n_in
                cell.hidden_size = cell.n_out
                cell.output_size = cell.n_out * 2 if cell.bidirectional else cell.n_out
                cell.num_matrices = cell.k
                cell.projection_size = cell.n_proj
                cell.layer_norm = None
                if cell.activation_type > 1:
                    raise Exception("ReLU or SeLU activation removed in sru>=2.1.9")

        # version <= 2.1.9
        if not hasattr(self, 'input_to_hidden'):
            self.input_to_hidden = None
            for cell in self.rnn_lst:
                cell.custom_m = None
