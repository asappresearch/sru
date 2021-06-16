import copy
import warnings
import math
from typing import List, Tuple, Union, Optional, Sequence, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from sru.ops import (elementwise_recurrence_inference,
                     elementwise_recurrence_gpu,
                     elementwise_recurrence_naive)


class SRUCell(nn.Module):
    """
    A single SRU layer as per `LSTMCell`, `GRUCell` in Pytorch.
    """

    __constants__ = ['input_size', 'hidden_size', 'output_size', 'rnn_dropout',
                     'dropout', 'bidirectional', 'has_skip_term', 'highway_bias',
                     'v1', 'rescale', 'activation_type', 'activation', 'transform_module',
                     'projection_size', 'num_matrices', 'layer_norm',
                     'scale_x', 'normalize_after', 'weight_c_init', ]

    scale_x: Tensor

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 dropout: float = 0.0,
                 rnn_dropout: float = 0.0,
                 bidirectional: bool = False,
                 projection_size: int = 0,
                 highway_bias: float = 0.0,
                 layer_norm: bool = False,
                 normalize_after: bool = True,
                 transform_module: Optional[nn.Module] = None,
                 rescale: bool = False,
                 has_skip_term: bool = True,
                 use_tanh: bool = False,
                 v1: bool = False,
                 amp_recurrence_fp16: bool = True,
                 weight_c_init: float = 1.0):
        """Initialize the SRUCell module.

        Parameters
        ----------
        input_size: int
            the number of features in the input `x`
        hidden_size: int
            the number of features in the hidden state *for each
            direction*
        dropout: float, optional
            the dropout value applied between layers (default=0)
        rnn_dropout: float, optional
            [DEPRECATED] the variational dropout value (default=0)
            This option is deprecated because minimal performance
            improvement, and increases codebase size. This option will
            be removed at the next major version upgrade
        bidirectional: bool, optional
            if True, set the module as a bidirectional SRU
            (default=False)
        projection_size: int, optional
            if non-zero, factorize the ``weight`` parameter matrix as a
            product of two parameter matrices, using an innder dimension
            ``projection_size`` (default=0)
        use_tanh: bool, optional
            [DEPRECATED] if True, apply `tanh` activation to the hidden
            state (default=False). `tanh` is deprecated because minimal
            performance improvement, and increases codebase size. This
            option will be removed at the next major version upgrade.
        highway_bias: float, optional
            the initial value of the bias used in the highway (sigmoid)
            gate (defulat=0)
        has_skip_term: bool, optional
            whether to include a residual connection for output hidden
            state `h` (default=True)
        layer_norm: bool, optional
            whether to apply pre- layer normalization for this layer
            (default=False)
        rescale: bool, optional
            whether to apply a constant rescaling multiplier for the
            residual term (default=True)
        v1: bool, optional
            [DEPRECATED] whether to use the an ealier v1 implementation
            of SRU (default=False)
        transform_module: nn.Module, optional
            use the give module instead of the batched matrix
            multiplication to compute the intermediate representations U
            needed for the elementwise recurrrence operation
            (default=None)
        amp_recurrence_fp16: Type, optional
            When using AMP autocast, selects which type to use
            for recurrence custom kernel.
            False: torch.float32, True: torch.float16
        normalize_after: bool, optional
            if True use post layer norm, else pre layer norm
            (default=True)
        weight_c_init: float, optional
            size of uniform initiatialization of weight_c
            (default=1.0)
        """
        super(SRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  # hidden size per direction
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.rnn_dropout = float(rnn_dropout)
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.has_skip_term = has_skip_term
        self.highway_bias = highway_bias
        self.v1 = v1
        self.rescale = rescale
        self.activation_type = 0
        self.activation = 'none'
        if use_tanh:
            self.activation_type = 1
            self.activation = 'tanh'
        self.amp_recurrence_fp16 = amp_recurrence_fp16
        self.normalize_after = normalize_after
        self.weight_c_init = weight_c_init

        # projection dimension
        self.projection_size = projection_size

        # number of sub-matrices used in SRU
        self.num_matrices = 3
        if has_skip_term and self.input_size != self.output_size:
            self.num_matrices = 4

        if transform_module is None:
            # create an appropriate transform_module, depending on whether we are using projection
            # or not
            if self.projection_size == 0:
                # use an nn.Linear
                transform_module = nn.Linear(
                    input_size, self.output_size * self.num_matrices, bias=False)
            else:
                # use a Sequential[nn.Linear, nn.Linear]
                transform_module = nn.Sequential(
                    nn.Linear(input_size, self.projection_size, bias=False),
                    nn.Linear(
                        self.projection_size, self.output_size * self.num_matrices, bias=False),
                )
        self.transform_module: nn.Module = transform_module

        self.weight_c = nn.Parameter(torch.Tensor(2 * self.output_size))
        self.bias = nn.Parameter(torch.Tensor(2 * self.output_size))

        # scaling constant used in highway connections when rescale=True
        self.register_buffer('scale_x', torch.FloatTensor([0]))

        self.layer_norm: Optional[nn.Module] = None
        if layer_norm:
            if normalize_after:
                self.layer_norm = nn.LayerNorm(self.output_size)
            else:
                self.layer_norm = nn.LayerNorm(self.input_size)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weights of SRU.
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

        def reset_module_parameters(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif hasattr(module, 'reset_parameters'):
                module.reset_parameters()  # type: ignore
            elif isinstance(module, nn.Sequential):
                for m in module:
                    reset_module_parameters(m)
            else:
                warnings.warn("Unable to reset parameters for custom module. "
                              "reset_parameters() method not found for custom module. "
                              + module.__class__.__name__)

        reset_module_parameters(self.transform_module)

        if not self.v1:
            self.weight_c.data.uniform_(-self.weight_c_init, self.weight_c_init)
        else:
            self.weight_c.data.zero_()
            self.weight_c.requires_grad = False

    def forward(self,
                input: Tensor,
                c0: Optional[Tensor] = None,
                mask_pad: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """The forward method of the SRU layer.
        """

        if input.dim() != 2 and input.dim() != 3:
            raise ValueError("Input must be 2 or 3 dimensional")

        batch_size = input.size(-2)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.output_size, dtype=input.dtype,
                             device=input.device)

        # apply layer norm before activation (i.e. before SRU computation)
        residual = input

        layer_norm = self.layer_norm
        if layer_norm is not None:
            if not self.normalize_after:
                input = layer_norm(input)

        # apply dropout for multiplication
        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((batch_size, input.size(-1)), self.rnn_dropout)
            input = input * mask.expand_as(input)

        # get the scaling constant; scale_x is a scalar
        scale_val: Optional[Tensor] = None
        scale_val = self.scale_x if self.rescale else None

        # get dropout mask
        mask_c: Optional[Tensor] = None
        if self.training and (self.dropout > 0):
            mask_c = self.get_dropout_mask_((batch_size, self.output_size),
                                            self.dropout)

        # compute U, V
        #   U is (length, batch_size, output_size * num_matrices)
        #   V is (output_size*2,) or (length, batch_size, output_size * 2) if provided
        U, V = self.compute_UV(input, c0, mask_pad)

        # apply elementwise recurrence to get hidden states h and c
        h, c = self.apply_recurrence(U, V, residual, c0, scale_val, mask_c, mask_pad)

        if layer_norm is not None:
            if self.normalize_after:
                h = layer_norm(h)

        return h, c

    def apply_recurrence(self,
                         U: Tensor,
                         V: Tensor,
                         residual: Tensor,
                         c0: Tensor,
                         scale_val: Optional[Tensor],
                         mask_c: Optional[Tensor],
                         mask_pad: Optional[Tensor]) -> List[Tensor]:
        """
        Apply the elementwise recurrence computation on given input
        tensors

        """
        if not torch.jit.is_scripting():
            if self.bias.is_cuda:
                return elementwise_recurrence_gpu(U, residual, V, self.bias, c0,
                                                  self.activation_type,
                                                  self.hidden_size,
                                                  self.bidirectional,
                                                  self.has_skip_term,
                                                  scale_val, mask_c, mask_pad,
                                                  self.amp_recurrence_fp16)
            else:
                return elementwise_recurrence_naive(U, residual, V, self.bias, c0,
                                                    self.activation_type,
                                                    self.hidden_size,
                                                    self.bidirectional,
                                                    self.has_skip_term,
                                                    scale_val, mask_c, mask_pad)
        else:
            return elementwise_recurrence_inference(U, residual, V, self.bias, c0,
                                                    self.activation_type,
                                                    self.hidden_size,
                                                    self.bidirectional,
                                                    self.has_skip_term,
                                                    scale_val, mask_c, mask_pad)

    def compute_UV(self,
                   input: Tensor,
                   c0: Optional[Tensor],
                   mask_pad: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        SRU performs grouped matrix multiplication to transform the
        input (length, batch_size, input_size) into a tensor U of size
        (length * batch_size, output_size * num_matrices).

        U will be computed by
        the given transform_module. The module can optionally return an
        additional tensor V (length, batch_size, output_size * 2) that
        will be added to the hidden-to-hidden coefficient terms in
        sigmoid gates, i.e., (V[t, b, d] + weight_c[d]) * c[t-1].

        """
        ret = self.transform_module(input)
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
        return U, V

    def get_dropout_mask_(self,
                          size: Tuple[int, int],
                          p: float) -> Tensor:
        """
        Composes the dropout mask for the `SRUCell`.
        """
        b = self.bias.data
        return b.new_empty(size).bernoulli_(1 - p).div_(1 - p)

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
        s += ",\n  transform_module=" + str(self.transform_module)
        return s.format(**self.__dict__)

    def __repr__(self):
        s = self.extra_repr()
        if len(s.split('\n')) == 1:
            return "{}({})".format(self.__class__.__name__, s)
        else:
            return "{}({}\n)".format(self.__class__.__name__, s)


class SRU(nn.Module):
    """
    Implementation of Simple Recurrent Unit (SRU)
    """

    __constants__ = ['input_size', 'hidden_size', 'output_size', 'num_layers',
                     'dropout', 'rnn_dropout', 'projection_size', 'rnn_lst',
                     'bidirectional', 'use_layer_norm', 'has_skip_term',
                     'num_directions', 'nn_rnn_compatible_return', 'input_to_hidden']

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 rnn_dropout: float = 0.0,
                 bidirectional: bool = False,
                 projection_size: Union[int, Sequence[int]] = 0,
                 highway_bias: float = 0.0,
                 layer_norm: bool = False,
                 normalize_after: bool = False,
                 transform_module: Optional[Union[nn.Module, Sequence[nn.Module]]] = None,
                 has_skip_term: bool = True,
                 rescale: bool = False,
                 use_tanh: bool = False,
                 v1: bool = False,
                 nn_rnn_compatible_return: bool = False,
                 proj_input_to_hidden_first: bool = False,
                 amp_recurrence_fp16: bool = True,
                 weight_c_init: float = 1.0):
        """Initialize the SRU module.

        Parameters
        ----------
        input_size: int
            the number of features in the input `x`
        hidden_size: int
            the number of features in the hidden state *for each
            direction*
        num_layers: int
            the number of stacked SRU layers (default=2)
        dropout: float, optional
            the dropout value applied between layers (default=0)
        rnn_dropout: float, optional
            [DEPRECATED] the variational dropout value (default=0)
            This option is deprecated because minimal performance
            improvement, and increases codebase size. This option will
            be removed at the next major version upgrade
        bidirectional: bool, optional
            if True, set the module as a bidirectional SRU
            (default=False)
        projection_size: Union[int, Sequence[int]]
            if non-zero, factorize the ``weight`` parameter in each
            layeras a product of two parameter matrices, using an inner
            dimension ``projection_size`` (default=0)
            If a sequence, length must equal number of layers, and
            values are projection size for each layer
        use_tanh: bool, optional
            [DEPRECATED] if True, apply `tanh` activation to the hidden
            state (default=False). `tanh` is deprecated because minimal
            performance improvement, and increases codebase size. This
            option will be removed at the next major version upgrade.
        layer_norm: bool, optional
            whether to apply pre- layer normalization for this layer
            (default=False)
        highway_bias: float, optional
            the initial value of the bias used in the highway (sigmoid)
            gate (defulat=0)
        has_skip_term: bool, optional
            whether to include a residual connection for output hidden
            state `h` (default=True)
        rescale: bool, optional
            whether to apply a constant rescaling multiplier for the
            residual term (default=False)
        v1: bool, optional
            [DEPRECATED] whether to use the an ealier v1 implementation
            of SRU (default=False)
        transform_module: Union[nn.Module, Sequence[nn.Module]], optional
            use the given module(s) instead of the batched matrix
            multiplication to compute the intermediate representations U
            needed for the elementwise recurrrence operation.  The
            module must take input x of shape (seq_len, batch_size,
            hidden_size). It returns a tensor U of shape (seq_len,
            batch_size, hidden_size * num_matrices), and one optional
            tensor V of shape (seq_len, batch_size, hidden_size * 2).
            (default=None)
        amp_recurrence_fp16: Type, optional
            When using AMP autocast, selects which type to use
            for recurrence custom kernel.
            False: torch.float32, True: torch.float16
        normalize_after: bool
            if True use post layer norm, else use pre layer norm
        weight_c_init: float, optional
            if not None, then size of uniform initiatialization of weight_c
            (default 1.0)
        """

        super(SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.projection_size = projection_size
        self.bidirectional = bidirectional
        self.use_layer_norm = layer_norm
        self.has_skip_term = has_skip_term
        self.num_directions = 2 if bidirectional else 1
        self.nn_rnn_compatible_return = nn_rnn_compatible_return
        self.input_to_hidden = None
        if proj_input_to_hidden_first and input_size != self.output_size:
            first_layer_input_size = self.output_size
            self.input_to_hidden = nn.Linear(input_size, self.output_size, bias=False)
        else:
            first_layer_input_size = input_size
        self.amp_recurrence_fp16 = amp_recurrence_fp16

        if rnn_dropout > 0:
            warnings.warn("rnn_dropout > 0 is deprecated and will be removed in"
                          "next major version of SRU. Please use dropout instead.")
        if use_tanh:
            warnings.warn("use_tanh = True is deprecated and will be removed in"
                          "next major version of SRU.")

        rnn_lst = nn.ModuleList()
        for i in range(num_layers):
            # get custom modules when provided
            transform_module_i = None
            if transform_module is not None:
                transform_module_i = transform_module[i] if isinstance(
                    transform_module, list) else copy.deepcopy(transform_module)
            _projection_size = projection_size if isinstance(
                projection_size, int) else projection_size[i]
            # create the i-th SRU layer
            layer_i = SRUCell(
                first_layer_input_size if i == 0 else self.output_size,
                self.hidden_size,
                dropout=dropout if i + 1 != num_layers else 0,
                rnn_dropout=rnn_dropout,
                bidirectional=bidirectional,
                projection_size=_projection_size,
                use_tanh=use_tanh,
                layer_norm=layer_norm,
                highway_bias=highway_bias,
                has_skip_term=has_skip_term,
                rescale=rescale,
                v1=v1,
                transform_module=transform_module_i,
                amp_recurrence_fp16=amp_recurrence_fp16,
                normalize_after=normalize_after,
                weight_c_init=weight_c_init,
            )
            rnn_lst.append(layer_i)
        self.rnn_lst = rnn_lst

    def __getitem__(self, n: int) -> SRUCell:
        """
        returns n'th layer srucell
        """
        return self.rnn_lst[n]

    def forward(self, input: Tensor,
                c0: Optional[Tensor] = None,
                mask_pad: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """The forward method of SRU module

        Parameters
        ----------
        input: Tensor
            the input feature. shape: (length, batch_size, input_size)
        c0: Tensor, optional
            the initial internal hidden state. shape: (num_layers,
            batch_size, output_size) where
            output_size = hidden_size * num_direction
        mask_pad: Tensor, optional
            the mask where a non-zero value indicates if an input token
            is pad token that should be ignored in forward and backward
            computation. shape: (length, batch_size)

        Returns
        ----------
        h: Tensor
            the output hidden state. shape: (length, batch_size,
            output_size) where
            output_size = hidden_size * num_direction
        c: Tensor
            the last internal hidden state. shape: (num_layers,
            batch_size, output_size), or (num_layers * num_directions,
            batch_size, hidden_size) if `nn_rnn_compatible_return` is
            set `True`

        """
        # unpack packed, if input is packed. packing and then unpacking will be slower than not
        # packing at all, but makes SRU usage compatible with nn.RNN usage
        orig_input = input
        if isinstance(orig_input, PackedSequence):
            input, lengths = nn.utils.rnn.pad_packed_sequence(input)
            max_length = lengths.max().item()
            mask_pad = torch.ByteTensor([[0] * length + [1] * (max_length - length)
                                        for length in lengths.tolist()])
            mask_pad = mask_pad.to(input.device).transpose(0, 1).contiguous()

        # The dimensions of `input` should be: `(sequence_length, batch_size, input_size)`.
        if input.dim() != 3:
            raise ValueError("There must be 3 dimensions for (length, batch_size, input_size)")

        if c0 is None:
            zeros = torch.zeros(input.size(1), self.output_size, dtype=input.dtype,
                                device=input.device)
            c0_ = [zeros for i in range(self.num_layers)]
        else:
            # The dimensions of `c0` should be: `(num_layers, batch_size, hidden_size * dir_)`.
            if c0.dim() != 3:
                raise ValueError("c0 must be 3 dim (num_layers, batch_size, output_size)")
            c0_ = [x.squeeze(0) for x in c0.chunk(self.num_layers, 0)]

        if self.input_to_hidden is None:
            prevx = input
        else:
            prevx = self.input_to_hidden(input)
        lstc = []
        i = 0
        for rnn in self.rnn_lst:
            h, c = rnn(prevx, c0_[i], mask_pad=mask_pad)
            prevx = h
            lstc.append(c)
            i += 1

        lstc_stack = torch.stack(lstc)
        if self.nn_rnn_compatible_return:
            batch_size = input.size(1)
            lstc_stack = lstc_stack.view(self.num_layers, batch_size,
                                         self.num_directions, self.hidden_size)
            lstc_stack = lstc_stack.transpose(1, 2).contiguous()
            lstc_stack = lstc_stack.view(self.num_layers * self.num_directions,
                                         batch_size, self.hidden_size)

        if isinstance(orig_input, PackedSequence):
            prevx = nn.utils.rnn.pack_padded_sequence(prevx, lengths, enforce_sorted=False)
            return prevx, lstc_stack
        else:
            return prevx, lstc_stack

    def reset_parameters(self):
        for rnn in self.rnn_lst:
            rnn.reset_parameters()
        if self.input_to_hidden is not None:
            self.input_to_hidden.reset_parameters()


class SRUppProjectedLinear(nn.Module):
    """
    Projected linear module used in SRU++ module.
    """

    __constants__ = ['in_features', 'out_features', 'proj_features']

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 proj_features: int,
                 dropout: float = 0.0,
                 layer_norm: bool = False):
        """Initialize the projected linear module.

        Parameters
        ----------
        in_features: int
            the number of input features.
        out_features: int
            the number of output features.
        proj_features: int
            the number of features used for attention computation. The input is projected into
            this dimension first. After that the module apply the query-key-value attention
            computation. The output is projected to dimension `out_features`.
        dropout: float, optional
            dropout probability applied after attention computation and before the final projection
            (default=0.0).
        layer_norm: bool, optional
            whether to apply layer normalization within the projected linear module.
        """
        super(SRUppProjectedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.proj_features = proj_features
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_features, proj_features, bias=False)
        self.linear2 = nn.Linear(proj_features, out_features, bias=False)
        self.layer_norm: Optional[nn.Module] = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(proj_features)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            self.linear1.bias.data.zero_()
        if self.linear2.bias is not None:
            self.linear2.bias.data.zero_()
        if self.dropout.p > 0:
            self.linear2.weight.data.mul_((1 - self.dropout.p)**0.5)

    def forward(self,
                input: Tensor,
                mask_pad: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                memory: Optional[Tensor] = None,
                memory_mask_pad: Optional[Tensor] = None) -> Tensor:
        """The forward method.
        """
        output = self.linear1(input)
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = self.linear2(self.dropout(output))
        return output


class SRUppAttention(nn.Module):
    """
    Self-attention module used in SRU++ module.
    """

    __constants__ = ['in_features', 'out_features', 'proj_features', 'num_heads',
                     'attn_dropout', 'rezero_init_alpha']

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 proj_features: int,
                 num_heads: int = 1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 rezero_init_alpha: float = 0.0,
                 layer_norm: bool = False,
                 normalize_after: bool = True):
        """Initialize the self-attention module.

        Parameters
        ----------
        in_features: int
            the number of input features.
        out_features: int
            the number of output features.
        proj_features: int
            the number of features used for attention computation. The input is projected into
            this dimension first. After that the module apply the query-key-value attention
            computation. The output is projected to dimension `out_features`.
        num_heads: int, optional
            the number of attention heads used. `proj_features` must be multipler of this value
            (default=1).
        dropout: float, optional
            dropout probability applied after attention computation and before the final projection
            (default=0.0).
        attn_dropout: float, optional
            dropout probability applied on attention map.
        rezero_init_alpha: float, optional
            initial scalar value for the attention transformation `x + alpha * Attention(x)`
            (default=0).
        normalize_after: bool, optional
            if True, apply post layer normalization; otherwise apply pre layer normalization
            (default=True).

        """
        super(SRUppAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.proj_features = proj_features
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = float(attn_dropout)
        self.rezero_init_alpha = float(rezero_init_alpha)
        self.linear1 = nn.Linear(in_features, proj_features, bias=False)
        self.linear2 = nn.Linear(proj_features, proj_features * 2, bias=False)
        self.linear3 = nn.Linear(proj_features, out_features, bias=False)
        # self.alpha = nn.Parameter(torch.Tensor([float(rezero_init_alpha)]))  # type: ignore
        self.normalize_after = normalize_after
        self.layer_norm: Optional[nn.Module] = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(proj_features)

        if proj_features % num_heads != 0:
            raise ValueError("proj_features ({}) must be divisible by num_heads ({})".format(
                proj_features, num_heads
            ))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        self.linear2.weight.data[self.proj_features:].mul_(0.0)
        # self.alpha.data[:] = self.rezero_init_alpha
        if self.linear1.bias is not None:
            self.linear1.bias.data.zero_()
        if self.linear2.bias is not None:
            self.linear2.bias.data.zero_()
        if self.linear3.bias is not None:
            self.linear3.bias.data.zero_()
        if self.dropout.p > 0:
            self.linear3.weight.data.mul_((1 - self.dropout.p)**0.5)

    def forward(self,
                input: Tensor,
                mask_pad: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                memory: Optional[Tensor] = None,
                memory_mask_pad: Optional[Tensor] = None) -> Tensor:
        """The forward method of SRU++ attention.
        """

        src_len = tgt_len = input.size(0)
        bsz = input.size(1)
        in_dim = input.size(2)
        proj_dim = self.proj_features
        num_heads = self.num_heads
        head_dim = proj_dim // num_heads
        scaling = float(head_dim) ** -0.5

        # concat memory and input as the key-value block when provided
        if memory is not None:
            if memory.dim() != 3 or list(memory.size()[-2:]) != [bsz, in_dim]:
                raise ValueError("memory has size {} but expect {}.".format(
                    list(memory.size()),
                    ['*', bsz, in_dim]
                ))
            mem_len = memory.size(0)
            src_len = memory.size(0) + input.size(0)
            input_ = torch.cat([memory, input], dim=0)
            z = self.linear1(input_)
            residual = z[memory.size(0):]
            layer_norm = self.layer_norm
            if layer_norm is not None:
                if not self.normalize_after:
                    z = layer_norm(z)
            q = z[memory.size(0):]
        else:
            mem_len = 0
            z = residual = self.linear1(input)
            layer_norm = self.layer_norm
            if layer_norm is not None:
                if not self.normalize_after:
                    z = layer_norm(z)
            q = z

        # query, key, value
        k, v = self.linear2(z).chunk(2, dim=-1)
        q = q.contiguous().view(tgt_len, -1, head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, -1, head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, -1, head_dim).transpose(0, 1)

        # (bsz * num_heads, tgt_len, src_len)
        q = q * scaling
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if list(attn_mask.size()) != [tgt_len, src_len]:
                raise ValueError("attn_mask has size {} but expect {}.".format(
                    list(attn_mask.size()),
                    [tgt_len, src_len]
                ))
            attn_output_weights += attn_mask.unsqueeze(0)

        if mask_pad is not None or memory_mask_pad is not None:
            if mask_pad is None:
                mask_pad = input.new_zeros(tgt_len, bsz, dtype=torch.bool)
            if mem_len > 0:
                if memory_mask_pad is None:
                    memory_mask_pad = input.new_zeros(mem_len, bsz, dtype=torch.bool)
                mask_pad = torch.cat([memory_mask_pad, mask_pad], dim=0)
            if list(mask_pad.size()) != [src_len, bsz]:
                raise ValueError("mask_pad has size {} but expect {}.".format(
                    list(mask_pad.size()),
                    [src_len, bsz]
                ))
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                mask_pad.transpose(0, 1).unsqueeze(1).unsqueeze(2),  # (bsz, 1, 1, src_len)
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.attn_dropout,
                                        training=self.training)

        # (bsz * num_heads, tgt_len, src_len) x (bsz * num_heads, src_len, head_dim)
        #     ---->  (bsz * num_heads, tgt_len, head_dim)
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, proj_dim)

        # attn_output = attn_output * self.alpha + residual
        attn_output = attn_output + residual
        layer_norm = self.layer_norm
        if layer_norm is not None:
            if self.normalize_after:
                attn_output = layer_norm(attn_output)

        # (tgt_len, bsz, out_dim)
        attn_output = self.linear3(self.dropout(attn_output))
        return attn_output


class SRUppCell(SRUCell):
    """
    A single layer of SRU++, inherited from SRUCell module
    """

    def forward(self,
                input: Tensor,
                c0: Optional[Tensor] = None,
                mask_pad: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                memory: Optional[Tensor] = None,
                memory_mask_pad: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """The forward method of the SRU++ layer.
        """

        if input.dim() != 2 and input.dim() != 3:
            raise ValueError("Input must be 2 or 3 dimensional")

        batch_size = input.size(-2)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.output_size, dtype=input.dtype,
                             device=input.device)

        # apply layer norm before activation (i.e. before SRU computation)
        residual = input
        layer_norm = self.layer_norm
        if layer_norm is not None:
            if not self.normalize_after:
                input = layer_norm(input)

        # apply dropout for multiplication
        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((batch_size, input.size(-1)), self.rnn_dropout)
            input = input * mask.expand_as(input)

        # get the scaling constant; scale_x is a scalar
        scale_val: Optional[Tensor] = None
        scale_val = self.scale_x if self.rescale else None

        # get dropout mask
        mask_c: Optional[Tensor] = None
        if self.training and (self.dropout > 0):
            mask_c = self.get_dropout_mask_((batch_size, self.output_size),
                                            self.dropout)

        # compute U
        #   U is (length, batch_size, output_size * num_matrices)
        transform_module = self.transform_module
        U = transform_module(input, mask_pad=mask_pad,
                             attn_mask=attn_mask,
                             memory=memory,
                             memory_mask_pad=memory_mask_pad)
        V = self.weight_c

        # apply elementwise recurrence to get hidden states h and c
        h, c = self.apply_recurrence(U, V,
                                     residual, c0,
                                     scale_val,
                                     mask_c,
                                     mask_pad)

        layer_norm = self.layer_norm
        if layer_norm is not None:
            if self.normalize_after:
                h = layer_norm(h)
        return h, c


class SRUpp(nn.Module):
    """
    Implementation of SRU++ module.
    """

    __constants__ = ['input_size', 'hidden_size', 'proj_size', 'output_size',
                     'num_layers', 'num_heads', 'dropout', 'bidirectional',
                     'use_layer_norm', 'num_directions', 'nn_rnn_compatible_return',
                     'input_to_hidden', 'rnn_lst', 'normalization_type']

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 proj_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 num_heads: int = 1,
                 bidirectional: bool = False,
                 layer_norm: bool = False,
                 normalize_after: bool = False,
                 attn_layer_norm: bool = True,
                 highway_bias: float = -2.0,
                 attention_every_n_layers: int = 1,
                 attention_last_n_layers: int = -1,
                 rescale: bool = False,
                 nn_rnn_compatible_return: bool = False,
                 proj_input_to_hidden_first: bool = False,
                 weight_c_init: float = 1.0):
        """Initialize the SRU++ module.

        Parameters
        ----------
        input_size: int
            the number of input features.
        hidden_size: int
            the number of features in the hidden state *for each direction*.
        proj_size: int
            the number of features used for attention.
        num_layers: int, optional
            the number of stacked SRU++ layers (default=2).
        dropout: float, optional
            dropout probability applied between sub-layers (default=0.0).
        attn_dropout: float, optional
            dropout probability applied on attention map (default=0.0).
        num_heads: int, optional
            number of attention heads (default=1).
        bidirectional: bool, optional
            if True, use bidirectional SRU++ (default=False).
        layer_norm: bool, optional
            whether to apply layer normalization to each SRU++ layer (default=False).
        highway_bias: float, optional
            the initial value of the bias used in the highway (sigmoid) gate (default=-1.0).
        attention_every_n_layers: int, optional
            only introduce attention every few layers of SRU++. by default, every SRU++ layer has
            attention (default=1).
        rescale: bool, optional
            whether to apply a constant rescaling multiplier for the residual term (default=False).
        proj_input_to_hidden_first: bool, optional
            if True, apply an nn.Linear module to the input of this module when input_size !=
            hidden_size (default=False).
        weight_c_init: float, optional
            size of uniform initiatialization of weight_c
            (default=1.0)

        """
        if attention_every_n_layers != 1 and attention_last_n_layers != -1:
            raise ValueError("Cannot set both attention_every_n_layers and "
                             "attention_last_n_layers in SRU++ module.")
        super(SRUpp, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.rnn_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.use_layer_norm = layer_norm
        self.num_directions = 2 if bidirectional else 1
        self.nn_rnn_compatible_return = nn_rnn_compatible_return
        self.input_to_hidden: Optional[nn.Module] = None
        if proj_input_to_hidden_first and input_size != self.output_size:
            first_layer_input_size = self.output_size
            self.input_to_hidden = nn.Linear(input_size, self.output_size, bias=False)
            nn.init.xavier_uniform_(self.input_to_hidden.weight)
        else:
            first_layer_input_size = input_size

        # attention configuration
        if attention_last_n_layers != -1:
            use_attention = lambda ind: num_layers - ind <= attention_last_n_layers  # noqa
        else:
            use_attention = lambda ind: (ind + 1) % attention_every_n_layers == 0  # noqa

        for i in range(num_layers):
            # create the i-th SRU layer
            in_features = first_layer_input_size if i == 0 else self.output_size
            proj_features = proj_size
            out_features = self.output_size * (3 if in_features == self.output_size else 4)
            custom_m: Optional[nn.Module] = None
            if use_attention(i):
                custom_m = SRUppAttention(
                    in_features,
                    out_features,
                    proj_features,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    num_heads=num_heads,
                    layer_norm=attn_layer_norm,
                )
            else:
                custom_m = SRUppProjectedLinear(
                    in_features,
                    out_features,
                    proj_features,
                    dropout=dropout,
                    layer_norm=attn_layer_norm,
                )
            layer = SRUppCell(
                in_features,
                self.hidden_size,
                dropout=dropout if i + 1 != num_layers else 0,
                bidirectional=bidirectional,
                layer_norm=layer_norm,
                normalize_after=normalize_after,
                highway_bias=highway_bias,
                rescale=rescale,
                transform_module=custom_m,
                weight_c_init=weight_c_init,
            )
            self.rnn_lst.append(layer)

    def __getitem__(self, n: int) -> SRUppCell:
        """
        returns n'th layer SRUppCell
        """
        return self.rnn_lst[n]

    def forward(self,
                input: Tensor,
                c0: Optional[Tensor] = None,
                mask_pad: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                memory: Optional[List[Optional[Tensor]]] = None,
                memory_mask_pad: Optional[Tensor] = None,
                ) -> Tuple[Tensor, Tensor, Dict[str, List[Tensor]]]:
        """
        The forward method of SRUpp module.

        Parameters
        ----------
        input: Tensor
            the input feature. shape: (length, batch_size, input_size)
        c0: Tensor, optional
            the initial internal hidden state. shape: (num_layers,
            batch_size, output_size) where
            output_size = hidden_size * num_direction
        mask_pad: Tensor, optional
            the mask where a non-zero value indicates if an input token
            is pad token that should be ignored in forward and backward
            computation. shape: (length, batch_size)
        attn_mask: Tensor, optional
            the additive attention mask. shape: (input_length, context_length)
            the mask is a float tensor that will be directly added to the
            attention weights before softmax normalization.
            input_length is the length of the input tensor, and context length
            is the total length of context states that each input can attend
            to. the context_length is equal to the sum of input_length and the
            lengths of extra memory states given by `memory`.
        memory: a list of optional tensors, optional
            a list of memory tensors as additional inputs for the attention
            to attend to. the size of the list is equal to the number of layers
            of SRUpp module. memory[i] is the memory tensor for the (i+1)-th
            layer and its second dimension (batch size) and third dimension
            (hidden size) must be compatible with the input tensor to the
            (i+1)-th layer.
        memory_mask_pad: tensor, optional
            the mask tensor indicate if a position in the memory tensors is
            an invalid / pad token that should be ignored in attention.
            shape: (memory_length, batch_size)

        Returns
        ----------
        h: Tensor
            the output hidden state. shape: (length, batch_size,
            output_size) where
            output_size = hidden_size * num_direction
        c: Tensor
            the last internal hidden state. shape: (num_layers,
            batch_size, output_size), or (num_layers * num_directions,
            batch_size, hidden_size) if `nn_rnn_compatible_return` is
            set `True`
        memory_bank: Dict[str, List[Tensor]]
            a dictionary that stores various internal states indexed
            by state names. each value is a list of tensors in which
            the i-th element is the state tensor of the (i+1)-th layer.
            these internal states can be reused for attention for the
            next forward call during training and decoding.

        """
        # unpack packed, if input is packed. packing and then unpacking will be slower than not
        # packing at all, but makes SRU++ usage compatible with nn.RNN usage
        orig_input = input
        if isinstance(orig_input, PackedSequence):
            input, lengths = nn.utils.rnn.pad_packed_sequence(input)
            max_length = lengths.max().item()
            mask_pad = torch.ByteTensor([[0] * length + [1] * (max_length - length)
                                        for length in lengths.tolist()])
            mask_pad = mask_pad.to(input.device).transpose(0, 1).contiguous()

        # The dimensions of `input` should be: `(sequence_length, batch_size, input_size)`.
        if input.dim() != 3:
            raise ValueError("There must be 3 dimensions for (length, batch_size, input_size)")

        length = input.size(0)
        bsz = input.size(1)
        input_size = input.size(2)
        num_layers = self.num_layers
        output_size = self.output_size

        if input_size != self.input_size:
            raise ValueError("Input has size (*, *, {}) but expect a last dimension of {}".format(
                input_size, self.input_size
            ))

        if c0 is None:
            zeros = torch.zeros(bsz, output_size, dtype=input.dtype, device=input.device)
            c0_ = [zeros for i in range(num_layers)]
        else:
            if list(c0.size()) != [num_layers, bsz, output_size]:
                raise ValueError("c0 has size {} but expect {}.".format(
                    list(c0.size()),
                    [num_layers, bsz, output_size]
                ))
            c0_ = [x.squeeze(0) for x in c0.chunk(self.num_layers, 0)]

        if mask_pad is not None and list(mask_pad.size()) != [length, bsz]:
            raise ValueError("mask_pad has size {} but expect {}.".format(
                list(mask_pad.size()),
                [length, bsz]
            ))

        if memory is not None and not isinstance(memory, list):
            raise ValueError("memory has type {} but expect List[Tensor].".format(
                type(memory)
            ))

        if memory is not None and len(memory) != num_layers:
            raise ValueError("memory has size {} but expect {}.".format(
                len(memory),
                num_layers
            ))

        if self.input_to_hidden is None:
            x = input
        else:
            x = self.input_to_hidden(input)
        prev_inputs = []
        lstc = []
        i = 0
        x = x.contiguous()
        for rnn in self.rnn_lst:
            prev_inputs.append(x)
            memory_i = memory[i] if memory is not None else None
            h, c = rnn(x, c0_[i],
                       mask_pad=mask_pad,
                       attn_mask=attn_mask,
                       memory=memory_i,
                       memory_mask_pad=memory_mask_pad)
            x = h
            lstc.append(c)
            i += 1

        lstc_stack = torch.stack(lstc)
        if self.nn_rnn_compatible_return:
            lstc_stack = lstc_stack.view(num_layers, bsz, self.num_directions, self.hidden_size)
            lstc_stack = lstc_stack.transpose(1, 2).contiguous()
            lstc_stack = lstc_stack.view(num_layers * self.num_directions, bsz, self.hidden_size)

        if isinstance(orig_input, PackedSequence):
            h = nn.utils.rnn.pack_padded_sequence(h, lengths, enforce_sorted=False)

        return (h, lstc_stack, {'saved_inputs': prev_inputs})

    def reset_parameters(self):
        for rnn in self.rnn_lst:
            rnn.reset_parameters()
        if self.input_to_hidden is not None:
            nn.init.xavier_uniform_(self.input_to_hidden.weight)
