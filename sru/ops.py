
from typing import List, Optional, Tuple
import os
import warnings

import torch
from torch import Tensor
from torch.utils.cpp_extension import load
from .cuda_functional import elementwise_recurrence_forward

# JIT compilation of elementwise fwd operator (CPU version)
cpu_source = os.path.join(os.path.dirname(__file__), "csrc", "sru_cpu_impl.cpp")
load(
    name="sru_cpu",
    sources=[cpu_source],
    extra_cflags=['-O3'],
    is_python_module=False,
    verbose=False
)


@torch.jit.script
def elementwise_recurrence_inference(U: Tensor,
                                     x: Tensor,
                                     weight_c: Tensor,
                                     bias: Tensor,
                                     c_init: Tensor,
                                     activation_type: int,
                                     hidden_size: int,
                                     bidirectional: bool,
                                     has_skip_term: bool,
                                     scale_x: Optional[Tensor] = None,
                                     dropout_mask_c: Optional[Tensor] = None,
                                     mask_pad: Optional[Tensor] = None) -> List[Tensor]:
    """Torchscripted elementwise forward operation of SRU for inference.

    """
    assert dropout_mask_c is None, "Dropout mask cannot be set during inference"
    bidir = 2 if bidirectional else 1
    length = x.size(0) if x.dim() == 3 else 1
    batch = x.size(-2)
    k = U.size(-1) // hidden_size // bidir
    is_custom = weight_c.dim() > 1
    mask_pad = None if mask_pad is None else mask_pad.to(dtype=torch.bool).contiguous()
    if U.is_cuda:
        h, last_hidden, c = elementwise_recurrence_forward(
            U,
            x,
            weight_c,
            bias,
            c_init,
            activation_type,
            hidden_size,
            bidirectional,
            has_skip_term,
            scale_x,
            dropout_mask_c,
            mask_pad
        )
        return h, last_hidden
    elif not bidirectional:
        return torch.ops.sru_cpu.cpu_forward(
            U.contiguous(),
            x.contiguous(),
            weight_c.contiguous(),
            bias.contiguous(),
            c_init.contiguous(),
            mask_pad,
            length,
            batch,
            hidden_size,
            k,
            activation_type,
            has_skip_term,
            scale_x.item() if scale_x is not None else 1.0,
            is_custom
        )
    else:
        return torch.ops.sru_cpu.cpu_bi_forward(
            U.contiguous(),
            x.contiguous(),
            weight_c.contiguous(),
            bias.contiguous(),
            c_init.contiguous(),
            mask_pad,
            length,
            batch,
            hidden_size,
            k,
            activation_type,
            has_skip_term,
            scale_x.item() if scale_x is not None else 1.0,
            is_custom
        )


@torch.jit.unused
def elementwise_recurrence_gpu(U: Tensor,
                               x: Tensor,
                               weight_c: Tensor,
                               bias: Tensor,
                               c_init: Tensor,
                               activation_type: int,
                               hidden_size: int,
                               bidirectional: bool,
                               has_skip_term: bool,
                               scale_x: Optional[Tensor] = None,
                               dropout_mask_c: Optional[Tensor] = None,
                               mask_pad: Optional[Tensor] = None,
                               amp_recurrence_fp16: bool = False) -> List[Tensor]:
    """Elementwise forward operation of SRU on GPU.

    """
    from .cuda_functional import ElementwiseRecurrence

    if amp_recurrence_fp16 and U.dtype == torch.float16:
        cast = torch.Tensor.half
    else:
        cast = torch.Tensor.float

    U = cast(U)
    x = cast(x)
    weight_c = cast(weight_c)
    bias = cast(bias)
    c_init = cast(c_init)
    scale_x = cast(scale_x) if scale_x is not None else scale_x
    dropout_mask_c = cast(dropout_mask_c) if dropout_mask_c is not None else dropout_mask_c

    return ElementwiseRecurrence.apply(
        U,
        x,
        weight_c,
        bias,
        c_init,
        activation_type,
        hidden_size,
        bidirectional,
        has_skip_term,
        scale_x,
        dropout_mask_c,
        mask_pad
    )


@torch.jit.unused
def elementwise_recurrence_naive(U: Tensor,
                                 x: Tensor,
                                 weight_c: Tensor,
                                 bias: Tensor,
                                 c_init: Tensor,
                                 activation_type: int,
                                 hidden_size: int,
                                 bidirectional: bool,
                                 has_skip_term: bool,
                                 scale_x: Optional[Tensor] = None,
                                 dropout_mask_c: Optional[Tensor] = None,
                                 mask_pad: Optional[Tensor] = None) -> List[Tensor]:
    """Elementwise forward operation of SRU in pure Python.

    """
    if torch.is_grad_enabled():
        warnings.warn("Running SRU on CPU with grad_enabled=True. Are you sure?")
    else:
        return elementwise_recurrence_inference(U, x, weight_c, bias, c_init,
                                                activation_type, hidden_size,
                                                bidirectional, has_skip_term,
                                                scale_x, dropout_mask_c, mask_pad)

    bidir = 2 if bidirectional else 1
    length = x.size(0) if x.dim() == 3 else 1
    batch = x.size(-2)
    k = U.size(-1) // hidden_size // bidir
    d = hidden_size
    is_custom = weight_c.dim() > 1

    U = U.contiguous().view(length, batch, bidir, d, k)

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
        x_prime = U[..., 3]

    if c_init is None:
        c_init = x.new_zeros(size=(batch, bidir, d))
    else:
        c_init = c_init.view(batch, bidir, d)

    mask_pad = mask_pad.view(length, batch, 1).float() if mask_pad is not None else None
    mask_c = dropout_mask_c.view(batch, bidir, d) if dropout_mask_c is not None else None

    h = x.new_zeros(length, batch, bidir, d)
    c_final = []
    for di in range(bidir):
        time_seq = range(length) if di == 0 else range(length - 1, -1, -1)
        mask_c_ = 1 if mask_c is None else mask_c[:, di, :]
        c_prev = c_init[:, di, :]
        fb, rb = forget_bias[di], reset_bias[di]
        if is_custom:
            fw = forget_wc[:, :, di, :].chunk(length)
            rw = reset_wc[:, :, di, :].chunk(length)
        else:
            fw = forget_wc[di].expand(batch, d)  # type: ignore
            rw = reset_wc[di].expand(batch, d)  # type: ignore
        u0 = U[:, :, di, :, 0].chunk(length)
        u1 = (U[:, :, di, :, 1] + fb).chunk(length)
        u2 = (U[:, :, di, :, 2] + rb).chunk(length)
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
            if mask_pad is not None:
                c_t = c_t * (1-mask_pad[t]) + c_prev * mask_pad[t]
            c_prev = c_t

            if activation_type == 0:
                g_c_t = c_t
            elif activation_type == 1:
                g_c_t = c_t.tanh()
            else:
                raise ValueError('Activation type must be 0 or 1, not {}'.format(activation_type))

            if x_prime is not None:
                h_t = xp[t] + (g_c_t - xp[t]) * mask_c_ * reset_t
            else:
                h_t = g_c_t * mask_c_ * reset_t
            if mask_pad is not None:
                h_t = h_t * (1-mask_pad[t])
            h[t, :, di, :] = h_t
        c_final.append(c_t.view(batch, d))

    return h.view(length, batch, -1), torch.stack(c_final, dim=1).view(batch, -1)  # type: ignore

