
from typing import Tuple, List, Optional, Union
import os
import time
import sys

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.cpp_extension import load
import sru

# JIT compilation of elementwise fwd operator (CPU version)
cpu_source = os.path.join(os.path.dirname(__file__), os.path.join("csrc", "sru_cpu_impl.cpp"))
load(
    name="sru_cpu",
    sources=[cpu_source],
    extra_cflags=['-O3'],
    is_python_module=False,
    verbose=False
)

@torch.jit.script
def elementwise_recurrence_cpu(U: Tensor,
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
    """Elementwise forward operation of SRU on CPU.

    """
    assert dropout_mask_c is None, "Dropout mask cannot be set during inference"
    bidir = 2 if bidirectional else 1
    length = x.size(0) if x.dim() == 3 else 1
    batch = x.size(-2)
    k = U.size(-1) // hidden_size // bidir
    is_custom = weight_c.dim() > 1
    mask_pad_ = torch.empty(0) if mask_pad is None else mask_pad.float()
    if not bidirectional:
        return torch.ops.sru_cpu.cpu_forward(
            U.contiguous(),
            x.contiguous(),
            weight_c.contiguous(),
            bias.contiguous(),
            c_init.contiguous(),
            mask_pad_.contiguous(),
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
            mask_pad_.contiguous(),
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
                               mask_pad: Optional[Tensor] = None) -> List[Tensor]:
    """Elementwise forward operation of SRU on GPU.

    """
    from .cuda_functional import SRU_Compute_GPU
    SRU_Compute_GPU.apply(
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
            fw = forget_wc[di].expand(batch, d)
            rw = reset_wc[di].expand(batch, d)
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

    return h.view(length, batch, -1), torch.stack(c_final, dim=1).view(batch, -1)



# some test code
if __name__ == "__main__":
    print(elementwise_recurrence_cpu)
    print(type(elementwise_recurrence_cpu))
    print(elementwise_recurrence_cpu.graph)
    print(elementwise_recurrence_cpu.code)

    srucell = sru.SRUCell(256, 256, rescale=False)
    x = torch.randn(16, 4, 256)
    U = srucell.compute_U(x)
    c0 = torch.zeros(4, 256)
    h, c = srucell(x)
    for func in [elementwise_recurrence_cpu,
                 elementwise_recurrence_naive]:
        h2, c2 = func(U, x, srucell.weight_c, srucell.bias,
                      c0, 0, 256, False, True, None, None, None)
        print((h-h2).abs().max())
        print((c-c2).abs().max())

    with torch.no_grad():
        start_time = time.time()
        for i in range(1000):
            U = srucell.compute_U(x)
            ret = elementwise_recurrence_cpu(U, x,
                                       srucell.weight_c,
                                       srucell.bias,
                                       c0,
                                       0,
                                       256,
                                       False,
                                       True,
                                       None,
                                       None,
                                       None)
        print('torchscript op:', (time.time() - start_time))

    with torch.no_grad():
        start_time = time.time()
        for i in range(1000):
            ret = srucell(x)
        print('srucell:', (time.time() - start_time))

