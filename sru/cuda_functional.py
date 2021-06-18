from typing import Optional, Tuple
import os
import warnings

import torch
from torch import Tensor
from torch.autograd import Function
from torch.utils.cpp_extension import load

try:
    sources = [
        os.path.join(os.path.dirname(__file__), "csrc", "sru_cuda_impl.cpp"),
        os.path.join(os.path.dirname(__file__), "csrc", "sru_cuda_kernel.cu"),
    ]
    load(
        name="sru_cuda",
        sources=sources,
        extra_cflags=['-O3'],
        is_python_module=False,
        verbose=False
    )
except Exception as e:
    warnings.warn("Just-in-time loading and compiling the CUDA kernels of SRU was unsuccessful. "
                  "Got the following error:\n" + str(e))
    sources_dummy = [
        os.path.join(os.path.dirname(__file__), "csrc", "sru_cuda_impl_dummy.cpp"),
    ]
    load(
        name="sru_cuda",
        sources=sources_dummy,
        extra_cflags=['-O3'],
        is_python_module=False,
        verbose=False
    )


@torch.jit.script
def elementwise_recurrence_forward(
    u: Tensor,
    x: Tensor,
    weight_c: Tensor,
    bias: Tensor,
    init: Tensor,
    activation_type: int,
    d_out: int,
    bidirectional: bool,
    has_skip_term: bool,
    scale_x: Optional[Tensor] = None,
    mask_c: Optional[Tensor] = None,
    mask_pad: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    """
    batch: int = x.size(-2)
    d: int = d_out
    length: int = x.size(0) if x.dim() == 3 else 1
    k: int = u.size(-1) // d
    k_: int = k // 2 if bidirectional else k
    is_custom: int = 1 if weight_c.dim() > 1 else 0
    skip_type: int = 0
    if has_skip_term:
        if k_ == 3:
            skip_type = 1
        else:
            skip_type = 2

    mask_pad_: Optional[Tensor] = None
    if mask_pad is not None:
        mask_pad_ = mask_pad.contiguous()
        assert mask_pad_.size(0) == length
        assert mask_pad_.size(1) == batch

    x_: Optional[Tensor] = None
    if skip_type > 0 and k_ == 3:
        if scale_x is not None:
            x_ = x.contiguous() * scale_x
        else:
            x_ = x.contiguous()

    # call faster / simple version if possible
    is_simple_version = ((k_ == 3) and has_skip_term and (not is_custom)
                         and (activation_type == 0))
    if is_simple_version:
        if bidirectional:
            h, c = torch.ops.sru_cuda.sru_bi_forward_simple(
                u.contiguous(),
                x_,
                weight_c.contiguous(),
                bias,
                init.contiguous(),
                mask_c,
                mask_pad_,
                length,
                batch,
                d
            )
        else:
            h, c = torch.ops.sru_cuda.sru_forward_simple(
                u.contiguous(),
                x_,
                weight_c.contiguous(),
                bias,
                init.contiguous(),
                mask_c,
                mask_pad_,
                length,
                batch,
                d
            )
    else:
        if bidirectional:
            h, c = torch.ops.sru_cuda.sru_bi_forward(
                u.contiguous(),
                x_,
                weight_c.contiguous(),
                bias,
                init.contiguous(),
                mask_c,
                mask_pad_,
                length,
                batch,
                d,
                k_,
                activation_type,
                skip_type,
                is_custom
            )
        else:
            h, c = torch.ops.sru_cuda.sru_forward(
                u.contiguous(),
                x_,
                weight_c.contiguous(),
                bias,
                init.contiguous(),
                mask_c,
                mask_pad_,
                length,
                batch,
                d,
                k_,
                activation_type,
                skip_type,
                is_custom
            )

    if x.dim() == 2:
        h, c = h.squeeze(0), c.squeeze(0)
        last_hidden = c
    elif bidirectional:
        last_hidden = torch.cat((c[-1, :, :d], c[0, :, d:]), dim=1)
    else:
        last_hidden = c[-1]
    return h, last_hidden, c


class ElementwiseRecurrence(Function):

    @staticmethod
    def forward(ctx, u, x, weight_c, bias,
                init,
                activation_type,
                d_out,
                bidirectional,
                has_skip_term,
                scale_x,
                mask_c=None,
                mask_pad=None):

        ctx.activation_type = activation_type
        ctx.d_out = d_out
        ctx.bidirectional = bidirectional
        ctx.has_skip_term = has_skip_term

        # ensure mask_pad is a bool tensor
        mask_pad = mask_pad.bool().contiguous() if mask_pad is not None else None
        h, last_hidden, c = elementwise_recurrence_forward(
            u,
            x,
            weight_c,
            bias,
            init,
            activation_type,
            d_out,
            bidirectional,
            has_skip_term,
            scale_x,
            mask_c,
            mask_pad
        )
        ctx.save_for_backward(u, x, weight_c, bias, init, mask_c, c, mask_pad, scale_x)
        return h, last_hidden

    @staticmethod
    def backward(ctx, grad_h, grad_last):
        u, x, weight_c, bias, init, mask_c, c, mask_pad, scale_x = ctx.saved_tensors
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = ctx.d_out
        k = u.size(-1) // d
        k_ = k // 2 if ctx.bidirectional else k
        skip_type = 0 if not ctx.has_skip_term else (1 if k_ == 3 else 2)
        is_custom = len(weight_c.size()) > 1

        if skip_type > 0 and k_ == 3:
            x_ = x.contiguous() * scale_x if scale_x is not None else x.contiguous()
        else:
            x_ = None

        # call faster / simple version if possible
        is_simple_version = ((k_ == 3) and ctx.has_skip_term and (not is_custom)
                             and (ctx.activation_type == 0))
        if is_simple_version:
            backward_func = torch.ops.sru_cuda.sru_bi_backward_simple if ctx.bidirectional else \
                torch.ops.sru_cuda.sru_backward_simple
            grad_u, grad_x, grad_wc, grad_bias, grad_init = backward_func(
                u.contiguous(),
                x_,
                weight_c.contiguous(),
                bias,
                init.contiguous(),
                mask_c,
                mask_pad.contiguous() if mask_pad is not None else None,
                c,
                grad_h.contiguous(),
                grad_last.contiguous(),
                length,
                batch,
                d,
            )
        else:
            backward_func = torch.ops.sru_cuda.sru_bi_backward if ctx.bidirectional else \
                torch.ops.sru_cuda.sru_backward
            grad_u, grad_x, grad_wc, grad_bias, grad_init = backward_func(
                u.contiguous(),
                x_,
                weight_c.contiguous(),
                bias,
                init.contiguous(),
                mask_c,
                mask_pad.contiguous() if mask_pad is not None else None,
                c,
                grad_h.contiguous(),
                grad_last.contiguous(),
                length,
                batch,
                d,
                k_,
                ctx.activation_type,
                skip_type,
                is_custom
            )

        if skip_type > 0 and k_ == 3:
            if scale_x is not None:
                grad_x.mul_(scale_x)
        else:
            grad_x = None
        if not is_custom:
            grad_wc = grad_wc.sum(1).view(-1)
        return grad_u, grad_x, grad_wc, grad_bias.sum(1).view(-1), grad_init, \
            None, None, None, None, None, None, None
