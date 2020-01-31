import os
import torch
from torch.autograd import Function

from torch.utils.cpp_extension import load
sources = [
    os.path.join(os.path.dirname(__file__), "sru_cuda_impl.cpp"),
    os.path.join(os.path.dirname(__file__), "sru_cuda_kernel.cu"),
]
sru_cuda_lib = load(
    name="sru_cuda_impl",
    sources=sources,
    extra_cflags=['-O3'],
    verbose=False
)

empty_btensor = torch.ByteTensor()
empty_ftensor = torch.FloatTensor()

class SRU_Compute_GPU(Function):

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
        ctx.scale_x = scale_x
         # ensure mask_pad is a byte tensor
        mask_pad = mask_pad.byte().contiguous() if mask_pad is not None else None
        ctx.mask_pad = mask_pad

        bidir = 2 if bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = d_out
        if mask_pad is not None:
            assert mask_pad.size(0) == length
            assert mask_pad.size(1) == batch
        k = u.size(-1) // d
        k_ = k // 2 if bidirectional else k
        skip_type = 0 if not has_skip_term else (1 if k_ == 3 else 2)
        ncols = batch * d * bidir

        is_custom = len(weight_c.size()) > 1

        size = (length, batch, d * bidir) if x.dim() == 3 else (batch, d * bidir)
        c = x.new_zeros(*size)
        h = x.new_zeros(*size)

        if skip_type > 0 and k_ == 3:
            x_ = x.contiguous() * scale_x if scale_x is not None else x.contiguous()
        else:
            x_ = empty_ftensor

        forward_func = sru_cuda_lib.sru_bi_forward if bidirectional else \
                sru_cuda_lib.sru_forward
        forward_func(
            h,
            c,
            u.contiguous(),
            x_,
            weight_c.contiguous(),
            bias,
            init.contiguous(),
            mask_c if mask_c is not None else empty_ftensor,
            mask_pad.contiguous() if mask_pad is not None else empty_btensor,
            length,
            batch,
            d,
            k_,
            activation_type,
            skip_type,
            is_custom
        )

        ctx.save_for_backward(u, x, weight_c, bias, init, mask_c)
        ctx.intermediate = c
        if x.dim() == 2:
            last_hidden = c
        elif bidirectional:
            last_hidden = torch.cat((c[-1, :, :d], c[0, :, d:]), dim=1)
        else:
            last_hidden = c[-1]
        return h, last_hidden

    @staticmethod
    def backward(ctx, grad_h, grad_last):
        bidir = 2 if ctx.bidirectional else 1
        u, x, weight_c, bias, init, mask_c = ctx.saved_tensors
        c = ctx.intermediate
        scale_x = ctx.scale_x
        mask_pad = ctx.mask_pad
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = ctx.d_out
        k = u.size(-1) // d
        k_ = k // 2 if ctx.bidirectional else k
        skip_type = 0 if not ctx.has_skip_term else (1 if k_ == 3 else 2)
        ncols = batch * d * bidir

        is_custom = len(weight_c.size()) > 1

        grad_u = u.new_zeros(*u.size())
        grad_init = x.new_zeros(batch, d * bidir)
        grad_x = x.new_zeros(*x.size()) if skip_type > 0 and k_ == 3 else None
        grad_bias = x.new_zeros(2, batch, bidir * d)
        if not is_custom:
            grad_wc = x.new_zeros(2, batch, bidir * d)
        else:
            grad_wc = weight_c.new_zeros(*weight_c.size())


        if skip_type > 0 and k_ == 3:
            x_ = x.contiguous() * scale_x if scale_x is not None else x.contiguous()
        else:
            x_ = empty_ftensor

        backward_func = sru_cuda_lib.sru_bi_backward if ctx.bidirectional else \
                sru_cuda_lib.sru_backward
        backward_func(
            grad_u,
            grad_x if skip_type > 0 and k_ == 3 else empty_ftensor,
            grad_wc,
            grad_bias,
            grad_init,
            u.contiguous(),
            x_,
            weight_c.contiguous(),
            bias,
            init.contiguous(),
            mask_c if mask_c is not None else empty_ftensor,
            mask_pad.contiguous() if mask_pad is not None else empty_btensor,
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

        if skip_type > 0 and k_ == 3 and scale_x is not None:
            grad_x.mul_(scale_x)
        if not is_custom:
            grad_wc = grad_wc.sum(1).view(-1)
        return grad_u, grad_x, grad_wc, grad_bias.sum(1).view(-1), grad_init, \
               None, None, None, None, None, None, None

