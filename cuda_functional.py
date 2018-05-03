import torch
from pynvrtc.compiler import Program
from torch.autograd import Function
from collections import namedtuple
from cupy.cuda import function


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
    __global__ void sru_fwd(const float * __restrict__ u,
                            const float * __restrict__ x,
                            const float * __restrict__ weight_c,
                            const float * __restrict__ bias,
                            const float * __restrict__ init,
                            const float * __restrict__ mask_h,
                            const char * __restrict__ mask_pad,
                            const int len,
                            const int batch,
                            const int d,
                            const int k,
                            float * __restrict__ h,
                            float * __restrict__ c,
                            const int activation_type,
                            const int skip_type)
    {
        assert ((skip_type >= 0) || (skip_type <= 2));
        assert ((skip_type != 1) || (k == 3));
        assert ((skip_type != 2) || (k == 4));

        const int ncols = batch*d;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        const int ncols_u = ncols*k;
        const int ncols_x = (k == 3) ? ncols : ncols_u;

        const float wc1 = *(weight_c + (col%d));
        const float wc2 = *(weight_c + (col%d) + d);
        const float bias1 = *(bias + (col%d));
        const float bias2 = *(bias + (col%d) + d);
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float cur = *(init + col);
        const float *up = u + (col*k);
        const float *xp = (skip_type == 0) ? NULL : ((skip_type == 1) ? (x + col) : (up + 3));
        const char *pad_p = (mask_pad == NULL) ? NULL : (mask_pad + (col/d));
        float *cp = c + col;
        float *hp = h + col;

        for (int row = 0; row < len; ++row)
        {
            if ((pad_p == NULL) || !(*pad_p)) {
                float g1 = sigmoidf((*(up+1)) + wc1*cur + bias1);
                float g2 = sigmoidf((*(up+2)) + wc2*cur + bias2);
                cur = (cur-(*up))*g1 + (*up);
                float val = calc_activation(activation_type, cur);
                if (skip_type)
                    *hp = (val*mask-(*xp))*g2 + (*xp);
                else
                    *hp = val*mask*g2;
            }
            *cp = cur;  // useful for backward
            up += ncols_u;
            cp += ncols;
            hp += ncols;
            if (skip_type) xp += ncols_x;
            if (pad_p) pad_p += batch;
        }
    }

    __global__ void sru_bwd(const float * __restrict__ u,
                            const float * __restrict__ x,
                            const float * __restrict__ weight_c,
                            const float * __restrict__ bias,
                            const float * __restrict__ init,
                            const float * __restrict__ mask_h,
                            const char * __restrict__ mask_pad,
                            const float * __restrict__ c,
                            const float * __restrict__ grad_h,
                            const float * __restrict__ grad_last,
                            const int len,
                            const int batch,
                            const int d,
                            const int k,
                            float * __restrict__ grad_u,
                            float * __restrict__ grad_x,
                            float * __restrict__ grad_wc,
                            float * __restrict__ grad_bias,
                            float * __restrict__ grad_init,
                            const int activation_type,
                            const int skip_type)
    {
        assert ((skip_type >= 0) || (skip_type <= 2));
        assert ((skip_type != 1) || (k == 3));
        assert ((skip_type != 2) || (k == 4));

        const int ncols = batch*d;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        const int ncols_u = ncols*k;
        const int ncols_x = (k == 3) ? ncols : ncols_u;

        const float wc1 = *(weight_c + (col%d));
        const float wc2 = *(weight_c + (col%d) + d);
        const float bias1 = *(bias + (col%d));
        const float bias2 = *(bias + (col%d) + d);
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float gwc1 = 0;
        float gwc2 = 0;
        float gbias1 = 0;
        float gbias2 = 0;
        float cur = *(grad_last + col);

        const float *up = u + (col*k) + (len-1)*ncols_u;
        const float *xp = (skip_type == 0) ? NULL : (
            (skip_type == 1) ? (x + col + (len-1)*ncols) : (up + 3)
        );
        const float *cp = c + col + (len-1)*ncols;
        const float *ghp = grad_h + col + (len-1)*ncols;
        const char *pad_p = (mask_pad == NULL) ? NULL : (mask_pad + (col/d) + (len-1)*batch);
        float *gup = grad_u + (col*k) + (len-1)*ncols_u;
        float *gxp = (skip_type == 0) ? NULL : (
            (skip_type == 1) ? (grad_x + col + (len-1)*ncols) : (gup + 3)
        );

        for (int row = len-1; row >= 0; --row)
        {
            if ((pad_p == NULL) || !(*pad_p)) {
                const float prev_c_val = (row>0) ? (*(cp-ncols)) : (*(init+col));
                const float g1 = sigmoidf((*(up+1)) + wc1*prev_c_val + bias1);
                const float g2 = sigmoidf((*(up+2)) + wc2*prev_c_val + bias2);
                const float c_val = calc_activation(activation_type, *cp);
                const float x_val = (skip_type) ? (*xp) : 0;
                const float u_val = *up;
                const float gh_val = *ghp;

                // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
                // c = c'*g1 + u0*(1-g1) = (c'-u0)*g1 + g0

                // gradient with respect to x[t]
                if (skip_type)
                    *gxp = gh_val*(1-g2);

                // gradient with respect to values in the second gate g2
                float gg2 = gh_val*(c_val*mask-x_val)*(g2*(1-g2));
                *(gup+2) = gg2;
                gbias2 += gg2;
                gwc2 += gg2*prev_c_val;

                // gradient with respect to c[t]
                const float tmp = g2*calc_grad_activation(activation_type, c_val);
                const float gc = gh_val*mask*tmp + cur;

                // gradient with respect to current input u0=W*x[t]
                *gup = gc*(1-g1);

                // gradient with respect to values in the first gate g1
                float gg1 = gc*(prev_c_val-u_val)*(g1*(1-g1));
                *(gup+1) = gg1;
                gbias1 += gg1;
                gwc1 += gg1*prev_c_val;

                // gradient with respect to c[t-1]
                cur = gc*g1 + gg1*wc1 + gg2*wc2;
            }

            up -= ncols_u;
            cp -= ncols;
            gup -= ncols_u;
            ghp -= ncols;
            if (skip_type) {
                xp -= ncols_x;
                gxp -= ncols_x;
            }
            if (pad_p) pad_p -= batch;
        }

        *(grad_wc + col) = gwc1;
        *(grad_wc + col + ncols) = gwc2;
        *(grad_bias + col) = gbias1;
        *(grad_bias + col + ncols) = gbias2;
        *(grad_init +col) = cur;
    }

    __global__ void sru_bi_fwd(const float * __restrict__ u,
                            const float * __restrict__ x,
                            const float * __restrict__ weight_c,
                            const float * __restrict__ bias,
                            const float * __restrict__ init,
                            const float * __restrict__ mask_h,
                            const char * __restrict__ mask_pad,
                            const int len,
                            const int batch,
                            const int d,
                            const int k,
                            float * __restrict__ h,
                            float * __restrict__ c,
                            const int activation_type,
                            const int skip_type)
    {
        assert ((skip_type >= 0) || (skip_type <= 2));
        assert ((skip_type != 1) || (k == 3));
        assert ((skip_type != 2) || (k == 4));

        const int ncols = batch*d*2;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        const int ncols_u = ncols*k;
        const int ncols_x = (k == 3) ? ncols : ncols_u;
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float cur = *(init + col);
        const int d2 = d*2;
        const float wc1 = *(weight_c + (col%d2));
        const float wc2 = *(weight_c + (col%d2) + d2);
        const float bias1 = *(bias + (col%d2));
        const float bias2 = *(bias + (col%d2) + d2);

        const float *up = u + (col*k);
        const float *xp = (skip_type == 0) ? NULL : ((skip_type == 1) ? (x + col) : (up + 3));
        const char *pad_p = (mask_pad == NULL) ? NULL : (mask_pad + (col/d2));
        float *cp = c + col;
        float *hp = h + col;
        const bool flip = (col%d2) >= d;
        if (flip) {
            up += (len-1)*ncols_u;
            cp += (len-1)*ncols;
            hp += (len-1)*ncols;
            if (skip_type) xp += (len-1)*ncols_x;
            if (pad_p) pad_p += (len-1)*batch;
        }
        const int ncols_u_ = flip ? -ncols_u : ncols_u;
        const int ncols_x_ = flip ? -ncols_x : ncols_x;
        const int ncols_ = flip ? -ncols : ncols;
        const int batch_ = flip ? -batch : batch;

        for (int cnt = 0; cnt < len; ++cnt)
        {
            if ((pad_p == NULL) || !(*pad_p)) {
                float g1 = sigmoidf((*(up+1)) + wc1*cur + bias1);
                float g2 = sigmoidf((*(up+2)) + wc2*cur + bias2);
                cur = (cur-(*up))*g1 + (*up);
                float val = calc_activation(activation_type, cur);
                if (skip_type)
                    *hp = (val*mask-(*xp))*g2 + (*xp);
                else
                    *hp = val*mask*g2;
            }
            *cp = cur;  // useful for backward
            up += ncols_u_;
            cp += ncols_;
            hp += ncols_;
            if (skip_type) xp += ncols_x_;
            if (pad_p) pad_p += batch_;
        }
    }

    __global__ void sru_bi_bwd(const float * __restrict__ u,
                               const float * __restrict__ x,
                               const float * __restrict__ weight_c,
                               const float * __restrict__ bias,
                               const float * __restrict__ init,
                               const float * __restrict__ mask_h,
                               const char * __restrict__ mask_pad,
                               const float * __restrict__ c,
                               const float * __restrict__ grad_h,
                               const float * __restrict__ grad_last,
                               const int len,
                               const int batch,
                               const int d,
                               const int k,
                               float * __restrict__ grad_u,
                               float * __restrict__ grad_x,
                               float * __restrict__ grad_wc,
                               float * __restrict__ grad_bias,
                               float * __restrict__ grad_init,
                               const int activation_type,
                               const int skip_type)
    {
        assert ((skip_type >= 0) || (skip_type <= 2));
        assert ((skip_type != 1) || (k == 3));
        assert ((skip_type != 2) || (k == 4));

        int ncols = batch*d*2;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float gwc1 = 0;
        float gwc2 = 0;
        float gbias1 = 0;
        float gbias2 = 0;
        float cur = *(grad_last + col);
        const int d2 = d*2;
        const float wc1 = *(weight_c + (col%d2));
        const float wc2 = *(weight_c + (col%d2) + d2);
        const float bias1 = *(bias + (col%d2));
        const float bias2 = *(bias + (col%d2) + d2);

        const float *up = u + (col*k);
        const float *xp = (skip_type == 0) ? NULL : (
            (skip_type == 1) ? (x + col) : (up + 3)
        );
        const float *cp = c + col;
        const float *ghp = grad_h + col;
        const char *pad_p = (mask_pad == NULL) ? NULL : (mask_pad + (col/d2));
        float *gup = grad_u + (col*k);
        float *gxp = (skip_type == 0) ? NULL : (
            (skip_type == 1) ? (grad_x + col) : (gup + 3)
        );

        const bool flip = ((col%d2) >= d);
        if (!flip) {
            up += (len-1)*ncols_u;
            cp += (len-1)*ncols;
            ghp += (len-1)*ncols;
            gup += (len-1)*ncols_u;
            if (skip_type) {
                xp += (len-1)*ncols_x;
                gxp += (len-1)*ncols_x;
            }
            if (pad_p) pad_p += (len-1)*batch;
        }
        const int ncols_u_ = flip ? -ncols_u : ncols_u;
        const int ncols_x_ = flip ? -ncols_x : ncols_x;
        const int ncols_ = flip ? -ncols : ncols;
        const int batch_ = flip ? -batch : batch;

        for (int cnt = 0; cnt < len; ++cnt)
        {
            if ((pad_p == NULL) || !(*pad_p)) {
                const float prev_c_val = (cnt<len-1) ? (*(cp-ncols_)) : (*(init+col));
                const float g1 = sigmoidf((*(up+1)) + wc1*prev_c_val + bias1);
                const float g2 = sigmoidf((*(up+2)) + wc2*prev_c_val + bias2);
                const float c_val = calc_activation(activation_type, *cp);
                const float x_val = (skip_type) ? (*xp) : 0;
                const float u_val = *up;
                const float gh_val = *ghp;

                // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
                // c = c'*g1 + u0*(1-g1) = (c'-u0)*g1 + u0

                // gradient with respect to x[t]
                if (skip_type)
                    *gxp = gh_val*(1-g2);

                // gradient with respect to values in the second gate g2
                float gg2 = gh_val*(c_val*mask-x_val)*(g2*(1-g2));
                *(gup+2) = gg2;
                gbias2 += gg2;
                gwc2 += gg2*prev_c_val;

                // gradient with respect to c[t]
                const float tmp = g2*calc_grad_activation(activation_type, c_val);
                const float gc = gh_val*mask*tmp + cur;

                // gradient with respect to u[0]=W*x[t]
                *gup = gc*(1-g1);

                // gradient with respect to values in the first gate g1
                float gg1 = gc*(prev_c_val-u_val)*(g1*(1-g1));
                *(gup+1) = gg1;
                gbias1 += gg1;
                gwc1 += gg1*prev_c_val;

                // gradient with respect to c[t-1]
                cur = gc*g1 + gg1*wc1 + gg2*wc2;
            }

            up -= ncols_u_;
            cp -= ncols_;
            gup -= ncols_u_;
            ghp -= ncols_;
            if (skip_type) {
                xp -= ncols_x_;
                gxp -= ncols_x_;
            }
            if (pad_p) pad_p -= batch_;
        }

        *(grad_wc + col) = gwc1;
        *(grad_wc + col + ncols) = gwc2;
        *(grad_bias + col) = gbias1;
        *(grad_bias + col + ncols) = gbias2;
        *(grad_init +col) = cur;
    }
}
"""


class SRU_Compute_GPU(Function):

    _SRU_PROG = Program(SRU_CODE.encode('utf-8'), 'sru_prog.cu'.encode())
    _SRU_PTX = _SRU_PROG.compile()
    _DEVICE2FUNC = {}

    def __init__(self,
                 activation_type,
                 d_out,
                 bidirectional=False,
                 has_skip_term=True,
                 scale_x=1):

        super(SRU_Compute_GPU, self).__init__()
        self.activation_type = activation_type
        self.d_out = d_out
        self.bidirectional = bidirectional
        self.has_skip_term = has_skip_term
        self.scale_x = scale_x
        self.mask_pad = None

    def compile_functions(self):
        device = torch.cuda.current_device()
        mod = function.Module()
        mod.load(bytes(self._SRU_PTX.encode()))
        fwd_func = mod.get_function('sru_fwd')
        bwd_func = mod.get_function('sru_bwd')
        bifwd_func = mod.get_function('sru_bi_fwd')
        bibwd_func = mod.get_function('sru_bi_bwd')

        Stream = namedtuple('Stream', ['ptr'])
        current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)

        self._DEVICE2FUNC[device] = (
            current_stream, fwd_func,
            bifwd_func, bwd_func, bibwd_func
        )
        return current_stream, fwd_func, bifwd_func, bwd_func, bibwd_func

    def get_functions(self):
        res = self._DEVICE2FUNC.get(torch.cuda.current_device(), None)
        return res if res else self.compile_functions()

    def forward(self, u, x, weight_c, bias, init=None, mask_h=None):
        bidir = 2 if self.bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        mask_pad = self.mask_pad
        if mask_pad is not None:
            assert mask_pad.size(0) == length
            assert mask_pad.size(1) == batch
        k = u.size(-1) // d
        k_ = k // 2 if self.bidirectional else k
        skip_type = 0 if not self.has_skip_term else (1 if k_ == 3 else 2)
        ncols = batch*d*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1) // thread_per_block + 1

        init_ = x.new(ncols).zero_() if init is None else init
        size = (length, batch, d * bidir) if x.dim() == 3 else (batch, d * bidir)
        c = x.new(*size)
        h = x.new(*size)

        scale_x = self.scale_x
        if skip_type > 0 and k_ == 3:
            x_ptr = x.contiguous() * scale_x if scale_x != 1 else x.contiguous()
            x_ptr = x_ptr.data_ptr()
        else:
            x_ptr = 0

        stream, fwd_func, bifwd_func, _, _ = self.get_functions()
        FUNC = fwd_func if not self.bidirectional else bifwd_func
        FUNC(args=[
            u.contiguous().data_ptr(),
            x_ptr,
            weight_c.data_ptr(),
            bias.data_ptr(),
            init_.contiguous().data_ptr(),
            mask_h.data_ptr() if mask_h is not None else 0,
            mask_pad.data_ptr() if mask_pad is not None else 0,
            length,
            batch,
            d,
            k_,
            h.data_ptr(),
            c.data_ptr(),
            self.activation_type,
            skip_type],
            block=(thread_per_block, 1, 1),
            grid=(num_block, 1, 1),
            stream=stream
        )

        self.save_for_backward(u, x, weight_c, bias, init, mask_h)
        self.intermediate = c
        if x.dim() == 2:
            last_hidden = c
        elif self.bidirectional:
            last_hidden = torch.cat((c[-1, :, :d], c[0, :, d:]), dim=1)
        else:
            last_hidden = c[-1]
        return h, last_hidden

    def backward(self, grad_h, grad_last):
        bidir = 2 if self.bidirectional else 1
        u, x, weight_c, bias, init, mask_h = self.saved_tensors
        c = self.intermediate
        scale_x = self.scale_x
        mask_pad = self.mask_pad
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k//2 if self.bidirectional else k
        skip_type = 0 if not self.has_skip_term else (1 if k_ == 3 else 2)
        ncols = batch*d*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size()).zero_()
        grad_wc = x.new(2, batch, d*bidir)
        grad_bias = x.new(2, batch, d*bidir)
        grad_init = x.new(batch, d*bidir)

        #  For DEBUG
        #  size = (length, batch, x.size(-1)) if x.dim() == 3 else (batch, x.size(-1))
        #  grad_x = x.new(*x.size()) if k_ == 3 else x.new(*size).zero_()

        #  Normal use
        grad_x = x.new(*x.size()).zero_() if skip_type > 0 and k_ == 3 else None

        if skip_type > 0 and k_ == 3:
            x_ptr = x.contiguous()*scale_x if scale_x != 1 else x.contiguous()
            x_ptr = x_ptr.data_ptr()
        else:
            x_ptr = 0

        stream, _, _, bwd_func, bibwd_func = self.get_functions()
        FUNC = bwd_func if not self.bidirectional else bibwd_func
        FUNC(args=[
            u.contiguous().data_ptr(),
            x_ptr,
            weight_c.data_ptr(),
            bias.data_ptr(),
            init_.contiguous().data_ptr(),
            mask_h.data_ptr() if mask_h is not None else 0,
            mask_pad.data_ptr() if mask_pad is not None else 0,
            c.data_ptr(),
            grad_h.contiguous().data_ptr(),
            grad_last.contiguous().data_ptr(),
            length,
            batch,
            d,
            k_,
            grad_u.data_ptr(),
            grad_x.data_ptr() if skip_type > 0 and k_ == 3 else 0,
            grad_wc.data_ptr(),
            grad_bias.data_ptr(),
            grad_init.data_ptr(),
            self.activation_type,
            skip_type],
            block=(thread_per_block, 1, 1),
            grid=(num_block, 1, 1),
            stream=stream
        )

        if skip_type > 0 and k_ == 3 and scale_x != 1:
            grad_x.mul_(scale_x)
        return grad_u, grad_x, grad_wc.sum(1).view(-1), grad_bias.sum(1).view(-1), grad_init, None
