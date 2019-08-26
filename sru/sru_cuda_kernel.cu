#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template <typename scalar_t>
__forceinline__ __device__ scalar_t sigmoidf(scalar_t x) {
    return 1.0 / (1.0 + exp(-x));
}

template <typename scalar_t>
__forceinline__ __device__ scalar_t calc_activation(int type, scalar_t x)
{
    return type ? tanh(x) : x;
}

template <typename scalar_t>
__forceinline__ __device__ scalar_t calc_grad_activation(int type, scalar_t x)
{
    return type ? (1 - x * x) : 1;
}

template <typename scalar_t>
__global__ void cuda_forward_kernel(
                        scalar_t* __restrict__ h,
                        scalar_t* __restrict__ c,
                        const scalar_t* __restrict__ u,
                        const scalar_t* __restrict__ x,
                        const scalar_t* __restrict__ weight_c,
                        const scalar_t* __restrict__ bias,
                        const scalar_t* __restrict__ init,
                        const scalar_t* __restrict__ mask_c,
                        const unsigned char* __restrict__ mask_pad,
                        const int len,
                        const int batch,
                        const int d,
                        const int k,
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

    const auto wc1 = *(weight_c + (col%d));
    const auto wc2 = *(weight_c + (col%d) + d);
    const auto bias1 = *(bias + (col%d));
    const auto bias2 = *(bias + (col%d) + d);
    const auto  mask = (mask_c == NULL) ? 1.0 : (*(mask_c + col));
    auto cur = *(init + col);
    const auto* __restrict__ up = u + (col*k);
    const auto* __restrict__ xp = (skip_type == 0) ? NULL : ((skip_type == 1) ? (x + col) : (up + 3));
    const unsigned char* __restrict__ pad_p = (mask_pad == NULL) ? NULL : (mask_pad + (col/d));
    auto* __restrict__ cp = c + col;
    auto* __restrict__ hp = h + col;

    for (int row = 0; row < len; ++row)
    {
        if ((pad_p == NULL) || !(*pad_p)) {
            const auto g1 = sigmoidf((*(up+1)) + wc1*cur + bias1);
            const auto g2 = sigmoidf((*(up+2)) + wc2*cur + bias2);
            cur = (cur-(*up))*g1 + (*up);
            const auto val = calc_activation(activation_type, cur);
            *hp = skip_type ? ((val * mask - (*xp)) * g2 + (*xp)) : (val * mask * g2);
        } 
        //else {
        //    *hp = 0;  // output 0 for a pad token
        //}
        *cp = cur;  // useful for backward
        up += ncols_u;
        cp += ncols;
        hp += ncols;
        if (skip_type) xp += ncols_x;
        if (pad_p) pad_p += batch;
    }
}

template <typename scalar_t>
__global__ void cuda_backward_kernel(
                        scalar_t* __restrict__ grad_u,
                        scalar_t* __restrict__ grad_x,
                        scalar_t* __restrict__ grad_wc,
                        scalar_t* __restrict__ grad_bias,
                        scalar_t* __restrict__ grad_init,
                        const scalar_t* __restrict__ u,
                        const scalar_t* __restrict__ x,
                        const scalar_t* __restrict__ weight_c,
                        const scalar_t* __restrict__ bias,
                        const scalar_t* __restrict__ init,
                        const scalar_t* __restrict__ mask_c,
                        const unsigned char * __restrict__ mask_pad,
                        const scalar_t* __restrict__ c,
                        const scalar_t* __restrict__ grad_h,
                        const scalar_t* __restrict__ grad_last,
                        const int len,
                        const int batch,
                        const int d,
                        const int k,
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

    const auto wc1 = *(weight_c + (col%d));
    const auto wc2 = *(weight_c + (col%d) + d);
    const auto bias1 = *(bias + (col%d));
    const auto bias2 = *(bias + (col%d) + d);
    const auto mask = (mask_c == NULL) ? 1.0 : (*(mask_c + col));
    scalar_t gwc1 = 0;
    scalar_t gwc2 = 0;
    scalar_t gbias1 = 0;
    scalar_t gbias2 = 0;
    auto cur = *(grad_last + col);

    const auto* __restrict__ up = u + (col*k) + (len-1)*ncols_u;
    const auto* __restrict__ xp = (skip_type == 0) ? NULL : (
        (skip_type == 1) ? (x + col + (len-1)*ncols) : (up + 3)
    );
    const auto* __restrict__ cp = c + col + (len-1)*ncols;
    const auto* __restrict__ ghp = grad_h + col + (len-1)*ncols;
    const unsigned char* __restrict__ pad_p = (mask_pad == NULL) ? NULL : (mask_pad + (col/d) + (len-1)*batch);
    auto* __restrict__ gup = grad_u + (col*k) + (len-1)*ncols_u;
    auto* __restrict__ gxp = (skip_type == 0) ? NULL : (
        (skip_type == 1) ? (grad_x + col + (len-1)*ncols) : (gup + 3)
    );

    for (int row = len-1; row >= 0; --row)
    {
        if ((pad_p == NULL) || !(*pad_p)) {
            const auto prev_c_val = (row>0) ? (*(cp-ncols)) : (*(init+col));
            const auto g1 = sigmoidf((*(up+1)) + wc1*prev_c_val + bias1);
            const auto g2 = sigmoidf((*(up+2)) + wc2*prev_c_val + bias2);
            const auto c_val = calc_activation(activation_type, *cp);
            const auto x_val = (skip_type) ? (*xp) : 0;
            const auto u_val = *up;
            const auto gh_val = *ghp;

            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + u0*(1-g1) = (c'-u0)*g1 + g0

            // gradient with respect to x[t]
            if (skip_type)
                *gxp = gh_val*(1-g2);

            // gradient with respect to values in the second gate g2
            auto gg2 = gh_val*(c_val*mask-x_val)*(g2*(1-g2));
            *(gup+2) = gg2;
            gbias2 += gg2;
            gwc2 += gg2*prev_c_val;

            // gradient with respect to c[t]
            const auto tmp = g2*calc_grad_activation(activation_type, c_val);
            const auto gc = gh_val*mask*tmp + cur;

            // gradient with respect to current input u0=W*x[t]
            *gup = gc*(1-g1);

            // gradient with respect to values in the first gate g1
            auto gg1 = gc*(prev_c_val-u_val)*(g1*(1-g1));
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
        if (skip_type) xp -= ncols_x;
        if (skip_type) gxp -= ncols_x;
        if (pad_p) pad_p -= batch;
    }

    //const int bias_idx = col % d;
    //atomicAdd(grad_wc + bias_idx, gwc1);
    //atomicAdd(grad_wc + bias_idx + d, gwc2);
    //atomicAdd(grad_bias + bias_idx, gbias1);
    //atomicAdd(grad_bias + bias_idx + d, gbias2);
    *(grad_wc + col) = gwc1;
    *(grad_wc + col + ncols) = gwc2;
    *(grad_bias + col) = gbias1;
    *(grad_bias + col + ncols) = gbias2;
    *(grad_init +col) = cur;
}

template <typename scalar_t>
__global__ void cuda_bi_forward_kernel(
                        scalar_t* __restrict__ h,
                        scalar_t* __restrict__ c,
                        const scalar_t* __restrict__ u,
                        const scalar_t* __restrict__ x,
                        const scalar_t* __restrict__ weight_c,
                        const scalar_t* __restrict__ bias,
                        const scalar_t* __restrict__ init,
                        const scalar_t* __restrict__ mask_c,
                        const unsigned char * __restrict__ mask_pad,
                        const int len,
                        const int batch,
                        const int d,
                        const int k,
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
    const scalar_t mask = (mask_c == NULL) ? 1.0 : (*(mask_c + col));
    auto cur = *(init + col);
    const int d2 = d*2;
    const auto wc1 = *(weight_c + (col%d2));
    const auto wc2 = *(weight_c + (col%d2) + d2);
    const auto bias1 = *(bias + (col%d2));
    const auto bias2 = *(bias + (col%d2) + d2);

    const auto *up = u + (col*k);
    const auto *xp = (skip_type == 0) ? NULL : ((skip_type == 1) ? (x + col) : (up + 3));
    const unsigned char *pad_p = (mask_pad == NULL) ? NULL : (mask_pad + (col/d2));
    auto *cp = c + col;
    auto *hp = h + col;
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
            auto g1 = sigmoidf((*(up+1)) + wc1*cur + bias1);
            auto g2 = sigmoidf((*(up+2)) + wc2*cur + bias2);
            cur = (cur-(*up))*g1 + (*up);
            auto val = calc_activation(activation_type, cur);
            if (skip_type)
                *hp = (val*mask-(*xp))*g2 + (*xp);
            else
                *hp = val*mask*g2;
        } else {
            *hp = 0;  // ouptut 0 for a pad token
        }
        *cp = cur;  // useful for backward
        up += ncols_u_;
        cp += ncols_;
        hp += ncols_;
        if (skip_type) xp += ncols_x_;
        if (pad_p) pad_p += batch_;
    }
}

template <typename scalar_t>
__global__ void cuda_bi_backward_kernel(
                           scalar_t* __restrict__ grad_u,
                           scalar_t* __restrict__ grad_x,
                           scalar_t* __restrict__ grad_wc,
                           scalar_t* __restrict__ grad_bias,
                           scalar_t* __restrict__ grad_init,
                           const scalar_t* __restrict__ u,
                           const scalar_t* __restrict__ x,
                           const scalar_t* __restrict__ weight_c,
                           const scalar_t* __restrict__ bias,
                           const scalar_t* __restrict__ init,
                           const scalar_t* __restrict__ mask_c,
                           const unsigned char * __restrict__ mask_pad,
                           const scalar_t* __restrict__ c,
                           const scalar_t* __restrict__ grad_h,
                           const scalar_t* __restrict__ grad_last,
                           const int len,
                           const int batch,
                           const int d,
                           const int k,
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
    const scalar_t mask = (mask_c == NULL) ? 1.0 : (*(mask_c + col));
    scalar_t gwc1 = 0;
    scalar_t gwc2 = 0;
    scalar_t gbias1 = 0;
    scalar_t gbias2 = 0;
    auto cur = *(grad_last + col);
    const int d2 = d*2;
    const auto wc1 = *(weight_c + (col%d2));
    const auto wc2 = *(weight_c + (col%d2) + d2);
    const auto bias1 = *(bias + (col%d2));
    const auto bias2 = *(bias + (col%d2) + d2);

    const auto *up = u + (col*k);
    const auto *xp = (skip_type == 0) ? NULL : (
        (skip_type == 1) ? (x + col) : (up + 3)
    );
    const auto *cp = c + col;
    const auto *ghp = grad_h + col;
    const unsigned char *pad_p = (mask_pad == NULL) ? NULL : (mask_pad + (col/d2));
    auto *gup = grad_u + (col*k);
    auto *gxp = (skip_type == 0) ? NULL : (
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
            const auto prev_c_val = (cnt<len-1) ? (*(cp-ncols_)) : (*(init+col));
            const auto g1 = sigmoidf((*(up+1)) + wc1*prev_c_val + bias1);
            const auto g2 = sigmoidf((*(up+2)) + wc2*prev_c_val + bias2);
            const auto c_val = calc_activation(activation_type, *cp);
            const auto x_val = (skip_type) ? (*xp) : 0;
            const auto u_val = *up;
            const auto gh_val = *ghp;

            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + u0*(1-g1) = (c'-u0)*g1 + u0

            // gradient with respect to x[t]
            if (skip_type)
                *gxp = gh_val*(1-g2);

            // gradient with respect to values in the second gate g2
            auto gg2 = gh_val*(c_val*mask-x_val)*(g2*(1-g2));
            *(gup+2) = gg2;
            gbias2 += gg2;
            gwc2 += gg2*prev_c_val;

            // gradient with respect to c[t]
            const auto tmp = g2*calc_grad_activation(activation_type, c_val);
            const auto gc = gh_val*mask*tmp + cur;

            // gradient with respect to u[0]=W*x[t]
            *gup = gc*(1-g1);

            // gradient with respect to values in the first gate g1
            auto gg1 = gc*(prev_c_val-u_val)*(g1*(1-g1));
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

    //const int bias_idx = col % d2;
    //atomicAdd(grad_wc + bias_idx, gwc1);
    //atomicAdd(grad_wc + bias_idx + d2, gwc2);
    //atomicAdd(grad_bias + bias_idx, gbias1);
    //atomicAdd(grad_bias + bias_idx + d2, gbias2);
    *(grad_wc + col) = gwc1;
    *(grad_wc + col + ncols) = gwc2;
    *(grad_bias + col) = gbias1;
    *(grad_bias + col + ncols) = gbias2;
    *(grad_init +col) = cur;
}

} //  end of namespace

//  unidirectional forward()
void sru_cuda_forward(
        torch::Tensor & h,
        torch::Tensor & c,
        const torch::Tensor & U,
        const torch::Tensor & x,
        const torch::Tensor & weight_c,
        const torch::Tensor & bias,
        const torch::Tensor & c_init,
        const torch::Tensor & mask_c,
        const torch::Tensor & mask_pad,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type) {

    const int threads = 512;
    const int total = batch_size * hidden_size;
    const dim3 blocks( (total - 1) / threads + 1 );

    AT_DISPATCH_FLOATING_TYPES(U.type(), "sru_forward_cuda", ([&] {
        cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            h.data<scalar_t>(),
            c.data<scalar_t>(),
            U.data<scalar_t>(),
            x.data<scalar_t>(),
            weight_c.data<scalar_t>(),
            bias.data<scalar_t>(),
            c_init.data<scalar_t>(),
            mask_c.numel() ? mask_c.data<scalar_t>() : NULL,
            mask_pad.numel() ? mask_pad.data<unsigned char>() : NULL,
            length,
            batch_size,
            hidden_size,
            k,
            activation_type,
            skip_type);
    }));
}

//  bidirectional forward()
void sru_cuda_bi_forward(
        torch::Tensor & h,
        torch::Tensor & c,
        const torch::Tensor & U,
        const torch::Tensor & x,
        const torch::Tensor & weight_c,
        const torch::Tensor & bias,
        const torch::Tensor & c_init,
        const torch::Tensor & mask_c,
        const torch::Tensor & mask_pad,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type) {

    const int threads = 512;
    const int total = batch_size * hidden_size * 2;
    const dim3 blocks( (total - 1) / threads + 1 );

    AT_DISPATCH_FLOATING_TYPES(U.type(), "sru_bi_forward_cuda", ([&] {
        cuda_bi_forward_kernel<scalar_t><<<blocks, threads>>>(
            h.data<scalar_t>(),
            c.data<scalar_t>(),
            U.data<scalar_t>(),
            x.data<scalar_t>(),
            weight_c.data<scalar_t>(),
            bias.data<scalar_t>(),
            c_init.data<scalar_t>(),
            mask_c.numel() ? mask_c.data<scalar_t>() : NULL,
            mask_pad.numel() ? mask_pad.data<unsigned char>() : NULL,
            length,
            batch_size,
            hidden_size,
            k,
            activation_type,
            skip_type);
    }));
}

//  unidirectional backward()
void sru_cuda_backward(
        torch::Tensor & grad_u,
        torch::Tensor & grad_x,
        torch::Tensor & grad_wc,
        torch::Tensor & grad_bias,
        torch::Tensor & grad_init,
        const torch::Tensor & U,
        const torch::Tensor & x,
        const torch::Tensor & weight_c,
        const torch::Tensor & bias,
        const torch::Tensor & c_init,
        const torch::Tensor & mask_c,
        const torch::Tensor & mask_pad,
        const torch::Tensor & c,
        const torch::Tensor & grad_h,
        const torch::Tensor & grad_last,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type) {

    const int threads = 512;
    const int total = batch_size * hidden_size;
    const dim3 blocks( (total - 1) / threads + 1 );

    AT_DISPATCH_FLOATING_TYPES(U.type(), "sru_backward_cuda", ([&] {
        cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_u.data<scalar_t>(),
            grad_x.numel() ? grad_x.data<scalar_t>() : NULL,
            grad_wc.data<scalar_t>(),
            grad_bias.data<scalar_t>(),
            grad_init.data<scalar_t>(),
            U.data<scalar_t>(),
            x.data<scalar_t>(),
            weight_c.data<scalar_t>(),
            bias.data<scalar_t>(),
            c_init.data<scalar_t>(),
            mask_c.numel() ? mask_c.data<scalar_t>() : NULL,
            mask_pad.numel() ? mask_pad.data<unsigned char>() : NULL,
            c.data<scalar_t>(),
            grad_h.data<scalar_t>(),
            grad_last.data<scalar_t>(),
            length,
            batch_size,
            hidden_size,
            k,
            activation_type,
            skip_type);
    }));
}

//  bidirectional backward()
void sru_cuda_bi_backward(
        torch::Tensor & grad_u,
        torch::Tensor & grad_x,
        torch::Tensor & grad_wc,
        torch::Tensor & grad_bias,
        torch::Tensor & grad_init,
        const torch::Tensor & U,
        const torch::Tensor & x,
        const torch::Tensor & weight_c,
        const torch::Tensor & bias,
        const torch::Tensor & c_init,
        const torch::Tensor & mask_c,
        const torch::Tensor & mask_pad,
        const torch::Tensor & c,
        const torch::Tensor & grad_h,
        const torch::Tensor & grad_last,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type) {

    const int threads = 512;
    const int total = batch_size * hidden_size * 2;
    const dim3 blocks( (total - 1) / threads + 1 );

    AT_DISPATCH_FLOATING_TYPES(U.type(), "sru_bi_backward_cuda", ([&] {
        cuda_bi_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_u.data<scalar_t>(),
            grad_x.numel() ? grad_x.data<scalar_t>() : NULL,
            grad_wc.data<scalar_t>(),
            grad_bias.data<scalar_t>(),
            grad_init.data<scalar_t>(),
            U.data<scalar_t>(),
            x.data<scalar_t>(),
            weight_c.data<scalar_t>(),
            bias.data<scalar_t>(),
            c_init.data<scalar_t>(),
            mask_c.numel() ? mask_c.data<scalar_t>() : NULL,
            mask_pad.numel() ? mask_pad.data<unsigned char>() : NULL,
            c.data<scalar_t>(),
            grad_h.data<scalar_t>(),
            grad_last.data<scalar_t>(),
            length,
            batch_size,
            hidden_size,
            k,
            activation_type,
            skip_type);
    }));
}


