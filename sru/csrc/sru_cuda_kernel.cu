#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template <typename scalar_t>
__forceinline__ __device__ scalar_t sigmoidf(scalar_t x) {
    return (scalar_t)1.f / ((scalar_t)1.f + expf(-x));
}

template <typename scalar_t>
__forceinline__ __device__ scalar_t calc_activation(int type, scalar_t x)
{
    return type ? (scalar_t)tanhf(x) : x;
}

template <typename scalar_t>
__forceinline__ __device__ scalar_t calc_grad_activation(int type, scalar_t x)
{
    return type ? ((scalar_t)1.f - x * x) : (scalar_t)1.f;
}

template <typename scalar_t>
__global__ void sru_cuda_forward_kernel_simple(
                        scalar_t* __restrict__ h,
                        scalar_t* __restrict__ c,
                        const scalar_t* __restrict__ u,
                        const scalar_t* __restrict__ x,
                        const scalar_t* __restrict__ weight_c,
                        const scalar_t* __restrict__ bias,
                        const scalar_t* __restrict__ init,
                        const scalar_t* __restrict__ mask_c,
                        const bool* __restrict__ mask_pad,
                        const int len,
                        const int batch,
                        const int d)
{
    const int ncols = batch * d;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    const int ncols_u = ncols * 3;
    const int ncols_x = ncols;

    const auto wc1 = *(weight_c + (col % d));
    const auto wc2 = *(weight_c + (col % d) + d);

    const auto bias1 = *(bias + (col % d));
    const auto bias2 = *(bias + (col % d) + d);
    const auto  mask = (mask_c == NULL) ? (scalar_t)1.f : (*(mask_c + col));
    auto cur = *(init + col);
    const auto* up = u + (col * 3);
    const auto* xp = x + col;
    const bool* pad_p = (mask_pad == NULL) ? NULL : (mask_pad + (col / d));
    auto* cp = c + col;
    auto* hp = h + col;

    for (int row = 0; row < len; ++row) {
        if ((pad_p == NULL) || !(*pad_p)) {
            const auto u0 = *up;
            const auto u1 = *(up + 1);
            const auto u2 = *(up + 2);

            const auto x_val = *xp;
            const auto g1 = sigmoidf(u1 + wc1 * cur + bias1);
            const auto g2 = sigmoidf(u2 + wc2 * cur + bias2);
            cur = (cur - u0) * g1 + u0;
            *hp = (cur - x_val) * mask * g2 + x_val;
        } 
        *cp = cur;  // useful for backward
        up += ncols_u;
        cp += ncols;
        hp += ncols;
        xp += ncols_x;
        pad_p = mask_pad ? (pad_p + batch) : NULL;
    }
}

template <typename scalar_t>
__global__ void sru_cuda_backward_kernel_simple(
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
                        const bool* __restrict__ mask_pad,
                        const scalar_t* __restrict__ c,
                        const scalar_t* __restrict__ grad_h,
                        const scalar_t* __restrict__ grad_last,
                        const int len,
                        const int batch,
                        const int d)
{
    const int ncols = batch * d;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    const int ncols_u = ncols * 3;
    const int ncols_x = ncols;

    const auto wc1 = *(weight_c + (col % d));
    const auto wc2 = *(weight_c + (col % d) + d);

    const auto bias1 = *(bias + (col % d));
    const auto bias2 = *(bias + (col % d) + d);
    const auto mask = (mask_c == NULL) ? (scalar_t)1.f : (*(mask_c + col));
    scalar_t gwc1 = 0;
    scalar_t gwc2 = 0;
    scalar_t gbias1 = 0;
    scalar_t gbias2 = 0;
    auto cur = *(grad_last + col);

    const auto* up = u + (col * 3) + (len - 1) * ncols_u;
    const auto* xp = x + col + (len - 1) * ncols;
    const auto* cp = c + col + (len - 1) * ncols;
    const auto* ghp = grad_h + col + (len - 1) * ncols;
    const bool* pad_p = (mask_pad == NULL) ? NULL : 
                                 (mask_pad + (col / d) + (len - 1) * batch);
    auto* gup = grad_u + (col * 3) + (len - 1) * ncols_u;
    auto* gxp = grad_x + col + (len - 1) * ncols;

    for (int row = len-1; row >= 0; --row) {
        if ((pad_p == NULL) || !(*pad_p)) {
            const auto prev_c_val = row ? (*(cp - ncols)) : (*(init + col));
            const auto c_val = *cp;
            const auto u0 = *up;
            const auto u1 = *(up + 1);
            const auto u2 = *(up + 2);

            const auto x_val = *xp;
            const auto gh_val = *ghp;
            const auto g1 = sigmoidf(u1 + wc1 * prev_c_val + bias1);
            const auto g2 = sigmoidf(u2 + wc2 * prev_c_val + bias2);

            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + u0*(1-g1) = (c'-u0)*g1 + g0

            // gradient with respect to values in the second gate g2
            const auto gg2 = gh_val * (c_val - x_val) * mask * (g2 * (1.f - g2));
            gbias2 += gg2;
            gwc2 += gg2 * prev_c_val;

            // gradient with respect to c[t]
            const auto gc = gh_val * mask * g2 + cur;

            // gradient with respect to values in the first gate g1
            const auto gg1 = gc * (prev_c_val - u0) * (g1 * (1.f - g1));
            gbias1 += gg1;
            gwc1 += gg1 * prev_c_val;

            // gradient with respect to c[t-1]
            cur = gc * g1 + gg1 * wc1 + gg2 * wc2;

            // gradient with respect to U
            *gup = gc * (1.f - g1);
            *(gup + 1) = gg1;
            *(gup + 2) = gg2;
 
            // gradient with respect to x[t]
            *gxp = gh_val * (1.f - g2 * mask);
        }

        up -= ncols_u;
        cp -= ncols;
        gup -= ncols_u;
        ghp -= ncols;
        xp -= ncols_x;
        gxp -= ncols_x;
        pad_p = mask_pad ? (pad_p - batch) : NULL;
    }

    *(grad_wc + col) = gwc1;
    *(grad_wc + col + ncols) = gwc2;
    *(grad_bias + col) = gbias1;
    *(grad_bias + col + ncols) = gbias2;
    *(grad_init + col) = cur;
}

template <typename scalar_t>
__global__ void sru_cuda_bi_forward_kernel_simple(
                        scalar_t* __restrict__ h,
                        scalar_t* __restrict__ c,
                        const scalar_t* __restrict__ u,
                        const scalar_t* __restrict__ x,
                        const scalar_t* __restrict__ weight_c,
                        const scalar_t* __restrict__ bias,
                        const scalar_t* __restrict__ init,
                        const scalar_t* __restrict__ mask_c,
                        const bool* __restrict__ mask_pad,
                        const int len,
                        const int batch,
                        const int d)
{
    const int ncols = batch * d * 2;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    const int ncols_u = ncols * 3;
    const int ncols_x = ncols;
    const scalar_t mask = (mask_c == NULL) ? (scalar_t)1.f : (*(mask_c + col));
    auto cur = *(init + col);
    const int d2 = d * 2;

    const auto wc1 = *(weight_c + (col % d2));
    const auto wc2 = *(weight_c + (col % d2) + d2);

    const auto bias1 = *(bias + (col % d2));
    const auto bias2 = *(bias + (col % d2) + d2);

    const auto *up = u + (col * 3);
    const auto *xp = x + col;
    const bool *pad_p = (mask_pad == NULL) ? NULL : (mask_pad + (col / d2));
    auto *cp = c + col;
    auto *hp = h + col;
    const bool flip = (col % d2) >= d;
    if (flip) {
        up += (len - 1) * ncols_u;
        cp += (len - 1) * ncols;
        hp += (len - 1) * ncols;
        xp += (len - 1) * ncols_x;
        if (pad_p) pad_p += (len - 1) * batch;
    }
    const int ncols_u_ = flip ? -ncols_u : ncols_u;
    const int ncols_x_ = flip ? -ncols_x : ncols_x;
    const int ncols_ = flip ? -ncols : ncols;
    const int batch_ = flip ? -batch : batch;

    for (int cnt = 0; cnt < len; ++cnt) {
        if ((pad_p == NULL) || !(*pad_p)) {
            const auto u0 = *up;
            const auto u1 = *(up + 1);
            const auto u2 = *(up + 2);

            const auto x_val = *xp;
            const auto g1 = sigmoidf(u1 + wc1 * cur + bias1);
            const auto g2 = sigmoidf(u2 + wc2 * cur + bias2);
            cur = (cur - u0) * g1 + u0;
            *hp = (cur - x_val) * mask * g2 + x_val;
        } 
        *cp = cur;  // useful for backward
        up += ncols_u_;
        cp += ncols_;
        hp += ncols_;
        xp += ncols_x_;
        pad_p = mask_pad ? (pad_p + batch_) : NULL;
    }
}

template <typename scalar_t>
__global__ void sru_cuda_bi_backward_kernel_simple(
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
                           const bool* __restrict__ mask_pad,
                           const scalar_t* __restrict__ c,
                           const scalar_t* __restrict__ grad_h,
                           const scalar_t* __restrict__ grad_last,
                           const int len,
                           const int batch,
                           const int d)
{
    int ncols = batch * d * 2;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int ncols_u = ncols * 3;
    int ncols_x = ncols;
    const scalar_t mask = (mask_c == NULL) ? (scalar_t)1.f : (*(mask_c + col));
    scalar_t gwc1 = 0;
    scalar_t gwc2 = 0;
    scalar_t gbias1 = 0;
    scalar_t gbias2 = 0;
    auto cur = *(grad_last + col);
    const int d2 = d * 2;

    const auto wc1 = *(weight_c + (col % d2));
    const auto wc2 = *(weight_c + (col % d2) + d2);

    const auto bias1 = *(bias + (col % d2));
    const auto bias2 = *(bias + (col % d2) + d2);

    const auto *up = u + (col * 3);
    const auto *xp = x + col;
    const auto *cp = c + col;
    const auto *ghp = grad_h + col;
    const bool *pad_p = (mask_pad == NULL) ? NULL : (mask_pad + (col / d2));
    auto *gup = grad_u + (col * 3);
    auto *gxp = grad_x + col;

    const bool flip = ((col % d2) >= d);
    if (!flip) {
        up += (len - 1) * ncols_u;
        cp += (len - 1) * ncols;
        ghp += (len - 1) * ncols;
        gup += (len - 1) * ncols_u;
        xp += (len - 1) * ncols_x;
        gxp += (len - 1) * ncols_x;
        if (pad_p) pad_p += (len - 1) * batch;
    }
    const int ncols_u_ = flip ? -ncols_u : ncols_u;
    const int ncols_x_ = flip ? -ncols_x : ncols_x;
    const int ncols_ = flip ? -ncols : ncols;
    const int batch_ = flip ? -batch : batch;

    for (int cnt = 0; cnt < len; ++cnt) {
        if ((pad_p == NULL) || !(*pad_p)) {
            const auto prev_c_val = (cnt < len - 1) ? (*(cp - ncols_)) : (*(init + col));
            const auto c_val = *cp;
            const auto u0 = *up;
            const auto u1 = *(up + 1);
            const auto u2 = *(up + 2);

            const auto x_val = *xp;
            const auto gh_val = *ghp;
            const auto g1 = sigmoidf(u1 + wc1 * prev_c_val + bias1);
            const auto g2 = sigmoidf(u2 + wc2 * prev_c_val + bias2);

            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + u0*(1-g1) = (c'-u0)*g1 + u0

            // gradient with respect to values in the second gate g2
            const auto gg2 = gh_val * (c_val - x_val) * mask * (g2 * (1.f - g2));
            gbias2 += gg2;
            gwc2 += gg2 * prev_c_val;

            // gradient with respect to c[t]
            const auto gc = gh_val * mask * g2 + cur;

            // gradient with respect to values in the first gate g1
            const auto gg1 = gc * (prev_c_val - u0) * (g1 * (1.f - g1));
            gbias1 += gg1;
            gwc1 += gg1 * prev_c_val;

            // gradient with respect to c[t-1]
            cur = gc * g1 + gg1 * wc1 + gg2 * wc2;

            // gradient with respect to U
            *gup = gc * (1.f - g1);
            *(gup + 1) = gg1;
            *(gup + 2) = gg2;

            // gradient with respect to x[t]
            *gxp = gh_val * (1.f - g2 * mask);
        }

        up -= ncols_u_;
        cp -= ncols_;
        gup -= ncols_u_;
        ghp -= ncols_;
        xp -= ncols_x_;
        gxp -= ncols_x_;
        pad_p = mask_pad ? (pad_p - batch_) : NULL;
    }

    *(grad_wc + col) = gwc1;
    *(grad_wc + col + ncols) = gwc2;
    *(grad_bias + col) = gbias1;
    *(grad_bias + col + ncols) = gbias2;
    *(grad_init +col) = cur;
}

template <typename scalar_t>
__global__ void sru_cuda_forward_kernel(
                        scalar_t* __restrict__ h,
                        scalar_t* __restrict__ c,
                        const scalar_t* __restrict__ u,
                        const scalar_t* __restrict__ x,
                        const scalar_t* __restrict__ weight_c,
                        const scalar_t* __restrict__ bias,
                        const scalar_t* __restrict__ init,
                        const scalar_t* __restrict__ mask_c,
                        const bool* __restrict__ mask_pad,
                        const int len,
                        const int batch,
                        const int d,
                        const int k,
                        const int activation_type,
                        const int skip_type,
                        const int is_custom)
{
    assert ((skip_type >= 0) && (skip_type <= 2));
    assert ((skip_type != 1) || (k == 3));
    assert ((skip_type != 2) || (k == 4));

    const int ncols = batch * d;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    const int ncols_u = ncols * k;
    const int ncols_x = (k == 3) ? ncols : ncols_u;

    const auto* vp1 = is_custom ? (weight_c + col * 2) : (weight_c + (col % d));
    const auto* vp2 = is_custom ? (weight_c + col * 2 + 1) : (weight_c + (col % d) + d);

    const auto bias1 = *(bias + (col % d));
    const auto bias2 = *(bias + (col % d) + d);
    const auto  mask = (mask_c == NULL) ? (scalar_t)1.f : (*(mask_c + col));
    auto cur = *(init + col);
    const auto* up = u + (col * k);
    const auto* xp = (skip_type == 0) ? NULL : ((skip_type == 1) ? (x + col) : (up + 3));
    const bool* pad_p = (mask_pad == NULL) ? NULL : (mask_pad + (col / d));
    auto* cp = c + col;
    auto* hp = h + col;

    for (int row = 0; row < len; ++row) {
        if ((pad_p == NULL) || !(*pad_p)) {
            const auto u0 = *up;
            const auto u1 = *(up + 1);
            const auto u2 = *(up + 2);
            const auto wc1 = *vp1;
            const auto wc2 = *vp2;

            const auto x_val = (skip_type) ? (*xp) : (scalar_t)0.f;
            const auto g1 = sigmoidf(u1 + wc1 * cur + bias1);
            const auto g2 = sigmoidf(u2 + wc2 * cur + bias2);
            cur = (cur - u0) * g1 + u0;
            const auto val = calc_activation(activation_type, cur);
            *hp = skip_type ? ((val - x_val) * mask * g2 + x_val) : (val * mask * g2);
        } 
        *cp = cur;  // useful for backward
        up += ncols_u;
        cp += ncols;
        hp += ncols;
        xp = skip_type ? (xp + ncols_x) : NULL;
        pad_p = mask_pad ? (pad_p + batch) : NULL;
        vp1 = is_custom ? (vp1 + ncols*2) : vp1;
        vp2 = is_custom ? (vp2 + ncols*2) : vp2;
    }
}

template <typename scalar_t>
__global__ void sru_cuda_backward_kernel(
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
                        const bool* __restrict__ mask_pad,
                        const scalar_t* __restrict__ c,
                        const scalar_t* __restrict__ grad_h,
                        const scalar_t* __restrict__ grad_last,
                        const int len,
                        const int batch,
                        const int d,
                        const int k,
                        const int activation_type,
                        const int skip_type,
                        const int is_custom)
{
    assert ((skip_type >= 0) && (skip_type <= 2));
    assert ((skip_type != 1) || (k == 3));
    assert ((skip_type != 2) || (k == 4));

    const int ncols = batch * d;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    const int ncols_u = ncols * k;
    const int ncols_x = (k == 3) ? ncols : ncols_u;

    const auto* vp1 = is_custom ? (weight_c + col * 2 + (len - 1) * ncols * 2) : (weight_c + (col % d));
    const auto* vp2 = is_custom ? (weight_c + col * 2 + 1 + (len - 1) * ncols * 2) : (weight_c + (col % d) + d);
    auto* gvp1 = is_custom ? (grad_wc + col * 2 + (len - 1) * ncols * 2) : (grad_wc + col);
    auto* gvp2 = is_custom ? (grad_wc + col * 2 + 1 + (len - 1) * ncols * 2) : (grad_wc + col + ncols);

    const auto bias1 = *(bias + (col % d));
    const auto bias2 = *(bias + (col % d) + d);
    const auto mask = (mask_c == NULL) ? (scalar_t)1.f : (*(mask_c + col));
    scalar_t gwc1 = 0;
    scalar_t gwc2 = 0;
    scalar_t gbias1 = 0;
    scalar_t gbias2 = 0;
    auto cur = *(grad_last + col);

    const auto* up = u + (col * k) + (len - 1) * ncols_u;
    const auto* xp = (skip_type == 0) ? NULL : (
        (skip_type == 1) ? (x + col + (len - 1) * ncols) : (up + 3)
    );
    const auto* cp = c + col + (len - 1) * ncols;
    const auto* ghp = grad_h + col + (len - 1) * ncols;
    const bool* pad_p = (mask_pad == NULL) ? NULL :
                                 (mask_pad + (col / d) + (len - 1) * batch);
    auto* gup = grad_u + (col * k) + (len - 1) * ncols_u;
    auto* gxp = (skip_type == 0) ? NULL : (
        (skip_type == 1) ? (grad_x + col + (len - 1) * ncols) : (gup + 3)
    );

    for (int row = len-1; row >= 0; --row) {
        if ((pad_p == NULL) || !(*pad_p)) {
            const auto prev_c_val = (row > 0) ? (*(cp - ncols)) : (*(init + col));
            const auto cp_val = *cp;
            const auto u0 = *up;
            const auto u1 = *(up + 1);
            const auto u2 = *(up + 2);
            const auto wc1 = *vp1;
            const auto wc2 = *vp2;

            const auto x_val = (skip_type) ? (*xp) : (scalar_t)0.f;
            const auto gh_val = *ghp;
            const auto g1 = sigmoidf(u1 + wc1 * prev_c_val + bias1);
            const auto g2 = sigmoidf(u2 + wc2 * prev_c_val + bias2);
            const auto c_val = calc_activation(activation_type, cp_val);

            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + u0*(1-g1) = (c'-u0)*g1 + g0

            // gradient with respect to values in the second gate g2
            const auto gg2 = gh_val * (c_val - x_val) * mask * (g2 * (1.f - g2));
            gbias2 += gg2;
            gwc2 += gg2 * prev_c_val;
            *gvp2 = gg2 * prev_c_val;

            // gradient with respect to c[t]
            const auto tmp = g2 * calc_grad_activation(activation_type, c_val);
            const auto gc = gh_val * mask * tmp + cur;

            // gradient with respect to values in the first gate g1
            const auto gg1 = gc * (prev_c_val - u0) * (g1 * (1.f - g1));
            gbias1 += gg1;
            gwc1 += gg1 * prev_c_val;
            *gvp1 = gg1 * prev_c_val;

            // gradient with respect to c[t-1]
            cur = gc * g1 + gg1 * wc1 + gg2 * wc2;

            // gradient with respect to U
            *gup = gc * (1.f - g1);
            *(gup + 1) = gg1;
            *(gup + 2) = gg2;
 
            // gradient with respect to x[t]
            if (skip_type)
                *gxp = gh_val * (1.f - g2 * mask);
        }

        up -= ncols_u;
        cp -= ncols;
        gup -= ncols_u;
        ghp -= ncols;
        xp = skip_type ? (xp - ncols_x) : NULL;
        gxp = skip_type ? (gxp - ncols_x) : NULL;
        pad_p = mask_pad ? (pad_p - batch) : NULL;
        vp1 = is_custom ? (vp1 - ncols*2) : vp1;
        vp2 = is_custom ? (vp2 - ncols*2) : vp2;
        gvp1 = is_custom ? (gvp1 - ncols*2) : gvp1;
        gvp2 = is_custom ? (gvp2 - ncols*2) : gvp2;
    }

    if (!is_custom) {
        *(grad_wc + col) = gwc1;
        *(grad_wc + col + ncols) = gwc2;
    }
    *(grad_bias + col) = gbias1;
    *(grad_bias + col + ncols) = gbias2;
    *(grad_init + col) = cur;
}

template <typename scalar_t>
__global__ void sru_cuda_bi_forward_kernel(
                        scalar_t* __restrict__ h,
                        scalar_t* __restrict__ c,
                        const scalar_t* __restrict__ u,
                        const scalar_t* __restrict__ x,
                        const scalar_t* __restrict__ weight_c,
                        const scalar_t* __restrict__ bias,
                        const scalar_t* __restrict__ init,
                        const scalar_t* __restrict__ mask_c,
                        const bool* __restrict__ mask_pad,
                        const int len,
                        const int batch,
                        const int d,
                        const int k,
                        const int activation_type,
                        const int skip_type,
                        const int is_custom)
{
    assert ((skip_type >= 0) && (skip_type <= 2));
    assert ((skip_type != 1) || (k == 3));
    assert ((skip_type != 2) || (k == 4));

    const int ncols = batch * d * 2;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    const int ncols_u = ncols * k;
    const int ncols_x = (k == 3) ? ncols : ncols_u;
    const scalar_t mask = (mask_c == NULL) ? (scalar_t)1.f : (*(mask_c + col));
    auto cur = *(init + col);
    const int d2 = d * 2;

    const auto* vp1 = is_custom ? (weight_c + col * 2) : (weight_c + (col % d2));
    const auto* vp2 = is_custom ? (weight_c + col * 2 + 1) : (weight_c + (col % d2) + d2);

    const auto bias1 = *(bias + (col % d2));
    const auto bias2 = *(bias + (col % d2) + d2);

    const auto *up = u + (col * k);
    const auto *xp = (skip_type == 0) ? NULL : ((skip_type == 1) ? (x + col) : (up + 3));
    const bool *pad_p = (mask_pad == NULL) ? NULL : (mask_pad + (col / d2));
    auto *cp = c + col;
    auto *hp = h + col;
    const bool flip = (col % d2) >= d;
    if (flip) {
        up += (len - 1) * ncols_u;
        cp += (len - 1) * ncols;
        hp += (len - 1) * ncols;
        if (skip_type) xp += (len - 1) * ncols_x;
        if (pad_p) pad_p += (len - 1) * batch;
        if (is_custom) {
            vp1 += (len - 1) * ncols * 2;
            vp2 += (len - 1) * ncols * 2;
        }
    }
    const int ncols_u_ = flip ? -ncols_u : ncols_u;
    const int ncols_x_ = flip ? -ncols_x : ncols_x;
    const int ncols_ = flip ? -ncols : ncols;
    const int batch_ = flip ? -batch : batch;

    for (int cnt = 0; cnt < len; ++cnt) {
        if ((pad_p == NULL) || !(*pad_p)) {
            const auto u0 = *up;
            const auto u1 = *(up + 1);
            const auto u2 = *(up + 2);
            const auto wc1 = *vp1;
            const auto wc2 = *vp2;

            const auto x_val = (skip_type) ? (*xp) : (scalar_t)0.f;
            const auto g1 = sigmoidf(u1 + wc1 * cur + bias1);
            const auto g2 = sigmoidf(u2 + wc2 * cur + bias2);
            cur = (cur - u0) * g1 + u0;
            const auto val = calc_activation(activation_type, cur);
            *hp = skip_type ? ((val - x_val) * mask * g2 + x_val) : (val * mask * g2);
        } 
        *cp = cur;  // useful for backward
        up += ncols_u_;
        cp += ncols_;
        hp += ncols_;
        xp = skip_type ? (xp + ncols_x_) : NULL;
        pad_p = mask_pad ? (pad_p + batch_) : NULL;
        vp1 = is_custom ? (vp1 + ncols_ * 2) : vp1;
        vp2 = is_custom ? (vp2 + ncols_ * 2) : vp2;
    }
}

template <typename scalar_t>
__global__ void sru_cuda_bi_backward_kernel(
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
                           const bool* __restrict__ mask_pad,
                           const scalar_t* __restrict__ c,
                           const scalar_t* __restrict__ grad_h,
                           const scalar_t* __restrict__ grad_last,
                           const int len,
                           const int batch,
                           const int d,
                           const int k,
                           const int activation_type,
                           const int skip_type,
                           const int is_custom)
{
    assert ((skip_type >= 0) && (skip_type <= 2));
    assert ((skip_type != 1) || (k == 3));
    assert ((skip_type != 2) || (k == 4));

    int ncols = batch * d * 2;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int ncols_u = ncols * k;
    int ncols_x = (k == 3) ? ncols : ncols_u;
    const scalar_t mask = (mask_c == NULL) ? (scalar_t)1.f : (*(mask_c + col));
    scalar_t gwc1 = 0;
    scalar_t gwc2 = 0;
    scalar_t gbias1 = 0;
    scalar_t gbias2 = 0;
    auto cur = *(grad_last + col);
    const int d2 = d * 2;

    const auto* vp1 = is_custom ? (weight_c + col * 2) : (weight_c + (col % d2));
    const auto* vp2 = is_custom ? (weight_c + col * 2 + 1) : (weight_c + (col % d2) + d2);
    auto* gvp1 = is_custom ? (grad_wc + col * 2) : (grad_wc + col);
    auto* gvp2 = is_custom ? (grad_wc + col * 2 + 1) : (grad_wc + col + ncols);

    const auto bias1 = *(bias + (col % d2));
    const auto bias2 = *(bias + (col % d2) + d2);

    const auto *up = u + (col * k);
    const auto *xp = (skip_type == 0) ? NULL : (
        (skip_type == 1) ? (x + col) : (up + 3)
    );
    const auto *cp = c + col;
    const auto *ghp = grad_h + col;
    const bool *pad_p = (mask_pad == NULL) ? NULL : (mask_pad + (col / d2));
    auto *gup = grad_u + (col * k);
    auto *gxp = (skip_type == 0) ? NULL : (
        (skip_type == 1) ? (grad_x + col) : (gup + 3)
    );

    const bool flip = ((col % d2) >= d);
    if (!flip) {
        up += (len - 1) * ncols_u;
        cp += (len - 1) * ncols;
        ghp += (len - 1) * ncols;
        gup += (len - 1) * ncols_u;
        if (skip_type) {
            xp += (len - 1) * ncols_x;
            gxp += (len - 1) * ncols_x;
        }
        if (pad_p) pad_p += (len - 1) * batch;
        if (is_custom) {
            vp1 += (len - 1) * ncols * 2;
            vp2 += (len - 1) * ncols * 2;
            gvp1 += (len - 1) * ncols * 2;
            gvp2 += (len - 1) * ncols * 2;
        }
    }
    const int ncols_u_ = flip ? -ncols_u : ncols_u;
    const int ncols_x_ = flip ? -ncols_x : ncols_x;
    const int ncols_ = flip ? -ncols : ncols;
    const int batch_ = flip ? -batch : batch;

    for (int cnt = 0; cnt < len; ++cnt)
    {
        if ((pad_p == NULL) || !(*pad_p)) {
            const auto prev_c_val = (cnt < len - 1) ? (*(cp - ncols_)) : (*(init + col));
            const auto cp_val = *cp;
            const auto u0 = *up;
            const auto u1 = *(up + 1);
            const auto u2 = *(up + 2);
            const auto wc1 = *vp1;
            const auto wc2 = *vp2;

            const auto x_val = (skip_type) ? (*xp) : (scalar_t)0.f;
            const auto gh_val = *ghp;
            const auto g1 = sigmoidf(u1 + wc1 * prev_c_val + bias1);
            const auto g2 = sigmoidf(u2 + wc2 * prev_c_val + bias2);
            const auto c_val = calc_activation(activation_type, cp_val);

            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + u0*(1-g1) = (c'-u0)*g1 + u0

            // gradient with respect to values in the second gate g2
            const auto gg2 = gh_val * (c_val - x_val) * mask * (g2 * (1.f - g2));
            gbias2 += gg2;
            gwc2 += gg2 * prev_c_val;
            *gvp2 = gg2 * prev_c_val;

            // gradient with respect to c[t]
            const auto tmp = g2 * calc_grad_activation(activation_type, c_val);
            const auto gc = gh_val * mask * tmp + cur;

            // gradient with respect to values in the first gate g1
            const auto gg1 = gc * (prev_c_val - u0) * (g1 * (1.f - g1));
            gbias1 += gg1;
            gwc1 += gg1 * prev_c_val;
            *gvp1 = gg1 * prev_c_val;

            // gradient with respect to c[t-1]
            cur = gc * g1 + gg1 * wc1 + gg2 * wc2;

            // gradient with respect to U
            *gup = gc * (1.f - g1);
            *(gup + 1) = gg1;
            *(gup + 2) = gg2;

            // gradient with respect to x[t]
            if (skip_type)
                *gxp = gh_val * (1.f - g2 * mask);
        }

        up -= ncols_u_;
        cp -= ncols_;
        gup -= ncols_u_;
        ghp -= ncols_;
        xp = skip_type ? (xp - ncols_x_) : NULL;
        gxp = skip_type ? (gxp - ncols_x_) : NULL;
        pad_p = mask_pad ? (pad_p - batch_) : NULL;
        vp1 = is_custom ? (vp1 - ncols_ * 2) : vp1;
        vp2 = is_custom ? (vp2 - ncols_ * 2) : vp2;
        gvp1 = is_custom ? (gvp1 - ncols_ * 2) : gvp1;
        gvp2 = is_custom ? (gvp2 - ncols_ * 2) : gvp2;
    }

    if (!is_custom) {
        *(grad_wc + col) = gwc1;
        *(grad_wc + col + ncols) = gwc2;
    }
    *(grad_bias + col) = gbias1;
    *(grad_bias + col + ncols) = gbias2;
    *(grad_init +col) = cur;
}

} //  end of namespace

//  unidirectional forward()
void sru_cuda_forward_simple(
        at::Tensor & h,
        at::Tensor & c,
        const at::Tensor & U,
        const at::optional<at::Tensor> & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::optional<at::Tensor> & mask_c,
        const at::optional<at::Tensor> & mask_pad,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size) {

    const int threads = 512;
    const int total = batch_size * hidden_size;
    const dim3 blocks( (total - 1) / threads + 1 );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(U.type(), "sru_forward_cuda_simple", ([&] {
        sru_cuda_forward_kernel_simple<scalar_t><<<blocks, threads>>>(
            h.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            U.data_ptr<scalar_t>(),
            x.has_value() ? x.value().data_ptr<scalar_t>() : NULL,
            weight_c.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            c_init.data_ptr<scalar_t>(),
            mask_c.has_value() ? mask_c.value().data_ptr<scalar_t>() : NULL,
            mask_pad.has_value() ? mask_pad.value().data_ptr<bool>() : NULL,
            length,
            batch_size,
            hidden_size);
    }));
}

//  unidirectional backward()
void sru_cuda_backward_simple(
        at::Tensor & grad_u,
        at::Tensor & grad_x,
        at::Tensor & grad_wc,
        at::Tensor & grad_bias,
        at::Tensor & grad_init,
        const at::Tensor & U,
        const at::optional<at::Tensor> & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::optional<at::Tensor> & mask_c,
        const at::optional<at::Tensor> & mask_pad,
        const at::Tensor & c,
        const at::Tensor & grad_h,
        const at::Tensor & grad_last,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size) {

    const int threads = 512;
    const int total = batch_size * hidden_size;
    const dim3 blocks( (total - 1) / threads + 1 );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(U.type(), "sru_backward_cuda_simple", ([&] {
        sru_cuda_backward_kernel_simple<scalar_t><<<blocks, threads>>>(
            grad_u.data_ptr<scalar_t>(),
            x.has_value() ? grad_x.data_ptr<scalar_t>() : NULL,
            grad_wc.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            grad_init.data_ptr<scalar_t>(),
            U.data_ptr<scalar_t>(),
            x.has_value() ? x.value().data_ptr<scalar_t>() : NULL,
            weight_c.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            c_init.data_ptr<scalar_t>(),
            mask_c.has_value() ? mask_c.value().data_ptr<scalar_t>() : NULL,
            mask_pad.has_value() ? mask_pad.value().data_ptr<bool>() : NULL,
            c.data_ptr<scalar_t>(),
            grad_h.data_ptr<scalar_t>(),
            grad_last.data_ptr<scalar_t>(),
            length,
            batch_size,
            hidden_size);
    }));
}

//  bidirectional forward()
void sru_cuda_bi_forward_simple(
        at::Tensor & h,
        at::Tensor & c,
        const at::Tensor & U,
        const at::optional<at::Tensor> & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::optional<at::Tensor> & mask_c,
        const at::optional<at::Tensor> & mask_pad,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size) {

    const int threads = 512;
    const int total = batch_size * hidden_size * 2;
    const dim3 blocks( (total - 1) / threads + 1 );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(U.type(), "sru_bi_forward_cuda_simple", ([&] {
        sru_cuda_bi_forward_kernel_simple<scalar_t><<<blocks, threads>>>(
            h.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            U.data_ptr<scalar_t>(),
            x.has_value() ? x.value().data_ptr<scalar_t>() : NULL,
            weight_c.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            c_init.data_ptr<scalar_t>(),
            mask_c.has_value() ? mask_c.value().data_ptr<scalar_t>() : NULL,
            mask_pad.has_value() ? mask_pad.value().data_ptr<bool>() : NULL,
            length,
            batch_size,
            hidden_size);
    }));
}

//  bidirectional backward()
void sru_cuda_bi_backward_simple(
        at::Tensor & grad_u,
        at::Tensor & grad_x,
        at::Tensor & grad_wc,
        at::Tensor & grad_bias,
        at::Tensor & grad_init,
        const at::Tensor & U,
        const at::optional<at::Tensor> & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::optional<at::Tensor> & mask_c,
        const at::optional<at::Tensor> & mask_pad,
        const at::Tensor & c,
        const at::Tensor & grad_h,
        const at::Tensor & grad_last,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size) {

    const int threads = 512;
    const int total = batch_size * hidden_size * 2;
    const dim3 blocks( (total - 1) / threads + 1 );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(U.type(), "sru_bi_backward_cuda_simple", ([&] {
        sru_cuda_bi_backward_kernel_simple<scalar_t><<<blocks, threads>>>(
            grad_u.data_ptr<scalar_t>(),
            x.has_value() ? grad_x.data_ptr<scalar_t>() : NULL,
            grad_wc.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            grad_init.data_ptr<scalar_t>(),
            U.data_ptr<scalar_t>(),
            x.has_value() ? x.value().data_ptr<scalar_t>() : NULL,
            weight_c.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            c_init.data_ptr<scalar_t>(),
            mask_c.has_value() ? mask_c.value().data_ptr<scalar_t>() : NULL,
            mask_pad.has_value() ? mask_pad.value().data_ptr<bool>() : NULL,
            c.data_ptr<scalar_t>(),
            grad_h.data_ptr<scalar_t>(),
            grad_last.data_ptr<scalar_t>(),
            length,
            batch_size,
            hidden_size);
    }));
}

//  unidirectional forward()
void sru_cuda_forward(
        at::Tensor & h,
        at::Tensor & c,
        const at::Tensor & U,
        const at::optional<at::Tensor> & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::optional<at::Tensor> & mask_c,
        const at::optional<at::Tensor> & mask_pad,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type,
        const int64_t is_custom) {

    const int threads = 512;
    const int total = batch_size * hidden_size;
    const dim3 blocks( (total - 1) / threads + 1 );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(U.type(), "sru_forward_cuda", ([&] {
        sru_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            h.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            U.data_ptr<scalar_t>(),
            x.has_value() ? x.value().data_ptr<scalar_t>() : NULL,
            weight_c.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            c_init.data_ptr<scalar_t>(),
            mask_c.has_value() ? mask_c.value().data_ptr<scalar_t>() : NULL,
            mask_pad.has_value() ? mask_pad.value().data_ptr<bool>() : NULL,
            length,
            batch_size,
            hidden_size,
            k,
            activation_type,
            skip_type,
            is_custom);
    }));
}

//  bidirectional forward()
void sru_cuda_bi_forward(
        at::Tensor & h,
        at::Tensor & c,
        const at::Tensor & U,
        const at::optional<at::Tensor> & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::optional<at::Tensor> & mask_c,
        const at::optional<at::Tensor> & mask_pad,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type,
        const int64_t is_custom) {

    const int threads = 512;
    const int total = batch_size * hidden_size * 2;
    const dim3 blocks( (total - 1) / threads + 1 );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(U.type(), "sru_bi_forward_cuda", ([&] {
        sru_cuda_bi_forward_kernel<scalar_t><<<blocks, threads>>>(
            h.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            U.data_ptr<scalar_t>(),
            x.has_value() ? x.value().data_ptr<scalar_t>() : NULL,
            weight_c.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            c_init.data_ptr<scalar_t>(),
            mask_c.has_value() ? mask_c.value().data_ptr<scalar_t>() : NULL,
            mask_pad.has_value() ? mask_pad.value().data_ptr<bool>() : NULL,
            length,
            batch_size,
            hidden_size,
            k,
            activation_type,
            skip_type,
            is_custom);
    }));
}

//  unidirectional backward()
void sru_cuda_backward(
        at::Tensor & grad_u,
        at::Tensor & grad_x,
        at::Tensor & grad_wc,
        at::Tensor & grad_bias,
        at::Tensor & grad_init,
        const at::Tensor & U,
        const at::optional<at::Tensor> & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::optional<at::Tensor> & mask_c,
        const at::optional<at::Tensor> & mask_pad,
        const at::Tensor & c,
        const at::Tensor & grad_h,
        const at::Tensor & grad_last,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type,
        const int64_t is_custom) {

    const int threads = 512;
    const int total = batch_size * hidden_size;
    const dim3 blocks( (total - 1) / threads + 1 );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(U.type(), "sru_backward_cuda", ([&] {
        sru_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_u.data_ptr<scalar_t>(),
            x.has_value() ? grad_x.data_ptr<scalar_t>() : NULL,
            grad_wc.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            grad_init.data_ptr<scalar_t>(),
            U.data_ptr<scalar_t>(),
            x.has_value() ? x.value().data_ptr<scalar_t>() : NULL,
            weight_c.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            c_init.data_ptr<scalar_t>(),
            mask_c.has_value() ? mask_c.value().data_ptr<scalar_t>() : NULL,
            mask_pad.has_value() ? mask_pad.value().data_ptr<bool>() : NULL,
            c.data_ptr<scalar_t>(),
            grad_h.data_ptr<scalar_t>(),
            grad_last.data_ptr<scalar_t>(),
            length,
            batch_size,
            hidden_size,
            k,
            activation_type,
            skip_type,
            is_custom);
    }));
}

//  bidirectional backward()
void sru_cuda_bi_backward(
        at::Tensor & grad_u,
        at::Tensor & grad_x,
        at::Tensor & grad_wc,
        at::Tensor & grad_bias,
        at::Tensor & grad_init,
        const at::Tensor & U,
        const at::optional<at::Tensor> & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::optional<at::Tensor> & mask_c,
        const at::optional<at::Tensor> & mask_pad,
        const at::Tensor & c,
        const at::Tensor & grad_h,
        const at::Tensor & grad_last,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type,
        const int64_t is_custom) {

    const int threads = 512;
    const int total = batch_size * hidden_size * 2;
    const dim3 blocks( (total - 1) / threads + 1 );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(U.type(), "sru_bi_backward_cuda", ([&] {
        sru_cuda_bi_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_u.data_ptr<scalar_t>(),
            x.has_value() ? grad_x.data_ptr<scalar_t>() : NULL,
            grad_wc.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            grad_init.data_ptr<scalar_t>(),
            U.data_ptr<scalar_t>(),
            x.has_value() ? x.value().data_ptr<scalar_t>() : NULL,
            weight_c.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            c_init.data_ptr<scalar_t>(),
            mask_c.has_value() ? mask_c.value().data_ptr<scalar_t>() : NULL,
            mask_pad.has_value() ? mask_pad.value().data_ptr<bool>() : NULL,
            c.data_ptr<scalar_t>(),
            grad_h.data_ptr<scalar_t>(),
            grad_last.data_ptr<scalar_t>(),
            length,
            batch_size,
            hidden_size,
            k,
            activation_type,
            skip_type,
            is_custom);
    }));
}

