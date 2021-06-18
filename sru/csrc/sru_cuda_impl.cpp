#include <torch/script.h>
#include <vector>

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
        const int64_t hidden_size);
//  U: the result of grouped multiplication
//  The size of U is [length, batch_size, hidden_size, 3]

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
        const int64_t hidden_size);

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
        const int64_t hidden_size);

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
        const int64_t hidden_size);

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
        const int64_t is_custom);
//  k: the number of sub-matrices in grouped multiplication
//  U: the result of grouped multiplication
//  The size of U is [length, batch_size, hidden_size, k]

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
        const int64_t is_custom);

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
        const int64_t is_custom);

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
        const int64_t is_custom);

/* Implementation starts here */

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//  unidirectional forward()
std::vector<at::Tensor> sru_forward_simple(
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

    at::Tensor h = at::zeros({length, batch_size, hidden_size}, U.options());
    at::Tensor c = at::zeros({length, batch_size, hidden_size}, U.options());

    sru_cuda_forward_simple(
        h,
        c,
        U,
        x,
        weight_c,
        bias,
        c_init,
        mask_c,
        mask_pad,
        length,
        batch_size,
        hidden_size);

    return {h, c};
}

//  unidirectional backward()
std::vector<at::Tensor> sru_backward_simple(
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

    at::Tensor grad_u = at::zeros_like(U);
    at::Tensor grad_x = x.has_value() ? at::zeros_like(x.value()) : at::zeros({1});
    at::Tensor grad_wc = at::zeros({2, batch_size, hidden_size}, U.options());
    at::Tensor grad_bias = at::zeros({2, batch_size, hidden_size}, U.options());
    at::Tensor grad_init = at::zeros({batch_size, hidden_size}, U.options());

    sru_cuda_backward_simple(
        grad_u,
        grad_x,
        grad_wc,
        grad_bias,
        grad_init,
        U,
        x,
        weight_c,
        bias,
        c_init,
        mask_c,
        mask_pad,
        c,
        grad_h,
        grad_last,
        length,
        batch_size,
        hidden_size);

    return {grad_u, grad_x, grad_wc, grad_bias, grad_init};
}

//  bidirectional forward()
std::vector<at::Tensor> sru_bi_forward_simple(
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

    at::Tensor h = at::zeros({length, batch_size, hidden_size * 2}, U.options());
    at::Tensor c = at::zeros({length, batch_size, hidden_size * 2}, U.options());

    sru_cuda_bi_forward_simple(
        h,
        c,
        U,
        x,
        weight_c,
        bias,
        c_init,
        mask_c,
        mask_pad,
        length,
        batch_size,
        hidden_size);
    
    return {h, c};
}

//  bidirectional backward()
std::vector<at::Tensor> sru_bi_backward_simple(
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

    at::Tensor grad_u = at::zeros_like(U);
    at::Tensor grad_x = x.has_value() ? at::zeros_like(x.value()) : at::zeros({1});
    at::Tensor grad_wc = at::zeros({2, batch_size, hidden_size * 2}, U.options());
    at::Tensor grad_bias = at::zeros({2, batch_size, hidden_size * 2}, U.options());
    at::Tensor grad_init = at::zeros({batch_size, hidden_size * 2}, U.options());

    sru_cuda_bi_backward_simple(
        grad_u,
        grad_x,
        grad_wc,
        grad_bias,
        grad_init,
        U,
        x,
        weight_c,
        bias,
        c_init,
        mask_c,
        mask_pad,
        c,
        grad_h,
        grad_last,
        length,
        batch_size,
        hidden_size);

    return {grad_u, grad_x, grad_wc, grad_bias, grad_init};
}

//  unidirectional forward()
std::vector<at::Tensor> sru_forward(
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

    at::Tensor h = at::zeros({length, batch_size, hidden_size}, U.options());
    at::Tensor c = at::zeros({length, batch_size, hidden_size}, U.options());

    sru_cuda_forward(
        h,
        c,
        U,
        x,
        weight_c,
        bias,
        c_init,
        mask_c,
        mask_pad,
        length,
        batch_size,
        hidden_size,
        k,
        activation_type,
        skip_type,
        is_custom);

    return {h, c};
}

//  bidirectional forward()
std::vector<at::Tensor> sru_bi_forward(
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

    at::Tensor h = at::zeros({length, batch_size, hidden_size * 2}, U.options());
    at::Tensor c = at::zeros({length, batch_size, hidden_size * 2}, U.options());

    sru_cuda_bi_forward(
        h,
        c,
        U,
        x,
        weight_c,
        bias,
        c_init,
        mask_c,
        mask_pad,
        length,
        batch_size,
        hidden_size,
        k,
        activation_type,
        skip_type,
        is_custom);

    return {h, c};
}

//  unidirectional backward()
std::vector<at::Tensor> sru_backward(
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

    at::Tensor grad_u = at::zeros_like(U);
    at::Tensor grad_x = x.has_value() ? at::zeros_like(x.value()) : at::zeros({1});
    at::Tensor grad_wc = is_custom ? at::zeros_like(weight_c) : at::zeros({2, batch_size, hidden_size}, U.options());
    at::Tensor grad_bias = at::zeros({2, batch_size, hidden_size}, U.options());
    at::Tensor grad_init = at::zeros({batch_size, hidden_size}, U.options());

    sru_cuda_backward(
        grad_u,
        grad_x,
        grad_wc,
        grad_bias,
        grad_init,
        U,
        x,
        weight_c,
        bias,
        c_init,
        mask_c,
        mask_pad,
        c,
        grad_h,
        grad_last,
        length,
        batch_size,
        hidden_size,
        k,
        activation_type,
        skip_type,
        is_custom);

    return {grad_u, grad_x, grad_wc, grad_bias, grad_init};
}

//  bidirectional backward()
std::vector<at::Tensor> sru_bi_backward(
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

    at::Tensor grad_u = at::zeros_like(U);
    at::Tensor grad_x = x.has_value() ? at::zeros_like(x.value()) : at::zeros({1});
    at::Tensor grad_wc = is_custom ? at::zeros_like(weight_c) : at::zeros({2, batch_size, hidden_size * 2}, U.options());
    at::Tensor grad_bias = at::zeros({2, batch_size, hidden_size * 2}, U.options());
    at::Tensor grad_init = at::zeros({batch_size, hidden_size * 2}, U.options());

    sru_cuda_bi_backward(
        grad_u,
        grad_x,
        grad_wc,
        grad_bias,
        grad_init,
        U,
        x,
        weight_c,
        bias,
        c_init,
        mask_c,
        mask_pad,
        c,
        grad_h,
        grad_last,
        length,
        batch_size,
        hidden_size,
        k,
        activation_type,
        skip_type,
        is_custom);

    return {grad_u, grad_x, grad_wc, grad_bias, grad_init};
}

// This way of registing custom op is based on earlier PRs of Pytorch:
// https://github.com/pytorch/pytorch/pull/28229
// 
// In Pytorch 1.6, the recommended way is to use TORCH_LIBRARY(), e.g.
//
//   TORCH_LIBRARY(sru_cpu, m) {
//       m.def("cpu_forward", &cpu_forward);
//       m.def("cpu_bi_forward", &cpu_bi_forward);
//   }
//
// We choose this way for backward compatibility.
static auto registory1 = 
    torch::RegisterOperators("sru_cuda::sru_forward_simple", &sru_forward_simple);
static auto registory2 = 
    torch::RegisterOperators("sru_cuda::sru_backward_simple", &sru_backward_simple);
static auto registory3 = 
    torch::RegisterOperators("sru_cuda::sru_bi_forward_simple", &sru_bi_forward_simple);
static auto registory4 = 
    torch::RegisterOperators("sru_cuda::sru_bi_backward_simple", &sru_bi_backward_simple);
static auto registory5 = 
    torch::RegisterOperators("sru_cuda::sru_forward", &sru_forward);
static auto registory6 = 
    torch::RegisterOperators("sru_cuda::sru_backward", &sru_backward);
static auto registory7 = 
    torch::RegisterOperators("sru_cuda::sru_bi_forward", &sru_bi_forward);
static auto registory8 = 
    torch::RegisterOperators("sru_cuda::sru_bi_backward", &sru_bi_backward);