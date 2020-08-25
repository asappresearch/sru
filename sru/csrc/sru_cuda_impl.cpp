#include <torch/extension.h>
#include <vector>

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
        const int64_t skip_type,
        const int64_t is_custom);
//  k: the number of sub-matrices in grouped multiplication
//  U: the result of grouped multiplication
//  The size of U is [length, batch_size, hidden_size, k]

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
        const int64_t skip_type,
        const int64_t is_custom);

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
        const int64_t skip_type,
        const int64_t is_custom);

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
        const int64_t skip_type,
        const int64_t is_custom);

/* Implementation starts here */

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//  unidirectional forward()
void sru_forward(
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
        const int64_t skip_type,
        const int64_t is_custom) {

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
}

//  bidirectional forward()
void sru_bi_forward(
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
        const int64_t skip_type,
        const int64_t is_custom) {

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
}

//  unidirectional backward()
void sru_backward(
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
        const int64_t skip_type,
        const int64_t is_custom) {

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
}

//  bidirectional backward()
void sru_bi_backward(
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
        const int64_t skip_type,
        const int64_t is_custom) {

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
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sru_forward", &sru_forward, "SRU forward (CUDA version)");
  m.def("sru_bi_forward", &sru_bi_forward, "SRU bidirectional forward (CUDA version)");
  m.def("sru_backward", &sru_backward, "SRU backward (CUDA version)");
  m.def("sru_bi_backward", &sru_bi_backward, "SRU bidirectional backward (CUDA version)");
}
