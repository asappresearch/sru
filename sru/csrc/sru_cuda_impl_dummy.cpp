#include <torch/script.h>
#include <vector>

/* Implementation starts here */

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

    throw "Failed to load SRU recurrence operators for GPU";
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

    throw "Failed to load SRU recurrence operators for GPU";
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

    throw "Failed to load SRU recurrence operators for GPU";
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

    throw "Failed to load SRU recurrence operators for GPU";
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

    throw "Failed to load SRU recurrence operators for GPU";
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

    throw "Failed to load SRU recurrence operators for GPU";
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

    throw "Failed to load SRU recurrence operators for GPU";
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

    throw "Failed to load SRU recurrence operators for GPU";
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
static auto registory_fwd_v1 = 
    torch::RegisterOperators("sru_cuda::sru_forward_simple", &sru_forward_simple);
static auto registory_bwd_v1 = 
    torch::RegisterOperators("sru_cuda::sru_backward_simple", &sru_backward_simple);
static auto registory_bi_fwd_v1 = 
    torch::RegisterOperators("sru_cuda::sru_bi_forward_simple", &sru_bi_forward_simple);
static auto registory_bi_bwd_v1 = 
    torch::RegisterOperators("sru_cuda::sru_bi_backward_simple", &sru_bi_backward_simple);
static auto registory_fwd_v2 = 
    torch::RegisterOperators("sru_cuda::sru_forward", &sru_forward);
static auto registory_bwd_v2 = 
    torch::RegisterOperators("sru_cuda::sru_backward", &sru_backward);
static auto registory_bi_fwd_v2 = 
    torch::RegisterOperators("sru_cuda::sru_bi_forward", &sru_bi_forward);
static auto registory_bi_bwd_v2 = 
    torch::RegisterOperators("sru_cuda::sru_bi_backward", &sru_bi_backward);
