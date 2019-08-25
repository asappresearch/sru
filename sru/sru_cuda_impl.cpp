
#include <torch/torch.h>
#include <vector>

//  unidirectional forward()
void sru_cuda_forward(
        at::Tensor & h,
        at::Tensor & c,
        const at::Tensor & U,
        const at::Tensor & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::Tensor & mask_c,
        const at::Tensor & mask_pad,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type);
//  k: the number of sub-matrices in grouped multiplication
//  U: the result of grouped multiplication
//  The size of U is [length, batch_size, hidden_size, k]

//  bidirectional forward()
void sru_cuda_bi_forward(
        at::Tensor & h,
        at::Tensor & c,
        const at::Tensor & U,
        const at::Tensor & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::Tensor & mask_c,
        const at::Tensor & mask_pad,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type);

//  unidirectional backward()
void sru_cuda_backward(
        at::Tensor & grad_u,
        at::Tensor & grad_x,
        at::Tensor & grad_wc,
        at::Tensor & grad_bias,
        at::Tensor & grad_init,
        const at::Tensor & U,
        const at::Tensor & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::Tensor & mask_c,
        const at::Tensor & mask_pad,
        const at::Tensor & c,
        const at::Tensor & grad_h,
        const at::Tensor & grad_last,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type);

//  bidirectional backward()
void sru_cuda_bi_backward(
        at::Tensor & grad_u,
        at::Tensor & grad_x,
        at::Tensor & grad_wc,
        at::Tensor & grad_bias,
        at::Tensor & grad_init,
        const at::Tensor & U,
        const at::Tensor & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::Tensor & mask_c,
        const at::Tensor & mask_pad,
        const at::Tensor & c,
        const at::Tensor & grad_h,
        const at::Tensor & grad_last,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type);


/* Implementation starts here */

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//  unidirectional forward()
void sru_forward(
        const at::Tensor & U,
        const at::Tensor & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::Tensor & mask_c,
        const at::Tensor & mask_pad,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type) {

    sru_cuda_forward(
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
        skip_type);
}

//  bidirectional forward()
void sru_bi_forward(
        const at::Tensor & U,
        const at::Tensor & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::Tensor & mask_c,
        const at::Tensor & mask_pad,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type) {

    sru_cuda_bi_forward(
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
        skip_type);
}

//  unidirectional backward()
void sru_backward(
        at::Tensor & grad_u,
        at::Tensor & grad_x,
        at::Tensor & grad_wc,
        at::Tensor & grad_bias,
        at::Tensor & grad_init,
        const at::Tensor & U,
        const at::Tensor & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::Tensor & mask_c,
        const at::Tensor & mask_pad,
        const at::Tensor & c,
        const at::Tensor & grad_h,
        const at::Tensor & grad_last,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type) {

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
        skip_type);
}

//  bidirectional backward()
void sru_bi_backward(
        at::Tensor & grad_u,
        at::Tensor & grad_x,
        at::Tensor & grad_wc,
        at::Tensor & grad_bias,
        at::Tensor & grad_init,
        const at::Tensor & U,
        const at::Tensor & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::Tensor & mask_c,
        const at::Tensor & mask_pad,
        const at::Tensor & c,
        const at::Tensor & grad_h,
        const at::Tensor & grad_last,
        const int64_t length, 
        const int64_t batch_size, 
        const int64_t hidden_size, 
        const int64_t k,
        const int64_t activation_type,
        const int64_t skip_type) {

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
        skip_type);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sru_forward", &sru_cuda_forward, "SRU forward (CUDA version)");
  m.def("sru_bi_forward", &sru_cuda_bi_forward, "SRU bidirectional forward (CUDA version)");
  m.def("sru_backward", &sru_cuda_backward, "SRU backward (CUDA version)");
  m.def("sru_bi_backward", &sru_cuda_bi_backward, "SRU bidirectional backward (CUDA version)");
}
