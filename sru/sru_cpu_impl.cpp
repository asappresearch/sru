
#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> cpu_forward(
        const at::Tensor & U,
        const at::Tensor & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::Tensor & mask_pad,
        int64_t length, 
        int64_t batch_size, 
        int64_t d, 
        int64_t k,
        int64_t activation_type,
        bool has_skip_term, 
        double scale_x);

std::vector<at::Tensor> cpu_bi_forward(
        const at::Tensor & U,
        const at::Tensor & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::Tensor & mask_pad,
        int64_t length, 
        int64_t batch_size, 
        int64_t d, 
        int64_t k,
        int64_t activation_type,
        bool has_skip_term, 
        double scale_x);

float sigmoidf(float x);
float reluf(float x);
float seluf(float x);
float apply_activation(int64_t type, float x);


/* Implementation starts here */

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) AT_ASSERTM(x.type().scalarType() == at::ScalarType::Float, #x " must be float")


std::vector<at::Tensor> cpu_forward(
        const at::Tensor & U,
        const at::Tensor & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::Tensor & mask_pad,
        int64_t length, 
        int64_t batch_size, 
        int64_t d, 
        int64_t k,
        int64_t activation_type,
        bool has_skip_term, 
        double scale_x) {
    
    CHECK_FLOAT(U);
    CHECK_CONTIGUOUS(U);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(weight_c);
    CHECK_CONTIGUOUS(bias);
    CHECK_CONTIGUOUS(c_init);
    CHECK_CONTIGUOUS(mask_pad);
    
    const int ncols = batch_size * d;
    const int ncols_u = batch_size * d * k;

    // pointers to parameters
    const float* forget_w_ptr = weight_c.data<float>();
    const float* reset_w_ptr = forget_w_ptr + d;
    const float* forget_b_ptr = bias.data<float>();
    const float* reset_b_ptr = forget_b_ptr + d;
    const float* U_ptr = U.data<float>();
    const float* x_ptr = x.data<float>();
    const float* pad_ptr = (mask_pad.numel() == 0) ? NULL : 
                                    mask_pad.data<float>();

    auto h = at::zeros({length, batch_size, d}, U.options());
    auto c = c_init.clone();
    auto h_ptr = h.data<float>();
    auto c_ptr = c.data<float>();

    for (int l = 0; l < length; l++) {
        for (int i = 0; i < batch_size; i++) {
            // skip pad tokens
            if (pad_ptr && (*(pad_ptr + l*batch_size))) continue;
            for (int j = 0; j < d; j++) {

                // U[l,i,j,*]
                const float* u = U_ptr + l*ncols_u + (i*d+j)*k;
                const float u0 = *(u);
                const float u1 = *(u+1);
                const float u2 = *(u+2);

                const float fw = *(forget_w_ptr + j);
                const float rw = *(reset_w_ptr + j);
                const float fb = *(forget_b_ptr + j);
                const float rb = *(reset_b_ptr + j);

                const float prev_c = *(c_ptr + (i*d+j));

                // forget gate
                const float fg = sigmoidf(u1 + fw*prev_c + fb);
                // reset gate
                const float rg = sigmoidf(u2 + rw*prev_c + rb);

                const float c_val = prev_c * fg + u0 * (1 - fg);
                const float gc_val = apply_activation(activation_type, c_val);
                
                const float x_val = has_skip_term ? ((k == 4) ? *(u + 3) : (*(x_ptr + l*ncols + (i*d+j)))*scale_x) : 0;
                const float h_val = gc_val * rg + x_val * (1 - rg);

                *(h_ptr + l*ncols + (i*d+j)) = h_val;
                *(c_ptr + (i*d+j)) = c_val; 
            }
        }
    }
    return {h, c};
}

std::vector<at::Tensor> cpu_bi_forward(
        const at::Tensor & U,
        const at::Tensor & x,
        const at::Tensor & weight_c,
        const at::Tensor & bias,
        const at::Tensor & c_init,
        const at::Tensor & mask_pad,
        int64_t length, 
        int64_t batch_size, 
        int64_t d, 
        int64_t k,
        int64_t activation_type,
        bool has_skip_term, 
        double scale_x) {
    
    CHECK_FLOAT(U);
    CHECK_CONTIGUOUS(U);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(weight_c);
    CHECK_CONTIGUOUS(bias);
    CHECK_CONTIGUOUS(c_init);
    CHECK_CONTIGUOUS(mask_pad);
    
    const int ncols = batch_size * d * 2;
    const int ncols_u = batch_size * d * 2 * k;
    const int d2 = d * 2;

    // pointers to parameters
    const float* forget_w_ptr = weight_c.data<float>();
    const float* reset_w_ptr = forget_w_ptr + d*2;
    const float* forget_b_ptr = bias.data<float>();
    const float* reset_b_ptr = forget_b_ptr + d*2;
    const float* U_ptr = U.data<float>();
    const float* x_ptr = x.data<float>();
    //const float* cinit_ptr = c_init.data<float>();
    const float* pad_ptr = (mask_pad.numel() == 0) ? NULL : 
                                    mask_pad.data<float>();

    auto h = at::zeros({length, batch_size, d*2}, U.options());
    auto c = c_init.clone();
    //auto c = at::zeros({batch_size, d*2}, U.options());
    auto h_ptr = h.data<float>();
    auto c_ptr = c.data<float>();
    for (int l = 0; l < length; l++) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < d2; j++) {
                // skip pad tokens
                int l_ = (j < d) ? l : (length - l - 1);
                if (pad_ptr && (*(pad_ptr + l_*batch_size))) continue;

                // U[l_,i,j,*]
                const float* u = U_ptr + l_*ncols_u + (i*d2+j)*k;
                const float u0 = *(u);
                const float u1 = *(u+1);
                const float u2 = *(u+2);

                const float fw = *(forget_w_ptr + j);
                const float rw = *(reset_w_ptr + j);
                const float fb = *(forget_b_ptr + j);
                const float rb = *(reset_b_ptr + j);

                const float prev_c = *(c_ptr + (i*d2+j));

                // forget gate
                const float fg = sigmoidf(u1 + fw*prev_c + fb);
                // reset gate
                const float rg = sigmoidf(u2 + rw*prev_c + rb);

                const float c_val = prev_c * fg + u0 * (1 - fg);
                const float gc_val = apply_activation(activation_type, c_val);
                
                const float x_val = has_skip_term ? ((k == 4) ? *(u + 3) : (*(x_ptr + l_*ncols + (i*d2+j)))*scale_x) : 0;
                const float h_val = gc_val * rg + x_val * (1 - rg);

                *(h_ptr + l_*ncols + (i*d2+j)) = h_val;
                *(c_ptr + (i*d2+j)) = c_val; 
            }
        }
    }
    return {h, c};
}


inline float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}

inline float reluf(float x) {
    return (x > 0.f) ? x : 0.f;
}

inline float seluf(float x) {
    return 1.0507009873554804934193349852946f * (
        (x > 0.f) ? x : 1.6732632423543772848170429916717f * (expf(x)-1.f)
    );
}

inline float apply_activation(int64_t type, float x) {
    switch (type) {
        case 0:
            return x;
        case 1:
            return tanhf(x);
        case 2:
            return reluf(x);
        case 3:
            return seluf(x);
    }
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cpu_forward", &cpu_forward, "SRU forward (CPU version)");
  m.def("cpu_bi_forward", &cpu_bi_forward, "SRU bidirectional forward (CPU version)");
}
