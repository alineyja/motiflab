// cpp/src/conv1d_cuda.cpp
#include <torch/extension.h>
#include <vector>

void launch_conv1d_forward_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    int batch_size, int C, int L, int F, int K, int output_length
);

torch::Tensor conv1d_forward_cuda( 
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias
) {
    // смежные тензоры для обеспечения непрерывности данных в памяти
    input_ = input.cuda().contiguous();
    weight_ = weight.cuda().contiguous();
    bias_ = bias.cuda().contiguous();
    
    // измеряем размеры входных тензоров
    int batch_size = input_.size(0);
    int C = input_.size(1);
    int L = input_.size(2);
    int F = weight_.size(0);
    int K = weight_.size(2);
    int output_length = L - K + 1;

    auto output = torch::zeros({batch_size, F, output_length}, torch::dtype(torch::kFloat).device(input_.device())); // создаем выходной тензор с нулями

    // Launch CUDA kernel
    launch_conv1d_forward_kernel(
        input_.data_ptr<float>(), weight_.data_ptr<float>(), bias_.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, C, L, F, K, output_length
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_forward_cuda", &conv1d_forward_cuda, "Conv1D forward (CUDA)");
}