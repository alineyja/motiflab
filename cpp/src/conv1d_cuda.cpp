// cpp/src/conv1d_cuda.cpp

#include <torch/extension.h>
#include <vector>
void launch_conv1d_forward_kernel(
    const float* X, const float* W, const float* bias, float* output,
    int B, int C, int L, int F, int K, int out_L
);
torch::Tensor conv1d_cuda_forward(
    torch::Tensor X,      // (B, C, L)
    torch::Tensor W,      // (F, C, K)
    torch::Tensor bias    // (F)
) {
    // смежные тензоры для обеспечения непрерывности данных в памяти
    auto input_contiguous = X.cuda().contiguous();
    auto W_contiguous = W.cuda().contiguous();
    auto bias_contiguous = bias.cuda().contiguous();
    // измеряем размеры входных тензоров
    int B = input_contiguous.size(0);
    int C = input_contiguous.size(1);
    int L = input_contiguous.size(2);
    int F = W_contiguous.size(0);
    int K = W_contiguous.size(2);

    int out_L = L - K + 1;

    // Безопасное создание тензора, наследуя все свойства (CUDA, float32) от input_contiguous
    auto output = torch::zeros({B, F, out_L}, input_contiguous.options());

    
    launch_conv1d_forward_kernel(
        input_contiguous.data_ptr<float>(),
        W_contiguous.data_ptr<float>(),
        bias_contiguous.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, L, F, K, out_L
    );

    return output;
}

// Регистрация модуля для PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv1d_cuda_forward, "Conv1D Forward (CUDA)");
}