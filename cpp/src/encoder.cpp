// cpp/src/encoder.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;

// функция кодирования
py::array_t<float> encode_one_hot(const std::string& seq, const std::string& layout) {
    size_t L = seq.length();
    size_t C = 4; // Стандартный алфавит ACGT

    //размеры и шаги для NumPy массива
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    if (layout == "channel_last") {
        shape = {L, C};
        strides = {C * sizeof(float), sizeof(float)};
    } else if (layout == "channel_first") {
        shape = {C, L};
        strides = {L * sizeof(float), sizeof(float)};
    } else {
        throw std::invalid_argument("Layout must be 'channel_last' or 'channel_first'");
    }

    // память под NumPy массив (инициализируем нулями)
    auto result = py::array_t<float>(shape, strides);
    auto ptr = static_cast<float*>(result.request().ptr);
    std::fill(ptr, ptr + L * C, 0.0f);
    for (size_t i = 0; i < L; ++i) {
        char nt = seq[i];
        int c = -1;

        // быстрый switch-case по символам
        switch (nt) {
            case 'A': case 'a': c = 0; break;
            case 'C': case 'c': c = 1; break;
            case 'G': case 'g': c = 2; break;
            case 'T': case 't': c = 3; break;
            default:
                throw std::invalid_argument(
                    std::string("C++ Encoder Error: Invalid nucleotide encountered: ") + nt
                );
        }

        // единицу в нужную область памяти
        if (layout == "channel_last") {
            ptr[i * C + c] = 1.0f;
        } else {
            ptr[c * L + i] = 1.0f;
        }
    }

    return result;
}
//свертка Conv1D на CPU
py::array_t<float> conv1d_forward(
    py::array_t<float> X,     
    py::array_t<float> W,     
    py::array_t<float> bias    
) {
    auto buf_X = X.request();
    auto buf_W = W.request();
    auto buf_b = bias.request();

    if (buf_X.ndim != 3 || buf_W.ndim != 3 || buf_b.ndim != 1) {
        throw std::invalid_argument("C++ Conv1D Error: неправильные размеры входных данных.");
    }

    size_t B = buf_X.shape[0];
    size_t C = buf_X.shape[1]; 
    size_t L = buf_X.shape[2];

    size_t F = buf_W.shape[0];
    size_t K = buf_W.shape[2];

    if (C != 4 || buf_W.shape[1] != 4) {
        throw std::invalid_argument("C++ Conv1D Error: входные данные должны иметь 4 канала (ACGT).");
    }

    if (L < K) {
        throw std::invalid_argument("C++ Conv1D Error: длина последовательности L должна быть >= размера ядра K.");
    }

    size_t out_L = L - K + 1; //Valid padding

    // Создаем выходной массив формы (B, F, out_L)
    std::vector<size_t> out_shape = {B, F, out_L};
    std::vector<size_t> out_strides = {
        F * out_L * sizeof(float), 
        out_L * sizeof(float), 
        sizeof(float)
    };
    
    auto Y = py::array_t<float>(out_shape, out_strides);

    //сырые указатели на данные
    const float* ptr_X = static_cast<const float*>(buf_X.ptr);
    const float* ptr_W = static_cast<const float*>(buf_W.ptr);
    const float* ptr_b = static_cast<const float*>(buf_b.ptr);
    float* ptr_Y = static_cast<float*>(Y.request().ptr);

    //вычисления по батчам и фильтрам с помощью OpenMP
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < static_cast<int>(B); ++b) {
        for (int f = 0; f < static_cast<int>(F); ++f) {
            float b_val = ptr_b[f];
            
            for (size_t i = 0; i < out_L; ++i) {
                float sum = b_val;
                for (size_t c = 0; c < C; ++c) {
                    //разворот цикла по размеру ядра
                    for (size_t j = 0; j < K; ++j) {
                        float x_val = ptr_X[b * (C * L) + c * L + (i + j)];
                        float w_val = ptr_W[f * (C * K) + c * K + j];
                        sum += x_val * w_val;
                    }
                }
                //результат свертки
                ptr_Y[b * (F * out_L) + f * out_L + i] = sum;
            }
        }
    }

    return Y;
}


// Регистрация модуля в pybind11
PYBIND11_MODULE(motiflab_cpp, m) {
    m.doc() = "High-performance C++ core MotifLab with OpenMP";
    
    m.def(
        "encode_one_hot", 
        &encode_one_hot, 
        "Fast C++ One-Hot Encoder",
        py::arg("seq"), 
        py::arg("layout") = "channel_first"
    );
    
    m.def(
        "conv1d_forward",
        &conv1d_forward,
        "Parallel CPU Conv1D Forward Pass",
        py::arg("X"),
        py::arg("W"),
        py::arg("bias")
    );
}