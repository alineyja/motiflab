// cpp/src/conv1d_kernel.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv1d_forward_kernel(
    const float* __restrict__ input, // входной тензор размерности (B, C, L)
    const float* __restrict__ weight, // весовой тензор размерности (F, C, K)
    const float* __restrict__ bias,  // смещение размерности (F)
    float* __restrict__ output, // выходной тензор размерности (B, F, output_length)
    int B, int C, int L, int K, int F, int output_length
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // вычисляем глобальный индекс потока 
    int total_threads = B * F * output_length; // общее количество потоков, необходимых для обработки всех элементов выходного тензора
    if (idx >= total_threads) {return;} // если индекс потока превышает общее количество элементов, выходим из функции

    int i = idx % output_length; // индекс элемента в выходной длине
    int temp = idx / output_length; // временная переменная для вычисления индекса фильтра
    int f = temp % F; // индекс фильтра
    int b = temp / F; // индекс батча

    float sum = bias[f]; // инициализируем сумму смещением для текущего фильтра

    for(int c = 0; c < C; ++c){
        for(int j = 0; j < K; ++j){
            int x_idx = b*C*L + c*L + (i+j); // вычисляем индекс элемента входного тензора
            int w_idx = f*C*K + c*K + j; // вычисляем индекс элемента весового тензора
            sum += input[x_idx] * weight[w_idx]; // суммируем произведение входного элемента и соответствующего весового элемента
        }
    }

    output[idx] = sum; // записываем результат в выходной тензор
}

void launch_conv1d_forward_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    int B, int C, int L, int K, int F, int output_length
){
    int total_threads = B * F * output_length; // общее количество потоков
    int threads_per_block = 256; // количество потоков в блоке
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block; // вычисляем количество блоков

    conv1d_forward_kernel<<<num_blocks, threads_per_block>>>(input, weight, bias, output, B, C, L, K, F, output_length); // запускаем ядро CUDA
}