# benchmarks/benchmark_conv1d.py
import sys
from pathlib import Path
# Добавляем пути
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))
import time
import torch
import torch.nn as nn
import numpy as np
import motiflab_cpp
try:
    import motiflab_cuda
except ImportError:
    print("Warning: 'motiflab_cuda' module not found.")
    motiflab_cuda = None


def main():
    B = 64
    C = 4
    L = 200
    F = 32
    K = 15
    num_runs = 100

    print(f"=== CONV1D FORWARD BENCHMARK ===")
    print(f"Batch Size: {B}, Channels: {C}, Length: {L}, Filters: {F}, Kernel Size: {K}")
    print(f"Running {num_runs} iterations for each engine...\n")

    #Генерируем случайные входные данные
    X_np = np.random.randn(B, C, L).astype(np.float32)
    W_np = np.random.randn(F, C, K).astype(np.float32)
    b_np = np.random.randn(F).astype(np.float32)

    X_torch = torch.from_numpy(X_np)
    W_torch = torch.from_numpy(W_np)
    b_torch = torch.from_numpy(b_np)

    #Настройка PyTorch CPU
    py_conv_cpu = nn.Conv1d(C, F, K)
    with torch.no_grad():
        py_conv_cpu.weight.copy_(W_torch)
        py_conv_cpu.bias.copy_(b_torch)

    #Настройка PyTorch GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        py_conv_gpu = nn.Conv1d(C, F, K).to(device)
        X_gpu = X_torch.to(device)
        with torch.no_grad():
            py_conv_gpu.weight.copy_(W_torch.to(device))
            py_conv_gpu.bias.copy_(b_torch.to(device))

    # pyTorch CPU
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            out_py_cpu = py_conv_cpu(X_torch)
    t_py_cpu = (time.perf_counter() - t_start) / num_runs
    # мой cpu
    t_start = time.perf_counter()
    for _ in range(num_runs):
        out_cpp_cpu = motiflab_cpp.conv1d_forward(X_np, W_np, b_np)
    t_cpp_cpu = (time.perf_counter() - t_start) / num_runs

   # запуск на GPU с использованием PyTorch
    t_py_gpu = 0.0
    pytorch_gpu_ok = False
    if torch.cuda.is_available():
        try:
            # Разогрев (Warm-up) GPU
            with torch.no_grad():
                for _ in range(10):
                    _ = py_conv_gpu(X_gpu)
            torch.cuda.synchronize()
            
            t_start = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_runs):
                    out_py_gpu = py_conv_gpu(X_gpu)
                torch.cuda.synchronize()
            t_py_gpu = (time.perf_counter() - t_start) / num_runs
            pytorch_gpu_ok = True
        except Exception as e:
            print("[Warning] Native PyTorch GPU failed (normal for Blackwell sm_120 on cu124 wheels).")
            print(f"Detail: {e}\n")

   # кастоммный cuda ядро
    t_cpp_cuda = 0.0
    custom_cuda_ok = False
    if motiflab_cuda is not None and torch.cuda.is_available() and pytorch_gpu_ok:
        try:
            # Извлекаем GPUвеса из PyTorch
            W_gpu = py_conv_gpu.weight.detach()
            b_gpu = py_conv_gpu.bias.detach()
            
            # Разогрев
            for _ in range(10):
                out_cpp_cuda = motiflab_cuda.forward(X_gpu, W_gpu, b_gpu)
            torch.cuda.synchronize()

            t_start = time.perf_counter()
            for _ in range(num_runs):
                out_cpp_cuda = motiflab_cuda.forward(X_gpu, W_gpu, b_gpu)
            torch.cuda.synchronize()
            t_cpp_cuda = (time.perf_counter() - t_start) / num_runs
            custom_cuda_ok = True
        except Exception as e:
            print(f"[Error] Custom CUDA Kernel failed: {e}\n")

    print("CORRECTNESS CHECKS")
    diff_cpp_cpu = np.abs(out_py_cpu.numpy() - out_cpp_cpu).max()
    print(f"Max absolute diff (PyTorch CPU vs Custom C++ CPU): {diff_cpp_cpu:.2e}")
    if custom_cuda_ok:
        diff_cuda = torch.abs(out_py_cpu - out_cpp_cuda.cpu()).max().item()
        print(f"Max absolute diff (PyTorch CPU vs Custom CUDA):    {diff_cuda:.2e}")
        
    if pytorch_gpu_ok and custom_cuda_ok:
        diff_cuda_gpu = torch.abs(out_py_gpu.cpu() - out_cpp_cuda.cpu()).max().item()
        print(f"Max absolute diff (PyTorch GPU vs Custom CUDA):    {diff_cuda_gpu:.2e}")
    print("-" * 26 + "\n")
    print("--- TIMING RESULTS (Average of 100 runs) ---")
    print(f"1. PyTorch CPU:             {t_py_cpu * 1000:.3f} ms")
    print(f"2. Custom C++ CPU (OpenMP): {t_cpp_cpu * 1000:.3f} ms")
    
    if pytorch_gpu_ok:
        print(f"3. PyTorch GPU (CUDA):      {t_py_gpu * 1000:.3f} ms")
    else:
        print("3. PyTorch GPU (CUDA):      [Skipped/Failed]")
        
    if custom_cuda_ok:
        print(f"4. Custom CUDA Kernel:      {t_cpp_cuda * 1000:.3f} ms")
    else:
        print("4. Custom CUDA Kernel:      [Failed]")


if __name__ == "__main__":
    main()