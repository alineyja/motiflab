# setup_cuda.py
import os
import sys
import shutil
from pathlib import Path

# Добавляем +PTX, чтобы драйвер 5070 Ti смог JIT-скомпилировать ядро при запуске
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
os.environ["DISTUTILS_USE_SDK"] = "1"

if "CUDA_HOME" not in os.environ:
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        os.environ["CUDA_HOME"] = str(Path(nvcc_path).resolve().parent.parent)
    elif "CUDA_PATH" in os.environ:
        os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]
    else:
        default_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
        if os.path.exists(default_path):
            os.environ["CUDA_HOME"] = default_path

try:
    import torch.utils.cpp_extension as ce
    if "CUDA_HOME" in os.environ: ce.CUDA_HOME = os.environ["CUDA_HOME"]
except ImportError:
    exit(1)

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="motiflab_cuda",
    ext_modules=[
        CUDAExtension(
            name="motiflab_cuda",
            sources=["cpp/src/conv1d_cuda.cpp", "cpp/src/conv1d_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)