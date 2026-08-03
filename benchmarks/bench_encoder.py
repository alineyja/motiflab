# benchmarks/benchmark_encoder.py
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src")) 
sys.path.append(str(root_dir))         
import time
import random
import numpy as np
from motiflab.bioinformatics.alphabet import DNAAlphabet
from motiflab.bioinformatics.sequence import DNASequence
from motiflab.preprocessing.encoder import OneHotEncoder
try:
    import motiflab_cpp
except ImportError:
    print("Error: 'motiflab_cpp' module not found.")
    print("Please compile it first by running: python setup.py build_ext --inplace")
    exit(1)


def generate_random_dna(length: int) -> str:
    return "".join(random.choices(["A", "C", "G", "T"], k=length))


def main():
    seq_length = 1_000_000  
    num_runs = 10           # Колво запусков для усреднения

    print(f"генерация днк длиной {seq_length:,} bp")
    raw_sequence = generate_random_dna(seq_length)

    #  для NumPy-энкодера
    alphabet = DNAAlphabet.standard()
    dna_seq = DNASequence(raw_sequence, alphabet)
    numpy_encoder = OneHotEncoder(alphabet, layout="channel_first")

    print(f"запуск бенчмарка ({num_runs} иттераций)")

    # Python и NumPy
    numpy_times = []
    for i in range(num_runs):
        t0 = time.perf_counter()
        _ = numpy_encoder.encode(dna_seq)
        t1 = time.perf_counter()
        numpy_times.append(t1 - t0)
    
    avg_numpy_time = np.mean(numpy_times)
    print(f"   NumPy среднее время {avg_numpy_time:.5f} seconds")

    # C++ мой
    cpp_times = []
    for i in range(num_runs):
        t0 = time.perf_counter()
        _ = motiflab_cpp.encode_one_hot(raw_sequence, "channel_first")
        t1 = time.perf_counter()
        cpp_times.append(t1 - t0)
        
    avg_cpp_time = np.mean(cpp_times)
    print(f"  C++  среднее время {avg_cpp_time:.5f} seconds")

    #Сравнение
    speedup = avg_numpy_time / avg_cpp_time
    print(f"BENCHMARK RESULTS (Sequence length: {seq_length:,} bp)")
    print(f"NumPy average time: {avg_numpy_time * 1000:.2f} ms")
    print(f"C++ average time:   {avg_cpp_time * 1000:.2f} ms")
    print(f" C++ Encoder is {speedup:.2f}x faster than NumPy")


if __name__ == "__main__":
    main()