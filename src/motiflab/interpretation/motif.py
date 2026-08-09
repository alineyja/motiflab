# src/motiflab/interpretation/motif.py

import numpy as np
from typing import List, Tuple
import torch

from motiflab.bioinformatics.base_alphabet import Alphabet
from motiflab.bioinformatics.sequence import DNASequence


def softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class MotifInterpreter:
    def __init__(self, alphabet: Alphabet):
        self.alphabet = alphabet

    def weights_to_ppm(self, weights: np.ndarray) -> np.ndarray:
        # применяем softmax по оси каналов (ось 1)
        # форма на входе F, 4, K - на выходе F, 4, K
        return softmax(weights, axis=1)

    def calculate_information_content(self, ppm: np.ndarray) -> np.ndarray:
        epsilon = 1e-9 # Защита от log(0)
        # H = -sum(p * log2(p))
        entropy = -np.sum(ppm * np.log2(ppm + epsilon), axis=0)
        ic = 2.0 - entropy
        return np.clip(ic, 0.0, 2.0)

    def extract_activating_seqlets(
        self,
        model: torch.nn.Module,
        sequences: List[DNASequence],
        filter_idx: int,
        threshold: float = 0.7
    ) -> List[str]:
        model.eval()
        kernel_size = model.kernel_size
        seqlets = []
        
        # глобальный максимум активации фильтра по всем данным
        all_activations = []
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for seq in sequences:
                # Кодирование последовательности в тензор формы (1, 4, L)
                # индексы от алфавита
                indices = np.asarray(self.alphabet.encode(seq.sequence), dtype=np.int64)
                matrix = np.eye(self.alphabet.size, dtype=np.float32)[indices].T
                tensor_x = torch.from_numpy(matrix).unsqueeze(0).to(device)
                
                # прогоняем только через сверточный слой (1, F, L)
                conv_out = model.relu(model.conv(tensor_x))
                # активации нужного фильтра (L,)
                act = conv_out[0, filter_idx].cpu().numpy()
                all_activations.append((seq.sequence, act))
        
        if not all_activations:
            return []
            
        # максимальное значение активации среди всех последовательностей
        global_max = max(np.max(act) for _, act in all_activations)
        cutoff = global_max * threshold
        
        # куски ДНК, которые превысили порог
        for raw_str, act in all_activations:
            # индексы позиций, где активация выше cutoff
            peak_indices = np.where(act >= cutoff)[0]
            
            for idx in peak_indices:
                # кусок ДНК центрированный вокруг максимума свертки
                start = idx - kernel_size // 2
                end = start + kernel_size
                
                # Защита от выхода за границы строки
                if start >= 0 and end <= len(raw_str):
                    seqlets.append(raw_str[start:end])
                    
        return seqlets

    def seqlets_to_ppm(self, seqlets: List[str]) -> np.ndarray:
        if not seqlets:
            raise ValueError("Seqlets list is empty.")
            
        kernel_size = len(seqlets[0])
        counts = np.zeros((4, kernel_size))
        
        # частота букв в каждой позиции
        for seqlet in seqlets:
            for col_idx, char in enumerate(seqlet):
                row_idx = self.alphabet.encode_symbol(char)
                counts[row_idx, col_idx] += 1
                
        # Нормализация частоты, добавляя Laplace pseudocounts
        pseudocount = 1.0
        normalized_ppm = (counts + pseudocount / 4) / (len(seqlets) + pseudocount)
        return normalized_ppm