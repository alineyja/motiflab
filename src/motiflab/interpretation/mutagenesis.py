# src/motiflab/interpretation/mutagenesis.py

import torch
import numpy as np
from typing import Tuple

from motiflab.bioinformatics.sequence import DNASequence
from motiflab.preprocessing.encoder import OneHotEncoder

class MutagenesisAnalyzer:
    def __init__(self, model: torch.nn.Module, encoder: OneHotEncoder):
        self.model = model
        self.encoder = encoder
        self.device = next(model.parameters()).device
        self.model.eval()

    def analyze(self, sequence: DNASequence) -> Tuple[np.ndarray, float]:
        L = len(sequence)
        vocab_size = self.encoder.alphabet.size

        #1 предикт для исходной последовательности
        wt_tensor = torch.from_numpy(self.encoder.encode(sequence)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            wt_logit = self.model(wt_tensor)
            baseline_p = torch.sigmoid(wt_logit).item()

        #2 Генерация мутанции
        wt_matrix = self.encoder.encode(sequence) 
        
        num_mutants = L * vocab_size
        mutant_batch = np.repeat(wt_matrix[np.newaxis, :, :], num_mutants, axis=0)

        valid_indices = [] 
        idx = 0
        
        for pos in range(L):
            wt_base_idx = self.encoder.alphabet.encode_symbol(sequence.sequence[pos])
            for mut_base_idx in range(vocab_size):
                if mut_base_idx == wt_base_idx:
                    idx += 1
                    continue
                
                mutant_batch[idx, :, pos] = 0.0
                mutant_batch[idx, mut_base_idx, pos] = 1.0
                
                valid_indices.append(idx)
                idx += 1

        mutant_batch_filtered = mutant_batch[valid_indices]
        tensor_batch = torch.from_numpy(mutant_batch_filtered).to(self.device)

        #3. Прогоняем через модель
        with torch.no_grad():
            mut_logits = self.model(tensor_batch)
            mut_probs = torch.sigmoid(mut_logits).cpu().numpy().flatten()

        #4 Создаем матрицу delta_p и возвращаем
        delta_p_matrix = np.full((vocab_size, L), np.nan, dtype=np.float32)
        
        prob_idx = 0
        idx = 0
        for pos in range(L):
            wt_base_idx = self.encoder.alphabet.encode_symbol(sequence.sequence[pos])
            for mut_base_idx in range(vocab_size):
                if mut_base_idx == wt_base_idx:
                    idx += 1
                    continue
                
                delta_p_matrix[mut_base_idx, pos] = mut_probs[prob_idx] - baseline_p
                prob_idx += 1
                idx += 1

        return delta_p_matrix, baseline_p