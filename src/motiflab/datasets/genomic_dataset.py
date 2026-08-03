# src/motiflab/datasets/genomic_dataset.py
from typing import List, Tuple
import torch
from torch.utils.data import Dataset

from motiflab.bioinformatics.coordinates import GenomeCoordinates
from motiflab.bioinformatics.fasta import FastaExtractor
from motiflab.preprocessing.encoder import OneHotEncoder
from motiflab.datasets.transforms import random_shuffle

class GenomicDataset(Dataset):
    def __init__(
        self,
        peaks: List[GenomeCoordinates],
        fasta_extractor: FastaExtractor,
        encoder: OneHotEncoder
    ):
        self.peaks = peaks
        self.fasta_extractor = fasta_extractor
        self.encoder = encoder
        self.num_peaks = len(peaks)

    def __len__(self) -> int:
        return self.num_peaks * 2

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # позитивный или негативный пример
        is_positive = idx < self.num_peaks
        
        #соответствующий пик (для негативных примеров с начала списка)
        peak_idx = idx if is_positive else idx - self.num_peaks
        coords = self.peaks[peak_idx]
        
        # ДНК из FASTA
        dna = self.fasta_extractor.extract(coords)
        
        if is_positive:
            label = 1.0
        else:
            label = 0.0
            # негативный пример 
            dna = random_shuffle(dna)
            
        # Конвертируем DNASequence - NumPy - PyTorch Tensor
        matrix_np = self.encoder.encode(dna)
        tensor_x = torch.from_numpy(matrix_np)
        
        # Таргет тоже должен быть тензором
        tensor_y = torch.tensor([label], dtype=torch.float32)
        
        return tensor_x, tensor_y