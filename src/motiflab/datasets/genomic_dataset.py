# src/motiflab/datasets/genomic_dataset.py

import random
from typing import List, Tuple
import torch
from torch.utils.data import Dataset

from motiflab.bioinformatics.coordinates import GenomeCoordinates
from motiflab.bioinformatics.fasta import FastaExtractor
from motiflab.preprocessing.encoder import OneHotEncoder
from motiflab.bioinformatics.sequence import DNASequence

class GenomicDataset(Dataset):
    def __init__(
        self,
        peaks: List[GenomeCoordinates],
        fasta_extractor: FastaExtractor,
        encoder: OneHotEncoder,
        shift_range: Tuple[int, int] = (10000, 100000)
    ):
        self.peaks = peaks
        self.fasta_extractor = fasta_extractor
        self.encoder = encoder
        self.shift_range = shift_range
        self.num_peaks = len(peaks)

    def _get_valid_background_dna(self, coords: GenomeCoordinates, max_attempts: int = 5) -> DNASequence:
        for _ in range(max_attempts):
            # Сдвиг влево или вправо
            direction = 1 if random.random() > 0.5 else -1
            shift_amount = random.randint(self.shift_range[0], self.shift_range[1]) * direction
            
            new_start = max(0, coords.start + shift_amount)
            new_end = new_start + coords.length
            
            bg_coords = GenomeCoordinates(
                chromosome=coords.chromosome,
                start=new_start,
                end=new_end,
                strand=coords.strand
            )
            
            try:
                dna = self.fasta_extractor.extract(bg_coords)
                if "N" in dna.sequence.upper():
                    continue
                    
                return dna
            except ValueError:
                continue
        dna = self.fasta_extractor.extract(coords)
        seq_list = list(dna.sequence)
        random.shuffle(seq_list)
        return DNASequence(
            sequence="".join(seq_list),
            alphabet=self.encoder.alphabet,
            coords=None
        )

    def __len__(self) -> int:
        return self.num_peaks * 2

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        is_positive = idx < self.num_peaks
        peak_idx = idx if is_positive else idx - self.num_peaks
        
        coords = self.peaks[peak_idx]
        
        if is_positive:
            label = 1.0
            dna = self.fasta_extractor.extract(coords)
        else:
            label = 0.0
            dna = self._get_valid_background_dna(coords)
            
        matrix_np = self.encoder.encode(dna)
        tensor_x = torch.from_numpy(matrix_np)
        tensor_y = torch.tensor([label], dtype=torch.float32)
        
        return tensor_x, tensor_y