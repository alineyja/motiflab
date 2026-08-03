import numpy as np
from typing import Literal, Optional

from motiflab.bioinformatics.base_alphabet import Alphabet
from motiflab.bioinformatics.sequence import DNASequence
from motiflab.bioinformatics.coordinates import GenomeCoordinates


class OneHotEncoder: 
    def __init__(
        self,
        alphabet: Alphabet,
        layout: Literal["channel_last", "channel_first"] = "channel_first",
        dtype: np.dtype = np.float32,
    ):
        if layout not in {"channel_last", "channel_first"}:
            raise ValueError("Layout must be 'channel_last' or 'channel_first'")
            
        self.alphabet = alphabet
        self.layout = layout
        self.dtype = dtype

    def encode(self, sequence: DNASequence) -> np.ndarray:
        # Получаем индексы от алфавита
        indices = np.asarray(self.alphabet.encode(sequence.sequence), dtype=np.int64)
        
        # Векторизованный One-Hot через взятие строк из единичной матрицы
        matrix = np.eye(self.alphabet.size, dtype=self.dtype)[indices]

        # Применяем нужный layout
        if self.layout == "channel_first":
            return matrix.T
            
        return matrix

    def decode(
        self, 
        matrix: np.ndarray, 
        coords: Optional[GenomeCoordinates] = None
    ) -> DNASequence:
        # Определяем ось, по которой ищем максимальную вероятность (индекс буквы)
        axis = 0 if self.layout == "channel_first" else 1
        
        indices = np.argmax(matrix, axis=axis)
        
        # Конвертируем обратно в строку
        sequence_str = self.alphabet.decode(tuple(indices))
        
        return DNASequence(
            sequence=sequence_str, 
            alphabet=self.alphabet,
            coords=coords
        )