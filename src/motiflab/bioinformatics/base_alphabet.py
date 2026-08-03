# src/motiflab/bioinformatics/base_alphabet.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


class Alphabet(ABC):
    symbols: Tuple[str, ...]

    @property
    def size(self) -> int:
        """Alphabet size."""
        return len(self.symbols)

    @abstractmethod
    def encode_symbol(self, symbol: str) -> int:
        ...

    @abstractmethod
    def decode_symbol(self, index: int) -> str:
        ...

    @abstractmethod
    def complement_symbol(self, symbol: str) -> str:
        ...

    @abstractmethod
    def is_valid_symbol(self, symbol: str) -> bool:
        ...

    def is_valid_sequence(self, sequence: str) -> bool:
        return all(self.is_valid_symbol(symbol) for symbol in sequence)