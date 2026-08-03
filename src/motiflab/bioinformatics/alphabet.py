from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Final, Tuple

from .base_alphabet import Alphabet


@dataclass(frozen=True, slots=True)
class DNAAlphabet(Alphabet):
    symbols: Tuple[str, ...]
    complements: Dict[str, str]

    def __post_init__(self) -> None:
        encode_map = {
            symbol: index
            for index, symbol in enumerate(self.symbols)
        }

        decode_map = {
            index: symbol
            for symbol, index in encode_map.items()
        }

        object.__setattr__(self, "_encode_map", encode_map)
        object.__setattr__(self, "_decode_map", decode_map)

    @classmethod
    def standard(cls) -> "DNAAlphabet":
        return cls(
            symbols=("A", "C", "G", "T"),
            complements={
                "A": "T",
                "T": "A",
                "C": "G",
                "G": "C",
            },
        )

    def encode_symbol(self, symbol: str) -> int:
        try:
            return self._encode_map[symbol]
        except KeyError:
            raise ValueError(
                f"Unknown DNA symbol '{symbol}'."
            )

    def decode_symbol(self, index: int) -> str:
        try:
            return self._decode_map[index]
        except KeyError:
            raise ValueError(
                f"Unknown DNA index '{index}'."
            )

    def complement_symbol(self, symbol: str) -> str:
        try:
            return self.complements[symbol]
        except KeyError:
            raise ValueError(
                f"Unknown DNA symbol '{symbol}'."
            )

    def encode(self, sequence: str) -> Tuple[int, ...]:
        return tuple(
            self.encode_symbol(symbol)
            for symbol in sequence
        )

    def decode(self, indices: Tuple[int, ...]) -> str:
        return "".join(
            self.decode_symbol(index)
            for index in indices
        )

    def complement(self, sequence: str) -> str:
        return "".join(
            self.complement_symbol(symbol)
            for symbol in sequence
        )

    def is_valid_symbol(self, symbol: str) -> bool:
        return symbol in self._encode_map