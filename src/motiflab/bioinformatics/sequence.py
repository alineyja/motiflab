from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from .alphabet import DNAAlphabet
from .coordinates import GenomeCoordinates, Strand


@dataclass(frozen=True, slots=True)
class DNASequence:

    sequence: str
    alphabet: DNAAlphabet
    coords: Optional[GenomeCoordinates] = None

    def __post_init__(self) -> None:
        # Автоматическая очистка к верхнему регистру
        cleaned_sequence = self.sequence.strip().upper()
        object.__setattr__(self, "sequence", cleaned_sequence)

        # Валидация алфавитом
        if not self.alphabet.is_valid_sequence(self.sequence):
            raise ValueError(
                "Sequence contains symbols that are not present in the alphabet."
            )

        # Синхронизация с координатами
        if self.coords is not None:
            if len(self.sequence) != len(self.coords):
                raise ValueError(
                    f"Length mismatch sequence is {len(self.sequence)} bp, "
                    f"but coordinates span {len(self.coords)} bp."
                )

    def reverse_complement(self) -> DNASequence:
        rc_seq = self.alphabet.complement(self.sequence)[::-1]
        new_coords = self.coords.flip() if self.coords else None

        return DNASequence(
            sequence=rc_seq,
            alphabet=self.alphabet,
            coords=new_coords,
        )

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, key: Union[int, slice]) -> Union[str, DNASequence]:
        if isinstance(key, int):
            return self.sequence[key]

        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            
            if step != 1:
                raise ValueError("Slicing with step != 1 is not supported.")

            new_sequence = self.sequence[start:stop]
            new_coords = None

            if self.coords is not None:
                # Математика пересчета координат (то, что мы обсуждали)
                if self.coords.strand is Strand.MINUS:
                    # На минус-цепи строка читается "справа налево" по отношению к геному
                    new_start = self.coords.end - stop
                    new_end = self.coords.end - start
                else:
                    # Для PLUS и UNKNOWN цепей отсчет идет слева направо
                    new_start = self.coords.start + start
                    new_end = self.coords.start + stop

                new_coords = GenomeCoordinates(
                    chromosome=self.coords.chromosome,
                    start=new_start,
                    end=new_end,
                    strand=self.coords.strand,
                )

            return DNASequence(
                sequence=new_sequence,
                alphabet=self.alphabet,
                coords=new_coords,
            )

        raise TypeError(f"Invalid argument type: {type(key)}")

    def __str__(self) -> str:
        return self.sequence

    def __repr__(self) -> str:
        coord_str = f", coords={self.coords!r}" if self.coords else ""
        return f"DNASequence(length={len(self)}{coord_str})"