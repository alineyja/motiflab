# src/motiflab/bioinformatics/coordinates.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Strand(str, Enum):
    PLUS = "+"
    MINUS = "-"
    UNKNOWN = "."

    def flip(self) -> "Strand":
        if self is Strand.PLUS:
            return Strand.MINUS
        if self is Strand.MINUS:
            return Strand.PLUS
        return Strand.UNKNOWN


@dataclass(frozen=True, slots=True)
class GenomeCoordinates:
    chromosome: str
    start: int
    end: int
    strand: Strand = Strand.UNKNOWN
    signal_value: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.chromosome, str):
            raise TypeError("Chromosome must be a string.")
        if not isinstance(self.start, int):
            raise TypeError("Start coordinate must be an integer.")
        if not isinstance(self.end, int):
            raise TypeError("End coordinate must be an integer.")

        if not self.chromosome.strip():
            raise ValueError("Chromosome name cannot be empty.")

        if self.start < 0:
            raise ValueError("Start coordinate cannot be negative.")
        if self.end <= self.start:
            raise ValueError("End coordinate must be greater than start.")

        if not isinstance(self.strand, Strand):
            try:
                object.__setattr__(self, "strand", Strand(self.strand))
            except ValueError:
                raise ValueError(f"Invalid strand: '{self.strand}'.")

    @property
    def length(self) -> int:
        return self.end - self.start

    def contains(self, position: int) -> bool:
        return self.start <= position < self.end

    def overlaps(self, other: object) -> bool:
        if not isinstance(other, GenomeCoordinates):
            return NotImplemented

        if self.chromosome != other.chromosome:
            return False
        return self.start < other.end and other.start < self.end

    def shift(self, offset: int) -> GenomeCoordinates:
        new_start = self.start + offset
        new_end = self.end + offset

        if new_start < 0:
            raise ValueError(f"Shift by {offset} results in negative coordinate ({new_start}).")

        return GenomeCoordinates(
            chromosome=self.chromosome,
            start=new_start,
            end=new_end,
            strand=self.strand,
        )

    def flip(self) -> GenomeCoordinates:
        return GenomeCoordinates(
            chromosome=self.chromosome,
            start=self.start,
            end=self.end,
            strand=self.strand.flip(),
        )

    def __contains__(self, position: int) -> bool:
        return self.contains(position)

    def __len__(self) -> int:
        return self.length

    def __str__(self) -> str:
        return f"{self.chromosome}:{self.start}-{self.end}({self.strand.value})"

    def __repr__(self) -> str:
        return f"GenomeCoordinates({self.chromosome}:{self.start}-{self.end}({self.strand.value}))"