# src/motiflab/bioinformatics/coordinates.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Strand(str, Enum):
    """
    Enumeration for genomic strands.
    Inherits from str to allow easy serialization, e.g., in BED files.
    """
    PLUS = "+"
    MINUS = "-"
    UNKNOWN = "."

    def flip(self) -> "Strand":
        """Invert the strand. Unknown remains unknown."""
        if self is Strand.PLUS:
            return Strand.MINUS
        if self is Strand.MINUS:
            return Strand.PLUS
        return Strand.UNKNOWN


@dataclass(frozen=True, slots=True)
class GenomeCoordinates:
    """
    Immutable genomic interval.

    Coordinates follow the BED convention:
        - 0-based
        - half-open interval [start, end)
    """

    chromosome: str
    start: int
    end: int
    strand: Strand = Strand.UNKNOWN

    def __post_init__(self) -> None:
        # 1. Проверка типов (ранняя защита от неверных данных)
        if not isinstance(self.chromosome, str):
            raise TypeError("Chromosome must be a string.")
        if not isinstance(self.start, int):
            raise TypeError("Start coordinate must be an integer.")
        if not isinstance(self.end, int):
            raise TypeError("End coordinate must be an integer.")

        # 2. Проверка на пустую строку хромосомы
        if not self.chromosome.strip():
            raise ValueError("Chromosome name cannot be empty.")

        # 3. Математические инварианты координат
        if self.start < 0:
            raise ValueError("Start coordinate cannot be negative.")
        if self.end <= self.start:
            raise ValueError("End coordinate must be greater than start.")

        # 4. Безопасный кастинг строки в Enum (UX улучшение)
        if not isinstance(self.strand, Strand):
            try:
                object.__setattr__(self, "strand", Strand(self.strand))
            except ValueError:
                raise ValueError(
                    f"Invalid strand: '{self.strand}'. "
                    f"Must be one of: '+', '-', '.'"
                )

    @property
    def length(self) -> int:
        return self.end - self.start

    def contains(self, position: int) -> bool:
        """Check whether a genomic position belongs to the interval."""
        return self.start <= position < self.end

    def overlaps(self, other: object) -> bool:
        """True if two intervals overlap."""
        if not isinstance(other, GenomeCoordinates):
            return NotImplemented

        if self.chromosome != other.chromosome:
            return False

        # Касание границ в полуоткрытых интервалах не считается пересечением
        return self.start < other.end and other.start < self.end

    def shift(self, offset: int) -> GenomeCoordinates:
        """
        Shift the interval along the chromosome.
        Raises ValueError if the shift results in negative coordinates.
        """
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
        """Return new coordinates with the strand flipped."""
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
        # Удобный формат для отладки в консоли
        return f"GenomeCoordinates({self.chromosome}:{self.start}-{self.end}({self.strand.value}))"