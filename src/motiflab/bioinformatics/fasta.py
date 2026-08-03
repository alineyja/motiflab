from pathlib import Path
from pyfaidx import Fasta

from motiflab.bioinformatics.base_alphabet import Alphabet
from motiflab.bioinformatics.sequence import DNASequence
from motiflab.bioinformatics.coordinates import GenomeCoordinates, Strand

class FastaExtractor:
    def __init__(self, fasta_path: str | Path, alphabet: Alphabet):
        self.fasta = Fasta(str(fasta_path))
        self.alphabet = alphabet

    def extract(self, coords: GenomeCoordinates) -> DNASequence:
        if coords.chromosome not in self.fasta:
            raise ValueError(f"Chromosome {coords.chromosome} not found in FASTA.")

        raw_seq = self.fasta[coords.chromosome][coords.start:coords.end].seq
        plus_coords = GenomeCoordinates(
            chromosome=coords.chromosome,
            start=coords.start,
            end=coords.end,
            strand=Strand.PLUS
        )
        
        dna_seq = DNASequence(
            sequence=raw_seq,
            alphabet=self.alphabet,
            coords=plus_coords
        )

        if coords.strand is Strand.MINUS:
            return dna_seq.reverse_complement()
        
        if coords.strand is Strand.UNKNOWN:
            return DNASequence(
                sequence=dna_seq.sequence,
                alphabet=self.alphabet,
                coords=coords
            )
            
        return dna_seq

    def close(self):
        self.fasta.close()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()