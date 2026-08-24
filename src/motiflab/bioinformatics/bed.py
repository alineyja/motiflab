from typing import Iterator
from pathlib import Path
from motiflab.bioinformatics.coordinates import GenomeCoordinates, Strand

def parse_bed(filepath: str | Path) -> Iterator[GenomeCoordinates]:
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Пропускаем пустые строки и комментарии
            if not line or line.startswith(('#', 'track', 'browser')):
                continue
            
            parts = line.split()
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            
            # В BED файле цепь обычно в 6-й колонке (индекс 5)
            strand = Strand.UNKNOWN
            if len(parts) >= 6:
                strand_str = parts[5]
                if strand_str in {"+", "-", "."}:
                    strand = Strand(strand_str)

            signal_val = 0.0
            if len(parts) >= 7:
                try:
                    signal_val = float(parts[6])
                except ValueError:
                    signal_val = 0.0


            yield GenomeCoordinates(
                chromosome=chrom,
                start=start,
                end=end,
                strand=strand,
                signal_value=signal_val
            )