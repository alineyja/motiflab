# src/motiflab/datasets/transforms.py
import random
from motiflab.bioinformatics.sequence import DNASequence

def random_shuffle(sequence: DNASequence) -> DNASequence:
    #строка в список,in-place, обратно
    seq_list = list(sequence.sequence)
    random.shuffle(seq_list)
    shuffled_str = "".join(seq_list)
    
    return DNASequence(
        sequence=shuffled_str,
        alphabet=sequence.alphabet,
        coords=None
    )