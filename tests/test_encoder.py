import pytest
import numpy as np

from motiflab.bioinformatics.alphabet import DNAAlphabet
from motiflab.bioinformatics.sequence import DNASequence
from motiflab.preprocessing.encoder import OneHotEncoder


@pytest.fixture
def alphabet():
    return DNAAlphabet.standard()

@pytest.fixture
def sequence(alphabet):
    return DNASequence("ACGT", alphabet)


def test_encode_channel_first(alphabet, sequence):
    encoder = OneHotEncoder(alphabet, layout="channel_first")
    matrix = encoder.encode(sequence)
    
    assert matrix.shape == (4, 4) # (Channels, Length)
    # A = index 0 -> [1, 0, 0, 0] in channel_last, so [1, 0, 0, 0] in column 0
    assert matrix[0, 0] == 1.0 
    assert matrix[1, 0] == 0.0


def test_encode_channel_last(alphabet, sequence):
    encoder = OneHotEncoder(alphabet, layout="channel_last")
    matrix = encoder.encode(sequence)
    
    assert matrix.shape == (4, 4) # (Length, Channels)
    # C = index 1 -> [0, 1, 0, 0] at row 1
    assert matrix[1, 1] == 1.0
    assert matrix[1, 0] == 0.0


def test_decode_identity(alphabet, sequence):
    # Тест на обратимость: encode -> decode должно вернуть то же самое
    encoder = OneHotEncoder(alphabet, layout="channel_first")
    
    matrix = encoder.encode(sequence)
    decoded_seq = encoder.decode(matrix)
    
    assert decoded_seq.sequence == sequence.sequence