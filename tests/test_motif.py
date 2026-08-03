# tests/interpretation/test_motif.py

import pytest
import numpy as np

from motiflab.bioinformatics.alphabet import DNAAlphabet
from motiflab.interpretation.motif import MotifInterpreter


@pytest.fixture
def interpreter():
    return MotifInterpreter(DNAAlphabet.standard())


def test_weights_to_ppm_sums_to_one(interpreter):
    dummy_weights = np.random.randn(2, 4, 9)
    ppm = interpreter.weights_to_ppm(dummy_weights)
    assert ppm.shape == (2, 4, 9)
    assert np.allclose(np.sum(ppm, axis=1), 1.0)


def test_information_content(interpreter):
    perfect_ppm = np.array([
        [1.0, 0.25],
        [0.0, 0.25],
        [0.0, 0.25],
        [0.0, 0.25]
    ])
    
    ic = interpreter.calculate_information_content(perfect_ppm)
    
    assert ic.shape == (2,)
    assert np.isclose(ic[0], 2.0)
    assert np.isclose(ic[1], 0.0) 


def test_seqlets_to_ppm(interpreter):
    #вырезанные seqlets
    seqlets = ["AAAA", "AAAC"] 
    
    ppm = interpreter.seqlets_to_ppm(seqlets)
    
    assert ppm.shape == (4, 4)
    # первой колонке доминирует буква 'A' (индекс 0)
    assert ppm[0, 0] > ppm[1, 0]