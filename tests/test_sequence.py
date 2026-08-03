import pytest
from motiflab.bioinformatics.alphabet import DNAAlphabet
from motiflab.bioinformatics.coordinates import GenomeCoordinates, Strand
from motiflab.bioinformatics.sequence import DNASequence


@pytest.fixture
def alphabet():
    return DNAAlphabet.standard()


def test_sequence_initialization(alphabet):
    # Успешная инициализация и авто-upper
    seq = DNASequence("acgt", alphabet)
    assert seq.sequence == "ACGT"
    assert len(seq) == 4


def test_sequence_validation_errors(alphabet):
    # Ошибка алфавита
    with pytest.raises(ValueError, match="not present in the alphabet"):
        DNASequence("ACGTN", alphabet)

    # Ошибка длины координат
    coords = GenomeCoordinates("chr1", 100, 200)
    with pytest.raises(ValueError, match="Length mismatch"):
        DNASequence("ACGT", alphabet, coords=coords)


def test_reverse_complement(alphabet):
    coords = GenomeCoordinates("chr1", 100, 104, Strand.PLUS)
    seq = DNASequence("ATCG", alphabet, coords=coords)
    
    rc_seq = seq.reverse_complement()
    
    # 5'-ATCG-3' -> complement: TAGC -> reverse: CGAT
    assert rc_seq.sequence == "CGAT"
    assert rc_seq.coords.strand is Strand.MINUS
    
    # Проверяем что физические границы интервала на геноме не изменились
    assert rc_seq.coords.start == 100
    assert rc_seq.coords.end == 104


def test_slice_plus_strand(alphabet):
    coords = GenomeCoordinates("chr1", 100, 110, Strand.PLUS)
    seq = DNASequence("ACGTACGTAC", alphabet, coords=coords)
    sliced = seq[2:6]
    
    assert sliced.sequence == "GTAC"
    assert sliced.coords.start == 102
    assert sliced.coords.end == 106
    assert sliced.coords.strand is Strand.PLUS


def test_slice_minus_strand(alphabet):
    coords = GenomeCoordinates("chr1", 100, 110, Strand.MINUS)
    seq = DNASequence("ACGTACGTAC", alphabet, coords=coords)
    sliced = seq[2:6]
    
    assert sliced.sequence == "GTAC"
    assert sliced.coords.start == 104
    assert sliced.coords.end == 108
    assert sliced.coords.strand is Strand.MINUS


def test_slice_unknown_strand(alphabet):
    coords = GenomeCoordinates("chr1", 100, 110, Strand.UNKNOWN)
    seq = DNASequence("ACGTACGTAC", alphabet, coords=coords)
    
    sliced = seq[2:6]
    assert sliced.coords.start == 102
    assert sliced.coords.end == 106
    assert sliced.coords.strand is Strand.UNKNOWN


def test_slice_no_coords(alphabet):
    seq = DNASequence("ACGTACGTAC", alphabet)
    sliced = seq[2:6]
    
    assert sliced.sequence == "GTAC"
    assert sliced.coords is None