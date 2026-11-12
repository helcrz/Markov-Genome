
import pytest

from src.dna_processing import (
    lire_fasta,
    coder_sequence,
    calculer_frequences,
)


def test_lecture_fasta(tmp_path):
    fichier = tmp_path / "test.fasta"
    fichier.write_text(">test\nacg\nGTA\n")

    assert lire_fasta(fichier) == "ACGGTA"


def test_codage():
    assert coder_sequence("ACGGTA") == [
        0, 1, 2, 2, 3, 0
    ]


def test_frequences():
    frequences = calculer_frequences("ACGGTA")

    assert frequences["A"] == pytest.approx(2 / 6)
    assert frequences["C"] == pytest.approx(1 / 6)
    assert frequences["G"] == pytest.approx(2 / 6)
    assert frequences["T"] == pytest.approx(1 / 6)


def test_nucleotide_invalide(tmp_path):
    fichier = tmp_path / "invalide.fasta"
    fichier.write_text(">test\nACNGT\n")

    with pytest.raises(ValueError):
        lire_fasta(fichier)
