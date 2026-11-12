
import numpy as np
import pytest

from src.markov_chain import (
    compter_transitions,
    estimer_transition,
    analyser_classes,
    calculer_periodes,
    distribution_invariante,
)


@pytest.fixture
def matrice_adn():
    N = compter_transitions("ACGGTA")
    return estimer_transition(N)


def test_irreductibilite(matrice_adn):
    analyse = analyser_classes(matrice_adn)

    assert analyse["irreductible"] is True
    assert len(analyse["classes"]) == 1


def test_recurrence(matrice_adn):
    analyse = analyser_classes(matrice_adn)

    assert analyse["recurrentes"] == {
        "A", "C", "G", "T"
    }
    assert analyse["transientes"] == set()


def test_aperiodicite(matrice_adn):
    periodes = calculer_periodes(matrice_adn)

    assert all(p == 1 for p in periodes.values())


def test_distribution_invariante(matrice_adn):
    pi = distribution_invariante(matrice_adn)

    attendu = np.array([0.2, 0.2, 0.4, 0.2])

    np.testing.assert_allclose(pi, attendu)
    np.testing.assert_allclose(pi @ matrice_adn, pi)
    assert np.isclose(pi.sum(), 1.0)


def test_chaine_reductible():
    P = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ], dtype=float)

    analyse = analyser_classes(P)

    assert analyse["irreductible"] is False
    assert analyse["recurrentes"] == {"A", "T"}
    assert analyse["transientes"] == {"C", "G"}

    with pytest.raises(ValueError):
        distribution_invariante(P)


def test_chaine_periodique():
    P = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)

    periodes = calculer_periodes(P)

    assert periodes["A"] == 2
    assert periodes["C"] == 2
    assert periodes["G"] == 1
    assert periodes["T"] == 1
