
import numpy as np

from src.markov_chain import (
    compter_transitions,
    estimer_transition,
    verifier_transition,
)


def test_comptage():
    N = compter_transitions("ACGGTA")

    attendu = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
    ])

    np.testing.assert_array_equal(N, attendu)


def test_estimation():
    N = compter_transitions("ACGGTA")
    P = estimer_transition(N)

    attendu = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0.5, 0.5],
        [1, 0, 0, 0],
    ])

    np.testing.assert_allclose(P, attendu)


def test_matrice_stochastique():
    N = compter_transitions("ACGGTA")
    P = estimer_transition(N)

    assert verifier_transition(P)


def test_etat_sans_depart():
    N = compter_transitions("AAA")
    P = estimer_transition(N)

    assert P[0, 0] == 1.0
    assert np.isnan(P[1:, :]).all()
    assert not verifier_transition(P)


def test_nombre_transitions():
    sequence = "ACGGTA"
    N = compter_transitions(sequence)

    assert N.sum() == len(sequence) - 1
