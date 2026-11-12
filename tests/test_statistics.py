import numpy as np
import pytest

from src.statistics import (
    P_REF,
    simuler_chaine,
    frequences_empiriques,
    erreur_estimation,
    experience_convergence,
)
from src.markov_chain import verifier_transition


def test_matrice_reference():
    assert verifier_transition(P_REF)


def test_longueur_simulation():
    rng = np.random.default_rng(42)

    sequence = simuler_chaine(P_REF, 100, rng)

    assert len(sequence) == 100
    assert set(sequence).issubset({"A", "C", "G", "T"})


def test_reproductibilite():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    s1 = simuler_chaine(P_REF, 100, rng1)
    s2 = simuler_chaine(P_REF, 100, rng2)

    assert s1 == s2


def test_chaine_deterministe():
    P = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ], dtype=float)

    rng = np.random.default_rng(42)

    sequence = simuler_chaine(P, 8, rng)

    assert sequence == "ACGTACGT"


def test_frequences():
    f = frequences_empiriques("ACGGTA")

    attendu = np.array([
        2 / 6,
        1 / 6,
        2 / 6,
        1 / 6,
    ])

    np.testing.assert_allclose(f, attendu)
    assert np.isclose(f.sum(), 1.0)


def test_erreur_nulle():
    assert erreur_estimation(P_REF, P_REF) == pytest.approx(0)


def test_matrice_incomplete():
    P = P_REF.copy()
    P[0, :] = np.nan

    assert np.isnan(erreur_estimation(P, P_REF))


def test_experience_reproductible():
    r1 = experience_convergence(
        P_REF, tailles=(50, 100),
        repetitions=3, graine=7
    )

    r2 = experience_convergence(
        P_REF, tailles=(50, 100),
        repetitions=3, graine=7
    )

    assert r1.equals(r2)
    assert list(r1["n"]) == [50, 100]


def test_longueur_invalide():
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        simuler_chaine(P_REF, 0, rng)
