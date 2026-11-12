
"""
Phase 3 - Statistiques des chaines de Markov.

Simulation, estimation, convergence et visualisations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dna_processing import NUCLEOTIDES
from src.markov_chain import (
    compter_transitions,
    estimer_transition,
    verifier_transition,
    distribution_invariante,
)


# Matrice de reference pour les simulations.
# Elle ne provient pas de donnees biologiques.
P_REF = np.array([
    [0.4, 0.3, 0.2, 0.1],
    [0.2, 0.4, 0.3, 0.1],
    [0.1, 0.2, 0.5, 0.2],
    [0.3, 0.2, 0.2, 0.3],
])


def simuler_chaine(P, n, rng, etat_initial="A"):
    """
    Simule n nucleotides selon une chaine de Markov.

    P : matrice de transition stochastique 4x4
    n : longueur de la sequence
    rng : generateur numpy.random.Generator
    etat_initial : premier nucleotide
    """
    if not verifier_transition(P):
        raise ValueError("Matrice P invalide.")

    if n < 1:
        raise ValueError("La longueur doit etre positive.")

    if etat_initial not in NUCLEOTIDES:
        raise ValueError("Etat initial inconnu.")

    indices = {e: i for i, e in enumerate(NUCLEOTIDES)}

    etat = indices[etat_initial]
    sequence = [etat_initial]

    for _ in range(n - 1):
        etat = int(rng.choice(4, p=P[etat]))
        sequence.append(NUCLEOTIDES[etat])

    return "".join(sequence)


def frequences_empiriques(sequence):
    """
    Calcule le vecteur des frequences empiriques
    dans l'ordre A, C, G, T.
    """
    if not sequence:
        raise ValueError("Sequence vide.")

    if any(x not in NUCLEOTIDES for x in sequence):
        raise ValueError("Sequence ADN invalide.")

    return np.array([
        sequence.count(e) / len(sequence)
        for e in NUCLEOTIDES
    ])


def erreur_estimation(P_estimee, P_reference):
    """
    Calcule ||P_estimee - P_reference||_F.

    Retourne NaN si une ligne est non estimable.
    """
    P_estimee = np.asarray(P_estimee, dtype=float)
    P_reference = np.asarray(P_reference, dtype=float)

    if not verifier_transition(P_reference):
        raise ValueError("Reference invalide.")

    if P_estimee.shape != P_reference.shape:
        raise ValueError("Dimensions incompatibles.")

    if not np.isfinite(P_estimee).all():
        return np.nan

    return float(
        np.linalg.norm(P_estimee - P_reference, ord="fro")
    )


def experience_convergence(
    P,
    tailles=(50, 100, 300, 1000, 3000, 10000),
    repetitions=20,
    graine=42,
):
    """
    Repete les simulations pour plusieurs longueurs.

    Chaque simulation utilise une trajectoire
    independante dans une meme experience.

    Retourne un DataFrame de statistiques.
    """
    if not verifier_transition(P):
        raise ValueError("Matrice P invalide.")

    if repetitions < 1:
        raise ValueError("Nombre de repetitions invalide.")

    rng = np.random.default_rng(graine)

    resultats = []

    for n in tailles:
        if n < 2:
            raise ValueError("Chaque taille doit etre >= 2.")

        erreurs = []

        for _ in range(repetitions):
            sequence = simuler_chaine(P, n, rng)

            N = compter_transitions(sequence)
            P_estimee = estimer_transition(N)

            erreur = erreur_estimation(P_estimee, P)

            if np.isfinite(erreur):
                erreurs.append(erreur)

        resultats.append({
            "n": n,
            "repetitions": repetitions,
            "estimations_valides": len(erreurs),
            "erreur_moyenne": (
                float(np.mean(erreurs))
                if erreurs else np.nan
            ),
            "ecart_type": (
                float(np.std(erreurs, ddof=1))
                if len(erreurs) > 1 else np.nan
            ),
        })

    return pd.DataFrame(resultats)


def produire_resultats():
    """
    Execute les experiences et exporte
    les tableaux et figures pour le rapport.
    """
    dossier_figures = Path("results/figures")
    dossier_tables = Path("results/tables")

    dossier_figures.mkdir(parents=True, exist_ok=True)
    dossier_tables.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2026)

    # 1. Etude de convergence
    resultats = experience_convergence(P_REF)

    resultats.to_csv(
        dossier_tables / "convergence_markov.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        resultats["n"],
        resultats["erreur_moyenne"],
        yerr=resultats["ecart_type"],
        fmt="o-",
        capsize=3,
        label="Moyenne +/- ecart-type",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Longueur de la sequence")
    ax.set_ylabel("Erreur de Frobenius")
    ax.set_title("Convergence de l'estimateur de P")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        dossier_figures / "convergence_markov.png",
        dpi=200
    )
    plt.close(fig)

    # 2. Une longue trajectoire
    sequence = simuler_chaine(
        P_REF, n=10000, rng=rng
    )

    N = compter_transitions(sequence)
    P_estimee = estimer_transition(N)

    # 3. Distribution invariante
    pi = distribution_invariante(P_REF)
    frequences = frequences_empiriques(sequence)

    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - 0.2, pi, width=0.4,
        label="Distribution invariante"
    )
    ax.bar(
        x + 0.2, frequences, width=0.4,
        label="Frequences empiriques"
    )

    ax.set_xticks(x, NUCLEOTIDES)
    ax.set_ylabel("Probabilite / frequence")
    ax.set_title("Distribution invariante et empirique")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        dossier_figures / "distribution_empirique.png",
        dpi=200
    )
    plt.close(fig)

    # 4. Comparaison des matrices
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4),
        constrained_layout=True
    )

    for ax, matrice, titre in zip(
        axes,
        (P_REF, P_estimee),
        ("P de reference", "P estimee"),
    ):
        im = ax.imshow(
            matrice, cmap="Blues",
            vmin=0, vmax=1
        )

        ax.set_title(titre)
        ax.set_xticks(range(4), NUCLEOTIDES)
        ax.set_yticks(range(4), NUCLEOTIDES)
        ax.set_xlabel("Arrivee")
        ax.set_ylabel("Depart")

        for i in range(4):
            for j in range(4):
                ax.text(
                    j, i, f"{matrice[i, j]:.2f}",
                    ha="center", va="center",
                    color=(
                        "white"
                        if matrice[i, j] > 0.5
                        else "black"
                    )
                )

    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(
        dossier_figures / "matrices_markov.png",
        dpi=200
    )
    plt.close(fig)

    # 5. Tableau des distributions
    comparaison = pd.DataFrame({
        "nucleotide": NUCLEOTIDES,
        "pi_theorique": pi,
        "frequence_empirique": frequences,
    })

    comparaison.to_csv(
        dossier_tables / "distribution_markov.csv",
        index=False
    )

    print("\nEtude de convergence :")
    print(resultats.to_string(index=False))

    print("\nMatrice de reference :")
    print(P_REF)

    print("\nMatrice estimee (n=10000) :")
    print(P_estimee)

    print("\nDistribution invariante :", pi)
    print("Distribution empirique :", frequences)
    print("\nFichiers exportes dans results/")


if __name__ == "__main__":
    produire_resultats()
