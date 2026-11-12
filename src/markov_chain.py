
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.dna_processing import lire_fasta, NUCLEOTIDES


def compter_transitions(sequence):
    """
    Construit la matrice des comptages N_ij.

    L'ordre des etats est A, C, G, T.
    """
    etats = {n: i for i, n in enumerate(NUCLEOTIDES)}
    N = np.zeros((4, 4), dtype=int)

    for actuel, suivant in zip(sequence[:-1], sequence[1:]):
        N[etats[actuel], etats[suivant]] += 1

    return N


def estimer_transition(N):
    """
    Estime P_ij = N_ij / somme_j N_ij.

    Si aucun depart n'est observe depuis un etat,
    la ligne est remplie de NaN.
    """
    N = np.asarray(N)

    if N.shape != (4, 4) or np.any(N < 0):
        raise ValueError("Matrice de comptages invalide.")

    totaux = N.sum(axis=1)

    P = np.full((4, 4), np.nan, dtype=float)

    for i in range(4):
        if totaux[i] > 0:
            P[i, :] = N[i, :] / totaux[i]

    return P


def verifier_transition(P):
    """
    Verifie les lignes estimables de P.

    Retourne True seulement si toutes les lignes
    sont definies et stochastiques.
    """
    P = np.asarray(P, dtype=float)

    if P.shape != (4, 4):
        return False

    if not np.all(np.isfinite(P)):
        return False

    if np.any(P < 0) or np.any(P > 1):
        return False

    return bool(np.allclose(P.sum(axis=1), 1.0))


def visualiser_markov(P, chemin=None):
    """Dessine le graphe oriente des transitions."""

    if not verifier_transition(P):
        raise ValueError(
            "Une matrice stochastique complete est requise."
        )

    graphe = nx.DiGraph()
    graphe.add_nodes_from(NUCLEOTIDES)

    for i, depart in enumerate(NUCLEOTIDES):
        for j, arrivee in enumerate(NUCLEOTIDES):
            if P[i, j] > 0:
                graphe.add_edge(
                    depart, arrivee, weight=float(P[i, j])
                )

    positions = {
        "A": (0, 1),
        "C": (1, 1),
        "G": (1, 0),
        "T": (0, 0),
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    nx.draw_networkx_nodes(
        graphe, positions, ax=ax,
        node_color="#dcefeb", node_size=2200
    )

    nx.draw_networkx_labels(
        graphe, positions, ax=ax,
        font_size=14, font_weight="bold"
    )

    nx.draw_networkx_edges(
        graphe, positions, ax=ax,
        arrows=True, arrowsize=22,
        node_size=2200, connectionstyle="arc3,rad=0.12"
    )

    etiquettes = {
        (u, v): f"{d['weight']:.2f}"
        for u, v, d in graphe.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        graphe, positions,
        edge_labels=etiquettes, ax=ax,
        font_size=10, rotate=False
    )

    ax.set_title("Transitions de la chaine de Markov")
    ax.axis("off")
    fig.tight_layout()

    if chemin is not None:
        chemin = Path(chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(chemin, dpi=200, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    sequence = lire_fasta(
        "data/examples/sequence_test.fasta"
    )

    N = compter_transitions(sequence)
    P = estimer_transition(N)

    print("Comptages :")
    print(N)

    print("\nMatrice de transition :")
    print(P)

    print("\nMatrice stochastique :")
    print(verifier_transition(P))

    visualiser_markov(
        P, "results/figures/graphe_markov.png"
    )
    plt.show()



import math

import networkx as nx
import numpy as np

from src.dna_processing import NUCLEOTIDES


def construire_graphe(P):
    """
    Construit le graphe de transition associe a P.
    """
    if not verifier_transition(P):
        raise ValueError(
            "P doit etre une matrice stochastique complete."
        )

    G = nx.DiGraph()
    G.add_nodes_from(NUCLEOTIDES)

    for i, depart in enumerate(NUCLEOTIDES):
        for j, arrivee in enumerate(NUCLEOTIDES):
            if P[i, j] > 0:
                G.add_edge(
                    depart,
                    arrivee,
                    weight=float(P[i, j])
                )

    return G


def analyser_classes(P):
    """
    Determine les classes de communication,
    les classes fermees et la nature des etats.

    Resultats valables pour une chaine finie.
    """
    G = construire_graphe(P)

    classes = [
        set(classe)
        for classe in nx.strongly_connected_components(G)
    ]

    # Tri uniquement pour rendre les sorties reproductibles.
    classes.sort(key=lambda c: min(c))

    fermees = []
    transientes = set()
    recurrentes = set()

    for classe in classes:
        sorties = any(
            voisin not in classe
            for etat in classe
            for voisin in G.successors(etat)
        )

        if not sorties:
            fermees.append(classe)
            recurrentes.update(classe)
        else:
            transientes.update(classe)

    return {
        "classes": classes,
        "irreductible": len(classes) == 1,
        "classes_fermees": fermees,
        "recurrentes": recurrentes,
        "transientes": transientes,
    }


def periode_classe(G, classe):
    """
    Calcule la periode d'une classe ayant un cycle.

    Pour une classe sans cycle, retourne None.
    """
    sous_graphe = G.subgraph(classe)
    racine = sorted(classe)[0]

    distances = nx.single_source_shortest_path_length(
        sous_graphe, racine
    )

    pgcd = 0

    for u, v in sous_graphe.edges():
        difference = distances[u] + 1 - distances[v]
        pgcd = math.gcd(pgcd, abs(difference))

    return pgcd if pgcd > 0 else None


def calculer_periodes(P):
    """
    Calcule la periode des etats a partir
    des classes de communication.

    None signifie : aucun retour positif possible.
    """
    G = construire_graphe(P)
    analyse = analyser_classes(P)

    periodes = {}

    for classe in analyse["classes"]:
        periode = periode_classe(G, classe)

        for etat in classe:
            periodes[etat] = periode

    return periodes


def distribution_invariante(P):
    """
    Calcule l'unique distribution invariante
    d'une chaine finie irreductible.

    Pour une chaine reductible, refuse de
    supposer l'unicite.
    """
    if not verifier_transition(P):
        raise ValueError("Matrice P invalide.")

    analyse = analyser_classes(P)

    if not analyse["irreductible"]:
        raise ValueError(
            "Chaine reductible : plusieurs distributions "
            "invariantes peuvent exister. "
            "Etudier d'abord les classes fermees."
        )

    n = P.shape[0]

    # Resolution de (P.T - I) pi = 0
    A = P.T - np.eye(n)
    b = np.zeros(n)

    # Remplacement d'une equation par sum(pi)=1.
    A[-1, :] = 1.0
    b[-1] = 1.0

    pi = np.linalg.solve(A, b)

    # Correction des seules erreurs numeriques.
    pi[np.abs(pi) < 1e-12] = 0.0

    if np.any(pi < 0):
        raise ValueError("Distribution negative inattendue.")

    pi = pi / pi.sum()

    if not np.allclose(pi @ P, pi, atol=1e-10):
        raise ValueError("Echec de la verification pi P = pi.")

    return pi
