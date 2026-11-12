
from pathlib import Path
from collections import Counter

NUCLEOTIDES = ("A", "C", "G", "T")

CODAGE = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


def lire_fasta(chemin):
    """
    Lit un fichier FASTA contenant une seule sequence ADN.
    Retourne une chaine de nucleotides validee.
    """
    chemin = Path(chemin)

    if not chemin.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}"
        )

    lignes = chemin.read_text(
        encoding="utf-8"
    ).splitlines()

    lignes = [
        ligne.strip()
        for ligne in lignes
        if ligne.strip()
    ]

    if not lignes or not lignes[0].startswith(">"):
        raise ValueError("En-tete FASTA manquant.")

    if any(ligne.startswith(">") for ligne in lignes[1:]):
        raise ValueError(
            "Ce lecteur accepte une seule sequence FASTA."
        )

    sequence = "".join(lignes[1:]).upper()

    if not sequence:
        raise ValueError("Sequence ADN vide.")

    invalides = set(sequence) - set(NUCLEOTIDES)

    if invalides:
        raise ValueError(
            f"Nucleotides non autorises : {sorted(invalides)}"
        )

    return sequence


def coder_sequence(sequence):
    """Convertit A, C, G, T en 0, 1, 2, 3."""
    return [CODAGE[nucleotide] for nucleotide in sequence]


def calculer_frequences(sequence):
    """Calcule les frequences empiriques des nucleotides."""
    compteurs = Counter(sequence)
    n = len(sequence)

    return {
        nucleotide: compteurs[nucleotide] / n
        for nucleotide in NUCLEOTIDES
    }


if __name__ == "__main__":
    chemin = Path("data/examples/sequence_test.fasta")

    sequence = lire_fasta(chemin)
    codes = coder_sequence(sequence)
    frequences = calculer_frequences(sequence)

    print("Sequence :", sequence)
    print("Longueur :", len(sequence))
    print("Codage   :", codes)
    print("Frequences :")

    for nucleotide, frequence in frequences.items():
        print(f"  {nucleotide} : {frequence:.4f}")
