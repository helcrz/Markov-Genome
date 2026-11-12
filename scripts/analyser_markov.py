
from src.dna_processing import lire_fasta
from src.markov_chain import (
    compter_transitions,
    estimer_transition,
    analyser_classes,
    calculer_periodes,
    distribution_invariante,
)


def main():
    sequence = lire_fasta(
        "data/examples/sequence_test.fasta"
    )

    N = compter_transitions(sequence)
    P = estimer_transition(N)

    analyse = analyser_classes(P)
    periodes = calculer_periodes(P)
    pi = distribution_invariante(P)

    print("Matrice de transition :")
    print(P)

    print("\nClasses :", analyse["classes"])
    print("Irreductible :", analyse["irreductible"])
    print("Classes fermees :", analyse["classes_fermees"])
    print("Etats recurrents :", analyse["recurrentes"])
    print("Etats transients :", analyse["transientes"])

    print("\nPeriodes :", periodes)
    print("Distribution invariante :", pi)


if __name__ == "__main__":
    main()
