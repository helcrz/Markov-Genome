# Markov-Genome — Comment lire l'ADN ?

**Projet semestriel GM4 — INSA Rouen Normandie, 2026–2027**

## Présentation

Ce projet étudie l'application des chaînes de Markov à l'analyse des séquences d'ADN, notamment pour l'annotation des régions codantes et non codantes.

Il s'appuie sur une démarche mathématique et informatique allant du codage des nucléotides à la construction de modèles markoviens, puis à l'implémentation de l'algorithme de Viterbi.

## Équipe

- Helmi BEN MAAOUI
- Bilel CHTOUROU
- Ahmed HAMROUNI

**Encadrante :** Mme Ioana CIOTIR

## Objectifs

1. Importer, nettoyer et coder les séquences ADN.
2. Construire une chaîne de Markov homogène et étudier ses propriétés.
3. Estimer les probabilités de transition et analyser leurs propriétés statistiques.
4. Construire un modèle de Markov caché (HMM).
5. Implémenter l'algorithme de Viterbi.
6. Valider les résultats et produire des visualisations.

## Technologies

- Python
- NumPy, SciPy, Pandas
- Matplotlib, NetworkX
- Jupyter Notebook
- Git et GitHub
- LaTeX pour la documentation scientifique

## Installation

Cloner le dépôt :

```bash
git clone https://github.com/helcrz/Markov-Genome.git
cd Markov-Genome
```

Créer et activer un environnement virtuel Python, puis installer les dépendances :

```bash
python -m pip install -r requirements.txt
```

## Organisation

- `src/` : implémentations Python
- `notebooks/` : expériences et visualisations
- `data/` : séquences ADN et simulations
- `tests/` : tests des algorithmes
- `results/` : figures et tableaux
- `docs/` : documentation mathématique et technique

## Références

É. Pardoux, *Processus de Markov et applications — Algorithmes, Réseaux, Génome et Finance*, chapitre 4 : « Chaînes de Markov et génome ».

## Échéance

**Soutenance : 11 décembre 2026**

Projet réalisé dans le cadre de la formation Génie Mathématique de l'INSA Rouen Normandie.
