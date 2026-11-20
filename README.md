# Architecture Transformer

Travail pratique sur les modèles de séquences : du RNN au Transformer, puis deux modèles préentraînés (GPT-2 et BERT) et trois exercices à rendre.

Le notebook `architecture_transformer.ipynb` contient le cours, les schémas, le code exécuté et les graphiques.

## Contenu

| Partie | Ce que fait le notebook |
|---|---|
| 1. RNN | Prédit la valeur suivante d’une sinusoïde à partir des 10 précédentes |
| 2. LSTM | Même tâche, avec une mémoire contrôlée par des portes |
| 3. Transformer | Même tâche, avec un encodeur et de l’auto-attention |
| GPT-2 | Génère la suite d’une phrase en anglais |
| BERT multilingue | Complète un mot masqué dans une phrase en français |
| Exercice 1 | Prédire la consommation électrique horaire avec un Transformer |
| Exercice 2 | Entraîner un mini-GPT caractère par caractère |
| Exercice 3 | Entraîner un mini-BERT pour retrouver des mots masqués |

Les parties 1 à 3 et les exemples GPT-2 / BERT sont exécutés. Les exercices 1 à 3 restent à compléter : le notebook prépare les données de l’exercice 1 et donne le squelette des exercices 2 et 3.

## Résultats des démonstrations

Sinusoïde, 300 époques, graine `42`, erreur quadratique moyenne :

| Modèle | Erreur d’entraînement (époque 300) | Erreur de test |
|---|---:|---:|
| RNN | 0.000025 | 0.000028 |
| LSTM | 0.000003 | 0.000003 |
| Transformer | 0.000004 | 0.000006 |

Pour le premier point de test (valeur réelle `-0.3639`) :

- RNN : `-0.3601`
- LSTM : `-0.3622`
- Transformer : `-0.3662`

Les courbes réel / prédit sont enregistrées dans le notebook.

GPT-2, à partir de *« Artificial intelligence can help people »*, a produit une suite d’environ 50 tokens. Le tirage est aléatoire (`do_sample=True`) : une nouvelle exécution donne un autre texte.

BERT (`bert-base-multilingual-cased`) sur *« Rabat est la [MASK] Maroc. »* propose en tête :

| Probabilité | Phrase |
|---:|---|
| 42.90 % | Rabat est la capitale Maroc. |
| 20.66 % | Rabat est la du Maroc. |
| 6.23 % | Rabat est la de Maroc. |
| 3.46 % | Rabat est la ville Maroc. |
| 1.67 % | Rabat est la Rabat Maroc. |

## Installation

Python 3.11. Le code tourne sur CPU.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Sous Windows : `.venv\Scripts\activate`.

## Lancer le notebook

Depuis la racine du projet, avec le venv activé :

```bash
python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3"
jupyter notebook architecture_transformer.ipynb
```

Choisir le kernel **Python 3** du venv. Pour tout ré-exécuter :

```bash
jupyter nbconvert --to notebook --execute --inplace architecture_transformer.ipynb
```

La première exécution télécharge GPT-2 et `bert-base-multilingual-cased` (environ 1,2 Go) dans le cache Hugging Face. Compter plusieurs minutes selon le réseau.

## Jeu de données de l’exercice 1

Fichier attendu à côté du notebook : `household_power_consumption.txt`.

Source : [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (UCI).

```bash
curl -L -o household.zip "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
unzip household.zip
```

Le fichier fait environ 127 Mo. GitHub refuse un fichier de plus de 100 Mo, donc il est listé dans `.gitignore`. Chacun le télécharge en local. Pour le versionner quand même, il faut [Git LFS](https://git-lfs.com/).

La cellule de préparation agrège `Global_active_power` en moyenne horaire, normalise avec la moyenne et l’écart-type du train, puis construit des fenêtres de 24 heures (70 % train, 15 % validation, 15 % test).

## Exercices à rendre

**Exercice 1 — consommation électrique.** La préparation des données est déjà dans le notebook. Il reste à coder `EnergyTransformer` (`Linear(1, 32)`, positions sur 24 heures, deux `TransformerEncoderLayer` avec `d_model=32`, `nhead=4`, `dim_feedforward=128`, `dropout=0.1`), à l’entraîner au plus 20 époques avec Adam (`lr=0.001`) et la MSE, à garder le meilleur modèle de validation, puis à comparer MAE et RMSE en kW à la persistance et à la référence de la veille.

**Exercice 2 — mini-GPT.** `corpus.txt` est fourni à la racine : un texte original en UTF-8, assez long pour la fenêtre de 64 caractères. On peut le remplacer par un corpus plus riche. Il reste à implémenter un Transformer autorégressif par caractères : deux blocs, quatre têtes, dimension 64, masque causal, perte d’entropie croisée, génération de 200 caractères aux températures 0.5, 1.0 et 1.5.

La cellule de préparation lit ce fichier et construit `x`. La cellule suivante en déduit le masque causal. Elle doit être exécutée après la préparation.

**Exercice 3 — mini-BERT.** Tokenisation par mots, 15 % de tokens masqués, encodeur bidirectionnel, perte seulement sur les positions sélectionnées. La cellule qui montre le calcul de la perte utilise des variables (`input_ids`, `positions_selectionnees`, `logits`, `taille_vocabulaire`) à définir dans l’implémentation : elle lève `NameError` tant que ce code n’est pas écrit.

## Arborescence

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── architecture_transformer.ipynb
└── household_power_consumption.txt   # local uniquement, non poussé
```

`.venv/` n’est pas versionné.
