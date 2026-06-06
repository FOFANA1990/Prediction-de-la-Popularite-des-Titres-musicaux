#  Prédiction de la Popularité des Titres Spotify 

## Description

Ce projet explore la prédiction de la **popularité d'un titre musical Spotify** (`track_popularity`, score de 0 à 100) à partir de ses caractéristiques audio et métadonnées.

Il s'agit d'un problème de **régression supervisée** traité de bout en bout : collecte -> nettoyage -> analyse exploratoire - feature engineering-> modélisation -> évaluation.

---

## Objectif

Construire un modèle de régression capable d'estimer la popularité d'un titre en fonction de ses attributs audio (énergie, danceability, loudness…) et contextuels (genre, année de sortie, artiste).

---

## Structure du Projet

```
.
├── TP_Regression_SPOTIFY_commented.ipynb   # Notebook principal (commenté)
├── README.md                                # Ce fichier
├── requirements.txt                         # Dépendances Python
├── high_popularity_spotify_data.csv         # Données : titres populaires
└── low_popularity_spotify_data.csv          # Données : titres peu populaires
```

---

## Pipeline du Notebook

Le notebook est organisé en étapes progressives :

### 1. Chargement des données
- Lecture de deux CSV distincts (titres à forte vs faible popularité)
- Fusion en un seul DataFrame unifié

### 2. Nettoyage (`ÉTAPE 3`)
- Suppression des valeurs manquantes (NaN)
- Suppression des doublons
- Suppression des colonnes techniques inutiles (`id`, `uri`, `playlist_id`…)

### 3. Analyse exploratoire (`ÉTAPES 4 & 5`)
- Détection des outliers par méthode **IQR**
- Distribution de la variable cible `track_popularity`
- Corrélations et heatmap
- Analyse par genre musical
- Scatter plots features vs popularité
- Analyse temporelle (évolution de la popularité par année)
- Vérification des bornes métier Spotify

### 4. Feature Engineering
- **Features artiste** : popularité moyenne/max de l'artiste, nombre de collaborations, comptage de titres dans le dataset
- **Features temporelles** : année de sortie, mois, ancienneté du titre en jours
- **Features textuelles** : longueur du nom du titre

### 5. Transformation des distributions
- Analyse du **skewness** de chaque feature audio
- Test de 7 transformations (log1p, sqrt, Yeo-Johnson, carré…)
- Application d'un **ColumnTransformer** avec la meilleure transformation par feature

### 6. Encodage des variables catégorielles
- Comparaison de 3 stratégies par validation croisée :
  - One-Hot Encoding
  - **Target Encoding** ✓ (retenu)
  - Approche hybride (OHE genre + Target subgenre)

### 7. Modélisation

| Modèle | Description |
|---|---|
| `LinearRegression` | Baseline OLS (sans régularisation) |
| `Lasso` | Régression L1 — sélection automatique de features |
| `ElasticNet` | Combinaison L1+L2 — robuste aux features corrélées |

Chaque modèle est intégré dans un **Pipeline sklearn** avec `StandardScaler` et optimisé via **GridSearchCV (CV=5)**.

### 8. Détection et correction du Data Leakage
- Identification des features construites à partir de la cible (`artist_avg_popularity`, `artist_max_popularity`)
- Ré-entraînement après suppression de ces colonnes

### 9. Analyse de la courbe d'apprentissage
- Visualisation du compromis **biais/variance**
- Diagnostic d'underfitting ou overfitting selon la convergence des courbes

---

## Métriques d'évaluation

| Métrique | Description |
|---|---|
| **MAE** | Erreur absolue moyenne (unités : points de popularité) |
| **RMSE** | Erreur quadratique moyenne (pénalise les grandes erreurs) |
| **R²** | Proportion de variance expliquée (1.0 = parfait) |

Une **baseline naive** (toujours prédire la moyenne) est calculée pour contextualiser les performances.

---

## Données

Les données proviennent de deux fichiers CSV issus de l'API Spotify :

- `high_popularity_spotify_data.csv` : titres avec `track_popularity` élevé
- `low_popularity_spotify_data.csv` : titres avec `track_popularity` faible

**Variable cible** : `track_popularity` (entier, 0–100)

**Features audio principales** :

| Feature | Description | Échelle |
|---|---|---|
| `energy` | Intensité et activité perçue | [0, 1] |
| `danceability` | Aptitude à la danse | [0, 1] |
| `loudness` | Volume moyen en dB | [-60, 0] |
| `acousticness` | Probabilité acoustique | [0, 1] |
| `instrumentalness` | Absence de voix | [0, 1] |
| `valence` | Positivité musicale | [0, 1] |
| `speechiness` | Présence de paroles | [0, 1] |
| `liveness` | Probabilité d'enregistrement live | [0, 1] |
| `tempo` | Tempo en BPM | [0, 300] |
| `duration_ms` | Durée en millisecondes | — |
| `key` | Tonalité musicale | [0, 11] |
| `mode` | Majeur (1) ou Mineur (0) | {0, 1} |
| `time_signature` | Signature temporelle | [0, 7] |

---

## Installation et Exécution

### Prérequis
- Python ≥ 3.8
- Jupyter Notebook ou JupyterLab

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Lancement

```bash
jupyter notebook TP_Regression_SPOTIFY_commented.ipynb
```

Ou dans Google Colab : importer le notebook et les deux CSV, puis exécuter toutes les cellules.

---

## Résultats Clés

- Les features les plus corrélées à la popularité sont liées à **l'artiste** (popularité historique) et au **genre musical**.
- Parmi les features audio pures, `loudness`, `energy` et `acousticness` ont la plus forte influence.
- Les titres récents (> 2020) sont en moyenne plus populaires dans ce dataset.
- La régression linéaire atteint ses limites (R² modeste) : la popularité dépend de facteurs non captés par les features audio seules (promotion, algorithmes de recommandation, réseaux sociaux…).

---

