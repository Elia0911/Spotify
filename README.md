# Spotify Track Popularity Prediction & Recommendation

Predicting a song's **popularity score** from its audio features, and building **recommender systems** on top of the same data — comparing a classical machine-learning approach against deep-learning approaches.

The project is split across two notebooks:

- **`Spotify_ML_Eli_Oun.ipynb`** — classical ML: Random Forest regressor + a cosine-similarity recommender
- **`Spotify_DL_Eli_Oun.ipynb`** — deep learning: a tuned DNN regressor, an autoencoder recommender, and a DNN "like/dislike" classifier

> Author: **Eli Oun**

---

## Overview

Given Spotify audio features for a track (danceability, energy, loudness, tempo, etc.), the main task is to **predict `track_popularity`** (a 0–100 score) as a regression problem. Two model families are compared:

- **Machine Learning** — a Random Forest regressor with exhaustive `GridSearchCV` tuning.
- **Deep Learning** — a fully-connected neural network with hyperparameters tuned via Keras Tuner.

Both notebooks share an identical preprocessing and feature-engineering pipeline so the model families can be compared fairly. Each notebook also generates test-set predictions, which are collected side-by-side in `Spotify_Prediction_Score-Eli_Oun.csv`.

Beyond popularity prediction, the project explores three **recommender** strategies on the same feature space (cosine similarity, autoencoder latent similarity, and a supervised DNN classifier).

## Dataset

- Training data: `spotify_songs_train.csv` (expects a `track_popularity` target column)
- Test data: `spotify_songs_X_test.csv` (features only, for generating predictions)

> The raw Spotify datasets are not included — the notebooks load them from local paths (originally Google Colab `/content/`). Update those paths to point at your own copies.

**Target:** `track_popularity` (0–100)

## Pipeline

Both notebooks apply the same steps:

1. **Handle missing values** — drop rows with `NaN`.
2. **Drop irrelevant columns** — IDs and names (`track_id`, `track_name`, `track_artist`, `track_album_id`, `track_album_name`, `track_album_release_date`, `playlist_name`, `playlist_id`, `Unnamed: 0`) that add noise without predictive value.
3. **Keep numeric features only** — the project focuses on numeric audio features to explore feature engineering rather than encoding categoricals.
4. **Feature engineering** — add interaction / non-linear terms:
   - `danceability_energy = danceability × energy`
   - `instrumentalness_duration = instrumentalness × duration_ms`
   - `tempo_squared = tempo²`
5. **Correlation analysis** — rank features by correlation with `track_popularity` to focus on the most predictive inputs.
6. **Scaling** — `RobustScaler` (robust to outliers).
7. **Split** — 80/20 train/test, `random_state=25`.

**Feature set used for modeling:**
`danceability`, `energy`, `loudness`, `valence`, `tempo`, `instrumentalness`, `duration_ms`, plus the three engineered interaction terms.

## Models

### Machine Learning — Random Forest Regressor

Tuned with `GridSearchCV` (5-fold, scoring on negative mean squared log error):

| Hyperparameter | Search grid |
|---|---|
| `n_estimators` | [100, 200] |
| `max_depth` | [None, 10, 20] |
| `min_samples_split` | [2, 5] |
| `min_samples_leaf` | [1, 2] |

**Best parameters found:** `n_estimators=200`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`

**Test performance:**

| Metric | Value |
|---|---|
| MSLE | 1.5649 |
| R² | 0.2552 |

### Deep Learning — Tuned DNN Regressor

A `Sequential` network of three `Dense` + `Dropout` blocks with a linear output, tuned with **Keras Tuner** (`RandomSearch`, `max_trials=10`, `executions_per_trial=2`, objective `val_loss`). Loss is mean squared logarithmic error; optimizer is Adam. Training uses `EarlyStopping` and `ReduceLROnPlateau`.

Search space:

| Hyperparameter | Range |
|---|---|
| Units (layer 1) | 32 → 512 (step 32) |
| Units (layer 2) | 32 → 512 (step 32) |
| Units (layer 3) | 32 → 256 (step 32) |
| Dropout (each block) | 0.2 → 0.5 (step 0.1) |
| Epochs | up to 100 (early-stopped) |

## Recommender Systems

Three approaches on the same feature space:

1. **Cosine-similarity recommender** (ML notebook) — compute a track-to-track cosine similarity matrix over scaled features; recommend by averaging similarity across a user's liked tracks and returning the most similar unseen tracks. Simple, efficient, interpretable, and domain-independent.

2. **Autoencoder recommender** (DL notebook) — an encoder–decoder network (latent dim 16) learns a compressed latent representation of each track; recommendations use cosine similarity in latent space. Captures non-linear relationships better than raw-feature similarity and needs no labels.

3. **DNN like/dislike classifier** (DL notebook) — synthetic binary labels (`liked = track_popularity > 70`) train a supervised classifier (sigmoid output, binary cross-entropy) that outputs a probability a user likes a track.

### DNN-based vs. Autoencoder-based (summary)

- **DNN (supervised):** needs labels, higher accuracy and stronger personalization when interaction data exists, but requires retraining when preferences change.
- **Autoencoder (unsupervised):** no labels needed, flexible and scalable for content-based recommendation, but less personalized and harder to interpret.

## Outputs

`Spotify_Prediction_Score-Eli_Oun.csv` holds the test-set popularity predictions from both model families (6,567 rows):

| Column | Description |
|---|---|
| `Unnamed: 0` | Row index |
| `ML prediction` | Random Forest predicted popularity |
| `DL prediction` | DNN predicted popularity |

Prediction summary (0–100 scale):

| | ML prediction | DL prediction |
|---|---|---|
| mean | 43.26 | 28.96 |
| std | 13.46 | 9.04 |
| min | 0.76 | 1.94 |
| max | 98.00 | 59.56 |

The two model families correlate at ~0.56. The Random Forest spans a wider range and centers higher, while the DNN predicts a narrower, lower band — worth keeping in mind given the modest R², which reflects how hard track popularity is to predict from audio features alone.

## Setup

### Prerequisites
- Python 3.9+
- pandas, numpy, scikit-learn, matplotlib
- tensorflow / keras, keras-tuner

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib tensorflow keras-tuner
```

### Usage

1. Place `spotify_songs_train.csv` and `spotify_songs_X_test.csv` where the notebooks expect them (update the `/content/...` paths).
2. Run `Spotify_ML_Eli_Oun.ipynb` for the Random Forest regressor + cosine recommender.
3. Run `Spotify_DL_Eli_Oun.ipynb` for the tuned DNN, autoencoder recommender, and DNN classifier.
4. Each notebook writes an `output.csv` of predictions; these are combined into `Spotify_Prediction_Score-Eli_Oun.csv`.

## Repository Structure

```
.
├── Spotify_ML_Eli_Oun.ipynb              # Random Forest regressor + cosine recommender
├── Spotify_DL_Eli_Oun.ipynb              # Tuned DNN + autoencoder + DNN classifier
├── Spotify_Prediction_Score-Eli_Oun.csv  # Combined ML & DL test predictions
└── README.md
```

## References

Selected references cited in the notebooks:

- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
- Kuhn, M., & Johnson, K. (2019). *Feature Engineering and Selection.* CRC Press.
- He, X., Liao, L., Zhang, H., et al. (2017). *Neural Collaborative Filtering.*
- Sedhain, S., Menon, A. K., et al. (2015). *AutoRec: Autoencoders Meet Collaborative Filtering.*
- Liang, D., Krishnan, R., et al. (2018). *Variational Autoencoders for Collaborative Filtering.*

## Author

**Eli Oun**
