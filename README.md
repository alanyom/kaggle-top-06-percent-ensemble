# Kaggle Playground S5E5 — Calorie Expenditure Prediction

**🥇 27th place out of 4,316 teams (Top 0.6%) | RMSLE: 0.05852**

Predicting calorie expenditure from exercise and biometric data using a Bayesian-tuned inverse-error weighted ensemble of gradient boosting models.

## Overview

This project documents my full competition workflow — from a simple GBM baseline to a fine-tuned ensemble that placed in the top 0.6% of 4,316 teams. The approach evolved iteratively, with each version improving on the last.

| Version | Approach | Val RMSLE |
|---|---|---|
| V1 | GBM baseline (3 features) | 0.09340 |
| V2 | Tuned ensemble (HistGB + LightGBM + XGBoost) | 0.05985 |
| **Final** | **Submission** | **0.05852** |

## Approach

**Preprocessing**
- Numerical features standardized with `StandardScaler` fit on training data only
- Categorical feature (`Sex`) one-hot encoded with `drop_first=True`

**Hyperparameter Optimization**
- Bayesian optimization via Hyperopt (Tree-structured Parzen Estimator) for all three models
- 100 trials per model minimizing RMSLE on a held-out validation set

**Ensemble**
- Inverse-error weighting: each model weighted by `1 / RMSLE`, normalized to sum to 1
- Error correlations between models (~0.89–0.96) confirmed meaningful diversity
- Models refit on full training data before generating final test predictions

## Results

| Model | Val RMSLE | Ensemble Weight |
|---|---|---|
| HistGradientBoosting | 0.06012 | 0.3382 |
| LightGBM | 0.06089 | 0.3326 |
| XGBoost | 0.06134 | 0.3292 |
| **Ensemble** | **0.05985** | — |
| **Leaderboard** | **0.05852** | — |

The leaderboard score outperforming the local validation score indicates strong generalization with no overfitting to the public leaderboard.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| ML | scikit-learn, LightGBM, XGBoost |
| Tuning | Hyperopt |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

## Project Structure

```
├── calorie-expenditure-ensemble.ipynb   # Full competition notebook
├── requirements.txt                      # Dependencies
└── README.md
```

## Usage

1. Download the dataset from [Kaggle Playground S5E5](https://www.kaggle.com/competitions/playground-series-s5e5)
2. Place `train.csv` and `test.csv` in `/kaggle/input/playground-series-s5e5/`
3. Run all cells in `calorie-expenditure-ensemble.ipynb`
